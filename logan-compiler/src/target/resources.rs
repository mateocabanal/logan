use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use logan_ir::{
    MemoryPoolBudget, MemoryPoolId, ResourceBudget, StoragePoolBudget, StoragePoolId,
};
use sha2::{Digest, Sha256};

use crate::{
    error::{ColicError, Result},
    target::MachineProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPoolKind {
    HostRam,
    UnifiedMemory,
    DeviceVram,
    PinnedStaging,
    AcceleratorPrivate,
}

impl MemoryPoolKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::HostRam => "host-ram",
            Self::UnifiedMemory => "uma",
            Self::DeviceVram => "device-vram",
            Self::PinnedStaging => "pinned-staging",
            Self::AcceleratorPrivate => "accelerator-private",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineMemoryPool {
    pub id: MemoryPoolId,
    pub kind: MemoryPoolKind,
    /// Physical capacity is a hardware/resource fact. `None` means the probe
    /// could not establish it; callers must not silently turn unknown into a
    /// stable hardware identity.
    pub capacity_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupport {
    Supported,
    Unsupported,
    Unknown,
}

impl CapabilitySupport {
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "yes",
            Self::Unsupported => "no",
            Self::Unknown => "unknown",
        }
    }
}

/// Stable facts about one storage pool. Volatile free space deliberately does
/// not live here; it is recorded in `StoragePoolObservation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePoolProfile {
    pub id: StoragePoolId,
    /// Filesystem/device identity reported by the host. This is diagnostic and
    /// contributes to the planning-pool fingerprint, but is not an execution
    /// ABI identifier.
    pub filesystem_identity: String,
    pub mount_point: String,
    pub capacity_bytes: u64,
    /// Proven by an actual create-new/delete probe in the selected directory.
    pub writable: CapabilitySupport,
    /// Keep capabilities unknown until Logan has actually probed or has a
    /// platform contract for them. Do not guess from filesystem type/name.
    pub pageable_mapping: CapabilitySupport,
    pub sparse_files: CapabilitySupport,
    pub direct_io: CapabilitySupport,
    pub preferred_alignment: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePoolObservation {
    pub profile: StoragePoolProfile,
    /// Volatile planner observation. This is checked again before allocation
    /// and must not be part of the stable execution ABI.
    pub available_bytes: u64,
    /// Existing directory used for the probe. Persist requirements, not this
    /// machine-specific absolute path, in package execution plans.
    pub probe_directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineResourceProfile {
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub unified_memory: bool,
    pub memory_pools: Vec<MachineMemoryPool>,
    /// Deterministic hardware/resource fingerprint. It includes memory
    /// topology/capacity but no volatile storage free-space observation.
    pub fingerprint: String,
}

impl MachineResourceProfile {
    /// Convert stable/observed resource facts into the optimizer envelope.
    ///
    /// `memory_limits` is optional per-pool usable capacity after higher-level
    /// OS/runtime/safety reservations. A supplied limit may only reduce a
    /// known physical capacity; it can never fabricate extra memory.
    pub fn resource_budget(
        &self,
        storage: &[StoragePoolObservation],
        memory_limits: &BTreeMap<MemoryPoolId, u64>,
    ) -> Result<ResourceBudget> {
        let mut memory_pools = Vec::with_capacity(self.memory_pools.len());
        for pool in &self.memory_pools {
            let physical = pool.capacity_bytes.ok_or_else(|| {
                ColicError::Usage(format!(
                    "memory pool `{}` capacity is unknown; cannot build a deterministic resource budget",
                    pool.id.0
                ))
            })?;
            let capacity = memory_limits.get(&pool.id).copied().unwrap_or(physical);
            if capacity == 0 || capacity > physical {
                return Err(ColicError::Usage(format!(
                    "memory pool `{}` usable limit {} is invalid for physical capacity {}",
                    pool.id.0, capacity, physical
                )));
            }
            memory_pools.push(MemoryPoolBudget {
                id: pool.id.clone(),
                capacity_bytes: capacity,
            });
        }

        let storage_pools = storage
            .iter()
            .map(|observation| StoragePoolBudget {
                id: observation.profile.id.clone(),
                available_bytes: observation.available_bytes,
                // Mutable backing is admitted only after writeability was
                // proven, never from a metadata/permission-bit guess.
                writable: observation.profile.writable.is_supported(),
            })
            .collect::<Vec<_>>();
        let budget = ResourceBudget {
            memory_pools,
            storage_pools,
        };
        budget.validate().map_err(ColicError::Usage)?;
        Ok(budget)
    }

    /// Stable fingerprint for the selected memory topology plus the stable
    /// storage-pool facts. Volatile available/free bytes are excluded.
    pub fn fingerprint_with_storage(&self, storage: &[StoragePoolObservation]) -> String {
        let mut canonical = format!("machine={}\n", self.fingerprint);
        let mut profiles = storage.iter().map(|item| &item.profile).collect::<Vec<_>>();
        profiles.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        for profile in profiles {
            canonical.push_str(&format!(
                "storage={}:{}:{}:{}:{}:{}:{}:{}\n",
                profile.id.0,
                profile.filesystem_identity,
                profile.mount_point,
                profile.capacity_bytes,
                profile.writable.as_str(),
                profile.pageable_mapping.as_str(),
                profile.sparse_files.as_str(),
                profile.direct_io.as_str(),
            ));
        }
        sha256_hex(canonical.as_bytes())
    }
}

impl MachineProfile {
    pub fn resource_profile(&self) -> MachineResourceProfile {
        let memory_pools = if self.unified_memory {
            vec![MachineMemoryPool {
                id: MemoryPoolId::new("uma0"),
                kind: MemoryPoolKind::UnifiedMemory,
                capacity_bytes: self.ram_bytes,
            }]
        } else {
            // Current machine probing does not yet enumerate discrete GPUs.
            // Do not invent VRAM pools; #56/#68 will append real pools when
            // device discovery provides their capacities.
            vec![MachineMemoryPool {
                id: MemoryPoolId::new("host"),
                kind: MemoryPoolKind::HostRam,
                capacity_bytes: self.ram_bytes,
            }]
        };
        let mut canonical = format!(
            "os={};arch={};uma={};metal={};apple8={};avx2={};apple_gpu_family_min={};",
            self.operating_system,
            self.architecture,
            u8::from(self.unified_memory),
            u8::from(self.metal_available),
            u8::from(self.apple8_abi),
            u8::from(self.avx2),
            self.apple_gpu_family_min,
        );
        for pool in &memory_pools {
            canonical.push_str(&format!(
                "mem={}:{}:{};",
                pool.id.0,
                pool.kind.as_str(),
                pool.capacity_bytes
                    .map(|bytes| bytes.to_string())
                    .unwrap_or_else(|| "unknown".into())
            ));
        }
        MachineResourceProfile {
            operating_system: self.operating_system,
            architecture: self.architecture,
            unified_memory: self.unified_memory,
            memory_pools,
            fingerprint: sha256_hex(canonical.as_bytes()),
        }
    }
}

/// Observe the filesystem containing `path`. The path itself need not exist;
/// Logan walks to the nearest existing parent directory so output/state paths
/// can be planned before creation.
pub fn observe_storage_path(path: &Path) -> Result<StoragePoolObservation> {
    let directory = nearest_existing_directory(path)?;
    let row = probe_df(&directory)?;
    let writable = prove_directory_writable(&directory);
    let storage_id = StoragePoolId::new(format!(
        "storage-{}",
        &sha256_hex(
            format!(
                "os={};fs={};mount={}",
                std::env::consts::OS,
                row.filesystem_identity,
                row.mount_point
            )
            .as_bytes()
        )[..16]
    ));
    Ok(StoragePoolObservation {
        profile: StoragePoolProfile {
            id: storage_id,
            filesystem_identity: row.filesystem_identity,
            mount_point: row.mount_point,
            capacity_bytes: row.capacity_bytes,
            writable,
            pageable_mapping: CapabilitySupport::Unknown,
            sparse_files: CapabilitySupport::Unknown,
            direct_io: CapabilitySupport::Unknown,
            preferred_alignment: None,
        },
        available_bytes: row.available_bytes,
        probe_directory: directory,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DfRow {
    filesystem_identity: String,
    capacity_bytes: u64,
    available_bytes: u64,
    mount_point: String,
}

fn nearest_existing_directory(path: &Path) -> Result<PathBuf> {
    let mut candidate = path;
    loop {
        if candidate.is_dir() {
            return fs::canonicalize(candidate).map_err(|source| ColicError::Io {
                path: candidate.to_owned(),
                source,
            });
        }
        if candidate.exists() {
            let parent = candidate.parent().ok_or_else(|| {
                ColicError::Usage(format!("storage path `{}` has no parent directory", path.display()))
            })?;
            return fs::canonicalize(parent).map_err(|source| ColicError::Io {
                path: parent.to_owned(),
                source,
            });
        }
        candidate = candidate.parent().ok_or_else(|| {
            ColicError::Usage(format!(
                "storage path `{}` has no existing parent directory",
                path.display()
            ))
        })?;
    }
}

#[cfg(unix)]
fn probe_df(directory: &Path) -> Result<DfRow> {
    let output = Command::new("df")
        .arg("-Pk")
        .arg(directory)
        .output()
        .map_err(|source| ColicError::Io {
            path: directory.to_owned(),
            source,
        })?;
    if !output.status.success() {
        return Err(ColicError::Usage(format!(
            "failed to inspect storage capacity for `{}` with df -Pk",
            directory.display()
        )));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        ColicError::Usage(format!("df -Pk returned non-UTF8 output: {error}"))
    })?;
    parse_df_pk(&stdout).ok_or_else(|| {
        ColicError::Usage(format!(
            "could not parse df -Pk output for `{}`",
            directory.display()
        ))
    })
}

#[cfg(not(unix))]
fn probe_df(directory: &Path) -> Result<DfRow> {
    Err(ColicError::unsupported(
        "storage capacity probe",
        format!(
            "path-bound storage probing is not implemented on {} for `{}` yet",
            std::env::consts::OS,
            directory.display()
        ),
    ))
}

fn parse_df_pk(text: &str) -> Option<DfRow> {
    let line = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with("Filesystem"))?;
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let percent_index = tokens.iter().position(|token| token.ends_with('%'))?;
    if percent_index < 4 || percent_index + 1 >= tokens.len() {
        return None;
    }
    let capacity_kib = tokens.get(percent_index - 3)?.parse::<u64>().ok()?;
    let available_kib = tokens.get(percent_index - 1)?.parse::<u64>().ok()?;
    let filesystem_identity = tokens[..percent_index - 3].join(" ");
    if filesystem_identity.is_empty() {
        return None;
    }
    let mount_point = tokens[percent_index + 1..].join(" ");
    if mount_point.is_empty() {
        return None;
    }
    Some(DfRow {
        filesystem_identity,
        capacity_bytes: capacity_kib.checked_mul(1024)?,
        available_bytes: available_kib.checked_mul(1024)?,
        mount_point,
    })
}

fn prove_directory_writable(directory: &Path) -> CapabilitySupport {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let probe = directory.join(format!(
        ".logan-storage-probe-{}-{nonce}",
        std::process::id()
    ));
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(file) => {
            drop(file);
            match fs::remove_file(&probe) {
                Ok(()) => CapabilitySupport::Supported,
                Err(_) => CapabilitySupport::Unknown,
            }
        }
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
            ) => CapabilitySupport::Unsupported,
        Err(_) => CapabilitySupport::Unknown,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apple_machine(ram_bytes: Option<u64>) -> MachineProfile {
        MachineProfile {
            operating_system: "macos",
            architecture: "aarch64",
            ram_bytes,
            unified_memory: true,
            metal_available: true,
            apple8_abi: true,
            avx2: false,
            apple_gpu_family_min: 8,
        }
    }

    fn linux_machine(ram_bytes: Option<u64>) -> MachineProfile {
        MachineProfile {
            operating_system: "linux",
            architecture: "x86_64",
            ram_bytes,
            unified_memory: false,
            metal_available: false,
            apple8_abi: false,
            avx2: true,
            apple_gpu_family_min: 8,
        }
    }

    fn storage(available_bytes: u64) -> StoragePoolObservation {
        StoragePoolObservation {
            profile: StoragePoolProfile {
                id: StoragePoolId::new("ssd0"),
                filesystem_identity: "/dev/test".into(),
                mount_point: "/models".into(),
                capacity_bytes: 1_000_000,
                writable: CapabilitySupport::Supported,
                pageable_mapping: CapabilitySupport::Unknown,
                sparse_files: CapabilitySupport::Unknown,
                direct_io: CapabilitySupport::Unknown,
                preferred_alignment: None,
            },
            available_bytes,
            probe_directory: PathBuf::from("/models"),
        }
    }

    #[test]
    fn apple_resource_profile_has_one_shared_uma_pool() {
        let profile = apple_machine(Some(16 << 30)).resource_profile();
        assert_eq!(profile.memory_pools.len(), 1);
        assert_eq!(profile.memory_pools[0].id, MemoryPoolId::new("uma0"));
        assert_eq!(profile.memory_pools[0].kind, MemoryPoolKind::UnifiedMemory);
        assert_eq!(profile.memory_pools[0].capacity_bytes, Some(16 << 30));
    }

    #[test]
    fn cpu_profile_uses_host_pool_without_inventing_vram() {
        let profile = linux_machine(Some(64 << 30)).resource_profile();
        assert_eq!(profile.memory_pools.len(), 1);
        assert_eq!(profile.memory_pools[0].id, MemoryPoolId::new("host"));
        assert_eq!(profile.memory_pools[0].kind, MemoryPoolKind::HostRam);
    }

    #[test]
    fn ram_capacity_changes_machine_resource_fingerprint_not_execution_abi() {
        let low = apple_machine(Some(16 << 30));
        let high = apple_machine(Some(64 << 30));
        assert_ne!(low.resource_profile().fingerprint, high.resource_profile().fingerprint);
        assert!(low.apple8_abi && high.apple8_abi);
    }

    #[test]
    fn df_parser_uses_capacity_and_available_fields_around_percent_column() {
        let linux = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/nvme0n1p2 1000000 250000 750000 25% /models\n";
        let parsed = parse_df_pk(linux).unwrap();
        assert_eq!(parsed.filesystem_identity, "/dev/nvme0n1p2");
        assert_eq!(parsed.capacity_bytes, 1_000_000 * 1024);
        assert_eq!(parsed.available_bytes, 750_000 * 1024);
        assert_eq!(parsed.mount_point, "/models");

        let spaced_mount = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/disk3s1 2000000 500000 1500000 25% /Volumes/Fast Models\n";
        assert_eq!(parse_df_pk(spaced_mount).unwrap().mount_point, "/Volumes/Fast Models");
    }

    #[test]
    fn volatile_free_space_does_not_change_stable_resource_fingerprint() {
        let machine = apple_machine(Some(16 << 30)).resource_profile();
        let low = storage(10_000);
        let high = storage(900_000);
        assert_eq!(
            machine.fingerprint_with_storage(&[low]),
            machine.fingerprint_with_storage(&[high])
        );
    }

    #[test]
    fn volatile_free_space_does_change_optimizer_storage_budget() {
        let machine = apple_machine(Some(16 << 30)).resource_profile();
        let low = machine
            .resource_budget(&[storage(10_000)], &BTreeMap::new())
            .unwrap();
        let high = machine
            .resource_budget(&[storage(900_000)], &BTreeMap::new())
            .unwrap();
        assert_eq!(low.storage_pools[0].available_bytes, 10_000);
        assert_eq!(high.storage_pools[0].available_bytes, 900_000);
    }

    #[test]
    fn memory_limit_can_reduce_but_never_inflate_physical_capacity() {
        let machine = apple_machine(Some(16 << 30)).resource_profile();
        let mut limits = BTreeMap::new();
        limits.insert(MemoryPoolId::new("uma0"), 12 << 30);
        let budget = machine.resource_budget(&[], &limits).unwrap();
        assert_eq!(budget.memory_pools[0].capacity_bytes, 12 << 30);

        limits.insert(MemoryPoolId::new("uma0"), 17 << 30);
        assert!(machine.resource_budget(&[], &limits).is_err());
    }

    #[test]
    fn unknown_memory_capacity_fails_instead_of_becoming_fake_hardware() {
        let machine = linux_machine(None).resource_profile();
        assert!(machine.resource_budget(&[], &BTreeMap::new()).is_err());
    }
}
