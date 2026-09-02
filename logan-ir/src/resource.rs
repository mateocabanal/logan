use serde::{Deserialize, Serialize};

/// Stable logical identifier for one memory-capacity domain used by a plan.
///
/// Device identity is deliberately separate: Apple CPU/Metal/ANE may share
/// one UMA pool, while discrete GPUs have independent VRAM pools.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryPoolId(pub String);

impl MemoryPoolId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Stable logical identifier for one backing-storage domain used by a plan.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoragePoolId(pub String);

impl StoragePoolId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataMutability {
    Immutable,
    Mutable,
}

/// Where the authoritative bytes live when they are not resident in a fast
/// memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackingKind {
    /// Immutable bytes already stored in the COLI package. This does not imply
    /// an additional runtime spill allocation.
    PackageRecord,
    /// Mutable runtime state (KV/QSA/etc.) backed by a writable state file.
    RuntimeStateFile,
    /// Small state/scratch that has no legal out-of-core representation and
    /// therefore contributes to the irreducible resident working set.
    ResidentOnly,
    /// Backend-owned persistent storage, only for runtimes that explicitly
    /// expose such a capacity domain.
    DevicePersistent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackingPlan {
    pub kind: BackingKind,
    pub storage_pool: Option<StoragePoolId>,
    /// Logical/physical bytes requiring this backing. Package-backed bytes are
    /// already present in the package and are not duplicate mutable spill.
    pub bytes: u64,
    pub alignment: u64,
    pub page_or_block_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyPlan {
    pub memory_pool: MemoryPoolId,
    /// True hard memory requirement to make forward progress. This is the
    /// capacity check; total logical/resource bytes are not.
    pub minimum_working_set_bytes: u64,
    /// Compiler-selected hot/cache target for performance. This is bounded by
    /// the pool budget but may be much smaller than the backed resource.
    pub target_resident_bytes: u64,
    pub pinned_bytes: u64,
    /// Lower values are cheaper eviction victims. Exact policy is runtime
    /// specific but the ordering is compiler-approved and persisted.
    pub eviction_priority: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessKind {
    DirectShared,
    Mapped,
    AsyncStream,
    Staged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessPlan {
    pub kind: AccessKind,
    pub prefetch_depth: u16,
    pub expected_read_bytes_per_step: u64,
    pub expected_write_bytes_per_step: u64,
}

/// Model-neutral physical resource contract. Representation/layout and
/// execution target remain separate compiler decisions; this struct only
/// describes backing, fast-tier residency and the access path between them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePlan {
    pub mutability: DataMutability,
    pub backing: BackingPlan,
    pub residency: ResidencyPlan,
    pub access: AccessPlan,
}

impl ResourcePlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.residency.minimum_working_set_bytes > self.residency.target_resident_bytes {
            return Err("minimum working set exceeds target resident bytes".into());
        }
        if self.residency.pinned_bytes > self.residency.target_resident_bytes {
            return Err("pinned bytes exceed target resident bytes".into());
        }
        if self.backing.alignment == 0 {
            return Err("backing alignment must be non-zero".into());
        }
        if self.backing.page_or_block_bytes == 0 {
            return Err("backing page/block size must be non-zero".into());
        }
        match (self.mutability, self.backing.kind) {
            (DataMutability::Mutable, BackingKind::PackageRecord) => {
                return Err("mutable state cannot use immutable package-record backing".into());
            }
            (DataMutability::Immutable, BackingKind::RuntimeStateFile) => {
                return Err("immutable package data should not require mutable state-file backing".into());
            }
            _ => {}
        }
        match self.backing.kind {
            BackingKind::PackageRecord | BackingKind::ResidentOnly => {
                if self.backing.storage_pool.is_some() {
                    return Err("package/resident-only backing must not bind a runtime storage pool".into());
                }
            }
            BackingKind::RuntimeStateFile | BackingKind::DevicePersistent => {
                if self.backing.storage_pool.is_none() {
                    return Err("runtime/device backing requires a storage-pool id".into());
                }
            }
        }
        Ok(())
    }

    pub fn immutable_package(
        bytes: u64,
        memory_pool: MemoryPoolId,
        minimum_working_set_bytes: u64,
        target_resident_bytes: u64,
        read_bytes_per_step: u64,
    ) -> Self {
        Self {
            mutability: DataMutability::Immutable,
            backing: BackingPlan {
                kind: BackingKind::PackageRecord,
                storage_pool: None,
                bytes,
                alignment: 1,
                page_or_block_bytes: 1,
            },
            residency: ResidencyPlan {
                memory_pool,
                minimum_working_set_bytes,
                target_resident_bytes,
                pinned_bytes: minimum_working_set_bytes,
                eviction_priority: 100,
            },
            access: AccessPlan {
                kind: AccessKind::AsyncStream,
                prefetch_depth: 1,
                expected_read_bytes_per_step: read_bytes_per_step,
                expected_write_bytes_per_step: 0,
            },
        }
    }

    pub fn mutable_file_backed(
        bytes: u64,
        storage_pool: StoragePoolId,
        memory_pool: MemoryPoolId,
        minimum_working_set_bytes: u64,
        target_resident_bytes: u64,
        block_bytes: u64,
        read_bytes_per_step: u64,
        write_bytes_per_step: u64,
    ) -> Self {
        Self {
            mutability: DataMutability::Mutable,
            backing: BackingPlan {
                kind: BackingKind::RuntimeStateFile,
                storage_pool: Some(storage_pool),
                bytes,
                alignment: block_bytes.max(1),
                page_or_block_bytes: block_bytes.max(1),
            },
            residency: ResidencyPlan {
                memory_pool,
                minimum_working_set_bytes,
                target_resident_bytes,
                pinned_bytes: minimum_working_set_bytes,
                eviction_priority: 80,
            },
            access: AccessPlan {
                kind: AccessKind::AsyncStream,
                prefetch_depth: 1,
                expected_read_bytes_per_step: read_bytes_per_step,
                expected_write_bytes_per_step: write_bytes_per_step,
            },
        }
    }

    pub fn mutable_backing_bytes(&self) -> u64 {
        if self.backing.kind == BackingKind::RuntimeStateFile {
            self.backing.bytes
        } else {
            0
        }
    }

    pub fn immutable_package_backing_bytes(&self) -> u64 {
        if self.backing.kind == BackingKind::PackageRecord {
            self.backing.bytes
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_state_can_exceed_ram_when_resident_target_is_bounded() {
        let plan = ResourcePlan::mutable_file_backed(
            21 * 1024,
            StoragePoolId::new("ssd0"),
            MemoryPoolId::new("uma0"),
            512,
            12 * 1024,
            4096,
            9 * 1024,
            64,
        );
        assert!(plan.validate().is_ok());
        assert_eq!(plan.backing.bytes, 21 * 1024);
        assert_eq!(plan.residency.target_resident_bytes, 12 * 1024);
        assert!(plan.backing.bytes > plan.residency.target_resident_bytes);
    }

    #[test]
    fn immutable_package_backing_does_not_become_mutable_spill() {
        let plan = ResourcePlan::immutable_package(
            100 * 1024,
            MemoryPoolId::new("uma0"),
            1024,
            8 * 1024,
            4 * 1024,
        );
        assert!(plan.validate().is_ok());
        assert_eq!(plan.mutable_backing_bytes(), 0);
        assert_eq!(plan.immutable_package_backing_bytes(), 100 * 1024);
    }

    #[test]
    fn irreducible_working_set_cannot_exceed_resident_target() {
        let mut plan = ResourcePlan::mutable_file_backed(
            32 * 1024,
            StoragePoolId::new("ssd0"),
            MemoryPoolId::new("uma0"),
            1024,
            4 * 1024,
            4096,
            8 * 1024,
            64,
        );
        plan.residency.minimum_working_set_bytes = 5 * 1024;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn mutable_state_requires_writable_runtime_backing_identity() {
        let mut plan = ResourcePlan::mutable_file_backed(
            4096,
            StoragePoolId::new("ssd0"),
            MemoryPoolId::new("uma0"),
            512,
            1024,
            4096,
            512,
            64,
        );
        plan.backing.storage_pool = None;
        assert!(plan.validate().is_err());
    }
}
