from pathlib import Path

# Wire the new target resource module without rewriting the large target file.
p = Path('logan-compiler/src/target/mod.rs')
s = p.read_text()
old = '''pub mod machine;

pub use machine::{MachineProfile, metal_available_for};'''
new = '''pub mod machine;
pub mod resources;

pub use machine::{MachineProfile, metal_available_for};
pub use resources::{
    CapabilitySupport, MachineMemoryPool, MachineResourceProfile, MemoryPoolKind,
    StoragePoolObservation, StoragePoolProfile, observe_storage_path,
};'''
if old not in s:
    raise SystemExit('target module export anchor missing')
p.write_text(s.replace(old, new, 1))

# Writeability is a path/user observation, not a stable storage identity fact.
p = Path('logan-compiler/src/target/resources.rs')
s = p.read_text()
replacements = [
    ('''    /// Proven by an actual create-new/delete probe in the selected directory.\n    pub writable: CapabilitySupport,\n    /// Keep capabilities unknown until Logan has actually probed or has a\n''', '''    /// Keep capabilities unknown until Logan has actually probed or has a\n'''),
    ('''pub struct StoragePoolObservation {\n    pub profile: StoragePoolProfile,\n    /// Volatile planner observation. This is checked again before allocation\n    /// and must not be part of the stable execution ABI.\n    pub available_bytes: u64,\n''', '''pub struct StoragePoolObservation {\n    pub profile: StoragePoolProfile,\n    /// Volatile planner observation. This is checked again before allocation\n    /// and must not be part of the stable execution ABI.\n    pub available_bytes: u64,\n    /// Proven by an actual create-new/delete probe in the selected directory.\n    /// This can change with user/ACL state, so it is deliberately excluded\n    /// from the stable storage fingerprint.\n    pub writable: CapabilitySupport,\n'''),
    ('''                writable: observation.profile.writable.is_supported(),\n''', '''                writable: observation.writable.is_supported(),\n'''),
    ('''                "storage={}:{}:{}:{}:{}:{}:{}:{}\\n",\n                profile.id.0,\n                profile.filesystem_identity,\n                profile.mount_point,\n                profile.capacity_bytes,\n                profile.writable.as_str(),\n                profile.pageable_mapping.as_str(),\n                profile.sparse_files.as_str(),\n                profile.direct_io.as_str(),\n''', '''                "storage={}:{}:{}:{}:{}:{}:{}\\n",\n                profile.id.0,\n                profile.filesystem_identity,\n                profile.mount_point,\n                profile.capacity_bytes,\n                profile.pageable_mapping.as_str(),\n                profile.sparse_files.as_str(),\n                profile.direct_io.as_str(),\n'''),
    ('''            capacity_bytes: row.capacity_bytes,\n            writable,\n            pageable_mapping: CapabilitySupport::Unknown,\n''', '''            capacity_bytes: row.capacity_bytes,\n            pageable_mapping: CapabilitySupport::Unknown,\n'''),
    ('''        available_bytes: row.available_bytes,\n        probe_directory: directory,\n''', '''        available_bytes: row.available_bytes,\n        writable,\n        probe_directory: directory,\n'''),
    ('''                capacity_bytes: 1_000_000,\n                writable: CapabilitySupport::Supported,\n                pageable_mapping: CapabilitySupport::Unknown,\n''', '''                capacity_bytes: 1_000_000,\n                pageable_mapping: CapabilitySupport::Unknown,\n'''),
    ('''            available_bytes,\n            probe_directory: PathBuf::from("/models"),\n''', '''            available_bytes,\n            writable: CapabilitySupport::Supported,\n            probe_directory: PathBuf::from("/models"),\n'''),
]
for old, new in replacements:
    if old not in s:
        raise SystemExit(f'resource identity anchor missing: {old[:80]!r}')
    s = s.replace(old, new, 1)

anchor = '''    #[test]\n    fn volatile_free_space_does_change_optimizer_storage_budget() {'''
extra = '''    #[test]\n    fn current_write_permission_is_not_part_of_stable_storage_fingerprint() {\n        let machine = apple_machine(Some(16 << 30)).resource_profile();\n        let writable = storage(100_000);\n        let mut readonly = writable.clone();\n        readonly.writable = CapabilitySupport::Unsupported;\n        assert_eq!(\n            machine.fingerprint_with_storage(&[writable]),\n            machine.fingerprint_with_storage(&[readonly.clone()])\n        );\n        let budget = machine\n            .resource_budget(&[readonly], &BTreeMap::new())\n            .unwrap();\n        assert!(!budget.storage_pools[0].writable);\n    }\n\n'''
if anchor not in s:
    raise SystemExit('writeability regression insertion anchor missing')
s = s.replace(anchor, extra + anchor, 1)
p.write_text(s)
