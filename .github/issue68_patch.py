from pathlib import Path

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
