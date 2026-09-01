//! Device registry and execution targets for issue #56.
//!
//! Pure identity/capability model: no platform backends (#57/#58), no native
//! APIs, no memory-budget math (that is the residency manager's job). One
//! `DeviceId` identifies one logical execution device and is shared with
//! `residency`, so a registered target and a residency-registered device are
//! the same identity by construction. Devices are never memory pools; pool
//! ownership lives in `ResidencyManager`.

use std::collections::HashMap;

use super::core::ActionKind;
pub use super::residency::DeviceId;

/// Broad execution-device class. Illustrative, not ABI-final (#56).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Neural,
    Io,
    Other,
}

/// A concrete compiler-approved execution choice attached to work: one device
/// plus the action classes it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionTarget {
    pub device: DeviceId,
    pub kind: DeviceKind,
    /// Action classes this target accepts.
    pub capabilities: Vec<ActionKind>,
}

impl ExecutionTarget {
    pub fn accepts(&self, kind: ActionKind) -> bool {
        self.capabilities.contains(&kind)
    }
}

/// Invalid registry operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceError {
    DuplicateDevice(DeviceId),
    UnknownDevice(DeviceId),
}

/// Runtime-owned registry of execution targets. Targets register and
/// unregister explicitly; the scheduler enumerates targets to pick where an
/// opaque action may run.
#[derive(Debug, Clone, Default)]
pub struct DeviceRegistry {
    targets: HashMap<DeviceId, ExecutionTarget>,
}

impl DeviceRegistry {
    pub fn new() -> Self {
        Self {
            targets: HashMap::new(),
        }
    }

    pub fn register(&mut self, target: ExecutionTarget) -> Result<(), DeviceError> {
        if self.targets.contains_key(&target.device) {
            return Err(DeviceError::DuplicateDevice(target.device));
        }
        self.targets.insert(target.device, target);
        Ok(())
    }

    pub fn unregister(&mut self, device: DeviceId) -> Result<(), DeviceError> {
        if self.targets.remove(&device).is_none() {
            return Err(DeviceError::UnknownDevice(device));
        }
        Ok(())
    }

    pub fn get(&self, device: DeviceId) -> Option<&ExecutionTarget> {
        self.targets.get(&device)
    }

    /// Every registered target in deterministic ascending `DeviceId` order.
    pub fn targets(&self) -> Vec<&ExecutionTarget> {
        let mut targets: Vec<_> = self.targets.values().collect();
        targets.sort_unstable_by_key(|target| target.device);
        targets
    }

    /// First target accepting `kind`, in ascending `DeviceId` order.
    pub fn find_capable(&self, kind: ActionKind) -> Option<&ExecutionTarget> {
        self.targets()
            .into_iter()
            .find(|target| target.accepts(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(device: u32, kind: DeviceKind, capabilities: Vec<ActionKind>) -> ExecutionTarget {
        ExecutionTarget {
            device: DeviceId(device),
            kind,
            capabilities,
        }
    }

    #[test]
    fn register_get_unregister_round_trip() {
        let mut registry = DeviceRegistry::new();
        let cpu = target(0, DeviceKind::Cpu, vec![ActionKind::Cpu]);
        registry.register(cpu.clone()).unwrap();
        assert_eq!(registry.get(DeviceId(0)), Some(&cpu));
        assert_eq!(
            registry.register(cpu),
            Err(DeviceError::DuplicateDevice(DeviceId(0)))
        );
        assert_eq!(
            registry.unregister(DeviceId(9)),
            Err(DeviceError::UnknownDevice(DeviceId(9)))
        );
        registry.unregister(DeviceId(0)).unwrap();
        assert_eq!(registry.get(DeviceId(0)), None);
    }

    #[test]
    fn enumeration_is_deterministic_by_device_id() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(target(2, DeviceKind::Gpu, vec![ActionKind::Accelerator]))
            .unwrap();
        registry
            .register(target(0, DeviceKind::Cpu, vec![ActionKind::Cpu]))
            .unwrap();
        registry
            .register(target(1, DeviceKind::Io, vec![ActionKind::Io]))
            .unwrap();
        let ids: Vec<_> = registry.targets().iter().map(|t| t.device).collect();
        assert_eq!(ids, vec![DeviceId(0), DeviceId(1), DeviceId(2)]);
    }

    #[test]
    fn caps_route_by_accepted_action_kind() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(target(5, DeviceKind::Gpu, vec![ActionKind::Cpu, ActionKind::Accelerator]))
            .unwrap();
        registry
            .register(target(3, DeviceKind::Cpu, vec![ActionKind::Cpu]))
            .unwrap();
        // Ascending id order: CPU-only device wins the CPU route.
        assert_eq!(registry.find_capable(ActionKind::Cpu).unwrap().device, DeviceId(3));
        assert_eq!(
            registry.find_capable(ActionKind::Accelerator).unwrap().device,
            DeviceId(5)
        );
        assert_eq!(registry.find_capable(ActionKind::Io), None);
        // Unregister removes the route.
        registry.unregister(DeviceId(3)).unwrap();
        assert_eq!(registry.find_capable(ActionKind::Cpu).unwrap().device, DeviceId(5));
    }

    #[test]
    fn target_accepts_checks_membership() {
        let gpu = target(1, DeviceKind::Gpu, vec![ActionKind::Accelerator]);
        assert!(gpu.accepts(ActionKind::Accelerator));
        assert!(!gpu.accepts(ActionKind::Cpu));
    }
}