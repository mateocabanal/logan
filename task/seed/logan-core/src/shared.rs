//! Backend-neutral shared-memory descriptors for heterogeneous execution.
//!
//! Logan runs on systems where multiple execution devices may share one
//! physical allocation (Apple CPU/GPU/ANE over UMA is the motivating case).
//! The core owns only identity + visibility metadata; concrete native handles
//! remain backend-owned. This keeps `logan-core` free of Metal/CoreML/IOSurface
//! APIs while still letting the planner and scheduler reason about zero-copy
//! handoffs.

use std::collections::BTreeMap;

use crate::sched::DeviceKind;

/// Runtime-stable identity for one shared physical allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SharedAllocationId(pub u64);

/// Platform-neutral description of the backing mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedMemoryKind {
    /// Ordinary host allocation. CPU-visible; accelerators may import it on a
    /// platform-specific zero-copy path but that is not implied here.
    Host,
    /// IOSurface / shared Apple UMA allocation.
    IoSurface,
    /// Device-owned persistent allocation whose exact API is backend-private.
    DevicePersistent,
    /// Backend extension value.
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// Device visibility for one allocation. Visibility is not synchronization:
/// an ANE->Metal handoff can be zero-copy and still require a completion fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceVisibility {
    pub cpu: bool,
    pub gpu: bool,
    pub neural: bool,
    pub io: bool,
}

impl DeviceVisibility {
    pub const CPU_ONLY: Self = Self {
        cpu: true,
        gpu: false,
        neural: false,
        io: false,
    };

    pub const APPLE_UMA: Self = Self {
        cpu: true,
        gpu: true,
        neural: true,
        io: false,
    };

    pub fn supports(self, kind: DeviceKind) -> bool {
        match kind {
            DeviceKind::Cpu => self.cpu,
            DeviceKind::Gpu => self.gpu,
            DeviceKind::Neural => self.neural,
            DeviceKind::Io => self.io,
            DeviceKind::Other => false,
        }
    }
}

/// Core-owned metadata for a backend-owned physical allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedAllocationDesc {
    pub id: SharedAllocationId,
    pub kind: SharedMemoryKind,
    pub bytes: usize,
    pub alignment: usize,
    pub visibility: DeviceVisibility,
    /// Optional backend-neutral diagnostic name.
    pub label: Option<String>,
}

impl SharedAllocationDesc {
    pub fn can_alias_between(&self, a: DeviceKind, b: DeviceKind) -> bool {
        a == b || (self.visibility.supports(a) && self.visibility.supports(b))
    }

    pub fn validate_range(&self, offset: usize, bytes: usize) -> Result<(), SharedMemoryError> {
        let end = offset
            .checked_add(bytes)
            .ok_or(SharedMemoryError::RangeOverflow)?;
        if end > self.bytes {
            return Err(SharedMemoryError::OutOfBounds {
                allocation: self.id,
                offset,
                bytes,
                allocation_bytes: self.bytes,
            });
        }
        Ok(())
    }
}

/// A tensor/value view into a shared allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedTensorView {
    pub allocation: SharedAllocationId,
    pub offset: usize,
    pub bytes: usize,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub access: AccessMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedMemoryError {
    Duplicate(SharedAllocationId),
    Unknown(SharedAllocationId),
    RangeOverflow,
    OutOfBounds {
        allocation: SharedAllocationId,
        offset: usize,
        bytes: usize,
        allocation_bytes: usize,
    },
}

impl std::fmt::Display for SharedMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate(id) => write!(f, "shared allocation {:?} is already registered", id),
            Self::Unknown(id) => write!(f, "shared allocation {:?} is not registered", id),
            Self::RangeOverflow => write!(f, "shared allocation byte range overflow"),
            Self::OutOfBounds {
                allocation,
                offset,
                bytes,
                allocation_bytes,
            } => write!(
                f,
                "shared allocation {:?} range [{offset}, {}) exceeds {allocation_bytes} bytes",
                allocation,
                offset.saturating_add(*bytes)
            ),
        }
    }
}

impl std::error::Error for SharedMemoryError {}

/// Metadata registry. Native handles are intentionally *not* stored here;
/// each backend maps `SharedAllocationId` to its own retained handle.
#[derive(Debug, Default)]
pub struct SharedAllocationRegistry {
    next_id: u64,
    allocations: BTreeMap<SharedAllocationId, SharedAllocationDesc>,
}

impl SharedAllocationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate_id(&mut self) -> SharedAllocationId {
        let id = SharedAllocationId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    pub fn register(&mut self, desc: SharedAllocationDesc) -> Result<(), SharedMemoryError> {
        if self.allocations.contains_key(&desc.id) {
            return Err(SharedMemoryError::Duplicate(desc.id));
        }
        self.next_id = self.next_id.max(desc.id.0.saturating_add(1));
        self.allocations.insert(desc.id, desc);
        Ok(())
    }

    pub fn get(&self, id: SharedAllocationId) -> Option<&SharedAllocationDesc> {
        self.allocations.get(&id)
    }

    pub fn unregister(
        &mut self,
        id: SharedAllocationId,
    ) -> Result<SharedAllocationDesc, SharedMemoryError> {
        self.allocations
            .remove(&id)
            .ok_or(SharedMemoryError::Unknown(id))
    }

    pub fn validate_view(&self, view: &SharedTensorView) -> Result<(), SharedMemoryError> {
        self.get(view.allocation)
            .ok_or(SharedMemoryError::Unknown(view.allocation))?
            .validate_range(view.offset, view.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apple_surface(id: u64, bytes: usize) -> SharedAllocationDesc {
        SharedAllocationDesc {
            id: SharedAllocationId(id),
            kind: SharedMemoryKind::IoSurface,
            bytes,
            alignment: 16 * 1024,
            visibility: DeviceVisibility::APPLE_UMA,
            label: Some("activation".into()),
        }
    }

    #[test]
    fn apple_uma_aliases_cpu_gpu_and_ane() {
        let d = apple_surface(1, 4096);
        assert!(d.can_alias_between(DeviceKind::Cpu, DeviceKind::Gpu));
        assert!(d.can_alias_between(DeviceKind::Gpu, DeviceKind::Neural));
        assert!(d.can_alias_between(DeviceKind::Cpu, DeviceKind::Neural));
        assert!(!d.can_alias_between(DeviceKind::Neural, DeviceKind::Io));
    }

    #[test]
    fn registry_validates_tensor_ranges() {
        let mut r = SharedAllocationRegistry::new();
        r.register(apple_surface(7, 1024)).unwrap();
        let ok = SharedTensorView {
            allocation: SharedAllocationId(7),
            offset: 256,
            bytes: 512,
            shape: vec![128],
            dtype: "f32".into(),
            access: AccessMode::ReadWrite,
        };
        r.validate_view(&ok).unwrap();
        let bad = SharedTensorView { bytes: 800, ..ok };
        assert!(matches!(
            r.validate_view(&bad),
            Err(SharedMemoryError::OutOfBounds { .. })
        ));
    }
}
