use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::rc::Rc;

use crate::raw;
use crate::{AneError, Result};
use logan_core::shared::{
    DeviceVisibility, SharedAllocationDesc, SharedAllocationId, SharedMemoryKind,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "logan_ane_async", kind = "static")]
unsafe extern "C" {
    fn logan_ane_surface_pack_bf16_transposed_f32(
        raw_surface: *mut c_void, total_spatial: usize, weight_offset: usize,
        src: *const u16, in_features: usize, out_features: usize,
    ) -> i32;
    fn logan_ane_surface_write_repeated_f32(
        raw_surface: *mut c_void, total_spatial: usize, token_spatial: usize,
        x: *const f32, in_features: usize,
    ) -> i32;
}


/// IOSurface-backed memory that can be shared with ANE without an extra copy.
pub struct AneSurface {
    raw: raw::IOSurfaceRef,
    logical_bytes: usize,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AneSurface {
    pub fn new(bytes: usize) -> Result<Self> {
        raw::ensure_frameworks()?;
        let surface = raw::create_surface(bytes)?;
        let allocated = raw::surface_alloc_size(surface);
        if allocated < bytes {
            raw::release_surface(surface);
            return Err(AneError::InvalidArgument(format!(
                "IOSurface allocated {allocated} bytes for a {bytes}-byte request"
            )));
        }
        Ok(Self {
            raw: surface,
            logical_bytes: bytes,
            _not_send_sync: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.logical_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.logical_bytes == 0
    }

    pub fn allocated_len(&self) -> usize {
        raw::surface_alloc_size(self.raw)
    }

    /// Process-global IOSurface identifier, useful for diagnostics and for
    /// correlating backend registrations. Ownership is still tied to `self`.
    pub fn iosurface_id(&self) -> u32 {
        raw::surface_id(self.raw)
    }

    /// Describe this IOSurface in Logan's backend-neutral shared-memory model.
    /// Native handle ownership remains in `AneSurface`; the descriptor only
    /// carries identity/visibility metadata for planning and scheduling.
    pub fn shared_allocation_desc(
        &self,
        id: SharedAllocationId,
        label: Option<String>,
    ) -> SharedAllocationDesc {
        SharedAllocationDesc {
            id,
            kind: SharedMemoryKind::IoSurface,
            bytes: self.allocated_len(),
            // The core treats this as a minimum guarantee. Individual backends
            // perform their own native alignment checks before importing.
            alignment: 1,
            visibility: DeviceVisibility::APPLE_UMA,
            label,
        }
    }

    pub fn read(&self) -> Result<SurfaceRead<'_>> {
        let ptr = raw::lock_surface(self.raw, true)?;
        Ok(SurfaceRead {
            surface: self,
            ptr,
            len: self.logical_bytes,
        })
    }

    pub fn write(&mut self) -> Result<SurfaceWrite<'_>> {
        let ptr = raw::lock_surface(self.raw, false)?;
        let len = self.logical_bytes;
        Ok(SurfaceWrite {
            surface: self,
            ptr,
            len,
        })
    }

    pub fn write_f32(&mut self, values: &[f32]) -> Result<()> {
        let want = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| AneError::InvalidArgument("f32 surface byte count overflow".into()))?;
        if want != self.logical_bytes {
            return Err(AneError::InvalidArgument(format!(
                "f32 payload is {want} bytes but surface is {} bytes",
                self.logical_bytes
            )));
        }
        let mut map = self.write()?;
        for (chunk, value) in map.chunks_exact_mut(4).zip(values.iter().copied()) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    pub fn read_f32(&self) -> Result<Vec<f32>> {
        if self.logical_bytes % 4 != 0 {
            return Err(AneError::InvalidArgument(format!(
                "surface byte size {} is not divisible by sizeof(f32)",
                self.logical_bytes
            )));
        }
        let map = self.read()?;
        Ok(map
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Blocked NEON BF16 [O,I] -> fp32 W^T [I,O] pack into a dynamic-weight region.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn pack_bf16_transposed_f32(
        &mut self, total_spatial: usize, weight_offset: usize,
        weights_bf16: &[u8], in_features: usize, out_features: usize,
    ) -> Result<()> {
        let want = in_features.checked_mul(out_features).and_then(|n| n.checked_mul(2))
            .ok_or_else(|| AneError::InvalidArgument("BF16 pack shape overflow".into()))?;
        if weights_bf16.len() != want || weights_bf16.as_ptr() as usize % 2 != 0 {
            return Err(AneError::InvalidArgument("BF16 pack byte size/alignment mismatch".into()));
        }
        let ok = unsafe { logan_ane_surface_pack_bf16_transposed_f32(
            self.raw.cast(), total_spatial, weight_offset, weights_bf16.as_ptr().cast(),
            in_features, out_features) };
        if ok == 1 { Ok(()) } else { Err(AneError::Surface { operation: "pack BF16 transpose", code: ok }) }
    }

    /// Update only the repeated decode-token tile in a packed dynamic surface.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn write_repeated_f32(
        &mut self, total_spatial: usize, token_spatial: usize, x: &[f32],
    ) -> Result<()> {
        let ok = unsafe { logan_ane_surface_write_repeated_f32(
            self.raw.cast(), total_spatial, token_spatial, x.as_ptr(), x.len()) };
        if ok == 1 { Ok(()) } else { Err(AneError::Surface { operation: "write repeated f32", code: ok }) }
    }

    /// Borrow the underlying `IOSurfaceRef` for interoperability with Metal,
    /// CoreVideo, or other low-level Apple APIs.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while `self` is alive. The caller
    /// must not release it or violate IOSurface locking/aliasing rules.
    pub unsafe fn as_raw_iosurface(&self) -> *mut c_void {
        self.raw
    }

    pub(crate) fn raw_surface(&self) -> raw::IOSurfaceRef {
        self.raw
    }
}

impl Drop for AneSurface {
    fn drop(&mut self) {
        raw::release_surface(self.raw);
    }
}

pub struct SurfaceRead<'a> {
    surface: &'a AneSurface,
    ptr: *mut u8,
    len: usize,
}

impl Deref for SurfaceRead<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for SurfaceRead<'_> {
    fn drop(&mut self) {
        raw::unlock_surface(self.surface.raw, true);
    }
}

pub struct SurfaceWrite<'a> {
    surface: &'a mut AneSurface,
    ptr: *mut u8,
    len: usize,
}

impl Deref for SurfaceWrite<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for SurfaceWrite<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for SurfaceWrite<'_> {
    fn drop(&mut self) {
        raw::unlock_surface(self.surface.raw, false);
    }
}
