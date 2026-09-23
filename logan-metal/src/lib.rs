//! Logan Metal backend: FFI to the proven C Metal stack
//! (backend_metal.mm / metalio.mm / apple8_metalio_direct.mm), engine-neutral.
//!
//! Every entry point returns 0/None on decline; the caller falls back to the
//! CPU reference path (Metal is never mandatory, matching the C engine's
//! fallback contract). Non-macOS builds get stub decliners.
//! Only the ops we measured as wins: quantized expert GEMV (fmt 7 = MXFP4,
//! byte-compatible with Apple8 tiles) + the small dense ops.
//!
//! ALSO the direct Apple8 execution seam (apple8_metalio_direct.mm):
//! slot-resident expert GEMV/SwiGLU, the fused one-command-buffer moe_topk
//! (with begin/finish split phase so CPU work overlaps the routed-GPU wait),
//! and the coalesced Metal GDN kernels. Every entry point returns 0/None on
//! decline; the caller falls back to the CPU reference path (Metal is never
//! mandatory, matching the C engine's fallback contract).
//!
//! Unsafe at the boundary only; every call checks the return code and
//! falls back to CPU on failure (Metal unavailable, invalid fmt, ...).

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct ColiMetalTensor {
        _private: [u8; 0],
    }

    /// Persistent zero-copy Metal view of an externally owned IOSurface.
    pub struct MetalSharedSurface {
        handle: *mut c_void,
        logical_bytes: usize,
        allocation_bytes: usize,
        _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    /// Persistent Metal continuation for an ANE-produced GDN qkv IOSurface.
    /// Lanes 0..kernel-2 contain causal history while the final spatial lane
    /// contains the current qkv projection. One dispatch computes Conv1D+SiLU.
    pub struct MetalGdnConvSilu {
        handle: *mut c_void,
        _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    /// One device-visible event shared by ANE and Metal. Metal owns the
    /// `MTLSharedEvent`; `ane_shared_event` is its private IOSurfaceSharedEvent
    /// backing, borrowed by the ANE request while this handle lives.
    pub struct MetalAneFence {
        handle: *mut c_void,
        ane_shared_event: *mut c_void,
        value: u64,
        _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    /// Reusable Metal packer for the model-global dynamic-weight ANE GDN path.
    /// It zero-copy wraps one layer's aligned BF16 qkv/z/a/b matrices and
    /// writes the transposed fp32 packed layout directly into ANE IOSurfaces.
    pub struct MetalAneDynamicPack {
        handle: *mut c_void,
        hidden: usize,
        _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    pub struct MetalAneDynamicPackPending {
        raw: Option<std::ptr::NonNull<c_void>>,
        _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    impl MetalAneFence {
        pub fn new(value: u64) -> Option<Self> {
            if value == 0 {
                return None;
            }
            let mut shared = std::ptr::null_mut();
            let handle = unsafe { coli_apple8_metalio_ane_fence_create(value, &mut shared) };
            if handle.is_null() || shared.is_null() {
                if !handle.is_null() {
                    unsafe { coli_apple8_metalio_ane_fence_free(handle) };
                }
                return None;
            }
            Some(Self {
                handle,
                ane_shared_event: shared,
                value,
                _not_send_sync: std::marker::PhantomData,
            })
        }

        pub fn ane_shared_event(&self) -> *mut c_void {
            self.ane_shared_event
        }
        /// Borrow the underlying MTLSharedEvent for Metal command encoding.
        /// Lifetime is tied to this fence; native pending objects retain it.
        pub fn metal_shared_event(&self) -> *mut c_void {
            unsafe { coli_apple8_metalio_ane_fence_metal_event(self.handle) }
        }
        pub fn value(&self) -> u64 {
            self.value
        }
        pub fn advance(&mut self) -> Option<u64> {
            let next = self.value.checked_add(1)?;
            if unsafe { coli_apple8_metalio_ane_fence_set_value(self.handle, next) } != 1 {
                return None;
            }
            self.value = next;
            Some(next)
        }
        fn raw_handle(&self) -> *mut c_void {
            self.handle
        }
    }

    impl Drop for MetalAneFence {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { coli_apple8_metalio_ane_fence_free(self.handle) };
                self.handle = std::ptr::null_mut();
            }
        }
    }

    #[repr(C)]
    struct ColiMetalMatmulDescRaw {
        tensor: *mut ColiMetalTensor,
        y: *mut f32,
        weights: *const c_void,
        scales: *const f32,
        fmt: i32,
        i: i32,
        o: i32,
        gs: i32,
        /// Per-descriptor activation. Null means "use the function-level shared
        /// `x`", which is what every caller other than the per-descriptor
        /// activation path wants.
        x: *const f32,
        /// Leading dimension for `x` when `x` is non-null.
        s: i32,
    }

    impl Default for ColiMetalMatmulDescRaw {
        fn default() -> Self {
            Self {
                tensor: std::ptr::null_mut(),
                y: std::ptr::null_mut(),
                weights: std::ptr::null(),
                scales: std::ptr::null(),
                fmt: 0,
                i: 0,
                o: 0,
                gs: 0,
                x: std::ptr::null(),
                // 1, not 0: the C side rejects a private-activation descriptor
                // with a non-positive leading dimension.
                s: 1,
            }
        }
    }

    pub struct MetalMatmulDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub y: &'a mut [f32],
        pub weights: &'a [u8],
        pub scales: &'a [u8],
        pub fmt: i32,
        pub i: usize,
        pub o: usize,
    }

    pub struct MlxAffineMatmulDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub y: &'a mut [f32],
        pub weights: &'a [u8],
        pub aux: &'a [u8],
        pub bits: u8,
        pub group_size: usize,
        /// True when MLX affine scale/bias sidecars are IEEE fp16 rather than BF16.
        pub aux_fp16: bool,
        pub i: usize,
        pub o: usize,
        /// Per-descriptor activation. `None` consumes the shared activation
        /// passed to `metal_matmul_mlx_affine_multi`; `Some(v)` lets a
        /// descriptor whose input differs (for example a routed expert's own
        /// SwiGLU output) join the same command buffer. `v.len() >= i`.
        pub x: Option<&'a [f32]>,
    }

    pub struct MetalWeightDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub weights: &'a [u8],
        pub scales: &'a [u8],
        pub fmt: i32,
        pub group_size: usize,
        pub i: usize,
        pub o: usize,
    }

    unsafe extern "C" {
        fn logan_metal_shared_surface_wrap(
            raw_surface: *mut c_void,
            logical_bytes: usize,
        ) -> *mut c_void;
        fn logan_metal_shared_surface_free(handle: *mut c_void);
        fn logan_metal_shared_surface_contents(handle: *mut c_void) -> *mut c_void;
        fn logan_metal_shared_surface_length(handle: *mut c_void) -> usize;
        fn logan_metal_shared_surface_allocation_length(handle: *mut c_void) -> usize;
        fn logan_metal_ane_dynamic_pack_create(
            qkv_surface: *mut c_void,
            aux_surface: *mut c_void,
            wqkv: *const u16,
            wz: *const u16,
            wa: *const u16,
            wb: *const u16,
            hidden: u32,
            spatial: u32,
            qkv_rows: u32,
            z_rows: u32,
            ab_rows: u32,
            qkv_stride: u32,
            aux_stride: u32,
            qkv_offset: u32,
            z_offset: u32,
            a_offset: u32,
            b_offset: u32,
        ) -> *mut c_void;
        fn logan_metal_ane_dynamic_pack_run(
            handle: *mut c_void,
            x: *const f32,
            gpu_ms: *mut f64,
        ) -> i32;
        fn logan_metal_ane_dynamic_pack_begin(
            handle: *mut c_void,
            x: *const f32,
            signal_event: *mut c_void,
            signal_value: u64,
        ) -> *mut c_void;
        fn logan_metal_ane_dynamic_pack_finish(pending: *mut c_void, gpu_ms: *mut f64) -> i32;
        fn logan_metal_ane_dynamic_pack_discard(pending: *mut c_void);
        fn logan_metal_ane_dynamic_pack_free(handle: *mut c_void);
        fn logan_metal_gdn_conv_silu_create(
            input_handle: *mut c_void,
            output_handle: *mut c_void,
            weights: *const f32,
            channels: usize,
            spatial: usize,
            kernel: usize,
        ) -> *mut c_void;
        fn logan_metal_gdn_conv_silu_run(handle: *mut c_void) -> i32;
        fn logan_metal_gdn_conv_silu_begin(handle: *mut c_void) -> *mut c_void;
        fn logan_metal_gdn_conv_silu_finish(handle: *mut c_void, gpu_ms: *mut f64) -> i32;
        fn logan_metal_gdn_conv_silu_free(handle: *mut c_void);

        pub fn coli_metal_init() -> i32;
        pub fn coli_metal_available() -> i32;
        fn coli_metal_profile_set_on(on: i32);
        fn coli_metal_profile_reset();
        fn coli_metal_profile_get(
            encode: *mut u64,
            submit: *mut u64,
            wait: *mut u64,
            kernel: *mut u64,
        );
        pub fn coli_metal_matmul(
            tensor: *mut *mut ColiMetalTensor,
            y: *mut f32,
            x: *const f32,
            weights: *const c_void,
            scales: *const f32,
            fmt: i32,
            s: i32,
            i: i32,
            o: i32,
            gs: i32,
        ) -> i32;
        fn coli_metal_wrap_stats(calls: *mut u64, zero_copy: *mut u64, copy_bytes: *mut u64, bytes_created: *mut u64);
        fn coli_metal_matmul_multi(
            x: *const f32,
            s: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
        ) -> i32;
        fn coli_metal_llama_layer(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            x: *mut f32,
            k_out: *mut f32,
            v_out: *mut f32,
            input_norm: *const f32,
            post_norm: *const f32,
            d: i32,
            inter: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            pos: i32,
            theta: f32,
            eps: f32,
        ) -> i32;
        fn coli_metal_llama_drop_model(model_id: u64);
        fn coli_metal_gdn_mxfp4(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            x: *const f32,
            out: *mut f32,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_w: *const f32,
            norm_w: *const f32,
            state: *mut f32,
            conv_state: *mut f32,
            d: i32,
            kheads: i32,
            kd: i32,
            vheads: i32,
            vd: i32,
            kk: i32,
            output_gate: i32,
            eps: f32,
        ) -> i32;
        fn coli_metal_gdn_mxfp4_drop_model(model_id: u64);
        fn coli_metal_hc_mix(
            model_id: u64,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            normed: *const f32,
            out: *mut f32,
            inject: *mut f32,
            d: i32,
            hc: i32,
            lr: i32,
        ) -> i32;
        fn coli_metal_hc_drop_model(model_id: u64);
        fn coli_metal_spark_layer(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            x: *mut f32,
            input_norm: *const f32,
            post_norm: *const f32,
            d: i32,
            inter: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            sliding: i32,
            window: i32,
            pos: i32,
            rotary_dim: i32,
            theta: f32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_token_begin(model_id: u64, x: *const f32, d: i32, pos: i32) -> i32;
        fn coli_metal_spark_layer_encode(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            input_norm: *const f32,
            post_norm: *const f32,
            d: i32,
            inter: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            sliding: i32,
            window: i32,
            pos: i32,
            rotary_dim: i32,
            theta: f32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_token_end(model_id: u64, x: *mut f32, d: i32, pos: i32) -> i32;
        fn coli_metal_spark_token_end_top1(
            model_id: u64,
            head: *mut ColiMetalMatmulDescRaw,
            norm: *const f32,
            token: *mut u32,
            d: i32,
            vocab: i32,
            pos: i32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_token_end_logits(
            model_id: u64,
            head: *mut ColiMetalMatmulDescRaw,
            norm: *const f32,
            logits: *mut f32,
            d: i32,
            vocab: i32,
            pos: i32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_token_abort(model_id: u64);
        fn coli_metal_spark_prefill_begin(
            model_id: u64,
            x: *const f32,
            s: i32,
            d: i32,
            base: i32,
        ) -> i32;
        fn coli_metal_spark_prefill_layer_encode(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            input_norm: *const f32,
            post_norm: *const f32,
            d: i32,
            inter: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            sliding: i32,
            window: i32,
            base: i32,
            s: i32,
            rotary_dim: i32,
            theta: f32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_prefill_end(model_id: u64, base: i32, s: i32) -> i32;
        fn coli_metal_spark_prefill_end_logits(
            model_id: u64,
            head: *mut ColiMetalMatmulDescRaw,
            norm: *const f32,
            logits: *mut f32,
            d: i32,
            vocab: i32,
            base: i32,
            s: i32,
            eps: f32,
        ) -> i32;
        fn coli_metal_spark_prefill_abort(model_id: u64);
        fn coli_metal_spark_drop_model(model_id: u64);
        fn coli_metal_shared_mxfp4(
            model_id: u64,
            layer: i32,
            descs: *mut ColiMetalMatmulDescRaw,
            count: i32,
            x: *const f32,
            out: *mut f32,
            d: i32,
            iinter: i32,
        ) -> i32;
        fn coli_metal_shared_mxfp4_drop_model(model_id: u64);
        pub fn coli_metal_rmsnorm(x: *mut f32, w: *const f32, n: i32, nrows: i32, eps: f32) -> i32;
        pub fn coli_metal_add(y: *mut f32, a: *const f32, n: i32) -> i32;
        pub fn coli_metal_silu_mul(g: *mut f32, u: *const f32, n: i32) -> i32;
        pub fn coli_metal_tensor_free(tensor: *mut ColiMetalTensor);
        pub fn coli_metal_shutdown();
        pub fn coli_bnns_bf16_matmul(
            w: *const u16,
            x: *const f32,
            y: *mut f32,
            o: i32,
            i: i32,
        ) -> i32;
        pub fn coli_bnns_bf16_matmul_batch(
            w: *const u16,
            x: *const f32,
            y: *mut f32,
            s: i32,
            o: i32,
            i: i32,
        ) -> i32;
    }

    impl MetalSharedSurface {
        /// Import an IOSurface into Metal without copying its backing bytes.
        ///
        /// # Safety
        /// `raw_surface` must be a valid `IOSurfaceRef` for this call. The
        /// native wrapper retains it after a successful import.
        pub unsafe fn from_iosurface(
            raw_surface: *mut c_void,
            logical_bytes: usize,
        ) -> Option<Self> {
            if raw_surface.is_null() || logical_bytes == 0 {
                return None;
            }
            let handle = unsafe { logan_metal_shared_surface_wrap(raw_surface, logical_bytes) };
            if handle.is_null() {
                return None;
            }
            let native_logical = unsafe { logan_metal_shared_surface_length(handle) };
            let allocation_bytes = unsafe { logan_metal_shared_surface_allocation_length(handle) };
            if native_logical != logical_bytes || allocation_bytes < logical_bytes {
                unsafe { logan_metal_shared_surface_free(handle) };
                return None;
            }
            Some(Self {
                handle,
                logical_bytes,
                allocation_bytes,
                _not_send_sync: std::marker::PhantomData,
            })
        }

        pub fn len(&self) -> usize {
            self.logical_bytes
        }

        pub fn is_empty(&self) -> bool {
            self.logical_bytes == 0
        }

        pub fn allocation_len(&self) -> usize {
            self.allocation_bytes
        }

        /// CPU-visible address returned by `MTLBuffer.contents`.
        ///
        /// # Safety
        /// The pointer aliases memory potentially used by Metal and ANE.
        /// Callers must establish device completion and Rust aliasing safety.
        pub unsafe fn contents_ptr(&self) -> *mut u8 {
            unsafe { logan_metal_shared_surface_contents(self.handle).cast() }
        }
    }

    impl Drop for MetalSharedSurface {
        fn drop(&mut self) {
            unsafe { logan_metal_shared_surface_free(self.handle) }
        }
    }

    impl MetalAneDynamicPack {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            qkv_dst: &MetalSharedSurface,
            aux_dst: &MetalSharedSurface,
            wqkv: &[u8],
            wz: &[u8],
            wa: &[u8],
            wb: &[u8],
            hidden: usize,
            spatial: usize,
            qkv_rows: usize,
            z_rows: usize,
            ab_rows: usize,
            qkv_stride: usize,
            aux_stride: usize,
            qkv_offset: usize,
            z_offset: usize,
            a_offset: usize,
            b_offset: usize,
        ) -> Option<Self> {
            if hidden == 0
                || spatial == 0
                || qkv_rows == 0
                || z_rows == 0
                || ab_rows == 0
                || wqkv.len() != qkv_rows.checked_mul(hidden)?.checked_mul(2)?
                || wz.len() != z_rows.checked_mul(hidden)?.checked_mul(2)?
                || wa.len() != ab_rows.checked_mul(hidden)?.checked_mul(2)?
                || wb.len() != ab_rows.checked_mul(hidden)?.checked_mul(2)?
            {
                return None;
            }
            let cv = |v: usize| u32::try_from(v).ok();
            let handle = unsafe {
                logan_metal_ane_dynamic_pack_create(
                    qkv_dst.handle,
                    aux_dst.handle,
                    wqkv.as_ptr().cast(),
                    wz.as_ptr().cast(),
                    wa.as_ptr().cast(),
                    wb.as_ptr().cast(),
                    cv(hidden)?,
                    cv(spatial)?,
                    cv(qkv_rows)?,
                    cv(z_rows)?,
                    cv(ab_rows)?,
                    cv(qkv_stride)?,
                    cv(aux_stride)?,
                    cv(qkv_offset)?,
                    cv(z_offset)?,
                    cv(a_offset)?,
                    cv(b_offset)?,
                )
            };
            if handle.is_null() {
                return None;
            }
            Some(Self {
                handle,
                hidden,
                _not_send_sync: std::marker::PhantomData,
            })
        }

        /// Commit the pack and signal the shared event when both ANE input surfaces are complete.
        pub fn begin(
            &mut self,
            x: &[f32],
            fence: &MetalAneFence,
        ) -> Option<MetalAneDynamicPackPending> {
            if x.len() != self.hidden || fence.value() == 0 {
                return None;
            }
            let event = fence.metal_shared_event();
            if event.is_null() {
                return None;
            }
            let raw = unsafe {
                logan_metal_ane_dynamic_pack_begin(self.handle, x.as_ptr(), event, fence.value())
            };
            Some(MetalAneDynamicPackPending {
                raw: Some(std::ptr::NonNull::new(raw)?),
                _not_send_sync: std::marker::PhantomData,
            })
        }

        /// Fill both dynamic-ANE packed IOSurfaces and wait for Metal completion.
        /// Returns actual GPU execution milliseconds; callers can separately
        /// measure exposed host wall time to diagnose queue/residency stalls.
        pub fn run(&mut self, x: &[f32]) -> Option<f64> {
            if x.len() != self.hidden {
                return None;
            }
            let mut gpu_ms = 0.0;
            let rc =
                unsafe { logan_metal_ane_dynamic_pack_run(self.handle, x.as_ptr(), &mut gpu_ms) };
            (rc > 0).then_some(gpu_ms)
        }
    }

    impl MetalAneDynamicPackPending {
        pub fn finish(mut self) -> Option<f64> {
            let raw = self.raw.take()?;
            let mut gpu_ms = 0.0;
            let rc = unsafe { logan_metal_ane_dynamic_pack_finish(raw.as_ptr(), &mut gpu_ms) };
            (rc > 0).then_some(gpu_ms)
        }
    }
    impl Drop for MetalAneDynamicPackPending {
        fn drop(&mut self) {
            if let Some(raw) = self.raw.take() {
                unsafe { logan_metal_ane_dynamic_pack_discard(raw.as_ptr()) }
            }
        }
    }

    impl Drop for MetalAneDynamicPack {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { logan_metal_ane_dynamic_pack_free(self.handle) };
                self.handle = std::ptr::null_mut();
            }
        }
    }

    /// Owns a committed command and borrows its continuation exclusively.
    /// Drop drains execution; native ownership also retains both IOSurfaces.
    #[must_use]
    pub struct MetalGdnConvPending<'a> {
        handle: *mut c_void,
        _owner: std::marker::PhantomData<&'a mut MetalGdnConvSilu>,
    }

    impl MetalGdnConvPending<'_> {
        pub fn finish(mut self) -> (bool, f64) {
            let mut gpu_ms = 0.0;
            let ok = unsafe { logan_metal_gdn_conv_silu_finish(self.handle, &mut gpu_ms) == 1 };
            self.handle = std::ptr::null_mut();
            (ok, gpu_ms)
        }
    }

    impl Drop for MetalGdnConvPending<'_> {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    logan_metal_gdn_conv_silu_finish(self.handle, std::ptr::null_mut());
                }
            }
        }
    }

    impl MetalGdnConvSilu {
        pub fn new(
            input: &MetalSharedSurface,
            output: &MetalSharedSurface,
            weights: &[f32],
            channels: usize,
            spatial: usize,
            kernel: usize,
        ) -> Option<Self> {
            if channels == 0
                || spatial == 0
                || kernel == 0
                || kernel > spatial
                || weights.len() != channels.checked_mul(kernel)?
                || input.len() < channels.checked_mul(spatial)?.checked_mul(4)?
                || output.len() < channels.checked_mul(4)?
            {
                return None;
            }
            let handle = unsafe {
                logan_metal_gdn_conv_silu_create(
                    input.handle,
                    output.handle,
                    weights.as_ptr(),
                    channels,
                    spatial,
                    kernel,
                )
            };
            if handle.is_null() {
                None
            } else {
                Some(Self {
                    handle,
                    _not_send_sync: std::marker::PhantomData,
                })
            }
        }

        /// # Safety
        /// External owners must not map or mutate the imported input/output
        /// surfaces until the returned ticket finishes or is dropped.
        pub unsafe fn begin(&mut self) -> Option<MetalGdnConvPending<'_>> {
            let handle = unsafe { logan_metal_gdn_conv_silu_begin(self.handle) };
            (!handle.is_null()).then_some(MetalGdnConvPending {
                handle,
                _owner: std::marker::PhantomData,
            })
        }

        pub fn run(&mut self) -> bool {
            unsafe { logan_metal_gdn_conv_silu_run(self.handle) == 1 }
        }
    }

    impl Drop for MetalGdnConvSilu {
        fn drop(&mut self) {
            unsafe { logan_metal_gdn_conv_silu_free(self.handle) }
        }
    }

    /// Lazily-initialized Metal availability. Returns true once init() succeeded.
    static INIT: std::sync::Once = std::sync::Once::new();
    static mut AVAILABLE: bool = false;

    pub fn metal_init() -> bool {
        INIT.call_once(|| {
            let ok = unsafe { coli_metal_init() } == 1;
            unsafe { AVAILABLE = ok };
        });
        metal_available()
    }

    pub fn metal_available() -> bool {
        unsafe { AVAILABLE && coli_metal_available() == 1 }
    }

    pub fn dense_profile_start() {
        if metal_available() {
            unsafe {
                coli_metal_profile_reset();
                coli_metal_profile_set_on(1);
            }
        }
    }
    pub fn dense_profile_stop() -> (u64, u64, u64, u64) {
        let (mut e, mut s, mut w, mut k) = (0, 0, 0, 0);
        if metal_available() {
            unsafe {
                coli_metal_profile_set_on(0);
                coli_metal_profile_get(&mut e, &mut s, &mut w, &mut k);
            }
        }
        (e, s, w, k)
    }

    /// y[O] = x[I] @ W^T for one token. Supported dense formats include
    /// raw BF16 (`fmt=5`), MXFP4 (`fmt=7`), FP8 (`fmt=8`), and Spark/MLX
    /// affine-8 (`fmt=15`).
    /// Returns true if Metal ran the matmul.
    pub fn metal_matmul(
        tensor: &mut *mut ColiMetalTensor,
        y: &mut [f32],
        x: &[f32],
        weights: &[u8],
        scales: &[u8],
        fmt: i32,
        i: usize,
        o: usize,
    ) -> bool {
        if !metal_available() {
            return false;
        }
        let (weight_bytes, scale_bytes) = match fmt {
            5 => (o * i * std::mem::size_of::<u16>(), 0),
            7 => (o * ((i + 1) / 2), o * ((i + 31) / 32)),
            8 => (
                o * i,
                o.div_ceil(128) * i.div_ceil(128) * std::mem::size_of::<f32>(),
            ),
            11 => (o * i, o * i.div_ceil(32) * std::mem::size_of::<f32>()),
            12 => (o * i, o * i.div_ceil(16) * std::mem::size_of::<f32>()),
            13 => (o * i, o * i.div_ceil(8) * std::mem::size_of::<f32>()),
            14 => (
                o * i + o * i.div_ceil(32) * 3,
                o * i.div_ceil(32) * std::mem::size_of::<f32>(),
            ),
            15 => (o * i, 2 * o * i.div_ceil(64) * std::mem::size_of::<u16>()),
            _ => return false,
        };
        if weights.len() < weight_bytes || scales.len() < scale_bytes {
            return false;
        }
        static BF16_DUMMY_SCALE: [f32; 1] = [1.0];
        let scale_ptr = if fmt == 5 {
            BF16_DUMMY_SCALE.as_ptr()
        } else {
            scales.as_ptr() as *const f32
        };
        let rc = unsafe {
            coli_metal_matmul(
                tensor,
                y.as_mut_ptr(),
                x.as_ptr(),
                weights.as_ptr() as *const c_void,
                scale_ptr,
                fmt,
                1,
                i as i32,
                o as i32,
                match fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    _ => 0,
                },
            )
        };
        rc == 1
    }

    /// Native MLX affine GEMV over the checkpoint's packed U32 bitstream.
    /// `aux` is `[BF16 scales][BF16 biases]`, each `[O, I/group_size]`.
    /// Formats 16..19 map to 4/5/6/8-bit respectively; experimental fmt20 is
    /// the Flash-MoE-inspired specialized Q4 nibble/FMA path.
    fn mlx_affine_fmt(bits: u8, q4_fma: bool, aux_fp16: bool) -> Option<i32> {
        if aux_fp16 {
            // 21..24 mirror generic MLX affine 16..19 but decode the
            // scale/bias sidecars as IEEE fp16. Keep the Q4-FMA experiment
            // BF16-only until it has its own fp16 qualification.
            return match bits {
                4 => Some(21),
                5 => Some(22),
                6 => Some(23),
                8 => Some(24),
                _ => None,
            };
        }
        match bits {
            4 => Some(if q4_fma { 20 } else { 16 }),
            5 => Some(17),
            6 => Some(18),
            8 => Some(19),
            _ => None,
        }
    }

    fn q4_fma_enabled() -> bool {
        std::env::var("LOGAN_Q4_FMA")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    }

    fn metal_matmul_mlx_affine_impl(
        tensor: &mut *mut ColiMetalTensor,
        y: &mut [f32],
        x: &[f32],
        weights: &[u8],
        aux: &[u8],
        bits: u8,
        group_size: usize,
        aux_fp16: bool,
        i: usize,
        o: usize,
        q4_fma: bool,
    ) -> bool {
        if !metal_available()
            || i == 0
            || o == 0
            || group_size == 0
            || i % group_size != 0
            || x.len() < i
            || y.len() < o
        {
            return false;
        }
        let Some(fmt) = mlx_affine_fmt(bits, q4_fma, aux_fp16) else {
            return false;
        };
        let row_bits = match i.checked_mul(bits as usize) {
            Some(v) if v % 32 == 0 => v,
            _ => return false,
        };
        let weight_bytes = match o.checked_mul(row_bits / 8) {
            Some(v) => v,
            None => return false,
        };
        let groups = i / group_size;
        let aux_bytes = match 2_usize
            .checked_mul(o)
            .and_then(|v| v.checked_mul(groups))
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        {
            Some(v) => v,
            None => return false,
        };
        if weights.len() < weight_bytes || aux.len() < aux_bytes {
            return false;
        }
        unsafe {
            coli_metal_matmul(
                tensor,
                y.as_mut_ptr(),
                x.as_ptr(),
                weights.as_ptr() as *const c_void,
                aux.as_ptr() as *const f32,
                fmt,
                1,
                i as i32,
                o as i32,
                group_size as i32,
            ) == 1
        }
    }

    pub fn metal_matmul_mlx_affine(
        tensor: &mut *mut ColiMetalTensor,
        y: &mut [f32],
        x: &[f32],
        weights: &[u8],
        aux: &[u8],
        bits: u8,
        group_size: usize,
        aux_fp16: bool,
        i: usize,
        o: usize,
    ) -> bool {
        metal_matmul_mlx_affine_impl(
            tensor,
            y,
            x,
            weights,
            aux,
            bits,
            group_size,
            aux_fp16,
            i,
            o,
            q4_fma_enabled(),
        )
    }

    pub fn metal_matmul_mlx_affine_multi(x: &[f32], descs: &mut [MlxAffineMatmulDesc<'_>]) -> bool {
        if !metal_available() || descs.is_empty() || descs.len() > 16 {
            return false;
        }
        let mut raw = Vec::with_capacity(descs.len());
        for d in descs.iter_mut() {
            // A descriptor with its own activation carries its own input and
            // leading dimension; one without falls back to the shared `x`,
            // which the C side reads as a single row `[1, I]`. Only the
            // fallback descriptors must agree on `I`, and the C side enforces
            // that.
            let xptr = match d.x {
                Some(v) => {
                    if v.len() < d.i {
                        return false;
                    }
                    v.as_ptr()
                }
                None if x.len() >= d.i => x.as_ptr(),
                None if x.is_empty() => std::ptr::null(),
                None => {
                    // The shared buffer is only read when at least one
                    // descriptor falls back to it, and then only at one row
                    // `[1, I]`. A batch made entirely of private-activation
                    // descriptors never touches it, so an empty `x` is valid
                    // there and must not be rejected here.
                    if x.len() < d.i {
                        return false;
                    }
                    x.as_ptr()
                }
            };
            if d.i == 0
                || d.o == 0
                || d.o > i32::MAX as usize
                || d.y.len() < d.o
                || d.group_size == 0
                || d.i % d.group_size != 0
                || d.group_size > i32::MAX as usize
            {
                return false;
            }
            let Some(fmt) = mlx_affine_fmt(d.bits, q4_fma_enabled(), d.aux_fp16) else {
                return false;
            };
            let row_bits = match d.i.checked_mul(d.bits as usize) {
                Some(v) if v % 32 == 0 => v,
                _ => return false,
            };
            let weight_bytes = match d.o.checked_mul(row_bits / 8) {
                Some(v) => v,
                None => return false,
            };
            let groups = d.i / d.group_size;
            let aux_bytes = match 2_usize
                .checked_mul(d.o)
                .and_then(|v| v.checked_mul(groups))
                .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            {
                Some(v) => v,
                None => return false,
            };
            if d.weights.len() < weight_bytes || d.aux.len() < aux_bytes {
                return false;
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: d.tensor,
                y: d.y.as_mut_ptr(),
                weights: d.weights.as_ptr() as *const c_void,
                scales: d.aux.as_ptr() as *const f32,
                fmt,
                i: d.i as i32,
                o: d.o as i32,
                gs: d.group_size as i32,
                x: xptr,
                s: 1,
            });
        }
        let ok = unsafe {
            coli_metal_matmul_multi(x.as_ptr(), 1, raw.as_mut_ptr(), raw.len() as i32) == 1
        };
        for (d, r) in descs.iter_mut().zip(raw.iter()) {
            d.tensor = r.tensor;
        }
        ok
    }

    #[cfg(test)]
    mod mlx_affine_tests {
        use super::{
            gdn_mxfp4, gdn_mxfp4_drop_model, metal_init, metal_matmul_mlx_affine,
            metal_matmul_mlx_affine_multi, ColiMetalTensor, MetalWeightDesc, MlxAffineMatmulDesc,
        };

        fn bf16(v: f32) -> [u8; 2] {
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        }

        fn pack_codes(codes: &[u32], bits: u8) -> Vec<u8> {
            let total_bits = codes.len() * bits as usize;
            assert_eq!(total_bits % 32, 0);
            let mut words = vec![0_u32; total_bits / 32];
            for (index, &code) in codes.iter().enumerate() {
                let bit = index * bits as usize;
                let word = bit / 32;
                let shift = bit % 32;
                words[word] |= code << shift;
                if shift + bits as usize > 32 {
                    words[word + 1] |= code >> (32 - shift);
                }
            }
            words.into_iter().flat_map(u32::to_le_bytes).collect()
        }

        #[test]
        fn fused_gdn_accepts_mixed_mlx_affine_formats_and_group_sizes() {
            assert!(
                metal_init(),
                "Metal backend must initialize on Apple Silicon"
            );
            const D: usize = 128;
            const KH: usize = 1;
            const KD: usize = 64;
            const VH: usize = 2;
            const VD: usize = 64;
            const KK: usize = 4;
            const KDIM: usize = KH * KD;
            const VDIM: usize = VH * VD;
            const CDIM: usize = 2 * KDIM + VDIM;

            fn zero_affine(o: usize, i: usize, bits: u8, group: usize) -> (Vec<u8>, Vec<u8>) {
                let weights = vec![0_u8; o * i * bits as usize / 8];
                let groups = i / group;
                let mut aux = Vec::with_capacity(2 * o * groups * 2);
                for _ in 0..o * groups {
                    aux.extend_from_slice(&bf16(1.0));
                }
                for _ in 0..o * groups {
                    aux.extend_from_slice(&bf16(0.0));
                }
                (weights, aux)
            }

            let (wq, aq) = zero_affine(CDIM, D, 4, 64);
            let (wz, az) = zero_affine(VDIM, D, 5, 128);
            let (wa, aa) = zero_affine(VH, D, 6, 64);
            let (wb, ab) = zero_affine(VH, D, 8, 128);
            let (wo, ao) = zero_affine(D, VDIM, 4, 64);
            let mut descs = [
                MetalWeightDesc {
                    tensor: std::ptr::null_mut(),
                    weights: &wq,
                    scales: &aq,
                    fmt: 16,
                    group_size: 64,
                    i: D,
                    o: CDIM,
                },
                MetalWeightDesc {
                    tensor: std::ptr::null_mut(),
                    weights: &wz,
                    scales: &az,
                    fmt: 17,
                    group_size: 128,
                    i: D,
                    o: VDIM,
                },
                MetalWeightDesc {
                    tensor: std::ptr::null_mut(),
                    weights: &wa,
                    scales: &aa,
                    fmt: 18,
                    group_size: 64,
                    i: D,
                    o: VH,
                },
                MetalWeightDesc {
                    tensor: std::ptr::null_mut(),
                    weights: &wb,
                    scales: &ab,
                    fmt: 19,
                    group_size: 128,
                    i: D,
                    o: VH,
                },
                MetalWeightDesc {
                    tensor: std::ptr::null_mut(),
                    weights: &wo,
                    scales: &ao,
                    fmt: 16,
                    group_size: 64,
                    i: VDIM,
                    o: D,
                },
            ];
            let x = vec![0.25_f32; D];
            let mut out = vec![123.0_f32; D];
            let a_log = vec![0.0_f32; VH];
            let dt_bias = vec![0.0_f32; VH];
            let conv_w = vec![1.0_f32; CDIM * KK];
            let norm_w = vec![1.0_f32; VD];
            unsafe fn aligned_zeroed_f32(len: usize) -> (*mut f32, std::alloc::Layout) {
                let logical = len.checked_mul(std::mem::size_of::<f32>()).unwrap();
                let bytes = logical.div_ceil(16_384) * 16_384;
                let layout =
                    std::alloc::Layout::from_size_align(bytes.max(16_384), 16_384).unwrap();
                let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
                assert!(!ptr.is_null());
                (ptr, layout)
            }
            let state_len = VH * KD * VD;
            let conv_state_len = CDIM * (KK - 1);
            let (state_ptr, state_layout) = unsafe { aligned_zeroed_f32(state_len) };
            let (conv_ptr, conv_layout) = unsafe { aligned_zeroed_f32(conv_state_len) };
            let state = unsafe { std::slice::from_raw_parts_mut(state_ptr, state_len) };
            let conv_state = unsafe { std::slice::from_raw_parts_mut(conv_ptr, conv_state_len) };
            let model_id = 0xaff1_u64;
            let rc = gdn_mxfp4(
                model_id, 0, &mut descs, &x, &mut out, &a_log, &dt_bias, &conv_w, &norm_w, state,
                conv_state, D, KH, KD, VH, VD, KK, 0, 1e-6,
            );
            assert_eq!(
                rc, 1,
                "full fused GDN must accept native MLX affine descriptors"
            );
            assert!(
                out.iter().all(|v| v.abs() <= 1e-6),
                "zero affine projections must produce zero output: {:?}",
                &out[..8]
            );
            for d in &descs {
                if !d.tensor.is_null() {
                    unsafe { super::coli_metal_tensor_free(d.tensor) };
                }
            }
            gdn_mxfp4_drop_model(model_id);
            unsafe {
                std::alloc::dealloc(state_ptr as *mut u8, state_layout);
                std::alloc::dealloc(conv_ptr as *mut u8, conv_layout);
            }
        }

        #[test]
        fn native_mlx_affine_multi_handles_mixed_bits_and_groups() {
            assert!(
                metal_init(),
                "Metal backend must initialize on Apple Silicon"
            );
            const I: usize = 128;
            let x: Vec<f32> = (0..I)
                .map(|i| ((i as i32 % 13) - 6) as f32 * 0.0625)
                .collect();

            let build = |bits: u8, group_size: usize, o: usize| {
                let mask = (1_u32 << bits) - 1;
                let groups = I / group_size;
                let mut weights = Vec::new();
                let mut scales = Vec::new();
                let mut biases = Vec::new();
                let mut expected = vec![0.0_f32; o];
                for row in 0..o {
                    let codes: Vec<u32> = (0..I)
                        .map(|col| (row as u32 * 11 + col as u32 * 5 + 1) & mask)
                        .collect();
                    weights.extend_from_slice(&pack_codes(&codes, bits));
                    for group in 0..groups {
                        let scale = 0.0625 * (row + group + 1) as f32;
                        let bias = -0.125 + 0.03125 * (row + group) as f32;
                        scales.extend_from_slice(&bf16(scale));
                        biases.extend_from_slice(&bf16(bias));
                        let start = group * group_size;
                        for col in start..start + group_size {
                            expected[row] += (codes[col] as f32 * scale + bias) * x[col];
                        }
                    }
                }
                let mut aux = scales;
                aux.extend_from_slice(&biases);
                (weights, aux, expected)
            };

            let (w5, a5, e5) = build(5, 128, 2);
            let (w6, a6, e6) = build(6, 64, 3);
            let mut y5 = vec![0.0_f32; 2];
            let mut y6 = vec![0.0_f32; 3];
            let mut t5: *mut ColiMetalTensor = std::ptr::null_mut();
            let mut t6: *mut ColiMetalTensor = std::ptr::null_mut();
            let mut descs = [
                MlxAffineMatmulDesc {
                    tensor: t5,
                    y: &mut y5,
                    weights: &w5,
                    aux: &a5,
                    bits: 5,
                    group_size: 128,
                    aux_fp16: false,
                    i: I,
                    o: 2,
                    x: None,
                },
                MlxAffineMatmulDesc {
                    tensor: t6,
                    y: &mut y6,
                    weights: &w6,
                    aux: &a6,
                    bits: 6,
                    group_size: 64,
                    aux_fp16: false,
                    i: I,
                    o: 3,
                    x: None,
                },
            ];
            assert!(metal_matmul_mlx_affine_multi(&x, &mut descs));
            t5 = descs[0].tensor;
            t6 = descs[1].tensor;
            drop(descs);
            for (got, expected) in y5.iter().zip(e5.iter()).chain(y6.iter().zip(e6.iter())) {
                let tol = 2e-3_f32.max(expected.abs() * 2e-4);
                assert!(
                    (got - expected).abs() <= tol,
                    "gpu={got} ref={expected} tol={tol}"
                );
            }
            if !t5.is_null() {
                unsafe { super::coli_metal_tensor_free(t5) };
            }
            if !t6.is_null() {
                unsafe { super::coli_metal_tensor_free(t6) };
            }
        }

        #[test]
        fn native_mlx_affine_gemv_matches_reference_for_all_supported_widths() {
            assert!(
                metal_init(),
                "Metal backend must initialize on Apple Silicon"
            );
            const O: usize = 3;
            const I: usize = 128;
            let x: Vec<f32> = (0..I)
                .map(|i| ((i as i32 % 11) - 5) as f32 * 0.125)
                .collect();

            for &(bits, group_size) in &[(4_u8, 64_usize), (5, 128), (6, 64), (8, 128)] {
                let mask = (1_u32 << bits) - 1;
                let groups = I / group_size;
                let mut weights = Vec::new();
                let mut expected = vec![0.0_f32; O];
                let mut scale_bytes = Vec::new();
                let mut bias_bytes = Vec::new();

                for row in 0..O {
                    let codes: Vec<u32> = (0..I)
                        .map(|col| (row as u32 * 17 + col as u32 * 7 + 3) & mask)
                        .collect();
                    weights.extend_from_slice(&pack_codes(&codes, bits));
                    for group in 0..groups {
                        let scale = 0.125 * (row + group + 1) as f32;
                        let bias = -0.25 + 0.0625 * (row + group) as f32;
                        scale_bytes.extend_from_slice(&bf16(scale));
                        bias_bytes.extend_from_slice(&bf16(bias));
                        let start = group * group_size;
                        let end = start + group_size;
                        for col in start..end {
                            expected[row] += (codes[col] as f32 * scale + bias) * x[col];
                        }
                    }
                }
                let mut aux = scale_bytes;
                aux.extend_from_slice(&bias_bytes);
                let mut out = vec![0.0_f32; O];
                let mut tensor: *mut ColiMetalTensor = std::ptr::null_mut();
                assert!(
                    metal_matmul_mlx_affine(
                        &mut tensor,
                        &mut out,
                        &x,
                        &weights,
                        &aux,
                        bits,
                        group_size,
                        false,
                        I,
                        O,
                    ),
                    "Metal declined MLX affine bits={bits} group={group_size}"
                );
                if !tensor.is_null() {
                    unsafe { super::coli_metal_tensor_free(tensor) };
                }
                for row in 0..O {
                    let tol = 2e-3_f32.max(expected[row].abs() * 2e-4);
                    assert!(
                        (out[row] - expected[row]).abs() <= tol,
                        "bits={bits} group={group_size} row={row}: gpu={} ref={} tol={tol}",
                        out[row],
                        expected[row]
                    );
                }
            }
        }

        #[test]
        fn q4_fma_variant_matches_reference_and_baseline() {
            assert!(
                metal_init(),
                "Metal backend must initialize on Apple Silicon"
            );
            const O: usize = 5;
            const I: usize = 256;
            const GROUP: usize = 64;
            let x: Vec<f32> = (0..I)
                .map(|i| ((i as i32 % 19) - 9) as f32 * 0.03125)
                .collect();
            let groups = I / GROUP;
            let mut weights = Vec::new();
            let mut scale_bytes = Vec::new();
            let mut bias_bytes = Vec::new();
            let mut expected = vec![0.0_f32; O];
            for row in 0..O {
                let codes: Vec<u32> = (0..I)
                    .map(|col| (row as u32 * 13 + col as u32 * 9 + 5) & 0xF)
                    .collect();
                weights.extend_from_slice(&pack_codes(&codes, 4));
                for group in 0..groups {
                    let scale = 0.03125 * (row + group + 1) as f32;
                    let bias = -0.1875 + 0.015625 * (row + group) as f32;
                    scale_bytes.extend_from_slice(&bf16(scale));
                    bias_bytes.extend_from_slice(&bf16(bias));
                    let start = group * GROUP;
                    for col in start..start + GROUP {
                        expected[row] += (codes[col] as f32 * scale + bias) * x[col];
                    }
                }
            }
            let mut aux = scale_bytes;
            aux.extend_from_slice(&bias_bytes);
            let mut baseline = vec![0.0_f32; O];
            let mut candidate = vec![0.0_f32; O];
            let mut baseline_tensor: *mut ColiMetalTensor = std::ptr::null_mut();
            let mut candidate_tensor: *mut ColiMetalTensor = std::ptr::null_mut();
            assert!(super::metal_matmul_mlx_affine_impl(
                &mut baseline_tensor,
                &mut baseline,
                &x,
                &weights,
                &aux,
                4,
                GROUP,
                false,
                I,
                O,
                false,
            ));
            assert!(super::metal_matmul_mlx_affine_impl(
                &mut candidate_tensor,
                &mut candidate,
                &x,
                &weights,
                &aux,
                4,
                GROUP,
                false,
                I,
                O,
                true,
            ));
            if !baseline_tensor.is_null() {
                unsafe { super::coli_metal_tensor_free(baseline_tensor) };
            }
            if !candidate_tensor.is_null() {
                unsafe { super::coli_metal_tensor_free(candidate_tensor) };
            }
            for row in 0..O {
                let tol = 3e-3_f32.max(expected[row].abs() * 3e-4);
                assert!(
                    (baseline[row] - expected[row]).abs() <= tol,
                    "baseline row={row}: gpu={} ref={} tol={tol}",
                    baseline[row],
                    expected[row]
                );
                assert!(
                    (candidate[row] - expected[row]).abs() <= tol,
                    "fma row={row}: gpu={} ref={} tol={tol}",
                    candidate[row],
                    expected[row]
                );
                assert!(
                    (candidate[row] - baseline[row]).abs() <= tol,
                    "row={row}: fma={} baseline={} tol={tol}",
                    candidate[row],
                    baseline[row]
                );
            }
        }
    }

    /// Several independent one-token quantized GEMVs that share the same
    /// activation vector, encoded into one Metal command buffer and waited once.
    /// Each descriptor retains its lazily-created opaque tensor handle.
    pub fn metal_matmul_multi(x: &[f32], descs: &mut [MetalMatmulDesc<'_>]) -> bool {
        if !metal_available() || descs.is_empty() || descs.len() > 16 {
            return false;
        }
        let common_i = descs[0].i;
        if common_i == 0 || common_i > i32::MAX as usize || x.len() < common_i {
            return false;
        }
        let mut raw = Vec::with_capacity(descs.len());
        for d in descs.iter_mut() {
            if d.i != common_i || d.o == 0 || d.o > i32::MAX as usize || d.y.len() < d.o {
                return false;
            }
            let (weight_bytes, scale_bytes) = match d.fmt {
                5 => (d.o * d.i * std::mem::size_of::<u16>(), 0),
                7 => (d.o * d.i.div_ceil(2), d.o * d.i.div_ceil(32)),
                8 => (
                    d.o * d.i,
                    d.o.div_ceil(128) * d.i.div_ceil(128) * std::mem::size_of::<f32>(),
                ),
                11 => (
                    d.o * d.i,
                    d.o * d.i.div_ceil(32) * std::mem::size_of::<f32>(),
                ),
                12 => (
                    d.o * d.i,
                    d.o * d.i.div_ceil(16) * std::mem::size_of::<f32>(),
                ),
                13 => (
                    d.o * d.i,
                    d.o * d.i.div_ceil(8) * std::mem::size_of::<f32>(),
                ),
                14 => (
                    d.o * d.i + d.o * d.i.div_ceil(32) * 3,
                    d.o * d.i.div_ceil(32) * std::mem::size_of::<f32>(),
                ),
                15 => (
                    d.o * d.i,
                    2 * d.o * d.i.div_ceil(64) * std::mem::size_of::<u16>(),
                ),
                _ => return false,
            };
            if d.weights.len() < weight_bytes || d.scales.len() < scale_bytes {
                return false;
            }
            static BF16_DUMMY_SCALE: [f32; 1] = [1.0];
            let scale_ptr = if d.fmt == 5 {
                BF16_DUMMY_SCALE.as_ptr()
            } else {
                d.scales.as_ptr() as *const f32
            };
            raw.push(ColiMetalMatmulDescRaw {
                tensor: d.tensor,
                y: d.y.as_mut_ptr(),
                weights: d.weights.as_ptr() as *const c_void,
                scales: scale_ptr,
                fmt: d.fmt,
                i: d.i as i32,
                o: d.o as i32,
                gs: match d.fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    _ => 0,
                },
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let ok = unsafe {
            coli_metal_matmul_multi(x.as_ptr(), 1, raw.as_mut_ptr(), raw.len() as i32) == 1
        };
        for (d, r) in descs.iter_mut().zip(raw.iter()) {
            d.tensor = r.tensor;
        }
        ok
    }

    #[allow(clippy::too_many_arguments)]
    pub fn llama_layer(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalMatmulDesc<'_>],
        x: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
        input_norm: &[f32],
        post_norm: &[f32],
        d: usize,
        inter: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || model_id == 0
            || descs.len() != 7
            || x.len() < d
            || k_out.len() < kv_heads.saturating_mul(head_dim)
            || v_out.len() < kv_heads.saturating_mul(head_dim)
            || input_norm.len() < d
            || post_norm.len() < d
            || d == 0
            || inter == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || heads % kv_heads != 0
            || layer > i32::MAX as usize
            || d > i32::MAX as usize
            || inter > i32::MAX as usize
            || heads > i32::MAX as usize
            || kv_heads > i32::MAX as usize
            || head_dim > i32::MAX as usize
            || pos > i32::MAX as usize
        {
            return 0;
        }
        let q = heads.saturating_mul(head_dim);
        let kv = kv_heads.saturating_mul(head_dim);
        let expected_i = [d, d, d, q, d, d, inter];
        let expected_o = [q, kv, kv, d, inter, inter, d];
        static BF16_DUMMY_SCALE: [f32; 1] = [1.0];
        let mut raw = Vec::with_capacity(descs.len());
        for (idx, desc) in descs.iter_mut().enumerate() {
            let (weight_bytes, scale_bytes, fmt, gs, scale_ptr) = match desc.fmt {
                5 => (
                    desc.i.saturating_mul(desc.o).saturating_mul(2),
                    0,
                    5,
                    0,
                    BF16_DUMMY_SCALE.as_ptr(),
                ),
                15 => (
                    desc.i.saturating_mul(desc.o),
                    2usize
                        .saturating_mul(desc.o)
                        .saturating_mul(desc.i.div_ceil(64))
                        .saturating_mul(std::mem::size_of::<u16>()),
                    15,
                    64,
                    desc.scales.as_ptr() as *const f32,
                ),
                _ => return 0,
            };
            if desc.i != expected_i[idx]
                || desc.o != expected_o[idx]
                || desc.weights.len() < weight_bytes
                || desc.scales.len() < scale_bytes
            {
                return 0;
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: desc.tensor,
                y: std::ptr::null_mut(),
                weights: desc.weights.as_ptr() as *const c_void,
                scales: scale_ptr,
                fmt,
                i: desc.i as i32,
                o: desc.o as i32,
                gs,
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_llama_layer(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                x.as_mut_ptr(),
                k_out.as_mut_ptr(),
                v_out.as_mut_ptr(),
                input_norm.as_ptr(),
                post_norm.as_ptr(),
                d as i32,
                inter as i32,
                heads as i32,
                kv_heads as i32,
                head_dim as i32,
                pos as i32,
                theta,
                eps,
            )
        };
        for (desc, item) in descs.iter_mut().zip(raw.iter()) {
            desc.tensor = item.tensor;
        }
        rc
    }

    pub fn llama_drop_model(model_id: u64) {
        if model_id != 0 && metal_available() {
            unsafe { coli_metal_llama_drop_model(model_id) };
        }
    }

    pub fn spark_token_begin(model_id: u64, x: &[f32], d: usize, pos: usize) -> bool {
        if !metal_available()
            || model_id == 0
            || d == 0
            || x.len() < d
            || d > i32::MAX as usize
            || pos > i32::MAX as usize
        {
            return false;
        }
        unsafe { coli_metal_spark_token_begin(model_id, x.as_ptr(), d as i32, pos as i32) == 1 }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_layer_encode(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalWeightDesc<'_>],
        input_norm: &[f32],
        post_norm: &[f32],
        d: usize,
        inter: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sliding: bool,
        window: usize,
        pos: usize,
        rotary_dim: usize,
        theta: f32,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || descs.len() != 6
            || input_norm.len() < d
            || post_norm.len() < d
            || [
                d,
                inter,
                heads,
                kv_heads,
                head_dim,
                window,
                pos + 1,
                rotary_dim,
            ]
            .into_iter()
            .any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        let mut raw = Vec::with_capacity(6);
        for desc in descs.iter() {
            if desc.fmt != 15 || desc.i > i32::MAX as usize || desc.o > i32::MAX as usize {
                return 0;
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: desc.tensor,
                y: std::ptr::null_mut(),
                weights: desc.weights.as_ptr() as *const c_void,
                scales: desc.scales.as_ptr() as *const f32,
                fmt: 15,
                i: desc.i as i32,
                o: desc.o as i32,
                gs: 64,
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_spark_layer_encode(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                input_norm.as_ptr(),
                post_norm.as_ptr(),
                d as i32,
                inter as i32,
                heads as i32,
                kv_heads as i32,
                head_dim as i32,
                sliding as i32,
                window as i32,
                pos as i32,
                rotary_dim as i32,
                theta,
                eps,
            )
        };
        for (desc, r) in descs.iter_mut().zip(raw.iter()) {
            desc.tensor = r.tensor;
        }
        rc
    }

    pub fn spark_token_end(model_id: u64, x: &mut [f32], d: usize, pos: usize) -> i32 {
        if !metal_available()
            || model_id == 0
            || d == 0
            || x.len() < d
            || d > i32::MAX as usize
            || pos > i32::MAX as usize
        {
            return 0;
        }
        unsafe { coli_metal_spark_token_end(model_id, x.as_mut_ptr(), d as i32, pos as i32) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_token_end_top1(
        model_id: u64,
        head: &mut MetalWeightDesc<'_>,
        norm: &[f32],
        token: &mut u32,
        d: usize,
        vocab: usize,
        pos: usize,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || model_id == 0
            || norm.len() < d
            || head.fmt != 15
            || head.i != d
            || head.o != vocab
            || [d, vocab, pos + 1]
                .into_iter()
                .any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        let mut raw = ColiMetalMatmulDescRaw {
            tensor: head.tensor,
            y: std::ptr::null_mut(),
            weights: head.weights.as_ptr() as *const c_void,
            scales: head.scales.as_ptr() as *const f32,
            fmt: 15,
            i: d as i32,
            o: vocab as i32,
            gs: 64,
                    // No per-descriptor activation on this path: every descriptor
            // reads the function-level shared `x`.
            x: std::ptr::null(),
            s: 1,
};
        let rc = unsafe {
            coli_metal_spark_token_end_top1(
                model_id,
                &mut raw,
                norm.as_ptr(),
                token as *mut u32,
                d as i32,
                vocab as i32,
                pos as i32,
                eps,
            )
        };
        head.tensor = raw.tensor;
        rc
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_token_end_logits(
        model_id: u64,
        head: &mut MetalWeightDesc<'_>,
        norm: &[f32],
        logits: &mut [f32],
        d: usize,
        vocab: usize,
        pos: usize,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || model_id == 0
            || norm.len() < d
            || logits.len() < vocab
            || [d, vocab, pos + 1]
                .into_iter()
                .any(|v| v > i32::MAX as usize)
            || head.fmt != 15
            || head.i != d
            || head.o != vocab
        {
            return 0;
        }
        let mut raw = ColiMetalMatmulDescRaw {
            tensor: head.tensor,
            y: std::ptr::null_mut(),
            weights: head.weights.as_ptr() as *const c_void,
            scales: head.scales.as_ptr() as *const f32,
            fmt: 15,
            i: d as i32,
            o: vocab as i32,
            gs: 64,
                    // No per-descriptor activation on this path: every descriptor
            // reads the function-level shared `x`.
            x: std::ptr::null(),
            s: 1,
};
        let rc = unsafe {
            coli_metal_spark_token_end_logits(
                model_id,
                &mut raw,
                norm.as_ptr(),
                logits.as_mut_ptr(),
                d as i32,
                vocab as i32,
                pos as i32,
                eps,
            )
        };
        head.tensor = raw.tensor;
        rc
    }

    pub fn spark_prefill_begin(model_id: u64, x: &[f32], s: usize, d: usize, base: usize) -> bool {
        if !metal_available()
            || model_id == 0
            || s <= 1
            || x.len() < s * d
            || [s, d, base + 1].into_iter().any(|v| v > i32::MAX as usize)
        {
            return false;
        }
        unsafe {
            coli_metal_spark_prefill_begin(model_id, x.as_ptr(), s as i32, d as i32, base as i32)
                == 1
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_prefill_layer_encode(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalWeightDesc<'_>],
        input_norm: &[f32],
        post_norm: &[f32],
        d: usize,
        inter: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sliding: bool,
        window: usize,
        base: usize,
        srows: usize,
        rotary_dim: usize,
        theta: f32,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || descs.len() != 6
            || input_norm.len() < d
            || post_norm.len() < d
            || [
                layer,
                d,
                inter,
                heads,
                kv_heads,
                head_dim,
                window,
                base + 1,
                srows,
                rotary_dim,
            ]
            .into_iter()
            .any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        let mut raw = Vec::with_capacity(6);
        for desc in descs.iter() {
            if desc.fmt != 15 || desc.i > i32::MAX as usize || desc.o > i32::MAX as usize {
                return 0;
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: desc.tensor,
                y: std::ptr::null_mut(),
                weights: desc.weights.as_ptr() as *const c_void,
                scales: desc.scales.as_ptr() as *const f32,
                fmt: 15,
                i: desc.i as i32,
                o: desc.o as i32,
                gs: 64,
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_spark_prefill_layer_encode(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                input_norm.as_ptr(),
                post_norm.as_ptr(),
                d as i32,
                inter as i32,
                heads as i32,
                kv_heads as i32,
                head_dim as i32,
                sliding as i32,
                window as i32,
                base as i32,
                srows as i32,
                rotary_dim as i32,
                theta,
                eps,
            )
        };
        for (desc, r) in descs.iter_mut().zip(raw.iter()) {
            desc.tensor = r.tensor;
        }
        rc
    }

    pub fn spark_prefill_end(model_id: u64, base: usize, srows: usize) -> i32 {
        if !metal_available()
            || model_id == 0
            || srows == 0
            || [base + 1, srows].into_iter().any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        unsafe { coli_metal_spark_prefill_end(model_id, base as i32, srows as i32) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_prefill_end_logits(
        model_id: u64,
        head: &mut MetalWeightDesc<'_>,
        norm: &[f32],
        logits: &mut [f32],
        d: usize,
        vocab: usize,
        base: usize,
        srows: usize,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || model_id == 0
            || norm.len() < d
            || logits.len() < vocab
            || head.fmt != 15
            || head.i != d
            || head.o != vocab
            || [d, vocab, base + 1, srows]
                .into_iter()
                .any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        let mut raw = ColiMetalMatmulDescRaw {
            tensor: head.tensor,
            y: std::ptr::null_mut(),
            weights: head.weights.as_ptr() as *const c_void,
            scales: head.scales.as_ptr() as *const f32,
            fmt: 15,
            i: d as i32,
            o: vocab as i32,
            gs: 64,
                    // No per-descriptor activation on this path: every descriptor
            // reads the function-level shared `x`.
            x: std::ptr::null(),
            s: 1,
};
        let rc = unsafe {
            coli_metal_spark_prefill_end_logits(
                model_id,
                &mut raw,
                norm.as_ptr(),
                logits.as_mut_ptr(),
                d as i32,
                vocab as i32,
                base as i32,
                srows as i32,
                eps,
            )
        };
        head.tensor = raw.tensor;
        rc
    }

    pub fn spark_prefill_abort(model_id: u64) {
        if model_id != 0 && metal_available() {
            unsafe { coli_metal_spark_prefill_abort(model_id) }
        }
    }

    pub fn spark_token_abort(model_id: u64) {
        if model_id != 0 && metal_available() {
            unsafe { coli_metal_spark_token_abort(model_id) };
        }
    }

    /// Full Spark-X2.5 dense layer decode in one Metal command buffer.
    /// `descs` = [qkv, attn_gate, out, mlp_gate, mlp_up, mlp_down], all fmt=15.
    /// Returns >0 on success, 0 on a pre-submit decline, <0 on a submitted GPU fault.
    #[allow(clippy::too_many_arguments)]
    pub fn spark_layer(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalWeightDesc<'_>],
        x: &mut [f32],
        input_norm: &[f32],
        post_norm: &[f32],
        d: usize,
        inter: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sliding: bool,
        window: usize,
        pos: usize,
        rotary_dim: usize,
        theta: f32,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || descs.len() != 6
            || x.len() < d
            || input_norm.len() < d
            || post_norm.len() < d
            || [
                d,
                inter,
                heads,
                kv_heads,
                head_dim,
                window,
                pos + 1,
                rotary_dim,
            ]
            .into_iter()
            .any(|v| v > i32::MAX as usize)
        {
            return 0;
        }
        let mut raw = Vec::with_capacity(6);
        for desc in descs.iter() {
            if desc.fmt != 15 || desc.i > i32::MAX as usize || desc.o > i32::MAX as usize {
                return 0;
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: desc.tensor,
                y: std::ptr::null_mut(),
                weights: desc.weights.as_ptr() as *const c_void,
                scales: desc.scales.as_ptr() as *const f32,
                fmt: 15,
                i: desc.i as i32,
                o: desc.o as i32,
                gs: 64,
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_spark_layer(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                x.as_mut_ptr(),
                input_norm.as_ptr(),
                post_norm.as_ptr(),
                d as i32,
                inter as i32,
                heads as i32,
                kv_heads as i32,
                head_dim as i32,
                sliding as i32,
                window as i32,
                pos as i32,
                rotary_dim as i32,
                theta,
                eps,
            )
        };
        for (desc, r) in descs.iter_mut().zip(raw.iter()) {
            desc.tensor = r.tensor;
        }
        rc
    }

    pub fn spark_drop_model(model_id: u64) {
        if model_id != 0 && metal_available() {
            unsafe { coli_metal_spark_drop_model(model_id) };
        }
    }

    /// Full one-command-buffer MXFP4 Qwen Gated DeltaNet decode. `descs`
    /// must be [qkv, z, a, b, out], all fmt=7. Persistent tensor handles are
    /// updated in place exactly like `metal_matmul_multi`. Returns >0 on
    /// success, 0 on a pre-submit decline, and <0 on a post-submit GPU fault.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_mxfp4(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalWeightDesc<'_>],
        x: &[f32],
        out: &mut [f32],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        output_gate: i32,
        eps: f32,
    ) -> i32 {
        if !metal_available()
            || model_id == 0
            || layer > i32::MAX as usize
            || descs.len() != 5
            || d == 0
            || d > i32::MAX as usize
            || kheads == 0
            || kheads > i32::MAX as usize
            || kd == 0
            || kd > i32::MAX as usize
            || vheads == 0
            || vheads > i32::MAX as usize
            || vd == 0
            || vd > i32::MAX as usize
            || kk == 0
            || kk > i32::MAX as usize
            || x.len() < d
            || out.len() < d
            || a_log.len() < vheads
            || dt_bias.len() < vheads
            || norm_w.len() < vd
            || !(eps > 0.0)
        {
            return 0;
        }
        let kdim = match kheads.checked_mul(kd) {
            Some(v) => v,
            None => return 0,
        };
        let vdim = match vheads.checked_mul(vd) {
            Some(v) => v,
            None => return 0,
        };
        let cdim = match kdim.checked_mul(2).and_then(|v| v.checked_add(vdim)) {
            Some(v) => v,
            None => return 0,
        };
        let conv_need = match cdim.checked_mul(kk) {
            Some(v) => v,
            None => return 0,
        };
        let state_need = match vheads.checked_mul(kd).and_then(|v| v.checked_mul(vd)) {
            Some(v) => v,
            None => return 0,
        };
        let conv_state_need = match cdim.checked_mul(kk.saturating_sub(1)) {
            Some(v) => v,
            None => return 0,
        };
        if conv_w.len() < conv_need
            || state.len() < state_need
            || conv_state.len() < conv_state_need
        {
            return 0;
        }
        let mut raw = Vec::with_capacity(5);
        for dsc in descs.iter_mut() {
            if (dsc.fmt != 5
                && dsc.fmt != 7
                && dsc.fmt != 9
                && dsc.fmt != 10
                && dsc.fmt != 11
                && dsc.fmt != 12
                && dsc.fmt != 13
                && dsc.fmt != 14
                && !(16..=24).contains(&dsc.fmt))
                || dsc.i == 0
                || dsc.o == 0
                || dsc.i > i32::MAX as usize
                || dsc.o > i32::MAX as usize
            {
                return 0;
            }
            let (weight_bytes, scale_bytes) = if dsc.fmt == 5 {
                (
                    dsc.o
                        .saturating_mul(dsc.i)
                        .saturating_mul(std::mem::size_of::<u16>()),
                    0,
                )
            } else if (16..=24).contains(&dsc.fmt) {
                // 16..20 are BF16-sidecar MLX affine (20 is the experimental Q4
                // FMA variant); 21..24 are the same widths with IEEE fp16
                // sidecars, which is what an FP16 checkpoint produces. Widening
                // this list makes the single-command-buffer full-GDN path
                // reachable for such a checkpoint (EXP-049).
                let bits = match dsc.fmt {
                    16 | 20 | 21 => 4usize,
                    17 | 22 => 5,
                    18 | 23 => 6,
                    19 | 24 => 8,
                    _ => unreachable!(),
                };
                if dsc.group_size == 0
                    || dsc.group_size > i32::MAX as usize
                    || dsc.i % dsc.group_size != 0
                    || dsc.i.checked_mul(bits).is_none_or(|v| v % 32 != 0)
                {
                    return 0;
                }
                (
                    dsc.o.saturating_mul(dsc.i.saturating_mul(bits) / 8),
                    2usize
                        .saturating_mul(dsc.o)
                        .saturating_mul(dsc.i / dsc.group_size)
                        .saturating_mul(std::mem::size_of::<u16>()),
                )
            } else if (11..=14).contains(&dsc.fmt) {
                let block = match dsc.fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    _ => unreachable!(),
                };
                let base = dsc.o.saturating_mul(dsc.i);
                let weights = if dsc.fmt == 14 {
                    base.saturating_add(dsc.o.saturating_mul(dsc.i.div_ceil(32)).saturating_mul(3))
                } else {
                    base
                };
                (
                    weights,
                    dsc.o
                        .saturating_mul(dsc.i.div_ceil(block))
                        .saturating_mul(std::mem::size_of::<f32>()),
                )
            } else {
                let planes = if dsc.fmt == 10 {
                    3
                } else if dsc.fmt == 9 {
                    2
                } else {
                    1
                };
                (
                    planes * dsc.o.saturating_mul(dsc.i.div_ceil(2)),
                    planes * dsc.o.saturating_mul(dsc.i.div_ceil(32)),
                )
            };
            if dsc.weights.len() < weight_bytes || dsc.scales.len() < scale_bytes {
                return 0;
            }
            static BF16_DUMMY_SCALE_GDN: [f32; 1] = [1.0];
            let scale_ptr = if dsc.fmt == 5 {
                BF16_DUMMY_SCALE_GDN.as_ptr()
            } else {
                dsc.scales.as_ptr() as *const f32
            };
            raw.push(ColiMetalMatmulDescRaw {
                tensor: dsc.tensor,
                y: std::ptr::null_mut(),
                weights: dsc.weights.as_ptr() as *const c_void,
                scales: scale_ptr,
                fmt: dsc.fmt,
                i: dsc.i as i32,
                o: dsc.o as i32,
                gs: match dsc.fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    16..=24 => dsc.group_size as i32,
                    _ => 0,
                },
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_gdn_mxfp4(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                x.as_ptr(),
                out.as_mut_ptr(),
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_w.as_ptr(),
                norm_w.as_ptr(),
                state.as_mut_ptr(),
                conv_state.as_mut_ptr(),
                d as i32,
                kheads as i32,
                kd as i32,
                vheads as i32,
                vd as i32,
                kk as i32,
                output_gate,
                eps,
            )
        };
        for (dsc, r) in descs.iter_mut().zip(raw.iter()) {
            dsc.tensor = r.tensor;
        }
        rc
    }

    pub fn gdn_mxfp4_drop_model(model_id: u64) {
        if model_id != 0 {
            unsafe { coli_metal_gdn_mxfp4_drop_model(model_id) };
        }
    }

    /// Qwen4 HyperConnection projection island. The caller supplies the
    /// canonical grouped-RMSNorm result so the existing f64 reduction order is
    /// preserved during qualification. The down/up/injection projections and
    /// all activation/mixing intermediates stay inside one Metal command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix(
        model_id: u64,
        descs: &mut [MetalWeightDesc<'_>],
        normed: &[f32],
        out: &mut [f32],
        inject: Option<&mut [f32]>,
        d: usize,
        hc: usize,
        lr: usize,
    ) -> Result<Option<()>, ()> {
        if !metal_available()
            || model_id == 0
            || !(descs.len() == 2 || descs.len() == 3)
            || d == 0
            || hc == 0
            || lr == 0
            || d > i32::MAX as usize
            || hc > i32::MAX as usize
            || lr > i32::MAX as usize
        {
            return Ok(None);
        }
        let Some(hcd) = d.checked_mul(hc) else {
            return Ok(None);
        };
        if hcd > i32::MAX as usize || normed.len() < hcd || out.len() < d {
            return Ok(None);
        }
        if descs.len() == 3 && inject.as_ref().is_none_or(|v| v.len() < hc) {
            return Ok(None);
        }
        let expected = [(hcd, lr), (lr, hcd), (hcd, hc)];
        let mut raw = Vec::with_capacity(descs.len());
        for (dsc, &(ei, eo)) in descs.iter_mut().zip(expected.iter()) {
            if dsc.i != ei || dsc.o != eo {
                return Ok(None);
            }
            let (weight_bytes, scale_bytes) = if dsc.fmt == 5 {
                (
                    dsc.o
                        .saturating_mul(dsc.i)
                        .saturating_mul(std::mem::size_of::<u16>()),
                    0,
                )
            } else if (16..=24).contains(&dsc.fmt) {
                let bits = match dsc.fmt {
                    16 | 20 | 21 => 4usize,
                    17 | 22 => 5,
                    18 | 23 => 6,
                    19 | 24 => 8,
                    _ => unreachable!(),
                };
                if dsc.group_size == 0
                    || dsc.i % dsc.group_size != 0
                    || dsc.i.checked_mul(bits).is_none_or(|v| v % 32 != 0)
                {
                    return Ok(None);
                }
                (
                    dsc.o.saturating_mul(dsc.i.saturating_mul(bits) / 8),
                    2usize
                        .saturating_mul(dsc.o)
                        .saturating_mul(dsc.i / dsc.group_size)
                        .saturating_mul(std::mem::size_of::<u16>()),
                )
            } else if (11..=14).contains(&dsc.fmt) {
                let block = match dsc.fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    _ => unreachable!(),
                };
                let base = dsc.o.saturating_mul(dsc.i);
                let weights = if dsc.fmt == 14 {
                    base.saturating_add(dsc.o.saturating_mul(dsc.i.div_ceil(32)).saturating_mul(3))
                } else {
                    base
                };
                (
                    weights,
                    dsc.o
                        .saturating_mul(dsc.i.div_ceil(block))
                        .saturating_mul(std::mem::size_of::<f32>()),
                )
            } else if matches!(dsc.fmt, 7 | 9 | 10) {
                let planes = if dsc.fmt == 10 {
                    3
                } else if dsc.fmt == 9 {
                    2
                } else {
                    1
                };
                (
                    planes * dsc.o.saturating_mul(dsc.i.div_ceil(2)),
                    planes * dsc.o.saturating_mul(dsc.i.div_ceil(32)),
                )
            } else {
                return Ok(None);
            };
            if dsc.weights.len() < weight_bytes || dsc.scales.len() < scale_bytes {
                return Ok(None);
            }
            static BF16_DUMMY_SCALE_HC: [f32; 1] = [1.0];
            let scale_ptr = if dsc.fmt == 5 {
                BF16_DUMMY_SCALE_HC.as_ptr()
            } else {
                dsc.scales.as_ptr() as *const f32
            };
            raw.push(ColiMetalMatmulDescRaw {
                tensor: dsc.tensor,
                y: std::ptr::null_mut(),
                weights: dsc.weights.as_ptr() as *const c_void,
                scales: scale_ptr,
                fmt: dsc.fmt,
                i: dsc.i as i32,
                o: dsc.o as i32,
                gs: match dsc.fmt {
                    11 | 14 => 32,
                    12 => 16,
                    13 => 8,
                    16..=24 => dsc.group_size as i32,
                    _ => 0,
                },
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let inject_ptr = inject
            .map(|v| v.as_mut_ptr())
            .unwrap_or(std::ptr::null_mut());
        let rc = unsafe {
            coli_metal_hc_mix(
                model_id,
                raw.as_mut_ptr(),
                raw.len() as i32,
                normed.as_ptr(),
                out.as_mut_ptr(),
                inject_ptr,
                d as i32,
                hc as i32,
                lr as i32,
            )
        };
        for (dsc, r) in descs.iter_mut().zip(raw.iter()) {
            dsc.tensor = r.tensor;
        }
        match rc {
            r if r > 0 => Ok(Some(())),
            0 => Ok(None),
            _ => Err(()),
        }
    }

    pub fn hc_drop_model(model_id: u64) {
        if model_id != 0 {
            unsafe { coli_metal_hc_drop_model(model_id) };
        }
    }

    /// Full one-command-buffer MXFP4 shared MLP. `descs` are
    /// [gate_proj, up_proj, down_proj]. The checkpoint's scalar shared gate is
    /// computed separately by the caller because it may use BF16. Returns
    /// Some(()) on success, None on pre-submit decline; post-submit faults are
    /// reported as Err.
    pub fn shared_mxfp4(
        model_id: u64,
        layer: usize,
        descs: &mut [MetalWeightDesc<'_>],
        x: &[f32],
        out: &mut [f32],
        d: usize,
        iinter: usize,
    ) -> Result<Option<()>, ()> {
        if !metal_available()
            || model_id == 0
            || layer > i32::MAX as usize
            || descs.len() != 3
            || d == 0
            || d > i32::MAX as usize
            || iinter == 0
            || iinter > i32::MAX as usize
            || x.len() < d
            || out.len() < d
        {
            return Ok(None);
        }
        let expected = [(d, iinter), (d, iinter), (iinter, d)];
        let mut raw = Vec::with_capacity(3);
        for (dsc, &(ei, eo)) in descs.iter_mut().zip(expected.iter()) {
            if dsc.i != ei || dsc.o != eo {
                return Ok(None);
            }
            // fmt 7 = MXFP4; 21..24 = MLX affine with fp16 sidecars. The sizing
            // MUST branch by format: the MXFP4 sizes (i/2 per row, one scale byte
            // per 32) are far smaller than the affine sizes, so reusing them for
            // an affine descriptor would let the length check pass while the C
            // side reads using its own larger stride -- an out-of-bounds read.
            let (weight_bytes, scale_bytes, fmt) = if dsc.fmt == 7 {
                (
                    dsc.o.saturating_mul(dsc.i.div_ceil(2)),
                    dsc.o.saturating_mul(dsc.i.div_ceil(32)),
                    7,
                )
            } else {
                let bits: usize = match dsc.fmt {
                    21 => 4,
                    22 => 5,
                    23 => 6,
                    24 => 8,
                    _ => return Ok(None),
                };
                if dsc.group_size == 0 || dsc.i % dsc.group_size != 0 {
                    return Ok(None);
                }
                (
                    dsc.o.saturating_mul((dsc.i * bits).div_ceil(8)),
                    // bf16/fp16 scale + bias per group, 2 bytes each.
                    2 * dsc.o.saturating_mul(dsc.i / dsc.group_size) * 2,
                    dsc.fmt,
                )
            };
            if dsc.weights.len() < weight_bytes || dsc.scales.len() < scale_bytes {
                return Ok(None);
            }
            raw.push(ColiMetalMatmulDescRaw {
                tensor: dsc.tensor,
                y: std::ptr::null_mut(),
                weights: dsc.weights.as_ptr() as *const c_void,
                scales: dsc.scales.as_ptr() as *const f32,
                fmt,
                i: dsc.i as i32,
                o: dsc.o as i32,
                gs: if fmt == 7 { 0 } else { dsc.group_size as i32 },
                            // No per-descriptor activation on this path: every descriptor
                // reads the function-level shared `x`.
                x: std::ptr::null(),
                s: 1,
});
        }
        let rc = unsafe {
            coli_metal_shared_mxfp4(
                model_id,
                layer as i32,
                raw.as_mut_ptr(),
                raw.len() as i32,
                x.as_ptr(),
                out.as_mut_ptr(),
                d as i32,
                iinter as i32,
            )
        };
        for (dsc, r) in descs.iter_mut().zip(raw.iter()) {
            dsc.tensor = r.tensor;
        }
        match rc {
            r if r > 0 => Ok(Some(())),
            0 => Ok(None),
            _ => Err(()),
        }
    }

    pub fn shared_mxfp4_drop_model(model_id: u64) {
        if model_id != 0 {
            unsafe { coli_metal_shared_mxfp4_drop_model(model_id) };
        }
    }

    /// CPU BF16 GEMV through Accelerate/BNNS. No weight copy is retained.
    pub fn bnns_bf16_matmul(w: &[u8], x: &[f32], y: &mut [f32], o: usize, i: usize) -> bool {
        if w.len() < o * i * 2 || x.len() < i || y.len() < o {
            return false;
        }
        unsafe {
            coli_bnns_bf16_matmul(
                w.as_ptr() as *const u16,
                x.as_ptr(),
                y.as_mut_ptr(),
                o as i32,
                i as i32,
            ) == 1
        }
    }

    /// CPU BF16 batched dense projection through BNNS. `x`/`y` are row-major
    /// `[rows, i]` and `[rows, o]`; the BF16 weight matrix remains caller-owned.
    pub fn bnns_bf16_matmul_batch(
        w: &[u8],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        o: usize,
        i: usize,
    ) -> bool {
        if rows == 0
            || rows > i32::MAX as usize
            || o > i32::MAX as usize
            || i > i32::MAX as usize
            || w.len() < o.saturating_mul(i).saturating_mul(2)
            || x.len() < rows.saturating_mul(i)
            || y.len() < rows.saturating_mul(o)
        {
            return false;
        }
        unsafe {
            coli_bnns_bf16_matmul_batch(
                w.as_ptr() as *const u16,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows as i32,
                o as i32,
                i as i32,
            ) == 1
        }
    }

    /// In-place rmsnorm over nrows rows of n. Returns true on GPU success.
    pub fn metal_rmsnorm(x: &mut [f32], w: &[f32], n: usize, nrows: usize, eps: f32) -> bool {
        if !metal_available() || x.len() < n * nrows || w.len() < n {
            return false;
        }
        unsafe { coli_metal_rmsnorm(x.as_mut_ptr(), w.as_ptr(), n as i32, nrows as i32, eps) == 1 }
    }

    /// y += a. Returns true on GPU success.
    pub fn metal_add(y: &mut [f32], a: &[f32]) -> bool {
        if !metal_available() || y.len() != a.len() {
            return false;
        }
        unsafe { coli_metal_add(y.as_mut_ptr(), a.as_ptr(), y.len() as i32) == 1 }
    }

    /// g *= silu(u), in place. Returns true on GPU success.
    pub fn metal_silu_mul(g: &mut [f32], u: &[f32]) -> bool {
        if !metal_available() || g.len() != u.len() {
            return false;
        }
        unsafe { coli_metal_silu_mul(g.as_mut_ptr(), u.as_ptr(), g.len() as i32) == 1 }
    }

    // -------------------------------------------------------------------------
    // MetalIO: async NVMe -> MTLBuffer expert streaming (from metalio.mm)
    // Never mandatory: every fn returns an error/0 and the caller falls back to
    // the pread path.
    // -------------------------------------------------------------------------

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ColiMetalioRegion {
        pub file: i32,
        pub src_off: u64,
        pub bytes: usize,
        pub dst_off: u64,
    }

    unsafe extern "C" {
        pub fn metalio_init() -> i32;
        pub fn metalio_active() -> i32;
        pub fn metalio_shutdown();
        pub fn metalio_file_add(path: *const std::os::raw::c_char) -> i32;
        pub fn metalio_slot_alloc(max_bytes: usize) -> i32;
        pub fn metalio_slot_free(slot: i32);
        pub fn metalio_slot_ptr(slot: i32) -> *mut std::os::raw::c_void;
        pub fn metalio_slot_bytes(slot: i32) -> usize;
        pub fn metalio_loadv(
            slot: i32,
            regions: *const ColiMetalioRegion,
            count: i32,
            kind: i32,
        ) -> i64;
        pub fn metalio_wait(event_value: i64) -> i32;
        pub fn metalio_batch_barrier() -> i64;
        pub fn metalio_batch_wait(event_value: i64, slots: *const i32, count: i32) -> i32;
        pub fn metalio_slot_consumed(slot: i32);
        pub fn metalio_prefetch_demanded(slot: i32);
        pub fn metalio_stats(out: *mut ColiMetalioStats);
    }

    /// MetalIO streaming counters (mirror of the C ColiMetalioStats).
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ColiMetalioStats {
        pub loads: u64,
        pub bytes: u64,
        pub waits: u64,
        pub fails: u64,
        pub prefetch_loads: u64,
        pub prefetch_used: u64,
        pub prefetch_wasted: u64,
        pub prefetch_ready_at_demand: u64,
        pub prefetch_late_at_demand: u64,
        pub outstanding: u64,
        pub peak_outstanding: u64,
        pub latency_samples: u64,
        pub total_latency_s: f64,
        pub lat_hist: [u64; 32],
    }

    pub fn mio_init() -> bool {
        static INIT: std::sync::Once = std::sync::Once::new();
        static mut ACTIVE: bool = false;
        INIT.call_once(|| {
            let ok = unsafe { metalio_init() } == 1;
            unsafe { ACTIVE = ok };
        });
        mio_active()
    }

    pub fn mio_active() -> bool {
        unsafe { metalio_active() == 1 }
    }

    /// Insert one queue-level completion point after all MetalIO loads issued
    /// so far. On a concurrent MTLIO queue this is the safe way to wait for a
    /// whole expert batch without assuming per-load event values complete in
    /// order.
    pub fn mio_batch_barrier() -> Option<i64> {
        if !mio_active() {
            return None;
        }
        let event = unsafe { metalio_batch_barrier() };
        (event > 0).then_some(event)
    }

    pub fn mio_batch_wait(event_value: i64, slots: &[i32]) -> bool {
        event_value > 0
            && unsafe { metalio_batch_wait(event_value, slots.as_ptr(), slots.len() as i32) == 0 }
    }

    /// The crate keeps ONE MTLIOFileHandle per shard file for the process
    /// lifetime. The native table is deliberately large enough for first-class
    /// sharded safetensors checkpoints (Qwen3.8-class exports can exceed 128
    /// shards); re-adding per miss would still waste handles and eventually
    /// force a POSIX fallback.
    static MIO_FILES: std::sync::Mutex<Option<std::collections::HashMap<String, i32>>> =
        std::sync::Mutex::new(None);

    /// Register a shard file once; returns its MetalIO file id (cached).
    /// None = MetalIO unavailable or the handle failed (caller falls back).
    pub fn mio_file(path: &str) -> Option<i32> {
        if !mio_init() {
            return None;
        }
        let mut guard = MIO_FILES.lock().unwrap();
        let map = guard.get_or_insert_with(std::collections::HashMap::new);
        if let Some(&fid) = map.get(path) {
            return Some(fid);
        }
        let cpath = std::ffi::CString::new(path).ok()?;
        let fid = unsafe { metalio_file_add(cpath.as_ptr()) };
        if fid < 0 {
            return None;
        }
        map.insert(path.to_string(), fid);
        Some(fid)
    }

    /// Engine-neutral MetalIO range. Unlike the legacy COLI helper, one
    /// request may span several source files and place each range at an
    /// arbitrary destination offset. That is what lets native .logan,
    /// safetensors/MLX, and legacy package readers share the same I/O engine.
    #[derive(Debug, Clone, Copy)]
    pub struct MioRegion {
        pub file: i32,
        pub src_off: u64,
        pub bytes: usize,
        pub dst_off: usize,
    }

    fn mio_load_regions_kind(regions: &[MioRegion], kind: i32) -> Option<(i32, i64)> {
        if !mio_init() || regions.is_empty() {
            return None;
        }
        let capacity = regions.iter().try_fold(0usize, |high, region| {
            region
                .dst_off
                .checked_add(region.bytes)
                .map(|end| high.max(end))
        })?;
        if capacity == 0 {
            return None;
        }
        let slot = unsafe { metalio_slot_alloc(capacity) };
        if slot < 0 {
            return None;
        }
        let native: Vec<ColiMetalioRegion> = regions
            .iter()
            .map(|region| ColiMetalioRegion {
                file: region.file,
                src_off: region.src_off,
                bytes: region.bytes,
                dst_off: region.dst_off as u64,
            })
            .collect();
        let event = unsafe { metalio_loadv(slot, native.as_ptr(), native.len() as i32, kind) };
        if event < 0 {
            unsafe { metalio_slot_free(slot) };
            return None;
        }
        Some((slot, event))
    }

    /// Submit arbitrary file ranges through MetalIO. `speculative` controls
    /// RouteScout accounting only; the physical I/O path is identical.
    pub fn mio_load_regions(
        regions: &[MioRegion],
        speculative: bool,
    ) -> Option<(i32, i64)> {
        mio_load_regions_kind(regions, if speculative { 2 } else { 1 })
    }

    /// Turn one completed MetalIO slot back into ordinary bytes and release the
    /// slot. Raw MLX currently copies from shared MTLBuffer into its native
    /// quantized `Wt` representation before dispatch; the I/O itself remains
    /// asynchronous and bypasses the POSIX page-cache path.
    /// Exact `wrap()` accounting: `(calls, zero_copy_calls, copied_bytes)`.
    /// Distinguishes a real memcpy cost from MTLBuffer-creation churn, which a
    /// timing probe cannot do at this host's noise level.
    pub fn wrap_stats() -> (u64, u64, u64, u64) {
        let mut calls = 0u64;
        let mut zc = 0u64;
        let mut bytes = 0u64;
        let mut created = 0u64;
        unsafe { coli_metal_wrap_stats(&mut calls, &mut zc, &mut bytes, &mut created) };
        (calls, zc, bytes, created)
    }

    pub fn mio_finish_slot(
        slot: i32,
        event: i64,
        used_bytes: usize,
        speculative: bool,
    ) -> Option<Vec<u8>> {
        if slot < 0 || event <= 0 {
            return None;
        }
        if speculative {
            unsafe { metalio_prefetch_demanded(slot) };
        }
        if unsafe { metalio_wait(event) } != 0 {
            unsafe { metalio_slot_free(slot) };
            return None;
        }
        let capacity = unsafe { metalio_slot_bytes(slot) };
        if used_bytes > capacity {
            unsafe { metalio_slot_free(slot) };
            return None;
        }
        let ptr = unsafe { metalio_slot_ptr(slot) } as *const u8;
        if ptr.is_null() {
            unsafe { metalio_slot_free(slot) };
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, used_bytes) }.to_vec();
        if speculative {
            unsafe { metalio_slot_consumed(slot) };
        }
        unsafe { metalio_slot_free(slot) };
        Some(bytes)
    }

    /// Release a speculative/demand slot without consuming it. MetalIO records
    /// an unused speculative slot as wasted, which is useful RouteScout data.
    pub fn mio_discard_slot(slot: i32) {
        if slot >= 0 {
            unsafe { metalio_slot_free(slot) };
        }
    }

    #[cfg(test)]
    mod metalio_range_tests {
        use super::{mio_file, mio_finish_slot, mio_load_regions, MioRegion};

        #[test]
        fn vectored_load_can_span_two_source_files() {
            let root = std::env::temp_dir().join(format!(
                "logan-metalio-ranges-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let a = root.join("a.bin");
            let b = root.join("b.bin");
            let a_bytes: Vec<u8> = (0..=127).collect();
            let b_bytes: Vec<u8> = (128..=255).collect();
            std::fs::write(&a, &a_bytes).unwrap();
            std::fs::write(&b, &b_bytes).unwrap();

            let Some(a_id) = mio_file(a.to_str().unwrap()) else {
                // MetalIO is an optional platform capability even on macOS;
                // the production path falls back rather than treating this as
                // model corruption.
                let _ = std::fs::remove_dir_all(&root);
                return;
            };
            let Some(b_id) = mio_file(b.to_str().unwrap()) else {
                let _ = std::fs::remove_dir_all(&root);
                return;
            };
            let regions = [
                MioRegion {
                    file: a_id,
                    src_off: 11,
                    bytes: 13,
                    dst_off: 0,
                },
                MioRegion {
                    file: b_id,
                    src_off: 7,
                    bytes: 17,
                    dst_off: 13,
                },
            ];
            let (slot, event) = mio_load_regions(&regions, false).expect("MetalIO range load");
            let got = mio_finish_slot(slot, event, 30, false).expect("MetalIO completion");
            let mut expected = a_bytes[11..24].to_vec();
            expected.extend_from_slice(&b_bytes[7..24]);
            assert_eq!(got, expected);

            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// Stream (offset, bytes) regions of one expert into a fresh slot, packed
    /// contiguously (dst offsets 0..total) so a single `moe_topk`/`swiglu`
    /// submission can consume the expert. Returns (slot, event) on success.
    /// The caller owns the slot until it frees it (or drops it into a cache).
    fn mio_load_expert_kind(fid: i32, regions: &[(u64, usize)], kind: i32) -> Option<(i32, i64)> {
        if !mio_init() || regions.is_empty() {
            return None;
        }
        let total: usize = regions.iter().map(|r| r.1).sum();
        let slot = unsafe { metalio_slot_alloc(total) };
        if slot < 0 {
            return None;
        }
        let coalesce = std::env::var("QWEN_MIO_COALESCE_EXPERT")
            .map(|v| v != "0")
            .unwrap_or(true);
        let contiguous = regions
            .windows(2)
            .all(|w| w[0].0.checked_add(w[0].1 as u64) == Some(w[1].0));
        let mut cr: Vec<ColiMetalioRegion> = if coalesce && contiguous {
            vec![ColiMetalioRegion {
                file: fid,
                src_off: regions[0].0,
                bytes: total,
                dst_off: 0,
            }]
        } else {
            let mut dst = 0usize;
            regions
                .iter()
                .map(|(off, len)| {
                    let r = ColiMetalioRegion {
                        file: fid,
                        src_off: *off,
                        bytes: *len,
                        dst_off: dst as u64,
                    };
                    dst += len;
                    r
                })
                .collect()
        };
        let ev = unsafe { metalio_loadv(slot, cr.as_mut_ptr(), cr.len() as i32, kind) };
        if ev < 0 {
            unsafe { metalio_slot_free(slot) };
            return None;
        }
        Some((slot, ev))
    }

    /// Demand/async expert load used by the canonical routed path.
    pub fn mio_load_expert(fid: i32, regions: &[(u64, usize)]) -> Option<(i32, i64)> {
        mio_load_expert_kind(fid, regions, 1) // MIO_LOAD_ASYNC
    }

    /// Speculative expert load. Same physical path, but tagged so MetalIO
    /// telemetry can distinguish prediction work from demand work.
    pub fn mio_prefetch_expert(fid: i32, regions: &[(u64, usize)]) -> Option<(i32, i64)> {
        mio_load_expert_kind(fid, regions, 2) // MIO_LOAD_SPEC
    }

    /// MetalIO streaming counters (loads/bytes/waits/fails + prefetch).
    pub fn mio_stats() -> ColiMetalioStats {
        let mut s: ColiMetalioStats = Default::default();
        if mio_active() {
            unsafe { metalio_stats(&mut s) };
        }
        s
    }

    // -------------------------------------------------------------------------
    // Direct Apple8 execution seam (apple8_metalio_direct.mm) — kernels run on
    // tile bytes already resident in a MetalIO slot, native tile order, no
    // host decode, no repack.
    // -------------------------------------------------------------------------

    /// One expert's three matrices inside one MetalIO slot (contiguous packing
    /// as produced by `mio_load_expert`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ColiApple8MetalioExpert {
        pub slot: i32,
        pub gate_offset: usize,
        pub gate_bytes: usize,
        pub up_offset: usize,
        pub up_bytes: usize,
        pub down_offset: usize,
        pub down_bytes: usize,
    }

    unsafe extern "C" {
        pub fn coli_apple8_metalio_direct_init() -> i32;
        pub fn coli_apple8_metalio_direct_shutdown();
        pub fn coli_apple8_metalio_matmul_slot(
            slot: i32,
            slot_offset: usize,
            matrix_bytes: usize,
            x: *const f32,
            y: *mut f32,
            s: i32,
            i: i32,
            o: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_swiglu_slot(
            slot: i32,
            gate_offset: usize,
            gate_bytes: usize,
            up_offset: usize,
            up_bytes: usize,
            down_offset: usize,
            down_bytes: usize,
            x: *const f32,
            y: *mut f32,
            s: i32,
            hidden: i32,
            intermediate: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_rows(
            experts: *const ColiApple8MetalioExpert,
            row_offsets: *const i32,
            expert_count: i32,
            x: *const f32,
            y: *mut f32,
            total_rows: i32,
            hidden: i32,
            intermediate: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk(
            experts: *const ColiApple8MetalioExpert,
            route_weights: *const f32,
            expert_count: i32,
            x: *const f32,
            y: *mut f32,
            hidden: i32,
            intermediate: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk_begin(
            experts: *const ColiApple8MetalioExpert,
            route_weights: *const f32,
            expert_count: i32,
            x: *const f32,
            hidden: i32,
            intermediate: i32,
            pending_out: *mut *mut c_void,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk_finish(pending: *mut c_void, y: *mut f32) -> i32;
        pub fn coli_apple8_metalio_moe_topk_discard(pending: *mut c_void);
        pub fn coli_apple8_metalio_profile_get(
            encode_ns: *mut u64,
            submit_ns: *mut u64,
            wait_ns: *mut u64,
            kernel_ns: *mut u64,
            fused_calls: *mut u64,
            fused_experts: *mut u64,
        );
        pub fn coli_apple8_metalio_profile_detail_get(
            gdn_wait_ns: *mut u64,
            gdn_kernel_ns: *mut u64,
            gdn_calls: *mut u64,
            moe_wait_ns: *mut u64,
            moe_kernel_ns: *mut u64,
            moe_calls: *mut u64,
        );
        // Generic BF16 GEMV (S x I rows -> O): attention q/k/v/o projections.
        pub fn coli_apple8_metalio_bf16_matmul(
            w: *const u16,
            x: *const f32,
            y: *mut f32,
            s: i32,
            o: i32,
            i: i32,
        ) -> i32;
        // GDN (coalesced Metal kernels, qwen_moe.c seam contract)
        pub fn coli_apple8_metalio_gdn_drop_model(model_id: u64);
        pub fn coli_apple8_metalio_ane_fence_create(
            value: u64,
            shared_backing_out: *mut *mut c_void,
        ) -> *mut c_void;
        pub fn coli_apple8_metalio_ane_fence_set_value(handle: *mut c_void, value: u64) -> i32;
        pub fn coli_apple8_metalio_ane_fence_metal_event(handle: *mut c_void) -> *mut c_void;
        pub fn coli_apple8_metalio_ane_fence_free(handle: *mut c_void);
        pub fn coli_apple8_metalio_gdn_begin(
            model_id: u64,
            layer: i32,
            x: *const f32,
            wqkv: *const u16,
            wz: *const u16,
            wa: *const u16,
            wb: *const u16,
            wout: *const u16,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_w: *const f32,
            norm_w: *const f32,
            state: *mut f32,
            conv_state: *mut f32,
            d: i32,
            kheads: i32,
            kd: i32,
            vheads: i32,
            vd: i32,
            kk: i32,
            output_gate: i32,
            eps: f32,
            pending_out: *mut *mut c_void,
        ) -> i32;
        pub fn coli_apple8_metalio_gdn_ane_begin(
            model_id: u64,
            layer: i32,
            qkv_surface: *mut c_void,
            qkv_bytes: usize,
            z_surface: *mut c_void,
            z_bytes: usize,
            a_surface: *mut c_void,
            a_bytes: usize,
            b_surface: *mut c_void,
            b_bytes: usize,
            spatial: i32,
            wqkv: *const u16,
            wz: *const u16,
            wa: *const u16,
            wb: *const u16,
            wout: *const u16,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_w: *const f32,
            norm_w: *const f32,
            state: *mut f32,
            conv_state: *mut f32,
            d: i32,
            kheads: i32,
            kd: i32,
            vheads: i32,
            vd: i32,
            kk: i32,
            output_gate: i32,
            eps: f32,
            ane_fence: *mut c_void,
            pending_out: *mut *mut c_void,
        ) -> i32;
        pub fn coli_apple8_metalio_gdn_finish(pending: *mut c_void, out: *mut f32) -> i32;
        pub fn coli_apple8_metalio_gdn_discard(pending: *mut c_void);
        pub fn coli_apple8_metalio_gdn_token(
            model_id: u64,
            layer: i32,
            x: *const f32,
            out: *mut f32,
            wqkv: *const u16,
            wz: *const u16,
            wa: *const u16,
            wb: *const u16,
            wout: *const u16,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_w: *const f32,
            norm_w: *const f32,
            state: *mut f32,
            conv_state: *mut f32,
            d: i32,
            kheads: i32,
            kd: i32,
            vheads: i32,
            vd: i32,
            kk: i32,
            output_gate: i32,
            eps: f32,
        ) -> i32;
    }

    /// Bring up the direct path (Metal device + command queue + pipelines).
    /// Requires MetalIO active (the C contract: slots are the weight source).
    static DIRECT_INIT: std::sync::Once = std::sync::Once::new();
    static mut DIRECT_OK: bool = false;

    pub fn direct_init() -> bool {
        if !mio_init() {
            return false;
        }
        DIRECT_INIT.call_once(|| {
            let ok = unsafe { coli_apple8_metalio_direct_init() } == 1;
            unsafe { DIRECT_OK = ok };
        });
        direct_available()
    }

    pub fn direct_available() -> bool {
        unsafe { DIRECT_OK }
    }

    /// Batched rows through one slot-resident Apple8 expert. `x` and `y` are
    /// row-major `[rows, hidden]`. The native kernel reuses the expert weights
    /// across all rows in one command buffer.
    pub fn swiglu_slot_batch(
        expert: &ColiApple8MetalioExpert,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        if !direct_available()
            || rows == 0
            || x.len() < rows.saturating_mul(hidden)
            || y.len() < rows.saturating_mul(hidden)
            || rows > i32::MAX as usize
            || hidden > i32::MAX as usize
            || intermediate > i32::MAX as usize
        {
            return false;
        }
        unsafe {
            coli_apple8_metalio_swiglu_slot(
                expert.slot,
                expert.gate_offset,
                expert.gate_bytes,
                expert.up_offset,
                expert.up_bytes,
                expert.down_offset,
                expert.down_bytes,
                x.as_ptr(),
                y.as_mut_ptr(),
                rows as i32,
                hidden as i32,
                intermediate as i32,
            ) == 1
        }
    }

    /// Prefill routed-MoE batch. `x` is grouped by expert according to
    /// `row_offsets`; native code encodes all expert groups into one command
    /// buffer and writes corresponding unweighted expert outputs to `y`.
    pub fn moe_rows(
        experts: &[ColiApple8MetalioExpert],
        row_offsets: &[i32],
        x: &[f32],
        y: &mut [f32],
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        if !direct_available()
            || experts.is_empty()
            || row_offsets.len() != experts.len() + 1
            || row_offsets.first().copied() != Some(0)
        {
            return false;
        }
        let Some(total_rows) = row_offsets.last().copied() else {
            return false;
        };
        if total_rows <= 0
            || x.len() < total_rows as usize * hidden
            || y.len() < total_rows as usize * hidden
            || experts.len() > i32::MAX as usize
            || hidden > i32::MAX as usize
            || intermediate > i32::MAX as usize
        {
            return false;
        }
        unsafe {
            coli_apple8_metalio_moe_rows(
                experts.as_ptr(),
                row_offsets.as_ptr(),
                experts.len() as i32,
                x.as_ptr(),
                y.as_mut_ptr(),
                total_rows,
                hidden as i32,
                intermediate as i32,
            ) == 1
        }
    }

    /// Fused decode-only routed layer: for K experts submits ONE command
    /// buffer (gate+up+swiglu -> down -> deterministic K-order weighted
    /// reduce) and waits once. `experts` in top-k order; `weights[i]` is the
    /// pre-renormalized route weight for experts[i] (C contract: consumed in
    /// caller order). K <= 64. Returns false -> caller runs the CPU path.
    pub fn moe_topk(
        experts: &[ColiApple8MetalioExpert],
        weights: &[f32],
        x: &[f32],
        y: &mut [f32],
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        if !direct_available()
            || experts.is_empty()
            || experts.len() > 64
            || experts.len() != weights.len()
            || x.len() < hidden
            || y.len() < hidden
        {
            return false;
        }
        unsafe {
            coli_apple8_metalio_moe_topk(
                experts.as_ptr(),
                weights.as_ptr(),
                experts.len() as i32,
                x.as_ptr(),
                y.as_mut_ptr(),
                hidden as i32,
                intermediate as i32,
            ) == 1
        }
    }

    /// Owning split-phase routed-MoE submission. The native pointer is never
    /// exposed through the safe API: finishing consumes this value, while
    /// dropping it retires the command without publishing output. This makes
    /// double-finish impossible in safe Rust and keeps scratch/slot lifetimes
    /// tied to actual native completion.
    pub struct MoePending {
        raw: Option<std::ptr::NonNull<c_void>>,
        hidden: usize,
    }

    impl Drop for MoePending {
        fn drop(&mut self) {
            if let Some(raw) = self.raw.take() {
                unsafe { coli_apple8_metalio_moe_topk_discard(raw.as_ptr()) };
            }
        }
    }

    /// Split-phase begin: encode+commit the fused block, return an owning
    /// pending handle, do NOT wait — the CPU can run the shared expert while
    /// the GPU works (the C engine's QWEN_APPLE8_OVERLAP=1 default path).
    pub fn moe_topk_begin(
        experts: &[ColiApple8MetalioExpert],
        weights: &[f32],
        x: &[f32],
        hidden: usize,
        intermediate: usize,
    ) -> Option<MoePending> {
        if !direct_available()
            || experts.is_empty()
            || experts.len() > 64
            || experts.len() != weights.len()
            || x.len() < hidden
        {
            return None;
        }
        let mut pending: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            coli_apple8_metalio_moe_topk_begin(
                experts.as_ptr(),
                weights.as_ptr(),
                experts.len() as i32,
                x.as_ptr(),
                hidden as i32,
                intermediate as i32,
                &mut pending,
            )
        };
        let raw = std::ptr::NonNull::new(pending)?;
        if rc == 1 {
            Some(MoePending {
                raw: Some(raw),
                hidden,
            })
        } else {
            // A non-null pointer on a failed begin would be a native contract
            // violation; retire it defensively rather than leaking resources.
            unsafe { coli_apple8_metalio_moe_topk_discard(raw.as_ptr()) };
            None
        }
    }

    /// Wait for the pending block, copy exactly `hidden` floats into y, and
    /// free the native handle. `pending` is consumed, so it cannot be finished
    /// twice. A too-short output is rejected before native memcpy; Drop still
    /// retires the submitted GPU work safely.
    pub fn moe_topk_finish(mut pending: MoePending, y: &mut [f32]) -> bool {
        if y.len() < pending.hidden {
            return false;
        }
        let Some(raw) = pending.raw.take() else {
            return false;
        };
        unsafe { coli_apple8_metalio_moe_topk_finish(raw.as_ptr(), y.as_mut_ptr()) == 1 }
    }

    /// Direct-path profile counters (encode/submit/wait/kernel ns +
    /// fused call/expert counts). Process-local, reset at direct_init.
    pub fn metal_profile() -> (u64, u64, u64, u64, u64, u64) {
        let (mut e, mut s, mut w, mut k, mut fc, mut fe) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        unsafe {
            coli_apple8_metalio_profile_get(&mut e, &mut s, &mut w, &mut k, &mut fc, &mut fe);
        }
        (e, s, w, k, fc, fe)
    }

    /// Per-island synchronization detail for the two decode-critical Metal
    /// blocks. These counters make driver/queue stalls visible separately
    /// from actual GPU execution.
    pub fn metal_profile_detail() -> (u64, u64, u64, u64, u64, u64) {
        let (mut gw, mut gk, mut gc, mut mw, mut mk, mut mc) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        unsafe {
            coli_apple8_metalio_profile_detail_get(
                &mut gw, &mut gk, &mut gc, &mut mw, &mut mk, &mut mc,
            );
        }
        (gw, gk, gc, mw, mk, mc)
    }

    /// Release native GDN wrappers for one model before its aligned Rust
    /// backing allocations are dropped. The native side serializes this with
    /// token execution and therefore cannot retain zero-copy buffers past the
    /// model lifetime.
    pub fn gdn_drop_model(model_id: u64) {
        if model_id != 0 && direct_available() {
            unsafe { coli_apple8_metalio_gdn_drop_model(model_id) };
        }
    }

    /// Owning split-phase full-GDN command. The handle must retire before the
    /// model-owned aligned weights/state are freed. Logan's Model::gdn_token
    /// keeps it stack-local; Drop waits/discards defensively on unwind.
    pub struct GdnPending {
        raw: Option<std::ptr::NonNull<c_void>>,
        hidden: usize,
    }

    impl Drop for GdnPending {
        fn drop(&mut self) {
            if let Some(raw) = self.raw.take() {
                unsafe { coli_apple8_metalio_gdn_discard(raw.as_ptr()) };
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token_begin(
        model_id: u64,
        layer: usize,
        x: &[f32],
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        wout: &[u8],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        output_gate: i32,
        eps: f32,
    ) -> Option<GdnPending> {
        if !direct_available() || x.len() < d || d > i32::MAX as usize {
            return None;
        }
        let mut pending: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            coli_apple8_metalio_gdn_begin(
                model_id,
                layer as i32,
                x.as_ptr(),
                wqkv.as_ptr() as *const u16,
                wz.as_ptr() as *const u16,
                wa.as_ptr() as *const u16,
                wb.as_ptr() as *const u16,
                wout.as_ptr() as *const u16,
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_w.as_ptr(),
                norm_w.as_ptr(),
                state.as_mut_ptr(),
                conv_state.as_mut_ptr(),
                d as i32,
                kheads as i32,
                kd as i32,
                vheads as i32,
                vd as i32,
                kk as i32,
                output_gate,
                eps,
                &mut pending,
            )
        };
        let raw = std::ptr::NonNull::new(pending)?;
        if rc == 1 {
            Some(GdnPending {
                raw: Some(raw),
                hidden: d,
            })
        } else {
            unsafe { coli_apple8_metalio_gdn_discard(raw.as_ptr()) };
            None
        }
    }

    pub fn gdn_token_finish(mut pending: GdnPending, out: &mut [f32]) -> i32 {
        if out.len() < pending.hidden {
            return 0;
        }
        let Some(raw) = pending.raw.take() else {
            return 0;
        };
        unsafe { coli_apple8_metalio_gdn_finish(raw.as_ptr(), out.as_mut_ptr()) }
    }

    /// Decode-only Metal GDN token, byte-exact C seam contract:
    /// - the five BF16 weight matrices, the recurrent state, and the conv
    ///   state MUST live in 16 KiB page-aligned host memory (the .mm wraps
    ///   them zero-copy via newBufferWithBytesNoCopy);
    /// - model_id is a process-unique owner identity; native layer reuse is
    ///   permitted only within that exact model;
    /// - x/out are copied;
    /// - rc > 0: done; rc == 0: declined BEFORE commit (CPU fallback safe);
    ///   rc < 0: failed AFTER submit — recurrent state may have advanced, the
    ///   C engine treats that as fatal (do not fall through).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token(
        model_id: u64,
        layer: usize,
        x: &[f32],
        out: &mut [f32],
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        wout: &[u8],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        output_gate: i32,
        eps: f32,
    ) -> i32 {
        if !direct_available() {
            return 0;
        }
        unsafe {
            coli_apple8_metalio_gdn_token(
                model_id,
                layer as i32,
                x.as_ptr(),
                out.as_mut_ptr(),
                wqkv.as_ptr() as *const u16,
                wz.as_ptr() as *const u16,
                wa.as_ptr() as *const u16,
                wb.as_ptr() as *const u16,
                wout.as_ptr() as *const u16,
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_w.as_ptr(),
                norm_w.as_ptr(),
                state.as_mut_ptr(),
                conv_state.as_mut_ptr(),
                d as i32,
                kheads as i32,
                kd as i32,
                vheads as i32,
                vd as i32,
                kk as i32,
                output_gate,
                eps,
            )
        }
    }
    /// Submit the Metal half of an ANE->Metal GDN island. When `fence` is
    /// supplied the command buffer is committed immediately but cannot execute
    /// its gather until ANE signals the shared event. This is the critical
    /// device-to-device dependency: the CPU does not wait between devices.
    ///
    /// # Safety
    /// Each raw pointer must denote a live IOSurface of the specified size.
    /// The surfaces and aligned model/state backing must remain valid until the
    /// returned `GdnPending` is finished or dropped.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_ane_token_begin(
        model_id: u64,
        layer: usize,
        surfaces: [(*mut c_void, usize); 4],
        spatial: usize,
        fence: Option<&MetalAneFence>,
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        wout: &[u8],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        output_gate: i32,
        eps: f32,
    ) -> Option<GdnPending> {
        if [layer, d, kheads, kd, vheads, vd, kk, spatial]
            .iter()
            .any(|&n| n > i32::MAX as usize)
            || d == 0
            || kd == 0
            || vd == 0
            || kheads == 0
            || vheads == 0
            || kk == 0
            || spatial == 0
            || vheads % kheads != 0
        {
            return None;
        }
        let Some(kdim) = kheads.checked_mul(kd) else {
            return None;
        };
        let Some(vdim) = vheads.checked_mul(vd) else {
            return None;
        };
        let Some(cdim) = kdim.checked_mul(2).and_then(|n| n.checked_add(vdim)) else {
            return None;
        };
        let fits = |actual: usize, rows: usize, cols: usize, bytes: usize| {
            rows.checked_mul(cols)
                .and_then(|n| n.checked_mul(bytes))
                .is_some_and(|n| actual >= n)
        };
        if !fits(wqkv.len(), cdim, d, 2)
            || !fits(wz.len(), vdim, d, 2)
            || !fits(wa.len(), vheads, d, 2)
            || !fits(wb.len(), vheads, d, 2)
            || !fits(wout.len(), d, vdim, 2)
            || a_log.len() < vheads
            || dt_bias.len() < vheads
            || !fits(conv_w.len(), cdim, kk, 1)
            || norm_w.len() < vd
            || !fits(state.len(), vdim, kd, 1)
            || !fits(conv_state.len(), cdim, kk - 1, 1)
        {
            return None;
        }
        if !direct_available() {
            return None;
        }
        let mut pending = std::ptr::null_mut();
        let rc = unsafe {
            coli_apple8_metalio_gdn_ane_begin(
                model_id,
                layer as i32,
                surfaces[0].0,
                surfaces[0].1,
                surfaces[1].0,
                surfaces[1].1,
                surfaces[2].0,
                surfaces[2].1,
                surfaces[3].0,
                surfaces[3].1,
                spatial as i32,
                wqkv.as_ptr() as *const u16,
                wz.as_ptr() as *const u16,
                wa.as_ptr() as *const u16,
                wb.as_ptr() as *const u16,
                wout.as_ptr() as *const u16,
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_w.as_ptr(),
                norm_w.as_ptr(),
                state.as_mut_ptr(),
                conv_state.as_mut_ptr(),
                d as i32,
                kheads as i32,
                kd as i32,
                vheads as i32,
                vd as i32,
                kk as i32,
                output_gate,
                eps,
                fence.map_or(std::ptr::null_mut(), MetalAneFence::raw_handle),
                &mut pending,
            )
        };
        if rc != 1 {
            return None;
        }
        let raw = std::ptr::NonNull::new(pending)?;
        Some(GdnPending {
            raw: Some(raw),
            hidden: d,
        })
    }

    /// Synchronous compatibility wrapper used by the existing GPU-tail A/B.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_ane_token(
        model_id: u64,
        layer: usize,
        surfaces: [(*mut c_void, usize); 4],
        spatial: usize,
        out: &mut [f32],
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        wout: &[u8],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        output_gate: i32,
        eps: f32,
    ) -> i32 {
        let pending = unsafe {
            gdn_ane_token_begin(
                model_id,
                layer,
                surfaces,
                spatial,
                None,
                wqkv,
                wz,
                wa,
                wb,
                wout,
                a_log,
                dt_bias,
                conv_w,
                norm_w,
                state,
                conv_state,
                d,
                kheads,
                kd,
                vheads,
                vd,
                kk,
                output_gate,
                eps,
            )
        };
        let Some(pending) = pending else {
            return 0;
        };
        gdn_token_finish(pending, out)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    // Non-macOS (CI/Linux): Metal is unavailable by definition. Every entry
    // point declines so callers take the CPU path — same contract as the C
    // engine's non-Darwin build.
    #[repr(C)]
    pub struct ColiMetalTensor {
        _private: [u8; 0],
    }

    pub unsafe fn coli_metal_tensor_free(_tensor: *mut ColiMetalTensor) {}

    pub struct MetalSharedSurface;
    pub struct MetalAneDynamicPack;
    pub struct MetalGdnConvSilu;

    impl MetalGdnConvSilu {
        pub fn new(
            _input: &MetalSharedSurface,
            _output: &MetalSharedSurface,
            _weights: &[f32],
            _channels: usize,
            _spatial: usize,
            _kernel: usize,
        ) -> Option<Self> {
            None
        }
        pub fn run(&mut self) -> bool {
            false
        }
    }

    impl MetalSharedSurface {
        pub unsafe fn from_iosurface(
            _raw_surface: *mut std::os::raw::c_void,
            _logical_bytes: usize,
        ) -> Option<Self> {
            None
        }
        pub fn len(&self) -> usize {
            0
        }
        pub fn is_empty(&self) -> bool {
            true
        }
        pub fn allocation_len(&self) -> usize {
            0
        }
        pub unsafe fn contents_ptr(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
    }

    impl MetalAneDynamicPack {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            _qkv_dst: &MetalSharedSurface,
            _aux_dst: &MetalSharedSurface,
            _wqkv: &[u8],
            _wz: &[u8],
            _wa: &[u8],
            _wb: &[u8],
            _hidden: usize,
            _spatial: usize,
            _qkv_rows: usize,
            _z_rows: usize,
            _ab_rows: usize,
            _qkv_stride: usize,
            _aux_stride: usize,
            _qkv_offset: usize,
            _z_offset: usize,
            _a_offset: usize,
            _b_offset: usize,
        ) -> Option<Self> {
            None
        }
        pub fn run(&mut self, _x: &[f32]) -> Option<f64> {
            None
        }
    }

    pub struct MetalMatmulDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub y: &'a mut [f32],
        pub weights: &'a [u8],
        pub scales: &'a [u8],
        pub fmt: i32,
        pub i: usize,
        pub o: usize,
    }

    pub struct MlxAffineMatmulDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub y: &'a mut [f32],
        pub weights: &'a [u8],
        pub aux: &'a [u8],
        pub bits: u8,
        pub group_size: usize,
        /// True when MLX affine scale/bias sidecars are IEEE fp16 rather than BF16.
        pub aux_fp16: bool,
        pub i: usize,
        pub o: usize,
        /// Per-descriptor activation. `None` consumes the shared activation
        /// passed to `metal_matmul_mlx_affine_multi`; `Some(v)` lets a
        /// descriptor whose input differs (for example a routed expert's own
        /// SwiGLU output) join the same command buffer. `v.len() >= i`.
        pub x: Option<&'a [f32]>,
    }

    pub struct MetalWeightDesc<'a> {
        pub tensor: *mut ColiMetalTensor,
        pub weights: &'a [u8],
        pub scales: &'a [u8],
        pub fmt: i32,
        pub group_size: usize,
        pub i: usize,
        pub o: usize,
    }

    pub fn metal_init() -> bool {
        false
    }
    pub fn metal_available() -> bool {
        false
    }
    pub fn dense_profile_start() {}
    pub fn dense_profile_stop() -> (u64, u64, u64, u64) {
        (0, 0, 0, 0)
    }
    pub fn metal_matmul(
        _tensor: &mut *mut ColiMetalTensor,
        _y: &mut [f32],
        _x: &[f32],
        _weights: &[u8],
        _scales: &[u8],
        _fmt: i32,
        _i: usize,
        _o: usize,
    ) -> bool {
        false
    }
    pub fn metal_matmul_mlx_affine(
        _tensor: &mut *mut ColiMetalTensor,
        _y: &mut [f32],
        _x: &[f32],
        _weights: &[u8],
        _aux: &[u8],
        _bits: u8,
        _group_size: usize,
        _aux_fp16: bool,
        _i: usize,
        _o: usize,
    ) -> bool {
        false
    }
    pub fn metal_matmul_mlx_affine_multi(
        _x: &[f32],
        _descs: &mut [MlxAffineMatmulDesc<'_>],
    ) -> bool {
        false
    }
    pub fn metal_matmul_multi(_x: &[f32], _descs: &mut [MetalMatmulDesc<'_>]) -> bool {
        false
    }
    #[allow(clippy::too_many_arguments)]
    pub fn llama_layer(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalMatmulDesc<'_>],
        _x: &mut [f32],
        _k_out: &mut [f32],
        _v_out: &mut [f32],
        _input_norm: &[f32],
        _post_norm: &[f32],
        _d: usize,
        _inter: usize,
        _heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos: usize,
        _theta: f32,
        _eps: f32,
    ) -> i32 {
        0
    }
    pub fn llama_drop_model(_model_id: u64) {}
    pub fn spark_token_begin(_model_id: u64, _x: &[f32], _d: usize, _pos: usize) -> bool {
        false
    }
    #[allow(clippy::too_many_arguments)]
    pub fn spark_layer_encode(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalWeightDesc<'_>],
        _input_norm: &[f32],
        _post_norm: &[f32],
        _d: usize,
        _inter: usize,
        _heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _sliding: bool,
        _window: usize,
        _pos: usize,
        _rotary_dim: usize,
        _theta: f32,
        _eps: f32,
    ) -> i32 {
        0
    }
    pub fn spark_token_end(_model_id: u64, _x: &mut [f32], _d: usize, _pos: usize) -> i32 {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn spark_token_end_top1(
        _model_id: u64,
        _head: &mut MetalWeightDesc<'_>,
        _norm: &[f32],
        _token: &mut u32,
        _d: usize,
        _vocab: usize,
        _pos: usize,
        _eps: f32,
    ) -> i32 {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_token_end_logits(
        _model_id: u64,
        _head: &mut MetalWeightDesc<'_>,
        _norm: &[f32],
        _logits: &mut [f32],
        _d: usize,
        _vocab: usize,
        _pos: usize,
        _eps: f32,
    ) -> i32 {
        0
    }

    pub fn spark_prefill_begin(
        _model_id: u64,
        _x: &[f32],
        _s: usize,
        _d: usize,
        _base: usize,
    ) -> bool {
        false
    }
    #[allow(clippy::too_many_arguments)]
    pub fn spark_prefill_layer_encode(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalWeightDesc<'_>],
        _input_norm: &[f32],
        _post_norm: &[f32],
        _d: usize,
        _inter: usize,
        _heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _sliding: bool,
        _window: usize,
        _base: usize,
        _srows: usize,
        _rotary_dim: usize,
        _theta: f32,
        _eps: f32,
    ) -> i32 {
        0
    }
    pub fn spark_prefill_end(_model_id: u64, _base: usize, _srows: usize) -> i32 {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spark_prefill_end_logits(
        _model_id: u64,
        _head: &mut MetalWeightDesc<'_>,
        _norm: &[f32],
        _logits: &mut [f32],
        _d: usize,
        _vocab: usize,
        _base: usize,
        _srows: usize,
        _eps: f32,
    ) -> i32 {
        0
    }
    pub fn spark_prefill_abort(_model_id: u64) {}

    pub fn spark_token_abort(_model_id: u64) {}

    #[allow(clippy::too_many_arguments)]
    pub fn spark_layer(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalWeightDesc<'_>],
        _x: &mut [f32],
        _input_norm: &[f32],
        _post_norm: &[f32],
        _d: usize,
        _inter: usize,
        _heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _sliding: bool,
        _window: usize,
        _pos: usize,
        _rotary_dim: usize,
        _theta: f32,
        _eps: f32,
    ) -> i32 {
        0
    }
    pub fn spark_drop_model(_model_id: u64) {}
    pub fn metal_rmsnorm(_x: &mut [f32], _w: &[f32], _n: usize, _nrows: usize, _eps: f32) -> bool {
        false
    }
    pub fn metal_add(_y: &mut [f32], _a: &[f32]) -> bool {
        false
    }
    pub fn metal_silu_mul(_g: &mut [f32], _u: &[f32]) -> bool {
        false
    }

    pub struct ColiApple8MetalioExpert {
        pub slot: i32,
        pub gate_offset: usize,
        pub gate_bytes: usize,
        pub up_offset: usize,
        pub up_bytes: usize,
        pub down_offset: usize,
        pub down_bytes: usize,
    }

    pub fn mio_init() -> bool {
        false
    }
    pub fn mio_active() -> bool {
        false
    }
    pub fn mio_batch_barrier() -> Option<i64> {
        None
    }
    pub fn mio_batch_wait(_event_value: i64, _slots: &[i32]) -> bool {
        false
    }
    pub fn mio_file(_path: &str) -> Option<i32> {
        None
    }
    #[derive(Debug, Clone, Copy)]
    pub struct MioRegion {
        pub file: i32,
        pub src_off: u64,
        pub bytes: usize,
        pub dst_off: usize,
    }
    pub fn mio_load_regions(
        _regions: &[MioRegion],
        _speculative: bool,
    ) -> Option<(i32, i64)> {
        None
    }
    pub fn mio_finish_slot(
        _slot: i32,
        _event: i64,
        _used_bytes: usize,
        _speculative: bool,
    ) -> Option<Vec<u8>> {
        None
    }
    pub fn mio_discard_slot(_slot: i32) {}
    pub fn mio_load_expert(_fid: i32, _regions: &[(u64, usize)]) -> Option<(i32, i64)> {
        None
    }
    pub fn mio_prefetch_expert(_fid: i32, _regions: &[(u64, usize)]) -> Option<(i32, i64)> {
        None
    }
    pub struct ColiMetalioStats {
        pub loads: u64,
        pub bytes: u64,
        pub waits: u64,
        pub fails: u64,
        pub prefetch_loads: u64,
        pub prefetch_used: u64,
        pub prefetch_wasted: u64,
        pub prefetch_ready_at_demand: u64,
        pub prefetch_late_at_demand: u64,
        pub outstanding: u64,
        pub peak_outstanding: u64,
        pub latency_samples: u64,
        pub total_latency_s: f64,
        pub lat_hist: [u64; 32],
    }
    impl Default for ColiMetalioStats {
        fn default() -> ColiMetalioStats {
            ColiMetalioStats {
                loads: 0,
                bytes: 0,
                waits: 0,
                fails: 0,
                prefetch_loads: 0,
                prefetch_used: 0,
                prefetch_wasted: 0,
                prefetch_ready_at_demand: 0,
                prefetch_late_at_demand: 0,
                outstanding: 0,
                peak_outstanding: 0,
                latency_samples: 0,
                total_latency_s: 0.0,
                lat_hist: [0; 32],
            }
        }
    }
    pub fn mio_stats() -> ColiMetalioStats {
        ColiMetalioStats::default()
    }
    pub fn metalio_wait(_ev: i64) -> i32 {
        -1
    }
    pub unsafe fn metalio_prefetch_demanded(_slot: i32) {}
    pub fn metalio_slot_free(_slot: i32) {}
    pub fn metalio_slot_ptr(_slot: i32) -> *mut std::os::raw::c_void {
        std::ptr::null_mut()
    }

    pub fn bnns_bf16_matmul(_w: &[u8], _x: &[f32], _y: &mut [f32], _o: usize, _i: usize) -> bool {
        false
    }
    pub fn bnns_bf16_matmul_batch(
        _w: &[u8],
        _x: &[f32],
        _y: &mut [f32],
        _rows: usize,
        _o: usize,
        _i: usize,
    ) -> bool {
        false
    }

    pub fn direct_init() -> bool {
        false
    }
    pub fn direct_available() -> bool {
        false
    }
    pub fn swiglu_slot_batch(
        _expert: &ColiApple8MetalioExpert,
        _x: &[f32],
        _y: &mut [f32],
        _rows: usize,
        _hidden: usize,
        _intermediate: usize,
    ) -> bool {
        false
    }
    pub fn moe_rows(
        _experts: &[ColiApple8MetalioExpert],
        _row_offsets: &[i32],
        _x: &[f32],
        _y: &mut [f32],
        _hidden: usize,
        _intermediate: usize,
    ) -> bool {
        false
    }
    pub fn moe_topk(
        _experts: &[ColiApple8MetalioExpert],
        _weights: &[f32],
        _x: &[f32],
        _y: &mut [f32],
        _hidden: usize,
        _inter: usize,
    ) -> bool {
        false
    }
    pub struct MoePending {
        _private: (),
    }
    pub fn moe_topk_begin(
        _experts: &[ColiApple8MetalioExpert],
        _weights: &[f32],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
    ) -> Option<MoePending> {
        None
    }
    pub fn moe_topk_finish(_pending: MoePending, _y: &mut [f32]) -> bool {
        false
    }
    pub fn metal_profile() -> (u64, u64, u64, u64, u64, u64) {
        (0, 0, 0, 0, 0, 0)
    }
    pub fn metal_profile_detail() -> (u64, u64, u64, u64, u64, u64) {
        (0, 0, 0, 0, 0, 0)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_mxfp4(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalWeightDesc<'_>],
        _x: &[f32],
        _out: &mut [f32],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _output_gate: i32,
        _eps: f32,
    ) -> i32 {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix(
        _model_id: u64,
        _descs: &mut [MetalWeightDesc<'_>],
        _normed: &[f32],
        _out: &mut [f32],
        _inject: Option<&mut [f32]>,
        _d: usize,
        _hc: usize,
        _lr: usize,
    ) -> Result<Option<()>, ()> {
        Ok(None)
    }
    pub fn hc_drop_model(_model_id: u64) {}

    pub fn shared_mxfp4(
        _model_id: u64,
        _layer: usize,
        _descs: &mut [MetalWeightDesc<'_>],
        _x: &[f32],
        _out: &mut [f32],
        _d: usize,
        _iinter: usize,
    ) -> Result<Option<()>, ()> {
        Ok(None)
    }
    pub fn shared_mxfp4_drop_model(_model_id: u64) {}

    pub fn gdn_mxfp4_drop_model(_model_id: u64) {}

    pub fn gdn_drop_model(_model_id: u64) {}

    pub struct GdnPending;

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token_begin(
        _model_id: u64,
        _layer: usize,
        _x: &[f32],
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _wout: &[u8],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _output_gate: i32,
        _eps: f32,
    ) -> Option<GdnPending> {
        None
    }

    pub fn gdn_token_finish(_pending: GdnPending, _out: &mut [f32]) -> i32 {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token(
        _model_id: u64,
        _layer: usize,
        _x: &[f32],
        _out: &mut [f32],
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _wout: &[u8],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _output_gate: i32,
        _eps: f32,
    ) -> i32 {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_ane_token(
        _model_id: u64,
        _layer: usize,
        _surfaces: [(*mut std::ffi::c_void, usize); 4],
        _spatial: usize,
        _out: &mut [f32],
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _wout: &[u8],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _output_gate: i32,
        _eps: f32,
    ) -> i32 {
        0
    }

    /// Non-macOS: Metal owns no shared event and there is no ANE to signal, so
    /// no fence can ever exist. Uninhabited and never constructible (there is
    /// no `new`), which keeps the Apple signatures identical — call sites that
    /// thread `Option<&MetalAneFence>` through simply always see `None`.
    pub struct MetalAneFence {
        _private: [u8; 0],
    }

    /// Non-macOS twin of the Apple zero-copy ANE GDN launch. There are no ANE
    /// IOSurfaces here, so this declines pre-submit exactly like
    /// `gdn_ane_token` above, and the caller's `if let Some(pending)` takes the
    /// CPU tail.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_ane_token_begin(
        _model_id: u64,
        _layer: usize,
        _surfaces: [(*mut std::ffi::c_void, usize); 4],
        _spatial: usize,
        _fence: Option<&MetalAneFence>,
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _wout: &[u8],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _output_gate: i32,
        _eps: f32,
    ) -> Option<GdnPending> {
        None
    }
}

pub use imp::*;

/// BF16 GEMV on the direct path (generic; wired for the attention
/// projections). w = BF16 bytes, O x I row-major; x = S x I f32; y = S x O.
/// rc > 0 done; rc == 0 declined pre-submit (CPU fallback); rc < 0 fatal.
#[cfg(target_os = "macos")]
pub fn bf16_matmul(w: &[u8], x: &[f32], y: &mut [f32], s: usize, o: usize, i: usize) -> i32 {
    if !direct_available() || !metal_available() {
        return 0;
    }
    unsafe {
        coli_apple8_metalio_bf16_matmul(
            w.as_ptr() as *const u16,
            x.as_ptr(),
            y.as_mut_ptr(),
            s as i32,
            o as i32,
            i as i32,
        )
    }
}

#[cfg(not(target_os = "macos"))]
pub fn bf16_matmul(_w: &[u8], _x: &[f32], _y: &mut [f32], _s: usize, _o: usize, _i: usize) -> i32 {
    0
}
