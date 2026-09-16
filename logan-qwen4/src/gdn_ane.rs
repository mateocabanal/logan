//! Experimental Apple Neural Engine GDN input-projection island.
//!
//! Opt-in only. `QWEN_GDN_ANE=1` enables layer 0 by default; select an
//! inclusive comma-separated set/range with `QWEN_GDN_ANE_LAYERS=0-2,4` or
//! explicitly request every eligible GDN layer with `all`.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use std::{path::PathBuf, time::Instant};

    use logan_ane::{
        mil::DenseProjection, AneAsyncChannel, AneChannelPending, AneModel, AneRequest, AneRuntime,
        AneSurface, CompileOptions,
    };
    use logan_metal::{
        MetalAneDynamicPack, MetalAneDynamicPackPending, MetalAneFence, MetalGdnConvSilu,
        MetalSharedSurface,
    };

    const SPATIAL: usize = 16;

    pub enum GdnAneState {
        Uninitialized,
        Ready(GdnAneLayer),
        Disabled,
    }

    impl Default for GdnAneState {
        fn default() -> Self {
            Self::Uninitialized
        }
    }

    pub struct GdnAneLayer {
        model: AneModel,
        input: AneSurface,
        qkv: AneSurface,
        z: AneSurface,
        a: AneSurface,
        b: AneSurface,
        conv_out: Option<AneSurface>,
        metal_conv: Option<MetalGdnConvSilu>,
        hidden: usize,
        qkv_rows: usize,
        z_rows: usize,
        ab_rows: usize,
        conv_kernel: usize,
        gpu_tail: bool,
        async_fence: Option<MetalAneFence>,
        async_channel: Option<AneAsyncChannel>,
        async_slow_score: u32,
        async_samples: u32,
    }

    /// Shared two-program dynamic-weight ANE engine. Unlike the original
    /// constant-weight path this is model-global: every GDN layer reuses the
    /// same qkv program and the same z+a+b program, avoiding the M2 firmware's
    /// severe 36-program residency/switching cliff.
    pub struct GdnAneDynamicEngine {
        qkv_model: AneModel,
        aux_model: AneModel,
        qkv_input: AneSurface,
        aux_input: AneSurface,
        qkv: AneSurface,
        z: AneSurface,
        a: AneSurface,
        b: AneSurface,
        hidden: usize,
        qkv_rows: usize,
        z_rows: usize,
        ab_rows: usize,
        qkv_spatial: usize,
        aux_spatial: usize,
        qkv_weight_offset: usize,
        aux_weight_offsets: [usize; 3],
        metal_packers: Vec<Option<MetalAneDynamicPack>>,
        async_fence: Option<MetalAneFence>,
        qkv_channel: Option<AneAsyncChannel>,
        aux_channel: Option<AneAsyncChannel>,
        async_started: bool,
    }

    pub fn dynamic_enabled() -> bool {
        switch_enabled("QWEN_GDN_ANE_DYNAMIC")
    }

    pub fn dynamic_async_enabled() -> bool {
        switch_enabled("QWEN_GDN_ANE_DYNAMIC_ASYNC")
    }

    /// Owns one fully device-chained dynamic GDN front half:
    /// Metal pack -> ANE qkv -> ANE z/a/b. The same monotonic shared event
    /// orders every edge; callers can submit the GPU tail against `gpu_fence()`
    /// before any of these stages complete on the host.
    pub struct GdnAneDynamicPending {
        pack: MetalAneDynamicPackPending,
        qkv: AneChannelPending,
        aux: AneChannelPending,
        submit_ms: f64,
    }

    impl GdnAneDynamicPending {
        pub fn submit_ms(&self) -> f64 {
            self.submit_ms
        }

        pub fn finish(self, timeout_ms: u64) -> Result<f64, String> {
            let pack_gpu_ms = self
                .pack
                .finish()
                .ok_or_else(|| "dynamic ANE Metal pack failed after submission".to_string())?;
            self.qkv.finish(timeout_ms).map_err(|e| e.to_string())?;
            self.aux.finish(timeout_ms).map_err(|e| e.to_string())?;
            Ok(pack_gpu_ms)
        }
    }

    impl GdnAneDynamicEngine {
        pub fn build(
            hidden: usize,
            qkv_rows: usize,
            z_rows: usize,
            ab_rows: usize,
        ) -> Result<Self, String> {
            let (qkv_program, qkv_layout) =
                logan_ane::mil::parallel_dense_packed_dynamic_f32_io(hidden, SPATIAL, &[qkv_rows])
                    .map_err(|e| e.to_string())?;
            let (aux_program, aux_layout) = logan_ane::mil::parallel_dense_packed_dynamic_f32_io(
                hidden,
                SPATIAL,
                &[z_rows, ab_rows, ab_rows],
            )
            .map_err(|e| e.to_string())?;
            // The ANE compiler currently rejects a packed tensor dimension
            // beyond 16384. Keep the production path fail-closed if a future
            // model geometry exceeds the hardware-qualified envelope.
            if qkv_layout.total_spatial > 16_384 || aux_layout.total_spatial > 16_384 {
                return Err(format!(
                    "dynamic ANE packed spatial exceeds 16384 (qkv={} aux={})",
                    qkv_layout.total_spatial, aux_layout.total_spatial,
                ));
            }
            let runtime = AneRuntime::load().map_err(|e| e.to_string())?;
            let mut options = CompileOptions::default();
            options.cache_directory = ane_cache_dir();
            let t0 = Instant::now();
            let mut qkv_model = runtime
                .compile(&qkv_program, options.clone())
                .map_err(|e| e.to_string())?;
            let qkv_compile_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = Instant::now();
            let mut aux_model = runtime
                .compile(&aux_program, options)
                .map_err(|e| e.to_string())?;
            let aux_compile_ms = t0.elapsed().as_secs_f64() * 1e3;
            qkv_model.load().map_err(|e| e.to_string())?;
            aux_model.load().map_err(|e| e.to_string())?;

            let qkv_input = AneSurface::new(hidden * qkv_layout.total_spatial * 4)
                .map_err(|e| e.to_string())?;
            let aux_input = AneSurface::new(hidden * aux_layout.total_spatial * 4)
                .map_err(|e| e.to_string())?;
            let qkv = AneSurface::new(qkv_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let z = AneSurface::new(z_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let a = AneSurface::new(ab_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let b = AneSurface::new(ab_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            eprintln!(
                "qwen4-rs: dynamic ANE GDN engine ready (2 programs; qkv_spatial={} aux_spatial={} input_mib={:.1}; compile={:.1}+{:.1}ms)",
                qkv_layout.total_spatial, aux_layout.total_spatial,
                (hidden * (qkv_layout.total_spatial + aux_layout.total_spatial) * 4) as f64 / 1048576.0,
                qkv_compile_ms, aux_compile_ms,
            );
            Ok(Self {
                qkv_model,
                aux_model,
                qkv_input,
                aux_input,
                qkv,
                z,
                a,
                b,
                hidden,
                qkv_rows,
                z_rows,
                ab_rows,
                qkv_spatial: qkv_layout.total_spatial,
                aux_spatial: aux_layout.total_spatial,
                qkv_weight_offset: qkv_layout.weight_offsets[0],
                aux_weight_offsets: [
                    aux_layout.weight_offsets[0],
                    aux_layout.weight_offsets[1],
                    aux_layout.weight_offsets[2],
                ],
                metal_packers: Vec::new(),
                async_fence: None,
                qkv_channel: None,
                aux_channel: None,
                async_started: false,
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn ensure_metal_packer(
            &mut self,
            layer: usize,
            wqkv: &[u8],
            wz: &[u8],
            wa: &[u8],
            wb: &[u8],
        ) -> Result<(), String> {
            if self.metal_packers.len() <= layer {
                self.metal_packers.resize_with(layer + 1, || None);
            }
            if self.metal_packers[layer].is_some() {
                return Ok(());
            }
            let qkv_dst = unsafe {
                MetalSharedSurface::from_iosurface(
                    self.qkv_input.as_raw_iosurface(),
                    self.qkv_input.len(),
                )
            }
            .ok_or_else(|| "dynamic ANE qkv Metal surface import failed".to_string())?;
            let aux_dst = unsafe {
                MetalSharedSurface::from_iosurface(
                    self.aux_input.as_raw_iosurface(),
                    self.aux_input.len(),
                )
            }
            .ok_or_else(|| "dynamic ANE aux Metal surface import failed".to_string())?;
            self.metal_packers[layer] = Some(
                MetalAneDynamicPack::new(
                    &qkv_dst,
                    &aux_dst,
                    wqkv,
                    wz,
                    wa,
                    wb,
                    self.hidden,
                    SPATIAL,
                    self.qkv_rows,
                    self.z_rows,
                    self.ab_rows,
                    self.qkv_spatial,
                    self.aux_spatial,
                    self.qkv_weight_offset,
                    self.aux_weight_offsets[0],
                    self.aux_weight_offsets[1],
                    self.aux_weight_offsets[2],
                )
                .ok_or_else(|| "dynamic ANE Metal packer creation failed".to_string())?,
            );
            Ok(())
        }

        fn ensure_async_channels(&mut self) -> Result<(), String> {
            if self.async_fence.is_none() {
                self.async_fence = MetalAneFence::new(1);
            }
            let fence = self
                .async_fence
                .as_ref()
                .ok_or_else(|| "dynamic ANE Metal shared event unavailable".to_string())?;
            let shared_event = fence.ane_shared_event();
            let submit_mode = if switch_enabled("QWEN_GDN_ANE_ASYNC_REALTIME") {
                2
            } else if switch_enabled("QWEN_GDN_ANE_ASYNC_DIRECT") {
                1
            } else {
                0
            };
            // Dynamic requests are large (~161 MiB across qkv+aux on
            // Qwen3.8-Flash-Next). Pre-mapping avoids repeated private-runtime
            // IOSurface mapping; hardware A/B on M2 cut warm GDN wait by ~25%.
            // Keep an explicit 0/false escape hatch for qualification.
            let premap = std::env::var("QWEN_GDN_ANE_ASYNC_PREMAP")
                .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            if self.qkv_channel.is_none() {
                self.qkv_channel = Some(
                    unsafe {
                        self.qkv_model.async_channel_wait_signal(
                            &[&self.qkv_input],
                            &[&self.qkv],
                            0,
                            shared_event,
                            shared_event,
                            submit_mode,
                            premap,
                        )
                    }
                    .map_err(|e| e.to_string())?,
                );
            }
            if self.aux_channel.is_none() {
                self.aux_channel = Some(
                    unsafe {
                        self.aux_model.async_channel_wait_signal(
                            &[&self.aux_input],
                            &[&self.z, &self.a, &self.b],
                            0,
                            shared_event,
                            shared_event,
                            submit_mode,
                            premap,
                        )
                    }
                    .map_err(|e| e.to_string())?,
                );
            }
            Ok(())
        }

        /// Submit Metal pack -> qkv ANE -> aux ANE without a host wait between
        /// stages. The caller should immediately submit the GPU tail against
        /// `gpu_fence()`, then finish this owner after that tail retires.
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_layer_async(
            &mut self,
            layer: usize,
            x: &[f32],
            wqkv: &[u8],
            wz: &[u8],
            wa: &[u8],
            wb: &[u8],
        ) -> Result<GdnAneDynamicPending, String> {
            if x.len() != self.hidden {
                return Err("dynamic ANE hidden vector size mismatch".into());
            }
            self.ensure_metal_packer(layer, wqkv, wz, wa, wb)?;
            self.ensure_async_channels()?;

            let submit_t0 = Instant::now();
            let pack_value = {
                let fence = self.async_fence.as_mut().unwrap();
                if self.async_started {
                    fence
                        .advance()
                        .ok_or_else(|| "dynamic ANE pack fence advance failed".to_string())?
                } else {
                    fence.value()
                }
            };
            let pack = {
                let fence = self.async_fence.as_ref().unwrap();
                self.metal_packers[layer]
                    .as_mut()
                    .unwrap()
                    .begin(x, fence)
                    .ok_or_else(|| "dynamic ANE Metal pack submit declined".to_string())?
            };
            self.async_started = true;

            let qkv_signal = self
                .async_fence
                .as_mut()
                .unwrap()
                .advance()
                .ok_or_else(|| "dynamic ANE qkv fence advance failed".to_string())?;
            let qkv = self
                .qkv_channel
                .as_mut()
                .unwrap()
                .submit_after(pack_value, qkv_signal)
                .map_err(|e| e.to_string())?;
            let aux_signal = self
                .async_fence
                .as_mut()
                .unwrap()
                .advance()
                .ok_or_else(|| "dynamic ANE aux fence advance failed".to_string())?;
            let aux = self
                .aux_channel
                .as_mut()
                .unwrap()
                .submit_after(qkv_signal, aux_signal)
                .map_err(|e| e.to_string())?;
            let submit_ms = submit_t0.elapsed().as_secs_f64() * 1e3;

            if switch_enabled("QWEN_GDN_ANE_TRACE") {
                eprintln!(
                    "qwen4-rs: dynamic ANE async layer={layer} pack_wait={pack_value} qkv_signal={qkv_signal} aux_signal={aux_signal} host_waits=0 submit_ms={submit_ms:.3}"
                );
            }
            Ok(GdnAneDynamicPending {
                pack,
                qkv,
                aux,
                submit_ms,
            })
        }

        pub fn gpu_fence(&self) -> Option<&MetalAneFence> {
            self.async_fence.as_ref()
        }

        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_layer(
            &mut self,
            layer: usize,
            x: &[f32],
            wqkv: &[u8],
            wz: &[u8],
            wa: &[u8],
            wb: &[u8],
        ) -> Result<(), String> {
            if x.len() != self.hidden {
                return Err("dynamic ANE hidden vector size mismatch".into());
            }
            let gpu_pack_enabled = std::env::var("QWEN_GDN_ANE_DYNAMIC_GPU_PACK")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            let pack_t0 = Instant::now();
            let mut pack_gpu_ms = None;
            if gpu_pack_enabled {
                if self.metal_packers.len() <= layer {
                    self.metal_packers.resize_with(layer + 1, || None);
                }
                if self.metal_packers[layer].is_none() {
                    let qkv_dst = unsafe {
                        MetalSharedSurface::from_iosurface(
                            self.qkv_input.as_raw_iosurface(),
                            self.qkv_input.len(),
                        )
                    };
                    let aux_dst = unsafe {
                        MetalSharedSurface::from_iosurface(
                            self.aux_input.as_raw_iosurface(),
                            self.aux_input.len(),
                        )
                    };
                    if let (Some(qkv_dst), Some(aux_dst)) = (qkv_dst.as_ref(), aux_dst.as_ref()) {
                        self.metal_packers[layer] = MetalAneDynamicPack::new(
                            qkv_dst,
                            aux_dst,
                            wqkv,
                            wz,
                            wa,
                            wb,
                            self.hidden,
                            SPATIAL,
                            self.qkv_rows,
                            self.z_rows,
                            self.ab_rows,
                            self.qkv_spatial,
                            self.aux_spatial,
                            self.qkv_weight_offset,
                            self.aux_weight_offsets[0],
                            self.aux_weight_offsets[1],
                            self.aux_weight_offsets[2],
                        );
                    }
                }
                if let Some(packer) = self.metal_packers[layer].as_mut() {
                    pack_gpu_ms = packer.run(x);
                }
            }
            let activation_ms = if pack_gpu_ms.is_some() {
                0.0
            } else {
                self.qkv_input
                    .pack_bf16_transposed_f32(
                        self.qkv_spatial,
                        self.qkv_weight_offset,
                        wqkv,
                        self.hidden,
                        self.qkv_rows,
                    )
                    .map_err(|e| e.to_string())?;
                self.aux_input
                    .pack_bf16_transposed_f32(
                        self.aux_spatial,
                        self.aux_weight_offsets[0],
                        wz,
                        self.hidden,
                        self.z_rows,
                    )
                    .map_err(|e| e.to_string())?;
                self.aux_input
                    .pack_bf16_transposed_f32(
                        self.aux_spatial,
                        self.aux_weight_offsets[1],
                        wa,
                        self.hidden,
                        self.ab_rows,
                    )
                    .map_err(|e| e.to_string())?;
                self.aux_input
                    .pack_bf16_transposed_f32(
                        self.aux_spatial,
                        self.aux_weight_offsets[2],
                        wb,
                        self.hidden,
                        self.ab_rows,
                    )
                    .map_err(|e| e.to_string())?;
                let activation_t0 = Instant::now();
                self.qkv_input
                    .write_repeated_f32(self.qkv_spatial, SPATIAL, x)
                    .map_err(|e| e.to_string())?;
                self.aux_input
                    .write_repeated_f32(self.aux_spatial, SPATIAL, x)
                    .map_err(|e| e.to_string())?;
                activation_t0.elapsed().as_secs_f64() * 1e3
            };
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1e3;

            let qkv_req =
                AneRequest::new(&[&self.qkv_input], &[&self.qkv], 0).map_err(|e| e.to_string())?;
            let aux_req = AneRequest::new(&[&self.aux_input], &[&self.z, &self.a, &self.b], 0)
                .map_err(|e| e.to_string())?;
            let eval_t0 = Instant::now();
            self.qkv_model
                .evaluate(&qkv_req)
                .map_err(|e| e.to_string())?;
            let qkv_ms = eval_t0.elapsed().as_secs_f64() * 1e3;
            let aux_t0 = Instant::now();
            self.aux_model
                .evaluate(&aux_req)
                .map_err(|e| e.to_string())?;
            let aux_ms = aux_t0.elapsed().as_secs_f64() * 1e3;
            if switch_enabled("QWEN_GDN_ANE_TRACE") {
                eprintln!(
                    "qwen4-rs: dynamic ANE GDN layer={layer} pack_ms={pack_ms:.3} pack_gpu_ms={:.3} activation_ms={activation_ms:.3} qkv_ms={qkv_ms:.3} aux_ms={aux_ms:.3}",
                    pack_gpu_ms.unwrap_or(0.0)
                );
            }
            Ok(())
        }

        pub fn gpu_surfaces(&self) -> [(*mut std::ffi::c_void, usize); 4] {
            unsafe {
                [
                    (self.qkv.as_raw_iosurface(), self.qkv.len()),
                    (self.z.as_raw_iosurface(), self.z.len()),
                    (self.a.as_raw_iosurface(), self.a.len()),
                    (self.b.as_raw_iosurface(), self.b.len()),
                ]
            }
        }

        pub fn materialize(
            &self,
            qkv: &mut [f32],
            z: &mut [f32],
            a: &mut [f32],
            b: &mut [f32],
        ) -> Result<(), String> {
            GdnAneLayer::read_column(&self.qkv, self.qkv_rows, qkv)?;
            GdnAneLayer::read_column(&self.z, self.z_rows, z)?;
            GdnAneLayer::read_column(&self.a, self.ab_rows, a)?;
            GdnAneLayer::read_column(&self.b, self.ab_rows, b)
        }
    }

    fn switch_enabled(name: &str) -> bool {
        std::env::var(name)
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    }

    fn fused_enabled() -> bool {
        std::env::var("QWEN_GDN_ANE_FUSED")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    }

    fn ephemeral_enabled() -> bool {
        switch_enabled("QWEN_GDN_ANE_EPHEMERAL")
    }

    fn ane_cache_dir() -> Option<PathBuf> {
        if switch_enabled("LOGAN_ANE_NOCACHE") {
            return None;
        }
        if let Some(path) = std::env::var_os("LOGAN_ANE_CACHE_DIR") {
            return Some(PathBuf::from(path));
        }
        if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
            return Some(PathBuf::from(path).join("logan/ane"));
        }
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/logan/ane"))
    }

    pub fn active(state: &GdnAneState) -> bool {
        matches!(state, GdnAneState::Ready(_))
    }

    pub fn fused_active(state: &GdnAneState) -> bool {
        matches!(state, GdnAneState::Ready(layer) if layer.metal_conv.is_some())
    }

    pub fn requested(layer: usize) -> bool {
        if !switch_enabled("QWEN_GDN_ANE") {
            return false;
        }
        let spec = std::env::var("QWEN_GDN_ANE_LAYERS").unwrap_or_else(|_| "0".into());
        if spec.trim().eq_ignore_ascii_case("all") {
            return true;
        }
        spec.split(',').any(|piece| {
            let piece = piece.trim();
            if piece.is_empty() {
                return false;
            }
            if let Some((lo, hi)) = piece.split_once('-') {
                match (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                    (Ok(lo), Ok(hi)) => lo <= layer && layer <= hi,
                    _ => false,
                }
            } else {
                piece.parse::<usize>().ok() == Some(layer)
            }
        })
    }

    fn bf16_to_fp16(bytes: &[u8]) -> Result<Vec<u16>, String> {
        if bytes.len() % 2 != 0 {
            return Err("BF16 byte count is not even".into());
        }
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for pair in bytes.chunks_exact(2) {
            let bf = u16::from_le_bytes([pair[0], pair[1]]);
            let value = f32::from_bits((bf as u32) << 16);
            if !value.is_finite() {
                return Err("non-finite BF16 GDN weight".into());
            }
            out.push(f32_to_f16_bits(value));
        }
        Ok(out)
    }

    // IEEE binary32 -> binary16, round-to-nearest-even.
    fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0xff {
            if mant == 0 {
                return sign | 0x7c00;
            }
            return sign | 0x7c00 | ((mant >> 13) as u16).max(1);
        }
        let half_exp = exp - 127 + 15;
        if half_exp >= 0x1f {
            return sign | 0x7c00;
        }
        if half_exp <= 0 {
            if half_exp < -10 {
                return sign;
            }
            let mantissa = mant | 0x80_0000;
            let shift = (14 - half_exp) as u32;
            let mut half_mant = mantissa >> shift;
            let remainder = mantissa & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && (half_mant & 1) != 0) {
                half_mant += 1;
            }
            return sign | half_mant as u16;
        }
        let mut half_exp_bits = (half_exp as u16) << 10;
        let mut half_mant = mant >> 13;
        let remainder = mant & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
            half_mant += 1;
            if half_mant == 0x400 {
                half_mant = 0;
                half_exp_bits += 0x400;
                if half_exp_bits >= 0x7c00 {
                    return sign | 0x7c00;
                }
            }
        }
        sign | half_exp_bits | half_mant as u16
    }

    impl GdnAneLayer {
        #[allow(clippy::too_many_arguments)]
        fn build(
            layer: usize,
            hidden: usize,
            qkv_rows: usize,
            z_rows: usize,
            ab_rows: usize,
            wqkv: &[u8],
            wz: &[u8],
            wa: &[u8],
            wb: &[u8],
            conv_weights: &[f32],
            conv_kernel: usize,
        ) -> Result<Self, String> {
            let t0 = Instant::now();
            let qkv16 = bf16_to_fp16(wqkv)?;
            let z16 = bf16_to_fp16(wz)?;
            let a16 = bf16_to_fp16(wa)?;
            let b16 = bf16_to_fp16(wb)?;
            let convert_ms = t0.elapsed().as_secs_f64() * 1e3;

            let program = logan_ane::mil::parallel_dense_fp16_f32_io(
                hidden,
                SPATIAL,
                &[
                    DenseProjection::new("qkv", qkv_rows, qkv16),
                    DenseProjection::new("z", z_rows, z16),
                    DenseProjection::new("a", ab_rows, a16),
                    DenseProjection::new("b", ab_rows, b16),
                ],
            )
            .map_err(|e| e.to_string())?;
            let runtime = AneRuntime::load().map_err(|e| e.to_string())?;
            let compile_t0 = Instant::now();
            let mut compile_options = CompileOptions::default();
            compile_options.cache_directory = ane_cache_dir();
            let mut model = runtime
                .compile(&program, compile_options)
                .map_err(|e| e.to_string())?;
            let compile_ms = compile_t0.elapsed().as_secs_f64() * 1e3;
            let load_t0 = Instant::now();
            model.load().map_err(|e| e.to_string())?;
            let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

            let input = AneSurface::new(hidden * SPATIAL * 4).map_err(|e| e.to_string())?;
            let qkv = AneSurface::new(qkv_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let z = AneSurface::new(z_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let a = AneSurface::new(ab_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
            let b = AneSurface::new(ab_rows * SPATIAL * 4).map_err(|e| e.to_string())?;

            let (conv_out, metal_conv) = if fused_enabled()
                && conv_kernel > 0
                && conv_kernel <= SPATIAL
                && conv_weights.len() == qkv_rows.saturating_mul(conv_kernel)
            {
                let out = AneSurface::new(qkv_rows * 4).map_err(|e| e.to_string())?;
                let input_metal = unsafe {
                    MetalSharedSurface::from_iosurface(qkv.as_raw_iosurface(), qkv.len())
                };
                let output_metal = unsafe {
                    MetalSharedSurface::from_iosurface(out.as_raw_iosurface(), out.len())
                };
                match (input_metal, output_metal) {
                    (Some(input_metal), Some(output_metal)) => {
                        let kernel = MetalGdnConvSilu::new(
                            &input_metal,
                            &output_metal,
                            conv_weights,
                            qkv_rows,
                            SPATIAL,
                            conv_kernel,
                        );
                        if kernel.is_none() {
                            eprintln!(
                                "qwen4-rs: ANE GDN layer {layer} Metal Conv1D+SiLU continuation unavailable; using CPU conv fallback"
                            );
                            (None, None)
                        } else {
                            (Some(out), kernel)
                        }
                    }
                    _ => {
                        eprintln!(
                            "qwen4-rs: ANE GDN layer {layer} IOSurface->Metal import failed; using CPU conv fallback"
                        );
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };
            let fused = metal_conv.is_some();
            let native_cache_hit = model.native_cache_hit();
            let ephemeral = ephemeral_enabled();
            if ephemeral {
                let unload_t0 = Instant::now();
                model.unload().map_err(|e| e.to_string())?;
                if switch_enabled("QWEN_GDN_ANE_TRACE") {
                    eprintln!(
                        "qwen4-rs: ANE GDN layer {layer} initial unload_ms={:.3}",
                        unload_t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            eprintln!(
                "qwen4-rs: ANE GDN layer {layer} ready (bf16->fp16={convert_ms:.1}ms compile={compile_ms:.1}ms load={load_ms:.1}ms fused_conv={fused} native_cache_hit={native_cache_hit} ephemeral={ephemeral})"
            );
            Ok(Self {
                model,
                input,
                qkv,
                z,
                a,
                b,
                conv_out,
                metal_conv,
                hidden,
                qkv_rows,
                z_rows,
                ab_rows,
                conv_kernel,
                gpu_tail: switch_enabled("QWEN_GDN_ANE_GPU_TAIL"),
                async_fence: None,
                async_channel: None,
                async_slow_score: 0,
                async_samples: 0,
            })
        }

        fn write_input(&mut self, x: &[f32]) -> Result<(), String> {
            if x.len() != self.hidden {
                return Err(format!(
                    "ANE GDN input {} != hidden {}",
                    x.len(),
                    self.hidden
                ));
            }
            let mut map = self.input.write().map_err(|e| e.to_string())?;
            for (channel, &value) in x.iter().enumerate() {
                let bytes = value.to_le_bytes();
                let base = channel * SPATIAL * 4;
                for s in 0..SPATIAL {
                    let off = base + s * 4;
                    map[off..off + 4].copy_from_slice(&bytes);
                }
            }
            Ok(())
        }

        fn read_column(surface: &AneSurface, rows: usize, dst: &mut [f32]) -> Result<(), String> {
            if dst.len() != rows {
                return Err("ANE GDN destination row count mismatch".into());
            }
            let map = surface.read().map_err(|e| e.to_string())?;
            for (row, value) in dst.iter_mut().enumerate() {
                let off = row * SPATIAL * 4;
                *value = f32::from_le_bytes(map[off..off + 4].try_into().unwrap());
            }
            Ok(())
        }

        fn evaluate_async_signal(&mut self, x: &[f32]) -> Result<AneChannelPending, String> {
            if !self.gpu_tail {
                return Err("ANE async GDN requires GPU tail".into());
            }
            if !self.model.is_loaded() {
                let load_t0 = Instant::now();
                self.model.load().map_err(|e| e.to_string())?;
                if switch_enabled("QWEN_GDN_ANE_TRACE") {
                    eprintln!(
                        "qwen4-rs: ANE GDN ephemeral reload_ms={:.3}",
                        load_t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            self.write_input(x)?;
            let signal_value = if let Some(fence) = self.async_fence.as_mut() {
                fence
                    .advance()
                    .ok_or_else(|| "ANE/Metal shared-event value advance failed".to_string())?
            } else {
                self.async_fence = MetalAneFence::new(1);
                if self.async_fence.is_none() {
                    return Err("Metal shared event unavailable".into());
                }
                1
            };
            if self.async_channel.is_none() {
                let fence = self.async_fence.as_ref().unwrap();
                let submit_mode = if switch_enabled("QWEN_GDN_ANE_ASYNC_REALTIME") {
                    2
                } else if switch_enabled("QWEN_GDN_ANE_ASYNC_DIRECT") {
                    1
                } else {
                    0
                };
                let premap = switch_enabled("QWEN_GDN_ANE_ASYNC_PREMAP");
                self.async_channel = Some(
                    unsafe {
                        self.model.async_channel(
                            &[&self.input],
                            &[&self.qkv, &self.z, &self.a, &self.b],
                            0,
                            fence.ane_shared_event(),
                            submit_mode,
                            premap,
                        )
                    }
                    .map_err(|e| e.to_string())?,
                );
            }
            self.async_channel
                .as_mut()
                .unwrap()
                .submit(signal_value)
                .map_err(|e| e.to_string())
        }

        fn evaluate(
            &mut self,
            x: &[f32],
            qkv: &mut [f32],
            z: &mut [f32],
            a: &mut [f32],
            b: &mut [f32],
        ) -> Result<(), String> {
            self.write_input(x)?;
            let request =
                AneRequest::new(&[&self.input], &[&self.qkv, &self.z, &self.a, &self.b], 0)
                    .map_err(|e| e.to_string())?;
            let eval_t0 = Instant::now();
            self.model.evaluate(&request).map_err(|e| e.to_string())?;
            if switch_enabled("QWEN_GDN_ANE_TRACE") {
                eprintln!(
                    "qwen4-rs: ANE eval sync submissions=1 host_waits=1 evaluate_ms={:.6}",
                    eval_t0.elapsed().as_secs_f64() * 1e3
                );
            }
            if self.gpu_tail {
                return Ok(());
            }
            Self::read_column(&self.qkv, self.qkv_rows, qkv)?;
            self.read_aux(z, a, b)?;
            Ok(())
        }

        fn read_aux(&self, z: &mut [f32], a: &mut [f32], b: &mut [f32]) -> Result<(), String> {
            Self::read_column(&self.z, self.z_rows, z)?;
            Self::read_column(&self.a, self.ab_rows, a)?;
            Self::read_column(&self.b, self.ab_rows, b)
        }

        fn conv_silu(
            &mut self,
            current_qkv: &[f32],
            conv_state: &mut [f32],
            y: &mut [f32],
        ) -> Result<(), String> {
            let k = self.conv_kernel;
            let hist = k.saturating_sub(1);
            if self.metal_conv.is_none() || self.conv_out.is_none() {
                return Err("ANE GDN fused Conv1D+SiLU continuation unavailable".into());
            }
            if current_qkv.len() != self.qkv_rows || y.len() != self.qkv_rows {
                return Err("ANE GDN fused Conv1D+SiLU vector shape mismatch".into());
            }
            if conv_state.len() != self.qkv_rows.saturating_mul(hist) {
                return Err(format!(
                    "ANE GDN conv state {} != expected {}x{}",
                    conv_state.len(),
                    self.qkv_rows,
                    hist
                ));
            }

            // ANE writes current qkv into every spatial lane. Preserve the
            // final lane as current and overwrite the first k-1 lanes with
            // the causal qkv history. Metal then consumes this SAME IOSurface.
            if hist != 0 {
                let mut map = self.qkv.write().map_err(|e| e.to_string())?;
                for ch in 0..self.qkv_rows {
                    let base = ch * SPATIAL * 4;
                    for j in 0..hist {
                        let off = base + j * 4;
                        map[off..off + 4].copy_from_slice(&conv_state[ch * hist + j].to_le_bytes());
                    }
                }
            }

            let ran = self.metal_conv.as_mut().expect("checked above").run();
            if !ran {
                return Err("Metal Conv1D+SiLU continuation failed".into());
            }

            let out = self.conv_out.as_ref().expect("checked above");
            let map = out.read().map_err(|e| e.to_string())?;
            for (ch, dst) in y.iter_mut().enumerate() {
                let off = ch * 4;
                *dst = f32::from_le_bytes(map[off..off + 4].try_into().unwrap());
            }
            drop(map);

            // Advance the authoritative host-side conv state only after the
            // Metal command has completed successfully, preserving fallback
            // correctness on a declined/failed accelerator path.
            if hist != 0 {
                for ch in 0..self.qkv_rows {
                    let row = &mut conv_state[ch * hist..(ch + 1) * hist];
                    row.rotate_left(1);
                    row[hist - 1] = current_qkv[ch];
                }
            }
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn warm(
        state: &mut GdnAneState,
        layer: usize,
        hidden: usize,
        qkv_rows: usize,
        z_rows: usize,
        ab_rows: usize,
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        conv_weights: &[f32],
        conv_kernel: usize,
    ) -> bool {
        if !requested(layer) || matches!(state, GdnAneState::Disabled) {
            return false;
        }
        if matches!(state, GdnAneState::Uninitialized) {
            match GdnAneLayer::build(
                layer,
                hidden,
                qkv_rows,
                z_rows,
                ab_rows,
                wqkv,
                wz,
                wa,
                wb,
                conv_weights,
                conv_kernel,
            ) {
                Ok(ready) => *state = GdnAneState::Ready(ready),
                Err(error) => {
                    eprintln!("qwen4-rs: ANE GDN layer {layer} disabled during warmup: {error}");
                    *state = GdnAneState::Disabled;
                    return false;
                }
            }
        }
        matches!(state, GdnAneState::Ready(_))
    }

    pub struct GdnAnePending {
        inner: AneChannelPending,
        submit_ms: f64,
    }

    impl GdnAnePending {
        pub fn submit_ms(&self) -> f64 {
            self.submit_ms
        }
        pub fn finish(self, timeout_ms: u64) -> Result<(), String> {
            self.inner.finish(timeout_ms).map_err(|e| e.to_string())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_input_async(
        state: &mut GdnAneState,
        layer: usize,
        hidden: usize,
        qkv_rows: usize,
        z_rows: usize,
        ab_rows: usize,
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        conv_weights: &[f32],
        conv_kernel: usize,
        x: &[f32],
    ) -> Option<GdnAnePending> {
        if !requested(layer) || matches!(state, GdnAneState::Disabled) {
            return None;
        }
        if !warm(
            state,
            layer,
            hidden,
            qkv_rows,
            z_rows,
            ab_rows,
            wqkv,
            wz,
            wa,
            wb,
            conv_weights,
            conv_kernel,
        ) {
            return None;
        }
        let GdnAneState::Ready(ready) = state else {
            return None;
        };
        if !ready.gpu_tail {
            return None;
        }
        let trace = switch_enabled("QWEN_GDN_ANE_TRACE");
        let t0 = Instant::now();
        match ready.evaluate_async_signal(x) {
            Ok(inner) => {
                let submit_ms = t0.elapsed().as_secs_f64() * 1e3;
                if trace {
                    eprintln!(
                        "qwen4-rs: ANE GDN async layer={layer} submissions=1 host_waits=0 submit_ms={submit_ms:.3}"
                    );
                }
                Some(GdnAnePending { inner, submit_ms })
            }
            Err(error) => {
                eprintln!("qwen4-rs: ANE GDN layer {layer} async submit declined: {error}");
                None
            }
        }
    }

    fn adaptive_enabled() -> bool {
        std::env::var("QWEN_GDN_ANE_ADAPTIVE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    }

    fn env_ms(name: &str, default: f64) -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(default)
    }

    /// Feed one completed async ANE->Metal sample into the opt-in path's
    /// residency circuit breaker. ANE program switches on the M2 can jump from
    /// ~1-5 ms to 75-150 ms when firmware residency thrashes. Rather than keep
    /// paying that cost forever, slowly accumulate a penalty and retire only
    /// persistently slow layers back to the validated Metal path.
    pub fn report_async_sample(
        state: &mut GdnAneState,
        layer: usize,
        submit_ms: f64,
        exposed_wait_ms: f64,
    ) -> bool {
        if ephemeral_enabled() {
            if let GdnAneState::Ready(ready) = state {
                // The pending request and Metal tail have both completed before
                // this hook. Drop the pre-mapped request object before unload;
                // the shared-event fence itself can be reused after reload.
                ready.async_channel = None;
                let unload_t0 = Instant::now();
                if let Err(error) = ready.model.unload() {
                    eprintln!("qwen4-rs: ANE GDN layer {layer} ephemeral unload failed: {error}");
                    *state = GdnAneState::Disabled;
                    return true;
                }
                if switch_enabled("QWEN_GDN_ANE_TRACE") {
                    eprintln!(
                        "qwen4-rs: ANE GDN layer {layer} ephemeral unload_ms={:.3}",
                        unload_t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            return false;
        }
        if !adaptive_enabled() {
            return false;
        }
        let submit_limit = env_ms("QWEN_GDN_ANE_SLOW_SUBMIT_MS", 25.0);
        let wait_limit = env_ms("QWEN_GDN_ANE_SLOW_WAIT_MS", 25.0);
        let retire_score = std::env::var("QWEN_GDN_ANE_SLOW_SCORE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        let mut retire = false;
        let mut score = 0u32;
        if let GdnAneState::Ready(ready) = state {
            ready.async_samples = ready.async_samples.saturating_add(1);
            let slow = submit_ms > submit_limit || exposed_wait_ms > wait_limit;
            if slow {
                ready.async_slow_score = ready.async_slow_score.saturating_add(2);
            } else {
                ready.async_slow_score = ready.async_slow_score.saturating_sub(1);
            }
            score = ready.async_slow_score;
            retire = ready.async_samples >= 2 && score >= retire_score;
        }
        if retire {
            eprintln!(
                "qwen4-rs: ANE GDN layer {layer} adaptively retired to Metal \
                 (submit_ms={submit_ms:.3} exposed_wait_ms={exposed_wait_ms:.3} score={score})"
            );
            *state = GdnAneState::Disabled;
            true
        } else {
            false
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_input(
        state: &mut GdnAneState,
        layer: usize,
        hidden: usize,
        qkv_rows: usize,
        z_rows: usize,
        ab_rows: usize,
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        conv_weights: &[f32],
        conv_kernel: usize,
        x: &[f32],
        qkv: &mut [f32],
        z: &mut [f32],
        a: &mut [f32],
        b: &mut [f32],
    ) -> bool {
        if !requested(layer) || matches!(state, GdnAneState::Disabled) {
            return false;
        }
        if !warm(
            state,
            layer,
            hidden,
            qkv_rows,
            z_rows,
            ab_rows,
            wqkv,
            wz,
            wa,
            wb,
            conv_weights,
            conv_kernel,
        ) {
            return false;
        }
        let GdnAneState::Ready(ready) = state else {
            return false;
        };
        let trace = switch_enabled("QWEN_GDN_ANE_TRACE");
        let t0 = trace.then(Instant::now);
        if let Err(error) = ready.evaluate(x, qkv, z, a, b) {
            eprintln!("qwen4-rs: ANE GDN layer {layer} disabled during evaluate: {error}");
            *state = GdnAneState::Disabled;
            return false;
        }
        if let Some(t0) = t0 {
            eprintln!(
                "qwen4-rs: ANE GDN trace layer={layer} input_ms={:.3} cpu_surface_maps={} cpu_output_bytes={} input_write_bytes={}",
                t0.elapsed().as_secs_f64() * 1e3,
                if ready.gpu_tail { 1 } else { 5 },
                if ready.gpu_tail { 0 } else { (ready.qkv_rows + ready.z_rows + 2 * ready.ab_rows) * 4 },
                ready.hidden * SPATIAL * 4
            );
        }
        true
    }

    /// Borrowed raw views; the caller must finish Metal use before mutating or
    /// dropping this ANE state. No CPU maps are opened here.
    pub fn gpu_surfaces(state: &GdnAneState) -> Option<[(*mut std::ffi::c_void, usize); 4]> {
        let GdnAneState::Ready(ready) = state else {
            return None;
        };
        if !ready.gpu_tail {
            return None;
        }
        Some(
            [&ready.qkv, &ready.z, &ready.a, &ready.b]
                .map(|s| (unsafe { s.as_raw_iosurface() }, s.len())),
        )
    }

    pub fn gpu_fence(state: &GdnAneState) -> Option<&MetalAneFence> {
        let GdnAneState::Ready(ready) = state else {
            return None;
        };
        ready.async_fence.as_ref()
    }

    pub fn materialize(
        state: &GdnAneState,
        qkv: &mut [f32],
        z: &mut [f32],
        a: &mut [f32],
        b: &mut [f32],
    ) {
        let GdnAneState::Ready(ready) = state else {
            return;
        };
        GdnAneLayer::read_column(&ready.qkv, ready.qkv_rows, qkv)
            .and_then(|_| ready.read_aux(z, a, b))
            .expect("ANE output mapping failed; cannot safely continue inference");
    }

    pub fn try_conv_silu(
        state: &mut GdnAneState,
        layer: usize,
        current_qkv: &[f32],
        conv_state: &mut [f32],
        y: &mut [f32],
    ) -> bool {
        let GdnAneState::Ready(ready) = state else {
            return false;
        };
        if ready.metal_conv.is_none() || ready.conv_out.is_none() {
            return false;
        }
        let trace = switch_enabled("QWEN_GDN_ANE_TRACE");
        let t0 = trace.then(Instant::now);
        if let Err(error) = ready.conv_silu(current_qkv, conv_state, y) {
            if fused_enabled() {
                eprintln!("qwen4-rs: ANE GDN layer {layer} fused Conv1D+SiLU declined: {error}");
            }
            return false;
        }
        if let Some(t0) = t0 {
            eprintln!(
                "qwen4-rs: ANE GDN trace layer={layer} conv_ms={:.3} cpu_surface_maps={} cpu_output_bytes={} history_write_bytes={}",
                t0.elapsed().as_secs_f64() * 1e3,
                1 + usize::from(ready.conv_kernel > 1),
                ready.qkv_rows * 4,
                ready.qkv_rows * ready.conv_kernel.saturating_sub(1) * 4
            );
        }
        true
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod imp {
    #[derive(Default)]
    pub enum GdnAneState {
        #[default]
        Uninitialized,
        Disabled,
    }

    pub fn active(_state: &GdnAneState) -> bool {
        false
    }

    /// The dynamic-weight engine does not exist off Apple silicon.
    ///
    /// A unit type rather than a struct with fields: nothing can construct it
    /// here, so there is no state to model, and `build` below always fails.
    pub struct GdnAneDynamicEngine;

    /// Off Apple silicon there is no ANE chain to be pending on. Only reachable
    /// if a caller ignored the `Err` from `evaluate_layer_async`; it mirrors the
    /// Apple type so that call site needs no cfg.
    pub struct GdnAneDynamicPending;

    impl GdnAneDynamicPending {
        pub fn submit_ms(&self) -> f64 {
            0.0
        }
        pub fn finish(self, _timeout_ms: u64) -> Result<f64, String> {
            Err("dynamic ANE is only available on macOS aarch64".into())
        }
    }

    impl GdnAneDynamicEngine {
        pub fn build(
            _hidden: usize,
            _qkv_rows: usize,
            _z_rows: usize,
            _ab_rows: usize,
        ) -> Result<Self, String> {
            Err("dynamic ANE is only available on macOS aarch64".into())
        }

        /// Only reachable through `dynamic_async_enabled()`, which is false
        /// here; returning `Err` matches the Apple type's shape without
        /// pretending a submission happened.
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_layer_async(
            &mut self,
            _layer: usize,
            _x: &[f32],
            _wqkv: &[u8],
            _wz: &[u8],
            _wa: &[u8],
            _wb: &[u8],
        ) -> Result<GdnAneDynamicPending, String> {
            Err("dynamic ANE is only available on macOS aarch64".into())
        }

        /// Apple returns the shared ANE/Metal fence it submits against. No ANE
        /// exists here, so there is never a fence.
        pub fn gpu_fence(&self) -> Option<&logan_metal::MetalAneFence> {
            None
        }

        /// Apple only calls this after a successful `evaluate_layer`; here that
        /// always fails, so this is unreachable in practice. Returning all-null
        /// keeps the caller's `gdn_ane_token` fallback (which also returns 0)
        /// rather than asserting.
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_layer(
            &mut self,
            _layer: usize,
            _x: &[f32],
            _wqkv: &[u8],
            _wz: &[u8],
            _wa: &[u8],
            _wb: &[u8],
        ) -> Result<(), String> {
            Err("dynamic ANE is only available on macOS aarch64".into())
        }

        /// Apple hands the four ANE IOSurfaces to the Metal tail. None exist
        /// off Apple silicon, so the caller's `gdn_ane_token` receives nulls
        /// and declines, taking the CPU tail.
        pub fn gpu_surfaces(&self) -> [(*mut std::ffi::c_void, usize); 4] {
            [(std::ptr::null_mut(), 0); 4]
        }
    }

    /// Always false: the switch exists on every platform so call sites do not
    /// need cfg gates, but only Apple silicon can honour it.
    pub fn dynamic_enabled() -> bool {
        false
    }

    pub fn dynamic_async_enabled() -> bool {
        false
    }

    /// No ANE surface exists to fence against, so there is never a fence.
    ///
    /// Returns the same `logan_metal::MetalAneFence` reference type the Apple
    /// build does, so the caller's `if let (Some(surfaces), Some(fence))` keeps
    /// compiling and simply takes the non-ANE branch.
    pub fn gpu_fence(_state: &GdnAneState) -> Option<&logan_metal::MetalAneFence> {
        None
    }

    pub fn fused_active(_state: &GdnAneState) -> bool {
        false
    }

    pub fn requested(_layer: usize) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn warm(
        _state: &mut GdnAneState,
        _layer: usize,
        _hidden: usize,
        _qkv_rows: usize,
        _z_rows: usize,
        _ab_rows: usize,
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _conv_weights: &[f32],
        _conv_kernel: usize,
    ) -> bool {
        false
    }

    pub struct GdnAnePending;
    impl GdnAnePending {
        pub fn submit_ms(&self) -> f64 {
            0.0
        }
        pub fn finish(self, _timeout_ms: u64) -> Result<(), String> {
            Err("ANE unavailable".into())
        }
    }
    pub fn report_async_sample(
        _state: &mut GdnAneState,
        _layer: usize,
        _submit_ms: f64,
        _exposed_wait_ms: f64,
    ) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_input_async(
        _state: &mut GdnAneState,
        _layer: usize,
        _hidden: usize,
        _qkv_rows: usize,
        _z_rows: usize,
        _ab_rows: usize,
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _conv_weights: &[f32],
        _conv_kernel: usize,
        _x: &[f32],
    ) -> Option<GdnAnePending> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_input(
        _state: &mut GdnAneState,
        _layer: usize,
        _hidden: usize,
        _qkv_rows: usize,
        _z_rows: usize,
        _ab_rows: usize,
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _conv_weights: &[f32],
        _conv_kernel: usize,
        _x: &[f32],
        _qkv: &mut [f32],
        _z: &mut [f32],
        _a: &mut [f32],
        _b: &mut [f32],
    ) -> bool {
        false
    }

    pub fn gpu_surfaces(_state: &GdnAneState) -> Option<[(*mut std::ffi::c_void, usize); 4]> {
        None
    }
    pub fn materialize(
        _state: &GdnAneState,
        _qkv: &mut [f32],
        _z: &mut [f32],
        _a: &mut [f32],
        _b: &mut [f32],
    ) {
    }

    pub fn try_conv_silu(
        _state: &mut GdnAneState,
        _layer: usize,
        _current_qkv: &[f32],
        _conv_state: &mut [f32],
        _y: &mut [f32],
    ) -> bool {
        false
    }
}

pub use imp::*;
