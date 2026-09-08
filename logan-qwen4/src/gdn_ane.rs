//! Experimental Apple Neural Engine GDN input-projection island.
//!
//! Opt-in only. `QWEN_GDN_ANE=1` enables layer 0 by default; select an
//! inclusive comma-separated set/range with `QWEN_GDN_ANE_LAYERS=0-2,4` or
//! explicitly request every eligible GDN layer with `all`.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use std::{path::PathBuf, time::Instant};

    use logan_ane::{
        AneModel, AneRequest, AneRuntime, AneSurface, CompileOptions, mil::DenseProjection,
    };
    use logan_metal::{MetalGdnConvSilu, MetalSharedSurface};

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
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".cache/logan/ane"))
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
            eprintln!(
                "qwen4-rs: ANE GDN layer {layer} ready (bf16->fp16={convert_ms:.1}ms compile={compile_ms:.1}ms load={load_ms:.1}ms fused_conv={fused})"
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
            self.model.evaluate(&request).map_err(|e| e.to_string())?;
            Self::read_column(&self.qkv, self.qkv_rows, qkv)?;
            Self::read_column(&self.z, self.z_rows, z)?;
            Self::read_column(&self.a, self.ab_rows, a)?;
            Self::read_column(&self.b, self.ab_rows, b)?;
            Ok(())
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
                "qwen4-rs: ANE GDN trace layer={layer} input_ms={:.3}",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        true
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
                "qwen4-rs: ANE GDN trace layer={layer} conv_ms={:.3}",
                t0.elapsed().as_secs_f64() * 1e3
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
