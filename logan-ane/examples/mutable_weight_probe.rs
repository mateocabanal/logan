use logan_ane::{AneRuntime, CompileOptions, DenseProjection};
use std::ffi::{CStr, c_char, c_void};
unsafe extern "C" {
    fn logan_ane_probe_mutable_weight_buffer(
        model: *mut c_void,
        symbol: *const c_char,
        buffer_id: u64,
        size: *mut u64,
        error: *mut i8,
        error_cap: usize,
    ) -> i32;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 16usize;
    let mut w = vec![0u16; n * n];
    for i in 0..n {
        w[i * n + i] = 0x3c00;
    }
    let p = logan_ane::mil::parallel_dense_fp16_f32_io(n, 16, &[DenseProjection::new("w", n, w)])?;
    let rt = AneRuntime::load()?;
    let mut model = rt.compile(&p, CompileOptions::default())?;
    model.load()?;
    for symbol in ["main", "main_0", "default"] {
        let cs = std::ffi::CString::new(symbol)?;
        for id in 0..8u64 {
            let mut sz = 0u64;
            let mut e = [0i8; 512];
            let rc = unsafe {
                logan_ane_probe_mutable_weight_buffer(
                    model.as_raw_object(),
                    cs.as_ptr(),
                    id,
                    &mut sz,
                    e.as_mut_ptr(),
                    e.len(),
                )
            };
            let msg = unsafe { CStr::from_ptr(e.as_ptr()) }.to_string_lossy();
            println!("symbol={symbol} id={id} rc={rc} size={sz} err={msg}");
        }
    }
    Ok(())
}
