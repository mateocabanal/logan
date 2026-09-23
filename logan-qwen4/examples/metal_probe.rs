//! Metal availability probe for the MLX affine (oQ4) safetensors path.
//!
//! `metal_available()` reads a lazily-initialized flag that only `metal_init()`
//! sets, so the probe must init first; a bare `metal_available()` call reports
//! false on a machine where Metal is fine.
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    let init = logan_metal::metal_init();
    println!("metal_init={init} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    println!("metal_available={}", logan_metal::metal_available());

    // Exercise the exact call the routed-expert GEMM makes: MLX affine 4-bit,
    // group 64, aux bf16, on a 512x2048 matrix.
    const O: usize = 512;
    const I: usize = 2048;
    const BITS: u8 = 4;
    const GROUP: usize = 64;
    let row_bytes = I * BITS as usize / 8;
    let groups = I / GROUP;
    let weights = vec![0x11u8; O * row_bytes];
    let scales = vec![0u8; O * groups * 2];
    let biases = vec![0u8; O * groups * 2];
    let x = vec![0.01f32; I];
    let mut y = vec![0.0f32; O];
    let mut tensor: *mut logan_metal::ColiMetalTensor = std::ptr::null_mut();

    let mut aux = vec![0u8; scales.len() + biases.len()];
    aux[..scales.len()].copy_from_slice(&scales);
    aux[scales.len()..].copy_from_slice(&biases);
    let ok = logan_metal::metal_matmul_mlx_affine(
        &mut tensor, &mut y, &x, &weights, &aux, BITS, GROUP, false, I, O,
    );
    println!("metal_matmul_mlx_affine={ok} y[0]={:.6}", y[0]);

    // Cost of one routed-expert GEMM as the engine issues it: a synchronous
    // dispatch of a [512x2048] 4-bit affine matrix. 600 expert calls per token
    // x 3 matrices = 1800 of these per token, so this number times 1800 is the
    // floor for the routed-expert phase on this path.
    for _ in 0..16 {
        logan_metal::metal_matmul_mlx_affine(
            &mut tensor, &mut y, &x, &weights, &aux, BITS, GROUP, false, I, O,
        );
    }
    let iterations = 200usize;
    let started = Instant::now();
    for _ in 0..iterations {
        logan_metal::metal_matmul_mlx_affine(
            &mut tensor, &mut y, &x, &weights, &aux, BITS, GROUP, false, I, O,
        );
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    println!("dispatch_us={us:.1} matrix_bytes={}", O * I / 2);
    println!(
        "projected_expert_gemm_ms_per_token={:.1}",
        us * 1800.0 / 1e3
    );
    println!("ROUTESCOUT_METAL_GATE: {}", if ok { "PASS" } else { "FALLBACK" });
}
