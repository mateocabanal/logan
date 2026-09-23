//! Microbenchmark: is the routed-expert MoE phase latency-bound or
//! bandwidth-bound?
//!
//! `cargo run --release -p logan-qwen4 --example affine_dispatch_probe`
//!
//! EXP-018 measured ~1191 MLX-affine dispatches per decode forward; EXP-029
//! corrected the expert *compute* term to ~335 ms/forward, implying roughly
//! 240 us per dispatch. Each expert projection is only 512x2048 at 4-bit
//! (~590 KB) — about 6 us of UMA traffic at this host's bandwidth. If that gap
//! is real, it is per-command-buffer overhead, and the fix is to encode several
//! independent GEMVs into ONE command buffer.
//!
//! `metal_matmul_mlx_affine_multi` already does that (up to 16 descriptors) and
//! is already used by the GDN fused-input path — but NOT by the expert path,
//! where `MlxLocalExpertSource::eval` calls `matmul` once per matrix and every
//! one of those commits and waits on its own command buffer.
//!
//! The descriptor contract requires every descriptor in one call to share the
//! same input width `I`. On this geometry that permits exactly one useful
//! grouping: all 8 experts' `gate_proj` and `up_proj` consume the SAME token
//! activation, so 16 dispatches collapse into one command buffer. The
//! `down_proj`s each consume their own expert's SwiGLU output, so they cannot
//! join a shared-activation batch.
//!
//! Shapes measured:
//!   serial  24 dispatches, one command buffer each  (today's expert path)
//!   batched  1 multi for the 16 gate/up + 8 single downs = 9 command buffers
//!
//! Output is `PROBE key=value` lines. `bit_identical` compares the two shapes'
//! outputs so a future switch can be justified as numerically neutral.

use logan_qwen4::ffi;

const D_MODEL: usize = 2048;
const D_HIDDEN: usize = 512;
const TOPK: usize = 8;
const BITS: u8 = 4;
const GROUP_SIZE: usize = 64;

struct Quant {
    weights: Vec<u8>,
    aux: Vec<u8>,
    i: usize,
    o: usize,
}

impl Quant {
    fn new(o: usize, i: usize, seed: u64) -> Self {
        let row_bytes = i * BITS as usize / 8;
        let groups = i / GROUP_SIZE;
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let weights: Vec<u8> = (0..o * row_bytes).map(|_| (next() >> 33) as u8).collect();
        // bf16 scale 1.0, bias 0.0. Magnitude does not affect dispatch timing.
        let one_bf16: u16 = 0x3F80;
        let mut aux = Vec::with_capacity(o * groups * 4);
        for _ in 0..o * groups {
            aux.extend_from_slice(&one_bf16.to_le_bytes());
            aux.extend_from_slice(&0u16.to_le_bytes());
        }
        Self { weights, aux, i, o }
    }
}

/// A prepared matmul: which quantized matrix, and the output buffer it targets.
struct Slot {
    q: usize,
    role: Role,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Gate,
    Up,
    Down,
}

/// The three quantized expert roles for every routed expert.
struct Experts {
    gates: Vec<Quant>,
    ups: Vec<Quant>,
    downs: Vec<Quant>,
}

impl Experts {
    fn quant(&self, s: &Slot) -> &Quant {
        match s.role {
            Role::Gate => &self.gates[s.q],
            Role::Up => &self.ups[s.q],
            Role::Down => &self.downs[s.q],
        }
    }
}

fn main() {
    let iters: usize = std::env::var("PROBE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let topk: usize = std::env::var("PROBE_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TOPK);

    if !ffi::metal_init() {
        eprintln!("PROBE metal_unavailable=1");
        std::process::exit(1);
    }
    println!(
        "PROBE d_model={D_MODEL} d_hidden={D_HIDDEN} topk={TOPK} bits={BITS} gs={GROUP_SIZE} iters={iters}"
    );

    let x: Vec<f32> = (0..D_MODEL).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect();

    let gates: Vec<Quant> = (0..topk).map(|e| Quant::new(D_HIDDEN, D_MODEL, 0x1000 + e as u64)).collect();
    let ups: Vec<Quant> = (0..topk).map(|e| Quant::new(D_HIDDEN, D_MODEL, 0x2000 + e as u64)).collect();
    let downs: Vec<Quant> = (0..topk).map(|e| Quant::new(D_MODEL, D_HIDDEN, 0x3000 + e as u64)).collect();

    // Slot order: 8 gate, 8 up, 8 down. gate/up take `x` (D_MODEL); down takes
    // the per-expert SwiGLU output (D_HIDDEN).
    let mut slots: Vec<Slot> = Vec::new();
    for e in 0..topk {
        slots.push(Slot { q: e, role: Role::Gate });
    }
    for e in 0..topk {
        slots.push(Slot { q: e, role: Role::Up });
    }
    for e in 0..topk {
        slots.push(Slot { q: e, role: Role::Down });
    }
    let n = slots.len(); // 24

    let experts = Experts { gates, ups, downs };
    let out_dim = |r: Role| if r == Role::Down { D_MODEL } else { D_HIDDEN };

    // Per-expert SwiGLU activation stands in for the real hidden vector; only
    // its width matters for dispatch cost.
    let down_x: Vec<f32> = (0..D_HIDDEN).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();

    // ---- Warm up both shapes once so buffer/pipeline setup is not timed ----
    {
        let mut ys: Vec<Vec<f32>> = slots.iter().map(|s| vec![0.0_f32; out_dim(s.role)]).collect();
        let mut ts: Vec<*mut ffi::ColiMetalTensor> = vec![std::ptr::null_mut(); n];
        run_serial(&slots, &experts, &mut ys, &mut ts, &x, &down_x);
        run_batched(&slots, &experts, &mut ys, &mut ts, &x, &down_x);
    }

    // ---- Arm: serial, one command buffer per matrix ------------------------
    let serial_us = {
        let mut ys: Vec<Vec<f32>> = slots.iter().map(|s| vec![0.0_f32; out_dim(s.role)]).collect();
        let mut ts: Vec<*mut ffi::ColiMetalTensor> = vec![std::ptr::null_mut(); n];
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            run_serial(&slots, &experts, &mut ys, &mut ts, &x, &down_x);
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    // ---- Arm: batched, 16 gate/up in one command buffer --------------------
    let (batched_us, groups) = {
        let mut ys: Vec<Vec<f32>> = slots.iter().map(|s| vec![0.0_f32; out_dim(s.role)]).collect();
        let mut ts: Vec<*mut ffi::ColiMetalTensor> = vec![std::ptr::null_mut(); n];
        let mut groups = 0usize;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            groups = run_batched(&slots, &experts, &mut ys, &mut ts, &x, &down_x);
        }
        (t0.elapsed().as_secs_f64() * 1e6 / iters as f64, groups)
    };

    println!("PROBE serial_us_per_token={serial_us:.2}");
    println!("PROBE batched_us_per_token={batched_us:.2}");
    println!("PROBE serial_cmd_buffers={n}");
    println!("PROBE batched_cmd_buffers={groups}");
    println!(
        "PROBE speedup={:.3}",
        if batched_us > 0.0 { serial_us / batched_us } else { 0.0 }
    );
    println!("PROBE serial_us_per_dispatch={:.2}", serial_us / n as f64);
    println!(
        "PROBE batched_us_per_dispatch={:.2}",
        if groups > 0 { batched_us / groups as f64 } else { 0.0 }
    );

    // ---- Numerical agreement ------------------------------------------------
    let mut ys_a: Vec<Vec<f32>> = slots.iter().map(|s| vec![0.0_f32; out_dim(s.role)]).collect();
    let mut ts_a: Vec<*mut ffi::ColiMetalTensor> = vec![std::ptr::null_mut(); n];
    run_serial(&slots, &experts, &mut ys_a, &mut ts_a, &x, &down_x);

    let mut ys_b: Vec<Vec<f32>> = slots.iter().map(|s| vec![0.0_f32; out_dim(s.role)]).collect();
    let mut ts_b: Vec<*mut ffi::ColiMetalTensor> = vec![std::ptr::null_mut(); n];
    run_batched(&slots, &experts, &mut ys_b, &mut ts_b, &x, &down_x);

    let mut max_abs = 0.0_f32;
    for (a, b) in ys_a.iter().zip(ys_b.iter()) {
        for (va, vb) in a.iter().zip(b.iter()) {
            max_abs = max_abs.max((va - vb).abs());
        }
    }
    // Bytes moved by the serial arm, so a caller can compute achieved GB/s.
    let bytes: usize = slots
        .iter()
        .map(|s| {
            let q = experts.quant(s);
            q.weights.len() + q.aux.len()
        })
        .sum();
    println!("PROBE moved_bytes={bytes}");
    println!(
        "PROBE serial_gbs={:.1}",
        bytes as f64 / serial_us / 1e3
    );
    println!(
        "PROBE batched_gbs={:.1}",
        bytes as f64 / batched_us / 1e3
    );
    println!("PROBE max_abs_diff={max_abs:.9}");
    println!("PROBE bit_identical={}", max_abs == 0.0);
}

fn run_serial(
    slots: &[Slot],
    experts: &Experts,
    ys: &mut [Vec<f32>],
    ts: &mut [*mut ffi::ColiMetalTensor],
    x: &[f32],
    down_x: &[f32],
) {
    for (m, slot) in slots.iter().enumerate() {
        let q = experts.quant(slot);
        let input = if slot.role == Role::Down { down_x } else { x };
        let ok = ffi::metal_matmul_mlx_affine(
            &mut ts[m],
            &mut ys[m],
            input,
            &q.weights,
            &q.aux,
            BITS,
            GROUP_SIZE,
            false,
            q.i,
            q.o,
        );
        assert!(ok, "serial MLX-affine dispatch declined");
    }
}

/// Batch every consecutive run of same-`I` gate/up matrices into one command
/// buffer. Returns the number of command buffers submitted.
fn run_batched(
    slots: &[Slot],
    experts: &Experts,
    ys: &mut [Vec<f32>],
    ts: &mut [*mut ffi::ColiMetalTensor],
    x: &[f32],
    down_x: &[f32],
) -> usize {
    let mut groups = 0usize;
    let mut m = 0usize;
    while m < slots.len() {
        // The gate/up block: every one of these shares `x` and has i == D_MODEL.
        if slots[m].role != Role::Down {
            let mut end = m;
            while end < slots.len() && slots[end].role != Role::Down {
                end += 1;
            }
            let mut descs: Vec<ffi::MlxAffineMatmulDesc> =
                Vec::with_capacity(end - m);
            // Split the output buffers so each descriptor owns its own `y`.
            for (k, y) in (m..end).zip(ys[m..end].iter_mut()) {
                let q = experts.quant(&slots[k]);
                descs.push(ffi::MlxAffineMatmulDesc {
                    tensor: ts[k],
                    y,
                    weights: &q.weights,
                    aux: &q.aux,
                    bits: BITS,
                    group_size: GROUP_SIZE,
                    aux_fp16: false,
                    i: q.i,
                    o: q.o,
                    x: None,
                });
            }
            assert!(
                ffi::metal_matmul_mlx_affine_multi(x, &mut descs),
                "batched MLX-affine dispatch declined"
            );
            for (k, d) in (m..end).zip(descs.iter()) {
                ts[k] = d.tensor;
            }
            groups += 1;
            m = end;
            continue;
        }
        // A down_proj consumes its own expert's activation, so it cannot join a
        // shared-activation batch.
        let q = experts.quant(&slots[m]);
        assert!(
            ffi::metal_matmul_mlx_affine(
                &mut ts[m],
                &mut ys[m],
                down_x,
                &q.weights,
                &q.aux,
                BITS,
                GROUP_SIZE,
                false,
                q.i,
                q.o,
            ),
            "batched down_proj dispatch declined"
        );
        groups += 1;
        m += 1;
    }
    groups
}
