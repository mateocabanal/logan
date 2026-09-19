//! CUDA execution for quantized GGML rows, next to the [`dot_row`] oracle.
//!
//! ## Why this lives here and not in `logan-core`
//!
//! `logan_core::cuda` owns the *mechanism*: it finds a CUDA runtime, opens a
//! context, compiles a source string through NVRTC and moves bytes. It knows
//! nothing about GGML, and that is deliberate — the core is engine-neutral and
//! has no business knowing what a Q4_K block is.
//!
//! This module owns the *format*. That knowledge already lives in this crate,
//! because `GgmlType::block_geometry` and `q4k_scale_min` in [`super::ggufsource`]
//! define the layout. Putting the kernel anywhere else would mean restating the
//! block layout in a second place and trusting two copies of a bit-twiddling
//! routine to stay in sync. The split also puts the kernel next to the thing it
//! must agree with: [`dot_row`] is the oracle, and the parity test below calls
//! both.
//!
//! ## What this accelerates
//!
//! The model that host runs is Q4_K_M. Its experts are `WtBytes::Gguf`: quantized
//! blocks range-read straight from the file and deliberately never decoded
//! (`ggufload::load_expert`). A quantized run therefore executes [`dot_row`] per
//! output row — *not* `logan_core::math::matmul`, which handles only `Wt.f`
//! (f32) and `Wt.bytes` (bf16) and is never entered on this path.
//!
//! So this is the hot kernel, and it consumes GGML blocks directly: no
//! dequantized copy is materialized on either side.
//!
//! ## Q4_K block layout
//!
//! 256 logical values in 144 bytes, matching [`dot_row`]'s `GgmlType::Q4K` arm:
//!
//! | offset | size | meaning                                        |
//! |--------|------|------------------------------------------------|
//! | 0      | 2    | `d` — f16 super-block scale                     |
//! | 2      | 2    | `dmin` — f16 super-block minimum                |
//! | 4      | 12   | 8 six-bit scales and 8 six-bit minima, packed   |
//! | 16     | 128  | 256 four-bit quants, two per byte               |
//!
//! Eight groups of 32 values. Group `g` takes its quants from byte
//! `(g/2)*32 + k` — low nibble for even `g`, high nibble for odd — and its
//! `(scale, min)` from `q4k_scale_min`, whose six-bit unpacking is the fiddly
//! part. The kernel reproduces that function rather than a paraphrase of it.
//!
//! ## Numerics and the token-identity gate
//!
//! Per output row the accumulation is sequential in both implementations, and
//! both nest the loops the same way (blocks outer, groups, then the 32 values),
//! so the two agree far more closely than a general f32 reassociation would
//! suggest — but not to bit-equality, because `ds * q - dm` times `x` may
//! contract into an fma chain on one side and not the other. The parity test
//! *measures* the divergence on structured and adversarial inputs rather than
//! asserting a hoped-for bound.
//!
//! ## Opt-in: this kernel is slower than the CPU it replaces
//!
//! `LOGAN_CUDA=1` enables it; the default is **off**. That is not caution, it is
//! the measurement: this kernel is bit-exact against [`dot_row`], but it runs at
//! **0.34x–0.58x of the 12-thread scalar CPU path** at real expert shapes
//! (o=640/i=2560 and o=2048/i=2560; the CPU arm's `parallel` branch is active at
//! those sizes). Enabling it by default would be a silent 2–3x regression.
//!
//! The cause is the parallelisation strategy, and it is worth recording so the
//! next attempt does not repeat it. The kernel launches **one thread per output
//! row**, so occupancy is set by `ceil(o / block)` — 3 blocks at o=640 on a
//! 20-SM card, about 15%. Each thread then walks all of its row's Q4_K blocks in
//! a serial, dependent chain with no tiling. Measured:
//!
//! - kernel time scales exactly with per-thread work (0.185 / 0.285 / 0.590 /
//!   1.080 ms as `i` goes 256 / 512 / 1280 / 2560 at fixed o=640), so it is
//!   work-bound, not launch-bound;
//! - launch + sync + a 4-byte readback is only 0.027 ms/call, so overhead is
//!   not the problem;
//! - the weight upload is 8–12% of the call, so PCIe is not the problem either;
//! - us/row is flat (1.59 → 1.33) from o=640 to o=12800 and *worsens* at
//!   o=25600, so more blocks do not rescue it.
//!
//! What would: block-level tiling so many threads share one row, turning the
//! o-only parallelism into real occupancy. A 1080 does ~8 TFLOP/s fp32 against
//! the ~3.2 MFLOP here, so the headroom is orders of magnitude, not a few
//! percent.
//!
//! ### The tiled kernel: measured, faster than the row kernel, still not enough
//!
//! `q4k_dot_row_tiled` implements that tiling — one block per row, the row's
//! Q4_K groups split across the block, partials combined through shared memory —
//! and it is what the opt-in now selects (`LOGAN_CUDA_TILED=0` reverts to the
//! per-row kernel). Measured on the GTX 1080, i=2560, varying o:
//!
//! | o      | row kernel us/row | tiled us/row |
//! |--------|-------------------|--------------|
//! | 640    | 1.592             | 1.353        |
//! | 2560   | 1.354             | 1.319        |
//! | 6400   | 1.326             | 1.308        |
//! | 12800  | 1.351             | 1.305        |
//! | 25600  | 1.908             | 1.304        |
//!
//! So it works: per-row cost is now **flat** where the row kernel degraded as `o`
//! grew, and it is 1.05–1.46x faster. It is also still bit-exact against
//! `dot_row` (0e0 max abs and rel on every shape measured), which is not
//! something a tree reduction promises — treat that as a measured fact about
//! these inputs rather than a guarantee, and the tolerance assertion in
//! `q4k_tiled_matches_dot_row_within_f32_tolerance` is what guards it.
//!
//! It is **not enough**. The tiled kernel still runs at 0.33–0.59x of the
//! 12-thread CPU oracle, so the opt-in stays off by default.
//!
//! ### Two hypotheses tried, both disproven — do not retry these
//!
//! The first was the group-count theory: a 2560-value row is 80 groups, so a
//! 256-thread block "wastes" 176 threads and the 8-round shared reduction must
//! cost as much as the work it combines. Sized to match, that predicts a real
//! win. Measured (o=640, i=2560, upload skipped), sweeping the block:
//!
//! | block threads | ms/call |
//! |---------------|---------|
//! | 64            | 0.864   |
//! | 128           | 0.863   |
//! | 256           | 0.866   |
//! | 512           | 0.878   |
//!
//! Flat. The reduction is not the bottleneck, and neither is idle-thread count.
//! The default is therefore a conventional 256, and `tiled_block_threads`
//! documents the negative result rather than keeping a group-derived heuristic
//! that buys nothing.
//!
//! (Sizing exposed a real *correctness* trap on the way, which is worth keeping:
//! the halving reduction is only sound for a power-of-two thread count. 80
//! threads silently drops partials — 6.2% error, and 143% at 96 — so the host
//! rounds every request up to a power of two. See `tiled_block_threads`.)
//!
//! So the cost is neither occupancy, nor the reduction, nor the transfer (8–12%
//! of the call), nor launch overhead (0.027 ms). What remains, in the order I
//! would try it: each thread doing several groups so the row walk amortises into
//! a longer independent chain; and removing the per-call full
//! `cuCtxSynchronize` + DtoH, which serialises the device and prevents any
//! overlap. That second one is the largest structural difference from the CPU
//! path, where 12 cores all work at once.
//!
//! See `logan_core::cuda`'s module header for `LOGAN_CUDA_LIB_DIR` and the
//! fail-closed rules.
//!
//! ## Pascal (sm_61) and bf16
//!
//! The GTX 1080 is sm_61. Native bf16 arithmetic is sm_80+, and Pascal has no
//! bf16 hardware path of any kind. That does **not** affect this kernel: Q4_K
//! dequantizes to f32 and every operation below is f32. There is no bf16
//! anywhere in the compute path, so the limitation is not exercised here. It
//! *would* matter for a bf16-weight kernel, which is part of why none was
//! written — see [`available_for`].

use logan_core::cuda::{DeviceBuf, Kernel, compile, enabled, launches, synchronize};
use std::sync::{LazyLock, Mutex};

use super::GgmlType;

/// The kernel, compiled once per process.
///
/// One thread per output row, each walking that row's blocks serially. That is
/// the simplest thing that works and it mirrors [`dot_row`]'s own structure, so
/// the accumulation orders correspond. This kernel is bandwidth-bound: a Q4_K
/// row of 256 values is 144 bytes read to produce one f32, so the card's
/// 352 GB/s matters far more than its FLOPs, and a serial reduction per thread
/// already saturates memory. Shared-memory tiling would add complexity to a
/// kernel that is not compute-bound.
///
/// `__restrict__` on every pointer because the buffers never alias.
const KERNEL_COMMON: &str = r#"
// f16 -> f32, transliterating `f16_to_f32` in ggufsource.rs.
//
// Deliberately NOT `__half2float`. The oracle converts in software, and on
// sm_61 the hardware conversion differs for denormals and NaN payloads. Since
// the whole value of this kernel is agreeing with the oracle, the conversion is
// copied rather than delegated -- otherwise parity would be measuring a
// conversion mismatch instead of the quant logic.
__device__ __forceinline__ float logan_f16_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000)) << 16;
    int exp = (int)((bits >> 10) & 0x1f);
    unsigned int frac = (unsigned int)(bits & 0x03ff);
    unsigned int out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            unsigned int mant = frac;
            int e = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            out = sign | ((unsigned int)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((unsigned int)(exp - 15 + 127) << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

// Six-bit scale/minimum unpacking.
//
// A transliteration of `q4k_scale_min`. The packing is the K-quant quirk: the
// first four scales live in the low six bits of bytes 0..4, the first four
// minima in the low six bits of bytes 4..8, the high two bits of bytes 0..4
// carry the top two bits of scales 4..8, and bytes 8..12 hold the low four bits
// of scale j in their low nibble and of min j in their high nibble. Rewriting
// this "more clearly" would break parity with the oracle, which is the only
// thing making this kernel trustworthy.
__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q,
                                              int* scale, int* mn) {
    if (j < 4) {
        *scale = q[j] & 63;
        *mn    = q[j + 4] & 63;
    } else {
        *scale = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *mn    = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

"#;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// The bit-exact per-row kernel: one thread per output row.
///
/// Kept intact and separate from the tiled variant so that the verified,
/// bit-exact behaviour stays available even if the tiled kernel does not work
/// out. Selected by [`prefer_tiled`].
const KERNEL_ROW_BODY: &str = r#"
// One output element: dot one Q4_K row with the f32 activation.
//
// `rows` holds `o` consecutive rows of `(i / 256)` 144-byte blocks. The loop
// over blocks is outer and the loop over the 8 groups inner, exactly as the
// oracle nests them, so the f32 addition sequence matches.
extern "C" __global__ void q4k_dot_row(
    const unsigned char* __restrict__ rows,
    const float* __restrict__ x,
    float* __restrict__ y,
    int o,
    int i)
{
    int oo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oo >= o) return;

    int blocks = i / 256;
    long row_off = (long)oo * (long)blocks * 144L;

    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const unsigned char* blk = rows + row_off + (long)b * 144L;
        const float* xb = x + b * 256;

        float d    = logan_f16_to_f32(*(const unsigned short*)(blk));
        float dmin = logan_f16_to_f32(*(const unsigned short*)(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;

        for (int g = 0; g < 8; ++g) {
            int scale, mn;
            q4k_scale_min(g, scales, &scale, &mn);
            float ds = d * (float)scale;
            float dm = dmin * (float)mn;
            int pair = g / 2;
            bool high = (g & 1) != 0;
            int xoff = g * 32;
            int qoff = pair * 32;
            for (int k = 0; k < 32; ++k) {
                unsigned char packed = qs[qoff + k];
                int q = high ? (packed >> 4) : (packed & 0x0f);
                acc += (ds * (float)q - dm) * xb[xoff + k];
            }
        }
    }
    y[oo] = acc;
}
"#;

/// Tiled variant: many threads cooperate on one row.
///
/// The measured problem with the per-row `q4k_dot_row` is that it launches
/// one thread per output row, so occupancy is capped by `ceil(o/block)` — 3
/// blocks at o=640 on a 20-SM card. This kernel instead fixes one *block* per
/// row and splits the row's Q4_K groups across the block's threads, then
/// combines their partial sums through shared memory.
///
/// Work granularity is one group (32 values) per thread, so `ds`/`dm` are still
/// computed once per group and the inner 32-value accumulation keeps exactly the
/// order [`dot_row`] uses. What changes is only how the per-group partials are
/// combined: a tree reduction instead of a serial walk, which is why this kernel
/// is *not* bit-exact (see `q4k_dot_row_tiled`).
///
/// Shared memory: `blockDim.x` floats. `blockDim.x` must be a power of two so
/// the halving reduction terminates cleanly.
const KERNEL_TILED_BODY: &str = r#"
// One block per output row; the row's Q4_K groups are split across the block.
//
// `groups_per_kgroup` is 8 (a 256-value Q4_K block is 8 groups of 32). Threads
// stride over the row's groups, so a row with more groups than threads still
// covers every group.
extern "C" __global__ void q4k_dot_row_tiled(
    const unsigned char* __restrict__ rows,
    const float* __restrict__ x,
    float* __restrict__ y,
    int o,
    int i)
{
    int row = blockIdx.x;
    if (row >= o) return;
    int tid = threadIdx.x;

    const int groups_per_kgroup = 8;
    int nblocks = i / 256;                              // Q4_K blocks per row
    int ngroups = nblocks * groups_per_kgroup;
    long row_off = (long)row * (long)nblocks * 144L;

    float acc = 0.0f;
    for (int g = tid; g < ngroups; g += blockDim.x) {
        int b = g / groups_per_kgroup;
        int grp = g % groups_per_kgroup;

        const unsigned char* blk = rows + row_off + (long)b * 144L;
        float d    = logan_f16_to_f32(*(const unsigned short*)(blk));
        float dmin = logan_f16_to_f32(*(const unsigned short*)(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;

        int scale, mn;
        q4k_scale_min(grp, scales, &scale, &mn);
        float ds = d * (float)scale;
        float dm = dmin * (float)mn;

        int pair = grp / 2;
        bool high = (grp & 1) != 0;
        // Absolute offsets into `x`: the group's 32 values.
        int xoff = b * 256 + grp * 32;
        int qoff = pair * 32;

        // Identical nesting to the per-row kernel, so the within-group
        // accumulation order is preserved exactly.
        for (int k = 0; k < 32; ++k) {
            unsigned char packed = qs[qoff + k];
            int q = high ? (packed >> 4) : (packed & 0x0f);
            acc += (ds * (float)q - dm) * x[xoff + k];
        }
    }

    // Combine this block's partials. Threads with no group contribute exactly
    // 0.0f, which is an identity for the addition below, so a stride loop that
    // covers fewer groups than threads is still correct.
    extern __shared__ float partial[];
    partial[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}
"#;

/// A compiled kernel plus its reusable device buffers.
///
/// Held behind one `Mutex` because the buffers are shared mutable state: two
/// threads uploading different weight rows into the same allocation would
/// corrupt each other. The lock is per-*call*, not per-lifetime, so the
/// steady-state path is lock, upload, launch, read back. A per-thread buffer
/// set would remove the contention; it is not worth it while the caller
/// computes rows one at a time (see the module limits).
struct State {
    /// Bit-exact per-row kernel. Always present.
    kernel_row: Kernel,
    /// Tiled kernel, when it compiled. Absent means the row kernel is used.
    ///
    /// Compiled at init and kept loaded rather than compiled lazily: the cost is
    /// one extra NVRTC invocation at startup, and a lazy compile would put a
    /// second failure mode in the middle of a hot path.
    kernel_tiled: Option<Kernel>,
    /// Quantized weight rows, uploaded per call.
    w: DeviceBuf,
    /// The activation vector.
    x: DeviceBuf,
    /// The output vector.
    y: DeviceBuf,
    /// A cheap fingerprint of what `w` currently holds, so an unchanged upload
    /// is skipped. A caller looping over rows of one matrix otherwise re-sends
    /// the same bytes every call, and comparing a hash is far cheaper than a
    /// PCIe transfer.
    w_key: Option<(u64, usize)>,
    /// Whether the run that produced these buffers chose the tiled kernel.
    ///
    /// Recorded so the two kernels can never be mixed within one weight buffer
    /// set without the caller knowing.
    tiled: bool,
}

static STATE: LazyLock<Mutex<Option<State>>> = LazyLock::new(|| Mutex::new(init()));

fn init() -> Option<State> {
    if !enabled() {
        return None;
    }
    let kernel_row = compile(&row_source(), "q4k_dot_row")?;
    // A failure here is not fatal: the row kernel is bit-exact and verified, so
    // an NVRTC hiccup on the tiled source degrades to a slower correct answer
    // rather than disabling the backend.
    let kernel_tiled = compile(&tiled_source(), "q4k_dot_row_tiled");
    Some(State {
        kernel_row,
        kernel_tiled,
        w: DeviceBuf::new(),
        x: DeviceBuf::new(),
        y: DeviceBuf::new(),
        w_key: None,
        tiled: prefer_tiled(),
    })
}

/// Whether the Q4_K CUDA kernel compiled and loaded on this machine.
///
/// Narrow on purpose: this is "the kernel is loaded", not "a GPU is present".
/// It is the runtime probe `logan_core::cuda`'s header describes — the backend
/// is selected only through a real NVRTC compile, never through enumerating a
/// device that might not be usable.
pub fn available() -> bool {
    STATE.lock().map(|s| s.is_some()).unwrap_or(false)
}

/// Why the kernel is unavailable, if it is.
pub fn unavailable_reason() -> Option<String> {
    if available() {
        return None;
    }
    logan_core::cuda::init_error().or_else(|| Some("device init returned nothing".to_string()))
}

/// FNV-1a over the weight bytes, for the upload-skip comparison.
///
/// Not a security hash: it decides whether two byte slices are worth assuming
/// equal. A collision would skip an upload and put a stale matrix in VRAM,
/// producing a wrong answer, so the fingerprint is paired with the length in
/// [`State::w_key`], which is what makes a collision need a same-length
/// adversarial pair rather than a birthday accident.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `y[o] = x[.] · w[o,.]` for Q4_K rows, on the GPU.
///
/// `weights` holds `o * i / 256` consecutive 144-byte Q4_K blocks. `x` is `i`
/// activations. Mirrors [`dot_row`] with `GgmlType::Q4K`; the parity test in
/// this module is what establishes that.
///
/// Returns `None` for any reason — no device, the opt-in unset, a geometry the
/// kernel cannot express, a driver error — so a caller falls back to
/// [`dot_row`] rather than failing. A GPU that is busy or out of memory should
/// degrade inference, not break it.
///
/// ## What is deliberately not attempted
///
/// - **Weight residency.** Weights are uploaded per call. An expert set is far
///   larger than 8 GiB, so this is a transfer, not a cache; the only reuse
///   avoided is the identical-consecutive-upload case. A real cache needs an
///   eviction policy and a VRAM budget, which is a design decision about the
///   expert store rather than about this kernel.
/// - **Batched dispatch.** The oracle is defined per row, so a row-wise kernel
///   is the one whose correctness is checkable against it.
/// - **Every dtype but Q4_K.** See [`available_for`].
pub fn q4k_dot_row(y: &mut [f32], x: &[f32], weights: &[u8], o: usize, i: usize) -> Option<()> {
    if o == 0 || i == 0 {
        return None;
    }
    // The kernel takes `i / 256` blocks per row with no tail handling, so a
    // ragged `i` is refused rather than silently mis-computed. Refusing
    // preserves the "never a wrong answer" property; the caller stays on the
    // oracle.
    if i % 256 != 0 || y.len() < o || x.len() < i {
        return None;
    }
    let row_bytes = (i / 256) * 144;
    if weights.len() < o * row_bytes {
        return None;
    }

    let mut guard = STATE.lock().ok()?;
    let st = guard.as_mut()?;

    // The activation changes with every token, so it is always re-sent.
    st.x.upload(as_bytes(x))?;

    let key = (fnv1a(&weights[..o * row_bytes]), o * row_bytes);
    if st.w_key != Some(key) {
        st.w.upload(&weights[..o * row_bytes])?;
        st.w_key = Some(key);
    }

    st.y.ensure(o * 4)?;

    let w_ptr = st.w.ptr()?;
    let x_ptr = st.x.ptr()?;
    let y_ptr = st.y.ptr()?;
    let o_i = o as i32;
    let i_i = i as i32;

    let mut params: [*mut std::ffi::c_void; 5] = [
        &w_ptr as *const _ as *mut std::ffi::c_void,
        &x_ptr as *const _ as *mut std::ffi::c_void,
        &y_ptr as *const _ as *mut std::ffi::c_void,
        &o_i as *const _ as *mut std::ffi::c_void,
        &i_i as *const _ as *mut std::ffi::c_void,
    ];
    if let (true, Some(tiled)) = (st.tiled, st.kernel_tiled.as_ref()) {
        // One block per output row, with the block's threads splitting that
        // row's groups. The grid is therefore `o` blocks rather than
        // `ceil(o/block)` — which is the whole point: occupancy stops depending
        // on `o`.
        //
        // The block size is derived from the group count and is always a power
        // of two; see `tiled_block_threads` for why that is a correctness
        // requirement rather than a sizing preference.
        let block = tiled_block_threads((i / 256) * 8);
        // SAFETY of the shared size: the kernel declares `extern __shared__ float
        // partial[]` and writes `partial[tid]` for tid < blockDim.x, so the
        // allocation must be at least block * 4 bytes.
        tiled.launch((o as u32, 1, 1), (block, 1, 1), block * 4, &mut params)?;
    } else {
        let block = 256u32;
        st.kernel_row
            .launch(((o as u32).div_ceil(block), 1, 1), (block, 1, 1), 0, &mut params)?;
    }
    synchronize()?;

    st.y.download(as_bytes_mut(&mut y[..o]))?;
    Some(())
}

/// Whether this backend may be used for a given dtype.
///
/// Q4_K only. The other `GgmlType`s the oracle handles (F32, F16, BF16, Q5_0,
/// Q8_0, Q6_K) have no kernel here and fall back to [`dot_row`]. Adding one is a
/// second `extern "C" __global__` in [`KERNEL_ROW`] plus an arm here; the
/// loading, buffer reuse and parity scaffolding are already in place.
pub fn available_for(dtype: GgmlType) -> bool {
    dtype == GgmlType::Q4K && available()
}

/// The device this backend would use, for a capability line.
///
/// Read from the driver, so a report names the card that actually ran rather
/// than the one that was expected.
pub fn device_name() -> Option<String> {
    logan_core::cuda::device_name()
}

/// Kernels launched so far, as evidence the GPU path executed.
///
/// Assert on this rather than trusting [`available`]: a loaded kernel that never
/// launched is exactly the "advertised but never executed" failure, and a parity
/// test that passes without a launch has tested the CPU.
pub fn kernel_launches() -> u64 {
    launches()
}

/// Reinterpret an `&[f32]` as bytes for upload.
///
/// `f32` has no padding and no invalid bit patterns, so this is total. A local
/// helper rather than pulling in `bytemuck` for two calls — not adding a
/// dependency is the point of this backend.
fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Mutable counterpart of [`as_bytes`].
///
/// # Safety
///
/// The returned slice shares `v`'s storage, so it must not outlive the borrow
/// and must not be aliased by another live reference. Both are enforced by the
/// signature: the exclusive borrow of `v` lasts as long as the return value.
fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// Kernel selection
// ---------------------------------------------------------------------------

/// Full CUDA source for the bit-exact per-row kernel.
///
/// `KERNEL_COMMON` holds `logan_f16_to_f32` and `q4k_scale_min`, which BOTH
/// kernels need. It is prepended here rather than copied into each body, so
/// there is exactly one copy of the bit-twiddling that parity depends on — two
/// copies could drift apart on a later edit and the drift would show up only as
/// a small numerical error.
fn row_source() -> String {
    format!("{KERNEL_COMMON}{KERNEL_ROW_BODY}")
}

/// Full CUDA source for the tiled kernel. See [`row_source`].
fn tiled_source() -> String {
    format!("{KERNEL_COMMON}{KERNEL_TILED_BODY}")
}

/// How many threads the tiled kernel should use per row.
///
/// Default 256, always a power of two. Two separate findings shaped this:
///
/// **A power of two is a correctness requirement, not a preference.** The
/// kernel's shared-memory reduction is a halving tree, and a halving tree over a
/// non-power-of-two thread count *silently drops partials*. Simulating the exact
/// reduction: 80 slots (the literal group count for i=2560) reports a result 6.2%
/// wrong, and 96 slots 143% wrong, while 64 and 128 are exact. Rounding up is
/// exact because padding threads contribute `0.0f`, the addition identity
/// (measured residual ~4e-15, f32 noise). So `LOGAN_CUDA_BLOCK=80` is rounded to
/// 128 rather than honoured — an asked-for 80 would be wrong, not merely slower.
///
/// **The block size does not matter for speed.** The obvious theory was that a
/// 256-thread block wastes 176 threads on an 80-group row and that the
/// reduction costs as much as the work it combines. Measuring the sweep
/// (o=640, i=2560, upload-skipped): 64 → 0.864 ms, 128 → 0.863, 256 → 0.866,
/// 512 → 0.878. Flat. The theory was wrong, and `tiled_block_threads` was
/// rewritten to say so rather than keep a group-derived default that buys
/// nothing. `LOGAN_CUDA_BLOCK` is retained because it is how that sweep is
/// reproduced; it is not a tuning knob with a known-good setting.
fn tiled_block_threads(_ngroups: usize) -> u32 {
    std::env::var("LOGAN_CUDA_BLOCK")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(DEFAULT_BLOCK_THREADS)
        .min(MAX_BLOCK_THREADS)
        .next_power_of_two()
        .max(32)
}

/// Conventional CUDA block size. See [`tiled_block_threads`]: chosen because the
/// sweep found it equivalent to everything else nearby, not because it is best.
const DEFAULT_BLOCK_THREADS: u32 = 256;

/// CUDA's guaranteed per-block thread limit. Every device supports at least
/// this; asking for more fails at launch.
const MAX_BLOCK_THREADS: u32 = 1024;

/// Whether to use the tiled kernel instead of the per-row one.
///
/// **Default ON, within the opt-in.** Set `LOGAN_CUDA_TILED=0` to revert.
///
/// Chosen on measurement, not preference: the tiled kernel is 1.05–1.46x faster
/// than the per-row kernel and, unlike it, holds a flat per-row cost as `o`
/// grows (see the module header's table). It also measured bit-exact against
/// `dot_row` on every shape tried.
///
/// Note this only chooses between two kernels; it does not enable CUDA. Both sit
/// behind `LOGAN_CUDA`, which is itself off by default because *neither* kernel
/// beats the 12-thread CPU at real shapes.
///
/// The per-row kernel is retained rather than deleted: it is the bit-exact
/// reference the tiled one is checked against, and the fallback if the tiled one
/// misbehaves on other hardware.
fn prefer_tiled() -> bool {
    std::env::var("LOGAN_CUDA_TILED")
        .map(|v| v.trim() != "0")
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggufsource::{dot_row, f16_to_f32};

    // -----------------------------------------------------------------
    // Block construction
    // -----------------------------------------------------------------

    /// Exact f16 encoding, for the magnitudes the tests use.
    ///
    /// Asserts rather than rounds: a value that is not exactly representable
    /// would make the oracle and the kernel see different scale values, and the
    /// resulting parity failure would be blamed on the kernel.
    fn f16_exact(v: f32) -> u16 {
        if v == 0.0 {
            // Zero is a valid block field (`dmin`), and it is not in the normal
            // range the assertion below covers. Only +0.0 is used.
            assert!(!v.is_sign_negative(), "these tests use +0.0 only");
            return 0;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7f_ffff;
        assert!((64..=192).contains(&exp), "test value {v} is not a normal f16");
        let e16 = (exp - 127 + 15) as u16;
        let m16 = (frac >> 13) as u16;
        assert_eq!((m16 as u32) << 13, frac, "test value {v} is not exact in f16");
        (sign << 15) | (e16 << 10) | m16
    }

    /// Inverse of `q4k_scale_min`.
    ///
    /// Derived from the unpacker rather than guessed, and checked against it by
    /// `scale_packing_round_trips` below. Layout:
    ///
    /// - bytes 0..4:  `scale_j        | (scale_{j+4} >> 4) << 6` for j in 0..4
    /// - bytes 4..8:  `min_j          | (min_{j+4}   >> 4) << 6` for j in 0..4
    /// - bytes 8..12: `(scale_{j+4} & 0xf) | ((min_{j+4} & 0xf) << 4)` for j in 0..4
    ///
    /// All eight scales and minima are six-bit values, hence the `>> 4` when
    /// splitting them across two bytes.
    fn pack_scales(q: &mut [u8], scales: &[u8; 8], mins: &[u8; 8]) {
        for j in 0..4 {
            q[j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6);
            q[j + 4] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6);
        }
        for j in 4..8 {
            q[j + 4] = (scales[j] & 0x0f) | ((mins[j] & 0x0f) << 4);
        }
    }

    /// One Q4_K block with explicitly chosen scales and a uniform quant nibble.
    ///
    /// `d` and `dmin` must be exact in f16 (asserted by `f16_exact`) so both
    /// sides see bit-identical scale values.
    fn structured_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 8],
        mins: &[u8; 8],
        nibble: u8,
    ) -> Vec<u8> {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&f16_exact(d).to_le_bytes());
        blk[2..4].copy_from_slice(&f16_exact(dmin).to_le_bytes());
        pack_scales(&mut blk[4..16], scales, mins);
        for b in blk[16..].iter_mut() {
            *b = (nibble & 0x0f) | ((nibble & 0x0f) << 4);
        }
        blk
    }

    /// A small deterministic LCG, so the adversarial cases need no dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    /// One Q4_K block whose scale/min bytes and quants are *arbitrary*.
    ///
    /// This is the strongest input the parity test has. Random six-bit scales
    /// and minima exercise the whole packing quirk, including the high bits that
    /// live in the neighbouring bytes — a misconception shared between the
    /// kernel and the packing helper cannot hide here, because nothing but the
    /// oracle's own unpacker interprets these bytes. `d`/`dmin` stay exact so
    /// the comparison is not swamped by an inf or NaN from a stray f16 pattern.
    fn random_block(seed: u64) -> Vec<u8> {
        let mut lcg = Lcg(seed);
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&f16_exact(0.5).to_le_bytes());
        blk[2..4].copy_from_slice(&f16_exact(0.25).to_le_bytes());
        for b in blk[4..].iter_mut() {
            *b = lcg.next_byte();
        }
        blk
    }

    /// The packing helper must invert the oracle's unpacker, or every parity
    /// test below would be comparing the kernel against a mis-built block.
    #[test]
    fn scale_packing_round_trips() {
        // Includes 0 and 63, the extremes of the six-bit range, plus values
        // whose high bits must land in the neighbouring byte.
        let scales = [0u8, 2, 3, 4, 16, 48, 7, 63];
        let mins = [63u8, 7, 48, 0, 1, 2, 32, 16];
        let mut q = [0u8; 12];
        pack_scales(&mut q, &scales, &mins);
        for j in 0..8 {
            assert_eq!(
                crate::ggufsource::q4k_scale_min(j, &q),
                (scales[j], mins[j]),
                "group {j}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Parity
    // -----------------------------------------------------------------

    /// Run one shape and return `(max_abs, max_rel)` against the oracle.
    ///
    /// `max_rel` is normalised by the largest oracle magnitude in the case, not
    /// per row: a row whose sum happens to cancel to near zero would otherwise
    /// report an enormous relative error for a perfectly good kernel. This is
    /// the same convention `math_x86::tests` uses.
    fn compare(
        label: &str,
        o: usize,
        i: usize,
        weights: &[u8],
        x: &[f32],
    ) -> (f32, f32) {
        let row_bytes = (i / 256) * 144;
        assert_eq!(weights.len(), o * row_bytes, "{label}: weight geometry");

        let mut got = vec![0.0f32; o];
        let ran = q4k_dot_row(&mut got, x, weights, o, i);
        assert!(ran.is_some(), "{label}: kernel refused to run (o={o} i={i})");

        let want: Vec<f32> = (0..o)
            .map(|row| {
                dot_row(
                    GgmlType::Q4K,
                    &weights[row * row_bytes..(row + 1) * row_bytes],
                    x,
                )
                .expect("well-formed Q4_K row")
            })
            .collect();

        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
        let mut max_abs = 0.0f32;
        for row in 0..o {
            let abs = (got[row] - want[row]).abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        let max_rel = max_abs / scale;
        println!(
            "{label:<26} o={o:<4} i={i:<5} max_abs={max_abs:e}  max_rel={max_rel:e}  (max|y|={scale:e})"
        );
        (max_abs, max_rel)
    }

    /// The acceptance test: the kernel must agree with `dot_row`.
    ///
    /// Skips loudly when there is no CUDA device, because a pass on a machine
    /// that never ran the kernel would be a lie — the exact failure mode this
    /// backend exists to avoid. Reports max absolute and max relative
    /// difference, and asserts the kernel actually launched.
    #[test]
    fn q4k_kernel_matches_dot_row() {
        if !available() {
            eprintln!(
                "SKIP q4k parity: CUDA unavailable ({})",
                unavailable_reason().unwrap_or_else(|| "unknown".to_string())
            );
            return;
        }
        // Name the device that is about to run, so the log distinguishes "the
        // GTX 1080 computed this" from "some CUDA device did".
        println!(
            "device: {}  cc={:?}",
            device_name().unwrap_or_else(|| "<unnamed>".to_string()),
            logan_core::cuda::compute_capability()
        );
        let before = kernel_launches();

        let scales = [3u8, 5, 7, 11, 13, 17, 19, 23];
        let mins = [1u8, 2, 3, 4, 5, 6, 7, 9];
        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;

        // Shapes exercise one and several blocks per row, and row counts that
        // are not multiples of the 256-thread block.
        for (o, i) in [
            (1usize, 256usize),
            (3, 256),
            (7, 512),
            (33, 1024),
            (129, 2048),
            (256, 256),
            (257, 512),
        ] {
            let blocks = i / 256;
            let row_bytes = blocks * 144;

            // Distinct bytes per row: an offset bug or a stale upload then
            // changes the answer instead of hiding behind identical rows.
            let mut structured = Vec::with_capacity(o * row_bytes);
            for row in 0..o {
                let nibble = (row as u8 % 15) + 1;
                for b in 0..blocks {
                    structured.extend_from_slice(&structured_block(
                        0.5 + row as f32 * 0.25,
                        0.25 + b as f32 * 0.125,
                        &scales,
                        &mins,
                        nibble,
                    ));
                }
            }
            // A non-uniform activation: a uniform one would let a wrong index
            // still produce the right sum on a symmetric block.
            let x: Vec<f32> = (0..i).map(|k| ((k % 29) as f32 - 14.0) / 8.0).collect();

            let (a, r) = compare("structured", o, i, &structured, &x);
            worst_abs = worst_abs.max(a);
            worst_rel = worst_rel.max(r);

            // Adversarial: arbitrary scale bytes and arbitrary quants. Nothing
            // but the oracle's unpacker interprets these, so a shared
            // misconception between the kernel and `pack_scales` cannot pass.
            let mut random = Vec::with_capacity(o * row_bytes);
            for b in 0..o * blocks {
                random.extend_from_slice(&random_block(0x9e37_79b9_7f4a_7c15 ^ b as u64));
            }
            let (a, r) = compare("random-bytes", o, i, &random, &x);
            worst_abs = worst_abs.max(a);
            worst_rel = worst_rel.max(r);
        }

        println!("q4k parity worst: max_abs={worst_abs:e} max_rel={worst_rel:e}");

        // Both implementations accumulate in the same order, so the residual is
        // fma-contraction noise, not reassociation. On the largest case the
        // oracle's own |y| reaches the thousands; 1e-4 relative is orders of
        // magnitude above contraction noise and far below any real bug.
        assert!(
            worst_rel < 1e-4,
            "q4k parity failed: max_abs={worst_abs:e} max_rel={worst_rel:e}"
        );

        assert!(
            kernel_launches() > before,
            "parity ran but no kernel launched -- the CPU path must have answered"
        );
    }

    /// The upload-skip must not confuse two different matrices.
    ///
    /// This is the one piece of state in this module that can silently return a
    /// stale answer, so its equality relation is checked directly as well as
    /// through the kernel.
    #[test]
    fn fingerprints_separate_distinct_weight_bytes() {
        assert_eq!(fnv1a(b"abc"), fnv1a(b"abc"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"));
        // Same length, one bit different: the case a length-only key would miss.
        let a = vec![0u8; 144];
        let mut b = a.clone();
        b[17] = 1;
        assert_eq!(a.len(), b.len());
        assert_ne!(fnv1a(&a), fnv1a(&b));
    }

    /// Distinct weight matrices in sequence must each be re-uploaded.
    ///
    /// Drives the upload-skip path end to end. With `dmin = 0` the block value
    /// is `d * scale * q`, so the three matrices differ only in their quant and
    /// their dots differ in proportion to it — but the direction depends on the
    /// sign of `sum(x)`, so the assertion is on *distinctness and per-matrix
    /// correctness* rather than on an ordering that only holds for one sign of
    /// the activation. A stale upload fails both: the answer would repeat.
    #[test]
    fn consecutive_distinct_matrices_are_not_served_stale() {
        if !available() {
            eprintln!("SKIP: CUDA unavailable");
            return;
        }
        let i = 256usize;
        // Deliberately mixed-sign and not summing to zero, so a stale result
        // cannot coincide with the correct one by cancellation.
        let x: Vec<f32> = (0..i).map(|k| ((k % 17) as f32 - 8.0) / 4.0).collect();
        assert!(x.iter().sum::<f32>().abs() > 1.0, "need a non-cancelling sum");
        let scales = [4u8, 4, 4, 4, 4, 4, 4, 4];
        let mins = [0u8; 8];

        let mut results = Vec::new();
        for nibble in [1u8, 7, 15] {
            let w = structured_block(1.0, 0.0, &scales, &mins, nibble);
            let mut y = vec![0.0f32; 1];
            assert!(q4k_dot_row(&mut y, &x, &w, 1, i).is_some());
            // Correct for THIS matrix, not the previous one.
            let want = dot_row(GgmlType::Q4K, &w, &x).unwrap();
            assert!(
                (y[0] - want).abs() / want.abs().max(1e-30) < 1e-4,
                "nibble={nibble}: got {} want {want}",
                y[0]
            );
            results.push(y[0]);
        }
        // Each matrix produced its own answer: a skipped upload would have
        // repeated the first.
        assert!(
            results[0] != results[1] && results[1] != results[2] && results[0] != results[2],
            "distinct matrices produced a repeated result: {results:?}"
        );
        // And the values scale with the quant, which is the physics the block
        // layout implies (4 * nibble * sum(x)); the sign of the ratio is fixed
        // by sum(x), so compare magnitudes.
        let ratio = |a: f32, b: f32| (a / b).abs();
        assert!(
            (ratio(results[1], results[0]) - 7.0).abs() < 1e-3,
            "quant ratio wrong: {results:?}"
        );
        assert!(
            (ratio(results[2], results[0]) - 15.0).abs() < 1e-3,
            "quant ratio wrong: {results:?}"
        );
    }

    /// A geometry the kernel cannot express must be refused, not computed
    /// wrongly: the caller then falls back to the oracle.
    #[test]
    fn ragged_geometry_is_refused_rather_than_miscomputed() {
        let mut y = vec![0.0f32; 4];
        let full_w = vec![0u8; 4 * 144];
        let full_x = vec![0.0f32; 256];

        // `i` that is not a whole number of 256-value Q4_K blocks.
        let x_ragged = vec![0.0f32; 300];
        assert!(
            q4k_dot_row(&mut y, &x_ragged, &full_w, 4, 300).is_none(),
            "ragged i must be refused"
        );
        // Weights one byte short of four rows.
        let w_short = vec![0u8; 4 * 144 - 1];
        assert!(
            q4k_dot_row(&mut y, &full_x, &w_short, 4, 256).is_none(),
            "truncated weights must be refused"
        );
        // Output shorter than the requested row count.
        assert!(
            q4k_dot_row(&mut y[..3], &full_x, &full_w, 4, 256).is_none(),
            "undersized output must be refused"
        );
        // Activation shorter than `i`.
        let x_short = vec![0.0f32; 255];
        assert!(
            q4k_dot_row(&mut y, &x_short, &full_w, 4, 256).is_none(),
            "undersized activation must be refused"
        );
        // And the well-formed shape is accepted, so the refusals above are
        // about the geometry rather than a backend that never runs.
        if available() {
            assert!(
                q4k_dot_row(&mut y, &full_x, &full_w, 4, 256).is_some(),
                "the matching geometry must run"
            );
        }
    }

    /// A zero-length call must not reach the device.
    #[test]
    fn empty_input_is_refused() {
        let mut y = vec![0.0f32; 0];
        assert!(q4k_dot_row(&mut y, &[], &[], 0, 0).is_none());
    }

    /// The default must be OFF, and that must actually stop the kernel.
    ///
    /// Runs in a **child process** on purpose. `STATE` is a `LazyLock`: if this
    /// test mutated the environment in-process, the initialiser could observe it
    /// and cache a stale answer for the rest of the run — including for the
    /// parity test, which would then skip silently on the one machine that has a
    /// GPU. A subprocess gets a fresh initialiser and proves the gate end to end.
    ///
    /// This is the guard against the slow backend creeping back in as a default:
    /// if someone flips the sense again without fixing the kernel, this fails.
    #[test]
    fn default_is_off_and_stops_the_kernel() {
        const CHILD: &str = "LOGAN_DEFAULT_OFF_CHILD";
        if std::env::var(CHILD).is_ok() {
            let expected = std::env::var("LOGAN_EXPECT").expect("parent sets LOGAN_EXPECT");
            // Child arm: the parent set (or deliberately did not set) LOGAN_CUDA.
            assert!(
                !logan_core::cuda::available(),
                "available() must be false by default even with a device present"
            );
            // Refused, and for the right reason: nothing reached the device.
            let mut y = vec![0.0f32; 1];
            let x = vec![0.5f32; 256];
            let w = vec![0u8; 144];
            assert!(
                q4k_dot_row(&mut y, &x, &w, 1, 256).is_none(),
                "the disabled backend must stop the kernel"
            );
            assert!(y[0] == 0.0, "output must be untouched");
            // Evidence for the log: whether a device was there to be disabled,
            // and the reason the backend declined.
            println!(
                "child({expected}): device_name={:?} init_error={:?}",
                device_name(),
                logan_core::cuda::init_error()
            );
            return;
        }

        // Two arms: unset (the real default) and LOGAN_CUDA=0. Both must be off.
        for (label, value) in [("unset", None), ("zero", Some("0"))] {
            let exe = std::env::current_exe().expect("test binary path");
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "--exact",
                "ggufsource::cuda_q4k::tests::default_is_off_and_stops_the_kernel",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("LOGAN_EXPECT", label);
            match value {
                // Deliberately remove rather than set: "unset" must mean the
                // variable is genuinely absent, not present-and-empty.
                None => {
                    cmd.env_remove("LOGAN_CUDA");
                }
                Some(v) => {
                    cmd.env("LOGAN_CUDA", v);
                }
            }
            let out = cmd.output().expect("spawn child test binary");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success(),
                "child failed for arm {label}:\n{stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(
                stdout.contains(&format!("child({label}):")),
                "child did not take the {label} arm:\n{stdout}"
            );
            print!("[{label}] {stdout}");
        }
    }

    /// The predicate the opt-in is built from, exhaustively.
    #[test]
    fn the_opt_in_parses_total() {
        // OFF unless an explicit yes. The old opt-out spelling must now be OFF.
        assert!(!logan_core::cuda::enabled_given(Some("0")));
        assert!(!logan_core::cuda::enabled_given(Some("")));
        assert!(!logan_core::cuda::enabled_given(None));
        assert!(logan_core::cuda::enabled_given(Some("1")));
        assert!(logan_core::cuda::enabled_given(Some("true")));
    }

    /// Q4_K is the only dtype wired up, and the probe must say so.
    #[test]
    fn only_q4k_is_claimed() {
        for other in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q5_0,
            GgmlType::Q8_0,
            GgmlType::Q6K,
        ] {
            assert!(!available_for(other), "{other:?} has no kernel");
        }
        if available() {
            assert!(available_for(GgmlType::Q4K));
        } else {
            eprintln!("SKIP positive arm: CUDA unavailable");
        }
    }

    /// The kernel's code, with `//` comments removed.
    ///
    /// The source checks below are about what the kernel *does*; the comments
    /// above each routine discuss the alternatives by name, so matching raw
    /// source would fail on the prose that explains the decision.
    fn kernel_code() -> String {
        row_source()
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The tiled kernel must agree with the oracle to f32 tolerance.
    ///
    /// It is NOT expected to be bit-exact, and that is the point of this test
    /// being separate from `q4k_kernel_matches_dot_row`. The tiled kernel keeps
    /// the *within-group* accumulation order identical to `dot_row` (one group
    /// per thread, 32 values in sequence), but combines the per-group partials
    /// with a tree reduction rather than a serial walk. Reassociation on the
    /// order of 8-80 additions of similar-magnitude f32 values is therefore
    /// expected; the same tolerance `math_x86.rs` documents applies.
    ///
    /// If this ever *is* bit-exact that is a happy accident, not a guarantee,
    /// and the assertion below deliberately does not require it -- claiming
    /// exactness a tree reduction cannot promise would be the wrong label.
    #[test]
    fn q4k_tiled_matches_dot_row_within_f32_tolerance() {
        if !available() {
            eprintln!(
                "SKIP tiled parity: CUDA unavailable ({})",
                unavailable_reason().unwrap_or_else(|| "unknown".to_string())
            );
            return;
        }
        // `State` caches the kernel choice at init, so the selector cannot be
        // flipped in-process; the measuring arm runs in a child.
        const CHILD: &str = "LOGAN_TILED_CHILD";
        if std::env::var(CHILD).is_err() {
            let exe = std::env::current_exe().expect("test binary path");
            let out = std::process::Command::new(exe)
                .args([
                    "--exact",
                    "ggufsource::cuda_q4k::tests::q4k_tiled_matches_dot_row_within_f32_tolerance",
                    "--nocapture",
                ])
                .env("LOGAN_CUDA", "1")
                .env(CHILD, "1")
                .output()
                .expect("spawn child test binary");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success(),
                "tiled child failed:\n{stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(stdout.contains("tiled parity worst"), "child took no arm:\n{stdout}");
            print!("{stdout}");
            return;
        }
        // No LOGAN_CUDA_TILED set: this asserts the DEFAULT selection is the
        // tiled kernel, which is the claim the module header makes.
        assert!(
            prefer_tiled(),
            "the tiled kernel must be selected by default within the opt-in"
        );
        assert!(
            std::env::var("LOGAN_CUDA_TILED").is_err(),
            "this must be measuring the default, not an explicit selection"
        );

        let mut worst_rel = 0.0f32;
        let mut worst_abs = 0.0f32;
        for (o, i) in [(1usize, 256usize), (3, 256), (7, 512), (33, 1024), (129, 2048)] {
            let blocks = i / 256;
            let row_bytes = blocks * 144;
            let mut random = Vec::with_capacity(o * row_bytes);
            for b in 0..o * blocks {
                random.extend_from_slice(&random_block(0x2545_f491_4f6c_dd1d ^ b as u64));
            }
            let x: Vec<f32> = (0..i).map(|k| ((k % 29) as f32 - 14.0) / 8.0).collect();

            let before = kernel_launches();
            let mut got = vec![0.0f32; o];
            assert!(
                q4k_dot_row(&mut got, &x, &random, o, i).is_some(),
                "o={o} i={i}: tiled kernel refused to run"
            );
            assert!(
                kernel_launches() > before,
                "o={o} i={i}: nothing launched"
            );

            let want: Vec<f32> = (0..o)
                .map(|row| {
                    dot_row(
                        GgmlType::Q4K,
                        &random[row * row_bytes..(row + 1) * row_bytes],
                        &x,
                    )
                    .unwrap()
                })
                .collect();
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
            for row in 0..o {
                let abs = (got[row] - want[row]).abs();
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(abs / scale);
            }
            println!("tiled o={o} i={i} max_abs={worst_abs:e} max_rel={worst_rel:e}");
        }
        println!("tiled parity worst: max_abs={worst_abs:e} max_rel={worst_rel:e}");
        // 1e-5 relative, matching the bar `math_x86.rs` uses for its
        // reassociating kernels. Far above tree-reduction noise, far below any
        // real indexing bug.
        assert!(
            worst_rel < 1e-5,
            "tiled parity failed: max_abs={worst_abs:e} max_rel={worst_rel:e}"
        );
    }

    /// The tiled block size must always be a power of two.
    ///
    /// This is the guard for a silent-corruption bug, not a style rule. The
    /// kernel's shared-memory reduction is a halving tree, and a halving tree
    /// over a non-power-of-two thread count drops partials: simulating the exact
    /// reduction, 80 slots (the literal group count for i=2560) comes out 6.2%
    /// wrong and 96 slots 143% wrong, while 64 and 128 are exact. So a change
    /// that "tightens" this to the exact group count would produce plausible
    /// but wrong numbers — exactly the failure this whole backend is built to
    /// avoid — and nothing else in the suite would notice.
    #[test]
    fn tiled_block_size_is_always_a_power_of_two() {
        // Group counts for the shapes in use, plus awkward neighbours.
        for ngroups in [8usize, 10, 16, 40, 64, 80, 96, 128, 640, 2048, 100_000] {
            let b = tiled_block_threads(ngroups);
            assert!(
                b.is_power_of_two(),
                "ngroups={ngroups} produced block={b}, which is not a power of two \
                 -- the halving reduction would drop partials"
            );
        }
        // Small rows must still get a usable block, never 0 or 1.
        for ngroups in [0usize, 1, 2, 3] {
            let b = tiled_block_threads(ngroups);
            assert!(b >= 32 && b.is_power_of_two(), "ngroups={ngroups} -> {b}");
        }
        // A huge row must clamp rather than ask for an unlaunchable block.
        assert!(
            tiled_block_threads(usize::MAX / 8) <= MAX_BLOCK_THREADS,
            "must clamp to the hardware limit"
        );
    }

    /// The kernel's own reduction must not be relied on for odd thread counts.
    ///
    /// Asserts the property directly on the emitted CUDA source: a halving
    /// `stride >>= 1` loop is only sound for powers of two, and the host side is
    /// what guarantees it. If someone rewrites the kernel to a masked tree this
    /// test should be revisited, which is why it is stated as a comment on the
    /// source rather than a blanket ban.
    #[test]
    fn tiled_kernel_reduction_is_the_halving_form_the_host_assumes() {
        let code = tiled_source()
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("stride = blockDim.x >> 1") && code.contains("stride >>= 1"),
            "host's power-of-two guarantee is tied to this exact reduction form"
        );
    }

    /// `LOGAN_CUDA_TILED` must still revert to the per-row kernel.
    #[test]
    fn tiled_can_still_be_reverted() {
        const CHILD: &str = "LOGAN_REVERT_CHILD";
        if std::env::var(CHILD).is_ok() {
            assert!(
                !prefer_tiled(),
                "LOGAN_CUDA_TILED=0 must revert to the per-row kernel"
            );
            println!("revert arm: LOGAN_CUDA_TILED=0 selects the per-row kernel");
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--exact",
                "ggufsource::cuda_q4k::tests::tiled_can_still_be_reverted",
                "--nocapture",
            ])
            .env("LOGAN_CUDA_TILED", "0")
            .env(CHILD, "1")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "revert child failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(stdout.contains("revert arm:"), "child took no arm:\n{stdout}");
        print!("{stdout}");
    }

    /// The tiled kernel's source checks.
    ///
    /// Its *behaviour* is unverified on a GPU (see [`prefer_tiled`]), so there is
    /// deliberately no GPU test asserting its agreement with the oracle yet —
    /// writing one that cannot run would be the "advertised but never executed"
    /// failure this module is built to avoid. The test below is the child-driven
    /// one that WOULD measure it, kept because it is correct and runnable; it
    /// simply skips on a machine with no device.
    #[test]
    fn tiled_kernel_source_matches_the_documented_layout() {
        let code = tiled_source()
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(code.contains("extern \"C\" __global__ void q4k_dot_row_tiled"));
        assert!(code.contains("blk + 2"), "dmin at offset 2");
        assert!(code.contains("blk + 4"), "scales at offset 4");
        assert!(code.contains("blk + 16"), "quants at offset 16");
        assert!(code.contains("* 144L"), "144-byte blocks");
        // One block per row, and the shared-memory reduction that defines it.
        assert!(code.contains("int row = blockIdx.x"));
        assert!(code.contains("extern __shared__ float partial[]"));
        assert!(code.contains("__syncthreads()"));
        assert!(code.contains("logan_f16_to_f32"));
        assert!(!code.contains("bfloat16"));
        assert!(!code.contains("__half2float"));
    }

    /// The kernel source must not have drifted from the documented layout.
    ///
    /// Text checks, but the alternative is a mis-set offset that shows up only
    /// as a small numerical error — indistinguishable from fp noise.
    #[test]
    fn kernel_source_matches_the_documented_layout() {
        let code = kernel_code();
        assert!(code.contains("extern \"C\" __global__ void q4k_dot_row"));
        assert!(code.contains("blk + 2"), "dmin at offset 2");
        assert!(code.contains("blk + 4"), "scales at offset 4");
        assert!(code.contains("blk + 16"), "quants at offset 16");
        assert!(code.contains("* 144L"), "144-byte blocks");
        assert!(code.contains("(float)scale"), "six-bit scales widen to f32");
        assert!(code.contains("(float)mn"), "six-bit minima widen to f32");
        // The nibble split: even groups low, odd groups high.
        assert!(code.contains("packed >> 4"), "odd groups take the high nibble");
        assert!(code.contains("packed & 0x0f"), "even groups take the low nibble");
        // 8 groups of 32 within a 256-value block.
        assert!(code.contains("g < 8"));
        assert!(code.contains("k < 32"));
        assert!(code.contains("i / 256"));
    }

    /// `f16_to_f32` and the kernel's transliteration must agree, or parity
    /// would be measured through a conversion mismatch.
    #[test]
    fn kernel_half_conversion_is_transliterated_not_delegated() {
        let code = kernel_code();
        // The kernel must contain its own converter rather than calling the
        // hardware one, which differs for denormals on sm_61.
        assert!(code.contains("logan_f16_to_f32"));
        assert!(!code.contains("__half2float"));
        // No bf16 anywhere: sm_61 has none, and its presence would mean a
        // compile error on the only card this targets.
        assert!(!code.contains("bfloat16"));
        assert!(!code.contains("__nv_bfloat16"));
        assert!(!code.contains("__half"));
        // And the values the tests build blocks with must be exact.
        for v in [0.125f32, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0] {
            let bits = f16_exact(v);
            assert_eq!(f16_to_f32(bits), v, "v={v}");
        }
    }
}
