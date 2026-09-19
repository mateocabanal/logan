//! CUDA GEMV for native MLX affine (oQ) weights.
//!
//! The checkpoint bytes stay packed in VRAM.  Only the activation is uploaded
//! and the result downloaded per call; weights/scales/biases are uploaded once
//! into the `ResidentAffine` owned by the corresponding `WtBytes`.

use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};

use logan_core::cuda::{self, DeviceBuf, Kernel};

const BLOCK: u32 = 256;

const SOURCE: &str = r#"
extern "C" __global__ void mlx_affine_gemv(
    const unsigned int* weights,
    const unsigned short* scales,
    const unsigned short* biases,
    const float* x,
    float* y,
    int rows,
    int cols,
    int bits,
    int group_size)
{
    const int row = (int)blockIdx.x;
    if (row >= rows) return;

    const int tid = (int)threadIdx.x;
    const int row_words = (cols * bits) >> 5;
    const int groups = cols / group_size;
    const unsigned int* wr = weights + ((long long)row * row_words);
    float sum = 0.0f;

    for (int col = tid; col < cols; col += (int)blockDim.x) {
        const int bit = col * bits;
        const int word = bit >> 5;
        const int shift = bit & 31;
        unsigned long long packed = (unsigned long long)wr[word];
        if (shift + bits > 32) {
            packed |= ((unsigned long long)wr[word + 1]) << 32;
        }
        const unsigned int mask = (1u << bits) - 1u;
        const unsigned int code = (unsigned int)((packed >> shift) & mask);
        const int group = row * groups + col / group_size;
        const float scale = __uint_as_float(((unsigned int)scales[group]) << 16);
        const float bias = __uint_as_float(((unsigned int)biases[group]) << 16);
        sum += x[col] * ((float)code * scale + bias);
    }

    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}
"#;

static KERNEL: OnceLock<Option<Kernel>> = OnceLock::new();
static WEIGHT_UPLOADS: AtomicU64 = AtomicU64::new(0);

fn kernel() -> Option<&'static Kernel> {
    KERNEL
        .get_or_init(|| cuda::compile(SOURCE, "mlx_affine_gemv"))
        .as_ref()
}

pub(crate) fn weight_uploads() -> u64 {
    WEIGHT_UPLOADS.load(Ordering::Relaxed)
}

pub struct ResidentAffine {
    weights: DeviceBuf,
    scales: DeviceBuf,
    biases: DeviceBuf,
    x: DeviceBuf,
    y: DeviceBuf,
}

impl ResidentAffine {
    fn upload(weights: &[u8], scales: &[u8], biases: &[u8]) -> Option<Self> {
        let mut out = Self {
            weights: DeviceBuf::new(),
            scales: DeviceBuf::new(),
            biases: DeviceBuf::new(),
            x: DeviceBuf::new(),
            y: DeviceBuf::new(),
        };
        out.weights.upload(weights)?;
        out.scales.upload(scales)?;
        out.biases.upload(biases)?;
        WEIGHT_UPLOADS.fetch_add(1, Ordering::Relaxed);
        Some(out)
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` is POD; the resulting byte slice has exactly the same
    // lifetime and covers exactly `len * size_of::<f32>()` initialized bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: same representation argument as `f32_bytes`; the caller owns the
    // mutable f32 slice for the duration of the device-to-host copy.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul(
    resident: &std::sync::Mutex<Option<ResidentAffine>>,
    y: &mut [f32],
    x: &[f32],
    weights: &[u8],
    scales: &[u8],
    biases: &[u8],
    bits: u8,
    group_size: usize,
    cols: usize,
    rows: usize,
) -> bool {
    if !matches!(bits, 4 | 5 | 6 | 8)
        || group_size == 0
        || cols == 0
        || rows == 0
        || cols % group_size != 0
        || x.len() != cols
        || y.len() != rows
        || cols * bits as usize % 32 != 0
    {
        return false;
    }
    let row_bytes = cols * bits as usize / 8;
    let groups = cols / group_size;
    if weights.len() != rows * row_bytes
        || scales.len() != rows * groups * 2
        || biases.len() != rows * groups * 2
    {
        return false;
    }
    if !cuda::available() {
        return false;
    }
    let Some(kernel) = kernel() else {
        return false;
    };

    let mut guard = resident.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        let Some(uploaded) = ResidentAffine::upload(weights, scales, biases) else {
            return false;
        };
        *guard = Some(uploaded);
    }
    let r = guard.as_mut().expect("resident initialized above");
    if r.x.upload(f32_bytes(x)).is_none() || r.y.ensure(std::mem::size_of_val(y)).is_none() {
        return false;
    }

    let Some(mut w_ptr) = r.weights.ptr() else {
        return false;
    };
    let Some(mut s_ptr) = r.scales.ptr() else {
        return false;
    };
    let Some(mut b_ptr) = r.biases.ptr() else {
        return false;
    };
    let Some(mut x_ptr) = r.x.ptr() else {
        return false;
    };
    let Some(mut y_ptr) = r.y.ptr() else {
        return false;
    };
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut bits_i = bits as i32;
    let mut group_i = group_size as i32;
    let mut params = [
        (&mut w_ptr as *mut u64).cast::<c_void>(),
        (&mut s_ptr as *mut u64).cast::<c_void>(),
        (&mut b_ptr as *mut u64).cast::<c_void>(),
        (&mut x_ptr as *mut u64).cast::<c_void>(),
        (&mut y_ptr as *mut u64).cast::<c_void>(),
        (&mut rows_i as *mut i32).cast::<c_void>(),
        (&mut cols_i as *mut i32).cast::<c_void>(),
        (&mut bits_i as *mut i32).cast::<c_void>(),
        (&mut group_i as *mut i32).cast::<c_void>(),
    ];
    if kernel
        .launch((rows as u32, 1, 1), (BLOCK, 1, 1), 0, &mut params)
        .is_none()
    {
        return false;
    }
    r.y.download(f32_bytes_mut(y)).is_some()
}
