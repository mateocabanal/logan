//! CUDA device access with no toolkit and no build-time CUDA.
//!
//! ## Why this file exists
//!
//! `MachineProfile` records the host's compute devices, and the Windows
//! x86_64 Windows box with a GeForce GTX 1080 (sm_61, 8 GiB) and a working
//! driver — but **no CUDA toolkit**. There is no `nvcc`, no CUDA_PATH, no
//! toolkit directory. A conventional CUDA backend would therefore compile into
//! nothing on exactly the machine that has the GPU, which is the same failure
//! `math_x86.rs` describes: planning for a capability that never executes.
//!
//! The route taken instead: NVRTC compiles the kernel from a source string at
//! first use, the driver API and NVRTC are reached through `LoadLibrary`/
//! `dlopen` rather than linked, and **no CUDA dependency appears in
//! Cargo.toml**. Both libraries ship inside torch, which is already installed
//! on that host for unrelated work:
//!
//! ```text
//! ...\Python313\Lib\site-packages\torch\lib\nvrtc64_120_0.dll
//! ...\Python313\Lib\site-packages\torch\lib\nvrtc-builtins64_126.dll
//! ...\Python313\Lib\site-packages\torch\lib\cudart64_12.dll
//! ```
//!
//! The tradeoff is explicit: this is a *runtime* dependency on a CUDA runtime
//! existing somewhere on the machine, detected at startup instead of linked at
//! build time.
//!
//! ## This module is mechanism, not policy
//!
//! Nothing here knows what a GGML block is, or what a matmul is. It resolves
//! the driver, creates a context, compiles a source string to PTX through
//! NVRTC, and moves bytes. The kernels live with the format that defines them
//! (`logan-qwen4::ggufsource::cuda_q4k`, next to the `dot_row` oracle they must
//! agree with). Keeping the split this way means the loader carries no
//! dependency on any engine's tensor representation.
//!
//! ## Fail closed
//!
//! Every entry point returns `Option`/`bool`. No toolkit, no DLL, no device, no
//! context, a compile error or any driver error returns `None` and the caller
//! stays on the CPU. Nothing in this file panics during capability detection:
//! the paths that could (missing env vars, unresolvable symbols, a device that
//! refuses a context) are all `None`. A GPU that is busy, wedged, or out of
//! memory must degrade inference, not break it.
//!
//! ## Reporting only what ran
//!
//! `available()` is deliberately narrow — it means "a kernel was compiled and
//! the module loaded", not "this machine has a GPU". [`device_name`] and
//! [`compute_capability`] read the name and CC back from the driver, so a
//! caller can print the actual device rather than a guess, and [`launches`]
//! counts kernels that reached `cuLaunchKernel` successfully. Advertising a
//! backend that never executed is the bug this file is written to avoid.
//!
//! ## Opt-in, and why
//!
//! Unlike the AVX2/NEON paths in `math.rs` and `math_x86.rs`, this backend is
//! **off unless explicitly enabled**:
//!
//! - `LOGAN_CUDA=1` (or `true`) — enable. Anything else, including unset and
//!   an unrecognised value, leaves it off.
//! - `LOGAN_CUDA_LIB_DIR=<dir>` — explicit CUDA library directory, searched
//!   before the interpreter query and the conventional installs.
//!
//! The sense is inverted from `QWEN_NEON_BF16` because the measurement said so.
//! The kernel is bit-exact against its oracle, but it is **slower than the CPU
//! it replaces**: 0.34x–0.58x of the 12-thread scalar path at real expert
//! shapes (o=640/i=2560, o=2048/i=2560). A verified backend that is 2–3x slower
//! must not be adopted silently, so it is opt-in until it earns the default.
//! The mechanism — the kernel parallelises over `o` only, so it runs a serial
//! per-row chain at ~15% occupancy — is documented in `logan-qwen4`'s
//! `ggufsource::cuda_q4k`.
//!
//! Because the backend is reached only through a runtime probe (this module's
//! `available()`) *and* the opt-in, a machine can hold a perfectly good GPU and
//! still take the CPU path. That is intended, and it is why [`device_name`] and
//! [`compute_capability`] answer independently of [`available`] — a capability
//! line should be able to report a card that is present but not in use without
//! that implying it is being used.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Whether the CUDA backend is permitted.
///
/// **Opt-in: disabled by default.** Set `LOGAN_CUDA=1` (or `true`) to enable.
///
/// This is deliberately the opposite sense from `QWEN_NEON_BF16`, and the
/// reason is measured, not stylistic: the Q4_K kernel here is bit-exact and its
/// parity is verified, but it is **slower than the CPU it replaces** — 0.34x to
/// 0.58x of the 12-thread scalar oracle at real expert shapes (o=640/i=2560 and
/// o=2048/i=2560; see `logan-qwen4`'s `ggufsource::cuda_q4k` header for the
/// mechanism). Enabling it by default would therefore be a silent regression
/// for every run on that host. It stays available, and correct, for the day the kernel
/// earns its place.
///
/// Gates *use*; [`available`] gates *existence*, and is itself gated by this.
/// [`device_name`] and [`compute_capability`] describe the hardware regardless,
/// so a capability line can still report a card that is present but not in use.
pub fn enabled() -> bool {
    enabled_given(std::env::var(GLOGAN_CUDA).ok().as_deref())
}

/// The opt-in variable, named so the predicate below can be tested without
/// mutating the process environment.
const GLOGAN_CUDA: &str = "LOGAN_CUDA";

/// [`enabled`] against a supplied value rather than the live environment.
///
/// Split out so the decision can be tested exhaustively without a test mutating
/// a process-global while other tests run in parallel — and so the parsing is
/// total, which is the property that keeps capability detection from panicking.
///
/// Deliberately not "anything but 0": with the backend slower than the CPU, an
/// unrecognised value must fall back to OFF. A typo like `LOGAN_CUDA=ture`
/// silently leaving the slow path enabled is exactly the failure mode this
/// opt-in exists to prevent.
pub fn enabled_given(value: Option<&str>) -> bool {
    match value {
        Some(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"),
        None => false,
    }
}

/// Kernels that have successfully reached `cuLaunchKernel`.
///
/// The evidence that the GPU path executed, as opposed to the inference that it
/// should have. A parity test that reports "CUDA" while this is zero is testing
/// nothing.
static LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// How many kernels have been launched successfully in this process.
pub fn launches() -> u64 {
    LAUNCHES.load(Ordering::Relaxed)
}

/// Why the backend is not available, if it is not.
///
/// Set once during the single initialisation; `None` means either "not yet
/// probed" or "available". Call [`available`] first if that distinction
/// matters. Kept for diagnostics on machines where the failure would otherwise
/// be indistinguishable from "no GPU here".
static INIT_ERROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn note_error(msg: String) {
    if let Ok(mut slot) = INIT_ERROR.lock() {
        if slot.is_none() {
            *slot = Some(msg);
        }
    }
}

/// The reason initialisation failed, once it has been attempted.
pub fn init_error() -> Option<String> {
    INIT_ERROR.lock().ok().and_then(|s| s.clone())
}

// ---------------------------------------------------------------------------
// Dynamic loading
// ---------------------------------------------------------------------------

/// Minimal driver/NVRTC surface, resolved by name through the OS loader.
///
/// Only what is used is declared: the point of this route is to avoid a
/// bindgen exercise, so the symbol list is short on purpose and lives in
/// [`Api::load`].
#[allow(non_snake_case, non_camel_case_types, dead_code)]
mod ffi {
    use super::*;

    pub type CUresult = c_int;
    pub type CUdevice = c_int;
    pub type CUcontext = *mut c_void;
    pub type CUmodule = *mut c_void;
    pub type CUfunction = *mut c_void;
    pub type CUdeviceptr = u64;
    pub type CUstream = *mut c_void;
    pub type nvrtcProgram = *mut c_void;
    pub type nvrtcResult = c_int;

    // Dynamic loading is spelled differently per platform. POSIX names on
    // Windows link but fail with LNK2019, because the MSVC CRT does not export
    // them -- which is how this kind of code first reaches the one machine that
    // has a GPU.
    #[cfg(unix)]
    unsafe extern "C" {
        #[cfg_attr(target_os = "linux", link(name = "dl"))]
        unsafe fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        unsafe fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    #[cfg(windows)]
    #[link(name = "kernel32")]
    unsafe extern "C" {
        fn LoadLibraryA(filename: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    #[cfg(unix)]
    const RTLD_NOW: c_int = 2;

    /// Open one shared library by path or bare name, or null.
    ///
    /// Bare names go through the OS search path, which is where the driver
    /// actually lives (`System32` on Windows, `ldconfig` on Linux). A path
    /// inside torch is used for NVRTC, which the OS would not find.
    pub fn open_library(path: &str) -> *mut c_void {
        let Ok(c) = CString::new(path) else {
            return std::ptr::null_mut();
        };
        unsafe {
            #[cfg(unix)]
            {
                dlopen(c.as_ptr(), RTLD_NOW)
            }
            #[cfg(windows)]
            {
                LoadLibraryA(c.as_ptr())
            }
        }
    }

    /// Look up one symbol in one library, or null.
    fn lookup(handle: *mut c_void, sym: *const c_char) -> *mut c_void {
        if handle.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            #[cfg(unix)]
            {
                dlsym(handle, sym)
            }
            #[cfg(windows)]
            {
                GetProcAddress(handle, sym)
            }
        }
    }

    /// Resolve a symbol from the first library that exports it.
    fn find(handles: &[*mut c_void], sym: &str) -> *mut c_void {
        let Ok(c) = CString::new(sym) else {
            return std::ptr::null_mut();
        };
        for &h in handles {
            let p = lookup(h, c.as_ptr());
            if !p.is_null() {
                return p;
            }
        }
        std::ptr::null_mut()
    }

    /// Every entry point this backend uses.
    ///
    /// Fields are bare fn pointers rather than `Option`s and [`Api::load`]
    /// returns `None` unless every one resolves, so a partially-populated `Api`
    /// is never observable. One struct covers both libraries because each name
    /// is searched across both handle sets.
    #[allow(non_snake_case)]
    pub struct Api {
        pub cuInit: unsafe extern "C" fn(u32) -> CUresult,
        pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
        pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
        pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
        pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult,
        pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
        pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
        pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
        pub cuModuleGetFunction:
            unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
        pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
        pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
        pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
        pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
        pub cuLaunchKernel: unsafe extern "C" fn(
            CUfunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            CUstream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult,
        pub cuGetErrorName: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
        pub nvrtcCreateProgram: unsafe extern "C" fn(
            *mut nvrtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> nvrtcResult,
        pub nvrtcCompileProgram:
            unsafe extern "C" fn(nvrtcProgram, c_int, *const *const c_char) -> nvrtcResult,
        pub nvrtcGetPTXSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult,
        pub nvrtcGetPTX: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult,
        pub nvrtcGetProgramLogSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult,
        pub nvrtcGetProgramLog: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult,
        pub nvrtcDestroyProgram: unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult,
    }

    impl Api {
        /// Resolve every symbol, or nothing.
        pub fn load(handles: &[*mut c_void], nvrtc_handles: &[*mut c_void]) -> Option<Api> {
            macro_rules! sym {
                ($name:ident : $ty:ty) => {{
                    let mut p = find(handles, stringify!($name));
                    if p.is_null() {
                        p = find(nvrtc_handles, stringify!($name));
                    }
                    if p.is_null() {
                        note_error(format!("missing CUDA symbol `{}`", stringify!($name)));
                        return None;
                    }
                    unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
                }};
            }
            Some(Api {
                cuInit: sym!(cuInit: unsafe extern "C" fn(u32) -> CUresult),
                cuDeviceGet: sym!(cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult),
                cuDeviceGetName: sym!(cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult),
                cuDeviceGetAttribute: sym!(cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult),
                cuCtxCreate_v2: sym!(cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult),
                cuCtxSetCurrent: sym!(cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult),
                cuCtxSynchronize: sym!(cuCtxSynchronize: unsafe extern "C" fn() -> CUresult),
                cuModuleLoadData: sym!(cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult),
                cuModuleGetFunction: sym!(cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult),
                cuMemAlloc_v2: sym!(cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult),
                cuMemFree_v2: sym!(cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult),
                cuMemcpyHtoD_v2: sym!(cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult),
                cuMemcpyDtoH_v2: sym!(cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult),
                cuLaunchKernel: sym!(cuLaunchKernel: unsafe extern "C" fn(
                    CUfunction, u32, u32, u32, u32, u32, u32,
                    u32, CUstream, *mut *mut c_void, *mut *mut c_void,
                ) -> CUresult),
                cuGetErrorName: sym!(cuGetErrorName: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult),
                nvrtcCreateProgram: sym!(nvrtcCreateProgram: unsafe extern "C" fn(
                    *mut nvrtcProgram, *const c_char, *const c_char, c_int,
                    *const *const c_char, *const *const c_char,
                ) -> nvrtcResult),
                nvrtcCompileProgram: sym!(nvrtcCompileProgram: unsafe extern "C" fn(nvrtcProgram, c_int, *const *const c_char) -> nvrtcResult),
                nvrtcGetPTXSize: sym!(nvrtcGetPTXSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult),
                nvrtcGetPTX: sym!(nvrtcGetPTX: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult),
                nvrtcGetProgramLogSize: sym!(nvrtcGetProgramLogSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult),
                nvrtcGetProgramLog: sym!(nvrtcGetProgramLog: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult),
                nvrtcDestroyProgram: sym!(nvrtcDestroyProgram: unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult),
            })
        }
    }
}

use ffi::*;

// ---------------------------------------------------------------------------
// Library discovery
// ---------------------------------------------------------------------------

/// Where a CUDA runtime might live on this machine.
///
/// Order matters: an explicit `LOGAN_CUDA_LIB_DIR` wins, then whatever the
/// installed interpreter reports for torch (asking python beats guessing
/// site-packages layouts across interpreters and versions), then conventional
/// install locations. Each directory is tried with every candidate file name,
/// so a directory that exists but holds nothing useful is simply skipped.
pub fn cuda_lib_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(d) = std::env::var("LOGAN_CUDA_LIB_DIR") {
        if !d.is_empty() {
            dirs.push(PathBuf::from(d));
        }
    }

    for py in ["python", "python3", "py"] {
        let Ok(out) = std::process::Command::new(py)
            .args([
                "-c",
                "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))",
            ])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !p.is_empty() && Path::new(&p).is_dir() {
            dirs.push(PathBuf::from(p));
        }
    }

    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let root = PathBuf::from(local).join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for e in entries.flatten() {
                let lib = e
                    .path()
                    .join("Lib")
                    .join("site-packages")
                    .join("torch")
                    .join("lib");
                if lib.is_dir() {
                    dirs.push(lib);
                }
            }
        }
    }

    dirs.sort();
    dirs.dedup();
    dirs
}

/// Candidate file names for the CUDA runtime and NVRTC, per platform.
///
/// Multiple versions because torch's bundled version is whatever CUDA torch was
/// built against; the loader tries them all rather than pinning one.
fn runtime_lib_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &[
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
            "cudart64_12.dll",
            "cudart64_11.dll",
        ]
    }
    #[cfg(not(windows))]
    {
        &[
            "libnvrtc.so",
            "libnvrtc.so.12",
            "libcudart.so",
            "libcudart.so.12",
        ]
    }
}

/// Driver names, opened by bare name so the OS search path finds the installed
/// driver (System32 / ldconfig) rather than a copy inside an application.
fn driver_lib_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["nvcuda.dll"]
    }
    #[cfg(not(windows))]
    {
        &["libcuda.so.1", "libcuda.so"]
    }
}

/// True when `dir` looks like it holds a CUDA runtime. Used by the capability
/// probe to avoid reporting a directory that exists but is empty.
pub fn lib_dir_has_cuda(dir: &Path) -> bool {
    runtime_lib_names().iter().any(|n| dir.join(n).exists())
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// The initialised driver, context and compiled kernels.
///
/// SAFETY: raw CUDA handles are not `Send`/`Sync` by construction. Everything
/// that touches the device goes through [`bind`], which makes the context
/// current on the calling thread first, so a shared context is usable from any
/// thread — CUDA contexts are per-thread current state, not per-thread
/// ownership. Buffer and module handles are plain addresses and carry no
/// thread-local state of their own.
pub struct Cuda {
    api: Api,
    ctx: CUcontext,
    device_name: String,
    cap: (i32, i32),
}

unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

static CUDA: LazyLock<Option<Cuda>> = LazyLock::new(init);

/// Make this thread's context current, or return `None`.
///
/// Every device call must be preceded by this: contexts are bound per thread,
/// so a thread that never called `cuCtxSetCurrent` has no current context and
/// every subsequent call fails with `CUDA_ERROR_INVALID_CONTEXT`. That looks
/// exactly like "no GPU" from the caller's side, which is why it is centralised
/// here rather than left to each entry point.
fn bind(c: &Cuda) -> Option<()> {
    if unsafe { (c.api.cuCtxSetCurrent)(c.ctx) } != 0 {
        return None;
    }
    Some(())
}

/// The initialised device, ignoring the env opt-out.
///
/// Split from [`with_device`] so that *cleanup* still works when `LOGAN_CUDA=0`
/// was set after a buffer was allocated: the opt-out governs whether work is
/// dispatched, not whether already-allocated memory is released.
fn cuda() -> Option<&'static Cuda> {
    CUDA.as_ref()
}

/// Run `f` with the CUDA device initialised and the context bound.
///
/// Returns `None` when the backend is unavailable, when the opt-in is not set,
/// or when `f` returns `None`. The single place the opt-in, the initialisation
/// and the thread-binding are applied, so no caller has to remember any of the
/// three.
pub fn with_device<R>(f: impl FnOnce(&Cuda) -> Option<R>) -> Option<R> {
    if !enabled() {
        note_error("not enabled: set LOGAN_CUDA=1".to_string());
        return None;
    }
    let c = cuda()?;
    bind(c)?;
    f(c)
}

fn init() -> Option<Cuda> {
    let dirs = cuda_lib_dirs();

    // NVRTC resolves its builtins by searching the process PATH and its own
    // directory -- not the directory the NVRTC DLL itself was loaded from.
    // Without this, loading succeeds, compilation starts, and then fails with
    // "failed to open nvrtc-builtins...", which reads like a broken install
    // rather than a search-path problem. Done during the single-threaded
    // initialisation, before any other thread can read the environment.
    #[cfg(windows)]
    if let Some(joined) = path_with_dirs(&dirs) {
        unsafe { std::env::set_var("PATH", joined) };
    }

    let mut handles: Vec<*mut c_void> = Vec::new();
    let mut nvrtc_handles: Vec<*mut c_void> = Vec::new();

    for name in driver_lib_names() {
        let h = open_library(name);
        if !h.is_null() {
            handles.push(h);
        }
    }
    if handles.is_empty() {
        note_error(format!(
            "no CUDA driver reachable (tried {:?})",
            driver_lib_names()
        ));
        return None;
    }
    // The runtime is opened only as an additional symbol source: every symbol
    // below is a driver or NVRTC entry point, but a driver stub that forwards to
    // the runtime still resolves this way.
    for dir in &dirs {
        for name in runtime_lib_names() {
            let p = dir.join(name);
            if !p.exists() {
                continue;
            }
            let h = open_library(&p.to_string_lossy());
            if h.is_null() {
                continue;
            }
            if name.contains("nvrtc") {
                nvrtc_handles.push(h);
            } else {
                handles.push(h);
            }
        }
    }

    if nvrtc_handles.is_empty() {
        note_error(format!(
            "no NVRTC library in {dirs:?} (set LOGAN_CUDA_LIB_DIR)"
        ));
        return None;
    }

    let api = Api::load(&handles, &nvrtc_handles)?;

    unsafe {
        // `cuInit(0)` is what turns a loadable driver into a usable one; a
        // machine with the DLL but no device fails here.
        if (api.cuInit)(0) != 0 {
            note_error("cuInit failed".to_string());
            return None;
        }
        let mut dev: CUdevice = 0;
        if (api.cuDeviceGet)(&mut dev, 0) != 0 {
            note_error("cuDeviceGet(0) failed".to_string());
            return None;
        }

        // Read the device's own name and compute capability back from the
        // driver. Both are reported rather than assumed: the CC drives the PTX
        // target, and a PTX built for a newer architecture would fail to JIT on
        // an older card -- a failure that otherwise looks like "CUDA
        // unavailable" instead of a version mismatch.
        let mut name_buf = [0 as c_char; 256];
        let device_name = if (api.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, dev) == 0 {
            CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .trim()
                .to_string()
        } else {
            "unknown CUDA device".to_string()
        };
        // 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        // 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR.
        let mut major = 0 as c_int;
        let mut minor = 0 as c_int;
        if (api.cuDeviceGetAttribute)(&mut major, 75, dev) != 0
            || (api.cuDeviceGetAttribute)(&mut minor, 76, dev) != 0
        {
            note_error("compute capability unavailable".to_string());
            return None;
        }
        if major <= 0 {
            note_error(format!("implausible compute capability {major}.{minor}"));
            return None;
        }

        let mut ctx: CUcontext = std::ptr::null_mut();
        if (api.cuCtxCreate_v2)(&mut ctx, 0, dev) != 0 {
            note_error("cuCtxCreate failed".to_string());
            return None;
        }
        if (api.cuCtxSetCurrent)(ctx) != 0 {
            note_error("cuCtxSetCurrent failed".to_string());
            return None;
        }

        Some(Cuda {
            api,
            ctx,
            device_name,
            cap: (major, minor),
        })
    }
}

/// `PATH` with `dirs` prepended, or `None` when there is nothing to add.
#[cfg(windows)]
fn path_with_dirs(dirs: &[PathBuf]) -> Option<String> {
    if dirs.is_empty() {
        return None;
    }
    let sep = ';';
    let prefix: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
    let mut joined = prefix.join(&sep.to_string());
    if let Ok(existing) = std::env::var("PATH") {
        joined.push(sep);
        joined.push_str(&existing);
    }
    Some(joined)
}

/// Whether a CUDA kernel can be compiled and loaded on this machine.
///
/// Narrow on purpose: this is not "there is a GPU here", it is "a device was
/// opened and NVRTC is usable". After the first call it is a cached lookup.
pub fn available() -> bool {
    enabled() && CUDA.is_some()
}

/// The driver's own name for device 0, e.g. `GeForce GTX 1080`.
///
/// Read from the driver rather than inferred, so a capability report can name
/// the device that actually ran instead of the one that was expected. Note this
/// describes the *device*, so it initialises the driver even when the opt-in is
/// unset — asking what hardware is present is a different question from asking
/// whether it is in use.
pub fn device_name() -> Option<String> {
    Some(CUDA.as_ref()?.device_name.clone())
}

/// Device compute capability as `(major, minor)`, e.g. `(6, 1)` for Pascal.
pub fn compute_capability() -> Option<(i32, i32)> {
    Some(CUDA.as_ref()?.cap)
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// A compiled entry point, usable from any thread (see [`Cuda`]).
pub struct Kernel {
    func: CUfunction,
    name: String,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Launch on a grid of `grid` blocks of `block` threads.
    ///
    /// `params` is the kernel-parameter array `cuLaunchKernel` expects: a
    /// pointer to each argument, **not** the arguments themselves. The caller
    /// owns those values and must keep them alive across the call.
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Option<()> {
        with_device(|c| {
            let rc = unsafe {
                (c.api.cuLaunchKernel)(
                    self.func,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                note_error(error_name(c, rc));
                return None;
            }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
            Some(())
        })
    }
}

/// Human-readable name for a driver error code.
///
/// Best-effort: a driver too old to export `cuGetErrorName` yields the raw
/// code, which is still better than nothing in a diagnostic line.
fn error_name(c: &Cuda, rc: c_int) -> String {
    unsafe {
        let mut p: *const c_char = std::ptr::null();
        let ok = (c.api.cuGetErrorName)(rc, &mut p);
        if ok == 0 && !p.is_null() {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {rc}")
        }
    }
}

/// Compile `source` and return its `entry` point.
///
/// The PTX is generated for this device's compute capability, derived from the
/// driver at initialisation. `source` must define `extern "C" __global__ void
/// entry(...)`: NVRTC compiles C++ with C linkage requested explicitly, so
/// without `extern "C"` the symbol is name-mangled and `cuModuleGetFunction`
/// fails at runtime rather than at compile time.
pub fn compile(source: &str, entry: &str) -> Option<Kernel> {
    with_device(|c| compile_inner(c, source, entry))
}

/// Compile and load, with every FFI call in one place.
///
/// A safe fn holding an explicit `unsafe` block rather than an `unsafe fn`:
/// nothing about the *caller's* obligations changes here, so an `unsafe fn`
/// would push unsafety onto callers for no reason (and in edition 2024 would
/// still need the inner block).
fn compile_inner(c: &Cuda, source: &str, entry: &str) -> Option<Kernel> {
    let api = &c.api;

    // `compute_XY`, not `sm_XY`: NVRTC emits PTX for the former and the driver
    // JITs it for the actual card, which keeps the same binary working across
    // cards of the same or newer architecture. Note the digits are
    // concatenated -- NVRTC rejects "compute_6.1".
    let arch = CString::new(format!("--gpu-architecture=compute_{}{}", c.cap.0, c.cap.1)).ok()?;
    let src = CString::new(source).ok()?;
    let fname = CString::new(format!("{entry}.cu")).ok()?;

    unsafe {
        let mut prog: nvrtcProgram = std::ptr::null_mut();
        if (api.nvrtcCreateProgram)(
            &mut prog,
            src.as_ptr(),
            fname.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ) != 0
        {
            note_error("nvrtcCreateProgram failed".to_string());
            return None;
        }

        let opts = [arch.as_ptr()];
        let rc = (api.nvrtcCompileProgram)(prog, opts.len() as c_int, opts.as_ptr());
        if rc != 0 {
            // The program log is the only way to find out why. A silent NVRTC
            // failure is indistinguishable from "no GPU", which is exactly the
            // confusion this whole module is written to avoid.
            let mut len = 0usize;
            let mut detail = String::new();
            if (api.nvrtcGetProgramLogSize)(prog, &mut len) == 0 && len > 1 {
                let mut buf = vec![0 as c_char; len];
                if (api.nvrtcGetProgramLog)(prog, buf.as_mut_ptr()) == 0 {
                    detail = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
                }
            }
            note_error(format!("nvrtc compile failed (rc={rc}): {detail}"));
            (api.nvrtcDestroyProgram)(&mut prog);
            return None;
        }

        let mut size = 0usize;
        if (api.nvrtcGetPTXSize)(prog, &mut size) != 0 || size == 0 {
            note_error("nvrtcGetPTXSize failed".to_string());
            (api.nvrtcDestroyProgram)(&mut prog);
            return None;
        }
        let mut ptx = vec![0 as c_char; size];
        let got = (api.nvrtcGetPTX)(prog, ptx.as_mut_ptr()) == 0;
        (api.nvrtcDestroyProgram)(&mut prog);
        if !got {
            note_error("nvrtcGetPTX failed".to_string());
            return None;
        }

        let mut module: CUmodule = std::ptr::null_mut();
        // NVRTC returns NUL-terminated PTX text; cuModuleLoadData wants a
        // pointer to it. Loading triggers the JIT for this device.
        if (api.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void) != 0 {
            note_error("cuModuleLoadData failed (PTX rejected by the driver)".to_string());
            return None;
        }

        let cname = CString::new(entry).ok()?;
        let mut func: CUfunction = std::ptr::null_mut();
        if (api.cuModuleGetFunction)(&mut func, module, cname.as_ptr()) != 0 {
            note_error(format!("entry point `{entry}` not found in module"));
            return None;
        }
        Some(Kernel {
            func,
            name: entry.to_string(),
        })
    }
}

impl Kernel {
    /// The entry-point name this handle was compiled for.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// A device buffer that grows on demand and is reused across calls.
///
/// Reuse is not an optimisation detail: a per-call `cuMemAlloc` costs on the
/// order of ten microseconds, and the kernels this exists for run in well under
/// a millisecond, so allocation would dominate the measurement. Growing rather
/// than reallocating keeps the steady state at allocation-free.
///
/// The address can change when the buffer grows. A caller caching anything
/// derived from [`Self::ptr`] must compare it, not assume it.
pub struct DeviceBuf {
    ptr: CUdeviceptr,
    /// Capacity in bytes.
    cap: usize,
}

unsafe impl Send for DeviceBuf {}
unsafe impl Sync for DeviceBuf {}

impl DeviceBuf {
    pub fn new() -> Self {
        Self { ptr: 0, cap: 0 }
    }

    /// Address of the buffer, or `None` while it is unallocated.
    pub fn ptr(&self) -> Option<CUdeviceptr> {
        if self.ptr == 0 { None } else { Some(self.ptr) }
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Ensure at least `bytes` of capacity, reallocating only when needed.
    pub fn ensure(&mut self, bytes: usize) -> Option<()> {
        if self.ptr != 0 && self.cap >= bytes {
            return Some(());
        }
        let c = cuda()?;
        let _ = bind(c);
        // Free before allocating: holding both would double the peak footprint,
        // and these buffers hold weight matrices.
        let old = std::mem::replace(&mut self.ptr, 0);
        self.cap = 0;
        unsafe {
            if old != 0 {
                (c.api.cuMemFree_v2)(old);
            }
        }
        let mut p: CUdeviceptr = 0;
        if unsafe { (c.api.cuMemAlloc_v2)(&mut p, bytes) } != 0 {
            note_error(format!("cuMemAlloc of {bytes} bytes failed"));
            return None;
        }
        self.ptr = p;
        self.cap = bytes;
        Some(())
    }

    /// Copy `data` to the device, growing the buffer if needed.
    pub fn upload(&mut self, data: &[u8]) -> Option<()> {
        self.ensure(data.len())?;
        let dst = self.ptr()?;
        with_device(|c| {
            let rc =
                unsafe { (c.api.cuMemcpyHtoD_v2)(dst, data.as_ptr() as *const c_void, data.len()) };
            if rc != 0 {
                note_error(error_name(c, rc));
                return None;
            }
            Some(())
        })
    }

    /// Copy `out.len()` bytes back from the device.
    ///
    /// Implies a device synchronisation: the copy does not begin until the
    /// preceding launches complete, which is the behaviour a caller reading a
    /// kernel result wants. A caller that needs the launches to overlap should
    /// use a stream, which this backend does not expose.
    pub fn download(&self, out: &mut [u8]) -> Option<()> {
        let src = self.ptr()?;
        if out.len() > self.cap {
            return None;
        }
        with_device(|c| {
            let rc =
                unsafe { (c.api.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, src, out.len()) };
            if rc != 0 {
                note_error(error_name(c, rc));
                return None;
            }
            Some(())
        })
    }
}

impl Default for DeviceBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // Deliberately `cuda()`, not `with_device()`: the env opt-out stops
            // new work being dispatched, and dropping a buffer is not new work.
            if let Some(c) = cuda() {
                let _ = bind(c);
                unsafe { (c.api.cuMemFree_v2)(self.ptr) };
            }
            self.ptr = 0;
            self.cap = 0;
        }
    }
}

/// Block a thread until every preceding launch on the context has completed.
pub fn synchronize() -> Option<()> {
    with_device(|c| {
        if unsafe { (c.api.cuCtxSynchronize)() } != 0 {
            note_error("cuCtxSynchronize failed".to_string());
            return None;
        }
        Some(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Availability must be answerable anywhere. The Mac this usually runs on
    /// has no CUDA driver at all; probing must return `false` rather than
    /// unwinding, because it happens during capability detection.
    #[test]
    fn probing_is_a_bool_and_never_panics() {
        let _ = available();
        let _ = device_name();
        let _ = compute_capability();
        let _ = init_error();
        let _ = launches();
    }

    /// The opt-in must be off by default and must fail closed on junk.
    ///
    /// Exercises the pure predicate rather than mutating the process
    /// environment: these tests run in parallel with the parity test, and a
    /// global set here could disable the backend mid-parity — or, worse, be
    /// observed by `STATE`'s `LazyLock` initialiser and cache an uninitialised
    /// backend for the rest of the run.
    #[test]
    fn the_opt_in_is_off_by_default_and_rejects_junk() {
        // Default OFF. This is the whole point: the backend is measured slower
        // than the CPU, so absence must mean "do not use it".
        assert!(!enabled_given(None), "unset must be OFF");
        // OFF for anything that is not an explicit yes, including the old
        // opt-out spelling and plausible typos.
        for junk in ["0", "", "no", "false", "ture", "yes", "on", "2", " "] {
            assert!(!enabled_given(Some(junk)), "{junk:?} must be OFF");
        }
        // ON only for an explicit, case-insensitive yes.
        for yes in ["1", "true", "TRUE", "True", " 1 ", "1\n"] {
            assert!(enabled_given(Some(yes)), "{yes:?} must be ON");
        }
        // And the live reader agrees with the predicate on the current value.
        assert_eq!(
            enabled(),
            enabled_given(std::env::var(GLOGAN_CUDA).ok().as_deref())
        );
    }

    #[test]
    fn lib_dir_detection_rejects_directories_without_a_runtime() {
        let tmp = std::env::temp_dir().join("logan-cuda-probe-test");
        let _ = std::fs::create_dir_all(&tmp);
        assert!(!lib_dir_has_cuda(&tmp));
        assert!(!lib_dir_has_cuda(Path::new("/nonexistent/nope")));
    }

    /// A kernel that cannot be compiled must fail closed rather than panic --
    /// including a syntactically broken one, which is the realistic case when
    /// editing kernel source.
    #[test]
    fn compiling_a_broken_kernel_fails_closed() {
        assert!(compile("this is not CUDA", "nope").is_none());
        assert!(compile("extern \"C\" __global__ void f(){}", "other").is_none());
    }

    /// A buffer that was never allocated must not claim an address, and must
    /// not panic when dropped.
    #[test]
    fn an_unallocated_buffer_is_inert() {
        let mut b = DeviceBuf::new();
        assert!(b.ptr().is_none());
        assert_eq!(b.capacity(), 0);
        let mut out = [0u8; 4];
        assert!(b.download(&mut out).is_none());
        drop(b);
    }
}
