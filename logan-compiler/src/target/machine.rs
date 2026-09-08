//! Machine profile: what the *host the compiler runs on* can execute, read
//! by a dependency-free, read-only probe (no Metal FFI — the runtime's
//! device init remains authoritative for execution; the compiler mirrors
//! its gates from static capability data + environment).
//!
//! The Apple8 ABI is available when the runtime's direct-path gate would
//! accept it: macOS/aarch64 (Metal + Apple-silicon) with the direct path
//! not disabled (`QWEN_APPLE8_DIRECT=0`). `COLI_METAL=0` mirrors the C
//! engine's global Metal kill-switch.

use crate::target_registry;

/// Host capabilities the planner decides targets and memory budgets from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineProfile {
    pub operating_system: &'static str,
    pub architecture: &'static str,
    /// Physical RAM in bytes (the ONE unified pool on Apple silicon).
    /// `None` when it cannot be read; `RAM_GB` (GiB) overrides, matching
    /// the C engine's RAM budget knob.
    pub ram_bytes: Option<u64>,
    /// Unified-memory host: CPU and GPU share one physical memory pool.
    pub unified_memory: bool,
    /// Metal backend reachable (static OS gate + `COLI_METAL` mirror).
    pub metal_available: bool,
    /// Private Apple Neural Engine runtime/compiler appears usable on this
    /// native host. Runtime ABI probing remains authoritative before execution.
    pub ane_available: bool,
    /// Metal and ANE can exchange activations through one IOSurface-backed UMA
    /// allocation without an explicit host copy on this host.
    pub ane_iosurface_interop: bool,
    /// Apple8/MXFP4 direct-execution ABI available. Mirrors the runtime's
    /// direct-path gate as far as a read-only probe can see (device-level
    /// FFI init is checked again at runtime before execution).
    pub apple8_abi: bool,
    /// x86_64 AVX2 detected on this host (the CPU profile's capability).
    pub avx2: bool,
    /// Minimum Apple GPU family the Apple8 ABI requires (registry contract,
    /// static capability data).
    pub apple_gpu_family_min: u32,
}

/// Default unified-pool budget used when the machine RAM cannot be read.
pub const DEFAULT_POOL_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

impl MachineProfile {
    pub fn probe() -> MachineProfile {
        let operating_system = std::env::consts::OS;
        let architecture = std::env::consts::ARCH;
        let coli_metal = std::env::var("COLI_METAL").ok();
        let metal_available = metal_available_for(operating_system, coli_metal.as_deref());
        let ane_gate = std::env::var("LOGAN_ANE").ok();
        let ane_available = ane_available_for(
            operating_system,
            architecture,
            ane_gate.as_deref(),
            private_ane_frameworks_present(),
        );
        let unified_memory = operating_system == "macos" && architecture == "aarch64";
        let direct = std::env::var("QWEN_APPLE8_DIRECT").ok();
        MachineProfile {
            operating_system,
            architecture,
            ram_bytes: detect_ram_bytes(),
            unified_memory,
            metal_available,
            ane_available,
            ane_iosurface_interop: ane_available && metal_available && unified_memory,
            apple8_abi: Self::apple8_abi_for(
                operating_system,
                architecture,
                metal_available,
                direct.as_deref(),
            ),
            avx2: current_has_avx2(),
            apple_gpu_family_min: target_registry::APPLE8_GPU_FAMILY_MIN,
        }
    }

    /// Pure selection gate exercised by `resolve`; factored out so the
    /// env-dependent parts stay testable without touching the process env.
    pub fn apple8_abi_for(
        operating_system: &str,
        architecture: &str,
        metal_available: bool,
        qwen_apple8_direct: Option<&str>,
    ) -> bool {
        metal_available
            && operating_system == "macos"
            && architecture == "aarch64"
            && qwen_apple8_direct.map(|v| v != "0").unwrap_or(true)
    }
}

/// macOS can run Metal; `COLI_METAL=0` mirrors the C engine's kill-switch.
pub fn metal_available_for(operating_system: &str, coli_metal: Option<&str>) -> bool {
    operating_system == "macos" && coli_metal.map(|v| v != "0").unwrap_or(true)
}

/// Read-only compiler-side ANE gate. The runtime still validates private
/// classes/selectors/type encodings before executing any ANE island.
pub fn ane_available_for(
    operating_system: &str,
    architecture: &str,
    logan_ane: Option<&str>,
    private_frameworks_present: bool,
) -> bool {
    operating_system == "macos"
        && architecture == "aarch64"
        && private_frameworks_present
        && logan_ane.map(|v| v != "0").unwrap_or(true)
}

fn private_ane_frameworks_present() -> bool {
    #[cfg(target_os = "macos")]
    {
        const ANE: &str =
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
        const COMPILER: &str =
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";
        std::path::Path::new(ANE).is_file() && std::path::Path::new(COMPILER).is_file()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Physical RAM: `RAM_GB` (GiB) env override, else `hw.memsize` on macOS,
/// else MemTotal on Linux. `None` when unreadable/other OS.
pub fn detect_ram_bytes() -> Option<u64> {
    if let Some(gib) = std::env::var("RAM_GB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
    {
        return gib.checked_mul(1024 * 1024 * 1024);
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if output.status.success() {
            let text = String::from_utf8(output.stdout).ok()?;
            return text.trim().parse::<u64>().ok();
        }
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        return kib.checked_mul(1024);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn current_has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn current_has_avx2() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_requires_macos_and_respects_kill_switch() {
        assert!(metal_available_for("macos", None));
        assert!(metal_available_for("macos", Some("1")));
        assert!(!metal_available_for("macos", Some("0")));
        assert!(!metal_available_for("linux", None));
        assert!(!metal_available_for("windows", None));
    }

    #[test]
    fn ane_requires_apple_silicon_private_frameworks_and_respects_kill_switch() {
        assert!(ane_available_for("macos", "aarch64", None, true));
        assert!(ane_available_for("macos", "aarch64", Some("1"), true));
        assert!(!ane_available_for("macos", "aarch64", Some("0"), true));
        assert!(!ane_available_for("macos", "x86_64", None, true));
        assert!(!ane_available_for("linux", "aarch64", None, true));
        assert!(!ane_available_for("macos", "aarch64", None, false));
    }

    #[test]
    fn apple8_abi_requires_macos_aarch64_metal_and_live_direct_path() {
        let ok = |metal: bool, direct: Option<&str>| {
            MachineProfile::apple8_abi_for("macos", "aarch64", metal, direct)
        };
        assert!(ok(true, None));
        assert!(ok(true, Some("1")));
        // mirrors the runtime direct-path gate exactly
        assert!(!ok(true, Some("0")));
        assert!(!ok(false, None));
        assert!(!MachineProfile::apple8_abi_for(
            "macos", "x86_64", true, None
        ));
        assert!(!MachineProfile::apple8_abi_for(
            "linux", "aarch64", true, None
        ));
    }

    #[test]
    fn unified_memory_is_an_apple_silicon_hardware_fact() {
        let machine = MachineProfile::probe();
        assert_eq!(
            machine.unified_memory,
            machine.operating_system == "macos" && machine.architecture == "aarch64"
        );
        assert_eq!(machine.apple_gpu_family_min, 8);
        assert_eq!(
            machine.ane_iosurface_interop,
            machine.unified_memory && machine.metal_available && machine.ane_available
        );
        // A machine that passes every static gate reports the Apple8 ABI.
        assert_eq!(
            machine.apple8_abi,
            MachineProfile::apple8_abi_for(
                machine.operating_system,
                machine.architecture,
                machine.metal_available,
                std::env::var("QWEN_APPLE8_DIRECT").ok().as_deref(),
            )
        );
    }

    #[test]
    fn ram_override_wins_over_detection() {
        // RAM_GB is read by probe(); set it, read, restore.
        unsafe {
            std::env::set_var("RAM_GB", "7");
        }
        let bytes = detect_ram_bytes().unwrap();
        unsafe {
            std::env::remove_var("RAM_GB");
        }
        assert_eq!(bytes, 7 * 1024 * 1024 * 1024);
    }
}
