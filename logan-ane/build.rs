fn main() {
    println!("cargo:rerun-if-changed=native/async_bridge.mm");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os != "macos" || arch != "aarch64" {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/async_bridge.mm");
    let obj = out.join("async_bridge.o");
    let lib = out.join("liblogan_ane_async.a");
    let status = std::process::Command::new("clang++")
        .args([
            "-x",
            "objective-c++",
            "-fobjc-arc",
            "-fblocks",
            "-O2",
            "-mmacosx-version-min=11.0",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang required on macOS");
    assert!(status.success(), "logan-ane async bridge failed to compile");
    let status = std::process::Command::new("ar")
        .args(["rcs"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("ar required on macOS");
    assert!(status.success(), "logan-ane async bridge archive failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=logan_ane_async");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}
