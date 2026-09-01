use std::path::PathBuf;

use logan_compiler::{
    ColicError, Result,
    recompile::{CodecMode, QuantMode, RecompileRequest},
};

const USAGE: &str = "Usage:\n  logan-recompile PACKAGE_DIR -o OUTPUT [--target source|PROFILE] [--quant keep|mxfp4] [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]\n\nExamples:\n  logan-recompile qwen.coli -o qwen-repacked.coli --repack --verify\n  logan-recompile linux.coli -o apple.coli --target macos-arm64-metal-apple8-v1 --quant mxfp4 --verify\n  logan-recompile int4.coli -o mxfp4.coli --quant mxfp4 --allow-requantize --verify";

fn main() {
    if let Err(error) = run() {
        eprintln!("logan-recompile: {error}");
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let request = parse(std::env::args().skip(1))?;
    let summary = logan_compiler::recompile::recompile(&request)?;
    println!("source_profile={}", summary.source_profile);
    println!("target_profile={}", summary.target_profile);
    println!("records={}", summary.records);
    println!("copied_records={}", summary.copied_records);
    println!("rewritten_experts={}", summary.rewritten_experts);
    println!("requantized_experts={}", summary.requantized_experts);
    println!("source_fingerprint={}", summary.source_fingerprint);
    Ok(())
}

fn parse<I>(mut args: I) -> Result<RecompileRequest>
where
    I: Iterator<Item = String>,
{
    let source = args
        .next()
        .ok_or_else(|| ColicError::Usage("recompile requires PACKAGE_DIR".into()))?;
    if matches!(source.as_str(), "help" | "--help" | "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }

    let mut output = None;
    let mut target = "source".to_owned();
    let mut quant = QuantMode::Keep;
    let mut codec = CodecMode::Keep;
    let mut allow_requantize = false;
    let mut repack = false;
    let mut verify = false;
    let mut force = false;

    while let Some(flag) = args.next() {
        let value = |args: &mut I, flag: &str| {
            args.next()
                .ok_or_else(|| ColicError::Usage(format!("{flag} requires a value")))
        };
        match flag.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(value(&mut args, &flag)?)),
            "--target" => target = value(&mut args, "--target")?,
            "--quant" => quant = QuantMode::parse(&value(&mut args, "--quant")?)?,
            "--codec" => codec = CodecMode::parse(&value(&mut args, "--codec")?)?,
            "--allow-requantize" => allow_requantize = true,
            "--repack" => repack = true,
            "--verify" => verify = true,
            "--force" => force = true,
            other => {
                return Err(ColicError::Usage(format!(
                    "unknown recompile option `{other}`"
                )))
            }
        }
    }

    let output = output.ok_or_else(|| ColicError::Usage("recompile requires -o/--output".into()))?;
    Ok(RecompileRequest {
        source: PathBuf::from(source),
        output,
        target,
        quant,
        codec,
        allow_requantize,
        repack,
        verify,
        force,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_requantization_without_environment_parameters() {
        let request = parse(
            [
                "old.coli",
                "-o",
                "new.coli",
                "--target",
                "macos-arm64-metal-apple8-v1",
                "--quant",
                "mxfp4",
                "--codec",
                "none",
                "--allow-requantize",
                "--repack",
                "--verify",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(request.source, PathBuf::from("old.coli"));
        assert_eq!(request.output, PathBuf::from("new.coli"));
        assert_eq!(request.target, "macos-arm64-metal-apple8-v1");
        assert_eq!(request.quant, QuantMode::Mxfp4);
        assert_eq!(request.codec, CodecMode::None);
        assert!(request.allow_requantize);
        assert!(request.repack);
        assert!(request.verify);
    }
}
