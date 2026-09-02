use std::path::PathBuf;

use logan_ir::ContextConstraint;

use crate::{
    error::{ColicError, Result},
    pipeline::{
        CodecRequest, CompileRequest, OptimizationProfile, QuantFloor, QuantRequest, TargetRequest,
    },
    recompile::{
        CodecMode as RecompileCodecMode, QuantMode as RecompileQuantMode, QuantRule,
        RecompileRequest,
    },
};

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    InspectSource {
        source: PathBuf,
    },
    Verify {
        package: PathBuf,
    },
    Compile(CompileRequest),
    Recompile(RecompileRequest),
    Run {
        package: std::path::PathBuf,
        prompt: String,
        max_new: usize,
    },
    Help,
}

pub const USAGE: &str = "Usage:\n  logan inspect-source MODEL_DIR\n  logan verify PACKAGE_DIR\n  logan compile MODEL_DIR (--max-context N | --require-context N) [--optimize] [--select-plan quality|balanced|long-context|latency|PLAN_ID] --target auto|native|PROFILE --quant exact|PROFILE --quant-floor bf16|exact --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--plan PLAN_PATH] [--dry-run] [--verify] [--force]\n  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|auto|native|PROFILE] [--optimize (--max-context N | --require-context N)] [--select-plan quality|balanced|long-context|latency|PLAN_ID] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]";

pub fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "help" | "--help" | "-h" => Ok(Command::Help),
        "inspect-source" => {
            let source = args
                .next()
                .ok_or_else(|| ColicError::Usage("inspect-source requires MODEL_DIR".into()))?;
            if args.next().is_some() {
                return Err(ColicError::Usage(
                    "inspect-source accepts exactly one MODEL_DIR".into(),
                ));
            }
            Ok(Command::InspectSource {
                source: PathBuf::from(source),
            })
        }
        "compile" => parse_compile(args),
        "recompile" => parse_recompile(args),
        "run" => {
            let package = std::path::PathBuf::from(
                args.next()
                    .ok_or_else(|| ColicError::Usage("run requires PACKAGE_DIR".into()))?,
            );
            let mut prompt = String::from("1 2 3 4 5");
            let mut max_new = 16;
            let mut it = args;
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--prompt" => {
                        prompt = it
                            .next()
                            .ok_or_else(|| ColicError::Usage("--prompt needs a value".into()))?
                    }
                    "--max-new" => {
                        max_new = it
                            .next()
                            .ok_or_else(|| ColicError::Usage("--max-new needs a value".into()))?
                            .parse()
                            .map_err(|_| ColicError::Usage("--max-new must be a number".into()))?
                    }
                    other => return Err(ColicError::Usage(format!("unknown run flag {other}"))),
                }
            }
            Ok(Command::Run {
                package,
                prompt,
                max_new,
            })
        }
        "verify" => {
            let package = args
                .next()
                .ok_or_else(|| ColicError::Usage("verify requires PACKAGE_DIR".into()))?;
            if args.next().is_some() {
                return Err(ColicError::Usage(
                    "verify accepts exactly one PACKAGE_DIR".into(),
                ));
            }
            Ok(Command::Verify {
                package: PathBuf::from(package),
            })
        }
        other => Err(ColicError::Usage(format!("unknown command `{other}`"))),
    }
}

fn parse_compile<I>(args: I) -> Result<Command>
where
    I: Iterator<Item = String>,
{
    let mut args = args.into_iter();
    let source = PathBuf::from(
        args.next()
            .ok_or_else(|| ColicError::Usage("compile requires MODEL_DIR".into()))?,
    );
    let mut request = CompileRequest::new(source);
    while let Some(flag) = args.next() {
        let value = |args: &mut I, flag: &str| {
            args.next()
                .ok_or_else(|| ColicError::Usage(format!("{flag} requires a value")))
        };
        match flag.as_str() {
            "--target" => request.target = TargetRequest::parse(&value(&mut args, "--target")?)?,
            "--quant" => {
                request.quant = QuantRequest::parse(&value(&mut args, "--quant")?)?;
                request.quant_explicit = true;
            }
            "--quant-floor" => {
                request.quant_floor = QuantFloor::parse(&value(&mut args, "--quant-floor")?)?
            }
            "--codec" => request.codec = CodecRequest::parse(&value(&mut args, "--codec")?)?,
            "--opt" => {
                request.optimization = OptimizationProfile::parse(&value(&mut args, "--opt")?)?;
            }
            "--optimize" => request.optimize = true,
            "--select-plan" => request.plan_choice = Some(value(&mut args, "--select-plan")?),
            "--max-context" => {
                if request.context.is_some() {
                    return Err(ColicError::Usage(
                        "--max-context and --require-context are mutually exclusive".into(),
                    ));
                }
                request.context = Some(ContextConstraint::maximum(parse_context_tokens(
                    &value(&mut args, "--max-context")?,
                    "--max-context",
                )?));
            }
            "--require-context" => {
                if request.context.is_some() {
                    return Err(ColicError::Usage(
                        "--max-context and --require-context are mutually exclusive".into(),
                    ));
                }
                request.context = Some(ContextConstraint::required(parse_context_tokens(
                    &value(&mut args, "--require-context")?,
                    "--require-context",
                )?));
            }
            "--plan" => request.plan = Some(PathBuf::from(value(&mut args, "--plan")?)),
            "-o" | "--output" => {
                request.output = Some(PathBuf::from(value(&mut args, "--output")?))
            }
            "--dry-run" => request.dry_run = true,
            "--verify" => request.verify = true,
            "--force" => request.force = true,
            other => {
                return Err(ColicError::Usage(format!(
                    "unknown compile option `{other}`"
                )));
            }
        }
    }
    if !request.dry_run && request.output.is_none() {
        return Err(ColicError::Usage(
            "compile requires -o/--output unless --dry-run is set".into(),
        ));
    }
    if request.context.is_none() {
        return Err(ColicError::Usage(
            "compile requires exactly one of --max-context N or --require-context N".into(),
        ));
    }
    if request.plan_choice.is_some() && !request.optimize {
        return Err(ColicError::Usage(
            "--select-plan requires --optimize".into(),
        ));
    }
    Ok(Command::Compile(request))
}

fn parse_context_tokens(value: &str, flag: &str) -> Result<u64> {
    let tokens = value
        .parse::<u64>()
        .map_err(|_| ColicError::Usage(format!("{flag} must be a positive integer")))?;
    if tokens == 0 {
        return Err(ColicError::Usage(format!(
            "{flag} must be greater than zero"
        )));
    }
    Ok(tokens)
}

fn parse_recompile<I>(args: I) -> Result<Command>
where
    I: Iterator<Item = String>,
{
    let mut args = args.into_iter();
    let source = PathBuf::from(
        args.next()
            .ok_or_else(|| ColicError::Usage("recompile requires PACKAGE_DIR".into()))?,
    );
    let mut output = None;
    let mut target = "source".to_owned();
    let mut target_explicit = false;
    let mut context = None;
    let mut optimize = false;
    let mut quant = RecompileQuantMode::Keep;
    let mut quant_rules = Vec::new();
    let mut codec = RecompileCodecMode::Keep;
    let mut allow_requantize = false;
    let mut repack = false;
    let mut verify = false;
    let mut force = false;
    let mut in_place = false;

    while let Some(flag) = args.next() {
        let value = |args: &mut I, flag: &str| {
            args.next()
                .ok_or_else(|| ColicError::Usage(format!("{flag} requires a value")))
        };
        match flag.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(value(&mut args, &flag)?)),
            "--target" => {
                target = value(&mut args, "--target")?;
                target_explicit = true;
            }
            "--optimize" => optimize = true,
            "--max-context" => {
                if context.is_some() {
                    return Err(ColicError::Usage(
                        "--max-context and --require-context are mutually exclusive".into(),
                    ));
                }
                context = Some(ContextConstraint::maximum(parse_context_tokens(
                    &value(&mut args, "--max-context")?,
                    "--max-context",
                )?));
            }
            "--require-context" => {
                if context.is_some() {
                    return Err(ColicError::Usage(
                        "--max-context and --require-context are mutually exclusive".into(),
                    ));
                }
                context = Some(ContextConstraint::required(parse_context_tokens(
                    &value(&mut args, "--require-context")?,
                    "--require-context",
                )?));
            }
            "--quant" => quant = RecompileQuantMode::parse(&value(&mut args, "--quant")?)?,
            "--quant-rule" => {
                quant_rules.push(QuantRule::parse(&value(&mut args, "--quant-rule")?)?)
            }
            "--codec" => codec = RecompileCodecMode::parse(&value(&mut args, "--codec")?)?,
            "--allow-requantize" => allow_requantize = true,
            "--repack" => repack = true,
            "--verify" => verify = true,
            "--force" => force = true,
            "--in-place" => in_place = true,
            other => {
                return Err(ColicError::Usage(format!(
                    "unknown recompile option `{other}`"
                )));
            }
        }
    }

    if optimize && context.is_none() {
        return Err(ColicError::Usage(
            "recompile --optimize requires exactly one of --max-context N or --require-context N"
                .into(),
        ));
    }
    if !optimize && context.is_some() {
        return Err(ColicError::Usage(
            "recompile context options require --optimize so they cannot be silently ignored"
                .into(),
        ));
    }
    if optimize && !target_explicit {
        target = "auto".to_owned();
    }

    if in_place && output.is_some() {
        return Err(ColicError::Usage(
            "recompile --in-place cannot be combined with -o/--output".into(),
        ));
    }
    let output = if in_place {
        source.clone()
    } else {
        output.ok_or_else(|| {
            ColicError::Usage("recompile requires -o/--output or --in-place".into())
        })?
    };
    Ok(Command::Recompile(RecompileRequest {
        source,
        output,
        target,
        quant,
        quant_rules,
        codec,
        context,
        optimize,
        allow_requantize,
        repack,
        verify,
        force: force || in_place,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_deterministic_compile_request() {
        let command = parse(
            [
                "compile",
                "fixture",
                "--target",
                "native",
                "--quant",
                "exact",
                "--codec",
                "none",
                "--opt",
                "latency",
                "--max-context",
                "65536",
                "-o",
                "out.coli",
                "--verify",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Compile(request) = command else {
            panic!("expected compile")
        };
        assert_eq!(request.target, TargetRequest::Native);
        assert_eq!(request.quant, QuantRequest::Exact);
        assert!(request.quant_explicit);
        assert_eq!(request.codec, CodecRequest::None);
        assert_eq!(request.optimization, OptimizationProfile::Latency);
        assert_eq!(request.context, Some(ContextConstraint::maximum(65_536)));
        assert!(request.verify);
    }

    #[test]
    fn parses_recompile_request_with_explicit_requantization() {
        let command = parse(
            [
                "recompile",
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
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Recompile(request) = command else {
            panic!("expected recompile")
        };
        assert_eq!(request.source, PathBuf::from("old.coli"));
        assert_eq!(request.output, PathBuf::from("new.coli"));
        assert_eq!(request.target, "macos-arm64-metal-apple8-v1");
        assert_eq!(request.quant, RecompileQuantMode::Mxfp4);
        assert_eq!(request.codec, RecompileCodecMode::None);
        assert!(request.allow_requantize);
        assert!(request.repack);
        assert!(request.verify);
    }

    #[test]
    fn parses_mixed_quant_rules_and_in_place() {
        let command = parse(
            [
                "recompile",
                "model.coli",
                "--in-place",
                "--quant",
                "keep",
                "--quant-rule",
                "layer:0-7=mxfp4",
                "--quant-rule",
                "expert:0=keep",
                "--allow-requantize",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Recompile(request) = command else {
            panic!("expected recompile")
        };
        assert_eq!(request.source, PathBuf::from("model.coli"));
        assert_eq!(request.output, request.source);
        assert_eq!(request.quant_rules.len(), 2);
        assert!(request.allow_requantize);
        assert!(request.force);
    }

    #[test]
    fn parses_optimized_recompile_with_same_context_contract() {
        let command = parse(
            [
                "recompile",
                "model.coli",
                "--in-place",
                "--optimize",
                "--require-context",
                "131072",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Recompile(request) = command else {
            panic!("expected recompile")
        };
        assert!(request.optimize);
        assert_eq!(request.target, "auto");
        assert_eq!(request.context, Some(ContextConstraint::required(131_072)));
    }

    #[test]
    fn compile_and_recompile_reject_ambiguous_context_constraints() {
        assert!(
            parse(
                [
                    "compile",
                    "fixture",
                    "--max-context",
                    "32768",
                    "--require-context",
                    "65536",
                    "--dry-run",
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
        assert!(
            parse(
                [
                    "recompile",
                    "model.coli",
                    "--in-place",
                    "--optimize",
                    "--max-context",
                    "32768",
                    "--require-context",
                    "65536",
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn optimized_recompile_requires_context() {
        assert!(
            parse(["recompile", "model.coli", "--in-place", "--optimize"].map(str::to_owned))
                .is_err()
        );
    }

    #[test]
    fn recompile_rejects_output_with_in_place() {
        assert!(
            parse(["recompile", "model.coli", "--in-place", "-o", "other.coli"].map(str::to_owned))
                .is_err()
        );
    }

    #[test]
    fn recompile_requires_an_output_argument() {
        assert!(parse(["recompile", "old.coli"].map(str::to_owned)).is_err());
    }

    #[test]
    fn portable_target_is_rejected() {
        assert!(
            parse(
                ["compile", "fixture", "--target", "portable-v1", "--dry-run"].map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn parses_standalone_verify_command() {
        assert_eq!(
            parse(["verify", "package.coli"].map(str::to_owned)).unwrap(),
            Command::Verify {
                package: PathBuf::from("package.coli")
            }
        );
    }
}
