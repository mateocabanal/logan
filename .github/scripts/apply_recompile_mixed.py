from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


# --- recompile.rs ---------------------------------------------------------
path = Path("logan-compiler/src/recompile.rs")
text = path.read_text()

marker = "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum CodecMode"
mixed_defs = r'''#[derive(Debug, Clone, PartialEq, Eq)]
enum QuantSelector {
    All,
    Layer { first: i32, last: i32 },
    Expert { first: i32, last: i32 },
    Pair { layer: i32, expert: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantRule {
    selector: QuantSelector,
    mode: QuantMode,
}

impl QuantRule {
    pub fn parse(value: &str) -> Result<Self> {
        let (selector, mode) = value.rsplit_once('=').ok_or_else(|| {
            ColicError::Usage(format!(
                "invalid quant rule `{value}` (expected SELECTOR=keep|mxfp4)"
            ))
        })?;
        let mode = QuantMode::parse(mode)?;
        let selector = if selector == "all" {
            QuantSelector::All
        } else if let Some((layer, expert)) = selector.split_once('/') {
            let layer = layer.strip_prefix("layer:").ok_or_else(|| {
                ColicError::Usage(format!("invalid quant-rule selector `{selector}`"))
            })?;
            let expert = expert.strip_prefix("expert:").ok_or_else(|| {
                ColicError::Usage(format!("invalid quant-rule selector `{selector}`"))
            })?;
            QuantSelector::Pair {
                layer: parse_nonnegative_index(layer, "layer")?,
                expert: parse_nonnegative_index(expert, "expert")?,
            }
        } else if let Some(range) = selector
            .strip_prefix("layer:")
            .or_else(|| selector.strip_prefix("layers:"))
        {
            let (first, last) = parse_nonnegative_range(range, "layer")?;
            QuantSelector::Layer { first, last }
        } else if let Some(range) = selector
            .strip_prefix("expert:")
            .or_else(|| selector.strip_prefix("experts:"))
        {
            let (first, last) = parse_nonnegative_range(range, "expert")?;
            QuantSelector::Expert { first, last }
        } else {
            return Err(ColicError::Usage(format!(
                "invalid quant-rule selector `{selector}` (expected all, layer:N[-M], expert:N[-M], or layer:N/expert:M)"
            )));
        };
        Ok(Self { selector, mode })
    }

    fn matches(&self, record: &RecordInfo) -> bool {
        if record.kind != 2 {
            return false;
        }
        match self.selector {
            QuantSelector::All => true,
            QuantSelector::Layer { first, last } => (first..=last).contains(&record.layer),
            QuantSelector::Expert { first, last } => (first..=last).contains(&record.expert),
            QuantSelector::Pair { layer, expert } => {
                record.layer == layer && record.expert == expert
            }
        }
    }

    fn as_spec(&self) -> String {
        let selector = match self.selector {
            QuantSelector::All => "all".to_owned(),
            QuantSelector::Layer { first, last } if first == last => format!("layer:{first}"),
            QuantSelector::Layer { first, last } => format!("layer:{first}-{last}"),
            QuantSelector::Expert { first, last } if first == last => format!("expert:{first}"),
            QuantSelector::Expert { first, last } => format!("expert:{first}-{last}"),
            QuantSelector::Pair { layer, expert } => format!("layer:{layer}/expert:{expert}"),
        };
        format!("{selector}={}", self.mode.as_str())
    }
}

fn parse_nonnegative_index(value: &str, what: &str) -> Result<i32> {
    let parsed = value.parse::<i32>().map_err(|_| {
        ColicError::Usage(format!("quant-rule {what} `{value}` is not an integer"))
    })?;
    if parsed < 0 {
        return Err(ColicError::Usage(format!(
            "quant-rule {what} must be non-negative"
        )));
    }
    Ok(parsed)
}

fn parse_nonnegative_range(value: &str, what: &str) -> Result<(i32, i32)> {
    let (first, last) = match value.split_once('-') {
        Some((first, last)) => (
            parse_nonnegative_index(first, what)?,
            parse_nonnegative_index(last, what)?,
        ),
        None => {
            let value = parse_nonnegative_index(value, what)?;
            (value, value)
        }
    };
    if first > last {
        return Err(ColicError::Usage(format!(
            "quant-rule {what} range starts after it ends"
        )));
    }
    Ok((first, last))
}

'''
text = replace_once(text, marker, mixed_defs + marker, "insert quant rules")
text = replace_once(
    text,
    "    pub quant: QuantMode,\n    pub codec: CodecMode,",
    "    pub quant: QuantMode,\n    /// Ordered routed-expert overrides. Later matching rules win.\n    pub quant_rules: Vec<QuantRule>,\n    pub codec: CodecMode,",
    "request quant_rules field",
)
text = replace_once(
    text,
    "            quant: QuantMode::Keep,\n            codec: CodecMode::Keep,",
    "            quant: QuantMode::Keep,\n            quant_rules: Vec::new(),\n            codec: CodecMode::Keep,",
    "request quant_rules default",
)
summary_marker = "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct RecompileSummary"
effective = r'''fn effective_quant(request: &RecompileRequest, record: &RecordInfo) -> QuantMode {
    request
        .quant_rules
        .iter()
        .filter(|rule| rule.matches(record))
        .last()
        .map_or(request.quant, |rule| rule.mode)
}

'''
text = replace_once(text, summary_marker, effective + summary_marker, "effective quant")
text = replace_once(
    text,
    "    let requantized = kinds.iter().any(|kind| *kind == MatrixKind::Int4G32);\n    if requantized && request.quant == QuantMode::Mxfp4 && !request.allow_requantize {",
    "    let quant = effective_quant(request, record);\n    let source_quantized = kinds.iter().any(|kind| *kind == MatrixKind::Int4G32);\n    let requantized = source_quantized && quant == QuantMode::Mxfp4;\n    if requantized && !request.allow_requantize {",
    "effective requantization",
)
plan_start = text.index("fn plan_record(")
plan_end = text.index("\nfn write_package(", plan_start)
plan = text[plan_start:plan_end]
plan = plan.replace("request.quant == QuantMode::Mxfp4", "quant == QuantMode::Mxfp4")
plan = plan.replace("request.quant == QuantMode::Keep", "quant == QuantMode::Keep")
text = text[:plan_start] + plan + text[plan_end:]

publish_start = text.index("    if request.force {", text.index("let write_result"))
summary_start = text.index("    Ok(RecompileSummary {", publish_start)
new_publish = '''    // Verify the complete sibling artifact before it becomes visible. In-place
    // recompilation always verifies, even without --verify, so a malformed
    // rewrite can never replace the source package.
    if request.verify || request.source == request.output {
        let verification = crate::verify::verify_package(&temporary)
            .and_then(|_| crate::verify_target::verify_target_layouts(&temporary));
        if let Err(error) = verification {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    }

    if request.force {
        storage::replace_package(&temporary, &request.output)?;
    } else {
        storage::publish_package(&temporary, &request.output)?;
    }

'''
text = text[:publish_start] + new_publish + text[summary_start:]
text = replace_once(
    text,
    '        "quant": request.quant.as_str(),\n',
    '        "quant": request.quant.as_str(),\n        "quant_rules": request.quant_rules.iter().map(|rule| rule.as_spec()).collect::<Vec<_>>(),\n        "in_place": request.source == request.output,\n',
    "provenance quant rules",
)

test_marker = "    fn zero_desc() -> MatrixDesc {"
tests = r'''    #[test]
    fn mixed_quant_rules_are_ordered_and_last_match_wins() {
        let mut request = RecompileRequest::new(
            PathBuf::from("old.coli"),
            PathBuf::from("new.coli"),
        );
        request.quant = QuantMode::Keep;
        request.quant_rules = vec![
            QuantRule::parse("layer:4-7=mxfp4").unwrap(),
            QuantRule::parse("expert:3=keep").unwrap(),
            QuantRule::parse("layer:6/expert:3=mxfp4").unwrap(),
        ];
        let record = RecordInfo {
            id: 1,
            kind: 2,
            codec: 0,
            math_format: 0,
            scale_format: 0,
            layout: 0,
            flags: 0,
            shard_id: 0,
            name: None,
            layer: 6,
            expert: 3,
            offset: 0,
            stored: 0,
            decoded: 0,
            stored_crc: 0,
            logical_crc: 0,
        };
        assert_eq!(effective_quant(&request, &record), QuantMode::Mxfp4);

        let mut other = record.clone();
        other.layer = 5;
        assert_eq!(effective_quant(&request, &other), QuantMode::Keep);

        other.expert = 8;
        assert_eq!(effective_quant(&request, &other), QuantMode::Mxfp4);
    }

    #[test]
    fn quant_rule_parser_rejects_reversed_and_unknown_selectors() {
        assert!(QuantRule::parse("layer:9-3=mxfp4").is_err());
        assert!(QuantRule::parse("dense=mxfp4").is_err());
        assert!(QuantRule::parse("layer:2=bogus").is_err());
    }

'''
text = replace_once(text, test_marker, tests + test_marker, "quant rule tests")
path.write_text(text)


# --- cli.rs ---------------------------------------------------------------
path = Path("logan-compiler/src/cli.rs")
text = path.read_text()
text = replace_once(
    text,
    "        CodecMode as RecompileCodecMode, QuantMode as RecompileQuantMode, RecompileRequest,\n",
    "        CodecMode as RecompileCodecMode, QuantMode as RecompileQuantMode, QuantRule, RecompileRequest,\n",
    "cli import",
)
text = replace_once(
    text,
    "  logan recompile PACKAGE_DIR -o OUTPUT [--target source|PROFILE] [--quant keep|mxfp4] [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]",
    "  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|PROFILE] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]",
    "cli usage",
)
parse_start = text.index("fn parse_recompile")
text = replace_once(
    text,
    "    let mut quant = RecompileQuantMode::Keep;\n    let mut codec = RecompileCodecMode::Keep;",
    "    let mut quant = RecompileQuantMode::Keep;\n    let mut quant_rules = Vec::new();\n    let mut codec = RecompileCodecMode::Keep;",
    "cli quant rule state",
)
text = replace_once(
    text,
    "    let mut verify = false;\n    let mut force = false;",
    "    let mut verify = false;\n    let mut force = false;\n    let mut in_place = false;",
    "cli in-place state",
)
quant_case = text.index('            "--quant" => {', parse_start)
codec_case = text.index('            "--codec" => {', quant_case)
quant_rule_case = '''            "--quant-rule" => {
                quant_rules.push(QuantRule::parse(&value(&mut args, "--quant-rule")?)?)
            }
'''
text = text[:codec_case] + quant_rule_case + text[codec_case:]
force_case = text.index('            "--force" => force = true,', codec_case)
force_end = text.index("\n", force_case) + 1
text = text[:force_end] + '            "--in-place" => in_place = true,\n' + text[force_end:]

output_start = text.index("    let output = output", parse_start)
function_end = text.index("\n}\n\n#[cfg(test)]", output_start)
new_tail = '''    if in_place && output.is_some() {
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
        allow_requantize,
        repack,
        verify,
        force: force || in_place,
    }))
'''
text = text[:output_start] + new_tail + text[function_end:]

cli_test_marker = "    #[test]\n    fn recompile_requires_an_output_argument() {"
cli_tests = r'''    #[test]
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
    fn recompile_rejects_output_with_in_place() {
        assert!(
            parse(
                ["recompile", "model.coli", "--in-place", "-o", "other.coli"]
                    .map(str::to_owned)
            )
            .is_err()
        );
    }

'''
text = replace_once(text, cli_test_marker, cli_tests + cli_test_marker, "cli tests")
path.write_text(text)
