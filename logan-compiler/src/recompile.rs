//! Offline COLI -> COLI recompilation.
//!
//! This path deliberately lives in the compiler rather than the runtime. It
//! can losslessly retarget MXFP4 experts between canonical row-major storage
//! and the Apple8 tile ABI, freshly quantize BF16 experts to MXFP4, and (only
//! with an explicit opt-in) requantize INT4-G32 experts to MXFP4. Unaffected
//! records are copied byte-for-byte and the original source-model fingerprint
//! is preserved.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use logan_format::{
    codecs::{self, INT4_MATH_FORMAT, INT4_SCALE_FORMAT, RANS_CODEC_ID, RANS_TABLE_ID, RansTable},
    package::{Package, RecordInfo},
};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{
    codec::rans256,
    error::{ColicError, Result},
    quant::mxfp4::{self, PackedMatrix},
    storage::{self, LoweredRecord, ManifestRecord, PlannedRecord, StoragePlan},
    target::{self, TargetProfile},
    target_registry,
};

const EXPERT_HEADER_BYTES: usize = 64;
const EXPERT_DESC_BYTES: usize = 128;
const EXPERT_MATRICES: usize = 3;
const EXPERT_DATA_OFFSET: usize = EXPERT_HEADER_BYTES + EXPERT_DESC_BYTES * EXPERT_MATRICES;
const DATA_ALIGNMENT: u64 = 16;
const MATH_BF16: u16 = 0x0003;
const MATH_MXFP4: u16 = 0x0020;
const SCALE_NONE: u16 = 0x0000;
const SCALE_E8M0: u16 = 0x0004;
const INPLACE_STATE_FILE: &str = ".logan-recompile-state.json";
const INPLACE_RECORD_META_FILE: &str = ".logan-recompile-record.json";
const INPLACE_RECORD_BACKUP_FILE: &str = ".logan-recompile-record.bin";
const INPLACE_MANIFEST_BACKUP_FILE: &str = ".logan-recompile-manifest.bak";
const INPLACE_FINAL_MANIFEST_BACKUP_FILE: &str = ".logan-recompile-final-manifest.bak";
const INPLACE_HEADERS_BACKUP_FILE: &str = ".logan-recompile-headers.bak";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Keep,
    Mxfp4,
}

impl QuantMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "keep" => Ok(Self::Keep),
            "mxfp4" => Ok(Self::Mxfp4),
            other => Err(ColicError::Usage(format!(
                "unknown recompile quant mode `{other}` (expected `keep` or `mxfp4`)"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Mxfp4 => "mxfp4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    let parsed = value
        .parse::<i32>()
        .map_err(|_| ColicError::Usage(format!("quant-rule {what} `{value}` is not an integer")))?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMode {
    /// Preserve compressed bytes for records that can be copied unchanged.
    /// Rewritten records are currently emitted raw.
    Keep,
    /// Rewrite supported compressed Apple8 experts as raw target-native bytes.
    None,
}

impl CodecMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "keep" => Ok(Self::Keep),
            "none" => Ok(Self::None),
            other => Err(ColicError::Usage(format!(
                "unknown recompile codec mode `{other}` (expected `keep` or `none`)"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecompileRequest {
    pub source: PathBuf,
    pub output: PathBuf,
    /// `source` preserves the package profile; otherwise this is an explicit
    /// registered target profile name. There is intentionally no env-var
    /// override or hidden native selection in this API.
    pub target: String,
    pub quant: QuantMode,
    /// Ordered routed-expert overrides. Later matching rules win.
    pub quant_rules: Vec<QuantRule>,
    pub codec: CodecMode,
    pub allow_requantize: bool,
    /// Force target-layout reconstruction even when the package already uses
    /// the requested representation.
    pub repack: bool,
    pub verify: bool,
    pub force: bool,
}

impl RecompileRequest {
    pub fn new(source: PathBuf, output: PathBuf) -> Self {
        Self {
            source,
            output,
            target: "source".into(),
            quant: QuantMode::Keep,
            quant_rules: Vec::new(),
            codec: CodecMode::Keep,
            allow_requantize: false,
            repack: false,
            verify: false,
            force: false,
        }
    }
}

fn effective_quant(request: &RecompileRequest, record: &RecordInfo) -> QuantMode {
    request
        .quant_rules
        .iter()
        .filter(|rule| rule.matches(record))
        .last()
        .map_or(request.quant, |rule| rule.mode)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecompileSummary {
    pub source_profile: String,
    pub target_profile: String,
    pub records: usize,
    pub copied_records: usize,
    pub rewritten_experts: usize,
    pub requantized_experts: usize,
    pub source_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertTarget {
    CanonicalMxfp4,
    Apple8Mxfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixKind {
    Bf16,
    CanonicalMxfp4,
    Apple8Mxfp4,
    Int4G32,
}

#[derive(Debug, Clone, Copy)]
struct MatrixDesc {
    role: u16,
    math: u16,
    scale: u16,
    weight_codec: u16,
    scale_codec: u16,
    layout: u16,
    rows: u32,
    cols: u32,
    weight_table: u32,
    scale_table: u32,
    weight_offset: u64,
    weight_stored: u64,
    weight_decoded: u64,
    scale_offset: u64,
    scale_stored: u64,
    scale_decoded: u64,
}

impl MatrixDesc {
    fn kind(self) -> Result<MatrixKind> {
        match (self.math, self.scale, self.layout) {
            (MATH_BF16, SCALE_NONE, 0) => Ok(MatrixKind::Bf16),
            (MATH_MXFP4, SCALE_E8M0, 0) => Ok(MatrixKind::CanonicalMxfp4),
            (MATH_MXFP4, SCALE_E8M0, target_registry::APPLE8_MXFP4_TILE_LAYOUT) => {
                Ok(MatrixKind::Apple8Mxfp4)
            }
            (INT4_MATH_FORMAT, INT4_SCALE_FORMAT, 0) => Ok(MatrixKind::Int4G32),
            _ => Err(ColicError::unsupported(
                "COLI recompilation",
                format!(
                    "unsupported expert matrix representation math=0x{:04x} scale=0x{:04x} layout=0x{:04x}",
                    self.math, self.scale, self.layout
                ),
            )),
        }
    }

    fn uses_rans(self) -> bool {
        self.weight_codec == RANS_CODEC_ID || self.scale_codec == RANS_CODEC_ID
    }
}

#[derive(Debug, Clone)]
enum ActionKind {
    Copy,
    Rewrite {
        descs: [MatrixDesc; EXPERT_MATRICES],
        target: ExpertTarget,
        requantized: bool,
    },
}

#[derive(Debug, Clone)]
struct Action {
    source: RecordInfo,
    kind: ActionKind,
    lowered: LoweredRecord,
    keeps_rans_table: bool,
}

pub fn recompile(request: &RecompileRequest) -> Result<RecompileSummary> {
    if request.source == request.output {
        if !request.force {
            return Err(ColicError::Usage(
                "recompiling in place requires --force; using a separate output path is safer"
                    .into(),
            ));
        }
        recover_in_place_if_needed(request)?;
    }

    let package = Package::open(&request.source)?;
    let target = resolve_target(&package, &request.target)?;
    let target_kind = if target.name == target_registry::APPLE8_PROFILE_NAME {
        ExpertTarget::Apple8Mxfp4
    } else {
        ExpertTarget::CanonicalMxfp4
    };

    let mut actions = Vec::with_capacity(package.records().len());
    let mut copied_records = 0_usize;
    let mut rewritten_experts = 0_usize;
    let mut requantized_experts = 0_usize;

    for record in package.records() {
        let action = plan_record(&package, record, request, target, target_kind)?;
        match action.kind {
            ActionKind::Copy => copied_records += 1,
            ActionKind::Rewrite { requantized, .. } => {
                rewritten_experts += 1;
                if requantized {
                    requantized_experts += 1;
                }
            }
        }
        actions.push(action);
    }

    let lowered = actions
        .iter()
        .map(|action| action.lowered.clone())
        .collect::<Vec<_>>();
    let fingerprint = *package.fingerprint();

    if request.source == request.output {
        // True low-space mode: preserve every record's existing physical slot
        // and rewrite only the selected records. This avoids materializing a
        // second package. A full space/layout preflight runs before any write.
        let (plan, shard_sizes) = plan_in_place(&package, &actions, target)?;
        ensure_in_place_state(request, package.profile(), target.name, plan.shards)?;
        write_package_in_place(
            &package,
            &actions,
            &plan,
            &shard_sizes,
            target,
            fingerprint,
            request,
        )?;
    } else {
        let plan = storage::plan_records(&lowered, target, 4 * 1024 * 1024 * 1024)?;
        let temporary = storage::temporary_package_path(&request.output)?;

        let write_result = write_package(
            &package,
            &actions,
            &plan,
            target,
            fingerprint,
            request,
            &temporary,
        );
        if let Err(error) = write_result {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }

        if request.verify {
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
    }

    Ok(RecompileSummary {
        source_profile: package.profile().to_owned(),
        target_profile: target.name.to_owned(),
        records: actions.len(),
        copied_records,
        rewritten_experts,
        requantized_experts,
        source_fingerprint: hex_fingerprint(package.fingerprint()),
    })
}

fn resolve_target(package: &Package, requested: &str) -> Result<TargetProfile> {
    let name = if requested == "source" {
        package.profile()
    } else {
        requested
    };
    let target = target::PROFILES
        .iter()
        .find(|profile| profile.name == name)
        .copied()
        .ok_or_else(|| ColicError::Usage(format!("unknown recompile target profile `{name}`")))?;
    if requested != "source" && !target.compiler_emission_supported {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            format!("target profile `{name}` does not support compiler emission"),
        ));
    }
    Ok(target)
}

fn plan_record(
    package: &Package,
    record: &RecordInfo,
    request: &RecompileRequest,
    target_profile: TargetProfile,
    target_kind: ExpertTarget,
) -> Result<Action> {
    if record.kind != 2 {
        if record.codec != 0 {
            return Err(ColicError::unsupported(
                "COLI recompilation",
                format!(
                    "record {} uses record-level codec {}; only expert-internal rANS is currently transcodable",
                    record.id, record.codec
                ),
            ));
        }
        return Ok(Action {
            source: record.clone(),
            kind: ActionKind::Copy,
            lowered: LoweredRecord {
                id: record.id,
                kind: record.kind,
                stored_bytes: record.stored,
                decoded_bytes: record.decoded,
            },
            keeps_rans_table: false,
        });
    }

    let descs = expert_descs(package, record)?;
    let kinds = [descs[0].kind()?, descs[1].kind()?, descs[2].kind()?];
    let all_mxfp4 = kinds
        .iter()
        .all(|kind| matches!(kind, MatrixKind::CanonicalMxfp4 | MatrixKind::Apple8Mxfp4));
    let quant = effective_quant(request, record);
    let source_quantized = kinds.iter().any(|kind| *kind == MatrixKind::Int4G32);
    let requantized = source_quantized && quant == QuantMode::Mxfp4;
    if requantized && !request.allow_requantize {
        return Err(ColicError::Usage(format!(
            "expert layer={} expert={} is already quantized INT4-G32; pass --allow-requantize to convert it to MXFP4",
            record.layer, record.expert
        )));
    }

    let target_changed = package.profile() != target_profile.name;
    let quantizes = quant == QuantMode::Mxfp4
        && kinds
            .iter()
            .any(|kind| !matches!(kind, MatrixKind::CanonicalMxfp4 | MatrixKind::Apple8Mxfp4));
    let layout_mismatch = all_mxfp4
        && kinds.iter().any(|kind| match target_kind {
            ExpertTarget::Apple8Mxfp4 => *kind != MatrixKind::Apple8Mxfp4,
            ExpertTarget::CanonicalMxfp4 => *kind != MatrixKind::CanonicalMxfp4,
        });
    let has_rans = descs.iter().any(|desc| desc.uses_rans());
    let strips_codec = request.codec == CodecMode::None && has_rans;
    let repacks = request.repack && all_mxfp4;

    if quant == QuantMode::Keep && (target_changed || layout_mismatch) && !all_mxfp4 {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            format!(
                "retargeting layer={} expert={} would change a non-MXFP4 representation; use --quant mxfp4{}",
                record.layer,
                record.expert,
                if source_quantized {
                    " --allow-requantize"
                } else {
                    ""
                }
            ),
        ));
    }

    if quant == QuantMode::Mxfp4 {
        for kind in kinds {
            if !matches!(
                kind,
                MatrixKind::Bf16
                    | MatrixKind::CanonicalMxfp4
                    | MatrixKind::Apple8Mxfp4
                    | MatrixKind::Int4G32
            ) {
                return Err(ColicError::unsupported(
                    "COLI recompilation",
                    "MXFP4 conversion does not support this source representation",
                ));
            }
        }
    }

    let rewrite =
        quantizes || layout_mismatch || strips_codec || repacks || (target_changed && all_mxfp4);
    if !rewrite {
        return Ok(Action {
            source: record.clone(),
            kind: ActionKind::Copy,
            lowered: LoweredRecord {
                id: record.id,
                kind: record.kind,
                stored_bytes: record.stored,
                decoded_bytes: record.decoded,
            },
            keeps_rans_table: has_rans,
        });
    }

    if quant == QuantMode::Keep && !all_mxfp4 {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            "requested transform requires MXFP4 but --quant keep forbids changing the mathematical format",
        ));
    }

    let (stored_bytes, decoded_bytes) = match target_kind {
        ExpertTarget::Apple8Mxfp4 => apple8_sizes(&descs)?,
        ExpertTarget::CanonicalMxfp4 => canonical_mxfp4_sizes(&descs)?,
    };
    Ok(Action {
        source: record.clone(),
        kind: ActionKind::Rewrite {
            descs,
            target: target_kind,
            requantized,
        },
        lowered: LoweredRecord {
            id: record.id,
            kind: record.kind,
            stored_bytes,
            decoded_bytes,
        },
        keeps_rans_table: false,
    })
}

/// Builds a storage plan that reuses the source package's shard IDs and record
/// offsets. The entire transform is rejected before mutation if any target
/// record would exceed the physical slot currently available to it.
fn plan_in_place(
    package: &Package,
    actions: &[Action],
    target: TargetProfile,
) -> Result<(StoragePlan, Vec<u64>)> {
    let shard_count = u32_at(package.manifest_ref(), 40)?;
    if shard_count == 0 {
        return Err(ColicError::Usage("COLI package has no data shards".into()));
    }
    let mut shard_sizes = Vec::with_capacity(shard_count as usize);
    for shard_id in 0..shard_count {
        let path = package
            .shard_path(shard_id)
            .ok_or_else(|| ColicError::Usage("source shard id is invalid".into()))?;
        let path = PathBuf::from(path);
        let bytes = fs::metadata(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .len();
        shard_sizes.push(bytes);
    }

    if actions.len() != package.records().len() {
        return Err(ColicError::Usage(
            "in-place action count does not match package records".into(),
        ));
    }

    let mut planned = Vec::with_capacity(actions.len());
    let mut projected_stored_bytes = 0_u64;
    for action in actions {
        let source = &action.source;
        let shard_size = *shard_sizes
            .get(source.shard_id as usize)
            .ok_or_else(|| ColicError::Usage("record refers to an invalid source shard".into()))?;
        if source.offset % target.record_alignment != 0 {
            return Err(ColicError::Usage(format!(
                "record {} offset {} is not aligned for target {} ({} bytes); low-space --in-place cannot relocate records",
                source.id, source.offset, target.name, target.record_alignment
            )));
        }
        let slot_end = package
            .records()
            .iter()
            .filter(|other| other.shard_id == source.shard_id && other.offset > source.offset)
            .map(|other| other.offset)
            .min()
            .unwrap_or(shard_size);
        if slot_end < source.offset {
            return Err(ColicError::Usage(format!(
                "record {} has an invalid physical slot",
                source.id
            )));
        }
        let capacity = slot_end - source.offset;
        if action.lowered.stored_bytes > capacity {
            return Err(ColicError::Usage(format!(
                "record {} (layer={} expert={}) needs {} bytes but its in-place slot holds {} ({} bytes short); use a separate output path or choose a non-growing transform",
                source.id,
                source.layer,
                source.expert,
                action.lowered.stored_bytes,
                capacity,
                action.lowered.stored_bytes - capacity
            )));
        }
        projected_stored_bytes = projected_stored_bytes
            .checked_add(action.lowered.stored_bytes)
            .ok_or_else(|| ColicError::Usage("in-place stored-byte total overflows u64".into()))?;
        planned.push(PlannedRecord {
            record: action.lowered.clone(),
            shard_id: source.shard_id,
            payload_offset: source.offset,
        });
    }

    let shard_size_limit = shard_sizes.iter().copied().max().unwrap_or(0);
    Ok((
        StoragePlan {
            record_alignment: target.record_alignment,
            shard_size_limit,
            shards: shard_count,
            records: planned,
            projected_stored_bytes,
            // Existing gaps are deliberately retained in low-space mode.
            projected_padding_bytes: 0,
        },
        shard_sizes,
    ))
}

/// Rewrites a package without creating a second package directory. Record
/// offsets and shard lengths stay fixed; only rewritten payload bytes, shard
/// headers, the manifest, and provenance are replaced.
fn write_package_in_place(
    package: &Package,
    actions: &[Action],
    plan: &StoragePlan,
    shard_sizes: &[u64],
    target: TargetProfile,
    fingerprint: [u8; 32],
    request: &RecompileRequest,
) -> Result<()> {
    if actions.len() != plan.records.len() {
        return Err(ColicError::Usage(
            "in-place action/plan order mismatch".into(),
        ));
    }

    let source_alignment = u64::from(u32_at(package.manifest_ref(), 24)?);
    let source_header_crcs = read_shard_header_crcs(package, plan.shards)?;
    let source_rans_table = source_rans_table(package)?;
    let mut checkpoint_plan = checkpoint_plan_from_package(package, shard_sizes)?;
    let mut metadata = actions.iter().map(copy_manifest_record).collect::<Vec<_>>();

    for (index, (action, planned)) in actions.iter().zip(&plan.records).enumerate() {
        if planned.shard_id != action.source.shard_id
            || planned.payload_offset != action.source.offset
        {
            return Err(ColicError::Usage(
                "low-space in-place plan attempted to relocate a record".into(),
            ));
        }
        let ActionKind::Rewrite {
            descs,
            target: expert_target,
            ..
        } = &action.kind
        else {
            continue;
        };

        // Build the complete replacement from the still-valid source record
        // before touching disk. Peak scratch is one expert payload in memory.
        let packed = [
            matrix_to_mxfp4(package, &action.source, descs[0], request.allow_requantize)?,
            matrix_to_mxfp4(package, &action.source, descs[1], request.allow_requantize)?,
            matrix_to_mxfp4(package, &action.source, descs[2], request.allow_requantize)?,
        ];
        let payload = match expert_target {
            ExpertTarget::Apple8Mxfp4 => build_apple8_expert(
                action.source.layer,
                action.source.expert,
                [&packed[0], &packed[1], &packed[2]],
            )?,
            ExpertTarget::CanonicalMxfp4 => build_canonical_mxfp4_expert(
                action.source.layer,
                action.source.expert,
                [&packed[0], &packed[1], &packed[2]],
            )?,
        };
        if payload.len() as u64 != planned.record.stored_bytes {
            return Err(ColicError::Usage(format!(
                "rewritten expert {}:{} produced {} bytes, planned {}",
                action.source.layer,
                action.source.expert,
                payload.len(),
                planned.record.stored_bytes
            )));
        }

        // Persist only the current source record plus the current manifest.
        // If the process dies anywhere before the new checkpoint manifest is
        // committed, the next `logan recompile --in-place` restores these few
        // bytes and resumes from the last structurally-valid checkpoint.
        begin_record_journal(&request.source, package, &action.source)?;
        let mutation = (|| -> Result<()> {
            write_record_payload(
                &request.source,
                planned.shard_id,
                planned.payload_offset,
                &payload,
            )?;
            checkpoint_plan.records[index].record = planned.record.clone();
            metadata[index] = rewritten_manifest_record(action, &payload);
            write_checkpoint_manifest(
                package,
                &checkpoint_plan,
                shard_sizes,
                source_alignment,
                &metadata,
                &source_header_crcs,
                source_rans_table.as_ref(),
                fingerprint,
                &request.source,
            )
        })();
        if let Err(error) = mutation {
            if let Err(recovery) = recover_record_journal(&request.source) {
                return Err(ColicError::Usage(format!(
                    "in-place rewrite failed: {error}; record rollback also failed: {recovery}"
                )));
            }
            return Err(error);
        }
        clear_record_journal(&request.source)?;
    }

    // Final target headers and profile are a tiny second transaction. Back up
    // only the current manifest and 128-byte shard headers so a crash while
    // flipping the package to the new target can roll back to the last hybrid
    // checkpoint without a model-sized copy.
    begin_finalization_journal(&request.source, plan.shards)?;
    set_in_place_phase(&request.source, "finalizing")?;
    let finalization = (|| -> Result<()> {
        let mut header_crcs = Vec::with_capacity(plan.shards as usize);
        for shard_id in 0..plan.shards {
            let file_bytes = *shard_sizes
                .get(shard_id as usize)
                .ok_or_else(|| ColicError::Usage("missing source shard size".into()))?;
            let header = storage::encode_data_shard_header(
                shard_id,
                file_bytes,
                plan.record_alignment,
                fingerprint,
            )?;
            write_shard_header(&request.source, shard_id, &header)?;
            header_crcs.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
        }

        let mut manifest = storage::encode_manifest_with_records(
            plan,
            target.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;
        if actions.iter().any(|action| action.keeps_rans_table) {
            let table =
                rans256::table_from_manifest(package.manifest_ref(), RANS_TABLE_ID, RANS_CODEC_ID)?;
            manifest = rans256::manifest_table_region(manifest, Some(&table))?;
        }
        patch_manifest_shard_sizes(&mut manifest, shard_sizes)?;
        replace_manifest(&request.source, &manifest)?;
        crate::verify::verify_package(&request.output)?;
        crate::verify_target::verify_target_layouts(&request.output)?;
        write_provenance(package, target, actions, request, &request.output)
    })();
    if let Err(error) = finalization {
        if let Err(recovery) = rollback_finalization(&request.source) {
            return Err(ColicError::Usage(format!(
                "in-place finalization failed: {error}; finalization rollback also failed: {recovery}"
            )));
        }
        return Err(error);
    }

    set_in_place_phase(&request.source, "committed")?;
    clear_finalization_journal(&request.source)?;
    clear_record_journal(&request.source)?;
    remove_if_exists(&request.source.join(INPLACE_STATE_FILE))?;
    Ok(())
}

fn request_signature(request: &RecompileRequest) -> String {
    serde_json::to_string(&json!({
        "target": request.target,
        "quant": request.quant.as_str(),
        "quant_rules": request.quant_rules.iter().map(|rule| rule.as_spec()).collect::<Vec<_>>(),
        "codec": request.codec.as_str(),
        "allow_requantize": request.allow_requantize,
        "repack": request.repack,
    }))
    .expect("recompile request signature is always JSON-serializable")
}

fn ensure_in_place_state(
    request: &RecompileRequest,
    source_profile: &str,
    target_profile: &str,
    shards: u32,
) -> Result<()> {
    let path = request.source.join(INPLACE_STATE_FILE);
    if path.exists() {
        let state = read_json_file(&path)?;
        ensure_state_signature(&state, request)?;
        return Ok(());
    }
    let state = json!({
        "version": 1,
        "phase": "records",
        "request": request_signature(request),
        "source_profile": source_profile,
        "target_profile": target_profile,
        "shards": shards,
    });
    write_json_synced(&path, &state)
}

fn recover_in_place_if_needed(request: &RecompileRequest) -> Result<()> {
    let state_path = request.source.join(INPLACE_STATE_FILE);
    if !state_path.exists() {
        // A crash before the record-journal marker is durable can leave only
        // orphan backup files. No model bytes were touched in that state.
        clear_record_journal(&request.source)?;
        clear_finalization_journal(&request.source)?;
        return Ok(());
    }
    let mut state = read_json_file(&state_path)?;
    let phase = state
        .get("phase")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ColicError::Usage("in-place recompile state has no phase".into()))?
        .to_owned();

    match phase.as_str() {
        "records" => recover_record_journal(&request.source)?,
        "finalizing" => {
            recover_record_journal(&request.source)?;
            rollback_finalization(&request.source)?;
            state = read_json_file(&state_path)?;
        }
        "committed" => {
            crate::verify::verify_package(&request.source)?;
            crate::verify_target::verify_target_layouts(&request.source)?;
            clear_record_journal(&request.source)?;
            clear_finalization_journal(&request.source)?;
            remove_if_exists(&state_path)?;
            return Ok(());
        }
        other => {
            return Err(ColicError::Usage(format!(
                "unknown in-place recompile recovery phase `{other}`"
            )));
        }
    }
    ensure_state_signature(&state, request)
}

fn ensure_state_signature(state: &serde_json::Value, request: &RecompileRequest) -> Result<()> {
    let stored = state
        .get("request")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            ColicError::Usage("in-place recompile state has no request signature".into())
        })?;
    let current = request_signature(request);
    if stored != current {
        return Err(ColicError::Usage(
            "an unfinished low-space in-place recompile exists; rerun with the same target/quant/codec options to resume it"
                .into(),
        ));
    }
    Ok(())
}

fn set_in_place_phase(root: &Path, phase: &str) -> Result<()> {
    let path = root.join(INPLACE_STATE_FILE);
    let mut state = read_json_file(&path)?;
    state["phase"] = serde_json::Value::String(phase.to_owned());
    write_json_synced(&path, &state)
}

fn begin_record_journal(root: &Path, package: &Package, record: &RecordInfo) -> Result<()> {
    clear_record_journal(root)?;
    let payload = package.read_record(record)?;
    let manifest = fs::read(root.join("manifest.coli")).map_err(|source| ColicError::Io {
        path: root.join("manifest.coli"),
        source,
    })?;
    write_synced(&root.join(INPLACE_RECORD_BACKUP_FILE), &payload)?;
    write_synced(&root.join(INPLACE_MANIFEST_BACKUP_FILE), &manifest)?;
    // The metadata marker is durable last. Until it exists no data shard is
    // modified, so orphan backup files can simply be discarded on recovery.
    write_json_synced(
        &root.join(INPLACE_RECORD_META_FILE),
        &json!({
            "version": 1,
            "shard": record.shard_id,
            "offset": record.offset,
            "bytes": record.stored,
        }),
    )
}

fn recover_record_journal(root: &Path) -> Result<()> {
    let meta_path = root.join(INPLACE_RECORD_META_FILE);
    if !meta_path.exists() {
        remove_if_exists(&root.join(INPLACE_RECORD_BACKUP_FILE))?;
        remove_if_exists(&root.join(INPLACE_MANIFEST_BACKUP_FILE))?;
        return Ok(());
    }
    let meta = read_json_file(&meta_path)?;
    let shard = json_u64(&meta, "shard")?;
    let shard: u32 = shard
        .try_into()
        .map_err(|_| ColicError::Usage("record journal shard id exceeds u32".into()))?;
    let offset = json_u64(&meta, "offset")?;
    let bytes = json_u64(&meta, "bytes")?;
    let backup_path = root.join(INPLACE_RECORD_BACKUP_FILE);
    let backup = fs::read(&backup_path).map_err(|source| ColicError::Io {
        path: backup_path.clone(),
        source,
    })?;
    if backup.len() as u64 != bytes {
        return Err(ColicError::Usage(
            "record journal payload length does not match its metadata".into(),
        ));
    }
    write_record_payload(root, shard, offset, &backup)?;
    let manifest_backup_path = root.join(INPLACE_MANIFEST_BACKUP_FILE);
    let manifest = fs::read(&manifest_backup_path).map_err(|source| ColicError::Io {
        path: manifest_backup_path.clone(),
        source,
    })?;
    replace_manifest(root, &manifest)?;
    clear_record_journal(root)
}

fn clear_record_journal(root: &Path) -> Result<()> {
    remove_if_exists(&root.join(INPLACE_RECORD_META_FILE))?;
    remove_if_exists(&root.join(INPLACE_RECORD_BACKUP_FILE))?;
    remove_if_exists(&root.join(INPLACE_MANIFEST_BACKUP_FILE))
}

fn begin_finalization_journal(root: &Path, shards: u32) -> Result<()> {
    clear_finalization_journal(root)?;
    let manifest_path = root.join("manifest.coli");
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path,
        source,
    })?;
    write_synced(&root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE), &manifest)?;
    let mut headers =
        Vec::with_capacity(shards as usize * storage::DATA_SHARD_HEADER_BYTES as usize);
    for shard_id in 0..shards {
        let path = root.join(format!("data-{shard_id:05}.coli"));
        let mut header = vec![0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io { path, source })?;
        headers.extend_from_slice(&header);
    }
    write_synced(&root.join(INPLACE_HEADERS_BACKUP_FILE), &headers)
}

fn rollback_finalization(root: &Path) -> Result<()> {
    let state = read_json_file(&root.join(INPLACE_STATE_FILE))?;
    let shards: u32 = json_u64(&state, "shards")?
        .try_into()
        .map_err(|_| ColicError::Usage("in-place state shard count exceeds u32".into()))?;
    let headers_path = root.join(INPLACE_HEADERS_BACKUP_FILE);
    let headers = fs::read(&headers_path).map_err(|source| ColicError::Io {
        path: headers_path.clone(),
        source,
    })?;
    let header_bytes = storage::DATA_SHARD_HEADER_BYTES as usize;
    if headers.len() != shards as usize * header_bytes {
        return Err(ColicError::Usage(
            "finalization header backup has the wrong length".into(),
        ));
    }
    for shard_id in 0..shards {
        let start = shard_id as usize * header_bytes;
        let header: [u8; storage::DATA_SHARD_HEADER_BYTES as usize] =
            headers[start..start + header_bytes].try_into().unwrap();
        write_shard_header(root, shard_id, &header)?;
    }
    let manifest_path = root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE);
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path.clone(),
        source,
    })?;
    replace_manifest(root, &manifest)?;
    set_in_place_phase(root, "records")?;
    clear_finalization_journal(root)
}

fn clear_finalization_journal(root: &Path) -> Result<()> {
    remove_if_exists(&root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE))?;
    remove_if_exists(&root.join(INPLACE_HEADERS_BACKUP_FILE))
}

fn checkpoint_plan_from_package(package: &Package, shard_sizes: &[u64]) -> Result<StoragePlan> {
    let records = package
        .records()
        .iter()
        .map(|record| PlannedRecord {
            record: LoweredRecord {
                id: record.id,
                kind: record.kind,
                stored_bytes: record.stored,
                decoded_bytes: record.decoded,
            },
            shard_id: record.shard_id,
            payload_offset: record.offset,
        })
        .collect::<Vec<_>>();
    let projected_stored_bytes = records.iter().try_fold(0_u64, |total, record| {
        total
            .checked_add(record.record.stored_bytes)
            .ok_or_else(|| ColicError::Usage("checkpoint stored-byte total overflows u64".into()))
    })?;
    Ok(StoragePlan {
        record_alignment: u64::from(u32_at(package.manifest_ref(), 24)?),
        shard_size_limit: shard_sizes.iter().copied().max().unwrap_or(0),
        shards: u32_at(package.manifest_ref(), 40)?,
        records,
        projected_stored_bytes,
        projected_padding_bytes: 0,
    })
}

fn read_shard_header_crcs(package: &Package, shards: u32) -> Result<Vec<u32>> {
    let mut result = Vec::with_capacity(shards as usize);
    for shard_id in 0..shards {
        let path = PathBuf::from(
            package
                .shard_path(shard_id)
                .ok_or_else(|| ColicError::Usage("source shard id is invalid".into()))?,
        );
        let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io { path, source })?;
        result.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
    }
    Ok(result)
}

fn source_rans_table(package: &Package) -> Result<Option<rans256::Table>> {
    if u32_at(package.manifest_ref(), 160)? == 0 {
        return Ok(None);
    }
    Ok(Some(rans256::table_from_manifest(
        package.manifest_ref(),
        RANS_TABLE_ID,
        RANS_CODEC_ID,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn write_checkpoint_manifest(
    package: &Package,
    plan: &StoragePlan,
    shard_sizes: &[u64],
    source_alignment: u64,
    metadata: &[ManifestRecord],
    header_crcs: &[u32],
    rans_table: Option<&rans256::Table>,
    fingerprint: [u8; 32],
    root: &Path,
) -> Result<()> {
    let mut checkpoint_plan = plan.clone();
    checkpoint_plan.record_alignment = source_alignment;
    let mut manifest = storage::encode_manifest_with_records(
        &checkpoint_plan,
        package.profile(),
        fingerprint,
        metadata,
        header_crcs,
    )?;
    if let Some(table) = rans_table {
        manifest = rans256::manifest_table_region(manifest, Some(table))?;
    }
    patch_manifest_shard_sizes(&mut manifest, shard_sizes)?;
    replace_manifest(root, &manifest)
}

fn write_record_payload(root: &Path, shard_id: u32, offset: u64, payload: &[u8]) -> Result<()> {
    let path = root.join(format!("data-{shard_id:05}.coli"));
    let mut shard = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| ColicError::Io {
            path: path.clone(),
            source,
        })?;
    shard
        .seek(SeekFrom::Start(offset))
        .map_err(|source| ColicError::Io {
            path: path.clone(),
            source,
        })?;
    shard.write_all(payload).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    shard
        .sync_data()
        .map_err(|source| ColicError::Io { path, source })
}

fn write_shard_header(
    root: &Path,
    shard_id: u32,
    header: &[u8; storage::DATA_SHARD_HEADER_BYTES as usize],
) -> Result<()> {
    write_record_payload(root, shard_id, 0, header)
}

fn replace_manifest(root: &Path, manifest: &[u8]) -> Result<()> {
    let path = root.join("manifest.coli");
    let next = root.join("manifest.coli.recompile-next");
    write_synced(&next, manifest)?;
    #[cfg(not(windows))]
    fs::rename(&next, &path).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    #[cfg(windows)]
    {
        fs::copy(&next, &path).map_err(|source| ColicError::Io {
            path: path.clone(),
            source,
        })?;
        File::open(&path)
            .and_then(|file| file.sync_all())
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?;
        remove_if_exists(&next)?;
    }
    Ok(())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.write_all(bytes).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.sync_all().map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })
}

fn write_json_synced(path: &Path, value: &serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| ColicError::Usage(format!("cannot encode recompile state: {error}")))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ColicError::Usage("recompile state path has no file name".into()))?;
    let next = path.with_file_name(format!("{name}.next"));
    write_synced(&next, &bytes)?;
    #[cfg(not(windows))]
    fs::rename(&next, path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    #[cfg(windows)]
    {
        fs::copy(&next, path).map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
        File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(|source| ColicError::Io {
                path: path.to_owned(),
                source,
            })?;
        remove_if_exists(&next)?;
    }
    Ok(())
}

fn read_json_file(path: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ColicError::Usage(format!("invalid recompile state JSON: {error}")))
}

fn json_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ColicError::Usage(format!("recompile state field `{field}` is invalid")))
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ColicError::Io {
            path: path.to_owned(),
            source,
        }),
    }
}

fn copy_manifest_record(action: &Action) -> ManifestRecord {
    ManifestRecord {
        id: action.source.id,
        name: action.source.name.clone(),
        layer: action.source.layer,
        expert: action.source.expert,
        kind: action.source.kind,
        codec: action.source.codec,
        math_format: action.source.math_format,
        scale_format: action.source.scale_format,
        layout: action.source.layout,
        flags: action.source.flags,
        stored_crc32c: action.source.stored_crc,
        logical_crc32c: action.source.logical_crc,
        codec_table_id: 0,
    }
}

fn rewritten_manifest_record(action: &Action, payload: &[u8]) -> ManifestRecord {
    ManifestRecord {
        id: action.source.id,
        name: action.source.name.clone(),
        layer: action.source.layer,
        expert: action.source.expert,
        kind: 2,
        codec: 0,
        math_format: 0xfffe,
        scale_format: 0xfffe,
        layout: 0xfffe,
        flags: 0,
        stored_crc32c: storage::crc32c(payload),
        logical_crc32c: 0,
        codec_table_id: 0,
    }
}

fn patch_manifest_shard_sizes(manifest: &mut [u8], shard_sizes: &[u64]) -> Result<()> {
    let shard_count = u32_at(manifest, 40)? as usize;
    if shard_count != shard_sizes.len() {
        return Err(ColicError::Usage(
            "in-place shard-size count does not match manifest".into(),
        ));
    }
    let table = usize::try_from(u64_at(manifest, 48)?)
        .map_err(|_| ColicError::Usage("manifest shard-table offset exceeds usize".into()))?;
    let table_bytes = usize::try_from(u64_at(manifest, 56)?)
        .map_err(|_| ColicError::Usage("manifest shard-table size exceeds usize".into()))?;
    if table
        .checked_add(table_bytes)
        .is_none_or(|end| end > manifest.len())
        || table_bytes != shard_count * 64
    {
        return Err(ColicError::Usage("manifest shard table is invalid".into()));
    }
    for (shard_id, file_bytes) in shard_sizes.iter().copied().enumerate() {
        put_u64(manifest, table + shard_id * 64 + 16, file_bytes);
    }
    manifest[144..148].fill(0);
    let crc = storage::crc32c(manifest);
    put_u32(manifest, 144, crc);
    Ok(())
}

fn write_package(
    package: &Package,
    actions: &[Action],
    plan: &storage::StoragePlan,
    target: TargetProfile,
    fingerprint: [u8; 32],
    request: &RecompileRequest,
    temporary: &Path,
) -> Result<()> {
    let mut metadata = Vec::with_capacity(actions.len());
    let mut header_crcs = Vec::with_capacity(plan.shards as usize);

    for shard_id in 0..plan.shards {
        let path = temporary.join(format!("data-{shard_id:05}.coli"));
        let mut writer =
            storage::DataShardWriter::create(&path, shard_id, plan.record_alignment, fingerprint)?;
        for (index, planned) in plan
            .records
            .iter()
            .enumerate()
            .filter(|(_, planned)| planned.shard_id == shard_id)
        {
            let action = actions
                .get(index)
                .ok_or_else(|| ColicError::Usage("recompile action/plan order mismatch".into()))?;
            metadata.push(write_action(
                package,
                &mut writer,
                planned,
                action,
                request,
            )?);
        }
        writer.finish()?;
        let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?;
        header_crcs.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
    }

    let mut manifest = storage::encode_manifest_with_records(
        plan,
        target.name,
        fingerprint,
        &metadata,
        &header_crcs,
    )?;
    if actions.iter().any(|action| action.keeps_rans_table) {
        let table =
            rans256::table_from_manifest(package.manifest_ref(), RANS_TABLE_ID, RANS_CODEC_ID)?;
        manifest = rans256::manifest_table_region(manifest, Some(&table))?;
    }
    let manifest_path = temporary.join("manifest.coli");
    fs::write(&manifest_path, manifest).map_err(|source| ColicError::Io {
        path: manifest_path,
        source,
    })?;
    copy_json_metadata(&request.source, temporary)?;
    write_provenance(package, target, actions, request, temporary)?;
    Ok(())
}

fn write_action(
    package: &Package,
    writer: &mut storage::DataShardWriter,
    planned: &PlannedRecord,
    action: &Action,
    request: &RecompileRequest,
) -> Result<ManifestRecord> {
    match &action.kind {
        ActionKind::Copy => {
            let shard = package
                .shard_path(action.source.shard_id)
                .ok_or_else(|| ColicError::Usage("source shard id is invalid".into()))?;
            let shard = PathBuf::from(shard);
            let offset = action.source.offset;
            let bytes = action.source.stored;
            writer.write_record_stream(planned, |output| {
                let mut input = File::open(&shard).map_err(|source| ColicError::Io {
                    path: shard.clone(),
                    source,
                })?;
                input
                    .seek(SeekFrom::Start(offset))
                    .map_err(|source| ColicError::Io {
                        path: shard.clone(),
                        source,
                    })?;
                let copied =
                    io::copy(&mut input.take(bytes), output).map_err(|source| ColicError::Io {
                        path: shard.clone(),
                        source,
                    })?;
                Ok(copied)
            })?;
            Ok(copy_manifest_record(action))
        }
        ActionKind::Rewrite { descs, target, .. } => {
            let packed = [
                matrix_to_mxfp4(package, &action.source, descs[0], request.allow_requantize)?,
                matrix_to_mxfp4(package, &action.source, descs[1], request.allow_requantize)?,
                matrix_to_mxfp4(package, &action.source, descs[2], request.allow_requantize)?,
            ];
            let payload = match target {
                ExpertTarget::Apple8Mxfp4 => build_apple8_expert(
                    action.source.layer,
                    action.source.expert,
                    [&packed[0], &packed[1], &packed[2]],
                )?,
                ExpertTarget::CanonicalMxfp4 => build_canonical_mxfp4_expert(
                    action.source.layer,
                    action.source.expert,
                    [&packed[0], &packed[1], &packed[2]],
                )?,
            };
            if payload.len() as u64 != planned.record.stored_bytes {
                return Err(ColicError::Usage(format!(
                    "rewritten expert {}:{} produced {} bytes, planned {}",
                    action.source.layer,
                    action.source.expert,
                    payload.len(),
                    planned.record.stored_bytes
                )));
            }
            writer.write_record(planned, &payload)?;
            Ok(rewritten_manifest_record(action, &payload))
        }
    }
}

fn expert_descs(package: &Package, record: &RecordInfo) -> Result<[MatrixDesc; EXPERT_MATRICES]> {
    let head = package.read_payload_range(record, 0, 32)?;
    if head.get(..8) != Some(b"COLIEXPT") || u16_at(&head, 8)? != 1 || u32_at(&head, 12)? != 64 {
        return Err(ColicError::Usage(format!(
            "expert record {} has an invalid COLIEXPT header",
            record.id
        )));
    }
    if u16_at(&head, 24)? as usize != EXPERT_MATRICES {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            "only three-matrix routed expert records are currently supported",
        ));
    }
    let desc_size = u32_at(&head, 28)? as usize;
    if desc_size != EXPERT_DESC_BYTES {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            format!("expert descriptor size {desc_size} is not the v1 128-byte descriptor"),
        ));
    }
    let prefix = package.read_payload_range(
        record,
        0,
        EXPERT_HEADER_BYTES + EXPERT_MATRICES * EXPERT_DESC_BYTES,
    )?;
    let mut result = Vec::with_capacity(EXPERT_MATRICES);
    for index in 0..EXPERT_MATRICES {
        let d = EXPERT_HEADER_BYTES + index * EXPERT_DESC_BYTES;
        let rows = u64_at(&prefix, d + 16)?;
        let cols = u64_at(&prefix, d + 24)?;
        result.push(MatrixDesc {
            role: u16_at(&prefix, d)?,
            math: u16_at(&prefix, d + 4)?,
            scale: u16_at(&prefix, d + 6)?,
            weight_codec: u16_at(&prefix, d + 8)?,
            scale_codec: u16_at(&prefix, d + 10)?,
            layout: u16_at(&prefix, d + 12)?,
            rows: rows
                .try_into()
                .map_err(|_| ColicError::Usage("matrix rows exceed u32".into()))?,
            cols: cols
                .try_into()
                .map_err(|_| ColicError::Usage("matrix columns exceed u32".into()))?,
            weight_table: u32_at(&prefix, d + 40)?,
            scale_table: u32_at(&prefix, d + 44)?,
            weight_offset: u64_at(&prefix, d + 48)?,
            weight_stored: u64_at(&prefix, d + 56)?,
            weight_decoded: u64_at(&prefix, d + 64)?,
            scale_offset: u64_at(&prefix, d + 72)?,
            scale_stored: u64_at(&prefix, d + 80)?,
            scale_decoded: u64_at(&prefix, d + 88)?,
        });
    }
    let result: [MatrixDesc; EXPERT_MATRICES] = result.try_into().unwrap();
    for (index, desc) in result.iter().enumerate() {
        if desc.role != (index + 1) as u16 {
            return Err(ColicError::Usage(format!(
                "expert record {} matrix {} has unexpected role {}",
                record.id, index, desc.role
            )));
        }
        validate_component(record, desc.weight_offset, desc.weight_stored)?;
        if desc.scale_stored != 0 {
            validate_component(record, desc.scale_offset, desc.scale_stored)?;
        }
    }
    Ok(result)
}

fn validate_component(record: &RecordInfo, offset: u64, bytes: u64) -> Result<()> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > record.stored)
    {
        return Err(ColicError::Usage(format!(
            "record {} contains an out-of-range expert component",
            record.id
        )));
    }
    Ok(())
}

fn matrix_to_mxfp4(
    package: &Package,
    record: &RecordInfo,
    desc: MatrixDesc,
    allow_requantize: bool,
) -> Result<PackedMatrix> {
    match desc.kind()? {
        MatrixKind::CanonicalMxfp4 => read_canonical_mxfp4(package, record, desc),
        MatrixKind::Apple8Mxfp4 => read_apple8_mxfp4(package, record, desc),
        MatrixKind::Bf16 => quantize_bf16(package, record, desc),
        MatrixKind::Int4G32 if allow_requantize => requantize_int4(package, record, desc),
        MatrixKind::Int4G32 => Err(ColicError::Usage(
            "INT4-G32 -> MXFP4 requires --allow-requantize".into(),
        )),
    }
}

fn read_canonical_mxfp4(
    package: &Package,
    record: &RecordInfo,
    desc: MatrixDesc,
) -> Result<PackedMatrix> {
    if desc.weight_codec != 0 || desc.scale_codec != 0 {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            "canonical MXFP4 with an inner codec is not currently supported",
        ));
    }
    let weights = read_component(package, record, desc.weight_offset, desc.weight_stored)?;
    let scales = read_component(package, record, desc.scale_offset, desc.scale_stored)?;
    validate_mxfp4_lengths(desc.rows, desc.cols, &weights, &scales)?;
    Ok(PackedMatrix {
        rows: desc.rows,
        columns: desc.cols,
        weights,
        scales,
    })
}

fn read_apple8_mxfp4(
    package: &Package,
    record: &RecordInfo,
    desc: MatrixDesc,
) -> Result<PackedMatrix> {
    if desc.scale_codec != 0 || desc.scale_stored != 0 || desc.scale_decoded != 0 {
        return Err(ColicError::Usage(
            "Apple8 MXFP4 must embed scales in the tile payload".into(),
        ));
    }
    let stored = read_component(package, record, desc.weight_offset, desc.weight_stored)?;
    let expected = target::apple8_tile_bytes(desc.rows, desc.cols)?;
    let tiles = match desc.weight_codec {
        0 => {
            if desc.weight_stored != expected || desc.weight_decoded != expected {
                return Err(ColicError::Usage(
                    "raw Apple8 matrix length does not match its geometry".into(),
                ));
            }
            stored
        }
        RANS_CODEC_ID => {
            if desc.weight_table != RANS_TABLE_ID {
                return Err(ColicError::unsupported(
                    "COLI recompilation",
                    format!("unsupported rANS table id {}", desc.weight_table),
                ));
            }
            let table = RansTable::from_manifest(
                package.manifest_ref(),
                desc.weight_table,
                desc.weight_codec,
            )?;
            let decoded =
                codecs::apple8_decode(&stored, &table, u64::from(desc.rows), u64::from(desc.cols))?;
            if decoded.len() as u64 != expected || desc.weight_decoded != expected {
                return Err(ColicError::Usage(
                    "decoded Apple8 matrix length does not match its geometry".into(),
                ));
            }
            decoded
        }
        other => {
            return Err(ColicError::unsupported(
                "COLI recompilation",
                format!("unsupported Apple8 weight codec {other}"),
            ));
        }
    };
    detile_apple8(desc.rows, desc.cols, &tiles)
}

fn quantize_bf16(package: &Package, record: &RecordInfo, desc: MatrixDesc) -> Result<PackedMatrix> {
    if desc.weight_codec != 0 || desc.scale_stored != 0 || desc.scale_codec != 0 {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            "BF16 expert quantization requires raw unscaled BF16 source matrices",
        ));
    }
    let source = read_component(package, record, desc.weight_offset, desc.weight_stored)?;
    let row_bytes = usize::try_from(u64::from(desc.cols) * 2)
        .map_err(|_| ColicError::Usage("BF16 row exceeds usize".into()))?;
    let expected = (desc.rows as usize)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("BF16 matrix size overflows usize".into()))?;
    if source.len() != expected {
        return Err(ColicError::Usage(
            "BF16 expert matrix length does not match geometry".into(),
        ));
    }
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    for row in source.chunks_exact(row_bytes) {
        mxfp4::quantize_bf16_row(row, &mut weights, &mut scales)?;
    }
    Ok(PackedMatrix {
        rows: desc.rows,
        columns: desc.cols,
        weights,
        scales,
    })
}

fn requantize_int4(
    package: &Package,
    record: &RecordInfo,
    desc: MatrixDesc,
) -> Result<PackedMatrix> {
    if desc.weight_codec != 0 || desc.scale_codec != 0 {
        return Err(ColicError::unsupported(
            "COLI recompilation",
            "INT4-G32 requantization currently requires raw weight and scale payloads",
        ));
    }
    let weights = read_component(package, record, desc.weight_offset, desc.weight_stored)?;
    let scales = read_component(package, record, desc.scale_offset, desc.scale_stored)?;
    let values =
        codecs::int4_grouped_decode(&weights, &scales, desc.rows as usize, desc.cols as usize)?;
    let mut packed = Vec::new();
    let mut e8m0 = Vec::new();
    for row in values.chunks_exact(desc.cols as usize) {
        quantize_f32_row(row, &mut packed, &mut e8m0)?;
    }
    Ok(PackedMatrix {
        rows: desc.rows,
        columns: desc.cols,
        weights: packed,
        scales: e8m0,
    })
}

fn read_component(
    package: &Package,
    record: &RecordInfo,
    offset: u64,
    bytes: u64,
) -> Result<Vec<u8>> {
    let bytes: usize = bytes
        .try_into()
        .map_err(|_| ColicError::Usage("expert component exceeds usize".into()))?;
    package
        .read_payload_range(record, offset, bytes)
        .map_err(Into::into)
}

fn validate_mxfp4_lengths(rows: u32, cols: u32, weights: &[u8], scales: &[u8]) -> Result<()> {
    let expected_weights = (rows as usize)
        .checked_mul((cols as usize).div_ceil(2))
        .ok_or_else(|| ColicError::Usage("MXFP4 weight size overflows usize".into()))?;
    let expected_scales = (rows as usize)
        .checked_mul((cols as usize).div_ceil(mxfp4::GROUP_SIZE))
        .ok_or_else(|| ColicError::Usage("MXFP4 scale size overflows usize".into()))?;
    if weights.len() != expected_weights || scales.len() != expected_scales {
        return Err(ColicError::Usage(format!(
            "MXFP4 buffers have {}/{} bytes; expected {expected_weights}/{expected_scales}",
            weights.len(),
            scales.len()
        )));
    }
    Ok(())
}

fn detile_apple8(rows: u32, cols: u32, tiles: &[u8]) -> Result<PackedMatrix> {
    let expected = target::apple8_tile_bytes(rows, cols)? as usize;
    if tiles.len() != expected {
        return Err(ColicError::Usage(
            "Apple8 tile buffer has the wrong size".into(),
        ));
    }
    let row_bytes = (cols as usize).div_ceil(2);
    let groups = (cols as usize).div_ceil(target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize);
    let mut weights = vec![0_u8; rows as usize * row_bytes];
    let mut scales = vec![0_u8; rows as usize * groups];
    for row in 0..rows as usize {
        let weight_row = row * row_bytes;
        let scale_row = row * groups;
        let tile_row = row / target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        let within_row = row % target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        for group in 0..groups {
            let tile_index = tile_row
                .checked_mul(groups)
                .and_then(|value| value.checked_add(group))
                .ok_or_else(|| ColicError::Usage("Apple8 tile index overflows usize".into()))?;
            let tile = tile_index
                .checked_mul(target_registry::APPLE8_MXFP4_TILE_BYTES as usize)
                .ok_or_else(|| ColicError::Usage("Apple8 tile offset overflows usize".into()))?;
            let column = group * target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize;
            let logical_columns =
                (cols as usize - column).min(target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize);
            let copy_bytes = logical_columns.div_ceil(2);
            let src = tile + within_row * target_registry::APPLE8_MXFP4_WEIGHT_ROW_BYTES as usize;
            let dst = weight_row + column / 2;
            weights[dst..dst + copy_bytes].copy_from_slice(&tiles[src..src + copy_bytes]);
            scales[scale_row + group] =
                tiles[tile + target_registry::APPLE8_MXFP4_WEIGHT_BYTES as usize + within_row];
        }
    }
    Ok(PackedMatrix {
        rows,
        columns: cols,
        weights,
        scales,
    })
}

fn pack_apple8(matrix: &PackedMatrix) -> Result<Vec<u8>> {
    validate_mxfp4_lengths(matrix.rows, matrix.columns, &matrix.weights, &matrix.scales)?;
    let row_bytes = (matrix.columns as usize).div_ceil(2);
    let groups =
        (matrix.columns as usize).div_ceil(target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize);
    let mut output = vec![
        0_u8;
        usize::try_from(target::apple8_tile_bytes(matrix.rows, matrix.columns)?)
            .map_err(|_| ColicError::Usage("Apple8 matrix exceeds usize".into()))?
    ];
    for row in 0..matrix.rows as usize {
        let weight_row = row * row_bytes;
        let scale_row = row * groups;
        let tile_row = row / target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        let within_row = row % target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        for group in 0..groups {
            let tile_index = tile_row
                .checked_mul(groups)
                .and_then(|value| value.checked_add(group))
                .ok_or_else(|| ColicError::Usage("Apple8 tile index overflows usize".into()))?;
            let tile = tile_index
                .checked_mul(target_registry::APPLE8_MXFP4_TILE_BYTES as usize)
                .ok_or_else(|| ColicError::Usage("Apple8 tile offset overflows usize".into()))?;
            let column = group * target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize;
            let logical_columns = (matrix.columns as usize - column)
                .min(target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize);
            let copy_bytes = logical_columns.div_ceil(2);
            let src = weight_row + column / 2;
            let dst = tile + within_row * target_registry::APPLE8_MXFP4_WEIGHT_ROW_BYTES as usize;
            output[dst..dst + copy_bytes].copy_from_slice(&matrix.weights[src..src + copy_bytes]);
            output[tile + target_registry::APPLE8_MXFP4_WEIGHT_BYTES as usize + within_row] =
                matrix.scales[scale_row + group];
        }
    }
    Ok(output)
}

fn build_canonical_mxfp4_expert(
    layer: i32,
    expert: i32,
    matrices: [&PackedMatrix; EXPERT_MATRICES],
) -> Result<Vec<u8>> {
    let mut payload = vec![0_u8; EXPERT_DATA_OFFSET];
    expert_header(&mut payload, layer, expert);
    let mut resident = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let weight_offset = append_aligned(&mut payload, &matrix.weights, DATA_ALIGNMENT)?;
        let scale_offset = append_aligned(&mut payload, &matrix.scales, DATA_ALIGNMENT)?;
        let d = EXPERT_HEADER_BYTES + index * EXPERT_DESC_BYTES;
        put_u16(&mut payload, d, (index + 1) as u16);
        put_u16(&mut payload, d + 4, MATH_MXFP4);
        put_u16(&mut payload, d + 6, SCALE_E8M0);
        put_u64(&mut payload, d + 16, matrix.rows as u64);
        put_u64(&mut payload, d + 24, matrix.columns as u64);
        put_u32(&mut payload, d + 32, 1);
        put_u32(&mut payload, d + 36, mxfp4::GROUP_SIZE as u32);
        put_u64(&mut payload, d + 48, weight_offset);
        put_u64(&mut payload, d + 56, matrix.weights.len() as u64);
        put_u64(&mut payload, d + 64, matrix.weights.len() as u64);
        put_u64(&mut payload, d + 72, scale_offset);
        put_u64(&mut payload, d + 80, matrix.scales.len() as u64);
        put_u64(&mut payload, d + 88, matrix.scales.len() as u64);
        let mut logical = Vec::with_capacity(matrix.weights.len() + matrix.scales.len());
        logical.extend_from_slice(&matrix.weights);
        logical.extend_from_slice(&matrix.scales);
        put_u32(&mut payload, d + 96, storage::crc32c(&logical));
        resident = resident
            .checked_add(logical.len() as u64)
            .ok_or_else(|| ColicError::Usage("MXFP4 resident size overflows u64".into()))?;
    }
    put_u64(&mut payload, 48, resident);
    Ok(payload)
}

fn build_apple8_expert(
    layer: i32,
    expert: i32,
    matrices: [&PackedMatrix; EXPERT_MATRICES],
) -> Result<Vec<u8>> {
    let mut payload = vec![0_u8; EXPERT_DATA_OFFSET];
    expert_header(&mut payload, layer, expert);
    let mut resident = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let tiles = pack_apple8(matrix)?;
        let weight_offset = append_aligned(
            &mut payload,
            &tiles,
            target_registry::APPLE8_MXFP4_MATRIX_ALIGNMENT,
        )?;
        let d = EXPERT_HEADER_BYTES + index * EXPERT_DESC_BYTES;
        put_u16(&mut payload, d, (index + 1) as u16);
        put_u16(
            &mut payload,
            d + 4,
            target_registry::APPLE8_MXFP4_MATH_FORMAT,
        );
        put_u16(
            &mut payload,
            d + 6,
            target_registry::APPLE8_MXFP4_SCALE_FORMAT,
        );
        put_u16(
            &mut payload,
            d + 12,
            target_registry::APPLE8_MXFP4_TILE_LAYOUT,
        );
        put_u64(&mut payload, d + 16, matrix.rows as u64);
        put_u64(&mut payload, d + 24, matrix.columns as u64);
        put_u32(
            &mut payload,
            d + 32,
            target_registry::APPLE8_MXFP4_SCALE_BLOCK_ROWS,
        );
        put_u32(
            &mut payload,
            d + 36,
            target_registry::APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
        );
        put_u64(&mut payload, d + 48, weight_offset);
        put_u64(&mut payload, d + 56, tiles.len() as u64);
        put_u64(&mut payload, d + 64, tiles.len() as u64);
        put_u32(&mut payload, d + 96, storage::crc32c(&tiles));
        put_u32(
            &mut payload,
            d + 104,
            target_registry::APPLE8_MXFP4_GROUP_SIZE,
        );
        resident = resident
            .checked_add(tiles.len() as u64)
            .ok_or_else(|| ColicError::Usage("Apple8 resident size overflows u64".into()))?;
    }
    put_u64(&mut payload, 48, resident);
    Ok(payload)
}

fn expert_header(payload: &mut [u8], layer: i32, expert: i32) {
    payload[..8].copy_from_slice(b"COLIEXPT");
    put_u16(payload, 8, 1);
    put_u32(payload, 12, EXPERT_HEADER_BYTES as u32);
    put_i32(payload, 16, layer);
    put_i32(payload, 20, expert);
    put_u16(payload, 24, EXPERT_MATRICES as u16);
    put_u32(payload, 28, EXPERT_DESC_BYTES as u32);
    put_u64(payload, 32, EXPERT_HEADER_BYTES as u64);
    put_u64(payload, 40, EXPERT_DATA_OFFSET as u64);
}

fn canonical_mxfp4_sizes(descs: &[MatrixDesc; EXPERT_MATRICES]) -> Result<(u64, u64)> {
    let mut stored = EXPERT_DATA_OFFSET as u64;
    let mut decoded = 0_u64;
    for desc in descs {
        let weights = u64::from(desc.rows)
            .checked_mul(u64::from(desc.cols).div_ceil(2))
            .ok_or_else(|| ColicError::Usage("MXFP4 weight size overflows u64".into()))?;
        let scales = u64::from(desc.rows)
            .checked_mul(u64::from(desc.cols).div_ceil(mxfp4::GROUP_SIZE as u64))
            .ok_or_else(|| ColicError::Usage("MXFP4 scale size overflows u64".into()))?;
        stored = storage::align_up(stored, DATA_ALIGNMENT)?
            .checked_add(weights)
            .ok_or_else(|| ColicError::Usage("MXFP4 record size overflows u64".into()))?;
        stored = storage::align_up(stored, DATA_ALIGNMENT)?
            .checked_add(scales)
            .ok_or_else(|| ColicError::Usage("MXFP4 record size overflows u64".into()))?;
        decoded = decoded
            .checked_add(weights)
            .and_then(|value| value.checked_add(scales))
            .ok_or_else(|| ColicError::Usage("MXFP4 decoded size overflows u64".into()))?;
    }
    Ok((stored, decoded))
}

fn apple8_sizes(descs: &[MatrixDesc; EXPERT_MATRICES]) -> Result<(u64, u64)> {
    let mut stored = EXPERT_DATA_OFFSET as u64;
    let mut decoded = 0_u64;
    for desc in descs {
        let bytes = target::apple8_tile_bytes(desc.rows, desc.cols)?;
        stored = storage::align_up(stored, target_registry::APPLE8_MXFP4_MATRIX_ALIGNMENT)?
            .checked_add(bytes)
            .ok_or_else(|| ColicError::Usage("Apple8 record size overflows u64".into()))?;
        decoded = decoded
            .checked_add(bytes)
            .ok_or_else(|| ColicError::Usage("Apple8 decoded size overflows u64".into()))?;
    }
    Ok((stored, decoded))
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8], alignment: u64) -> Result<u64> {
    let offset = storage::align_up(output.len() as u64, alignment)?;
    output.resize(
        offset
            .try_into()
            .map_err(|_| ColicError::Usage("record offset exceeds usize".into()))?,
        0,
    );
    output.extend_from_slice(bytes);
    Ok(offset)
}

fn quantize_f32_row(values: &[f32], weights: &mut Vec<u8>, scales: &mut Vec<u8>) -> Result<()> {
    let mut nibbles = Vec::with_capacity(values.len());
    for (group_index, group) in values.chunks(mxfp4::GROUP_SIZE).enumerate() {
        if group.iter().any(|value| !value.is_finite()) {
            return Err(ColicError::Usage(format!(
                "MXFP4 requantization refuses non-finite value in group {group_index}"
            )));
        }
        let (scale_code, scale) = choose_scale(group);
        scales.push(scale_code);
        for value in group {
            nibbles.push(quantize_value(*value, scale));
        }
    }
    for pair in nibbles.chunks(2) {
        let low = pair[0] & 0x0f;
        let high = pair.get(1).copied().unwrap_or(0) & 0x0f;
        weights.push(low | (high << 4));
    }
    Ok(())
}

fn choose_scale(values: &[f32]) -> (u8, f32) {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        return (127, 1.0);
    }
    let bits = max_abs.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let max_exp = if biased == 0 {
        let mantissa = bits & 0x007f_ffff;
        (31 - mantissa.leading_zeros() as i32) - 149
    } else {
        biased - 127
    };
    let mut scale_exp = (max_exp - 2).clamp(-126, 127);
    let mut scale_code = (scale_exp + 127) as u8;
    let mut scale = mxfp4::runtime_e8m0_to_f32(scale_code);
    if max_abs > mxfp4::MAX_E2M1 * scale && scale_exp < 127 {
        scale_exp += 1;
        scale_code = (scale_exp + 127) as u8;
        scale = mxfp4::runtime_e8m0_to_f32(scale_code);
    }
    (scale_code, scale)
}

fn quantize_value(value: f32, scale: f32) -> u8 {
    let magnitude = (value.abs() / scale).min(mxfp4::MAX_E2M1);
    let mut best_code = 0_u8;
    let mut best_error = f32::INFINITY;
    for (code, candidate) in mxfp4::E2M1_MAGNITUDES.iter().copied().enumerate() {
        let error = (magnitude - candidate).abs();
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) != 0) {
            best_error = error;
            best_code = code as u8;
        }
    }
    if value.is_sign_negative() {
        best_code | 0x8
    } else {
        best_code
    }
}

fn copy_json_metadata(source: &Path, output: &Path) -> Result<()> {
    let entries = fs::read_dir(source).map_err(|source_error| ColicError::Io {
        path: source.to_owned(),
        source: source_error,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source_error| ColicError::Io {
            path: source.to_owned(),
            source: source_error,
        })?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name == "recompile.json" || !name.ends_with(".json") {
            continue;
        }
        if entry
            .file_type()
            .map_err(|source_error| ColicError::Io {
                path: path.clone(),
                source: source_error,
            })?
            .is_file()
        {
            files.push((name.to_owned(), path));
        }
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, source_path) in files {
        let destination = output.join(name);
        fs::copy(&source_path, &destination).map_err(|source_error| ColicError::Io {
            path: source_path,
            source: source_error,
        })?;
    }
    Ok(())
}

fn write_provenance(
    package: &Package,
    target: TargetProfile,
    actions: &[Action],
    request: &RecompileRequest,
    output: &Path,
) -> Result<()> {
    let parent_manifest = package.manifest_ref();
    let parent_manifest_sha256 = format!("{:x}", Sha256::digest(parent_manifest));
    let rewritten = actions
        .iter()
        .filter(|action| matches!(action.kind, ActionKind::Rewrite { .. }))
        .count();
    let requantized = actions
        .iter()
        .filter(|action| {
            matches!(
                action.kind,
                ActionKind::Rewrite {
                    requantized: true,
                    ..
                }
            )
        })
        .count();
    let provenance = json!({
        "version": 1,
        "operation": "coli-recompile",
        "source_model_fingerprint": hex_fingerprint(package.fingerprint()),
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_profile": package.profile(),
        "target_profile": target.name,
        "quant": request.quant.as_str(),
        "quant_rules": request.quant_rules.iter().map(|rule| rule.as_spec()).collect::<Vec<_>>(),
        "in_place": request.source == request.output,
        "codec": request.codec.as_str(),
        "allow_requantize": request.allow_requantize,
        "repack": request.repack,
        "rewritten_experts": rewritten,
        "requantized_experts": requantized,
    });
    let path = output.join("recompile.json");
    let bytes = serde_json::to_vec_pretty(&provenance).map_err(|error| {
        ColicError::Usage(format!("cannot encode recompile provenance: {error}"))
    })?;
    fs::write(&path, bytes).map_err(|source| ColicError::Io { path, source })
}

fn hex_fingerprint(bytes: &[u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| ColicError::Usage("truncated u16 in expert record".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| ColicError::Usage("truncated u32 in expert record".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| ColicError::Usage("truncated u64 in expert record".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(rows: u32, columns: u32) -> PackedMatrix {
        let row_bytes = (columns as usize).div_ceil(2);
        let groups = (columns as usize).div_ceil(mxfp4::GROUP_SIZE);
        PackedMatrix {
            rows,
            columns,
            weights: (0..rows as usize * row_bytes)
                .map(|index| index.wrapping_mul(37) as u8)
                .collect(),
            scales: (0..rows as usize * groups)
                .map(|index| 120_u8.wrapping_add(index as u8))
                .collect(),
        }
    }

    #[test]
    fn apple8_tile_roundtrip_is_bit_exact() {
        for (rows, columns) in [(1, 32), (8, 32), (9, 33), (17, 97)] {
            let source = packed(rows, columns);
            let tiles = pack_apple8(&source).unwrap();
            let decoded = detile_apple8(rows, columns, &tiles).unwrap();
            assert_eq!(decoded, source);
        }
    }

    #[test]
    fn canonical_and_apple8_size_plans_match_builders() {
        let matrices = [packed(9, 33), packed(9, 33), packed(33, 9)];
        let descs = [
            MatrixDesc {
                role: 1,
                math: MATH_MXFP4,
                scale: SCALE_E8M0,
                weight_codec: 0,
                scale_codec: 0,
                layout: 0,
                rows: 9,
                cols: 33,
                weight_table: 0,
                scale_table: 0,
                weight_offset: 0,
                weight_stored: 0,
                weight_decoded: 0,
                scale_offset: 0,
                scale_stored: 0,
                scale_decoded: 0,
            },
            MatrixDesc {
                role: 2,
                rows: 9,
                cols: 33,
                ..zero_desc()
            },
            MatrixDesc {
                role: 3,
                rows: 33,
                cols: 9,
                ..zero_desc()
            },
        ];
        let canonical =
            build_canonical_mxfp4_expert(0, 0, [&matrices[0], &matrices[1], &matrices[2]]).unwrap();
        let apple = build_apple8_expert(0, 0, [&matrices[0], &matrices[1], &matrices[2]]).unwrap();
        assert_eq!(
            canonical.len() as u64,
            canonical_mxfp4_sizes(&descs).unwrap().0
        );
        assert_eq!(apple.len() as u64, apple8_sizes(&descs).unwrap().0);
    }

    #[test]
    fn mixed_quant_rules_are_ordered_and_last_match_wins() {
        let mut request =
            RecompileRequest::new(PathBuf::from("old.coli"), PathBuf::from("new.coli"));
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
    fn low_space_in_place_retarget_preserves_shard_size_and_record_offset() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "logan-recompile-in-place-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();

        let matrices = [packed(9, 32), packed(9, 32), packed(32, 32)];
        let payload =
            build_apple8_expert(0, 0, [&matrices[0], &matrices[1], &matrices[2]]).unwrap();
        let fingerprint = [0x5a; 32];
        let lowered = LoweredRecord {
            id: 1,
            kind: 2,
            stored_bytes: payload.len() as u64,
            decoded_bytes: u64::from_le_bytes(payload[48..56].try_into().unwrap()),
        };
        let source_plan =
            storage::plan_records(&[lowered], target::MACOS_ARM64_METAL_APPLE8_V1, 64 * 1024)
                .unwrap();
        let shard_path = root.join("data-00000.coli");
        let mut writer = storage::DataShardWriter::create(
            &shard_path,
            0,
            source_plan.record_alignment,
            fingerprint,
        )
        .unwrap();
        writer
            .write_record(&source_plan.records[0], &payload)
            .unwrap();
        writer.finish().unwrap();
        let before_shard_bytes = fs::metadata(&shard_path).unwrap().len();
        let before_offset = source_plan.records[0].payload_offset;

        let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&shard_path)
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        let metadata = [ManifestRecord {
            id: 1,
            name: Some("layers.0.ffn.experts.0".into()),
            layer: 0,
            expert: 0,
            kind: 2,
            codec: 0,
            math_format: 0xfffe,
            scale_format: 0xfffe,
            layout: 0xfffe,
            flags: 0,
            stored_crc32c: storage::crc32c(&payload),
            logical_crc32c: 0,
            codec_table_id: 0,
        }];
        let manifest = storage::encode_manifest_with_records(
            &source_plan,
            target::MACOS_ARM64_METAL_APPLE8_V1.name,
            fingerprint,
            &metadata,
            &[u32::from_le_bytes(header[72..76].try_into().unwrap())],
        )
        .unwrap();
        fs::write(root.join("manifest.coli"), manifest).unwrap();
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        let mut request = RecompileRequest::new(root.clone(), root.clone());
        request.target = target::LINUX_X86_64_AVX2_V1.name.into();
        request.force = true;
        ensure_in_place_state(
            &request,
            target::MACOS_ARM64_METAL_APPLE8_V1.name,
            target::LINUX_X86_64_AVX2_V1.name,
            1,
        )
        .unwrap();

        // Simulate a crash after a record overwrite but before its checkpoint
        // manifest is committed. Recovery restores exactly the old record and
        // manifest using only one-record scratch.
        {
            let source = Package::open(&root).unwrap();
            begin_record_journal(&root, &source, &source.records()[0]).unwrap();
            write_record_payload(&root, 0, before_offset, &[0_u8; 32]).unwrap();
        }
        recover_in_place_if_needed(&request).unwrap();
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        // Simulate a crash in the tiny finalization transaction after shard
        // headers start changing. The header+manifest backup rolls it back.
        begin_finalization_journal(&root, 1).unwrap();
        set_in_place_phase(&root, "finalizing").unwrap();
        write_shard_header(&root, 0, &[0_u8; storage::DATA_SHARD_HEADER_BYTES as usize]).unwrap();
        recover_in_place_if_needed(&request).unwrap();
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        let before = Package::open(&root).unwrap();
        assert_eq!(before.records()[0].offset, before_offset);
        let before_stored = before.records()[0].stored;

        let summary = recompile(&request).unwrap();
        assert_eq!(summary.rewritten_experts, 1);

        let after = Package::open(&root).unwrap();
        assert_eq!(after.profile(), target::LINUX_X86_64_AVX2_V1.name);
        assert_eq!(after.records()[0].shard_id, 0);
        assert_eq!(after.records()[0].offset, before_offset);
        assert!(after.records()[0].stored < before_stored);
        assert_eq!(fs::metadata(&shard_path).unwrap().len(), before_shard_bytes);
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quant_rule_parser_rejects_reversed_and_unknown_selectors() {
        assert!(QuantRule::parse("layer:9-3=mxfp4").is_err());
        assert!(QuantRule::parse("dense=mxfp4").is_err());
        assert!(QuantRule::parse("layer:2=bogus").is_err());
    }

    #[test]
    fn patch_manifest_shard_sizes_preserves_physical_file_lengths() {
        let plan = StoragePlan {
            record_alignment: 4096,
            shard_size_limit: 16 * 1024,
            shards: 1,
            records: vec![PlannedRecord {
                record: LoweredRecord {
                    id: 1,
                    kind: 1,
                    stored_bytes: 32,
                    decoded_bytes: 32,
                },
                shard_id: 0,
                payload_offset: 4096,
            }],
            projected_stored_bytes: 32,
            projected_padding_bytes: 0,
        };
        let records = [ManifestRecord {
            id: 1,
            name: None,
            layer: -1,
            expert: -1,
            kind: 1,
            codec: 0,
            math_format: 0,
            scale_format: 0,
            layout: 0,
            flags: 0,
            stored_crc32c: 0,
            logical_crc32c: 0,
            codec_table_id: 0,
        }];
        let mut manifest =
            storage::encode_manifest_with_records(&plan, "test-profile", [7; 32], &records, &[123])
                .unwrap();
        patch_manifest_shard_sizes(&mut manifest, &[12 * 1024]).unwrap();
        let table = u64_at(&manifest, 48).unwrap() as usize;
        assert_eq!(u64_at(&manifest, table + 16).unwrap(), 12 * 1024);
        let expected = u32_at(&manifest, 144).unwrap();
        let mut crc_bytes = manifest.clone();
        crc_bytes[144..148].fill(0);
        assert_eq!(storage::crc32c(&crc_bytes), expected);
    }

    fn zero_desc() -> MatrixDesc {
        MatrixDesc {
            role: 0,
            math: MATH_MXFP4,
            scale: SCALE_E8M0,
            weight_codec: 0,
            scale_codec: 0,
            layout: 0,
            rows: 0,
            cols: 0,
            weight_table: 0,
            scale_table: 0,
            weight_offset: 0,
            weight_stored: 0,
            weight_decoded: 0,
            scale_offset: 0,
            scale_stored: 0,
            scale_decoded: 0,
        }
    }
}
