use super::{
    DsparkGeometry, DsparkProjection, DsparkWeights, VerificationOptions, VerificationResult,
    verify_greedy_logits,
};
use crate::{
    BackendPreference,
    kv::KvCheckpoint,
    metal::BackendReport,
    model::{DenseModel, DenseSession, ForwardOutput},
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetHistory {
    tokens: Vec<u32>,
    capacity: usize,
}

impl TargetHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            tokens: Vec::new(),
            capacity,
        }
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn commit(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        if self.capacity != 0 && self.tokens.len() > self.capacity {
            let drop = self.tokens.len() - self.capacity;
            self.tokens.drain(..drop);
        }
    }
    pub fn clear(&mut self) {
        self.tokens.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftAttentionState {
    width: usize,
    mask_token: u32,
    committed: Vec<u32>,
    provisional: Vec<u32>,
}

impl DraftAttentionState {
    pub fn new(width: usize, mask_token: u32) -> Result<Self, String> {
        if width == 0 {
            return Err("draft attention width must be non-zero".into());
        }
        Ok(Self {
            width,
            mask_token,
            committed: Vec::new(),
            provisional: Vec::new(),
        })
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }
    pub fn committed(&self) -> &[u32] {
        &self.committed
    }
    pub fn provisional(&self) -> &[u32] {
        &self.provisional
    }

    /// Always returns the complete width.  Unused proposal slots are masks;
    /// they are not shortened to the number of proposals.
    pub fn full_width_context(&self) -> Vec<u32> {
        let mut context = vec![self.mask_token; self.width];
        let mut values = self
            .committed
            .iter()
            .chain(self.provisional.iter())
            .copied();
        let total = self.committed.len().saturating_add(self.provisional.len());
        let skip = total.saturating_sub(self.width);
        let retained = total.min(self.width);
        let start = self.width - retained;
        for (slot, token) in values.by_ref().skip(skip).enumerate() {
            context[start + slot] = token;
        }
        context
    }

    pub fn begin(&mut self, proposals: &[u32]) -> Result<Vec<u32>, String> {
        if proposals.len() > self.width {
            return Err(format!(
                "{} proposals exceed draft width {}",
                proposals.len(),
                self.width
            ));
        }
        self.provisional.clear();
        self.provisional.extend_from_slice(proposals);
        Ok(self.full_width_context())
    }
    /// Materialize a draft block with the anchor fixed at slot zero.  The
    /// remaining slots stay present and are filled with the model mask.
    pub fn begin_with_anchor(
        &mut self,
        anchor: u32,
        proposals: &[u32],
    ) -> Result<Vec<u32>, String> {
        if proposals.len().saturating_add(1) > self.width {
            return Err(format!(
                "anchor plus {} proposals exceed draft width {}",
                proposals.len(),
                self.width
            ));
        }
        self.provisional.clear();
        self.provisional.push(anchor);
        self.provisional.extend_from_slice(proposals);
        let mut context = vec![self.mask_token; self.width];
        context[..self.provisional.len()].copy_from_slice(&self.provisional);
        Ok(context)
    }

    pub fn commit_input_rows(&mut self, rows: &[u32]) {
        self.committed.extend_from_slice(rows);
        self.provisional.clear();
        if self.committed.len() > self.width {
            let drop = self.committed.len() - self.width;
            self.committed.drain(..drop);
        }
    }
    pub fn rollback(&mut self) {
        self.provisional.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalContext {
    pub anchor: u32,
    pub proposals: Vec<u32>,
    pub attention: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftProposal {
    pub context: ProposalContext,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedTaps {
    pub rows: usize,
    pub width: usize,
    pub taps: BTreeMap<usize, Vec<f32>>,
}

impl MaterializedTaps {
    pub fn from_forward(
        output: &ForwardOutput,
        configured: &[usize],
        width: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        let rows = output.rows.min(max_rows);
        let mut taps = BTreeMap::new();
        for &layer in configured {
            if let Some(values) = output.taps.get(&layer) {
                let expected = output.rows.checked_mul(width).ok_or("tap size overflow")?;
                if values.len() != expected {
                    return Err(format!(
                        "tap {layer} has {} values, expected {expected}",
                        values.len()
                    ));
                }
                taps.insert(layer, values[..rows * width].to_vec());
            }
        }
        Ok(Self { rows, width, taps })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedTargetStates {
    pub rows: usize,
    pub width: usize,
    pub values: Vec<f32>,
    pub backend: BackendReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkStateCheckpoint {
    pub target: KvCheckpoint,
    pub target_history: Vec<u32>,
    pub draft_committed: Vec<u32>,
    pub draft_provisional: Vec<u32>,
    pub anchor: Option<u32>,
    pub cancelled: bool,
}

#[derive(Debug, Clone)]
pub struct DsparkSession {
    target: DenseSession,
    weights: DsparkWeights,
    geometry: DsparkGeometry,
    target_history: TargetHistory,
    draft: DraftAttentionState,
    anchor: Option<u32>,
    cancelled: bool,
    max_materialized_rows: usize,
}
/// Compatibility name for callers that treat the target-plus-drafter as a model.
pub type DsparkModel = DsparkSession;
impl DsparkSession {
    pub fn new(target: Arc<DenseModel>, weights: DsparkWeights) -> Result<Self, String> {
        let geometry = weights.geometry.clone();
        geometry.validate()?;
        let draft = DraftAttentionState::new(geometry.block_width + 1, geometry.mask_token_id)?;
        let max_materialized_rows = geometry.block_width + 1;
        Ok(Self {
            target: DenseSession::new(target),
            weights,
            geometry,
            target_history: TargetHistory::new(4096),
            draft,
            anchor: None,
            cancelled: false,
            max_materialized_rows,
        })
    }
    pub fn from_target(target: DenseSession, weights: DsparkWeights) -> Result<Self, String> {
        let geometry = weights.geometry.clone();
        geometry.validate()?;
        let draft = DraftAttentionState::new(geometry.block_width + 1, geometry.mask_token_id)?;
        let max_materialized_rows = geometry.block_width + 1;
        Ok(Self {
            target,
            weights,
            geometry,
            target_history: TargetHistory::new(4096),
            draft,
            anchor: None,
            cancelled: false,
            max_materialized_rows,
        })
    }
    pub fn target(&self) -> &DenseSession {
        &self.target
    }
    pub fn target_mut(&mut self) -> &mut DenseSession {
        &mut self.target
    }
    pub fn geometry(&self) -> &DsparkGeometry {
        &self.geometry
    }
    pub fn weights(&self) -> &DsparkWeights {
        &self.weights
    }
    pub fn target_history(&self) -> &TargetHistory {
        &self.target_history
    }
    pub fn draft_state(&self) -> &DraftAttentionState {
        &self.draft
    }
    pub fn anchor(&self) -> Option<u32> {
        self.anchor
    }
    pub fn set_cancelled(&mut self, cancelled: bool) {
        self.cancelled = cancelled;
    }
    pub fn set_max_materialized_rows(&mut self, rows: usize) {
        self.max_materialized_rows = rows;
    }
    pub fn set_backend(&mut self, backend: BackendPreference) {
        self.target.set_backend(backend);
    }

    /// Prefill committed target history.  The final token becomes the caller's
    /// anchor; it is not implicitly proposed or emitted by this method.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        tap_ids: &[usize],
    ) -> Result<MaterializedTaps, String> {
        if tokens.is_empty() {
            return Ok(MaterializedTaps {
                rows: 0,
                width: self.geometry.hidden_size,
                taps: BTreeMap::new(),
            });
        }
        let output = self.target.forward(tokens, tap_ids)?;
        self.target_history.commit(tokens);
        self.draft.commit_input_rows(tokens);
        self.anchor = tokens.last().copied();
        MaterializedTaps::from_forward(
            &output,
            &self.geometry.taps,
            self.geometry.hidden_size,
            self.max_materialized_rows,
        )
    }
    /// Prefill and immediately execute the official target-tap projection.
    pub fn prefill_projected(
        &mut self,
        tokens: &[u32],
        tap_ids: &[usize],
    ) -> Result<(MaterializedTaps, ProjectedTargetStates), String> {
        let taps = self.prefill(tokens, tap_ids)?;
        let projected = self.project_target_taps(&taps)?;
        Ok((taps, projected))
    }

    /// Project target hidden states through the official DSpark `fc` tensor.
    ///
    /// The input is the row-wise concatenation of target taps in
    /// `geometry.taps` order. Metal receives the whole row batch at once;
    /// declined or non-BF16 execution falls back to the decoded CPU matrix.
    pub fn project_target_taps(
        &self,
        taps: &MaterializedTaps,
    ) -> Result<ProjectedTargetStates, String> {
        let projection = self
            .weights
            .fc
            .as_ref()
            .ok_or("DSpark weights do not contain fc projection")?;
        if taps.width != self.geometry.hidden_size {
            return Err(format!(
                "tap width {} != DSpark hidden size {}",
                taps.width, self.geometry.hidden_size
            ));
        }
        let input_width = self
            .geometry
            .taps
            .len()
            .checked_mul(taps.width)
            .ok_or("DSpark projection input width overflow")?;
        if projection.cols != input_width || projection.rows != self.geometry.hidden_size {
            return Err(format!(
                "DSpark projection is {}x{}, expected {}x{}",
                projection.rows, projection.cols, self.geometry.hidden_size, input_width
            ));
        }
        let input_len = taps
            .rows
            .checked_mul(input_width)
            .ok_or("DSpark projection input size overflow")?;
        let mut input = Vec::with_capacity(input_len);
        for row in 0..taps.rows {
            for &layer in &self.geometry.taps {
                let values = taps
                    .taps
                    .get(&layer)
                    .ok_or_else(|| format!("missing configured DSpark tap layer {layer}"))?;
                let start = row
                    .checked_mul(taps.width)
                    .ok_or("DSpark tap row offset overflow")?;
                let end = start + taps.width;
                if end > values.len() {
                    return Err(format!(
                        "tap {layer} has {} values, needs {end}",
                        values.len()
                    ));
                }
                input.extend_from_slice(&values[start..end]);
            }
        }
        let (values, backend) = project_rows(
            projection,
            self.target.backend_preference(),
            taps.rows,
            &input,
        )?;
        Ok(ProjectedTargetStates {
            rows: taps.rows,
            width: projection.rows,
            values,
            backend,
        })
    }

    pub fn propose(&mut self, anchor: u32, count: usize) -> Result<DraftProposal, String> {
        if count > self.geometry.block_width {
            return Err(format!(
                "proposal count {count} exceeds block width {}",
                self.geometry.block_width
            ));
        }
        if anchor as usize >= self.geometry.vocab_size {
            return Err("anchor exceeds DSpark vocabulary".into());
        }
        let mut tokens = Vec::with_capacity(count);
        let mut previous = anchor;
        for _ in 0..count {
            let next = self.weights.predict_markov(previous)?;
            tokens.push(next);
            previous = next;
        }
        let attention = self.draft.begin_with_anchor(anchor, &tokens)?;
        Ok(DraftProposal {
            context: ProposalContext {
                anchor,
                proposals: tokens.clone(),
                attention,
            },
            tokens,
        })
    }

    /// Verify using the target model.  The full `[anchor,drafts]` block is
    /// evaluated first; rejected rows are then discarded by restoring the
    /// checkpoint and replaying only the retained prefix.
    pub fn verify_block(
        &mut self,
        anchor: u32,
        drafts: &[u32],
        options: &VerificationOptions,
    ) -> Result<(VerificationResult, MaterializedTaps), String> {
        if drafts.len() > self.geometry.block_width {
            return Err(format!(
                "{} drafts exceed block width {}",
                drafts.len(),
                self.geometry.block_width
            ));
        }
        if self.cancelled || matches!(options.control, super::DecodeControl::Cancel) {
            self.draft.rollback();
            return Ok((
                super::verify_greedy_with_options(&[], drafts, options),
                MaterializedTaps {
                    rows: 0,
                    width: self.geometry.hidden_size,
                    taps: BTreeMap::new(),
                },
            ));
        }
        let checkpoint = self.target.checkpoint();
        let mut input = Vec::with_capacity(drafts.len() + 1);
        input.push(anchor);
        input.extend_from_slice(drafts);
        let output = self.target.forward(&input, &self.geometry.taps)?;
        let logits = (0..output.rows)
            .filter_map(|row| output.logits_row(row).map(|x| x.to_vec()))
            .collect::<Vec<_>>();
        let result = verify_greedy_logits(&logits, drafts, options);
        let taps = MaterializedTaps::from_forward(
            &output,
            &self.geometry.taps,
            self.geometry.hidden_size,
            self.max_materialized_rows,
        )?;
        if result.error.is_some() || result.cancelled {
            self.target.restore(checkpoint)?;
            self.draft.rollback();
            return Ok((result, taps));
        }
        self.target.restore(checkpoint)?;
        if result.retained_rows > 0 {
            self.target.forward(&input[..result.retained_rows], &[])?;
        }
        self.target_history.commit(&input[..result.retained_rows]);
        self.draft.commit_input_rows(&input[..result.retained_rows]);
        self.anchor = result.next_anchor;
        self.cancelled = false;
        Ok((result, taps))
    }

    /// Apply a verifier result produced by a synthetic target oracle.  This is
    /// the same state transition as `verify_block`, without running target
    /// math, and is useful for CPU-reference tests.
    pub fn commit_verification(
        &mut self,
        anchor: u32,
        drafts: &[u32],
        result: &VerificationResult,
    ) -> Result<(), String> {
        if result.cancelled || result.error.is_some() {
            self.draft.rollback();
            return Ok(());
        }
        if result.retained_rows > drafts.len() + 1 {
            return Err("verification retained more rows than its input block".into());
        }
        let mut input = Vec::with_capacity(drafts.len() + 1);
        input.push(anchor);
        input.extend_from_slice(drafts);
        self.target_history.commit(&input[..result.retained_rows]);
        self.draft.commit_input_rows(&input[..result.retained_rows]);
        self.anchor = result.next_anchor;
        Ok(())
    }

    pub fn checkpoint(&self) -> DsparkStateCheckpoint {
        DsparkStateCheckpoint {
            target: self.target.checkpoint(),
            target_history: self.target_history.tokens.clone(),
            draft_committed: self.draft.committed.clone(),
            draft_provisional: self.draft.provisional.clone(),
            anchor: self.anchor,
            cancelled: self.cancelled,
        }
    }
    pub fn restore(&mut self, checkpoint: &DsparkStateCheckpoint) -> Result<(), String> {
        self.target.restore(checkpoint.target)?;
        self.target_history.tokens = checkpoint.target_history.clone();
        self.draft.committed = checkpoint.draft_committed.clone();
        self.draft.provisional = checkpoint.draft_provisional.clone();
        self.anchor = checkpoint.anchor;
        self.cancelled = checkpoint.cancelled;
        Ok(())
    }
}
fn project_rows(
    projection: &DsparkProjection,
    requested: BackendPreference,
    rows: usize,
    input: &[f32],
) -> Result<(Vec<f32>, BackendReport), String> {
    let input_len = rows
        .checked_mul(projection.cols)
        .ok_or("DSpark projection input size overflow")?;
    if input.len() != input_len {
        return Err(format!(
            "DSpark projection input has {}, expected {input_len}",
            input.len()
        ));
    }
    let output_len = rows
        .checked_mul(projection.rows)
        .ok_or("DSpark projection output size overflow")?;
    if projection.values.len() != projection.rows * projection.cols {
        return Err("DSpark projection decoded weight size is invalid".into());
    }
    let mut output = vec![0.0; output_len];
    let backend = if rows == 0 {
        BackendReport::cpu(requested, "empty DSpark projection")
    } else {
        crate::metal::matmul_bf16(
            requested,
            projection.dtype,
            &projection.bytes,
            input,
            &mut output,
            rows,
            projection.rows,
            projection.cols,
        )
    };
    if matches!(backend.used, crate::metal::BackendUsed::Cpu) {
        for row in 0..rows {
            let x = &input[row * projection.cols..(row + 1) * projection.cols];
            let y = &mut output[row * projection.rows..(row + 1) * projection.rows];
            for (out, weights) in y
                .iter_mut()
                .zip(projection.values.chunks_exact(projection.cols))
            {
                *out = weights
                    .iter()
                    .zip(x)
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
        }
    }
    Ok((output, backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dspark_projection_batches_rows_in_tap_order() {
        let projection = DsparkProjection {
            rows: 2,
            cols: 3,
            dtype: crate::DType::BF16,
            values: vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0],
            bytes: Vec::new(),
        };
        let (output, backend) = project_rows(
            &projection,
            BackendPreference::Cpu,
            2,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(backend.used, crate::metal::BackendUsed::Cpu);
        assert_eq!(output, vec![14.0, 6.0, 32.0, 10.5]);
    }

    #[test]
    fn dspark_projection_rejects_missing_rows() {
        let projection = DsparkProjection {
            rows: 2,
            cols: 3,
            dtype: crate::DType::BF16,
            values: vec![0.0; 6],
            bytes: Vec::new(),
        };
        let error = project_rows(&projection, BackendPreference::Cpu, 1, &[1.0, 2.0]).unwrap_err();
        assert!(error.contains("expected 3"));
    }
}
