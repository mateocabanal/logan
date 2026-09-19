use std::fmt;

/// Controls a verification attempt without making cancellation a hidden global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeControl {
    Continue,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationOptions {
    pub eos_token_ids: Vec<u32>,
    pub output_cap: Option<usize>,
    pub control: DecodeControl,
}

impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            eos_token_ids: Vec::new(),
            output_cap: None,
            control: DecodeControl::Continue,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentDiagnostic {
    pub expected_rows: usize,
    pub actual_rows: usize,
    /// Target row `i` predicts draft `i`; the anchor is input row zero.
    pub comparison_offset: usize,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    pub accepted_drafts: usize,
    pub retained_rows: usize,
    pub next_anchor: Option<u32>,
    pub emitted: Vec<u32>,
    pub rejected_drafts: Vec<u32>,
    pub target_predictions: Vec<u32>,
    pub cancelled: bool,
    pub all_match: bool,
    pub stopped_on_eos: bool,
    pub output_cap_reached: bool,
    pub alignment: AlignmentDiagnostic,
    pub error: Option<String>,
}

impl VerificationResult {
    fn invalid(expected: usize, actual: usize, message: impl Into<String>) -> Self {
        Self {
            accepted_drafts: 0,
            retained_rows: 0,
            next_anchor: None,
            emitted: Vec::new(),
            rejected_drafts: Vec::new(),
            target_predictions: Vec::new(),
            cancelled: false,
            all_match: false,
            stopped_on_eos: false,
            output_cap_reached: false,
            alignment: AlignmentDiagnostic {
                expected_rows: expected,
                actual_rows: actual,
                comparison_offset: 0,
                valid: false,
            },
            error: Some(message.into()),
        }
    }
}

/// Verify token predictions produced by the target for inputs `[anchor,d1,..]`.
/// This convenience form has no EOS or output cap.  A mismatch at row `i`
/// commits only input rows `0..=i` (the anchor plus accepted drafts).
pub fn verify_greedy(target_predictions: &[u32], drafts: &[u32]) -> VerificationResult {
    verify_greedy_with_options(target_predictions, drafts, &VerificationOptions::default())
}

/// Exact greedy verifier with EOS, output-cap, and cancellation semantics.
pub fn verify_greedy_with_options(
    target_predictions: &[u32],
    drafts: &[u32],
    options: &VerificationOptions,
) -> VerificationResult {
    let expected = drafts.len();
    let alignment = AlignmentDiagnostic {
        expected_rows: expected,
        actual_rows: target_predictions.len(),
        comparison_offset: 0,
        valid: target_predictions.len() >= expected,
    };
    if matches!(options.control, DecodeControl::Cancel) {
        return VerificationResult {
            accepted_drafts: 0,
            retained_rows: 0,
            next_anchor: None,
            emitted: Vec::new(),
            rejected_drafts: drafts.to_vec(),
            target_predictions: target_predictions.to_vec(),
            cancelled: true,
            all_match: false,
            stopped_on_eos: false,
            output_cap_reached: false,
            alignment,
            error: None,
        };
    }
    if target_predictions.len() < expected {
        return VerificationResult::invalid(
            expected,
            target_predictions.len(),
            "target output has fewer rows than drafts",
        );
    }
    let alignment = AlignmentDiagnostic {
        expected_rows: expected,
        actual_rows: target_predictions.len(),
        comparison_offset: 0,
        valid: true,
    };
    let cap = options.output_cap.unwrap_or(usize::MAX);
    if cap == 0 {
        return VerificationResult {
            accepted_drafts: 0,
            retained_rows: 0,
            next_anchor: None,
            emitted: Vec::new(),
            rejected_drafts: drafts.to_vec(),
            target_predictions: target_predictions.to_vec(),
            cancelled: false,
            all_match: false,
            stopped_on_eos: false,
            output_cap_reached: true,
            alignment,
            error: None,
        };
    }

    let mut accepted = 0;
    let mut emitted = Vec::new();
    let mut rejected = Vec::new();
    let mut next_anchor = None;
    let mut all_match = true;
    let mut stopped_on_eos = false;
    let mut cap_reached = false;

    for (row, &draft) in drafts.iter().enumerate() {
        let prediction = target_predictions[row];
        if prediction != draft {
            all_match = false;
            next_anchor = Some(prediction);
            rejected.extend_from_slice(&drafts[row..]);
            if emitted.len() < cap {
                emitted.push(prediction);
                if options.eos_token_ids.contains(&prediction) {
                    stopped_on_eos = true;
                }
            } else {
                cap_reached = true;
            }
            break;
        }
        accepted += 1;
        if emitted.len() >= cap {
            cap_reached = true;
            break;
        }
        emitted.push(draft);
        if options.eos_token_ids.contains(&draft) {
            stopped_on_eos = true;
            break;
        }
        if emitted.len() == cap {
            cap_reached = true;
            break;
        }
    }

    if all_match && !stopped_on_eos && !cap_reached {
        next_anchor = target_predictions.get(drafts.len()).copied();
        if next_anchor.is_none() {
            return VerificationResult::invalid(
                expected + 1,
                target_predictions.len(),
                "target output lacks the post-block prediction",
            );
        }
    }

    let retained_rows = if stopped_on_eos || cap_reached || !all_match {
        1 + accepted
    } else {
        drafts.len() + 1
    };
    if all_match && !stopped_on_eos && !cap_reached {
        rejected.clear();
    }
    VerificationResult {
        accepted_drafts: accepted,
        retained_rows,
        next_anchor,
        emitted,
        rejected_drafts: rejected,
        target_predictions: target_predictions.to_vec(),
        cancelled: false,
        all_match,
        stopped_on_eos,
        output_cap_reached: cap_reached,
        alignment,
        error: None,
    }
}
/// applying the same verifier.  Empty rows are rejected as an alignment error.
pub fn verify_greedy_logits(
    target_logits: &[Vec<f32>],
    drafts: &[u32],
    options: &VerificationOptions,
) -> VerificationResult {
    let mut predictions = Vec::with_capacity(target_logits.len());
    for row in target_logits {
        if row.is_empty() {
            return VerificationResult::invalid(
                drafts.len(),
                target_logits.len(),
                "target logits contain an empty row",
            );
        }
        let mut best = 0usize;
        for (index, &value) in row.iter().enumerate().skip(1) {
            if value > row[best] {
                best = index;
            }
        }
        if best > u32::MAX as usize {
            return VerificationResult::invalid(
                drafts.len(),
                target_logits.len(),
                "vocabulary index exceeds u32",
            );
        }
        predictions.push(best as u32);
    }
    verify_greedy_with_options(&predictions, drafts, options)
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = &self.error {
            return f.write_str(error);
        }
        write!(
            f,
            "accepted {} of {} draft tokens",
            self.accepted_drafts, self.alignment.expected_rows
        )
    }
}
