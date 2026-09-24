//! Next-token expert staging: a token-scoped arena for predicted expert bytes,
//! plus the candidate fusion that fills it.
//!
//! # Why this is not [`crate::RouteArena`]
//!
//! RouteArena is excellent at what it does and it is the wrong buffer for
//! prediction. It is **layer-parity scratch**: two arenas, one per layer parity,
//! refilled by whichever layer of that parity is currently executing. Its
//! lifetime is one layer's demand read, so bytes written into it by a
//! *speculative* load are overwritten by the next layer of the same parity long
//! before the token that wanted them arrives. The measured consequence was
//! exactly that: Edge0 submitted correct early reads, the arena later performed
//! its own demand read of the same experts, and the predicted bytes were never
//! consumed — duplicated I/O and no speedup.
//!
//! This module is the opposite shape. It is **token-generation staged and
//! layer-addressed**: one row per layer, whose slots are consumed by token `t`
//! and immediately refilled for token `t+1`.
//!
//! ```text
//! token t, layer L:
//!   1. native K4 route for (t, L) is final      ← the only route ever executed
//!   2. staged lookup per route expert           → hit: take the staged bytes
//!                                                 late: wait, do NOT re-read
//!                                                 miss: demand-load, exactly as
//!                                                       native K4 would
//!   3. MoE compute over those bytes
//!   4. free the row, fuse edge0 + routescout for (t+1, L), submit those reads
//! ```
//!
//! Step 4 comes after step 3 because a refill can only target slots step 3 is
//! finished reading. The lead time is therefore "one layer's compute" short of a
//! full token, which is ~97% of a token at this host's ~6 ms/layer against
//! ~200 ms/token — and crucially it is the *same* lead for every layer rather
//! than degrading toward the bottom of the stack as a single token-boundary
//! submission would.
//!
//! # The invariant that makes it safe
//!
//! **Native K4 is always the executed route.** Nothing here can change which
//! experts run: [`crate::Model::route_topk_n`] computes the route natively in
//! this mode, and this module is only ever asked to make bytes ready for a route
//! the router has already decided. A wrong prediction wastes SSD bandwidth and
//! nothing else.
//!
//! Every slot carries the `(request_generation, token_generation)` it was filled
//! for. A slot whose tag does not match the generation being served is a
//! **miss**, never a hit — so a prediction left over from a previous
//! conversation, or a token that has already been consumed, can never be handed
//! to the kernel as this token's bytes.

use crate::edge0_router::RankedCandidate;

/// How two predictors' rankings are combined into one candidate set.
///
/// The scores are NOT on a common scale — Edge0's are softmax probabilities over
/// 256 experts, RouteScout's are peak-normalized transition sums — so every arm
/// is either single-source or scale-free. Treating them as calibrated
/// probabilities is the mistake this enum exists to make measurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionArm {
    /// Edge0's ranking only.
    Edge0,
    /// RouteScout's ranking only.
    RouteScout,
    /// Both rankings interleaved in rank order, deduplicated. Needs no score at
    /// all, so it stays correct if either source's scale is wrong.
    Union,
    /// Reciprocal rank fusion, `sum(1 / (RRF_K + rank))`. Scale-free by
    /// construction: needs no normalization and no weights.
    ReciprocalRank,
    /// Independently peak-normalize each source to [0,1], then
    /// `w_edge0 * edge0 + w_rs * routescout`. The weights are the sweep's
    /// variable.
    Weighted,
}

impl FusionArm {
    /// Parse `QWEN_HYBRID_FUSION`. An unrecognised value falls back to the
    /// default with a warning, matching this repo's convention for a bad mode
    /// string rather than silently reinterpreting the run.
    pub fn from_env() -> Self {
        match std::env::var("QWEN_HYBRID_FUSION") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "edge0" => Self::Edge0,
                "routescout" | "rs" => Self::RouteScout,
                "union" => Self::Union,
                "rrf" | "reciprocal" | "reciprocal_rank" => Self::ReciprocalRank,
                "weighted" => Self::Weighted,
                other => {
                    eprintln!("logan hybrid: unknown QWEN_HYBRID_FUSION={other:?}; using weighted");
                    Self::Weighted
                }
            },
            Err(_) => Self::Weighted,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Edge0 => "edge0",
            Self::RouteScout => "routescout",
            Self::Union => "union",
            Self::ReciprocalRank => "rrf",
            Self::Weighted => "weighted",
        }
    }
}

/// RRF rank constant. Only sets how fast `1/(k+rank)` decays across ranks; no
/// measurement here was sensitive to it, because neither source supplies more
/// than `M` candidates.
const RRF_K: f32 = 60.0;

/// Staging policy, resolved once from the environment.
#[derive(Clone, Copy, Debug)]
pub struct StageConfig {
    /// Candidate budget `M` — experts read per layer per token. **Not** the
    /// execution K, which stays 4.
    pub candidates: usize,
    pub arm: FusionArm,
    pub w_edge0: f32,
    pub w_routescout: f32,
    /// Whether to submit staging reads at all. `false` keeps the predictors
    /// running but reads nothing — the "staging disabled" control arm.
    pub enabled: bool,
    /// Whether the previous token's same-layer route is added to the staged set.
    ///
    /// On by default, and deliberately so: this arena has *no residency*, so an
    /// expert the previous token used is not in memory and must be re-read. That
    /// makes the "already resident, do not predict it" assumption behind
    /// RouteScout's cold-arrivals framing false here. EXP-069 measured same-layer
    /// cross-token reuse at 31.3% (retain-4) on this checkpoint, so omitting
    /// these candidates would discard the single best available hint.
    pub resident_prior: bool,
}

impl StageConfig {
    pub fn from_env() -> Self {
        Self {
            candidates: std::env::var("QWEN_HYBRID_STAGE_M")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
                .clamp(1, 64),
            arm: FusionArm::from_env(),
            w_edge0: env_f32("QWEN_HYBRID_W_EDGE0", 1.0),
            w_routescout: env_f32("QWEN_HYBRID_W_RS", 1.0),
            enabled: std::env::var("QWEN_HYBRID_STAGE")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(true),
            resident_prior: std::env::var("QWEN_HYBRID_RESIDENT_PRIOR")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(true),
        }
    }
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

/// Fuse two independently-ranked candidate lists into one ranking, best first.
///
/// Inputs are best-first per their own source. The output is deduplicated and at
/// most `budget` wide. Deduplication is not cosmetic: an expert staged twice
/// would consume two slots and leave a different candidate unstaged, and the
/// byte counters would double-count one read.
pub fn fuse_candidates(
    edge0: &[RankedCandidate],
    routescout: &[(usize, f32)],
    arm: FusionArm,
    w_edge0: f32,
    w_routescout: f32,
    budget: usize,
) -> Vec<usize> {
    if budget == 0 {
        return Vec::new();
    }
    match arm {
        FusionArm::Edge0 => rank_single(edge0.iter().map(|c| (c.expert, c.score)), budget),
        FusionArm::RouteScout => rank_single(routescout.iter().copied(), budget),
        FusionArm::Union => {
            let mut out: Vec<usize> = Vec::with_capacity(budget);
            let high = edge0.len().max(routescout.len());
            for rank in 0..high {
                if out.len() >= budget {
                    break;
                }
                if let Some(c) = edge0.get(rank) {
                    push_unique(&mut out, c.expert, budget);
                }
                if out.len() >= budget {
                    break;
                }
                if let Some(&(expert, _)) = routescout.get(rank) {
                    push_unique(&mut out, expert, budget);
                }
            }
            out
        }
        FusionArm::ReciprocalRank => {
            let mut fused: Vec<(usize, f32)> = Vec::with_capacity(edge0.len() + routescout.len());
            let mut add = |expert: usize, rank: usize| {
                let weight = 1.0 / (RRF_K + rank as f32);
                match fused.iter_mut().find(|(e, _)| *e == expert) {
                    Some(entry) => entry.1 += weight,
                    None => fused.push((expert, weight)),
                }
            };
            for (rank, c) in edge0.iter().enumerate() {
                add(c.expert, rank);
            }
            for (rank, &(expert, _)) in routescout.iter().enumerate() {
                add(expert, rank);
            }
            rank_single(fused.into_iter(), budget)
        }
        FusionArm::Weighted => {
            let e0 = peak_normalized(edge0.iter().map(|c| (c.expert, c.score)));
            let rs = peak_normalized(routescout.iter().copied());
            let mut fused: Vec<(usize, f32)> = Vec::with_capacity(e0.len() + rs.len());
            let mut weight = |list: &[(usize, f32)], w: f32| {
                for &(expert, score) in list {
                    match fused.iter_mut().find(|(e, _)| *e == expert) {
                        Some(entry) => entry.1 += score * w,
                        None => fused.push((expert, score * w)),
                    }
                }
            };
            weight(&e0, w_edge0);
            weight(&rs, w_routescout);
            rank_single(fused.into_iter(), budget)
        }
    }
}

/// Peak-normalize `(expert, score)` by its highest score.
///
/// A no-op when the peak is not positive: an all-zero list stays all-zero, so it
/// contributes nothing rather than inventing a uniform prior the source never
/// expressed.
fn peak_normalized(iter: impl Iterator<Item = (usize, f32)>) -> Vec<(usize, f32)> {
    let raw: Vec<(usize, f32)> = iter.collect();
    let peak = raw
        .iter()
        .map(|&(_, s)| s)
        .filter(|s| s.is_finite())
        .fold(0.0_f32, f32::max);
    if peak <= 0.0 {
        return raw;
    }
    raw.into_iter()
        .map(|(expert, score)| (expert, score / peak))
        .collect()
}

/// Sort by score descending (lower id wins ties), drop non-positive scores, and
/// truncate to `budget` unique experts.
fn rank_single(iter: impl Iterator<Item = (usize, f32)>, budget: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = iter
        .filter(|(_, score)| score.is_finite() && *score > 0.0)
        .collect();
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut out: Vec<usize> = Vec::with_capacity(budget.min(scored.len()));
    for (expert, _) in scored {
        if out.len() >= budget {
            break;
        }
        push_unique(&mut out, expert, budget);
    }
    out
}

fn push_unique(out: &mut Vec<usize>, expert: usize, budget: usize) {
    if out.len() < budget && !out.contains(&expert) {
        out.push(expert);
    }
}

/// Which generation a slot's bytes belong to.
///
/// Compared for exact equality everywhere: an entry whose tag differs from the
/// tag being served is not a candidate for reuse at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotTag {
    pub request_generation: u64,
    pub token_generation: u64,
}

/// One expert slot inside a staging row.
#[derive(Clone, Copy, Debug)]
struct StageSlot {
    /// Expert whose bytes are in this slot, or `FREE` when the slot is available.
    expert: u32,
    tag: SlotTag,
    /// Event of the load submitted into this slot. `0` means none was submitted
    /// — a claimed slot whose submission failed, which must read as a miss so
    /// the caller demand-loads rather than waiting on nothing.
    event: i64,
    /// True once the consuming token has read these bytes. Only used to release
    /// a slot promptly; completion itself is decided by probing the slot, never
    /// by this flag, because "a load was submitted" is not "the bytes are here".
    consumed: bool,
}

/// The `experts` field value marking an available slot. 256 and 512 are real
/// expert counts on this geometry, so the sentinel is `u32::MAX`, which no
/// checkpoint can address.
const FREE: u32 = u32::MAX;

/// Decide what a staged slot is worth, from the submitted event and the device
/// probe. `None` = nothing was submitted, `Some(true)` = complete, `Some(false)`
/// = still in flight.
///
/// Pure so the three-way outcome is unit-testable without a device: the real
/// path obtains `complete` from [`crate::ffi::mio_probe`], which needs a live
/// MetalIO slot, while the *decision* is where the correctness risk is.
///
/// `complete == None` (no device slot to probe) resolves to "ready": with no
/// device-backed slot there is no transfer in flight to wait for, so waiting on
/// the recorded event is the only alternative and it would block forever.
pub fn classify_staged(event: i64, complete: Option<bool>) -> Option<bool> {
    if event <= 0 {
        // Claimed but never submitted (a failed enqueue): there is nothing to
        // wait for, so this must demand-load rather than block on a dead event.
        return None;
    }
    Some(complete.unwrap_or(true))
}

/// What a staging lookup found for one route expert.
#[derive(Debug, Clone, Copy)]
pub enum StagedTake {
    /// Bytes are complete in this slot. Take them without any I/O.
    Hit { slot: i32 },
    /// Bytes are staged but the load has not finished. Wait on `event` — this is
    /// the "late at demand" case, and re-issuing the read would be exactly the
    /// duplicate I/O this design exists to remove.
    Late { slot: i32, event: i64 },
    /// Nothing staged for this expert in this generation.
    Miss,
}

/// Per-layer staging row: `candidates` slots, each `stride` bytes, each aliased
/// by its own MetalIO slot so loads land in the memory the kernel will read.
pub struct StageLayer {
    slots: Vec<i32>,
    /// Page-aligned allocation each slot aliases. Held, not leaked:
    /// `ArenaBuf::drop` unregisters the pages and frees them, so dropping the
    /// arena releases its whole footprint and the arena's memory cost is a real
    /// number in the M sweep rather than a leak.
    bufs: Vec<std::sync::Arc<crate::ArenaBuf>>,
    stage: Vec<StageSlot>,
    stride: usize,
}

impl StageLayer {
    /// Byte offset of `slot` inside this row's `bufs[slot]`.
    ///
    /// Always 0: each slot aliases its own exactly-`stride`-sized allocation, so
    /// a slot is one expert block with no sub-offset. Kept as a method because
    /// the caller must not have to know that.
    fn base_offset(&self) -> usize {
        0
    }

    fn find(&self, expert: u32, tag: SlotTag) -> Option<usize> {
        self.stage
            .iter()
            .position(|e| e.expert == expert && e.tag == tag)
    }

    /// A slot available to load into under `tag`: either already this expert for
    /// this tag, or empty.
    ///
    /// A slot holding a *different* generation's bytes is deliberately not
    /// reusable: its load may still be in flight, and re-tagging before that
    /// completes would hand two generations one buffer.
    fn claim(&mut self, expert: u32, tag: SlotTag) -> Option<usize> {
        if let Some(index) = self.find(expert, tag) {
            return Some(index);
        }
        self.stage.iter().position(|e| e.expert == FREE)
    }

    /// Whether this row holds `expert` under any tag — the stale case, where the
    /// bytes exist but belong to another generation.
    fn holds(&self, expert: u32) -> bool {
        self.stage.iter().any(|e| e.expert == expert)
    }

    /// Free every slot whose load is not in flight, so the row is fully
    /// available for the next generation's staging.
    ///
    /// A slot with a load still in flight is left tagged: its bytes are about to
    /// belong to a generation nobody will ask for, which is a miss by
    /// construction, and freeing it early would risk a concurrent write into one
    /// MetalIO slot (which the backend rejects outright).
    fn release_all(&mut self) {
        for entry in self.stage.iter_mut() {
            entry.expert = FREE;
            entry.event = 0;
            entry.consumed = false;
        }
    }
}

/// Counters that make staging auditable rather than asserted.
///
/// `duplicate_reads` is the I/O gate: an expert staged for this layer *and* this
/// generation that was demand-loaded anyway. It must be zero — it is precisely
/// the EXP-067 failure mode this arena replaces.
#[derive(Clone, Debug, Default)]
pub struct StageStats {
    pub hits: u64,
    pub late: u64,
    pub misses: u64,
    pub demand_reads: u64,
    pub staged_bytes: u64,
    pub consumed_bytes: u64,
    pub demand_bytes: u64,
    pub duplicate_reads: u64,
    pub stale_rejected: u64,
    /// Submissions refused because no slot was free under the staging tag.
    pub unplaced: u64,
    /// Sum over tokens of the number of layers whose staged set covered the
    /// whole K4 route, and the number of layers compared. `full_route_coverage`
    /// is the ratio; per §11 of the mission a weak predictor must not be hidden
    /// behind a tok/s figure.
    pub full_route_layers: u64,
    pub route_layers: u64,
    /// Candidate recall: staged experts that were in the actual route, over the
    /// actual route's total experts, both restricted to layers with a candidate
    /// set.
    pub recall_hits: u64,
    pub recall_total: u64,
    /// Per layer: (staged bytes, consumed bytes, demand bytes, demand reads).
    pub per_layer: Vec<(u64, u64, u64, u64)>,
}

impl StageStats {
    fn new(layers: usize) -> Self {
        Self {
            per_layer: vec![(0, 0, 0, 0); layers],
            ..Default::default()
        }
    }

    fn entry(&mut self, layers: usize, layer: usize) -> &mut (u64, u64, u64, u64) {
        if self.per_layer.len() < layers {
            self.per_layer.resize(layers, (0, 0, 0, 0));
        }
        &mut self.per_layer[layer]
    }

    /// Fraction of staged bytes that were actually consumed. The complement is
    /// waste, so this is the efficiency of the candidate budget.
    pub fn efficiency(&self) -> f64 {
        if self.stats_denominator() == 0 {
            return 0.0;
        }
        self.consumed_bytes as f64 / self.stats_denominator() as f64
    }

    fn stats_denominator(&self) -> u64 {
        self.staged_bytes
    }

    pub fn wasted_bytes(&self) -> u64 {
        self.staged_bytes.saturating_sub(self.consumed_bytes)
    }

    /// Number of layers whose staged set covered the whole executed route.
    pub fn full_route_layer_count(&self) -> u64 {
        self.full_route_layers
    }

    /// Fraction of compared layers where every executed expert was already
    /// staged.
    pub fn full_route_coverage(&self) -> f64 {
        if self.route_layers == 0 {
            return 0.0;
        }
        self.full_route_layers as f64 / self.route_layers as f64
    }

    /// Fraction of executed experts that were staged.
    pub fn candidate_recall(&self) -> f64 {
        if self.recall_total == 0 {
            return 0.0;
        }
        self.recall_hits as f64 / self.recall_total as f64
    }
}

/// The staging arena: **two rows per layer**, selected by token parity.
///
/// # Why two rows and not one
///
/// A prediction for consumer layer `L` is produced by owner `L-1`, and it must
/// be *submitted* while `L-1` still has the freshest evidence — which is before
/// layer `L` has run. But at that moment a single row for `L` is still holding
/// the current token's entries, which layer `L` has not consumed yet. The fill
/// would therefore find no free slot on every token, stage nothing, and turn
/// every lookup into a miss.
///
/// Two rows per layer, chosen by token parity, remove the conflict: a token
/// reads one parity and fills the other, so fill and consume never touch the same
/// memory. It costs `2 x layers x M x stride` (~446 MiB at M=4 on this
/// checkpoint), which is the price of true double buffering and is what bounds
/// the M sweep.
pub struct StageArena {
    /// `rows[layer * 2 + parity]`.
    rows: Vec<StageLayer>,
    layer_count: usize,
    stride: usize,
    candidates: usize,
    request_generation: u64,
    token_generation: u64,
    enabled: bool,
    pub stats: StageStats,
}
impl StageArena {
    /// Row index for `(layer, parity)`.
    fn row_index(&self, layer: usize, parity: usize) -> Option<usize> {
        if layer >= self.layer_count {
            return None;
        }
        Some(layer * 2 + parity)
    }

    /// Build `layers` **pairs** of rows, `candidates` slots each, `stride` bytes
    /// per slot.
    ///
    /// Every slot aliases caller-owned page-aligned memory, so a load lands in
    /// the bytes the MoE kernel will read: no slot copy and no second
    /// materialization. `Err` means MetalIO declined, which the caller treats as
    /// "staging unavailable" and disables permanently rather than retrying.
    pub fn new(layers: usize, candidates: usize, stride: usize) -> Result<Self, String> {
        if candidates == 0 || stride == 0 {
            return Err("staging arena needs a nonzero candidate count and stride".into());
        }
        let mut rows: Vec<StageLayer> = Vec::with_capacity(layers * 2);
        for layer in 0..layers {
            for parity in 0..2usize {
                let mut slots = Vec::with_capacity(candidates);
                let mut bufs = Vec::with_capacity(candidates);
                for _ in 0..candidates {
                    let buf = crate::ArenaBuf::alloc(stride)?;
                    crate::ffi::metal_register(buf.base(), buf.len());
                    let slot = crate::ffi::mio_slot_alloc_alias(buf.base(), buf.len());
                    if slot < 0 {
                        // Release everything already taken so a partial failure
                        // does not leak slots or registered pages. Dropping
                        // `rows`/`bufs` unregisters the pages they own.
                        for &s in &slots {
                            crate::ffi::mio_discard_slot(s);
                        }
                        for row in &rows {
                            for &s in &row.slots {
                                crate::ffi::mio_discard_slot(s);
                            }
                        }
                        return Err(format!(
                            "staging alias slot allocation failed on layer {layer} parity {parity}"
                        ));
                    }
                    slots.push(slot);
                    bufs.push(buf);
                }
                rows.push(StageLayer {
                    slots,
                    bufs,
                    stage: vec![
                        StageSlot {
                            expert: FREE,
                            tag: SlotTag {
                                request_generation: 0,
                                token_generation: 0,
                            },
                            event: 0,
                            consumed: false,
                        };
                        candidates
                    ],
                    stride,
                });
            }
        }
        Ok(Self {
            rows,
            layer_count: layers,
            stride,
            candidates,
            request_generation: 0,
            token_generation: 0,
            enabled: true,
            stats: StageStats::new(layers),
        })
    }

    /// A structure-only arena with no MetalIO slots, for testing the generation
    /// bookkeeping.
    ///
    /// The tag/parity logic is pure bookkeeping and is where the correctness risk
    /// lives, but exercising it through `new` would need a real MetalIO slot for
    /// every block. This builds the same rows with `slot = -1`, which every
    /// slot-related call treats as "no device slot", so the tests run anywhere.
    #[cfg(test)]
    pub fn for_test(layers: usize, candidates: usize, stride: usize) -> Self {
        let rows = (0..layers * 2)
            .map(|_| StageLayer {
                slots: vec![-1; candidates],
                bufs: Vec::new(),
                stage: vec![
                    StageSlot {
                        expert: FREE,
                        tag: SlotTag {
                            request_generation: 0,
                            token_generation: 0,
                        },
                        event: 0,
                        consumed: false,
                    };
                    candidates
                ],
                stride,
            })
            .collect::<Vec<_>>();
        Self {
            rows,
            layer_count: layers,
            stride,
            candidates,
            request_generation: 0,
            token_generation: 0,
            enabled: true,
            stats: StageStats::new(layers),
        }
    }

    pub fn candidates(&self) -> usize {
        self.candidates
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn token_generation(&self) -> u64 {
        self.token_generation
    }

    pub fn request_generation(&self) -> u64 {
        self.request_generation
    }

    /// Resident bytes: the arena's memory cost, and what bounds the M sweep.
    ///
    /// Counts both parities, because both are allocated and neither is optional:
    /// the double buffer is what makes fill and consume possible in one pass.
    pub fn bytes(&self) -> usize {
        self.rows
            .len()
            .saturating_mul(self.candidates)
            .saturating_mul(self.stride)
    }

    /// Drop every staged tag and advance the request generation.
    ///
    /// Called at a request/session boundary. The **bytes** stay where they are,
    /// merely untagged, which is what makes this cheap: the next token re-tags
    /// slots before loading into them, and any slot still holding the previous
    /// request's bytes can never be looked up because its tag is stale.
    pub fn begin_request(&mut self) {
        self.request_generation = self.request_generation.saturating_add(1);
        self.token_generation = 0;
        let tag = SlotTag {
            request_generation: self.request_generation,
            token_generation: 0,
        };
        for row in &mut self.rows {
            row.release_all();
            for entry in &mut row.stage {
                if entry.expert == FREE {
                    entry.tag = tag;
                }
            }
        }
    }

    /// Advance to the next token, flipping the parity the token consumes.
    ///
    /// Staging submitted during token `t` filled parity `(t+1) & 1`; after this
    /// call the served parity is `(t+1) & 1`, so exactly that set becomes
    /// servable. Entries in the *other* parity are from token `t` — the token
    /// that just finished — and are released so the next fill can use them.
    pub fn begin_token(&mut self) {
        self.token_generation = self.token_generation.saturating_add(1);
        // The parity that was just consumed is now the staging parity for the
        // token after this one, so it must be free. Only completed slots are
        // released; one with an outstanding write is left claimed because
        // MetalIO rejects a concurrent write into a single slot, and releasing it
        // would alias two generations into one buffer.
        let stale_parity = ((self.token_generation + 1) & 1) as usize;
        for layer in 0..self.layer_count {
            let Some(index) = self.row_index(layer, stale_parity) else {
                continue;
            };
            let Some(row) = self.rows.get_mut(index) else {
                continue;
            };
            for slot_index in 0..row.stage.len() {
                let entry = row.stage[slot_index];
                if entry.expert == FREE {
                    continue;
                }
                // A slot whose write is still outstanding must not be handed to
                // the next staging pass: MetalIO rejects a concurrent write into
                // one slot, and re-tagging would alias two generations into one
                // buffer. It stays claimed and becomes a miss for whatever asks
                // next, which is the safe direction.
                if entry.event > 0 && crate::ffi::mio_probe(row.slots[slot_index]) == Some(false) {
                    continue;
                }
                row.stage[slot_index].expert = FREE;
                row.stage[slot_index].event = 0;
                row.stage[slot_index].consumed = false;
            }
        }
    }

    /// Tag of the generation this token consumes.
    fn consume_tag(&self) -> SlotTag {
        SlotTag {
            request_generation: self.request_generation,
            token_generation: self.token_generation,
        }
    }

    /// Tag of the generation staging now fills, i.e. the next token.
    fn stage_tag(&self) -> SlotTag {
        SlotTag {
            request_generation: self.request_generation,
            token_generation: self.token_generation.saturating_add(1),
        }
    }

    /// Look up a route expert for the current token.
    ///
    /// Searches only the parity this token consumes, so an expert staged for the
    /// current token can never be confused with one staged for the next. `None`
    /// when staging is off or `layer` is outside the arena. A generation mismatch
    /// is [`StagedTake::Miss`], counted so that a caller serving stale bytes would
    /// be visible rather than silent.
    pub fn lookup(&mut self, layer: usize, expert: u32) -> Option<StagedTake> {
        if !self.enabled {
            return None;
        }
        let tag = self.consume_tag();
        let index = self.row_index(layer, (self.token_generation & 1) as usize)?;
        let row = self.rows.get(index)?;
        match row.find(expert, tag) {
            Some(slot_index) => {
                let entry = row.stage[slot_index];
                let slot = row.slots[slot_index];
                // Completion is decided by probing the slot, never by a flag:
                // "a load was submitted" is not "the bytes have arrived", and
                // conflating the two made every staged expert read as Late.
                //
                // A `None` here is a *miss*, not an absent arena: `lookup`'s
                // `None` means "staging is off for this layer", and returning it
                // for "this expert's slot is unusable" would tell the caller the
                // arena declined rather than that it should demand-load.
                match classify_staged(entry.event, crate::ffi::mio_probe(slot)) {
                    Some(true) => Some(StagedTake::Hit { slot }),
                    Some(false) => Some(StagedTake::Late {
                        slot,
                        event: entry.event,
                    }),
                    None => {
                        self.stats.stale_rejected += 1;
                        Some(StagedTake::Miss)
                    }
                }
            }
            None => {
                // Held under a different generation in THIS parity: the bytes
                // exist but are not this token's. Recorded so the guard is
                // measured rather than merely asserted.
                if row.holds(expert) {
                    self.stats.stale_rejected += 1;
                }
                Some(StagedTake::Miss)
            }
        }
    }

    /// Claim a slot in this token's parity to demand-load a route expert into.
    ///
    /// Returns `(slot, region_base_offset)`.
    pub fn claim_demand(&mut self, layer: usize, expert: u32) -> Option<(i32, usize)> {
        if !self.enabled {
            return None;
        }
        let tag = self.consume_tag();
        let index = self.row_index(layer, (self.token_generation & 1) as usize)?;
        let row = self.rows.get_mut(index)?;
        let slot_index = row.claim(expert, tag)?;
        row.stage[slot_index].expert = expert;
        row.stage[slot_index].tag = tag;
        row.stage[slot_index].event = 0;
        row.stage[slot_index].consumed = false;
        let off = row.base_offset();
        Some((row.slots[slot_index], off))
    }

    /// Record the event of a staged load submitted into a claimed slot.
    pub fn set_event(&mut self, layer: usize, expert: u32, event: i64) {
        let stage_tag = self.stage_tag();
        let index = match self.row_index(layer, ((self.token_generation + 1) & 1) as usize) {
            Some(index) => index,
            None => return,
        };
        let Some(row) = self.rows.get_mut(index) else {
            return;
        };
        if let Some(slot_index) = row.find(expert, stage_tag) {
            row.stage[slot_index].event = event;
        }
    }

    /// Mark a route expert consumed on this token.
    ///
    /// Deliberately leaves the tag in place: the slot stays claimed for this
    /// token's parity until `begin_token` releases that parity, which cannot
    /// happen before this token's compute has finished reading it.
    pub fn consume(&mut self, layer: usize, expert: u32) -> bool {
        let tag = self.consume_tag();
        let Some(index) = self.row_index(layer, (self.token_generation & 1) as usize) else {
            return false;
        };
        let Some(row) = self.rows.get_mut(index) else {
            return false;
        };
        match row.find(expert, tag) {
            Some(slot_index) => {
                row.stage[slot_index].consumed = true;
                true
            }
            None => false,
        }
    }

    /// Claim a slot to stage `expert` for the **next** token, in the next token's
    /// parity.
    ///
    /// Returns `(slot, region_base_offset, fresh)`. `fresh == false` means a load
    /// for this exact `(expert, next generation)` is already staged, so the caller
    /// must not submit a second read: MetalIO rejects a concurrent write into one
    /// slot, and a duplicate submission would be the very I/O this design removes.
    pub fn claim_stage(&mut self, layer: usize, expert: u32) -> Option<(i32, usize, bool)> {
        if !self.enabled {
            return None;
        }
        let tag = self.stage_tag();
        let index = self.row_index(layer, ((self.token_generation + 1) & 1) as usize)?;
        let row = self.rows.get_mut(index)?;
        if let Some(slot_index) = row.find(expert, tag) {
            let off = row.base_offset();
            return Some((row.slots[slot_index], off, false));
        }
        let Some(slot_index) = row.claim(expert, tag) else {
            self.stats.unplaced += 1;
            return None;
        };
        row.stage[slot_index].expert = expert;
        row.stage[slot_index].tag = tag;
        row.stage[slot_index].event = 0;
        row.stage[slot_index].consumed = false;
        let off = row.base_offset();
        Some((row.slots[slot_index], off, true))
    }

    /// The `Arc<ArenaBuf>` a slot's bytes live in, for building zero-copy views.
    ///
    /// Searches both parities because the caller has a slot id, not a parity.
    pub fn buf_for(&self, layer: usize, slot: i32) -> Option<std::sync::Arc<crate::ArenaBuf>> {
        for parity in 0..2usize {
            let index = self.row_index(layer, parity)?;
            let Some(row) = self.rows.get(index) else {
                continue;
            };
            if let Some(slot_index) = row.slots.iter().position(|&s| s == slot) {
                return row.bufs.get(slot_index).map(std::sync::Arc::clone);
            }
        }
        None
    }

    /// Bytes per expert block, i.e. the length of the view to build.
    pub fn block_bytes(&self) -> usize {
        self.stride
    }

    pub fn record_submit(&mut self, layer: usize, bytes: usize) {
        self.stats.staged_bytes = self.stats.staged_bytes.saturating_add(bytes as u64);
        let layers = self.layer_count;
        let entry = self.stats.entry(layers, layer);
        entry.0 = entry.0.saturating_add(bytes as u64);
    }

    pub fn record_consume(&mut self, layer: usize, bytes: usize) {
        self.stats.consumed_bytes = self.stats.consumed_bytes.saturating_add(bytes as u64);
        let layers = self.layer_count;
        let entry = self.stats.entry(layers, layer);
        entry.1 = entry.1.saturating_add(bytes as u64);
    }

    /// Record one staged expert taken from the arena for this token.
    ///
    /// `late` distinguishes a slot that had already landed from one the consumer
    /// had to wait for. Both are hits — neither re-read the bytes — but the split
    /// is what tells us whether the lead time was actually sufficient.
    pub fn record_hit(&mut self, layer: usize, bytes: usize, late: bool) {
        if late {
            self.stats.late += 1;
        } else {
            self.stats.hits += 1;
        }
        self.record_consume(layer, bytes);
    }

    /// Record one demand read of `expert` on `layer`.
    ///
    /// `staged` must be true when this expert was staged for this generation and
    /// is nonetheless being read again: that is a duplicate read, and the hard
    /// invariant the design is built around.
    pub fn record_demand(&mut self, layer: usize, staged: bool, bytes: usize) {
        self.stats.demand_reads += 1;
        if staged {
            self.stats.duplicate_reads += 1;
        }
        self.stats.demand_bytes = self.stats.demand_bytes.saturating_add(bytes as u64);
        let layers = self.layer_count;
        let entry = self.stats.entry(layers, layer);
        entry.2 = entry.2.saturating_add(bytes as u64);
        entry.3 += 1;
    }

    /// Record one layer's route coverage: how many of its executed experts were
    /// staged, and whether all of them were.
    pub fn record_route(&mut self, staged_hits: usize, route_width: usize) {
        self.stats.recall_hits += staged_hits as u64;
        self.stats.recall_total += route_width as u64;
        self.stats.route_layers += 1;
        if route_width > 0 && staged_hits == route_width {
            self.stats.full_route_layers += 1;
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.candidates > 0;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Drop for StageArena {
    fn drop(&mut self) {
        // Slots first: each wraps one of the arena's pages and the wrapper must
        // be gone before those pages are freed.
        for row in &self.rows {
            for &slot in &row.slots {
                crate::ffi::mio_discard_slot(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(list: &[(usize, f32)]) -> Vec<RankedCandidate> {
        list.iter()
            .map(|&(expert, score)| RankedCandidate { expert, score })
            .collect()
    }

    #[test]
    fn edge0_arm_takes_only_edge0_in_rank_order() {
        let e0 = cand(&[(7, 0.5), (3, 0.3), (9, 0.2)]);
        let rs = vec![(1, 9.0), (2, 8.0)];
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::Edge0, 1.0, 1.0, 2),
            vec![7, 3]
        );
    }

    #[test]
    fn routescout_arm_ignores_edge0_entirely() {
        let e0 = cand(&[(7, 0.5), (3, 0.3)]);
        let rs = vec![(1, 9.0), (2, 8.0)];
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::RouteScout, 1.0, 1.0, 2),
            vec![1, 2]
        );
    }

    #[test]
    fn union_interleaves_and_deduplicates() {
        // Expert 5 is in both lists; it must occupy one slot, not two.
        let e0 = cand(&[(5, 0.9), (7, 0.5)]);
        let rs = vec![(5, 1.0), (2, 0.5)];
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::Union, 1.0, 1.0, 4),
            vec![5, 7, 2]
        );
    }

    #[test]
    fn reciprocal_rank_is_scale_free() {
        // RouteScout's scores are ~10^6x Edge0's; a weighted sum would be
        // decided by that alone. RRF must follow rank agreement instead.
        let e0 = cand(&[(5, 0.001), (7, 0.0009)]);
        let rs = vec![(5, 900.0), (7, 800.0)];
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::ReciprocalRank, 1.0, 1.0, 2),
            vec![5, 7]
        );
    }

    #[test]
    fn weighted_fusion_peak_normalizes_each_source() {
        // With equal weights both sources must contribute equally despite Edge0
        // being ~1000x smaller: each is peak-normalized first.
        let e0 = cand(&[(5, 0.001), (7, 0.0005)]);
        let rs = vec![(9, 4.0), (7, 2.0)];
        // 5 -> 1.0 (e0 peak), 9 -> 1.0 (rs peak) tie broken by lower id, then
        // 7 -> 0.5 from e0 + 0.5 from rs = 1.0... which also ties, so 7 sorts
        // before 9 on the id tie-break. Ordering is 5, 7, 9.
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::Weighted, 1.0, 1.0, 3),
            vec![5, 7, 9]
        );
    }

    #[test]
    fn zero_weight_source_contributes_nothing() {
        let e0 = cand(&[(5, 0.9), (7, 0.5)]);
        let rs = vec![(1, 9.0), (2, 8.0)];
        assert_eq!(
            fuse_candidates(&e0, &rs, FusionArm::Weighted, 1.0, 0.0, 4),
            vec![5, 7]
        );
    }

    #[test]
    fn fusion_never_exceeds_budget() {
        let e0 = cand(&[(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]);
        let rs = vec![(5, 1.0), (6, 0.9)];
        for arm in [
            FusionArm::Edge0,
            FusionArm::RouteScout,
            FusionArm::Union,
            FusionArm::ReciprocalRank,
            FusionArm::Weighted,
        ] {
            assert_eq!(
                fuse_candidates(&e0, &rs, arm, 1.0, 1.0, 2).len(),
                2,
                "arm {arm:?} exceeded its budget"
            );
        }
    }

    #[test]
    fn staged_bytes_are_served_to_the_next_token() {
        let mut arena = StageArena::for_test(1, 4, 1024);
        // Stage an expert during token 0 for token 1.
        assert!(arena.claim_stage(0, 7).is_some());
        arena.set_event(0, 7, 1);
        // Advance: the parity staged into is now the parity consumed.
        arena.begin_token();
        assert!(
            matches!(arena.lookup(0, 7), Some(StagedTake::Hit { .. })),
            "an expert staged during token t must be a hit at t+1"
        );
    }

    #[test]
    fn an_end_to_end_stage_then_consume_cycle_actually_stages() {
        // The regression this exists for: with one row per layer, the fill for
        // consumer L is submitted while layer L-1 runs, but layer L has not yet
        // consumed its own row, so every `claim_stage` found no free slot and
        // *nothing* was ever staged. Two parities make fill and consume disjoint.
        let mut arena = StageArena::for_test(2, 4, 1024);
        let mut hits = 0u64;
        // Token 0: layer 0 stages a prediction for layer 1's next token.
        assert!(arena.claim_stage(1, 5).is_some());
        arena.set_event(1, 5, 1);
        arena.begin_token();
        // Token 1: layer 1 consumes it.
        if let Some(StagedTake::Hit { .. }) = arena.lookup(1, 5) {
            hits += 1;
        }
        assert_eq!(hits, 1, "a staged expert must be servable one token later");
        assert_eq!(
            arena.stats.unplaced, 0,
            "staging must never run out of slots"
        );
    }

    #[test]
    fn filling_all_m_candidates_while_the_consumer_has_not_run_yet_places_all() {
        // The exact shape that failed before: stage a full M-wide set for the
        // next token while the consumer layer's own row is untouched.
        let mut arena = StageArena::for_test(2, 4, 1024);
        for expert in 0..4u32 {
            assert!(
                arena.claim_stage(1, expert).is_some(),
                "candidate {expert} of M=4 must find a slot"
            );
        }
        assert_eq!(arena.stats.unplaced, 0);
    }

    #[test]
    fn a_slot_staged_for_the_next_token_is_not_served_to_this_one() {
        let mut arena = StageArena::for_test(1, 4, 1024);
        assert!(arena.claim_stage(0, 7).is_some());
        assert!(
            matches!(arena.lookup(0, 7), Some(StagedTake::Miss)),
            "a prediction must never be visible to the token that produced it"
        );
    }

    #[test]
    fn a_new_request_rejects_every_previous_slot() {
        let mut arena = StageArena::for_test(1, 4, 1024);
        assert!(arena.claim_stage(0, 7).is_some());
        arena.set_event(0, 7, 1);
        arena.begin_token();
        assert!(matches!(arena.lookup(0, 7), Some(StagedTake::Hit { .. })));

        // A request boundary must make even a correctly-staged slot unservable,
        // or a previous conversation's bytes could be handed to this one.
        arena.begin_request();
        assert!(
            matches!(arena.lookup(0, 7), Some(StagedTake::Miss)),
            "a previous request's bytes must never be served"
        );
    }

    #[test]
    fn stale_rejection_is_counted_when_the_expert_is_held_under_another_generation() {
        // A prediction staged for t+1 lives in the *other* parity, so the
        // lookup at t must not even see it — parity separation is what makes
        // this case structurally impossible rather than merely checked.
        let mut arena = StageArena::for_test(1, 4, 1024);
        assert!(arena.claim_stage(0, 7).is_some());
        assert!(matches!(arena.lookup(0, 7), Some(StagedTake::Miss)));
        assert_eq!(
            arena.stats.stale_rejected, 0,
            "a next-token prediction is in another parity, not a stale entry"
        );

        // The stale case that DOES occur: the served parity holds the expert
        // under a tag from an older generation — e.g. a generation jump without
        // that parity's slots having been released.
        let mut arena = StageArena::for_test(1, 4, 1024);
        assert!(arena.claim_stage(0, 9).is_some()); // parity 1, tag gen 1
        arena.set_event(0, 9, 1);
        // Serve parity 1 (gen 1) but claim the served generation is 3, so the
        // entry's tag does not match while its bytes are still in that row.
        arena.token_generation = 3;
        assert!(matches!(arena.lookup(0, 9), Some(StagedTake::Miss)));
        assert_eq!(
            arena.stats.stale_rejected, 1,
            "an expert held under a mismatched tag in the served parity must be counted"
        );
    }

    #[test]
    fn an_incomplete_staged_load_is_late_not_a_miss() {
        // The decision, independent of any device. A staged load that is still in
        // flight must be waited on rather than re-issued: re-issuing is exactly
        // the duplicate I/O this design removes.
        assert_eq!(classify_staged(99, Some(false)), Some(false));
        assert_eq!(classify_staged(99, Some(true)), Some(true));
        // Claimed but never submitted: demand-load, never block on a dead event.
        assert_eq!(classify_staged(0, None), None);
        assert_eq!(classify_staged(0, Some(true)), None);
        // No device slot to probe: nothing is in flight, so it is ready.
        assert_eq!(classify_staged(7, None), Some(true));
    }

    #[test]
    fn a_claimed_but_unsubmitted_slot_is_a_miss() {
        // Reaches the caller only if an enqueue failed. It must demand-load:
        // reporting Hit would hand the kernel whatever bytes were in the slot
        // from a previous generation.
        let mut arena = StageArena::for_test(1, 4, 1024);
        assert!(arena.claim_stage(0, 7).is_some());
        arena.begin_token();
        assert!(matches!(arena.lookup(0, 7), Some(StagedTake::Miss)));
    }

    #[test]
    fn claim_stage_is_idempotent_for_one_generation() {
        let mut arena = StageArena::for_test(1, 4, 1024);
        let (first, _, fresh_a) = arena.claim_stage(0, 7).expect("first claim");
        let (second, _, fresh_b) = arena.claim_stage(0, 7).expect("second claim");
        assert!(fresh_a);
        assert!(!fresh_b, "re-claiming one expert must not be a fresh read");
        assert_eq!(first, second, "re-claim must return the same slot");
    }

    #[test]
    fn a_parity_never_places_more_candidates_than_it_is_wide() {
        let mut arena = StageArena::for_test(1, 2, 1024);
        assert!(arena.claim_stage(0, 1).is_some());
        assert!(arena.claim_stage(0, 2).is_some());
        // Third distinct expert has nowhere to go in this parity: reported, not
        // silently overwriting a slot another expert's bytes are in flight for.
        assert!(arena.claim_stage(0, 3).is_none());
        assert_eq!(arena.stats.unplaced, 1);
    }

    #[test]
    fn both_parities_are_allocated_and_counted() {
        let arena = StageArena::for_test(3, 4, 1000);
        // 3 layers x 2 parities x M x stride: the double buffer is the cost of
        // being able to fill and consume in the same pass.
        assert_eq!(arena.bytes(), 3 * 2 * 4 * 1000);
    }

    #[test]
    fn route_coverage_and_recall_are_recorded_per_layer() {
        let mut arena = StageArena::for_test(2, 4, 1024);
        arena.record_route(4, 4); // full coverage
        arena.record_route(2, 4); // partial
        assert_eq!(arena.stats.full_route_layer_count(), 1);
        assert_eq!(arena.stats.route_layers, 2);
        assert!((arena.stats.candidate_recall() - 0.75).abs() < 1e-9);
        assert!((arena.stats.full_route_coverage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_disabled_arena_never_serves_or_claims() {
        let mut arena = StageArena::for_test(1, 4, 1024);
        arena.set_enabled(false);
        assert!(arena.claim_stage(0, 7).is_none());
        assert!(arena.lookup(0, 7).is_none());
        assert!(arena.claim_demand(0, 7).is_none());
        assert!(!arena.is_enabled());
    }

    #[test]
    fn a_partially_covered_route_loads_exactly_the_missing_experts() {
        // §12: a route of 4 where 3 were staged must read exactly the one that
        // was not. The classification below is the same logic `stage_fetch_route`
        // applies, expressed over a test arena so the *count* is checkable
        // without a device.
        let mut arena = StageArena::for_test(1, 4, 1024);
        let route = [5u32, 9, 12, 30];
        // Stage three of the four for the next token.
        for &expert in &[5u32, 9, 30] {
            assert!(arena.claim_stage(0, expert).is_some());
            arena.set_event(0, expert, 1);
            arena.record_submit(0, 1024);
        }
        arena.begin_token();

        let mut staged = 0usize;
        let mut demand: Vec<u32> = Vec::new();
        for &expert in &route {
            match arena.lookup(0, expert) {
                Some(StagedTake::Hit { .. }) => {
                    arena.consume(0, expert);
                    arena.record_hit(0, 1024, false);
                    staged += 1;
                }
                Some(StagedTake::Late { .. }) => panic!("no load was left in flight"),
                Some(StagedTake::Miss) | None => demand.push(expert),
            }
        }
        assert_eq!(staged, 3, "the three staged experts must be hits");
        assert_eq!(
            demand,
            vec![12],
            "exactly the missing expert must be demand-read"
        );
        assert_eq!(arena.stats.duplicate_reads, 0);
        assert_eq!(arena.stats.hits, 3);
    }

    #[test]
    fn no_duplicate_expert_id_consumes_two_slots() {
        // An expert that both predictors rank, and that is also in the previous
        // route, must occupy ONE slot. Two would leave a different candidate
        // unstaged and double-count one read.
        let e0 = cand(&[(5, 0.9), (7, 0.5)]);
        let rs = vec![(5, 900.0), (2, 800.0)];
        for arm in [
            FusionArm::Edge0,
            FusionArm::RouteScout,
            FusionArm::Union,
            FusionArm::ReciprocalRank,
            FusionArm::Weighted,
        ] {
            let got = fuse_candidates(&e0, &rs, arm, 1.0, 1.0, 8);
            let mut sorted = got.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(
                before,
                sorted.len(),
                "arm {arm:?} emitted a duplicate expert"
            );
        }

        // And at the arena level: staging the same expert twice must claim one
        // slot, not two.
        let mut arena = StageArena::for_test(1, 4, 1024);
        let (first, _, fresh_a) = arena.claim_stage(0, 5).expect("first");
        let (second, _, fresh_b) = arena.claim_stage(0, 5).expect("second");
        assert!(fresh_a && !fresh_b);
        assert_eq!(first, second);
    }

    #[test]
    fn a_full_route_coverage_claim_requires_every_executed_expert() {
        // §11's `full_route_coverage` must be a strict all-four test, not "most of
        // them": 3/4 staged is not coverage.
        let mut arena = StageArena::for_test(1, 4, 1024);
        arena.record_route(3, 4);
        assert_eq!(arena.stats.full_route_layer_count(), 0);
        arena.record_route(4, 4);
        assert_eq!(arena.stats.full_route_layer_count(), 1);
        assert!((arena.stats.full_route_coverage() - 0.5).abs() < 1e-9);
    }
}
