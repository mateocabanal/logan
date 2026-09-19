use serde::{Deserialize, Serialize};

/// User intent for context planning, shared by compile and recompile.
///
/// `Maximum` means the optimizer may trade context away for another useful
/// Pareto point, but must never plan beyond `tokens`. `Required` makes
/// `tokens` a hard lower bound for every admissible plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextConstraintKind {
    Maximum,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextConstraint {
    pub kind: ContextConstraintKind,
    pub tokens: u64,
}

impl ContextConstraint {
    pub const fn maximum(tokens: u64) -> Self {
        Self {
            kind: ContextConstraintKind::Maximum,
            tokens,
        }
    }

    pub const fn required(tokens: u64) -> Self {
        Self {
            kind: ContextConstraintKind::Required,
            tokens,
        }
    }

    /// Validate the user constraint against the model's architectural ceiling.
    pub const fn valid_for_model(self, model_max_tokens: u64) -> bool {
        self.tokens > 0 && model_max_tokens > 0 && self.tokens <= model_max_tokens
    }

    /// Whether a candidate compiled context is admissible under this intent.
    pub const fn allows(self, compiled_max_tokens: u64) -> bool {
        if compiled_max_tokens == 0 {
            return false;
        }
        match self.kind {
            ContextConstraintKind::Maximum => compiled_max_tokens <= self.tokens,
            ContextConstraintKind::Required => compiled_max_tokens >= self.tokens,
        }
    }
}

/// Context-dependent runtime state. Keep architecture-specific components
/// separate so hybrid recurrent/attention models are not reduced to a generic
/// `bytes_per_token` estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextStateBytes {
    pub full_attention_kv: u64,
    pub gdn_recurrent: u64,
    pub gdn_conv: u64,
    pub qsa_index: u64,
    pub ple: u64,
    pub mtp_speculative: u64,
    pub other: u64,
}

impl ContextStateBytes {
    pub fn total(self) -> Option<u64> {
        [
            self.full_attention_kv,
            self.gdn_recurrent,
            self.gdn_conv,
            self.qsa_index,
            self.ple,
            self.mtp_speculative,
            self.other,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }

    pub fn total_bytes(self) -> u64 {
        self.total().unwrap_or(u64::MAX)
    }
}

/// The context portion of a selected physical deployment plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPlan {
    pub constraint: ContextConstraint,
    pub model_max_tokens: u64,
    pub compiled_max_tokens: u64,
    pub state_bytes: ContextStateBytes,
}

impl ContextPlan {
    pub fn is_valid(&self) -> bool {
        self.constraint.valid_for_model(self.model_max_tokens)
            && self.compiled_max_tokens <= self.model_max_tokens
            && self.constraint.allows(self.compiled_max_tokens)
            && self.state_bytes.total().is_some()
    }
}

/// Capacity accounting that precedes weight residency/cache optimization.
///
/// The optimizer may only spend `available_for_weights_and_cache()` on
/// optional residency/cache after fixed/context/scratch state and host safety
/// reserves are removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerMemoryBudget {
    pub physical_memory: u64,
    pub os_reserve: u64,
    pub runtime_reserve: u64,
    pub safety_reserve: u64,
    pub fixed_model_state: u64,
    pub context_state: ContextStateBytes,
    pub execution_scratch: u64,
}

impl PlannerMemoryBudget {
    pub fn reserved_bytes(self) -> Option<u64> {
        let context = self.context_state.total()?;
        [
            self.os_reserve,
            self.runtime_reserve,
            self.safety_reserve,
            self.fixed_model_state,
            context,
            self.execution_scratch,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }

    pub fn available_for_weights_and_cache(self) -> Option<u64> {
        self.physical_memory.checked_sub(self.reserved_bytes()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_context_admits_smaller_pareto_points_but_not_larger_ones() {
        let constraint = ContextConstraint::maximum(65_536);
        assert!(constraint.valid_for_model(262_144));
        assert!(constraint.allows(32_768));
        assert!(constraint.allows(65_536));
        assert!(!constraint.allows(131_072));
    }

    #[test]
    fn required_context_is_a_hard_lower_bound() {
        let constraint = ContextConstraint::required(65_536);
        assert!(constraint.valid_for_model(262_144));
        assert!(!constraint.allows(32_768));
        assert!(constraint.allows(65_536));
        assert!(constraint.allows(131_072));
    }

    #[test]
    fn requested_context_must_fit_the_model_ceiling() {
        assert!(!ContextConstraint::maximum(0).valid_for_model(262_144));
        assert!(!ContextConstraint::required(262_145).valid_for_model(262_144));
    }

    #[test]
    fn hybrid_context_state_keeps_components_distinct_and_checked() {
        let state = ContextStateBytes {
            full_attention_kv: 4_000,
            gdn_recurrent: 300,
            gdn_conv: 200,
            qsa_index: 500,
            ple: 100,
            mtp_speculative: 50,
            other: 25,
        };
        assert_eq!(state.total(), Some(5_175));
        assert_eq!(
            ContextStateBytes {
                full_attention_kv: u64::MAX,
                gdn_recurrent: 1,
                ..ContextStateBytes::default()
            }
            .total(),
            None
        );
    }

    #[test]
    fn context_state_reduces_the_budget_available_to_weights_and_cache() {
        let budget = PlannerMemoryBudget {
            physical_memory: 16_000,
            os_reserve: 2_000,
            runtime_reserve: 500,
            safety_reserve: 1_000,
            fixed_model_state: 1_000,
            context_state: ContextStateBytes {
                full_attention_kv: 3_000,
                gdn_recurrent: 250,
                gdn_conv: 250,
                qsa_index: 250,
                ..ContextStateBytes::default()
            },
            execution_scratch: 500,
        };
        assert_eq!(budget.reserved_bytes(), Some(8_750));
        assert_eq!(budget.available_for_weights_and_cache(), Some(7_250));
    }

    #[test]
    fn impossible_memory_budget_fails_closed() {
        let budget = PlannerMemoryBudget {
            physical_memory: 1_000,
            os_reserve: 900,
            runtime_reserve: 200,
            ..PlannerMemoryBudget::default()
        };
        assert_eq!(budget.available_for_weights_and_cache(), None);
    }
}
