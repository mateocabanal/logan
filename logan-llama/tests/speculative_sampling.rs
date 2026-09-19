use logan_llama::dspark::{
    ConditionalDistribution, DecodeRoute, Distribution, MarkovModel, SamplingError, SequenceRng,
    route_for_sampling, route_for_temperature, sample_speculative, supports_speculative_sampling,
};

fn dist(values: &[f32]) -> Distribution {
    Distribution::new(values).unwrap()
}

#[test]
fn tiny_distributions_use_exact_residual_and_acceptance_ratio() {
    let p = dist(&[0.2, 0.3, 0.5]);
    let q = dist(&[0.5, 0.25, 0.25]);
    assert!((p.acceptance_probability(&q, 0).unwrap() - 0.4).abs() < 1e-6);
    let residual = p.residual_distribution(&q).unwrap();
    assert!((residual.probability(0).unwrap() - 0.0).abs() < 1e-6);
    assert!((residual.probability(1).unwrap() - (1.0 / 6.0)).abs() < 1e-6);
    assert!((residual.probability(2).unwrap() - (5.0 / 6.0)).abs() < 1e-6);
}

#[test]
fn zero_proposal_probability_is_recovered_by_residual() {
    let p = dist(&[0.0, 0.8, 0.2]);
    let q = dist(&[1.0, 0.0, 0.0]);
    assert_eq!(p.acceptance_probability(&q, 1).unwrap(), 0.0);
    let mut rng = SequenceRng::new(vec![0.0, 0.9, 0.0]);
    let result = sample_speculative(
        &[],
        1,
        &|_: &[u32]| Ok(p.clone()),
        &|_: &[u32]| Ok(q.clone()),
        &mut rng,
    )
    .unwrap();
    assert_eq!(result.tokens, vec![1]);
    assert!(result.rejected);
}

#[test]
fn filtered_support_preserves_zero_mass_and_exact_correction() {
    let p =
        Distribution::from_logits_filtered(&[2.0, 1.0, 9.0], 1.0, &[true, true, false]).unwrap();
    let q =
        Distribution::from_logits_filtered(&[1.0, 2.0, 9.0], 1.0, &[true, false, true]).unwrap();
    assert_eq!(p.probability(2).unwrap(), 0.0);
    assert_eq!(q.probability(1).unwrap(), 0.0);
    let residual = p.residual_distribution(&q).unwrap();
    assert!(residual.probability(1).unwrap() > 0.0);
    assert!(residual.probability(2).unwrap() == 0.0);
}

#[test]
fn markov_proposal_uses_the_generated_prefix() {
    let proposal = MarkovModel::new(vec![8.0, 0.0], vec![vec![0.0, 8.0], vec![8.0, 0.0]]).unwrap();
    let initial = proposal.distribution(&[]).unwrap();
    let after_zero = proposal.distribution(&[0]).unwrap();
    assert!(initial.probability(0).unwrap() > 0.99);
    assert!(after_zero.probability(1).unwrap() > 0.99);
}

#[test]
fn repetition_penalty_is_recomputed_for_each_prefix() {
    let model = MarkovModel::new(vec![2.0, 1.0], vec![vec![2.0, 1.0], vec![2.0, 1.0]])
        .unwrap()
        .with_repetition_penalty(2.0)
        .unwrap();
    let before = model.distribution(&[]).unwrap();
    let after = model.distribution(&[0]).unwrap();
    assert!(before.probability(0).unwrap() > after.probability(0).unwrap());
}

#[test]
fn acceptance_boundaries_are_deterministic() {
    let p = dist(&[0.5, 0.5]);
    let q = dist(&[1.0, 0.0]);
    let mut accepted_rng = SequenceRng::new(vec![0.0, 0.5, 0.0]);
    let accepted = sample_speculative(
        &[],
        1,
        &|_: &[u32]| Ok(p.clone()),
        &|_: &[u32]| Ok(q.clone()),
        &mut accepted_rng,
    )
    .unwrap();
    assert!(!accepted.rejected);
    assert_eq!(accepted.accepted, 1);

    let mut rejected_rng = SequenceRng::new(vec![0.0, 0.500_001, 0.0]);
    let rejected = sample_speculative(
        &[],
        1,
        &|_: &[u32]| Ok(p.clone()),
        &|_: &[u32]| Ok(q.clone()),
        &mut rejected_rng,
    )
    .unwrap();
    assert!(rejected.rejected);
    assert_eq!(rejected.tokens, vec![1]);
}

#[test]
fn all_accepted_gets_a_target_bonus_token() {
    let p = dist(&[0.0, 1.0]);
    let q = dist(&[0.0, 1.0]);
    let mut rng = SequenceRng::new(vec![0.0, 0.0, 0.0]);
    let result = sample_speculative(
        &[],
        1,
        &|_: &[u32]| Ok(p.clone()),
        &|_: &[u32]| Ok(q.clone()),
        &mut rng,
    )
    .unwrap();
    assert!(result.bonus);
    assert_eq!(result.tokens, vec![1, 1]);
}

#[test]
fn invalid_distributions_are_explicit_errors() {
    assert!(matches!(
        Distribution::new(&[0.0, 0.0]),
        Err(SamplingError::ZeroMass)
    ));
    assert!(matches!(
        Distribution::new(&[1.0, f32::NAN]),
        Err(SamplingError::InvalidProbability { .. })
    ));
    assert!(matches!(
        Distribution::new(&[-1.0, 2.0]),
        Err(SamplingError::InvalidProbability { .. })
    ));
}

#[test]
fn unsupported_temperature_uses_ordinary_target_route() {
    assert_eq!(route_for_temperature(0.0), DecodeRoute::OrdinaryTarget);
    assert_eq!(route_for_temperature(-1.0), DecodeRoute::OrdinaryTarget);
    assert_eq!(route_for_temperature(f32::NAN), DecodeRoute::OrdinaryTarget);
    assert_eq!(route_for_temperature(0.7), DecodeRoute::DsparkSampling);
    assert_eq!(route_for_sampling(0.7, false), DecodeRoute::OrdinaryTarget);
    assert_eq!(route_for_sampling(0.7, true), DecodeRoute::DsparkSampling);
}
