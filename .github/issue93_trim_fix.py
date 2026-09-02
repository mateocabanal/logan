from pathlib import Path

p = Path('logan-ir/src/tiered_optimizer.rs')
s = p.read_text()
old = '''fn state_sort_key(state: &State) -> (u64, u64, u64, u64, u64, u16, Vec<usize>) {
    let usage = state.resources.usage().ok();
    (
        state.quality_loss_ppm,
        state.latency_cost,
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::target_resident_bytes),
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::mutable_backing_bytes),
        state.package_bytes,
        state.last_dispatch_class,
        state.choices.clone(),
    )
}
'''
new = '''fn state_sort_key(
    state: &State,
) -> (u64, u64, u64, u64, u64, u64, u64, u16, Vec<usize>) {
    let usage = state.resources.usage().ok();
    (
        state.quality_loss_ppm,
        state.latency_cost,
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::minimum_working_set_bytes),
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::target_resident_bytes),
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::mutable_backing_bytes),
        usage
            .as_ref()
            .map_or(u64::MAX, TieredResourceUsage::storage_traffic_bytes_per_step),
        state.package_bytes,
        state.last_dispatch_class,
        state.choices.clone(),
    )
}
'''
if old not in s:
    raise SystemExit('state_sort_key anchor missing')
p.write_text(s.replace(old, new, 1))
