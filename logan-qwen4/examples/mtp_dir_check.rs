// Verify a drafter DIRECTORY loads: every tensor the loader needs must resolve.
// Usage: mtp_dir_check <drafter-dir>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(a.get(1).map(String::as_str)
        .unwrap_or("/home/mateo/models/qwen38-mtp-drafter"));
    let st = match logan_qwen4::StFile::open_dir(dir) {
        Ok(v) => v,
        Err(e) => { println!("  open_dir FAILED: {e}"); std::process::exit(1); }
    };
    println!("  shards={} dtypes={:?}", st.shard_count(), st.dtype_counts());
    let (d, hcd, e, mi, hd, hcl, hc) = (2560u64, 10240u64, 512u64, 640u64, 256u64, 320u64, 4u64);
    let lp = "layers.0";
    let checks: Vec<(String, Vec<u64>)> = vec![
        (format!("{lp}.self_attn.q_proj.weight"), vec![2*24*hd, d]),
        (format!("{lp}.self_attn.k_proj.weight"), vec![2*hd, d]),
        (format!("{lp}.self_attn.v_proj.weight"), vec![2*hd, d]),
        (format!("{lp}.self_attn.o_proj.weight"), vec![d, 24*hd]),
        (format!("{lp}.attn_hyper_connection.hc_norm.weight"), vec![hcd]),
        (format!("{lp}.mlp_hyper_connection.hc_norm.weight"), vec![hcd]),
        (format!("{lp}.mlp.gate.weight"), vec![e, d]),
        (format!("{lp}.mlp.shared_expert.gate_proj.weight"), vec![mi, d]),
        (format!("{lp}.mlp.shared_expert.down_proj.weight"), vec![d, mi]),
        (format!("{lp}.mlp.switch_mlp.gate_proj.weight"), vec![e, mi, d]),
        (format!("{lp}.mlp.switch_mlp.down_proj.weight"), vec![e, d, mi]),
        ("fc_embedding.weight".into(), vec![d, d]),
        ("fc_hidden.weight".into(), vec![d, d]),
        ("pre_fc_norm_embedding.weight".into(), vec![d]),
        ("pre_fc_norm_hidden.weight".into(), vec![hcd]),
        ("hyper_connection_mixer.hc_norm.weight".into(), vec![hcd]),
        ("hyper_connection_mixer.input_mix_weight_down.weight".into(), vec![hcl, hcd]),
        ("hyper_connection_mixer.input_mix_weight_up.weight".into(), vec![hcd, hcl]),
        (format!("{lp}.mlp.switch_mlp.up_proj.weight"), vec![e, mi, d]),
        (format!("{lp}.self_attn.q_norm.weight"), vec![hd]),
        (format!("{lp}.self_attn.k_norm.weight"), vec![hd]),
        (format!("{lp}.mlp.shared_expert_gate.weight"), vec![1, d]),
        (format!("{lp}.attn_hyper_connection.block_inject_weight.weight"), vec![hc, hcd]),
        (format!("{lp}.mlp_hyper_connection.block_inject_weight.weight"), vec![hc, hcd]),
        (format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"), vec![hcl, hcd]),
        (format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"), vec![hcd, hcl]),
        (format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"), vec![hcl, hcd]),
        (format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"), vec![hcd, hcl]),
        (format!("{lp}.mlp.shared_expert.up_proj.weight"), vec![mi, d]),
    ];
    let (mut ok, mut bad) = (0, 0);
    for (name, shape) in &checks {
        match st.f32(name, shape) {
            Ok(v) => { ok += 1;
                if v.iter().all(|x| *x == 0.0) || v.iter().any(|x| !x.is_finite()) {
                    println!("  SUSPECT {name}: all-zero or non-finite"); } }
            Err(err) => { bad += 1; println!("  FAIL {name}: {err}"); }
        }
    }
    println!("  resolved {ok}, failed {bad} of {}", checks.len());
    println!("  {}", if bad == 0 { "drafter loads on this host" } else { "DRAPTER INCOMPLETE" });
}
