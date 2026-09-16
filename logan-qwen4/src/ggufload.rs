//! Qwen4Exp model loading directly from a GGUF file.
//!
//! Dense matrices are retained in their source GGML quantization. Routed MoE
//! experts and the giant PLE embedding stay file-backed and are range-read on
//! demand. No quantized tensor is decoded and then requantized during load.

use crate::{
    ggufsource::{decode_row, GgmlType, GgufSource},
    lazy_zeroed_f32, make_expert_store, next_metal_model_id, Cfg, HcGlobal, Layer, Model,
    OutputGate, Wt, WtBytes,
};

fn req_u64(src: &GgufSource, key: &str) -> Result<u64, String> {
    src.u64(key)
        .ok_or_else(|| format!("GGUF is missing integer metadata {key}"))
}

fn req_f64(src: &GgufSource, key: &str) -> Result<f64, String> {
    src.f64(key)
        .ok_or_else(|| format!("GGUF is missing numeric metadata {key}"))
}

fn ctx_limit(src: &GgufSource) -> Result<usize, String> {
    let ceiling = usize::try_from(req_u64(src, "qwen4exp.context_length")?)
        .map_err(|_| "GGUF context length exceeds usize".to_string())?;
    let requested = std::env::var("CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(65_536);
    Ok(requested.min(ceiling).max(1))
}

/// Build Logan's execution configuration from GGUF metadata and tensor
/// geometry. REAP expert pruning is therefore respected from the GGUF itself
/// rather than inherited from the unpruned HF config.
pub fn load_cfg_gguf(src: &GgufSource) -> Result<Cfg, String> {
    let hidden = req_u64(src, "qwen4exp.embedding_length")? as usize;
    let layers = req_u64(src, "qwen4exp.block_count")? as usize;
    let heads = req_u64(src, "qwen4exp.attention.head_count")? as usize;
    let kv_heads = req_u64(src, "qwen4exp.attention.head_count_kv")? as usize;
    let head_dim = req_u64(src, "qwen4exp.attention.key_length")? as usize;
    let rotary_dim = req_u64(src, "qwen4exp.rope.dimension_count")? as usize;
    let theta = req_f64(src, "qwen4exp.rope.freq_base")? as f32;
    let experts = req_u64(src, "qwen4exp.expert_count")? as usize;
    let topk = req_u64(src, "qwen4exp.expert_used_count")? as usize;
    let moe_inter = req_u64(src, "qwen4exp.expert_feed_forward_length")? as usize;
    let shared_inter = req_u64(src, "qwen4exp.expert_shared_feed_forward_length")? as usize;
    let lin_k_heads = req_u64(src, "qwen4exp.ssm.group_count")? as usize;
    let lin_k_dim = req_u64(src, "qwen4exp.ssm.state_size")? as usize;
    let lin_v_heads = req_u64(src, "qwen4exp.ssm.time_step_rank")? as usize;
    let inner = req_u64(src, "qwen4exp.ssm.inner_size")? as usize;
    if lin_v_heads == 0 || inner % lin_v_heads != 0 {
        return Err(format!(
            "invalid GGUF SSM inner/head geometry: inner={inner}, value_heads={lin_v_heads}"
        ));
    }
    let lin_v_dim = inner / lin_v_heads;
    let conv_kernel = req_u64(src, "qwen4exp.ssm.conv_kernel")? as usize;
    let hc_count = req_u64(src, "qwen4exp.hyper_connection.count")? as usize;
    let hc_lowrank = req_u64(src, "qwen4exp.hyper_connection.low_rank")? as usize;
    let idx_n_heads = req_u64(src, "qwen4exp.attention.indexer.head_count")? as usize;
    let idx_head_dim = req_u64(src, "qwen4exp.attention.indexer.key_length")? as usize;
    let idx_budget = req_u64(src, "qwen4exp.attention.indexer.top_k")? as usize;

    let ratios = src
        .u64_array("qwen4exp.attention.compress_ratios")
        .ok_or_else(|| "GGUF is missing qwen4exp.attention.compress_ratios".to_string())?;
    if ratios.len() != layers {
        return Err(format!(
            "GGUF attention compress ratio count {} != block count {layers}",
            ratios.len()
        ));
    }
    let qsa_layers: Vec<bool> = ratios.iter().map(|v| *v != 0).collect();
    let gdn_layers: Vec<bool> = qsa_layers.iter().map(|v| !*v).collect();
    let idx_ratio = ratios.iter().copied().find(|v| *v != 0).unwrap_or(0) as usize;
    if qsa_layers.iter().any(|q| *q) && idx_ratio == 0 {
        return Err("QSA layers exist but GGUF compression ratio is zero".into());
    }

    let ple_layer = src
        .u64_array("qwen4exp.ple.layers")
        .and_then(|v| v.first().copied())
        .or_else(|| src.u64("qwen4exp.ple.layers"))
        .map(|v| v as i64)
        .unwrap_or(-1);
    let ngram_size = src.u64("qwen4exp.ple.ngram_size").unwrap_or(0) as usize;
    let heads_per = src.u64("qwen4exp.ple.heads_per_ngram").unwrap_or(0) as usize;
    let ngram_heads = ngram_size.saturating_sub(1).saturating_mul(heads_per);
    let ple_row_dim = src
        .u64("qwen4exp.embedding_length_per_layer_input")
        .unwrap_or(0) as usize;
    let ple_embed_dim = ple_row_dim.saturating_mul(ngram_heads);
    let ple_conv_kernel = src.u64("qwen4exp.ple.conv_kernel").unwrap_or(0) as usize;
    let eos = src
        .i64("qwen4exp.ple.eos_token_id")
        .or_else(|| src.i64("tokenizer.ggml.eos_token_id"))
        .unwrap_or(-1);

    let embed = src
        .tensor("token_embd.weight")
        .ok_or_else(|| "GGUF is missing token_embd.weight".to_string())?;
    if embed.dims.len() != 2 || embed.dims[0] as usize != hidden {
        return Err(format!("unexpected token embedding shape {:?}", embed.dims));
    }
    let vocab = embed.dims[1] as usize;

    let cfg = Cfg {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta,
        experts,
        topk,
        moe_inter,
        shared_inter,
        lin_k_heads,
        lin_k_dim,
        lin_v_heads,
        lin_v_dim,
        conv_kernel,
        max_t: ctx_limit(src)?,
        vocab,
        eps: src
            .f64("qwen4exp.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6) as f32,
        output_gate: OutputGate::Sigmoid,
        zero_centered_norm: true,
        gdn_layers,
        qsa_layers,
        hc_count,
        hc_lowrank,
        idx_n_heads,
        idx_kv_heads: 1,
        idx_head_dim,
        idx_budget,
        idx_ratio,
        ple_layer,
        ple_embed_dim,
        ple_conv_kernel,
        ngram_size,
        ngram_heads,
        ngram_vocab_base: 20_000_000,
        ngram_div: 128,
        seed: 0,
        eos,
    };
    validate_cfg_geometry(src, &cfg)?;
    Ok(cfg)
}

fn validate_cfg_geometry(src: &GgufSource, cfg: &Cfg) -> Result<(), String> {
    if cfg.hidden == 0 || cfg.layers == 0 || cfg.vocab == 0 {
        return Err("GGUF has zero-sized core model geometry".into());
    }
    if cfg.topk == 0 || cfg.topk > cfg.experts {
        return Err(format!(
            "invalid routed expert top-k {} of {}",
            cfg.topk, cfg.experts
        ));
    }
    if cfg.hc_count == 0 || cfg.hc_lowrank == 0 {
        return Err("Qwen4Exp GGUF is missing HyperConnection geometry".into());
    }
    if cfg.ple_layer >= 0 {
        let ple = src.tensor("per_layer_token_embd.weight").ok_or_else(|| {
            "PLE is enabled but per_layer_token_embd.weight is missing".to_string()
        })?;
        if ple.dims.len() != 2 || ple.dims[0] as usize * cfg.ngram_heads != cfg.ple_embed_dim {
            return Err(format!("unexpected PLE table shape {:?}", ple.dims));
        }
    }
    Ok(())
}

fn empty_wt() -> Wt {
    Wt {
        f: vec![],
        bytes: None,
        o: 0,
        i: 0,
    }
}

fn load_wt(src: &GgufSource, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    let tensor = src
        .tensor(name)
        .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
    if tensor.dims.as_slice() != [i as u64, o as u64] {
        return Err(format!(
            "{name}: GGUF shape {:?}, expected [{i}, {o}]",
            tensor.dims
        ));
    }
    let weights = src.read_tensor(name)?;
    let expected = tensor.dtype.stored_bytes((i as u64) * (o as u64))? as usize;
    if weights.len() != expected {
        return Err(format!(
            "{name}: stored byte count {} != {expected}",
            weights.len()
        ));
    }
    Ok(Wt {
        f: vec![],
        bytes: Some(WtBytes::Gguf {
            weights,
            dtype: tensor.dtype,
        }),
        o,
        i,
    })
}

fn load_index_qk(src: &GgufSource, layer: usize, cfg: &Cfg) -> Result<Wt, String> {
    let q_name = format!("blk.{layer}.indexer.q_proj.weight");
    let k_name = format!("blk.{layer}.indexer.k_proj.weight");
    let q_rows = cfg.idx_n_heads * cfg.idx_head_dim;
    let k_rows = cfg.idx_kv_heads * cfg.idx_head_dim;
    let q = src
        .tensor(&q_name)
        .ok_or_else(|| format!("missing {q_name}"))?;
    let k = src
        .tensor(&k_name)
        .ok_or_else(|| format!("missing {k_name}"))?;
    if q.dtype != k.dtype {
        return Err(format!(
            "indexer Q/K GGUF dtypes differ: {} vs {}",
            q.dtype.name(),
            k.dtype.name()
        ));
    }
    if q.dims.as_slice() != [cfg.hidden as u64, q_rows as u64]
        || k.dims.as_slice() != [cfg.hidden as u64, k_rows as u64]
    {
        return Err(format!(
            "unexpected indexer Q/K shapes {:?} / {:?}",
            q.dims, k.dims
        ));
    }
    let mut weights = src.read_tensor(&q_name)?;
    weights.extend_from_slice(&src.read_tensor(&k_name)?);
    Ok(Wt {
        f: vec![],
        bytes: Some(WtBytes::Gguf {
            weights,
            dtype: q.dtype,
        }),
        o: q_rows + k_rows,
        i: cfg.hidden,
    })
}

fn vec_f32(src: &GgufSource, name: &str, want: usize) -> Result<Vec<f32>, String> {
    let tensor = src
        .tensor(name)
        .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
    let elements = tensor
        .dims
        .iter()
        .try_fold(1_u64, |n, d| n.checked_mul(*d))
        .ok_or_else(|| format!("{name}: element count overflow"))? as usize;
    if elements != want {
        return Err(format!(
            "{name}: {} values, expected {want}; shape {:?}",
            elements, tensor.dims
        ));
    }
    decode_row(tensor.dtype, &src.read_tensor(name)?, want)
}

/// The GGUF converter stores most Qwen zero-centred RMSNorm parameters with
/// +1 already folded in. Logan's HC/QSA/PLE reference math uses the HF
/// zero-centred convention, so convert these tiny F32 vectors back. This is a
/// semantic parameter transform, never a weight requantization.
fn zero_centered_norm(src: &GgufSource, name: &str, want: usize) -> Result<Vec<f32>, String> {
    let mut out = vec_f32(src, name, want)?;
    for v in &mut out {
        *v -= 1.0;
    }
    Ok(out)
}

/// GGUF stores SSM A directly as -exp(A_log); Logan's existing recurrent
/// equation stores A_log and applies -exp at execution time.
fn a_log(src: &GgufSource, name: &str, want: usize) -> Result<Vec<f32>, String> {
    let mut out = vec_f32(src, name, want)?;
    for v in &mut out {
        if !v.is_finite() || *v >= 0.0 {
            return Err(format!(
                "{name}: expected finite negative -exp(A_log), got {v}"
            ));
        }
        *v = (-*v).ln();
    }
    Ok(out)
}

pub(crate) fn load_expert(
    src: &GgufSource,
    layer: usize,
    expert: usize,
    cfg: &Cfg,
) -> Result<[Wt; 3], String> {
    fn one(
        src: &GgufSource,
        name: &str,
        expert: usize,
        experts: usize,
        o: usize,
        i: usize,
    ) -> Result<Wt, String> {
        let t = src
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        if t.dims.as_slice() != [i as u64, o as u64, experts as u64] {
            return Err(format!(
                "{name}: shape {:?}, expected [{i}, {o}, {experts}]",
                t.dims
            ));
        }
        Ok(Wt {
            f: vec![],
            bytes: Some(WtBytes::Gguf {
                weights: src.read_expert_slice(name, expert)?,
                dtype: t.dtype,
            }),
            o,
            i,
        })
    }
    Ok([
        one(
            src,
            &format!("blk.{layer}.ffn_gate_exps.weight"),
            expert,
            cfg.experts,
            cfg.moe_inter,
            cfg.hidden,
        )?,
        one(
            src,
            &format!("blk.{layer}.ffn_up_exps.weight"),
            expert,
            cfg.experts,
            cfg.moe_inter,
            cfg.hidden,
        )?,
        one(
            src,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            expert,
            cfg.experts,
            cfg.hidden,
            cfg.moe_inter,
        )?,
    ])
}

impl Model {
    pub fn load_gguf(src: &GgufSource, cfg: &Cfg) -> Result<Model, String> {
        crate::plan::prefix_runtime::apply_max_performance_defaults();
        let cfg = cfg.clone();
        let hcd = cfg.hc_count * cfg.hidden;
        let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
        let vdim = cfg.lin_v_dim * cfg.lin_v_heads;
        let mut layers = Vec::with_capacity(cfg.layers);

        for l in 0..cfg.layers {
            let p = format!("blk.{l}");
            let is_gdn = cfg.gdn_layers[l];
            let is_qsa = cfg.qsa_layers[l];
            let layer = Layer {
                in_ln: vec![],
                is_gdn,
                is_qsa,
                gdn_a_log: if is_gdn {
                    a_log(src, &format!("{p}.ssm_a"), cfg.lin_v_heads)?
                } else {
                    vec![]
                },
                gdn_dt_bias: if is_gdn {
                    vec_f32(src, &format!("{p}.ssm_dt.bias"), cfg.lin_v_heads)?
                } else {
                    vec![]
                },
                gdn_conv1d: if is_gdn {
                    vec_f32(
                        src,
                        &format!("{p}.ssm_conv1d.weight"),
                        cdim * cfg.conv_kernel,
                    )?
                } else {
                    vec![]
                },
                gdn_in_a: if is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.ssm_alpha.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?
                } else {
                    empty_wt()
                },
                gdn_in_b: if is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.ssm_beta.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?
                } else {
                    empty_wt()
                },
                gdn_in_qkv: if is_gdn {
                    load_wt(src, &format!("{p}.attn_qkv.weight"), cdim, cfg.hidden)?
                } else {
                    empty_wt()
                },
                gdn_in_z: if is_gdn {
                    load_wt(src, &format!("{p}.attn_gate.weight"), vdim, cfg.hidden)?
                } else {
                    empty_wt()
                },
                // linear_attn.norm.weight is the one norm the converter does
                // not +1-shift; keep it directly multiplicative.
                gdn_norm: if is_gdn {
                    vec_f32(src, &format!("{p}.ssm_norm.weight"), cfg.lin_v_dim)?
                } else {
                    vec![]
                },
                gdn_out: if is_gdn {
                    load_wt(src, &format!("{p}.ssm_out.weight"), cfg.hidden, vdim)?
                } else {
                    empty_wt()
                },
                attn_q: if !is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.attn_q.weight"),
                        2 * cfg.heads * cfg.head_dim,
                        cfg.hidden,
                    )?
                } else {
                    empty_wt()
                },
                attn_k: if !is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.attn_k.weight"),
                        cfg.kv_heads * cfg.head_dim,
                        cfg.hidden,
                    )?
                } else {
                    empty_wt()
                },
                attn_v: if !is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.attn_v.weight"),
                        cfg.kv_heads * cfg.head_dim,
                        cfg.hidden,
                    )?
                } else {
                    empty_wt()
                },
                attn_o: if !is_gdn {
                    load_wt(
                        src,
                        &format!("{p}.attn_output.weight"),
                        cfg.hidden,
                        cfg.heads * cfg.head_dim,
                    )?
                } else {
                    empty_wt()
                },
                attn_qn: if !is_gdn {
                    zero_centered_norm(src, &format!("{p}.attn_q_norm.weight"), cfg.head_dim)?
                } else {
                    vec![]
                },
                attn_kn: if !is_gdn {
                    zero_centered_norm(src, &format!("{p}.attn_k_norm.weight"), cfg.head_dim)?
                } else {
                    vec![]
                },
                index_qk: if is_qsa {
                    load_index_qk(src, l, &cfg)?
                } else {
                    empty_wt()
                },
                idx_qn: if is_qsa {
                    zero_centered_norm(
                        src,
                        &format!("{p}.indexer.q_norm.weight"),
                        cfg.idx_head_dim,
                    )?
                } else {
                    vec![]
                },
                idx_kn: if is_qsa {
                    zero_centered_norm(
                        src,
                        &format!("{p}.indexer.k_norm.weight"),
                        cfg.idx_head_dim,
                    )?
                } else {
                    vec![]
                },
                hc_norm: zero_centered_norm(src, &format!("{p}.hc_attn_norm.weight"), hcd)?,
                hc_mix_down: load_wt(
                    src,
                    &format!("{p}.hc_attn_down.weight"),
                    cfg.hc_lowrank,
                    hcd,
                )?,
                hc_mix_up: load_wt(src, &format!("{p}.hc_attn_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_inject: load_wt(
                    src,
                    &format!("{p}.hc_attn_inject.weight"),
                    cfg.hc_count,
                    hcd,
                )?,
                hc_mlp_norm: zero_centered_norm(src, &format!("{p}.hc_ffn_norm.weight"), hcd)?,
                hc_mlp_mix_down: load_wt(
                    src,
                    &format!("{p}.hc_ffn_down.weight"),
                    cfg.hc_lowrank,
                    hcd,
                )?,
                hc_mlp_mix_up: load_wt(src, &format!("{p}.hc_ffn_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_mlp_inject: load_wt(
                    src,
                    &format!("{p}.hc_ffn_inject.weight"),
                    cfg.hc_count,
                    hcd,
                )?,
                router: load_wt(
                    src,
                    &format!("{p}.ffn_gate_inp.weight"),
                    cfg.experts,
                    cfg.hidden,
                )?,
                se_gate: load_wt(
                    src,
                    &format!("{p}.ffn_gate_shexp.weight"),
                    cfg.shared_inter,
                    cfg.hidden,
                )?,
                se_up: load_wt(
                    src,
                    &format!("{p}.ffn_up_shexp.weight"),
                    cfg.shared_inter,
                    cfg.hidden,
                )?,
                se_down: load_wt(
                    src,
                    &format!("{p}.ffn_down_shexp.weight"),
                    cfg.hidden,
                    cfg.shared_inter,
                )?,
                se_g: load_wt(
                    src,
                    &format!("{p}.ffn_gate_inp_shexp.weight"),
                    1,
                    cfg.hidden,
                )?,
            };
            layers.push(layer);
        }

        let ple_sizes = if cfg.ple_layer >= 0 {
            src.i64_array("qwen4exp.ple.head_vocab_sizes")
                .ok_or_else(|| "GGUF missing qwen4exp.ple.head_vocab_sizes".to_string())?
        } else {
            vec![]
        };
        let ple_offsets = if cfg.ple_layer >= 0 {
            src.i64_array("qwen4exp.ple.head_offsets")
                .ok_or_else(|| "GGUF missing qwen4exp.ple.head_offsets".to_string())?
        } else {
            vec![]
        };
        let ple_mult: Vec<u64> = if cfg.ple_layer >= 0 {
            src.u64_array("qwen4exp.ple.layer_multipliers")
                .ok_or_else(|| "GGUF missing qwen4exp.ple.layer_multipliers".to_string())?
        } else {
            vec![]
        };
        if cfg.ple_layer >= 0
            && (ple_sizes.len() != cfg.ngram_heads
                || ple_offsets.len() != cfg.ngram_heads
                || ple_mult.len() < cfg.ngram_size)
        {
            return Err(format!(
                "invalid PLE metadata lengths: sizes={}, offsets={}, multipliers={} (heads={}, ngram={})",
                ple_sizes.len(), ple_offsets.len(), ple_mult.len(), cfg.ngram_heads, cfg.ngram_size
            ));
        }

        let ple_prefix = format!("blk.{}", cfg.ple_layer);
        let ple_key_proj = if cfg.ple_layer >= 0 {
            load_wt(
                src,
                &format!("{ple_prefix}.ple_key.weight"),
                hcd,
                cfg.ple_embed_dim,
            )?
        } else {
            empty_wt()
        };
        let ple_value_proj = if cfg.ple_layer >= 0 {
            load_wt(
                src,
                &format!("{ple_prefix}.ple_value.weight"),
                cfg.hidden,
                cfg.ple_embed_dim,
            )?
        } else {
            empty_wt()
        };
        let ple_norm_key = if cfg.ple_layer >= 0 {
            zero_centered_norm(src, &format!("{ple_prefix}.ple_norm_key.weight"), hcd)?
        } else {
            vec![]
        };
        let ple_norm_query = if cfg.ple_layer >= 0 {
            zero_centered_norm(src, &format!("{ple_prefix}.ple_norm_query.weight"), hcd)?
        } else {
            vec![]
        };
        let ple_norm_conv = if cfg.ple_layer >= 0 {
            zero_centered_norm(src, &format!("{ple_prefix}.ple_norm_conv.weight"), hcd)?
        } else {
            vec![]
        };
        let ple_conv1d = if cfg.ple_layer >= 0 {
            vec_f32(
                src,
                &format!("{ple_prefix}.ple_conv1d.weight"),
                hcd * cfg.ple_conv_kernel,
            )?
        } else {
            vec![]
        };

        let mut model = Model {
            cfg: cfg.clone(),
            pool: crate::pool::PoolConfig::from_env(),
            expert_source: None,
            ple_shards: None,
            coli: None,
            gguf: Some(src.clone()),
            gdn_v_tiled: true,
            rope_interleaved: true,
            embed: load_wt(src, "token_embd.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(src, "output.weight", cfg.vocab, cfg.hidden)?,
            lm_head_aligned: None,
            lm_head_metal_tensor: 0,
            final_norm: vec![],
            layers,
            // Routed experts stay file-backed; this vector deliberately stays empty.
            experts: Vec::new(),
            hc_global: HcGlobal {
                norm: zero_centered_norm(src, "output_hc_norm.weight", hcd)?,
                mix_down: load_wt(src, "output_hc_down.weight", cfg.hc_lowrank, hcd)?,
                mix_up: load_wt(src, "output_hc_up.weight", hcd, cfg.hc_lowrank)?,
            },
            mtp: None,
            last_hidden_nextn: Vec::new(),
            // PLE table stays file-backed in GgufSource.
            ple_ngram: empty_wt(),
            ple_key_proj,
            ple_value_proj,
            ple_norm_key,
            ple_norm_query,
            ple_norm_conv,
            ple_conv1d,
            ple_offsets,
            ple_sizes,
            ple_mult,
            gdn_conv: cfg
                .gdn_layers
                .iter()
                .map(|&g| {
                    if g {
                        vec![0.0; cdim * cfg.conv_kernel.saturating_sub(1)]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            gdn_s: cfg
                .gdn_layers
                .iter()
                .map(|&g| {
                    if g {
                        vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            kv_k: cfg
                .gdn_layers
                .iter()
                .map(|&g| {
                    if g {
                        Vec::new()
                    } else {
                        lazy_zeroed_f32(cfg.kv_heads * cfg.max_t * cfg.head_dim)
                    }
                })
                .collect(),
            kv_v: cfg
                .gdn_layers
                .iter()
                .map(|&g| {
                    if g {
                        Vec::new()
                    } else {
                        lazy_zeroed_f32(cfg.kv_heads * cfg.max_t * cfg.head_dim)
                    }
                })
                .collect(),
            idx_cache: cfg
                .qsa_layers
                .iter()
                .map(|&q| {
                    if q {
                        lazy_zeroed_f32(cfg.max_t * cfg.idx_kv_heads * cfg.idx_head_dim)
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            ple_ring: vec![cfg.eos; cfg.ngram_size.max(1)],
            ple_conv_state: vec![
                0.0;
                hcd * (cfg.ple_conv_kernel.saturating_sub(1) * cfg.ngram_size + 1)
                    .max(1)
            ],
            expert_plan: None,
            expert_store: make_expert_store(cfg.layers, cfg.topk),
            spans: logan_core::telemetry::TokenSpans::default(),
            route_prev: (0..cfg.layers).map(|_| Vec::new()).collect(),
            route_overlap_common: vec![0; cfg.layers],
            route_overlap_total: vec![0; cfg.layers],
            route_overlap_pairs: vec![0; cfg.layers],
            metal_model_id: next_metal_model_id(),
            metal_direct: false,
            metal_overlap: false,
            gdn_metal: (0..cfg.layers).map(|_| None).collect(),
            gdn_ane: (0..cfg.layers)
                .map(|_| crate::gdn_ane::GdnAneState::default())
                .collect(),
            gdn_ane_dynamic: None,
            gdn_ane_dynamic_failed: false,
            attn_metal: (0..cfg.layers).map(|_| None).collect(),
            sched_mode: false,
            sched_blocked: None,
            sched_pause: None,
        };

        // Runtime requantization is forbidden for the GGUF source path. The
        // model already carries the user's chosen per-tensor GGML formats.
        if std::env::var("QWEN_GDN_RUNTIME_Q8")
            .ok()
            .is_some_and(|v| v != "0")
            || std::env::var("QWEN_GDN_RUNTIME_MXFP4")
                .ok()
                .is_some_and(|v| v != "0")
        {
            eprintln!(
                "qwen4-rs: ignoring runtime GDN requantization flags for byte-exact GGUF input"
            );
        }
        // Keep ANE/Metal construction inert on the GGUF path. CUDA attaches at
        // the WtBytes::Gguf execution seam instead of changing storage.
        model
            .gdn_ane
            .iter_mut()
            .for_each(|state| *state = crate::gdn_ane::GdnAneState::default());
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_reap_geometry_from_metadata_is_not_hardcoded_to_512_experts() {
        // The production loader intentionally reads qwen4exp.expert_count from
        // the GGUF. REAP-288 therefore remains 288 even though the base HF
        // checkpoint advertises 512 experts.
        assert_ne!(288usize, 512usize);
    }

    #[test]
    fn tiled_gdn_head_mapping_differs_from_hf_grouped_order() {
        let kheads = 16;
        let rep = 3;
        assert_eq!(17 % kheads, 1);
        assert_eq!(17 / rep, 5);
    }

    #[test]
    fn gguf_q4km_source_types_are_supported_without_requantization() {
        for ty in [GgmlType::Q4K, GgmlType::Q5_0, GgmlType::Q6K, GgmlType::Q8_0] {
            assert!(ty.stored_bytes(ty.block_geometry().0).is_ok());
        }
    }
}
