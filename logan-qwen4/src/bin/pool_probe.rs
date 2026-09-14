//! Drive the pool client against a live coordinator and check the values.
//!
//! The unit tests prove the encoder and parser are self-consistent; they cannot
//! prove the pool agrees. This sends a real batch for a real expert, compares
//! the returned vector against the same expert decoded and evaluated locally
//! from the checkpoint, and fails if they differ.
//!
//! A client that returned well-formed but wrong numbers would pass every format
//! test, which is exactly the failure mode worth guarding here.
//!
//! Usage:
//!   cargo run -p logan-qwen4 --bin pool_probe -- \
//!     --coordinator http://127.0.0.1:8080 --model ~/models/Qwen3.6-35B-A3B-mxfp4

use std::path::PathBuf;

use logan_qwen4::pool::{self, ExpertCall, PoolConfig};

fn main() -> Result<(), String> {
    let mut coordinator = std::env::var("LOGAN_POOL_COORDINATOR")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let mut model = PathBuf::from(
        std::env::var("LOGAN_POOL_MODEL")
            .unwrap_or_else(|_| "~/models/Qwen3.6-35B-A3B-mxfp4".into()),
    );
    let mut layer = 34u32;
    let mut expert = 3u32;
    let mut family = "qwen36".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() + 1 {
        match args.get(i).map(String::as_str) {
            Some("--coordinator") => {
                coordinator = args[i + 1].clone();
                i += 2;
            }
            Some("--model") => {
                model = PathBuf::from(shellexpand(&args[i + 1]));
                i += 2;
            }
            Some("--layer") => {
                layer = args[i + 1].parse().map_err(|e| format!("layer: {e}"))?;
                i += 2;
            }
            Some("--expert") => {
                expert = args[i + 1].parse().map_err(|e| format!("expert: {e}"))?;
                i += 2;
            }
            Some("--family") => {
                family = args[i + 1].clone();
                i += 2;
            }
            _ => break,
        }
    }

    if !model.is_dir() {
        return Err(format!("model dir not found: {}", model.display()));
    }

    // A deterministic activation: the check is about agreement, not about the
    // input being interesting.
    let d_model = 2048usize;
    let input: Vec<f32> = (0..d_model)
        .map(|j| (((j * 37) % 101) as f32 - 50.0) / 500.0)
        .collect();

    let cfg = PoolConfig::new(coordinator.clone())
        .with_family(family.clone());
    println!("coordinator {coordinator}  family {family}");
    println!("layer {layer} expert {expert}  d_model {d_model}");

    // Reference: decode + evaluate locally, using the same checkpoint reader the
    // rest of the crate uses.
    let local = local_expert(&model, layer, expert, &input)?;
    println!("local  rms {:.6}", rms(&local));

    // Pool: send the same call, get the same expert from a worker.
    let calls = vec![ExpertCall {
        layer,
        expert,
        input: input.clone(),
    }];
    let got = pool::run_expert_batch(&cfg, &calls, d_model, 512, "silu")?;
    if got.len() != 1 {
        return Err(format!("expected 1 output, got {}", got.len()));
    }
    let remote = &got[0];
    if remote.len() != local.len() {
        return Err(format!(
            "shape mismatch: local {} vs pool {}",
            local.len(),
            remote.len()
        ));
    }
    let diff = local
        .iter()
        .zip(remote.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("pool   rms {:.6}", rms(remote));
    println!("max|local - pool| = {diff:.3e}");
    if diff > 1e-3 {
        return Err(format!("pool disagrees with local decode by {diff:.3e}"));
    }
    println!("OK: pool client returns the same expert the checkpoint holds");
    Ok(())
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn shellexpand(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

/// Evaluate one expert from the checkpoint directly.
///
/// Deliberately a second implementation rather than a call into Model: the
/// point is to compare two paths, so sharing code would make the comparison
/// vacuous.
fn local_expert(
    model: &std::path::Path,
    layer: u32,
    expert: u32,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let prefix = format!("language_model.model.layers.{layer}.mlp.switch_mlp");
    let mut shards: Vec<PathBuf> = std::fs::read_dir(model)
        .map_err(|e| format!("read_dir: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
        })
        .collect();
    shards.sort();

    let mut decoded: Vec<Vec<f32>> = Vec::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let name = format!("{prefix}.{proj}");
        let mut found = None;
        for shard in &shards {
            let mut f = File::open(shard).map_err(|e| format!("open: {e}"))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len).map_err(|e| format!("{e}"))?;
            let hlen = u64::from_le_bytes(len);
            let mut hbuf = vec![0u8; hlen as usize];
            f.read_exact(&mut hbuf).map_err(|e| format!("{e}"))?;
            let header: serde_json::Value =
                serde_json::from_slice(&hbuf).map_err(|e| format!("json: {e}"))?;
            let data_start = 8 + hlen;
            let wkey = format!("{name}.weight");
            let skey = format!("{name}.scales");
            let (Some(w), Some(s)) = (header.get(&wkey), header.get(&skey)) else {
                continue;
            };
            let wshape: Vec<u64> = w["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let sshape: Vec<u64> = s["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            if wshape.len() != 3 || expert as u64 >= wshape[0] {
                continue;
            }
            let rows = wshape[1] as usize;
            let cols = wshape[2] as usize; // u32 words
            let per_w = rows * cols * 4;
            let sgroups = (sshape[2] as usize) * (sshape[1] as usize) / (sshape[0] as usize).max(1);
            let per_s = sgroups;

            let woff = w["data_offsets"].as_array().unwrap()[0].as_u64().unwrap();
            let soff = s["data_offsets"].as_array().unwrap()[0].as_u64().unwrap();
            f.seek(SeekFrom::Start(data_start + woff + expert as u64 * per_w as u64))
                .map_err(|e| format!("{e}"))?;
            let mut wbuf = vec![0u8; per_w];
            f.read_exact(&mut wbuf).map_err(|e| format!("{e}"))?;
            f.seek(SeekFrom::Start(data_start + soff + expert as u64 * per_s as u64))
                .map_err(|e| format!("{e}"))?;
            let mut sbuf = vec![0u8; per_s];
            f.read_exact(&mut sbuf).map_err(|e| format!("{e}"))?;

            // MXFP4: E2M1 nibbles, low nibble first, one E8M0 scale per 32 cols.
            const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let mut out = vec![0.0f32; rows * (cols * 8)];
            let mut col = 0usize;
            'outer: for (bi, b) in wbuf.iter().enumerate() {
                for nib in [b & 0x0F, (b >> 4) & 0x0F] {
                    if col >= out.len() {
                        break 'outer;
                    }
                    // One scale byte per 32 OUTPUT columns. Indexing by the
                    // packed byte index instead inflates values by ~2^5.
                    let scale_idx = col / 32;
                    let scale = sbuf
                        .get(scale_idx)
                        .map(|&s| f32::from_bits((s as u32) << 23))
                        .unwrap_or(1.0);
                    let mag = MAG[(nib & 0x07) as usize];
                    out[col] = if nib & 0x08 != 0 { -mag } else { mag } * scale;
                    col += 1;
                }
            }
            found = Some(out);
            break;
        }
        let Some(w) = found else {
            return Err(format!("tensor {name} not found for expert {expert}"));
        };
        decoded.push(w);
    }

    let [gate, up, down] = [&decoded[0], &decoded[1], &decoded[2]];
    let inter = 512usize;
    let hidden = 2048usize;
    if gate.len() != inter * hidden || up.len() != inter * hidden || down.len() != hidden * inter {
        return Err(format!(
            "unexpected shapes: gate {} up {} down {}",
            gate.len(),
            up.len(),
            down.len()
        ));
    }

    let mut act = vec![0.0f32; inter];
    for r in 0..inter {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for c in 0..hidden {
            g += gate[r * hidden + c] * input[c];
            u += up[r * hidden + c] * input[c];
        }
        act[r] = (g / (1.0 + (-g).exp())) * u;
    }
    let mut out = vec![0.0f32; hidden];
    for r in 0..hidden {
        let mut acc = 0.0f32;
        for c in 0..inter {
            acc += down[r * inter + c] * act[c];
        }
        out[r] = acc;
    }
    Ok(out)
}
