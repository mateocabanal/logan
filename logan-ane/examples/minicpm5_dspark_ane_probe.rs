//! Executes one real MiniCPM5-DSpark projection through the fixed-weight ANE path.
//!
//! This is intentionally a model-island qualification probe, not a full DSpark
//! decoder: the official draft package contains five transformer blocks plus
//! Markov/confidence heads, while Logan's full DSpark target/draft loader is not
//! yet wired to the official checkpoint. The probe does load a real DSpark
//! tensor, compiles it as a constant-weight ANE convolution, evaluates it, and
//! compares the result with an fp16 CPU reference.

use std::{env, fs, path::PathBuf, time::Instant};

use half::{bf16, f16};
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions, DenseProjection};
use safetensors::{tensor::Dtype, SafeTensors};

const HIDDEN: usize = 2048;
const SPATIAL: usize = 16;
const DEFAULT_ROOT: &str = "models/MiniCPM5-2B-DSpark-BF16";

fn checkpoint_root() -> PathBuf {
    if let Some(path) = env::var_os("MINICPM5_DSPARK_DIR") {
        return PathBuf::from(path);
    }
    let home = env::var_os("HOME").unwrap_or_default();
    PathBuf::from(home).join(DEFAULT_ROOT)
}

fn projection_name() -> &'static str {
    match env::var("MINICPM5_DSPARK_PROJECTION")
        .unwrap_or_else(|_| "fc".into())
        .as_str()
    {
        "gate" => "layers.0.mlp.gate_proj.weight",
        "up" => "layers.0.mlp.up_proj.weight",
        "down" => "layers.0.mlp.down_proj.weight",
        _ => "fc.weight",
    }
}

fn max_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = checkpoint_root();
    let model_path = root.join("model.safetensors");
    let bytes = fs::read(&model_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let name = projection_name();
    let tensor = tensors.tensor(name)?;
    let expected_shape = if name == "fc.weight" {
        [HIDDEN, HIDDEN * 5]
    } else if name.ends_with("down_proj.weight") {
        [HIDDEN, 6144]
    } else {
        [6144, HIDDEN]
    };
    if tensor.dtype() != Dtype::BF16 || tensor.shape() != expected_shape {
        return Err(format!(
            "unexpected {name} tensor: dtype={:?} shape={:?}",
            tensor.dtype(),
            tensor.shape()
        )
        .into());
    }
    let out_features = tensor.shape()[0];
    let in_features = tensor.shape()[1];

    let weights_fp16 = tensor
        .data()
        .chunks_exact(2)
        .map(|raw| {
            let value = bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
            f16::from_f32(value).to_bits()
        })
        .collect::<Vec<_>>();
    let projection = DenseProjection::new(name, out_features, weights_fp16.clone());
    let program = mil::parallel_dense_fp16_f32_io(in_features, SPATIAL, &[projection])?;
    let runtime = AneRuntime::load()?;
    if !runtime.device_info().has_ane {
        println!("ANE unavailable; DSpark projection remains on CPU/Metal");
        return Ok(());
    }
    let compile_start = Instant::now();
    let mut model = match runtime.compile(&program, CompileOptions::default()) {
        Ok(model) => model,
        Err(error) => {
            println!("DSpark projection ANE compile declined: {error}");
            return Ok(());
        }
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1e3;
    model.load()?;

    let input_values = (0..in_features * SPATIAL)
        .map(|index| ((index as i32 % 29) - 14) as f32 / 32.0)
        .collect::<Vec<_>>();
    let mut input = AneSurface::new(input_values.len() * 4)?;
    let output = AneSurface::new(out_features * SPATIAL * 4)?;
    input.write_f32(&input_values)?;
    let request = AneRequest::new(&[&input], &[&output], 0)?;
    for _ in 0..3 {
        model.evaluate(&request)?;
    }
    let iterations = 10;
    let eval_start = Instant::now();
    for _ in 0..iterations {
        model.evaluate(&request)?;
    }
    let eval_us = eval_start.elapsed().as_secs_f64() * 1e6 / iterations as f64;

    let output_values = output.read_f32()?;
    let mut expected = vec![0.0f32; output_values.len()];
    for out in 0..out_features {
        for spatial in 0..SPATIAL {
            let mut sum = 0.0f32;
            for input_channel in 0..in_features {
                let weight_offset = out * in_features + input_channel;
                let weight = f16::from_bits(weights_fp16[weight_offset]).to_f32();
                let input_value =
                    f16::from_f32(input_values[input_channel * SPATIAL + spatial]).to_f32();
                sum += weight * input_value;
            }
            expected[out * SPATIAL + spatial] = sum;
        }
    }
    let error = max_error(&output_values, &expected);
    println!("device: {:#?}", runtime.device_info());
    println!("checkpoint: {}", model_path.display());
    println!("tensor: {name} {:?}", tensor.shape());
    println!("compile: {compile_ms:.3} ms");
    println!("evaluate: {eval_us:.3} us/dispatch");
    println!("max abs error: {error:.8}");
    if error > 0.25 {
        return Err(format!("DSpark ANE projection mismatch: {error}").into());
    }
    Ok(())
}
