use std::time::Instant;

use logan_ane::{
    AneRequest, AneRuntime, AneSurface, BlobV2Builder, CompileOptions, MilProgram, WeightBlob,
};

const WEIGHT_PATH: &str = "@model_path/weights/weight.bin";

fn identity_weights(channels: usize) -> Vec<u16> {
    let mut values = vec![0u16; channels * channels];
    // [O, I, 1, 1] identity convolution. 1.0 in IEEE fp16 is 0x3c00.
    for channel in 0..channels {
        values[channel * channels + channel] = 0x3c00;
    }
    values
}

fn identity_conv_mil(channels: usize, spatial: usize, blob_offset: u64) -> String {
    format!(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]\n\
{{\n\
 func main<ios18>(tensor<fp32, [1, {channels}, 1, {spatial}]> x) {{\n\
  string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n\
  tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
  tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
  tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
  int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n\
  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n\
  tensor<fp16, [1, {channels}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n\
  tensor<fp16, [{channels}, {channels}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{channels}, {channels}, 1, 1]>(BLOBFILE(path = string(\"{WEIGHT_PATH}\"), offset = uint64({blob_offset})))];\n\
  tensor<fp16, [1, {channels}, 1, {spatial}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n\
  string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n\
  tensor<fp32, [1, {channels}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n\
 }} -> (y);\n\
}}\n"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    let channels = 256usize;
    let spatial = 64usize;
    let elements = channels * spatial;
    let bytes = elements * std::mem::size_of::<f32>();

    let mut blob = BlobV2Builder::new();
    let blob_offset = blob.push_fp16(&identity_weights(channels))?;
    let blob_bytes = blob.into_bytes();
    let program = MilProgram::new(identity_conv_mil(channels, spatial, blob_offset.get()))
        .with_weight(WeightBlob::new(WEIGHT_PATH, blob_bytes.clone()).descriptor_offset(0));

    let compile_t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    let compile_ms = compile_t0.elapsed().as_secs_f64() * 1e3;
    model.load()?;

    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    // Exactly representable fp16 values, so an identity fp16 convolution should
    // round-trip exactly apart from any backend implementation detail.
    let values: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 17) - 8) as f32 / 8.0)
        .collect();
    input.write_f32(&values)?;

    let request = AneRequest::new(&[&input], &[&output], 0)?;
    for _ in 0..5 {
        model.evaluate(&request)?;
    }
    let iters = 100usize;
    let eval_t0 = Instant::now();
    for _ in 0..iters {
        model.evaluate(&request)?;
    }
    let eval_us = eval_t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let result = output.read_f32()?;
    let max_abs_error = result
        .iter()
        .zip(&values)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    println!("device: {:#?}", runtime.device_info());
    println!("weight blob: {} bytes", blob_bytes.len());
    println!("MIL BLOBFILE metadata offset: {}", blob_offset.get());
    println!("compile: {compile_ms:.3} ms");
    println!("evaluate: {eval_us:.3} us/dispatch");
    println!("max abs error: {max_abs_error:.8}");

    if !max_abs_error.is_finite() || max_abs_error > 0.001 {
        return Err(format!("weighted identity convolution failed: {max_abs_error}").into());
    }
    Ok(())
}
