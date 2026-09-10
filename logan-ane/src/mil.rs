use crate::{AneError, BlobV2Builder, Result};

/// Raw MIL source plus any BLOBFILE payloads referenced by it.
#[derive(Clone, Debug, Default)]
pub struct MilProgram {
    text: String,
    weights: Vec<WeightBlob>,
}

impl MilProgram {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            weights: Vec::new(),
        }
    }

    pub fn with_weight(mut self, weight: WeightBlob) -> Self {
        self.weights.push(weight);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn weights(&self) -> &[WeightBlob] {
        &self.weights
    }

    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(AneError::InvalidArgument("MIL text is empty".into()));
        }
        for weight in &self.weights {
            weight.validate()?;
        }
        Ok(())
    }
}

/// One file exposed to MIL through `BLOBFILE(path = "@model_path/...")`.
///
/// `data` must contain the CoreML weight-file representation expected by the
/// MIL BLOBFILE declaration. Use [`crate::BlobV2Builder`] to construct ordinary
/// Blob v2 `weight.bin` files without hand-encoding their metadata records.
#[derive(Clone, Debug)]
pub struct WeightBlob {
    path: String,
    data: Vec<u8>,
    descriptor_offset: u64,
}

impl WeightBlob {
    pub fn new(path: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            path: path.into(),
            data,
            descriptor_offset: 0,
        }
    }

    /// Offset used by `_ANEInMemoryModelDescriptor`'s per-file dictionary.
    /// This is distinct from a BLOBFILE offset written in MIL source.
    pub fn descriptor_offset(mut self, offset: u64) -> Self {
        self.descriptor_offset = offset;
        self
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn descriptor_file_offset(&self) -> u64 {
        self.descriptor_offset
    }

    fn validate(&self) -> Result<()> {
        let Some(relative) = self.path.strip_prefix("@model_path/") else {
            return Err(AneError::InvalidArgument(format!(
                "weight path {:?} must begin with @model_path/",
                self.path
            )));
        };
        if relative.is_empty()
            || relative.starts_with('/')
            || relative
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(AneError::InvalidArgument(format!(
                "weight path {:?} is not a safe model-relative path",
                self.path
            )));
        }
        Ok(())
    }
}

/// One fixed fp16 dense projection in a parallel ANE island.
///
/// We express a matrix `[out_features, in_features]` as a 1x1 convolution
/// weight `[O, I, 1, 1]`. This maps cleanly onto the ANE compiler and lets a
/// single input activation feed several projections in one compiled MIL
/// program — the shape used by Qwen GDN's qkv/z/a/b input side.
#[derive(Clone, Debug)]
pub struct DenseProjection {
    pub name: String,
    pub out_features: usize,
    /// IEEE fp16 row-major `[out_features, in_features]` values.
    pub weights_fp16: Vec<u16>,
}

impl DenseProjection {
    pub fn new(name: impl Into<String>, out_features: usize, weights_fp16: Vec<u16>) -> Self {
        Self {
            name: name.into(),
            out_features,
            weights_fp16,
        }
    }
}

/// Build one fixed-shape, multi-projection ANE dense island.
///
/// Input is fp32 `[1, I, 1, S]`; it is cast once to fp16, every projection is
/// evaluated from that shared activation, and each output is cast back to fp32
/// `[1, O_n, 1, S]`. All weights share one CoreML Blob v2 file.
pub fn parallel_dense_fp16_f32_io(
    in_features: usize,
    spatial: usize,
    projections: &[DenseProjection],
) -> Result<MilProgram> {
    if in_features == 0 || spatial == 0 || projections.is_empty() {
        return Err(AneError::InvalidArgument(
            "parallel dense island requires non-zero input/spatial dimensions and projections"
                .into(),
        ));
    }
    if spatial < 16 || spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(format!(
            "parallel dense spatial dimension {spatial} must be >= 16 and a multiple of 16"
        )));
    }

    const WEIGHT_PATH: &str = "@model_path/weights/dense.bin";
    let mut blob = BlobV2Builder::new();
    let mut offsets = Vec::with_capacity(projections.len());
    for (idx, projection) in projections.iter().enumerate() {
        if projection.out_features == 0 {
            return Err(AneError::InvalidArgument(format!(
                "dense projection {idx} has zero output features"
            )));
        }
        let expected = projection
            .out_features
            .checked_mul(in_features)
            .ok_or_else(|| AneError::InvalidArgument("dense weight shape overflow".into()))?;
        if projection.weights_fp16.len() != expected {
            return Err(AneError::InvalidArgument(format!(
                "dense projection {:?} has {} fp16 weights, expected {}x{} = {expected}",
                projection.name,
                projection.weights_fp16.len(),
                projection.out_features,
                in_features
            )));
        }
        offsets.push(blob.push_fp16(&projection.weights_fp16)?.get());
    }

    let mut body = String::new();
    body.push_str(&format!(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]\n\
{{\n\
 func main<ios18>(tensor<fp32, [1, {in_features}, 1, {spatial}]> x) {{\n\
  string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n\
  tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
  tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
  tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
  int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n\
  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n\
  tensor<fp16, [1, {in_features}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    for (idx, (projection, offset)) in projections.iter().zip(offsets.iter()).enumerate() {
        body.push_str(&format!(
            "  tensor<fp16, [{out}, {in_features}, 1, 1]> W{idx} = const()[name = string(\"W{idx}\"), val = tensor<fp16, [{out}, {in_features}, 1, 1]>(BLOBFILE(path = string(\"{WEIGHT_PATH}\"), offset = uint64({offset})))];\n\
  tensor<fp16, [1, {out}, 1, {spatial}]> y{idx}_16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W{idx}, x = x16)[name = string(\"conv_{idx}\")];\n\
  string to_fp32_{idx} = const()[name = string(\"to_fp32_{idx}\"), val = string(\"fp32\")];\n\
  tensor<fp32, [1, {out}, 1, {spatial}]> y{idx} = cast(dtype = to_fp32_{idx}, x = y{idx}_16)[name = string(\"cast_out_{idx}\")];\n",
            out = projection.out_features
        ));
    }

    let outputs = (0..projections.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(" }} -> ({outputs});\n}}\n"));

    Ok(MilProgram::new(body)
        .with_weight(WeightBlob::new(WEIGHT_PATH, blob.into_bytes()).descriptor_offset(0)))
}

/// Build Qwen4Exp MTP's fixed dense input projections as one ANE island.
///
/// This intentionally begins *after* the two MTP RMSNorms. Qwen4Exp's MTP
/// front-end applies `fc_embedding` once to the normalized new-token embedding
/// and applies one shared `fc_hidden` matrix independently to every gated-
/// residual (HC) hidden branch. The outputs remain separate: the projected
/// embedding becomes `prev_block_output`, while the projected HC branches form
/// the draft layer's multi-stream hidden state.
///
/// The private ANE compiler is most reliable with one activation input, so the
/// caller packs both normalized values into one fp32 IOSurface:
/// `packed = embedding[S] | hidden[hc_count*S]`, shape
/// `[1, H, 1, (hc_count+1)*S]`. Both HxH projections currently evaluate over
/// that whole packed surface and return the same packed shape; callers consume
/// only the `fc_embedding` result from the first `S` lanes and the `fc_hidden`
/// result from the remaining HC lanes. This intentionally trades about 2x the
/// ideal arithmetic for a single hardware-qualified ANE dispatch.
///
/// Note that Qwen4Exp's hidden RMSNorm is over the concatenated `hc_count*H`
/// vector *before* this packing; it is not a per-branch normalization.
pub fn qwen4_mtp_input_projections_fp16_f32_io(
    hidden_size: usize,
    token_spatial: usize,
    hc_count: usize,
    fc_embedding: DenseProjection,
    fc_hidden: DenseProjection,
) -> Result<MilProgram> {
    if hidden_size == 0 || hc_count == 0 {
        return Err(AneError::InvalidArgument(
            "Qwen4 MTP input projections require non-zero hidden size and HC count".into(),
        ));
    }
    if token_spatial < 16 || token_spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(format!(
            "Qwen4 MTP token spatial dimension {token_spatial} must be >= 16 and a multiple of 16"
        )));
    }
    let hidden_spatial = token_spatial
        .checked_mul(hc_count)
        .ok_or_else(|| AneError::InvalidArgument("Qwen4 MTP hidden spatial overflow".into()))?;
    let packed_spatial = token_spatial
        .checked_add(hidden_spatial)
        .ok_or_else(|| AneError::InvalidArgument("Qwen4 MTP packed spatial overflow".into()))?;

    for (role, projection) in [("fc_embedding", &fc_embedding), ("fc_hidden", &fc_hidden)] {
        if projection.out_features != hidden_size {
            return Err(AneError::InvalidArgument(format!(
                "Qwen4 MTP {role} has {} output features, expected hidden size {hidden_size}",
                projection.out_features
            )));
        }
    }

    // Keep the first production probe on the already hardware-qualified ANE
    // primitive: both HxH projections evaluate over the whole packed spatial
    // surface and therefore return the same shape. The caller consumes only
    // fc_embedding[:, 0..S] and fc_hidden[:, S..packed_spatial]. This spends
    // roughly 2x the ideal projection arithmetic, but avoids unsupported MIL
    // slicing and gives us one dispatch with stable compiled weights.
    parallel_dense_fp16_f32_io(
        hidden_size,
        packed_spatial,
        &[fc_embedding, fc_hidden],
    )

}

/// Build the decode front-half of a GDN block as one ANE program.
///
/// Inputs:
/// - `x`: fp32 `[1, I, 1, S]`; callers repeat the current token across `S`.
/// - one fp32 `packed` input `[1, I, 1, 2*S]`: first S lanes are x; the
///   second S lanes flatten `[C, kernel]` conv history. The first
///   `kernel-1` entries per qkv channel carry prior values; the final slot is
///   ignored and replaced by current qkv.
///
/// The program computes qkv/z/a/b projections, forms the exact causal
/// `kernel`-tap qkv window, runs a depthwise convolution, applies SiLU, and
/// returns three fp32 outputs: `y`, `qkv`, and packed `z | a | b`.
/// `y` is repeated across S so every public ANE tensor keeps an ANE-friendly
/// spatial multiple; callers consume column zero only.
pub fn gdn_front_fp16_f32_io(
    in_features: usize,
    spatial: usize,
    projections: &[DenseProjection],
    conv_weights_fp16: &[u16],
    kernel: usize,
) -> Result<MilProgram> {
    if projections.is_empty() || projections.len() > 4 {
        return Err(AneError::InvalidArgument(format!(
            "GDN front island requires qkv plus up to three auxiliary projections, got {}",
            projections.len()
        )));
    }
    if in_features == 0 || spatial < 16 || spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(
            "GDN front island requires non-zero input and spatial >=16 multiple of 16".into(),
        ));
    }
    if kernel == 0 || kernel > spatial {
        return Err(AneError::InvalidArgument(format!(
            "invalid GDN convolution kernel {kernel} for spatial {spatial}"
        )));
    }
    let qkv_rows = projections[0].out_features;
    if qkv_rows == 0 || conv_weights_fp16.len() != qkv_rows * kernel {
        return Err(AneError::InvalidArgument(format!(
            "GDN conv has {} fp16 taps, expected {}x{} = {}",
            conv_weights_fp16.len(),
            qkv_rows,
            kernel,
            qkv_rows * kernel
        )));
    }
    if qkv_rows * kernel != in_features * spatial {
        return Err(AneError::InvalidArgument(format!(
            "GDN history pack requires qkv_rows*kernel == in_features*spatial ({} != {})",
            qkv_rows * kernel,
            in_features * spatial
        )));
    }

    const WEIGHT_PATH: &str = "@model_path/weights/gdn-front.bin";
    let mut blob = BlobV2Builder::new();
    let mut offsets = Vec::with_capacity(4);
    for (idx, projection) in projections.iter().enumerate() {
        if projection.out_features == 0 {
            return Err(AneError::InvalidArgument(format!(
                "GDN projection {idx} has zero output features"
            )));
        }
        let expected = projection
            .out_features
            .checked_mul(in_features)
            .ok_or_else(|| AneError::InvalidArgument("GDN dense weight shape overflow".into()))?;
        if projection.weights_fp16.len() != expected {
            return Err(AneError::InvalidArgument(format!(
                "GDN projection {:?} has {} fp16 weights, expected {expected}",
                projection.name,
                projection.weights_fp16.len()
            )));
        }
        offsets.push(blob.push_fp16(&projection.weights_fp16)?.get());
    }
    // The private ANE compiler on current macOS rejects grouped/depthwise
    // conv. Store one per-channel vector per temporal tap so the exact same
    // depthwise dot product can be expressed as elementwise mul + add.
    let mut tap_offsets = Vec::with_capacity(kernel);
    for tap in 0..kernel {
        let weights = (0..qkv_rows)
            .map(|ch| conv_weights_fp16[ch * kernel + tap])
            .collect::<Vec<_>>();
        tap_offsets.push(blob.push_fp16(&weights)?.get());
    }

    let packed_spatial = spatial * 2;
    let mut body = String::new();
    body.push_str(&format!(
        "program(1.3)\n\\
[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]\n\\
{{\n\\
 func main<ios18>(tensor<fp32, [1, {in_features}, 1, {packed_spatial}]> packed) {{\n\\
  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n\\
  tensor<fp16, [1, {in_features}, 1, {packed_spatial}]> packed16 = cast(dtype = to_fp16, x = packed)[name = string(\"cast_packed\")];\n\\
  tensor<int32, [4]> xb = const()[name = string(\"xb\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\\
  tensor<int32, [4]> xs = const()[name = string(\"xs\"), val = tensor<int32, [4]>([1, {in_features}, 1, {spatial}])];\n\\
  tensor<fp16, [1, {in_features}, 1, {spatial}]> x16 = slice_by_size(x = packed16, begin = xb, size = xs)[name = string(\"slice_x\")];\n\\
  tensor<int32, [4]> hbpack = const()[name = string(\"hbpack\"), val = tensor<int32, [4]>([0, 0, 0, {spatial}])];\n\\
  tensor<fp16, [1, {in_features}, 1, {spatial}]> hp16 = slice_by_size(x = packed16, begin = hbpack, size = xs)[name = string(\"slice_history\")];\n\\
  tensor<int32, [4]> hshape = const()[name = string(\"hshape\"), val = tensor<int32, [4]>([1, {qkv_rows}, 1, {kernel}])];\n\\
  tensor<fp16, [1, {qkv_rows}, 1, {kernel}]> h16 = reshape(shape = hshape, x = hp16)[name = string(\"reshape_history\")];\n\\
  string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n\\
  tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n\\
  tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\\
  tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n\\
  int32 gr1 = const()[name = string(\"gr1\"), val = int32(1)];\n"
    ));

    for (idx, (projection, offset)) in projections.iter().zip(offsets.iter()).enumerate() {
        body.push_str(&format!(
            "  tensor<fp16, [{out}, {in_features}, 1, 1]> W{idx} = const()[name = string(\"W{idx}\"), val = tensor<fp16, [{out}, {in_features}, 1, 1]>(BLOBFILE(path = string(\"{WEIGHT_PATH}\"), offset = uint64({offset})))];\n\\
  tensor<fp16, [1, {out}, 1, {spatial}]> p{idx} = conv(dilations = dl, groups = gr1, pad = pd, pad_type = pt, strides = st, weight = W{idx}, x = x16)[name = string(\"proj_{idx}\")];\n",
            out = projection.out_features,
        ));
    }

    let hist_len = kernel.saturating_sub(1);
    body.push_str(&format!(
        "  tensor<int32, [4]> one = const()[name = string(\"one\"), val = tensor<int32, [4]>([1, {qkv_rows}, 1, 1])];\n\\
  tensor<int32, [4]> bq = const()[name = string(\"bq\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\\
  tensor<fp16, [1, {qkv_rows}, 1, 1]> qc = slice_by_size(x = p0, begin = bq, size = one)[name = string(\"qkv_current\")];\n"
    ));

    let mut terms = Vec::with_capacity(kernel);
    for (tap, offset) in tap_offsets.iter().enumerate() {
        body.push_str(&format!(
            "  tensor<fp16, [1, {qkv_rows}, 1, 1]> T{tap} = const()[name = string(\"T{tap}\"), val = tensor<fp16, [1, {qkv_rows}, 1, 1]>(BLOBFILE(path = string(\"{WEIGHT_PATH}\"), offset = uint64({offset})))];\n"
        ));
        let source = if tap < hist_len {
            body.push_str(&format!(
                "  tensor<int32, [4]> bh{tap} = const()[name = string(\"bh{tap}\"), val = tensor<int32, [4]>([0, 0, 0, {tap}])];\n\\
  tensor<fp16, [1, {qkv_rows}, 1, 1]> h{tap} = slice_by_size(x = h16, begin = bh{tap}, size = one)[name = string(\"hist_{tap}\")];\n"
            ));
            format!("h{tap}")
        } else {
            "qc".to_string()
        };
        body.push_str(&format!(
            "  tensor<fp16, [1, {qkv_rows}, 1, 1]> m{tap} = mul(x = {source}, y = T{tap})[name = string(\"tap_mul_{tap}\")];\n"
        ));
        terms.push(format!("m{tap}"));
    }
    let mut acc = terms[0].clone();
    for tap in 1..kernel {
        let next = format!("cv{tap}");
        body.push_str(&format!(
            "  tensor<fp16, [1, {qkv_rows}, 1, 1]> {next} = add(x = {acc}, y = m{tap})[name = string(\"tap_add_{tap}\")];\n"
        ));
        acc = next;
    }
    body.push_str(&format!(
        "  tensor<fp16, [1, {qkv_rows}, 1, 1]> sg = sigmoid(x = {acc})[name = string(\"sigmoid\")];\n\\
  tensor<fp16, [1, {qkv_rows}, 1, 1]> y1 = mul(x = {acc}, y = sg)[name = string(\"silu\")];\n\\
  int32 sax = const()[name = string(\"sax\"), val = int32(3)];\n\\
  bool sint = const()[name = string(\"sint\"), val = bool(false)];\n"
    ));

    let repeat_y = std::iter::repeat_n("y1", spatial)
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(
        "  tensor<fp16, [1, {qkv_rows}, 1, {spatial}]> y16 = concat(axis = sax, interleave = sint, values = ({repeat_y}))[name = string(\"repeat_y\")];\n\\
  string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n\\
  tensor<fp32, [1, {qkv_rows}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_y\")];\n\\
  tensor<fp32, [1, {qkv_rows}, 1, {spatial}]> qkv = cast(dtype = to_fp32, x = p0)[name = string(\"cast_qkv\")];\n"
    ));
    if projections.len() == 1 {
        body.push_str(" } -> (y, qkv);\n}\n");
    } else {
        let aux_rows = projections[1..]
            .iter()
            .map(|projection| projection.out_features)
            .sum::<usize>();
        body.push_str(
            "  int32 cax = const()[name = string(\"cax\"), val = int32(1)];\n\\
  bool cint = const()[name = string(\"cint\"), val = bool(false)];\n",
        );
        let aux_values = (1..projections.len())
            .map(|idx| format!("p{idx}"))
            .collect::<Vec<_>>()
            .join(", ");
        if projections.len() == 2 {
            body.push_str(&format!(
                "  tensor<fp32, [1, {aux_rows}, 1, {spatial}]> aux = cast(dtype = to_fp32, x = p1)[name = string(\"cast_aux\")];\n\\
 }} -> (y, qkv, aux);\n}}\n"
            ));
        } else {
            body.push_str(&format!(
                "  tensor<fp16, [1, {aux_rows}, 1, {spatial}]> aux16 = concat(axis = cax, interleave = cint, values = ({aux_values}))[name = string(\"pack_aux\")];\n\\
  tensor<fp32, [1, {aux_rows}, 1, {spatial}]> aux = cast(dtype = to_fp32, x = aux16)[name = string(\"cast_aux\")];\n\\
 }} -> (y, qkv, aux);\n}}\n"
            ));
        }
    }

    Ok(MilProgram::new(body)
        .with_weight(WeightBlob::new(WEIGHT_PATH, blob.into_bytes()).descriptor_offset(0)))
}

/// Build one fixed-shape, multi-projection ANE dense island with fp16 I/O.
///
/// Input is fp16 `[1, I, 1, S]`; outputs are fp16 `[1, O_n, 1, S]`.
/// The projection weights use the same Blob v2 constant representation as
/// [`parallel_dense_fp16_f32_io`], but no boundary casts are inserted. This is
/// the preferred decode form when the host only needs to convert one token at
/// the island boundary.
pub fn parallel_dense_fp16_io(
    in_features: usize,
    spatial: usize,
    projections: &[DenseProjection],
) -> Result<MilProgram> {
    // Start from the already hardware-qualified fp32-boundary program so the
    // private compiler sees byte-for-byte identical weight declarations and
    // convolution structure. Then remove only the boundary casts.
    let base = parallel_dense_fp16_f32_io(in_features, spatial, projections)?;
    let mut text = base.text;

    text = text.replacen(
        &format!("func main<ios18>(tensor<fp32, [1, {in_features}, 1, {spatial}]> x)"),
        &format!("func main<ios18>(tensor<fp16, [1, {in_features}, 1, {spatial}]> x)"),
        1,
    );
    text = text.replace(
        "  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
        "",
    );
    text = text.replace(
        &format!(
            "  tensor<fp16, [1, {in_features}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        ),
        "",
    );
    text = text.replace("x = x16", "x = x");

    for (idx, projection) in projections.iter().enumerate() {
        text = text.replace(
            &format!(
                "  string to_fp32_{idx} = const()[name = string(\"to_fp32_{idx}\"), val = string(\"fp32\")];\n"
            ),
            "",
        );
        text = text.replace(
            &format!(
                "  tensor<fp32, [1, {out}, 1, {spatial}]> y{idx} = cast(dtype = to_fp32_{idx}, x = y{idx}_16)[name = string(\"cast_out_{idx}\")];\n",
                out = projection.out_features,
            ),
            "",
        );
    }

    let old_outputs = (0..projections.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let fp16_outputs = (0..projections.len())
        .map(|idx| format!("y{idx}_16"))
        .collect::<Vec<_>>()
        .join(", ");
    text = text.replacen(
        &format!(" }} -> ({old_outputs});"),
        &format!(" }} -> ({fp16_outputs});"),
        1,
    );

    Ok(MilProgram {
        text,
        weights: base.weights,
    })
}

/// Build one reusable fixed-shape dense island whose projection weights are
/// runtime inputs instead of BLOBFILE constants.
///
/// Input order is `x, W0, W1, ...`; each `Wn` is fp16 `[O_n, I, 1, 1]`.
/// This lets one compiled ANE program execute the same geometry for many model
/// layers while each layer keeps its own persistent IOSurface-backed weights.
pub fn parallel_dense_dynamic_fp16_f32_io(
    in_features: usize,
    spatial: usize,
    out_features: &[usize],
) -> Result<MilProgram> {
    if in_features == 0 || spatial == 0 || out_features.is_empty() {
        return Err(AneError::InvalidArgument(
            "dynamic parallel dense island requires non-zero input/spatial dimensions and projections"
                .into(),
        ));
    }
    if spatial < 16 || spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(format!(
            "dynamic parallel dense spatial dimension {spatial} must be >= 16 and a multiple of 16"
        )));
    }
    if out_features.iter().any(|&out| out == 0) {
        return Err(AneError::InvalidArgument(
            "dynamic parallel dense projection output sizes must be non-zero".into(),
        ));
    }

    let mut signature = format!("tensor<fp32, [1, {in_features}, 1, {spatial}]> x");
    for (idx, &out) in out_features.iter().enumerate() {
        signature.push_str(&format!(
            ", tensor<fp16, [{out}, {in_features}, 1, 1]> W{idx}"
        ));
    }

    let mut body = String::from(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n",
    );
    body.push_str(&format!(" func main<ios18>({signature}) {{\n"));
    body.push_str(
        "  string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n\
  tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
  tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
  tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
  int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n\
  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    body.push_str(&format!(
        "  tensor<fp16, [1, {in_features}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    for (idx, &out) in out_features.iter().enumerate() {
        body.push_str(&format!(
            "  tensor<fp16, [1, {out}, 1, {spatial}]> y{idx}_16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W{idx}, x = x16)[name = string(\"conv_{idx}\")];\n"
        ));
        body.push_str(&format!(
            "  string to_fp32_{idx} = const()[name = string(\"to_fp32_{idx}\"), val = string(\"fp32\")];\n"
        ));
        body.push_str(&format!(
            "  tensor<fp32, [1, {out}, 1, {spatial}]> y{idx} = cast(dtype = to_fp32_{idx}, x = y{idx}_16)[name = string(\"cast_out_{idx}\")];\n"
        ));
    }

    let outputs = (0..out_features.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(" }} -> ({outputs});\n}}\n"));
    Ok(MilProgram::new(body))
}

/// Spatial packing contract for [`parallel_dense_packed_dynamic_fp16_f32_io`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedDenseLayout {
    pub in_features: usize,
    pub token_spatial: usize,
    pub total_spatial: usize,
    /// Spatial start for each transposed `[I, O]` weight matrix.
    pub weight_offsets: Vec<usize>,
    pub out_features: Vec<usize>,
}

/// Build one reusable single-input ANE dense island with packed dynamic weights.
///
/// ANE's private runtime rejects the obvious multi-input dynamic-weight form on
/// the currently validated M2 (`0x1d` request cancellation). This form follows
/// the working low-level contract used by dynamic ANE research kernels: one
/// fp16 IOSurface `[1, I, 1, S + sum(O)]` contains both activations and weights.
/// The graph slices activation `[I,S]` and each transposed weight matrix `[I,O]`,
/// reshapes them to matmul-friendly rank-4 tensors, and emits fp32 outputs.
///
/// Packed memory for every input channel `i` is:
/// `x[i, 0..S], W0^T[i, 0..O0], W1^T[i, 0..O1], ...`.
pub fn parallel_dense_packed_dynamic_fp16_f32_io(
    in_features: usize,
    token_spatial: usize,
    out_features: &[usize],
) -> Result<(MilProgram, PackedDenseLayout)> {
    if in_features == 0 || token_spatial == 0 || out_features.is_empty() {
        return Err(AneError::InvalidArgument(
            "packed dynamic dense requires non-zero input/spatial dimensions and projections"
                .into(),
        ));
    }
    if token_spatial < 16 || token_spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(format!(
            "packed dynamic dense token spatial {token_spatial} must be >= 16 and a multiple of 16"
        )));
    }
    if out_features.iter().any(|&out| out == 0) {
        return Err(AneError::InvalidArgument(
            "packed dynamic dense projection output sizes must be non-zero".into(),
        ));
    }

    let mut weight_offsets = Vec::with_capacity(out_features.len());
    let mut total_spatial = token_spatial;
    for &out in out_features {
        weight_offsets.push(total_spatial);
        total_spatial = total_spatial
            .checked_add(out)
            .ok_or_else(|| AneError::InvalidArgument("packed dense spatial overflow".into()))?;
    }

    let mut body = String::from(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n",
    );
    body.push_str(&format!(
        " func main<ios18>(tensor<fp16, [1, {in_features}, 1, {total_spatial}]> packed) {{\n"
    ));
    // Slice/prepare the activation once; all projections share it.
    body.push_str(&format!(
        "  tensor<int32, [4]> act_b = const()[name=string(\"act_b\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
  tensor<int32, [4]> act_s = const()[name=string(\"act_s\"), val=tensor<int32, [4]>([1,{in_features},1,{token_spatial}])];\n\
  tensor<fp16, [1,{in_features},1,{token_spatial}]> act = slice_by_size(x=packed,begin=act_b,size=act_s)[name=string(\"act\")];\n\
  tensor<int32, [4]> act_r = const()[name=string(\"act_r\"), val=tensor<int32, [4]>([1,1,{in_features},{token_spatial}])];\n\
  tensor<fp16, [1,1,{in_features},{token_spatial}]> act2 = reshape(shape=act_r,x=act)[name=string(\"act2\")];\n\
  tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n\
  tensor<fp16, [1,1,{token_spatial},{in_features}]> act3 = transpose(perm=pm,x=act2)[name=string(\"act3\")];\n\
  bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
    ));

    for (idx, (&out, &offset)) in out_features.iter().zip(&weight_offsets).enumerate() {
        body.push_str(&format!(
            "  tensor<int32, [4]> w{idx}_b = const()[name=string(\"w{idx}_b\"), val=tensor<int32, [4]>([0,0,0,{offset}])];\n\
  tensor<int32, [4]> w{idx}_s = const()[name=string(\"w{idx}_s\"), val=tensor<int32, [4]>([1,{in_features},1,{out}])];\n\
  tensor<fp16, [1,{in_features},1,{out}]> w{idx}_slice = slice_by_size(x=packed,begin=w{idx}_b,size=w{idx}_s)[name=string(\"w{idx}_slice\")];\n\
  tensor<int32, [4]> w{idx}_r = const()[name=string(\"w{idx}_r\"), val=tensor<int32, [4]>([1,1,{in_features},{out}])];\n\
  tensor<fp16, [1,1,{in_features},{out}]> w{idx} = reshape(shape=w{idx}_r,x=w{idx}_slice)[name=string(\"w{idx}\")];\n\
  tensor<fp16, [1,1,{token_spatial},{out}]> mm{idx} = matmul(transpose_x=bF,transpose_y=bF,x=act3,y=w{idx})[name=string(\"mm{idx}\")];\n\
  tensor<fp16, [1,1,{out},{token_spatial}]> mm{idx}_t = transpose(perm=pm,x=mm{idx})[name=string(\"mm{idx}_t\")];\n\
  tensor<int32, [4]> y{idx}_r = const()[name=string(\"y{idx}_r\"), val=tensor<int32, [4]>([1,{out},1,{token_spatial}])];\n\
  tensor<fp16, [1,{out},1,{token_spatial}]> y{idx}_16 = reshape(shape=y{idx}_r,x=mm{idx}_t)[name=string(\"y{idx}_16\")];\n\
  string y{idx}_dtype = const()[name=string(\"y{idx}_dtype\"), val=string(\"fp32\")];\n\
  tensor<fp32, [1,{out},1,{token_spatial}]> y{idx} = cast(dtype=y{idx}_dtype,x=y{idx}_16)[name=string(\"y{idx}\")];\n"
        ));
    }

    let outputs = (0..out_features.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(" }} -> ({outputs});\n}}\n"));

    Ok((
        MilProgram::new(body),
        PackedDenseLayout {
            in_features,
            token_spatial,
            total_spatial,
            weight_offsets,
            out_features: out_features.to_vec(),
        },
    ))
}

/// Build the hardware-qualified packed dynamic dense contract used by current
/// ANE reverse-engineering kernels: one **fp32** IOSurface contains both the
/// activation tile and transposed fp32 weights. The graph casts the entire
/// packed tensor to fp16 internally, performs the matmul(s), and casts outputs
/// back to fp32. On the validated M2 this boundary contract is materially
/// different from an fp16 IOSurface input: the latter is rejected at evaluate
/// time with ANE status 0x1d.
///
/// Packed memory for input channel `i` is:
/// `x[i, 0..S], W0^T[i, 0..O0], W1^T[i, 0..O1], ...`.
pub fn parallel_dense_packed_dynamic_f32_io(
    in_features: usize,
    token_spatial: usize,
    out_features: &[usize],
) -> Result<(MilProgram, PackedDenseLayout)> {
    if in_features == 0 || token_spatial == 0 || out_features.is_empty() {
        return Err(AneError::InvalidArgument(
            "packed dynamic f32 dense requires non-zero input/spatial dimensions and projections"
                .into(),
        ));
    }
    if out_features.iter().any(|&out| out == 0) {
        return Err(AneError::InvalidArgument(
            "packed dynamic f32 projection output sizes must be non-zero".into(),
        ));
    }

    let mut weight_offsets = Vec::with_capacity(out_features.len());
    let mut total_spatial = token_spatial;
    for &out in out_features {
        weight_offsets.push(total_spatial);
        total_spatial = total_spatial
            .checked_add(out)
            .ok_or_else(|| AneError::InvalidArgument("packed dynamic f32 spatial overflow".into()))?;
    }

    let mut body = String::from(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n",
    );
    body.push_str(&format!(
        " func main<ios18>(tensor<fp32, [1, {in_features}, 1, {total_spatial}]> x) {{\n"
    ));
    // Match the upstream hardware-qualified contract: cast the complete packed
    // IOSurface once, then slice activation/weight regions from the fp16 tensor.
    body.push_str(&format!(
        "  string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n\
  tensor<fp16, [1, {in_features}, 1, {total_spatial}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n\
  tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n\
  tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{in_features},1,{token_spatial}])];\n\
  tensor<fp16, [1,{in_features},1,{token_spatial}]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n\
  tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,{in_features},{token_spatial}])];\n\
  tensor<fp16, [1,1,{in_features},{token_spatial}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n\
  tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n\
  tensor<fp16, [1,1,{token_spatial},{in_features}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
    ));

    for (idx, (&out, &offset)) in out_features.iter().zip(&weight_offsets).enumerate() {
        body.push_str(&format!(
            "  tensor<int32, [4]> bw{idx} = const()[name = string(\"bw{idx}\"), val = tensor<int32, [4]>([0,0,0,{offset}])];\n\
  tensor<int32, [4]> sw{idx} = const()[name = string(\"sw{idx}\"), val = tensor<int32, [4]>([1,{in_features},1,{out}])];\n\
  tensor<fp16, [1,{in_features},1,{out}]> wt{idx} = slice_by_size(x=xh,begin=bw{idx},size=sw{idx})[name=string(\"wt{idx}\")];\n\
  tensor<int32, [4]> rw{idx} = const()[name = string(\"rw{idx}\"), val = tensor<int32, [4]>([1,1,{in_features},{out}])];\n\
  tensor<fp16, [1,1,{in_features},{out}]> W{idx} = reshape(shape=rw{idx},x=wt{idx})[name=string(\"W{idx}\")];\n\
  bool bF{idx} = const()[name = string(\"bF{idx}\"), val = bool(false)];\n\
  tensor<fp16, [1,1,{token_spatial},{out}]> yh{idx} = matmul(transpose_x=bF{idx},transpose_y=bF{idx},x=a3,y=W{idx})[name=string(\"mm{idx}\")];\n\
  tensor<fp16, [1,1,{out},{token_spatial}]> yt{idx} = transpose(perm=pm,x=yh{idx})[name=string(\"yt{idx}\")];\n\
  tensor<int32, [4]> ro{idx} = const()[name = string(\"ro{idx}\"), val = tensor<int32, [4]>([1,{out},1,{token_spatial}])];\n\
  tensor<fp16, [1,{out},1,{token_spatial}]> yr{idx} = reshape(shape=ro{idx},x=yt{idx})[name=string(\"yr{idx}\")];\n\
  string to32_{idx} = const()[name = string(\"to32_{idx}\"), val = string(\"fp32\")];\n\
  tensor<fp32, [1,{out},1,{token_spatial}]> y{idx} = cast(dtype=to32_{idx},x=yr{idx})[name=string(\"cout{idx}\")];\n"
        ));
    }

    let outputs = (0..out_features.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(" }} -> ({outputs});\n}}\n"));

    Ok((
        MilProgram::new(body),
        PackedDenseLayout {
            in_features,
            token_spatial,
            total_spatial,
            weight_offsets,
            out_features: out_features.to_vec(),
        },
    ))
}

/// Build a reusable single-input dynamic dense island.
///
/// The fp16 input is `[1, I, 1, S + sum(O_n)]`. For every input channel, the
/// spatial row contains the activation tile first, followed by one row from
/// each `[I, O_n]` weight matrix. MIL slices the packed regions, reshapes them
/// into matrices, and evaluates `activation @ weight` with `matmul`.
///
/// This mirrors the single-IOSurface dynamic-weight pattern used by current
/// ANE transformer experiments and avoids the private runtime's multi-input
/// request failure (`0x1d`). Outputs are fp16 `[1, O_n, 1, S]`.
pub fn parallel_dense_packed_fp16_io(
    in_features: usize,
    spatial: usize,
    out_features: &[usize],
) -> Result<MilProgram> {
    if in_features == 0 || spatial == 0 || out_features.is_empty() {
        return Err(AneError::InvalidArgument(
            "packed parallel dense island requires non-zero dimensions and projections".into(),
        ));
    }
    if out_features.iter().any(|&out| out == 0) {
        return Err(AneError::InvalidArgument(
            "packed parallel dense projection output sizes must be non-zero".into(),
        ));
    }
    let packed_spatial = out_features.iter().try_fold(spatial, |acc, &out| {
        acc.checked_add(out)
            .ok_or_else(|| AneError::InvalidArgument("packed dense spatial overflow".into()))
    })?;

    let mut body = String::from(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n",
    );
    body.push_str(&format!(
        " func main<ios18>(tensor<fp16, [1, {in_features}, 1, {packed_spatial}]> x) {{\n"
    ));
    body.push_str(
        "  tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
  bool bF = const()[name=string(\"bF\"), val=bool(false)];\n\
  tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n",
    );
    body.push_str(&format!(
        "  tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{in_features},1,{spatial}])];\n\
  tensor<fp16, [1,{in_features},1,{spatial}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n\
  tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{in_features},{spatial}])];\n\
  tensor<fp16, [1,1,{in_features},{spatial}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n\
  tensor<fp16, [1,1,{spatial},{in_features}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
    ));

    let mut offset = spatial;
    for (idx, &out) in out_features.iter().enumerate() {
        body.push_str(&format!(
            "  tensor<int32, [4]> bw{idx} = const()[name=string(\"bw{idx}\"), val=tensor<int32, [4]>([0,0,0,{offset}])];\n\
  tensor<int32, [4]> sw{idx} = const()[name=string(\"sw{idx}\"), val=tensor<int32, [4]>([1,{in_features},1,{out}])];\n\
  tensor<fp16, [1,{in_features},1,{out}]> wt{idx} = slice_by_size(x=x,begin=bw{idx},size=sw{idx})[name=string(\"wt{idx}\")];\n\
  tensor<int32, [4]> rw{idx} = const()[name=string(\"rw{idx}\"), val=tensor<int32, [4]>([1,1,{in_features},{out}])];\n\
  tensor<fp16, [1,1,{in_features},{out}]> W{idx} = reshape(shape=rw{idx},x=wt{idx})[name=string(\"W{idx}\")];\n\
  tensor<fp16, [1,1,{spatial},{out}]> yh{idx} = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W{idx})[name=string(\"yh{idx}\")];\n\
  tensor<fp16, [1,1,{out},{spatial}]> yt{idx} = transpose(perm=pm,x=yh{idx})[name=string(\"yt{idx}\")];\n\
  tensor<int32, [4]> ro{idx} = const()[name=string(\"ro{idx}\"), val=tensor<int32, [4]>([1,{out},1,{spatial}])];\n\
  tensor<fp16, [1,{out},1,{spatial}]> y{idx} = reshape(shape=ro{idx},x=yt{idx})[name=string(\"y{idx}\")];\n"
        ));
        offset += out;
    }

    let outputs = (0..out_features.len())
        .map(|idx| format!("y{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(" }} -> ({outputs});\n}}\n"));
    Ok(MilProgram::new(body))
}

/// Weight-free fp32 -> fp16 -> ReLU -> fp32 fixture.
///
/// It is intentionally simple and contains no BLOBFILE constants, making it a
/// good end-to-end ABI/IOSurface correctness probe. The private ANE compiler is
/// sensitive to tensor geometry; spatial dimensions below 16 or not aligned to
/// 16 are rejected here rather than delegated to an undocumented compiler
/// failure mode.
pub fn relu_fp32(channels: usize, spatial: usize) -> Result<MilProgram> {
    if channels == 0 || spatial == 0 {
        return Err(AneError::InvalidArgument(
            "relu_fp32 dimensions must be non-zero".into(),
        ));
    }
    if spatial < 16 || spatial % 16 != 0 {
        return Err(AneError::InvalidArgument(format!(
            "relu_fp32 spatial dimension {spatial} must be >= 16 and a multiple of 16"
        )));
    }
    let text = format!(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]\n\
{{\n\
 func main<ios18>(tensor<fp32, [1, {channels}, 1, {spatial}]> x) {{\n\
  string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n\
  tensor<fp16, [1, {channels}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n\
  tensor<fp16, [1, {channels}, 1, {spatial}]> r16 = relu(x = x16)[name = string(\"relu\")];\n\
  string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n\
  tensor<fp32, [1, {channels}, 1, {spatial}]> y = cast(dtype = to_fp32, x = r16)[name = string(\"cast_out\")];\n\
 }} -> (y);\n\
}}\n"
    );
    Ok(MilProgram::new(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_fixture_has_expected_shape_and_no_weights() {
        let p = relu_fp32(16, 16).unwrap();
        assert!(p.text().contains("tensor<fp32, [1, 16, 1, 16]>"));
        assert!(p.text().contains("relu(x = x16)"));
        assert!(p.weights().is_empty());
    }

    #[test]
    fn relu_fixture_requires_ane_friendly_spatial_alignment() {
        assert!(relu_fp32(16, 15).is_err());
        assert!(relu_fp32(16, 17).is_err());
        assert!(relu_fp32(16, 32).is_ok());
    }

    #[test]
    fn parallel_dense_packs_multiple_projections_into_one_blob() {
        let identity = (0..16 * 16)
            .map(|i| if i / 16 == i % 16 { 0x3c00 } else { 0 })
            .collect();
        let half = vec![0x3800; 8 * 16];
        let p = parallel_dense_fp16_f32_io(
            16,
            16,
            &[
                DenseProjection::new("identity", 16, identity),
                DenseProjection::new("half", 8, half),
            ],
        )
        .unwrap();
        assert_eq!(p.weights().len(), 1);
        assert_eq!(
            u32::from_le_bytes(p.weights()[0].data()[0..4].try_into().unwrap()),
            2
        );
        assert!(p.text().contains("offset = uint64(64)"));
        assert!(p.text().contains("y0, y1"));
        p.validate().unwrap();
    }

    #[test]
    fn parallel_dense_rejects_wrong_weight_shape() {
        let bad = DenseProjection::new("bad", 16, vec![0; 15]);
        assert!(parallel_dense_fp16_f32_io(16, 16, &[bad]).is_err());
    }

    #[test]
    fn qwen4_mtp_input_projections_preserve_hc_stream_geometry() {
        let identity = (0..16 * 16)
            .map(|i| if i / 16 == i % 16 { 0x3c00 } else { 0 })
            .collect::<Vec<_>>();
        let half = (0..16 * 16)
            .map(|i| if i / 16 == i % 16 { 0x3800 } else { 0 })
            .collect::<Vec<_>>();
        let p = qwen4_mtp_input_projections_fp16_f32_io(
            16,
            16,
            4,
            DenseProjection::new("fc_embedding", 16, identity),
            DenseProjection::new("fc_hidden", 16, half),
        )
        .unwrap();

        assert_eq!(p.weights().len(), 1);
        assert!(p.text().contains("tensor<fp32, [1, 16, 1, 80]> x"));
        assert!(p.text().contains("tensor<fp32, [1, 16, 1, 80]> y0"));
        assert!(p.text().contains("tensor<fp32, [1, 16, 1, 80]> y1"));
        assert!(p.text().contains("y0, y1"));
        p.validate().unwrap();
    }

    #[test]
    fn qwen4_mtp_input_projections_reject_bad_geometry() {
        let good = vec![0x3c00; 16 * 16];
        assert!(qwen4_mtp_input_projections_fp16_f32_io(
            16,
            15,
            4,
            DenseProjection::new("fc_embedding", 16, good.clone()),
            DenseProjection::new("fc_hidden", 16, good.clone()),
        )
        .is_err());
        assert!(qwen4_mtp_input_projections_fp16_f32_io(
            16,
            16,
            0,
            DenseProjection::new("fc_embedding", 16, good.clone()),
            DenseProjection::new("fc_hidden", 16, good),
        )
        .is_err());
    }

    #[test]
    fn rejects_unsafe_weight_path() {
        let p = MilProgram::new("program(1.3) {}")
            .with_weight(WeightBlob::new("@model_path/../oops", vec![1]));
        assert!(p.validate().is_err());
    }
}
