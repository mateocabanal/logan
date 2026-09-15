pub fn lower_exact_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    let _ = math_format_for_dtype(&tensor.dtype)?;
    let data = read_tensor(tensor)?;
    let mut payload = vec![0_u8; TENSOR_HEADER_BYTES];
    payload[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut payload, 8, 1);
    put_u32(&mut payload, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut payload, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut payload, 32 + index * 8, *dimension);
    }
    put_u64(&mut payload, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut payload, 104, data.len() as u64);
    put_u64(&mut payload, 112, data.len() as u64);
    put_u32(&mut payload, 120, crc32c(&data));
    payload.extend_from_slice(&data);
    Ok(payload)
}

pub fn exact_tensor_stored_bytes(tensor: &source::TensorRef) -> Result<u64> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    let _ = math_format_for_dtype(&tensor.dtype)?;
    (TENSOR_HEADER_BYTES as u64)
        .checked_add(tensor.len)
        .ok_or_else(|| ColicError::Usage("projected tensor payload size overflows u64".into()))
}

pub fn stream_exact_tensor<W: Write + Seek>(
    tensor: &source::TensorRef,
    output: &mut W,
) -> Result<(u32, u32)> {
    exact_tensor_stored_bytes(tensor)?;
    let record_start = output.stream_position().map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut header = [0_u8; TENSOR_HEADER_BYTES];
    header[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut header, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut header, 32 + index * 8, *dimension);
    }
    put_u64(&mut header, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut header, 104, tensor.len);
    put_u64(&mut header, 112, tensor.len);
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut logical_state = !0_u32;
    let mut payload_state = !0_u32;
    copy_tensor_stream(tensor, output, &mut payload_state, Some(&mut logical_state))?;
    let logical_crc32c = !logical_state;
    put_u32(&mut header, 120, logical_crc32c);
    output.seek(SeekFrom::Start(record_start)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    output.seek(SeekFrom::Start(
        record_start
            .checked_add(TENSOR_HEADER_BYTES as u64)
            .and_then(|offset| offset.checked_add(tensor.len))
            .ok_or_else(|| ColicError::Usage("tensor output offset overflows u64".into()))?,
    )).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    Ok((logical_crc32c, crc32c_combine(crc32c(&header), !payload_state, tensor.len)))
}

fn crc32c_state(state: u32, bytes: &[u8]) -> u32 {
    logan_format::crc32c_update(state, bytes)
}

fn write_padding<W: Write>(
    output: &mut W,
    mut bytes: u64,
    state: &mut u32,
    path: &std::path::Path,
) -> Result<()> {
    const ZEROES: [u8; 16] = [0; 16];
    while bytes != 0 {
        let count = bytes.min(ZEROES.len() as u64) as usize;
        output.write_all(&ZEROES[..count]).map_err(|source| ColicError::Io {
            path: path.to_owned(), source,
        })?;
        *state = crc32c_state(*state, &ZEROES[..count]);
        bytes -= count as u64;
    }
    Ok(())
}

fn copy_tensor_stream<W: Write>(
    tensor: &source::TensorRef,
    output: &mut W,
    output_state: &mut u32,
    logical_state: Option<&mut u32>,
) -> Result<()> {
    if let Some(split) = parse_qwen3_next_split(&tensor.dtype) {
        return copy_qwen3_next_split(tensor, split, output, output_state, logical_state);
    }
    if let Some(bits) = parse_const_bf16_dtype(&tensor.dtype) {
        let bytes = bits.to_le_bytes();
        output.write_all(&bytes).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        *output_state = crc32c_state(*output_state, &bytes);
        if let Some(state) = logical_state {
            *state = crc32c_state(*state, &bytes);
        }
        return Ok(());
    }
    if let Some(scale_bits) = parse_ple_quant_dtype(&tensor.dtype) {
        return copy_bf16_to_e4m3(tensor, scale_bits, output, output_state, logical_state);
    }

    let mut input = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    input.seek(SeekFrom::Start(tensor.offset)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut remaining = tensor.len;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut logical_state = logical_state;
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        input.read_exact(&mut buffer[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        output.write_all(&buffer[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        *output_state = crc32c_state(*output_state, &buffer[..count]);
        if let Some(state) = logical_state.as_deref_mut() {
            *state = crc32c_state(*state, &buffer[..count]);
        }
        remaining -= count as u64;
    }
    Ok(())
}

fn copy_bf16_to_e4m3<W: Write>(
    tensor: &source::TensorRef,
    scale_bits: u16,
    output: &mut W,
    output_state: &mut u32,
    logical_state: Option<&mut u32>,
) -> Result<()> {
    let input_bytes = tensor
        .len
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("PLE BF16 input byte size overflows u64".into()))?;
    let mut input = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    input.seek(SeekFrom::Start(tensor.offset)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let lut = ple_bf16_to_e4m3_lut(scale_bits);
    let mut remaining = input_bytes;
    let mut src = vec![0_u8; 8 * 1024 * 1024];
    let mut dst = vec![0_u8; src.len() / 2];
    let mut logical_state = logical_state;
    while remaining != 0 {
        let mut count = remaining.min(src.len() as u64) as usize;
        count &= !1;
        if count == 0 {
            return Err(ColicError::Usage("odd BF16 PLE input length".into()));
        }
        input.read_exact(&mut src[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        let elements = count / 2;
        for (index, pair) in src[..count].chunks_exact(2).enumerate() {
            let bf16 = u16::from_le_bytes([pair[0], pair[1]]);
            let encoded = lut[bf16 as usize];
            if encoded > 0xff {
                let value = f32::from_bits((bf16 as u32) << 16);
                let scale = f32::from_bits((scale_bits as u32) << 16);
                return Err(ColicError::unsupported(
                    "Qwen4 PLE BF16->E4M3 lowering",
                    format!("value {value} exceeds derived E4M3 scale {scale}; refusing saturation"),
                ));
            }
            dst[index] = encoded as u8;
        }
        let bytes = &dst[..elements];
        output.write_all(bytes).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        *output_state = crc32c_state(*output_state, bytes);
        if let Some(state) = logical_state.as_deref_mut() {
            *state = crc32c_state(*state, bytes);
        }
        remaining -= count as u64;
    }
    Ok(())
}

fn ple_bf16_to_e4m3_lut(scale_bits: u16) -> std::sync::Arc<Vec<u16>> {
    use std::{collections::HashMap, sync::{Arc, Mutex, OnceLock}};
    static LUTS: OnceLock<Mutex<HashMap<u16, Arc<Vec<u16>>>>> = OnceLock::new();
    let cache = LUTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(lut) = cache.lock().unwrap().get(&scale_bits).cloned() {
        return lut;
    }
    let scale = f32::from_bits((scale_bits as u32) << 16);
    let lut = Arc::new(
        (0..=u16::MAX)
            .map(|bits| encode_bf16_e4m3(bits, scale))
            .collect::<Vec<_>>(),
    );
    cache.lock().unwrap().insert(scale_bits, lut.clone());
    lut
}

fn encode_bf16_e4m3(bits: u16, scale: f32) -> u16 {
    let value = f32::from_bits((bits as u32) << 16);
    if !value.is_finite() || !(scale.is_finite() && scale > 0.0) {
        return 0x100;
    }
    let scaled = value.abs() / scale;
    if scaled > 240.0 + f32::EPSILON * 240.0 {
        return 0x100;
    }
    let mut best = 0_u8;
    let mut best_error = f32::INFINITY;
    for code in 0_u8..=0x77 {
        let candidate = e4m3_positive(code);
        let error = (candidate - scaled).abs();
        if error < best_error {
            best_error = error;
            best = code;
        }
    }
    (best | if value.is_sign_negative() { 0x80 } else { 0 }) as u16
}

fn e4m3_positive(code: u8) -> f32 {
    let exp = ((code >> 3) & 0x0f) as i32;
    let mant = (code & 0x07) as f32;
    match exp {
        0 => mant * 0.001953125,
        e => (1.0 + mant * 0.125) * 2f32.powi(e - 7),
    }
}

fn parse_ple_quant_dtype(dtype: &str) -> Option<u16> {
    dtype
        .strip_prefix(crate::model::qwen4_exp::PLE_BF16_TO_F8_PREFIX)
        .and_then(|hex| u16::from_str_radix(hex, 16).ok())
}

fn parse_const_bf16_dtype(dtype: &str) -> Option<u16> {
    dtype
        .strip_prefix(crate::model::qwen4_exp::PLE_CONST_BF16_PREFIX)
        .and_then(|hex| u16::from_str_radix(hex, 16).ok())
}

pub fn crc32c_combine(mut left: u32, right: u32, mut right_len: u64) -> u32 {
    if right_len == 0 {
        return left;
    }
    const POLY: u32 = 0x82f6_3b78;
    let mut odd = [0_u32; 32];
    odd[0] = POLY;
    let mut row = 1_u32;
    for item in odd.iter_mut().skip(1) {
        *item = row;
        row <<= 1;
    }
    let mut even = gf2_matrix_square(&odd);
    odd = gf2_matrix_square(&even);
    loop {
        even = gf2_matrix_square(&odd);
        if right_len & 1 != 0 {
            left = gf2_matrix_times(&even, left);
        }
        right_len >>= 1;
        if right_len == 0 { break; }
        odd = gf2_matrix_square(&even);
        if right_len & 1 != 0 {
            left = gf2_matrix_times(&odd, left);
        }
        right_len >>= 1;
        if right_len == 0 { break; }
    }
    left ^ right
}

fn gf2_matrix_times(matrix: &[u32; 32], mut vector: u32) -> u32 {
    let mut sum = 0_u32;
    let mut index = 0;
    while vector != 0 {
        if vector & 1 != 0 { sum ^= matrix[index]; }
        vector >>= 1;
        index += 1;
    }
    sum
}

fn gf2_matrix_square(matrix: &[u32; 32]) -> [u32; 32] {
    std::array::from_fn(|index| gf2_matrix_times(matrix, matrix[index]))
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, 16)?;
    output.resize(offset as usize, 0);
    output.extend_from_slice(bytes);
    Ok(offset)
}


#[derive(Clone, Copy)]
enum Qwen3NextSplit {
    Qkvz {
        qkv: bool,
        key_heads: u32,
        key_dim: u32,
        value_heads: u32,
        value_dim: u32,
        hidden: u32,
    },
    Ba {
        b: bool,
        key_heads: u32,
        value_heads: u32,
        hidden: u32,
    },
}

fn parse_qwen3_next_split(dtype: &str) -> Option<Qwen3NextSplit> {
    if let Some(spec) = dtype.strip_prefix(crate::model::qwen3_next::QKVZ_SPLIT_PREFIX) {
        let mut p = spec.split(':');
        let role = p.next()?;
        let key_heads = p.next()?.parse().ok()?;
        let key_dim = p.next()?.parse().ok()?;
        let value_heads = p.next()?.parse().ok()?;
        let value_dim = p.next()?.parse().ok()?;
        let hidden = p.next()?.parse().ok()?;
        if p.next().is_some() || !matches!(role, "qkv" | "z") {
            return None;
        }
        return Some(Qwen3NextSplit::Qkvz {
            qkv: role == "qkv",
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            hidden,
        });
    }
    if let Some(spec) = dtype.strip_prefix(crate::model::qwen3_next::BA_SPLIT_PREFIX) {
        let mut p = spec.split(':');
        let role = p.next()?;
        let key_heads = p.next()?.parse().ok()?;
        let value_heads = p.next()?.parse().ok()?;
        let hidden = p.next()?.parse().ok()?;
        if p.next().is_some() || !matches!(role, "b" | "a") {
            return None;
        }
        return Some(Qwen3NextSplit::Ba {
            b: role == "b",
            key_heads,
            value_heads,
            hidden,
        });
    }
    None
}

fn copy_qwen3_next_split<W: Write>(
    tensor: &source::TensorRef,
    split: Qwen3NextSplit,
    output: &mut W,
    output_state: &mut u32,
    logical_state: Option<&mut u32>,
) -> Result<()> {
    let (key_heads, hidden, spans): (u32, u32, Vec<(u32, u32)>) = match split {
        Qwen3NextSplit::Qkvz {
            qkv,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            hidden,
        } => {
            if key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 {
                return Err(ColicError::InvalidSource {
                    path: tensor.source.clone(),
                    detail: "invalid Qwen3-Next qkvz head geometry".into(),
                });
            }
            let rep = value_heads / key_heads;
            let value_group = rep.checked_mul(value_dim).ok_or_else(|| ColicError::InvalidSource {
                path: tensor.source.clone(),
                detail: "Qwen3-Next qkvz value-group size overflow".into(),
            })?;
            let group_rows = key_dim
                .checked_mul(2)
                .and_then(|v| value_group.checked_mul(2).and_then(|vv| v.checked_add(vv)))
                .ok_or_else(|| ColicError::InvalidSource {
                    path: tensor.source.clone(),
                    detail: "Qwen3-Next qkvz group size overflow".into(),
                })?;
            let components = if qkv {
                vec![(0, key_dim), (key_dim, key_dim), (2 * key_dim, value_group)]
            } else {
                vec![(2 * key_dim + value_group, value_group)]
            };
            let mut spans = Vec::with_capacity(components.len() * key_heads as usize);
            for (component_start, rows) in components {
                for group in 0..key_heads {
                    spans.push((group * group_rows + component_start, rows));
                }
            }
            (key_heads, hidden, spans)
        }
        Qwen3NextSplit::Ba {
            b,
            key_heads,
            value_heads,
            hidden,
        } => {
            if key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 {
                return Err(ColicError::InvalidSource {
                    path: tensor.source.clone(),
                    detail: "invalid Qwen3-Next ba head geometry".into(),
                });
            }
            let rep = value_heads / key_heads;
            let group_rows = rep.checked_mul(2).ok_or_else(|| ColicError::InvalidSource {
                path: tensor.source.clone(),
                detail: "Qwen3-Next ba group size overflow".into(),
            })?;
            let component_start = if b { 0 } else { rep };
            let spans = (0..key_heads)
                .map(|group| (group * group_rows + component_start, rep))
                .collect();
            (key_heads, hidden, spans)
        }
    };
    let _ = key_heads; // geometry is validated above; spans are the actual copy plan.
    if hidden == 0 {
        return Err(ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split hidden size is zero".into(),
        });
    }
    let row_bytes = u64::from(hidden)
        .checked_mul(2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split row size overflow".into(),
        })?;
    let mut input = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(),
        source,
    })?;
    let mut logical_state = logical_state;
    let mut written = 0_u64;
    for (start_row, rows) in spans {
        let relative = u64::from(start_row).checked_mul(row_bytes).ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split source offset overflow".into(),
        })?;
        let offset = tensor.offset.checked_add(relative).ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split source offset overflow".into(),
        })?;
        let bytes = u64::from(rows).checked_mul(row_bytes).ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split block size overflow".into(),
        })?;
        let bytes_usize: usize = bytes.try_into().map_err(|_| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split block exceeds address space".into(),
        })?;
        let mut buffer = vec![0_u8; bytes_usize];
        input.seek(SeekFrom::Start(offset)).map_err(|source| ColicError::Io {
            path: tensor.source.clone(),
            source,
        })?;
        input.read_exact(&mut buffer).map_err(|source| ColicError::Io {
            path: tensor.source.clone(),
            source,
        })?;
        output.write_all(&buffer).map_err(|source| ColicError::Io {
            path: tensor.source.clone(),
            source,
        })?;
        *output_state = crc32c_state(*output_state, &buffer);
        if let Some(state) = logical_state.as_deref_mut() {
            *state = crc32c_state(*state, &buffer);
        }
        written = written.checked_add(bytes).ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next split output size overflow".into(),
        })?;
    }
    if written != tensor.len {
        return Err(ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: format!("Qwen3-Next split emitted {written} bytes, expected {}", tensor.len),
        });
    }
    Ok(())
}

fn read_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    if let Some(split) = parse_qwen3_next_split(&tensor.dtype) {
        let mut output = std::io::Cursor::new(Vec::with_capacity(tensor.len.try_into().map_err(|_| ColicError::Usage("Qwen3-Next split output too large".into()))?));
        let mut state = !0_u32;
        copy_qwen3_next_split(tensor, split, &mut output, &mut state, None)?;
        return Ok(output.into_inner());
    }
    if let Some(bits) = parse_const_bf16_dtype(&tensor.dtype) {
        return Ok(bits.to_le_bytes().to_vec());
    }
    if let Some(scale_bits) = parse_ple_quant_dtype(&tensor.dtype) {
        let mut output = std::io::Cursor::new(Vec::with_capacity(
            tensor
                .len
                .try_into()
                .map_err(|_| ColicError::Usage("PLE output too large".into()))?,
        ));
        let mut state = !0_u32;
        copy_bf16_to_e4m3(tensor, scale_bits, &mut output, &mut state, None)?;
        return Ok(output.into_inner());
    }
    let mut bytes = vec![0; tensor.len.try_into().map_err(|_| ColicError::Usage(
        "tensor is too large for the current record-lowering address space".into()
    ))?];
    source::read_range(tensor, 0..tensor.len, &mut bytes)?;
    Ok(bytes)
}

pub fn math_format_for_dtype(dtype: &str) -> Result<u16> {
    if parse_qwen3_next_split(dtype).is_some() {
        return Ok(3);
    }
    if parse_ple_quant_dtype(dtype).is_some() {
        return Ok(0x10);
    }
    if parse_const_bf16_dtype(dtype).is_some() {
        return Ok(3);
    }
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" => Ok(5),
        "F8_E8M0" | "F8_E8M0FNU" => Ok(5),
        "I64" => Ok(0x0a),
        "I8" => Ok(0x20),
        "F8_E4M3" | "F8_E4M3FN" => Ok(0x10),
        "F8_E5M2" => Ok(0x11),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported matrix dtype `{dtype}`"),
        )),
    }
}

fn expert_math_format(dtype: &str) -> Result<u16> {
    match dtype {
        "I8" => Ok(0x20),
        other => math_format_for_dtype(other),
    }
}

fn scale_format(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" | "F8_E8M0" | "F8_E8M0FNU" => Ok(4),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported scale dtype `{dtype}`"),
        )),
    }
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod qwen4_ple_quant_tests {
    use super::*;

    #[test]
    fn qwen4_pseudo_dtype_parsing_round_trips_scale() {
        let dtype = format!("{}3d80", crate::model::qwen4_exp::PLE_BF16_TO_F8_PREFIX);
        assert_eq!(parse_ple_quant_dtype(&dtype), Some(0x3d80));
        let dtype = format!("{}3d80", crate::model::qwen4_exp::PLE_CONST_BF16_PREFIX);
        assert_eq!(parse_const_bf16_dtype(&dtype), Some(0x3d80));
    }

    #[test]
    fn e4m3_encoder_refuses_saturation() {
        assert!(encode_bf16_e4m3(0x4180, 0.0625) > 0xff); // BF16 16.0 / 1/16 = 256
    }
}


#[cfg(test)]
mod qwen3_next_split_tests {
    use super::*;
    use std::{fs, time::{SystemTime, UNIX_EPOCH}};

    fn tmp_file(values: &[u16]) -> std::path::PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("logan-qwen3-next-split-{nonce}.bin"));
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, bytes).unwrap();
        path
    }

    fn payload_u16(tensor: &source::TensorRef) -> Vec<u16> {
        let payload = lower_exact_tensor(tensor).unwrap();
        payload[TENSOR_HEADER_BYTES..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn qkvz_virtual_views_restore_runtime_qkv_and_z_order() {
        // Fused HF row order by key-head group is:
        //   g0 [q0,k0,v0,v1,z0,z1], g1 [q1,k1,v2,v3,z2,z3].
        // Logan's canonical runtime wants all Q, then all K, then all V.
        let path = tmp_file(&(1_u16..=12).collect::<Vec<_>>());
        let qkv = source::TensorRef {
            source: path.clone(),
            offset: 0,
            len: 8 * 2,
            dtype: format!("{}qkv:2:1:4:1:1", crate::model::qwen3_next::QKVZ_SPLIT_PREFIX),
            shape: vec![8, 1],
        };
        let z = source::TensorRef {
            source: path.clone(),
            offset: 0,
            len: 4 * 2,
            dtype: format!("{}z:2:1:4:1:1", crate::model::qwen3_next::QKVZ_SPLIT_PREFIX),
            shape: vec![4, 1],
        };
        assert_eq!(payload_u16(&qkv), vec![1, 7, 2, 8, 3, 4, 9, 10]);
        assert_eq!(payload_u16(&z), vec![5, 6, 11, 12]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn ba_virtual_views_restore_runtime_b_and_a_order() {
        // rep=2: g0 [b0,b1,a0,a1], g1 [b2,b3,a2,a3].
        let path = tmp_file(&(101_u16..=108).collect::<Vec<_>>());
        let b = source::TensorRef {
            source: path.clone(),
            offset: 0,
            len: 4 * 2,
            dtype: format!("{}b:2:4:1", crate::model::qwen3_next::BA_SPLIT_PREFIX),
            shape: vec![4, 1],
        };
        let a = source::TensorRef {
            source: path.clone(),
            offset: 0,
            len: 4 * 2,
            dtype: format!("{}a:2:4:1", crate::model::qwen3_next::BA_SPLIT_PREFIX),
            shape: vec![4, 1],
        };
        assert_eq!(payload_u16(&b), vec![101, 102, 105, 106]);
        assert_eq!(payload_u16(&a), vec![103, 104, 107, 108]);
        fs::remove_file(path).unwrap();
    }
}
