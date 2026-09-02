const PLE_F8_FROM_BF16_S16: &str = "BF16_TO_F8_E4M3_S16";
const PLE_SCALE_CONST_S16: &str = "CONST_BF16_PLE_SCALE_S16";
const PLE_E4M3_SCALE_S16: f32 = 0.0625; // 1/16, exactly representable in BF16.

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

fn crc32c_state(mut crc: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    crc
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
    if tensor.dtype == PLE_SCALE_CONST_S16 {
        let bytes = bf16_from_f32(PLE_E4M3_SCALE_S16).to_le_bytes();
        output.write_all(&bytes).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        *output_state = crc32c_state(*output_state, &bytes);
        if let Some(state) = logical_state {
            *state = crc32c_state(*state, &bytes);
        }
        return Ok(());
    }
    if tensor.dtype == PLE_F8_FROM_BF16_S16 {
        return copy_bf16_to_e4m3_s16(tensor, output, output_state, logical_state);
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

fn copy_bf16_to_e4m3_s16<W: Write>(
    tensor: &source::TensorRef,
    output: &mut W,
    output_state: &mut u32,
    logical_state: Option<&mut u32>,
) -> Result<()> {
    let input_bytes = tensor.len.checked_mul(2).ok_or_else(|| {
        ColicError::Usage("PLE BF16 input byte size overflows u64".into())
    })?;
    let mut input = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    input.seek(SeekFrom::Start(tensor.offset)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;

    let lut = ple_bf16_to_e4m3_s16_lut();
    let mut remaining = input_bytes;
    let mut src = vec![0_u8; 8 * 1024 * 1024];
    let mut dst = vec![0_u8; src.len() / 2];
    let mut logical_state = logical_state;
    while remaining != 0 {
        let mut count = remaining.min(src.len() as u64) as usize;
        if count & 1 != 0 { count -= 1; }
        if count == 0 { return Err(ColicError::Usage("odd BF16 PLE input length".into())); }
        input.read_exact(&mut src[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        let elements = count / 2;
        for (index, pair) in src[..count].chunks_exact(2).enumerate() {
            let bf = u16::from_le_bytes([pair[0], pair[1]]);
            let encoded = lut[bf as usize];
            if encoded > 0xff {
                let value = f32::from_bits((bf as u32) << 16);
                return Err(ColicError::unsupported(
                    "Qwen4 PLE BF16->E4M3 lowering",
                    format!(
                        "value {value} cannot be represented with fixed scale {PLE_E4M3_SCALE_S16}; refusing to saturate"
                    ),
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

fn ple_bf16_to_e4m3_s16_lut() -> &'static [u16] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Vec<u16>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..=u16::MAX)
            .map(|bits| encode_bf16_e4m3_s16(bits))
            .collect()
    })
}

fn encode_bf16_e4m3_s16(bits: u16) -> u16 {
    let value = f32::from_bits((bits as u32) << 16);
    if !value.is_finite() {
        return 0x100;
    }
    let scaled = value.abs() / PLE_E4M3_SCALE_S16;
    if scaled > 240.0 {
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

fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
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

fn read_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    if tensor.dtype == PLE_SCALE_CONST_S16 {
        return Ok(bf16_from_f32(PLE_E4M3_SCALE_S16).to_le_bytes().to_vec());
    }
    if tensor.dtype == PLE_F8_FROM_BF16_S16 {
        let mut output = std::io::Cursor::new(Vec::with_capacity(
            tensor.len.try_into().map_err(|_| ColicError::Usage("PLE output too large".into()))?,
        ));
        let mut state = !0_u32;
        copy_bf16_to_e4m3_s16(tensor, &mut output, &mut state, None)?;
        return Ok(output.into_inner());
    }
    let mut bytes = vec![0; tensor.len.try_into().map_err(|_| ColicError::Usage(
        "tensor is too large for the current record-lowering address space".into()
    ))?];
    source::read_range(tensor, 0..tensor.len, &mut bytes)?;
    Ok(bytes)
}

pub fn math_format_for_dtype(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        PLE_SCALE_CONST_S16 => Ok(3),
        "U8" => Ok(5),
        "F8_E8M0" | "F8_E8M0FNU" => Ok(5),
        "I64" => Ok(0x0a),
        "I8" => Ok(0x20),
        "F8_E4M3" | "F8_E4M3FN" | PLE_F8_FROM_BF16_S16 => Ok(0x10),
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
    fn fixed_ple_scale_is_exact_bf16() {
        let bits = bf16_from_f32(PLE_E4M3_SCALE_S16);
        assert_eq!(f32::from_bits((bits as u32) << 16), PLE_E4M3_SCALE_S16);
    }

    #[test]
    fn e4m3_lut_rejects_saturation_instead_of_clamping() {
        let huge = bf16_from_f32(16.0);
        assert!(encode_bf16_e4m3_s16(huge) > 0xff);
    }

    #[test]
    fn e4m3_lut_roundtrips_representable_scaled_values() {
        for code in [0x00_u8, 0x01, 0x08, 0x20, 0x38, 0x57, 0x77] {
            let source = e4m3_positive(code) * PLE_E4M3_SCALE_S16;
            let bf = bf16_from_f32(source);
            assert_eq!(encode_bf16_e4m3_s16(bf) as u8, code);
        }
    }
}
