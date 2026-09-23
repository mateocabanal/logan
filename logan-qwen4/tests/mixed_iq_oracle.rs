use logan_qwen4::ggufsource::{decode_row, dot_row, GgmlType};

fn raw_block(bytes: usize, seed: u8, q5k: bool) -> Vec<u8> {
    let mut raw = (0..bytes)
        .map(|i| ((i * 37 + usize::from(seed) * 13 + 11) & 0xff) as u8)
        .collect::<Vec<_>>();
    raw[0..2].copy_from_slice(&0x3400u16.to_le_bytes()); // f16 0.25
    if q5k {
        raw[2..4].copy_from_slice(&0x3000u16.to_le_bytes()); // f16 0.125
    }
    raw
}

fn fnv_f32_bits(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[test]
fn mixed_iq_dequant_matches_upstream_ggml_reference() {
    // Golden hashes generated with current upstream llama.cpp dequantize_row_*
    // from commit 5836771 (ggml 0.24.0), using the exact deterministic bytes
    // constructed above. Hashing every output f32 bit keeps this regression
    // fixture compact while covering the complete dequantized row.
    let cases = [
        (
            GgmlType::Q2_0,
            18usize,
            64usize,
            1u8,
            false,
            0x649b15adb443169b,
        ),
        (GgmlType::Q5K, 176, 256, 2, true, 0xc80bea08e01fc995),
        (GgmlType::Iq2Xxs, 66, 256, 3, false, 0x3510966a7ba0b62d),
        (GgmlType::Iq2Xs, 74, 256, 4, false, 0xda159c82f6cc1296),
        (GgmlType::Iq2S, 82, 256, 5, false, 0x8ab29bdeacd52122),
        (GgmlType::Iq3Xxs, 98, 256, 6, false, 0x778838afb440a130),
        (GgmlType::Iq3S, 110, 256, 7, false, 0xf0fc042c1ca99657),
        (GgmlType::Iq4Nl, 18, 32, 8, false, 0x6527de72db14a4ce),
        (GgmlType::Iq4Xs, 136, 256, 9, false, 0xd9d825132d7588d5),
    ];

    for (dtype, bytes, elements, seed, q5k, expected) in cases {
        assert_eq!(dtype.block_geometry(), (elements as u64, bytes as u64));
        let raw = raw_block(bytes, seed, q5k);
        let decoded = decode_row(dtype, &raw, elements)
            .unwrap_or_else(|err| panic!("{} decode failed: {err}", dtype.name()));
        assert_eq!(
            fnv_f32_bits(&decoded),
            expected,
            "{} diverged from upstream GGML",
            dtype.name()
        );
    }
}

#[test]
fn mixed_iq_dot_row_matches_dequantized_reference() {
    let cases = [
        (GgmlType::Q2_0, 18usize, 64usize, 11u8, false),
        (GgmlType::Q5K, 176, 256, 12, true),
        (GgmlType::Iq2Xxs, 66, 256, 13, false),
        (GgmlType::Iq2Xs, 74, 256, 14, false),
        (GgmlType::Iq2S, 82, 256, 15, false),
        (GgmlType::Iq3Xxs, 98, 256, 16, false),
        (GgmlType::Iq3S, 110, 256, 17, false),
        (GgmlType::Iq4Nl, 18, 32, 18, false),
        (GgmlType::Iq4Xs, 136, 256, 19, false),
    ];

    for (dtype, bytes, elements, seed, q5k) in cases {
        let raw = raw_block(bytes, seed, q5k);
        let decoded = decode_row(dtype, &raw, elements).unwrap();
        let x = (0..elements)
            .map(|i| (((i * 17 + 3) % 31) as f32 - 15.0) * (1.0 / 32.0))
            .collect::<Vec<_>>();
        let mut expected = 0.0f32;
        for (&w, &v) in decoded.iter().zip(&x) {
            expected += w * v;
        }
        let actual = dot_row(dtype, &raw, &x).unwrap();
        let tol = 2e-5f32 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "{} dot mismatch: actual={actual} expected={expected} tol={tol}",
            dtype.name()
        );
    }
}
