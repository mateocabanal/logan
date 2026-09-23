use logan_qwen4::ggufsource::{decode_row, GgmlType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 5, "type raw elements output");
    let ty = match args[1].as_str() {
        "q2_0" => GgmlType::Q2_0,
        "q5_K" => GgmlType::Q5K,
        "iq2_xxs" => GgmlType::Iq2Xxs,
        "iq2_xs" => GgmlType::Iq2Xs,
        "iq2_s" => GgmlType::Iq2S,
        "iq3_xxs" => GgmlType::Iq3Xxs,
        "iq3_s" => GgmlType::Iq3S,
        "iq4_nl" => GgmlType::Iq4Nl,
        "iq4_xs" => GgmlType::Iq4Xs,
        other => panic!("unknown type {other}"),
    };
    let raw = std::fs::read(&args[2]).unwrap();
    let n: usize = args[3].parse().unwrap();
    let decoded = decode_row(ty, &raw, n).unwrap();
    let mut bytes = Vec::with_capacity(decoded.len() * 4);
    for v in decoded {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&args[4], bytes).unwrap();
}
