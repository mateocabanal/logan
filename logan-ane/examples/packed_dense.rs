use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, mil};
use std::time::Instant;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - exp;
        let mut half = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half = half.wrapping_add(1);
        }
        sign | half
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut half = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        if (mant & 0x1000) != 0 {
            half = half.wrapping_add(1);
        }
        half
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let f = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if f == 0 {
            s
        } else {
            let mut frac = f;
            let mut exp = 127 - 14;
            while (frac & 0x400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x3ff;
            s | ((exp as u32) << 23) | (frac << 13)
        }
    } else if e == 31 {
        s | 0x7f800000 | (f << 13)
    } else {
        s | ((e + (127 - 15)) << 23) | (f << 13)
    };
    f32::from_bits(bits)
}

fn write_u16s(surface: &mut AneSurface, values: &[u16]) -> Result<(), Box<dyn std::error::Error>> {
    let mut map = surface.write()?;
    if map.len() != values.len() * 2 {
        return Err(format!("surface bytes {} != {}", map.len(), values.len() * 2).into());
    }
    for (dst, &v) in map.chunks_exact_mut(2).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

fn read_first_spatial(
    surface: &AneSurface,
    out: usize,
    spatial: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let map = surface.read()?;
    let mut result = Vec::with_capacity(out);
    for o in 0..out {
        let i = (o * spatial) * 2;
        result.push(f16_bits_to_f32(u16::from_le_bytes([map[i], map[i + 1]])));
    }
    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const I: usize = 16;
    const S: usize = 16;
    const O0: usize = 16;
    const O1: usize = 8;
    let outputs = [O0, O1];
    let packed_s = S + outputs.iter().sum::<usize>();

    let program = mil::parallel_dense_packed_fp16_io(I, S, &outputs)?;
    let runtime = AneRuntime::load()?;
    let t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    model.load()?;
    println!("compile+load: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);

    // Per-input-channel packed rows: [activation x16 | identity W0 row | double-first-8 W1 row].
    let mut packed = vec![0u16; I * packed_s];
    for i in 0..I {
        let x = (i as f32 - 8.0) / 8.0;
        for s in 0..S {
            packed[i * packed_s + s] = f32_to_f16_bits(x);
        }
        for o in 0..O0 {
            packed[i * packed_s + S + o] = if i == o { 0x3c00 } else { 0 };
        }
        let off1 = S + O0;
        for o in 0..O1 {
            packed[i * packed_s + off1 + o] = if i == o { 0x4000 } else { 0 };
        }
    }

    let mut input = AneSurface::new(packed.len() * 2)?;
    write_u16s(&mut input, &packed)?;
    let output0 = AneSurface::new(O0 * S * 2)?;
    let output1 = AneSurface::new(O1 * S * 2)?;
    let request = AneRequest::new(&[&input], &[&output0, &output1], 0)?;

    for _ in 0..5 {
        model.evaluate(&request)?;
    }
    let t1 = Instant::now();
    for _ in 0..100 {
        model.evaluate(&request)?;
    }
    println!(
        "evaluate: {:.3} us",
        t1.elapsed().as_secs_f64() * 1e6 / 100.0
    );

    let y0 = read_first_spatial(&output0, O0, S)?;
    let y1 = read_first_spatial(&output1, O1, S)?;
    let expected0: Vec<f32> = (0..O0).map(|i| (i as f32 - 8.0) / 8.0).collect();
    let expected1: Vec<f32> = (0..O1).map(|i| 2.0 * (i as f32 - 8.0) / 8.0).collect();
    let e0 = y0
        .iter()
        .zip(&expected0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let e1 = y1
        .iter()
        .zip(&expected1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("identity max abs error: {e0:.8}");
    println!("double max abs error:   {e1:.8}");
    if e0 > 0.001 || e1 > 0.001 {
        return Err("packed dynamic matmul mismatch".into());
    }
    Ok(())
}
