use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, MilProgram};
use std::time::Instant;

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 113u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const IC: usize = 256;
    const OC: usize = 256;
    const SEQ: usize = 16;
    const SP: usize = SEQ + OC;
    let mil = format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {IC}, 1, {SP}]> x) {{
        tensor<int32, [4]> mm_ba = const()[name=string("mm_ba"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> mm_sa = const()[name=string("mm_sa"), val=tensor<int32, [4]>([1,{IC},1,{SEQ}])];
        tensor<fp16, [1,{IC},1,{SEQ}]> mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)[name=string("mm_act")];
        tensor<int32, [4]> mm_bw = const()[name=string("mm_bw"), val=tensor<int32, [4]>([0,0,0,{SEQ}])];
        tensor<int32, [4]> mm_sw = const()[name=string("mm_sw"), val=tensor<int32, [4]>([1,{IC},1,{OC}])];
        tensor<fp16, [1,{IC},1,{OC}]> mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)[name=string("mm_wt")];
        tensor<int32, [4]> mm_ra = const()[name=string("mm_ra"), val=tensor<int32, [4]>([1,1,{IC},{SEQ}])];
        tensor<fp16, [1,1,{IC},{SEQ}]> mm_a2 = reshape(shape=mm_ra,x=mm_act)[name=string("mm_a2")];
        tensor<int32, [4]> mm_pm = const()[name=string("mm_pm"), val=tensor<int32, [4]>([0,1,3,2])];
        tensor<fp16, [1,1,{SEQ},{IC}]> mm_a3 = transpose(perm=mm_pm,x=mm_a2)[name=string("mm_a3")];
        tensor<int32, [4]> mm_rw = const()[name=string("mm_rw"), val=tensor<int32, [4]>([1,1,{IC},{OC}])];
        tensor<fp16, [1,1,{IC},{OC}]> mm_W = reshape(shape=mm_rw,x=mm_wt)[name=string("mm_W")];
        bool bF = const()[name=string("bF"), val=bool(false)];
        tensor<fp16, [1,1,{SEQ},{OC}]> mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)[name=string("mm_yh")];
        tensor<fp16, [1,1,{OC},{SEQ}]> mm_yt = transpose(perm=mm_pm,x=mm_yh)[name=string("mm_yt")];
        tensor<int32, [4]> mm_ro = const()[name=string("mm_ro"), val=tensor<int32, [4]>([1,{OC},1,{SEQ}])];
        tensor<fp16, [1,{OC},1,{SEQ}]> mm_y = reshape(shape=mm_ro,x=mm_yt)[name=string("mm_y")];
    }} -> (mm_y);
}}
"#
    );
    let runtime = AneRuntime::load()?;
    let t = Instant::now();
    let mut model = runtime.compile(&MilProgram::new(mil), CompileOptions::default())?;
    println!("compile {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    model.load()?;
    let mut buf = vec![0u16; IC * SP];
    for c in 0..IC {
        let base = c * SP;
        for s in 0..SEQ {
            buf[base + s] = 0x3c00;
        }
        buf[base + SEQ + c] = 0x3c00;
    }
    let mut input = AneSurface::new(buf.len() * 2)?;
    {
        let mut m = input.write()?;
        for (d, v) in m.chunks_exact_mut(2).zip(buf) {
            d.copy_from_slice(&v.to_le_bytes());
        }
    }
    let output = AneSurface::new(OC * SEQ * 2)?;
    let req = AneRequest::new(&[&input], &[&output], 0)?;
    for _ in 0..5 {
        model.evaluate(&req)?;
    }
    let t = Instant::now();
    for _ in 0..100 {
        model.evaluate(&req)?;
    }
    println!("eval {:.3} us", t.elapsed().as_secs_f64() * 1e6 / 100.0);
    let map = output.read()?;
    let mut max = 0f32;
    for ch in map.chunks_exact(2) {
        max = max.max((f16_to_f32(u16::from_le_bytes([ch[0], ch[1]])) - 1.0).abs());
    }
    println!("max error {max}");
    if max > 1e-5 {
        return Err("bad output".into());
    }
    Ok(())
}
