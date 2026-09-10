use logan_ane::{AneRequest,AneRuntime,AneSurface,CompileOptions};
fn main()->Result<(),Box<dyn std::error::Error>>{
 let c=16usize; let s=16usize;
 let text=r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
 func main<ios18>(tensor<fp32, [1, $C, 1, $S]> x) {
  string t16 = const()[name = string("t16"), val = string("fp16")];
  tensor<fp16, [1, $C, 1, $S]> x16 = cast(dtype=t16,x=x)[name=string("cast0")];
  string t32 = const()[name = string("t32"), val = string("fp32")];
  tensor<fp32, [1, $C, 1, $S]> y = cast(dtype=t32,x=x16)[name=string("cast1")];
 } -> (y);
 func relu_proc<ios18>(tensor<fp32, [1, $C, 1, $S]> x) {
  string t16b = const()[name = string("t16b"), val = string("fp16")];
  tensor<fp16, [1, $C, 1, $S]> x16b = cast(dtype=t16b,x=x)[name=string("cast2")];
  tensor<fp16, [1, $C, 1, $S]> r16 = relu(x=x16b)[name=string("relu")];
  string t32b = const()[name = string("t32b"), val = string("fp32")];
  tensor<fp32, [1, $C, 1, $S]> yb = cast(dtype=t32b,x=r16)[name=string("cast3")];
 } -> (yb);
}
"#.replace("$C",&c.to_string()).replace("$S",&s.to_string());
 let p=logan_ane::mil::MilProgram::new(text); let rt=AneRuntime::load()?; let mut model=rt.compile(&p,CompileOptions::default())?; model.load()?;
 let vals:Vec<f32>=(0..c*s).map(|i|((i as i32%11)-5) as f32/4.0).collect(); let mut x=AneSurface::new(vals.len()*4)?; x.write_f32(&vals)?;
 for proc in 0..3u64 { let y=AneSurface::new(vals.len()*4)?; let req=AneRequest::new(&[&x],&[&y],proc); match req { Ok(req)=>match model.evaluate(&req){Ok(())=>{let out=y.read_f32()?; let id=out.iter().zip(&vals).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max); let relu=out.iter().zip(&vals).map(|(a,b)|(a-b.max(0.0)).abs()).fold(0f32,f32::max); println!("proc={proc} ok identity_err={id:.6} relu_err={relu:.6}");},Err(e)=>println!("proc={proc} eval_err={e}")}, Err(e)=>println!("proc={proc} req_err={e}") } }
 Ok(())
}
