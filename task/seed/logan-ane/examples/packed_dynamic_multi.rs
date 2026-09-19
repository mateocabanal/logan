use logan_ane::{AneRequest,AneRuntime,AneSurface,CompileOptions};
fn main()->Result<(),Box<dyn std::error::Error>>{
 let i=64usize; let s=64usize; let outs=[64usize,32usize,8usize,8usize];
 let (program,layout)=logan_ane::mil::parallel_dense_packed_dynamic_f32_io(i,s,&outs)?;
 let rt=AneRuntime::load()?; let opts=CompileOptions{reuse_compiled_model:false,keep_temporary_files:true,..CompileOptions::default()}; let mut model=rt.compile(&program,opts)?; model.load()?;
 let mut packed=vec![0.0f32;i*layout.total_spatial];
 for d in 0..i {
  let base=d*layout.total_spatial;
  for j in 0..s {packed[base+j]=((d*s+j)%97) as f32*0.001;}
  for (k,&o) in outs.iter().enumerate(){let off=layout.weight_offsets[k];for c in 0..o{packed[base+off+c]=if c==d%o{(k+1) as f32}else{0.0};}}
 }
 let mut input=AneSurface::new(packed.len()*4)?;input.write_f32(&packed)?;
 let ys:Vec<AneSurface>=outs.iter().map(|&o|AneSurface::new(o*s*4).unwrap()).collect();
 let refs:Vec<&AneSurface>=ys.iter().collect();let req=AneRequest::new(&[&input],&refs,0)?;
 for _ in 0..5{model.evaluate(&req)?;}
 for (k,y) in ys.iter().enumerate(){let got=y.read_f32()?;let mut max=0f32;for c in 0..outs[k]{for j in 0..s{let mut exp=0f32;for d in 0..i{let w=if c==d%outs[k]{(k+1) as f32}else{0.0};exp+=packed[d*layout.total_spatial+j]*w;}max=max.max((got[c*s+j]-exp).abs());}}println!("out{k} max_err={max:.8}");if max>0.03{return Err(format!("output {k} mismatch {max}").into());}}
 Ok(())
}
