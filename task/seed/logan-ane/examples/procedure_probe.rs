use logan_ane::{AneRuntime,CompileOptions,DenseProjection};
fn main()->Result<(),Box<dyn std::error::Error>>{
 let n=16usize; let mut w=vec![0u16;n*n]; for i in 0..n { w[i*n+i]=0x3c00; }
 let p=logan_ane::mil::parallel_dense_fp16_f32_io(n,16,&[DenseProjection::new("w",n,w)])?;
 let rt=AneRuntime::load()?; let mut model=rt.compile(&p,CompileOptions::default())?; model.load()?;
 for id in 0..16u64 {
   match model.probe_mutable_weight_buffer(id) {
     Ok(size)=>println!("buffer_id={id} size={size}"),
     Err(e)=>println!("buffer_id={id} unavailable: {e}"),
   }
 }
 Ok(())
}
