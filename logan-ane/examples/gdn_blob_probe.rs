use logan_ane::{
    AneRuntime, BlobV2Builder, CompileOptions,
    mil::{MilProgram, WeightBlob},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const HDR: &str = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n";
    let mut blob = BlobV2Builder::new();
    let off = blob.push_fp16(&vec![0x3800u16; 256])?.get();
    let mil = format!(
        r#"{HDR} func main<ios18>(tensor<fp32, [1,256,1,16]> x) {{
  string d=const()[name=string("d"),val=string("fp16")]; tensor<fp16,[1,256,1,16]> x16=cast(dtype=d,x=x)[name=string("cx")];
  tensor<int32,[4]> b=const()[name=string("b"),val=tensor<int32,[4]>([0,0,0,0])]; tensor<int32,[4]> sz=const()[name=string("sz"),val=tensor<int32,[4]>([1,256,1,1])];
  tensor<fp16,[1,256,1,1]> x1=slice_by_size(x=x16,begin=b,size=sz)[name=string("sl")];
  tensor<fp16,[1,256,1,1]> W=const()[name=string("W"),val=tensor<fp16,[1,256,1,1]>(BLOBFILE(path=string("@model_path/weights/tap.bin"),offset=uint64({off})))];
  tensor<fp16,[1,256,1,1]> m=mul(x=x1,y=W)[name=string("m")];
  int32 ax=const()[name=string("ax"),val=int32(3)]; bool il=const()[name=string("il"),val=bool(false)];
  tensor<fp16,[1,256,1,16]> rep=concat(axis=ax,interleave=il,values=(m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m))[name=string("rep")];
  string d32=const()[name=string("d32"),val=string("fp32")]; tensor<fp32,[1,256,1,16]> y=cast(dtype=d32,x=rep)[name=string("co")];
 }} -> (y);
}}
"#
    );
    let p = MilProgram::new(mil).with_weight(WeightBlob::new(
        "@model_path/weights/tap.bin",
        blob.into_bytes(),
    ));
    let rt = AneRuntime::load()?;
    let mut model = rt.compile(&p, CompileOptions::default())?;
    model.load()?;
    println!("PASS blob_vector_mul");
    Ok(())
}
