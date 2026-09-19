//! Shape/ABI qualification probe for the MiniCPM5 dense FFN island.
//!
//! This probe never enables placement. A generated graph or successful compile
//! is not treated as native support: the current dynamic-weight ABI can still
//! decline evaluation, in which case the caller remains on CPU/Metal.

use logan_ane::{mil, AneRuntime, CompileOptions};

fn main() {
    let widths = [1usize, 7, 16, 64, 128, 256];
    let runtime = match AneRuntime::load() {
        Ok(runtime) => {
            println!("ANE runtime: available (qualification only; placement remains disabled)");
            Some(runtime)
        }
        Err(error) => {
            println!("ANE runtime: unsupported ({error}); using CPU/Metal fallback");
            None
        }
    };

    for logical_width in widths {
        let padded_width = match mil::FusedDenseFfnShape::padded_width_for(logical_width) {
            Ok(width) => width,
            Err(error) => {
                println!("width logical={logical_width}: invalid shape ({error}); fallback");
                continue;
            }
        };
        let shape = match mil::FusedDenseFfnShape::new(
            logical_width,
            padded_width,
            padded_width * 4,
            16,
        ) {
            Ok(shape) => shape,
            Err(error) => {
                println!("width logical={logical_width} padded={padded_width}: invalid shape ({error}); fallback");
                continue;
            }
        };
        let program = match mil::minicpm5_dense_ffn_fp16_f32_io(shape) {
            Ok(program) => program,
            Err(error) => {
                println!("width logical={logical_width} padded={padded_width}: MIL rejected ({error}); fallback");
                continue;
            }
        };

        let Some(runtime) = runtime.as_ref() else {
            println!("width logical={logical_width} padded={padded_width}: MIL generated; native unavailable; fallback");
            continue;
        };
        match runtime.compile(&program, CompileOptions::default()) {
            Ok(_) => println!(
                "width logical={logical_width} padded={padded_width}: MIL compiled; evaluation ABI unqualified; fallback retained"
            ),
            Err(error) => println!(
                "width logical={logical_width} padded={padded_width}: native compile declined ({error}); fallback"
            ),
        }
    }
}
