use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: logan-spark <MLX_DIR|COLI_DIR> [token ids ...]");
        std::process::exit(2)
    }
    let prompt: Vec<u32> = if a.len() > 2 {
        a[2..]
            .iter()
            .map(|x| x.parse().expect("token id"))
            .collect()
    } else {
        vec![1, 2, 3, 4, 5]
    };
    match logan_spark::run_greedy(Path::new(&a[1]), &prompt, 4) {
        Ok(x) => println!("generated: {x:?}"),
        Err(e) => {
            eprintln!("spark error: {e}");
            std::process::exit(1)
        }
    }
}
