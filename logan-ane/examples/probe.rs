use logan_ane::AneRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    println!("device: {:#?}", runtime.device_info());
    println!("capabilities: {:#?}", runtime.capabilities());
    Ok(())
}
