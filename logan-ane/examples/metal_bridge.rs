use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, mil};
use logan_core::sched::DeviceKind;
use logan_core::shared::{SharedAllocationId, SharedAllocationRegistry};

fn metal_add_in_place(
    shared: &logan_metal::MetalSharedSurface,
    addend: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if shared.len() != addend.len() * std::mem::size_of::<f32>() {
        return Err("Metal shared surface/addend length mismatch".into());
    }
    let ptr = unsafe { shared.contents_ptr() };
    if ptr.is_null() {
        return Err("MTLBuffer.contents returned null".into());
    }
    // SAFETY: `shared` retains the IOSurface/MTLBuffer for this call, the ANE
    // is not executing concurrently, and metal_add is synchronous before this
    // temporary mutable slice is released.
    let values = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f32>(), addend.len()) };
    if !logan_metal::metal_add(values, addend) {
        return Err("Metal add declined on IOSurface-backed MTLBuffer memory".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    if !runtime.device_info().has_ane {
        return Err("ANE unavailable".into());
    }
    if !logan_metal::metal_init() {
        return Err("Metal unavailable".into());
    }

    let channels = 256usize;
    let spatial = 64usize;
    let elements = channels * spatial;
    let bytes = elements * std::mem::size_of::<f32>();

    let mut model = runtime.compile(
        &mil::relu_fp32(channels, spatial)?,
        CompileOptions::default(),
    )?;
    model.load()?;

    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    let original: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 41) - 20) as f32 / 8.0)
        .collect();
    input.write_f32(&original)?;

    let input_metal =
        unsafe { logan_metal::MetalSharedSurface::from_iosurface(input.as_raw_iosurface(), bytes) }
            .ok_or("Metal could not zero-copy import the ANE input IOSurface")?;
    let output_metal = unsafe {
        logan_metal::MetalSharedSurface::from_iosurface(output.as_raw_iosurface(), bytes)
    }
    .ok_or("Metal could not zero-copy import the ANE output IOSurface")?;

    // Register the same allocations in Logan's backend-neutral metadata model.
    let mut registry = SharedAllocationRegistry::new();
    let input_id = SharedAllocationId(input.iosurface_id() as u64);
    let output_id = SharedAllocationId(output.iosurface_id() as u64);
    registry.register(input.shared_allocation_desc(input_id, Some("bridge.input".into())))?;
    registry.register(output.shared_allocation_desc(output_id, Some("bridge.output".into())))?;
    assert!(
        registry
            .get(input_id)
            .unwrap()
            .can_alias_between(DeviceKind::Gpu, DeviceKind::Neural)
    );

    // Metal -> ANE: mutate the input in-place on the GPU, then have the ANE
    // consume that exact IOSurface as its request input.
    let plus_half = vec![0.5f32; elements];
    metal_add_in_place(&input_metal, &plus_half)?;

    let request = AneRequest::new(&[&input], &[&output], 0)?;
    model.evaluate(&request)?;

    let after_ane = output.read_f32()?;
    let expected_ane = original.iter().map(|&x| (x + 0.5).max(0.0));
    let ane_error = after_ane
        .iter()
        .copied()
        .zip(expected_ane)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // ANE -> Metal: mutate the ANE output through its persistent MTLBuffer
    // view and verify the CPU sees the same physical bytes afterward.
    let plus_one = vec![1.0f32; elements];
    metal_add_in_place(&output_metal, &plus_one)?;
    let final_values = output.read_f32()?;
    let expected_final = original.iter().map(|&x| (x + 0.5).max(0.0) + 1.0);
    let final_error = final_values
        .iter()
        .copied()
        .zip(expected_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("device: {:#?}", runtime.device_info());
    println!("input IOSurface id: {}", input.iosurface_id());
    println!("output IOSurface id: {}", output.iosurface_id());
    println!(
        "Metal import bytes: logical={} allocated={}",
        input_metal.len(),
        input_metal.allocation_len()
    );
    println!("Metal -> ANE max abs error: {ane_error:.8}");
    println!("ANE -> Metal -> CPU max abs error: {final_error:.8}");

    if ane_error > 0.001 || final_error > 0.001 {
        return Err("Metal/ANE shared-IOSurface validation failed".into());
    }
    Ok(())
}
