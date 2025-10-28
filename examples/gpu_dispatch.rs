use std::error::Error;
use std::mem::size_of;

use bytemuck::cast_slice;
use futures_channel::oneshot;
use pollster::block_on;
use sobol_burley::{gpu, NUM_DIMENSIONS};

fn main() -> Result<(), Box<dyn Error>> {
    block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        Ok(adapter) => adapter,
        Err(err) => return Err(format!("No compatible GPU adapter found: {err}").into()),
    };

    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sobol-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
    {
        Ok(pair) => pair,
        Err(err) => return Err(format!("Failed to request device: {err}").into()),
    };

    let sobol = gpu::SobolGpu::new(&device);

    let sample_count: u32 = 1024;
    let dim_count: u32 = 8;
    let output_elems = (sample_count * dim_count) as usize;
    let params = gpu::Params {
        sample_base: 0,
        dim_base: 0,
        sample_count,
        dim_count,
        seed: 0,
        num_dims: NUM_DIMENSIONS,
        flags: gpu::FLAG_OUTPUT_DIMS_MAJOR,
        ..Default::default()
    };

    let output = gpu::create_output_buffer(&device, &params);
    let params_buffer = gpu::create_params_buffer(&device, &params);

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sobol-readback"),
        size: (output_elems * size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = sobol.create_bind_group(&device, &output, &params_buffer);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sobol-encoder"),
    });

    sobol.encode(&mut encoder, &bind_group, &params);

    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, readback.size());
    queue.submit(Some(encoder.finish()));

    let buffer_slice = readback.slice(..);
    let (sender, receiver) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    match receiver.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(format!("failed to map readback buffer: {err:?}").into()),
        Err(_) => return Err("map_async receiver dropped".into()),
    }

    let data = buffer_slice.get_mapped_range();
    let floats: &[f32] = cast_slice(&data);

    let layout_is_dims_major = (params.flags & gpu::FLAG_OUTPUT_DIMS_MAJOR) != 0;
    let mut first_sample = Vec::with_capacity(dim_count as usize);
    for dim in 0..dim_count as usize {
        let idx = if layout_is_dims_major {
            dim * sample_count as usize
        } else {
            dim + 0 * dim_count as usize
        };
        first_sample.push(floats[idx]);
    }

    println!("first sample across {} dims: {:?}", dim_count, first_sample);

    drop(data);
    readback.unmap();

    Ok(())
}
