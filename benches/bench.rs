use bencher::{benchmark_group, benchmark_main, black_box, Bencher};
use rand::prelude::*;
use sobol_burley::{sample, sample_8d};

#[cfg(feature = "wgpu")]
use pollster::block_on;
#[cfg(feature = "wgpu")]
use sobol_burley::gpu;

//----

fn gen_1000_samples_8d(bench: &mut Bencher) {
    bench.iter(|| {
        for d in 0..120 {
            for i in 0..250 {
                black_box(sample_8d(i, d, 1234567890));
            }
        }
    });
}

fn gen_1000_samples_incoherent_8d(bench: &mut Bencher) {
    let mut rng = rand::rng();
    bench.iter(|| {
        let s = rng.random::<u32>();
        let d = rng.random::<u32>();
        let seed = rng.random::<u32>();
        for i in 0..250u32 {
            black_box(sample_8d(
                s.wrapping_add(i).wrapping_mul(512),
                d.wrapping_add(i).wrapping_mul(97) % 32,
                seed,
            ));
        }
    });
}

fn gen_1000_samples(bench: &mut Bencher) {
    bench.iter(|| {
        for d in 0..120 {
            for i in 0..1000 {
                black_box(sample(i, d, 1234567890));
            }
        }
    });
}

fn gen_1000_samples_incoherent(bench: &mut Bencher) {
    let mut rng = rand::rng();
    bench.iter(|| {
        let s = rng.random::<u32>();
        let d = rng.random::<u32>();
        let seed = rng.random::<u32>();
        for i in 0..1000u32 {
            black_box(sample(
                s.wrapping_add(i).wrapping_mul(512),
                d.wrapping_add(i).wrapping_mul(97) % 128,
                seed,
            ));
        }
    });
}

#[cfg(feature = "wgpu")]
fn gen_1000_samples_gpu(bench: &mut Bencher) {
    let maybe_device = block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sobol-bench-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()
    });

    let Some((device, queue)) = maybe_device else {
        eprintln!("skipping gen_1000_samples_gpu: no compatible GPU found");
        bench.iter(|| {});
        return;
    };

    let sobol = gpu::SobolGpu::new(&device);
    let params = gpu::Params {
        sample_base: 0,
        dim_base: 0,
        sample_count: 1000,
        dim_count: 120,
        seed: 1234567890,
        num_dims: sobol_burley::NUM_DIMENSIONS,
        flags: gpu::FLAG_OUTPUT_DIMS_MAJOR,
        ..Default::default()
    };
    let output = gpu::create_output_buffer(&device, &params);
    let params_buffer = gpu::create_params_buffer(&device, &params);
    let bind_group = sobol.create_bind_group(&device, &output, &params_buffer);

    bench.iter(|| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sobol-bench-encoder"),
        });
        sobol.encode(&mut encoder, &bind_group, &params);
        queue.submit(Some(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
    });
}

//----

#[cfg(feature = "wgpu")]
benchmark_group!(
    benches,
    gen_1000_samples,
    gen_1000_samples_incoherent,
    gen_1000_samples_8d,
    gen_1000_samples_incoherent_8d,
    gen_1000_samples_gpu,
);

#[cfg(not(feature = "wgpu"))]
benchmark_group!(
    benches,
    gen_1000_samples,
    gen_1000_samples_incoherent,
    gen_1000_samples_8d,
    gen_1000_samples_incoherent_8d,
);
benchmark_main!(benches);
