//! GPU compute support using `wgpu` and the WGSL shader in `shaders/sobol.wgsl`.
//!
//! This module mirrors the CPU-side sampler but executes batches of samples on
//! the GPU. It is compiled when the crate is built with the `wgpu` feature.

use core::{mem, slice};

use wgpu::{self, util::DeviceExt};

use crate::{NUM_DIMENSIONS, REV_VECTORS_BIN, SOBOL_DEPTH};

/// Number of threads per workgroup used by the WGSL compute shader.
pub const THREADS_PER_GROUP: u32 = 256;

/// Write samples with dimensions as the major axis when set on [`Params::flags`].
pub const FLAG_OUTPUT_DIMS_MAJOR: u32 = 1 << 0;

/// Uniform parameter layout shared with the WGSL shader.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Params {
    pub sample_base: u32,
    pub dim_base: u32,
    pub sample_count: u32,
    pub dim_count: u32,
    pub seed: u32,
    pub num_dims: u32,
    pub flags: u32,
    pub _pad0: u32,
}

impl Params {
    /// Borrow the raw bytes of the params for uniform buffer uploads.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self as *const Self as *const u8, mem::size_of::<Self>()) }
    }
}

/// Helper object that owns the GPU pipeline and immutable direction vectors.
pub struct SobolGpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    vectors: wgpu::Buffer,
}

impl SobolGpu {
    /// Create the shader module, pipeline layout, and upload the static direction vectors.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/sobol.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sobol-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sobol-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sobol-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sobol_kernel"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let vectors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sobol-direction-vectors"),
            contents: REV_VECTORS_BIN,
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            bind_group_layout,
            pipeline,
            vectors,
        }
    }

    /// Access the bind group layout for additional customization.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Access the compute pipeline for advanced usage.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Create a bind group that targets the provided output buffer and params buffer.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        output: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobol-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode a compute pass that dispatches the Sobol kernel for the supplied parameters.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &Params,
    ) {
        if params.sample_count == 0 || params.dim_count == 0 {
            return;
        }

        assert!(
            params.num_dims <= NUM_DIMENSIONS,
            "Requested num_dims={} exceeds available {}",
            params.num_dims,
            NUM_DIMENSIONS
        );
        let max_dim = params
            .dim_base
            .checked_add(params.dim_count)
            .expect("dimension range overflow");
        assert!(
            max_dim <= params.num_dims && max_dim <= NUM_DIMENSIONS,
            "Requested dimensions [{}..{}) exceed available {}",
            params.dim_base,
            max_dim,
            params.num_dims
        );

        // Match WGSL constants
        const SAMPLES_PER_THREAD: u32 = 8;
        let groups_x = (params.sample_count + (THREADS_PER_GROUP * SAMPLES_PER_THREAD) - 1)
            / (THREADS_PER_GROUP * SAMPLES_PER_THREAD);
        let groups_y = params.dim_count;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sobol-compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    /// Returns the number of bytes required for an output buffer covering the range described by `params`.
    ///
    /// The layout flag does not affect the total size, only how elements are indexed.
    pub fn output_buffer_size(params: &Params) -> u64 {
        (params.sample_count as u64) * (params.dim_count as u64) * mem::size_of::<f32>() as u64
    }

    /// Number of direction numbers per dimension.
    pub const fn sobol_depth() -> usize {
        SOBOL_DEPTH
    }
}

/// Convenience helper to construct an output buffer suitable for the parameters provided.
pub fn create_output_buffer(device: &wgpu::Device, params: &Params) -> wgpu::Buffer {
    let size = SobolGpu::output_buffer_size(params);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sobol-output"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Construct a uniform buffer containing the provided parameters.
pub fn create_params_buffer(device: &wgpu::Device, params: &Params) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sobol-params"),
        contents: params.as_bytes(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use crate::sample;
    use bytemuck::cast_slice;
    use futures_channel::oneshot;

    #[test]
    fn gpu_matches_cpu_samples() {
        pollster::block_on(async {
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
                Err(err) => {
                    eprintln!("skipping gpu_matches_cpu_samples: adapter request failed ({err})");
                    return;
                }
            };

            let (device, queue) = match adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("sobol-test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                })
                .await
            {
                Ok(pair) => pair,
                Err(err) => {
                    panic!("failed to acquire test device: {err}");
                }
            };

            let sobol = SobolGpu::new(&device);
            let params = Params {
                sample_base: 0,
                dim_base: 0,
                sample_count: 1024,
                dim_count: 64,
                seed: 123,
                num_dims: NUM_DIMENSIONS,
                ..Default::default()
            };

            let output = create_output_buffer(&device, &params);
            let params_buffer = create_params_buffer(&device, &params);
            let readback_size = SobolGpu::output_buffer_size(&params);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sobol-test-readback"),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let bind_group = sobol.create_bind_group(&device, &output, &params_buffer);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sobol-test-encoder"),
            });

            sobol.encode(&mut encoder, &bind_group, &params);
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, readback_size);

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
                Ok(Err(err)) => panic!("failed to map readback buffer: {err:?}"),
                Err(_) => panic!("map_async receiver dropped"),
            }

            let data = buffer_slice.get_mapped_range();
            let gpu_values: &[f32] = cast_slice(&data);

            for sample_idx in 0..params.sample_count as usize {
                for dim_idx in 0..params.dim_count as usize {
                    let expected = sample(
                        params.sample_base + sample_idx as u32,
                        params.dim_base + dim_idx as u32,
                        params.seed,
                    );
                    let actual = gpu_values[sample_idx * params.dim_count as usize + dim_idx];
                    assert_eq!(
                        expected, actual,
                        "mismatch at sample {}, dim {}",
                        sample_idx, dim_idx
                    );
                }
            }

            drop(data);
            readback.unmap();
        });
    }
}
