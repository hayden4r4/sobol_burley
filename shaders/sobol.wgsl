// Sobol Owen-scrambled sequence compute shader in WGSL.
//
// Translates the CPU implementation from `sobol_burley::sample` into a GPU
// compute pipeline. The shader consumes reversed-bit direction vectors laid
// out as `[NUM_DIMENSIONS][SOBOL_DEPTH]` (flattened) and writes Sobol samples
// into an output buffer. By default writes are sample-major; setting
// `Params.flags & 1u` switches to dims-major (dimension contiguous) layout.

const SOBOL_DEPTH : u32 = 32u;
const SAMPLES_PER_THREAD : u32 = 8u;
const SCRAMBLE_TABLE : array<u32, 8> = array<u32, 8>(
    0x912f69bau,
    0x174f18abu,
    0x691e72cau,
    0xb40cc1b8u,
    0x912f69bau,
    0x174f18abu,
    0x691e72cau,
    0xb40cc1b8u,
);

struct Params {
    sample_base : u32,
    dim_base : u32,
    sample_count : u32,
    dim_count : u32,
    seed : u32,
    num_dims : u32,
    flags : u32,
    _pad0 : u32,
};

struct DirectionVectors {
    data : array<u32>,
};

struct OutputBuffer {
    data : array<f32>,
};

@group(0) @binding(0) var<storage, read> vectors : DirectionVectors;
@group(0) @binding(1) var<storage, read_write> output : OutputBuffer;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> staged_dir : array<u32, SOBOL_DEPTH>;

fn hash(n : u32) -> u32 {
    var value = n;
    value = value ^ 0xe6fe3bebu;
    value = value ^ (value >> 16u);
    value = value * 0x7feb352du;
    value = value ^ (value >> 15u);
    value = value * 0x846ca68bu;
    value = value ^ (value >> 16u);
    return value;
}

fn owen_scramble_rev(n_rev : u32, scramble : u32) -> u32 {
    var value = n_rev;
    value = value ^ (value * 0x3d20adeau);
    value = value + scramble;
    value = value * ((scramble >> 16u) | 1u);
    value = value ^ (value * 0x05526c56u);
    value = value ^ (value * 0x53a22864u);
    return value;
}

fn sobol_rev(sample_index_rev : u32, dimension : u32) -> u32 {
    let base = dimension * SOBOL_DEPTH;
    var sobol = 0u;
    var index = sample_index_rev;
    var i = 0u;

    loop {
        if (index == 0u) {
            break;
        }

        let j = countLeadingZeros(index);
        sobol = sobol ^ vectors.data[base + i + j];
        i = i + j + 1u;
        index = index << (j + 1u);
    }

    return sobol;
}

fn u32_to_f32_norm(n : u32) -> f32 {
    return bitcast<f32>((n >> 9u) | 0x3f800000u) - 1.0;
}

@compute @workgroup_size(256, 1, 1)
fn sobol_kernel(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    // Guard dimensions first (Y axis is dimensions)
    if (gid.y >= params.dim_count) {
        return;
    }

    let dimension = params.dim_base + gid.y;
    if (dimension >= params.num_dims) {
        return;
    }

    // Stage the 32 direction numbers for this dimension into workgroup memory once.
    if (lid.x < SOBOL_DEPTH) {
        let base = dimension * SOBOL_DEPTH;
        staged_dir[lid.x] = vectors.data[base + lid.x];
    }
    workgroupBarrier();

    // Each thread computes a small block of consecutive samples to amortize overhead.
    let block_base = params.sample_base + gid.x * SAMPLES_PER_THREAD;
    let hashed_seed = hash(params.seed ^ 0x79c68e4au);
    let scramble_seed = params.seed * 0x9c8f2d3bu;
    let dim_set = dimension >> 3u;
    let scramble = dim_set ^ scramble_seed ^ SCRAMBLE_TABLE[dimension & 7u];
    let scramble_hash = hash(scramble);

    for (var t : u32 = 0u; t < SAMPLES_PER_THREAD; t = t + 1u) {
        let sample_index = block_base + t;
        if (sample_index >= params.sample_count) {
            break;
        }

        let shuffled_rev_index = owen_scramble_rev(reverseBits(sample_index), hashed_seed);
        var sobol_val = 0u;
        var idx = shuffled_rev_index;
        var offset = 0u;
        loop {
            if (idx == 0u) {
                break;
            }
            let leading = countLeadingZeros(idx);
            let dir_index = offset + leading;
            if (dir_index < SOBOL_DEPTH) {
                sobol_val = sobol_val ^ staged_dir[dir_index];
            }
            offset = dir_index + 1u;
            idx = idx << (leading + 1u);
        }

        let sobol_owen = owen_scramble_rev(sobol_val, scramble_hash);
        let value = u32_to_f32_norm(reverseBits(sobol_owen));

        if ((params.flags & 1u) != 0u) {
            let out_index = gid.y * params.sample_count + sample_index;
            output.data[out_index] = value;
        } else {
            let out_index = sample_index * params.dim_count + gid.y;
            output.data[out_index] = value;
        }
    }
}
