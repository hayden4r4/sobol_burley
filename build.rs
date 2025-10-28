//! This file generates the Sobol direction vectors used by this crate's
//! Sobol sequence.

use std::{fs::File, io::Write, path::Path};

/// How many components to generate.
const NUM_DIMENSIONS: usize = 21201;
/// How many dimensions to generate at once when using SIMD.
const SIMD_WIDTH: usize = 8;

/// What file to generate the numbers from.
const DIRECTION_NUMBERS_TEXT: &str = include_str!("direction_numbers/new-joe-kuo-6.21201.txt");

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR env var");
    let dest_path = Path::new(&out_dir).join("vectors.inc");
    let mut f = File::create(&dest_path).unwrap();

    // Init direction vectors.
    let vectors = generate_direction_vectors(NUM_DIMENSIONS);

    // Pre-compute reversed-bit variants once so we can materialize both the
    // legacy block layout and the new flat GPU-friendly buffers without
    // repeating work.
    let mut rev_vectors_flat = Vec::with_capacity(NUM_DIMENSIONS);
    for vec in &vectors {
        let mut reversed = [0u32; SOBOL_DEPTH];
        for (dst, &src) in reversed.iter_mut().zip(vec.iter()) {
            *dst = src.reverse_bits();
        }
        rev_vectors_flat.push(reversed);
    }

    f.write_all(
        format!(
            concat!(
                "/// The number of available dimensions.\n",
                "pub const NUM_DIMENSIONS: u32 = {0};\n\n",
                "/// The number of dimensions processed per block.\n",
                "pub const SIMD_WIDTH: usize = {1};\n\n",
                "/// The number of direction numbers per dimension.\n",
                "pub const SOBOL_DEPTH: usize = {3};\n\n",
                "const REV_VECTORS: &[[[u{2}; {4}]; {3}]] = &[\n"
            ),
            NUM_DIMENSIONS as u32, SIMD_WIDTH, SOBOL_BITS, SOBOL_DEPTH, SIMD_WIDTH
        )
        .as_bytes(),
    )
    .unwrap();
    for chunk in rev_vectors_flat.chunks_exact(SIMD_WIDTH) {
        f.write_all(b"  [\n").unwrap();
        for depth_idx in 0..SOBOL_DEPTH {
            let mut line = String::from("    [");
            for (lane_idx, row) in chunk.iter().enumerate() {
                line.push_str(&format!("0x{:08x}", row[depth_idx]));
                if lane_idx + 1 != SIMD_WIDTH {
                    line.push_str(", ");
                }
            }
            line.push_str("],\n");
            f.write_all(line.as_bytes()).unwrap();
        }
        f.write_all(b"  ],\n").unwrap();
    }
    f.write_all(b"];\n").unwrap();

    // Emit a flat include!() file for host- or test-side usage where the GPU
    // layout is more appropriate than the SIMD-interleaved structure.
    let flat_inc_path = Path::new(&out_dir).join("vectors_flat.inc");
    let mut flat_inc = File::create(&flat_inc_path).unwrap();
    flat_inc
        .write_all(
            format!(
                "/// Flat reversed-bit direction vectors laid out as \
                 /// [NUM_DIMENSIONS][SOBOL_DEPTH].\n\
                 pub const REV_VECTORS_FLAT: &[[u{0}; {1}]] = &[\n",
                SOBOL_BITS, SOBOL_DEPTH
            )
            .as_bytes(),
        )
        .unwrap();
    for row in &rev_vectors_flat {
        let mut line = String::from("  [");
        for (idx, value) in row.iter().enumerate() {
            line.push_str(&format!("0x{:08x}", value));
            if idx + 1 != SOBOL_DEPTH {
                line.push_str(", ");
            }
        }
        line.push_str("],\n");
        flat_inc.write_all(line.as_bytes()).unwrap();
    }
    flat_inc.write_all(b"];\n").unwrap();

    // Write a raw binary blob (little-endian u32) for direct upload via GPU
    // APIs without requiring an include!() on the host side.
    let mut bin = File::create(Path::new(&out_dir).join("vectors.bin")).unwrap();
    for row in &rev_vectors_flat {
        for value in row {
            bin.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

//======================================================================

// The following is adapted from the code on this webpage:
//
// http://web.maths.unsw.edu.au/~fkuo/sobol/
//
// From these papers:
//
//     * S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's
//       quasirandom sequence generator, ACM Trans. Math. Softw. 29,
//       49-57 (2003)
//
//     * S. Joe and F. Y. Kuo, Constructing Sobol sequences with better
//       two-dimensional projections, SIAM J. Sci. Comput. 30, 2635-2654 (2008)
//
// It is under the 3-clause BSD license, copyright Stephen Joe and Frances
// Y. Kuo.  See `licenses/JOE_KUO.txt` for details.

type SobolInt = u32;
const SOBOL_BITS: usize = std::mem::size_of::<SobolInt>() * 8; // Bits per vector element.
const SOBOL_DEPTH: usize = 32; // Number of vector elements.

pub fn generate_direction_vectors(dimensions: usize) -> Vec<[SobolInt; SOBOL_DEPTH]> {
    let mut vectors = Vec::new();

    // Calculate first dimension, which is just the van der Corput sequence.
    let mut dim_0 = [0 as SobolInt; SOBOL_DEPTH];
    for i in 0..SOBOL_DEPTH {
        dim_0[i] = 1 << (SOBOL_BITS - 1 - i);
    }
    vectors.push(dim_0);

    // Do the rest of the dimensions.
    let mut lines = DIRECTION_NUMBERS_TEXT.lines();
    for _ in 1..dimensions {
        let mut v = [0 as SobolInt; SOBOL_DEPTH];

        // Get data from the next valid line from the direction numbers text
        // file.
        let (s, a, m) = loop {
            if let Ok((a, m)) = parse_direction_numbers(
                lines
                    .next()
                    .expect("Not enough direction numbers for the requested number of dimensions."),
            ) {
                break (m.len(), a, m);
            }
        };

        // Generate the direction numbers for this dimension.
        for i in 0..s.min(SOBOL_DEPTH) {
            v[i] = (m[i] << (SOBOL_BITS - (i + 1))) as SobolInt;
        }
        if s < SOBOL_DEPTH {
            for i in s..SOBOL_DEPTH {
                v[i] = v[i - s as usize] ^ (v[i - s as usize] >> s);

                for k in 1..s {
                    v[i] ^= ((a >> (s - 1 - k)) & 1) as SobolInt * v[i - k as usize];
                }
            }
        }

        vectors.push(v);
    }

    vectors
}

/// Parses the direction numbers from a single line of the direction numbers
/// text file.  Returns the `a` and `m` parts.
fn parse_direction_numbers(text: &str) -> Result<(u32, Vec<u32>), Box<dyn std::error::Error>> {
    let mut numbers = text.split_whitespace();
    if numbers.clone().count() < 4 || text.starts_with("#") {
        return Err(Box::new(ParseError(())));
    }

    // Skip the first two numbers, which are just the dimension and the count
    // of direction numbers for this dimension.
    let _ = numbers.next().unwrap().parse::<u32>()?;
    let _ = numbers.next().unwrap().parse::<u32>()?;

    let a = numbers.next().unwrap().parse::<u32>()?;

    let mut m = Vec::new();
    for n in numbers {
        m.push(n.parse::<u32>()?);
    }

    Ok((a, m))
}

#[derive(Debug, Copy, Clone)]
struct ParseError(());
impl std::error::Error for ParseError {}
impl std::fmt::Display for ParseError {
    fn fmt(&self, _f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Ok(())
    }
}
