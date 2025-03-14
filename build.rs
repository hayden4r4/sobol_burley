//! This f generates the Sobol direction vectors used by this crate's
//! Sobol sequence.

use std::{env, fs::File, io::Write, path::Path};

/// How many components to generate.
const NUM_DIMENSIONS: usize = 21201;

/// How many dimensions to generate at once when using SIMD.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub const SOBOL_WIDTH: usize = 4;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub const SOBOL_WIDTH: usize = 8;

/// What f to generate the numbers from.
const DIRECTION_NUMBERS_TEXT: &str = include_str!("direction_numbers/new-joe-kuo-6.21201.txt");

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("vectors.inc");
    let mut f = File::create(&dest_path).unwrap();

    // Init direction vectors.
    let vectors = generate_direction_vectors(NUM_DIMENSIONS);

    // Write dimensions limit.
    f.write_all(
        format!(
            "/// The number of available dimensions.\npub const NUM_DIMENSIONS: u32 = {};\n\n",
            NUM_DIMENSIONS
        )
        .as_bytes(),
    )
    .unwrap();

    // Write SIMD width.
    f.write_all(
        format!(
            "/// The number of dimensions to generate at once when using SIMD.\npub const SOBOL_WIDTH: usize = {};\n\n",
            SOBOL_WIDTH
        )
        .as_bytes(),
    ).unwrap();

    // Write the vectors.
    // We write them in a rather atypical way because of how the library
    // uses them.  First, we interleave the numbers of each set of 8
    // dimensions, for SIMD evaluation.  Second, each number is written
    // with reversed bits, to avoid needing to reverse them before scrambling.
    f.write_all(
        format!(
            "const REV_VECTORS: &[[[u{SOBOL_BITS}; {SOBOL_WIDTH}]; {SOBOL_DEPTH}]] = &[\n",
        )
        .as_bytes(),
    )
    .unwrap();
    for d4 in vectors.chunks_exact(SOBOL_WIDTH) {
        f.write_all("  [\n".as_bytes()).unwrap();
        for i in 0..SOBOL_DEPTH {
            f.write_all("    [".as_bytes()).unwrap();
            for j in 0..SOBOL_WIDTH {
                let value = d4[j][i].reverse_bits();
                if j < SOBOL_WIDTH - 1 {
                    f.write_all(format!("0x{:08x}, ", value).as_bytes()).unwrap();
                } else {
                    f.write_all(format!("0x{:08x}", value).as_bytes()).unwrap();
                }
            }
            f.write_all("],\n".as_bytes()).unwrap();
        }
        f.write_all("  ],\n".as_bytes()).unwrap();
    }
    f.write_all("];\n".as_bytes()).unwrap();
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
        // f.
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
/// text f.  Returns the `a` and `m` parts.
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
