//! A seedable Owen-scrambled Sobol sequence.
//!
//! This crate is based on the paper [Practical Hash-based Owen
//! Scrambling](http://www.jcgt.org/published/0009/04/01/) by Brent Burley,
//! but with an improved hash from [Building a Better LK
//! Hash](https://psychopath.io/post/2021_01_30_building_a_better_lk_hash)
//! and more dimensions due to
//! [Kuo et al.](http://web.maths.unsw.edu.au/~fkuo/sobol/)
//!
//! The below restrictions apply:
//!
//! * The maximum sequence length is 2^32.
//! * The maximum number of dimensions is 21201 (although this can be worked
//!   around with seeding).
//! * Only `f32` output is supported.
//!
//! These are all trade-offs for the sake of better performance and a smaller
//! memory footprint.
//!
//!
//! ## Basic usage
//!
//! Basic usage is pretty straightforward:
//!
//! ```rust
//! use sobol_burley::sample;
//!
//! // Print 1024 3-dimensional points.
//! for i in 0..1024 {
//!     let x = sample(i, 0, 0);
//!     let y = sample(i, 1, 0);
//!     let z = sample(i, 2, 0);
//!     println!("({}, {}, {})", x, y, z);
//! }
//! ```
//!
//! The first parameter of `sample()` is the index of the sample you want,
//! and the second parameter is the index of the dimension you want.  The
//! parameters are zero-indexed, and outputs are in the interval [0, 1).
//!
//! If all you want is a single Owen-scrambled Sobol sequence, then this is
//! all you need.  You can ignore the third parameter.
//!
//!
//! ## Seeding
//!
//! *(Note: the `sample()` function automatically uses a different Owen
//! scramble for each dimension, so seeding is unnecessary if you just want
//! a single Sobol sequence.)*
//!
//! The third parameter of `sample()` is a seed that produces statistically
//! independent Sobol sequences via the scrambling+shuffling technique from
//! Brent Burley's paper.
//!
//! One of the applications for this is to decorrelate the error between
//! related integral estimates.  For example, in a 3d renderer you might
//! pass a different seed to each pixel so that error in the pixel colors
//! shows up as noise instead of as structured artifacts.
//!
//! Another important application is "padding" the dimensions of a Sobol
//! sequence.  By changing the seed we can re-use the same dimensions over
//! and over to create an arbitrarily high-dimensional sequence.  For example:
//!
//! ```rust
//! # use sobol_burley::sample;
//! // Print 10000 dimensions of a single sample.
//! for dimension in 0..10000 {
//!     let seed = dimension / SOBOL_WIDTH as u32;
//!     let n = sample(0, dimension % SOBOL_WIDTH as u32, seed);
//!     println!("{}", n);
//! }
//!```
//!
//! In this example we change seeds every SOBOL_WIDTH dimensions.  This allows us to
//! re-use the same SOBOL_WIDTH dimensions over and over, extending the sequence to as
//! many dimensions as we like.  Each set of SOBOL_WIDTH dimensions is stratified within
//! itself, but is randomly decorrelated from the other sets.
//!
//! See Burley's paper for justification of this padding approach as well as
//! recommendations about its use.
//!
//!
//! # SIMD
//!
//! You can use `sample_simd()` to compute SOBOL_WIDTH dimensions at once, returned as
//! an array of floats.
//!
//! On x86-64 architectures `sample_simd()` utilizes SIMD for a roughly SOBOL_WIDTHx
//! speed-up.  On other architectures it still computes correct results, but
//! SIMD isn't supported yet.
//!
//! Importantly, `sample()` and `sample_simd()` always compute identical results:
//!
//! ```rust
//! # use sobol_burley::{sample, sample_simd};
//! for dimension_set in 0..10 {
//!     let a = [
//!         sample(0, dimension_set * SOBOL_WIDTH, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 1, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 2, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 3, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 4, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 5, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 6, 0),
//!         sample(0, dimension_set * SOBOL_WIDTH + 7, 0)
//!     ];
//!     let b = sample_simd(0, dimension_set, 0);
//!
//!     assert_eq!(a, b);
//! }
//! ```
//!
//! The difference is only in performance and how the dimensions are indexed.

#![no_std]
#![allow(clippy::unreadable_literal)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod parts;
mod wide_32ix256;

// This `include` provides `NUM_DIMENSIONS` and `REV_VECTORS`.
// See the build.rs file for how this included file is generated.
include!(concat!(env!("OUT_DIR"), "/vectors.inc"));

/// The number of available 4d dimension sets.
///
/// This is just `NUM_DIMENSIONS / SOBOL_WIDTH`, for convenience.
pub const NUM_DIMENSION_SETS_SIMD: u32 = NUM_DIMENSIONS / SOBOL_WIDTH as u32;

/// Compute one dimension of a single sample in the Sobol sequence.
///
/// `sample_index` specifies which sample in the Sobol sequence to compute.
/// A maxmimum of 2^32 samples is supported.
///
/// `dimension` specifies which dimension to compute.
///
/// `seed` produces statistically independent Sobol sequences.  Passing two
/// different seeds will produce two different sequences that are only randomly
/// associated, with no stratification or correlation between them.
///
/// Returns a number in the interval [0, 1).
///
/// # Panics
///
/// * Panics if `dimension` is greater than or equal to [`NUM_DIMENSIONS`].
/// * In debug, panics if `sample_index` is greater than or equal to 2^32.
///   In release, returns unspecified floats in the interval [0, 1).
#[inline]
pub fn sample(sample_index: u32, dimension: u32, seed: u32) -> f32 {
    use parts::*;
    debug_assert!(sample_index < 4_294_967_295);

    // Shuffle the index using the given seed to produce a unique statistically
    // independent Sobol sequence.
    let shuffled_rev_index =
        owen_scramble_rev(sample_index.reverse_bits(), hash(seed ^ 0x79c68e4a));

    let sobol = sobol_rev(shuffled_rev_index, dimension);

    // Compute the scramble value for doing Owen scrambling.
    // The multiply on `seed` is to avoid accidental cancellation
    // with `dimension` on an incrementing or otherwise structured
    // seed.
    let scramble = {
        let seed = seed.wrapping_mul(0x9c8f2d3b);
        let ds = dimension >> 3;
        // These should probably be different for each channel, but I just duplicate
        // the existing 4D ones as the difference is almost certainly negligible
        // Current max support for 16 dimensions
        let scramble_arr: [u32; SOBOL_WIDTH] = [
            0x912f69ba, 0x174f18ab, 0x691e72ca, 0xb40cc1b8, 0x912f69ba, 0x174f18ab, 0x691e72ca,
            0xb40cc1b8, 0x912f69ba, 0x174f18ab, 0x691e72ca, 0xb40cc1b8, 0x912f69ba, 0x174f18ab,
            0x691e72ca, 0xb40cc1b8,
        ][0..SOBOL_WIDTH]
            .try_into()
            .unwrap();

        ds ^ seed
            ^ scramble_arr[dimension as usize & 0b111]
    };

    let sobol_owen_rev = owen_scramble_rev(sobol, hash(scramble));

    u32_to_f32_norm(sobol_owen_rev.reverse_bits())
}

/// Compute SOBOL_WIDTH dimensions of a single sample in the Sobol sequence.
///
/// This is identical to [`sample()`], but computes SOBOL_WIDTH dimensions at once.
/// On x86-64 architectures it utilizes SIMD for a roughly 8x speed-up.
/// On other architectures it still computes correct results, but doesn't
/// utilize SIMD.
///
/// `dimension_set` specifies which SOBOL_WIDTH dimensions to compute. `0` yields the
/// first SOBOL_WIDTH dimensions, `1` the second SOBOL_WIDTH dimensions, and so on.
///
/// # Panics
///
/// * Panics if `dimension_set` is greater than or equal to
///   [`NUM_DIMENSION_SETS_SIMD`].
/// * In debug, panics if `sample_index` is greater than or equal to 2^32.
///   In release, returns unspecified floats in the interval [0, 1).
#[inline]
pub fn sample_simd(sample_index: u32, dimension_set: u32, seed: u32) -> [f32; SOBOL_WIDTH] {
    use parts::*;
    debug_assert!(sample_index < 4_294_967_295);

    // Shuffle the index using the given seed to produce a unique statistically
    // independent Sobol sequence.
    let shuffled_rev_index =
        owen_scramble_rev(sample_index.reverse_bits(), hash(seed ^ 0x79c68e4a));

    let sobol = sobol_simd_rev(shuffled_rev_index, dimension_set);

    // Compute the scramble values for doing Owen scrambling.
    // The multiply on `seed` is to avoid accidental cancellation
    // with `dimension` on an incrementing or otherwise structured
    // seed.
    let scramble = {
        let seed: PackedInt = [seed.wrapping_mul(0x9c8f2d3b); SOBOL_WIDTH].into();
        let ds: PackedInt = [dimension_set; SOBOL_WIDTH].into();
        let scramble_arr: [u32; SOBOL_WIDTH] = [
            0x912f69ba, 0x174f18ab, 0x691e72ca, 0xb40cc1b8, 0x912f69ba, 0x174f18ab, 0x691e72ca,
            0xb40cc1b8, 0x912f69ba, 0x174f18ab, 0x691e72ca, 0xb40cc1b8, 0x912f69ba, 0x174f18ab,
            0x691e72ca, 0xb40cc1b8,
        ][0..SOBOL_WIDTH]
            .try_into()
            .unwrap();
        // These random values should probably be different for each channel,
        // but I just duplicate the existing 4D ones as the difference is almost certainly
        // negligible
        seed ^ ds ^ scramble_arr.into()
    };

    let sobol_owen_rev = owen_scramble_simd_rev(sobol, hash_simd(scramble));

    // Un-reverse the bits and convert to floating point in [0, 1).
    sobol_owen_rev.reverse_bits().to_f32_norm()
}

//----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_1d_and_simd_match() {
        for s in 0..4 {
            for d in 0..SOBOL_WIDTH as u32 {
                for n in 0..21201 {
                    let a1 = sample(n, d * SOBOL_WIDTH as u32, s);
                    let b1 = sample(n, d * SOBOL_WIDTH as u32 + 1, s);
                    let c1 = sample(n, d * SOBOL_WIDTH as u32 + 2, s);
                    let d1 = sample(n, d * SOBOL_WIDTH as u32 + 3, s);
                    let e1 = sample(n, d * SOBOL_WIDTH as u32 + 4, s);
                    let f1 = sample(n, d * SOBOL_WIDTH as u32 + 5, s);
                    let g1 = sample(n, d * SOBOL_WIDTH as u32 + 6, s);
                    let h1 = sample(n, d * SOBOL_WIDTH as u32 + 7, s);

                    let [a2, b2, c2, d2, e2, f2, g2, h2] = sample_simd(n, d, s);

                    assert_eq!(a1, a2);
                    assert_eq!(b1, b2);
                    assert_eq!(c1, c2);
                    assert_eq!(d1, d2);
                    assert_eq!(e1, e2);
                    assert_eq!(f1, f2);
                    assert_eq!(g1, g2);
                    assert_eq!(h1, h2);
                }
            }
        }
    }
}
