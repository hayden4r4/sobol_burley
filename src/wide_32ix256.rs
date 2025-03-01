//--------------------------------------------------------------------------
// x86/64 AVX2
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub(crate) mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_or_si256, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_setzero_si256, _mm256_sll_epi32, _mm256_slli_epi32,
        _mm256_srl_epi32, _mm256_srli_epi32, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_si256,
        _mm_set_epi32,
    };

    /// A packed set of 16 `u32`s.
    ///
    /// Addition, subtraction, and multiplication are all wrapping.
    ///
    /// Uses SIMD for computation on supported platforms.
    #[derive(Debug, Copy, Clone)]
    pub struct PackedInt {
        v: __m256i,
    }

    impl PackedInt {
        #[inline(always)]
        pub(crate) fn zero() -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_setzero_si256() },
            }
        }

        /// For testing.
        #[allow(dead_code)]
        fn get(self, i: usize) -> u32 {
            let n: [u32; 8] = unsafe { core::mem::transmute(self) };
            n[i]
        }

        /// Convert each integer to a float in [0.0, 1.0).
        ///
        /// Same behavior as
        /// [`parts::u32_to_f32_norm()`](`crate::parts::u32_to_f32_norm()`),
        /// applied to each integer individually.
        #[inline(always)]
        pub fn to_f32_norm(self) -> [f32; 8] {
            let n8 = unsafe {
                let a = _mm256_srli_epi32(self.v, 9);
                let b = _mm256_or_si256(a, _mm256_set1_epi32(core::mem::transmute(0x3f800000u32)));
                _mm256_sub_ps(core::mem::transmute(b), _mm256_set1_ps(1.0))
            };

            unsafe { core::mem::transmute(n8) }
        }

        /// Reverse the order of the bits in each integer.
        ///
        /// Same behavior as `reverse_bits()` in the Rust standard
        /// library, applied to each integer individually.
        #[inline]
        pub fn reverse_bits(self) -> PackedInt {
            let mut n = self.v;
            unsafe {
                // From http://aggregate.org/MAGIC/#Bit%20Reversal but SIMD
                // on 16 numbers at once.

                let y0 = _mm256_set1_epi32(core::mem::transmute(0x55555555u32));
                n = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32(n, 1), y0),
                    _mm256_slli_epi32(_mm256_and_si256(n, y0), 1),
                );

                let y1 = _mm256_set1_epi32(core::mem::transmute(0x33333333u32));
                n = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32(n, 2), y1),
                    _mm256_slli_epi32(_mm256_and_si256(n, y1), 2),
                );

                let y2 = _mm256_set1_epi32(core::mem::transmute(0x0f0f0f0fu32));
                n = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32(n, 4), y2),
                    _mm256_slli_epi32(_mm256_and_si256(n, y2), 4),
                );

                let y3 = _mm256_set1_epi32(core::mem::transmute(0x00ff00ffu32));
                n = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32(n, 8), y3),
                    _mm256_slli_epi32(_mm256_and_si256(n, y3), 8),
                );

                n = _mm256_or_si256(_mm256_srli_epi32(n, 16), _mm256_slli_epi32(n, 16));

                PackedInt { v: n }
            }
        }
    }

    impl core::ops::Mul for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn mul(self, other: Self) -> PackedInt {
            unsafe {
                use core::arch::x86_64::_mm256_mullo_epi32;
                PackedInt {
                    v: _mm256_mullo_epi32(self.v, other.v),
                }
            }
        }
    }

    impl core::ops::MulAssign for PackedInt {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl core::ops::Add for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            PackedInt {
                v: unsafe { _mm256_add_epi32(self.v, other.v) },
            }
        }
    }

    impl core::ops::AddAssign for PackedInt {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl core::ops::Sub for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            PackedInt {
                v: unsafe { _mm256_sub_epi32(self.v, other.v) },
            }
        }
    }

    impl core::ops::SubAssign for PackedInt {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl core::ops::BitAnd for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_and_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitAndAssign for PackedInt {
        #[inline(always)]
        fn bitand_assign(&mut self, other: Self) {
            *self = *self & other;
        }
    }

    impl core::ops::BitOr for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_or_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitOrAssign for PackedInt {
        #[inline(always)]
        fn bitor_assign(&mut self, other: Self) {
            *self = *self | other;
        }
    }

    impl core::ops::BitXor for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_xor_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitXorAssign for PackedInt {
        #[inline(always)]
        fn bitxor_assign(&mut self, other: Self) {
            *self = *self ^ other;
        }
    }

    impl core::ops::Shl<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shl(self, other: i32) -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_sll_epi32(self.v, _mm_set_epi32(0, 0, 0, other)) },
            }
        }
    }

    impl core::ops::Shr<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shr(self, other: i32) -> PackedInt {
            PackedInt {
                v: unsafe { _mm256_srl_epi32(self.v, _mm_set_epi32(0, 0, 0, other)) },
            }
        }
    }

    impl From<[u32; 8]> for PackedInt {
        #[inline(always)]
        fn from(v: [u32; 8]) -> Self {
            PackedInt {
                v: unsafe { core::mem::transmute(v) },
            }
        }
    }

    impl From<PackedInt> for [u32; 8] {
        #[inline(always)]
        fn from(i: PackedInt) -> [u32; 8] {
            unsafe { core::mem::transmute(i.v) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn from_array() {
            let a = PackedInt::from([1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(a.get(0), 1);
            assert_eq!(a.get(1), 2);
            assert_eq!(a.get(2), 3);
            assert_eq!(a.get(3), 4);
            assert_eq!(a.get(4), 5);
            assert_eq!(a.get(5), 6);
            assert_eq!(a.get(6), 7);
            assert_eq!(a.get(7), 8);
        }

        #[test]
        fn shr() {
            let a = PackedInt::from([0xffffffff; 8]) >> 16;
            assert_eq!(a.get(0), 0x0000ffff);
            assert_eq!(a.get(1), 0x0000ffff);
            assert_eq!(a.get(2), 0x0000ffff);
            assert_eq!(a.get(3), 0x0000ffff);
            assert_eq!(a.get(4), 0x0000ffff);
            assert_eq!(a.get(5), 0x0000ffff);
            assert_eq!(a.get(6), 0x0000ffff);
            assert_eq!(a.get(7), 0x0000ffff);
        }

        #[test]
        fn shl() {
            let a = PackedInt::from([0xffffffff; 8]) << 16;
            assert_eq!(a.get(0), 0xffff0000);
            assert_eq!(a.get(1), 0xffff0000);
            assert_eq!(a.get(2), 0xffff0000);
            assert_eq!(a.get(3), 0xffff0000);
            assert_eq!(a.get(4), 0xffff0000);
            assert_eq!(a.get(5), 0xffff0000);
            assert_eq!(a.get(6), 0xffff0000);
            assert_eq!(a.get(7), 0xffff0000);
        }

        #[test]
        fn to_f32_norm() {
            let a = PackedInt::from([0x00000000; 8]);
            let b = PackedInt::from([0x80000000; 8]);
            let c = PackedInt::from([0xffffffff; 8]);

            let a2 = a.to_f32_norm();
            let b2 = b.to_f32_norm();
            let c2 = c.to_f32_norm();

            assert_eq!(a2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            assert_eq!(b2, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
            assert!(c2[0] > 0.99999 && c2[0] < 1.0);
            assert!(c2[1] > 0.99999 && c2[1] < 1.0);
            assert!(c2[2] > 0.99999 && c2[2] < 1.0);
            assert!(c2[3] > 0.99999 && c2[3] < 1.0);
            assert!(c2[4] > 0.99999 && c2[4] < 1.0);
            assert!(c2[5] > 0.99999 && c2[5] < 1.0);
            assert!(c2[6] > 0.99999 && c2[6] < 1.0);
            assert!(c2[7] > 0.99999 && c2[7] < 1.0);
        }

        #[test]
        fn reverse_bits() {
            let a = 0xcde7a64e_u32;
            let b = 0xdc69fbd9_u32;
            let c = 0x3238fec6_u32;
            let d = 0x1fb9ba8f_u32;

            assert_eq!(
                PackedInt::from([a; 8]).reverse_bits().get(0),
                a.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([b; 8]).reverse_bits().get(0),
                b.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([c; 8]).reverse_bits().get(0),
                c.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([d; 8]).reverse_bits().get(0),
                d.reverse_bits()
            );
        }
    }
}
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub use avx2::PackedInt;

//--------------------------------------------------------------------------
// ARM NEON
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub(crate) mod neon {
    use core::arch::aarch64::{
        uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_f32, vdupq_n_s32, vdupq_n_u32, veorq_u32,
        vmovq_n_u32, vnegq_s32, vorrq_u32, vreinterpretq_f32_u32, vshlq_n_u32, vshlq_u32,
        vshrq_n_u32, vsubq_f32, vsubq_u32,
    };

    /// A packed set of 8 `u32`s.
    ///
    /// Addition, subtraction, and multiplication are all wrapping.
    ///
    /// Uses SIMD for computation on supported platforms.
    #[derive(Debug, Copy, Clone)]
    pub struct PackedInt {
        v: uint32x4_t,
    }

    impl PackedInt {
        #[inline(always)]
        pub(crate) fn zero() -> PackedInt {
            PackedInt {
                v: unsafe { vmovq_n_u32(0) },
            }
        }

        /// For testing.
        #[allow(dead_code)]
        fn get(self, i: usize) -> u32 {
            let n: [u32; 4] = unsafe { core::mem::transmute(self) };
            n[i]
        }

        /// Convert each integer to a float in [0.0, 1.0).
        ///
        /// Same behavior as
        /// [`parts::u32_to_f32_norm()`](`crate::parts::u32_to_f32_norm()`),
        /// applied to each integer individually.
        #[inline(always)]
        pub fn to_f32_norm(self) -> [f32; 4] {
            let n4 = unsafe {
                let a = vshrq_n_u32(self.v, 9);
                let b = vorrq_u32(a, vdupq_n_u32(core::mem::transmute(0x3f800000u32)));
                vsubq_f32(vreinterpretq_f32_u32(b), vdupq_n_f32(1.0))
            };

            unsafe { core::mem::transmute(n4) }
        }

        /// Reverse the order of the bits in each integer.
        ///
        /// Same behavior as `reverse_bits()` in the Rust standard
        /// library, applied to each integer individually.
        #[inline]
        pub fn reverse_bits(self) -> PackedInt {
            let mut n = self.v;
            unsafe {
                // From http://aggregate.org/MAGIC/#Bit%20Reversal but SIMD
                // on 8 numbers at once.

                let y0 = vdupq_n_u32(core::mem::transmute(0x55555555u32));
                n = vorrq_u32(
                    vandq_u32(vshrq_n_u32(n, 1), y0),
                    vshlq_n_u32(vandq_u32(n, y0), 1),
                );

                let y1 = vdupq_n_u32(core::mem::transmute(0x33333333u32));
                n = vorrq_u32(
                    vandq_u32(vshrq_n_u32(n, 2), y1),
                    vshlq_n_u32(vandq_u32(n, y1), 2),
                );

                let y2 = vdupq_n_u32(core::mem::transmute(0x0f0f0f0fu32));
                n = vorrq_u32(
                    vandq_u32(vshrq_n_u32(n, 4), y2),
                    vshlq_n_u32(vandq_u32(n, y2), 4),
                );

                let y3 = vdupq_n_u32(core::mem::transmute(0x00ff00ffu32));
                n = vorrq_u32(
                    vandq_u32(vshrq_n_u32(n, 8), y3),
                    vshlq_n_u32(vandq_u32(n, y3), 8),
                );

                n = vorrq_u32(vshrq_n_u32(n, 16), vshlq_n_u32(n, 16));

                PackedInt { v: n }
            }
        }
    }

    impl core::ops::Mul for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn mul(self, other: Self) -> PackedInt {
            unsafe {
                use core::arch::aarch64::vmulq_u32;
                PackedInt {
                    v: vmulq_u32(self.v, other.v),
                }
            }
        }
    }

    impl core::ops::MulAssign for PackedInt {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl core::ops::Add for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            PackedInt {
                v: unsafe { vaddq_u32(self.v, other.v) },
            }
        }
    }

    impl core::ops::AddAssign for PackedInt {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl core::ops::Sub for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            PackedInt {
                v: unsafe { vsubq_u32(self.v, other.v) },
            }
        }
    }

    impl core::ops::SubAssign for PackedInt {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl core::ops::BitAnd for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { vandq_u32(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitAndAssign for PackedInt {
        #[inline(always)]
        fn bitand_assign(&mut self, other: Self) {
            *self = *self & other;
        }
    }

    impl core::ops::BitOr for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { vorrq_u32(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitOrAssign for PackedInt {
        #[inline(always)]
        fn bitor_assign(&mut self, other: Self) {
            *self = *self | other;
        }
    }

    impl core::ops::BitXor for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, other: Self) -> PackedInt {
            PackedInt {
                v: unsafe { veorq_u32(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitXorAssign for PackedInt {
        #[inline(always)]
        fn bitxor_assign(&mut self, other: Self) {
            *self = *self ^ other;
        }
    }

    impl core::ops::Shl<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shl(self, other: i32) -> PackedInt {
            let shift_vec = unsafe { vdupq_n_s32(other) };
            PackedInt {
                v: unsafe { vshlq_u32(self.v, shift_vec) },
            }
        }
    }

    impl core::ops::Shr<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shr(self, other: i32) -> PackedInt {
            let shift_vec = unsafe { vdupq_n_s32(other) };
            PackedInt {
                v: unsafe { vshlq_u32(self.v, vnegq_s32(shift_vec)) },
            }
        }
    }

    impl From<[u32; 4]> for PackedInt {
        #[inline(always)]
        fn from(v: [u32; 4]) -> Self {
            PackedInt {
                v: unsafe { core::mem::transmute(v) },
            }
        }
    }

    impl From<PackedInt> for [u32; 4] {
        #[inline(always)]
        fn from(i: PackedInt) -> [u32; 4] {
            unsafe { core::mem::transmute(i.v) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn from_array() {
            let a = PackedInt::from([1, 2, 3, 4]);
            assert_eq!(a.get(0), 1);
            assert_eq!(a.get(1), 2);
            assert_eq!(a.get(2), 3);
            assert_eq!(a.get(3), 4);
        }

        #[test]
        fn shr() {
            let a = PackedInt::from([0xffffffff; 4]) >> 16;
            assert_eq!(a.get(0), 0x0000ffff);
            assert_eq!(a.get(1), 0x0000ffff);
            assert_eq!(a.get(2), 0x0000ffff);
            assert_eq!(a.get(3), 0x0000ffff);
        }

        #[test]
        fn shl() {
            let a = PackedInt::from([0xffffffff; 4]) << 16;
            assert_eq!(a.get(0), 0xffff0000);
            assert_eq!(a.get(1), 0xffff0000);
            assert_eq!(a.get(2), 0xffff0000);
            assert_eq!(a.get(3), 0xffff0000);
        }

        #[test]
        fn to_f32_norm() {
            let a = PackedInt::from([0x00000000; 4]);
            let b = PackedInt::from([0x80000000; 4]);
            let c = PackedInt::from([0xffffffff; 4]);

            let a2 = a.to_f32_norm();
            let b2 = b.to_f32_norm();
            let c2 = c.to_f32_norm();

            assert_eq!(a2, [0.0, 0.0, 0.0, 0.0]);
            assert_eq!(b2, [0.5, 0.5, 0.5, 0.5]);
            assert!(c2[0] > 0.99999 && c2[0] < 1.0);
            assert!(c2[1] > 0.99999 && c2[1] < 1.0);
            assert!(c2[2] > 0.99999 && c2[2] < 1.0);
            assert!(c2[3] > 0.99999 && c2[3] < 1.0);
        }

        #[test]
        fn reverse_bits() {
            let a = 0xcde7a64e_u32;
            let b = 0xdc69fbd9_u32;
            let c = 0x3238fec6_u32;
            let d = 0x1fb9ba8f_u32;

            assert_eq!(
                PackedInt::from([a; 4]).reverse_bits().get(0),
                a.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([b; 4]).reverse_bits().get(0),
                b.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([c; 4]).reverse_bits().get(0),
                c.reverse_bits()
            );
            assert_eq!(
                PackedInt::from([d; 4]).reverse_bits().get(0),
                d.reverse_bits()
            );
        }
    }
}
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub use neon::PackedInt;

//--------------------------------------------------------------------------
// Fallback if no SIMD
#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub(crate) mod fallback {
    /// A packed set of 16 `u32`s.
    ///
    /// Uses SIMD for computation on supported platforms.
    #[derive(Debug, Copy, Clone)]
    #[repr(align(16))]
    pub struct PackedInt {
        v: [u32; 8],
    }

    impl PackedInt {
        #[inline(always)]
        pub(crate) fn zero() -> PackedInt {
            PackedInt {
                v: [0, 0, 0, 0, 0, 0, 0, 0],
            }
        }

        /// Convert each integer to a float in [0.0, 1.0).
        ///
        /// Same behavior as
        /// [`parts::u32_to_f32_norm()`](`crate::parts::u32_to_f32_norm()`),
        /// applied to each integer individually.
        #[inline(always)]
        pub fn to_f32_norm(self) -> [f32; 8] {
            [
                f32::from_bits((self.v[0] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[1] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[2] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[3] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[4] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[5] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[6] >> 9) | 0x3f800000) - 1.0,
                f32::from_bits((self.v[7] >> 9) | 0x3f800000) - 1.0,
            ]
        }

        /// Reverse the order of the bits in each integer.
        ///
        /// Same behavior as `reverse_bits()` in the Rust standard
        /// library, applied to each integer individually.
        #[inline(always)]
        pub fn reverse_bits(self) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0].reverse_bits(),
                    self.v[1].reverse_bits(),
                    self.v[2].reverse_bits(),
                    self.v[3].reverse_bits(),
                    self.v[4].reverse_bits(),
                    self.v[5].reverse_bits(),
                    self.v[6].reverse_bits(),
                    self.v[7].reverse_bits(),
                ],
            }
        }
    }

    impl core::ops::Mul for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn mul(self, other: Self) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0].wrapping_mul(other.v[0]),
                    self.v[1].wrapping_mul(other.v[1]),
                    self.v[2].wrapping_mul(other.v[2]),
                    self.v[3].wrapping_mul(other.v[3]),
                    self.v[4].wrapping_mul(other.v[4]),
                    self.v[5].wrapping_mul(other.v[5]),
                    self.v[6].wrapping_mul(other.v[6]),
                    self.v[7].wrapping_mul(other.v[7]),
                ],
            }
        }
    }

    impl core::ops::MulAssign for PackedInt {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl core::ops::Add for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            PackedInt {
                v: [
                    self.v[0].wrapping_add(other.v[0]),
                    self.v[1].wrapping_add(other.v[1]),
                    self.v[2].wrapping_add(other.v[2]),
                    self.v[3].wrapping_add(other.v[3]),
                    self.v[4].wrapping_add(other.v[4]),
                    self.v[5].wrapping_add(other.v[5]),
                    self.v[6].wrapping_add(other.v[6]),
                    self.v[7].wrapping_add(other.v[7]),
                ],
            }
        }
    }

    impl core::ops::AddAssign for PackedInt {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl core::ops::Sub for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            PackedInt {
                v: [
                    self.v[0].wrapping_sub(other.v[0]),
                    self.v[1].wrapping_sub(other.v[1]),
                    self.v[2].wrapping_sub(other.v[2]),
                    self.v[3].wrapping_sub(other.v[3]),
                    self.v[4].wrapping_sub(other.v[4]),
                    self.v[5].wrapping_sub(other.v[5]),
                    self.v[6].wrapping_sub(other.v[6]),
                    self.v[7].wrapping_sub(other.v[7]),
                ],
            }
        }
    }

    impl core::ops::SubAssign for PackedInt {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl core::ops::BitAnd for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, other: Self) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0] & other.v[0],
                    self.v[1] & other.v[1],
                    self.v[2] & other.v[2],
                    self.v[3] & other.v[3],
                    self.v[4] & other.v[4],
                    self.v[5] & other.v[5],
                    self.v[6] & other.v[6],
                    self.v[7] & other.v[7],
                ],
            }
        }
    }

    impl core::ops::BitAndAssign for PackedInt {
        #[inline(always)]
        fn bitand_assign(&mut self, other: Self) {
            *self = *self & other;
        }
    }

    impl core::ops::BitOr for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, other: Self) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0] | other.v[0],
                    self.v[1] | other.v[1],
                    self.v[2] | other.v[2],
                    self.v[3] | other.v[3],
                    self.v[4] | other.v[4],
                    self.v[5] | other.v[5],
                    self.v[6] | other.v[6],
                    self.v[7] | other.v[7],
                ],
            }
        }
    }

    impl core::ops::BitOrAssign for PackedInt {
        #[inline(always)]
        fn bitor_assign(&mut self, other: Self) {
            *self = *self | other;
        }
    }

    impl core::ops::BitXor for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, other: Self) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0] ^ other.v[0],
                    self.v[1] ^ other.v[1],
                    self.v[2] ^ other.v[2],
                    self.v[3] ^ other.v[3],
                    self.v[4] ^ other.v[4],
                    self.v[5] ^ other.v[5],
                    self.v[6] ^ other.v[6],
                    self.v[7] ^ other.v[7],
                ],
            }
        }
    }

    impl core::ops::BitXorAssign for PackedInt {
        #[inline(always)]
        fn bitxor_assign(&mut self, other: Self) {
            *self = *self ^ other;
        }
    }

    impl core::ops::Shl<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shl(self, other: i32) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0] << other,
                    self.v[1] << other,
                    self.v[2] << other,
                    self.v[3] << other,
                    self.v[4] << other,
                    self.v[5] << other,
                    self.v[6] << other,
                    self.v[7] << other,
                ],
            }
        }
    }

    impl core::ops::Shr<i32> for PackedInt {
        type Output = Self;
        #[inline(always)]
        fn shr(self, other: i32) -> PackedInt {
            PackedInt {
                v: [
                    self.v[0] >> other,
                    self.v[1] >> other,
                    self.v[2] >> other,
                    self.v[3] >> other,
                    self.v[4] >> other,
                    self.v[5] >> other,
                    self.v[6] >> other,
                    self.v[7] >> other,
                ],
            }
        }
    }

    impl From<[u32; 8]> for PackedInt {
        #[inline(always)]
        fn from(v: [u32; 8]) -> Self {
            PackedInt { v }
        }
    }

    impl From<PackedInt> for [u32; 8] {
        #[inline(always)]
        fn from(i: PackedInt) -> [u32; 8] {
            i.v
        }
    }
}
#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub use fallback::PackedInt;
