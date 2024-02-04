//--------------------------------------------------------------------------
// x86/64 AVX2
#[cfg(all(target_arch = "x86_64", feature = "simd", target_feature = "+avx2"))]
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
    pub struct Int8 {
        v: __m256i,
    }

    impl Int8 {
        #[inline(always)]
        pub(crate) fn zero() -> Int8 {
            Int8 {
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
        pub fn reverse_bits(self) -> Int8 {
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

                Int8 { v: n }
            }
        }
    }

    impl core::ops::Mul for Int8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, other: Self) -> Int8 {
            unsafe {
                use core::arch::x86_64::_mm256_mullo_epi32;
                Int8 {
                    v: _mm256_mullo_epi32(self.v, other.v),
                }
            }
        }
    }

    impl core::ops::MulAssign for Int8 {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl core::ops::Add for Int8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            Int8 {
                v: unsafe { _mm256_add_epi32(self.v, other.v) },
            }
        }
    }

    impl core::ops::AddAssign for Int8 {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl core::ops::Sub for Int8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            Int8 {
                v: unsafe { _mm256_sub_epi32(self.v, other.v) },
            }
        }
    }

    impl core::ops::SubAssign for Int8 {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl core::ops::BitAnd for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, other: Self) -> Int8 {
            Int8 {
                v: unsafe { _mm256_and_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitAndAssign for Int8 {
        #[inline(always)]
        fn bitand_assign(&mut self, other: Self) {
            *self = *self & other;
        }
    }

    impl core::ops::BitOr for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, other: Self) -> Int8 {
            Int8 {
                v: unsafe { _mm256_or_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitOrAssign for Int8 {
        #[inline(always)]
        fn bitor_assign(&mut self, other: Self) {
            *self = *self | other;
        }
    }

    impl core::ops::BitXor for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, other: Self) -> Int8 {
            Int8 {
                v: unsafe { _mm256_xor_si256(self.v, other.v) },
            }
        }
    }

    impl core::ops::BitXorAssign for Int8 {
        #[inline(always)]
        fn bitxor_assign(&mut self, other: Self) {
            *self = *self ^ other;
        }
    }

    impl core::ops::Shl<i32> for Int8 {
        type Output = Self;
        #[inline(always)]
        fn shl(self, other: i32) -> Int8 {
            Int8 {
                v: unsafe { _mm256_sll_epi32(self.v, _mm_set_epi32(0, 0, 0, other)) },
            }
        }
    }

    impl core::ops::Shr<i32> for Int8 {
        type Output = Self;
        #[inline(always)]
        fn shr(self, other: i32) -> Int8 {
            Int8 {
                v: unsafe { _mm256_srl_epi32(self.v, _mm_set_epi32(0, 0, 0, other)) },
            }
        }
    }

    impl From<[u32; 8]> for Int8 {
        #[inline(always)]
        fn from(v: [u32; 8]) -> Self {
            Int8 {
                v: unsafe { core::mem::transmute(v) },
            }
        }
    }

    impl From<Int8> for [u32; 8] {
        #[inline(always)]
        fn from(i: Int8) -> [u32; 8] {
            unsafe { core::mem::transmute(i.v) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn from_array() {
            let a = Int8::from([1, 2, 3, 4, 5, 6, 7, 8]);
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
            let a = Int8::from([0xffffffff; 8]) >> 16;
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
            let a = Int8::from([0xffffffff; 8]) << 16;
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
            let a = Int8::from([0x00000000; 8]);
            let b = Int8::from([0x80000000; 8]);
            let c = Int8::from([0xffffffff; 8]);

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

            assert_eq!(Int8::from([a; 8]).reverse_bits().get(0), a.reverse_bits());
            assert_eq!(Int8::from([b; 8]).reverse_bits().get(0), b.reverse_bits());
            assert_eq!(Int8::from([c; 8]).reverse_bits().get(0), c.reverse_bits());
            assert_eq!(Int8::from([d; 8]).reverse_bits().get(0), d.reverse_bits());
        }
    }
}
#[cfg(all(target_arch = "x86_64", feature = "simd", target_feature = "+avx2",))]
pub use avx2::Int8;

//--------------------------------------------------------------------------
// Fallback
#[cfg(not(all(target_arch = "x86_64", feature = "simd", target_feature = "+avx2",)))]
pub(crate) mod fallback {
    /// A packed set of 16 `u32`s.
    ///
    /// Uses SIMD for computation on supported platforms.
    #[derive(Debug, Copy, Clone)]
    #[repr(align(16))]
    pub struct Int8 {
        v: [u32; 8],
    }

    impl Int8 {
        #[inline(always)]
        pub(crate) fn zero() -> Int8 {
            Int8 {
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
        pub fn reverse_bits(self) -> Int8 {
            Int8 {
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

    impl core::ops::Mul for Int8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, other: Self) -> Int8 {
            Int8 {
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

    impl core::ops::MulAssign for Int8 {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl core::ops::Add for Int8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            Int8 {
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

    impl core::ops::AddAssign for Int8 {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl core::ops::Sub for Int8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            Int8 {
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

    impl core::ops::SubAssign for Int8 {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl core::ops::BitAnd for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, other: Self) -> Int8 {
            Int8 {
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

    impl core::ops::BitAndAssign for Int8 {
        #[inline(always)]
        fn bitand_assign(&mut self, other: Self) {
            *self = *self & other;
        }
    }

    impl core::ops::BitOr for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, other: Self) -> Int8 {
            Int8 {
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

    impl core::ops::BitOrAssign for Int8 {
        #[inline(always)]
        fn bitor_assign(&mut self, other: Self) {
            *self = *self | other;
        }
    }

    impl core::ops::BitXor for Int8 {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, other: Self) -> Int8 {
            Int8 {
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

    impl core::ops::BitXorAssign for Int8 {
        #[inline(always)]
        fn bitxor_assign(&mut self, other: Self) {
            *self = *self ^ other;
        }
    }

    impl core::ops::Shl<i32> for Int8 {
        type Output = Self;
        #[inline(always)]
        fn shl(self, other: i32) -> Int8 {
            Int8 {
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

    impl core::ops::Shr<i32> for Int8 {
        type Output = Self;
        #[inline(always)]
        fn shr(self, other: i32) -> Int8 {
            Int8 {
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

    impl From<[u32; 8]> for Int8 {
        #[inline(always)]
        fn from(v: [u32; 8]) -> Self {
            Int8 { v }
        }
    }

    impl From<Int8> for [u32; 8] {
        #[inline(always)]
        fn from(i: Int8) -> [u32; 8] {
            i.v
        }
    }
}
#[cfg(not(all(target_arch = "x86_64", feature = "simd", target_feature = "+avx2",)))]
pub use fallback::Int8;
