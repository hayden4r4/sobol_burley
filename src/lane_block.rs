//! GPU-friendly lane block abstraction used in place of CPU SIMD types.

#![cfg_attr(not(test), allow(dead_code))]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign,
    Shl, Shr, Sub, SubAssign,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneBlock<const N: usize> {
    lanes: [u32; N],
}

impl<const N: usize> LaneBlock<N> {
    #[inline(always)]
    pub const fn new(lanes: [u32; N]) -> Self {
        Self { lanes }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { lanes: [0; N] }
    }

    #[inline(always)]
    pub fn into_inner(self) -> [u32; N] {
        self.lanes
    }

    #[inline(always)]
    pub fn reverse_bits(self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].reverse_bits();
            i += 1;
        }
        Self { lanes }
    }

    #[inline(always)]
    pub fn to_f32_norm(self) -> [f32; N] {
        let mut out = [0.0f32; N];
        let mut i = 0;
        while i < N {
            out[i] = f32::from_bits((self.lanes[i] >> 9) | 0x3f80_0000) - 1.0;
            i += 1;
        }
        out
    }
}

impl<const N: usize> From<[u32; N]> for LaneBlock<N> {
    #[inline(always)]
    fn from(lanes: [u32; N]) -> Self {
        Self { lanes }
    }
}

impl<const N: usize> From<LaneBlock<N>> for [u32; N] {
    #[inline(always)]
    fn from(block: LaneBlock<N>) -> Self {
        block.lanes
    }
}

impl<const N: usize> Add for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].wrapping_add(rhs.lanes[i]);
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> AddAssign for LaneBlock<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] = self.lanes[i].wrapping_add(rhs.lanes[i]);
            i += 1;
        }
    }
}

impl<const N: usize> Sub for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].wrapping_sub(rhs.lanes[i]);
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> SubAssign for LaneBlock<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] = self.lanes[i].wrapping_sub(rhs.lanes[i]);
            i += 1;
        }
    }
}

impl<const N: usize> Mul for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].wrapping_mul(rhs.lanes[i]);
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> MulAssign for LaneBlock<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] = self.lanes[i].wrapping_mul(rhs.lanes[i]);
            i += 1;
        }
    }
}

impl<const N: usize> BitAnd for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] &= rhs.lanes[i];
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> BitAndAssign for LaneBlock<N> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] &= rhs.lanes[i];
            i += 1;
        }
    }
}

impl<const N: usize> BitOr for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] |= rhs.lanes[i];
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> BitOrAssign for LaneBlock<N> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] |= rhs.lanes[i];
            i += 1;
        }
    }
}

impl<const N: usize> BitXor for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] ^= rhs.lanes[i];
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> BitXorAssign for LaneBlock<N> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        let mut i = 0;
        while i < N {
            self.lanes[i] ^= rhs.lanes[i];
            i += 1;
        }
    }
}

impl<const N: usize> Shl<i32> for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        let shift = rhs as u32;
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].wrapping_shl(shift);
            i += 1;
        }
        Self { lanes }
    }
}

impl<const N: usize> Shr<i32> for LaneBlock<N> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        let shift = rhs as u32;
        let mut lanes = self.lanes;
        let mut i = 0;
        while i < N {
            lanes[i] = lanes[i].wrapping_shr(shift);
            i += 1;
        }
        Self { lanes }
    }
}

#[cfg(test)]
mod tests {
    use super::LaneBlock;

    const N: usize = 8;

    #[test]
    fn construction_and_into() {
        let block = LaneBlock::<N>::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let arr: [u32; N] = block.into();
        assert_eq!(arr, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn reverse_bits_matches_scalar() {
        let block = LaneBlock::<N>::new([0x0123_4567; N]);
        let rev = block.reverse_bits();
        assert_eq!(rev.into_inner()[0], 0xE6A2_C480);
    }

    #[test]
    fn arithmetic_and_shifts() {
        let a = LaneBlock::<N>::new([1; N]);
        let b = LaneBlock::<N>::new([2; N]);
        assert_eq!((a + b).into_inner(), [3; N]);
        assert_eq!((a * b).into_inner(), [2; N]);
        assert_eq!((a << 1).into_inner(), [2; N]);
    }
}
