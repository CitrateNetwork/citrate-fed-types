//! Q16.16 deterministic fixed-point — the quantization grid every committed byte rides.
//!
//! Faithful extraction of `nat-types::fixed` (the canonical definition). Integer
//! arithmetic is bit-identical across platforms and federated nodes where IEEE-754 float
//! is not; this is the property that lets federated results reconcile and on-chain
//! provenance verify. Representation: a value `v` is stored as `round(v * 2^16)` in an
//! `i64`; multiplication uses an `i128` intermediate so the product cannot overflow before
//! the right-shift.

use serde::{Deserialize, Serialize};

const FRAC_BITS: u32 = 16;
const ONE_RAW: i64 = 1 << FRAC_BITS; // 65536

/// A Q16.16 fixed-point number. Serializes as its raw integer so the encoding is exact
/// and platform-independent (no float ever touches the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Q16(i64);

impl Q16 {
    pub const ZERO: Q16 = Q16(0);
    pub const ONE: Q16 = Q16(ONE_RAW);

    /// Construct from a raw Q16.16 integer (value = raw / 2^16).
    pub const fn from_raw(raw: i64) -> Self {
        Q16(raw)
    }

    /// The underlying raw integer. This is what gets hashed and committed.
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Quantize an `f32` onto the Q16.16 grid — the *only* lossy boundary. Round to
    /// nearest, ties away from zero, deterministically.
    pub fn from_f32(v: f32) -> Self {
        let scaled = (v as f64) * (ONE_RAW as f64);
        Q16(scaled.round() as i64)
    }

    /// Dequantize back to `f32` for display / non-deterministic downstream use.
    pub fn to_f32(self) -> f32 {
        (self.0 as f64 / ONE_RAW as f64) as f32
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Q16) -> Q16 {
        Q16(self.0 + other.0)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Q16) -> Q16 {
        Q16(self.0 - other.0)
    }

    /// Fixed-point multiply: (a * b) >> 16, via i128 to avoid mid-product overflow.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Q16) -> Q16 {
        let prod = (self.0 as i128) * (other.0 as i128);
        Q16((prod >> FRAC_BITS) as i64)
    }

    /// Fixed-point divide: (a << 16) / b, via i128. Caller guarantees `other != 0`.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Q16) -> Q16 {
        debug_assert!(other.0 != 0, "Q16 division by zero");
        let num = (self.0 as i128) << FRAC_BITS;
        Q16((num / other.0 as i128) as i64)
    }
}

impl std::iter::Sum for Q16 {
    fn sum<I: Iterator<Item = Q16>>(iter: I) -> Self {
        iter.fold(Q16::ZERO, |acc, x| acc.add(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_round_trips() {
        assert_eq!(Q16::ONE.raw(), 65536);
        assert_eq!(Q16::from_f32(1.0), Q16::ONE);
        assert_eq!(Q16::ONE.to_f32(), 1.0);
    }

    #[test]
    fn mul_is_exact_on_grid() {
        let half = Q16::from_f32(0.5);
        assert_eq!(half.mul(half), Q16::from_f32(0.25));
    }

    #[test]
    fn from_f32_rounds_to_nearest_ties_away() {
        // 0.9 * 65536 = 58982.4 -> 58982 (nearest); the property the ledger bps math needs.
        assert_eq!(Q16::from_f32(0.9).raw(), 58982);
    }

    #[test]
    fn determinism_same_inputs_same_bits() {
        let xs = [0.1f32, 0.2, 0.3, 0.4];
        let run = || -> i64 {
            xs.iter()
                .map(|&x| Q16::from_f32(x))
                .sum::<Q16>()
                .mul(Q16::from_f32(7.0))
                .raw()
        };
        assert_eq!(run(), run());
    }
}
