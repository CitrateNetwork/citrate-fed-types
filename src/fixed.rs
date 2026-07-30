//! Q16.16 deterministic fixed-point — the quantization grid every committed byte rides.
//!
//! Faithful extraction of `nat-types::fixed` (the canonical definition). Integer
//! arithmetic is bit-identical across platforms, federated nodes, *and build profiles*
//! where IEEE-754 float is not; this is the property that lets federated results reconcile
//! and on-chain provenance verify. Representation: a value `v` is stored as `round(v * 2^16)`
//! in an `i64`; multiplication uses an `i128` intermediate so the product cannot overflow
//! before the right-shift.
//!
//! **Overflow contract (Tier-1 remediation, finding H3/L2/M2):** every operation is
//! *saturating* and *total* — `add`/`sub`/`mul` saturate to `i64::MIN`/`i64::MAX` rather
//! than panicking (debug) or wrapping (release), and `div` by zero saturates by sign. This
//! removes the only build-profile-divergent behavior, so two honest nodes on `debug` and
//! `release` cannot fork. Committed reductions still ride `i128` intermediates and never
//! reach saturation with realistic magnitudes; the saturation is the fail-safe boundary.

use serde::{Deserialize, Serialize};

/// **Do not raise this. The `i64` headroom is for range, never for resolution.**
///
/// The widening from `i32` to `i64` bought representable *range*. It is tempting to read
/// the spare bits as room for a finer grid — `Q32.32` — and that would silently destroy
/// the property cross-backend settlement depends on.
///
/// Two honest members training identical work on different hardware do not land on
/// bit-identical weights; CUDA, Metal and CPU differ in reduction order and fused kernels.
/// Measured across two machines and two vendors (2026-07-30), the worst honest divergence
/// is **1.34e-07** — about one ULP of f32 — against a grid step of `2^-16` = **1.53e-05**.
/// The grid is ~114× coarser than the disagreement, and that is exactly why it works: the
/// quantization *absorbs* honest hardware drift.
///
/// The share of coordinates landing on opposite sides of a grid boundary is roughly
/// `divergence / grid_step`:
///
/// | `FRAC_BITS` | grid step | straddling coordinates |
/// |---|---|---|
/// | 16 (today) | 1.53e-05 | ~0.9% |
/// | 32 | 2.33e-10 | **all of them, by ~565 steps** |
///
/// At 32 fractional bits the grid is finer than the noise floor of the hardware, so it
/// stops quantizing the disagreement away and starts recording it. Every committed value
/// would differ between honest members, by hundreds of raw units rather than one.
///
/// The float precision is not the lever either. Divergence is already at one ULP of f32,
/// the smallest disagreement the format can express, so there is no sloppiness to recover;
/// and f64 is not an option in any case — Apple GPUs have no f64 ALU, so it would exclude
/// every Mac in the co-op from accelerated work.
///
/// Range is the safe axis. Resolution is not.
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
    ///
    /// Non-finite inputs (`NaN`, `±∞`) have no grid point; they map deterministically to
    /// [`Q16::ZERO`] (finding L1) so a poison float can neither split nodes nor inject an
    /// `i64::MAX` raw that would reach the saturating-arithmetic band. Huge *finite* inputs
    /// still saturate to `±i64::MAX` via the deterministic float→int cast.
    pub fn from_f32(v: f32) -> Self {
        if !v.is_finite() {
            return Q16::ZERO;
        }
        let scaled = (v as f64) * (ONE_RAW as f64);
        Q16(scaled.round() as i64)
    }

    /// Dequantize back to `f32` for display / non-deterministic downstream use.
    pub fn to_f32(self) -> f32 {
        (self.0 as f64 / ONE_RAW as f64) as f32
    }

    /// Saturating fixed-point add — never panics (debug) or wraps (release); clamps to
    /// `i64::MIN`/`i64::MAX` (finding H3).
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Q16) -> Q16 {
        Q16(self.0.saturating_add(other.0))
    }

    /// Saturating fixed-point sub — same total/saturating contract as [`Q16::add`] (H3).
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Q16) -> Q16 {
        Q16(self.0.saturating_sub(other.0))
    }

    /// Fixed-point multiply: `(a * b) >> 16`, via `i128` so the mid-product cannot overflow.
    /// The post-shift down-cast to `i64` *saturates* rather than truncating/sign-flipping at
    /// extreme magnitude (finding L2).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Q16) -> Q16 {
        let prod = (self.0 as i128) * (other.0 as i128);
        Q16(saturating_i128_to_i64(prod >> FRAC_BITS))
    }

    /// Fixed-point divide: `(a << 16) / b`, via `i128`. Division by zero has no grid point,
    /// so it saturates by sign (`+i64::MAX` / `-i64::MIN`) rather than panicking in *any*
    /// build profile (finding M2). No committed path divides; this is the fail-safe boundary.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Q16) -> Q16 {
        if other.0 == 0 {
            return Q16(if self.0 >= 0 { i64::MAX } else { i64::MIN });
        }
        let num = (self.0 as i128) << FRAC_BITS;
        Q16(saturating_i128_to_i64(num / other.0 as i128))
    }
}

/// Clamp an `i128` into `i64` range deterministically (saturating down-cast).
fn saturating_i128_to_i64(v: i128) -> i64 {
    v.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

impl std::iter::Sum for Q16 {
    fn sum<I: Iterator<Item = Q16>>(iter: I) -> Self {
        iter.fold(Q16::ZERO, |acc, x| acc.add(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid step is load-bearing for cross-backend settlement, so it is pinned
    /// rather than left implicit. See the note on `FRAC_BITS`: honest hardware
    /// divergence is ~1.34e-07 and this grid is ~114× coarser, which is what lets
    /// quantization absorb the disagreement. Making the grid finer would make every
    /// honest member's commitment differ.
    ///
    /// If you are here because this test failed, you changed `FRAC_BITS`. That is a
    /// settlement-breaking change, not a precision improvement — read the note first.
    #[test]
    fn the_grid_step_is_pinned_at_2_pow_minus_16() {
        assert_eq!(
            FRAC_BITS, 16,
            "FRAC_BITS is load-bearing — see its doc comment"
        );
        assert_eq!(ONE_RAW, 65_536);
        let step = 1.0f64 / ONE_RAW as f64;
        assert!(
            (step - 1.525_878_906_25e-5).abs() < f64::EPSILON,
            "grid step moved: {step:e}"
        );
        // The measured worst honest cross-vendor divergence must stay well under one
        // step, or exact-equality settlement stops being merely wrong and starts
        // being wrong in a way a tolerance cannot rescue.
        let worst_measured_divergence = 1.341e-7;
        assert!(
            worst_measured_divergence * 100.0 < step,
            "grid step {step:e} is no longer 100x above measured divergence"
        );
    }

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

    // --- Tier-1 remediation ratchets (each fails on the pre-fix code) ---

    /// H3: add/sub saturate instead of debug-panic / release-wrap on overflow.
    #[test]
    fn add_sub_saturate_not_panic_or_wrap() {
        assert_eq!(
            Q16::from_raw(i64::MAX).add(Q16::from_raw(1)).raw(),
            i64::MAX
        );
        assert_eq!(
            Q16::from_raw(i64::MIN).sub(Q16::from_raw(1)).raw(),
            i64::MIN
        );
        // a saturated value stays put under further same-direction ops (idempotent ceiling).
        assert_eq!(
            Q16::from_raw(i64::MAX).add(Q16::from_raw(i64::MAX)).raw(),
            i64::MAX
        );
    }

    /// L2: the mul post-shift down-cast saturates rather than sign-flipping.
    #[test]
    fn mul_saturates_on_extreme_not_sign_flip() {
        let r = Q16::from_raw(i64::MAX).mul(Q16::from_raw(i64::MAX)).raw();
        assert_eq!(
            r,
            i64::MAX,
            "extreme mul must saturate positive, not wrap negative"
        );
        let mixed = Q16::from_raw(i64::MAX).mul(Q16::from_raw(i64::MIN)).raw();
        assert_eq!(
            mixed,
            i64::MIN,
            "extreme opposite-sign mul saturates negative"
        );
    }

    /// M2: div by zero saturates by sign in every build profile (no panic).
    #[test]
    fn div_by_zero_saturates_by_sign() {
        assert_eq!(Q16::ONE.div(Q16::ZERO).raw(), i64::MAX);
        assert_eq!(Q16::from_f32(-1.0).div(Q16::ZERO).raw(), i64::MIN);
        // a normal divide is unaffected.
        assert_eq!(Q16::ONE.div(Q16::from_f32(2.0)), Q16::from_f32(0.5));
    }

    /// L1: non-finite floats map deterministically to ZERO, not 0/±i64::MAX poison.
    #[test]
    fn from_f32_maps_non_finite_to_zero() {
        assert_eq!(Q16::from_f32(f32::NAN), Q16::ZERO);
        assert_eq!(Q16::from_f32(f32::INFINITY), Q16::ZERO);
        assert_eq!(Q16::from_f32(f32::NEG_INFINITY), Q16::ZERO);
        // finite values are unchanged (golden-preserving).
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
