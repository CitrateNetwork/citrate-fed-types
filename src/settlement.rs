//! Settlement / patronage math — the bit-exact replica of the on-chain
//! `PatronageLedger.recordContribution` unit calculation (g4-settlement).
//!
//! Faithful extraction of `nat-federated::seam::SettlementRow`. The chain mints patronage
//! units `units = (compute·wCompute/1e4)·(qualityBps·wData/1e4)/1e4`; this is the replica
//! both sides reconcile against. The `data_quality_bps` round-to-nearest is load-bearing:
//! floor would map `0.9 → 8999`, a 1-bps drift that desyncs the Rust seam from Solidity.
//! `patronage_units_match_onchain_ledger` reproduces the `FederatedSettlement.t.sol`
//! golden cases (`4000 / 1800 / 500`) — the proof this extraction did not drift.

use crate::fixed::{Q16Error, Q16};

/// Basis-points scale (`1.0 == 10_000 bps`), matching the Solidity ledger.
pub const BPS_SCALE: u32 = 10_000;

/// Why a [`SettlementRow`] could not be constructed from untrusted factors (finding
/// FT-B-007/FT-B-008). The infallible [`SettlementRow::new`] stays for internal, already-valid
/// callers; the money-path ingest uses [`SettlementRow::try_new`] so an invalid row is refused
/// at birth rather than silently corrected by one accessor and not another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettlementError {
    /// `compute_metered` is negative — negative work is not billable.
    NegativeCompute,
    /// `data_quality` is outside `[0, 1]` — an honesty factor cannot exceed unity nor go below zero.
    QualityOutOfRange,
    /// A factor was a non-finite / saturating `f32`.
    BadFactor(Q16Error),
}

impl std::fmt::Display for SettlementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettlementError::NegativeCompute => {
                write!(f, "settlement: compute_metered is negative")
            }
            SettlementError::QualityOutOfRange => {
                write!(f, "settlement: data_quality outside [0,1]")
            }
            SettlementError::BadFactor(e) => write!(f, "settlement: {e}"),
        }
    }
}
impl std::error::Error for SettlementError {}

/// A single unified settlement record carrying the **two factors explicitly** —
/// `compute_metered` and `data_quality` — so the co-op ledger computes units itself and
/// the honesty factor is never pre-collapsed. Field shape mirrors the on-chain
/// `PatronageLedger.recordContribution(roundId, member, computeMetered, dataQualityBps)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementRow {
    pub node_id: String,
    pub compute_metered: Q16,
    pub data_quality: Q16,
    /// Optional zone discriminant the work targeted (NAT `ZoneId as u8`); `None` for dense.
    pub zone_tag: Option<u8>,
    pub trace_hash: String,
}

/// The exact argument shape of the on-chain `recordContribution`: `computeMetered` as an
/// integer unit count and `dataQualityBps` as basis points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerRecord {
    pub compute_metered: u128,
    pub data_quality_bps: u16,
}

impl SettlementRow {
    pub fn new(
        node_id: impl Into<String>,
        compute_metered: Q16,
        data_quality: Q16,
        zone_tag: Option<u8>,
        trace_hash: impl Into<String>,
    ) -> Self {
        SettlementRow {
            node_id: node_id.into(),
            compute_metered,
            data_quality,
            zone_tag,
            trace_hash: trace_hash.into(),
        }
    }

    /// Parse-don't-validate constructor for untrusted factors (finding FT-B-007/FT-B-008):
    /// rejects a negative `compute_metered` and a `data_quality` outside `[0,1]` at
    /// construction, so a row that reaches the accessors is already valid and the two
    /// settlement seams (`reward_weight` vs `patronage_units`) cannot disagree on it. The
    /// infallible [`SettlementRow::new`] remains for callers holding already-validated Q16s.
    pub fn try_new(
        node_id: impl Into<String>,
        compute_metered: Q16,
        data_quality: Q16,
        zone_tag: Option<u8>,
        trace_hash: impl Into<String>,
    ) -> Result<Self, SettlementError> {
        if compute_metered.raw() < 0 {
            return Err(SettlementError::NegativeCompute);
        }
        if data_quality.raw() < 0 || data_quality.raw() > Q16::ONE.raw() {
            return Err(SettlementError::QualityOutOfRange);
        }
        Ok(SettlementRow::new(
            node_id,
            compute_metered,
            data_quality,
            zone_tag,
            trace_hash,
        ))
    }

    /// The collapsed reward weight `compute × data_quality`.
    ///
    /// Fail-closed like its sibling accessors (`compute_units`, `data_quality_bps`): the
    /// compute factor is floored at 0 and the quality factor is clamped to `[0,1]` *before*
    /// the multiply (finding FT-B-002). Without these guards a bare `Q16::mul` turns two
    /// negatives into a positive reward and lets `quality>1` inflate the weight — the legacy
    /// `seam::Settlement` payout path consumes this value directly.
    pub fn reward_weight(&self) -> Q16 {
        let compute = Q16::from_raw(self.compute_metered.raw().max(0));
        let quality = Q16::from_raw(self.data_quality.raw().clamp(0, Q16::ONE.raw()));
        compute.mul(quality)
    }

    /// `data_quality` (Q16 in [0,1]) as the on-chain `dataQualityBps` (u16 in [0,10000]).
    /// Round-to-nearest (`0.9 → 9000`, not `8999`); clamped fail-safe so a score outside
    /// [0,1] cannot inflate units.
    pub fn data_quality_bps(&self) -> u16 {
        let one = Q16::ONE.raw() as i128;
        let raw = self.data_quality.raw().clamp(0, Q16::ONE.raw()) as i128;
        ((raw * BPS_SCALE as i128 + one / 2) / one) as u16
    }

    /// `compute_metered` (Q16) as the on-chain integer `computeMetered` unit count.
    ///
    /// Derives the truncation from the one exported grid scale (`Q16::SCALE`) rather than a
    /// second hardcoded `>> 16` (finding FT-B-005). For a non-negative raw, `/ Q16::SCALE`
    /// equals the old arithmetic shift bit-for-bit, so the golden is preserved.
    pub fn compute_units(&self) -> u128 {
        (self.compute_metered.raw().max(0) / Q16::SCALE) as u128
    }

    /// The on-chain `recordContribution` call shape for this row.
    pub fn to_ledger_record(&self) -> LedgerRecord {
        LedgerRecord {
            compute_metered: self.compute_units(),
            data_quality_bps: self.data_quality_bps(),
        }
    }

    /// Bit-exact replica of `PatronageLedger.recordContribution`'s unit math:
    /// `p = (compute·wCompute/1e4)·(qualityBps·wData/1e4)/1e4`.
    pub fn patronage_units(&self, w_compute_bps: u32, w_data_bps: u32) -> u128 {
        let weighted_compute = self.compute_units() * w_compute_bps as u128 / BPS_SCALE as u128;
        let weighted_quality =
            self.data_quality_bps() as u128 * w_data_bps as u128 / BPS_SCALE as u128;
        weighted_compute * weighted_quality / BPS_SCALE as u128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(compute: f32, quality: f32) -> SettlementRow {
        SettlementRow::new(
            "n",
            Q16::from_f32(compute),
            Q16::from_f32(quality),
            Some(3),
            "t",
        )
    }

    #[test]
    fn data_quality_bps_rounds_to_nearest() {
        assert_eq!(row(1.0, 0.9).data_quality_bps(), 9000); // not 8999
        assert_eq!(row(1.0, 0.5).data_quality_bps(), 5000);
        assert_eq!(row(1.0, 1.0).data_quality_bps(), 10000);
    }

    /// FT-B-002 tripwire: `reward_weight()` — consumed by the legacy `seam::Settlement`
    /// payout path — must apply the same [0,1] quality clamp and non-negativity guard its
    /// sibling accessors (`compute_units`, `data_quality_bps`) already apply. Without them a
    /// bare `Q16::mul` turns two negatives into a positive reward and lets `quality>1` inflate.
    #[test]
    fn reward_weight_is_fail_closed_like_its_siblings() {
        // (-4000) * (-1.0) must NOT become +4000: negative work is worth zero, not full pay.
        let bad = row(-4000.0, -1.0);
        assert_eq!(
            bad.reward_weight().raw(),
            0,
            "two negatives must not multiply into a positive reward"
        );
        // quality clamped to [0,1] — it cannot inflate the weight above compute * 1.0.
        let inflated = row(10.0, 1000.0);
        assert!(
            inflated.reward_weight().raw() <= Q16::from_f32(10.0).raw(),
            "out-of-range quality must be clamped, not passed through as a 1000x multiplier"
        );
        // sign-consistency with the unified/on-chain replica path.
        assert_eq!(
            bad.reward_weight().raw() > 0,
            bad.patronage_units(BPS_SCALE, BPS_SCALE) > 0,
            "the two settlement seams must agree on the sign of the reward"
        );
    }

    /// FT-B-007/FT-B-008: `try_new` refuses an invalid row at construction — the precondition
    /// that a bare `new()` lets slip through to exactly one of four accessors.
    #[test]
    fn try_new_rejects_invalid_rows_at_construction() {
        assert_eq!(
            SettlementRow::try_new("n", Q16::from_f32(-1.0), Q16::from_f32(1.0), None, "t"),
            Err(SettlementError::NegativeCompute)
        );
        assert_eq!(
            SettlementRow::try_new("n", Q16::from_f32(10.0), Q16::from_f32(1000.0), None, "t"),
            Err(SettlementError::QualityOutOfRange)
        );
        assert_eq!(
            SettlementRow::try_new("n", Q16::from_f32(10.0), Q16::from_f32(-0.5), None, "t"),
            Err(SettlementError::QualityOutOfRange)
        );
        // a well-formed row constructs, and both seams then agree on its sign.
        let ok = SettlementRow::try_new("n", Q16::from_f32(10.0), Q16::from_f32(0.5), None, "t")
            .expect("valid row");
        assert!(ok.reward_weight().raw() > 0);
        assert!(ok.patronage_units(BPS_SCALE, BPS_SCALE) > 0);
    }

    /// FT-B-005: `compute_units` truncates via the exported `Q16::SCALE`, and a Q16 `1.0`
    /// must map to exactly one unit — the ratchet that pins the second copy of the scale to
    /// the first. If `FRAC_BITS` ever moves, both this and the grid-step test fail together.
    #[test]
    fn compute_units_derives_from_the_exported_scale() {
        assert_eq!(row(1.0, 1.0).compute_units(), 1);
        // the derivation equals the historical `>> 16` for every non-negative raw.
        for raw in [65_535_i64, 65_536, 131_071, i64::MAX] {
            let r = SettlementRow::new("n", Q16::from_raw(raw), Q16::ONE, None, "t");
            assert_eq!(r.compute_units(), (raw.max(0) >> 16) as u128);
        }
    }

    // PARITY ANCHOR — reproduces FederatedSettlement.t.sol::test_coordinator_settles_round
    // (4000 / 1800 / 500), via nat-federated::seam::patronage_units_match_onchain_ledger.
    #[test]
    fn patronage_units_match_onchain_ledger() {
        let cases = [
            (4000.0_f32, 1.0_f32, 4000_u128),
            (2000.0, 0.9, 1800),
            (1000.0, 0.5, 500),
        ];
        for (compute, quality, expected) in cases {
            assert_eq!(
                row(compute, quality).patronage_units(BPS_SCALE, BPS_SCALE),
                expected
            );
        }
    }

    /// L3: lock the three-step integer-division *order* under NON-unit weights, where two of
    /// the divisions actually floor (the default-weight anchor above leaves them exact). The
    /// expected value is hand-derived from `PatronageLedger.recordContribution`
    /// (`citrate-coop/contracts/src/PatronageLedger.sol:96-98`) in the contract's exact order:
    ///   compute_units      = floor(1234)                 = 1234
    ///   data_quality_bps   = round(0.9 * 10_000)         = 9000
    ///   weighted_compute   = 1234 * 6500 / 10_000        = 802   (floor of 802.1)
    ///   weighted_quality   = 9000 * 4200 / 10_000        = 3780
    ///   units              = 802 * 3780 / 10_000         = 303   (floor of 303.156)
    /// If the operation order or flooring ever drifts from the contract, this fails.
    #[test]
    fn patronage_units_match_onchain_under_non_unit_weights() {
        assert_eq!(row(1234.0, 0.9).patronage_units(6500, 4200), 303);
    }
}
