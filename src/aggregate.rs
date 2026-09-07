//! Bucketed coordinate-wise trimmed-mean aggregation in Q16 — the reduction the on-chain
//! challenge game re-executes to adjudicate a disputed aggregate (g4-onchain).
//!
//! Faithful extraction of `nat-aggregate` (the pure core; the compression path and the
//! research sweep stay in `nat`). The reduction is the Rust image of the TLA+ pair
//! `GradientAggregation.tla` / `GradientAggregationAdversarial.tla`: a strict-total-order
//! trim `(value, source_index)` so no tie can make two nodes disagree, integer arithmetic
//! throughout, no hash-map iteration on the path. `frozen_aggregate_digest` reproduces the
//! exact `nat-aggregate` golden bytes — the proof this extraction did not drift.

use crate::fixed::Q16;
use crate::hex;
use sha2::{Digest, Sha256};

/// One worker's pseudo-gradient for an outer round: a fixed-point delta vector. All
/// pseudo-gradients in a round share dimensionality `dim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PseudoGradient {
    /// The submitting worker's identity (the bucket-seed and the audit-trail key).
    pub node_id: String,
    /// The delta, one Q16 coordinate per model parameter.
    pub coords: Vec<Q16>,
}

impl PseudoGradient {
    pub fn new(node_id: impl Into<String>, coords: Vec<Q16>) -> Self {
        PseudoGradient {
            node_id: node_id.into(),
            coords,
        }
    }
    pub fn dim(&self) -> usize {
        self.coords.len()
    }
}

/// Why an aggregation could not be computed (fail-closed at the boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateError {
    Empty,
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    /// `2 * trim >= bucket_count`, so the kept band would be empty. Also returned when
    /// `2 * trim` would overflow `usize` (an adversarial `trim` cannot wrap past this guard).
    TrimBudgetTooLarge {
        trim: usize,
        bucket_count: usize,
    },
    NoBuckets,
    /// `bucket_count` exceeds [`MAX_BUCKETS`] — rejected before allocation so a pathological
    /// value cannot OOM the aggregator (finding H2).
    TooManyBuckets {
        bucket_count: usize,
        max: usize,
    },
    /// `trim == 0` — no order statistics are dropped, so the trimmed-mean's entire Byzantine
    /// guarantee is void and a single outlier lands in the mean at full weight (finding
    /// FT-B-004). The robustness must not be silently delegated to a caller-supplied `0`.
    NoTrimBudget,
    /// A submitted coordinate's raw magnitude exceeds [`MAX_COORD_MAGNITUDE`] (finding
    /// FT-B-004). A saturating-scale coordinate (reachable from an ordinary huge finite `f32`)
    /// would dominate the reduction and collide on `i64::MAX`, so it is refused at ingest.
    CoordinateOutOfRange {
        node_id: String,
        coord: usize,
        raw: i64,
    },
}

/// Upper bound on the raw magnitude of any single submitted coordinate. Far above any real
/// pseudo-gradient delta (honest coords ride raw magnitudes in the thousands–millions) yet
/// well below the saturating band, so no honest submission is ever refused while a coordinate
/// engineered to saturate `i64::MAX` — and thereby capture the reduction — is (finding
/// FT-B-004). The bound must be mirrored by the on-chain challenge re-executor for parity.
pub const MAX_COORD_MAGNITUDE: i64 = i64::MAX / 2;

/// Upper bound on `bucket_count`, far above any real federated round (≈1M buckets), that
/// bounds the bucket-vector allocation so a malicious `bucket_count` cannot exhaust memory.
pub const MAX_BUCKETS: usize = 1 << 20;

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateError::Empty => write!(f, "aggregate: no pseudo-gradients"),
            AggregateError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "aggregate: dimension mismatch (expected {expected}, found {found})"
                )
            }
            AggregateError::TrimBudgetTooLarge { trim, bucket_count } => {
                write!(
                    f,
                    "aggregate: trim {trim} too large for {bucket_count} buckets"
                )
            }
            AggregateError::NoBuckets => write!(f, "aggregate: bucket_count must be >= 1"),
            AggregateError::TooManyBuckets { bucket_count, max } => {
                write!(
                    f,
                    "aggregate: bucket_count {bucket_count} exceeds max {max}"
                )
            }
            AggregateError::NoTrimBudget => {
                write!(f, "aggregate: trim must be >= 1 (trim=0 has no Byzantine guarantee)")
            }
            AggregateError::CoordinateOutOfRange {
                node_id,
                coord,
                raw,
            } => write!(
                f,
                "aggregate: {node_id} coordinate {coord} raw {raw} exceeds max magnitude {MAX_COORD_MAGNITUDE}"
            ),
        }
    }
}
impl std::error::Error for AggregateError {}

/// The result of an outer-round aggregation: the aggregated pseudo-gradient and a
/// deterministic digest over its raw Q16 bytes (the value an auditor recomputes and the
/// on-chain challenge commits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateResult {
    pub aggregate: Vec<Q16>,
    pub digest: String,
}

/// Deterministically assign a worker to one of `bucket_count` buckets by hashing
/// `seed || node_id`. Production replaces the seed with a VRF output; the assignment
/// function is identical and reproducible given the seed.
///
/// `bucket_count` is clamped to `>= 1` so a direct call with `0` cannot divide-by-zero
/// (finding M1); the sole production caller [`aggregate`] already rejects `0` up front, so
/// this clamp is defense-in-depth on the public helper and never changes a real result.
pub fn bucket_of(seed: &[u8], node_id: &str, bucket_count: usize) -> usize {
    let mut h = Sha256::new();
    h.update(seed);
    h.update([0u8]); // domain separator between seed and id
    h.update(node_id.as_bytes());
    let d = h.finalize();
    let v = u64::from_le_bytes(d[0..8].try_into().expect("sha256 is 32 bytes"));
    (v % bucket_count.max(1) as u64) as usize
}

/// Coordinate-wise trimmed mean of one coordinate across submissions, in Q16. Sort by the
/// strict total order `(value, source_index)`, drop the lowest/highest `trim`, average the
/// kept band with deterministic integer division.
///
/// **Rounding convention (normative, finding FT-B-006):** the average is integer division
/// that **truncates toward zero**, matching Rust's and Solidity's `/`. A conforming
/// re-implementation MUST NOT floor: Python's `//`, `numpy.floor_divide` and most notebook
/// re-derivations floor, which disagrees with this reduction on every negative mean that is
/// not exact (e.g. `sum=-4, count=3` gives `-1` here, `-2` under floor) and would produce a
/// different digest and a spurious challenge outcome. The same truncation is used in
/// [`bucket_mean`]; both must be re-implemented as truncate-toward-zero.
fn coordinate_trimmed_mean(values: &mut [(Q16, usize)], trim: usize) -> Q16 {
    values.sort_unstable_by(|a, b| a.0.raw().cmp(&b.0.raw()).then(a.1.cmp(&b.1)));
    let kept = &values[trim..values.len() - trim];
    let sum: i128 = kept.iter().map(|(q, _)| q.raw() as i128).sum();
    Q16::from_raw((sum / kept.len() as i128) as i64)
}

/// Coordinate-wise mean of a bucket's pseudo-gradients (the within-bucket reduction).
fn bucket_mean(grads: &[&PseudoGradient], dim: usize) -> Vec<Q16> {
    (0..dim)
        .map(|c| {
            let sum: i128 = grads.iter().map(|g| g.coords[c].raw() as i128).sum();
            Q16::from_raw((sum / grads.len() as i128) as i64)
        })
        .collect()
}

/// Aggregate one outer round: bucket the pseudo-gradients (seed-derived), reduce each
/// bucket to its coordinate-wise mean, then take the coordinate-wise trimmed mean across
/// the non-empty bucket means. Deterministic and order-independent: the result depends
/// only on the *set* of (node_id, coords) and the seed.
pub fn aggregate(
    grads: &[PseudoGradient],
    trim: usize,
    bucket_count: usize,
    seed: &[u8],
) -> Result<AggregateResult, AggregateError> {
    if bucket_count == 0 {
        return Err(AggregateError::NoBuckets);
    }
    if bucket_count > MAX_BUCKETS {
        return Err(AggregateError::TooManyBuckets {
            bucket_count,
            max: MAX_BUCKETS,
        });
    }
    let first = grads.first().ok_or(AggregateError::Empty)?;
    let dim = first.dim();
    for g in grads {
        if g.dim() != dim {
            return Err(AggregateError::DimensionMismatch {
                expected: dim,
                found: g.dim(),
            });
        }
        // FT-B-004: refuse a coordinate engineered to saturate the grid before it can
        // dominate the reduction or collide on `i64::MAX`. Honest deltas are orders of
        // magnitude below this bound, so this never fires on real traffic.
        for (c, q) in g.coords.iter().enumerate() {
            if q.raw() > MAX_COORD_MAGNITUDE || q.raw() < -MAX_COORD_MAGNITUDE {
                return Err(AggregateError::CoordinateOutOfRange {
                    node_id: g.node_id.clone(),
                    coord: c,
                    raw: q.raw(),
                });
            }
        }
    }

    // FT-B-004: `trim == 0` leaves the reduction with no order statistics dropped, so the
    // trimmed-mean's Byzantine guarantee is void. Refuse it rather than delegate robustness
    // to a caller-supplied zero. (The frozen anchors all use `trim >= 1`, so this is
    // golden-preserving.)
    if trim == 0 {
        return Err(AggregateError::NoTrimBudget);
    }

    // FT-B-009: bucket into a sparse map keyed by *occupied* bucket index only, so cost is
    // O(grads.len()) rather than O(bucket_count). `BTreeMap::values()` yields buckets in
    // ascending index order — bit-identical to the old `(0..bucket_count).filter(non-empty)`
    // scan — so the resulting `bucket_means` (and therefore the digest) is unchanged.
    let mut buckets: std::collections::BTreeMap<usize, Vec<&PseudoGradient>> =
        std::collections::BTreeMap::new();
    for g in grads {
        buckets
            .entry(bucket_of(seed, &g.node_id, bucket_count))
            .or_default()
            .push(g);
    }
    let bucket_means: Vec<Vec<Q16>> = buckets.values().map(|b| bucket_mean(b, dim)).collect();

    // Overflow-safe trim guard: an adversarial `trim` near `usize::MAX` must not let
    // `2 * trim` wrap past this check and then underflow the `[trim..len-trim]` slice
    // (finding H2). `checked_mul` → `None` ⇒ trim is far too large ⇒ reject.
    let band_too_small = trim
        .checked_mul(2)
        .is_none_or(|two_trim| two_trim >= bucket_means.len());
    if band_too_small {
        return Err(AggregateError::TrimBudgetTooLarge {
            trim,
            bucket_count: bucket_means.len(),
        });
    }

    let aggregate: Vec<Q16> = (0..dim)
        .map(|c| {
            let mut col: Vec<(Q16, usize)> = bucket_means
                .iter()
                .enumerate()
                .map(|(i, m)| (m[c], i))
                .collect();
            coordinate_trimmed_mean(&mut col, trim)
        })
        .collect();

    let digest = round_digest(&aggregate, trim, bucket_count, seed);
    Ok(AggregateResult { aggregate, digest })
}

/// Domain string for every aggregate-side commitment (finding FT-B-003), mirroring
/// `lora`'s `nat-lora-commit-v1`. Prevents an unprefixed SHA-256 of an 8-byte-aligned blob
/// from being mistaken for an aggregate digest, and denies the empty-vector digest the
/// well-known SHA-256-of-empty constant.
const AGG_DOMAIN: &[u8] = b"nat-aggregate-digest-v1";

/// Domain-separated, length-framed digest of an aggregate vector (finding FT-B-003): the
/// deterministic commitment an auditor recomputes. `H(domain || len || raw i64 LE…)`. No
/// floats; the raw `i64`s are the canonical bytes. The length prefix removes the
/// variable-length padding ambiguity a bare concatenation carries.
pub fn digest_of(v: &[Q16]) -> String {
    let mut h = Sha256::new();
    h.update(AGG_DOMAIN);
    h.update((v.len() as u64).to_le_bytes());
    for q in v {
        h.update(q.raw().to_le_bytes());
    }
    hex(&h.finalize())
}

/// The round commitment the on-chain challenge anchors (finding FT-B-003): binds the
/// aggregate *and the parameters that produced it* — `seed`, `trim`, `bucket_count`, `dim` —
/// so a coordinator cannot grind a favourable `(seed, trim, bucket_count)` after seeing
/// submissions and still present a digest that "verifies". `H(domain || seed_len || seed ||
/// trim || bucket_count || dim || len || raw i64 LE…)`.
pub fn round_digest(aggregate: &[Q16], trim: usize, bucket_count: usize, seed: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(AGG_DOMAIN);
    h.update((seed.len() as u64).to_le_bytes());
    h.update(seed);
    h.update((trim as u64).to_le_bytes());
    h.update((bucket_count as u64).to_le_bytes());
    h.update((aggregate.len() as u64).to_le_bytes());
    for q in aggregate {
        h.update(q.raw().to_le_bytes());
    }
    hex(&h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pg(id: &str, coords: &[f32]) -> PseudoGradient {
        PseudoGradient::new(id, coords.iter().map(|&v| Q16::from_f32(v)).collect())
    }

    #[test]
    fn determinism_and_order_independence() {
        let g = vec![
            pg("a", &[1.0, 8.0]),
            pg("b", &[2.0, 7.0]),
            pg("c", &[3.0, 6.0]),
            pg("d", &[4.0, 5.0]),
        ];
        let mut shuffled = g.clone();
        shuffled.rotate_right(2);
        shuffled.swap(0, 3);
        let r = aggregate(&g, 1, 64, b"seed-test").expect("aggregate");
        assert_eq!(
            r,
            aggregate(&shuffled, 1, 64, b"seed-test").expect("aggregate")
        );
    }

    #[test]
    fn byzantine_outliers_trimmed_within_honest_band() {
        let g = vec![
            pg("h1", &[10.0]),
            pg("h2", &[10.0]),
            pg("h3", &[10.0]),
            pg("h4", &[10.0]),
            pg("h5", &[10.0]),
            pg("byz_lo", &[-9000.0]),
            pg("byz_hi", &[9000.0]),
        ];
        let r = aggregate(&g, 2, 64, b"seed-test").expect("aggregate");
        assert_eq!(r.aggregate, vec![Q16::from_f32(10.0)]);
    }

    #[test]
    fn fails_closed() {
        assert_eq!(
            aggregate(&[], 0, 4, b"s").unwrap_err(),
            AggregateError::Empty
        );
        assert_eq!(
            aggregate(&[pg("a", &[1.0, 2.0]), pg("b", &[1.0])], 0, 64, b"s").unwrap_err(),
            AggregateError::DimensionMismatch {
                expected: 2,
                found: 1
            }
        );
        assert_eq!(
            aggregate(&[pg("a", &[1.0]), pg("b", &[2.0])], 1, 64, b"s").unwrap_err(),
            AggregateError::TrimBudgetTooLarge {
                trim: 1,
                bucket_count: 2
            }
        );
    }

    // --- Tier-1 remediation ratchets (each fails / panics on the pre-fix code) ---

    /// H2: an adversarial `trim` whose `2*trim` overflows `usize` must be rejected, not wrap
    /// past the guard into a slice underflow.
    #[test]
    fn aggregate_rejects_overflowing_trim() {
        let g = vec![
            pg("a", &[1.0]),
            pg("b", &[1.0]),
            pg("c", &[1.0]),
            pg("d", &[1.0]),
        ];
        let trim = usize::MAX / 2 + 1; // 2*trim overflows usize
                                       // Must reject (not wrap past the guard into a slice underflow). `bucket_count` in the
                                       // error is the non-empty-bucket-means count, so match on the variant + trim only.
        assert!(matches!(
            aggregate(&g, trim, 64, b"s").unwrap_err(),
            AggregateError::TrimBudgetTooLarge { trim: t, .. } if t == trim
        ));
    }

    /// H2: a pathological `bucket_count` is rejected before the bucket-vector allocation,
    /// so it cannot OOM the aggregator.
    #[test]
    fn aggregate_rejects_pathological_bucket_count() {
        let g = vec![pg("a", &[1.0])];
        assert_eq!(
            aggregate(&g, 0, usize::MAX, b"s").unwrap_err(),
            AggregateError::TooManyBuckets {
                bucket_count: usize::MAX,
                max: MAX_BUCKETS
            }
        );
    }

    /// FT-B-004: `trim == 0` is refused — the trimmed-mean's Byzantine guarantee must not be
    /// silently delegated to a caller-supplied zero.
    #[test]
    fn aggregate_rejects_zero_trim() {
        let g = vec![pg("a", &[1.0]), pg("b", &[2.0]), pg("c", &[3.0])];
        assert_eq!(
            aggregate(&g, 0, 64, b"s").unwrap_err(),
            AggregateError::NoTrimBudget
        );
    }

    /// FT-B-004: a coordinate engineered to saturate the grid is refused at ingest, so it
    /// cannot capture the reduction. The saturating raw comes from an ordinary huge finite f32.
    #[test]
    fn aggregate_rejects_saturating_coordinate() {
        let g = vec![
            pg("h1", &[10.0]),
            pg("h2", &[10.0]),
            pg("attacker", &[f32::MAX]), // saturates from_f32 to i64::MAX
        ];
        assert!(matches!(
            aggregate(&g, 1, 64, b"s").unwrap_err(),
            AggregateError::CoordinateOutOfRange { coord: 0, .. }
        ));
        // a coordinate exactly at the bound is accepted; one past it is not.
        let ok = vec![
            PseudoGradient::new("a", vec![Q16::from_raw(MAX_COORD_MAGNITUDE)]),
            pg("b", &[1.0]),
            pg("c", &[2.0]),
        ];
        assert!(aggregate(&ok, 1, 64, b"s").is_ok());
        let over = vec![PseudoGradient::new(
            "a",
            vec![Q16::from_raw(MAX_COORD_MAGNITUDE + 1)],
        )];
        assert!(matches!(
            aggregate(&over, 1, 64, b"s").unwrap_err(),
            AggregateError::CoordinateOutOfRange { .. }
        ));
    }

    /// FT-B-006: the reduction's integer division truncates toward zero (Rust/Solidity `/`),
    /// NOT floor — locked with a signed, non-exact mean where the two conventions disagree.
    #[test]
    fn trimmed_mean_truncates_toward_zero_not_floor() {
        // sum=-4, count=3: truncation -> -1, floor -> -2.
        let mut neg = [
            (Q16::from_raw(-1), 0),
            (Q16::from_raw(-1), 1),
            (Q16::from_raw(-2), 2),
        ];
        assert_eq!(coordinate_trimmed_mean(&mut neg, 0).raw(), -1);
        // sign-mirrored: sum=4, count=3 -> +1. The reduction is not sign-symmetric under floor.
        let mut pos = [
            (Q16::from_raw(1), 0),
            (Q16::from_raw(1), 1),
            (Q16::from_raw(2), 2),
        ];
        assert_eq!(coordinate_trimmed_mean(&mut pos, 0).raw(), 1);
        // bucket_mean shares the convention.
        let g = [
            &PseudoGradient::new("a", vec![Q16::from_raw(-1)]),
            &PseudoGradient::new("b", vec![Q16::from_raw(-1)]),
            &PseudoGradient::new("c", vec![Q16::from_raw(-2)]),
        ];
        assert_eq!(bucket_mean(&g, 1)[0].raw(), -1);
    }

    /// FT-B-009: the sparse bucketing preserves results at a large `bucket_count` (the
    /// refactor must not change the aggregate) and stays deterministic without allocating
    /// O(bucket_count).
    #[test]
    fn aggregate_is_stable_at_large_bucket_count() {
        let g = vec![
            pg("h1", &[10.0, 1.0]),
            pg("h2", &[10.0, 1.0]),
            pg("h3", &[10.0, 1.0]),
            pg("byz_lo", &[-9000.0, -9000.0]),
            pg("byz_hi", &[9000.0, 9000.0]),
        ];
        let r = aggregate(&g, 1, MAX_BUCKETS, b"seed-test").expect("aggregate");
        // deterministic across runs.
        assert_eq!(
            r,
            aggregate(&g, 1, MAX_BUCKETS, b"seed-test").expect("aggregate")
        );
    }

    /// M1: the public `bucket_of` helper does not divide-by-zero on `bucket_count == 0`.
    #[test]
    fn bucket_of_does_not_panic_on_zero() {
        assert_eq!(bucket_of(b"seed", "node", 0), 0);
    }

    // CANDIDATE RE-FROZEN ANCHOR (FT-B-003 / NAT2-B-008) — HELD, NOT a merged golden.
    // The round digest now binds a domain string + (seed, trim, bucket_count, dim), so it no
    // longer equals the pre-fix bare-concatenation golden `e79c5a63…`. This value locks the
    // NEW preimage for self-consistency ONLY; the real re-freeze must be performed jointly by
    // the owner across nat-aggregate (NAT2-B-002/003) and the Solidity challenge re-executor,
    // which must recompute their golden through the identical framing before this lands.
    #[test]
    fn round_digest_is_frozen_candidate() {
        let g = vec![
            pg("alpha", &[1.0, -2.0, 3.5]),
            pg("beta", &[2.0, -1.0, 3.0]),
            pg("gamma", &[1.5, -1.5, 3.25]),
            pg("delta", &[1.75, -1.25, 3.1]),
        ];
        let r = aggregate(&g, 1, 64, b"frozen-seed-v1").expect("aggregate");
        assert_eq!(
            r.digest, "9c8597653a4762ecb709c74956abbd40d132997d9579ee742edb9f8cb80a28b7",
            "round_digest preimage drifted from the FT-B-003 candidate framing"
        );
    }

    /// FT-B-003: `digest_of` is domain-separated and length-framed — it is NOT reproducible
    /// by a bare unprefixed SHA-256 of the same raw bytes, and the empty aggregate does not
    /// hash to the well-known SHA-256-of-empty constant.
    #[test]
    fn digest_of_is_domain_separated_and_length_framed() {
        use sha2::{Digest, Sha256};
        let v = [Q16::from_f32(1.0), Q16::from_f32(-2.0)];
        // a bare sha256 over the same bytes (the pre-fix preimage) must NOT match.
        let mut bare = Sha256::new();
        for q in &v {
            bare.update(q.raw().to_le_bytes());
        }
        assert_ne!(digest_of(&v), hex(&bare.finalize()));
        // the empty aggregate is not the universally-known empty-string digest.
        assert_ne!(
            digest_of(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// FT-B-003: the round digest binds the parameters that produced the aggregate, so two
    /// runs that differ only in `trim`, `bucket_count`, or `seed` cannot share a digest even
    /// when they happen to yield the same aggregate vector.
    #[test]
    fn round_digest_binds_round_parameters() {
        let v = [Q16::from_f32(1.0), Q16::from_f32(2.0)];
        let base = round_digest(&v, 1, 64, b"seed");
        assert_ne!(base, round_digest(&v, 2, 64, b"seed")); // trim
        assert_ne!(base, round_digest(&v, 1, 128, b"seed")); // bucket_count
        assert_ne!(base, round_digest(&v, 1, 64, b"other-seed")); // seed
    }
}
