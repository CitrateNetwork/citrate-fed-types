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
    /// `2 * trim >= bucket_count`, so the kept band would be empty.
    TrimBudgetTooLarge {
        trim: usize,
        bucket_count: usize,
    },
    NoBuckets,
}

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
pub fn bucket_of(seed: &[u8], node_id: &str, bucket_count: usize) -> usize {
    let mut h = Sha256::new();
    h.update(seed);
    h.update([0u8]); // domain separator between seed and id
    h.update(node_id.as_bytes());
    let d = h.finalize();
    let v = u64::from_le_bytes(d[0..8].try_into().expect("sha256 is 32 bytes"));
    (v % bucket_count as u64) as usize
}

/// Coordinate-wise trimmed mean of one coordinate across submissions, in Q16. Sort by the
/// strict total order `(value, source_index)`, drop the lowest/highest `trim`, average the
/// kept band with deterministic integer division.
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
    let first = grads.first().ok_or(AggregateError::Empty)?;
    let dim = first.dim();
    for g in grads {
        if g.dim() != dim {
            return Err(AggregateError::DimensionMismatch {
                expected: dim,
                found: g.dim(),
            });
        }
    }

    let mut buckets: Vec<Vec<&PseudoGradient>> = vec![Vec::new(); bucket_count];
    for g in grads {
        buckets[bucket_of(seed, &g.node_id, bucket_count)].push(g);
    }
    let bucket_means: Vec<Vec<Q16>> = buckets
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| bucket_mean(b, dim))
        .collect();

    if 2 * trim >= bucket_means.len() {
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

    let digest = digest_of(&aggregate);
    Ok(AggregateResult { aggregate, digest })
}

/// `H(raw Q16 little-endian bytes)` of an aggregate vector — the deterministic commitment
/// an auditor recomputes (and the on-chain challenge anchors). No floats; the raw `i64`s
/// are the canonical bytes.
pub fn digest_of(v: &[Q16]) -> String {
    let mut h = Sha256::new();
    for q in v {
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

    // PARITY ANCHOR — must reproduce the `nat-aggregate` frozen golden bit-for-bit. If
    // this differs, the extraction drifted from the source of truth (the whole point of
    // the boundary is that it cannot). Inputs identical to nat-aggregate::frozen_aggregate_digest.
    #[test]
    fn frozen_aggregate_digest_matches_nat() {
        let g = vec![
            pg("alpha", &[1.0, -2.0, 3.5]),
            pg("beta", &[2.0, -1.0, 3.0]),
            pg("gamma", &[1.5, -1.5, 3.25]),
            pg("delta", &[1.75, -1.25, 3.1]),
        ];
        let r = aggregate(&g, 1, 64, b"frozen-seed-v1").expect("aggregate");
        assert_eq!(
            r.digest, "e79c5a6381c2e761f264d1c64dfdf12016c08ca3494ee909736ec84d00aa59a1",
            "fed-types aggregate digest diverged from nat-aggregate — extraction drifted"
        );
    }
}
