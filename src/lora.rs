//! LoRA adapter commitment — the Q16-exact digest the chain stores as
//! `LoRAFactory.adapterModelCommitment` and verifies on registration (g4-settlement).
//!
//! Faithful extraction of `nat-lora::commit` (the commitment only — the generator + its
//! meta-training stay in `nat`). Q16-quantized, rank-atom-order-independent (the rank-`K`
//! factorization has no canonical atom order, so the sorted multiset of per-atom
//! `(B column, A row)` blobs is hashed), tamper-detecting. The domain string is preserved
//! verbatim (`nat-lora-commit-v1`) so when `nat-lora` adopts this kernel its frozen golden
//! stays byte-identical — the migration does not move the committed bytes.

use crate::fixed::Q16;
use crate::hex;
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"nat-lora-commit-v1";

/// The committed shape of a generated adapter: a low-rank `ΔW = alpha · (B · A)` over a
/// zone, with `matrix_a` `[rank][dim_in]` and `matrix_b` `[dim_out][rank]`. `zone_tag` is
/// the NAT `ZoneId as u8` discriminant (so the kernel does not depend on the `nat` enum).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraFactors {
    pub zone_tag: u8,
    pub rank: usize,
    pub dim_out: usize,
    pub dim_in: usize,
    pub alpha: f32,
    pub matrix_a: Vec<Vec<f32>>,
    pub matrix_b: Vec<Vec<f32>>,
}

/// Why a [`LoraFactors`] could not be committed: its matrices do not match the declared
/// `rank` / `dim_out` / `dim_in` (finding H1). The chain recomputes this commitment to
/// verify an untrusted registration, so a shape-malformed payload must be *rejected*, not
/// allowed to panic the verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    ShapeMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    /// A weight (`alpha`, `matrix_a[..]` or `matrix_b[..]`) is non-finite (`NaN`/`±∞`).
    /// `from_f32` maps every non-finite value to raw `0`, so committing one would forge a
    /// benign all-zero digest for an adapter that actually serves poison weights (FT-B-001).
    NonFiniteWeight { field: &'static str, index: usize },
    /// A finite weight whose Q16 image saturates the `i64` grid (`|v| ≳ 1.407e14`). Every
    /// saturating magnitude collapses to the same raw, so distinct adapters would share one
    /// committed digest — the commitment stops being binding (FT-B-001).
    WeightOutOfRange { field: &'static str, index: usize },
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoraError::ShapeMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "lora: {field} shape mismatch (expected {expected}, found {found})"
            ),
            LoraError::NonFiniteWeight { field, index } => {
                write!(f, "lora: {field}[{index}] is non-finite (NaN/±inf)")
            }
            LoraError::WeightOutOfRange { field, index } => {
                write!(f, "lora: {field}[{index}] saturates the Q16 grid")
            }
        }
    }
}
impl std::error::Error for LoraError {}

fn q(v: f32) -> [u8; 8] {
    Q16::from_f32(v).raw().to_le_bytes()
}

/// A weight is committed injectively only while it is finite *and* its Q16 image does not
/// saturate the `i64` grid; outside that domain `from_f32` is many-to-one (finding
/// FT-B-001). Reject either case fail-closed. The scale is read from `Q16::ONE.raw()` so
/// this stays tied to the one grid definition rather than hardcoding it.
fn check_weight(field: &'static str, index: usize, v: f32) -> Result<(), LoraError> {
    if !v.is_finite() {
        return Err(LoraError::NonFiniteWeight { field, index });
    }
    let scaled = (v as f64) * (Q16::ONE.raw() as f64);
    if scaled.abs() >= i64::MAX as f64 {
        return Err(LoraError::WeightOutOfRange { field, index });
    }
    Ok(())
}

/// Validate that every committed weight lands injectively on the Q16 grid (FT-B-001), so
/// the digest binds the values it hashes. Runs after [`validate_shape`], so the per-matrix
/// index arithmetic reflects the declared dims.
fn validate_values(a: &LoraFactors) -> Result<(), LoraError> {
    check_weight("alpha", 0, a.alpha)?;
    for (k, row) in a.matrix_a.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            check_weight("matrix_a", k * a.dim_in + i, v)?;
        }
    }
    for (o, row) in a.matrix_b.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            check_weight("matrix_b", o * a.rank + k, v)?;
        }
    }
    Ok(())
}

/// Validate that the matrices match the declared dims, fail-closed (finding H1). `matrix_a`
/// is `[rank][dim_in]`, `matrix_b` is `[dim_out][rank]`.
fn validate_shape(a: &LoraFactors) -> Result<(), LoraError> {
    if a.matrix_a.len() != a.rank {
        return Err(LoraError::ShapeMismatch {
            field: "matrix_a.rows(rank)",
            expected: a.rank,
            found: a.matrix_a.len(),
        });
    }
    for row in &a.matrix_a {
        if row.len() != a.dim_in {
            return Err(LoraError::ShapeMismatch {
                field: "matrix_a.cols(dim_in)",
                expected: a.dim_in,
                found: row.len(),
            });
        }
    }
    if a.matrix_b.len() != a.dim_out {
        return Err(LoraError::ShapeMismatch {
            field: "matrix_b.rows(dim_out)",
            expected: a.dim_out,
            found: a.matrix_b.len(),
        });
    }
    for row in &a.matrix_b {
        if row.len() != a.rank {
            return Err(LoraError::ShapeMismatch {
                field: "matrix_b.cols(rank)",
                expected: a.rank,
                found: row.len(),
            });
        }
    }
    Ok(())
}

/// The Q16-exact, rank-atom-order-independent, tamper-detecting LoRA commitment.
///
/// Fail-closed: returns [`LoraError::ShapeMismatch`] if the matrices do not match the
/// declared dims (finding H1), and [`LoraError::NonFiniteWeight`] /
/// [`LoraError::WeightOutOfRange`] if any weight falls outside the domain on which
/// `from_f32` is injective (finding FT-B-001), so an untrusted registration can neither
/// panic the verifier nor forge a colliding digest. The committed bytes are unchanged for a
/// well-formed adapter — the frozen golden holds.
pub fn lora_commitment(a: &LoraFactors) -> Result<String, LoraError> {
    validate_shape(a)?;
    validate_values(a)?;
    let mut atoms: Vec<Vec<u8>> = (0..a.rank)
        .map(|k| {
            let mut blob = Vec::with_capacity((a.dim_out + a.dim_in) * 8);
            for o in 0..a.dim_out {
                blob.extend_from_slice(&q(a.matrix_b[o][k]));
            }
            for i in 0..a.dim_in {
                blob.extend_from_slice(&q(a.matrix_a[k][i]));
            }
            blob
        })
        .collect();
    atoms.sort_unstable();

    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update([a.zone_tag]);
    h.update((a.rank as u64).to_le_bytes());
    h.update((a.dim_out as u64).to_le_bytes());
    h.update((a.dim_in as u64).to_le_bytes());
    h.update(q(a.alpha));
    for blob in &atoms {
        h.update((blob.len() as u32).to_le_bytes());
        h.update(blob);
    }
    Ok(hex(&h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LoraFactors {
        LoraFactors {
            zone_tag: 3, // ZoneId::PF
            rank: 2,
            dim_out: 4,
            dim_in: 3,
            alpha: 1.0,
            matrix_a: vec![vec![0.3, -0.2, 0.1], vec![-0.3, 0.2, 0.5]],
            matrix_b: vec![
                vec![1.0, 0.0],
                vec![-0.5, 0.5],
                vec![0.25, -0.5],
                vec![0.0, 1.0],
            ],
        }
    }

    #[test]
    fn deterministic_and_tamper_detecting() {
        let a = sample();
        assert_eq!(lora_commitment(&a), lora_commitment(&a));
        let mut t = a.clone();
        t.matrix_a[0][0] += 0.01;
        assert_ne!(lora_commitment(&a), lora_commitment(&t));
    }

    #[test]
    fn rank_atom_order_independent() {
        let a = sample();
        let mut swapped = a.clone();
        for o in 0..a.dim_out {
            swapped.matrix_b[o].swap(0, 1);
        }
        swapped.matrix_a.swap(0, 1);
        assert_eq!(lora_commitment(&a), lora_commitment(&swapped));
    }

    /// H1: a shape-malformed `LoraFactors` is rejected fail-closed, not panicked.
    #[test]
    fn lora_commitment_rejects_malformed_shape() {
        // declares rank=2/dim_out=4/dim_in=3 but supplies 1-row matrices.
        let bad = LoraFactors {
            zone_tag: 0,
            rank: 2,
            dim_out: 4,
            dim_in: 3,
            alpha: 1.0,
            matrix_a: vec![vec![0.1, 0.2, 0.3]],
            matrix_b: vec![vec![1.0, 0.0]],
        };
        assert!(matches!(
            lora_commitment(&bad),
            Err(LoraError::ShapeMismatch { .. })
        ));
        // a too-short row inside an otherwise-correct outer length is also caught.
        let mut bad_row = sample();
        bad_row.matrix_a[0].pop();
        assert!(matches!(
            lora_commitment(&bad_row),
            Err(LoraError::ShapeMismatch {
                field: "matrix_a.cols(dim_in)",
                ..
            })
        ));
    }

    // --- FT-B-001 tripwires: the commitment must bind its f32 value domain ---

    /// FT-B-001: a non-finite weight has no grid point and collapses to raw 0 under
    /// `from_f32`, so a poison adapter would otherwise share a benign digest. Refuse it.
    #[test]
    fn lora_commitment_refuses_non_finite_weights() {
        let mut nan_a = sample();
        nan_a.matrix_a[0][0] = f32::NAN;
        let mut inf_b = sample();
        inf_b.matrix_b[0][0] = f32::INFINITY;
        let mut ninf_alpha = sample();
        ninf_alpha.alpha = f32::NEG_INFINITY;
        assert!(lora_commitment(&nan_a).is_err());
        assert!(lora_commitment(&inf_b).is_err());
        assert!(lora_commitment(&ninf_alpha).is_err());
    }

    /// FT-B-001: any finite magnitude whose Q16 image saturates the i64 grid collapses
    /// onto the same raw as every other saturating magnitude — a commitment collision.
    #[test]
    fn lora_commitment_refuses_saturating_weights() {
        let mut big = sample();
        big.matrix_a[0][0] = 2.0e14; // saturates Q16::from_f32 to i64::MAX
        let mut bigger = sample();
        bigger.matrix_a[0][0] = 3.0e38; // a wildly different value, same saturated raw
        assert!(lora_commitment(&big).is_err());
        assert!(lora_commitment(&bigger).is_err());
    }

    /// FT-B-001 (the binding property): an all-zero no-op adapter still commits, but an
    /// adapter that actually serves NaN weights must NOT reproduce the zero digest.
    #[test]
    fn lora_commitment_binds_poison_apart_from_zero() {
        let zero = LoraFactors {
            zone_tag: 3,
            rank: 2,
            dim_out: 4,
            dim_in: 3,
            alpha: 0.0,
            matrix_a: vec![vec![0.0; 3]; 2],
            matrix_b: vec![vec![0.0; 2]; 4],
        };
        let mut poison = zero.clone();
        poison.matrix_a[0][0] = f32::NAN;
        assert!(lora_commitment(&zero).is_ok());
        assert_ne!(lora_commitment(&zero), lora_commitment(&poison));
    }

    // Frozen golden over an explicit fixture (the kernel's own ratchet). The cross-crate
    // parity with nat-lora's `bd08b278…` is preserved by the identical domain +
    // serialization, and is re-asserted when nat-lora adopts this kernel (migration guard).
    #[test]
    fn lora_commitment_is_frozen() {
        assert_eq!(
            lora_commitment(&sample()).unwrap(),
            "9bda1b5bccb365446d998a71f48c6852a85d1a94657022cd02bc9a3742a6716d"
        );
    }
}
