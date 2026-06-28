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
        }
    }
}
impl std::error::Error for LoraError {}

fn q(v: f32) -> [u8; 8] {
    Q16::from_f32(v).raw().to_le_bytes()
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
/// declared dims (finding H1), so an untrusted registration cannot panic the verifier. The
/// committed bytes are unchanged from the pre-remediation digest — the frozen golden holds.
pub fn lora_commitment(a: &LoraFactors) -> Result<String, LoraError> {
    validate_shape(a)?;
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
