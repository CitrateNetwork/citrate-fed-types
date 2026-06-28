//! Seam traits — the integration interfaces `citrate-chain` / `citrate-compute-pool`
//! implement to land a federated round (g4-onchain + g4-settlement).
//!
//! These are *interface shapes only* (no logic): the deterministic computation lives in
//! [`crate::aggregate`] / [`crate::settlement`] / [`crate::lora`]; these traits are the
//! thin boundary the production adapters fill. Faithful extraction of the
//! `nat-federated` `ChainCommit` / `Settlement` traits plus the richer `UnifiedSettlement`
//! that carries both factors (compute + data_quality) instead of a pre-collapsed weight.

use crate::fixed::Q16;
use crate::settlement::SettlementRow;

/// A boundary error. Deployment adapters map their own failures into this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FedError(pub String);

impl FedError {
    pub fn new(msg: impl Into<String>) -> Self {
        FedError(msg.into())
    }
}

impl std::fmt::Display for FedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "federation seam error: {}", self.0)
    }
}
impl std::error::Error for FedError {}

/// Commits the merged trace-hash on-chain (g4-onchain). The real impl writes to
/// `citrate-chain`; an auditor replays the gather and checks the committed hash reproduces.
pub trait ChainCommit {
    /// An opaque receipt (e.g. a tx hash) the caller can present for audit.
    fn commit_trace_hash(&self, merged_hash: &str) -> Result<String, FedError>;
}

/// Settles an accepted contribution through compute-pool with a collapsed reward weight
/// (the legacy seam). NAT proposes the weight; compute-pool converts weight → payout.
pub trait Settlement {
    fn settle(&self, node_id: &str, reward_weight: Q16) -> Result<(), FedError>;
}

/// The richer settlement: carries both factors explicitly via [`SettlementRow`] so the
/// patronage ledger computes `units = (compute·wCompute)·(quality·wData)` itself (the
/// `citrate-coop` `FederatedSettlement` / `PatronageLedger` adapter implements this).
pub trait UnifiedSettlement {
    fn settle_row(&self, row: &SettlementRow) -> Result<(), FedError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // A reference impl proves the trait shapes are usable end-to-end with the kernel math.
    struct Recorder {
        committed: std::cell::RefCell<Vec<String>>,
        settled: std::cell::RefCell<Vec<(String, u128)>>,
    }

    impl ChainCommit for Recorder {
        fn commit_trace_hash(&self, merged_hash: &str) -> Result<String, FedError> {
            self.committed.borrow_mut().push(merged_hash.to_string());
            Ok(format!("receipt:{merged_hash}"))
        }
    }
    impl UnifiedSettlement for Recorder {
        fn settle_row(&self, row: &SettlementRow) -> Result<(), FedError> {
            self.settled
                .borrow_mut()
                .push((row.node_id.clone(), row.patronage_units(10_000, 10_000)));
            Ok(())
        }
    }

    #[test]
    fn seam_traits_drive_the_kernel_math() {
        let r = Recorder {
            committed: Default::default(),
            settled: Default::default(),
        };
        assert_eq!(r.commit_trace_hash("abc").unwrap(), "receipt:abc");
        let row = SettlementRow::new(
            "nodeA",
            Q16::from_f32(4000.0),
            Q16::from_f32(1.0),
            Some(3),
            "t",
        );
        r.settle_row(&row).unwrap();
        assert_eq!(r.settled.borrow()[0], ("nodeA".to_string(), 4000));
    }
}
