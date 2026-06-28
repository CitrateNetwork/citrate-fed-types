//! `citrate-fed-types` — the audited deterministic boundary (Gate-4 WP-W0).
//!
//! The federated meta-learning program built its machinery in the `nat` repo (the
//! research crates: aggregation, distillation, the GMN weight encoder, the LoRA
//! generator). The production chain (`citrate-chain`) must, at Gate-4, execute the *same
//! deterministic computations* a `nat` node did — re-run an aggregation to adjudicate a
//! challenge, recompute a committed digest, reconcile patronage units — **without** the
//! chain build depending on the maturing research repo.
//!
//! This crate is that boundary: it carries **only the contracts both sides must agree on
//! bit-for-bit, and none of the research ML**. Every symbol here is a faithful extraction
//! of proven `nat` code — same code, so it reproduces the existing frozen golden digests
//! (`aggregate` → `e79c5a63…`; settlement → `4000/1800/500`), which is the audit evidence
//! that the extraction did not drift. Deps are `serde` + `sha2` only; nothing pulls in a
//! model, an encoder, a generator, or an `f32` research path.
//!
//! See `ADR-2026-06-28-fed-types-boundary` for the decision and the migration plan
//! (the `nat` crates and `citrate-chain` both come to depend on this kernel).

pub mod aggregate;
pub mod fixed;
pub mod lora;
pub mod seam;
pub mod settlement;

pub use fixed::Q16;

/// Shared hex encoder for the digest paths (lowercase, no allocation surprises).
pub(crate) fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}
