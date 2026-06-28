# citrate-fed-types

The **audited deterministic boundary** between the federated meta-learning research crates
(in the `nat` repo) and the production chain (`citrate-chain`) — Gate-4 WP-W0 of the
Federated Meta-Learning program.

At Gate-4 the chain must execute the *same deterministic computations* a `nat` node did —
re-run an aggregation to adjudicate a challenge, recompute a committed digest, reconcile
patronage units — **without** the consensus-critical chain build depending on the maturing
research repo. This crate is that boundary: **only the contracts both sides must agree on
bit-for-bit, and none of the research ML.**

## Modules
| Module | Contract |
|--------|----------|
| `fixed` | `Q16` fixed-point grid (the quantization boundary) |
| `aggregate` | bucketed coordinate trimmed-mean + `digest_of` + `bucket_of` (the challenge re-executes this) |
| `settlement` | `SettlementRow` patronage math — the `PatronageLedger` unit replica |
| `lora` | `lora_commitment` over explicit `(A,B)` factors — `LoRAFactory.adapterModelCommitment` |
| `seam` | `ChainCommit` / `Settlement` / `UnifiedSettlement` trait shapes (interface only) |

## Faithful-extraction guarantee
Every symbol is a faithful copy of proven `nat` code, so it reproduces the existing frozen
golden digests — the audit evidence that the extraction did not drift:
- `aggregate::frozen_aggregate_digest_matches_nat` → `e79c5a63…` (== `nat-aggregate`)
- `settlement::patronage_units_match_onchain_ledger` → `4000 / 1800 / 500` (== `FederatedSettlement.t.sol`)

Domain strings + serialization are preserved verbatim, so when the `nat` crates adopt this
kernel and delete their copies, **no committed byte moves**.

## What is deliberately NOT here
The GMN encoder, the LoRA generator + meta-training, federated distillation, the model, and
every `f32` research path. The chain links the kernel, not the laboratory.

See `citrate-federation/.agentile/adrs/ADR-2026-06-28-fed-types-boundary.md`.
