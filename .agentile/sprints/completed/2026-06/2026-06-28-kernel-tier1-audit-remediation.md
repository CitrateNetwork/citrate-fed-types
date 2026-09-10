---
created: 2026-06-28T00:00:00Z
closed: 2026-06-28T00:00:00Z
branch: fix/tier1-remediation
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: completed
sprint: KERNEL-AUDIT-S1 (Tier-1 audit + remediation)
companion: ../../../../../citrate-federation/.agentile/planset/2026-06-28-i64-s1-q16-unification-and-deterministic-reroll.md
---

# citrate-fed-types — Tier-1 audit + remediation

The kernel is the one place the chain, nat, and the federated gradient path all agree
on what a number *is*. Before it could become that shared source of truth, it had to
survive a Tier-1 audit — the outsider's-lens, reproduce-or-retract, adversarial-by-
default standard (AGENTILE_AUDIT_STANDARD v0.1) — not a self-review. This journal
records that audit and the remediation sprint that closed it. Merged to `main` at
**dbbfbff** (PR #1), the rev the whole I64-S1 program is now pinned to.

## What the audit was looking for

Not "does it pass its tests" — it passed its tests. The question was: **where does a
hostile or merely-weird input get a silent wrong answer instead of a loud refusal?**
A consensus kernel has no room for "shouldn't happen" — every path is reachable by
someone who wants it to be.

## Findings: 3 HIGH (all fail-closed gaps) + minors

Every HIGH was the same species of bug — a path that *silently mis-handled* an extreme
input rather than failing closed. None were "wrong math"; all were "missing refusal."

| ID | Surface | Gap | Remediation |
|----|---------|-----|-------------|
| **H1** | `aggregate.rs` | Unbounded bucket count — a pathological input could allocate without limit / wrap | `MAX_BUCKETS = 1<<20`, `TooManyBuckets` error, `checked_mul(2).is_none_or(...)` trim guard, `bucket_of` clamp |
| **H2** | `lora.rs` | `lora_commitment` consumed un-validated matrix shapes (rank/dim mismatch → garbage digest) | `LoraError` + `validate_shape`; `lora_commitment` now returns `Result` |
| **H3** | `fixed.rs` | Fixed-point edges could panic / produce non-deterministic values (non-finite `f32`, div-by-zero) | `from_f32` non-finite → `ZERO`; `div` by zero saturates by numerator sign (`±i64::MAX`); `saturating_i128_to_i64` down-cast |

The minors were of a piece: a non-unit `patronage` weight path in `settlement.rs` that
deserved an explicit test, and the saturating-vs-truncating boundary on the i128→i64
mul down-cast.

## How it was remediated: red → green ratchets, anchors held

Each fix landed as a **failing test first, then the fix** — so the test proves the gap
was real, not just that the patch compiles. The discipline that mattered most:

- **Frozen golden digests were preserved, not regenerated.** `aggregate` (e79c5a63),
  `lora` (9bda1b5b), and the patronage constants (4000/1800/500) are consensus anchors —
  any downstream commitment assumes those exact bytes. The remediation added *refusals*
  around the edges without moving a single committed digest. A fix that changed a golden
  would have been a fix that broke consensus.
- **One over-specified test caught me honestly.** An `aggregate_rejects_overflowing_trim`
  assertion pinned `bucket_count == 4` when a hash collision legitimately produced 3;
  relaxed to `matches!(... trim: t, .. if t == trim)` — asserting the *property* (the trim
  fired), not an incidental count. A reminder that adversarial tests must assert the
  invariant, not the coincidence.

## Q16 is i64 here — the decision that propagated

The kernel's `Q16` is `i64`-backed (private field, `from_raw`/`raw` access), saturating
`add`/`sub`/`mul`/`div` with i128 intermediates, div-by-zero saturating by sign. That
i64 choice is what later forced — and justified — the chain-wide widening: once the
kernel is the source of truth and it's i64, the chain *must* meet it there. This audit
is the reason the chain could trust the kernel enough to pin to it.

## Close

23 tests green; 3 HIGH + minors remediated with anchors intact; merged at **dbbfbff**.
The rev is now a dev-dependency of `citrate-chain`'s `core/federated` parity crate,
where a 6-test proof asserts the chain's `Q16` is byte-identical to this one — the
audit's payoff, made executable.
