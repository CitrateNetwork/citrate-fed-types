# citrate-fed-types

*Part of the **[Citrate Network](https://citrate.ai)** — own the means of computation. · [Docs](https://docs.citrate.ai) · [Run a node](https://citrate.ai/download) · [Contribute → free membership](https://github.com/CitrateNetwork/.github/blob/main/CONTRIBUTING.md)*
> The audited, deterministic Rust boundary crate shared between the Citrate chain and the federated-meta-learning research code — the math both sides must agree on **bit-for-bit**.

## What it is
`citrate-fed-types` carries only the deterministic contracts the production chain
([citrate-chain](https://github.com/CitrateNetwork/citrate-chain)) and the federated
meta-learning research crates must reproduce identically: the `Q16` fixed-point grid, the
bucketed coordinate trimmed-mean aggregation + digest (`aggregate` / `digest_of` /
`bucket_of`), the `SettlementRow` patronage-unit math, the LoRA `(A,B)` commitment, and the
`ChainCommit` / `Settlement` / `UnifiedSettlement` seam traits. No research ML — no model,
encoder, generator, or `f32` path. Every symbol is a faithful extraction that reproduces the
frozen golden digests (aggregate → `e79c5a63…`; settlement units → `4000 / 1800 / 500`), which
is the audit evidence the extraction did not drift. Dependencies are `serde` + `sha2` only.

See the concepts in the docs: <https://docs.citrate.ai>.

## Prerequisites
```bash
# Toolchain is pinned by rust-toolchain.toml (channel 1.96.0, with rustfmt + clippy).
# rustup will auto-install the pinned toolchain on first cargo invocation.
rustup --version
cargo --version   # resolves to the pinned 1.96.0 inside the repo
```

## Build from source
```bash
git clone https://github.com/CitrateNetwork/citrate-fed-types.git
cd citrate-fed-types
cargo build            # compiles the library crate
cargo test             # runs unit tests INCLUDING the frozen-digest parity checks
./scripts/ci-local.sh  # full local gate: fmt --check, clippy -D warnings, test
```
Expected: a library rlib under `target/`. There is no binary. The parity tests
(`frozen_aggregate_digest_matches_nat`, `patronage_units_match_onchain_ledger`) must pass —
they are the guarantee that committed bytes have not moved. Build is small and fast.

## Run locally
This is a pure library crate — nothing to run, no ports, no network, no env. Exercise it via
its tests, or depend on it from another crate:

```rust
use citrate_fed_types::Q16;
use citrate_fed_types::aggregate::{aggregate, digest_of};
// deterministic: same inputs → same digest on every machine and every node.
```
Verify it's working: `cargo test` is green and prints the parity tests passing.

## Connect it locally  ← the differentiator
This crate is "connected" by being a **dependency**, not a service. To wire it into a local
Citrate build so both sides share one deterministic kernel:

1. Clone it beside your other checkouts (e.g. next to `citrate-chain`).
2. Add it as a path (or git) dependency in the consuming crate's `Cargo.toml`:
   ```toml
   [dependencies]
   citrate-fed-types = { path = "../citrate-fed-types" }
   # or, pinned by revision:
   # citrate-fed-types = { git = "https://github.com/CitrateNetwork/citrate-fed-types.git", rev = "<sha>" }
   ```
3. Build the consumer (`cargo build` in citrate-chain). Because the toolchain is pinned to
   `1.96.0` on both sides, the aggregation/settlement/LoRA results are byte-identical to what
   a federated node produced — which is exactly what a Gate-4 challenge re-execution needs.
4. End-to-end check: `cargo test` in the consuming crate; the shared golden digests
   (`e79c5a63…`, `4000/1800/500`) match on both sides.

For the full multi-repo bring-up, see the LOCAL_STACK guide at <https://docs.citrate.ai>.

## Configuration
None. No env vars, no config files, no network endpoints — determinism is the point. The only
"configuration" is the pinned toolchain in `rust-toolchain.toml` (channel `1.96.0`), which
every consumer must match to reproduce the golden digests.

## Links
- Docs: <https://docs.citrate.ai>
- Consumed by: [citrate-chain](https://github.com/CitrateNetwork/citrate-chain) (challenge re-execution + settlement) and the federated meta-learning research crates
- Contributing (DCO): `CONTRIBUTING.md` · Security: `SECURITY.md` · License: [`LICENSE`](LICENSE)

## License

Licensed under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE)). This is the open-source infrastructure tier of Citrate's open-core model. The commercial application layer is source-available under BUSL-1.1. Licensor: Citrate Inc.
