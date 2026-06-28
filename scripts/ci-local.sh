#!/usr/bin/env bash
# Local CI for the citrate-fed-types kernel — the gate until production (no GitHub
# Actions / paid minutes yet). This crate is the audited deterministic boundary, so its
# checks include the frozen-digest parity tests (e79c5a63 / 4000-1800-500) that prove it
# has not drifted from the nat source of truth.
#
# Usage: scripts/ci-local.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
export CARGO_TERM_COLOR=always

fail=0
guard() { # guard <name> <cmd...>
  local name="$1"; shift
  if "$@"; then printf '   \033[32m✓ %s\033[0m\n' "$name"
  else printf '   \033[31m✗ %s FAILED\033[0m\n' "$name"; fail=1; fi
}

printf '\033[1m>> citrate-fed-types local CI\033[0m\n'
guard fmt        cargo fmt --all -- --check
guard clippy     cargo clippy --all-targets -- -D warnings
guard test       cargo test
# cargo-deny is optional here (no deny.toml yet); run it if a config exists.
if [[ -f deny.toml ]] && command -v cargo-deny >/dev/null 2>&1; then
  guard cargo-deny cargo deny check advisories bans licenses sources
fi

echo
if [[ "$fail" == 0 ]]; then printf '\033[32m=== KERNEL CI PASSED ===\033[0m\n'
else printf '\033[31m=== KERNEL CI FAILED ===\033[0m\n'; fi
exit "$fail"
