#!/usr/bin/env bash
# Regenerates every result table in docs/results/ (deterministic: same seeds, same numbers).
set -euo pipefail
cd "$(dirname "$0")/.."

MESSAGES=${MESSAGES:-100000}
SEEDS=${SEEDS:-20}
OUT=docs/results
mkdir -p "$OUT"
cargo build --release -q
BIN=target/release/orderflow-ml

# 1. The four configurations: traded signal (model or baseline) x cost gate (off or on).
for signal in model baseline; do
  "$BIN" backtest --messages "$MESSAGES" --seed 1 --seeds "$SEEDS" --signal "$signal" \
    > "$OUT/$signal-gate-off.md" 2>/dev/null
  "$BIN" backtest --messages "$MESSAGES" --seed 1 --seeds "$SEEDS" --signal "$signal" --cost-gate \
    > "$OUT/$signal-gate-on.md" 2>/dev/null
done

# 2. Held-out seeds (21-40): never looked at while choosing defaults, so a cleaner out-of-sample check.
"$BIN" backtest --messages "$MESSAGES" --seed 21 --seeds "$SEEDS" > "$OUT/holdout-model-gate-off.md" 2>/dev/null
"$BIN" backtest --messages "$MESSAGES" --seed 21 --seeds "$SEEDS" --signal baseline \
  > "$OUT/holdout-baseline-gate-off.md" 2>/dev/null

echo "Wrote $(ls "$OUT" | wc -l) tables to $OUT/"
