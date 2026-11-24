# orderflow-ml: technical report

Online machine learning and paper trading on a simulated limit order book, with an optional LLM risk advisor.

This document describes what the system does, how it was evaluated, and how far the results can be trusted. Every number in it can be regenerated with `./scripts/reproduce.sh`.

## 1. Summary

`orderflow-ml` is a Rust system with five parts: a simulated exchange feed, a limit order book with a matching engine, an online-learning pipeline that predicts the next mid price move, a paper-trading bot with risk controls, and an optional advisor in which Claude chooses the bot's risk mode.

Main findings, all measured on the built-in simulator:

1. The online logistic regression predicts the direction of the next mid price move with **74.0% accuracy** (±0.7 pp across seeds) and a log-loss of **0.511**, against 53.8% and 0.756 for a simple order-book-imbalance rule. The model has the better log-loss in 40 of 40 seeded runs.
2. A bot trading that signal made money in 40 of 40 runs (mean +566 and +647 USDC on two disjoint sets of 20 seeds). The same bot trading the baseline's signal lost money in 40 of 40 runs (about -52 USDC), so the profit comes from the model and not from the accounting.
3. These results do **not** show a tradable edge in real markets. The simulator is synthetic, and the bot is filled at the best quote with unlimited size (section 11).
4. A "cost gate" that skips signals whose expected move is below the spread did not help here; it removed the profit (section 8.4). It is off by default.

## 2. Scope and goals

| In scope | Out of scope |
| --- | --- |
| Reconstructing a limit order book from a message feed | Real exchange connectivity |
| Online (streaming) learning and honest evaluation against a baseline | Deep learning, hyperparameter search |
| Paper trading with risk limits | Real money, latency, market impact |
| A bounded role for an LLM in a trading loop | Using an LLM to predict prices |

Design goals, in order: correctness, testability, simplicity of the code, then scalability (bounded memory, clean shutdown, retry behavior).

## 3. Architecture

```
live:      feed ──HTTP──► FeedClient ──► consumer
                          (poll, backoff, save model)
                                              │
backtest:  FeedState (seeded, in-process) ────┤
                                              ▼
      ┌─ Pipeline (pure, no I/O) ──────────────────────────┐
      │ OrderBook ─► Features ─► Model ─► Bot (paper)      │
      └────────────────────────────────────────────────────┘
                                │ run report
                                ▼
                  Advisor (Claude, optional)
                                │ risk mode: normal / reduced / paused
                                └──────────► back to the Bot
```

| Module | Responsibility |
| --- | --- |
| `feed.rs` | Message types, seeded message generator, bounded message buffer, HTTP API |
| `client.rs` | HTTP client with timeouts; errors include the server's message |
| `book.rs`, `trade.rs` | Order book, price-time matching, trade records |
| `ml/features.rs` | Per-symbol feature extraction with rolling state |
| `ml/model.rs` | `Predictor` trait, online logistic regression, imbalance baseline, JSON persistence |
| `ml/metrics.rs` | Rolling (last 1000) and cumulative accuracy and log-loss |
| `bot.rs` | Paper trading, position accounting, risk limits, kill switch |
| `advisor.rs` | Claude-backed risk advisor |
| `pipeline.rs` | Ties book, features, model, metrics and bot together; no I/O |
| `consumer.rs` | Live loop: poll, feed the pipeline, report, persist, consult the advisor |
| `backtest.rs` | Seeded in-process simulation and multi-seed sweeps |

**Key design choice.** `Pipeline` has no I/O. The live bot and the backtest are thin drivers around the same struct, so the backtest exercises the production code path and results are deterministic given a seed.

**Failure model.** Network errors back off exponentially (capped at 10 s) and retry. An advisor failure keeps the current risk mode. A corrupt model file starts a fresh model with a warning. Ctrl-C stops the loops, saves the model and prints a final report. The feed holds at most 100,000 messages.

## 4. Data: the simulated feed

The feed generates one message per tick. With probability 0.15 (when candidates exist) it cancels a random order among the last 50 generated, which may already have traded. Otherwise it creates a limit order:

- a symbol chosen uniformly from SOL-USDC (10), ETH-USDC (100), BTC-USDC (1000);
- the symbol's mid price takes a multiplicative step uniform in ±0.2%;
- the order price is that mid times a factor uniform in ±0.5%, rounded to cents;
- quantity uniform in [1, 10], rounded to one decimal; side 50/50; account uniform.

Prices are integer ticks (cents) inside the book, so price levels are exact map keys. A seed fully determines the stream (apart from wall-clock timestamps, which no component uses for decisions).

## 5. Order book and matching

Each symbol keeps `bids` and `asks` as `BTreeMap<price, VecDeque<order>>`. The best price is a map end and time priority is queue order. An index from order ID to (symbol, side, price) finds a cancel's level directly, then the order is removed from its queue.

A new order is matched against the opposite side while it crosses the best price; each fill trades at the resting order's price, and any remainder rests at the limit price. Quantities below `1e-9` are treated as zero, which fixes float residue such as 0.1 + 0.2 ≠ 0.3 leaving "dust" orders in the book (covered by a test).

## 6. Features

Seven features are computed whenever a symbol's book changes and both sides are non-empty. All are scale-free so one model serves all three symbols.

| # | Name | Definition |
| --- | --- | --- |
| 0 | `spread_bps` | (ask − bid) / mid × 10⁴ |
| 1 | `imbalance_top` | (bidQty − askQty) / (bidQty + askQty) at the best level |
| 2 | `imbalance_depth` | same, summed over the best 5 levels per side |
| 3 | `last_trade_vs_mid_bps` | (last trade price − mid) / mid × 10⁴, 0 before the first trade |
| 4 | `trade_flow` | net aggressor volume / total volume over the last 20 trades, in [−1, 1] |
| 5 | `momentum_bps` | (mid − oldest mid in the last 20 snapshots) / mid × 10⁴ |
| 6 | `volatility_bps` | std. dev. of mid-to-mid returns over the last 20 snapshots, in bps |

## 7. Model and evaluation

### 7.1 Model

Online logistic regression, `P(up) = σ(w·z + b)`, where `z` is the feature vector standardized with running mean and variance (Welford's algorithm) and clipped to ±5 standard deviations. Each labeled example triggers one SGD step on the log-loss with L2 decay:

```
w ← w − η · ((p − y) · z + λ · w)        b ← b − η · (p − y)        η = 0.05, λ = 1e-4
```

The scaler observes an example before it is used for the update, so inputs stay centered as the market drifts.

### 7.2 Labeling

When a symbol's mid changes, the pipeline stores a pending prediction (features, mid, model probability, baseline probability). The label is the direction of the **next** mid change for that symbol; mids are compared as exact integers (twice the mid in ticks), never as floats. When the label arrives, the model is updated and both predictions are scored. About 27,000 predictions are scored per 100,000 messages.

### 7.3 Evaluation protocol

Evaluation is **prequential** (test-then-train): every prediction is made and recorded before the model sees its label, so no example is scored by a model that trained on it. Reported metrics are cumulative over the whole run, with a rolling 1000-prediction window available for live monitoring.

The yardstick is an **imbalance baseline**: `P(up) = 0.5 + 0.4 · imbalance_top`. A model that cannot beat it is not adding information.

## 8. The bot

### 8.1 Execution model

The bot trades on a paper account. A signal above `threshold` (0.55) goes long, below `1 − threshold` goes short; otherwise it waits. Fills are taker orders at the best ask (buy) or best bid (sell), with a 1 bp fee on notional. The bot does not trade until the model has learned from 500 samples.

### 8.2 Sizing and limits

Trade size is 100 USDC of notional and the position limit is 500 USDC per symbol, so BTC, ETH and SOL carry comparable risk. (An earlier version sized in units, which made BTC 100 times riskier than SOL and tripped the kill switch on ordinary noise.) The bot only moves toward the signal, never against it, and never exceeds the limit.

### 8.3 Risk controls

Equity is realized PnL, minus fees, plus open positions marked at the mid. If equity falls more than `max_drawdown` (50 USDC) below its peak, the bot halts and closes all positions at the current quotes. A halt lasts until restart. Risk modes `reduced` (half size and limit) and `paused` (flatten and stop) can be set by the advisor.

### 8.4 Ablation: the cost gate

Intuition says a bot that crosses the spread on every trade should skip signals whose expected move is smaller than the spread. The "cost gate" implements that: it trades only if `|2p − 1| × (average mid move)` exceeds the spread plus fees. It was tested as an ablation (section 10) and **removed the profit** in this simulator, so it is off by default (`--cost-gate` enables it). Its estimator (an average of mid moves) is crude, and the result should not be read as "cost gates never help".

## 9. The AI advisor

With `--advisor claude`, at each report interval the run report (model and baseline metrics, PnL, drawdown, positions, current mode) is sent to the Anthropic Messages API as JSON. The system prompt defines a closed answer set, and the reply is parsed as `{"mode": "normal|reduced|paused", "reason": "..."}`.

Safety properties:

- The LLM never places orders and never sees or changes limits; it only picks one of three modes.
- It cannot make the bot riskier than its configured limits (`normal` is the configured maximum).
- Any failure (network, HTTP error, unparsable reply, unknown mode) keeps the current mode.
- The reply parser tolerates code fences and casing but rejects unknown modes.

**Evaluation status.** The advisor is tested against a mock API server that checks headers, model name and request shape (3 tests). It has **not** been run against the real API and its decisions have **not** been evaluated for usefulness. The backtest does not call it (cost and determinism).

## 10. Results

Setup: 100,000 simulated messages per run, 20 consecutive seeds per table, default parameters unless stated. Raw per-seed tables are in [docs/results/](results/).

### 10.1 Prediction quality (cumulative over each run, mean ± std across 20 seeds)

| Seeds | Model accuracy | Baseline accuracy | Model log-loss | Baseline log-loss | Model better |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–20 | 0.739 ± 0.006 | 0.538 ± 0.009 | 0.5130 ± 0.0092 | 0.7558 ± 0.0114 | 20/20 |
| 21–40 (held out) | 0.740 ± 0.007 | 0.538 ± 0.012 | 0.5113 ± 0.0093 | 0.7564 ± 0.0136 | 20/20 |

### 10.2 Paper-trading PnL in USDC (mean ± std across 20 seeds)

| Traded signal | Cost gate | Seeds | PnL | Positive runs | Kill switch tripped |
| --- | --- | --- | ---: | ---: | ---: |
| Model | off (default) | 1–20 | **+566 ± 213** | 20/20 | 3/20 |
| Model | off (default) | 21–40 (held out) | **+647 ± 195** | 20/20 | 6/20 |
| Baseline (control) | off | 1–20 | −52 ± 4 | 0/20 | 20/20 |
| Baseline (control) | off | 21–40 (held out) | −53 ± 2 | 0/20 | 20/20 |
| Model | on | 1–20 | −0.1 ± 63 | 9/20 | 19/20 |
| Baseline | on | 1–20 | +14 ± 68 | 10/20 | 18/20 |

### 10.3 How to read these

- The control matters. The same bot, fees and accounting lose money on the baseline's signal and make money on the model's, so the profit is attributable to what the model learned.
- The model's log-loss of 0.51 against 0.69 for a coin flip is a large, stable improvement.
- Seeds 1–20 were looked at while choosing the cost-gate default, so seeds 21–40 are the cleaner out-of-sample check; the results match.
- Across seeds the PnL varies a lot (std ≈ 200), and in 3 to 6 of 20 runs the kill switch fires. A single run says little; use the sweeps.

## 11. Threats to validity

1. **Synthetic data.** The simulator's random order placement and 15% cancel process create short-term structure the model can learn. The most likely source is mean reversion after a new best quote is cancelled, but this was not isolated. A real market would be different, and the results say nothing about it.
2. **Optimistic fills.** Fills happen at the best quote for the full size, even when the resting quantity is smaller, with no latency, queue position or market impact. This overstates PnL, probably most for SOL, where 100 USDC is about 10 units, as much as a whole price level can hold (BTC's 0.1 units always fits). That was not measured.
3. **Same generator for training and testing.** Fresh seeds give fresh random streams, but all share one data-generating process. This tests overfitting to a stream, not robustness to a different market.
4. **Marking at mid.** Open positions are valued at the mid, slightly more favorable than liquidation value.
5. **Decisions informed by seeds 1–20.** The cost-gate default was set after looking at them, and the mechanism behind the gate's failure was not investigated.
6. **Advisor unevaluated** (section 9).
7. **No statistical test.** Win counts (20/20, 0/20) and standard deviations are reported, but no formal test or confidence interval.

## 12. Testing

52 automated tests (`cargo test`), plus `cargo clippy -D warnings` and `cargo fmt --check` in CI.

| Area | Tests | What they check |
| --- | ---: | --- |
| Order book | 3 | best price, cancel by ID, partial fills and time priority |
| Matching | 4 | resting vs crossing, sweeping levels, float residue |
| Features | 4 | empty side, spread and imbalance values, trade flow, momentum |
| Model | 5 | learns a signal, wildly different feature scales, save/load round trip, neutral when untrained, baseline |
| Metrics | 4 | accuracy and log-loss values, window eviction, cumulative equals a never-dropping window |
| Bot | 14 | fills at bid/ask, notional sizing, limits, fees, kill switch, pause, reduced mode, position flips, cost gate |
| Pipeline | 4 | labeling after a mid move, no label when the mid is unchanged, cancels |
| Feed | 4 | seeded determinism, increasing IDs, bounded buffer, JSON wire format |
| Advisor | 3 | parsing, code fences and casing, rejection of bad replies |
| Advisor (mock API) | 3 | request headers and shape, API errors, advisor off |
| Backtest | 1 + 3 | sweep seeds and table; determinism, learning within limits, signal source changes behavior |

Manual end-to-end checks were also run: a live feed and bot over HTTP, model saved and reloaded, feed killed mid-run (backoff), Ctrl-C shutdown, invalid order rejection.

A bug found by the evaluation work itself: the first version of the control-signal option was silently ignored (identical results for both signals). It was caught because the numbers matched to the cent, fixed, and a regression test was added.

## 13. Reproducibility

```bash
cargo test                      # 52 tests
./scripts/reproduce.sh          # regenerates every table in docs/results/ (about 20 seconds)
cargo run --release -- backtest --messages 100000 --seed 1 --seeds 20   # one table
cargo run --release -- backtest --seed 3 --signal baseline              # one run
```

Toolchain pinned to Rust 1.91.1 (`rust-toolchain.toml`); dependencies locked in `Cargo.lock`. Runs are deterministic for a given seed and configuration.

## 14. Limitations and future work

- Feed restarts reset message IDs and a running bot will wait for IDs it has already passed (restart the bot).
- A consumer that falls more than 100,000 messages behind loses the gap.
- Model comparison: add more `Predictor` implementations (gradient-boosted trees, a small neural net) and compare on the same prequential protocol.
- Realistic execution: consume book liquidity when filling, add latency, and test a maker strategy.
- Isolate the source of the learnable structure (for example, by disabling cancels in the simulator) and report how much of the edge disappears.
- Evaluate the advisor: run it against the real API on recorded sessions and compare PnL and drawdown with and without it.
- Replace the simulator with recorded exchange data to test on a real market.

## Appendix A: default parameters

| Parameter | Value | Where |
| --- | --- | --- |
| Learning rate η | 0.05 | `ml/model.rs` |
| L2 λ | 1e-4 | `ml/model.rs` |
| Feature clip | ±5 σ | `ml/model.rs` |
| Depth levels / flow window / mid window | 5 / 20 / 20 | `ml/features.rs` |
| Rolling metrics window | 1000 | `pipeline.rs` |
| Signal threshold | 0.55 | `bot.rs` |
| Trade notional / max notional per symbol | 100 / 500 USDC | `bot.rs` |
| Fee | 1 bp | `bot.rs` |
| Max drawdown | 50 USDC | `bot.rs` |
| Warm-up samples | 500 | `bot.rs` |
| Cancel probability / candidate window | 0.15 / 50 | `feed.rs` |
| Retained feed messages | 100,000 | `feed.rs` |
| Advisor model (override with `ANTHROPIC_MODEL`) | `claude-haiku-5-5` | `advisor.rs` |
