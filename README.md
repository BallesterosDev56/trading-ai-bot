# orderflow-ml

A Rust side project that turns a simulated exchange order feed into an online machine learning pipeline and a paper-trading bot: simulate market data, rebuild the order book, extract features, learn to predict short-term mid price moves, and trade the signal under risk limits, with an optional LLM (Claude) acting as a risk advisor.

The predictions come from classic machine learning (an online logistic regression), which runs locally and deterministically. Generative AI is a separate, optional layer that only supervises risk; it never predicts prices or places orders. For the full method, evaluation and caveats, read the technical report ([Markdown](docs/technical-report.md), [PDF](docs/technical-report.pdf)).

## What it does

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

| Stage | Module | Notes |
| --- | --- | --- |
| Order feed simulator + HTTP API | `src/feed.rs` | Seedable, bounded memory, graceful shutdown |
| HTTP client for the feed | `src/client.rs` | Timeouts, errors include the server's message |
| Order book + matching engine | `src/book.rs`, `src/trade.rs` | Integer price ticks, price-time priority, indexed cancels |
| Feature extraction | `src/ml/features.rs` | 7 scale-free features: spread, top/depth imbalance, last trade vs mid, trade flow, momentum, volatility |
| Online logistic regression | `src/ml/model.rs` | Streaming standardization, L2, JSON save/load |
| Rolling metrics | `src/ml/metrics.rs` | Accuracy and log-loss over the last 1000 predictions |
| Paper-trading bot | `src/bot.rs` | Notional sizing, position limits, fees, cost gate, drawdown kill switch |
| AI risk advisor | `src/advisor.rs` | Claude chooses `normal`/`reduced`/`paused`; it never places orders |
| Glue | `src/pipeline.rs`, `src/consumer.rs`, `src/backtest.rs` | One `Pipeline` drives both the live bot and the backtest |

### How the ML works

Every time a symbol's book changes, the pipeline builds a feature vector and predicts P(next mid price move is up). The label arrives at the next mid price change; then the model takes one SGD step on it. Quality is tracked against a simple baseline (lean towards the heavier side of the top of the book), so you always see whether the model is actually adding something.

### How the bot works

The bot turns P(up) into taker orders at the best bid/ask on a paper account:

- trades only on confident signals (`--threshold`) and after a warm-up;
- optionally skips signals whose expected move would not cover the spread and fees (`--cost-gate`, off by default because it removed the profit in the ablation);
- sizes in USDC notional, so BTC, ETH and SOL carry comparable risk, with a per-symbol limit;
- a drawdown kill switch halts the bot and closes everything.

### How the AI advisor works

With `--advisor claude`, every report interval the run report (model vs baseline metrics, PnL, drawdown, positions) is sent to Claude, which answers with one of three risk modes. `reduced` halves size and limits, `paused` closes everything and stops trading. The LLM is deliberately boxed in: it picks from a closed set, can only make the bot more cautious than its configured limits, and if the call fails the bot keeps its current mode and carries on.

## Getting started

Requires Rust 1.85 or newer (the toolchain is pinned in `rust-toolchain.toml`).

```bash
cargo test

# Fast, deterministic simulation: same seed, same result
cargo run --release -- backtest --messages 100000 --seed 42

# Twenty seeds as a markdown table
cargo run --release -- backtest --seeds 20

# Live: feed in one terminal, bot in another
cargo run --release -- feed --num-accounts 20 --rate 100
cargo run --release -- bot --model-path model.model.json

# Same, with Claude adjusting risk every 30 seconds
export ANTHROPIC_API_KEY=...
cargo run --release -- bot --advisor claude
```

Set `RUST_LOG=debug` for per-trade logs. `ANTHROPIC_MODEL` and `ANTHROPIC_BASE_URL` override the advisor's defaults. Use `--feed-url` (or `FEED_URL`) and `feed --addr` to run on another host or port.

The feed API and the other commands (`submit`, `cancel`, `orders`) are documented in [docs/feed-api.md](docs/feed-api.md). Available symbols: `SOL-USDC`, `ETH-USDC`, `BTC-USDC`.

## Results

Evaluation is prequential (every prediction is scored before the model learns its label) and compared against a simple order-book-imbalance rule. Each row is the mean ± std over 20 seeded runs of 100,000 simulated messages. Reproduce everything with `./scripts/reproduce.sh` (about 20 seconds); per-seed tables are in [docs/results/](docs/results/).

**Prediction quality** (next mid price direction)

| Seeds | Model accuracy | Baseline accuracy | Model log-loss | Baseline log-loss | Model better |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–20 | 0.739 ± 0.006 | 0.538 ± 0.009 | 0.513 ± 0.009 | 0.756 ± 0.011 | 20/20 |
| 21–40 (held out) | 0.740 ± 0.007 | 0.538 ± 0.012 | 0.511 ± 0.009 | 0.756 ± 0.014 | 20/20 |

**Paper-trading PnL** (USDC, same bot and accounting; only the traded signal changes)

| Traded signal | Seeds | PnL | Positive runs |
| --- | --- | ---: | ---: |
| Model | 1–20 | +566 ± 213 | 20/20 |
| Model | 21–40 (held out) | +647 ± 195 | 20/20 |
| Baseline (control) | 1–20 | -52 ± 4 | 0/20 |
| Baseline (control) | 21–40 (held out) | -53 ± 2 | 0/20 |

**How to read this.** The control shows the profit comes from what the model learned, not from the accounting. It does not show a tradable edge in a real market: the data is synthetic, and the bot is filled at the best quote with unlimited size, no latency and no market impact. The simulator's random order placement and cancels create short-term structure the model picks up. The technical report lists all the threats to validity, and the advisor has only been tested against a mock API.

## Bot options

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--threshold` | 0.55 | Trade only when P(up) is at least this (long) or at most 1 minus this (short) |
| `--trade-notional` | 100 | USDC traded per signal |
| `--max-notional` | 500 | Largest position per symbol, in USDC |
| `--fee-bps` | 1 | Taker fee per fill, in basis points |
| `--max-drawdown` | 50 | Drop from peak equity (USDC) that halts the bot and closes positions |
| `--warmup` | 500 | Samples the model must learn from before the bot may trade |
| `--cost-gate` | off | Skip signals whose expected move would not cover the spread and fees |
| `--advisor` | off | `claude` lets Claude set the risk mode (needs `ANTHROPIC_API_KEY`) |
| `--model-path` | none | Load the model on start and save it periodically |
| `backtest --seeds N` | 1 | Run N consecutive seeds and print a markdown table |
| `backtest --signal` | model | `baseline` trades the rule's signal instead (control) |

## Testing

52 automated tests plus `cargo clippy -D warnings` and `cargo fmt --check` in CI.

| Area | Tests |
| --- | ---: |
| Order book and matching | 7 |
| Features | 4 |
| Model and metrics | 9 |
| Bot (fills, limits, fees, kill switch, risk modes, cost gate) | 14 |
| Pipeline (labeling, cancels) | 4 |
| Feed (determinism, bounded buffer, wire format) | 4 |
| Advisor (parsing, mock Claude API) | 6 |
| Backtest (determinism, sweeps, control signal) | 4 |

The reports themselves are checked too: the backtest is deterministic for a seed, and a regression test makes sure the control signal really changes what the bot trades.

## Known limitations and next steps

- If the feed restarts, its message IDs start over and a running bot will wait for IDs it already passed. Restart the bot too (a feed epoch or `/status` endpoint would fix it).
- The feed keeps the last 100,000 messages in memory; a consumer that falls further behind than that loses the gap.
- The kill switch is permanent until restart by design.
- Fills ignore book depth, latency and market impact, so PnL is optimistic. See the [technical report](docs/technical-report.md) for the full list.
- Ideas: more models behind the `Predictor` trait (the baseline already uses it), multi-step horizons for labels, a maker strategy instead of crossing the spread.
