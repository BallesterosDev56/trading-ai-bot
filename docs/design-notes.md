# Design notes

## Principles

- **One pure core.** `Pipeline` takes feed messages and does everything else (book, features, model, bot) with no I/O. The live consumer and the backtest are thin drivers around it, so what you test is what runs.
- **Small closed interfaces.** `Predictor` (predict, update) for models; `Advisor` (off or Claude) for risk advice; `RiskMode` is a three-value enum. Nothing is abstracted before a second user exists.
- **Failure never stops trading logic.** Network errors back off and retry; advisor errors keep the current mode; a corrupt model file starts a fresh model with a warning.
- **Determinism where possible.** The feed takes a seed; the bot uses ordered maps; the backtest is reproducible and tested for it.

## Order book (`src/book.rs`)

- `Data` holds what is needed per resting order (id, account, timestamp, quantity).
- `Order` keeps `bids` and `asks` as `BTreeMap<Price, VecDeque<Data>>` (one side per symbol), so best price is a map end and time priority is queue order.
- `OrderBook` maps each symbol to its `Order` and indexes orders by id for O(1) lookup on cancel.
- Prices are integer ticks (cents); quantities below `QTY_EPS` count as zero so float residue never rests in the book.

## Labeling

A prediction is made when a symbol's mid changes, and labeled by the direction of the next mid change. Cancels that move the mid count as moves too. Mids are compared as integer ticks, never as floats.

## The cost gate (opt-in)

The idea: predicting direction well is not enough if every trade crosses the spread, so skip signals whose expected move (`|2p - 1|` times the average mid move) is below the spread plus fees.

It was measured, and it did not help. In the 20-seed ablation the model-driven bot made +566 USDC without the gate and -0.1 USDC with it (see `docs/results/`). The first version of this note claimed the opposite; that claim came from a run made before sizing moved to notional, and it was wrong. The gate stays as an option (`--cost-gate`) because on real data with realistic fills it could matter, but it is off by default.
