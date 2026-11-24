use crate::book::{Data, Order, Price, from_price};
use crate::feed::Side;
use crate::trade::Trade;
use std::collections::VecDeque;

/// Number of features produced by `extract`, kept in sync with `FEATURE_NAMES`.
pub const NUM_FEATURES: usize = 7;

/// A fixed-size numeric snapshot of one symbol's order book, ready for a model.
pub type FeatureVector = [f64; NUM_FEATURES];

/// Human-readable name of each feature, in vector order.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "spread_bps",
    "imbalance_top",
    "imbalance_depth",
    "last_trade_vs_mid_bps",
    "trade_flow",
    "momentum_bps",
    "volatility_bps",
];

/// Index of the top-of-book imbalance, which the baseline model reads directly.
pub const IMBALANCE_TOP: usize = 1;

/// How many price levels per side count towards depth imbalance.
const DEPTH_LEVELS: usize = 5;
/// How many recent trades feed the trade-flow feature.
const FLOW_WINDOW: usize = 20;
/// How many recent mid prices feed momentum and volatility.
const MID_WINDOW: usize = 20;
/// Converts a ratio into basis points, so features are scale-free across symbols.
const BPS: f64 = 10_000.0;

/// One symbol's book state at a point in time, plus the features derived from it.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub best_bid_ticks: Price,
    pub best_ask_ticks: Price,
    pub features: FeatureVector,
}

impl Snapshot {
    pub fn best_bid(&self) -> f64 {
        from_price(self.best_bid_ticks)
    }

    pub fn best_ask(&self) -> f64 {
        from_price(self.best_ask_ticks)
    }

    /// Twice the mid price in ticks; an exact integer, so mid moves compare without float error.
    pub fn mid_key(&self) -> i64 {
        self.best_bid_ticks + self.best_ask_ticks
    }
}

/// Turns one symbol's book plus its recent history into a `Snapshot`.
/// Keep one extractor per symbol: it remembers recent mids and trades.
#[derive(Debug, Default)]
pub struct FeatureExtractor {
    mids: VecDeque<f64>,
    /// Signed trade quantities: positive when the buyer was the aggressor.
    flow: VecDeque<f64>,
    last_trade_price: Option<f64>,
}

impl FeatureExtractor {
    /// Records an executed trade so later snapshots can see recent trade flow.
    pub fn on_trade(&mut self, trade: &Trade) {
        self.last_trade_price = Some(from_price(trade.price));
        let signed = match trade.aggressor {
            Side::Buy => trade.quantity,
            Side::Sell => -trade.quantity,
        };
        self.flow.push_back(signed);
        if self.flow.len() > FLOW_WINDOW {
            self.flow.pop_front();
        }
    }

    /// Builds a snapshot for the book, or `None` if either side is empty.
    pub fn extract(&mut self, book: &Order) -> Option<Snapshot> {
        // 1. Need both sides of the book, otherwise mid price and spread are undefined.
        let best_bid_ticks = book.best_bid()?;
        let best_ask_ticks = book.best_ask()?;
        let bid = from_price(best_bid_ticks);
        let ask = from_price(best_ask_ticks);
        let mid = (bid + ask) / 2.0;

        // 2. Spread and the last trade's distance from mid, in basis points.
        let spread_bps = (ask - bid) / mid * BPS;
        let last_trade_vs_mid_bps = self
            .last_trade_price
            .map_or(0.0, |price| (price - mid) / mid * BPS);

        // 3. Imbalance in [-1, 1]: positive means more buy pressure (top of book, then deeper).
        let imbalance_top = imbalance(
            level_quantity(&book.bids[&best_bid_ticks]),
            level_quantity(&book.asks[&best_ask_ticks]),
        );
        let imbalance_depth = imbalance(
            depth(book.bids.iter().rev(), DEPTH_LEVELS),
            depth(book.asks.iter(), DEPTH_LEVELS),
        );

        // 4. Trade flow in [-1, 1]: net aggressor volume over the recent trade window.
        let flow_total: f64 = self.flow.iter().map(|q| q.abs()).sum();
        let trade_flow = if flow_total > 0.0 {
            self.flow.iter().sum::<f64>() / flow_total
        } else {
            0.0
        };

        // 5. Momentum and volatility come from the rolling window of recent mids.
        self.mids.push_back(mid);
        if self.mids.len() > MID_WINDOW {
            self.mids.pop_front();
        }
        let momentum_bps = (mid / self.mids[0] - 1.0) * BPS;
        let volatility_bps = self.volatility_bps();

        Some(Snapshot {
            best_bid_ticks,
            best_ask_ticks,
            features: [
                spread_bps,
                imbalance_top,
                imbalance_depth,
                last_trade_vs_mid_bps,
                trade_flow,
                momentum_bps,
                volatility_bps,
            ],
        })
    }

    /// Standard deviation of the recent mid-to-mid returns, in basis points.
    fn volatility_bps(&self) -> f64 {
        let returns: Vec<f64> = self
            .mids
            .iter()
            .zip(self.mids.iter().skip(1))
            .map(|(prev, next)| (next / prev - 1.0) * BPS)
            .collect();
        if returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    }
}

/// Total resting quantity at one price level.
fn level_quantity(queue: &VecDeque<Data>) -> f64 {
    queue.iter().map(|d| d.quantity).sum()
}

/// Total resting quantity across the first `n` price levels of one side.
fn depth<'a>(levels: impl Iterator<Item = (&'a Price, &'a VecDeque<Data>)>, n: usize) -> f64 {
    levels.take(n).map(|(_, queue)| level_quantity(queue)).sum()
}

/// `(bid - ask) / (bid + ask)`, or 0 when both are empty.
fn imbalance(bid_quantity: f64, ask_quantity: f64) -> f64 {
    let total = bid_quantity + ask_quantity;
    if total > 0.0 {
        (bid_quantity - ask_quantity) / total
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::OrderBook;

    fn data(id: u64, quantity: f64) -> Data {
        Data {
            id,
            account: 0,
            timestamp: 0,
            quantity,
        }
    }

    #[test]
    fn empty_side_yields_none() {
        let mut book = OrderBook::default();
        book.create_order("SOL-USDC", Side::Buy, 10.0, data(1, 1.0));
        let mut extractor = FeatureExtractor::default();
        assert!(extractor.extract(&book.order_book["SOL-USDC"]).is_none());
    }

    #[test]
    fn computes_spread_and_imbalance() {
        let mut book = OrderBook::default();
        book.create_order("SOL-USDC", Side::Buy, 9.90, data(1, 3.0));
        book.create_order("SOL-USDC", Side::Sell, 10.10, data(2, 1.0));
        let snap = FeatureExtractor::default()
            .extract(&book.order_book["SOL-USDC"])
            .unwrap();
        // Spread is 0.20 on a mid of 10.00, i.e. 200 bps.
        assert!((snap.features[0] - 200.0).abs() < 1e-6);
        assert!((snap.features[1] - 0.5).abs() < 1e-9);
        assert!((snap.features[2] - 0.5).abs() < 1e-9);
        assert_eq!(snap.mid_key(), 2000);
    }

    #[test]
    fn trade_flow_follows_the_aggressor() {
        let mut book = OrderBook::default();
        book.create_order("X", Side::Buy, 9.0, data(1, 1.0));
        book.create_order("X", Side::Sell, 11.0, data(2, 1.0));
        let trade = |aggressor, quantity| Trade {
            symbol: "X".into(),
            price: 1000,
            quantity,
            timestamp: 0,
            buy_order_id: 0,
            buy_account: 0,
            sell_order_id: 0,
            sell_account: 0,
            aggressor,
        };
        let mut extractor = FeatureExtractor::default();
        extractor.on_trade(&trade(Side::Buy, 3.0));
        extractor.on_trade(&trade(Side::Sell, 1.0));
        let snap = extractor.extract(&book.order_book["X"]).unwrap();
        // Net aggressor volume is +2 out of 4 traded.
        assert!((snap.features[4] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn momentum_tracks_mid_changes() {
        let mut extractor = FeatureExtractor::default();
        let mut last = None;
        for price in [10.0, 10.1, 10.2] {
            let mut book = OrderBook::default();
            book.create_order("X", Side::Buy, price - 0.1, data(1, 1.0));
            book.create_order("X", Side::Sell, price + 0.1, data(2, 1.0));
            last = extractor.extract(&book.order_book["X"]);
        }
        assert!(last.unwrap().features[5] > 0.0);
    }
}
