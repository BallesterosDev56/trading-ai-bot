use crate::book::QTY_EPS;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::{info, warn};

/// Tunable limits for the paper-trading bot.
#[derive(Debug, Clone)]
pub struct BotConfig {
    /// Trade only when P(up) is at least this (long) or at most `1 - threshold` (short).
    pub threshold: f64,
    /// Value traded per signal, in quote currency (USDC), so every symbol carries similar risk.
    pub trade_notional: f64,
    /// Largest absolute position value per symbol, in quote currency.
    pub max_notional: f64,
    /// Taker fee charged on every fill, in basis points of notional.
    pub fee_bps: f64,
    /// Drop from peak equity (in quote currency) that trips the kill switch.
    pub max_drawdown: f64,
    /// Number of samples the model must learn from before the bot may trade.
    pub warmup_samples: u64,
    /// Skip signals whose expected move would not cover the spread and fees (off by default; see docs/technical-report.md).
    pub cost_gate: bool,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            threshold: 0.55,
            trade_notional: 100.0,
            max_notional: 500.0,
            fee_bps: 1.0,
            max_drawdown: 50.0,
            warmup_samples: 500,
            cost_gate: false,
        }
    }
}

impl BotConfig {
    /// Rejects settings that would make the bot misbehave.
    pub fn validate(&self) -> Result<()> {
        if !(self.threshold > 0.5 && self.threshold < 1.0) {
            bail!("threshold must be between 0.5 and 1.0 (exclusive)");
        }
        if !(self.trade_notional > 0.0 && self.max_notional >= self.trade_notional) {
            bail!("trade_notional must be positive and not larger than max_notional");
        }
        if self.fee_bps < 0.0 || self.max_drawdown <= 0.0 {
            bail!("fee_bps must be >= 0 and max_drawdown must be > 0");
        }
        Ok(())
    }
}

/// How much risk the bot may take right now; the advisor can change this at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskMode {
    Normal,
    /// Half the usual trade size and position limit.
    Reduced,
    /// No trading; open positions are closed.
    Paused,
}

impl RiskMode {
    fn scale(self) -> f64 {
        match self {
            RiskMode::Normal => 1.0,
            RiskMode::Reduced => 0.5,
            RiskMode::Paused => 0.0,
        }
    }
}

/// Weight of the newest mid move in the running average move size.
const MOVE_EWMA_ALPHA: f64 = 0.01;

/// Running average size of mid price moves for one symbol.
#[derive(Debug, Clone, Copy)]
struct MoveStats {
    last_mid: f64,
    avg_move_bps: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Position {
    qty: f64,
    avg_price: f64,
}

/// A snapshot of the bot's state, for logs and for the advisor.
#[derive(Debug, Clone, Serialize)]
pub struct BotReport {
    pub equity: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub trades: u64,
    pub drawdown: f64,
    pub mode: RiskMode,
    pub halted: bool,
    pub positions: BTreeMap<String, f64>,
}

/// A paper-trading bot: turns P(up) signals into taker fills at the best bid/ask, with risk limits.
#[derive(Debug)]
pub struct Bot {
    config: BotConfig,
    mode: RiskMode,
    halted: bool,
    positions: BTreeMap<String, Position>,
    /// Latest (bid, ask) per symbol; used for fills and marking positions.
    quotes: BTreeMap<String, (f64, f64)>,
    moves: BTreeMap<String, MoveStats>,
    realized: f64,
    fees: f64,
    peak_equity: f64,
    trades: u64,
}

impl Bot {
    pub fn new(config: BotConfig) -> Self {
        Self {
            config,
            mode: RiskMode::Normal,
            halted: false,
            positions: BTreeMap::new(),
            quotes: BTreeMap::new(),
            moves: BTreeMap::new(),
            realized: 0.0,
            fees: 0.0,
            peak_equity: 0.0,
            trades: 0,
        }
    }

    pub fn config(&self) -> &BotConfig {
        &self.config
    }

    /// Records a new best bid/ask for a symbol and re-checks the kill switch.
    pub fn on_quote(&mut self, symbol: &str, bid: f64, ask: f64) {
        self.quotes.insert(symbol.to_string(), (bid, ask));
        self.track_move(symbol, (bid + ask) / 2.0);
        self.peak_equity = self.peak_equity.max(self.equity());
        if !self.halted && self.drawdown() > self.config.max_drawdown {
            warn!(
                "max drawdown exceeded ({:.2}); halting and closing positions",
                self.drawdown()
            );
            self.halted = true;
            self.flatten();
        }
    }

    /// Reacts to a prediction: may open or grow a position in the signal's direction.
    pub fn on_signal(&mut self, symbol: &str, p_up: f64, model_samples: u64) {
        // 1. Never trade while halted, paused, or before the model has seen enough data.
        let scale = self.mode.scale();
        if self.halted || scale == 0.0 || model_samples < self.config.warmup_samples {
            return;
        }
        if self.config.cost_gate && !self.edge_covers_costs(symbol, p_up) {
            return;
        }
        let Some(&(bid, ask)) = self.quotes.get(symbol) else {
            return;
        };
        let mid = (bid + ask) / 2.0;
        let position = self.positions.get(symbol).map_or(0.0, |p| p.qty);
        let limit = self.config.max_notional * scale / mid;
        let size = self.config.trade_notional * scale / mid;

        // 2. Only act on confident signals, and only ever move towards the signal.
        let delta = if p_up >= self.config.threshold {
            ((position + size).min(limit) - position).max(0.0)
        } else if p_up <= 1.0 - self.config.threshold {
            ((position - size).max(-limit) - position).min(0.0)
        } else {
            0.0
        };
        if delta.abs() > QTY_EPS {
            self.trade(symbol, delta);
        }
    }

    /// Changes the risk mode; pausing closes everything immediately.
    pub fn set_mode(&mut self, mode: RiskMode) {
        if mode == self.mode {
            return;
        }
        info!("risk mode: {:?} -> {:?}", self.mode, mode);
        self.mode = mode;
        if mode == RiskMode::Paused {
            self.flatten();
        }
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Realized PnL, minus fees, plus open PnL marked at the mid price.
    pub fn equity(&self) -> f64 {
        let open: f64 = self
            .positions
            .iter()
            .filter_map(|(symbol, pos)| {
                let (bid, ask) = self.quotes.get(symbol)?;
                Some(pos.qty * ((bid + ask) / 2.0 - pos.avg_price))
            })
            .sum();
        self.realized - self.fees + open
    }

    pub fn report(&self) -> BotReport {
        BotReport {
            equity: self.equity(),
            realized_pnl: self.realized,
            fees_paid: self.fees,
            trades: self.trades,
            drawdown: self.drawdown(),
            mode: self.mode,
            halted: self.halted,
            positions: self
                .positions
                .iter()
                .filter(|(_, p)| p.qty.abs() > QTY_EPS)
                .map(|(s, p)| (s.clone(), p.qty))
                .collect(),
        }
    }

    /// Updates the running average of how far the mid moves each time it changes.
    fn track_move(&mut self, symbol: &str, mid: f64) {
        match self.moves.get_mut(symbol) {
            None => {
                self.moves.insert(
                    symbol.to_string(),
                    MoveStats {
                        last_mid: mid,
                        avg_move_bps: 0.0,
                    },
                );
            }
            Some(stats) if mid != stats.last_mid => {
                let move_bps = (mid / stats.last_mid - 1.0).abs() * 10_000.0;
                stats.avg_move_bps = if stats.avg_move_bps == 0.0 {
                    move_bps
                } else {
                    stats.avg_move_bps * (1.0 - MOVE_EWMA_ALPHA) + move_bps * MOVE_EWMA_ALPHA
                };
                stats.last_mid = mid;
            }
            Some(_) => {}
        }
    }

    /// Expected move is `|2p - 1|` times the typical move; it must beat the round-trip spread plus fees.
    fn edge_covers_costs(&self, symbol: &str, p_up: f64) -> bool {
        let (Some(&(bid, ask)), Some(stats)) = (self.quotes.get(symbol), self.moves.get(symbol))
        else {
            return false;
        };
        let mid = (bid + ask) / 2.0;
        let cost_bps = (ask - bid) / mid * 10_000.0 + 2.0 * self.config.fee_bps;
        (2.0 * p_up - 1.0).abs() * stats.avg_move_bps > cost_bps
    }

    fn drawdown(&self) -> f64 {
        self.peak_equity - self.equity()
    }

    /// Closes every open position at the latest quotes.
    fn flatten(&mut self) {
        let open: Vec<(String, f64)> = self
            .positions
            .iter()
            .filter(|(_, p)| p.qty.abs() > QTY_EPS)
            .map(|(s, p)| (s.clone(), p.qty))
            .collect();
        for (symbol, qty) in open {
            self.trade(&symbol, -qty);
        }
    }

    /// Fills a signed quantity at the best ask (buying) or best bid (selling), charging the fee.
    fn trade(&mut self, symbol: &str, signed_qty: f64) {
        let Some(&(bid, ask)) = self.quotes.get(symbol) else {
            return;
        };
        let price = if signed_qty > 0.0 { ask } else { bid };
        self.fees += price * signed_qty.abs() * self.config.fee_bps / 10_000.0;
        self.apply_fill(symbol, signed_qty, price);
        self.trades += 1;
        info!("paper trade {symbol}: {signed_qty:+.2} @ {price:.2}");
    }

    /// Updates the position and realized PnL for one fill, handling adds, reductions and flips.
    fn apply_fill(&mut self, symbol: &str, signed_qty: f64, price: f64) {
        let pos = self.positions.entry(symbol.to_string()).or_default();
        if pos.qty == 0.0 || pos.qty.signum() == signed_qty.signum() {
            // 1. Adding to (or opening) a position: blend the average entry price.
            let total = pos.qty.abs() + signed_qty.abs();
            pos.avg_price = (pos.avg_price * pos.qty.abs() + price * signed_qty.abs()) / total;
            pos.qty += signed_qty;
            return;
        }
        // 2. Reducing: realize PnL on the part that closes existing exposure.
        let closing = signed_qty.abs().min(pos.qty.abs());
        self.realized += closing * (price - pos.avg_price) * pos.qty.signum();
        let new_qty = pos.qty + signed_qty;
        if new_qty.abs() <= QTY_EPS {
            *pos = Position::default();
        } else if new_qty.signum() != pos.qty.signum() {
            // 3. Flipped through zero: the remainder opens a fresh position at this price.
            *pos = Position {
                qty: new_qty,
                avg_price: price,
            };
        } else {
            pos.qty = new_qty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config where one unit at a mid of 10.0 is one trade (10 notional) and 5 units is the limit.
    fn config() -> BotConfig {
        BotConfig {
            trade_notional: 10.0,
            max_notional: 50.0,
            fee_bps: 0.0,
            warmup_samples: 0,
            cost_gate: false,
            ..BotConfig::default()
        }
    }

    fn bot() -> Bot {
        Bot::new(config())
    }

    #[test]
    fn config_validation() {
        assert!(BotConfig::default().validate().is_ok());
        assert!(
            BotConfig {
                threshold: 0.4,
                ..BotConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            BotConfig {
                max_notional: 50.0,
                ..BotConfig::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn does_not_trade_during_warmup() {
        let mut bot = Bot::new(BotConfig {
            warmup_samples: 100,
            ..config()
        });
        bot.on_quote("X", 9.9, 10.1);
        bot.on_signal("X", 0.9, 10);
        assert_eq!(bot.report().trades, 0);
    }

    #[test]
    fn buys_at_ask_and_sells_at_bid() {
        let mut bot = bot();
        bot.on_quote("X", 9.9, 10.1);
        bot.on_signal("X", 0.9, 0);
        assert_eq!(bot.report().positions["X"], 1.0);
        // Immediately marked at mid (10.0) against a 10.1 entry.
        assert!((bot.equity() + 0.1).abs() < 1e-9);
        bot.on_signal("X", 0.1, 0);
        assert!(bot.report().positions.is_empty());
        // Round trip paid the full spread: bought 10.1, sold 9.9.
        assert!((bot.report().realized_pnl + 0.2).abs() < 1e-9);
    }

    #[test]
    fn sizing_is_in_notional_so_symbols_carry_similar_risk() {
        let mut bot = bot();
        bot.on_quote("BIG", 999.0, 1001.0);
        bot.on_signal("BIG", 0.9, 0);
        // 10 notional at a mid of 1000 is 0.01 units.
        assert!((bot.report().positions["BIG"] - 0.01).abs() < 1e-9);
    }

    #[test]
    fn respects_max_position() {
        let mut bot = bot();
        bot.on_quote("X", 9.9, 10.1);
        for _ in 0..20 {
            bot.on_signal("X", 0.9, 0);
        }
        assert!((bot.report().positions["X"] - 5.0).abs() < 1e-9);
        assert_eq!(bot.report().trades, 5);
    }

    #[test]
    fn ignores_weak_signals() {
        let mut bot = bot();
        bot.on_quote("X", 9.9, 10.1);
        bot.on_signal("X", 0.52, 0);
        bot.on_signal("X", 0.48, 0);
        assert_eq!(bot.report().trades, 0);
    }

    #[test]
    fn fees_reduce_equity() {
        let mut bot = Bot::new(BotConfig {
            fee_bps: 100.0,
            ..config()
        });
        bot.on_quote("X", 10.0, 10.0);
        bot.on_signal("X", 0.9, 0);
        // 1% of a 10.0 notional.
        assert!((bot.report().fees_paid - 0.1).abs() < 1e-9);
    }

    #[test]
    fn kill_switch_halts_and_flattens() {
        let mut bot = Bot::new(BotConfig {
            max_drawdown: 1.0,
            ..config()
        });
        bot.on_quote("X", 10.0, 10.0);
        for _ in 0..5 {
            bot.on_signal("X", 0.9, 0);
        }
        // Price collapses by 1.0 on a 5.0 position: a 5.0 drawdown, above the 1.0 limit.
        bot.on_quote("X", 9.0, 9.0);
        assert!(bot.is_halted());
        assert!(bot.report().positions.is_empty());
        let trades = bot.report().trades;
        bot.on_signal("X", 0.9, 0);
        assert_eq!(bot.report().trades, trades);
    }

    #[test]
    fn paused_mode_closes_positions_and_blocks_trading() {
        let mut bot = bot();
        bot.on_quote("X", 9.9, 10.1);
        bot.on_signal("X", 0.9, 0);
        bot.set_mode(RiskMode::Paused);
        assert!(bot.report().positions.is_empty());
        bot.on_signal("X", 0.9, 0);
        assert!(bot.report().positions.is_empty());
    }

    #[test]
    fn reduced_mode_halves_limits() {
        let mut bot = bot();
        bot.set_mode(RiskMode::Reduced);
        bot.on_quote("X", 9.9, 10.1);
        for _ in 0..20 {
            bot.on_signal("X", 0.9, 0);
        }
        assert!((bot.report().positions["X"] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn flipping_through_zero_resets_entry_price() {
        let mut bot = bot();
        bot.apply_fill("X", 2.0, 10.0);
        bot.apply_fill("X", -5.0, 11.0);
        // Closed 2 at +1.0 each; the remaining -3 opened at 11.0.
        assert!((bot.report().realized_pnl - 2.0).abs() < 1e-9);
        assert_eq!(bot.positions["X"].qty, -3.0);
        assert_eq!(bot.positions["X"].avg_price, 11.0);
    }

    #[test]
    fn cost_gate_blocks_trades_when_moves_are_smaller_than_the_spread() {
        let mut bot = Bot::new(BotConfig {
            cost_gate: true,
            ..config()
        });
        // Spread is 200 bps but the mid only wobbles by about 10 bps.
        bot.on_quote("X", 9.9, 10.1);
        bot.on_quote("X", 9.905, 10.105);
        bot.on_quote("X", 9.9, 10.1);
        bot.on_signal("X", 0.95, 0);
        assert_eq!(bot.report().trades, 0);
    }

    #[test]
    fn cost_gate_allows_trades_when_moves_beat_the_spread() {
        let mut bot = Bot::new(BotConfig {
            cost_gate: true,
            ..config()
        });
        // Spread is 20 bps and the mid moves by about 100 bps each step.
        bot.on_quote("X", 9.99, 10.01);
        bot.on_quote("X", 10.49, 10.51);
        bot.on_quote("X", 9.99, 10.01);
        bot.on_signal("X", 0.95, 0);
        assert_eq!(bot.report().trades, 1);
    }

    #[test]
    fn cost_gate_blocks_trades_before_any_move_is_observed() {
        let mut bot = Bot::new(BotConfig {
            cost_gate: true,
            ..config()
        });
        bot.on_quote("X", 9.99, 10.01);
        bot.on_signal("X", 0.95, 0);
        assert_eq!(bot.report().trades, 0);
    }
}
