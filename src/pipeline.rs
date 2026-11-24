use crate::book::OrderBook;
use crate::bot::{Bot, BotConfig, BotReport};
use crate::feed::OrderMessage;
use crate::ml::features::{FeatureExtractor, FeatureVector};
use crate::ml::metrics::{CumulativeMetrics, RollingMetrics, Summary};
use crate::ml::model::{ImbalanceBaseline, LogisticModel, Predictor};
use crate::trade::match_order;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use tracing::debug;

/// How many recent predictions the rolling quality metrics cover.
const METRICS_WINDOW: usize = 1_000;

/// Which prediction the bot trades on; `Baseline` exists as a control to compare against the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalSource {
    #[default]
    Model,
    Baseline,
}

/// A prediction waiting for the next mid price move to be labeled.
struct Pending {
    features: FeatureVector,
    mid_key: i64,
    p_model: f64,
    p_baseline: f64,
}

/// Everything worth knowing about a run so far; also what the advisor reads.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub messages: u64,
    pub model_samples: u64,
    /// Quality over the last `METRICS_WINDOW` predictions.
    pub model: Summary,
    pub baseline: Summary,
    /// Quality over every prediction since the start (each made before the model saw its label).
    pub model_overall: Summary,
    pub baseline_overall: Summary,
    pub bot: BotReport,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "messages processed : {}", self.messages)?;
        writeln!(f, "model samples      : {}", self.model_samples)?;
        writeln!(
            f,
            "logistic model     : accuracy {:.3}, log-loss {:.4} (last {} predictions)",
            self.model.accuracy, self.model.log_loss, self.model.samples
        )?;
        writeln!(
            f,
            "imbalance baseline : accuracy {:.3}, log-loss {:.4}",
            self.baseline.accuracy, self.baseline.log_loss
        )?;
        writeln!(
            f,
            "overall (model)    : accuracy {:.3}, log-loss {:.4} ({} predictions)",
            self.model_overall.accuracy, self.model_overall.log_loss, self.model_overall.samples
        )?;
        writeln!(
            f,
            "overall (baseline) : accuracy {:.3}, log-loss {:.4}",
            self.baseline_overall.accuracy, self.baseline_overall.log_loss
        )?;
        write!(
            f,
            "bot                : equity {:+.2}, realized {:+.2}, fees {:.2}, trades {}, drawdown {:.2}, mode {:?}{}",
            self.bot.equity,
            self.bot.realized_pnl,
            self.bot.fees_paid,
            self.bot.trades,
            self.bot.drawdown,
            self.bot.mode,
            if self.bot.halted { " (HALTED)" } else { "" }
        )
    }
}

/// The whole market-data path, with no I/O: feed message -> book -> features -> model -> bot.
/// Both the live consumer and the backtest drive this same struct.
pub struct Pipeline {
    book: OrderBook,
    extractors: HashMap<String, FeatureExtractor>,
    pending: HashMap<String, Pending>,
    model: LogisticModel,
    baseline: ImbalanceBaseline,
    model_metrics: RollingMetrics,
    baseline_metrics: RollingMetrics,
    model_overall: CumulativeMetrics,
    baseline_overall: CumulativeMetrics,
    bot: Bot,
    signal: SignalSource,
    messages: u64,
}

impl Pipeline {
    pub fn new(model: LogisticModel, bot_config: BotConfig) -> Self {
        Self {
            book: OrderBook::default(),
            extractors: HashMap::new(),
            pending: HashMap::new(),
            model,
            baseline: ImbalanceBaseline,
            model_metrics: RollingMetrics::new(METRICS_WINDOW),
            baseline_metrics: RollingMetrics::new(METRICS_WINDOW),
            model_overall: CumulativeMetrics::default(),
            baseline_overall: CumulativeMetrics::default(),
            bot: Bot::new(bot_config),
            signal: SignalSource::default(),
            messages: 0,
        }
    }

    /// Chooses which prediction the bot trades on (the model by default).
    pub fn with_signal(mut self, signal: SignalSource) -> Self {
        self.signal = signal;
        self
    }

    pub fn model(&self) -> &LogisticModel {
        &self.model
    }

    pub fn bot_mut(&mut self) -> &mut Bot {
        &mut self.bot
    }

    /// Applies one feed message to the book and runs the ML + bot steps for the symbol it touched.
    pub fn on_message(&mut self, msg: &OrderMessage) {
        self.messages += 1;

        // 1. Update the book; new orders may trade, cancels may remove a resting order.
        let symbol = match msg {
            OrderMessage::New(order) => {
                let trades = match_order(&mut self.book, order);
                let extractor = self.extractors.entry(order.symbol.clone()).or_default();
                for trade in &trades {
                    debug!(
                        "trade {} {} @ {}",
                        trade.symbol, trade.quantity, trade.price
                    );
                    extractor.on_trade(trade);
                }
                order.symbol.clone()
            }
            OrderMessage::Cancel { target_id, .. } => match self.book.remove_order(*target_id) {
                Some((symbol, _)) => symbol,
                // A cancel for an order that already traded (or never existed) changes nothing.
                None => return,
            },
        };

        // 2. Learn and predict for the symbol whose book just changed.
        self.observe(&symbol);
    }

    /// Labels the previous prediction if the mid moved, then makes (and acts on) a new one.
    fn observe(&mut self, symbol: &str) {
        let Some(book) = self.book.order_book.get(symbol) else {
            return;
        };
        let extractor = self.extractors.entry(symbol.to_string()).or_default();
        let Some(snapshot) = extractor.extract(book) else {
            return;
        };
        self.bot
            .on_quote(symbol, snapshot.best_bid(), snapshot.best_ask());

        // 1. The label for the pending prediction is the direction of the next mid price change.
        if let Some(pending) = self.pending.get(symbol) {
            if pending.mid_key == snapshot.mid_key() {
                return;
            }
            let went_up = snapshot.mid_key() > pending.mid_key;
            self.model_metrics.record(pending.p_model, went_up);
            self.baseline_metrics.record(pending.p_baseline, went_up);
            self.model_overall.record(pending.p_model, went_up);
            self.baseline_overall.record(pending.p_baseline, went_up);
            self.model.update(&pending.features, went_up);
            self.pending.remove(symbol);
        }

        // 2. Predict from the fresh snapshot, let the bot react, and wait for the next move.
        let p_model = self.model.predict(&snapshot.features);
        let p_baseline = self.baseline.predict(&snapshot.features);
        let p_signal = match self.signal {
            SignalSource::Model => p_model,
            SignalSource::Baseline => p_baseline,
        };
        self.bot.on_signal(symbol, p_signal, self.model.samples());
        self.pending.insert(
            symbol.to_string(),
            Pending {
                features: snapshot.features,
                mid_key: snapshot.mid_key(),
                p_model,
                p_baseline,
            },
        );
    }

    pub fn report(&self) -> Report {
        Report {
            messages: self.messages,
            model_samples: self.model.samples(),
            model: self.model_metrics.summary(),
            baseline: self.baseline_metrics.summary(),
            model_overall: self.model_overall.summary(),
            baseline_overall: self.baseline_overall.summary(),
            bot: self.bot.report(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feed::{NewOrder, Side};

    fn new(id: u64, side: Side, price: f64, quantity: f64) -> OrderMessage {
        OrderMessage::New(NewOrder {
            id,
            timestamp: 0,
            account: 0,
            symbol: "X".into(),
            side,
            price,
            quantity,
        })
    }

    fn pipeline() -> Pipeline {
        Pipeline::new(LogisticModel::new(0.05), BotConfig::default())
    }

    #[test]
    fn labels_after_the_mid_moves() {
        let mut p = pipeline();
        p.on_message(&new(1, Side::Buy, 9.9, 1.0));
        p.on_message(&new(2, Side::Sell, 10.1, 1.0));
        // First snapshot only makes a prediction; nothing to label yet.
        assert_eq!(p.report().model_samples, 0);
        // A better bid moves the mid up, which labels the first prediction.
        p.on_message(&new(3, Side::Buy, 10.0, 1.0));
        assert_eq!(p.report().model_samples, 1);
        assert_eq!(p.report().model.samples, 1);
    }

    #[test]
    fn unchanged_mid_does_not_label() {
        let mut p = pipeline();
        p.on_message(&new(1, Side::Buy, 9.9, 1.0));
        p.on_message(&new(2, Side::Sell, 10.1, 1.0));
        // Extra size behind the best bid does not move the mid.
        p.on_message(&new(3, Side::Buy, 9.9, 5.0));
        assert_eq!(p.report().model_samples, 0);
    }

    #[test]
    fn cancel_of_unknown_order_is_ignored() {
        let mut p = pipeline();
        p.on_message(&OrderMessage::Cancel {
            id: 1,
            timestamp: 0,
            account: 0,
            target_id: 99,
        });
        assert_eq!(p.report().messages, 1);
    }

    #[test]
    fn cancel_changes_the_book() {
        let mut p = pipeline();
        p.on_message(&new(1, Side::Buy, 9.9, 1.0));
        p.on_message(&new(2, Side::Sell, 10.1, 1.0));
        p.on_message(&new(3, Side::Buy, 10.0, 1.0));
        p.on_message(&OrderMessage::Cancel {
            id: 4,
            timestamp: 0,
            account: 0,
            target_id: 3,
        });
        // The mid fell back to 10.0 after the cancel, so a second sample was labeled (down move).
        assert_eq!(p.report().model_samples, 2);
    }
}
