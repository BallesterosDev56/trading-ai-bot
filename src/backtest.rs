use crate::bot::BotConfig;
use crate::feed::FeedState;
use crate::ml::model::{DEFAULT_LEARNING_RATE, LogisticModel};
use crate::pipeline::{Pipeline, Report, SignalSource};

/// Settings for a deterministic, in-process simulation (no HTTP, no waiting).
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub messages: usize,
    pub seed: u64,
    pub accounts: u32,
    pub bot: BotConfig,
    /// Which prediction the bot trades on.
    pub signal: SignalSource,
}

/// Generates `messages` feed messages from a seeded simulator and runs them through the pipeline.
/// The same config always produces the same report.
pub fn run(config: &BacktestConfig) -> Report {
    let mut feed = FeedState::new(config.accounts, Some(config.seed));
    let mut pipeline = Pipeline::new(
        LogisticModel::new(DEFAULT_LEARNING_RATE),
        config.bot.clone(),
    )
    .with_signal(config.signal);
    for _ in 0..config.messages {
        let msg = feed.next_message();
        pipeline.on_message(&msg);
    }
    pipeline.report()
}

/// Aggregate view of several seeded runs.
#[derive(Debug, Clone)]
pub struct Sweep {
    pub runs: Vec<(u64, Report)>,
}

/// Runs the backtest for `count` consecutive seeds starting at `config.seed`.
pub fn sweep(config: &BacktestConfig, count: u64) -> Sweep {
    let runs = (0..count)
        .map(|offset| {
            let seed = config.seed + offset;
            (
                seed,
                run(&BacktestConfig {
                    seed,
                    ..config.clone()
                }),
            )
        })
        .collect();
    Sweep { runs }
}

impl Sweep {
    /// How many runs the model beat the baseline on whole-run log-loss.
    pub fn model_wins(&self) -> usize {
        self.runs
            .iter()
            .filter(|(_, r)| r.model_overall.log_loss < r.baseline_overall.log_loss)
            .count()
    }

    /// Mean and sample standard deviation of a per-run value.
    fn mean_std(&self, value: impl Fn(&Report) -> f64) -> (f64, f64) {
        let values: Vec<f64> = self.runs.iter().map(|(_, r)| value(r)).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = if values.len() > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        (mean, var.sqrt())
    }

    /// Renders one row per seed plus a summary, as GitHub-flavored markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| seed | predictions | model acc | baseline acc | model log-loss | baseline log-loss | trades | PnL | halted |\n");
        out.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n");
        for (seed, r) in &self.runs {
            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} | {:.4} | {:.4} | {} | {:+.2} | {} |\n",
                seed,
                r.model_overall.samples,
                r.model_overall.accuracy,
                r.baseline_overall.accuracy,
                r.model_overall.log_loss,
                r.baseline_overall.log_loss,
                r.bot.trades,
                r.bot.equity,
                if r.bot.halted { "yes" } else { "no" },
            ));
        }
        let (acc, acc_sd) = self.mean_std(|r| r.model_overall.accuracy);
        let (base_acc, base_acc_sd) = self.mean_std(|r| r.baseline_overall.accuracy);
        let (loss, loss_sd) = self.mean_std(|r| r.model_overall.log_loss);
        let (base_loss, base_loss_sd) = self.mean_std(|r| r.baseline_overall.log_loss);
        let (pnl, pnl_sd) = self.mean_std(|r| r.bot.equity);
        let positive = self.runs.iter().filter(|(_, r)| r.bot.equity > 0.0).count();
        let halted = self.runs.iter().filter(|(_, r)| r.bot.halted).count();
        let n = self.runs.len();
        out.push_str(&format!(
            "\n{n} runs. Model accuracy {acc:.3} ± {acc_sd:.3} vs baseline {base_acc:.3} ± {base_acc_sd:.3}. \
Model log-loss {loss:.4} ± {loss_sd:.4} vs baseline {base_loss:.4} ± {base_loss_sd:.4}; model better in {}/{n} runs. \
PnL {pnl:+.2} ± {pnl_sd:.2}, positive in {positive}/{n} runs, kill switch tripped in {halted}/{n}.\n",
            self.model_wins(),
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_runs_consecutive_seeds_and_renders_a_table() {
        let config = BacktestConfig {
            messages: 5_000,
            seed: 10,
            accounts: 5,
            bot: BotConfig::default(),
            signal: SignalSource::Model,
        };
        let sweep = sweep(&config, 3);
        let seeds: Vec<_> = sweep.runs.iter().map(|(s, _)| *s).collect();
        assert_eq!(seeds, vec![10, 11, 12]);
        let table = sweep.to_markdown();
        assert!(table.contains("| 11 |"));
        assert!(table.contains("3 runs."));
    }
}
