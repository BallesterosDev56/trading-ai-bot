use orderflow_ml::backtest::{BacktestConfig, run};
use orderflow_ml::bot::BotConfig;
use orderflow_ml::pipeline::SignalSource;

fn config(seed: u64) -> BacktestConfig {
    BacktestConfig {
        messages: 20_000,
        seed,
        accounts: 10,
        bot: BotConfig::default(),
        signal: SignalSource::Model,
    }
}

#[test]
fn same_seed_gives_identical_results() {
    let a = run(&config(7));
    let b = run(&config(7));
    assert_eq!(format!("{a}"), format!("{b}"));
}

#[test]
fn model_learns_and_bot_stays_within_limits() {
    let report = run(&config(7));
    assert!(
        report.model_samples > 1_000,
        "model should have learned from many samples"
    );
    assert!(report.model.accuracy.is_finite() && report.model.log_loss.is_finite());
    // Whatever the PnL, the kill switch must have tripped if drawdown went past the limit.
    assert!(report.bot.drawdown <= BotConfig::default().max_drawdown || report.bot.halted);
}

#[test]
fn signal_source_changes_what_the_bot_trades() {
    let model = run(&config(7));
    let baseline = run(&BacktestConfig {
        signal: SignalSource::Baseline,
        ..config(7)
    });
    // Same market, same learning; only the traded signal differs, so the bot's results must differ.
    assert_eq!(model.model_overall.samples, baseline.model_overall.samples);
    assert_ne!(model.bot.trades, baseline.bot.trades);
}
