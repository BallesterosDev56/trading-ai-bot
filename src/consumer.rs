use crate::advisor::Advisor;
use crate::bot::BotConfig;
use crate::client::FeedClient;
use crate::feed::OrderId;
use crate::ml::model::{DEFAULT_LEARNING_RATE, LogisticModel};
use crate::pipeline::Pipeline;
use anyhow::Result;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::{Instant, sleep};
use tracing::{info, warn};

/// The longest we wait between retries when the feed is unreachable.
const MAX_BACKOFF: Duration = Duration::from_secs(10);

/// Settings for the live bot that consumes the feed over HTTP.
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// How often to poll the feed for new messages.
    pub poll_interval: Duration,
    /// How often to log a report, save the model and ask the advisor.
    pub report_interval: Duration,
    /// Where to load the model from on start and save it to periodically.
    pub model_path: Option<PathBuf>,
    pub bot: BotConfig,
    pub advisor: Advisor,
}

/// Polls the feed, runs every message through the pipeline, and reports periodically until Ctrl-C.
/// Uses the `since` query parameter to only fetch messages published after the last one seen.
pub async fn run(client: FeedClient, config: ConsumerConfig) -> Result<()> {
    config.bot.validate()?;
    let mut pipeline = Pipeline::new(load_model(&config), config.bot.clone());
    let mut since: OrderId = 0;
    let mut wait = config.poll_interval;
    let mut last_report = Instant::now();

    let shutdown = crate::shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        // 1. Sleep until the next poll, or stop right away on Ctrl-C.
        tokio::select! {
            _ = &mut shutdown => break,
            _ = sleep(wait) => {}
        }

        // 2. Fetch new messages; back off exponentially while the feed is down.
        match client.orders_since(since).await {
            Ok(messages) => {
                wait = config.poll_interval;
                for msg in &messages {
                    pipeline.on_message(msg);
                    since = msg.id();
                }
            }
            Err(e) => {
                warn!("{e:#}; retrying in {:?}", (wait * 2).min(MAX_BACKOFF));
                wait = (wait * 2).min(MAX_BACKOFF);
            }
        }

        // 3. On a timer: log a summary, persist the model, and let the advisor adjust risk.
        if last_report.elapsed() >= config.report_interval {
            last_report = Instant::now();
            report(&mut pipeline, &config).await;
        }
    }

    info!("shutting down");
    report(&mut pipeline, &config).await;
    Ok(())
}

/// Loads the saved model if there is one, otherwise starts a fresh model.
fn load_model(config: &ConsumerConfig) -> LogisticModel {
    if let Some(path) = &config.model_path
        && path.exists()
    {
        match LogisticModel::load(path) {
            Ok(model) => {
                info!(
                    "loaded model from {} ({} samples)",
                    path.display(),
                    model.samples()
                );
                return model;
            }
            Err(e) => warn!("{e:#}; starting with a fresh model"),
        }
    }
    LogisticModel::new(DEFAULT_LEARNING_RATE)
}

/// Logs the current report, saves the model, and applies the advisor's risk mode.
async fn report(pipeline: &mut Pipeline, config: &ConsumerConfig) {
    let report = pipeline.report();
    info!("\n{report}");

    if let Some(path) = &config.model_path
        && let Err(e) = pipeline.model().save(path)
    {
        warn!("{e:#}");
    }

    // A failed advisor call keeps the current mode: the bot never depends on the network.
    match config.advisor.advise(&report).await {
        Ok(Some(advice)) => {
            info!("advisor says {:?}: {}", advice.mode, advice.reason);
            pipeline.bot_mut().set_mode(advice.mode);
        }
        Ok(None) => {}
        Err(e) => warn!("advisor unavailable: {e:#}"),
    }
}
