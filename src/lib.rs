pub mod advisor;
pub mod backtest;
pub mod book;
pub mod bot;
pub mod client;
pub mod consumer;
pub mod feed;
pub mod ml;
pub mod pipeline;
pub mod trade;

/// Resolves when the process receives Ctrl-C, so long-running loops can stop cleanly.
pub async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::error!("failed to listen for Ctrl-C: {e}");
        // If we can't listen for the signal, never resolve instead of shutting down by mistake.
        std::future::pending::<()>().await;
    }
}
