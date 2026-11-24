mod cli;

use anyhow::Result;
use clap::Parser;
use cli::{AdvisorKind, Cli, Command};
use orderflow_ml::advisor::{Advisor, ClaudeAdvisor};
use orderflow_ml::backtest::{self, BacktestConfig};
use orderflow_ml::client::FeedClient;
use orderflow_ml::consumer::{self, ConsumerConfig};
use orderflow_ml::feed;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

/// The entry point of the application: parses the CLI and runs the chosen command.
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Logs go to stderr so stdout stays clean for results; set RUST_LOG to change verbosity (default: info).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // 2. Every command that talks to the feed shares one client.
    let cli = Cli::parse();
    let client = FeedClient::new(&cli.feed_url);

    match cli.command {
        Command::Feed {
            addr,
            num_accounts,
            rate,
            seed,
        } => {
            feed::start_feed(addr, num_accounts, rate, seed).await?;
        }
        Command::Submit {
            account,
            symbol,
            side,
            price,
            quantity,
        } => {
            let id = client
                .submit_order(account, &symbol.to_uppercase(), side, price, quantity)
                .await?;
            println!("Order submitted to feed successfully: id {id}");
        }
        Command::Cancel { account, order_id } => {
            let id = client.cancel(account, order_id).await?;
            println!("Cancel submitted to feed successfully: id {id}");
        }
        Command::Orders { n } => {
            let messages = client.last_orders(n).await?;
            println!("Most recent {} messages:", messages.len());
            println!("{:#?}", messages);
        }
        Command::Bot(args) => {
            let advisor = match args.advisor {
                AdvisorKind::Off => Advisor::Off,
                AdvisorKind::Claude => Advisor::Claude(ClaudeAdvisor::from_env()?),
            };
            let config = ConsumerConfig {
                poll_interval: Duration::from_millis(args.poll_ms),
                report_interval: Duration::from_secs(args.report_secs),
                model_path: args.model_path,
                bot: args.bot.into(),
                advisor,
            };
            consumer::run(client, config).await?;
        }
        Command::Backtest {
            messages,
            seed,
            seeds,
            accounts,
            signal,
            bot,
        } => {
            let config = BacktestConfig {
                messages,
                seed,
                accounts,
                bot: bot.into(),
                signal: signal.into(),
            };
            config.bot.validate()?;
            if seeds > 1 {
                println!("{}", backtest::sweep(&config, seeds).to_markdown());
            } else {
                println!("{}", backtest::run(&config));
            }
        }
    }
    Ok(())
}
