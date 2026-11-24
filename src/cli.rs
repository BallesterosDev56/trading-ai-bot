use clap::{Args, Parser, Subcommand, ValueEnum};
use orderflow_ml::bot::BotConfig;
use orderflow_ml::feed::Side;
use orderflow_ml::pipeline::SignalSource;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Order-flow simulator, online ML pipeline and paper-trading bot.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Base URL of the order feed (for every command except `feed` and `backtest`).
    #[arg(
        long,
        global = true,
        env = "FEED_URL",
        default_value = "http://127.0.0.1:3000"
    )]
    pub feed_url: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the simulated order feed and its HTTP API.
    Feed {
        /// Address the feed API listens on.
        #[arg(long, default_value = "127.0.0.1:3000")]
        addr: SocketAddr,
        /// The number of simulated accounts placing orders on the feed.
        #[arg(long, default_value_t = 10)]
        num_accounts: u32,
        /// The average number of feed messages generated per second.
        #[arg(long, default_value_t = 2.0)]
        rate: f64,
        /// Seed for the random generator, for reproducible feeds.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Submit a new limit order to the feed.
    Submit {
        account: u32,
        symbol: String,
        /// Buy or Sell (also accepts bid/ask/b/s).
        side: Side,
        price: f64,
        quantity: f64,
    },
    /// Submit a cancel for a previously placed order.
    Cancel { account: u32, order_id: u64 },
    /// Fetch and print the most recent feed messages.
    Orders {
        #[arg(default_value_t = 10)]
        n: usize,
    },
    /// Run the live bot: rebuild the book, learn online, and paper trade the model's signals.
    Bot(BotCommand),
    /// Run the pipeline and bot on a seeded in-process simulation, as fast as possible.
    Backtest {
        /// How many feed messages to simulate.
        #[arg(long, default_value_t = 50_000)]
        messages: usize,
        /// Seed for the simulated feed; the same seed gives the same result.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Run this many consecutive seeds (starting at --seed) and print a markdown table.
        #[arg(long, default_value_t = 1)]
        seeds: u64,
        #[arg(long, default_value_t = 10)]
        accounts: u32,
        /// Which prediction the bot trades on; `baseline` is a control to compare against the model.
        #[arg(long, value_enum, default_value_t = SignalArg::Model)]
        signal: SignalArg,
        #[command(flatten)]
        bot: BotArgs,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum SignalArg {
    Model,
    Baseline,
}

impl From<SignalArg> for SignalSource {
    fn from(arg: SignalArg) -> Self {
        match arg {
            SignalArg::Model => SignalSource::Model,
            SignalArg::Baseline => SignalSource::Baseline,
        }
    }
}

#[derive(Args, Debug)]
pub struct BotCommand {
    /// How often to poll the feed, in milliseconds.
    #[arg(long, default_value_t = 500)]
    pub poll_ms: u64,
    /// How often to log a report, save the model and consult the advisor, in seconds.
    #[arg(long, default_value_t = 30)]
    pub report_secs: u64,
    /// Load the model from this file on start and save it there periodically.
    #[arg(long)]
    pub model_path: Option<PathBuf>,
    /// Who adjusts the bot's risk mode. `claude` needs ANTHROPIC_API_KEY.
    #[arg(long, value_enum, default_value_t = AdvisorKind::Off)]
    pub advisor: AdvisorKind,
    #[command(flatten)]
    pub bot: BotArgs,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum AdvisorKind {
    Off,
    Claude,
}

/// Bot limits shared by `bot` and `backtest`.
#[derive(Args, Debug)]
pub struct BotArgs {
    /// Trade only when P(up) >= threshold (long) or <= 1 - threshold (short).
    #[arg(long, default_value_t = 0.55)]
    pub threshold: f64,
    /// Value traded per signal, in USDC.
    #[arg(long, default_value_t = 100.0)]
    pub trade_notional: f64,
    /// Largest absolute position value per symbol, in USDC.
    #[arg(long, default_value_t = 500.0)]
    pub max_notional: f64,
    /// Taker fee per fill, in basis points of notional.
    #[arg(long, default_value_t = 1.0)]
    pub fee_bps: f64,
    /// Drawdown from peak equity that halts the bot.
    #[arg(long, default_value_t = 50.0)]
    pub max_drawdown: f64,
    /// Samples the model must learn from before the bot may trade.
    #[arg(long, default_value_t = 500)]
    pub warmup: u64,
    /// Skip signals whose expected move would not cover the spread and fees.
    #[arg(long)]
    pub cost_gate: bool,
}

impl From<BotArgs> for BotConfig {
    fn from(args: BotArgs) -> Self {
        BotConfig {
            threshold: args.threshold,
            trade_notional: args.trade_notional,
            max_notional: args.max_notional,
            fee_bps: args.fee_bps,
            max_drawdown: args.max_drawdown,
            warmup_samples: args.warmup,
            cost_gate: args.cost_gate,
        }
    }
}
