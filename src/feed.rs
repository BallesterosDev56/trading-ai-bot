use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;
use tracing::info;

/// The trading pairs available on the feed, along with their initial mid prices.
/// Prices are quoted in the second token of the pair (e.g. USDC per ETH).
pub const SYMBOLS: [(&str, f64); 3] = [
    ("SOL-USDC", 10.0),
    ("ETH-USDC", 100.0),
    ("BTC-USDC", 1000.0),
];

/// How many recent open orders the generator remembers as candidates for
/// random cancellation.
const CANCEL_CANDIDATE_WINDOW: usize = 50;

/// The probability that any given generated message is a cancel instead of a
/// new order.
const CANCEL_PROBABILITY: f64 = 0.15;

/// How many messages the feed keeps in memory; older ones are dropped.
const MAX_RETAINED_MESSAGES: usize = 100_000;

/// Initializes and starts the order feed simulation.
/// Spawns the order generator and serves the HTTP API until Ctrl-C.
pub async fn start_feed(
    addr: SocketAddr,
    num_accounts: u32,
    rate: f64,
    seed: Option<u64>,
) -> Result<()> {
    // 1. The feed state is shared between the generator task and the HTTP handlers.
    let shared_state = Arc::new(Mutex::new(FeedState::new(num_accounts, seed)));

    // 2. The generator runs in its own task so it never blocks the API.
    let generator_state = Arc::clone(&shared_state);
    tokio::spawn(async move {
        produce_orders(generator_state, rate).await;
    });

    // 3. The API server runs in the foreground until we are told to stop.
    run_server(shared_state, addr).await
}

/// Represents the side of an order in the market.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// Returns the other side of the market.
    pub fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

/// This implementation allows parsing a string into a Side.
/// It is case-insensitive and supports single-letter shorthands.
impl FromStr for Side {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BUY" | "BID" | "B" => Ok(Side::Buy),
            "SELL" | "ASK" | "S" => Ok(Side::Sell),
            _ => Err(format!("'{}' is not a valid side", s)),
        }
    }
}

/// A type alias for account identifiers for clarity.
pub type AccountId = u32;

/// A type alias for order/message identifiers. IDs are strictly increasing
/// across all messages on the feed and start at 1.
pub type OrderId = u64;

/// A new limit order entering the market.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NewOrder {
    pub id: OrderId,
    pub timestamp: u64,
    pub account: AccountId,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
}

/// Defines the types of messages published on the order feed.
/// The JSON shape is `{"New": {...}}` or `{"Cancel": {...}}`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OrderMessage {
    /// A new limit order entering the market.
    New(NewOrder),
    /// A request to cancel a previously placed order.
    /// `target_id` refers to the `id` of the order to cancel.
    Cancel {
        id: OrderId,
        timestamp: u64,
        account: AccountId,
        target_id: OrderId,
    },
}

impl OrderMessage {
    /// Returns the message's own feed sequence ID.
    pub fn id(&self) -> OrderId {
        match self {
            OrderMessage::New(order) => order.id,
            OrderMessage::Cancel { id, .. } => *id,
        }
    }
}

/// Represents the entire state of the order feed at a given moment.
pub struct FeedState {
    /// The most recent messages, in feed order (bounded by `MAX_RETAINED_MESSAGES`).
    messages: VecDeque<OrderMessage>,
    /// The ID to be assigned to the next message.
    next_id: OrderId,
    /// The current mid price for each symbol; each drifts as a random walk.
    mids: HashMap<String, f64>,
    /// Recently generated (order id, account) pairs, used to pick cancel targets.
    cancel_candidates: Vec<(OrderId, AccountId)>,
    /// The number of simulated accounts placing orders.
    num_accounts: u32,
    /// Random source; seedable so simulations are reproducible.
    rng: StdRng,
}

impl FeedState {
    /// Creates a new feed state with the initial mid price for each symbol.
    pub fn new(num_accounts: u32, seed: Option<u64>) -> Self {
        let mut mids = HashMap::new();
        for (symbol, mid) in SYMBOLS {
            mids.insert(symbol.to_string(), mid);
        }
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        FeedState {
            messages: VecDeque::new(),
            next_id: 1,
            mids,
            cancel_candidates: Vec::new(),
            num_accounts: num_accounts.max(1),
            rng,
        }
    }

    /// Hands out the next strictly increasing message ID.
    fn allocate_id(&mut self) -> OrderId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Publishes a message, dropping the oldest one if the buffer is full.
    pub fn push(&mut self, msg: OrderMessage) {
        if self.messages.len() >= MAX_RETAINED_MESSAGES {
            self.messages.pop_front();
        }
        self.messages.push_back(msg);
    }

    /// Generates the next random feed message without publishing it.
    /// Most messages are new limit orders priced around a per-symbol drifting
    /// mid price; occasionally a recent order is cancelled instead.
    pub fn next_message(&mut self) -> OrderMessage {
        let id = self.allocate_id();
        let timestamp = now_millis();
        let rng = &mut self.rng;

        // 1. Occasionally cancel a recent order instead of placing a new one.
        if !self.cancel_candidates.is_empty() && rng.gen_bool(CANCEL_PROBABILITY) {
            let idx = rng.gen_range(0..self.cancel_candidates.len());
            let (target_id, account) = self.cancel_candidates.swap_remove(idx);
            return OrderMessage::Cancel {
                id,
                timestamp,
                account,
                target_id,
            };
        }

        // 2. Pick a random symbol and let its mid price drift slightly.
        let (symbol, _) = SYMBOLS[rng.gen_range(0..SYMBOLS.len())];
        let mid = self.mids.get_mut(symbol).expect("every symbol has a mid");
        *mid *= 1.0 + rng.gen_range(-0.002..0.002);

        // 3. Price the order randomly around the mid so buys and sells frequently cross.
        let price = round2(*mid * (1.0 + rng.gen_range(-0.005..0.005)));
        let quantity = round1(rng.gen_range(1.0..10.0));
        let side = if rng.gen_bool(0.5) {
            Side::Buy
        } else {
            Side::Sell
        };
        let account = rng.gen_range(0..self.num_accounts);

        // 4. Remember this order as a potential cancel target.
        self.cancel_candidates.push((id, account));
        if self.cancel_candidates.len() > CANCEL_CANDIDATE_WINDOW {
            self.cancel_candidates.remove(0);
        }

        OrderMessage::New(NewOrder {
            id,
            timestamp,
            account,
            symbol: symbol.to_string(),
            side,
            price,
            quantity,
        })
    }
}

type SharedState = Arc<Mutex<FeedState>>;

/// Locks the feed state, recovering the data if another thread panicked while holding it.
fn lock(state: &SharedState) -> MutexGuard<'_, FeedState> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Returns the current UNIX timestamp in milliseconds.
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Rounds a value to two decimal places (used for prices).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Rounds a value to one decimal place (used for quantities).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// A loop that publishes a new random message on the feed at a fixed rate.
async fn produce_orders(state: SharedState, rate: f64) {
    let interval_ms = (1000.0 / rate.clamp(0.1, 1000.0)) as u64;
    loop {
        sleep(Duration::from_millis(interval_ms)).await;
        let mut state = lock(&state);
        let msg = state.next_message();
        info!("Publishing message: {:?}", msg);
        state.push(msg);
    }
}

/// Runs the API server to handle HTTP requests until Ctrl-C.
/// The server serves the order feed and accepts order submissions.
async fn run_server(shared_state: SharedState, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/orders", get(get_orders))
        .route("/order", post(submit_order))
        .route("/cancel", post(submit_cancel))
        .route("/symbols", get(get_symbols))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    info!("listening on {}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(crate::shutdown_signal())
        .await
        .context("feed server failed")
}

/// Defines the query parameters for the /orders endpoint.
#[derive(Deserialize)]
struct GetOrdersQuery {
    /// Return only messages with an ID strictly greater than this value.
    /// Use the highest ID you have seen so far to poll for new messages.
    since: Option<OrderId>,
    /// If provided, return only the last n messages instead.
    n: Option<usize>,
}

/// Handles GET requests for retrieving feed messages.
/// With `since`, returns all messages after the given ID (use this to poll).
/// With `n`, returns the last n messages.
async fn get_orders(
    State(state): State<SharedState>,
    Query(params): Query<GetOrdersQuery>,
) -> Json<Vec<OrderMessage>> {
    let state = lock(&state);
    if let Some(n) = params.n {
        let start = state.messages.len().saturating_sub(n);
        return Json(state.messages.iter().skip(start).cloned().collect());
    }
    let since = params.since.unwrap_or(0);
    // Messages are stored in ID order, so we can skip everything up to `since`.
    let start = state.messages.partition_point(|m| m.id() <= since);
    Json(state.messages.iter().skip(start).cloned().collect())
}

/// Defines the request body for submitting a new order.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitOrderRequest {
    pub account: AccountId,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
}

/// Defines the response returned when a message is accepted onto the feed.
#[derive(Serialize, Deserialize)]
pub struct SubmitResponse {
    pub id: OrderId,
}

/// Handles POST requests for submitting a new order onto the feed.
/// The order is validated, assigned an ID, and published like any other
/// feed message.
async fn submit_order(
    State(state): State<SharedState>,
    Json(req): Json<SubmitOrderRequest>,
) -> Result<Json<SubmitResponse>, (StatusCode, String)> {
    if !SYMBOLS.iter().any(|(s, _)| *s == req.symbol) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("'{}' is not a valid symbol", req.symbol),
        ));
    }
    // NaN and infinity must be rejected explicitly: `NaN <= 0.0` is false.
    let valid = |x: f64| x.is_finite() && x > 0.0;
    if !valid(req.price) || !valid(req.quantity) {
        return Err((
            StatusCode::BAD_REQUEST,
            "price and quantity must be positive finite numbers".to_string(),
        ));
    }

    let mut state = lock(&state);
    let id = state.allocate_id();
    let msg = OrderMessage::New(NewOrder {
        id,
        timestamp: now_millis(),
        account: req.account,
        symbol: req.symbol,
        side: req.side,
        price: req.price,
        quantity: req.quantity,
    });
    info!("Received order: {:?}", msg);
    state.push(msg);
    Ok(Json(SubmitResponse { id }))
}

/// Defines the request body for submitting a cancel.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitCancelRequest {
    pub account: AccountId,
    pub target_id: OrderId,
}

/// Handles POST requests for submitting a cancel onto the feed.
/// The feed does not check whether the target order is still open; consumers
/// of the feed decide what a cancel means for their own book.
async fn submit_cancel(
    State(state): State<SharedState>,
    Json(req): Json<SubmitCancelRequest>,
) -> Json<SubmitResponse> {
    let mut state = lock(&state);
    let id = state.allocate_id();
    let msg = OrderMessage::Cancel {
        id,
        timestamp: now_millis(),
        account: req.account,
        target_id: req.target_id,
    };
    info!("Received cancel: {:?}", msg);
    state.push(msg);
    Json(SubmitResponse { id })
}

/// Handles GET requests for the list of available symbols.
async fn get_symbols() -> Json<Vec<String>> {
    Json(SYMBOLS.iter().map(|(s, _)| s.to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_prices() {
        let prices = |seed| {
            let mut state = FeedState::new(5, Some(seed));
            (0..200)
                .filter_map(|_| match state.next_message() {
                    OrderMessage::New(o) => Some((o.symbol, o.side, o.price, o.quantity)),
                    OrderMessage::Cancel { .. } => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(prices(7), prices(7));
        assert_ne!(prices(7), prices(8));
    }

    #[test]
    fn ids_are_strictly_increasing() {
        let mut state = FeedState::new(3, Some(1));
        let ids: Vec<_> = (0..100).map(|_| state.next_message().id()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn buffer_is_bounded() {
        let mut state = FeedState::new(3, Some(1));
        for _ in 0..MAX_RETAINED_MESSAGES + 10 {
            let msg = state.next_message();
            state.push(msg);
        }
        assert_eq!(state.messages.len(), MAX_RETAINED_MESSAGES);
    }

    #[test]
    fn wire_format_is_unchanged() {
        let msg = OrderMessage::New(NewOrder {
            id: 42,
            timestamp: 1,
            account: 3,
            symbol: "ETH-USDC".into(),
            side: Side::Buy,
            price: 100.25,
            quantity: 5.0,
        });
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["New"]["id"], 42);
        assert_eq!(json["New"]["side"], "Buy");
    }
}
