use crate::feed::{
    AccountId, OrderId, OrderMessage, Side, SubmitCancelRequest, SubmitOrderRequest, SubmitResponse,
};
use anyhow::{Context, Result, bail};
use reqwest::Response;
use std::time::Duration;

/// How long to wait for the feed before giving up on a request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// A small HTTP client for the order feed API.
#[derive(Debug, Clone)]
pub struct FeedClient {
    base_url: String,
    http: reqwest::Client,
}

impl FeedClient {
    pub fn new(base_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("a default reqwest client always builds");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Fetches every message with an ID strictly greater than `since`.
    pub async fn orders_since(&self, since: OrderId) -> Result<Vec<OrderMessage>> {
        let res = self
            .http
            .get(format!("{}/orders", self.base_url))
            .query(&[("since", since)])
            .send()
            .await
            .context("failed to reach the feed")?;
        parse(res).await
    }

    /// Fetches the most recent `n` messages.
    pub async fn last_orders(&self, n: usize) -> Result<Vec<OrderMessage>> {
        let res = self
            .http
            .get(format!("{}/orders", self.base_url))
            .query(&[("n", n)])
            .send()
            .await
            .context("failed to reach the feed")?;
        parse(res).await
    }

    /// Submits a limit order and returns the ID the feed assigned to it.
    pub async fn submit_order(
        &self,
        account: AccountId,
        symbol: &str,
        side: Side,
        price: f64,
        quantity: f64,
    ) -> Result<OrderId> {
        let body = SubmitOrderRequest {
            account,
            symbol: symbol.to_string(),
            side,
            price,
            quantity,
        };
        let res = self
            .http
            .post(format!("{}/order", self.base_url))
            .json(&body)
            .send()
            .await
            .context("failed to reach the feed")?;
        Ok(parse::<SubmitResponse>(res).await?.id)
    }

    /// Submits a cancel for a previously placed order and returns the cancel's own ID.
    pub async fn cancel(&self, account: AccountId, target_id: OrderId) -> Result<OrderId> {
        let body = SubmitCancelRequest { account, target_id };
        let res = self
            .http
            .post(format!("{}/cancel", self.base_url))
            .json(&body)
            .send()
            .await
            .context("failed to reach the feed")?;
        Ok(parse::<SubmitResponse>(res).await?.id)
    }
}

/// Turns a non-2xx response into an error that includes the server's message.
async fn parse<T: serde::de::DeserializeOwned>(res: Response) -> Result<T> {
    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        bail!("feed returned {status}: {body}");
    }
    res.json()
        .await
        .context("failed to parse the feed response")
}
