use crate::bot::RiskMode;
use crate::pipeline::Report;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// A small, fast model is plenty for a periodic risk check; override with `ANTHROPIC_MODEL`.
const DEFAULT_MODEL: &str = "claude-haiku-5-5";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

const SYSTEM_PROMPT: &str = "You are a risk advisor for a paper-trading bot that trades from an online \
logistic-regression model predicting the next mid price move. You receive a JSON report with the model's \
rolling accuracy and log-loss, a simple imbalance baseline's accuracy and log-loss over the same window, \
and the bot's PnL, drawdown, positions and current risk mode. Choose the risk mode for the next period: \
\"normal\", \"reduced\" (half size) or \"paused\" (flat, no trading). Prefer \"reduced\" when the model is \
not clearly beating the baseline or drawdown is growing, and \"paused\" only for clear trouble. Keep \
\"normal\" when the model beats the baseline and risk is contained. With fewer than 100 evaluated \
predictions, keep \"normal\". Reply with a single JSON object and nothing else: \
{\"mode\": \"normal|reduced|paused\", \"reason\": \"one short sentence\"}";

/// A risk-mode recommendation with a short explanation.
#[derive(Debug, Clone, PartialEq)]
pub struct Advice {
    pub mode: RiskMode,
    pub reason: String,
}

/// Asks Claude to pick a risk mode from a run report.
#[derive(Debug, Clone)]
pub struct ClaudeAdvisor {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl ClaudeAdvisor {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("a default reqwest client always builds");
        Self {
            http,
            api_key,
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Builds an advisor from `ANTHROPIC_API_KEY` (required), `ANTHROPIC_MODEL` and `ANTHROPIC_BASE_URL`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY must be set to use the Claude advisor")?;
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(api_key, model, base_url))
    }

    async fn advise(&self, report: &Report) -> Result<Advice> {
        // 1. Send the report as the user message; the system prompt defines the closed answer set.
        let body = json!({
            "model": self.model,
            "max_tokens": 200,
            "system": SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": serde_json::to_string_pretty(report)?,
            }],
        });
        let res = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .context("failed to reach the Anthropic API")?;
        let status = res.status();
        if !status.is_success() {
            bail!(
                "Anthropic API returned {status}: {}",
                res.text().await.unwrap_or_default()
            );
        }

        // 2. Pull the text out of the first content block and parse the advice from it.
        let value: Value = res.json().await.context("invalid Anthropic API response")?;
        let text = value["content"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow!("Anthropic API response had no text block"))?;
        parse_advice(text)
    }
}

/// Who decides the bot's risk mode: nobody (static), or Claude.
#[derive(Debug, Clone)]
pub enum Advisor {
    Off,
    Claude(ClaudeAdvisor),
}

impl Advisor {
    /// Returns advice for the report, or `None` when the advisor is off.
    pub async fn advise(&self, report: &Report) -> Result<Option<Advice>> {
        match self {
            Advisor::Off => Ok(None),
            Advisor::Claude(claude) => claude.advise(report).await.map(Some),
        }
    }
}

/// Extracts `{"mode": ..., "reason": ...}` from the model's reply, tolerating text around the JSON.
pub fn parse_advice(text: &str) -> Result<Advice> {
    let start = text
        .find('{')
        .ok_or_else(|| anyhow!("no JSON object in advisor reply"))?;
    let end = text
        .rfind('}')
        .ok_or_else(|| anyhow!("no JSON object in advisor reply"))?;
    if end < start {
        bail!("malformed JSON in advisor reply");
    }
    let value: Value =
        serde_json::from_str(&text[start..=end]).context("malformed JSON in advisor reply")?;
    let mode = match value["mode"].as_str().map(str::to_lowercase).as_deref() {
        Some("normal") => RiskMode::Normal,
        Some("reduced") => RiskMode::Reduced,
        Some("paused") => RiskMode::Paused,
        other => bail!("unknown risk mode in advisor reply: {other:?}"),
    };
    let reason = value["reason"]
        .as_str()
        .unwrap_or("no reason given")
        .to_string();
    Ok(Advice { mode, reason })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json() {
        let advice =
            parse_advice(r#"{"mode": "reduced", "reason": "model near baseline"}"#).unwrap();
        assert_eq!(advice.mode, RiskMode::Reduced);
        assert_eq!(advice.reason, "model near baseline");
    }

    #[test]
    fn tolerates_code_fences_and_case() {
        let advice =
            parse_advice("```json\n{\"mode\": \"PAUSED\", \"reason\": \"x\"}\n```").unwrap();
        assert_eq!(advice.mode, RiskMode::Paused);
    }

    #[test]
    fn rejects_unknown_modes_and_garbage() {
        assert!(parse_advice(r#"{"mode": "yolo"}"#).is_err());
        assert!(parse_advice("no json here").is_err());
        assert!(parse_advice("} {").is_err());
    }
}
