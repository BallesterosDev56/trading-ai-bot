use axum::{Json, Router, http::HeaderMap, http::StatusCode, routing::post};
use orderflow_ml::advisor::{Advisor, ClaudeAdvisor};
use orderflow_ml::bot::{BotConfig, RiskMode};
use orderflow_ml::ml::model::LogisticModel;
use orderflow_ml::pipeline::Pipeline;
use serde_json::{Value, json};

/// Starts a fake Anthropic API that checks the request shape and answers with canned advice.
async fn spawn_mock(reply_text: &'static str) -> String {
    let app = Router::new().route(
        "/v1/messages",
        post(
            move |headers: HeaderMap, Json(body): Json<Value>| async move {
                let authed = headers.get("x-api-key").is_some_and(|v| v == "test-key")
                    && headers.get("anthropic-version").is_some();
                let shaped = body["model"] == "test-model"
                    && body["messages"][0]["role"] == "user"
                    && body["system"].is_string();
                if !authed || !shaped {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": "bad request"})),
                    );
                }
                (
                    StatusCode::OK,
                    Json(json!({"content": [{"type": "text", "text": reply_text}]})),
                )
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

fn report() -> orderflow_ml::pipeline::Report {
    Pipeline::new(LogisticModel::new(0.05), BotConfig::default()).report()
}

#[tokio::test]
async fn claude_advisor_returns_parsed_advice() {
    let url = spawn_mock(r#"{"mode": "reduced", "reason": "drawdown rising"}"#).await;
    let advisor = Advisor::Claude(ClaudeAdvisor::new(
        "test-key".into(),
        "test-model".into(),
        url,
    ));
    let advice = advisor.advise(&report()).await.unwrap().unwrap();
    assert_eq!(advice.mode, RiskMode::Reduced);
    assert_eq!(advice.reason, "drawdown rising");
}

#[tokio::test]
async fn api_errors_surface_as_errors() {
    let url = spawn_mock("{}").await;
    let advisor = Advisor::Claude(ClaudeAdvisor::new(
        "wrong-key".into(),
        "test-model".into(),
        url,
    ));
    assert!(advisor.advise(&report()).await.is_err());
}

#[tokio::test]
async fn off_advisor_gives_no_advice() {
    assert!(Advisor::Off.advise(&report()).await.unwrap().is_none());
}
