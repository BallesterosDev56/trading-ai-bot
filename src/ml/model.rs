use crate::ml::features::{FeatureVector, IMBALANCE_TOP, NUM_FEATURES};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default SGD step size.
pub const DEFAULT_LEARNING_RATE: f64 = 0.05;
/// Default L2 penalty; small, just enough to keep weights from drifting on noise.
const DEFAULT_L2: f64 = 1e-4;
/// Standardized features are clipped to this many standard deviations.
const CLIP_SIGMAS: f64 = 5.0;

/// Anything that predicts P(mid price goes up) and can learn from the outcome.
pub trait Predictor {
    /// Returns the predicted probability (0..1) that the next mid price move is up.
    fn predict(&self, x: &FeatureVector) -> f64;
    /// Learns from one labeled example.
    fn update(&mut self, x: &FeatureVector, went_up: bool);
}

/// Running mean and variance per feature (Welford), used to standardize inputs online.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Scaler {
    count: u64,
    mean: [f64; NUM_FEATURES],
    m2: [f64; NUM_FEATURES],
}

impl Scaler {
    fn new() -> Self {
        Self {
            count: 0,
            mean: [0.0; NUM_FEATURES],
            m2: [0.0; NUM_FEATURES],
        }
    }

    fn observe(&mut self, x: &FeatureVector) {
        self.count += 1;
        for (i, &value) in x.iter().enumerate() {
            let delta = value - self.mean[i];
            self.mean[i] += delta / self.count as f64;
            self.m2[i] += delta * (value - self.mean[i]);
        }
    }

    /// Centers and scales each feature, clipping outliers. Constant features become 0.
    fn transform(&self, x: &FeatureVector) -> FeatureVector {
        let mut out = [0.0; NUM_FEATURES];
        if self.count < 2 {
            return out;
        }
        for i in 0..NUM_FEATURES {
            let std = (self.m2[i] / (self.count - 1) as f64).sqrt();
            if std > 1e-9 {
                out[i] = ((x[i] - self.mean[i]) / std).clamp(-CLIP_SIGMAS, CLIP_SIGMAS);
            }
        }
        out
    }
}

/// Online logistic regression that predicts P(mid price goes up) from book features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticModel {
    weights: [f64; NUM_FEATURES],
    bias: f64,
    learning_rate: f64,
    l2: f64,
    scaler: Scaler,
}

impl LogisticModel {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: [0.0; NUM_FEATURES],
            bias: 0.0,
            learning_rate,
            l2: DEFAULT_L2,
            scaler: Scaler::new(),
        }
    }

    /// How many labeled examples the model has learned from.
    pub fn samples(&self) -> u64 {
        self.scaler.count
    }

    /// Saves the model as JSON, writing to a temp file first so a crash never leaves a half-written file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_vec_pretty(self).context("failed to serialize model")?;
        std::fs::write(&tmp, json).with_context(|| format!("failed to write {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("failed to move model to {}", path.display()))
    }

    /// Loads a model previously written by `save`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid model file {}", path.display()))
    }

    fn score(&self, z: &FeatureVector) -> f64 {
        self.bias + self.weights.iter().zip(z).map(|(w, v)| w * v).sum::<f64>()
    }
}

impl Predictor for LogisticModel {
    fn predict(&self, x: &FeatureVector) -> f64 {
        // 1. Standardize, then squash the linear score with a sigmoid so the output is a probability.
        let z = self.scaler.transform(x);
        1.0 / (1.0 + (-self.score(&z)).exp())
    }

    /// One SGD step on the log-loss, with L2 weight decay.
    fn update(&mut self, x: &FeatureVector, went_up: bool) {
        // 1. Fold the example into the scaler first, so features stay centered as the market drifts.
        self.scaler.observe(x);
        let z = self.scaler.transform(x);
        // 2. For log-loss the gradient w.r.t. the score is simply (prediction - label).
        let label = if went_up { 1.0 } else { 0.0 };
        let error = 1.0 / (1.0 + (-self.score(&z)).exp()) - label;
        // 3. Move each weight against its gradient (plus decay), then the bias.
        for (w, v) in self.weights.iter_mut().zip(&z) {
            *w -= self.learning_rate * (error * v + self.l2 * *w);
        }
        self.bias -= self.learning_rate * error;
    }
}

/// A simple rule used as a yardstick: lean towards the side with more resting size at the top.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImbalanceBaseline;

impl Predictor for ImbalanceBaseline {
    fn predict(&self, x: &FeatureVector) -> f64 {
        0.5 + 0.4 * x[IMBALANCE_TOP].clamp(-1.0, 1.0)
    }

    fn update(&mut self, _x: &FeatureVector, _went_up: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_signal(value: f64) -> FeatureVector {
        let mut x = [0.0; NUM_FEATURES];
        x[2] = value;
        x
    }

    #[test]
    fn learns_a_separable_signal() {
        let mut model = LogisticModel::new(0.5);
        for _ in 0..500 {
            model.update(&with_signal(1.0), true);
            model.update(&with_signal(-1.0), false);
        }
        assert!(model.predict(&with_signal(1.0)) > 0.9);
        assert!(model.predict(&with_signal(-1.0)) < 0.1);
    }

    #[test]
    fn handles_features_on_very_different_scales() {
        // Feature 0 is huge and useless; feature 2 is tiny and informative.
        let mut model = LogisticModel::new(0.1);
        for i in 0..2000 {
            let up = i % 2 == 0;
            let mut x = with_signal(if up { 0.01 } else { -0.01 });
            x[0] = 5000.0 + (i % 7) as f64;
            model.update(&x, up);
        }
        let mut x = with_signal(0.01);
        x[0] = 5003.0;
        assert!(model.predict(&x) > 0.8);
    }

    #[test]
    fn untrained_model_is_neutral() {
        let model = LogisticModel::new(0.1);
        assert!((model.predict(&with_signal(3.0)) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut model = LogisticModel::new(0.5);
        for _ in 0..100 {
            model.update(&with_signal(1.0), true);
            model.update(&with_signal(-1.0), false);
        }
        let path =
            std::env::temp_dir().join(format!("orderflow-model-{}.json", std::process::id()));
        model.save(&path).unwrap();
        let loaded = LogisticModel::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.samples(), model.samples());
        assert_eq!(
            loaded.predict(&with_signal(1.0)),
            model.predict(&with_signal(1.0))
        );
    }

    #[test]
    fn baseline_follows_imbalance() {
        let mut x = [0.0; NUM_FEATURES];
        x[IMBALANCE_TOP] = 1.0;
        assert!(ImbalanceBaseline.predict(&x) > 0.5);
        x[IMBALANCE_TOP] = -1.0;
        assert!(ImbalanceBaseline.predict(&x) < 0.5);
    }
}
