use serde::Serialize;
use std::collections::VecDeque;

/// Accuracy and log-loss over a window of predictions.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Summary {
    pub samples: usize,
    pub accuracy: f64,
    pub log_loss: f64,
}

/// Tracks the most recent `capacity` predictions so quality reflects current behavior.
#[derive(Debug, Clone)]
pub struct RollingMetrics {
    window: VecDeque<(f64, bool)>,
    capacity: usize,
}

impl RollingMetrics {
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Records one prediction `p_up` and whether the price actually went up.
    pub fn record(&mut self, p_up: f64, went_up: bool) {
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back((p_up, went_up));
    }

    pub fn summary(&self) -> Summary {
        let n = self.window.len();
        if n == 0 {
            return Summary::default();
        }
        let correct = self
            .window
            .iter()
            .filter(|(p, up)| (*p >= 0.5) == *up)
            .count();
        let loss: f64 = self
            .window
            .iter()
            .map(|(p, up)| {
                // Clamp so a confident wrong prediction costs a lot but never infinity.
                let p = p.clamp(1e-12, 1.0 - 1e-12);
                if *up { -p.ln() } else { -(1.0 - p).ln() }
            })
            .sum();
        Summary {
            samples: n,
            accuracy: correct as f64 / n as f64,
            log_loss: loss / n as f64,
        }
    }
}

/// Accuracy and log-loss over every prediction since the start, using constant memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct CumulativeMetrics {
    count: usize,
    correct: usize,
    loss: f64,
}

impl CumulativeMetrics {
    /// Records one prediction `p_up` and whether the price actually went up.
    pub fn record(&mut self, p_up: f64, went_up: bool) {
        let p = p_up.clamp(1e-12, 1.0 - 1e-12);
        self.count += 1;
        self.correct += ((p_up >= 0.5) == went_up) as usize;
        self.loss -= if went_up { p.ln() } else { (1.0 - p).ln() };
    }

    pub fn summary(&self) -> Summary {
        if self.count == 0 {
            return Summary::default();
        }
        Summary {
            samples: self.count,
            accuracy: self.correct as f64 / self.count as f64,
            log_loss: self.loss / self.count as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_is_zeroed() {
        assert_eq!(RollingMetrics::new(10).summary().samples, 0);
    }

    #[test]
    fn computes_accuracy_and_log_loss() {
        let mut m = RollingMetrics::new(10);
        m.record(0.9, true);
        m.record(0.8, false);
        let s = m.summary();
        assert_eq!(s.samples, 2);
        assert!((s.accuracy - 0.5).abs() < 1e-12);
        let expected = (-(0.9f64).ln() - (0.2f64).ln()) / 2.0;
        assert!((s.log_loss - expected).abs() < 1e-12);
    }

    #[test]
    fn window_drops_oldest() {
        let mut m = RollingMetrics::new(2);
        m.record(0.9, false);
        m.record(0.9, true);
        m.record(0.9, true);
        assert_eq!(m.summary().samples, 2);
        assert_eq!(m.summary().accuracy, 1.0);
    }

    #[test]
    fn cumulative_matches_a_window_that_never_drops() {
        let mut rolling = RollingMetrics::new(1_000);
        let mut total = CumulativeMetrics::default();
        for i in 0..200 {
            let p = 0.05 + 0.9 * (i % 10) as f64 / 10.0;
            rolling.record(p, i % 3 == 0);
            total.record(p, i % 3 == 0);
        }
        let (a, b) = (rolling.summary(), total.summary());
        assert_eq!(a.samples, b.samples);
        assert!((a.accuracy - b.accuracy).abs() < 1e-12);
        assert!((a.log_loss - b.log_loss).abs() < 1e-12);
    }
}
