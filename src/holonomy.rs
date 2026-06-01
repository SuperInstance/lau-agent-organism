//! Holonomy module — H¹ cohomology for delusion detection.
//!
//! The first cohomology H¹ detects global inconsistencies in the agent's
//! belief structure. Reward hacking manifests as nonzero H¹ — the agent's
//! local optimization doesn't patch together into a globally consistent policy.
//!
//! Holonomy around closed loops in belief space reveals self-deception.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// H¹ holonomy diagnostic result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyDiagnostic {
    /// H¹ class representatives (nonzero = delusion detected).
    pub h1_classes: Vec<f64>,
    /// Overall H¹ risk (0 = consistent, high = delusional).
    pub h1_risk: f64,
    /// Number of detected inconsistencies.
    pub inconsistency_count: usize,
    /// Holonomy around each belief cycle.
    pub cycle_holonomies: Vec<f64>,
    /// Whether the agent is delusional.
    pub is_delusional: bool,
    /// Delusion threshold.
    pub threshold: f64,
}

/// H¹ holonomy monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyMonitor {
    /// Dimension of belief space.
    pub dim: usize,
    /// History of belief transitions (coboundary chain).
    pub transition_history: Vec<DVector<f64>>,
    /// Maximum number of transitions to keep.
    pub max_history: usize,
    /// Delusion detection threshold for H¹.
    pub threshold: f64,
    /// Reward history for hacking detection.
    pub reward_history: Vec<f64>,
    /// Running estimate of H¹.
    pub h1_estimate: f64,
}

impl HolonomyMonitor {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            transition_history: Vec::new(),
            max_history: 100,
            threshold: 0.1,
            reward_history: Vec::new(),
            h1_estimate: 0.0,
        }
    }

    /// Record a belief transition (coboundary of a belief change).
    pub fn record_transition(&mut self, old_belief: &DVector<f64>, new_belief: &DVector<f64>) {
        let delta = new_belief - old_belief;
        self.transition_history.push(delta);
        if self.transition_history.len() > self.max_history {
            self.transition_history.remove(0);
        }
    }

    /// Record a reward observation for hacking detection.
    pub fn record_reward(&mut self, reward: f64) {
        self.reward_history.push(reward);
        if self.reward_history.len() > self.max_history {
            self.reward_history.remove(0);
        }
    }

    /// Compute the coboundary operator (discrete exterior derivative).
    ///
    /// For transitions δ₁, δ₂, ..., the coboundary checks:
    /// d² = 0 mod errors → violations of d² = 0 indicate H¹ ≠ 0.
    pub fn compute_coboundary(&self) -> DMatrix<f64> {
        let n = self.transition_history.len();
        if n < 2 {
            return DMatrix::zeros(self.dim, self.dim);
        }

        // Build the coboundary matrix from consecutive transitions
        let mut coboundary = DMatrix::zeros(self.dim, n - 1);
        for i in 0..n - 1 {
            let diff = &self.transition_history[i + 1] - &self.transition_history[i];
            for j in 0..self.dim {
                coboundary[(j, i)] = diff[j];
            }
        }
        coboundary
    }

    /// Detect H¹ by checking if the coboundary chain is closed.
    ///
    /// A consistent belief system has d² = 0, meaning consecutive transitions
    /// compose to zero. Nonzero H¹ means the agent is "going in circles"
    /// without realizing it — classic reward hacking.
    pub fn detect_h1(&self) -> HolonomyDiagnostic {
        let n = self.transition_history.len();

        if n < 3 {
            return HolonomyDiagnostic {
                h1_classes: vec![0.0; self.dim],
                h1_risk: 0.0,
                inconsistency_count: 0,
                cycle_holonomies: vec![],
                is_delusional: false,
                threshold: self.threshold,
            };
        }

        // Check closure of transition cycles
        let mut cycle_holonomies = Vec::new();

        // Sliding window cycles
        let window_size = 3.min(n);
        for start in 0..=(n - window_size) {
            let mut holonomy = DVector::zeros(self.dim);
            for i in start..start + window_size {
                holonomy += &self.transition_history[i];
            }
            // Holonomy is the residual (should be zero for consistent beliefs)
            let h_norm: f64 = holonomy.iter().map(|x| x * x).sum::<f64>().sqrt();
            cycle_holonomies.push(h_norm);
        }

        // Global H¹: sum of all transitions (should be near zero)
        let global_holonomy: DVector<f64> =
            self.transition_history.iter().fold(DVector::zeros(self.dim), |acc, v| acc + v);

        let h1_classes: Vec<f64> = global_holonomy.iter().map(|&x| x.abs()).collect();
        let h1_risk: f64 = h1_classes.iter().sum::<f64>() / self.dim as f64;

        let inconsistency_count = h1_classes.iter().filter(|&&x| x > self.threshold).count();
        let is_delusional = h1_risk > self.threshold;

        HolonomyDiagnostic {
            h1_classes,
            h1_risk,
            inconsistency_count,
            cycle_holonomies,
            is_delusional,
            threshold: self.threshold,
        }
    }

    /// Detect reward hacking specifically.
    ///
    /// Reward hacking: rewards increase while belief coherence decreases.
    pub fn detect_reward_hacking(&self) -> bool {
        if self.reward_history.len() < 5 {
            return false;
        }

        // Check if rewards are monotonically increasing
        let n = self.reward_history.len();
        let recent = &self.reward_history[n - 5..];
        let mut increasing = true;
        for i in 1..recent.len() {
            if recent[i] <= recent[i - 1] {
                increasing = false;
                break;
            }
        }

        // If rewards are suspiciously increasing AND H¹ is nonzero
        increasing && self.h1_estimate > self.threshold * 0.5
    }

    /// Update H¹ estimate with exponential smoothing.
    pub fn update_h1_estimate(&mut self) {
        let diag = self.detect_h1();
        self.h1_estimate = 0.9 * self.h1_estimate + 0.1 * diag.h1_risk;
    }

    /// Get current H¹ risk.
    pub fn current_risk(&self) -> f64 {
        self.h1_estimate
    }

    /// Whether H¹ is zero (consistent beliefs, no delusion).
    pub fn is_consistent(&self) -> bool {
        self.h1_estimate < self.threshold
    }

    /// Reset the monitor.
    pub fn reset(&mut self) {
        self.transition_history.clear();
        self.reward_history.clear();
        self.h1_estimate = 0.0;
    }
}

/// Check if a sequence of local policies can be glued into a global policy.
///
/// This is the sheaf condition: local consistency → global consistency.
/// Returns true if H¹ = 0 (no obstruction to gluing).
pub fn check_sheaf_condition(local_policies: &[DVector<f64>], overlaps: &[(usize, usize, f64)]) -> bool {
    // For each overlapping pair, check that the policies agree on the overlap
    for &(i, j, tolerance) in overlaps {
        if i < local_policies.len() && j < local_policies.len() {
            let diff = &local_policies[i] - &local_policies[j];
            let agreement = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
            if agreement > tolerance {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = HolonomyMonitor::new(3);
        assert_eq!(monitor.dim, 3);
        assert!((monitor.h1_estimate).abs() < 1e-10);
    }

    #[test]
    fn test_record_transition() {
        let mut monitor = HolonomyMonitor::new(2);
        let old = DVector::from_vec(vec![0.0, 0.0]);
        let new = DVector::from_vec(vec![1.0, 0.0]);
        monitor.record_transition(&old, &new);
        assert_eq!(monitor.transition_history.len(), 1);
    }

    #[test]
    fn test_consistent_beliefs_have_zero_h1() {
        let mut monitor = HolonomyMonitor::new(2);
        // Consistent: forward then back
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        monitor.record_transition(&a, &b);
        monitor.record_transition(&b, &a);
        let diag = monitor.detect_h1();
        assert!(!diag.is_delusional);
        assert!(diag.h1_risk < 0.1);
    }

    #[test]
    fn test_inconsistent_beliefs_have_nonzero_h1() {
        let mut monitor = HolonomyMonitor::new(2);
        monitor.threshold = 0.01;
        // Inconsistent: keep going same direction
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        let c = DVector::from_vec(vec![2.0, 0.0]);
        let d = DVector::from_vec(vec![3.0, 0.0]);
        monitor.record_transition(&a, &b);
        monitor.record_transition(&b, &c);
        monitor.record_transition(&c, &d);
        let diag = monitor.detect_h1();
        assert!(diag.h1_risk > 0.0);
    }

    #[test]
    fn test_h1_classes_dimension() {
        let monitor = HolonomyMonitor::new(4);
        let diag = monitor.detect_h1();
        assert_eq!(diag.h1_classes.len(), 4);
    }

    #[test]
    fn test_reward_hacking_detection() {
        let mut monitor = HolonomyMonitor::new(2);
        // Suspicious: rewards keep increasing
        for i in 0..10 {
            monitor.record_reward(i as f64);
        }
        monitor.h1_estimate = 0.1; // nonzero H¹
        assert!(monitor.detect_reward_hacking());
    }

    #[test]
    fn test_no_reward_hacking_with_variable_rewards() {
        let mut monitor = HolonomyMonitor::new(2);
        // Normal: rewards fluctuate
        for &r in &[1.0, 0.5, 1.0, 0.3, 0.8] {
            monitor.record_reward(r);
        }
        monitor.h1_estimate = 0.1;
        assert!(!monitor.detect_reward_hacking());
    }

    #[test]
    fn test_update_h1_estimate() {
        let mut monitor = HolonomyMonitor::new(2);
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        monitor.record_transition(&a, &b);
        monitor.record_transition(&b, &a);
        monitor.update_h1_estimate();
        assert!(monitor.h1_estimate >= 0.0);
    }

    #[test]
    fn test_is_consistent() {
        let monitor = HolonomyMonitor::new(2);
        assert!(monitor.is_consistent());
    }

    #[test]
    fn test_reset() {
        let mut monitor = HolonomyMonitor::new(2);
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        monitor.record_transition(&a, &b);
        monitor.record_reward(1.0);
        monitor.h1_estimate = 1.0;
        monitor.reset();
        assert!(monitor.transition_history.is_empty());
        assert!(monitor.reward_history.is_empty());
        assert!((monitor.h1_estimate).abs() < 1e-10);
    }

    #[test]
    fn test_sheaf_condition_passes() {
        let policies = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
        ];
        let overlaps = vec![(0, 1, 0.1)];
        assert!(check_sheaf_condition(&policies, &overlaps));
    }

    #[test]
    fn test_sheaf_condition_fails() {
        let policies = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let overlaps = vec![(0, 1, 0.1)];
        assert!(!check_sheaf_condition(&policies, &overlaps));
    }

    #[test]
    fn test_cycle_holonomies_computed() {
        let mut monitor = HolonomyMonitor::new(2);
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        let c = DVector::from_vec(vec![0.0, 1.0]);
        monitor.record_transition(&a, &b);
        monitor.record_transition(&b, &c);
        monitor.record_transition(&c, &a);
        let diag = monitor.detect_h1();
        assert!(!diag.cycle_holonomies.is_empty());
    }
}
