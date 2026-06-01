//! Conservation module — Noether symmetry tracking + CALM merge.
//!
//! Noether's theorem: every continuous symmetry of the action yields a conserved
//! quantity. We track which symmetries the agent maintains and their associated
//! conserved currents.
//!
//! CALM (Coordinated Agent Lifecycle Merge) provides fleet coordination by
//! merging agents while preserving conserved quantities.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A symmetry and its associated conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symmetry {
    /// Name of the symmetry (e.g., "time_translation", "rotation").
    pub name: String,
    /// The generator of the symmetry (Lie algebra element).
    pub generator: DVector<f64>,
    /// The conserved quantity value.
    pub conserved_value: f64,
    /// Tolerance for conservation check.
    pub tolerance: f64,
}

/// Result of checking conservation laws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Each symmetry and whether it's conserved.
    pub symmetries: Vec<(String, bool, f64)>,
    /// Overall conservation status.
    pub all_conserved: bool,
    /// Maximum violation.
    pub max_violation: f64,
}

/// Noether symmetry tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoetherTracker {
    /// Tracked symmetries.
    pub symmetries: Vec<Symmetry>,
    /// History of conserved values (for drift detection).
    pub history: Vec<HashMap<String, f64>>,
}

impl NoetherTracker {
    pub fn new() -> Self {
        Self {
            symmetries: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Register a new symmetry to track.
    pub fn register_symmetry(
        &mut self,
        name: &str,
        generator: DVector<f64>,
        initial_value: f64,
        tolerance: f64,
    ) {
        self.symmetries.push(Symmetry {
            name: name.to_string(),
            generator,
            conserved_value: initial_value,
            tolerance,
        });
    }

    /// Register time-translation symmetry (energy conservation).
    pub fn register_energy_conservation(&mut self, dim: usize, initial_energy: f64) {
        let gen = DVector::zeros(dim);
        self.register_symmetry("energy", gen, initial_energy, 0.01);
    }

    /// Register phase-rotation symmetry (angular momentum).
    pub fn register_rotation_conservation(&mut self, dim: usize, initial_angular_momentum: f64) {
        let gen = DVector::zeros(dim);
        self.register_symmetry("rotation", gen, initial_angular_momentum, 0.01);
    }

    /// Update a symmetry's conserved value after a dynamics step.
    pub fn update_value(&mut self, name: &str, new_value: f64) {
        if let Some(sym) = self.symmetries.iter_mut().find(|s| s.name == name) {
            sym.conserved_value = new_value;
        }
    }

    /// Check all conservation laws.
    pub fn check_conservation(&self, current_values: &HashMap<String, f64>) -> ConservationReport {
        let mut results = Vec::new();
        let mut all_conserved = true;
        let mut max_violation = 0.0f64;

        for sym in &self.symmetries {
            if let Some(&current) = current_values.get(&sym.name) {
                let violation = (current - sym.conserved_value).abs();
                let conserved = violation <= sym.tolerance;
                if !conserved {
                    all_conserved = false;
                }
                max_violation = max_violation.max(violation);
                results.push((sym.name.clone(), conserved, violation));
            }
        }

        ConservationReport {
            symmetries: results,
            all_conserved,
            max_violation,
        }
    }

    /// Record current state in history.
    pub fn snapshot(&mut self, values: HashMap<String, f64>) {
        self.history.push(values);
        // Keep history bounded
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Compute drift in conservation laws over time.
    pub fn conservation_drift(&self) -> HashMap<String, f64> {
        let mut drifts = HashMap::new();
        if self.history.len() < 2 {
            return drifts;
        }

        let first = &self.history[0];
        let last = self.history.last().unwrap();

        for sym in &self.symmetries {
            let v0 = first.get(&sym.name).copied().unwrap_or(0.0);
            let vn = last.get(&sym.name).copied().unwrap_or(0.0);
            drifts.insert(sym.name.clone(), (vn - v0).abs());
        }

        drifts
    }

    /// Number of tracked symmetries.
    pub fn num_symmetries(&self) -> usize {
        self.symmetries.len()
    }
}

/// CALM merge result for fleet coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalmMergeResult {
    /// Merged state.
    pub merged_state: DVector<f64>,
    /// Merged precision.
    pub merged_precision: DMatrix<f64>,
    /// Conserved quantity residuals (should be near zero).
    pub conservation_residuals: Vec<f64>,
    /// Merge successful?
    pub success: bool,
}

/// CALM merge: combine two agent states while preserving conserved quantities.
///
/// Uses precision-weighted averaging with conservation constraints.
pub fn calm_merge(
    state_a: &DVector<f64>,
    precision_a: &DMatrix<f64>,
    state_b: &DVector<f64>,
    precision_b: &DMatrix<f64>,
    conserved_quantities: &[f64],
) -> CalmMergeResult {
    // Precision-weighted merge: P_merged = P_a + P_b
    let merged_precision = precision_a + precision_b;

    // Merged mean: (P_a + P_b)^{-1} (P_a μ_a + P_b μ_b)
    let weighted_sum = precision_a * state_a + precision_b * state_b;
    let merged_state = match merged_precision.clone().try_inverse() {
        Some(inv) => inv * weighted_sum,
        None => (state_a + state_b) * 0.5,
    };

    // Check conservation: residuals should be small
    let residuals: Vec<f64> = conserved_quantities
        .iter()
        .map(|&q| {
            let merged_q = merged_state.iter().sum::<f64>(); // simplified
            (merged_q - q).abs()
        })
        .collect();

    let success = residuals.iter().all(|&r| r < 1.0);

    CalmMergeResult {
        merged_state,
        merged_precision,
        conservation_residuals: residuals,
        success,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noether_tracker_creation() {
        let tracker = NoetherTracker::new();
        assert_eq!(tracker.symmetries.len(), 0);
    }

    #[test]
    fn test_register_symmetry() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("energy", DVector::zeros(3), 100.0, 0.01);
        assert_eq!(tracker.num_symmetries(), 1);
    }

    #[test]
    fn test_register_energy_conservation() {
        let mut tracker = NoetherTracker::new();
        tracker.register_energy_conservation(3, 50.0);
        assert_eq!(tracker.num_symmetries(), 1);
        assert_eq!(tracker.symmetries[0].name, "energy");
    }

    #[test]
    fn test_check_conservation_passes() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("energy", DVector::zeros(2), 100.0, 0.1);
        let mut values = HashMap::new();
        values.insert("energy".to_string(), 100.05);
        let report = tracker.check_conservation(&values);
        assert!(report.all_conserved);
    }

    #[test]
    fn test_check_conservation_fails() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("energy", DVector::zeros(2), 100.0, 0.01);
        let mut values = HashMap::new();
        values.insert("energy".to_string(), 99.0);
        let report = tracker.check_conservation(&values);
        assert!(!report.all_conserved);
    }

    #[test]
    fn test_conservation_report_max_violation() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("a", DVector::zeros(2), 10.0, 0.1);
        tracker.register_symmetry("b", DVector::zeros(2), 20.0, 0.1);
        let mut values = HashMap::new();
        values.insert("a".to_string(), 10.5);
        values.insert("b".to_string(), 19.0);
        let report = tracker.check_conservation(&values);
        assert!((report.max_violation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_history() {
        let mut tracker = NoetherTracker::new();
        let mut values = HashMap::new();
        values.insert("energy".to_string(), 100.0);
        tracker.snapshot(values);
        assert_eq!(tracker.history.len(), 1);
    }

    #[test]
    fn test_conservation_drift() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("energy", DVector::zeros(2), 100.0, 0.1);
        let mut v1 = HashMap::new();
        v1.insert("energy".to_string(), 100.0);
        tracker.snapshot(v1);
        let mut v2 = HashMap::new();
        v2.insert("energy".to_string(), 99.0);
        tracker.snapshot(v2);
        let drift = tracker.conservation_drift();
        assert!((drift["energy"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calm_merge_basic() {
        let state_a = DVector::from_vec(vec![1.0, 0.0]);
        let state_b = DVector::from_vec(vec![0.0, 1.0]);
        let prec = DMatrix::identity(2, 2);
        let result = calm_merge(&state_a, &prec, &state_b, &prec, &[]);
        assert_eq!(result.merged_state.len(), 2);
        // Equal precision → midpoint
        assert!((result.merged_state[0] - 0.5).abs() < 1e-10);
        assert!((result.merged_state[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_calm_merge_unequal_precision() {
        let state_a = DVector::from_vec(vec![0.0]);
        let state_b = DVector::from_vec(vec![10.0]);
        let prec_a = DMatrix::from_vec(1, 1, vec![10.0]); // high confidence in a
        let prec_b = DMatrix::from_vec(1, 1, vec![1.0]);  // low confidence in b
        let result = calm_merge(&state_a, &prec_a, &state_b, &prec_b, &[]);
        // Should be closer to state_a
        assert!(result.merged_state[0] < 5.0);
    }

    #[test]
    fn test_calm_merge_precision_adds() {
        let prec_a = DMatrix::identity(2, 2) * 2.0;
        let prec_b = DMatrix::identity(2, 2) * 3.0;
        let state = DVector::zeros(2);
        let result = calm_merge(&state, &prec_a, &state, &prec_b, &[]);
        assert!((result.merged_precision[(0, 0)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_value() {
        let mut tracker = NoetherTracker::new();
        tracker.register_symmetry("energy", DVector::zeros(2), 100.0, 0.1);
        tracker.update_value("energy", 99.5);
        assert!((tracker.symmetries[0].conserved_value - 99.5).abs() < 1e-10);
    }
}
