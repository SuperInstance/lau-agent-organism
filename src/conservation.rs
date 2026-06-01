//! Conservation law — the fundamental invariant of the organism.
//!
//! Conservation law: Landauer + free_energy + H¹_risk ≈ constant across
//! the lifecycle. This is the bookkeeping identity that makes the agent
//! thermodynamically closed.

use serde::{Deserialize, Serialize};

/// Conservation state tracking the fundamental invariant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationState {
    /// Landauer cost (cumulative information erasure cost).
    pub landauer: f64,
    /// Current free energy.
    pub free_energy: f64,
    /// H¹ risk (delusion/coherence measure).
    pub h1_risk: f64,
    /// The constant that should be conserved.
    pub conservation_constant: f64,
    /// Tolerance for conservation check.
    pub tolerance: f64,
}

impl ConservationState {
    /// Create a new conservation state from initial conditions.
    pub fn new(initial_free_energy: f64) -> Self {
        // Initial: Landauer=0, free_energy=E, H¹_risk≈0
        // Conservation constant = 0 + E + 0 = E
        Self {
            landauer: 0.0,
            free_energy: initial_free_energy,
            h1_risk: 0.0,
            conservation_constant: initial_free_energy,
            tolerance: 0.1,
        }
    }

    /// Compute the conservation quantity.
    pub fn conservation_quantity(&self) -> f64 {
        self.landauer + self.free_energy + self.h1_risk
    }

    /// Check if conservation law holds.
    pub fn is_conserved(&self) -> bool {
        let qty = self.conservation_quantity();
        (qty - self.conservation_constant).abs() < self.tolerance * self.conservation_constant
    }

    /// Conservation violation magnitude.
    pub fn violation(&self) -> f64 {
        (self.conservation_quantity() - self.conservation_constant).abs()
    }

    /// Relative violation as fraction.
    pub fn relative_violation(&self) -> f64 {
        if self.conservation_constant > 0.0 {
            self.violation() / self.conservation_constant
        } else {
            0.0
        }
    }

    /// Update state and check conservation.
    pub fn update(&mut self, landauer: f64, free_energy: f64, h1_risk: f64) -> ConservationReport {
        self.landauer = landauer;
        self.free_energy = free_energy;
        self.h1_risk = h1_risk;

        ConservationReport {
            conserved: self.is_conserved(),
            quantity: self.conservation_quantity(),
            constant: self.conservation_constant,
            violation: self.violation(),
            components: ConservationComponents {
                landauer: self.landauer,
                free_energy: self.free_energy,
                h1_risk: self.h1_risk,
            },
        }
    }

    /// Verify the conservation law analytically.
    ///
    /// When the agent dies (Landauer = initial_free_energy):
    ///   Landauer + free_energy + H¹_risk
    ///   = initial_free_energy + (free_energy - initial_free_energy) + H¹_risk
    ///   ≈ initial_free_energy (when free_energy ≈ 0 and H¹_risk ≈ 0)
    pub fn verify_death_conservation(&self, initial_free_energy: f64) -> DeathConservationCheck {
        let at_death = self.landauer >= initial_free_energy;
        let expected_remaining = (initial_free_energy - self.landauer).max(0.0);
        let actual_remaining = self.free_energy + self.h1_risk;
        let death_violation = (expected_remaining - actual_remaining).abs();

        DeathConservationCheck {
            at_death: at_death,
            landauer_equals_budget: (self.landauer - initial_free_energy).abs() < self.tolerance * initial_free_energy,
            death_conservation_holds: death_violation < self.tolerance * initial_free_energy,
            death_violation,
        }
    }
}

/// Conservation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Whether conservation holds.
    pub conserved: bool,
    /// Current conservation quantity.
    pub quantity: f64,
    /// Expected constant.
    pub constant: f64,
    /// Violation magnitude.
    pub violation: f64,
    /// Component breakdown.
    pub components: ConservationComponents,
}

/// Components of the conservation law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationComponents {
    pub landauer: f64,
    pub free_energy: f64,
    pub h1_risk: f64,
}

/// Death conservation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeathConservationCheck {
    /// Whether the agent is at death.
    pub at_death: bool,
    /// Whether Landauer cost equals the initial budget.
    pub landauer_equals_budget: bool,
    /// Whether conservation holds at death.
    pub death_conservation_holds: bool,
    /// Violation at death.
    pub death_violation: f64,
}

/// Verify conservation law across a sequence of states.
pub fn verify_conservation_across_lifecycle(
    states: &[(f64, f64, f64)], // (landauer, free_energy, h1_risk)
    initial_free_energy: f64,
    tolerance: f64,
) -> LifecycleConservationReport {
    let constant = initial_free_energy;
    let mut violations = Vec::new();
    let mut max_violation = 0.0f64;
    let mut all_conserved = true;

    for &(landauer, free_energy, h1_risk) in states {
        let qty = landauer + free_energy + h1_risk;
        let violation = (qty - constant).abs();
        violations.push(violation);
        if violation > tolerance * constant {
            all_conserved = false;
        }
        max_violation = max_violation.max(violation);
    }

    let avg_violation = if violations.is_empty() {
        0.0
    } else {
        violations.iter().sum::<f64>() / violations.len() as f64
    };

    LifecycleConservationReport {
        total_states: states.len(),
        all_conserved,
        max_violation,
        avg_violation,
        violations,
        conservation_constant: constant,
    }
}

/// Report of conservation across the full lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConservationReport {
    pub total_states: usize,
    pub all_conserved: bool,
    pub max_violation: f64,
    pub avg_violation: f64,
    pub violations: Vec<f64>,
    pub conservation_constant: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_conservation() {
        let state = ConservationState::new(100.0);
        assert!((state.conservation_constant - 100.0).abs() < 1e-10);
        assert!(state.is_conserved());
    }

    #[test]
    fn test_conservation_quantity() {
        let state = ConservationState::new(100.0);
        // Initial: 0 + 100 + 0 = 100
        assert!((state.conservation_quantity() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_with_spending() {
        let mut state = ConservationState::new(100.0);
        // Spend 30 on landauer: 30 + 70 + 0 = 100 ✓
        let report = state.update(30.0, 70.0, 0.0);
        assert!(report.conserved);
    }

    #[test]
    fn test_conservation_with_h1_risk() {
        let mut state = ConservationState::new(100.0);
        // 20 + 70 + 10 = 100 ✓
        let report = state.update(20.0, 70.0, 10.0);
        assert!(report.conserved);
    }

    #[test]
    fn test_conservation_violation() {
        let mut state = ConservationState::new(100.0);
        // 50 + 30 + 0 = 80 ≠ 100 ✗
        let report = state.update(50.0, 30.0, 0.0);
        assert!(!report.conserved);
        assert!(report.violation > 0.0);
    }

    #[test]
    fn test_relative_violation() {
        let mut state = ConservationState::new(100.0);
        state.update(50.0, 30.0, 0.0);
        assert!((state.relative_violation() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_death_conservation() {
        let mut state = ConservationState::new(100.0);
        state.update(100.0, 0.0, 0.0);
        let check = state.verify_death_conservation(100.0);
        assert!(check.at_death);
        assert!(check.landauer_equals_budget);
        assert!(check.death_conservation_holds);
    }

    #[test]
    fn test_death_conservation_with_h1() {
        let mut state = ConservationState::new(100.0);
        // At death: 90 + 5 + 5 = 100
        state.update(90.0, 5.0, 5.0);
        let check = state.verify_death_conservation(100.0);
        // Not quite at death (90 < 100)
        assert!(!check.at_death);
    }

    #[test]
    fn test_lifecycle_conservation() {
        let states = vec![
            (0.0, 100.0, 0.0),
            (10.0, 90.0, 0.0),
            (30.0, 70.0, 0.0),
            (50.0, 50.0, 0.0),
            (80.0, 20.0, 0.0),
            (100.0, 0.0, 0.0),
        ];
        let report = verify_conservation_across_lifecycle(&states, 100.0, 0.1);
        assert!(report.all_conserved);
        assert_eq!(report.total_states, 6);
    }

    #[test]
    fn test_lifecycle_conservation_with_h1() {
        let states = vec![
            (0.0, 90.0, 10.0),
            (20.0, 70.0, 10.0),
            (50.0, 40.0, 10.0),
            (90.0, 0.0, 10.0),
        ];
        let report = verify_conservation_across_lifecycle(&states, 100.0, 0.1);
        assert!(report.all_conserved);
    }

    #[test]
    fn test_lifecycle_conservation_violation() {
        let states = vec![
            (0.0, 100.0, 0.0),
            (10.0, 80.0, 0.0), // 90 ≠ 100
        ];
        let report = verify_conservation_across_lifecycle(&states, 100.0, 0.05);
        assert!(!report.all_conserved);
        assert!(report.max_violation > 0.0);
    }

    #[test]
    fn test_conservation_report_serialization() {
        let mut state = ConservationState::new(100.0);
        let report = state.update(30.0, 70.0, 0.0);
        let json = serde_json::to_string(&report).unwrap();
        let restored: ConservationReport = serde_json::from_str(&json).unwrap();
        assert!(restored.conserved);
    }
}
