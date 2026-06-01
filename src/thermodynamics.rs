//! Thermodynamics module — Landauer cost + Varadhan rate tracking.
//!
//! Landauer's principle: erasing one bit costs kT ln(2) of energy.
//! Varadhan's lemma: the rate function of large deviations gives the
//! thermodynamic cost of unlikely transitions.
//!
//! The agent must "pay rent" — track and budget its thermodynamic costs.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Boltzmann constant in natural units (kT at room temp ≈ 4.11e-21 J).
pub const BOLTZMANN_KT: f64 = 4.11e-21;

/// Landauer cost per bit erasure: kT ln(2).
pub const LANDAUER_PER_BIT: f64 = BOLTZMANN_KT * std::f64::consts::LN_2;

/// Thermodynamic budget and cost tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicBudget {
    /// Initial free energy budget.
    pub initial_free_energy: f64,
    /// Current free energy remaining.
    pub free_energy: f64,
    /// Cumulative Landauer cost (information erasure).
    pub cumulative_landauer: f64,
    /// Cumulative Varadhan cost (unlikely transition penalty).
    pub cumulative_varadhan: f64,
    /// Number of bits processed (erased).
    pub bits_erased: u64,
    /// Number of state transitions.
    pub transitions: u64,
}

/// Result of a thermodynamic cost computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicCost {
    /// Landauer cost for this operation.
    pub landauer_cost: f64,
    /// Varadhan rate cost for this transition.
    pub varadhan_cost: f64,
    /// Remaining budget after this operation.
    pub remaining_budget: f64,
    /// Whether the agent can afford this operation.
    pub affordable: bool,
}

impl ThermodynamicBudget {
    /// Create a new budget with given initial free energy.
    pub fn new(initial_free_energy: f64) -> Self {
        Self {
            initial_free_energy,
            free_energy: initial_free_energy,
            cumulative_landauer: 0.0,
            cumulative_varadhan: 0.0,
            bits_erased: 0,
            transitions: 0,
        }
    }

    /// Create a budget scaled for simulation (not physical units).
    pub fn simulation_budget(units: f64) -> Self {
        Self::new(units)
    }

    /// Compute Landauer cost for erasing n bits.
    pub fn landauer_cost(&self, bits: f64) -> f64 {
        bits * LANDAUER_PER_BIT
    }

    /// Compute Landauer cost in simulation units (scaled).
    pub fn landauer_cost_sim(&self, bits: f64) -> f64 {
        // Scale so costs are meaningful in simulation
        bits * 0.1
    }

    /// Compute Varadhan rate function for a state transition.
    ///
    /// Given old and new states, computes the large deviation rate:
    /// I(x→y) = sup_θ [θ·(y-x) - Λ(θ)]
    /// Simplified: rate ≈ |y-x|² / (2σ²) for Gaussian transitions.
    pub fn varadhan_rate(&self, old_state: &DVector<f64>, new_state: &DVector<f64>, sigma: f64) -> f64 {
        let diff = new_state - old_state;
        let sigma2 = sigma * sigma;
        diff.iter().map(|d| d * d / (2.0 * sigma2)).sum()
    }

    /// Spend energy on an operation.
    pub fn spend(&mut self, landauer: f64, varadhan: f64) -> ThermodynamicCost {
        let total = landauer + varadhan;
        let affordable = total <= self.free_energy;

        if affordable {
            self.free_energy -= total;
            self.cumulative_landauer += landauer;
            self.cumulative_varadhan += varadhan;
        }

        ThermodynamicCost {
            landauer_cost: landauer,
            varadhan_cost: varadhan,
            remaining_budget: self.free_energy,
            affordable,
        }
    }

    /// Erase bits and pay the Landauer cost (simulation units).
    pub fn erase_bits(&mut self, bits: f64) -> ThermodynamicCost {
        let cost = self.landauer_cost_sim(bits);
        let result = self.spend(cost, 0.0);
        if result.affordable {
            self.bits_erased += bits as u64;
        }
        result
    }

    /// Execute a state transition and pay Varadhan cost.
    pub fn transition(&mut self, old: &DVector<f64>, new: &DVector<f64>, sigma: f64) -> ThermodynamicCost {
        let cost = self.varadhan_rate(old, new, sigma) * 0.01; // scaled
        let result = self.spend(0.0, cost);
        if result.affordable {
            self.transitions += 1;
        }
        result
    }

    /// Total thermodynamic cost expended.
    pub fn total_cost(&self) -> f64 {
        self.cumulative_landauer + self.cumulative_varadhan
    }

    /// Fraction of budget spent.
    pub fn budget_fraction_spent(&self) -> f64 {
        if self.initial_free_energy > 0.0 {
            self.total_cost() / self.initial_free_energy
        } else {
            0.0
        }
    }

    /// Is the agent bankrupt (can't afford any operation)?
    pub fn is_bankrupt(&self) -> bool {
        self.free_energy <= 0.0
    }

    /// Death condition: cumulative Landauer cost equals initial free energy.
    pub fn should_die(&self) -> bool {
        self.cumulative_landauer >= self.initial_free_energy
    }

    /// Landauer cost as fraction of initial budget.
    pub fn landauer_fraction(&self) -> f64 {
        if self.initial_free_energy > 0.0 {
            self.cumulative_landauer / self.initial_free_energy
        } else {
            0.0
        }
    }

    /// Energy efficiency: useful work per unit of free energy spent.
    pub fn efficiency(&self) -> f64 {
        if self.total_cost() > 0.0 {
            1.0 - self.cumulative_varadhan / self.total_cost()
        } else {
            1.0
        }
    }

    /// Harvest free energy (e.g., from environment reward).
    pub fn harvest(&mut self, energy: f64) {
        self.free_energy += energy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_creation() {
        let budget = ThermodynamicBudget::new(100.0);
        assert!((budget.free_energy - 100.0).abs() < 1e-10);
        assert!((budget.initial_free_energy - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_affordable() {
        let mut budget = ThermodynamicBudget::new(100.0);
        let cost = budget.spend(30.0, 10.0);
        assert!(cost.affordable);
        assert!((budget.free_energy - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_unaffordable() {
        let mut budget = ThermodynamicBudget::new(10.0);
        let cost = budget.spend(50.0, 50.0);
        assert!(!cost.affordable);
        // Should not deduct
        assert!((budget.free_energy - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_erase_bits() {
        let mut budget = ThermodynamicBudget::simulation_budget(100.0);
        let cost = budget.erase_bits(10.0);
        assert!(cost.affordable);
        assert_eq!(budget.bits_erased, 10);
    }

    #[test]
    fn test_varadhan_rate() {
        let budget = ThermodynamicBudget::new(100.0);
        let old = DVector::from_vec(vec![0.0, 0.0]);
        let new = DVector::from_vec(vec![1.0, 1.0]);
        let rate = budget.varadhan_rate(&old, &new, 1.0);
        assert!(rate > 0.0);
        assert!((rate - 1.0).abs() < 1e-10); // (1+1)/(2*1) = 1.0
    }

    #[test]
    fn test_transition_cost() {
        let mut budget = ThermodynamicBudget::simulation_budget(100.0);
        let old = DVector::from_vec(vec![0.0]);
        let new = DVector::from_vec(vec![1.0]);
        let cost = budget.transition(&old, &new, 1.0);
        assert!(cost.affordable);
        assert_eq!(budget.transitions, 1);
    }

    #[test]
    fn test_total_cost() {
        let mut budget = ThermodynamicBudget::new(100.0);
        budget.spend(30.0, 20.0);
        assert!((budget.total_cost() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_bankruptcy() {
        let mut budget = ThermodynamicBudget::new(10.0);
        assert!(!budget.is_bankrupt());
        budget.spend(10.0, 0.0);
        assert!(budget.is_bankrupt());
    }

    #[test]
    fn test_death_condition() {
        let mut budget = ThermodynamicBudget::simulation_budget(10.0);
        // In simulation, landauer_cost_sim(10 bits) = 10 * 0.1 = 1.0
        assert!(!budget.should_die());
        // Erase enough bits to trigger death
        for _ in 0..100 {
            budget.erase_bits(10.0);
        }
        // Cumulative landauer = 100 * 1.0 = 100, which >= 10
        assert!(budget.should_die());
    }

    #[test]
    fn test_harvest_energy() {
        let mut budget = ThermodynamicBudget::new(10.0);
        budget.harvest(50.0);
        assert!((budget.free_energy - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_efficiency() {
        let mut budget = ThermodynamicBudget::new(100.0);
        budget.spend(80.0, 20.0);
        assert!((budget.efficiency() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_budget_fraction() {
        let mut budget = ThermodynamicBudget::new(100.0);
        budget.spend(25.0, 25.0);
        assert!((budget.budget_fraction_spent() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_landauer_fraction() {
        let mut budget = ThermodynamicBudget::simulation_budget(100.0);
        budget.erase_bits(50.0); // 50 * 0.1 = 5.0
        assert!((budget.landauer_fraction() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_physical_landauer_cost() {
        let budget = ThermodynamicBudget::new(1.0);
        let cost = budget.landauer_cost(1.0);
        assert!(cost > 0.0);
        assert!((cost - LANDAUER_PER_BIT).abs() < 1e-20);
    }
}
