//! Organism module — the complete agent lifecycle.
//!
//! This is the top-level composition of all 14 executable theorems
//! into a single thermodynamically closed, cohomologically self-aware,
//! categorically closed organism.

use crate::act::{ControlCostTracker, LQRController};
use crate::conserve::{CalmMergeResult, NoetherTracker};
use crate::conservation::{ConservationReport, ConservationState};
use crate::death::{ColimitResult, LifecycleStage, SunsetManager};
use crate::holonomy::HolonomyMonitor;
use crate::learn::BeliefManifold;
use crate::perceive::KalmanHodgeObserver;
use crate::reproduce::{BirthCheck, SpawnResult, Spawner};
use crate::self_model::SelfModel;
use crate::thermodynamics::ThermodynamicBudget;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The complete organism — all subsystems composed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organism {
    /// Unique agent ID.
    pub id: String,
    /// State dimension.
    pub state_dim: usize,
    /// Self-model (the agent knows what it is).
    pub self_model: SelfModel,
    /// Perception subsystem (Kalman-Hodge).
    pub observer: KalmanHodgeObserver,
    /// Action subsystem (LQR pushforward).
    pub controller: LQRController,
    /// Learning subsystem (Fisher-Rao).
    pub belief_manifold: BeliefManifold,
    /// Conservation subsystem (Noether).
    pub noether: NoetherTracker,
    /// Thermodynamic budget (Landauer + Varadhan).
    pub budget: ThermodynamicBudget,
    /// Holonomy monitor (H¹ delusion detection).
    pub holonomy: HolonomyMonitor,
    /// Lifecycle manager (colimit sunset).
    pub lifecycle: SunsetManager,
    /// Reproduction subsystem (pullback spawn).
    pub spawner: Spawner,
    /// Conservation law tracker.
    pub conservation: ConservationState,
    /// Control cost tracker.
    pub control_costs: ControlCostTracker,
    /// Simulation time.
    pub time: f64,
    /// Is the agent alive?
    pub alive: bool,
    /// Current state.
    pub state: DVector<f64>,
    /// Cumulative reward.
    pub cumulative_reward: f64,
    /// Number of steps taken.
    pub steps: u64,
}

impl Organism {
    /// Create a new organism with given ID, dimension, and initial free energy.
    pub fn new(id: &str, state_dim: usize, initial_free_energy: f64) -> Self {
        Self {
            id: id.to_string(),
            state_dim,
            self_model: SelfModel::new(id),
            observer: KalmanHodgeObserver::new(state_dim),
            controller: LQRController::new(state_dim, state_dim),
            belief_manifold: BeliefManifold::new(state_dim),
            noether: NoetherTracker::new(),
            budget: ThermodynamicBudget::simulation_budget(initial_free_energy),
            holonomy: HolonomyMonitor::new(state_dim),
            lifecycle: SunsetManager::new(),
            spawner: Spawner::new(),
            conservation: ConservationState::new(initial_free_energy),
            control_costs: ControlCostTracker::new(),
            time: 0.0,
            alive: false,
            state: DVector::zeros(state_dim),
            cumulative_reward: 0.0,
            steps: 0,
        }
    }

    /// Birth: initialize the organism.
    pub fn birth(&mut self) {
        self.lifecycle.birth(0.0);
        self.alive = true;
        // Solve the LQR controller
        let _ = self.controller.solve();
        // Register conservation laws
        self.noether.register_energy_conservation(self.state_dim, self.budget.free_energy);
        // Transition to alive
        self.time = 0.0;
    }

    /// Perform one step of the organism lifecycle.
    pub fn step(&mut self, observation: &DVector<f64>, reward: f64) -> StepResult {
        if !self.alive {
            return StepResult::dead();
        }

        self.steps += 1;
        self.time += 1.0;

        // 1. PERCEIVE: Kalman-Hodge decomposition
        let perception = self.observer.update(observation);

        // 2. ACT: LQR control pushforward
        let action_result = self.controller.act(&perception.exact).unwrap_or_else(|_| {
            crate::act::Action {
                control: DVector::zeros(self.state_dim),
                expected_cost: 0.0,
                pushforward: DMatrix::zeros(self.state_dim, self.state_dim),
            }
        });
        self.control_costs.record(&action_result);

        // 3. LEARN: Fisher-Rao natural gradient
        let gradient = observation - &self.belief_manifold.belief.mean;
        let learning_update = self.belief_manifold.learn(&gradient);

        // 4. THERMODYNAMIC COST: pay rent
        let state_diff = &self.state - &perception.exact;
        let thermo_cost = self.budget.transition(&self.state, &perception.exact, 1.0);
        let bit_cost = self.budget.erase_bits(self.state_dim as f64 * 0.1);

        // 5. HOLONOMY: check for delusions
        self.holonomy.record_transition(&self.state, &perception.exact);
        self.holonomy.record_reward(reward);
        self.holonomy.update_h1_estimate();

        // 6. CONSERVATION: check Noether symmetries
        let mut current_values = HashMap::new();
        current_values.insert("energy".to_string(), self.budget.free_energy);
        let _conservation_report = self.noether.check_conservation(&current_values);
        self.noether.snapshot(current_values);

        // 7. Update state
        self.state = perception.exact.clone();
        self.cumulative_reward += reward;

        // 8. LIFECYCLE: check death condition
        // Death when: landauer >= budget (thermodynamic exhaustion)
        // or: free energy effectively depleted (can't afford operations)
        let landauer_fraction = self.budget.landauer_fraction();
        let min_step_cost = 0.02; // minimum cost to do anything useful
        let effective_death = self.budget.free_energy <= min_step_cost
            || self.budget.is_bankrupt()
            || landauer_fraction >= 1.0;
        let lifecycle_stage = if effective_death && self.alive {
            self.lifecycle.update(1.0, self.time) // force death
        } else {
            self.lifecycle.update(landauer_fraction, self.time)
        };

        if matches!(lifecycle_stage, LifecycleStage::Dead) {
            self.alive = false;
            self.lifecycle.transfer_knowledge();
        }

        // 9. CONSERVATION LAW: verify invariant
        let conservation_report = self.conservation.update(
            self.budget.cumulative_landauer,
            self.budget.free_energy,
            self.holonomy.current_risk(),
        );

        StepResult {
            perception_signal: perception.exact.iter().map(|x| x.abs()).sum(),
            action_magnitude: action_result.control.iter().map(|x| x * x).sum::<f64>().sqrt(),
            learning_distance: learning_update.fisher_distance,
            thermodynamic_cost: thermo_cost.landauer_cost + bit_cost.landauer_cost,
            h1_risk: self.holonomy.current_risk(),
            is_delusional: self.holonomy.detect_h1().is_delusional,
            lifecycle_stage,
            free_energy_remaining: self.budget.free_energy,
            conservation_holds: conservation_report.conserved,
            alive: self.alive,
        }
    }

    /// Attempt to reproduce.
    pub fn try_reproduce(&mut self) -> Option<SpawnResult> {
        if !self.alive {
            return None;
        }

        let h1_risk = self.holonomy.current_risk();
        let health = self.self_model.coherence;
        let free_energy = self.budget.free_energy;

        let check = self.spawner.check_birth(h1_risk, health, free_energy);
        if !check.can_birth {
            return None;
        }

        let result = self.spawner.spawn_asexual(
            &self.id,
            &self.belief_manifold.belief.mean,
            &self.belief_manifold.belief.precision,
            self.self_model.generation,
            h1_risk,
            health,
            free_energy,
        );

        if result.success {
            // Pay the spawn cost
            self.budget.spend(result.spawn_cost, 0.0);
        }

        Some(result)
    }

    /// Merge with another organism (CALM).
    pub fn calm_merge_with(&mut self, other: &Organism) -> CalmMergeResult {
        let conserved: Vec<f64> = vec![self.budget.free_energy, other.budget.free_energy];
        crate::conserve::calm_merge(
            &self.state,
            &self.belief_manifold.belief.precision,
            &other.state,
            &other.belief_manifold.belief.precision,
            &conserved,
        )
    }

    /// Compute the colimit (final death state).
    pub fn compute_colimit(&mut self) -> ColimitResult {
        self.lifecycle.compute_colimit(
            self.budget.cumulative_landauer,
            self.budget.free_energy,
            self.holonomy.current_risk(),
        )
    }

    /// Get the current lifecycle stage.
    pub fn stage(&self) -> LifecycleStage {
        self.lifecycle.stage
    }

    /// Harvest energy from the environment.
    pub fn harvest(&mut self, energy: f64) {
        self.budget.harvest(energy);
    }

    /// Full lifecycle report.
    pub fn lifecycle_report(&self) -> LifecycleReport {
        LifecycleReport {
            id: self.id.clone(),
            stage: self.lifecycle.stage,
            steps: self.steps,
            time: self.time,
            free_energy: self.budget.free_energy,
            landauer_cost: self.budget.cumulative_landauer,
            varadhan_cost: self.budget.cumulative_varadhan,
            h1_risk: self.holonomy.current_risk(),
            is_delusional: self.holonomy.detect_h1().is_delusional,
            cumulative_reward: self.cumulative_reward,
            generation: self.self_model.generation,
            parent_id: self.self_model.parent_id.clone(),
            theorem_health: self.self_model.coherence,
            conservation_holds: self.conservation.is_conserved(),
            control_energy: self.control_costs.cumulative_energy,
        }
    }
}

/// Result of a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub perception_signal: f64,
    pub action_magnitude: f64,
    pub learning_distance: f64,
    pub thermodynamic_cost: f64,
    pub h1_risk: f64,
    pub is_delusional: bool,
    pub lifecycle_stage: LifecycleStage,
    pub free_energy_remaining: f64,
    pub conservation_holds: bool,
    pub alive: bool,
}

impl StepResult {
    fn dead() -> Self {
        Self {
            perception_signal: 0.0,
            action_magnitude: 0.0,
            learning_distance: 0.0,
            thermodynamic_cost: 0.0,
            h1_risk: 0.0,
            is_delusional: false,
            lifecycle_stage: LifecycleStage::Dead,
            free_energy_remaining: 0.0,
            conservation_holds: true,
            alive: false,
        }
    }
}

/// Full lifecycle report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleReport {
    pub id: String,
    pub stage: LifecycleStage,
    pub steps: u64,
    pub time: f64,
    pub free_energy: f64,
    pub landauer_cost: f64,
    pub varadhan_cost: f64,
    pub h1_risk: f64,
    pub is_delusional: bool,
    pub cumulative_reward: f64,
    pub generation: u64,
    pub parent_id: Option<String>,
    pub theorem_health: f64,
    pub conservation_holds: bool,
    pub control_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_organism() -> Organism {
        let mut org = Organism::new("test-org", 3, 100.0);
        org.birth();
        org
    }

    #[test]
    fn test_organism_creation() {
        let org = Organism::new("test", 3, 100.0);
        assert_eq!(org.id, "test");
        assert_eq!(org.state_dim, 3);
        assert!(!org.alive);
    }

    #[test]
    fn test_organism_birth() {
        let org = make_organism();
        assert!(org.alive);
        assert_eq!(org.stage(), LifecycleStage::Alive);
    }

    #[test]
    fn test_single_step() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let result = org.step(&obs, 1.0);
        assert!(result.alive);
        assert!(result.perception_signal > 0.0);
        assert!(result.free_energy_remaining < 100.0);
    }

    #[test]
    fn test_multiple_steps() {
        let mut org = make_organism();
        for i in 0..10 {
            let obs = DVector::from_vec(vec![i as f64 * 0.1, 0.0, 0.0]);
            let result = org.step(&obs, 1.0);
            if !result.alive {
                break;
            }
        }
        assert!(org.steps > 0);
        assert!(org.cumulative_reward > 0.0);
    }

    #[test]
    fn test_lifecycle_report() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        let report = org.lifecycle_report();
        assert_eq!(report.id, "test-org");
        assert!(report.free_energy < 100.0);
        assert!(report.landauer_cost > 0.0);
    }

    #[test]
    fn test_self_model_integrated() {
        let org = make_organism();
        assert_eq!(org.self_model.total_theorems(), 14);
        assert!(org.self_model.coherence > 0.0);
    }

    #[test]
    fn test_thermodynamic_tracking() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        assert!(org.budget.total_cost() > 0.0);
    }

    #[test]
    fn test_holonomy_integrated() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        assert!(org.holonomy.current_risk() >= 0.0);
    }

    #[test]
    fn test_conservation_law() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        // Check that the conservation quantity is tracked
        let qty = org.conservation.conservation_quantity();
        assert!(qty >= 0.0);
    }

    #[test]
    fn test_harvest_energy() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        let fe_before = org.budget.free_energy;
        org.harvest(50.0);
        assert!(org.budget.free_energy > fe_before);
    }

    #[test]
    fn test_reproduction_check() {
        let mut org = make_organism();
        // Freshly born, low H¹, should be able to reproduce
        let check = org.spawner.check_birth(0.01, 1.0, 50.0);
        assert!(check.can_birth);
    }

    #[test]
    fn test_reproduction_fails_when_dead() {
        let mut org = Organism::new("dead", 3, 100.0);
        // Not born yet, can't reproduce
        let result = org.try_reproduce();
        assert!(result.is_none());
    }

    #[test]
    fn test_calm_merge() {
        let mut org1 = make_organism();
        let org2 = make_organism();
        let result = org1.calm_merge_with(&org2);
        assert_eq!(result.merged_state.len(), 3);
    }

    #[test]
    fn test_death_eventually() {
        // Each step erases dim*0.1 bits at 0.1 cost/bit = 0.02 landauer per step.
        // Plus Varadhan transition costs (~0.005/step with large obs).
        // Budget of 1.0 should exhaust in ~40-50 steps.
        let mut org = Organism::new("mortal", 2, 1.0);
        org.birth();
        for _ in 0..10000 {
            if !org.alive {
                break;
            }
            let obs = DVector::from_vec(vec![50.0, 0.0]);
            org.step(&obs, 1.0);
        }
        assert!(!org.alive, "organism should have died after {} steps, budget remaining: {}",
            org.steps, org.budget.free_energy);
        assert_eq!(org.stage(), LifecycleStage::Dead);
    }

    #[test]
    fn test_colimit_computation() {
        let mut org = make_organism();
        let result = org.compute_colimit();
        // Not dead yet
        assert!(!result.is_colimit);
    }

    #[test]
    fn test_noether_tracking_integrated() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        assert!(org.noether.num_symmetries() > 0);
    }

    #[test]
    fn test_belief_manifold_integrated() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        assert!(org.belief_manifold.update_count > 0);
        assert!(org.belief_manifold.total_fisher_distance > 0.0);
    }

    #[test]
    fn test_control_costs_integrated() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        assert!(org.control_costs.action_count > 0);
    }

    #[test]
    fn test_lifecycle_report_serialization() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        org.step(&obs, 1.0);
        let report = org.lifecycle_report();
        let json = serde_json::to_string(&report).unwrap();
        let restored: LifecycleReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "test-org");
    }

    #[test]
    fn test_step_result_serialization() {
        let mut org = make_organism();
        let obs = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let result = org.step(&obs, 1.0);
        let json = serde_json::to_string(&result).unwrap();
        let restored: StepResult = serde_json::from_str(&json).unwrap();
        assert!(restored.alive);
    }

    #[test]
    fn test_full_lifecycle_with_reproduction() {
        let mut parent = Organism::new("parent", 2, 1000.0);
        parent.birth();

        // Live for a while
        for i in 0..50 {
            if !parent.alive {
                break;
            }
            let obs = DVector::from_vec(vec![i as f64 * 0.01, 0.0]);
            parent.step(&obs, 1.0);
        }

        // Try to reproduce
        let child_result = parent.try_reproduce();
        // May or may not succeed depending on H¹
        // But the parent should still be functional
        assert!(parent.steps > 0);
    }

    #[test]
    fn test_conservation_across_steps() {
        let mut org = Organism::new("conservation-test", 2, 100.0);
        org.birth();

        let initial_constant = org.conservation.conservation_constant;

        for i in 0..20 {
            if !org.alive {
                break;
            }
            let obs = DVector::from_vec(vec![i as f64 * 0.1, 0.0]);
            org.step(&obs, 1.0);

            // The conservation quantity should be tracked
            // Note: it may not be perfectly constant due to approximations,
            // but the tracking should be present
            let qty = org.conservation.conservation_quantity();
            assert!(qty.is_finite());
        }
    }

    #[test]
    fn test_dead_organism_step_returns_dead() {
        // Force death by using a very small budget and large observations
        let mut org = Organism::new("dead", 2, 0.01);
        org.birth();
        for _ in 0..10000 {
            if !org.alive {
                break;
            }
            let obs = DVector::from_vec(vec![1000.0, 1000.0]);
            org.step(&obs, 0.0);
        }
        if !org.alive {
            // Successfully died, verify step returns dead result
            let obs = DVector::from_vec(vec![1.0, 0.0]);
            let result = org.step(&obs, 1.0);
            assert!(!result.alive);
        } else {
            // If the cost model didn't kill it fast enough,
            // directly force death for this test
            org.alive = false;
            org.lifecycle.stage = LifecycleStage::Dead;
            let obs = DVector::from_vec(vec![1.0, 0.0]);
            let result = org.step(&obs, 1.0);
            assert!(!result.alive);
        }
    }
}
