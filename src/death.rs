//! Death module — colimit sunset when Landauer cost exceeds free energy.
//!
//! The lifecycle of the agent is a diagram in a category. Death is the
//! colimit of this diagram — the universal object that all lifecycle
//! stages map into. The agent dies when its cumulative thermodynamic
//! cost equals its initial free energy budget.

use serde::{Deserialize, Serialize};

/// Lifecycle stage of the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleStage {
    /// Agent is being spawned (pullback birth).
    Gestating,
    /// Agent is alive and operational.
    Alive,
    /// Agent's free energy is running low.
    Declining,
    /// Agent has entered sunset protocol.
    Sunset,
    /// Agent is dead (colimit reached).
    Dead,
}

impl std::fmt::Display for LifecycleStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleStage::Gestating => write!(f, "Gestating"),
            LifecycleStage::Alive => write!(f, "Alive"),
            LifecycleStage::Declining => write!(f, "Declining"),
            LifecycleStage::Sunset => write!(f, "Sunset"),
            LifecycleStage::Dead => write!(f, "Dead"),
        }
    }
}

/// Death condition parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeathCondition {
    /// The threshold fraction at which sunset begins.
    pub sunset_threshold: f64,
    /// The threshold fraction at which declining begins.
    pub declining_threshold: f64,
    /// Whether death is irreversible.
    pub irreversible: bool,
}

impl Default for DeathCondition {
    fn default() -> Self {
        Self {
            sunset_threshold: 0.9,
            declining_threshold: 0.7,
            irreversible: true,
        }
    }
}

/// Colimit sunset manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunsetManager {
    /// Current lifecycle stage.
    pub stage: LifecycleStage,
    /// Death condition parameters.
    pub condition: DeathCondition,
    /// Fraction of free energy spent (0..1).
    pub spent_fraction: f64,
    /// Whether sunset has been initiated.
    pub sunset_initiated: bool,
    /// Final message/state before death.
    pub final_state: Option<String>,
    /// Timestamps of stage transitions.
    pub transitions: Vec<(LifecycleStage, f64)>,
    /// Whether knowledge has been transferred (for reproduction).
    pub knowledge_transferred: bool,
}

impl SunsetManager {
    pub fn new() -> Self {
        Self {
            stage: LifecycleStage::Gestating,
            condition: DeathCondition::default(),
            spent_fraction: 0.0,
            sunset_initiated: false,
            final_state: None,
            transitions: Vec::new(),
            knowledge_transferred: false,
        }
    }

    /// Transition to alive stage.
    pub fn birth(&mut self, time: f64) {
        self.stage = LifecycleStage::Alive;
        self.transitions.push((LifecycleStage::Alive, time));
    }

    /// Update lifecycle stage based on free energy fraction spent.
    pub fn update(&mut self, landauer_fraction: f64, time: f64) -> LifecycleStage {
        self.spent_fraction = landauer_fraction;

        let new_stage = if landauer_fraction >= 1.0 {
            LifecycleStage::Dead
        } else if landauer_fraction >= self.condition.sunset_threshold && !self.sunset_initiated {
            self.sunset_initiated = true;
            LifecycleStage::Sunset
        } else if landauer_fraction >= self.condition.declining_threshold {
            LifecycleStage::Declining
        } else {
            self.stage
        };

        if new_stage != self.stage {
            self.transitions.push((new_stage, time));
            self.stage = new_stage;
        }

        self.stage
    }

    /// Check if the agent should die.
    pub fn should_die(&self) -> bool {
        matches!(self.stage, LifecycleStage::Dead)
    }

    /// Check if in sunset phase.
    pub fn in_sunset(&self) -> bool {
        matches!(self.stage, LifecycleStage::Sunset) || matches!(self.stage, LifecycleStage::Declining)
    }

    /// Record final state before death.
    pub fn set_final_state(&mut self, state: &str) {
        self.final_state = Some(state.to_string());
    }

    /// Mark knowledge as transferred (prerequisite for clean death).
    pub fn transfer_knowledge(&mut self) {
        self.knowledge_transferred = true;
    }

    /// Execute the colimit: compute the final universal object.
    ///
    /// The colimit of the lifecycle diagram is the agent's complete
    /// history, collapsed into a single point. This is death.
    pub fn compute_colimit(&mut self, landauer_cost: f64, free_energy: f64, h1_risk: f64) -> ColimitResult {
        let dead = landauer_cost >= free_energy;

        let result = ColimitResult {
            is_colimit: dead,
            landauer_at_death: if dead { landauer_cost } else { 0.0 },
            free_energy_remaining: free_energy - landauer_cost,
            h1_risk_at_death: h1_risk,
            knowledge_transferred: self.knowledge_transferred,
            stage: self.stage,
        };

        if dead {
            self.stage = LifecycleStage::Dead;
        }

        result
    }

    /// Get the number of lifecycle transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    /// Time alive (difference between birth and death transitions).
    pub fn lifespan(&self) -> f64 {
        let birth_time = self
            .transitions
            .iter()
            .find(|(s, _)| *s == LifecycleStage::Alive)
            .map(|&(_, t)| t);
        let death_time = self
            .transitions
            .iter()
            .find(|(s, _)| *s == LifecycleStage::Dead)
            .map(|&(_, t)| t);

        match (birth_time, death_time) {
            (Some(b), Some(d)) => d - b,
            (Some(b), None) => -b, // still alive
            _ => 0.0,
        }
    }
}

/// Result of computing the lifecycle colimit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColimitResult {
    /// Whether the colimit condition is met (death).
    pub is_colimit: bool,
    /// Landauer cost at the moment of death.
    pub landauer_at_death: f64,
    /// Free energy remaining (should be ≈ 0 at death).
    pub free_energy_remaining: f64,
    /// H¹ risk at death.
    pub h1_risk_at_death: f64,
    /// Whether knowledge was transferred before death.
    pub knowledge_transferred: bool,
    /// Current lifecycle stage.
    pub stage: LifecycleStage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_stage_is_gestating() {
        let mgr = SunsetManager::new();
        assert_eq!(mgr.stage, LifecycleStage::Gestating);
    }

    #[test]
    fn test_birth_transition() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        assert_eq!(mgr.stage, LifecycleStage::Alive);
        assert_eq!(mgr.num_transitions(), 1);
    }

    #[test]
    fn test_declining_stage() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        let stage = mgr.update(0.75, 1.0);
        assert_eq!(stage, LifecycleStage::Declining);
    }

    #[test]
    fn test_sunset_stage() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        let stage = mgr.update(0.95, 1.0);
        assert_eq!(stage, LifecycleStage::Sunset);
        assert!(mgr.sunset_initiated);
    }

    #[test]
    fn test_death_stage() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        let stage = mgr.update(1.0, 1.0);
        assert_eq!(stage, LifecycleStage::Dead);
    }

    #[test]
    fn test_should_die() {
        let mut mgr = SunsetManager::new();
        assert!(!mgr.should_die());
        mgr.stage = LifecycleStage::Dead;
        assert!(mgr.should_die());
    }

    #[test]
    fn test_in_sunset() {
        let mut mgr = SunsetManager::new();
        assert!(!mgr.in_sunset());
        mgr.stage = LifecycleStage::Sunset;
        assert!(mgr.in_sunset());
    }

    #[test]
    fn test_colimit_computation() {
        let mut mgr = SunsetManager::new();
        let result = mgr.compute_colimit(100.0, 50.0, 0.01);
        assert!(result.is_colimit);
        assert!((result.free_energy_remaining - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_colimit_not_yet() {
        let mut mgr = SunsetManager::new();
        let result = mgr.compute_colimit(50.0, 100.0, 0.01);
        assert!(!result.is_colimit);
    }

    #[test]
    fn test_knowledge_transfer() {
        let mut mgr = SunsetManager::new();
        assert!(!mgr.knowledge_transferred);
        mgr.transfer_knowledge();
        assert!(mgr.knowledge_transferred);
    }

    #[test]
    fn test_final_state() {
        let mut mgr = SunsetManager::new();
        mgr.set_final_state("all knowledge transferred");
        assert_eq!(mgr.final_state, Some("all knowledge transferred".to_string()));
    }

    #[test]
    fn test_lifecycle_display() {
        assert_eq!(format!("{}", LifecycleStage::Alive), "Alive");
        assert_eq!(format!("{}", LifecycleStage::Dead), "Dead");
    }

    #[test]
    fn test_lifespan() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        mgr.update(1.0, 100.0);
        let ls = mgr.lifespan();
        assert!((ls - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut mgr = SunsetManager::new();
        mgr.birth(0.0);
        assert_eq!(mgr.stage, LifecycleStage::Alive);

        mgr.update(0.5, 10.0);
        assert_eq!(mgr.stage, LifecycleStage::Alive);

        mgr.update(0.75, 20.0);
        assert_eq!(mgr.stage, LifecycleStage::Declining);

        mgr.update(0.92, 30.0);
        assert_eq!(mgr.stage, LifecycleStage::Sunset);

        mgr.update(1.0, 40.0);
        assert_eq!(mgr.stage, LifecycleStage::Dead);

        assert_eq!(mgr.num_transitions(), 4); // birth, declining, sunset, dead
    }
}
