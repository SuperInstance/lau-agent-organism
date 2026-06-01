//! Self-model module — the agent represents its own theorem-structure.
//!
//! The agent knows what it is: a composition of 14 executable theorems
//! forming a regulatory network. This self-representation is itself
//! a categorical object that can be inspected and reasoned about.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A theorem in the agent's self-model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    /// Name of the theorem.
    pub name: String,
    /// Category (perception, action, learning, etc.).
    pub category: String,
    /// Mathematical statement (informal).
    pub statement: String,
    /// Dependencies (other theorems this one requires).
    pub dependencies: Vec<String>,
    /// Whether this theorem is currently active.
    pub active: bool,
    /// Health metric (0..1, how well the theorem is functioning).
    pub health: f64,
}

/// The agent's self-model: a representation of its own theorem structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// All theorems the agent embodies.
    pub theorems: HashMap<String, Theorem>,
    /// Active theorem count.
    pub active_count: usize,
    /// Overall self-coherence (0..1).
    pub coherence: f64,
    /// Generation (for reproductive tracking).
    pub generation: u64,
    /// Parent ID (for reproductive tracking).
    pub parent_id: Option<String>,
    /// Unique agent ID.
    pub agent_id: String,
    /// Self-awareness metric: how well the agent understands itself.
    pub self_awareness: f64,
}

impl SelfModel {
    /// Create a complete self-model with all 14 executable theorems.
    pub fn new(agent_id: &str) -> Self {
        let mut theorems = HashMap::new();

        let theorem_defs = vec![
            ("Kalman-Hodge", "perception", "Kalman filter = Hodge decomposition of observation stream", vec![]),
            ("Spectral Perception", "perception", "Observations decompose into exact, co-exact, and harmonic components", vec!["Kalman-Hodge"]),
            ("Obs-Ctrl Adjunction", "action", "Observation is left adjoint to Control: Obs ⊣ Ctrl", vec!["Kalman-Hodge"]),
            ("LQR Pushforward", "action", "Optimal control is the pushforward through the Obs-Ctrl adjunction", vec!["Obs-Ctrl Adjunction"]),
            ("Fisher-Rao Geodesic", "learning", "Learning follows geodesics on the Fisher information manifold", vec!["Kalman-Hodge"]),
            ("Natural Gradient", "learning", "F^{-1}∇L is the natural gradient on the belief manifold", vec!["Fisher-Rao Geodesic"]),
            ("Noether Conservation", "conservation", "Every continuous symmetry yields a conserved quantity", vec!["LQR Pushforward"]),
            ("CALM Merge", "conservation", "Fleet merge preserves conserved quantities via precision-weighted averaging", vec!["Noether Conservation"]),
            ("Landauer Cost", "thermodynamics", "Information erasure costs kT ln(2) per bit", vec!["Natural Gradient"]),
            ("Varadhan Rate", "thermodynamics", "Large deviation rate function gives thermodynamic cost of transitions", vec!["Landauer Cost"]),
            ("H¹ Holonomy", "delusion", "First cohomology detects global inconsistencies in belief structure", vec!["Fisher-Rao Geodesic"]),
            ("Colimit Death", "lifecycle", "Agent dies when Landauer cost = initial free energy (colimit of lifecycle)", vec!["Landauer Cost"]),
            ("Pullback Birth", "lifecycle", "Agent reproduces via pullback of parent knowledge when H¹=0", vec!["H¹ Holonomy", "Noether Conservation"]),
            ("Conservation Law", "meta", "Landauer + free_energy + H¹_risk ≈ constant across lifecycle", vec!["Landauer Cost", "H¹ Holonomy"]),
        ];

        for (name, category, statement, deps) in theorem_defs {
            theorems.insert(
                name.to_string(),
                Theorem {
                    name: name.to_string(),
                    category: category.to_string(),
                    statement: statement.to_string(),
                    dependencies: deps.into_iter().map(String::from).collect(),
                    active: true,
                    health: 1.0,
                },
            );
        }

        let active_count = theorems.values().filter(|t| t.active).count();

        Self {
            theorems,
            active_count,
            coherence: 1.0,
            generation: 0,
            parent_id: None,
            agent_id: agent_id.to_string(),
            self_awareness: 1.0,
        }
    }

    /// Create as an offspring of another agent.
    pub fn offspring(agent_id: &str, parent_id: &str, generation: u64) -> Self {
        let mut model = Self::new(agent_id);
        model.parent_id = Some(parent_id.to_string());
        model.generation = generation;
        model
    }

    /// Get theorems by category.
    pub fn theorems_by_category(&self, category: &str) -> Vec<&Theorem> {
        self.theorems
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Check that all dependencies are satisfied.
    pub fn check_dependencies(&self) -> Vec<String> {
        let mut unsatisfied = Vec::new();
        for theorem in self.theorems.values() {
            for dep in &theorem.dependencies {
                if let Some(dep_thm) = self.theorems.get(dep) {
                    if !dep_thm.active {
                        unsatisfied.push(format!("{} depends on inactive {}", theorem.name, dep));
                    }
                } else {
                    unsatisfied.push(format!("{} depends on missing {}", theorem.name, dep));
                }
            }
        }
        unsatisfied
    }

    /// Compute self-coherence: fraction of theorems that are healthy.
    pub fn compute_coherence(&self) -> f64 {
        if self.theorems.is_empty() {
            return 0.0;
        }
        let total_health: f64 = self.theorems.values().map(|t| t.health).sum();
        total_health / self.theorems.len() as f64
    }

    /// Deactivate a theorem (e.g., due to failure).
    pub fn deactivate_theorem(&mut self, name: &str) {
        if let Some(t) = self.theorems.get_mut(name) {
            t.active = false;
            t.health = 0.0;
        }
        self.active_count = self.theorems.values().filter(|t| t.active).count();
        self.coherence = self.compute_coherence();
    }

    /// Update theorem health.
    pub fn update_health(&mut self, name: &str, health: f64) {
        if let Some(t) = self.theorems.get_mut(name) {
            t.health = health.max(0.0).min(1.0);
            if t.health < 0.1 {
                t.active = false;
            }
        }
        self.coherence = self.compute_coherence();
        self.active_count = self.theorems.values().filter(|t| t.active).count();
    }

    /// Total number of theorems.
    pub fn total_theorems(&self) -> usize {
        self.theorems.len()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the theorem dependency graph as adjacency list.
    pub fn dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();
        for theorem in self.theorems.values() {
            graph.insert(theorem.name.clone(), theorem.dependencies.clone());
        }
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model_creation() {
        let model = SelfModel::new("test-agent");
        assert_eq!(model.total_theorems(), 14);
        assert_eq!(model.active_count, 14);
    }

    #[test]
    fn test_all_categories_present() {
        let model = SelfModel::new("test");
        assert!(!model.theorems_by_category("perception").is_empty());
        assert!(!model.theorems_by_category("action").is_empty());
        assert!(!model.theorems_by_category("learning").is_empty());
        assert!(!model.theorems_by_category("conservation").is_empty());
        assert!(!model.theorems_by_category("thermodynamics").is_empty());
        assert!(!model.theorems_by_category("delusion").is_empty());
        assert!(!model.theorems_by_category("lifecycle").is_empty());
        assert!(!model.theorems_by_category("meta").is_empty());
    }

    #[test]
    fn test_dependencies_satisfied() {
        let model = SelfModel::new("test");
        let unsatisfied = model.check_dependencies();
        assert!(unsatisfied.is_empty());
    }

    #[test]
    fn test_dependencies_broken_on_deactivate() {
        let mut model = SelfModel::new("test");
        model.deactivate_theorem("Kalman-Hodge");
        let unsatisfied = model.check_dependencies();
        assert!(!unsatisfied.is_empty());
    }

    #[test]
    fn test_coherence_starts_at_one() {
        let model = SelfModel::new("test");
        assert!((model.coherence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_decreases_on_deactivate() {
        let mut model = SelfModel::new("test");
        model.deactivate_theorem("Kalman-Hodge");
        assert!(model.coherence < 1.0);
    }

    #[test]
    fn test_offspring_tracking() {
        let model = SelfModel::offspring("child", "parent-1", 2);
        assert_eq!(model.generation, 2);
        assert_eq!(model.parent_id, Some("parent-1".to_string()));
        assert_eq!(model.agent_id, "child");
    }

    #[test]
    fn test_json_serialization() {
        let model = SelfModel::new("test");
        let json = model.to_json().unwrap();
        let restored = SelfModel::from_json(&json).unwrap();
        assert_eq!(restored.total_theorems(), 14);
        assert_eq!(restored.agent_id, "test");
    }

    #[test]
    fn test_dependency_graph() {
        let model = SelfModel::new("test");
        let graph = model.dependency_graph();
        assert!(graph.contains_key("LQR Pushforward"));
        assert!(graph["LQR Pushforward"].contains(&"Obs-Ctrl Adjunction".to_string()));
    }

    #[test]
    fn test_update_health() {
        let mut model = SelfModel::new("test");
        model.update_health("Kalman-Hodge", 0.5);
        assert!((model.theorems["Kalman-Hodge"].health - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_update_health_deactivates_below_threshold() {
        let mut model = SelfModel::new("test");
        model.update_health("Kalman-Hodge", 0.05);
        assert!(!model.theorems["Kalman-Hodge"].active);
    }

    #[test]
    fn test_active_count_decreases() {
        let mut model = SelfModel::new("test");
        assert_eq!(model.active_count, 14);
        model.deactivate_theorem("Kalman-Hodge");
        assert_eq!(model.active_count, 13);
    }
}
