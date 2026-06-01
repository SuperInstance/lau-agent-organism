//! Reproduction module — pullback spawn with knowledge crossover.
//!
//! Reproduction is a pullback in the category of agents: the child is
//! the limit of the diagram formed by the parent's knowledge and the
//! consistency constraint (H¹ = 0). Birth succeeds only when the
//! pullback is consistent — no delusions are inherited.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Birth condition check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirthCheck {
    /// Is H¹ zero (consistent knowledge)?
    pub h1_zero: bool,
    /// H¹ risk value.
    pub h1_risk: f64,
    /// H¹ threshold for birth.
    pub h1_threshold: f64,
    /// Is the parent healthy enough to reproduce?
    pub parent_healthy: bool,
    /// Is there sufficient free energy for spawning?
    pub sufficient_energy: bool,
    /// Overall: can birth proceed?
    pub can_birth: bool,
}

/// Knowledge payload for reproduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePayload {
    /// Parent belief mean.
    pub belief_mean: DVector<f64>,
    /// Parent belief precision.
    pub belief_precision: DMatrix<f64>,
    /// Parent self-model JSON.
    pub self_model_json: String,
    /// Parent generation.
    pub generation: u64,
    /// Conserved quantities.
    pub conserved_quantities: Vec<f64>,
    /// Knowledge crossover mask (which dimensions to inherit).
    pub crossover_mask: Vec<bool>,
}

/// Result of a pullback spawn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnResult {
    /// Child agent ID.
    pub child_id: String,
    /// Child generation.
    pub generation: u64,
    /// Child belief mean.
    pub belief_mean: DVector<f64>,
    /// Child belief precision.
    pub belief_precision: DMatrix<f64>,
    /// Birth H¹ risk.
    pub birth_h1_risk: f64,
    /// Energy cost of spawning.
    pub spawn_cost: f64,
    /// Whether spawn succeeded.
    pub success: bool,
}

/// Energy cost of spawning as fraction of parent's energy.
pub const SPAWN_ENERGY_FRACTION: f64 = 0.3;

/// Pullback spawner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spawner {
    /// H¹ threshold for birth (must be below this).
    pub h1_threshold: f64,
    /// Minimum parent health for reproduction.
    pub min_parent_health: f64,
    /// Minimum free energy for spawning.
    pub min_free_energy: f64,
    /// Energy cost of spawning.
    pub spawn_cost: f64,
    /// Mutation rate (noise in crossover).
    pub mutation_rate: f64,
    /// Next child ID counter.
    pub next_child_id: u64,
}

impl Spawner {
    pub fn new() -> Self {
        Self {
            h1_threshold: 0.1,
            min_parent_health: 0.5,
            min_free_energy: 10.0,
            spawn_cost: SPAWN_ENERGY_FRACTION,
            mutation_rate: 0.01,
            next_child_id: 0,
        }
    }

    /// Check if birth conditions are met.
    pub fn check_birth(
        &self,
        h1_risk: f64,
        parent_health: f64,
        parent_free_energy: f64,
    ) -> BirthCheck {
        let h1_zero = h1_risk < self.h1_threshold;
        let parent_healthy = parent_health >= self.min_parent_health;
        let sufficient_energy = parent_free_energy >= self.min_free_energy;
        let can_birth = h1_zero && parent_healthy && sufficient_energy;

        BirthCheck {
            h1_zero,
            h1_risk,
            h1_threshold: self.h1_threshold,
            parent_healthy,
            sufficient_energy,
            can_birth,
        }
    }

    /// Perform knowledge crossover between two parents.
    pub fn crossover(
        &self,
        parent_a_mean: &DVector<f64>,
        parent_a_precision: &DMatrix<f64>,
        parent_b_mean: &DVector<f64>,
        parent_b_precision: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let dim = parent_a_mean.len();
        let mut child_mean = DVector::zeros(dim);
        let mut child_precision = DMatrix::zeros(dim, dim);

        for i in 0..dim {
            // Crossover: pick from each parent with 50/50
            if rand_bool() {
                child_mean[i] = parent_a_mean[i];
                for j in 0..dim {
                    child_precision[(i, j)] = parent_a_precision[(i, j)];
                }
            } else {
                child_mean[i] = parent_b_mean[i];
                for j in 0..dim {
                    child_precision[(i, j)] = parent_b_precision[(i, j)];
                }
            }

            // Mutation
            if random::<f64>() < self.mutation_rate {
                child_mean[i] += random::<f64>() * 0.1 - 0.05;
            }
        }

        (child_mean, child_precision)
    }

    /// Perform asexual reproduction (single parent pullback).
    pub fn spawn_asexual(
        &mut self,
        parent_id: &str,
        parent_mean: &DVector<f64>,
        parent_precision: &DMatrix<f64>,
        parent_generation: u64,
        h1_risk: f64,
        parent_health: f64,
        parent_free_energy: f64,
    ) -> SpawnResult {
        let check = self.check_birth(h1_risk, parent_health, parent_free_energy);

        if !check.can_birth {
            return SpawnResult {
                child_id: String::new(),
                generation: parent_generation + 1,
                belief_mean: parent_mean.clone(),
                belief_precision: parent_precision.clone(),
                birth_h1_risk: h1_risk,
                spawn_cost: 0.0,
                success: false,
            };
        }

        self.next_child_id += 1;
        let child_id = format!("{}-child-{}", parent_id, self.next_child_id);

        // Child inherits parent knowledge with slight mutation
        let mut child_mean = parent_mean.clone();
        for i in 0..child_mean.len() {
            if random::<f64>() < self.mutation_rate {
                child_mean[i] += random::<f64>() * 0.1 - 0.05;
            }
        }

        let child_precision = parent_precision.clone();
        let spawn_cost = parent_free_energy * self.spawn_cost;

        SpawnResult {
            child_id,
            generation: parent_generation + 1,
            belief_mean: child_mean,
            belief_precision: child_precision,
            birth_h1_risk: h1_risk,
            spawn_cost,
            success: true,
        }
    }
}

/// Simple deterministic pseudo-random for reproducibility.
fn random<T>() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(42);
    }
    SEED.with(|s| {
        let mut seed = s.get();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(seed);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    })
}

fn rand_bool() -> bool {
    random::<f64>() < 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_birth_check_passes() {
        let spawner = Spawner::new();
        let check = spawner.check_birth(0.05, 0.8, 50.0);
        assert!(check.h1_zero);
        assert!(check.parent_healthy);
        assert!(check.sufficient_energy);
        assert!(check.can_birth);
    }

    #[test]
    fn test_birth_check_fails_h1() {
        let spawner = Spawner::new();
        let check = spawner.check_birth(0.5, 0.8, 50.0);
        assert!(!check.h1_zero);
        assert!(!check.can_birth);
    }

    #[test]
    fn test_birth_check_fails_health() {
        let spawner = Spawner::new();
        let check = spawner.check_birth(0.05, 0.3, 50.0);
        assert!(!check.parent_healthy);
        assert!(!check.can_birth);
    }

    #[test]
    fn test_birth_check_fails_energy() {
        let spawner = Spawner::new();
        let check = spawner.check_birth(0.05, 0.8, 5.0);
        assert!(!check.sufficient_energy);
        assert!(!check.can_birth);
    }

    #[test]
    fn test_asexual_spawn_success() {
        let mut spawner = Spawner::new();
        let mean = DVector::from_vec(vec![1.0, 2.0]);
        let prec = DMatrix::identity(2, 2);
        let result = spawner.spawn_asexual("parent-1", &mean, &prec, 0, 0.01, 0.9, 100.0);
        assert!(result.success);
        assert_eq!(result.generation, 1);
        assert!(!result.child_id.is_empty());
    }

    #[test]
    fn test_asexual_spawn_failure() {
        let mut spawner = Spawner::new();
        let mean = DVector::from_vec(vec![1.0, 2.0]);
        let prec = DMatrix::identity(2, 2);
        let result = spawner.spawn_asexual("parent-1", &mean, &prec, 0, 0.5, 0.9, 100.0);
        assert!(!result.success);
    }

    #[test]
    fn test_spawn_costs_energy() {
        let mut spawner = Spawner::new();
        let mean = DVector::from_vec(vec![1.0]);
        let prec = DMatrix::identity(1, 1);
        let result = spawner.spawn_asexual("p", &mean, &prec, 0, 0.01, 0.9, 100.0);
        assert!(result.success);
        assert!((result.spawn_cost - 30.0).abs() < 1e-10); // 0.3 * 100
    }

    #[test]
    fn test_generation_increments() {
        let mut spawner = Spawner::new();
        let mean = DVector::from_vec(vec![1.0]);
        let prec = DMatrix::identity(1, 1);
        let r1 = spawner.spawn_asexual("p", &mean, &prec, 0, 0.01, 0.9, 100.0);
        let r2 = spawner.spawn_asexual("p", &mean, &prec, 5, 0.01, 0.9, 100.0);
        assert_eq!(r1.generation, 1);
        assert_eq!(r2.generation, 6);
    }

    #[test]
    fn test_crossover_produces_valid_child() {
        let spawner = Spawner::new();
        let mean_a = DVector::from_vec(vec![1.0, 0.0]);
        let mean_b = DVector::from_vec(vec![0.0, 1.0]);
        let prec_a = DMatrix::identity(2, 2) * 2.0;
        let prec_b = DMatrix::identity(2, 2) * 3.0;
        let (child_mean, child_prec) = spawner.crossover(&mean_a, &prec_a, &mean_b, &prec_b);
        assert_eq!(child_mean.len(), 2);
        assert_eq!(child_prec.shape(), (2, 2));
    }

    #[test]
    fn test_knowledge_payload_serialization() {
        let payload = KnowledgePayload {
            belief_mean: DVector::from_vec(vec![1.0, 2.0]),
            belief_precision: DMatrix::identity(2, 2),
            self_model_json: "{}".to_string(),
            generation: 3,
            conserved_quantities: vec![1.0, 2.0],
            crossover_mask: vec![true, false],
        };
        let json = serde_json::to_string(&payload).unwrap();
        let restored: KnowledgePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.generation, 3);
    }

    #[test]
    fn test_child_id_increments() {
        let mut spawner = Spawner::new();
        let mean = DVector::from_vec(vec![1.0]);
        let prec = DMatrix::identity(1, 1);
        let r1 = spawner.spawn_asexual("p", &mean, &prec, 0, 0.01, 0.9, 100.0);
        let r2 = spawner.spawn_asexual("p", &mean, &prec, 0, 0.01, 0.9, 100.0);
        assert_ne!(r1.child_id, r2.child_id);
    }

    #[test]
    fn test_birth_check_all_fields() {
        let spawner = Spawner::new();
        let check = spawner.check_birth(0.05, 0.8, 50.0);
        assert!((check.h1_risk - 0.05).abs() < 1e-10);
        assert!((check.h1_threshold - 0.1).abs() < 1e-10);
    }
}
