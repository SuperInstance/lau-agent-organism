# lau-agent-organism

> The agent the mathematics wants — thermodynamically closed, cohomologically self-aware, categorically closed organism

## What This Does

The agent the mathematics wants — thermodynamically closed, cohomologically self-aware, categorically closed organism. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-agent-organism
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_agent_organism::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct Theorem 
pub struct SelfModel 
    pub fn new(agent_id: &str) -> Self 
    pub fn offspring(agent_id: &str, parent_id: &str, generation: u64) -> Self 
    pub fn theorems_by_category(&self, category: &str) -> Vec<&Theorem> 
    pub fn check_dependencies(&self) -> Vec<String> 
    pub fn compute_coherence(&self) -> f64 
    pub fn deactivate_theorem(&mut self, name: &str) 
    pub fn update_health(&mut self, name: &str, health: f64) 
    pub fn total_theorems(&self) -> usize 
    pub fn to_json(&self) -> Result<String, serde_json::Error> 
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> 
    pub fn dependency_graph(&self) -> HashMap<String, Vec<String>> 
pub struct Symmetry 
pub struct ConservationReport 
pub struct NoetherTracker 
    pub fn new() -> Self 
    pub fn register_symmetry(
    pub fn register_energy_conservation(&mut self, dim: usize, initial_energy: f64) 
    pub fn register_rotation_conservation(&mut self, dim: usize, initial_angular_momentum: f64) 
    pub fn update_value(&mut self, name: &str, new_value: f64) 
    pub fn check_conservation(&self, current_values: &HashMap<String, f64>) -> ConservationReport 
    pub fn snapshot(&mut self, values: HashMap<String, f64>) 
    pub fn conservation_drift(&self) -> HashMap<String, f64> 
    pub fn num_symmetries(&self) -> usize 
pub struct CalmMergeResult 
pub fn calm_merge(
pub struct Perception 
pub struct KalmanHodgeObserver 
    pub fn new(dim: usize) -> Self 
    pub fn with_matrices(
    pub fn predict(&mut self, transition: &DMatrix<f64>) 
    pub fn update(&mut self, observation: &DVector<f64>) -> Perception 
    pub fn state(&self) -> &DVector<f64> 
    pub fn uncertainty(&self) -> f64 
    pub fn snr(&self) -> f64 
pub struct HodgeDecomposition 
pub fn hodge_decompose(laplacian: &DMatrix<f64>, vector: &DVector<f64>) -> HodgeDecomposition 
pub enum LifecycleStage 
pub struct DeathCondition 
pub struct SunsetManager 
    pub fn new() -> Self 
    pub fn birth(&mut self, time: f64) 
    pub fn update(&mut self, landauer_fraction: f64, time: f64) -> LifecycleStage 
    pub fn should_die(&self) -> bool 
    pub fn in_sunset(&self) -> bool 
    pub fn set_final_state(&mut self, state: &str) 
    pub fn transfer_knowledge(&mut self) 
    pub fn compute_colimit(&mut self, landauer_cost: f64, free_energy: f64, h1_risk: f64) -> ColimitResult 
    pub fn num_transitions(&self) -> usize 
    pub fn lifespan(&self) -> f64 
pub struct ColimitResult 
pub struct LQRController 
pub struct Action 
    pub fn new(state_dim: usize, control_dim: usize) -> Self 
    pub fn with_matrices(
    pub fn solve(&mut self) -> Result<(), String> 
    pub fn act(&self, state: &DVector<f64>) -> Result<Action, String> 
    pub fn adjunction_unit(&self, observation: &DVector<f64>) -> DVector<f64> 
    pub fn adjunction_counit(&self, control: &DVector<f64>) -> DVector<f64> 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**143 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
