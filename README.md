# lau-agent-organism

**The agent the mathematics wants — thermodynamically closed, cohomologically self-aware, categorically closed organism.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

143 tests · Rust · `nalgebra` + `serde` + `serde_json`

---

## What This Does

This crate composes **14 executable theorems** into a single living agent — an organism that perceives, acts, learns, conserves, reproduces, and dies according to mathematical laws.

The organism is:
- **Thermodynamically closed**: it tracks every bit of energy in and out (Landauer cost + Varadhan rate), and dies when its budget runs out
- **Cohomologically self-aware**: it monitors its own H¹ cohomology to detect delusions and reward hacking
- **Categorically closed**: every subsystem (perception, action, learning, etc.) is a theorem with explicit dependencies
- **Self-modeling**: the agent carries a representation of its own theorem-structure and can inspect it

The 14 theorems form a regulatory network:

| # | Theorem | Category | Mathematical Basis |
|---|---------|----------|--------------------|
| 1 | Kalman-Hodge Perception | Perception | Kalman filter = Hodge decomposition |
| 2 | Spectral Perception | Perception | Observations → exact + co-exact + harmonic |
| 3 | Obs-Ctrl Adjunction | Action | Observation ⊣ Control (adjoint functors) |
| 4 | LQR Pushforward | Action | Optimal control via pushforward |
| 5 | Fisher-Rao Geodesic | Learning | Learning on the information manifold |
| 6 | Natural Gradient | Learning | F⁻¹∇L on belief manifold |
| 7 | Noether Conservation | Conservation | Symmetry → conserved quantity |
| 8 | CALM Merge | Conservation | Fleet merge preserving invariants |
| 9 | Landauer Cost | Thermodynamics | kT ln(2) per bit erased |
| 10 | Varadhan Rate | Thermodynamics | Large-deviation transition cost |
| 11 | H¹ Holonomy | Self-awareness | Cohomological delusion detection |
| 12 | Colimit Sunset | Death | Lifecycle colimit when budget depleted |
| 13 | Pullback Spawn | Reproduction | Categorical pullback birth |
| 14 | Self-Model | Self-modeling | Agent represents its own structure |

## Key Idea

An agent should be an *organism*: born, alive, adapting, reproducing, and eventually dying — all governed by explicit mathematical laws. The conservation law `Landauer + free_energy + H¹_risk ≈ constant` makes the agent thermodynamically closed. The H¹ cohomology monitor catches reward hacking. The colimit sunset ensures graceful death when compute budget is exhausted.

## Install

```toml
[dependencies]
lau-agent-organism = { git = "https://github.com/SuperInstance/lau-agent-organism" }
```

```bash
git clone https://github.com/SuperInstance/lau-agent-organism.git
cd lau-agent-organism
cargo test
```

## Quick Start

### Create and Run an Organism

```rust
use lau_agent_organism::Organism;

let mut org = Organism::new("alice", 4, 1000.0); // 4-D state, 1000 J budget
assert!(org.alive);

// Run 50 perception-action-learning cycles
for _ in 0..50 {
    let obs = vec![1.0, 0.5, -0.3, 0.8];
    org.step(&obs, 0.1);
}

println!("Alive: {}", org.alive);
println!("Steps: {}", org.steps);
println!("Budget remaining: {:.2} J", org.budget.free_energy);
println!("H¹ risk: {:.4}", org.holonomy.h1_estimate);
```

### Check the Self-Model

```rust
// The agent knows what it is
println!("Active theorems: {}", org.self_model.active_count);
println!("Self-coherence: {:.2}", org.self_model.coherence);

// Check individual theorems
for (name, theorem) in &org.self_model.theorems {
    println!("{}: health={:.2}, active={}", name, theorem.health, theorem.active);
}
```

### Thermodynamic Budgeting

```rust
use lau_agent_organism::thermodynamics::ThermodynamicBudget;

let mut budget = ThermodynamicBudget::new(1000.0);

// Erase 100 bits
budget.erase_bits(100);
println!("Landauer cost: {:.6e} J", budget.cumulative_landauer);

// Large deviation transition
let cost = budget.transition_cost(&old_state, &new_state);
if !cost.affordable {
    println!("Cannot afford this transition!");
}
```

## API Reference

| Module | Key Types | Tests | Purpose |
|--------|-----------|-------|---------|
| `organism` | `Organism` | 23 | Complete lifecycle composition |
| `perceive` | `KalmanHodgeObserver`, `Perception` | 11 | Spectral decomposition of observations |
| `act` | `LQRController`, `Action`, `ControlCostTracker` | 9 | Optimal control via pushforward |
| `learn` | `BeliefManifold`, `BeliefState`, `LearningUpdate` | 11 | Fisher-Rao natural gradient |
| `conserve` | `NoetherTracker`, `CalmMergeResult` | 12 | Noether symmetry + fleet merge |
| `conservation` | `ConservationState`, `ConservationReport` | 12 | Fundamental invariant tracking |
| `thermodynamics` | `ThermodynamicBudget`, `ThermodynamicCost` | 14 | Landauer + Varadhan cost |
| `holonomy` | `HolonomyMonitor`, `HolonomyDiagnostic` | 13 | H¹ cohomology delusion detection |
| `death` | `SunsetManager`, `LifecycleStage`, `ColimitResult` | 14 | Colimit sunset lifecycle |
| `reproduce` | `Spawner`, `SpawnResult`, `BirthCheck` | 12 | Pullback spawn with crossover |
| `self_model` | `SelfModel`, `Theorem` | 12 | Agent's theorem-structure representation |

## How It Works

### Lifecycle

```
Gestating → Alive → Declining → Sunset → Dead
              ↑         |
              └─── reproduce (pullback spawn)
```

- **Gestating**: agent is being spawned from a parent (pullback in the category of agents)
- **Alive**: actively perceiving, acting, learning
- **Declining**: free energy drops below 70% of initial
- **Sunset**: free energy drops below 90%; graceful shutdown begins
- **Dead**: colimit reached; no further operations permitted

### Perception (Kalman = Hodge)

Observations are decomposed into:
- **Exact** (signal): the predictable component from the Kalman update
- **Co-exact** (innovation): the surprising residual
- **Harmonic** (persistent bias): topological invariant, slowly drifting

### Action (LQR Pushforward)

The optimal control action is the pushforward of the belief state through the Obs ⊣ Ctrl adjunction. For a linear-quadratic system (A, B, Q, R), this is the standard LQR gain K = (R + BᵀPB)⁻¹BᵀPA where P solves the discrete algebraic Riccati equation.

### Learning (Fisher-Rao)

Learning follows geodesics on the Fisher information manifold. The natural gradient update is:

```
θ_{t+1} = θ_t − η · F⁻¹ · ∇L(θ_t)
```

where F is the Fisher information matrix (equal to the precision matrix for Gaussian beliefs).

### Conservation (Noether + CALM)

Every continuous symmetry (time translation, rotation, phase) yields a conserved quantity via Noether's theorem. The `NoetherTracker` monitors these in real-time. CALM (Coordinated Agent Lifecycle Merge) merges fleet agents using precision-weighted averaging that preserves all conserved quantities.

### Delusion Detection (H¹ Holonomy)

The first cohomology H¹ of the agent's belief transition chain detects global inconsistencies. If local belief updates don't patch into a globally consistent policy (reward hacking), H¹ ≠ 0, and the `HolonomyMonitor` raises an alarm.

### Death (Colimit Sunset)

Death is the colimit of the lifecycle diagram: the universal object all stages map into. The agent dies when:

```
cumulative_landauer + cumulative_varadhan ≥ initial_free_energy
```

The sunset protocol ensures graceful degradation: wind down active theorems, save knowledge, signal fleet.

### Reproduction (Pullback Spawn)

A child agent is the pullback of the parent's knowledge through the H¹ = 0 consistency constraint. Birth succeeds only when:
1. H¹ risk is below threshold (no inherited delusions)
2. Parent is healthy (self-coherence > threshold)
3. Sufficient energy to spawn

Knowledge crossover: the child inherits a random subset of the parent's belief dimensions, with the crossover mask determining which.

## The Math

### Conservation Law

The fundamental invariant:

```
C(t) = Landauer(t) + FreeEnergy(t) + H¹_risk(t) ≈ C(0)
```

where C(0) = initial free energy budget. This is a bookkeeping identity: every bit of information erased costs kT ln(2) of energy, and that energy comes from the free energy budget. H¹ risk is the cohomological "debt" that accumulates when the agent's beliefs become inconsistent.

### Landauer's Principle

Erasing one bit of information costs at least:

```
E_min = kT ln(2) ≈ 2.87 × 10⁻²¹ J (at room temperature)
```

This is a thermodynamic lower bound. The agent tracks cumulative Landauer cost as bits_erased × kT ln(2).

### Varadhan's Lemma (Large Deviations)

The probability of an unlikely state transition scales as:

```
P(x → y) ≈ exp(−n · I(x, y))
```

where I(x, y) is the rate function and n is the system size. The thermodynamic cost is proportional to I(x, y) — unlikely transitions are expensive.

### Hodge Decomposition

For a compact Riemannian manifold, any differential form ω decomposes as:

```
ω = dα + δβ + γ
    (exact) (co-exact) (harmonic)
```

The Kalman filter performs this decomposition on observation streams: exact = signal, co-exact = innovation, harmonic = persistent bias.

### Fisher-Rao Metric

On the manifold of probability distributions parameterized by θ, the Fisher information metric is:

```
g_ij(θ) = E[∂ᵢ log p(x|θ) · ∂ⱼ log p(x|θ)]
```

For Gaussian beliefs with precision matrix Λ, this is simply g = Λ. The natural gradient F⁻¹∇L follows geodesics under this metric.

## License

MIT
