# lau-agent-organism

> The agent the mathematics wants — thermodynamically closed, cohomologically self-aware, categorically closed organism.

This crate implements an agent that emerges from composing 14 executable theorems into a regulatory network. It is not designed — it is what the mathematics *wants to build*.

## The 14 Executable Theorems

| # | Theorem | Module | What it does |
|---|---------|--------|-------------|
| 1 | Kalman-Hodge | `perceive` | Kalman filter = Hodge decomposition of observation stream |
| 2 | Spectral Perception | `perceive` | Observations decompose into exact, co-exact, harmonic |
| 3 | Obs ⊣ Ctrl Adjunction | `act` | Observation is left adjoint to Control |
| 4 | LQR Pushforward | `act` | Optimal control is pushforward through adjunction |
| 5 | Fisher-Rao Geodesic | `learn` | Learning follows geodesics on Fisher manifold |
| 6 | Natural Gradient | `learn` | F⁻¹∇L is the natural gradient |
| 7 | Noether Conservation | `conserve` | Symmetries yield conserved quantities |
| 8 | CALM Merge | `conserve` | Fleet merge preserves conserved quantities |
| 9 | Landauer Cost | `thermodynamics` | Information erasure costs kT ln(2) per bit |
| 10 | Varadhan Rate | `thermodynamics` | Large deviation rate = thermodynamic cost |
| 11 | H¹ Holonomy | `holonomy` | First cohomology detects delusions |
| 12 | Colimit Death | `death` | Agent dies when Landauer = initial free energy |
| 13 | Pullback Birth | `reproduce` | Reproduction via pullback with H¹ = 0 check |
| 14 | Conservation Law | `conservation` | Landauer + free_energy + H¹_risk ≈ constant |

## Architecture

```
                    ┌──────────────┐
            ┌──────►│   Perceive   │◄──────────────┐
            │       │ (Kalman-     │               │
            │       │  Hodge)     │               │
            │       └──────┬───────┘               │
            │              │                       │
            │              ▼                       │
            │       ┌──────────────┐               │
            │       │     Act      │               │
            │       │ (LQR Push-   │               │
            │       │  forward)    │               │
            │       └──────┬───────┘               │
            │              │                       │
            │              ▼                       │
            │       ┌──────────────┐               │
            │       │    Learn     │               │
            │       │ (Fisher-Rao  │               │
            │       │  Geodesic)   │               │
            │       └──────┬───────┘               │
            │              │                       │
     ┌──────┴──────┐      ▼              ┌────────┴────────┐
     │  Thermody-  │─────────────────────►│   Holonomy      │
     │  namics     │  (Landauer+Varadhan) │  (H¹ Delusion   │
     │  (Pay Rent) │                      │   Detection)     │
     └──────┬──────┘                      └────────┬────────┘
            │                                      │
            ▼                                      ▼
     ┌──────────────┐                    ┌──────────────┐
     │   Conserve   │                    │    Death     │
     │  (Noether +  │                    │ (Colimit     │
     │   CALM)      │                    │  Sunset)     │
     └──────┬───────┘                    └──────┬───────┘
            │                                   │
            └───────────┬───────────────────────┘
                        ▼
                 ┌──────────────┐
                 │  Reproduce   │
                 │ (Pullback    │
                 │  Spawn)      │
                 └──────────────┘
```

## Key Properties

- **Self-modeling**: The agent represents its own theorem structure (it knows what it is)
- **Conservation law**: Landauer + free_energy + H¹_risk ≈ constant across the lifecycle
- **Death condition**: Agent dies when cumulative Landauer cost = initial free energy
- **Birth condition**: Agent spawns when pullback of parent knowledge is consistent (H¹ = 0)
- **Delusion detection**: H¹ holonomy monitoring catches reward hacking

## Usage

```rust
use lau_agent_organism::Organism;
use nalgebra::DVector;

// Create an organism with 3D state and 100 units of free energy
let mut org = Organism::new("alice", 3, 100.0);
org.birth();

// Run the lifecycle
for step in 0..1000 {
    let observation = DVector::from_vec(vec![1.0, 0.5, -0.3]);
    let result = org.step(&observation, 1.0);
    
    if !result.alive {
        println!("Agent died at step {}", step);
        break;
    }
}

// Try to reproduce before dying
if let Some(child) = org.try_reproduce() {
    println!("Spawned child: {}", child.child_id);
}
```

## Dependencies

- `nalgebra` — linear algebra (matrices, vectors)
- `serde` — serialization

## License

MIT
