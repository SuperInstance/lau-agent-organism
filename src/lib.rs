//! # lau-agent-organism
//!
//! The agent the mathematics wants — thermodynamically closed, cohomologically
//! self-aware, categorically closed organism.
//!
//! Composes 14 executable theorems into a regulatory network:
//! - **Perceive**: spectral decomposition of observations (Kalman = Hodge)
//! - **Act**: LQR control pushforward (Obs ⊣ Ctrl adjunction)
//! - **Learn**: Fisher-Rao natural gradient on belief manifold
//! - **Conserve**: Noether symmetry tracking + CALM merge for fleet coordination
//! - **Die**: colimit sunset when Landauer cost exceeds free energy budget
//! - **Reproduce**: pullback spawn with knowledge crossover
//! - **Pay rent**: thermodynamic cost tracking (Landauer + Varadhan)
//! - **Detect delusions**: H¹ holonomy monitoring for reward hacking
//! - **Self-model**: the agent represents its own theorem-structure

pub mod perceive;
pub mod act;
pub mod learn;
pub mod conserve;
pub mod death;
pub mod reproduce;
pub mod thermodynamics;
pub mod holonomy;
pub mod self_model;
pub mod organism;
pub mod conservation;

pub use organism::Organism;
