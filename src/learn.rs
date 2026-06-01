//! Learning module — Fisher-Rao natural gradient on the belief manifold.
//!
//! The Fisher information metric defines a Riemannian structure on the space
//! of belief distributions. Learning follows geodesics (natural gradient descent)
//! on this manifold, ensuring information-theoretic optimal updates.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A point on the belief manifold (parameterized as multivariate Gaussian).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// Mean of the belief distribution.
    pub mean: DVector<f64>,
    /// Precision matrix (inverse covariance) — the Fisher metric.
    pub precision: DMatrix<f64>,
    /// Learning rate for natural gradient steps.
    pub learning_rate: f64,
}

/// Result of a learning step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningUpdate {
    /// New belief mean.
    pub new_mean: DVector<f64>,
    /// Change in mean (natural gradient step).
    pub delta_mean: DVector<f64>,
    /// Fisher information distance traveled.
    pub fisher_distance: f64,
    /// KL divergence between old and new beliefs.
    pub kl_divergence: f64,
}

impl BeliefState {
    /// Create a new belief state with identity precision.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: DVector::zeros(dim),
            precision: DMatrix::identity(dim, dim),
            learning_rate: 0.1,
        }
    }

    /// Create with specific mean and precision.
    pub fn with_params(mean: DVector<f64>, precision: DMatrix<f64>) -> Self {
        Self {
            mean,
            precision,
            learning_rate: 0.1,
        }
    }

    /// Get covariance (inverse of precision).
    pub fn covariance(&self) -> DMatrix<f64> {
        self.precision
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(self.dim(), self.dim()))
    }

    /// Dimension of the belief state.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Compute the Fisher information matrix (which is the precision for Gaussians).
    pub fn fisher_information(&self) -> &DMatrix<f64> {
        &self.precision
    }

    /// Natural gradient step: follows the geodesic on the belief manifold.
    ///
    /// The natural gradient is: F^{-1} ∇L, where F is the Fisher information.
    /// For Gaussians, this simplifies to: Σ ∇L = P^{-1} ∇L.
    pub fn natural_gradient_step(&mut self, gradient: &DVector<f64>) -> LearningUpdate {
        let fisher = self.fisher_information().clone();
        let fisher_inv = fisher
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(self.dim(), self.dim()));

        // Natural gradient: F^{-1} ∇L
        let natural_grad = &fisher_inv * gradient;

        // Step along the geodesic
        let delta = &natural_grad * self.learning_rate;
        let _old_mean = self.mean.clone();
        self.mean += &delta;

        // Fisher distance: δ^T F δ
        let fisher_distance = delta.transpose() * &fisher * &delta;
        let fisher_distance = fisher_distance[(0, 0)];

        // KL divergence approximation: 0.5 * δ^T F δ
        let kl = 0.5 * fisher_distance;

        LearningUpdate {
            new_mean: self.mean.clone(),
            delta_mean: delta,
            fisher_distance,
            kl_divergence: kl,
        }
    }

    /// Update precision based on new information (Bayesian update).
    pub fn update_precision(&mut self, information: &DMatrix<f64>) {
        self.precision += information;
    }

    /// Compute Fisher-Rao distance to another belief state.
    pub fn fisher_rao_distance(&self, other: &BeliefState) -> f64 {
        // For two Gaussians with same precision, FR distance ≈ Mahalanobis distance
        let diff = &self.mean - &other.mean;
        let mahal = diff.transpose() * &self.precision * &diff;
        mahal[(0, 0)].sqrt()
    }

    /// Geodesic interpolation between two belief states.
    pub fn geodesic(&self, other: &BeliefState, t: f64) -> BeliefState {
        let mean = (1.0 - t) * &self.mean + t * &other.mean;
        // Precision interpolation (simplified: log-Euclidean)
        let p_self = self.precision.map(|x| x.ln().max(-50.0));
        let p_other = other.precision.map(|x| x.ln().max(-50.0));
        let log_interp = (1.0 - t) * &p_self + t * &p_other;
        let precision = log_interp.map(|x| x.exp());

        BeliefState {
            mean,
            precision,
            learning_rate: (1.0 - t) * self.learning_rate + t * other.learning_rate,
        }
    }
}

/// Belief manifold tracker — manages the history of learning updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefManifold {
    /// Current belief.
    pub belief: BeliefState,
    /// Total Fisher distance traveled.
    pub total_fisher_distance: f64,
    /// Total KL divergence accumulated.
    pub total_kl: f64,
    /// Number of updates.
    pub update_count: u64,
}

impl BeliefManifold {
    pub fn new(dim: usize) -> Self {
        Self {
            belief: BeliefState::new(dim),
            total_fisher_distance: 0.0,
            total_kl: 0.0,
            update_count: 0,
        }
    }

    /// Perform a learning update on the manifold.
    pub fn learn(&mut self, gradient: &DVector<f64>) -> LearningUpdate {
        let update = self.belief.natural_gradient_step(gradient);
        self.total_fisher_distance += update.fisher_distance;
        self.total_kl += update.kl_divergence;
        self.update_count += 1;
        update
    }

    /// Average Fisher distance per update.
    pub fn average_fisher_distance(&self) -> f64 {
        if self.update_count == 0 {
            0.0
        } else {
            self.total_fisher_distance / self.update_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_creation() {
        let b = BeliefState::new(3);
        assert_eq!(b.dim(), 3);
        assert_eq!(b.mean.len(), 3);
    }

    #[test]
    fn test_belief_covariance() {
        let b = BeliefState::new(2);
        let cov = b.covariance();
        // Identity precision → identity covariance
        assert!((cov[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_information_is_precision() {
        let b = BeliefState::new(3);
        let fi = b.fisher_information();
        assert!((fi[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_gradient_step() {
        let mut b = BeliefState::new(2);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        let update = b.natural_gradient_step(&grad);
        assert!((update.delta_mean[0] - 0.1).abs() < 1e-10);
        assert!(update.fisher_distance > 0.0);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let mut b = BeliefState::new(2);
        let grad = DVector::from_vec(vec![1.0, 1.0]);
        let update = b.natural_gradient_step(&grad);
        assert!(update.kl_divergence > 0.0);
    }

    #[test]
    fn test_fisher_distance_proportional_to_step() {
        let mut b = BeliefState::new(2);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        let update1 = b.natural_gradient_step(&grad);
        b.learning_rate = 0.2;
        let update2 = b.natural_gradient_step(&grad);
        assert!(update2.fisher_distance > update1.fisher_distance);
    }

    #[test]
    fn test_fisher_rao_distance() {
        let b1 = BeliefState::with_params(
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::identity(2, 2),
        );
        let b2 = BeliefState::with_params(
            DVector::from_vec(vec![3.0, 4.0]),
            DMatrix::identity(2, 2),
        );
        let d = b1.fisher_rao_distance(&b2);
        assert!((d - 5.0).abs() < 1e-10); // sqrt(9+16) = 5
    }

    #[test]
    fn test_geodesic_interpolation() {
        let b1 = BeliefState::with_params(
            DVector::from_vec(vec![0.0]),
            DMatrix::identity(1, 1),
        );
        let b2 = BeliefState::with_params(
            DVector::from_vec(vec![10.0]),
            DMatrix::identity(1, 1),
        );
        let mid = b1.geodesic(&b2, 0.5);
        assert!((mid.mean[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_manifold_tracking() {
        let mut manifold = BeliefManifold::new(2);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        manifold.learn(&grad);
        manifold.learn(&grad);
        assert_eq!(manifold.update_count, 2);
        assert!(manifold.total_fisher_distance > 0.0);
        assert!(manifold.total_kl > 0.0);
    }

    #[test]
    fn test_precision_update() {
        let mut b = BeliefState::new(2);
        let info = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]);
        b.update_precision(&info);
        assert!((b.precision[(0, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_fisher_distance() {
        let mut manifold = BeliefManifold::new(2);
        assert!((manifold.average_fisher_distance()).abs() < 1e-10);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        manifold.learn(&grad);
        assert!(manifold.average_fisher_distance() > 0.0);
    }
}
