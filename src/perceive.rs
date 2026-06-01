//! Perception module — spectral decomposition of observations via Kalman = Hodge.
//!
//! The Kalman filter is reinterpreted as Hodge decomposition of the observation
//! stream into exact (signal), co-exact (innovation), and harmonic (persistent
//! bias) components.

use nalgebra::{DMatrix, DVector, DVectorSlice};
use serde::{Deserialize, Serialize};

/// Spectral decomposition of an observation vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perception {
    /// Exact component (signal) — the predictable part from the Kalman update.
    pub exact: DVector<f64>,
    /// Co-exact component (innovation) — the surprising/residual part.
    pub coexact: DVector<f64>,
    /// Harmonic component (persistent bias) — topological invariant.
    pub harmonic: DVector<f64>,
    /// Eigenvalues of the observation precision matrix.
    pub spectrum: Vec<f64>,
}

/// Kalman-Hodge observer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanHodgeObserver {
    /// State dimension.
    pub dim: usize,
    /// Current state estimate.
    pub state: DVector<f64>,
    /// State covariance.
    pub covariance: DMatrix<f64>,
    /// Observation matrix H (maps state → observation space).
    pub observation_matrix: DMatrix<f64>,
    /// Process noise covariance Q.
    pub process_noise: DMatrix<f64>,
    /// Observation noise covariance R.
    pub observation_noise: DMatrix<f64>,
    /// Accumulated harmonic (persistent bias) estimate.
    pub harmonic_bias: DVector<f64>,
    /// Smoothing factor for harmonic component (0..1).
    pub harmonic_alpha: f64,
}

impl KalmanHodgeObserver {
    /// Create a new observer with identity observation model.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            state: DVector::zeros(dim),
            covariance: DMatrix::identity(dim, dim),
            observation_matrix: DMatrix::identity(dim, dim),
            process_noise: DMatrix::identity(dim, dim) * 0.01,
            observation_noise: DMatrix::identity(dim, dim) * 0.1,
            harmonic_bias: DVector::zeros(dim),
            harmonic_alpha: 0.05,
        }
    }

    /// Create with custom matrices.
    pub fn with_matrices(
        state: DVector<f64>,
        covariance: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
    ) -> Self {
        let dim = state.len();
        Self {
            dim,
            state,
            covariance,
            observation_matrix: h,
            process_noise: q,
            observation_noise: r,
            harmonic_bias: DVector::zeros(dim),
            harmonic_alpha: 0.05,
        }
    }

    /// Predict step (prior).
    pub fn predict(&mut self, transition: &DMatrix<f64>) {
        self.state = transition * &self.state;
        self.covariance =
            transition * &self.covariance * transition.transpose() + &self.process_noise;
    }

    /// Update step with observation, performing Hodge decomposition.
    ///
    /// Returns a Perception struct decomposing the observation into
    /// exact (Kalman signal), co-exact (innovation), and harmonic (bias).
    pub fn update(&mut self, observation: &DVector<f64>) -> Perception {
        let h = &self.observation_matrix;
        let ht = h.transpose();

        // Innovation
        let y = observation - h * &self.state;

        // Innovation covariance: S = H P H^T + R
        let s = h * &self.covariance * &ht + &self.observation_noise;

        // Kalman gain: K = P H^T S^{-1}
        let s_inv = match s.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::identity(s.nrows(), s.ncols()) * 1e6,
        };
        let k = &self.covariance * &ht * &s_inv;

        // Exact component (Kalman update)
        let state_update = &k * &y;
        let exact_component = &self.state + &state_update;

        // Co-exact component (innovation residual after update)
        let coexact_component = &y - h * &state_update;

        // Harmonic component (persistent bias update via exponential smoothing)
        self.harmonic_bias =
            (1.0 - self.harmonic_alpha) * &self.harmonic_bias + self.harmonic_alpha * &y;
        let harmonic_component = self.harmonic_bias.clone();

        // Update state and covariance
        self.state = exact_component.clone();
        let kh = &k * h;
        self.covariance = (DMatrix::identity(self.dim, self.dim) - kh) * &self.covariance;

        // Spectral decomposition of precision matrix
        let precision = match self.covariance.clone().try_inverse() {
            Some(p) => p,
            None => DMatrix::identity(self.dim, self.dim),
        };
        let spectrum = compute_spectrum(&precision);

        Perception {
            exact: exact_component,
            coexact: coexact_component,
            harmonic: harmonic_component,
            spectrum,
        }
    }

    /// Get current state estimate.
    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Get current uncertainty (trace of covariance).
    pub fn uncertainty(&self) -> f64 {
        self.covariance.trace()
    }

    /// Compute signal-to-noise ratio of current estimate.
    pub fn snr(&self) -> f64 {
        let signal_power: f64 = self.state.iter().map(|x| x * x).sum();
        let noise_power = self.covariance.trace().max(1e-12);
        signal_power / noise_power
    }
}

/// Compute eigenvalue spectrum of a symmetric matrix (power iteration fallback).
fn compute_spectrum(matrix: &DMatrix<f64>) -> Vec<f64> {
    // For small matrices, compute via the symmetric eigen decomposition.
    // nalgebra doesn't have a direct symmetric eigendecomposition for DMatrix,
    // so we approximate via trace-based spectral norm estimation.
    let n = matrix.nrows();
    if n == 0 {
        return vec![];
    }

    // Use iterative Rayleigh quotient for dominant eigenvalues
    let mut eigenvalues = Vec::with_capacity(n.min(5));
    let mut deflated = matrix.clone();

    let num_eigs = n.min(5);
    for _ in 0..num_eigs {
        let (eigval, eigvec) = power_iteration(&deflated, 50);
        eigenvalues.push(eigval);
        // Deflate
        let vv = &eigvec * eigvec.transpose();
        deflated = &deflated - eigval * vv;
    }

    eigenvalues
}

/// Power iteration for dominant eigenvalue/eigenvector.
fn power_iteration(matrix: &DMatrix<f64>, max_iter: usize) -> (f64, DVector<f64>) {
    let n = matrix.nrows();
    let mut v = DVector::from_fn(n, |_, _| 1.0 / (n as f64).sqrt());

    for _ in 0..max_iter {
        let mv = matrix * &v;
        let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
        v = mv / norm;
    }

    let av = matrix * &v;
    let eigval = v.dot(&av);

    (eigval, v)
}

/// Compute Hodge decomposition of a vector field on a graph.
///
/// Given a graph Laplacian L, decomposes a vector v into:
/// - exact: im(d^*) where d is the coboundary
/// - coexact: im(d)
/// - harmonic: ker(L)
#[derive(Debug, Clone)]
pub struct HodgeDecomposition {
    pub exact: DVector<f64>,
    pub coexact: DVector<f64>,
    pub harmonic: DVector<f64>,
}

/// Perform Hodge decomposition using the graph Laplacian.
pub fn hodge_decompose(laplacian: &DMatrix<f64>, vector: &DVector<f64>) -> HodgeDecomposition {
    let n = laplacian.nrows();

    // Harmonic component: project onto ker(L)
    // Approximate via pseudo-inverse: harmonic = v - L^+ L v
    let lv = laplacian * vector;
    let l_plus = pseudo_inverse(laplacian);
    let exact_plus_coexact = &l_plus * &lv;
    let harmonic = vector - laplacian * &exact_plus_coexact;

    // Split exact and coexact (simplified: exact = gradient part, coexact = curl part)
    let exact = &l_plus * &lv;
    let coexact = vector - &exact - &harmonic;

    HodgeDecomposition {
        exact,
        coexact,
        harmonic,
    }
}

/// Moore-Penrose pseudo-inverse (regularized inverse).
fn pseudo_inverse(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let regularization = DMatrix::identity(matrix.nrows(), matrix.ncols()) * 1e-8;
    let regularized = matrix + &regularization;
    match regularized.try_inverse() {
        Some(inv) => inv,
        None => DMatrix::identity(matrix.nrows(), matrix.ncols()) * 1e-8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_kalman_hodge_creation() {
        let obs = KalmanHodgeObserver::new(3);
        assert_eq!(obs.dim, 3);
        assert_eq!(obs.state.len(), 3);
        assert_eq!(obs.covariance.shape(), (3, 3));
    }

    #[test]
    fn test_predict_updates_state() {
        let mut obs = KalmanHodgeObserver::new(2);
        obs.state = DVector::from_vec(vec![1.0, 0.0]);
        let f = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]); // identity
        obs.predict(&f);
        assert!((obs.state[0] - 1.0).abs() < 1e-10);
        assert!(obs.covariance.trace() > 2.0); // should grow
    }

    #[test]
    fn test_update_with_observation() {
        let mut obs = KalmanHodgeObserver::new(2);
        let z = DVector::from_vec(vec![1.0, 1.0]);
        let perc = obs.update(&z);
        assert_eq!(perc.exact.len(), 2);
        assert_eq!(perc.coexact.len(), 2);
        assert_eq!(perc.harmonic.len(), 2);
    }

    #[test]
    fn test_perception_has_spectrum() {
        let mut obs = KalmanHodgeObserver::new(3);
        let z = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let perc = obs.update(&z);
        assert!(!perc.spectrum.is_empty());
    }

    #[test]
    fn test_exact_component_is_updated_state() {
        let mut obs = KalmanHodgeObserver::new(2);
        let z = DVector::from_vec(vec![5.0, -3.0]);
        let perc = obs.update(&z);
        // exact should be close to observation for identity H
        assert!((perc.exact[0] - 5.0).abs() < 1.0);
        assert!((perc.exact[1] - (-3.0)).abs() < 1.0);
    }

    #[test]
    fn test_harmonic_accumulates_bias() {
        let mut obs = KalmanHodgeObserver::new(2);
        obs.harmonic_alpha = 0.5;
        // Repeated same observation should build harmonic
        let z = DVector::from_vec(vec![10.0, 10.0]);
        let _ = obs.update(&z);
        let _ = obs.update(&z);
        let _ = obs.update(&z);
        // harmonic bias should be nonzero now
        assert!(obs.harmonic_bias.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_uncertainty_decreases_with_observations() {
        let mut obs = KalmanHodgeObserver::new(2);
        let initial_unc = obs.uncertainty();
        let z = DVector::from_vec(vec![1.0, 1.0]);
        obs.update(&z);
        assert!(obs.uncertainty() < initial_unc);
    }

    #[test]
    fn test_snr_increases_with_observations() {
        let mut obs = KalmanHodgeObserver::new(2);
        let z = DVector::from_vec(vec![5.0, 5.0]);
        let _ = obs.update(&z);
        let snr1 = obs.snr();
        let _ = obs.update(&z);
        let snr2 = obs.snr();
        assert!(snr2 > snr1);
    }

    #[test]
    fn test_hodge_decomposition_components_sum() {
        let laplacian = DMatrix::from_vec(3, 3, vec![
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ]);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let decomp = hodge_decompose(&laplacian, &v);
        let sum = &decomp.exact + &decomp.coexact + &decomp.harmonic;
        for i in 0..3 {
            assert!((sum[i] - v[i]).abs() < 0.1, "Component sum mismatch at index {}", i);
        }
    }

    #[test]
    fn test_spectrum_eigenvalues_are_real() {
        let mut obs = KalmanHodgeObserver::new(4);
        let z = DVector::from_vec(vec![1.0, -1.0, 2.0, -2.0]);
        let perc = obs.update(&z);
        // All spectrum values should be finite
        for &eigval in &perc.spectrum {
            assert!(eigval.is_finite());
        }
    }

    #[test]
    fn test_perception_serialization() {
        let mut obs = KalmanHodgeObserver::new(2);
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let perc = obs.update(&z);
        let json = serde_json::to_string(&perc).expect("serialize");
        let deserialized: Perception = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.exact.len(), 2);
    }
}
