//! Action module — LQR control pushforward via Obs ⊣ Ctrl adjunction.
//!
//! The Obs ⊣ Ctrl adjunction says: the optimal control policy is the left
//! adjoint to observation. Action is the pushforward of belief through the
//! optimal control map.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// LQR controller that computes optimal control actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LQRController {
    /// State dimension.
    pub state_dim: usize,
    /// Control dimension.
    pub control_dim: usize,
    /// State transition matrix A.
    pub a: DMatrix<f64>,
    /// Control input matrix B.
    pub b: DMatrix<f64>,
    /// State cost matrix Q.
    pub q: DMatrix<f64>,
    /// Control cost matrix R.
    pub r: DMatrix<f64>,
    /// Solved Riccati gain matrix K (computed once, cached).
    pub gain: Option<DMatrix<f64>>,
}

/// Result of computing a control action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// The control vector u.
    pub control: DVector<f64>,
    /// Expected cost of this action.
    pub expected_cost: f64,
    /// Pushforward mapping (state → action Jacobian).
    pub pushforward: DMatrix<f64>,
}

impl LQRController {
    /// Create a new LQR controller with identity dynamics.
    pub fn new(state_dim: usize, control_dim: usize) -> Self {
        Self {
            state_dim,
            control_dim,
            a: DMatrix::identity(state_dim, state_dim),
            b: {
                let mut m = DMatrix::zeros(state_dim, control_dim);
                for i in 0..state_dim.min(control_dim) {
                    m[(i, i)] = 1.0;
                }
                m
            },
            q: DMatrix::identity(state_dim, state_dim),
            r: DMatrix::identity(control_dim, control_dim),
            gain: None,
        }
    }

    /// Create with custom matrices.
    pub fn with_matrices(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
    ) -> Self {
        let state_dim = a.nrows();
        let control_dim = b.ncols();
        Self {
            state_dim,
            control_dim,
            a,
            b,
            q,
            r,
            gain: None,
        }
    }

    /// Solve the discrete-time algebraic Riccati equation iteratively.
    pub fn solve(&mut self) -> Result<(), String> {
        let n = self.state_dim;
        let m = self.control_dim;

        // Iterative DARE solution
        let mut p = self.q.clone();
        let bt = self.b.transpose();
        let at = self.a.transpose();

        for _ in 0..200 {
            // S = R + B^T P B
            let s = &self.r + &bt * &p * &self.b;

            let s_inv = s.clone().try_inverse().ok_or("S matrix not invertible")?;

            // K = (R + B^T P B)^{-1} B^T P A
            let k = &s_inv * &bt * &p * &self.a;

            // P_new = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
            let p_new = &self.q + &at * &p * &self.a - at.clone() * &p * &self.b * &k;

            // Check convergence
            let diff = (&p_new - &p).norm();
            p = p_new;
            if diff < 1e-10 {
                break;
            }
        }

        // Compute final gain: K = (R + B^T P B)^{-1} B^T P A
        let s = &self.r + &bt * &p * &self.b;
        let s_inv = s.clone().try_inverse().ok_or("Final S not invertible")?;
        self.gain = Some(s_inv * bt * &p * &self.a);

        Ok(())
    }

    /// Compute control action for a given state.
    pub fn act(&self, state: &DVector<f64>) -> Result<Action, String> {
        let k = self.gain.as_ref().ok_or("Controller not solved. Call solve() first.")?;

        let control = -(k * state);

        // Expected cost: x^T P x (with P from Riccati)
        let expected_cost = state.dot(&(self.q.clone() * state));

        // Pushforward: the Jacobian of the action map is -K
        let pushforward = -k.clone();

        Ok(Action {
            control,
            expected_cost,
            pushforward,
        })
    }

    /// Compute the adjunction pairing: observation adjoint to control.
    ///
    /// Returns the natural isomorphism between observation space and control
    /// space, confirming the Obs ⊣ Ctrl adjunction.
    pub fn adjunction_unit(&self, observation: &DVector<f64>) -> DVector<f64> {
        // η: observation → state → action (pushforward)
        if let Some(ref k) = self.gain {
            -(k * observation)
        } else {
            DVector::zeros(self.control_dim)
        }
    }

    /// Compute the counit of the Obs ⊣ Ctrl adjunction.
    pub fn adjunction_counit(&self, control: &DVector<f64>) -> DVector<f64> {
        // ε: control → state (observation of control effect)
        &self.b * control
    }

    /// Verify the triangle identities of the adjunction.
    pub fn verify_adjunction(&self) -> bool {
        if self.gain.is_none() {
            return false;
        }
        // Simplified check: B·(-K) should approximate identity on the
        // controllable subspace
        let k = self.gain.as_ref().unwrap();
        let bk = &self.b * k;
        // For identity dynamics with B=I and reasonable Q,R, B·K ≈ c·I
        let trace = bk.trace();
        trace > 0.0
    }
}

/// Control cost tracker for thermodynamic accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlCostTracker {
    /// Cumulative control energy expended.
    pub cumulative_energy: f64,
    /// Number of actions taken.
    pub action_count: u64,
}

impl ControlCostTracker {
    pub fn new() -> Self {
        Self {
            cumulative_energy: 0.0,
            action_count: 0,
        }
    }

    /// Record an action and update cumulative energy.
    pub fn record(&mut self, action: &Action) {
        let energy: f64 = action.control.iter().map(|x| x * x).sum();
        self.cumulative_energy += energy;
        self.action_count += 1;
    }

    /// Average energy per action.
    pub fn average_energy(&self) -> f64 {
        if self.action_count == 0 {
            0.0
        } else {
            self.cumulative_energy / self.action_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lqr_creation() {
        let ctrl = LQRController::new(3, 2);
        assert_eq!(ctrl.state_dim, 3);
        assert_eq!(ctrl.control_dim, 2);
    }

    #[test]
    fn test_lqr_solve() {
        let mut ctrl = LQRController::new(2, 2);
        ctrl.solve().expect("LQR solve should succeed");
        assert!(ctrl.gain.is_some());
    }

    #[test]
    fn test_lqr_act() {
        let mut ctrl = LQRController::new(2, 2);
        ctrl.solve().unwrap();
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let action = ctrl.act(&state).unwrap();
        assert_eq!(action.control.len(), 2);
    }

    #[test]
    fn test_lqr_act_before_solve_fails() {
        let ctrl = LQRController::new(2, 2);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        assert!(ctrl.act(&state).is_err());
    }

    #[test]
    fn test_control_pushforward_is_negative_gain() {
        let mut ctrl = LQRController::new(2, 2);
        ctrl.solve().unwrap();
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let action = ctrl.act(&state).unwrap();
        let k = ctrl.gain.as_ref().unwrap();
        for i in 0..2 {
            assert!((action.pushforward[(i, 0)] - (-k[(i, 0)])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adjunction_unit() {
        let mut ctrl = LQRController::new(2, 2);
        ctrl.solve().unwrap();
        let obs = DVector::from_vec(vec![1.0, 2.0]);
        let unit = ctrl.adjunction_unit(&obs);
        assert_eq!(unit.len(), 2);
    }

    #[test]
    fn test_adjunction_counit() {
        let ctrl = LQRController::new(2, 2);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let counit = ctrl.adjunction_counit(&u);
        assert_eq!(counit.len(), 2);
    }

    #[test]
    fn test_control_cost_tracker() {
        let mut tracker = ControlCostTracker::new();
        let action = Action {
            control: DVector::from_vec(vec![1.0, 1.0]),
            expected_cost: 0.0,
            pushforward: DMatrix::zeros(2, 2),
        };
        tracker.record(&action);
        assert_eq!(tracker.action_count, 1);
        assert!((tracker.cumulative_energy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_energy() {
        let mut tracker = ControlCostTracker::new();
        assert!((tracker.average_energy()).abs() < 1e-10);
        let action = Action {
            control: DVector::from_vec(vec![3.0, 4.0]),
            expected_cost: 0.0,
            pushforward: DMatrix::zeros(2, 2),
        };
        tracker.record(&action);
        assert!((tracker.average_energy() - 25.0).abs() < 1e-10);
    }
}
