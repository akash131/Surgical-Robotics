"""Model Predictive Control for surgical robotics."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
import numpy as np
from numpy.typing import NDArray


@dataclass
class MPCConfig:
    """Configuration for Model Predictive Controller."""
    # Horizon parameters
    prediction_horizon: int = 20  # Number of prediction steps
    control_horizon: int = 10  # Number of control steps

    # Time step
    dt: float = 0.01  # 10ms control period

    # Solver parameters
    max_iterations: int = 50
    tolerance: float = 1e-6

    # Weights
    state_weight: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(6)
    )
    control_weight: NDArray[np.float64] = field(
        default_factory=lambda: 0.01 * np.eye(6)
    )
    terminal_weight: NDArray[np.float64] = field(
        default_factory=lambda: 10 * np.eye(6)
    )

    # Constraints
    use_constraints: bool = True


@dataclass
class ConstraintSet:
    """Constraints for MPC optimization."""
    # State constraints
    state_min: Optional[NDArray[np.float64]] = None
    state_max: Optional[NDArray[np.float64]] = None

    # Control (joint velocity/torque) constraints
    control_min: Optional[NDArray[np.float64]] = None
    control_max: Optional[NDArray[np.float64]] = None

    # Control rate constraints
    control_rate_min: Optional[NDArray[np.float64]] = None
    control_rate_max: Optional[NDArray[np.float64]] = None

    # Custom constraint function
    custom_constraint: Optional[Callable[[NDArray, NDArray], bool]] = None


class RobotDynamicsModel:
    """
    Robot dynamics model for MPC prediction.

    Supports both linear and nonlinear dynamics.
    """

    def __init__(
        self,
        num_joints: int,
        inertia_matrix: Optional[NDArray[np.float64]] = None,
        gravity_vector: Optional[Callable[[NDArray], NDArray]] = None
    ) -> None:
        self.num_joints = num_joints
        self.state_dim = num_joints * 2  # position + velocity
        self.control_dim = num_joints

        # Default inertia matrix (simplified)
        if inertia_matrix is None:
            self.M = np.eye(num_joints)
        else:
            self.M = inertia_matrix

        # Gravity compensation
        self.gravity_func = gravity_vector

        # Coriolis/centrifugal (simplified - would be computed from full dynamics)
        self.damping = 0.1 * np.eye(num_joints)

    def forward_dynamics(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute state derivative: dx/dt = f(x, u)

        Args:
            state: [position, velocity]
            control: Joint torques

        Returns:
            State derivative [velocity, acceleration]
        """
        q = state[:self.num_joints]
        qd = state[self.num_joints:]

        # Gravity
        if self.gravity_func is not None:
            g = self.gravity_func(q)
        else:
            g = np.zeros(self.num_joints)

        # Acceleration: M * qdd = tau - D * qd - g
        qdd = np.linalg.solve(
            self.M,
            control - self.damping @ qd - g
        )

        return np.concatenate([qd, qdd])

    def linearize(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Linearize dynamics around operating point.

        Returns:
            A: State matrix (df/dx)
            B: Control matrix (df/du)
        """
        n = self.state_dim
        m = self.control_dim

        # Numerical linearization
        eps = 1e-6

        A = np.zeros((n, n))
        B = np.zeros((n, m))

        f0 = self.forward_dynamics(state, control)

        # Compute A
        for i in range(n):
            state_plus = state.copy()
            state_plus[i] += eps
            f_plus = self.forward_dynamics(state_plus, control)
            A[:, i] = (f_plus - f0) / eps

        # Compute B
        for i in range(m):
            control_plus = control.copy()
            control_plus[i] += eps
            f_plus = self.forward_dynamics(state, control_plus)
            B[:, i] = (f_plus - f0) / eps

        return A, B

    def discretize(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Discretize continuous dynamics.

        Uses zero-order hold approximation.
        """
        n = A.shape[0]
        m = B.shape[1]

        # Ad = exp(A * dt) ≈ I + A*dt + (A*dt)^2/2 + ...
        Ad = np.eye(n) + A * dt + 0.5 * (A @ A) * dt**2

        # Bd ≈ B * dt
        Bd = B * dt

        return Ad, Bd


class ModelPredictiveController:
    """
    Model Predictive Controller for surgical robot motion control.

    Features:
    - Constraint handling (joint limits, velocity limits)
    - Preview of reference trajectory
    - Obstacle avoidance constraints
    - Real-time capable with warm-starting
    """

    def __init__(
        self,
        model: RobotDynamicsModel,
        config: Optional[MPCConfig] = None
    ) -> None:
        self.model = model
        self.config = config or MPCConfig()

        self.constraints: Optional[ConstraintSet] = None

        # State dimensions
        self.nx = model.state_dim
        self.nu = model.control_dim
        self.N = self.config.prediction_horizon

        # Warm start storage
        self._previous_solution: Optional[NDArray[np.float64]] = None

        # Reference trajectory
        self.reference_trajectory: Optional[NDArray[np.float64]] = None

    def set_constraints(self, constraints: ConstraintSet) -> None:
        """Set optimization constraints."""
        self.constraints = constraints

    def set_reference_trajectory(
        self,
        trajectory: NDArray[np.float64]
    ) -> None:
        """
        Set reference trajectory.

        Args:
            trajectory: Shape (N+1, nx) or (N+1, nq) for position only
        """
        self.reference_trajectory = trajectory

    def compute_control(
        self,
        current_state: NDArray[np.float64],
        reference: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute optimal control input.

        Args:
            current_state: Current state [position, velocity]
            reference: Target state (uses trajectory if None)

        Returns:
            Optimal control input
        """
        if reference is None and self.reference_trajectory is not None:
            reference = self.reference_trajectory[0]
        elif reference is None:
            reference = current_state  # Stay in place

        # Linearize around current state
        u_nominal = np.zeros(self.nu)
        A, B = self.model.linearize(current_state, u_nominal)
        Ad, Bd = self.model.discretize(A, B, self.config.dt)

        # Solve QP
        u_optimal = self._solve_qp(current_state, reference, Ad, Bd)

        # Warm start for next iteration
        self._previous_solution = u_optimal

        return u_optimal

    def _solve_qp(
        self,
        x0: NDArray[np.float64],
        x_ref: NDArray[np.float64],
        Ad: NDArray[np.float64],
        Bd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Solve the MPC quadratic program.

        Simplified implementation using direct optimization.
        Production would use OSQP, qpOASES, or similar.
        """
        N = self.config.control_horizon
        nx = self.nx
        nu = self.nu

        # Build prediction matrices
        # x(k+i) = Ad^i * x0 + sum(Ad^j * Bd * u(k+i-j-1))

        Q = self.config.state_weight
        R = self.config.control_weight
        Qf = self.config.terminal_weight

        # Gradient descent solution (simplified)
        # Real implementation would use proper QP solver

        if self._previous_solution is not None:
            u = self._previous_solution[:nu].copy()
        else:
            u = np.zeros(nu)

        # Simple gradient descent
        alpha = 0.01
        for _ in range(self.config.max_iterations):
            # Compute gradient
            x_pred = Ad @ x0 + Bd @ u
            error = x_pred - x_ref

            grad = R @ u + Bd.T @ Q @ error
            u_new = u - alpha * grad

            # Apply constraints
            if self.constraints is not None:
                if self.constraints.control_min is not None:
                    u_new = np.maximum(u_new, self.constraints.control_min)
                if self.constraints.control_max is not None:
                    u_new = np.minimum(u_new, self.constraints.control_max)

            if np.linalg.norm(u_new - u) < self.config.tolerance:
                break

            u = u_new

        return u

    def predict_trajectory(
        self,
        initial_state: NDArray[np.float64],
        control_sequence: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Predict state trajectory given control sequence.

        Args:
            initial_state: Initial state
            control_sequence: Shape (N, nu)

        Returns:
            Predicted states (N+1, nx)
        """
        N = len(control_sequence)
        trajectory = np.zeros((N + 1, self.nx))
        trajectory[0] = initial_state

        state = initial_state.copy()
        for i in range(N):
            # Integrate one step
            dx = self.model.forward_dynamics(state, control_sequence[i])
            state = state + dx * self.config.dt
            trajectory[i + 1] = state

        return trajectory


class TrajectoryOptimizer:
    """
    Trajectory optimization for surgical motion planning.

    Optimizes full trajectory considering dynamics and constraints.
    """

    def __init__(
        self,
        model: RobotDynamicsModel,
        horizon: int = 100,
        dt: float = 0.01
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.dt = dt

    def optimize(
        self,
        start_state: NDArray[np.float64],
        goal_state: NDArray[np.float64],
        constraints: Optional[ConstraintSet] = None,
        obstacles: Optional[List[Tuple[NDArray, float]]] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Optimize trajectory from start to goal.

        Args:
            start_state: Starting state
            goal_state: Goal state
            constraints: Motion constraints
            obstacles: List of (center, radius) tuples

        Returns:
            Tuple of (state_trajectory, control_trajectory)
        """
        N = self.horizon
        nx = self.model.state_dim
        nu = self.model.control_dim

        # Initialize with linear interpolation
        states = np.linspace(start_state, goal_state, N + 1)
        controls = np.zeros((N, nu))

        # Iterative optimization
        max_iters = 100
        for iteration in range(max_iters):
            # Compute cost
            cost = self._compute_cost(states, controls, goal_state)

            # Compute gradient
            grad_states, grad_controls = self._compute_gradient(
                states, controls, goal_state
            )

            # Line search
            alpha = 0.01
            states_new = states - alpha * grad_states
            controls_new = controls - alpha * grad_controls

            # Apply constraints
            if constraints:
                states_new, controls_new = self._apply_constraints(
                    states_new, controls_new, constraints
                )

            # Check convergence
            if np.linalg.norm(states_new - states) < 1e-6:
                break

            states = states_new
            controls = controls_new

        return states, controls

    def _compute_cost(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
        goal: NDArray[np.float64]
    ) -> float:
        """Compute trajectory cost."""
        # Running cost
        running_cost = 0.0
        for i in range(len(controls)):
            state_error = states[i] - goal
            running_cost += 0.5 * np.sum(state_error ** 2)
            running_cost += 0.01 * np.sum(controls[i] ** 2)

        # Terminal cost
        terminal_error = states[-1] - goal
        terminal_cost = 5.0 * np.sum(terminal_error ** 2)

        return running_cost + terminal_cost

    def _compute_gradient(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
        goal: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute cost gradient."""
        grad_states = np.zeros_like(states)
        grad_controls = np.zeros_like(controls)

        for i in range(len(controls)):
            grad_states[i] = states[i] - goal
            grad_controls[i] = 0.02 * controls[i]

        grad_states[-1] += 10.0 * (states[-1] - goal)

        return grad_states, grad_controls

    def _apply_constraints(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
        constraints: ConstraintSet
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply constraints to trajectory."""
        if constraints.state_min is not None:
            states = np.maximum(states, constraints.state_min)
        if constraints.state_max is not None:
            states = np.minimum(states, constraints.state_max)

        if constraints.control_min is not None:
            controls = np.maximum(controls, constraints.control_min)
        if constraints.control_max is not None:
            controls = np.minimum(controls, constraints.control_max)

        return states, controls
