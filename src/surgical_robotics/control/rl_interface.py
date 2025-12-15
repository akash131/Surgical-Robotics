"""Reinforcement Learning interfaces for surgical robotics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Callable
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class ActionSpace(Enum):
    """Action space types."""
    DISCRETE = auto()
    CONTINUOUS = auto()
    MULTI_DISCRETE = auto()
    MULTI_BINARY = auto()


@dataclass
class SpaceDefinition:
    """Definition of observation or action space."""
    space_type: ActionSpace
    shape: Tuple[int, ...]
    low: Optional[NDArray[np.float64]] = None
    high: Optional[NDArray[np.float64]] = None
    n_actions: Optional[int] = None  # For discrete spaces

    def sample(self) -> NDArray:
        """Sample from the space."""
        if self.space_type == ActionSpace.CONTINUOUS:
            if self.low is not None and self.high is not None:
                return np.random.uniform(self.low, self.high)
            return np.random.randn(*self.shape)
        elif self.space_type == ActionSpace.DISCRETE:
            return np.array([np.random.randint(0, self.n_actions or 1)])
        else:
            return np.zeros(self.shape)

    def contains(self, x: NDArray) -> bool:
        """Check if value is in space."""
        if self.space_type == ActionSpace.CONTINUOUS:
            if self.low is not None and self.high is not None:
                return bool(np.all(x >= self.low) and np.all(x <= self.high))
        return True


@dataclass
class StepResult:
    """Result of environment step."""
    observation: NDArray[np.float64]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


class RLEnvironment(ABC):
    """
    Abstract base class for RL environments.

    Compatible with Gymnasium interface for use with
    standard RL libraries (Stable-Baselines3, RLlib, etc.)
    """

    def __init__(self) -> None:
        self.observation_space: Optional[SpaceDefinition] = None
        self.action_space: Optional[SpaceDefinition] = None
        self._step_count = 0
        self._max_steps = 1000

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Reset environment to initial state.

        Returns:
            Tuple of (observation, info)
        """
        pass

    @abstractmethod
    def step(self, action: NDArray) -> StepResult:
        """
        Take action in environment.

        Args:
            action: Action to take

        Returns:
            StepResult with observation, reward, done flags, info
        """
        pass

    @abstractmethod
    def render(self) -> Optional[NDArray]:
        """Render environment (optional)."""
        pass

    def close(self) -> None:
        """Clean up environment."""
        pass

    @property
    def unwrapped(self) -> "RLEnvironment":
        """Return unwrapped environment."""
        return self


class RewardFunction:
    """
    Reward function for surgical RL tasks.

    Combines multiple reward components with configurable weights.
    """

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {
            "distance_to_goal": 1.0,
            "smoothness": 0.1,
            "safety": 10.0,
            "time_penalty": 0.01,
            "success_bonus": 100.0,
            "collision_penalty": -50.0,
        }

        self._last_action: Optional[NDArray] = None
        self._last_position: Optional[NDArray] = None

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set reward component weights."""
        self.weights.update(weights)

    def compute(
        self,
        current_position: NDArray[np.float64],
        goal_position: NDArray[np.float64],
        action: NDArray[np.float64],
        is_collision: bool = False,
        is_success: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward.

        Args:
            current_position: Current position
            goal_position: Goal position
            action: Action taken
            is_collision: Whether collision occurred
            is_success: Whether task completed successfully

        Returns:
            Tuple of (total_reward, component_dict)
        """
        components: Dict[str, float] = {}

        # Distance to goal (negative distance = reward approaches 0)
        distance = float(np.linalg.norm(current_position - goal_position))
        components["distance_to_goal"] = -distance

        # Smoothness (penalize large action changes)
        if self._last_action is not None:
            action_diff = float(np.linalg.norm(action - self._last_action))
            components["smoothness"] = -action_diff
        else:
            components["smoothness"] = 0.0

        # Time penalty
        components["time_penalty"] = -1.0

        # Collision penalty
        if is_collision:
            components["collision_penalty"] = 1.0
        else:
            components["collision_penalty"] = 0.0

        # Success bonus
        if is_success:
            components["success_bonus"] = 1.0
        else:
            components["success_bonus"] = 0.0

        # Safety (distance from obstacles, joint limits, etc.)
        components["safety"] = 0.0  # Would be computed from safety checks

        # Compute weighted sum
        total_reward = sum(
            self.weights.get(k, 0.0) * v
            for k, v in components.items()
        )

        # Update state
        self._last_action = action.copy()
        self._last_position = current_position.copy()

        return total_reward, components

    def reset(self) -> None:
        """Reset reward function state."""
        self._last_action = None
        self._last_position = None


class SurgicalRLEnv(RLEnvironment):
    """
    Reinforcement learning environment for surgical robotics.

    Provides a Gymnasium-compatible interface for training
    RL agents on surgical tasks.
    """

    def __init__(
        self,
        task: str = "reach",
        use_force_feedback: bool = True,
        max_steps: int = 500
    ) -> None:
        super().__init__()
        self.task = task
        self.use_force_feedback = use_force_feedback
        self._max_steps = max_steps

        # Robot state
        self._joint_positions = np.zeros(7)
        self._joint_velocities = np.zeros(7)
        self._ee_position = np.zeros(3)
        self._ee_orientation = np.array([1, 0, 0, 0])
        self._ee_force = np.zeros(6)

        # Goal
        self._goal_position = np.zeros(3)
        self._goal_orientation = np.array([1, 0, 0, 0])

        # Reward function
        self.reward_function = RewardFunction()

        # Define spaces
        self._define_spaces()

    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: joint positions, velocities, EE pose, goal, (force)
        obs_dim = 7 + 7 + 7 + 3  # joints + velocities + ee_pose + goal
        if self.use_force_feedback:
            obs_dim += 6  # force/torque

        self.observation_space = SpaceDefinition(
            space_type=ActionSpace.CONTINUOUS,
            shape=(obs_dim,),
            low=-np.inf * np.ones(obs_dim),
            high=np.inf * np.ones(obs_dim)
        )

        # Action: joint velocity commands
        self.action_space = SpaceDefinition(
            space_type=ActionSpace.CONTINUOUS,
            shape=(7,),
            low=-1.0 * np.ones(7),
            high=1.0 * np.ones(7)
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0

        # Reset robot to random initial configuration
        self._joint_positions = np.random.uniform(-0.5, 0.5, 7)
        self._joint_velocities = np.zeros(7)

        # Set random goal
        self._goal_position = np.random.uniform(-0.3, 0.3, 3)
        self._goal_position[2] = np.random.uniform(0.1, 0.4)  # Above table

        # Reset reward function
        self.reward_function.reset()

        # Compute forward kinematics (simplified)
        self._update_ee_pose()

        obs = self._get_observation()
        info = {"goal": self._goal_position.copy()}

        return obs, info

    def step(self, action: NDArray) -> StepResult:
        """Execute action in environment."""
        self._step_count += 1

        # Scale action to velocity limits
        max_velocity = 0.5  # rad/s
        joint_velocity = action * max_velocity

        # Integrate joint positions
        dt = 0.01
        self._joint_positions += joint_velocity * dt
        self._joint_velocities = joint_velocity

        # Apply joint limits
        self._joint_positions = np.clip(self._joint_positions, -np.pi, np.pi)

        # Update end-effector pose
        self._update_ee_pose()

        # Check termination conditions
        distance_to_goal = float(np.linalg.norm(
            self._ee_position - self._goal_position
        ))
        success = distance_to_goal < 0.01  # 1cm threshold
        collision = self._check_collision()

        terminated = success or collision
        truncated = self._step_count >= self._max_steps

        # Compute reward
        reward, reward_info = self.reward_function.compute(
            current_position=self._ee_position,
            goal_position=self._goal_position,
            action=action,
            is_collision=collision,
            is_success=success
        )

        obs = self._get_observation()
        info = {
            "success": success,
            "collision": collision,
            "distance_to_goal": distance_to_goal,
            "reward_components": reward_info
        }

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )

    def render(self) -> Optional[NDArray]:
        """Render environment (placeholder)."""
        return None

    def _get_observation(self) -> NDArray[np.float64]:
        """Get current observation."""
        ee_pose = np.concatenate([
            self._ee_position,
            self._ee_orientation
        ])

        obs_parts = [
            self._joint_positions,
            self._joint_velocities,
            ee_pose,
            self._goal_position
        ]

        if self.use_force_feedback:
            obs_parts.append(self._ee_force)

        return np.concatenate(obs_parts)

    def _update_ee_pose(self) -> None:
        """Update end-effector pose from joint positions (simplified FK)."""
        # Simplified forward kinematics
        # In practice, would use actual robot kinematics
        self._ee_position = np.array([
            0.2 * np.sin(self._joint_positions[0]) +
            0.2 * np.sin(self._joint_positions[0] + self._joint_positions[1]),
            0.2 * np.cos(self._joint_positions[0]) +
            0.2 * np.cos(self._joint_positions[0] + self._joint_positions[1]),
            0.3 + 0.1 * np.sin(self._joint_positions[2])
        ])

    def _check_collision(self) -> bool:
        """Check for collisions (simplified)."""
        # Check workspace bounds
        if np.any(self._ee_position < -0.5) or np.any(self._ee_position > 0.5):
            return True
        # Check table collision
        if self._ee_position[2] < 0.0:
            return True
        return False

    def set_goal(self, position: NDArray[np.float64]) -> None:
        """Manually set goal position."""
        self._goal_position = position.copy()


class PolicyInterface(ABC):
    """
    Abstract interface for RL policies.

    Allows integration of trained policies with the surgical robot system.
    """

    @abstractmethod
    def load(self, path: str) -> bool:
        """Load policy from file."""
        pass

    @abstractmethod
    def predict(
        self,
        observation: NDArray[np.float64],
        deterministic: bool = True
    ) -> Tuple[NDArray[np.float64], Optional[Dict]]:
        """
        Predict action from observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, info)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset policy state (for recurrent policies)."""
        pass


class NeuralNetworkPolicy(PolicyInterface):
    """
    Neural network policy wrapper.

    Wraps trained neural network models for inference.
    """

    def __init__(self) -> None:
        self.model = None
        self._observation_normalizer: Optional[Callable] = None

    def load(self, path: str) -> bool:
        """
        Load policy from file.

        Supports common formats (ONNX, PyTorch, TensorFlow SavedModel).
        """
        # Would load model from path
        return True

    def predict(
        self,
        observation: NDArray[np.float64],
        deterministic: bool = True
    ) -> Tuple[NDArray[np.float64], Optional[Dict]]:
        """Predict action from observation."""
        if self.model is None:
            return np.zeros(7), None

        # Normalize observation if normalizer is set
        if self._observation_normalizer:
            observation = self._observation_normalizer(observation)

        # Would run inference on model
        # Placeholder: return zero action
        action = np.zeros(7)

        return action, None

    def reset(self) -> None:
        """Reset policy state."""
        pass

    def set_observation_normalizer(
        self,
        normalizer: Callable[[NDArray], NDArray]
    ) -> None:
        """Set observation normalization function."""
        self._observation_normalizer = normalizer


class ImitationLearningWrapper:
    """
    Wrapper for imitation learning from demonstrations.

    Collects expert demonstrations and provides interface
    for behavior cloning or inverse RL.
    """

    def __init__(self) -> None:
        self.demonstrations: List[Dict[str, NDArray]] = []
        self._current_episode: Dict[str, List] = {
            "observations": [],
            "actions": [],
            "rewards": []
        }
        self._recording = False

    def start_recording(self) -> None:
        """Start recording demonstration."""
        self._recording = True
        self._current_episode = {
            "observations": [],
            "actions": [],
            "rewards": []
        }

    def record_step(
        self,
        observation: NDArray[np.float64],
        action: NDArray[np.float64],
        reward: float
    ) -> None:
        """Record single step."""
        if not self._recording:
            return

        self._current_episode["observations"].append(observation.copy())
        self._current_episode["actions"].append(action.copy())
        self._current_episode["rewards"].append(reward)

    def stop_recording(self) -> None:
        """Stop recording and save demonstration."""
        if not self._recording:
            return

        self._recording = False

        if self._current_episode["observations"]:
            demonstration = {
                "observations": np.array(self._current_episode["observations"]),
                "actions": np.array(self._current_episode["actions"]),
                "rewards": np.array(self._current_episode["rewards"])
            }
            self.demonstrations.append(demonstration)

    def get_demonstrations(self) -> List[Dict[str, NDArray]]:
        """Get all recorded demonstrations."""
        return self.demonstrations

    def save_demonstrations(self, path: str) -> bool:
        """Save demonstrations to file."""
        # Would save to NPZ or similar format
        return True

    def load_demonstrations(self, path: str) -> bool:
        """Load demonstrations from file."""
        # Would load from file
        return True
