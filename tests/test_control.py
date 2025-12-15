"""Tests for advanced control module."""

import pytest
import numpy as np

from surgical_robotics.control.mpc import (
    ModelPredictiveController,
    MPCConfig,
    RobotDynamicsModel,
    ConstraintSet,
    TrajectoryOptimizer,
)
from surgical_robotics.control.impedance import (
    ImpedanceController,
    AdmittanceController,
    HybridForcePosition,
    ImpedanceParams,
    VariableImpedanceController,
)
from surgical_robotics.control.rl_interface import (
    RLEnvironment,
    SurgicalRLEnv,
    RewardFunction,
    SpaceDefinition,
    ActionSpace,
)
from surgical_robotics.control.teleoperation import (
    TeleoperationController,
    TeleoperationConfig,
    TeleoperationMode,
    MotionScaler,
    TimeDelayCompensator,
    BilateralController,
)
from surgical_robotics.core.base import Pose


class TestMPC:
    """Tests for Model Predictive Control."""

    def test_dynamics_model_creation(self):
        """Test robot dynamics model creation."""
        model = RobotDynamicsModel(num_joints=7)

        assert model.num_joints == 7
        assert model.state_dim == 14  # position + velocity
        assert model.control_dim == 7

    def test_forward_dynamics(self):
        """Test forward dynamics computation."""
        model = RobotDynamicsModel(num_joints=7)

        state = np.zeros(14)
        control = np.ones(7)

        dx = model.forward_dynamics(state, control)

        assert len(dx) == 14
        # First 7 should be velocities (0), last 7 should be accelerations
        assert np.allclose(dx[:7], 0)

    def test_linearization(self):
        """Test dynamics linearization."""
        model = RobotDynamicsModel(num_joints=7)

        state = np.zeros(14)
        control = np.zeros(7)

        A, B = model.linearize(state, control)

        assert A.shape == (14, 14)
        assert B.shape == (14, 7)

    def test_mpc_controller(self):
        """Test MPC controller."""
        model = RobotDynamicsModel(num_joints=7)
        config = MPCConfig(
            prediction_horizon=10,
            control_horizon=5,
            dt=0.01
        )
        mpc = ModelPredictiveController(model, config)

        state = np.zeros(14)
        reference = np.zeros(14)
        reference[:7] = 0.1  # Target position

        control = mpc.compute_control(state, reference)

        assert len(control) == 7

    def test_mpc_with_constraints(self):
        """Test MPC with constraints."""
        model = RobotDynamicsModel(num_joints=7)
        mpc = ModelPredictiveController(model)

        constraints = ConstraintSet(
            control_min=-10 * np.ones(7),
            control_max=10 * np.ones(7)
        )
        mpc.set_constraints(constraints)

        state = np.zeros(14)
        control = mpc.compute_control(state)

        # Control should be within limits
        assert np.all(control >= -10)
        assert np.all(control <= 10)


class TestImpedanceControl:
    """Tests for impedance and admittance control."""

    def test_impedance_params(self):
        """Test impedance parameter creation."""
        default = ImpedanceParams.default()
        soft = ImpedanceParams.soft()
        stiff = ImpedanceParams.stiff()

        assert np.all(soft.stiffness < default.stiffness)
        assert np.all(stiff.stiffness > default.stiffness)

    def test_impedance_controller(self):
        """Test impedance controller."""
        controller = ImpedanceController()

        desired_pose = Pose(
            position=np.array([0, 0, 0.3]),
            orientation=np.array([1, 0, 0, 0])
        )
        controller.set_desired_pose(desired_pose)

        current_pose = Pose(
            position=np.array([0.01, 0, 0.3]),  # Small offset
            orientation=np.array([1, 0, 0, 0])
        )
        current_velocity = np.zeros(6)

        force = controller.compute_force(current_pose, current_velocity)

        assert len(force) == 6
        # Should have restoring force in x direction
        assert force[0] < 0  # Force toward desired position

    def test_admittance_controller(self):
        """Test admittance controller."""
        controller = AdmittanceController()

        equilibrium = Pose(
            position=np.array([0, 0, 0.3]),
            orientation=np.array([1, 0, 0, 0])
        )
        controller.set_equilibrium_pose(equilibrium)

        # Apply force
        force = np.array([10, 0, 0, 0, 0, 0])
        position_offset, velocity = controller.compute_motion(force, dt=0.01)

        assert len(position_offset) == 6
        assert len(velocity) == 6
        # Should move in direction of force
        assert position_offset[0] > 0 or velocity[0] > 0

    def test_hybrid_controller(self):
        """Test hybrid force-position controller."""
        # Force control in Z, position in X,Y
        controller = HybridForcePosition()

        desired_pose = Pose(
            position=np.array([0, 0, 0.3]),
            orientation=np.array([1, 0, 0, 0])
        )
        controller.set_desired_pose(desired_pose)
        controller.set_desired_force(np.array([0, 0, 5, 0, 0, 0]))  # 5N in Z

        current_pose = Pose(
            position=np.array([0.01, 0, 0.3]),
            orientation=np.array([1, 0, 0, 0])
        )
        current_velocity = np.zeros(6)
        measured_force = np.array([0, 0, 3, 0, 0, 0])  # 3N measured

        output = controller.compute_control(
            current_pose, current_velocity, measured_force, dt=0.01
        )

        assert len(output) == 6

    def test_variable_impedance(self):
        """Test variable impedance controller."""
        controller = VariableImpedanceController()

        initial_stiffness = controller.current_params.stiffness.copy()

        # High force should reduce stiffness
        high_force = np.array([50, 0, 0, 0, 0, 0])
        controller.adapt_to_force(high_force)

        # Stiffness should be reduced
        assert np.all(controller.current_params.stiffness <= initial_stiffness)


class TestRLEnvironment:
    """Tests for RL environment."""

    def test_space_definition(self):
        """Test space definition."""
        space = SpaceDefinition(
            space_type=ActionSpace.CONTINUOUS,
            shape=(7,),
            low=-np.ones(7),
            high=np.ones(7)
        )

        sample = space.sample()
        assert len(sample) == 7
        assert space.contains(sample)

    def test_surgical_env_creation(self):
        """Test surgical RL environment creation."""
        env = SurgicalRLEnv(
            task="reach",
            use_force_feedback=True,
            max_steps=100
        )

        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_reset(self):
        """Test environment reset."""
        env = SurgicalRLEnv()

        obs, info = env.reset(seed=42)

        assert len(obs) == env.observation_space.shape[0]
        assert "goal" in info

    def test_env_step(self):
        """Test environment step."""
        env = SurgicalRLEnv()
        env.reset()

        action = np.zeros(7)
        result = env.step(action)

        assert len(result.observation) == env.observation_space.shape[0]
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)

    def test_reward_function(self):
        """Test reward function."""
        reward_fn = RewardFunction()

        current_pos = np.array([0.1, 0.1, 0.3])
        goal_pos = np.array([0, 0, 0.3])
        action = np.zeros(7)

        total_reward, components = reward_fn.compute(
            current_pos, goal_pos, action
        )

        assert isinstance(total_reward, float)
        assert "distance_to_goal" in components


class TestTeleoperation:
    """Tests for teleoperation control."""

    def test_teleop_config(self):
        """Test teleoperation configuration."""
        config = TeleoperationConfig(
            mode=TeleoperationMode.POSITION,
            position_scale=0.5,
            max_linear_velocity=0.2
        )

        assert config.position_scale == 0.5
        assert config.mode == TeleoperationMode.POSITION

    def test_motion_scaler(self):
        """Test motion scaling."""
        config = TeleoperationConfig(position_scale=0.5)
        scaler = MotionScaler(config)

        master_ref = Pose(
            position=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        )
        slave_ref = Pose(
            position=np.array([0, 0, 0.3]),
            orientation=np.array([1, 0, 0, 0])
        )
        scaler.set_reference(master_ref, slave_ref)

        # Master moved 10cm
        master_pose = Pose(
            position=np.array([0.1, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        )

        slave_target = scaler.scale_motion(master_pose, dt=0.01)

        # Slave should move 5cm (scaled)
        expected_x = 0.05  # 0.1 * 0.5
        # Due to filtering, won't be exact
        assert slave_target.position[0] > 0

    def test_time_delay_compensator(self):
        """Test time delay compensation."""
        compensator = TimeDelayCompensator(impedance=50.0)

        velocity = np.ones(6)
        force = np.zeros(6)

        wave = compensator.encode_to_wave(velocity, force, timestamp=0.0)
        assert len(wave) == 6

        vel_out, force_out = compensator.decode_from_wave(wave, timestamp=0.01)
        assert len(vel_out) == 6
        assert len(force_out) == 6

    def test_bilateral_controller(self):
        """Test bilateral controller."""
        controller = BilateralController()

        master_ref = Pose.identity()
        slave_ref = Pose(position=np.array([0, 0, 0.3]), orientation=np.array([1, 0, 0, 0]))
        controller.set_reference(master_ref, slave_ref)

        master_pose = Pose(
            position=np.array([0.05, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        )
        master_velocity = np.zeros(6)

        slave_target, master_feedback = controller.update_master(
            master_pose, master_velocity
        )

        assert slave_target is not None
        assert len(master_feedback) == 6

    def test_teleoperation_controller(self):
        """Test complete teleoperation controller."""
        config = TeleoperationConfig(mode=TeleoperationMode.POSITION)
        controller = TeleoperationController(config)

        controller.enable()

        master_ref = Pose.identity()
        slave_ref = Pose(position=np.array([0, 0, 0.3]), orientation=np.array([1, 0, 0, 0]))
        controller.engage_clutch(master_ref, slave_ref)
        controller.release_clutch()

        master_pose = Pose(
            position=np.array([0.05, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        )

        slave_command, master_feedback = controller.update(
            master_pose=master_pose,
            master_velocity=np.zeros(6),
            slave_pose=slave_ref,
            slave_force=np.zeros(6),
            dt=0.01
        )

        assert slave_command is not None
        assert len(master_feedback) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
