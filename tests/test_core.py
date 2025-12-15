"""Tests for core surgical robotics components."""

import numpy as np
import pytest

from surgical_robotics.core.base import (
    Pose,
    JointState,
    JointLimits,
    EndEffector,
    RobotArm,
    WorkspaceVolume,
    RobotState,
    SafetyLevel,
)
from surgical_robotics.core.kinematics import (
    DHParameters,
    ForwardKinematics,
    InverseKinematics,
    TrajectoryPlanner,
)
from surgical_robotics.core.safety import (
    SafetyController,
    CollisionDetector,
    CollisionType,
    SafetyZone,
    PlaneFixture,
    PathFixture,
)


class TestPose:
    """Tests for Pose class."""

    def test_identity_pose(self):
        """Test identity pose creation."""
        pose = Pose.identity()
        np.testing.assert_array_almost_equal(pose.position, [0, 0, 0])
        np.testing.assert_array_almost_equal(pose.orientation, [1, 0, 0, 0])

    def test_pose_normalization(self):
        """Test quaternion normalization in Pose."""
        pose = Pose(
            position=np.array([1, 2, 3]),
            orientation=np.array([2, 0, 0, 0])  # Non-normalized
        )
        # Should be normalized to [1, 0, 0, 0]
        np.testing.assert_array_almost_equal(pose.orientation, [1, 0, 0, 0])

    def test_pose_to_matrix(self):
        """Test conversion to transformation matrix."""
        pose = Pose.identity()
        matrix = pose.to_matrix()

        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix, np.eye(4))


class TestDHParameters:
    """Tests for DH parameters."""

    def test_dh_transformation(self):
        """Test DH transformation matrix computation."""
        # Simple rotation about Z
        dh = DHParameters(theta=np.pi/2, d=0, a=0, alpha=0)
        T = dh.transformation_matrix()

        expected_rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T[:3, :3], expected_rotation)

    def test_dh_with_offset(self):
        """Test DH transformation with joint offset."""
        dh = DHParameters(theta=0, d=0, a=1.0, alpha=0)
        T = dh.transformation_matrix(joint_offset=0)

        # Translation along X axis
        assert T[0, 3] == pytest.approx(1.0)


class TestForwardKinematics:
    """Tests for forward kinematics solver."""

    def test_simple_chain(self):
        """Test FK for simple 2-link planar arm."""
        # Two links of length 1, rotating about Z
        dh_params = [
            DHParameters(0, 0, 1.0, 0),
            DHParameters(0, 0, 1.0, 0),
        ]
        fk = ForwardKinematics(dh_params)

        # Both joints at 0
        pose = fk.compute(np.array([0, 0]))
        np.testing.assert_array_almost_equal(pose.position, [2, 0, 0])

        # First joint at 90 degrees
        pose = fk.compute(np.array([np.pi/2, 0]))
        np.testing.assert_array_almost_equal(pose.position, [0, 2, 0], decimal=5)

    def test_all_frames(self):
        """Test computation of all intermediate frames."""
        dh_params = [
            DHParameters(0, 0, 1.0, 0),
            DHParameters(0, 0, 1.0, 0),
        ]
        fk = ForwardKinematics(dh_params)

        frames = fk.compute_all_frames(np.array([0, 0]))
        assert len(frames) == 3  # Base + 2 joints


class TestInverseKinematics:
    """Tests for inverse kinematics solver."""

    def test_ik_solution(self):
        """Test IK can find solution for reachable pose."""
        dh_params = [
            DHParameters(0, 0, 1.0, np.pi/2),
            DHParameters(0, 0, 1.0, 0),
            DHParameters(0, 0, 1.0, 0),
        ]
        fk = ForwardKinematics(dh_params)
        ik = InverseKinematics(fk, max_iterations=200, tolerance=1e-4)

        # Start from a known configuration
        initial_joints = np.array([0.1, 0.2, 0.3])
        target_pose = fk.compute(initial_joints)

        # Try to find solution from different starting point
        solution, success = ik.compute(
            target_pose,
            initial_guess=np.zeros(3)
        )

        # Verify solution reaches target
        achieved_pose = fk.compute(solution)
        position_error = np.linalg.norm(achieved_pose.position - target_pose.position)
        assert position_error < 0.001  # 1mm tolerance


class TestTrajectoryPlanner:
    """Tests for trajectory planning."""

    def test_linear_interpolation(self):
        """Test linear interpolation between configurations."""
        start = np.array([0, 0, 0])
        end = np.array([1, 2, 3])
        num_points = 11

        trajectory = TrajectoryPlanner.linear_interpolation(start, end, num_points)

        assert trajectory.shape == (11, 3)
        np.testing.assert_array_almost_equal(trajectory[0], start)
        np.testing.assert_array_almost_equal(trajectory[-1], end)
        np.testing.assert_array_almost_equal(trajectory[5], [0.5, 1, 1.5])

    def test_quintic_polynomial(self):
        """Test quintic polynomial trajectory generation."""
        start = np.array([0, 0])
        end = np.array([1, 1])
        duration = 1.0
        dt = 0.1

        positions, velocities, accelerations = TrajectoryPlanner.quintic_polynomial(
            start, end, duration, dt
        )

        # Check boundary conditions
        np.testing.assert_array_almost_equal(positions[0], start)
        np.testing.assert_array_almost_equal(positions[-1], end, decimal=5)

        # Velocities should be zero at endpoints
        np.testing.assert_array_almost_equal(velocities[0], [0, 0], decimal=3)
        np.testing.assert_array_almost_equal(velocities[-1], [0, 0], decimal=3)


class TestSafetyController:
    """Tests for safety controller."""

    def test_force_limits(self):
        """Test force limit checking."""
        controller = SafetyController()
        controller.force_limits.max_force = 10.0

        # Within limits
        level = controller.check_force_limits(
            forces=np.array([5, 0, 0]),
            torques=np.array([0, 0, 0])
        )
        assert level == SafetyLevel.NORMAL

        # Approaching limits (>70%)
        level = controller.check_force_limits(
            forces=np.array([8, 0, 0]),
            torques=np.array([0, 0, 0])
        )
        assert level == SafetyLevel.CAUTION

    def test_trajectory_validation(self):
        """Test trajectory validation."""
        controller = SafetyController()

        positions = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        velocities = np.array([[0, 0], [1, 1], [0, 0]])
        accelerations = np.array([[0, 0], [0, 0], [0, 0]])

        joint_limits = (np.array([-1, -1]), np.array([2, 2]))
        velocity_limits = np.array([2, 2])
        acceleration_limits = np.array([5, 5])

        valid, message = controller.validate_trajectory(
            positions, velocities, accelerations,
            joint_limits, velocity_limits, acceleration_limits
        )

        assert valid


class TestCollisionDetector:
    """Tests for collision detection."""

    def test_no_collision(self):
        """Test detection with no collision."""
        detector = CollisionDetector()
        result = detector.check_collision(np.array([0, 0, 0]))

        assert not result.collision_detected

    def test_path_collision_check(self):
        """Test collision checking along path."""
        detector = CollisionDetector()

        result = detector.check_path_collision(
            start=np.array([0, 0, 0]),
            end=np.array([1, 1, 1])
        )

        assert not result.collision_detected


class TestWorkspaceVolume:
    """Tests for workspace volume."""

    def test_cuboid_contains(self):
        """Test point containment for cuboid workspace."""
        workspace = WorkspaceVolume(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            shape="cuboid"
        )

        assert workspace.contains_point(np.array([0, 0, 0]))
        assert workspace.contains_point(np.array([0.9, 0.9, 0.9]))
        assert not workspace.contains_point(np.array([1.5, 0, 0]))

    def test_sphere_contains(self):
        """Test point containment for spherical workspace."""
        workspace = WorkspaceVolume(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),  # Diameter 2
            shape="sphere"
        )

        assert workspace.contains_point(np.array([0, 0, 0]))
        assert workspace.contains_point(np.array([0.5, 0.5, 0.5]))
        assert not workspace.contains_point(np.array([1.5, 0, 0]))


class TestVirtualFixtures:
    """Tests for virtual fixtures."""

    def test_plane_fixture(self):
        """Test plane constraint force computation."""
        fixture = PlaneFixture(
            name="test_plane",
            point_on_plane=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1]),
            stiffness=1000.0
        )
        fixture.is_active = True

        # Point on plane - no force
        force = fixture.compute_constraint_force(np.array([1, 1, 0]))
        np.testing.assert_array_almost_equal(force, [0, 0, 0])

        # Point above plane - force pushing down
        force = fixture.compute_constraint_force(np.array([0, 0, 0.1]))
        assert force[2] < 0  # Pushing back toward plane

    def test_path_fixture(self):
        """Test path constraint force computation."""
        path_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ])
        fixture = PathFixture(
            name="test_path",
            path_points=path_points,
            stiffness=1000.0,
            tube_radius=0.05
        )
        fixture.is_active = True

        # Point on path - no force
        force = fixture.compute_constraint_force(np.array([0.5, 0, 0]))
        np.testing.assert_array_almost_equal(force, [0, 0, 0])

        # Point off path - force toward path
        force = fixture.compute_constraint_force(np.array([0.5, 0.1, 0]))
        assert force[1] < 0  # Pushing back toward path
