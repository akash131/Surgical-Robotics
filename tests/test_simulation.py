"""Tests for simulation module."""

import pytest
import numpy as np

from surgical_robotics.simulation.physics import (
    PhysicsEngine,
    InternalPhysicsEngine,
    PhysicsConfig,
    PhysicsBackend,
    RigidBody,
    CollisionShape,
)
from surgical_robotics.simulation.scene import (
    SurgicalScene,
    SceneObject,
    ObjectType,
    OperatingTable,
    PatientModel,
)
from surgical_robotics.simulation.tissue import (
    TissueSimulator,
    TissueProperties,
    TissueType,
    DeformableTissue,
    CuttableTissue,
)
from surgical_robotics.simulation.robot_sim import (
    RobotSimulator,
    SimulatedJoint,
    JointType,
)


class TestPhysicsEngine:
    """Tests for physics engine."""

    def test_internal_engine_initialization(self):
        """Test internal physics engine initialization."""
        config = PhysicsConfig(backend=PhysicsBackend.INTERNAL)
        engine = InternalPhysicsEngine(config)
        assert engine.initialize()
        assert engine._initialized

    def test_add_rigid_body(self):
        """Test adding rigid body to simulation."""
        engine = InternalPhysicsEngine()
        engine.initialize()

        body = RigidBody(
            name="test_body",
            mass=1.0,
            position=np.array([0, 0, 1]),
            orientation=np.array([1, 0, 0, 0])
        )

        body_id = engine.add_rigid_body(body)
        assert body_id >= 0
        assert "test_body" in engine.bodies

    def test_physics_step(self):
        """Test physics simulation step."""
        engine = InternalPhysicsEngine()
        engine.initialize()

        body = RigidBody(
            name="falling_body",
            mass=1.0,
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1, 0, 0, 0]),
            is_static=False
        )
        engine.add_rigid_body(body)

        # Step simulation
        initial_z = body.position[2]
        engine.step(0.1)

        # Body should fall due to gravity
        assert body.position[2] < initial_z

    def test_apply_force(self):
        """Test applying force to body."""
        engine = InternalPhysicsEngine()
        engine.initialize()

        body = RigidBody(
            name="pushed_body",
            mass=1.0,
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1, 0, 0, 0]),
            is_static=False
        )
        engine.add_rigid_body(body)

        # Apply upward force to counteract gravity
        engine.apply_force("pushed_body", np.array([0, 0, 20]))
        initial_pos = body.position.copy()
        engine.step(0.01)

        # Body should move upward (force > mg)
        assert body.position[2] > initial_pos[2]


class TestSurgicalScene:
    """Tests for surgical scene."""

    def test_scene_creation(self):
        """Test creating surgical scene."""
        scene = SurgicalScene()
        assert scene is not None
        assert scene.physics is not None

    def test_setup_operating_room(self):
        """Test operating room setup."""
        scene = SurgicalScene()
        scene.setup_operating_room()

        assert "floor" in scene.objects
        assert "operating_table" in scene.objects
        assert scene.operating_table is not None

    def test_add_patient(self):
        """Test adding patient to scene."""
        scene = SurgicalScene()
        scene.setup_operating_room()
        scene.add_patient()

        assert scene.patient is not None
        assert "patient_body" in scene.objects

    def test_operating_table(self):
        """Test operating table functionality."""
        table = OperatingTable()

        initial_height = table.get_surface_height()
        table.set_height(1.0)
        new_height = table.get_surface_height()

        assert new_height != initial_height

    def test_add_tool(self):
        """Test adding surgical tool."""
        scene = SurgicalScene()
        scene.add_tool(
            "scalpel",
            "cutting",
            position=np.array([0, 0, 1]),
            properties={"length": 0.1}
        )

        assert "scalpel" in scene.tools


class TestTissueSimulation:
    """Tests for tissue simulation."""

    def test_tissue_properties(self):
        """Test tissue property creation."""
        soft = TissueProperties.soft_tissue()
        assert soft.tissue_type == TissueType.SOFT_TISSUE
        assert soft.youngs_modulus > 0

        bone = TissueProperties.bone()
        assert bone.tissue_type == TissueType.BONE
        assert bone.youngs_modulus > soft.youngs_modulus

    def test_deformable_tissue(self):
        """Test deformable tissue creation."""
        tissue = DeformableTissue(
            name="test_tissue",
            properties=TissueProperties.soft_tissue(),
            position=np.array([0, 0, 0]),
            vertices=np.random.rand(10, 3),
            triangles=np.zeros((0, 3), dtype=np.int32)
        )
        tissue.initialize()

        assert tissue.rest_vertices is not None
        assert tissue.velocities is not None

    def test_tissue_tool_interaction(self):
        """Test tool-tissue interaction."""
        tissue = DeformableTissue(
            name="test_tissue",
            properties=TissueProperties.soft_tissue(),
            position=np.array([0, 0, 0]),
            vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
            triangles=np.array([[0, 1, 2]], dtype=np.int32)
        )
        tissue.initialize()

        reaction = tissue.apply_tool_interaction(
            tool_position=np.array([0.05, 0.05, 0]),
            tool_force=np.array([0, 0, 1]),
            tool_radius=0.1
        )

        # Should get some reaction force
        assert reaction is not None

    def test_cuttable_tissue(self):
        """Test cuttable tissue."""
        tissue = CuttableTissue(
            name="cuttable",
            properties=TissueProperties.soft_tissue(),
            position=np.array([0, 0, 0]),
            vertices=np.random.rand(10, 3),
            triangles=np.zeros((0, 3), dtype=np.int32)
        )
        tissue.initialize()

        # Apply cut
        cut_made = tissue.apply_cut(
            cut_start=np.array([0, 0, 0]),
            cut_end=np.array([0.1, 0, 0]),
            cut_force=10.0  # Above threshold
        )

        assert cut_made
        assert tissue.is_cut
        assert len(tissue.cuts) == 1


class TestRobotSimulator:
    """Tests for robot simulator."""

    def test_robot_creation(self):
        """Test robot simulator creation."""
        robot = RobotSimulator(name="test_robot", num_joints=7)

        assert robot.name == "test_robot"
        assert robot.num_joints == 7
        assert len(robot.joints) == 7

    def test_forward_kinematics(self):
        """Test forward kinematics."""
        robot = RobotSimulator(num_joints=7)

        # Set all joints to zero
        joint_pos = np.zeros(7)
        ee_pose = robot.forward_kinematics(joint_pos)

        assert ee_pose is not None
        assert len(ee_pose.position) == 3
        assert len(ee_pose.orientation) == 4

    def test_jacobian(self):
        """Test Jacobian computation."""
        robot = RobotSimulator(num_joints=7)

        J = robot.get_jacobian()

        assert J.shape == (6, 7)  # 6 DOF Cartesian, 7 joints

    def test_inverse_kinematics(self):
        """Test inverse kinematics."""
        robot = RobotSimulator(num_joints=7)

        # Get current EE pose
        current_pose = robot.get_end_effector_pose()

        # Small perturbation
        target_pose = current_pose
        target_pose.position += np.array([0.01, 0.01, 0])

        # Solve IK
        joint_positions, success = robot.inverse_kinematics(target_pose)

        # Should find a solution for small perturbation
        assert success
        assert len(joint_positions) == 7

    def test_position_control(self):
        """Test position control."""
        robot = RobotSimulator(num_joints=7)
        robot.control_mode = "position"

        target = np.array([0.1, -0.1, 0.2, -0.5, 0.1, 0.3, -0.1])
        robot.set_joint_positions(target)

        # Step several times
        for _ in range(100):
            robot.step(0.01)

        # Should converge toward target
        current = robot.get_joint_positions()
        error = np.linalg.norm(current - target)
        assert error < 0.5  # Should be close

    def test_joint_limits(self):
        """Test joint limit enforcement."""
        robot = RobotSimulator(num_joints=7)

        # Try to set position beyond limits
        beyond_limits = np.array([10, 10, 10, 10, 10, 10, 10])
        robot.set_joint_positions(beyond_limits)
        robot.control_mode = "position"

        for _ in range(100):
            robot.step(0.01)

        # Should be clamped to limits
        current = robot.get_joint_positions()
        for i, joint in enumerate(robot.joints.values()):
            assert current[i] <= joint.position_max
            assert current[i] >= joint.position_min


class TestTissueSimulator:
    """Tests for tissue simulator manager."""

    def test_create_box_tissue(self):
        """Test creating box-shaped tissue."""
        sim = TissueSimulator()

        tissue = sim.create_box_tissue(
            name="test_box",
            center=np.array([0, 0, 1]),
            size=np.array([0.1, 0.1, 0.1]),
            properties=TissueProperties.soft_tissue(),
            resolution=3
        )

        assert tissue is not None
        assert "test_box" in sim.tissues

    def test_create_organ(self):
        """Test creating organ model."""
        sim = TissueSimulator()

        liver = sim.create_organ(
            name="liver",
            organ_type="liver",
            center=np.array([0, 0, 1]),
            scale=1.0
        )

        assert liver is not None
        assert "liver" in sim.cuttable_tissues

    def test_simulator_step(self):
        """Test tissue simulator step."""
        sim = TissueSimulator()
        sim.create_organ("test_organ", "liver", np.array([0, 0, 1]))

        stats = sim.step(0.01)

        assert "blood_loss" in stats
        assert "total_blood_loss" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
