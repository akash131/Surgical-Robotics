"""Tests for Orthopedic robot system."""

import numpy as np
import pytest

from surgical_robotics.orthopedic.robot import (
    OrthopedicRobot,
    OrthopedicArm,
    CuttingMode,
    BoneRegistration,
    CuttingBoundary,
)
from surgical_robotics.orthopedic.cutting import (
    MillingTool,
    SawBlade,
    CuttingPath,
    BoneCuttingSystem,
    ToolType,
    AdaptiveCuttingController,
)
from surgical_robotics.orthopedic.force_feedback import (
    TissueForceSensor,
    TissueClassifier,
    ForceController,
    TissueType,
    ForceReading,
)
from surgical_robotics.orthopedic.planning import (
    PatientModel,
    ImplantModel,
    BoneCut,
    SurgicalPlan,
    BoneCutPlanner,
    BoneType,
    ImplantType,
)
from surgical_robotics.core.base import RobotState


class TestOrthopedicRobot:
    """Tests for orthopedic robot system."""

    def test_initialization(self):
        """Test robot initialization."""
        robot = OrthopedicRobot()

        assert robot.name == "OrthopedicRobot"
        assert robot.cutting_mode == CuttingMode.IDLE
        assert len(robot.arms) == 1

    def test_initialize_calibrate(self):
        """Test initialization and calibration."""
        robot = OrthopedicRobot()

        assert robot.initialize()
        assert robot.state == RobotState.READY

        assert robot.calibrate()
        assert robot.state == RobotState.READY

    def test_cutting_mode_change(self):
        """Test cutting mode changes."""
        robot = OrthopedicRobot()
        robot.initialize()

        assert robot.set_cutting_mode(CuttingMode.BURR)
        assert robot.cutting_mode == CuttingMode.BURR
        assert robot.spindle_speed > 0

    def test_spindle_control(self):
        """Test spindle start/stop."""
        robot = OrthopedicRobot()
        robot.initialize()
        robot.set_cutting_mode(CuttingMode.BURR)

        assert robot.start_spindle()
        assert robot.spindle_running

        robot.stop_spindle()
        assert not robot.spindle_running

    def test_bone_registration(self):
        """Test bone registration."""
        robot = OrthopedicRobot()

        ct_landmarks = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
        ]
        # Same points (identity registration)
        robot_landmarks = ct_landmarks.copy()

        result = robot.register_bone("femur", ct_landmarks, robot_landmarks)

        assert result
        assert "femur" in robot.bone_registrations


class TestCuttingBoundary:
    """Tests for cutting boundaries."""

    def test_boundary_check(self):
        """Test boundary containment check."""
        boundary = CuttingBoundary(
            name="test_boundary",
            boundary_points=[
                np.array([0, 0, 0]),
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                np.array([0, 1, 0]),
            ],
            normal_direction=np.array([0, 0, 1]),
            max_depth=0.01
        )

        # Point within boundary
        assert boundary.is_within_boundary(np.array([0.5, 0.5, 0]))


class TestCuttingTools:
    """Tests for cutting tools."""

    def test_milling_tool(self):
        """Test milling tool properties."""
        tool = MillingTool(
            name="test_burr",
            tool_type=ToolType.SPHERICAL_BURR,
            diameter=0.003,
            length=0.050
        )

        assert tool.radius == 0.0015
        assert tool.num_flutes == 4

    def test_material_removal_rate(self):
        """Test MRR calculation."""
        tool = MillingTool(
            name="test_burr",
            tool_type=ToolType.SPHERICAL_BURR,
            diameter=0.003,
            length=0.050
        )

        mrr = tool.compute_material_removal_rate(
            feed_rate=0.005,  # 5mm/s
            depth_of_cut=0.001  # 1mm
        )

        expected_mrr = 0.005 * 0.001 * 0.003
        assert mrr == pytest.approx(expected_mrr)


class TestCuttingPath:
    """Tests for cutting paths."""

    def test_waypoint_addition(self):
        """Test waypoint addition."""
        path = CuttingPath(name="test_path")

        path.add_waypoint(np.array([0, 0, 0]))
        path.add_waypoint(np.array([1, 0, 0]))
        path.add_waypoint(np.array([2, 0, 0]))

        assert len(path.waypoints) == 3
        assert path.total_length == pytest.approx(2.0)

    def test_path_interpolation(self):
        """Test path interpolation."""
        path = CuttingPath(name="test_path")

        path.add_waypoint(np.array([0, 0, 0]))
        path.add_waypoint(np.array([1, 0, 0]))

        interpolated = path.interpolate(11)

        assert len(interpolated) == 11
        np.testing.assert_array_almost_equal(interpolated[0][0], [0, 0, 0])
        np.testing.assert_array_almost_equal(interpolated[-1][0], [1, 0, 0])


class TestBoneCuttingSystem:
    """Tests for bone cutting system."""

    def test_planar_cut_generation(self):
        """Test planar cut path generation."""
        system = BoneCuttingSystem()

        tool = MillingTool(
            name="test_burr",
            tool_type=ToolType.SPHERICAL_BURR,
            diameter=0.003,
            length=0.050
        )
        system.load_tool(tool)

        paths = system.generate_planar_cut_paths(
            plane_origin=np.array([0, 0, 0]),
            plane_normal=np.array([0, 0, 1]),
            width=0.020,
            height=0.020,
            depth=0.005
        )

        assert len(paths) > 0

    def test_path_validation(self):
        """Test path validation."""
        system = BoneCuttingSystem()

        tool = MillingTool(
            name="test_burr",
            tool_type=ToolType.SPHERICAL_BURR,
            diameter=0.003,
            length=0.050
        )
        system.load_tool(tool)

        path = CuttingPath(name="test_path")
        path.add_waypoint(np.array([0, 0, 0]), feed_rate=0.005)
        path.add_waypoint(np.array([0.01, 0, 0]), feed_rate=0.005)

        valid, issues = system.validate_path(path)

        assert valid
        assert len(issues) == 0


class TestAdaptiveCuttingController:
    """Tests for adaptive cutting control."""

    def test_feed_rate_update(self):
        """Test feed rate adjustment based on force."""
        controller = AdaptiveCuttingController()
        controller.target_force = 10.0

        # Force below target - should increase feed
        initial_feed = controller.current_feed_rate
        new_feed = controller.update(5.0, dt=0.01)

        assert new_feed >= initial_feed

    def test_controller_reset(self):
        """Test controller reset."""
        controller = AdaptiveCuttingController()

        controller.integral_error = 10.0
        controller.reset()

        assert controller.integral_error == 0.0


class TestTissueForceSensor:
    """Tests for force sensing."""

    def test_force_update(self):
        """Test force sensor update."""
        sensor = TissueForceSensor(sample_rate=1000.0)

        reading = sensor.update(
            raw_forces=np.array([1, 2, 3]),
            raw_torques=np.array([0.1, 0.2, 0.3]),
            timestamp=0.001
        )

        assert reading.force_magnitude > 0
        assert reading.torque_magnitude > 0

    def test_average_force(self):
        """Test force averaging."""
        sensor = TissueForceSensor(sample_rate=1000.0)

        # Add multiple readings
        for i in range(100):
            sensor.update(
                raw_forces=np.array([1, 0, 0]),
                raw_torques=np.array([0, 0, 0]),
                timestamp=i * 0.001
            )

        avg = sensor.get_average_force(window_ms=50)
        np.testing.assert_array_almost_equal(avg, [1, 0, 0], decimal=1)


class TestTissueClassifier:
    """Tests for tissue classification."""

    def test_stiffness_estimation(self):
        """Test stiffness estimation."""
        classifier = TissueClassifier()

        stiffness = classifier.estimate_stiffness(
            force_delta=np.array([10, 0, 0]),  # 10N force change
            position_delta=np.array([0.001, 0, 0])  # 1mm displacement
        )

        assert stiffness == pytest.approx(10000)  # 10N/1mm = 10000 N/m


class TestForceController:
    """Tests for force control."""

    def test_force_limit_check(self):
        """Test force limit checking."""
        controller = ForceController()
        controller.max_force = 100.0

        reading = ForceReading(
            forces=np.array([50, 0, 0]),
            torques=np.array([0, 0, 0]),
            timestamp=0.0
        )

        safe, message = controller.check_force_limits(reading, TissueType.CORTICAL_BONE)

        assert safe
        assert "within limits" in message.lower()

    def test_compliance_velocity(self):
        """Test compliance velocity computation."""
        controller = ForceController()
        controller.compliance_stiffness = 1000.0

        reading = ForceReading(
            forces=np.array([10, 0, 0]),
            torques=np.array([0, 0, 0]),
            timestamp=0.0
        )

        velocity = controller.compute_compliance_velocity(reading, target_force=0.0)

        # Velocity should be in direction of force
        assert velocity[0] > 0


class TestPatientModel:
    """Tests for patient model."""

    def test_landmark_addition(self):
        """Test landmark addition."""
        model = PatientModel(
            patient_id="test_patient",
            bone_type=BoneType.FEMUR
        )

        model.add_landmark("test_point", np.array([1, 2, 3]))

        assert "test_point" in model.landmarks
        np.testing.assert_array_equal(model.landmarks["test_point"], [1, 2, 3])

    def test_femur_axes_computation(self):
        """Test femoral axis computation."""
        model = PatientModel(
            patient_id="test_patient",
            bone_type=BoneType.FEMUR
        )

        model.add_landmark("femoral_head_center", np.array([0, 0, 0.4]))
        model.add_landmark("medial_epicondyle", np.array([-0.03, 0, 0]))
        model.add_landmark("lateral_epicondyle", np.array([0.03, 0, 0]))

        result = model.compute_axes()

        assert result
        assert model.mechanical_axis is not None
        assert model.transverse_axis is not None


class TestBoneCut:
    """Tests for bone cut definition."""

    def test_cut_corners(self):
        """Test cut corner computation."""
        cut = BoneCut(
            name="test_cut",
            plane_point=np.array([0, 0, 0]),
            plane_normal=np.array([0, 0, 1]),
            depth=0.01,
            width=0.02,
            height=0.02
        )

        corners = cut.get_corners()

        assert len(corners) == 4
        # Corners should be at ±half_width, ±half_height
        for corner in corners:
            assert abs(corner[0]) == pytest.approx(0.01) or abs(corner[1]) == pytest.approx(0.01)


class TestSurgicalPlan:
    """Tests for surgical planning."""

    def test_plan_validation(self):
        """Test plan validation."""
        model = PatientModel(
            patient_id="test",
            bone_type=BoneType.FEMUR
        )
        model.add_landmark("test", np.array([0, 0, 0]))

        plan = SurgicalPlan(
            plan_id="test_plan",
            patient_model=model
        )

        # Empty plan should fail validation
        valid, issues = plan.validate()

        assert not valid
        assert len(issues) > 0


class TestBoneCutPlanner:
    """Tests for bone cut planning."""

    def test_tka_femoral_cuts(self):
        """Test TKA femoral cut planning."""
        planner = BoneCutPlanner()

        model = PatientModel(
            patient_id="test",
            bone_type=BoneType.FEMUR
        )
        model.add_landmark("knee_center", np.array([0, 0, 0]))
        model.mechanical_axis = np.array([0, 0, 1])

        planner.set_patient_model(model)

        cuts = planner.plan_tka_femoral_cuts()

        assert len(cuts) == 5  # 5 standard cuts
        cut_names = [c.name for c in cuts]
        assert "distal_femoral" in cut_names
        assert "anterior_femoral" in cut_names
        assert "posterior_femoral" in cut_names

    def test_tha_acetabular_reaming(self):
        """Test THA acetabular reaming planning."""
        planner = BoneCutPlanner()

        model = PatientModel(
            patient_id="test",
            bone_type=BoneType.PELVIS
        )
        model.add_landmark("acetabulum_center", np.array([0, 0, 0]))

        planner.set_patient_model(model)

        params = planner.plan_tha_acetabular_reaming(
            target_cup_size=0.054,
            anteversion=np.radians(20),
            inclination=np.radians(45)
        )

        assert "center" in params
        assert "axis" in params
        assert "reaming_sequence" in params
        assert len(params["reaming_sequence"]) > 0
