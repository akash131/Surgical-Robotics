"""Tests for Da Vinci surgical robot system."""

import numpy as np
import pytest

from surgical_robotics.davinci.robot import (
    DaVinciRobot,
    DaVinciArm,
    MasterConsole,
    ArmRole,
)
from surgical_robotics.davinci.haptics import (
    HapticDevice,
    HapticFeedbackController,
    ForceReading,
    TissueInteractionModel,
)
from surgical_robotics.davinci.vision import (
    StereoscopicVision,
    EndoscopeController,
    CameraParameters,
    StereoParameters,
)
from surgical_robotics.davinci.instruments import (
    SurgicalInstrument,
    InstrumentType,
    InstrumentTracker,
    INSTRUMENT_CATALOG,
)
from surgical_robotics.core.base import Pose, RobotState


class TestDaVinciRobot:
    """Tests for Da Vinci robot system."""

    def test_initialization(self):
        """Test robot initialization."""
        robot = DaVinciRobot(num_patient_arms=3, has_endoscope=True)

        assert robot.name == "DaVinci"
        assert robot.num_patient_arms == 3
        assert robot.has_endoscope
        assert len(robot.arms) == 4  # 3 patient + 1 endoscope

    def test_arm_roles(self):
        """Test arm role assignment."""
        robot = DaVinciRobot(num_patient_arms=3, has_endoscope=True)

        assert robot.arms[0].role == ArmRole.PATIENT_SIDE_LEFT
        assert robot.arms[1].role == ArmRole.PATIENT_SIDE_RIGHT
        assert robot.arms[2].role == ArmRole.PATIENT_SIDE_AUX
        assert robot.arms[3].role == ArmRole.ENDOSCOPE

    def test_initialize(self):
        """Test system initialization."""
        robot = DaVinciRobot()
        result = robot.initialize()

        assert result
        assert robot.state == RobotState.READY

    def test_calibrate(self):
        """Test calibration."""
        robot = DaVinciRobot()
        robot.initialize()
        result = robot.calibrate()

        assert result
        assert robot.state == RobotState.READY

    def test_teleoperation_start_stop(self):
        """Test teleoperation mode."""
        robot = DaVinciRobot()
        robot.initialize()

        assert robot.start_teleoperation()
        assert robot.teleoperation_active
        assert robot.state == RobotState.OPERATING

        robot.stop_teleoperation()
        assert not robot.teleoperation_active
        assert robot.state == RobotState.READY

    def test_rcm_setting(self):
        """Test remote center of motion setting."""
        robot = DaVinciRobot()
        rcm_pos = np.array([0.1, 0.2, 0.3])

        robot.set_rcm(0, rcm_pos)

        arm = robot.arms[0]
        assert isinstance(arm, DaVinciArm)
        np.testing.assert_array_equal(arm.remote_center_of_motion, rcm_pos)

    def test_arm_swap(self):
        """Test controller-to-arm mapping swap."""
        robot = DaVinciRobot()

        original_mapping = robot.arm_to_controller_mapping.copy()
        robot.swap_arms(0, 1)

        assert robot.arm_to_controller_mapping[0] == original_mapping[1]
        assert robot.arm_to_controller_mapping[1] == original_mapping[0]


class TestMasterConsole:
    """Tests for master console."""

    def test_motion_scaling(self):
        """Test motion scaling."""
        console = MasterConsole()
        console.scaling_factor = 0.5

        prev_pose = Pose.identity()
        current_pose = Pose(
            position=np.array([0.1, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        )

        motion = console.get_scaled_motion(current_pose, prev_pose)
        np.testing.assert_array_almost_equal(motion, [0.05, 0, 0])

    def test_clutch_engaged(self):
        """Test clutch engagement blocks motion."""
        console = MasterConsole()
        console.clutch_engaged = True

        motion = console.get_scaled_motion(
            Pose(position=np.array([0.1, 0, 0]), orientation=np.array([1, 0, 0, 0])),
            Pose.identity()
        )

        np.testing.assert_array_equal(motion, [0, 0, 0])

    def test_scaling_limits(self):
        """Test scaling factor limits."""
        console = MasterConsole()

        console.set_scaling(0.1)  # Below minimum
        assert console.scaling_factor == 0.2

        console.set_scaling(1.5)  # Above maximum
        assert console.scaling_factor == 1.0


class TestHapticFeedback:
    """Tests for haptic feedback system."""

    def test_haptic_device(self):
        """Test haptic device operation."""
        device = HapticDevice("test_device", max_force=5.0)

        device.enable()
        assert device.is_enabled

        device.set_force(np.array([2, 0, 0]))
        assert device.state.force_output[0] > 0

        device.disable()
        assert not device.is_enabled
        np.testing.assert_array_equal(device.state.force_output, [0, 0, 0])

    def test_force_saturation(self):
        """Test force saturation to max."""
        device = HapticDevice("test_device", max_force=5.0)
        device.enable()

        device.set_force(np.array([10, 0, 0]))

        # Should be saturated to max_force
        force_magnitude = np.linalg.norm(device.state.force_output)
        assert force_magnitude <= 5.0

    def test_force_reading(self):
        """Test force reading properties."""
        reading = ForceReading(
            forces=np.array([3, 4, 0]),
            torques=np.array([1, 0, 0]),
            timestamp=0.0
        )

        assert reading.force_magnitude == pytest.approx(5.0)
        assert reading.torque_magnitude == pytest.approx(1.0)


class TestTissueInteraction:
    """Tests for tissue interaction model."""

    def test_no_contact(self):
        """Test no force when not in contact."""
        model = TissueInteractionModel()

        force = model.compute_contact_force(
            penetration_depth=-0.001,  # Not in contact
            velocity=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1])
        )

        np.testing.assert_array_equal(force, [0, 0, 0])

    def test_contact_force(self):
        """Test force computation in contact."""
        model = TissueInteractionModel(stiffness=1000)

        force = model.compute_contact_force(
            penetration_depth=0.001,  # 1mm penetration
            velocity=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1])
        )

        # Should produce force in normal direction
        assert force[2] > 0  # Positive Z force


class TestStereoscopicVision:
    """Tests for stereoscopic vision system."""

    def test_pixel_to_3d(self):
        """Test pixel to 3D conversion."""
        vision = StereoscopicVision()

        point_3d = vision.pixel_to_3d((960, 540), depth=0.1)

        # Center pixel should project to roughly (0, 0, depth)
        assert point_3d[2] == pytest.approx(0.1)

    def test_project_3d_to_pixel(self):
        """Test 3D to pixel projection."""
        vision = StereoscopicVision()

        # Point at center of image
        pixel = vision.project_3d_to_pixel(np.array([0, 0, 0.1]))

        # Should be near principal point
        assert abs(pixel[0] - 960) < 10
        assert abs(pixel[1] - 540) < 10


class TestEndoscopeController:
    """Tests for endoscope controller."""

    def test_zoom_controls(self):
        """Test zoom in/out."""
        controller = EndoscopeController()

        initial_zoom = controller.zoom_level
        controller.zoom_in()
        assert controller.zoom_level > initial_zoom

        controller.zoom_out()
        assert controller.zoom_level == pytest.approx(initial_zoom)

    def test_lock_unlock(self):
        """Test position locking."""
        controller = EndoscopeController()

        controller.lock()
        assert controller.is_locked

        # Should not be able to move when locked
        result = controller.move_to(Pose.identity())
        assert not result

        controller.unlock()
        assert not controller.is_locked


class TestInstruments:
    """Tests for surgical instruments."""

    def test_instrument_catalog(self):
        """Test instrument catalog contains expected instruments."""
        assert InstrumentType.MARYLAND_DISSECTOR in INSTRUMENT_CATALOG
        assert InstrumentType.LARGE_NEEDLE_DRIVER in INSTRUMENT_CATALOG

    def test_instrument_energy(self):
        """Test instrument energy activation."""
        spec = INSTRUMENT_CATALOG[InstrumentType.MARYLAND_DISSECTOR]
        instrument = SurgicalInstrument(
            name="test_instrument",
            tool_type="dissector",
            specification=spec
        )

        result = instrument.activate_energy(0.5)
        assert result
        assert instrument.energy_active
        assert instrument.energy_power == pytest.approx(spec.max_power * 0.5)

        instrument.deactivate_energy()
        assert not instrument.energy_active
        assert instrument.energy_power == 0

    def test_instrument_tracker(self):
        """Test instrument collision tracking."""
        tracker = InstrumentTracker()

        spec = INSTRUMENT_CATALOG[InstrumentType.PROGRASP_FORCEPS]
        instrument1 = SurgicalInstrument(
            name="instrument1",
            tool_type="forceps",
            specification=spec
        )
        instrument2 = SurgicalInstrument(
            name="instrument2",
            tool_type="forceps",
            specification=spec
        )

        tracker.register_instrument(0, instrument1)
        tracker.register_instrument(1, instrument2)

        # Update positions far apart - no collision
        tracker.update_instrument_pose(0, Pose(
            position=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        ))
        tracker.update_instrument_pose(1, Pose(
            position=np.array([0.5, 0, 0]),
            orientation=np.array([1, 0, 0, 0])
        ))

        collisions = tracker.check_instrument_collisions()
        assert len(collisions) == 0
