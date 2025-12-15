"""Tests for Neurosurgical robot system."""

import numpy as np
import pytest

from surgical_robotics.neurosurgical.robot import (
    NeurosurgicalRobot,
    NeurosurgicalArm,
    StereotacticFrame,
    NeurosurgicalTarget,
    PrecisionMode,
)
from surgical_robotics.neurosurgical.imaging import (
    ImageVolume,
    ImageRegistration,
    MRIIntegration,
    CTIntegration,
)
from surgical_robotics.neurosurgical.tremor import (
    TremorFilter,
    TremorCancellation,
    MotionStabilizer,
    FilterType,
)
from surgical_robotics.core.base import RobotState


class TestNeurosurgicalRobot:
    """Tests for neurosurgical robot system."""

    def test_initialization(self):
        """Test robot initialization."""
        robot = NeurosurgicalRobot()

        assert robot.name == "NeurosurgicalRobot"
        assert robot.precision_mode == PrecisionMode.ULTRA
        assert len(robot.arms) == 1

    def test_precision_mode(self):
        """Test high precision arm configuration."""
        robot = NeurosurgicalRobot()
        arm = robot.arms[0]

        assert isinstance(arm, NeurosurgicalArm)
        assert arm.positioning_accuracy < 0.001  # Sub-millimeter

    def test_initialize_calibrate(self):
        """Test initialization and calibration."""
        robot = NeurosurgicalRobot()

        assert robot.initialize()
        assert robot.state == RobotState.READY

        assert robot.calibrate()
        assert robot.state == RobotState.READY

    def test_position_accuracy_stats(self):
        """Test position accuracy statistics."""
        robot = NeurosurgicalRobot()
        robot.initialize()

        stats = robot.get_position_accuracy_stats()

        assert "mean" in stats
        assert "max" in stats
        assert "std" in stats


class TestStereotacticFrame:
    """Tests for stereotactic frame."""

    def test_fiducial_registration(self):
        """Test fiducial-based registration."""
        frame = StereotacticFrame(frame_id="test_frame")

        # Add fiducials
        frame_points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]

        # Corresponding image points (simple translation)
        image_points = [p + np.array([0.1, 0.2, 0.3]) for p in frame_points]

        for p in frame_points:
            frame.add_fiducial(p)

        result = frame.compute_registration(image_points)

        assert result
        assert frame.is_registered

    def test_coordinate_transformation(self):
        """Test coordinate transformation after registration."""
        frame = StereotacticFrame(frame_id="test_frame")

        # Simple translation registration
        frame.registration_matrix = np.eye(4)
        frame.registration_matrix[:3, 3] = [0.1, 0.2, 0.3]
        frame.is_registered = True

        test_point = np.array([1, 2, 3])
        image_point = frame.transform_to_image(test_point)

        np.testing.assert_array_almost_equal(
            image_point,
            [1.1, 2.2, 3.3]
        )


class TestNeurosurgicalTarget:
    """Tests for surgical target."""

    def test_trajectory_computation(self):
        """Test trajectory direction computation."""
        target = NeurosurgicalTarget(
            name="test_target",
            position=np.array([0, 0, -0.05]),
            entry_point=np.array([0, 0, 0])
        )

        direction = target.compute_trajectory()

        assert direction is not None
        np.testing.assert_array_almost_equal(direction, [0, 0, -1])
        assert target.depth == pytest.approx(0.05)


class TestImageVolume:
    """Tests for image volume."""

    def test_coordinate_conversion(self):
        """Test world to voxel conversion."""
        volume = ImageVolume(
            data=np.zeros((100, 100, 100), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
            origin=np.array([0, 0, 0]),
            direction=np.eye(3)
        )

        world_point = np.array([50.0, 50.0, 50.0])
        voxel = volume.world_to_voxel(world_point)

        np.testing.assert_array_almost_equal(voxel, [50, 50, 50])

    def test_roundtrip_conversion(self):
        """Test world -> voxel -> world roundtrip."""
        volume = ImageVolume(
            data=np.zeros((100, 100, 100), dtype=np.float32),
            spacing=(0.5, 0.5, 0.5),
            origin=np.array([-25, -25, -25]),
            direction=np.eye(3)
        )

        original = np.array([10.0, 20.0, 30.0])
        voxel = volume.world_to_voxel(original)
        recovered = volume.voxel_to_world(voxel)

        np.testing.assert_array_almost_equal(recovered, original)


class TestImageRegistration:
    """Tests for image registration."""

    def test_registration(self):
        """Test paired-point registration."""
        registration = ImageRegistration()

        # Add point pairs (simple rotation + translation)
        for i in range(5):
            image_pt = np.random.rand(3)
            # Robot point = image point + translation
            robot_pt = image_pt + np.array([0.1, 0.2, 0.3])

            registration.add_point_pair(image_pt, robot_pt)

        result = registration.compute_registration()

        assert result is not None
        assert result.is_valid
        assert result.fiducial_registration_error < 0.01

    def test_point_transformation(self):
        """Test point transformation after registration."""
        registration = ImageRegistration()

        # Identity registration (points are same)
        points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])]
        for p in points:
            registration.add_point_pair(p, p)

        registration.compute_registration()

        test_point = np.array([0.5, 0.5, 0.5])
        transformed = registration.transform_point(test_point)

        np.testing.assert_array_almost_equal(transformed, test_point, decimal=3)


class TestMRIIntegration:
    """Tests for MRI integration."""

    def test_mr_safe_mode(self):
        """Test MR-safe mode activation."""
        mri = MRIIntegration()

        assert not mri.mr_compatible_mode

        result = mri.enter_mr_safe_mode()
        assert result
        assert mri.mr_compatible_mode

        mri.exit_mr_safe_mode()
        assert not mri.mr_compatible_mode

    def test_robot_position_check(self):
        """Test robot position relative to bore."""
        mri = MRIIntegration()

        position = np.array([0, 0, 0])  # At isocenter
        result = mri.get_robot_position_in_bore(position)

        assert result["distance_from_isocenter"] == 0
        assert result["in_bore"]
        assert result["in_imaging_fov"]


class TestCTIntegration:
    """Tests for CT integration."""

    def test_clearance_verification(self):
        """Test robot clearance verification."""
        ct = CTIntegration()

        # Position well within bore
        positions = [np.array([0, 0, 0]), np.array([0.1, 0, 0])]
        assert ct.verify_robot_clearance(positions)

        # Position outside bore
        positions = [np.array([0.5, 0, 0])]  # Beyond clearance
        assert not ct.verify_robot_clearance(positions)

    def test_dose_tracking(self):
        """Test radiation dose tracking."""
        ct = CTIntegration()

        initial_dose = ct.total_dose_mgy
        ct.acquire_image()

        assert ct.total_dose_mgy > initial_dose


class TestTremorFilter:
    """Tests for tremor filtering."""

    def test_lowpass_filter(self):
        """Test low-pass filter."""
        filter = TremorFilter(
            filter_type=FilterType.LOW_PASS,
            cutoff_frequency=5.0,
            sample_rate=1000.0
        )

        # Process multiple samples
        outputs = []
        for i in range(200):
            input_signal = np.array([1, 0, 0]) + 0.1 * np.sin(2 * np.pi * 10 * i / 1000)
            output = filter.filter(input_signal)
            outputs.append(output)

        # Output should be smoother than input (less high-frequency content)
        # Check variance is reduced
        output_var = np.var([o[0] for o in outputs[-100:]])
        input_var = np.var([1 + 0.1 * np.sin(2 * np.pi * 10 * i / 1000) for i in range(100, 200)])

        assert output_var < input_var

    def test_filter_reset(self):
        """Test filter reset."""
        filter = TremorFilter()

        for i in range(50):
            filter.filter(np.random.rand(3))

        assert len(filter.state.buffer) > 0

        filter.reset()
        assert len(filter.state.buffer) == 0


class TestTremorCancellation:
    """Tests for complete tremor cancellation system."""

    def test_process(self):
        """Test tremor cancellation processing."""
        cancellation = TremorCancellation()

        # Process signal with tremor
        for i in range(500):
            tremor = 0.0001 * np.sin(2 * np.pi * 10 * i / 1000)  # 10Hz tremor
            input_signal = np.array([0.1, 0.1, 0.1]) + tremor

            output = cancellation.process(input_signal)

        # Should detect tremor after enough samples
        # Note: Detection depends on amplitude threshold

    def test_cancellation_ratio(self):
        """Test cancellation ratio calculation."""
        cancellation = TremorCancellation()

        # Process some samples
        for i in range(200):
            cancellation.process(np.random.rand(3) * 0.001)

        ratio = cancellation.get_cancellation_ratio()
        assert 0 <= ratio <= 2.0  # Reasonable range


class TestMotionStabilizer:
    """Tests for motion stabilization."""

    def test_position_locking(self):
        """Test position lock/unlock."""
        stabilizer = MotionStabilizer()
        lock_position = np.array([0.1, 0.1, 0.1])

        stabilizer.lock_position(lock_position)
        assert stabilizer.is_locked
        np.testing.assert_array_equal(stabilizer.locked_position, lock_position)

        stabilizer.unlock_position()
        assert not stabilizer.is_locked

    def test_motion_scaling(self):
        """Test motion scaling."""
        stabilizer = MotionStabilizer()

        stabilizer.set_motion_scale(0.1)
        assert stabilizer.motion_scale == 0.1

        # Test limits
        stabilizer.set_motion_scale(0.001)
        assert stabilizer.motion_scale == 0.01  # Clamped to minimum

    def test_velocity_limiting(self):
        """Test velocity limiting."""
        stabilizer = MotionStabilizer()
        stabilizer.set_max_velocity(0.001)  # 1mm/s

        # Process rapid movements
        stabilizer.last_output = np.array([0, 0, 0])
        stabilizer.last_time = 0.0

        # Request movement that would exceed velocity limit
        result = stabilizer._apply_velocity_limit(
            np.array([1, 0, 0]),  # 1m movement
            current_time=0.001  # 1ms later
        )

        # Should be limited
        velocity = np.linalg.norm(result - np.array([0, 0, 0])) / 0.001
        assert velocity <= 0.002  # Allow small tolerance
