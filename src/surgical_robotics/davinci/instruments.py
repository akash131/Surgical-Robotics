"""Surgical instruments and tracking for Da Vinci systems."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose, EndEffector


class InstrumentType(Enum):
    """Types of surgical instruments."""
    # Grasping instruments
    MARYLAND_DISSECTOR = auto()
    PROGRASP_FORCEPS = auto()
    CADIERE_FORCEPS = auto()
    FENESTRATED_BIPOLAR = auto()

    # Cutting instruments
    CURVED_SCISSORS = auto()
    ROUND_TIP_SCISSORS = auto()
    MONOPOLAR_SCISSORS = auto()

    # Needle drivers
    LARGE_NEEDLE_DRIVER = auto()
    MEGA_NEEDLE_DRIVER = auto()
    BLACK_DIAMOND_NEEDLE_DRIVER = auto()

    # Specialized
    CLIP_APPLIER = auto()
    STAPLER = auto()
    VESSEL_SEALER = auto()
    SUCTION_IRRIGATOR = auto()

    # Energy instruments
    MONOPOLAR_CAUTERY = auto()
    BIPOLAR_CAUTERY = auto()
    HARMONIC_SCALPEL = auto()


@dataclass
class InstrumentSpecification:
    """Technical specifications for a surgical instrument."""
    instrument_type: InstrumentType
    name: str
    shaft_length: float  # meters
    shaft_diameter: float  # meters
    tip_width: float  # meters

    # Articulation capabilities
    wrist_pitch_range: tuple[float, float] = (-90, 90)  # degrees
    wrist_yaw_range: tuple[float, float] = (-90, 90)
    grip_range: tuple[float, float] = (0, 60)  # degrees

    # Force limits
    max_grip_force: float = 10.0  # Newtons
    max_tip_force: float = 5.0

    # Energy capabilities
    has_energy: bool = False
    energy_type: Optional[str] = None
    max_power: float = 0.0  # Watts


# Catalog of standard Da Vinci instruments
INSTRUMENT_CATALOG: dict[InstrumentType, InstrumentSpecification] = {
    InstrumentType.MARYLAND_DISSECTOR: InstrumentSpecification(
        instrument_type=InstrumentType.MARYLAND_DISSECTOR,
        name="Maryland Bipolar Dissector",
        shaft_length=0.47,
        shaft_diameter=0.008,
        tip_width=0.012,
        has_energy=True,
        energy_type="bipolar",
        max_power=40.0
    ),
    InstrumentType.PROGRASP_FORCEPS: InstrumentSpecification(
        instrument_type=InstrumentType.PROGRASP_FORCEPS,
        name="ProGrasp Forceps",
        shaft_length=0.47,
        shaft_diameter=0.008,
        tip_width=0.015,
        max_grip_force=15.0
    ),
    InstrumentType.CURVED_SCISSORS: InstrumentSpecification(
        instrument_type=InstrumentType.CURVED_SCISSORS,
        name="Curved Scissors",
        shaft_length=0.47,
        shaft_diameter=0.008,
        tip_width=0.010,
        has_energy=True,
        energy_type="monopolar",
        max_power=35.0
    ),
    InstrumentType.LARGE_NEEDLE_DRIVER: InstrumentSpecification(
        instrument_type=InstrumentType.LARGE_NEEDLE_DRIVER,
        name="Large Needle Driver",
        shaft_length=0.47,
        shaft_diameter=0.008,
        tip_width=0.008,
        max_grip_force=20.0,
        grip_range=(0, 45)
    ),
    InstrumentType.VESSEL_SEALER: InstrumentSpecification(
        instrument_type=InstrumentType.VESSEL_SEALER,
        name="Vessel Sealer Extend",
        shaft_length=0.47,
        shaft_diameter=0.010,
        tip_width=0.020,
        has_energy=True,
        energy_type="bipolar_advanced",
        max_power=80.0,
        max_grip_force=25.0
    ),
}


@dataclass
class SurgicalInstrument(EndEffector):
    """Active surgical instrument with full state tracking."""
    specification: Optional[InstrumentSpecification] = None
    wrist_pitch: float = 0.0  # degrees
    wrist_yaw: float = 0.0
    jaw_angle: float = 0.0  # 0 = closed

    # Energy state
    energy_active: bool = False
    energy_power: float = 0.0

    # Sensor readings
    measured_grip_force: float = 0.0
    measured_tip_force: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )

    # Usage tracking
    activation_count: int = 0
    total_energy_delivered: float = 0.0  # Joules

    def activate_energy(self, power_fraction: float = 1.0) -> bool:
        """Activate energy delivery (cautery, sealing, etc.)."""
        if self.specification is None or not self.specification.has_energy:
            return False

        self.energy_active = True
        self.energy_power = self.specification.max_power * np.clip(power_fraction, 0, 1)
        self.activation_count += 1
        return True

    def deactivate_energy(self) -> None:
        """Deactivate energy delivery."""
        self.energy_active = False
        self.energy_power = 0.0

    def set_wrist_angles(self, pitch: float, yaw: float) -> bool:
        """Set wrist articulation angles."""
        if self.specification is None:
            return False

        pitch_min, pitch_max = self.specification.wrist_pitch_range
        yaw_min, yaw_max = self.specification.wrist_yaw_range

        self.wrist_pitch = np.clip(pitch, pitch_min, pitch_max)
        self.wrist_yaw = np.clip(yaw, yaw_min, yaw_max)
        return True

    def set_jaw_angle(self, angle: float) -> bool:
        """Set jaw opening angle."""
        if self.specification is None:
            return False

        grip_min, grip_max = self.specification.grip_range
        self.jaw_angle = np.clip(angle, grip_min, grip_max)
        return True

    def get_tip_transform(self) -> NDArray[np.float64]:
        """Get transformation matrix from instrument base to tip."""
        # Compose transformation from wrist angles
        pitch_rad = np.radians(self.wrist_pitch)
        yaw_rad = np.radians(self.wrist_yaw)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        rotation = Rz @ Rx

        transform = np.eye(4)
        transform[:3, :3] = rotation

        if self.specification:
            transform[2, 3] = self.specification.shaft_length

        return transform


@dataclass
class InstrumentCollisionVolume:
    """Collision geometry for an instrument."""
    instrument: SurgicalInstrument
    shaft_points: list[NDArray[np.float64]] = field(default_factory=list)
    tip_radius: float = 0.005  # meters

    def update_geometry(self, base_pose: Pose) -> None:
        """Update collision geometry based on current pose."""
        if self.instrument.specification is None:
            return

        # Generate points along instrument shaft
        num_points = 10
        shaft_length = self.instrument.specification.shaft_length

        base_matrix = base_pose.to_matrix()
        tip_transform = self.instrument.get_tip_transform()
        full_transform = base_matrix @ tip_transform

        self.shaft_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            # Interpolate along shaft
            local_point = np.array([0, 0, t * shaft_length, 1])
            world_point = full_transform @ local_point
            self.shaft_points.append(world_point[:3])

    def check_collision(
        self,
        point: NDArray[np.float64],
        clearance: float = 0.002
    ) -> bool:
        """Check if a point collides with this instrument."""
        for shaft_point in self.shaft_points:
            distance = float(np.linalg.norm(point - shaft_point))
            if distance < (self.tip_radius + clearance):
                return True
        return False


class InstrumentTracker:
    """
    Track multiple instruments and detect collisions.

    Provides collision avoidance between instruments.
    """

    def __init__(self) -> None:
        self.instruments: dict[int, SurgicalInstrument] = {}
        self.collision_volumes: dict[int, InstrumentCollisionVolume] = {}
        self.collision_clearance = 0.005  # 5mm minimum clearance

        # Collision callbacks
        self._collision_callbacks: list = []

    def register_instrument(
        self,
        arm_index: int,
        instrument: SurgicalInstrument
    ) -> None:
        """Register an instrument for tracking."""
        self.instruments[arm_index] = instrument
        self.collision_volumes[arm_index] = InstrumentCollisionVolume(
            instrument=instrument
        )

    def update_instrument_pose(self, arm_index: int, pose: Pose) -> None:
        """Update instrument pose and collision geometry."""
        if arm_index in self.instruments:
            self.instruments[arm_index].pose = pose
            self.collision_volumes[arm_index].update_geometry(pose)

    def check_instrument_collisions(self) -> list[tuple[int, int, float]]:
        """
        Check for collisions between all tracked instruments.

        Returns:
            List of (arm1, arm2, distance) tuples for potential collisions
        """
        collisions: list[tuple[int, int, float]] = []
        arm_indices = list(self.instruments.keys())

        for i, arm1 in enumerate(arm_indices):
            for arm2 in arm_indices[i + 1:]:
                min_distance = self._compute_minimum_distance(arm1, arm2)
                if min_distance < self.collision_clearance:
                    collisions.append((arm1, arm2, min_distance))

        return collisions

    def _compute_minimum_distance(self, arm1: int, arm2: int) -> float:
        """Compute minimum distance between two instruments."""
        vol1 = self.collision_volumes.get(arm1)
        vol2 = self.collision_volumes.get(arm2)

        if vol1 is None or vol2 is None:
            return float('inf')

        min_dist = float('inf')

        for p1 in vol1.shaft_points:
            for p2 in vol2.shaft_points:
                dist = float(np.linalg.norm(p1 - p2))
                min_dist = min(min_dist, dist)

        return min_dist

    def get_safe_trajectory(
        self,
        arm_index: int,
        start_pose: Pose,
        end_pose: Pose,
        num_waypoints: int = 20
    ) -> Optional[list[Pose]]:
        """
        Generate collision-free trajectory between poses.

        Args:
            arm_index: Index of moving arm
            start_pose: Starting pose
            end_pose: Target pose
            num_waypoints: Number of interpolation points

        Returns:
            List of waypoint poses, or None if no safe path found
        """
        waypoints: list[Pose] = []

        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)

            # Linear interpolation of position
            position = start_pose.position + t * (end_pose.position - start_pose.position)

            # SLERP for orientation
            orientation = self._slerp(
                start_pose.orientation,
                end_pose.orientation,
                t
            )

            test_pose = Pose(position=position, orientation=orientation)

            # Check for collisions at this waypoint
            self.collision_volumes[arm_index].update_geometry(test_pose)

            collision_found = False
            for other_arm in self.collision_volumes:
                if other_arm != arm_index:
                    min_dist = self._compute_minimum_distance(arm_index, other_arm)
                    if min_dist < self.collision_clearance:
                        collision_found = True
                        break

            if collision_found:
                # Simple path not possible, would need path planning
                return None

            waypoints.append(test_pose)

        return waypoints

    @staticmethod
    def _slerp(
        q1: NDArray[np.float64],
        q2: NDArray[np.float64],
        t: float
    ) -> NDArray[np.float64]:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)

        # Handle opposite quaternions
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Linear interpolation for very close quaternions
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return s1 * q1 + s2 * q2


class InstrumentUsageMonitor:
    """Monitor instrument usage for maintenance and safety."""

    def __init__(self) -> None:
        self.usage_data: dict[int, dict] = {}
        self.max_activations = 10000
        self.max_energy_joules = 100000.0

    def log_activation(self, arm_index: int) -> None:
        """Log an instrument activation event."""
        if arm_index not in self.usage_data:
            self.usage_data[arm_index] = {
                "activations": 0,
                "energy_delivered": 0.0,
                "operation_time": 0.0
            }

        self.usage_data[arm_index]["activations"] += 1

    def log_energy(self, arm_index: int, energy_joules: float) -> None:
        """Log energy delivered."""
        if arm_index in self.usage_data:
            self.usage_data[arm_index]["energy_delivered"] += energy_joules

    def check_maintenance_needed(self, arm_index: int) -> tuple[bool, str]:
        """Check if instrument needs maintenance or replacement."""
        if arm_index not in self.usage_data:
            return False, "No usage data"

        data = self.usage_data[arm_index]

        if data["activations"] > self.max_activations:
            return True, "Maximum activation count exceeded"

        if data["energy_delivered"] > self.max_energy_joules:
            return True, "Maximum energy delivery exceeded"

        return False, "Instrument within normal parameters"
