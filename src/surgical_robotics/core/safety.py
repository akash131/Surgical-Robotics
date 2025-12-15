"""Safety systems for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose, SafetyLevel, WorkspaceVolume


class CollisionType(Enum):
    """Types of collision detection."""
    SELF_COLLISION = auto()
    ENVIRONMENT_COLLISION = auto()
    PATIENT_COLLISION = auto()
    INSTRUMENT_COLLISION = auto()


@dataclass
class CollisionResult:
    """Result of collision detection."""
    collision_detected: bool
    collision_type: Optional[CollisionType] = None
    collision_point: Optional[NDArray[np.float64]] = None
    minimum_distance: float = float('inf')
    objects_involved: list[str] = field(default_factory=list)


@dataclass
class SafetyZone:
    """Defines a safety zone with different clearance requirements."""
    name: str
    volume: WorkspaceVolume
    required_clearance: float  # meters
    priority: int = 0  # Higher priority = more important


class CollisionDetector:
    """Detects potential collisions between robot components and environment."""

    def __init__(self) -> None:
        self.safety_zones: list[SafetyZone] = []
        self.arm_geometries: dict[str, list[NDArray[np.float64]]] = {}
        self.obstacle_geometries: list[NDArray[np.float64]] = []

    def add_safety_zone(self, zone: SafetyZone) -> None:
        """Register a safety zone."""
        self.safety_zones.append(zone)
        self.safety_zones.sort(key=lambda z: z.priority, reverse=True)

    def set_arm_geometry(
        self, arm_name: str, link_positions: list[NDArray[np.float64]]
    ) -> None:
        """Update arm geometry for collision checking."""
        self.arm_geometries[arm_name] = link_positions

    def add_obstacle(self, obstacle_points: NDArray[np.float64]) -> None:
        """Add obstacle geometry."""
        self.obstacle_geometries.append(obstacle_points)

    def check_collision(
        self,
        point: NDArray[np.float64],
        arm_name: Optional[str] = None
    ) -> CollisionResult:
        """
        Check for collisions at a given point.

        Args:
            point: 3D point to check
            arm_name: Name of arm for self-collision checking

        Returns:
            CollisionResult with detection status and details
        """
        result = CollisionResult(collision_detected=False)

        # Check safety zones
        for zone in self.safety_zones:
            if zone.volume.contains_point(point):
                # Check if too close to zone boundary
                distance_to_boundary = self._distance_to_boundary(point, zone.volume)
                if distance_to_boundary < zone.required_clearance:
                    result.collision_detected = True
                    result.collision_type = CollisionType.PATIENT_COLLISION
                    result.collision_point = point
                    result.minimum_distance = distance_to_boundary
                    result.objects_involved = [zone.name]
                    return result

        # Check self-collision with other arms
        if arm_name:
            for other_arm, links in self.arm_geometries.items():
                if other_arm != arm_name:
                    for link_point in links:
                        distance = float(np.linalg.norm(point - link_point))
                        if distance < result.minimum_distance:
                            result.minimum_distance = distance
                        if distance < 0.01:  # 1cm collision threshold
                            result.collision_detected = True
                            result.collision_type = CollisionType.SELF_COLLISION
                            result.collision_point = point
                            result.objects_involved = [arm_name, other_arm]
                            return result

        # Check obstacle collisions
        for obstacle in self.obstacle_geometries:
            for obs_point in obstacle:
                distance = float(np.linalg.norm(point - obs_point))
                if distance < result.minimum_distance:
                    result.minimum_distance = distance
                if distance < 0.005:  # 5mm collision threshold
                    result.collision_detected = True
                    result.collision_type = CollisionType.ENVIRONMENT_COLLISION
                    result.collision_point = point
                    return result

        return result

    def check_path_collision(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        num_samples: int = 20
    ) -> CollisionResult:
        """Check for collisions along a path."""
        for t in np.linspace(0, 1, num_samples):
            point = start + t * (end - start)
            result = self.check_collision(point)
            if result.collision_detected:
                return result
        return CollisionResult(collision_detected=False)

    @staticmethod
    def _distance_to_boundary(
        point: NDArray[np.float64], volume: WorkspaceVolume
    ) -> float:
        """Calculate distance from point to volume boundary."""
        if volume.shape == "cuboid":
            half_dims = volume.dimensions / 2
            relative = np.abs(point - volume.center)
            distances = half_dims - relative
            return float(np.min(distances))
        elif volume.shape == "sphere":
            radius = volume.dimensions[0] / 2
            distance_to_center = float(np.linalg.norm(point - volume.center))
            return radius - distance_to_center
        return float('inf')


@dataclass
class ForceLimit:
    """Force limits for safe operation."""
    max_force: float  # Newtons
    max_torque: float  # Newton-meters
    impact_force_limit: float  # Newtons (for unexpected contacts)


class SafetyController:
    """Central safety controller for surgical robot."""

    def __init__(self) -> None:
        self.collision_detector = CollisionDetector()
        self.force_limits = ForceLimit(
            max_force=10.0,  # 10N default
            max_torque=2.0,  # 2Nm default
            impact_force_limit=5.0  # 5N impact limit
        )
        self.current_safety_level = SafetyLevel.NORMAL
        self._emergency_callbacks: list[Callable[[], None]] = []
        self._watchdog_active = True
        self._last_heartbeat = 0.0

    def register_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Register callback to be called on emergency stop."""
        self._emergency_callbacks.append(callback)

    def check_force_limits(
        self, forces: NDArray[np.float64], torques: NDArray[np.float64]
    ) -> SafetyLevel:
        """
        Check if forces/torques are within safe limits.

        Args:
            forces: 3D force vector [Fx, Fy, Fz]
            torques: 3D torque vector [Tx, Ty, Tz]

        Returns:
            Current safety level based on force readings
        """
        force_magnitude = float(np.linalg.norm(forces))
        torque_magnitude = float(np.linalg.norm(torques))

        # Check against limits
        force_ratio = force_magnitude / self.force_limits.max_force
        torque_ratio = torque_magnitude / self.force_limits.max_torque

        max_ratio = max(force_ratio, torque_ratio)

        if max_ratio > 1.0:
            self._trigger_emergency("Force/torque limit exceeded")
            return SafetyLevel.CRITICAL
        elif max_ratio > 0.9:
            self.current_safety_level = SafetyLevel.WARNING
            return SafetyLevel.WARNING
        elif max_ratio > 0.7:
            self.current_safety_level = SafetyLevel.CAUTION
            return SafetyLevel.CAUTION

        self.current_safety_level = SafetyLevel.NORMAL
        return SafetyLevel.NORMAL

    def check_velocity_limits(
        self,
        joint_velocities: NDArray[np.float64],
        max_velocities: NDArray[np.float64]
    ) -> SafetyLevel:
        """Check if joint velocities are within limits."""
        velocity_ratios = np.abs(joint_velocities) / max_velocities
        max_ratio = float(np.max(velocity_ratios))

        if max_ratio > 1.0:
            self._trigger_emergency("Velocity limit exceeded")
            return SafetyLevel.CRITICAL
        elif max_ratio > 0.9:
            return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def validate_trajectory(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        accelerations: NDArray[np.float64],
        joint_limits: tuple[NDArray[np.float64], NDArray[np.float64]],
        velocity_limits: NDArray[np.float64],
        acceleration_limits: NDArray[np.float64]
    ) -> tuple[bool, str]:
        """
        Validate entire trajectory before execution.

        Returns:
            Tuple of (is_valid, error_message)
        """
        min_pos, max_pos = joint_limits

        # Check position limits
        if np.any(positions < min_pos) or np.any(positions > max_pos):
            return False, "Trajectory exceeds joint position limits"

        # Check velocity limits
        if np.any(np.abs(velocities) > velocity_limits):
            return False, "Trajectory exceeds velocity limits"

        # Check acceleration limits
        if np.any(np.abs(accelerations) > acceleration_limits):
            return False, "Trajectory exceeds acceleration limits"

        return True, "Trajectory validated successfully"

    def _trigger_emergency(self, reason: str) -> None:
        """Trigger emergency stop."""
        self.current_safety_level = SafetyLevel.CRITICAL
        for callback in self._emergency_callbacks:
            callback()

    def heartbeat(self, timestamp: float) -> bool:
        """
        Watchdog heartbeat - must be called periodically.

        Returns:
            True if watchdog is healthy, False if timeout detected
        """
        if self._watchdog_active:
            if self._last_heartbeat > 0:
                delta = timestamp - self._last_heartbeat
                if delta > 0.1:  # 100ms timeout
                    self._trigger_emergency("Watchdog timeout")
                    return False
            self._last_heartbeat = timestamp
        return True

    def enable_watchdog(self, enabled: bool = True) -> None:
        """Enable or disable watchdog timer."""
        self._watchdog_active = enabled
        if enabled:
            self._last_heartbeat = 0.0


class VirtualFixture:
    """
    Virtual fixture (active constraint) for guiding robot motion.

    Used to constrain motion to a surface, path, or region.
    """

    def __init__(self, name: str, stiffness: float = 1000.0) -> None:
        self.name = name
        self.stiffness = stiffness  # N/m
        self.is_active = False

    def compute_constraint_force(
        self, current_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute force to maintain constraint (override in subclasses)."""
        return np.zeros(3)


class PlaneFixture(VirtualFixture):
    """Constrain motion to a plane."""

    def __init__(
        self,
        name: str,
        point_on_plane: NDArray[np.float64],
        normal: NDArray[np.float64],
        stiffness: float = 1000.0
    ) -> None:
        super().__init__(name, stiffness)
        self.point = np.asarray(point_on_plane)
        self.normal = np.asarray(normal) / np.linalg.norm(normal)

    def compute_constraint_force(
        self, current_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute force to push position back to plane."""
        if not self.is_active:
            return np.zeros(3)

        # Distance from plane
        displacement = current_position - self.point
        distance = float(np.dot(displacement, self.normal))

        # Force proportional to deviation
        return -self.stiffness * distance * self.normal


class PathFixture(VirtualFixture):
    """Constrain motion to follow a path."""

    def __init__(
        self,
        name: str,
        path_points: NDArray[np.float64],
        stiffness: float = 1000.0,
        tube_radius: float = 0.01
    ) -> None:
        super().__init__(name, stiffness)
        self.path_points = path_points
        self.tube_radius = tube_radius

    def compute_constraint_force(
        self, current_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute force to keep position on path."""
        if not self.is_active:
            return np.zeros(3)

        # Find closest point on path
        min_distance = float('inf')
        closest_point = self.path_points[0]

        for i in range(len(self.path_points) - 1):
            p1, p2 = self.path_points[i], self.path_points[i + 1]
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if segment_length > 0:
                t = np.clip(
                    np.dot(current_position - p1, segment) / (segment_length ** 2),
                    0, 1
                )
                point_on_segment = p1 + t * segment
                distance = float(np.linalg.norm(current_position - point_on_segment))

                if distance < min_distance:
                    min_distance = distance
                    closest_point = point_on_segment

        # Force toward path if outside tube
        if min_distance > self.tube_radius:
            direction = closest_point - current_position
            direction = direction / np.linalg.norm(direction)
            deviation = min_distance - self.tube_radius
            return self.stiffness * deviation * direction

        return np.zeros(3)
