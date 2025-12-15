"""Bone cutting and milling systems for orthopedic robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class ToolType(Enum):
    """Types of cutting tools."""
    SPHERICAL_BURR = auto()
    CYLINDRICAL_BURR = auto()
    TAPERED_BURR = auto()
    RECIPROCATING_SAW = auto()
    OSCILLATING_SAW = auto()
    DRILL_BIT = auto()
    REAMER = auto()


@dataclass
class MillingTool:
    """Milling/burring tool specifications."""
    name: str
    tool_type: ToolType
    diameter: float  # meters
    length: float  # meters
    num_flutes: int = 4

    # Operating parameters
    recommended_rpm_min: float = 20000
    recommended_rpm_max: float = 80000
    recommended_feed_min: float = 0.001  # m/s
    recommended_feed_max: float = 0.010

    # Current state
    current_rpm: float = 0.0
    wear_level: float = 0.0  # 0 to 1

    @property
    def radius(self) -> float:
        """Get tool radius."""
        return self.diameter / 2

    def compute_material_removal_rate(
        self,
        feed_rate: float,
        depth_of_cut: float
    ) -> float:
        """
        Compute material removal rate (MRR).

        Args:
            feed_rate: Feed rate in m/s
            depth_of_cut: Depth of cut in meters

        Returns:
            MRR in m³/s
        """
        # Simplified MRR calculation
        return feed_rate * depth_of_cut * self.diameter


@dataclass
class SawBlade:
    """Saw blade specifications."""
    name: str
    blade_width: float  # meters (kerf width)
    blade_length: float  # meters
    blade_thickness: float  # meters

    # Operating parameters
    stroke_length: float = 0.010  # meters
    recommended_speed_min: float = 10000  # strokes per minute
    recommended_speed_max: float = 20000

    # Current state
    current_speed: float = 0.0
    oscillation_angle: float = 0.0  # for oscillating saws


@dataclass
class CuttingPath:
    """Path for robotic cutting operation."""
    name: str
    waypoints: list[NDArray[np.float64]] = field(default_factory=list)
    normals: list[NDArray[np.float64]] = field(default_factory=list)  # Surface normals
    feed_rates: list[float] = field(default_factory=list)  # m/s per segment

    # Path parameters
    is_closed: bool = False
    total_length: float = 0.0

    def add_waypoint(
        self,
        position: NDArray[np.float64],
        normal: Optional[NDArray[np.float64]] = None,
        feed_rate: float = 0.005
    ) -> None:
        """Add waypoint to path."""
        self.waypoints.append(np.asarray(position))

        if normal is not None:
            self.normals.append(np.asarray(normal))
        else:
            self.normals.append(np.array([0, 0, 1]))

        self.feed_rates.append(feed_rate)
        self._update_length()

    def _update_length(self) -> None:
        """Update total path length."""
        if len(self.waypoints) < 2:
            self.total_length = 0.0
            return

        length = 0.0
        for i in range(len(self.waypoints) - 1):
            length += float(np.linalg.norm(
                self.waypoints[i + 1] - self.waypoints[i]
            ))

        if self.is_closed and len(self.waypoints) > 2:
            length += float(np.linalg.norm(
                self.waypoints[0] - self.waypoints[-1]
            ))

        self.total_length = length

    def interpolate(self, num_points: int) -> list[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Interpolate path to specified number of points.

        Returns:
            List of (position, normal) tuples
        """
        if len(self.waypoints) < 2:
            return []

        result: list[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []

        # Compute cumulative distances
        distances = [0.0]
        for i in range(len(self.waypoints) - 1):
            d = float(np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]))
            distances.append(distances[-1] + d)

        # Interpolate
        target_distances = np.linspace(0, distances[-1], num_points)

        for target_d in target_distances:
            # Find segment
            for i in range(len(distances) - 1):
                if distances[i] <= target_d <= distances[i + 1]:
                    segment_d = distances[i + 1] - distances[i]
                    if segment_d > 0:
                        t = (target_d - distances[i]) / segment_d
                    else:
                        t = 0

                    position = (
                        self.waypoints[i] +
                        t * (self.waypoints[i + 1] - self.waypoints[i])
                    )
                    normal = (
                        self.normals[i] +
                        t * (self.normals[i + 1] - self.normals[i])
                    )
                    normal = normal / np.linalg.norm(normal)

                    result.append((position, normal))
                    break

        return result


class BoneCuttingSystem:
    """
    Complete bone cutting system controller.

    Manages cutting tools, paths, and execution.
    """

    def __init__(self) -> None:
        self.current_tool: Optional[MillingTool | SawBlade] = None
        self.cutting_paths: dict[str, CuttingPath] = {}
        self.current_path: Optional[CuttingPath] = None

        # Cutting parameters
        self.target_depth: float = 0.0
        self.stepover: float = 0.5  # Fraction of tool diameter

        # State
        self.is_cutting = False
        self.current_position: NDArray[np.float64] = np.zeros(3)
        self.material_removed: float = 0.0  # m³

        # Safety limits
        self.max_depth = 0.050  # 50mm max depth
        self.max_feed_rate = 0.020  # 20mm/s

    def load_tool(self, tool: MillingTool | SawBlade) -> None:
        """Load cutting tool."""
        self.current_tool = tool
        if isinstance(tool, MillingTool):
            self.stepover = tool.diameter * 0.5

    def add_cutting_path(self, path: CuttingPath) -> None:
        """Add a cutting path."""
        self.cutting_paths[path.name] = path

    def select_path(self, path_name: str) -> bool:
        """Select active cutting path."""
        if path_name in self.cutting_paths:
            self.current_path = self.cutting_paths[path_name]
            return True
        return False

    def generate_planar_cut_paths(
        self,
        plane_origin: NDArray[np.float64],
        plane_normal: NDArray[np.float64],
        width: float,
        height: float,
        depth: float
    ) -> list[CuttingPath]:
        """
        Generate paths for a planar cut.

        Args:
            plane_origin: Origin point of cutting plane
            plane_normal: Normal direction of plane
            width: Cut width in meters
            height: Cut height in meters
            depth: Cut depth in meters

        Returns:
            List of cutting paths for each depth layer
        """
        if self.current_tool is None:
            return []

        tool_diameter = 0.003  # Default 3mm if saw
        if isinstance(self.current_tool, MillingTool):
            tool_diameter = self.current_tool.diameter

        paths: list[CuttingPath] = []

        # Compute plane coordinate system
        normal = plane_normal / np.linalg.norm(plane_normal)

        # Create orthogonal vectors
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])

        x_axis = np.cross(up, normal)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)

        # Generate layers
        depth_step = tool_diameter * 0.3  # 30% of tool diameter per pass
        num_layers = int(np.ceil(depth / depth_step))

        for layer in range(num_layers):
            current_depth = min((layer + 1) * depth_step, depth)
            path = CuttingPath(name=f"planar_cut_layer_{layer}")

            # Raster pattern
            step = tool_diameter * self.stepover
            num_passes = int(np.ceil(height / step))

            for i in range(num_passes):
                y_offset = i * step - height / 2

                if i % 2 == 0:
                    # Forward pass
                    start = plane_origin + x_axis * (-width/2) + y_axis * y_offset - normal * current_depth
                    end = plane_origin + x_axis * (width/2) + y_axis * y_offset - normal * current_depth
                else:
                    # Reverse pass
                    start = plane_origin + x_axis * (width/2) + y_axis * y_offset - normal * current_depth
                    end = plane_origin + x_axis * (-width/2) + y_axis * y_offset - normal * current_depth

                path.add_waypoint(start, -normal)
                path.add_waypoint(end, -normal)

            paths.append(path)

        return paths

    def generate_contour_path(
        self,
        contour_points: list[NDArray[np.float64]],
        depth: float,
        closed: bool = True
    ) -> CuttingPath:
        """
        Generate path for contour cutting.

        Args:
            contour_points: Points defining the contour
            depth: Cutting depth
            closed: Whether contour is closed

        Returns:
            Cutting path for contour
        """
        path = CuttingPath(name="contour_cut", is_closed=closed)

        for point in contour_points:
            # Offset point by depth in Z direction (simplified)
            cut_point = point.copy()
            cut_point[2] -= depth
            path.add_waypoint(cut_point)

        if closed and len(contour_points) > 0:
            # Add first point again to close
            cut_point = contour_points[0].copy()
            cut_point[2] -= depth
            path.add_waypoint(cut_point)

        return path

    def compute_cut_time(self, path: CuttingPath) -> float:
        """Estimate cutting time for a path."""
        if not path.waypoints or not path.feed_rates:
            return 0.0

        total_time = 0.0
        for i in range(len(path.waypoints) - 1):
            segment_length = float(np.linalg.norm(
                path.waypoints[i + 1] - path.waypoints[i]
            ))
            feed_rate = path.feed_rates[i] if i < len(path.feed_rates) else 0.005
            if feed_rate > 0:
                total_time += segment_length / feed_rate

        return total_time

    def validate_path(self, path: CuttingPath) -> Tuple[bool, list[str]]:
        """
        Validate cutting path.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues: list[str] = []

        if not path.waypoints:
            issues.append("Path has no waypoints")
            return False, issues

        if self.current_tool is None:
            issues.append("No tool loaded")
            return False, issues

        # Check feed rates
        for i, feed in enumerate(path.feed_rates):
            if feed > self.max_feed_rate:
                issues.append(f"Feed rate at point {i} exceeds maximum")

        # Check for sharp corners (may cause tool breakage)
        for i in range(1, len(path.waypoints) - 1):
            v1 = path.waypoints[i] - path.waypoints[i - 1]
            v2 = path.waypoints[i + 1] - path.waypoints[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                if angle < np.radians(30):  # Sharp corner
                    issues.append(f"Sharp corner at point {i} (angle: {np.degrees(angle):.1f}°)")

        return len(issues) == 0, issues


@dataclass
class CuttingForceModel:
    """Model for predicting cutting forces in bone."""
    # Material properties (cortical bone defaults)
    specific_cutting_force: float = 100e6  # N/m² (100 MPa)
    friction_coefficient: float = 0.3

    # Tool geometry effects
    rake_angle: float = 0.0  # radians
    helix_angle: float = 0.0  # radians

    def predict_force(
        self,
        tool: MillingTool,
        feed_rate: float,
        depth_of_cut: float,
        width_of_cut: float
    ) -> NDArray[np.float64]:
        """
        Predict cutting forces.

        Args:
            tool: Cutting tool
            feed_rate: Feed rate in m/s
            depth_of_cut: Depth of cut in m
            width_of_cut: Width of cut in m

        Returns:
            Force vector [Fx, Fy, Fz] in Newtons
        """
        # Simplified mechanistic force model
        # F = Kc * h * b
        # where h = chip thickness, b = width of cut

        # Chip thickness (simplified)
        if tool.current_rpm > 0:
            feed_per_tooth = feed_rate / (tool.current_rpm / 60 * tool.num_flutes)
        else:
            feed_per_tooth = 0.0001

        # Tangential force
        Ft = self.specific_cutting_force * feed_per_tooth * width_of_cut

        # Radial force (typically 30-50% of tangential)
        Fr = 0.4 * Ft

        # Axial force
        Fa = depth_of_cut * self.specific_cutting_force * 0.1

        return np.array([Fr, Ft, Fa])

    def predict_torque(
        self,
        tool: MillingTool,
        tangential_force: float
    ) -> float:
        """Predict spindle torque."""
        return tangential_force * tool.radius


class AdaptiveCuttingController:
    """
    Adaptive controller for bone cutting.

    Adjusts feed rate and spindle speed based on
    measured cutting forces.
    """

    def __init__(self) -> None:
        self.force_model = CuttingForceModel()
        self.target_force = 20.0  # Target cutting force in Newtons
        self.force_tolerance = 5.0  # Acceptable force variation

        # PID gains for feed rate control
        self.kp = 0.001
        self.ki = 0.0001
        self.kd = 0.00001

        # State
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.current_feed_rate = 0.005  # m/s

        # Limits
        self.min_feed_rate = 0.001
        self.max_feed_rate = 0.020

    def update(
        self,
        measured_force: float,
        dt: float
    ) -> float:
        """
        Update feed rate based on measured force.

        Args:
            measured_force: Measured cutting force in Newtons
            dt: Time step in seconds

        Returns:
            Updated feed rate in m/s
        """
        # Force error
        error = self.target_force - measured_force

        # PID control
        self.integral_error += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0

        # Compute adjustment
        adjustment = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative
        )

        # Update feed rate
        self.current_feed_rate += adjustment
        self.current_feed_rate = np.clip(
            self.current_feed_rate,
            self.min_feed_rate,
            self.max_feed_rate
        )

        self.previous_error = error
        return self.current_feed_rate

    def reset(self) -> None:
        """Reset controller state."""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.current_feed_rate = 0.005
