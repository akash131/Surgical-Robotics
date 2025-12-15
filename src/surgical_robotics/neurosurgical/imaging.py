"""Intraoperative imaging integration for neurosurgical robots."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class ImagingModality(Enum):
    """Types of intraoperative imaging."""
    MRI = auto()
    CT = auto()
    ULTRASOUND = auto()
    FLUOROSCOPY = auto()
    OPTICAL = auto()


@dataclass
class ImageVolume:
    """3D medical image volume."""
    data: NDArray[np.float32]  # 3D image data
    spacing: Tuple[float, float, float]  # Voxel spacing in mm
    origin: NDArray[np.float64]  # World coordinates of first voxel
    direction: NDArray[np.float64]  # 3x3 direction cosine matrix

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image dimensions."""
        return self.data.shape  # type: ignore

    def world_to_voxel(self, world_point: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert world coordinates to voxel indices."""
        relative = world_point - self.origin
        voxel_coords = np.linalg.inv(self.direction) @ relative
        voxel_indices = voxel_coords / np.array(self.spacing)
        return voxel_indices

    def voxel_to_world(self, voxel_indices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert voxel indices to world coordinates."""
        scaled = voxel_indices * np.array(self.spacing)
        world_point = self.direction @ scaled + self.origin
        return world_point

    def sample_at(self, world_point: NDArray[np.float64]) -> float:
        """Sample image intensity at world coordinate using trilinear interpolation."""
        voxel = self.world_to_voxel(world_point)

        # Check bounds
        if np.any(voxel < 0) or np.any(voxel >= np.array(self.shape) - 1):
            return 0.0

        # Trilinear interpolation
        x0, y0, z0 = int(voxel[0]), int(voxel[1]), int(voxel[2])
        xd, yd, zd = voxel[0] - x0, voxel[1] - y0, voxel[2] - z0

        c00 = self.data[x0, y0, z0] * (1 - xd) + self.data[x0 + 1, y0, z0] * xd
        c01 = self.data[x0, y0, z0 + 1] * (1 - xd) + self.data[x0 + 1, y0, z0 + 1] * xd
        c10 = self.data[x0, y0 + 1, z0] * (1 - xd) + self.data[x0 + 1, y0 + 1, z0] * xd
        c11 = self.data[x0, y0 + 1, z0 + 1] * (1 - xd) + self.data[x0 + 1, y0 + 1, z0 + 1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return float(c0 * (1 - zd) + c1 * zd)


@dataclass
class RegistrationResult:
    """Result of image-to-robot registration."""
    transformation_matrix: NDArray[np.float64]
    fiducial_registration_error: float  # FRE in mm
    target_registration_error: float  # TRE estimate in mm
    num_points_used: int
    is_valid: bool

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Extract rotation matrix."""
        return self.transformation_matrix[:3, :3]

    @property
    def translation(self) -> NDArray[np.float64]:
        """Extract translation vector."""
        return self.transformation_matrix[:3, 3]


class ImageRegistration:
    """
    Register images to robot coordinate system.

    Supports both fiducial-based and surface-based registration.
    """

    def __init__(self) -> None:
        self.image_points: list[NDArray[np.float64]] = []
        self.robot_points: list[NDArray[np.float64]] = []
        self.result: Optional[RegistrationResult] = None

    def add_point_pair(
        self,
        image_point: NDArray[np.float64],
        robot_point: NDArray[np.float64]
    ) -> None:
        """Add a corresponding point pair for registration."""
        self.image_points.append(np.asarray(image_point))
        self.robot_points.append(np.asarray(robot_point))

    def clear_points(self) -> None:
        """Clear all registration points."""
        self.image_points = []
        self.robot_points = []
        self.result = None

    def compute_registration(self, min_points: int = 3) -> Optional[RegistrationResult]:
        """
        Compute registration transformation.

        Uses paired-point registration with SVD.
        """
        if len(self.image_points) < min_points:
            return None

        image_pts = np.array(self.image_points)
        robot_pts = np.array(self.robot_points)

        # Compute centroids
        image_centroid = np.mean(image_pts, axis=0)
        robot_centroid = np.mean(robot_pts, axis=0)

        # Center points
        image_centered = image_pts - image_centroid
        robot_centered = robot_pts - robot_centroid

        # SVD
        H = image_centered.T @ robot_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = robot_centroid - R @ image_centroid

        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Compute FRE
        transformed_image = (R @ image_pts.T).T + t
        errors = np.linalg.norm(transformed_image - robot_pts, axis=1)
        fre = float(np.sqrt(np.mean(errors ** 2)))

        # Estimate TRE (simplified - would use proper TRE formula in practice)
        tre = fre * 1.5  # Conservative estimate

        self.result = RegistrationResult(
            transformation_matrix=transform,
            fiducial_registration_error=fre,
            target_registration_error=tre,
            num_points_used=len(self.image_points),
            is_valid=fre < 2.0  # 2mm threshold
        )

        return self.result

    def transform_point(self, image_point: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Transform a point from image to robot coordinates."""
        if self.result is None or not self.result.is_valid:
            return None

        point_h = np.append(image_point, 1)
        transformed = self.result.transformation_matrix @ point_h
        return transformed[:3]


class IntraoperativeImaging(ABC):
    """Abstract base class for intraoperative imaging systems."""

    def __init__(self, modality: ImagingModality) -> None:
        self.modality = modality
        self.current_image: Optional[ImageVolume] = None
        self.registration = ImageRegistration()
        self.is_acquiring = False

    @abstractmethod
    def acquire_image(self) -> Optional[ImageVolume]:
        """Acquire a new image."""
        pass

    @abstractmethod
    def is_robot_compatible(self) -> bool:
        """Check if robot can be in imaging field."""
        pass

    def register_to_robot(
        self,
        fiducial_image_coords: list[NDArray[np.float64]],
        fiducial_robot_coords: list[NDArray[np.float64]]
    ) -> Optional[RegistrationResult]:
        """Register image to robot coordinate system."""
        self.registration.clear_points()

        for img_pt, robot_pt in zip(fiducial_image_coords, fiducial_robot_coords):
            self.registration.add_point_pair(img_pt, robot_pt)

        return self.registration.compute_registration()


class MRIIntegration(IntraoperativeImaging):
    """
    Integration with intraoperative MRI systems.

    Handles MR-compatible robot positioning and image-guided updates.
    """

    def __init__(self) -> None:
        super().__init__(ImagingModality.MRI)

        # MRI-specific parameters
        self.field_strength = 1.5  # Tesla (1.5T or 3T typical)
        self.bore_diameter = 0.7  # meters
        self.imaging_fov = 0.4  # meters

        # Safety zones
        self.gauss_5_line_distance = 2.0  # meters from isocenter
        self.robot_safe_zone_start = 1.5  # meters

        # MR compatibility
        self.mr_compatible_mode = False
        self.active_shielding_enabled = False

    def acquire_image(self) -> Optional[ImageVolume]:
        """Acquire MRI image."""
        if not self.mr_compatible_mode:
            return None

        # In real implementation, would interface with MRI scanner
        # Create placeholder image
        shape = (256, 256, 128)
        self.current_image = ImageVolume(
            data=np.zeros(shape, dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
            origin=np.array([-128.0, -128.0, -64.0]),
            direction=np.eye(3)
        )

        return self.current_image

    def is_robot_compatible(self) -> bool:
        """Check if robot can safely operate in MRI environment."""
        # Robot must use MR-compatible materials
        return self.mr_compatible_mode

    def enter_mr_safe_mode(self) -> bool:
        """Configure robot for MR-safe operation."""
        # Would disable ferromagnetic components, switch to piezoelectric actuators, etc.
        self.mr_compatible_mode = True
        return True

    def exit_mr_safe_mode(self) -> None:
        """Exit MR-safe mode."""
        self.mr_compatible_mode = False

    def get_robot_position_in_bore(
        self,
        robot_position: NDArray[np.float64]
    ) -> dict[str, float]:
        """Calculate robot position relative to MRI bore."""
        # Assuming isocenter at origin
        distance_from_isocenter = float(np.linalg.norm(robot_position))
        radial_distance = float(np.linalg.norm(robot_position[:2]))

        return {
            "distance_from_isocenter": distance_from_isocenter,
            "radial_distance": radial_distance,
            "in_bore": radial_distance < self.bore_diameter / 2,
            "in_imaging_fov": distance_from_isocenter < self.imaging_fov / 2,
            "in_5_gauss_zone": distance_from_isocenter < self.gauss_5_line_distance,
        }


class CTIntegration(IntraoperativeImaging):
    """
    Integration with intraoperative CT systems.

    Supports cone-beam CT and O-arm imaging.
    """

    def __init__(self, ct_type: str = "cone_beam") -> None:
        super().__init__(ImagingModality.CT)

        self.ct_type = ct_type  # "cone_beam", "o_arm", "standard"

        # CT parameters
        self.gantry_clearance = 0.7  # meters
        self.scan_fov = 0.4  # meters
        self.rotation_time = 5.0  # seconds

        # Radiation safety
        self.is_emitting = False
        self.total_dose_mgy = 0.0

    def acquire_image(self) -> Optional[ImageVolume]:
        """Acquire CT image."""
        self.is_emitting = True

        # Simulate acquisition
        shape = (512, 512, 256)
        self.current_image = ImageVolume(
            data=np.zeros(shape, dtype=np.float32),
            spacing=(0.5, 0.5, 0.5),
            origin=np.array([-128.0, -128.0, -64.0]),
            direction=np.eye(3)
        )

        # Track dose (simplified)
        self.total_dose_mgy += 10.0  # Typical dose per scan

        self.is_emitting = False
        return self.current_image

    def is_robot_compatible(self) -> bool:
        """Check if robot can be in CT field during imaging."""
        # Robot must be radiolucent to avoid artifacts
        return True

    def verify_robot_clearance(
        self,
        robot_positions: list[NDArray[np.float64]]
    ) -> bool:
        """Verify robot won't collide with CT gantry during rotation."""
        for pos in robot_positions:
            # Check radial distance from isocenter
            radial = float(np.linalg.norm(pos[:2]))
            if radial > self.gantry_clearance / 2 - 0.05:  # 5cm margin
                return False
        return True

    def get_dose_report(self) -> dict[str, float]:
        """Get radiation dose report."""
        return {
            "total_dose_mgy": self.total_dose_mgy,
            "estimated_effective_dose_msv": self.total_dose_mgy * 0.015,  # Rough estimate
        }


@dataclass
class NavigationTarget:
    """Target point for image-guided navigation."""
    name: str
    image_position: NDArray[np.float64]
    robot_position: Optional[NDArray[np.float64]] = None
    error_mm: Optional[float] = None

    def update_robot_position(self, registration: ImageRegistration) -> bool:
        """Update robot position using registration."""
        result = registration.transform_point(self.image_position)
        if result is not None:
            self.robot_position = result
            return True
        return False


class ImageGuidedNavigation:
    """
    Real-time image-guided navigation system.

    Fuses imaging data with robot position for surgical guidance.
    """

    def __init__(self, imaging: IntraoperativeImaging) -> None:
        self.imaging = imaging
        self.targets: list[NavigationTarget] = []
        self.current_trajectory: Optional[list[NDArray[np.float64]]] = None

        # Display parameters
        self.display_plane = "axial"  # "axial", "sagittal", "coronal"
        self.display_slice_offset = 0.0

    def add_target(self, target: NavigationTarget) -> None:
        """Add navigation target."""
        target.update_robot_position(self.imaging.registration)
        self.targets.append(target)

    def plan_trajectory(
        self,
        entry_point: NDArray[np.float64],
        target_point: NDArray[np.float64],
        num_points: int = 100
    ) -> list[NDArray[np.float64]]:
        """Plan straight-line trajectory from entry to target."""
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = entry_point + t * (target_point - entry_point)
            trajectory.append(point)

        self.current_trajectory = trajectory
        return trajectory

    def get_distance_to_target(
        self,
        current_position: NDArray[np.float64],
        target_index: int = 0
    ) -> Optional[float]:
        """Calculate distance from current position to target."""
        if target_index >= len(self.targets):
            return None

        target = self.targets[target_index]
        if target.robot_position is None:
            return None

        return float(np.linalg.norm(current_position - target.robot_position))

    def get_trajectory_deviation(
        self,
        current_position: NDArray[np.float64]
    ) -> Optional[float]:
        """Calculate perpendicular distance from planned trajectory."""
        if self.current_trajectory is None or len(self.current_trajectory) < 2:
            return None

        # Find closest point on trajectory
        min_distance = float('inf')

        for i in range(len(self.current_trajectory) - 1):
            p1 = self.current_trajectory[i]
            p2 = self.current_trajectory[i + 1]

            # Project current position onto line segment
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if segment_length > 0:
                t = np.clip(
                    np.dot(current_position - p1, segment) / (segment_length ** 2),
                    0, 1
                )
                closest = p1 + t * segment
                distance = float(np.linalg.norm(current_position - closest))
                min_distance = min(min_distance, distance)

        return min_distance

    def update_display_plane_from_position(
        self,
        position: NDArray[np.float64]
    ) -> None:
        """Update display slice to show current position."""
        if self.imaging.current_image is None:
            return

        # Get slice offset based on display plane
        if self.display_plane == "axial":
            self.display_slice_offset = position[2]
        elif self.display_plane == "sagittal":
            self.display_slice_offset = position[0]
        elif self.display_plane == "coronal":
            self.display_slice_offset = position[1]
