"""3D Stereoscopic Vision System for Da Vinci robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class CameraMode(Enum):
    """Endoscope camera operational modes."""
    STANDARD = auto()
    FLUORESCENCE = auto()  # ICG/Firefly imaging
    NEAR_INFRARED = auto()
    ENHANCED_VISION = auto()


@dataclass
class CameraParameters:
    """Intrinsic camera parameters."""
    focal_length: Tuple[float, float]  # fx, fy in pixels
    principal_point: Tuple[float, float]  # cx, cy in pixels
    resolution: Tuple[int, int]  # width, height
    distortion_coefficients: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(5)
    )

    @property
    def intrinsic_matrix(self) -> NDArray[np.float64]:
        """Get 3x3 camera intrinsic matrix."""
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])


@dataclass
class StereoParameters:
    """Stereo vision calibration parameters."""
    baseline: float  # Distance between cameras in meters
    left_camera: CameraParameters
    right_camera: CameraParameters
    rectification_R1: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3)
    )
    rectification_R2: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3)
    )

    def compute_disparity_to_depth(self, disparity: float) -> float:
        """Convert disparity to depth using baseline and focal length."""
        if disparity <= 0:
            return float('inf')
        fx = self.left_camera.focal_length[0]
        return (self.baseline * fx) / disparity


class StereoscopicVision:
    """
    3D Stereoscopic Vision System.

    Provides depth perception for the surgical field using
    dual cameras on the endoscope.
    """

    def __init__(self, stereo_params: Optional[StereoParameters] = None) -> None:
        if stereo_params is None:
            # Default HD stereo parameters
            default_camera = CameraParameters(
                focal_length=(1000.0, 1000.0),
                principal_point=(960.0, 540.0),
                resolution=(1920, 1080)
            )
            stereo_params = StereoParameters(
                baseline=0.005,  # 5mm baseline typical for endoscope
                left_camera=default_camera,
                right_camera=default_camera
            )

        self.stereo_params = stereo_params
        self.mode = CameraMode.STANDARD

        # Image buffers (would normally hold actual image data)
        self._left_image: Optional[NDArray[np.uint8]] = None
        self._right_image: Optional[NDArray[np.uint8]] = None
        self._depth_map: Optional[NDArray[np.float32]] = None

        # Processing parameters
        self.min_disparity = 0
        self.max_disparity = 128
        self.block_size = 5

    def capture_stereo_pair(self) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Capture synchronized stereo image pair."""
        # In real implementation, would capture from cameras
        return self._left_image, self._right_image

    def compute_depth_map(
        self,
        left_image: NDArray[np.uint8],
        right_image: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        """
        Compute depth map from stereo image pair.

        Uses block matching for disparity estimation.

        Args:
            left_image: Left camera image (grayscale)
            right_image: Right camera image (grayscale)

        Returns:
            Depth map in meters
        """
        # Simplified disparity computation
        # Real implementation would use SGM or other advanced methods
        h, w = left_image.shape[:2]
        disparity = np.zeros((h, w), dtype=np.float32)

        half_block = self.block_size // 2

        for y in range(half_block, h - half_block):
            for x in range(self.max_disparity + half_block, w - half_block):
                left_block = left_image[
                    y - half_block:y + half_block + 1,
                    x - half_block:x + half_block + 1
                ]

                best_disparity = 0
                best_score = float('inf')

                for d in range(self.min_disparity, min(self.max_disparity, x - half_block)):
                    right_block = right_image[
                        y - half_block:y + half_block + 1,
                        x - d - half_block:x - d + half_block + 1
                    ]

                    # Sum of absolute differences
                    score = float(np.sum(np.abs(
                        left_block.astype(np.float32) - right_block.astype(np.float32)
                    )))

                    if score < best_score:
                        best_score = score
                        best_disparity = d

                disparity[y, x] = best_disparity

        # Convert disparity to depth
        depth_map = np.zeros_like(disparity)
        valid_mask = disparity > 0
        depth_map[valid_mask] = (
            self.stereo_params.baseline *
            self.stereo_params.left_camera.focal_length[0] /
            disparity[valid_mask]
        )

        self._depth_map = depth_map
        return depth_map

    def pixel_to_3d(
        self,
        pixel_uv: Tuple[int, int],
        depth: float
    ) -> NDArray[np.float64]:
        """
        Convert pixel coordinates to 3D point.

        Args:
            pixel_uv: Pixel coordinates (u, v)
            depth: Depth in meters

        Returns:
            3D point in camera frame
        """
        u, v = pixel_uv
        K_inv = np.linalg.inv(self.stereo_params.left_camera.intrinsic_matrix)
        pixel_homogeneous = np.array([u, v, 1.0])

        ray_direction = K_inv @ pixel_homogeneous
        point_3d = ray_direction * depth

        return point_3d

    def project_3d_to_pixel(
        self,
        point_3d: NDArray[np.float64],
        use_left: bool = True
    ) -> Tuple[int, int]:
        """
        Project 3D point to pixel coordinates.

        Args:
            point_3d: 3D point in camera frame
            use_left: Use left camera (True) or right camera (False)

        Returns:
            Pixel coordinates (u, v)
        """
        camera = (
            self.stereo_params.left_camera if use_left
            else self.stereo_params.right_camera
        )

        K = camera.intrinsic_matrix
        if point_3d[2] <= 0:
            return (0, 0)

        pixel_homogeneous = K @ (point_3d / point_3d[2])
        u = int(round(pixel_homogeneous[0]))
        v = int(round(pixel_homogeneous[1]))

        return (u, v)

    def set_mode(self, mode: CameraMode) -> None:
        """Set camera imaging mode."""
        self.mode = mode


class EndoscopeController:
    """
    Controller for the endoscope camera arm.

    Manages camera positioning, zoom, and focus.
    """

    def __init__(self) -> None:
        self.pose = Pose.identity()
        self.zoom_level = 1.0  # 1.0 = standard, up to 10x
        self.focus_distance = 0.1  # meters
        self.is_locked = False

        # Movement limits
        self.max_pan = np.radians(60)  # ±60 degrees
        self.max_tilt = np.radians(60)
        self.min_zoom = 1.0
        self.max_zoom = 10.0

        # Auto-focus parameters
        self.auto_focus_enabled = True
        self.focus_target: Optional[NDArray[np.float64]] = None

    def move_to(self, target_pose: Pose) -> bool:
        """Move endoscope to target pose."""
        if self.is_locked:
            return False

        # Validate target is within limits
        # (simplified - would normally check pan/tilt angles)
        self.pose = target_pose
        return True

    def pan_tilt(self, pan_delta: float, tilt_delta: float) -> bool:
        """
        Adjust camera pan and tilt.

        Args:
            pan_delta: Pan angle change in radians
            tilt_delta: Tilt angle change in radians

        Returns:
            True if movement was successful
        """
        if self.is_locked:
            return False

        # Apply pan/tilt to orientation
        current_euler = self._quaternion_to_euler(self.pose.orientation)
        new_yaw = np.clip(current_euler[2] + pan_delta, -self.max_pan, self.max_pan)
        new_pitch = np.clip(current_euler[1] + tilt_delta, -self.max_tilt, self.max_tilt)

        self.pose.orientation = self._euler_to_quaternion(
            current_euler[0], new_pitch, new_yaw
        )
        return True

    def set_zoom(self, zoom_level: float) -> None:
        """Set camera zoom level."""
        self.zoom_level = np.clip(zoom_level, self.min_zoom, self.max_zoom)

    def zoom_in(self, factor: float = 1.5) -> None:
        """Zoom in by factor."""
        self.set_zoom(self.zoom_level * factor)

    def zoom_out(self, factor: float = 1.5) -> None:
        """Zoom out by factor."""
        self.set_zoom(self.zoom_level / factor)

    def set_focus(self, distance: float) -> None:
        """Set manual focus distance."""
        self.auto_focus_enabled = False
        self.focus_distance = max(0.01, distance)  # Minimum 1cm

    def enable_auto_focus(self, target_point: Optional[NDArray[np.float64]] = None) -> None:
        """Enable auto-focus, optionally targeting a specific 3D point."""
        self.auto_focus_enabled = True
        self.focus_target = target_point

    def lock(self) -> None:
        """Lock endoscope position."""
        self.is_locked = True

    def unlock(self) -> None:
        """Unlock endoscope position."""
        self.is_locked = False

    def follow_instrument(
        self,
        instrument_pose: Pose,
        lead_distance: float = 0.02
    ) -> None:
        """
        Adjust camera to follow an instrument.

        Args:
            instrument_pose: Current instrument pose
            lead_distance: How far ahead of instrument to look
        """
        if self.is_locked:
            return

        # Point camera toward instrument tip with some lead
        target_point = instrument_pose.position
        camera_position = self.pose.position

        # Compute look direction
        look_direction = target_point - camera_position
        distance = float(np.linalg.norm(look_direction))

        if distance > 0.001:
            look_direction = look_direction / distance

            # Update orientation to look at target
            up = np.array([0, 0, 1])
            right = np.cross(look_direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, look_direction)

            rotation_matrix = np.column_stack([right, up, -look_direction])
            self.pose.orientation = self._rotation_matrix_to_quaternion(rotation_matrix)

            # Auto-adjust focus
            if self.auto_focus_enabled:
                self.focus_distance = distance

    @staticmethod
    def _quaternion_to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
        """Convert Euler angles to quaternion."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    @staticmethod
    def _rotation_matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])


@dataclass
class VisualMarker:
    """Visual marker for instrument/anatomical tracking."""
    marker_id: int
    position_3d: NDArray[np.float64]
    orientation: NDArray[np.float64]
    size: float  # meters
    confidence: float = 1.0


class MarkerTracker:
    """
    Track visual markers in stereo images.

    Used for instrument tracking and registration.
    """

    def __init__(self, stereo_vision: StereoscopicVision) -> None:
        self.stereo_vision = stereo_vision
        self.tracked_markers: dict[int, VisualMarker] = {}
        self.marker_history: dict[int, list[NDArray[np.float64]]] = {}
        self.history_length = 10

    def detect_markers(
        self,
        left_image: NDArray[np.uint8],
        right_image: NDArray[np.uint8]
    ) -> list[VisualMarker]:
        """
        Detect and localize markers in stereo images.

        Args:
            left_image: Left camera image
            right_image: Right camera image

        Returns:
            List of detected markers with 3D positions
        """
        # Simplified marker detection
        # Real implementation would use ArUco, AprilTag, or custom markers
        detected: list[VisualMarker] = []

        # Placeholder for actual detection logic
        # Would involve:
        # 1. Detect markers in both images
        # 2. Match corresponding markers
        # 3. Triangulate 3D positions
        # 4. Estimate orientations

        return detected

    def update_tracking(self, markers: list[VisualMarker]) -> None:
        """Update tracked markers with new detections."""
        for marker in markers:
            self.tracked_markers[marker.marker_id] = marker

            # Update history
            if marker.marker_id not in self.marker_history:
                self.marker_history[marker.marker_id] = []

            history = self.marker_history[marker.marker_id]
            history.append(marker.position_3d.copy())

            if len(history) > self.history_length:
                history.pop(0)

    def get_smoothed_position(self, marker_id: int) -> Optional[NDArray[np.float64]]:
        """Get smoothed marker position from history."""
        if marker_id not in self.marker_history:
            return None

        history = self.marker_history[marker_id]
        if not history:
            return None

        # Simple moving average
        return np.mean(history, axis=0)
