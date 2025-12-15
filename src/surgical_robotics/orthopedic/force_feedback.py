"""Force feedback and tissue differentiation for orthopedic robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray


class TissueType(Enum):
    """Types of tissue encountered in orthopedic surgery."""
    SOFT_TISSUE = auto()
    CARTILAGE = auto()
    CANCELLOUS_BONE = auto()  # Spongy bone
    CORTICAL_BONE = auto()  # Hard outer bone
    METAL_IMPLANT = auto()
    CEMENT = auto()
    AIR = auto()  # No contact


@dataclass
class TissueProperties:
    """Mechanical properties of tissue types."""
    tissue_type: TissueType
    stiffness: float  # N/m
    damping: float  # Ns/m
    yield_force: float  # N (force at which tissue yields)
    typical_frequency: float  # Hz (vibration signature)

    @classmethod
    def get_default_properties(cls, tissue_type: TissueType) -> "TissueProperties":
        """Get default properties for a tissue type."""
        defaults = {
            TissueType.SOFT_TISSUE: cls(TissueType.SOFT_TISSUE, 1000, 50, 5, 50),
            TissueType.CARTILAGE: cls(TissueType.CARTILAGE, 10000, 100, 20, 200),
            TissueType.CANCELLOUS_BONE: cls(TissueType.CANCELLOUS_BONE, 50000, 200, 50, 500),
            TissueType.CORTICAL_BONE: cls(TissueType.CORTICAL_BONE, 200000, 500, 200, 2000),
            TissueType.METAL_IMPLANT: cls(TissueType.METAL_IMPLANT, 1000000, 1000, 1000, 5000),
            TissueType.CEMENT: cls(TissueType.CEMENT, 100000, 300, 100, 1000),
            TissueType.AIR: cls(TissueType.AIR, 0, 0, 0, 0),
        }
        return defaults.get(tissue_type, defaults[TissueType.AIR])


@dataclass
class ForceReading:
    """Force/torque sensor reading."""
    forces: NDArray[np.float64]  # [Fx, Fy, Fz] in Newtons
    torques: NDArray[np.float64]  # [Tx, Ty, Tz] in Nm
    timestamp: float

    @property
    def force_magnitude(self) -> float:
        """Total force magnitude."""
        return float(np.linalg.norm(self.forces))

    @property
    def torque_magnitude(self) -> float:
        """Total torque magnitude."""
        return float(np.linalg.norm(self.torques))


class TissueForceSensor:
    """
    Force/torque sensor for tissue interaction.

    Provides force feedback for tissue differentiation
    and safe cutting control.
    """

    def __init__(self, sample_rate: float = 1000.0) -> None:
        self.sample_rate = sample_rate

        # Sensor specifications
        self.force_range = 500.0  # ±500N
        self.torque_range = 50.0  # ±50Nm
        self.resolution = 0.01  # 0.01N resolution
        self.noise_level = 0.1  # 0.1N RMS noise

        # Calibration
        self.calibration_matrix = np.eye(6)
        self.bias = np.zeros(6)

        # Buffer for signal processing
        self.buffer_size = 1000
        self.force_buffer: list[ForceReading] = []

        # Current reading
        self.current_reading = ForceReading(
            forces=np.zeros(3),
            torques=np.zeros(3),
            timestamp=0.0
        )

    def update(
        self,
        raw_forces: NDArray[np.float64],
        raw_torques: NDArray[np.float64],
        timestamp: float
    ) -> ForceReading:
        """
        Update sensor with new reading.

        Args:
            raw_forces: Raw force measurement [Fx, Fy, Fz]
            raw_torques: Raw torque measurement [Tx, Ty, Tz]
            timestamp: Measurement timestamp

        Returns:
            Processed force reading
        """
        # Apply calibration
        raw = np.concatenate([raw_forces, raw_torques])
        calibrated = self.calibration_matrix @ (raw - self.bias)

        self.current_reading = ForceReading(
            forces=calibrated[:3],
            torques=calibrated[3:],
            timestamp=timestamp
        )

        # Update buffer
        self.force_buffer.append(self.current_reading)
        if len(self.force_buffer) > self.buffer_size:
            self.force_buffer.pop(0)

        return self.current_reading

    def get_average_force(self, window_ms: float = 100) -> NDArray[np.float64]:
        """Get average force over time window."""
        window_samples = int(window_ms * self.sample_rate / 1000)
        if len(self.force_buffer) < window_samples:
            window_samples = len(self.force_buffer)

        if window_samples == 0:
            return np.zeros(3)

        forces = np.array([r.forces for r in self.force_buffer[-window_samples:]])
        return np.mean(forces, axis=0)

    def get_force_variance(self, window_ms: float = 100) -> NDArray[np.float64]:
        """Get force variance over time window."""
        window_samples = int(window_ms * self.sample_rate / 1000)
        if len(self.force_buffer) < window_samples:
            window_samples = len(self.force_buffer)

        if window_samples < 2:
            return np.zeros(3)

        forces = np.array([r.forces for r in self.force_buffer[-window_samples:]])
        return np.var(forces, axis=0)

    def compute_frequency_spectrum(
        self,
        axis: int = 2  # Z-axis by default
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute frequency spectrum of force signal.

        Returns:
            Tuple of (frequencies, amplitudes)
        """
        if len(self.force_buffer) < 64:
            return np.array([]), np.array([])

        # Get force signal for specified axis
        forces = np.array([r.forces[axis] for r in self.force_buffer])

        # FFT
        n = len(forces)
        fft = np.fft.fft(forces)
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)

        # Only positive frequencies
        mask = frequencies >= 0
        frequencies = frequencies[mask]
        amplitudes = np.abs(fft[mask]) / n

        return frequencies, amplitudes


class TissueClassifier:
    """
    Classify tissue type based on force measurements.

    Uses force magnitude, stiffness, and vibration signatures
    to identify tissue type.
    """

    def __init__(self) -> None:
        self.tissue_properties = {
            tissue: TissueProperties.get_default_properties(tissue)
            for tissue in TissueType
        }

        # Classification thresholds
        self.stiffness_history: list[float] = []
        self.history_size = 50

        # Current classification
        self.current_tissue = TissueType.AIR
        self.confidence = 0.0

    def estimate_stiffness(
        self,
        force_delta: NDArray[np.float64],
        position_delta: NDArray[np.float64]
    ) -> float:
        """
        Estimate contact stiffness from force and position changes.

        Args:
            force_delta: Change in force [N]
            position_delta: Change in position [m]

        Returns:
            Estimated stiffness [N/m]
        """
        position_magnitude = float(np.linalg.norm(position_delta))
        force_magnitude = float(np.linalg.norm(force_delta))

        if position_magnitude > 0.00001:  # 10 micrometers minimum
            stiffness = force_magnitude / position_magnitude
        else:
            stiffness = 0.0

        self.stiffness_history.append(stiffness)
        if len(self.stiffness_history) > self.history_size:
            self.stiffness_history.pop(0)

        return stiffness

    def classify(
        self,
        force_sensor: TissueForceSensor,
        current_stiffness: Optional[float] = None
    ) -> tuple[TissueType, float]:
        """
        Classify current tissue type.

        Args:
            force_sensor: Force sensor for measurements
            current_stiffness: Pre-computed stiffness (optional)

        Returns:
            Tuple of (tissue_type, confidence)
        """
        # Get average stiffness
        if current_stiffness is not None:
            stiffness = current_stiffness
        elif self.stiffness_history:
            stiffness = np.mean(self.stiffness_history[-10:])
        else:
            stiffness = 0.0

        # Get force variance (vibration indicator)
        variance = force_sensor.get_force_variance()
        total_variance = float(np.sum(variance))

        # Get dominant frequency
        frequencies, amplitudes = force_sensor.compute_frequency_spectrum()
        if len(frequencies) > 0 and len(amplitudes) > 0:
            dominant_freq = frequencies[np.argmax(amplitudes)]
        else:
            dominant_freq = 0.0

        # Classification logic
        if stiffness < 100:
            self.current_tissue = TissueType.AIR
            self.confidence = 0.9
        elif stiffness < 5000:
            self.current_tissue = TissueType.SOFT_TISSUE
            self.confidence = 0.7
        elif stiffness < 20000:
            self.current_tissue = TissueType.CARTILAGE
            self.confidence = 0.6
        elif stiffness < 100000:
            self.current_tissue = TissueType.CANCELLOUS_BONE
            self.confidence = 0.7
        elif stiffness < 500000:
            self.current_tissue = TissueType.CORTICAL_BONE
            self.confidence = 0.8
        else:
            # High stiffness - could be metal or cement
            if total_variance > 10:  # High vibration
                self.current_tissue = TissueType.METAL_IMPLANT
                self.confidence = 0.6
            else:
                self.current_tissue = TissueType.CEMENT
                self.confidence = 0.5

        return self.current_tissue, self.confidence


class ForceController:
    """
    Force controller for safe cutting operations.

    Implements force limiting and compliance control.
    """

    def __init__(self) -> None:
        # Force limits
        self.max_force = 100.0  # N
        self.max_torque = 10.0  # Nm
        self.soft_tissue_force_limit = 10.0  # N
        self.bone_force_limit = 50.0  # N

        # Compliance control
        self.compliance_enabled = True
        self.compliance_stiffness = 1000.0  # Virtual stiffness N/m
        self.compliance_damping = 50.0  # Virtual damping Ns/m

        # Safety callbacks
        self._force_limit_callbacks: list[Callable[[float, TissueType], None]] = []
        self._tissue_transition_callbacks: list[Callable[[TissueType, TissueType], None]] = []

        # State
        self.previous_tissue = TissueType.AIR
        self.force_limited = False

    def register_force_limit_callback(
        self,
        callback: Callable[[float, TissueType], None]
    ) -> None:
        """Register callback for force limit events."""
        self._force_limit_callbacks.append(callback)

    def register_tissue_transition_callback(
        self,
        callback: Callable[[TissueType, TissueType], None]
    ) -> None:
        """Register callback for tissue transitions."""
        self._tissue_transition_callbacks.append(callback)

    def check_force_limits(
        self,
        force_reading: ForceReading,
        tissue_type: TissueType
    ) -> tuple[bool, str]:
        """
        Check if forces are within safe limits.

        Args:
            force_reading: Current force reading
            tissue_type: Current tissue type

        Returns:
            Tuple of (is_safe, message)
        """
        force_magnitude = force_reading.force_magnitude
        torque_magnitude = force_reading.torque_magnitude

        # Check absolute limits
        if force_magnitude > self.max_force:
            self.force_limited = True
            for callback in self._force_limit_callbacks:
                callback(force_magnitude, tissue_type)
            return False, f"Force exceeded maximum: {force_magnitude:.1f}N > {self.max_force}N"

        if torque_magnitude > self.max_torque:
            self.force_limited = True
            return False, f"Torque exceeded maximum: {torque_magnitude:.2f}Nm > {self.max_torque}Nm"

        # Check tissue-specific limits
        if tissue_type == TissueType.SOFT_TISSUE:
            if force_magnitude > self.soft_tissue_force_limit:
                return False, f"Force too high for soft tissue: {force_magnitude:.1f}N"

        elif tissue_type in [TissueType.CARTILAGE, TissueType.CANCELLOUS_BONE, TissueType.CORTICAL_BONE]:
            if force_magnitude > self.bone_force_limit:
                return False, f"Force too high for bone: {force_magnitude:.1f}N"

        self.force_limited = False
        return True, "Forces within limits"

    def compute_compliance_velocity(
        self,
        force_reading: ForceReading,
        target_force: float = 0.0
    ) -> NDArray[np.float64]:
        """
        Compute velocity adjustment for compliance control.

        Implements admittance control: velocity = (force - target) / impedance

        Args:
            force_reading: Current force reading
            target_force: Target interaction force

        Returns:
            Velocity adjustment [m/s]
        """
        if not self.compliance_enabled:
            return np.zeros(3)

        # Force error
        force_error = force_reading.forces - target_force

        # Admittance control (simplified)
        velocity = force_error / self.compliance_stiffness

        # Limit velocity
        velocity_magnitude = float(np.linalg.norm(velocity))
        max_velocity = 0.01  # 10mm/s maximum
        if velocity_magnitude > max_velocity:
            velocity = velocity / velocity_magnitude * max_velocity

        return velocity

    def handle_tissue_transition(
        self,
        new_tissue: TissueType,
        classifier: TissueClassifier
    ) -> None:
        """Handle transition between tissue types."""
        if new_tissue != self.previous_tissue:
            for callback in self._tissue_transition_callbacks:
                callback(self.previous_tissue, new_tissue)
            self.previous_tissue = new_tissue

            # Adjust limits based on new tissue
            self._adjust_limits_for_tissue(new_tissue)

    def _adjust_limits_for_tissue(self, tissue: TissueType) -> None:
        """Adjust force limits based on tissue type."""
        if tissue == TissueType.SOFT_TISSUE:
            self.compliance_stiffness = 500.0  # More compliant
        elif tissue == TissueType.CARTILAGE:
            self.compliance_stiffness = 1000.0
        elif tissue in [TissueType.CANCELLOUS_BONE, TissueType.CORTICAL_BONE]:
            self.compliance_stiffness = 2000.0  # Stiffer
        elif tissue == TissueType.METAL_IMPLANT:
            self.compliance_stiffness = 5000.0  # Very stiff


@dataclass
class HapticBoundary:
    """Haptic boundary (virtual fixture) for safe cutting."""
    name: str
    points: list[NDArray[np.float64]]
    stiffness: float = 10000.0  # N/m
    damping: float = 100.0  # Ns/m
    is_active: bool = True

    def compute_boundary_force(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute force to keep position within boundary.

        Args:
            position: Current position
            velocity: Current velocity

        Returns:
            Boundary force
        """
        if not self.is_active or len(self.points) < 3:
            return np.zeros(3)

        # Find closest point on boundary
        min_distance = float('inf')
        closest_point = self.points[0]
        closest_normal = np.array([0, 0, 1])

        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]

            # Project onto edge
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)
            if edge_length > 0:
                t = np.clip(np.dot(position - p1, edge) / (edge_length ** 2), 0, 1)
                point_on_edge = p1 + t * edge
                distance = float(np.linalg.norm(position - point_on_edge))

                if distance < min_distance:
                    min_distance = distance
                    closest_point = point_on_edge
                    # Normal perpendicular to edge
                    edge_normalized = edge / edge_length
                    closest_normal = np.cross(edge_normalized, np.array([0, 0, 1]))
                    if np.linalg.norm(closest_normal) > 0:
                        closest_normal = closest_normal / np.linalg.norm(closest_normal)

        # Compute penetration
        direction_to_center = closest_point - position
        penetration = float(np.dot(direction_to_center, closest_normal))

        if penetration > 0:  # Inside boundary (no force needed)
            return np.zeros(3)

        # Spring-damper force
        spring_force = -self.stiffness * penetration * closest_normal
        damper_force = -self.damping * np.dot(velocity, closest_normal) * closest_normal

        return spring_force + damper_force
