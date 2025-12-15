"""Tremor cancellation and motion stabilization for neurosurgical robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray


class FilterType(Enum):
    """Types of tremor filters."""
    LOW_PASS = auto()
    NOTCH = auto()
    ADAPTIVE = auto()
    KALMAN = auto()
    WEIGHTED_MOVING_AVERAGE = auto()


@dataclass
class TremorCharacteristics:
    """Characteristics of physiological tremor."""
    # Typical physiological tremor: 8-12 Hz
    frequency_range: tuple[float, float] = (8.0, 12.0)  # Hz
    amplitude_range: tuple[float, float] = (0.00005, 0.0005)  # 50-500 micrometers
    dominant_frequency: float = 10.0  # Hz


@dataclass
class FilterState:
    """State for digital filters."""
    buffer: list[NDArray[np.float64]] = field(default_factory=list)
    buffer_size: int = 100
    coefficients: Optional[NDArray[np.float64]] = None
    output_history: list[NDArray[np.float64]] = field(default_factory=list)


class TremorFilter:
    """
    Digital filter for tremor removal.

    Implements various filtering strategies for removing
    physiological tremor from motion commands.
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.LOW_PASS,
        cutoff_frequency: float = 6.0,
        sample_rate: float = 1000.0
    ) -> None:
        self.filter_type = filter_type
        self.cutoff_frequency = cutoff_frequency
        self.sample_rate = sample_rate

        self.state = FilterState()
        self._design_filter()

    def _design_filter(self) -> None:
        """Design filter coefficients."""
        if self.filter_type == FilterType.LOW_PASS:
            self._design_butterworth_lowpass()
        elif self.filter_type == FilterType.NOTCH:
            self._design_notch_filter()
        elif self.filter_type == FilterType.WEIGHTED_MOVING_AVERAGE:
            self._design_weighted_average()

    def _design_butterworth_lowpass(self, order: int = 4) -> None:
        """Design Butterworth low-pass filter."""
        # Normalized cutoff frequency
        wc = 2 * self.cutoff_frequency / self.sample_rate

        # Design using bilinear transform (simplified)
        # In practice, would use scipy.signal.butter
        self.state.buffer_size = order + 1

        # Approximate coefficients for 4th order Butterworth
        alpha = np.tan(np.pi * wc / 2)
        alpha2 = alpha * alpha
        alpha4 = alpha2 * alpha2

        # Simplified coefficient calculation
        b = np.array([alpha4, 4*alpha4, 6*alpha4, 4*alpha4, alpha4])
        a = np.array([
            1 + 2.613*alpha + 3.414*alpha2 + 2.613*alpha*alpha2 + alpha4,
            -4 - 5.226*alpha + 5.226*alpha*alpha2 + 4*alpha4,
            6 - 6.828*alpha2 + 6*alpha4,
            -4 + 5.226*alpha - 5.226*alpha*alpha2 + 4*alpha4,
            1 - 2.613*alpha + 3.414*alpha2 - 2.613*alpha*alpha2 + alpha4
        ])

        # Normalize
        b = b / a[0]
        a = a / a[0]

        self.state.coefficients = np.vstack([b, a])

    def _design_notch_filter(self, notch_freq: float = 10.0, Q: float = 30.0) -> None:
        """Design notch filter to remove specific tremor frequency."""
        w0 = 2 * np.pi * notch_freq / self.sample_rate
        alpha = np.sin(w0) / (2 * Q)

        b = np.array([1, -2 * np.cos(w0), 1])
        a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])

        b = b / a[0]
        a = a / a[0]

        self.state.coefficients = np.vstack([b, a])
        self.state.buffer_size = 3

    def _design_weighted_average(self, window_size: int = 10) -> None:
        """Design weighted moving average filter."""
        # Gaussian-like weights
        weights = np.exp(-0.5 * np.linspace(-2, 0, window_size) ** 2)
        weights = weights / np.sum(weights)

        self.state.coefficients = weights.reshape(1, -1)
        self.state.buffer_size = window_size

    def filter(self, input_signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply filter to input signal.

        Args:
            input_signal: 3D position signal [x, y, z]

        Returns:
            Filtered position signal
        """
        self.state.buffer.append(input_signal.copy())

        if len(self.state.buffer) > self.state.buffer_size:
            self.state.buffer.pop(0)

        if len(self.state.buffer) < self.state.buffer_size:
            return input_signal

        if self.filter_type == FilterType.WEIGHTED_MOVING_AVERAGE:
            return self._apply_weighted_average()
        else:
            return self._apply_iir_filter()

    def _apply_weighted_average(self) -> NDArray[np.float64]:
        """Apply weighted moving average filter."""
        if self.state.coefficients is None:
            return self.state.buffer[-1]

        weights = self.state.coefficients[0]
        buffer_array = np.array(self.state.buffer[-len(weights):])

        return np.sum(weights[:, np.newaxis] * buffer_array, axis=0)

    def _apply_iir_filter(self) -> NDArray[np.float64]:
        """Apply IIR filter."""
        if self.state.coefficients is None:
            return self.state.buffer[-1]

        b = self.state.coefficients[0]
        a = self.state.coefficients[1]

        buffer_array = np.array(self.state.buffer)

        # Initialize output history if needed
        while len(self.state.output_history) < len(a) - 1:
            self.state.output_history.append(buffer_array[0])

        # Apply filter equation
        output = np.zeros(3)
        for i, bi in enumerate(b):
            if i < len(buffer_array):
                output += bi * buffer_array[-(i + 1)]

        for i, ai in enumerate(a[1:], 1):
            if i <= len(self.state.output_history):
                output -= ai * self.state.output_history[-i]

        self.state.output_history.append(output)
        if len(self.state.output_history) > len(a):
            self.state.output_history.pop(0)

        return output

    def reset(self) -> None:
        """Reset filter state."""
        self.state.buffer = []
        self.state.output_history = []


class AdaptiveTremorFilter:
    """
    Adaptive filter that learns tremor characteristics.

    Uses LMS (Least Mean Squares) algorithm to adapt to
    individual tremor patterns.
    """

    def __init__(
        self,
        filter_length: int = 32,
        step_size: float = 0.01,
        sample_rate: float = 1000.0
    ) -> None:
        self.filter_length = filter_length
        self.step_size = step_size
        self.sample_rate = sample_rate

        # Filter weights (3 channels for x, y, z)
        self.weights = np.zeros((3, filter_length))

        # Reference signal buffer (typically a delayed version of input)
        self.reference_buffer: list[NDArray[np.float64]] = []

        # Adaptation control
        self.is_adapting = True
        self.adaptation_rate = step_size

    def filter(
        self,
        input_signal: NDArray[np.float64],
        reference_signal: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Apply adaptive filter.

        Args:
            input_signal: Noisy input signal
            reference_signal: Reference signal correlated with noise

        Returns:
            Filtered output signal
        """
        if reference_signal is None:
            # Use delayed input as reference (for tremor)
            reference_signal = input_signal

        self.reference_buffer.append(reference_signal.copy())
        if len(self.reference_buffer) > self.filter_length:
            self.reference_buffer.pop(0)

        if len(self.reference_buffer) < self.filter_length:
            return input_signal

        # Form reference vector
        ref_matrix = np.array(self.reference_buffer).T  # 3 x filter_length

        # Compute filter output (estimate of tremor)
        tremor_estimate = np.sum(self.weights * ref_matrix, axis=1)

        # Compute error (desired signal)
        error = input_signal - tremor_estimate

        # Update weights using LMS
        if self.is_adapting:
            for i in range(3):
                self.weights[i] += (
                    2 * self.adaptation_rate * error[i] * ref_matrix[i]
                )

        return error

    def set_adaptation_rate(self, rate: float) -> None:
        """Set adaptation rate (step size)."""
        self.adaptation_rate = np.clip(rate, 0.0001, 0.1)

    def enable_adaptation(self, enable: bool = True) -> None:
        """Enable or disable weight adaptation."""
        self.is_adapting = enable

    def get_estimated_tremor_frequency(self) -> float:
        """Estimate dominant tremor frequency from filter weights."""
        # FFT of filter weights
        avg_weights = np.mean(self.weights, axis=0)
        fft = np.fft.fft(avg_weights)
        freqs = np.fft.fftfreq(len(avg_weights), 1 / self.sample_rate)

        # Find peak in physiological tremor range
        mask = (freqs > 4) & (freqs < 15)
        if np.any(mask):
            masked_fft = np.abs(fft[mask])
            masked_freqs = freqs[mask]
            peak_idx = np.argmax(masked_fft)
            return float(masked_freqs[peak_idx])

        return 10.0  # Default


class TremorCancellation:
    """
    Complete tremor cancellation system.

    Combines multiple filtering strategies for optimal
    tremor removal while preserving intended motion.
    """

    def __init__(self, sample_rate: float = 1000.0) -> None:
        self.sample_rate = sample_rate

        # Multi-stage filtering
        self.lowpass_filter = TremorFilter(
            FilterType.LOW_PASS,
            cutoff_frequency=5.0,
            sample_rate=sample_rate
        )
        self.notch_filter = TremorFilter(
            FilterType.NOTCH,
            cutoff_frequency=10.0,
            sample_rate=sample_rate
        )
        self.adaptive_filter = AdaptiveTremorFilter(
            filter_length=32,
            step_size=0.01,
            sample_rate=sample_rate
        )

        # Tremor detection
        self.tremor_detected = False
        self.tremor_amplitude = 0.0
        self.detection_threshold = 0.0001  # 0.1mm

        # Statistics
        self.input_variance_history: list[float] = []
        self.output_variance_history: list[float] = []

    def process(self, input_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Process position signal to cancel tremor.

        Args:
            input_position: Raw position signal [x, y, z]

        Returns:
            Tremor-cancelled position signal
        """
        # Stage 1: Notch filter to remove dominant frequency
        stage1 = self.notch_filter.filter(input_position)

        # Stage 2: Adaptive filter for residual tremor
        stage2 = self.adaptive_filter.filter(stage1)

        # Stage 3: Low-pass filter for smoothing
        output = self.lowpass_filter.filter(stage2)

        # Update statistics
        self._update_statistics(input_position, output)

        return output

    def _update_statistics(
        self,
        input_signal: NDArray[np.float64],
        output_signal: NDArray[np.float64]
    ) -> None:
        """Update tremor statistics."""
        # Track signal variance
        input_var = float(np.var(input_signal))
        output_var = float(np.var(output_signal))

        self.input_variance_history.append(input_var)
        self.output_variance_history.append(output_var)

        # Keep history bounded
        max_history = 1000
        if len(self.input_variance_history) > max_history:
            self.input_variance_history.pop(0)
            self.output_variance_history.pop(0)

        # Detect tremor
        if len(self.input_variance_history) > 10:
            avg_input_var = np.mean(self.input_variance_history[-10:])
            self.tremor_amplitude = np.sqrt(avg_input_var)
            self.tremor_detected = self.tremor_amplitude > self.detection_threshold

    def get_cancellation_ratio(self) -> float:
        """Get ratio of output to input variance (lower is better)."""
        if not self.input_variance_history or not self.output_variance_history:
            return 1.0

        avg_input = np.mean(self.input_variance_history[-100:])
        avg_output = np.mean(self.output_variance_history[-100:])

        if avg_input > 0:
            return avg_output / avg_input
        return 1.0

    def reset(self) -> None:
        """Reset all filters."""
        self.lowpass_filter.reset()
        self.notch_filter.reset()
        self.input_variance_history = []
        self.output_variance_history = []


class MotionStabilizer:
    """
    Motion stabilization for steady-hand microsurgery.

    Provides position locking and scaled motion control
    for ultra-precise positioning.
    """

    def __init__(self) -> None:
        self.tremor_cancellation = TremorCancellation()

        # Position locking
        self.is_locked = False
        self.locked_position: Optional[NDArray[np.float64]] = None
        self.lock_radius = 0.0005  # 0.5mm lock radius

        # Motion scaling
        self.motion_scale = 0.1  # 10:1 reduction
        self.reference_position: Optional[NDArray[np.float64]] = None

        # Velocity limiting
        self.max_velocity = 0.001  # 1mm/s
        self.last_output: Optional[NDArray[np.float64]] = None
        self.last_time: Optional[float] = None

        # Dead zone for fine positioning
        self.dead_zone_radius = 0.00005  # 50 micrometers

    def process(
        self,
        input_position: NDArray[np.float64],
        current_time: float
    ) -> NDArray[np.float64]:
        """
        Process input position with stabilization.

        Args:
            input_position: Raw input position
            current_time: Current timestamp in seconds

        Returns:
            Stabilized output position
        """
        # Apply tremor cancellation
        filtered = self.tremor_cancellation.process(input_position)

        # Handle position locking
        if self.is_locked and self.locked_position is not None:
            distance = float(np.linalg.norm(filtered - self.locked_position))
            if distance < self.lock_radius:
                return self.locked_position.copy()
            else:
                # Breakout from lock
                self.is_locked = False

        # Apply motion scaling
        if self.reference_position is not None:
            delta = filtered - self.reference_position
            scaled_delta = delta * self.motion_scale

            # Apply dead zone
            delta_magnitude = float(np.linalg.norm(scaled_delta))
            if delta_magnitude < self.dead_zone_radius:
                scaled_delta = np.zeros(3)

            output = self.locked_position + scaled_delta if self.locked_position is not None else filtered
        else:
            output = filtered

        # Apply velocity limiting
        output = self._apply_velocity_limit(output, current_time)

        return output

    def _apply_velocity_limit(
        self,
        target_position: NDArray[np.float64],
        current_time: float
    ) -> NDArray[np.float64]:
        """Limit velocity to maximum safe value."""
        if self.last_output is None or self.last_time is None:
            self.last_output = target_position.copy()
            self.last_time = current_time
            return target_position

        dt = current_time - self.last_time
        if dt <= 0:
            return self.last_output

        delta = target_position - self.last_output
        velocity = delta / dt
        speed = float(np.linalg.norm(velocity))

        if speed > self.max_velocity:
            # Scale velocity to maximum
            velocity = velocity / speed * self.max_velocity
            output = self.last_output + velocity * dt
        else:
            output = target_position

        self.last_output = output.copy()
        self.last_time = current_time

        return output

    def lock_position(self, position: Optional[NDArray[np.float64]] = None) -> None:
        """Lock to current or specified position."""
        if position is not None:
            self.locked_position = position.copy()
        elif self.last_output is not None:
            self.locked_position = self.last_output.copy()

        self.is_locked = True

    def unlock_position(self) -> None:
        """Unlock position."""
        self.is_locked = False

    def set_reference_position(self, position: NDArray[np.float64]) -> None:
        """Set reference for scaled motion."""
        self.reference_position = position.copy()
        self.locked_position = position.copy()

    def set_motion_scale(self, scale: float) -> None:
        """Set motion scaling factor (0.01 to 1.0)."""
        self.motion_scale = np.clip(scale, 0.01, 1.0)

    def set_max_velocity(self, velocity: float) -> None:
        """Set maximum velocity in m/s."""
        self.max_velocity = max(0.0001, velocity)  # Minimum 0.1mm/s

    def get_stabilization_stats(self) -> dict[str, float]:
        """Get stabilization performance statistics."""
        return {
            "tremor_detected": float(self.tremor_cancellation.tremor_detected),
            "tremor_amplitude_um": self.tremor_cancellation.tremor_amplitude * 1e6,
            "cancellation_ratio": self.tremor_cancellation.get_cancellation_ratio(),
            "motion_scale": self.motion_scale,
            "is_locked": float(self.is_locked),
        }
