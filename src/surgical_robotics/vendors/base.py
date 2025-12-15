"""Base classes for vendor integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose, RobotState, SafetyLevel


class ConnectionStatus(Enum):
    """Connection status with vendor system."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATED = auto()
    ERROR = auto()
    TIMEOUT = auto()


class VendorType(Enum):
    """Supported vendor types."""
    INTUITIVE_DAVINCI = auto()
    MEDTRONIC_HUGO = auto()
    MEDTRONIC_MAZOR = auto()
    STRYKER_MAKO = auto()
    ZIMMER_ROSA = auto()
    SMITH_NEPHEW_CORI = auto()
    BRAINLAB = auto()
    GENERIC = auto()


@dataclass
class VendorCapabilities:
    """Capabilities of a vendor system."""
    vendor_type: VendorType
    vendor_name: str
    model_name: str

    # Hardware capabilities
    num_arms: int = 1
    has_haptic_feedback: bool = False
    has_force_sensing: bool = False
    has_vision_system: bool = False
    has_navigation: bool = False

    # Degrees of freedom
    arm_dof: int = 6
    wrist_dof: int = 3

    # Precision specifications
    positioning_accuracy: float = 0.001  # meters
    repeatability: float = 0.0005  # meters

    # Communication
    supports_ros: bool = False
    supports_ros2: bool = False
    supports_udp: bool = True
    supports_tcp: bool = True

    # Safety
    max_speed: float = 0.5  # m/s
    max_force: float = 50.0  # N
    has_collision_detection: bool = True

    # Imaging integration
    supports_ct: bool = False
    supports_mri: bool = False
    supports_fluoroscopy: bool = False
    supports_ultrasound: bool = False


@dataclass
class VendorConfig:
    """Configuration for vendor connection."""
    host: str = "localhost"
    port: int = 5000
    timeout: float = 5.0
    retry_count: int = 3
    retry_delay: float = 1.0

    # Authentication
    api_key: Optional[str] = None
    certificate_path: Optional[str] = None

    # Protocol settings
    protocol: str = "tcp"  # "tcp", "udp", "ros", "ros2"
    message_format: str = "json"  # "json", "protobuf", "binary"

    # Performance
    control_rate: float = 1000.0  # Hz
    telemetry_rate: float = 100.0  # Hz

    # Custom settings
    custom_settings: dict = field(default_factory=dict)


@dataclass
class VendorTelemetry:
    """Telemetry data from vendor system."""
    timestamp: float
    robot_state: RobotState
    safety_level: SafetyLevel

    # Arm states
    arm_poses: list[Pose] = field(default_factory=list)
    joint_positions: list[NDArray[np.float64]] = field(default_factory=list)
    joint_velocities: list[NDArray[np.float64]] = field(default_factory=list)
    joint_torques: list[NDArray[np.float64]] = field(default_factory=list)

    # Force/torque
    end_effector_forces: list[NDArray[np.float64]] = field(default_factory=list)
    end_effector_torques: list[NDArray[np.float64]] = field(default_factory=list)

    # System health
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 25.0
    error_codes: list[int] = field(default_factory=list)


class VendorInterface(ABC):
    """
    Abstract base class for vendor system integration.

    Provides a unified interface for interacting with different
    surgical robot manufacturers' systems.
    """

    def __init__(self, config: Optional[VendorConfig] = None) -> None:
        self.config = config or VendorConfig()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.capabilities: Optional[VendorCapabilities] = None
        self.last_telemetry: Optional[VendorTelemetry] = None

        # Callbacks
        self._on_connect_callbacks: list[Callable[[], None]] = []
        self._on_disconnect_callbacks: list[Callable[[], None]] = []
        self._on_error_callbacks: list[Callable[[str], None]] = []
        self._on_telemetry_callbacks: list[Callable[[VendorTelemetry], None]] = []
        self._on_state_change_callbacks: list[Callable[[RobotState], None]] = []

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to vendor system."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from vendor system."""
        pass

    @abstractmethod
    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with vendor system."""
        pass

    @abstractmethod
    def get_capabilities(self) -> VendorCapabilities:
        """Get system capabilities."""
        pass

    @abstractmethod
    def get_telemetry(self) -> VendorTelemetry:
        """Get current telemetry data."""
        pass

    @abstractmethod
    def send_command(self, command: dict) -> dict:
        """Send command to vendor system."""
        pass

    @abstractmethod
    def set_arm_pose(self, arm_index: int, pose: Pose) -> bool:
        """Set target pose for an arm."""
        pass

    @abstractmethod
    def set_joint_positions(
        self, arm_index: int, positions: NDArray[np.float64]
    ) -> bool:
        """Set target joint positions."""
        pass

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Trigger emergency stop."""
        pass

    @abstractmethod
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop."""
        pass

    # Callback registration
    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register connection callback."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register disconnection callback."""
        self._on_disconnect_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]) -> None:
        """Register error callback."""
        self._on_error_callbacks.append(callback)

    def on_telemetry(self, callback: Callable[[VendorTelemetry], None]) -> None:
        """Register telemetry callback."""
        self._on_telemetry_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[RobotState], None]) -> None:
        """Register state change callback."""
        self._on_state_change_callbacks.append(callback)

    # Helper methods for triggering callbacks
    def _trigger_connect(self) -> None:
        for cb in self._on_connect_callbacks:
            cb()

    def _trigger_disconnect(self) -> None:
        for cb in self._on_disconnect_callbacks:
            cb()

    def _trigger_error(self, message: str) -> None:
        for cb in self._on_error_callbacks:
            cb(message)

    def _trigger_telemetry(self, telemetry: VendorTelemetry) -> None:
        self.last_telemetry = telemetry
        for cb in self._on_telemetry_callbacks:
            cb(telemetry)

    def _trigger_state_change(self, state: RobotState) -> None:
        for cb in self._on_state_change_callbacks:
            cb(state)

    @property
    def is_connected(self) -> bool:
        """Check if connected to vendor system."""
        return self.connection_status in [
            ConnectionStatus.CONNECTED,
            ConnectionStatus.AUTHENTICATED
        ]


class VendorSimulator:
    """
    Simulator for vendor systems.

    Used for testing and development without actual hardware.
    """

    def __init__(self, capabilities: VendorCapabilities) -> None:
        self.capabilities = capabilities
        self.robot_state = RobotState.IDLE
        self.safety_level = SafetyLevel.NORMAL

        # Simulated arm states
        self.arm_poses = [Pose.identity() for _ in range(capabilities.num_arms)]
        self.joint_positions = [
            np.zeros(capabilities.arm_dof) for _ in range(capabilities.num_arms)
        ]

    def simulate_telemetry(self, timestamp: float) -> VendorTelemetry:
        """Generate simulated telemetry."""
        return VendorTelemetry(
            timestamp=timestamp,
            robot_state=self.robot_state,
            safety_level=self.safety_level,
            arm_poses=self.arm_poses.copy(),
            joint_positions=[jp.copy() for jp in self.joint_positions],
            joint_velocities=[np.zeros_like(jp) for jp in self.joint_positions],
            joint_torques=[np.zeros_like(jp) for jp in self.joint_positions],
            end_effector_forces=[np.zeros(3) for _ in range(self.capabilities.num_arms)],
            end_effector_torques=[np.zeros(3) for _ in range(self.capabilities.num_arms)],
            cpu_usage=np.random.uniform(10, 30),
            memory_usage=np.random.uniform(20, 40),
            temperature=25.0 + np.random.uniform(-2, 2),
        )

    def apply_command(self, arm_index: int, target_pose: Pose) -> bool:
        """Apply movement command in simulation."""
        if arm_index >= len(self.arm_poses):
            return False

        # Simple interpolation toward target
        current = self.arm_poses[arm_index]
        alpha = 0.1  # Interpolation factor

        new_position = current.position + alpha * (target_pose.position - current.position)
        self.arm_poses[arm_index] = Pose(
            position=new_position,
            orientation=target_pose.orientation
        )

        return True
