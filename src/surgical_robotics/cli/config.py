"""Configuration management for surgical robotics."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
import os

import numpy as np


class RobotType(str, Enum):
    """Supported robot types."""
    DAVINCI = "davinci"
    DAVINCI_XI = "davinci_xi"
    DAVINCI_X = "davinci_x"
    HUGO = "hugo"
    MAKO = "mako"
    ROSA = "rosa"
    MAZOR = "mazor"
    NEUROMATE = "neuromate"
    CUSTOM = "custom"


class ControlMode(str, Enum):
    """Robot control modes."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    TELEOPERATION = "teleoperation"


@dataclass
class JointConfig:
    """Configuration for a single joint."""
    name: str
    type: str = "revolute"  # revolute, prismatic, fixed

    # Limits
    position_min: float = -3.14159
    position_max: float = 3.14159
    velocity_max: float = 2.0
    torque_max: float = 100.0
    acceleration_max: float = 10.0

    # DH parameters
    d: float = 0.0
    a: float = 0.0
    alpha: float = 0.0
    theta_offset: float = 0.0

    # Dynamics
    inertia: float = 0.1
    damping: float = 0.1
    friction: float = 0.05

    # Calibration
    home_position: float = 0.0
    encoder_offset: float = 0.0


@dataclass
class SafetyConfig:
    """Safety configuration."""
    # Emergency stop
    enable_e_stop: bool = True
    e_stop_reaction_time: float = 0.01  # seconds

    # Force limits
    max_contact_force: float = 50.0  # N
    max_tool_force: float = 20.0  # N

    # Velocity limits
    max_cartesian_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 1.0  # rad/s

    # Workspace limits
    workspace_enabled: bool = True
    workspace_center: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.5]
    )
    workspace_radius: float = 0.5

    # Virtual fixtures
    virtual_fixtures_enabled: bool = True

    # Collision detection
    collision_detection_enabled: bool = True
    collision_threshold: float = 0.01  # meters

    # Watchdog
    watchdog_timeout: float = 0.1  # seconds
    heartbeat_required: bool = True


@dataclass
class CommunicationConfig:
    """Communication configuration."""
    # Control interface
    control_interface: str = "ethernet"  # ethernet, usb, can, ros2
    control_ip: str = "192.168.1.100"
    control_port: int = 30003

    # Real-time
    control_frequency: float = 1000.0  # Hz
    use_realtime: bool = True

    # ROS2
    ros2_enabled: bool = False
    ros2_namespace: str = ""
    ros2_qos_profile: str = "sensor_data"

    # DICOM
    dicom_enabled: bool = False
    dicom_ae_title: str = "SURGICAL_ROBOT"
    dicom_port: int = 11112

    # HL7
    hl7_enabled: bool = False
    hl7_host: str = "localhost"
    hl7_port: int = 2575


@dataclass
class CalibrationConfig:
    """Calibration configuration."""
    # Joint calibration
    joint_offsets: List[float] = field(default_factory=list)
    joint_scales: List[float] = field(default_factory=list)

    # Tool calibration
    tool_center_point: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    tool_orientation: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0]
    )

    # Registration
    registration_matrix: Optional[List[List[float]]] = None

    # Camera calibration
    camera_intrinsics: Optional[Dict[str, Any]] = None
    camera_extrinsics: Optional[Dict[str, Any]] = None

    # Last calibration
    last_calibration_date: Optional[str] = None
    calibration_valid: bool = False


@dataclass
class TelemetryConfig:
    """Telemetry and logging configuration."""
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "/var/log/surgical_robotics"
    log_max_size_mb: int = 100
    log_retention_days: int = 30

    # Metrics
    metrics_enabled: bool = True
    metrics_interval: float = 1.0  # seconds
    metrics_export: str = "prometheus"  # prometheus, statsd, none

    # Recording
    record_enabled: bool = False
    record_directory: str = "/data/recordings"
    record_format: str = "hdf5"  # hdf5, csv, rosbag

    # Alerts
    alerts_enabled: bool = True
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    enabled: bool = False
    physics_backend: str = "internal"  # internal, pybullet, mujoco
    time_step: float = 0.001
    real_time: bool = True
    render_enabled: bool = True
    render_rate: float = 60.0


@dataclass
class RobotConfig:
    """Complete robot configuration."""
    name: str = "surgical_robot"
    robot_type: RobotType = RobotType.CUSTOM
    serial_number: str = ""
    firmware_version: str = ""

    # Number of joints/arms
    num_joints: int = 7
    num_arms: int = 1

    # Joint configurations
    joints: List[JointConfig] = field(default_factory=list)

    # Sub-configurations
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Control
    default_control_mode: ControlMode = ControlMode.POSITION

    # Vendor-specific settings
    vendor_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize joints if not provided."""
        if not self.joints:
            self.joints = [
                JointConfig(name=f"joint_{i}")
                for i in range(self.num_joints)
            ]


@dataclass
class Configuration:
    """
    Top-level configuration for surgical robotics system.

    Includes robot, environment, and procedure settings.
    """
    version: str = "1.0.0"

    # Robot configuration
    robot: RobotConfig = field(default_factory=RobotConfig)

    # Multiple robots (for multi-arm systems)
    additional_robots: Dict[str, RobotConfig] = field(default_factory=dict)

    # Environment
    environment: Dict[str, Any] = field(default_factory=dict)

    # Procedure settings
    procedure_type: str = ""
    procedure_settings: Dict[str, Any] = field(default_factory=dict)

    # User preferences
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manages loading, saving, and validating configurations."""

    DEFAULT_CONFIG_PATHS = [
        Path.home() / ".surgical_robotics" / "config.json",
        Path("/etc/surgical_robotics/config.json"),
        Path("./config.json"),
    ]

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[Configuration] = None

    def load(
        self,
        path: Optional[Union[str, Path]] = None
    ) -> Configuration:
        """Load configuration from file."""
        if path:
            config_path = Path(path)
        elif self.config_path:
            config_path = self.config_path
        else:
            config_path = self._find_config_file()

        if config_path is None or not config_path.exists():
            # Return default configuration
            self._config = Configuration()
            return self._config

        with open(config_path, 'r') as f:
            data = json.load(f)

        self._config = self._dict_to_config(data)
        return self._config

    def save(
        self,
        config: Configuration,
        path: Optional[Union[str, Path]] = None
    ) -> bool:
        """Save configuration to file."""
        if path:
            config_path = Path(path)
        elif self.config_path:
            config_path = self.config_path
        else:
            config_path = self.DEFAULT_CONFIG_PATHS[0]

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config_to_dict(config)

        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return True

    def validate(self, config: Configuration) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check robot configuration
        if config.robot.num_joints < 1:
            issues.append("Robot must have at least 1 joint")

        if len(config.robot.joints) != config.robot.num_joints:
            issues.append(
                f"Joint count mismatch: {len(config.robot.joints)} "
                f"joints configured, {config.robot.num_joints} expected"
            )

        # Check safety limits
        if config.robot.safety.max_contact_force <= 0:
            issues.append("Max contact force must be positive")

        if config.robot.safety.watchdog_timeout <= 0:
            issues.append("Watchdog timeout must be positive")

        # Check communication
        if config.robot.communication.control_frequency <= 0:
            issues.append("Control frequency must be positive")

        # Check joint limits
        for joint in config.robot.joints:
            if joint.position_min >= joint.position_max:
                issues.append(
                    f"Joint {joint.name}: position_min must be less than position_max"
                )
            if joint.velocity_max <= 0:
                issues.append(
                    f"Joint {joint.name}: velocity_max must be positive"
                )

        return issues

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in default locations."""
        # Check environment variable first
        env_path = os.environ.get("SURGICAL_ROBOTICS_CONFIG")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        # Check default paths
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path

        return None

    def _config_to_dict(self, config: Configuration) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def convert(obj: Any) -> Any:
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(config)

    def _dict_to_config(self, data: Dict[str, Any]) -> Configuration:
        """Convert dictionary to configuration."""
        # Parse robot config
        robot_data = data.get('robot', {})
        joints_data = robot_data.pop('joints', [])

        joints = [
            JointConfig(**j) if isinstance(j, dict) else j
            for j in joints_data
        ]

        # Parse nested configs
        safety = SafetyConfig(**robot_data.pop('safety', {}))
        communication = CommunicationConfig(**robot_data.pop('communication', {}))
        calibration = CalibrationConfig(**robot_data.pop('calibration', {}))
        telemetry = TelemetryConfig(**robot_data.pop('telemetry', {}))
        simulation = SimulationConfig(**robot_data.pop('simulation', {}))

        # Handle enums
        if 'robot_type' in robot_data:
            robot_data['robot_type'] = RobotType(robot_data['robot_type'])
        if 'default_control_mode' in robot_data:
            robot_data['default_control_mode'] = ControlMode(
                robot_data['default_control_mode']
            )

        robot = RobotConfig(
            **robot_data,
            joints=joints,
            safety=safety,
            communication=communication,
            calibration=calibration,
            telemetry=telemetry,
            simulation=simulation
        )

        return Configuration(
            version=data.get('version', '1.0.0'),
            robot=robot,
            additional_robots=data.get('additional_robots', {}),
            environment=data.get('environment', {}),
            procedure_type=data.get('procedure_type', ''),
            procedure_settings=data.get('procedure_settings', {}),
            user_preferences=data.get('user_preferences', {})
        )


# Convenience functions
def load_config(path: Optional[Union[str, Path]] = None) -> Configuration:
    """Load configuration from file."""
    manager = ConfigurationManager(path)
    return manager.load()


def save_config(
    config: Configuration,
    path: Optional[Union[str, Path]] = None
) -> bool:
    """Save configuration to file."""
    manager = ConfigurationManager(path)
    return manager.save(config, path)


def create_default_config(robot_type: RobotType = RobotType.CUSTOM) -> Configuration:
    """Create default configuration for robot type."""
    if robot_type == RobotType.DAVINCI_XI:
        robot = RobotConfig(
            name="da_vinci_xi",
            robot_type=RobotType.DAVINCI_XI,
            num_joints=7,
            num_arms=4,
            safety=SafetyConfig(
                max_contact_force=30.0,
                max_tool_force=10.0,
            ),
            communication=CommunicationConfig(
                control_frequency=1000.0,
            ),
        )
    elif robot_type == RobotType.MAKO:
        robot = RobotConfig(
            name="stryker_mako",
            robot_type=RobotType.MAKO,
            num_joints=6,
            safety=SafetyConfig(
                max_contact_force=100.0,
                virtual_fixtures_enabled=True,
            ),
        )
    elif robot_type == RobotType.ROSA:
        robot = RobotConfig(
            name="zimmer_rosa",
            robot_type=RobotType.ROSA,
            num_joints=6,
            safety=SafetyConfig(
                max_contact_force=20.0,
            ),
        )
    else:
        robot = RobotConfig()

    return Configuration(robot=robot)
