"""Intuitive Surgical Da Vinci system integration."""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.vendors.base import (
    VendorInterface,
    VendorCapabilities,
    VendorConfig,
    VendorTelemetry,
    VendorType,
    VendorSimulator,
    ConnectionStatus,
)
from surgical_robotics.core.base import Pose, RobotState, SafetyLevel


class DaVinciModel(Enum):
    """Da Vinci system models."""
    XI = auto()  # Da Vinci Xi
    X = auto()  # Da Vinci X
    SP = auto()  # Da Vinci SP (Single Port)
    SI = auto()  # Da Vinci Si (Legacy)


class DaVinciArmType(Enum):
    """Da Vinci arm types."""
    PSM1 = auto()  # Patient Side Manipulator 1
    PSM2 = auto()
    PSM3 = auto()
    PSM4 = auto()
    ECM = auto()  # Endoscope Camera Manipulator


@dataclass
class DaVinciInstrumentInfo:
    """Information about installed Da Vinci instrument."""
    arm_type: DaVinciArmType
    instrument_name: str
    instrument_id: str
    uses_remaining: int
    is_sterile: bool
    has_energy: bool
    energy_type: Optional[str] = None


class IntuitiveDaVinciInterface(VendorInterface):
    """
    Interface for Intuitive Surgical Da Vinci systems.

    Supports Da Vinci Xi, X, SP, and Si models.
    Provides access to:
    - Multi-arm control (PSM1-4, ECM)
    - Instrument management
    - Vision system
    - Teleoperation state
    """

    def __init__(
        self,
        config: Optional[VendorConfig] = None,
        model: DaVinciModel = DaVinciModel.XI
    ) -> None:
        super().__init__(config)
        self.model = model
        self._setup_capabilities()

        # Simulation mode
        self._simulator: Optional[VendorSimulator] = None
        self.simulation_mode = False

        # System state
        self.instruments: dict[DaVinciArmType, Optional[DaVinciInstrumentInfo]] = {}
        self.teleoperation_active = False
        self.clutch_engaged = False
        self.camera_control_active = False

        # Motion scaling
        self.motion_scale = 0.5  # Default 2:1 scaling

    def _setup_capabilities(self) -> None:
        """Set up Da Vinci capabilities based on model."""
        num_arms = 4 if self.model in [DaVinciModel.XI, DaVinciModel.X] else 3

        self.capabilities = VendorCapabilities(
            vendor_type=VendorType.INTUITIVE_DAVINCI,
            vendor_name="Intuitive Surgical",
            model_name=f"Da Vinci {self.model.name}",
            num_arms=num_arms + 1,  # PSMs + ECM
            has_haptic_feedback=True,
            has_force_sensing=True,
            has_vision_system=True,
            has_navigation=False,
            arm_dof=7,
            wrist_dof=3,
            positioning_accuracy=0.001,
            repeatability=0.0005,
            supports_ros=True,
            supports_ros2=True,
            max_speed=0.3,
            max_force=15.0,
            has_collision_detection=True,
            supports_fluoroscopy=True,
        )

    def connect(self) -> bool:
        """Connect to Da Vinci system."""
        self.connection_status = ConnectionStatus.CONNECTING

        try:
            if self.simulation_mode:
                self._simulator = VendorSimulator(self.capabilities)
                self.connection_status = ConnectionStatus.CONNECTED
                self._trigger_connect()
                return True

            # Real connection logic would go here
            # This would use Intuitive's ISI API or dVRK
            self.connection_status = ConnectionStatus.CONNECTED
            self._trigger_connect()
            return True

        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self._trigger_error(str(e))
            return False

    def disconnect(self) -> None:
        """Disconnect from Da Vinci system."""
        self.connection_status = ConnectionStatus.DISCONNECTED
        self._simulator = None
        self._trigger_disconnect()

    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with Da Vinci system."""
        if not self.is_connected:
            return False

        # Verify credentials
        if "api_key" in credentials:
            self.connection_status = ConnectionStatus.AUTHENTICATED
            return True

        return False

    def get_capabilities(self) -> VendorCapabilities:
        """Get Da Vinci capabilities."""
        return self.capabilities

    def get_telemetry(self) -> VendorTelemetry:
        """Get current telemetry from Da Vinci."""
        if self._simulator:
            return self._simulator.simulate_telemetry(time.time())

        # Real telemetry retrieval would go here
        return VendorTelemetry(
            timestamp=time.time(),
            robot_state=RobotState.READY,
            safety_level=SafetyLevel.NORMAL,
        )

    def send_command(self, command: dict) -> dict:
        """Send command to Da Vinci system."""
        cmd_type = command.get("type", "")
        response = {"success": False, "error": None}

        if cmd_type == "set_motion_scale":
            self.motion_scale = command.get("scale", 0.5)
            response["success"] = True

        elif cmd_type == "engage_clutch":
            self.clutch_engaged = command.get("engaged", False)
            response["success"] = True

        elif cmd_type == "start_teleoperation":
            if self._can_start_teleoperation():
                self.teleoperation_active = True
                response["success"] = True
            else:
                response["error"] = "Cannot start teleoperation - check system state"

        elif cmd_type == "stop_teleoperation":
            self.teleoperation_active = False
            response["success"] = True

        return response

    def set_arm_pose(self, arm_index: int, pose: Pose) -> bool:
        """Set target pose for Da Vinci arm."""
        if not self.is_connected:
            return False

        if self._simulator:
            return self._simulator.apply_command(arm_index, pose)

        # Real arm control would go here
        return True

    def set_joint_positions(
        self, arm_index: int, positions: NDArray[np.float64]
    ) -> bool:
        """Set joint positions for Da Vinci arm."""
        if not self.is_connected:
            return False

        if self._simulator and arm_index < len(self._simulator.joint_positions):
            self._simulator.joint_positions[arm_index] = positions.copy()
            return True

        return True

    def emergency_stop(self) -> bool:
        """Trigger emergency stop."""
        self.teleoperation_active = False
        if self._simulator:
            self._simulator.robot_state = RobotState.EMERGENCY_STOP
            self._simulator.safety_level = SafetyLevel.CRITICAL
        self._trigger_state_change(RobotState.EMERGENCY_STOP)
        return True

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop."""
        if self._simulator:
            self._simulator.robot_state = RobotState.IDLE
            self._simulator.safety_level = SafetyLevel.NORMAL
        self._trigger_state_change(RobotState.IDLE)
        return True

    # Da Vinci specific methods
    def enable_simulation_mode(self) -> None:
        """Enable simulation mode for testing."""
        self.simulation_mode = True

    def get_instrument_info(self, arm_type: DaVinciArmType) -> Optional[DaVinciInstrumentInfo]:
        """Get information about installed instrument."""
        return self.instruments.get(arm_type)

    def set_instrument(
        self,
        arm_type: DaVinciArmType,
        instrument_name: str,
        instrument_id: str
    ) -> bool:
        """Register instrument installation."""
        self.instruments[arm_type] = DaVinciInstrumentInfo(
            arm_type=arm_type,
            instrument_name=instrument_name,
            instrument_id=instrument_id,
            uses_remaining=10,  # Typical use count
            is_sterile=True,
            has_energy="cautery" in instrument_name.lower() or "bipolar" in instrument_name.lower(),
            energy_type="monopolar" if "monopolar" in instrument_name.lower() else "bipolar" if "bipolar" in instrument_name.lower() else None
        )
        return True

    def set_motion_scale(self, scale: float) -> None:
        """Set motion scaling factor (0.2 to 1.0)."""
        self.motion_scale = np.clip(scale, 0.2, 1.0)

    def get_motion_scale(self) -> float:
        """Get current motion scaling factor."""
        return self.motion_scale

    def start_teleoperation(self) -> bool:
        """Start teleoperation mode."""
        if self._can_start_teleoperation():
            self.teleoperation_active = True
            self._trigger_state_change(RobotState.OPERATING)
            return True
        return False

    def stop_teleoperation(self) -> None:
        """Stop teleoperation mode."""
        self.teleoperation_active = False
        self._trigger_state_change(RobotState.READY)

    def engage_clutch(self, engaged: bool) -> None:
        """Engage or disengage clutch."""
        self.clutch_engaged = engaged

    def enable_camera_control(self, enabled: bool) -> None:
        """Enable camera control mode."""
        self.camera_control_active = enabled

    def swap_arms(self, arm1: DaVinciArmType, arm2: DaVinciArmType) -> bool:
        """Swap instrument control between arms."""
        if arm1 == DaVinciArmType.ECM or arm2 == DaVinciArmType.ECM:
            return False  # Cannot swap with camera arm

        # Swap instrument registrations
        inst1 = self.instruments.get(arm1)
        inst2 = self.instruments.get(arm2)

        if inst1:
            inst1.arm_type = arm2
        if inst2:
            inst2.arm_type = arm1

        self.instruments[arm1] = inst2
        self.instruments[arm2] = inst1

        return True

    def _can_start_teleoperation(self) -> bool:
        """Check if teleoperation can be started."""
        if not self.is_connected:
            return False

        # Check at least one PSM has instrument
        has_instrument = any(
            arm_type != DaVinciArmType.ECM and info is not None
            for arm_type, info in self.instruments.items()
        )

        return has_instrument

    def get_endoscope_pose(self) -> Optional[Pose]:
        """Get current endoscope pose."""
        telemetry = self.get_telemetry()
        ecm_index = self.capabilities.num_arms - 1  # ECM is last arm

        if telemetry.arm_poses and ecm_index < len(telemetry.arm_poses):
            return telemetry.arm_poses[ecm_index]

        return None

    def focus_endoscope(self, distance: float) -> bool:
        """Set endoscope focus distance."""
        return self.send_command({
            "type": "endoscope_focus",
            "distance": distance
        }).get("success", False)

    def set_endoscope_zoom(self, zoom_level: float) -> bool:
        """Set endoscope zoom level."""
        return self.send_command({
            "type": "endoscope_zoom",
            "zoom": np.clip(zoom_level, 1.0, 10.0)
        }).get("success", False)
