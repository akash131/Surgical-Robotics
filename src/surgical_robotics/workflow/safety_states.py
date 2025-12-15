"""Safety state management for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Set
import time
from datetime import datetime


class SafetyState(Enum):
    """Safety system states."""
    NORMAL = "normal"  # All systems nominal
    WARNING = "warning"  # Non-critical issue detected
    REDUCED = "reduced"  # Operating with reduced capabilities
    STOPPED = "stopped"  # Motion stopped, can resume
    EMERGENCY = "emergency"  # Emergency stop active
    FAULT = "fault"  # System fault, requires intervention


class SafetyLevel(Enum):
    """Safety integrity levels."""
    LOW = 0  # Informational
    MEDIUM = 1  # Requires attention
    HIGH = 2  # Requires immediate action
    CRITICAL = 3  # Safety-critical


@dataclass
class SafetyViolation:
    """Records a safety violation."""
    id: str
    level: SafetyLevel
    category: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_action: Optional[str] = None

    # Context
    robot_state: Optional[Dict[str, Any]] = None
    joint_positions: Optional[List[float]] = None
    end_effector_pose: Optional[Dict[str, float]] = None

    def resolve(self, action: str = "") -> None:
        """Mark violation as resolved."""
        self.resolved = True
        self.resolution_time = datetime.now()
        self.resolution_action = action


class SafetyStateMachine:
    """
    Manages safety states and transitions.

    Implements IEC 62443 compliant safety state management.
    """

    def __init__(self) -> None:
        self._state = SafetyState.NORMAL
        self._previous_state = SafetyState.NORMAL

        # Active violations
        self._violations: Dict[str, SafetyViolation] = {}
        self._violation_history: List[SafetyViolation] = []

        # Monitored conditions
        self._conditions: Dict[str, bool] = {}
        self._condition_checks: Dict[str, Callable[[], bool]] = {}

        # Limits
        self._limits: Dict[str, Dict[str, float]] = {}

        # Callbacks
        self._on_state_change: Optional[Callable[[SafetyState, SafetyState], None]] = None
        self._on_violation: Optional[Callable[[SafetyViolation], None]] = None
        self._on_emergency: Optional[Callable[[], None]] = None

        # State entry times
        self._state_entry_time = time.time()

    @property
    def state(self) -> SafetyState:
        """Current safety state."""
        return self._state

    @property
    def is_safe_to_operate(self) -> bool:
        """Check if safe to operate."""
        return self._state in [SafetyState.NORMAL, SafetyState.WARNING]

    @property
    def is_emergency(self) -> bool:
        """Check if in emergency state."""
        return self._state == SafetyState.EMERGENCY

    @property
    def active_violations(self) -> List[SafetyViolation]:
        """Get active (unresolved) violations."""
        return [v for v in self._violations.values() if not v.resolved]

    def add_condition_check(
        self,
        name: str,
        check: Callable[[], bool]
    ) -> None:
        """Add condition to monitor."""
        self._condition_checks[name] = check
        self._conditions[name] = True

    def set_limit(
        self,
        name: str,
        warning: float,
        critical: float
    ) -> None:
        """Set warning and critical limits for a value."""
        self._limits[name] = {
            "warning": warning,
            "critical": critical
        }

    def check_limit(self, name: str, value: float) -> Optional[SafetyLevel]:
        """Check value against limits."""
        if name not in self._limits:
            return None

        limits = self._limits[name]

        if abs(value) >= limits["critical"]:
            return SafetyLevel.CRITICAL
        elif abs(value) >= limits["warning"]:
            return SafetyLevel.MEDIUM

        return None

    def update(self) -> SafetyState:
        """
        Update safety state based on conditions.

        Should be called periodically.
        """
        # Run all condition checks
        for name, check in self._condition_checks.items():
            try:
                self._conditions[name] = check()
            except Exception:
                self._conditions[name] = False

        # Evaluate state
        new_state = self._evaluate_state()

        if new_state != self._state:
            self._transition_to(new_state)

        return self._state

    def _evaluate_state(self) -> SafetyState:
        """Evaluate what state should be active."""
        # Check for active critical violations
        critical_violations = [
            v for v in self._violations.values()
            if not v.resolved and v.level == SafetyLevel.CRITICAL
        ]
        if critical_violations:
            return SafetyState.EMERGENCY

        # Check for high-level violations
        high_violations = [
            v for v in self._violations.values()
            if not v.resolved and v.level == SafetyLevel.HIGH
        ]
        if high_violations:
            return SafetyState.FAULT

        # Check conditions
        failed_conditions = [
            name for name, ok in self._conditions.items()
            if not ok
        ]
        if failed_conditions:
            return SafetyState.STOPPED

        # Check for warnings
        warnings = [
            v for v in self._violations.values()
            if not v.resolved and v.level in [SafetyLevel.LOW, SafetyLevel.MEDIUM]
        ]
        if warnings:
            return SafetyState.WARNING

        return SafetyState.NORMAL

    def _transition_to(self, new_state: SafetyState) -> None:
        """Transition to new safety state."""
        self._previous_state = self._state
        self._state = new_state
        self._state_entry_time = time.time()

        # Callback
        if self._on_state_change:
            self._on_state_change(self._previous_state, new_state)

        # Emergency callback
        if new_state == SafetyState.EMERGENCY and self._on_emergency:
            self._on_emergency()

    def report_violation(
        self,
        violation_id: str,
        level: SafetyLevel,
        category: str,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyViolation:
        """Report a safety violation."""
        violation = SafetyViolation(
            id=violation_id,
            level=level,
            category=category,
            description=description,
            robot_state=context
        )

        self._violations[violation_id] = violation
        self._violation_history.append(violation)

        if self._on_violation:
            self._on_violation(violation)

        # Trigger state update
        self.update()

        return violation

    def resolve_violation(
        self,
        violation_id: str,
        action: str = ""
    ) -> bool:
        """Resolve a safety violation."""
        if violation_id not in self._violations:
            return False

        self._violations[violation_id].resolve(action)

        # Trigger state update
        self.update()

        return True

    def clear_all_violations(self) -> None:
        """Clear all resolved violations."""
        self._violations = {
            k: v for k, v in self._violations.items()
            if not v.resolved
        }

    def trigger_emergency_stop(self, reason: str = "") -> None:
        """Trigger emergency stop."""
        self.report_violation(
            f"e_stop_{int(time.time())}",
            SafetyLevel.CRITICAL,
            "emergency_stop",
            f"Emergency stop triggered: {reason}"
        )

    def reset(self) -> bool:
        """
        Reset from emergency/fault state.

        Returns True if reset successful.
        """
        if self._state not in [SafetyState.EMERGENCY, SafetyState.FAULT]:
            return False

        # Check if all critical violations are resolved
        unresolved = [
            v for v in self._violations.values()
            if not v.resolved and v.level in [SafetyLevel.CRITICAL, SafetyLevel.HIGH]
        ]
        if unresolved:
            return False

        self._transition_to(SafetyState.STOPPED)
        return True

    def acknowledge(self) -> bool:
        """
        Acknowledge warnings and resume.

        Returns True if acknowledgment successful.
        """
        if self._state not in [SafetyState.WARNING, SafetyState.STOPPED]:
            return False

        # Mark low/medium violations as acknowledged
        for violation in self._violations.values():
            if violation.level in [SafetyLevel.LOW, SafetyLevel.MEDIUM]:
                violation.resolve("acknowledged")

        self.update()
        return True

    def set_callbacks(
        self,
        on_state_change: Optional[Callable[[SafetyState, SafetyState], None]] = None,
        on_violation: Optional[Callable[[SafetyViolation], None]] = None,
        on_emergency: Optional[Callable[[], None]] = None
    ) -> None:
        """Set safety callbacks."""
        self._on_state_change = on_state_change
        self._on_violation = on_violation
        self._on_emergency = on_emergency

    def get_status(self) -> Dict[str, Any]:
        """Get safety status."""
        return {
            "state": self._state.value,
            "previous_state": self._previous_state.value,
            "is_safe_to_operate": self.is_safe_to_operate,
            "is_emergency": self.is_emergency,
            "active_violation_count": len(self.active_violations),
            "conditions": dict(self._conditions),
            "time_in_state": time.time() - self._state_entry_time
        }


class EmergencyHandler:
    """
    Handles emergency situations in surgical robotics.

    Coordinates response to emergency conditions.
    """

    def __init__(self, safety_machine: SafetyStateMachine) -> None:
        self.safety = safety_machine

        # Emergency actions
        self._stop_actions: List[Callable[[], None]] = []
        self._recovery_actions: List[Callable[[], bool]] = []

        # State
        self._emergency_active = False
        self._emergency_start: Optional[datetime] = None
        self._emergency_cause: Optional[str] = None

        # Callbacks
        self._on_emergency_start: Optional[Callable[[str], None]] = None
        self._on_emergency_end: Optional[Callable[[float], None]] = None

        # Register with safety machine
        safety_machine.set_callbacks(
            on_emergency=self._handle_emergency
        )

    def register_stop_action(self, action: Callable[[], None]) -> None:
        """Register action to execute on emergency stop."""
        self._stop_actions.append(action)

    def register_recovery_action(self, action: Callable[[], bool]) -> None:
        """Register action for recovery sequence."""
        self._recovery_actions.append(action)

    def trigger_emergency(
        self,
        cause: str,
        source: str = "manual"
    ) -> None:
        """Trigger emergency stop."""
        self._emergency_active = True
        self._emergency_start = datetime.now()
        self._emergency_cause = cause

        # Report to safety machine
        self.safety.trigger_emergency_stop(f"{source}: {cause}")

        # Execute stop actions
        self._execute_stop_sequence()

        if self._on_emergency_start:
            self._on_emergency_start(cause)

    def _handle_emergency(self) -> None:
        """Internal handler for safety machine emergency."""
        if not self._emergency_active:
            self._emergency_active = True
            self._emergency_start = datetime.now()
            self._execute_stop_sequence()

    def _execute_stop_sequence(self) -> None:
        """Execute emergency stop sequence."""
        for action in self._stop_actions:
            try:
                action()
            except Exception as e:
                # Log but continue with other actions
                print(f"Stop action failed: {e}")

    def can_recover(self) -> bool:
        """Check if recovery is possible."""
        # Cannot recover if safety violations are unresolved
        if self.safety.active_violations:
            return False

        return True

    def recover(self) -> bool:
        """
        Attempt recovery from emergency.

        Returns True if recovery successful.
        """
        if not self.can_recover():
            return False

        # Execute recovery actions
        for action in self._recovery_actions:
            try:
                if not action():
                    return False
            except Exception:
                return False

        # Reset safety state
        if not self.safety.reset():
            return False

        # Calculate emergency duration
        duration = 0.0
        if self._emergency_start:
            duration = (datetime.now() - self._emergency_start).total_seconds()

        self._emergency_active = False
        self._emergency_start = None
        self._emergency_cause = None

        if self._on_emergency_end:
            self._on_emergency_end(duration)

        return True

    def get_emergency_info(self) -> Dict[str, Any]:
        """Get emergency information."""
        return {
            "active": self._emergency_active,
            "cause": self._emergency_cause,
            "start_time": self._emergency_start.isoformat() if self._emergency_start else None,
            "duration": (
                (datetime.now() - self._emergency_start).total_seconds()
                if self._emergency_start else 0
            ),
            "can_recover": self.can_recover(),
            "active_violations": [
                {
                    "id": v.id,
                    "level": v.level.name,
                    "description": v.description
                }
                for v in self.safety.active_violations
            ]
        }

    def set_callbacks(
        self,
        on_start: Optional[Callable[[str], None]] = None,
        on_end: Optional[Callable[[float], None]] = None
    ) -> None:
        """Set emergency callbacks."""
        self._on_emergency_start = on_start
        self._on_emergency_end = on_end


class SafetyMonitor:
    """
    Continuous safety monitoring for surgical robots.

    Monitors various safety parameters and conditions.
    """

    def __init__(self, safety_machine: SafetyStateMachine) -> None:
        self.safety = safety_machine

        # Monitoring parameters
        self._monitors: Dict[str, Dict[str, Any]] = {}
        self._monitor_interval = 0.01  # 100 Hz default
        self._last_check = time.time()

    def add_force_monitor(
        self,
        name: str,
        get_force: Callable[[], float],
        warning_limit: float,
        critical_limit: float
    ) -> None:
        """Add force monitoring."""
        self._monitors[f"force_{name}"] = {
            "type": "force",
            "getter": get_force,
            "warning": warning_limit,
            "critical": critical_limit
        }
        self.safety.set_limit(f"force_{name}", warning_limit, critical_limit)

    def add_velocity_monitor(
        self,
        name: str,
        get_velocity: Callable[[], float],
        limit: float
    ) -> None:
        """Add velocity monitoring."""
        self._monitors[f"velocity_{name}"] = {
            "type": "velocity",
            "getter": get_velocity,
            "limit": limit
        }
        self.safety.set_limit(f"velocity_{name}", limit * 0.9, limit)

    def add_position_monitor(
        self,
        name: str,
        get_position: Callable[[], float],
        min_limit: float,
        max_limit: float
    ) -> None:
        """Add position monitoring (joint limits)."""
        self._monitors[f"position_{name}"] = {
            "type": "position",
            "getter": get_position,
            "min": min_limit,
            "max": max_limit
        }

    def add_watchdog(
        self,
        name: str,
        get_heartbeat: Callable[[], float],
        timeout: float
    ) -> None:
        """Add watchdog monitoring."""
        self._monitors[f"watchdog_{name}"] = {
            "type": "watchdog",
            "getter": get_heartbeat,
            "timeout": timeout,
            "last_beat": time.time()
        }

    def check(self) -> List[str]:
        """
        Run all safety checks.

        Returns list of violation IDs.
        """
        current_time = time.time()

        if current_time - self._last_check < self._monitor_interval:
            return []

        self._last_check = current_time
        violations = []

        for name, monitor in self._monitors.items():
            try:
                violation = self._check_monitor(name, monitor)
                if violation:
                    violations.append(violation)
            except Exception as e:
                # Monitor failure is itself a violation
                self.safety.report_violation(
                    f"monitor_fail_{name}",
                    SafetyLevel.HIGH,
                    "monitor_failure",
                    f"Monitor {name} failed: {e}"
                )
                violations.append(f"monitor_fail_{name}")

        # Update safety state
        self.safety.update()

        return violations

    def _check_monitor(
        self,
        name: str,
        monitor: Dict[str, Any]
    ) -> Optional[str]:
        """Check single monitor."""
        monitor_type = monitor["type"]

        if monitor_type == "force":
            value = monitor["getter"]()
            level = self.safety.check_limit(name, value)
            if level:
                return self._report_limit_violation(name, value, level, "force")

        elif monitor_type == "velocity":
            value = monitor["getter"]()
            level = self.safety.check_limit(name, value)
            if level:
                return self._report_limit_violation(name, value, level, "velocity")

        elif monitor_type == "position":
            value = monitor["getter"]()
            if value < monitor["min"]:
                self.safety.report_violation(
                    f"pos_low_{name}",
                    SafetyLevel.HIGH,
                    "position_limit",
                    f"Position {name} below minimum: {value:.3f} < {monitor['min']:.3f}"
                )
                return f"pos_low_{name}"
            if value > monitor["max"]:
                self.safety.report_violation(
                    f"pos_high_{name}",
                    SafetyLevel.HIGH,
                    "position_limit",
                    f"Position {name} above maximum: {value:.3f} > {monitor['max']:.3f}"
                )
                return f"pos_high_{name}"

        elif monitor_type == "watchdog":
            heartbeat = monitor["getter"]()
            if time.time() - heartbeat > monitor["timeout"]:
                self.safety.report_violation(
                    f"watchdog_{name}",
                    SafetyLevel.CRITICAL,
                    "watchdog_timeout",
                    f"Watchdog {name} timed out"
                )
                return f"watchdog_{name}"
            monitor["last_beat"] = heartbeat

        return None

    def _report_limit_violation(
        self,
        name: str,
        value: float,
        level: SafetyLevel,
        category: str
    ) -> str:
        """Report limit violation."""
        violation_id = f"{category}_{name}_{int(time.time())}"
        self.safety.report_violation(
            violation_id,
            level,
            f"{category}_limit",
            f"{category.capitalize()} {name} exceeded: {value:.3f}"
        )
        return violation_id
