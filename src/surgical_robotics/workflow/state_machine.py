"""State machine implementation for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional, Dict, Any, List, Callable, Set, TypeVar, Generic
)
import time
from datetime import datetime


class StateType(Enum):
    """Types of states in the machine."""
    INITIAL = auto()
    NORMAL = auto()
    FINAL = auto()
    ERROR = auto()


@dataclass
class State:
    """
    Represents a state in the state machine.

    Attributes:
        name: Unique state identifier
        state_type: Type of state (initial, normal, final, error)
        on_enter: Callback when entering state
        on_exit: Callback when exiting state
        on_update: Callback during state update
    """
    name: str
    state_type: StateType = StateType.NORMAL

    # Callbacks
    on_enter: Optional[Callable[["State", Dict[str, Any]], None]] = None
    on_exit: Optional[Callable[["State", Dict[str, Any]], None]] = None
    on_update: Optional[Callable[["State", float, Dict[str, Any]], None]] = None

    # Metadata
    description: str = ""
    timeout: Optional[float] = None  # Auto-transition on timeout
    timeout_target: Optional[str] = None

    # State data
    data: Dict[str, Any] = field(default_factory=dict)
    entry_time: Optional[float] = None

    def enter(self, context: Dict[str, Any]) -> None:
        """Called when entering the state."""
        self.entry_time = time.time()
        if self.on_enter:
            self.on_enter(self, context)

    def exit(self, context: Dict[str, Any]) -> None:
        """Called when exiting the state."""
        if self.on_exit:
            self.on_exit(self, context)
        self.entry_time = None

    def update(self, dt: float, context: Dict[str, Any]) -> Optional[str]:
        """
        Update the state.

        Returns target state name if transition should occur.
        """
        if self.on_update:
            self.on_update(self, dt, context)

        # Check timeout
        if self.timeout and self.entry_time:
            elapsed = time.time() - self.entry_time
            if elapsed > self.timeout:
                return self.timeout_target

        return None


@dataclass
class Transition:
    """
    Represents a transition between states.

    Attributes:
        source: Source state name
        target: Target state name
        event: Event that triggers transition
        condition: Guard condition
        action: Action to execute during transition
    """
    source: str
    target: str
    event: Optional[str] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None
    priority: int = 0

    def can_trigger(self, event: Optional[str], context: Dict[str, Any]) -> bool:
        """Check if transition can be triggered."""
        if self.event and event != self.event:
            return False
        if self.condition and not self.condition(context):
            return False
        return True

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute transition action."""
        if self.action:
            self.action(context)


class StateMachine:
    """
    Generic state machine implementation.

    Features:
    - Hierarchical states (substates)
    - Guard conditions
    - Entry/exit actions
    - Timeout transitions
    - Event-driven and conditional transitions
    """

    def __init__(self, name: str = "state_machine") -> None:
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []

        self._current_state: Optional[State] = None
        self._initial_state: Optional[str] = None
        self._context: Dict[str, Any] = {}

        # History
        self._state_history: List[tuple] = []  # (state_name, timestamp)
        self._transition_history: List[tuple] = []

        # Callbacks
        self._on_state_change: Optional[Callable[[str, str], None]] = None

    def add_state(self, state: State) -> None:
        """Add state to machine."""
        self.states[state.name] = state
        if state.state_type == StateType.INITIAL:
            self._initial_state = state.name

    def add_transition(self, transition: Transition) -> None:
        """Add transition to machine."""
        self.transitions.append(transition)
        # Sort by priority
        self.transitions.sort(key=lambda t: -t.priority)

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)

    def start(self) -> bool:
        """Start the state machine."""
        if not self._initial_state:
            # Find initial state
            for state in self.states.values():
                if state.state_type == StateType.INITIAL:
                    self._initial_state = state.name
                    break

        if not self._initial_state:
            return False

        self._transition_to(self._initial_state)
        return True

    def stop(self) -> None:
        """Stop the state machine."""
        if self._current_state:
            self._current_state.exit(self._context)
        self._current_state = None

    def update(self, dt: float) -> None:
        """Update the state machine."""
        if not self._current_state:
            return

        # Update current state
        next_state = self._current_state.update(dt, self._context)
        if next_state:
            self._transition_to(next_state)

    def trigger(self, event: str) -> bool:
        """Trigger an event."""
        if not self._current_state:
            return False

        # Find matching transition
        for transition in self.transitions:
            if transition.source != self._current_state.name:
                continue
            if transition.can_trigger(event, self._context):
                self._execute_transition(transition)
                return True

        return False

    def can_transition(self, target: str) -> bool:
        """Check if transition to target is possible."""
        if not self._current_state:
            return False

        for transition in self.transitions:
            if transition.source == self._current_state.name and transition.target == target:
                if transition.can_trigger(None, self._context):
                    return True
        return False

    def force_transition(self, target: str) -> bool:
        """Force transition to target state."""
        if target not in self.states:
            return False
        self._transition_to(target)
        return True

    @property
    def current_state(self) -> Optional[str]:
        """Get current state name."""
        return self._current_state.name if self._current_state else None

    @property
    def is_running(self) -> bool:
        """Check if machine is running."""
        return self._current_state is not None

    @property
    def is_final(self) -> bool:
        """Check if in final state."""
        if not self._current_state:
            return False
        return self._current_state.state_type == StateType.FINAL

    @property
    def is_error(self) -> bool:
        """Check if in error state."""
        if not self._current_state:
            return False
        return self._current_state.state_type == StateType.ERROR

    def get_history(self) -> List[tuple]:
        """Get state transition history."""
        return self._state_history.copy()

    def set_state_change_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def _transition_to(self, target: str) -> None:
        """Internal transition to target state."""
        if target not in self.states:
            return

        previous_name = self._current_state.name if self._current_state else None

        # Exit current state
        if self._current_state:
            self._current_state.exit(self._context)

        # Enter new state
        self._current_state = self.states[target]
        self._current_state.enter(self._context)

        # Record history
        self._state_history.append((target, time.time()))

        # Callback
        if self._on_state_change and previous_name:
            self._on_state_change(previous_name, target)

    def _execute_transition(self, transition: Transition) -> None:
        """Execute a transition."""
        # Execute transition action
        transition.execute(self._context)

        # Record
        self._transition_history.append((
            transition.source,
            transition.target,
            transition.event,
            time.time()
        ))

        # Transition to target
        self._transition_to(transition.target)


class RobotState(Enum):
    """Standard robot operational states."""
    POWERED_OFF = "powered_off"
    INITIALIZING = "initializing"
    IDLE = "idle"
    HOMING = "homing"
    READY = "ready"
    OPERATING = "operating"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"
    CALIBRATING = "calibrating"
    SHUTTING_DOWN = "shutting_down"


class RobotStateMachine(StateMachine):
    """
    State machine for robot operational states.

    Implements standard surgical robot state transitions.
    """

    def __init__(self, name: str = "robot_state_machine") -> None:
        super().__init__(name)
        self._setup_states()
        self._setup_transitions()

    def _setup_states(self) -> None:
        """Set up robot states."""
        states = [
            State(
                name=RobotState.POWERED_OFF.value,
                state_type=StateType.INITIAL,
                description="Robot is powered off"
            ),
            State(
                name=RobotState.INITIALIZING.value,
                description="Robot is initializing systems",
                timeout=30.0,
                timeout_target=RobotState.ERROR.value
            ),
            State(
                name=RobotState.IDLE.value,
                description="Robot is idle, waiting for commands"
            ),
            State(
                name=RobotState.HOMING.value,
                description="Robot is homing joints",
                timeout=60.0,
                timeout_target=RobotState.ERROR.value
            ),
            State(
                name=RobotState.READY.value,
                description="Robot is ready for operation"
            ),
            State(
                name=RobotState.OPERATING.value,
                description="Robot is actively operating"
            ),
            State(
                name=RobotState.PAUSED.value,
                description="Operation is paused"
            ),
            State(
                name=RobotState.ERROR.value,
                state_type=StateType.ERROR,
                description="Robot is in error state"
            ),
            State(
                name=RobotState.EMERGENCY_STOP.value,
                state_type=StateType.ERROR,
                description="Emergency stop activated"
            ),
            State(
                name=RobotState.MAINTENANCE.value,
                description="Robot is in maintenance mode"
            ),
            State(
                name=RobotState.CALIBRATING.value,
                description="Robot is being calibrated"
            ),
            State(
                name=RobotState.SHUTTING_DOWN.value,
                description="Robot is shutting down",
                timeout=10.0,
                timeout_target=RobotState.POWERED_OFF.value
            ),
        ]

        for state in states:
            self.add_state(state)

    def _setup_transitions(self) -> None:
        """Set up standard transitions."""
        transitions = [
            # Power on sequence
            Transition(
                source=RobotState.POWERED_OFF.value,
                target=RobotState.INITIALIZING.value,
                event="power_on"
            ),
            Transition(
                source=RobotState.INITIALIZING.value,
                target=RobotState.IDLE.value,
                event="init_complete"
            ),

            # Homing
            Transition(
                source=RobotState.IDLE.value,
                target=RobotState.HOMING.value,
                event="home"
            ),
            Transition(
                source=RobotState.HOMING.value,
                target=RobotState.READY.value,
                event="home_complete"
            ),

            # Operation
            Transition(
                source=RobotState.READY.value,
                target=RobotState.OPERATING.value,
                event="start"
            ),
            Transition(
                source=RobotState.OPERATING.value,
                target=RobotState.PAUSED.value,
                event="pause"
            ),
            Transition(
                source=RobotState.PAUSED.value,
                target=RobotState.OPERATING.value,
                event="resume"
            ),
            Transition(
                source=RobotState.OPERATING.value,
                target=RobotState.READY.value,
                event="stop"
            ),
            Transition(
                source=RobotState.PAUSED.value,
                target=RobotState.READY.value,
                event="stop"
            ),

            # Calibration
            Transition(
                source=RobotState.READY.value,
                target=RobotState.CALIBRATING.value,
                event="calibrate"
            ),
            Transition(
                source=RobotState.IDLE.value,
                target=RobotState.CALIBRATING.value,
                event="calibrate"
            ),
            Transition(
                source=RobotState.CALIBRATING.value,
                target=RobotState.READY.value,
                event="calibration_complete"
            ),

            # Maintenance
            Transition(
                source=RobotState.IDLE.value,
                target=RobotState.MAINTENANCE.value,
                event="enter_maintenance"
            ),
            Transition(
                source=RobotState.READY.value,
                target=RobotState.MAINTENANCE.value,
                event="enter_maintenance"
            ),
            Transition(
                source=RobotState.MAINTENANCE.value,
                target=RobotState.IDLE.value,
                event="exit_maintenance"
            ),

            # Emergency stop - from any operating state
            Transition(
                source=RobotState.OPERATING.value,
                target=RobotState.EMERGENCY_STOP.value,
                event="e_stop",
                priority=100
            ),
            Transition(
                source=RobotState.READY.value,
                target=RobotState.EMERGENCY_STOP.value,
                event="e_stop",
                priority=100
            ),
            Transition(
                source=RobotState.HOMING.value,
                target=RobotState.EMERGENCY_STOP.value,
                event="e_stop",
                priority=100
            ),
            Transition(
                source=RobotState.CALIBRATING.value,
                target=RobotState.EMERGENCY_STOP.value,
                event="e_stop",
                priority=100
            ),

            # Error handling
            Transition(
                source=RobotState.OPERATING.value,
                target=RobotState.ERROR.value,
                event="error"
            ),
            Transition(
                source=RobotState.HOMING.value,
                target=RobotState.ERROR.value,
                event="error"
            ),

            # Recovery
            Transition(
                source=RobotState.ERROR.value,
                target=RobotState.IDLE.value,
                event="reset"
            ),
            Transition(
                source=RobotState.EMERGENCY_STOP.value,
                target=RobotState.IDLE.value,
                event="reset"
            ),

            # Shutdown
            Transition(
                source=RobotState.IDLE.value,
                target=RobotState.SHUTTING_DOWN.value,
                event="shutdown"
            ),
            Transition(
                source=RobotState.READY.value,
                target=RobotState.SHUTTING_DOWN.value,
                event="shutdown"
            ),
            Transition(
                source=RobotState.ERROR.value,
                target=RobotState.SHUTTING_DOWN.value,
                event="shutdown"
            ),
            Transition(
                source=RobotState.SHUTTING_DOWN.value,
                target=RobotState.POWERED_OFF.value,
                event="shutdown_complete"
            ),
        ]

        for transition in transitions:
            self.add_transition(transition)

    def power_on(self) -> bool:
        """Power on the robot."""
        return self.trigger("power_on")

    def home(self) -> bool:
        """Start homing sequence."""
        return self.trigger("home")

    def start_operation(self) -> bool:
        """Start robot operation."""
        return self.trigger("start")

    def pause(self) -> bool:
        """Pause operation."""
        return self.trigger("pause")

    def resume(self) -> bool:
        """Resume operation."""
        return self.trigger("resume")

    def stop(self) -> bool:
        """Stop operation."""
        return self.trigger("stop")

    def emergency_stop(self) -> bool:
        """Trigger emergency stop."""
        return self.trigger("e_stop")

    def reset(self) -> bool:
        """Reset from error state."""
        return self.trigger("reset")

    def shutdown(self) -> bool:
        """Shutdown the robot."""
        return self.trigger("shutdown")

    def is_operational(self) -> bool:
        """Check if robot is in operational state."""
        return self.current_state == RobotState.OPERATING.value

    def is_ready(self) -> bool:
        """Check if robot is ready."""
        return self.current_state in [
            RobotState.READY.value,
            RobotState.OPERATING.value,
            RobotState.PAUSED.value
        ]
