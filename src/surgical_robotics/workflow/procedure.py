"""Surgical procedure workflow management."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable
import time
from datetime import datetime, timedelta


class ProcedurePhase(Enum):
    """Surgical procedure phases."""
    # Pre-operative
    SETUP = "setup"
    PATIENT_POSITIONING = "patient_positioning"
    REGISTRATION = "registration"
    PLANNING_REVIEW = "planning_review"
    DRAPING = "draping"

    # Intra-operative
    INCISION = "incision"
    ACCESS = "access"
    DISSECTION = "dissection"
    MAIN_PROCEDURE = "main_procedure"
    HEMOSTASIS = "hemostasis"
    VERIFICATION = "verification"
    CLOSURE = "closure"

    # Post-operative
    DRESSING = "dressing"
    DOCUMENTATION = "documentation"
    CLEANUP = "cleanup"

    # Special
    PAUSE = "pause"
    EMERGENCY = "emergency"


@dataclass
class SurgicalTask:
    """
    Represents a single surgical task.

    Can be a manual step or automated robot action.
    """
    id: str
    name: str
    description: str = ""
    phase: ProcedurePhase = ProcedurePhase.MAIN_PROCEDURE

    # Task type
    is_automated: bool = False
    requires_confirmation: bool = True
    is_critical: bool = False

    # Execution
    action: Optional[Callable[[], bool]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Timing
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Status
    completed: bool = False
    skipped: bool = False
    error: Optional[str] = None

    # Dependencies
    prerequisites: List[str] = field(default_factory=list)

    # Metadata
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def start(self) -> None:
        """Mark task as started."""
        self.start_time = datetime.now()

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark task as completed."""
        self.end_time = datetime.now()
        if self.start_time:
            self.actual_duration = self.end_time - self.start_time
        self.completed = success
        self.error = error

    def skip(self, reason: str = "") -> None:
        """Skip this task."""
        self.skipped = True
        self.notes = reason

    def execute(self) -> bool:
        """Execute the task action."""
        if not self.action:
            return True

        self.start()
        try:
            result = self.action()
            self.complete(success=result)
            return result
        except Exception as e:
            self.complete(success=False, error=str(e))
            return False


class TaskSequencer:
    """
    Sequences and executes surgical tasks.

    Manages task dependencies and execution order.
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, SurgicalTask] = {}
        self._task_order: List[str] = []
        self._current_index: int = 0

        # Callbacks
        self._on_task_start: Optional[Callable[[SurgicalTask], None]] = None
        self._on_task_complete: Optional[Callable[[SurgicalTask, bool], None]] = None
        self._on_sequence_complete: Optional[Callable[[bool], None]] = None

    def add_task(self, task: SurgicalTask) -> None:
        """Add task to sequence."""
        self.tasks[task.id] = task
        if task.id not in self._task_order:
            self._task_order.append(task.id)

    def insert_task(self, task: SurgicalTask, position: int) -> None:
        """Insert task at specific position."""
        self.tasks[task.id] = task
        if task.id in self._task_order:
            self._task_order.remove(task.id)
        self._task_order.insert(position, task.id)

    def remove_task(self, task_id: str) -> bool:
        """Remove task from sequence."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._task_order.remove(task_id)
            return True
        return False

    def get_current_task(self) -> Optional[SurgicalTask]:
        """Get current task."""
        if self._current_index < len(self._task_order):
            task_id = self._task_order[self._current_index]
            return self.tasks.get(task_id)
        return None

    def get_next_task(self) -> Optional[SurgicalTask]:
        """Get next task in sequence."""
        if self._current_index + 1 < len(self._task_order):
            task_id = self._task_order[self._current_index + 1]
            return self.tasks.get(task_id)
        return None

    def can_proceed(self) -> bool:
        """Check if can proceed to next task."""
        current = self.get_current_task()
        if not current:
            return False

        # Check if current task is complete or skipped
        if not (current.completed or current.skipped):
            return False

        # Check next task's prerequisites
        next_task = self.get_next_task()
        if next_task:
            for prereq_id in next_task.prerequisites:
                if prereq_id in self.tasks:
                    prereq = self.tasks[prereq_id]
                    if not (prereq.completed or prereq.skipped):
                        return False

        return True

    def advance(self) -> bool:
        """Advance to next task."""
        if not self.can_proceed():
            return False

        self._current_index += 1

        if self._current_index >= len(self._task_order):
            if self._on_sequence_complete:
                self._on_sequence_complete(True)
            return False

        return True

    def execute_current(self) -> bool:
        """Execute current task."""
        task = self.get_current_task()
        if not task:
            return False

        if self._on_task_start:
            self._on_task_start(task)

        success = task.execute()

        if self._on_task_complete:
            self._on_task_complete(task, success)

        return success

    def skip_current(self, reason: str = "") -> bool:
        """Skip current task."""
        task = self.get_current_task()
        if not task:
            return False

        if task.is_critical:
            return False  # Cannot skip critical tasks

        task.skip(reason)
        return True

    def run_automated(self) -> bool:
        """Run all automated tasks in sequence."""
        while self._current_index < len(self._task_order):
            task = self.get_current_task()
            if not task:
                break

            if task.is_automated:
                if not self.execute_current():
                    return False
            else:
                # Stop at manual task
                break

            if not self.advance():
                break

        return True

    def reset(self) -> None:
        """Reset sequencer to beginning."""
        self._current_index = 0
        for task in self.tasks.values():
            task.completed = False
            task.skipped = False
            task.error = None
            task.start_time = None
            task.end_time = None
            task.actual_duration = None

    def get_progress(self) -> Dict[str, Any]:
        """Get sequence progress."""
        total = len(self._task_order)
        completed = sum(
            1 for task in self.tasks.values()
            if task.completed or task.skipped
        )

        return {
            "total": total,
            "completed": completed,
            "current_index": self._current_index,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "current_task": self.get_current_task().name if self.get_current_task() else None
        }

    def set_callbacks(
        self,
        on_start: Optional[Callable[[SurgicalTask], None]] = None,
        on_complete: Optional[Callable[[SurgicalTask, bool], None]] = None,
        on_sequence_complete: Optional[Callable[[bool], None]] = None
    ) -> None:
        """Set execution callbacks."""
        self._on_task_start = on_start
        self._on_task_complete = on_complete
        self._on_sequence_complete = on_sequence_complete


class ProcedureWorkflow:
    """
    Complete surgical procedure workflow manager.

    Manages procedure phases, tasks, and documentation.
    """

    def __init__(
        self,
        procedure_name: str,
        procedure_type: str = ""
    ) -> None:
        self.procedure_name = procedure_name
        self.procedure_type = procedure_type

        # Phases and tasks
        self.phases: Dict[ProcedurePhase, List[SurgicalTask]] = {}
        self.task_sequencer = TaskSequencer()

        # Current state
        self.current_phase: Optional[ProcedurePhase] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Documentation
        self.patient_id: str = ""
        self.surgeon: str = ""
        self.notes: List[str] = []
        self.events: List[Dict[str, Any]] = []

        # Status
        self.is_active: bool = False
        self.is_complete: bool = False
        self.is_aborted: bool = False

    def setup_phases(self, phases: List[ProcedurePhase]) -> None:
        """Set up procedure phases."""
        for phase in phases:
            self.phases[phase] = []

    def add_task_to_phase(
        self,
        phase: ProcedurePhase,
        task: SurgicalTask
    ) -> None:
        """Add task to a specific phase."""
        task.phase = phase
        if phase not in self.phases:
            self.phases[phase] = []
        self.phases[phase].append(task)
        self.task_sequencer.add_task(task)

    def start_procedure(
        self,
        patient_id: str = "",
        surgeon: str = ""
    ) -> bool:
        """Start the surgical procedure."""
        if self.is_active:
            return False

        self.patient_id = patient_id
        self.surgeon = surgeon
        self.start_time = datetime.now()
        self.is_active = True

        # Start with first phase
        if self.phases:
            self.current_phase = list(self.phases.keys())[0]

        self._log_event("procedure_start", {
            "procedure": self.procedure_name,
            "patient_id": patient_id,
            "surgeon": surgeon
        })

        return True

    def advance_phase(self) -> bool:
        """Advance to next phase."""
        if not self.is_active or not self.current_phase:
            return False

        phase_list = list(self.phases.keys())
        current_index = phase_list.index(self.current_phase)

        if current_index + 1 >= len(phase_list):
            return False

        previous_phase = self.current_phase
        self.current_phase = phase_list[current_index + 1]

        self._log_event("phase_change", {
            "from": previous_phase.value,
            "to": self.current_phase.value
        })

        return True

    def set_phase(self, phase: ProcedurePhase) -> bool:
        """Set current phase directly."""
        if phase not in self.phases:
            return False

        previous = self.current_phase
        self.current_phase = phase

        self._log_event("phase_change", {
            "from": previous.value if previous else None,
            "to": phase.value
        })

        return True

    def pause_procedure(self, reason: str = "") -> None:
        """Pause the procedure."""
        self._log_event("procedure_pause", {"reason": reason})
        self.current_phase = ProcedurePhase.PAUSE

    def resume_procedure(self) -> None:
        """Resume the procedure."""
        self._log_event("procedure_resume", {})
        # Return to appropriate phase (simplified)
        if ProcedurePhase.MAIN_PROCEDURE in self.phases:
            self.current_phase = ProcedurePhase.MAIN_PROCEDURE

    def abort_procedure(self, reason: str) -> None:
        """Abort the procedure."""
        self.is_aborted = True
        self.is_active = False
        self.end_time = datetime.now()

        self._log_event("procedure_abort", {"reason": reason})

    def complete_procedure(self) -> None:
        """Mark procedure as complete."""
        self.is_complete = True
        self.is_active = False
        self.end_time = datetime.now()

        self._log_event("procedure_complete", {
            "duration_minutes": self.get_duration().total_seconds() / 60
            if self.get_duration() else 0
        })

    def add_note(self, note: str) -> None:
        """Add clinical note."""
        self.notes.append(note)
        self._log_event("note_added", {"note": note})

    def get_duration(self) -> Optional[timedelta]:
        """Get procedure duration."""
        if not self.start_time:
            return None

        end = self.end_time or datetime.now()
        return end - self.start_time

    def get_status(self) -> Dict[str, Any]:
        """Get procedure status."""
        duration = self.get_duration()
        progress = self.task_sequencer.get_progress()

        return {
            "procedure": self.procedure_name,
            "type": self.procedure_type,
            "patient_id": self.patient_id,
            "surgeon": self.surgeon,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "is_active": self.is_active,
            "is_complete": self.is_complete,
            "is_aborted": self.is_aborted,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_minutes": duration.total_seconds() / 60 if duration else 0,
            "task_progress": progress,
            "event_count": len(self.events)
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get procedure summary for documentation."""
        return {
            "procedure_name": self.procedure_name,
            "procedure_type": self.procedure_type,
            "patient_id": self.patient_id,
            "surgeon": self.surgeon,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.get_duration()) if self.get_duration() else None,
            "completed": self.is_complete,
            "aborted": self.is_aborted,
            "phases_completed": [
                p.value for p in self.phases.keys()
                if all(t.completed or t.skipped for t in self.phases[p])
            ],
            "notes": self.notes,
            "events": self.events
        }

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log procedure event."""
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "phase": self.current_phase.value if self.current_phase else None,
            "data": data
        })

    @classmethod
    def create_laparoscopic_workflow(cls) -> "ProcedureWorkflow":
        """Create standard laparoscopic procedure workflow."""
        workflow = cls("Laparoscopic Procedure", "laparoscopic")

        phases = [
            ProcedurePhase.SETUP,
            ProcedurePhase.PATIENT_POSITIONING,
            ProcedurePhase.DRAPING,
            ProcedurePhase.ACCESS,
            ProcedurePhase.MAIN_PROCEDURE,
            ProcedurePhase.HEMOSTASIS,
            ProcedurePhase.CLOSURE,
            ProcedurePhase.DOCUMENTATION
        ]
        workflow.setup_phases(phases)

        # Add standard tasks
        workflow.add_task_to_phase(
            ProcedurePhase.SETUP,
            SurgicalTask(
                id="check_equipment",
                name="Equipment Check",
                description="Verify all equipment is functional",
                is_critical=True
            )
        )

        workflow.add_task_to_phase(
            ProcedurePhase.ACCESS,
            SurgicalTask(
                id="trocar_placement",
                name="Trocar Placement",
                description="Insert trocars at marked positions",
                is_critical=True
            )
        )

        return workflow

    @classmethod
    def create_orthopedic_workflow(cls) -> "ProcedureWorkflow":
        """Create orthopedic surgery workflow."""
        workflow = cls("Orthopedic Procedure", "orthopedic")

        phases = [
            ProcedurePhase.SETUP,
            ProcedurePhase.REGISTRATION,
            ProcedurePhase.PLANNING_REVIEW,
            ProcedurePhase.INCISION,
            ProcedurePhase.MAIN_PROCEDURE,
            ProcedurePhase.VERIFICATION,
            ProcedurePhase.CLOSURE,
            ProcedurePhase.DOCUMENTATION
        ]
        workflow.setup_phases(phases)

        # Add bone registration task
        workflow.add_task_to_phase(
            ProcedurePhase.REGISTRATION,
            SurgicalTask(
                id="bone_registration",
                name="Bone Registration",
                description="Register bone anatomy to CT model",
                is_critical=True,
                is_automated=True
            )
        )

        return workflow
