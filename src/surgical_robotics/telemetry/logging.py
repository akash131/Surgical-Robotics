"""Logging system for surgical robotics with audit trail support."""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import threading
from queue import Queue
import traceback


class LogLevel(Enum):
    """Log levels with medical significance."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CLINICAL = 25  # Clinical events (between INFO and WARNING)
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    AUDIT = 35  # Audit events (above WARNING)


# Register custom levels
logging.addLevelName(LogLevel.CLINICAL.value, "CLINICAL")
logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger: str
    message: str

    # Context
    procedure_id: Optional[str] = None
    patient_id: Optional[str] = None
    user_id: Optional[str] = None
    robot_id: Optional[str] = None

    # Technical details
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None

    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)

    # Safety
    safety_relevant: bool = False
    requires_acknowledgment: bool = False

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def __init__(
        self,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__()
        self.context = context or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            procedure_id=self.context.get("procedure_id"),
            patient_id=self.context.get("patient_id"),
            user_id=self.context.get("user_id"),
            robot_id=self.context.get("robot_id"),
        )

        # Add extra data
        if hasattr(record, "data"):
            entry.data = record.data
        if hasattr(record, "safety_relevant"):
            entry.safety_relevant = record.safety_relevant

        return entry.to_json()


class SurgicalLogger:
    """
    Enhanced logger for surgical robotics.

    Features:
    - Structured logging
    - Context tracking (procedure, patient, user)
    - Clinical and audit log levels
    - Safety event highlighting
    """

    def __init__(
        self,
        name: str = "surgical_robotics",
        level: LogLevel = LogLevel.INFO
    ) -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        # Context
        self._context: Dict[str, str] = {}

        # Handlers
        self._handlers: List[logging.Handler] = []

        # Callbacks for special events
        self._on_error: Optional[Callable[[LogEntry], None]] = None
        self._on_audit: Optional[Callable[[LogEntry], None]] = None

    def set_context(
        self,
        procedure_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
        robot_id: Optional[str] = None
    ) -> None:
        """Set logging context."""
        if procedure_id:
            self._context["procedure_id"] = procedure_id
        if patient_id:
            self._context["patient_id"] = patient_id
        if user_id:
            self._context["user_id"] = user_id
        if robot_id:
            self._context["robot_id"] = robot_id

        # Update formatter context
        for handler in self._handlers:
            if isinstance(handler.formatter, StructuredFormatter):
                handler.formatter.context = self._context

    def clear_context(self) -> None:
        """Clear logging context."""
        self._context.clear()

    def add_console_handler(
        self,
        level: LogLevel = LogLevel.INFO,
        structured: bool = False
    ) -> None:
        """Add console output handler."""
        handler = logging.StreamHandler()
        handler.setLevel(level.value)

        if structured:
            handler.setFormatter(StructuredFormatter(self._context))
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))

        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def add_file_handler(
        self,
        filepath: str,
        level: LogLevel = LogLevel.DEBUG,
        max_bytes: int = 10_000_000,  # 10MB
        backup_count: int = 5
    ) -> None:
        """Add rotating file handler."""
        from logging.handlers import RotatingFileHandler

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(level.value)
        handler.setFormatter(StructuredFormatter(self._context))

        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def clinical(self, message: str, **kwargs: Any) -> None:
        """Log clinical event."""
        self._log(LogLevel.CLINICAL, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical error."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def audit(self, message: str, **kwargs: Any) -> None:
        """Log audit event."""
        self._log(LogLevel.AUDIT, message, **kwargs)

    def safety_event(
        self,
        message: str,
        level: LogLevel = LogLevel.WARNING,
        **kwargs: Any
    ) -> None:
        """Log safety-relevant event."""
        kwargs["safety_relevant"] = True
        self._log(level, f"[SAFETY] {message}", **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        kwargs["data"] = kwargs.get("data", {})
        kwargs["data"]["traceback"] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Internal log method."""
        extra = {"data": kwargs.get("data", {})}
        if "safety_relevant" in kwargs:
            extra["safety_relevant"] = kwargs["safety_relevant"]

        self._logger.log(level.value, message, extra=extra)

    def set_error_callback(
        self,
        callback: Callable[[LogEntry], None]
    ) -> None:
        """Set callback for error events."""
        self._on_error = callback

    def set_audit_callback(
        self,
        callback: Callable[[LogEntry], None]
    ) -> None:
        """Set callback for audit events."""
        self._on_audit = callback


class AuditLogger:
    """
    Medical audit trail logger.

    Compliant with medical device audit requirements.
    Provides tamper-evident logging with checksums.
    """

    def __init__(
        self,
        audit_file: str,
        device_id: str = ""
    ) -> None:
        self.audit_file = Path(audit_file)
        self.device_id = device_id

        # Ensure directory exists
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

        # Entry counter for sequence
        self._sequence = 0
        self._last_hash = ""

        # Thread-safe queue
        self._queue: Queue = Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start audit logger."""
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True
        )
        self._writer_thread.start()

    def stop(self) -> None:
        """Stop audit logger."""
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

    def log_action(
        self,
        action: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        procedure_id: Optional[str] = None
    ) -> None:
        """Log auditable action."""
        entry = {
            "sequence": self._sequence,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": self.device_id,
            "action": action,
            "user_id": user_id,
            "patient_id": patient_id,
            "procedure_id": procedure_id,
            "details": details or {},
            "previous_hash": self._last_hash
        }

        # Compute hash for chain
        import hashlib
        entry_json = json.dumps(entry, sort_keys=True)
        entry["hash"] = hashlib.sha256(entry_json.encode()).hexdigest()

        self._sequence += 1
        self._last_hash = entry["hash"]

        self._queue.put(entry)

    def log_login(self, user_id: str, success: bool) -> None:
        """Log user login attempt."""
        self.log_action(
            "login_attempt",
            user_id,
            {"success": success}
        )

    def log_procedure_start(
        self,
        user_id: str,
        procedure_id: str,
        patient_id: str,
        procedure_type: str
    ) -> None:
        """Log procedure start."""
        self.log_action(
            "procedure_start",
            user_id,
            {"procedure_type": procedure_type},
            patient_id=patient_id,
            procedure_id=procedure_id
        )

    def log_procedure_end(
        self,
        user_id: str,
        procedure_id: str,
        patient_id: str,
        outcome: str
    ) -> None:
        """Log procedure end."""
        self.log_action(
            "procedure_end",
            user_id,
            {"outcome": outcome},
            patient_id=patient_id,
            procedure_id=procedure_id
        )

    def log_configuration_change(
        self,
        user_id: str,
        setting: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        """Log configuration change."""
        self.log_action(
            "config_change",
            user_id,
            {
                "setting": setting,
                "old_value": str(old_value),
                "new_value": str(new_value)
            }
        )

    def log_safety_event(
        self,
        user_id: str,
        event_type: str,
        description: str,
        severity: str = "warning"
    ) -> None:
        """Log safety event."""
        self.log_action(
            "safety_event",
            user_id,
            {
                "event_type": event_type,
                "description": description,
                "severity": severity
            }
        )

    def _writer_loop(self) -> None:
        """Background writer loop."""
        while self._running or not self._queue.empty():
            try:
                entry = self._queue.get(timeout=1.0)
                self._write_entry(entry)
            except Exception:
                continue

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write entry to audit file."""
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def verify_integrity(self) -> List[int]:
        """
        Verify audit log integrity.

        Returns list of sequence numbers with integrity issues.
        """
        issues = []

        if not self.audit_file.exists():
            return issues

        import hashlib
        previous_hash = ""

        with open(self.audit_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Check chain
                    if entry.get("previous_hash") != previous_hash:
                        issues.append(entry.get("sequence", -1))

                    # Verify hash
                    stored_hash = entry.pop("hash", "")
                    entry_json = json.dumps(entry, sort_keys=True)
                    computed_hash = hashlib.sha256(entry_json.encode()).hexdigest()

                    if computed_hash != stored_hash:
                        issues.append(entry.get("sequence", -1))

                    previous_hash = stored_hash

                except json.JSONDecodeError:
                    issues.append(-1)

        return issues


def setup_logging(
    log_dir: str = "/var/log/surgical_robotics",
    level: LogLevel = LogLevel.INFO,
    console: bool = True,
    structured: bool = True
) -> SurgicalLogger:
    """
    Set up logging for surgical robotics application.

    Returns configured SurgicalLogger instance.
    """
    logger = SurgicalLogger("surgical_robotics", level)

    if console:
        logger.add_console_handler(level, structured=False)

    # Main log file
    logger.add_file_handler(
        os.path.join(log_dir, "surgical_robotics.log"),
        level=LogLevel.DEBUG
    )

    # Error log file
    logger.add_file_handler(
        os.path.join(log_dir, "errors.log"),
        level=LogLevel.ERROR
    )

    return logger
