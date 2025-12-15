"""Data recording and replay for surgical robotics."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Iterator
import json
import time
import threading
from queue import Queue
import struct


@dataclass
class RecordingMetadata:
    """Metadata for a recording session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None

    # Context
    procedure_id: Optional[str] = None
    patient_id: Optional[str] = None
    surgeon: Optional[str] = None
    robot_type: str = ""

    # Recording info
    sample_rate: float = 100.0  # Hz
    total_samples: int = 0
    channels: List[str] = field(default_factory=list)

    # File info
    format: str = "binary"  # binary, csv, hdf5
    compression: Optional[str] = None


@dataclass
class DataSample:
    """Single data sample."""
    timestamp: float
    data: Dict[str, Any]


class RecordingSession:
    """
    Manages a single recording session.

    Records time-series data from surgical robot.
    """

    def __init__(
        self,
        session_id: str,
        output_dir: str,
        channels: List[str],
        sample_rate: float = 100.0
    ) -> None:
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.channels = channels
        self.sample_rate = sample_rate

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        self.metadata = RecordingMetadata(
            session_id=session_id,
            start_time=datetime.utcnow().isoformat() + "Z",
            channels=channels,
            sample_rate=sample_rate
        )

        # State
        self._recording = False
        self._samples: List[DataSample] = []
        self._sample_count = 0

        # Files
        self._data_file = self.output_dir / f"{session_id}_data.bin"
        self._metadata_file = self.output_dir / f"{session_id}_metadata.json"
        self._file_handle: Optional[Any] = None

    def start(self) -> None:
        """Start recording."""
        self._recording = True
        self._file_handle = open(self._data_file, 'wb')

        # Write header
        header = {
            "version": 1,
            "channels": self.channels,
            "sample_rate": self.sample_rate
        }
        header_json = json.dumps(header).encode('utf-8')
        self._file_handle.write(struct.pack('I', len(header_json)))
        self._file_handle.write(header_json)

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        # Update metadata
        self.metadata.end_time = datetime.utcnow().isoformat() + "Z"
        self.metadata.total_samples = self._sample_count

        # Write metadata
        with open(self._metadata_file, 'w') as f:
            json.dump({
                "session_id": self.metadata.session_id,
                "start_time": self.metadata.start_time,
                "end_time": self.metadata.end_time,
                "procedure_id": self.metadata.procedure_id,
                "patient_id": self.metadata.patient_id,
                "surgeon": self.metadata.surgeon,
                "robot_type": self.metadata.robot_type,
                "sample_rate": self.metadata.sample_rate,
                "total_samples": self.metadata.total_samples,
                "channels": self.metadata.channels,
                "format": self.metadata.format
            }, f, indent=2)

    def record_sample(self, data: Dict[str, float]) -> None:
        """Record a single sample."""
        if not self._recording or not self._file_handle:
            return

        timestamp = time.time()

        # Pack data
        values = [data.get(ch, 0.0) for ch in self.channels]

        # Write: timestamp (double) + values (doubles)
        self._file_handle.write(struct.pack('d', timestamp))
        self._file_handle.write(struct.pack(f'{len(values)}d', *values))

        self._sample_count += 1

    def set_context(
        self,
        procedure_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        surgeon: Optional[str] = None,
        robot_type: Optional[str] = None
    ) -> None:
        """Set recording context."""
        if procedure_id:
            self.metadata.procedure_id = procedure_id
        if patient_id:
            self.metadata.patient_id = patient_id
        if surgeon:
            self.metadata.surgeon = surgeon
        if robot_type:
            self.metadata.robot_type = robot_type

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording

    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        return self._sample_count / self.sample_rate


class DataRecorder:
    """
    High-level data recorder for surgical robotics.

    Features:
    - Automatic session management
    - Configurable channels
    - Multiple output formats
    - Background recording
    """

    def __init__(
        self,
        output_dir: str = "/data/recordings",
        default_sample_rate: float = 100.0
    ) -> None:
        self.output_dir = Path(output_dir)
        self.default_sample_rate = default_sample_rate

        # Active session
        self._session: Optional[RecordingSession] = None

        # Background recording
        self._queue: Queue = Queue()
        self._recorder_thread: Optional[threading.Thread] = None
        self._running = False

        # Data sources
        self._data_sources: Dict[str, Callable[[], Dict[str, float]]] = {}

    def register_data_source(
        self,
        name: str,
        source: Callable[[], Dict[str, float]]
    ) -> None:
        """Register a data source for recording."""
        self._data_sources[name] = source

    def start_recording(
        self,
        session_name: Optional[str] = None,
        channels: Optional[List[str]] = None,
        sample_rate: Optional[float] = None
    ) -> str:
        """Start a new recording session."""
        if self._session and self._session.is_recording:
            self.stop_recording()

        # Generate session ID
        session_id = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine channels from data sources
        if channels is None:
            channels = []
            for source_func in self._data_sources.values():
                try:
                    sample = source_func()
                    channels.extend(sample.keys())
                except Exception:
                    pass
            channels = list(set(channels))

        # Create session
        self._session = RecordingSession(
            session_id=session_id,
            output_dir=str(self.output_dir),
            channels=channels,
            sample_rate=sample_rate or self.default_sample_rate
        )

        self._session.start()

        # Start background thread
        self._running = True
        self._recorder_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self._recorder_thread.start()

        return session_id

    def stop_recording(self) -> Optional[str]:
        """Stop current recording session."""
        self._running = False

        if self._recorder_thread:
            self._recorder_thread.join(timeout=5.0)

        if self._session:
            self._session.stop()
            session_id = self._session.session_id
            self._session = None
            return session_id

        return None

    def set_context(self, **kwargs: Any) -> None:
        """Set recording context."""
        if self._session:
            self._session.set_context(**kwargs)

    def _recording_loop(self) -> None:
        """Background recording loop."""
        if not self._session:
            return

        interval = 1.0 / self._session.sample_rate

        while self._running and self._session:
            start_time = time.time()

            # Collect data from all sources
            data = {}
            for name, source in self._data_sources.items():
                try:
                    source_data = source()
                    data.update(source_data)
                except Exception:
                    pass

            # Record sample
            self._session.record_sample(data)

            # Sleep to maintain sample rate
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get list of recorded sessions."""
        sessions = []

        for metadata_file in self.output_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    sessions.append(json.load(f))
            except Exception:
                pass

        return sorted(sessions, key=lambda x: x.get("start_time", ""))


class DataReplayer:
    """
    Replays recorded data from surgical robot sessions.

    Features:
    - Variable playback speed
    - Seeking to specific times
    - Callback-based data delivery
    """

    def __init__(self, recording_dir: str) -> None:
        self.recording_dir = Path(recording_dir)

        # State
        self._metadata: Optional[RecordingMetadata] = None
        self._data_file: Optional[Any] = None
        self._channels: List[str] = []

        # Playback
        self._playing = False
        self._playback_speed = 1.0
        self._current_time = 0.0

        # Callbacks
        self._on_sample: Optional[Callable[[float, Dict[str, float]], None]] = None

    def load(self, session_id: str) -> bool:
        """Load a recording session."""
        metadata_file = self.recording_dir / f"{session_id}_metadata.json"
        data_file = self.recording_dir / f"{session_id}_data.bin"

        if not metadata_file.exists() or not data_file.exists():
            return False

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self._metadata = RecordingMetadata(
            session_id=metadata["session_id"],
            start_time=metadata["start_time"],
            end_time=metadata.get("end_time"),
            procedure_id=metadata.get("procedure_id"),
            patient_id=metadata.get("patient_id"),
            surgeon=metadata.get("surgeon"),
            robot_type=metadata.get("robot_type", ""),
            sample_rate=metadata["sample_rate"],
            total_samples=metadata["total_samples"],
            channels=metadata["channels"]
        )

        self._channels = self._metadata.channels

        # Open data file
        self._data_file = open(data_file, 'rb')

        # Read header
        header_len = struct.unpack('I', self._data_file.read(4))[0]
        self._data_file.read(header_len)  # Skip header

        return True

    def close(self) -> None:
        """Close loaded recording."""
        if self._data_file:
            self._data_file.close()
            self._data_file = None
        self._metadata = None

    def play(
        self,
        on_sample: Callable[[float, Dict[str, float]], None],
        speed: float = 1.0
    ) -> None:
        """
        Start playback.

        Args:
            on_sample: Callback for each sample (timestamp, data)
            speed: Playback speed multiplier
        """
        if not self._data_file or not self._metadata:
            return

        self._on_sample = on_sample
        self._playback_speed = speed
        self._playing = True

        # Playback thread
        thread = threading.Thread(target=self._playback_loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop playback."""
        self._playing = False

    def seek(self, time_offset: float) -> None:
        """Seek to time offset in seconds."""
        if not self._metadata or not self._data_file:
            return

        # Calculate sample number
        sample_num = int(time_offset * self._metadata.sample_rate)
        sample_num = max(0, min(sample_num, self._metadata.total_samples - 1))

        # Calculate byte offset (after header)
        header_len = struct.unpack('I', self._data_file.read(4))[0]
        self._data_file.seek(4 + header_len)  # Reset

        bytes_per_sample = 8 + 8 * len(self._channels)  # timestamp + values
        self._data_file.seek(4 + header_len + sample_num * bytes_per_sample)

    def _playback_loop(self) -> None:
        """Playback loop."""
        if not self._data_file or not self._metadata:
            return

        n_channels = len(self._channels)
        bytes_per_sample = 8 + 8 * n_channels

        last_timestamp = None
        last_real_time = time.time()

        while self._playing:
            # Read sample
            data = self._data_file.read(bytes_per_sample)
            if len(data) < bytes_per_sample:
                self._playing = False
                break

            # Unpack
            timestamp = struct.unpack('d', data[:8])[0]
            values = struct.unpack(f'{n_channels}d', data[8:])

            # Create data dict
            sample_data = dict(zip(self._channels, values))

            # Timing
            if last_timestamp is not None:
                real_elapsed = time.time() - last_real_time
                desired_elapsed = (timestamp - last_timestamp) / self._playback_speed

                if desired_elapsed > real_elapsed:
                    time.sleep(desired_elapsed - real_elapsed)

            # Deliver sample
            if self._on_sample:
                self._on_sample(timestamp, sample_data)

            last_timestamp = timestamp
            last_real_time = time.time()

    def iterate_samples(self) -> Iterator[tuple]:
        """
        Iterate through all samples.

        Yields (timestamp, data_dict) tuples.
        """
        if not self._data_file or not self._metadata:
            return

        # Reset to start
        self._data_file.seek(0)
        header_len = struct.unpack('I', self._data_file.read(4))[0]
        self._data_file.read(header_len)

        n_channels = len(self._channels)
        bytes_per_sample = 8 + 8 * n_channels

        while True:
            data = self._data_file.read(bytes_per_sample)
            if len(data) < bytes_per_sample:
                break

            timestamp = struct.unpack('d', data[:8])[0]
            values = struct.unpack(f'{n_channels}d', data[8:])
            sample_data = dict(zip(self._channels, values))

            yield timestamp, sample_data

    def get_metadata(self) -> Optional[RecordingMetadata]:
        """Get recording metadata."""
        return self._metadata

    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        if not self._metadata:
            return {}

        return {
            "session_id": self._metadata.session_id,
            "duration_seconds": self._metadata.total_samples / self._metadata.sample_rate,
            "sample_rate": self._metadata.sample_rate,
            "total_samples": self._metadata.total_samples,
            "channels": self._channels,
            "start_time": self._metadata.start_time,
            "end_time": self._metadata.end_time
        }
