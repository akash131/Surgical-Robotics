"""Real-time communication protocols for surgical robotics."""

import json
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any, List
import numpy as np
from numpy.typing import NDArray


class MessageType(Enum):
    """Types of real-time messages."""
    JOINT_STATE = auto()
    CARTESIAN_STATE = auto()
    COMMAND = auto()
    STATUS = auto()
    ERROR = auto()
    HEARTBEAT = auto()
    CONFIG = auto()


@dataclass
class RealtimeMessage:
    """Real-time message container."""
    message_type: MessageType
    timestamp: float
    sequence_number: int
    payload: Dict[str, Any]

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        data = {
            "type": self.message_type.value,
            "timestamp": self.timestamp,
            "sequence": self.sequence_number,
            "payload": self.payload
        }
        json_bytes = json.dumps(data).encode('utf-8')
        # Prefix with length (4 bytes)
        return struct.pack('>I', len(json_bytes)) + json_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["RealtimeMessage"]:
        """Deserialize from bytes."""
        try:
            # Skip length prefix
            json_data = json.loads(data[4:].decode('utf-8'))
            return cls(
                message_type=MessageType(json_data["type"]),
                timestamp=json_data["timestamp"],
                sequence_number=json_data["sequence"],
                payload=json_data["payload"]
            )
        except Exception:
            return None


class MessageCodec:
    """
    Codec for efficient message serialization.

    Supports JSON and binary formats.
    """

    def __init__(self, format: str = "json") -> None:
        self.format = format
        self._sequence = 0

    def encode_joint_state(
        self,
        positions: NDArray[np.float64],
        velocities: Optional[NDArray[np.float64]] = None,
        torques: Optional[NDArray[np.float64]] = None
    ) -> bytes:
        """Encode joint state message."""
        self._sequence += 1

        payload = {
            "positions": positions.tolist(),
            "velocities": velocities.tolist() if velocities is not None else [],
            "torques": torques.tolist() if torques is not None else []
        }

        msg = RealtimeMessage(
            message_type=MessageType.JOINT_STATE,
            timestamp=time.time(),
            sequence_number=self._sequence,
            payload=payload
        )

        return msg.to_bytes()

    def encode_cartesian_state(
        self,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64],
        linear_velocity: Optional[NDArray[np.float64]] = None,
        angular_velocity: Optional[NDArray[np.float64]] = None
    ) -> bytes:
        """Encode Cartesian state message."""
        self._sequence += 1

        payload = {
            "position": position.tolist(),
            "orientation": orientation.tolist(),
            "linear_velocity": linear_velocity.tolist() if linear_velocity is not None else [],
            "angular_velocity": angular_velocity.tolist() if angular_velocity is not None else []
        }

        msg = RealtimeMessage(
            message_type=MessageType.CARTESIAN_STATE,
            timestamp=time.time(),
            sequence_number=self._sequence,
            payload=payload
        )

        return msg.to_bytes()

    def encode_command(
        self,
        command_type: str,
        target: NDArray[np.float64],
        **kwargs
    ) -> bytes:
        """Encode command message."""
        self._sequence += 1

        payload = {
            "command_type": command_type,
            "target": target.tolist(),
            **kwargs
        }

        msg = RealtimeMessage(
            message_type=MessageType.COMMAND,
            timestamp=time.time(),
            sequence_number=self._sequence,
            payload=payload
        )

        return msg.to_bytes()

    def decode(self, data: bytes) -> Optional[RealtimeMessage]:
        """Decode message from bytes."""
        return RealtimeMessage.from_bytes(data)


class Transport(ABC):
    """Abstract base class for transport protocols."""

    def __init__(self) -> None:
        self._connected = False
        self._on_receive_callback: Optional[Callable[[bytes], None]] = None

    @abstractmethod
    def connect(self, host: str, port: int) -> bool:
        """Connect to remote endpoint."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from remote endpoint."""
        pass

    @abstractmethod
    def send(self, data: bytes) -> bool:
        """Send data."""
        pass

    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive data."""
        pass

    def set_receive_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for received data."""
        self._on_receive_callback = callback

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


class UDPTransport(Transport):
    """
    UDP transport for real-time communication.

    Provides low-latency, connectionless communication.
    Best for high-frequency state updates where occasional
    packet loss is acceptable.
    """

    def __init__(self, local_port: Optional[int] = None) -> None:
        super().__init__()
        self.local_port = local_port
        self.remote_host: Optional[str] = None
        self.remote_port: Optional[int] = None
        self._socket = None

    def connect(self, host: str, port: int) -> bool:
        """Set up UDP communication."""
        try:
            # Would create actual UDP socket
            self.remote_host = host
            self.remote_port = port
            self._connected = True
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Close UDP socket."""
        self._connected = False
        self._socket = None

    def send(self, data: bytes) -> bool:
        """Send UDP datagram."""
        if not self._connected:
            return False

        # Would send via socket
        return True

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive UDP datagram."""
        if not self._connected:
            return None

        # Would receive with timeout
        return None

    def bind(self, port: int) -> bool:
        """Bind to local port for receiving."""
        self.local_port = port
        return True


class TCPTransport(Transport):
    """
    TCP transport for reliable communication.

    Provides guaranteed delivery and ordering.
    Best for commands and configuration where
    reliability is more important than latency.
    """

    def __init__(self) -> None:
        super().__init__()
        self._socket = None
        self._buffer = b""

    def connect(self, host: str, port: int) -> bool:
        """Establish TCP connection."""
        try:
            # Would create TCP socket and connect
            self._connected = True
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Close TCP connection."""
        self._connected = False
        self._socket = None
        self._buffer = b""

    def send(self, data: bytes) -> bool:
        """Send data over TCP."""
        if not self._connected:
            return False

        # Would send via socket
        return True

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive data from TCP."""
        if not self._connected:
            return None

        # Would receive with buffering for message framing
        return None


@dataclass
class RealtimeConfig:
    """Configuration for real-time protocol."""
    control_rate_hz: float = 1000.0
    state_rate_hz: float = 500.0
    heartbeat_rate_hz: float = 10.0

    # Timeouts
    connection_timeout: float = 5.0
    heartbeat_timeout: float = 1.0

    # Buffer sizes
    send_buffer_size: int = 65536
    receive_buffer_size: int = 65536

    # Protocol options
    use_compression: bool = False
    use_encryption: bool = False


class RealtimeProtocol:
    """
    Complete real-time communication protocol.

    Manages bidirectional communication with heartbeat monitoring.
    """

    def __init__(
        self,
        transport: Transport,
        config: Optional[RealtimeConfig] = None
    ) -> None:
        self.transport = transport
        self.config = config or RealtimeConfig()
        self.codec = MessageCodec()

        self._running = False
        self._last_heartbeat_sent = 0.0
        self._last_heartbeat_received = 0.0

        # Callbacks
        self._message_callbacks: Dict[MessageType, List[Callable]] = {
            mt: [] for mt in MessageType
        }

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    def connect(self, host: str, port: int) -> bool:
        """Connect to remote endpoint."""
        if not self.transport.connect(host, port):
            return False

        self._running = True
        self._last_heartbeat_received = time.time()
        return True

    def disconnect(self) -> None:
        """Disconnect from remote endpoint."""
        self._running = False
        self.transport.disconnect()

    def register_callback(
        self,
        message_type: MessageType,
        callback: Callable[[RealtimeMessage], None]
    ) -> None:
        """Register callback for message type."""
        self._message_callbacks[message_type].append(callback)

    def send_joint_state(
        self,
        positions: NDArray[np.float64],
        velocities: Optional[NDArray[np.float64]] = None,
        torques: Optional[NDArray[np.float64]] = None
    ) -> bool:
        """Send joint state update."""
        data = self.codec.encode_joint_state(positions, velocities, torques)
        return self._send(data)

    def send_cartesian_state(
        self,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64]
    ) -> bool:
        """Send Cartesian state update."""
        data = self.codec.encode_cartesian_state(position, orientation)
        return self._send(data)

    def send_command(
        self,
        command_type: str,
        target: NDArray[np.float64],
        **kwargs
    ) -> bool:
        """Send command."""
        data = self.codec.encode_command(command_type, target, **kwargs)
        return self._send(data)

    def send_heartbeat(self) -> bool:
        """Send heartbeat message."""
        msg = RealtimeMessage(
            message_type=MessageType.HEARTBEAT,
            timestamp=time.time(),
            sequence_number=0,
            payload={"status": "alive"}
        )
        return self._send(msg.to_bytes())

    def process(self) -> None:
        """Process incoming messages and send heartbeat."""
        # Check heartbeat
        current_time = time.time()
        if current_time - self._last_heartbeat_sent > 1.0 / self.config.heartbeat_rate_hz:
            self.send_heartbeat()
            self._last_heartbeat_sent = current_time

        # Check for received messages
        data = self.transport.receive(timeout=0.001)
        if data:
            self._handle_received(data)

        # Check heartbeat timeout
        if current_time - self._last_heartbeat_received > self.config.heartbeat_timeout:
            # Connection lost
            self._running = False

    def _send(self, data: bytes) -> bool:
        """Send data and update statistics."""
        if self.transport.send(data):
            self.messages_sent += 1
            self.bytes_sent += len(data)
            return True
        return False

    def _handle_received(self, data: bytes) -> None:
        """Handle received data."""
        self.bytes_received += len(data)

        msg = self.codec.decode(data)
        if msg is None:
            return

        self.messages_received += 1

        if msg.message_type == MessageType.HEARTBEAT:
            self._last_heartbeat_received = time.time()

        # Notify callbacks
        for callback in self._message_callbacks[msg.message_type]:
            callback(msg)

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "connected": self.transport.is_connected,
            "running": self._running
        }

    @property
    def is_connected(self) -> bool:
        """Check if protocol is connected and healthy."""
        return self._running and self.transport.is_connected


class RealtimeServer:
    """
    Real-time communication server.

    Accepts multiple client connections and broadcasts state.
    """

    def __init__(self, port: int, config: Optional[RealtimeConfig] = None) -> None:
        self.port = port
        self.config = config or RealtimeConfig()
        self.clients: List[RealtimeProtocol] = []
        self._running = False

    def start(self) -> bool:
        """Start server."""
        # Would bind to port and start accepting connections
        self._running = True
        return True

    def stop(self) -> None:
        """Stop server."""
        self._running = False
        for client in self.clients:
            client.disconnect()
        self.clients.clear()

    def broadcast_joint_state(
        self,
        positions: NDArray[np.float64],
        velocities: Optional[NDArray[np.float64]] = None,
        torques: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Broadcast joint state to all clients."""
        for client in self.clients:
            client.send_joint_state(positions, velocities, torques)

    def broadcast_cartesian_state(
        self,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64]
    ) -> None:
        """Broadcast Cartesian state to all clients."""
        for client in self.clients:
            client.send_cartesian_state(position, orientation)

    def process(self) -> None:
        """Process all client connections."""
        # Remove disconnected clients
        self.clients = [c for c in self.clients if c.is_connected]

        # Process each client
        for client in self.clients:
            client.process()

    @property
    def num_clients(self) -> int:
        """Get number of connected clients."""
        return len(self.clients)
