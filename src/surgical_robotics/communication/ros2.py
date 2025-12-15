"""ROS2 integration for surgical robotics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Dict, Any
import json
import time
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose, JointState


class QoSProfile(Enum):
    """ROS2 QoS profiles."""
    SENSOR_DATA = auto()  # Best effort, volatile
    RELIABLE = auto()  # Reliable, transient local
    SERVICES = auto()  # Reliable, volatile
    PARAMETERS = auto()  # Reliable, transient local


@dataclass
class ROS2Config:
    """ROS2 configuration."""
    node_name: str = "surgical_robot"
    namespace: str = ""
    domain_id: int = 0
    use_sim_time: bool = False

    # Topic configuration
    joint_state_topic: str = "joint_states"
    robot_state_topic: str = "robot_state"
    command_topic: str = "robot_command"
    tf_topic: str = "tf"

    # QoS settings
    sensor_qos: QoSProfile = QoSProfile.SENSOR_DATA
    command_qos: QoSProfile = QoSProfile.RELIABLE


@dataclass
class JointStateMessage:
    """ROS2 JointState message equivalent."""
    header_stamp: float
    header_frame_id: str
    name: List[str]
    position: List[float]
    velocity: List[float]
    effort: List[float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "header": {
                "stamp": self.header_stamp,
                "frame_id": self.header_frame_id
            },
            "name": self.name,
            "position": self.position,
            "velocity": self.velocity,
            "effort": self.effort
        }

    @classmethod
    def from_joint_states(
        cls,
        joint_states: List[JointState],
        joint_names: List[str],
        frame_id: str = "base_link"
    ) -> "JointStateMessage":
        """Create from JointState objects."""
        return cls(
            header_stamp=time.time(),
            header_frame_id=frame_id,
            name=joint_names,
            position=[js.position for js in joint_states],
            velocity=[js.velocity for js in joint_states],
            effort=[js.torque for js in joint_states]
        )


@dataclass
class PoseStampedMessage:
    """ROS2 PoseStamped message equivalent."""
    header_stamp: float
    header_frame_id: str
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [x, y, z, w] quaternion

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "header": {
                "stamp": self.header_stamp,
                "frame_id": self.header_frame_id
            },
            "pose": {
                "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
                "orientation": {
                    "x": self.orientation[0],
                    "y": self.orientation[1],
                    "z": self.orientation[2],
                    "w": self.orientation[3]
                }
            }
        }

    @classmethod
    def from_pose(cls, pose: Pose, frame_id: str = "base_link") -> "PoseStampedMessage":
        """Create from Pose object."""
        return cls(
            header_stamp=time.time(),
            header_frame_id=frame_id,
            position=pose.position.tolist(),
            orientation=[
                pose.orientation[1],  # x
                pose.orientation[2],  # y
                pose.orientation[3],  # z
                pose.orientation[0]   # w
            ]
        )


@dataclass
class TransformMessage:
    """ROS2 Transform message equivalent."""
    header_stamp: float
    header_frame_id: str
    child_frame_id: str
    translation: List[float]
    rotation: List[float]  # quaternion [x, y, z, w]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "header": {
                "stamp": self.header_stamp,
                "frame_id": self.header_frame_id
            },
            "child_frame_id": self.child_frame_id,
            "transform": {
                "translation": {
                    "x": self.translation[0],
                    "y": self.translation[1],
                    "z": self.translation[2]
                },
                "rotation": {
                    "x": self.rotation[0],
                    "y": self.rotation[1],
                    "z": self.rotation[2],
                    "w": self.rotation[3]
                }
            }
        }


class ROS2Publisher(ABC):
    """Abstract base class for ROS2 publishers."""

    def __init__(self, topic: str, qos: QoSProfile = QoSProfile.SENSOR_DATA) -> None:
        self.topic = topic
        self.qos = qos
        self._callbacks: List[Callable[[dict], None]] = []

    def register_callback(self, callback: Callable[[dict], None]) -> None:
        """Register callback for published messages."""
        self._callbacks.append(callback)

    @abstractmethod
    def publish(self, message: Any) -> None:
        """Publish message."""
        pass

    def _notify_callbacks(self, message_dict: dict) -> None:
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            cb(message_dict)


class ROS2Subscriber(ABC):
    """Abstract base class for ROS2 subscribers."""

    def __init__(self, topic: str, qos: QoSProfile = QoSProfile.SENSOR_DATA) -> None:
        self.topic = topic
        self.qos = qos
        self._message_callback: Optional[Callable[[dict], None]] = None

    def set_callback(self, callback: Callable[[dict], None]) -> None:
        """Set message callback."""
        self._message_callback = callback

    @abstractmethod
    def spin_once(self) -> None:
        """Process one message if available."""
        pass


class ROS2JointStatePublisher(ROS2Publisher):
    """Publisher for joint state messages."""

    def __init__(
        self,
        topic: str = "joint_states",
        joint_names: Optional[List[str]] = None
    ) -> None:
        super().__init__(topic, QoSProfile.SENSOR_DATA)
        self.joint_names = joint_names or []

    def publish(self, joint_states: List[JointState]) -> None:
        """Publish joint states."""
        if not self.joint_names:
            self.joint_names = [f"joint_{i}" for i in range(len(joint_states))]

        message = JointStateMessage.from_joint_states(
            joint_states, self.joint_names
        )
        self._notify_callbacks(message.to_dict())

    def publish_from_arrays(
        self,
        positions: NDArray[np.float64],
        velocities: Optional[NDArray[np.float64]] = None,
        efforts: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Publish from numpy arrays."""
        n = len(positions)
        if not self.joint_names:
            self.joint_names = [f"joint_{i}" for i in range(n)]

        message = JointStateMessage(
            header_stamp=time.time(),
            header_frame_id="base_link",
            name=self.joint_names[:n],
            position=positions.tolist(),
            velocity=velocities.tolist() if velocities is not None else [0.0] * n,
            effort=efforts.tolist() if efforts is not None else [0.0] * n
        )
        self._notify_callbacks(message.to_dict())


class ROS2PosePublisher(ROS2Publisher):
    """Publisher for pose messages."""

    def __init__(
        self,
        topic: str = "end_effector_pose",
        frame_id: str = "base_link"
    ) -> None:
        super().__init__(topic, QoSProfile.SENSOR_DATA)
        self.frame_id = frame_id

    def publish(self, pose: Pose) -> None:
        """Publish pose."""
        message = PoseStampedMessage.from_pose(pose, self.frame_id)
        self._notify_callbacks(message.to_dict())


class ROS2TFPublisher(ROS2Publisher):
    """Publisher for TF transforms."""

    def __init__(self, topic: str = "tf") -> None:
        super().__init__(topic, QoSProfile.SENSOR_DATA)
        self._static_transforms: List[TransformMessage] = []

    def publish_transform(
        self,
        parent_frame: str,
        child_frame: str,
        translation: NDArray[np.float64],
        rotation: NDArray[np.float64]
    ) -> None:
        """Publish a single transform."""
        message = TransformMessage(
            header_stamp=time.time(),
            header_frame_id=parent_frame,
            child_frame_id=child_frame,
            translation=translation.tolist(),
            rotation=[rotation[1], rotation[2], rotation[3], rotation[0]]  # xyzw
        )
        self._notify_callbacks({"transforms": [message.to_dict()]})

    def publish(self, transforms: List[TransformMessage]) -> None:
        """Publish multiple transforms."""
        self._notify_callbacks({
            "transforms": [t.to_dict() for t in transforms]
        })

    def add_static_transform(
        self,
        parent_frame: str,
        child_frame: str,
        translation: NDArray[np.float64],
        rotation: NDArray[np.float64]
    ) -> None:
        """Add a static transform."""
        self._static_transforms.append(TransformMessage(
            header_stamp=time.time(),
            header_frame_id=parent_frame,
            child_frame_id=child_frame,
            translation=translation.tolist(),
            rotation=[rotation[1], rotation[2], rotation[3], rotation[0]]
        ))


class ROS2Bridge:
    """
    Bridge for ROS2 communication.

    Provides a complete interface for publishing and subscribing
    to ROS2 topics without requiring ROS2 installation.
    Can be extended to use actual rclpy when available.
    """

    def __init__(self, config: Optional[ROS2Config] = None) -> None:
        self.config = config or ROS2Config()
        self.publishers: Dict[str, ROS2Publisher] = {}
        self.subscribers: Dict[str, ROS2Subscriber] = {}

        self._running = False
        self._message_queue: List[dict] = []

        # Standard publishers
        self.joint_state_publisher: Optional[ROS2JointStatePublisher] = None
        self.pose_publisher: Optional[ROS2PosePublisher] = None
        self.tf_publisher: Optional[ROS2TFPublisher] = None

    def initialize(self) -> bool:
        """Initialize ROS2 bridge."""
        try:
            # Set up standard publishers
            self.joint_state_publisher = ROS2JointStatePublisher(
                self.config.joint_state_topic
            )
            self.pose_publisher = ROS2PosePublisher()
            self.tf_publisher = ROS2TFPublisher()

            self.publishers[self.config.joint_state_topic] = self.joint_state_publisher
            self.publishers["end_effector_pose"] = self.pose_publisher
            self.publishers[self.config.tf_topic] = self.tf_publisher

            self._running = True
            return True

        except Exception:
            return False

    def shutdown(self) -> None:
        """Shutdown ROS2 bridge."""
        self._running = False

    def add_publisher(self, topic: str, publisher: ROS2Publisher) -> None:
        """Add custom publisher."""
        self.publishers[topic] = publisher

    def add_subscriber(self, topic: str, subscriber: ROS2Subscriber) -> None:
        """Add custom subscriber."""
        self.subscribers[topic] = subscriber

    def spin_once(self) -> None:
        """Process one iteration of callbacks."""
        for subscriber in self.subscribers.values():
            subscriber.spin_once()

    def get_time(self) -> float:
        """Get current ROS time."""
        if self.config.use_sim_time:
            # Would use simulation time
            return time.time()
        return time.time()

    def publish_robot_state(
        self,
        joint_states: List[JointState],
        end_effector_pose: Optional[Pose] = None,
        transforms: Optional[List[tuple]] = None
    ) -> None:
        """Publish complete robot state."""
        if self.joint_state_publisher:
            self.joint_state_publisher.publish(joint_states)

        if end_effector_pose and self.pose_publisher:
            self.pose_publisher.publish(end_effector_pose)

        if transforms and self.tf_publisher:
            for parent, child, trans, rot in transforms:
                self.tf_publisher.publish_transform(parent, child, trans, rot)


class URDFParser:
    """Parser for URDF robot descriptions."""

    def __init__(self) -> None:
        self.links: Dict[str, dict] = {}
        self.joints: Dict[str, dict] = {}
        self.robot_name: str = ""

    def parse(self, urdf_content: str) -> bool:
        """Parse URDF XML content."""
        # Simplified URDF parsing
        # Real implementation would use proper XML parsing
        try:
            # Would parse XML and extract links/joints
            return True
        except Exception:
            return False

    def get_joint_names(self) -> List[str]:
        """Get all joint names."""
        return list(self.joints.keys())

    def get_link_names(self) -> List[str]:
        """Get all link names."""
        return list(self.links.keys())

    def get_joint_limits(self, joint_name: str) -> Optional[dict]:
        """Get limits for a joint."""
        if joint_name in self.joints:
            return self.joints[joint_name].get("limits")
        return None
