"""Surgical scene management for simulation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.simulation.physics import (
    PhysicsEngine,
    RigidBody,
    SoftBody,
    CollisionShape,
    create_physics_engine,
    PhysicsConfig,
)


class ObjectType(Enum):
    """Types of scene objects."""
    STATIC = auto()  # Immovable objects
    DYNAMIC = auto()  # Movable rigid bodies
    SOFT = auto()  # Deformable objects
    TOOL = auto()  # Surgical tools
    ANATOMY = auto()  # Patient anatomy


@dataclass
class SceneObject:
    """Base class for scene objects."""
    name: str
    object_type: ObjectType
    position: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    orientation: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1, 0, 0, 0])
    )
    scale: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(3)
    )

    # Visual properties
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.7, 1.0])
    )
    mesh_path: Optional[str] = None
    visible: bool = True

    # Physics
    mass: float = 1.0
    collision_enabled: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal state
    _physics_body: Optional[RigidBody] = None


@dataclass
class OperatingTable:
    """Operating table model."""
    position: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, 0.9])  # Standard height
    )
    dimensions: NDArray[np.float64] = field(
        default_factory=lambda: np.array([2.0, 0.6, 0.05])  # L x W x H
    )
    height_adjustable: bool = True
    tilt_angle: float = 0.0  # degrees
    rotation_angle: float = 0.0  # degrees

    # Padding
    has_padding: bool = True
    padding_thickness: float = 0.05

    def get_surface_height(self) -> float:
        """Get height of table surface."""
        height = self.position[2] + self.dimensions[2] / 2
        if self.has_padding:
            height += self.padding_thickness
        return height

    def set_height(self, height: float) -> None:
        """Adjust table height."""
        if self.height_adjustable:
            self.position[2] = height - self.dimensions[2] / 2

    def apply_tilt(self, angle: float) -> None:
        """Apply Trendelenburg tilt."""
        self.tilt_angle = np.clip(angle, -30, 30)

    def apply_rotation(self, angle: float) -> None:
        """Rotate table."""
        self.rotation_angle = angle % 360


@dataclass
class PatientModel:
    """Patient anatomy model for simulation."""
    name: str = "patient"
    position: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    orientation: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1, 0, 0, 0])
    )

    # Body dimensions (approximate)
    height: float = 1.7  # meters
    weight: float = 70.0  # kg

    # Anatomy regions
    anatomy_regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Surface mesh for collision
    surface_mesh: Optional[str] = None

    # Internal structures
    organ_meshes: Dict[str, str] = field(default_factory=dict)

    def get_region_bounds(self, region: str) -> Optional[Dict[str, NDArray]]:
        """Get bounding box of anatomical region."""
        if region in self.anatomy_regions:
            return self.anatomy_regions[region].get('bounds')
        return None

    def add_region(
        self,
        name: str,
        center: NDArray[np.float64],
        size: NDArray[np.float64],
        tissue_type: str = "soft"
    ) -> None:
        """Add anatomical region."""
        self.anatomy_regions[name] = {
            'center': center,
            'size': size,
            'tissue_type': tissue_type,
            'bounds': {
                'min': center - size / 2,
                'max': center + size / 2
            }
        }


class SurgicalScene:
    """
    Manages complete surgical simulation scene.

    Includes:
    - Operating room setup
    - Patient positioning
    - Robot(s) placement
    - Surgical tools
    - Lighting and cameras
    """

    def __init__(
        self,
        physics_config: Optional[PhysicsConfig] = None
    ) -> None:
        self.physics = create_physics_engine(physics_config)
        self.objects: Dict[str, SceneObject] = {}

        # Scene components
        self.operating_table: Optional[OperatingTable] = None
        self.patient: Optional[PatientModel] = None
        self.robots: Dict[str, Any] = {}
        self.tools: Dict[str, SceneObject] = {}
        self.cameras: Dict[str, Dict[str, Any]] = {}

        # Scene bounds
        self.workspace_bounds = {
            'min': np.array([-2.0, -2.0, 0.0]),
            'max': np.array([2.0, 2.0, 3.0])
        }

        # Callbacks
        self._on_collision: Optional[Callable[[str, str], None]] = None

    def setup_operating_room(self) -> None:
        """Set up standard operating room environment."""
        # Add floor
        floor = SceneObject(
            name="floor",
            object_type=ObjectType.STATIC,
            position=np.array([0, 0, -0.05]),
            color=np.array([0.3, 0.3, 0.3, 1.0]),
            mass=0.0
        )
        self.add_object(floor)

        # Add operating table
        self.operating_table = OperatingTable()
        table_obj = SceneObject(
            name="operating_table",
            object_type=ObjectType.STATIC,
            position=self.operating_table.position,
            color=np.array([0.8, 0.8, 0.9, 1.0]),
            mass=0.0
        )
        self.add_object(table_obj)

        # Add OR lights (visual only)
        light_positions = [
            np.array([0, 0, 2.5]),
            np.array([0.5, 0.3, 2.3]),
            np.array([-0.5, 0.3, 2.3])
        ]
        for i, pos in enumerate(light_positions):
            self.cameras[f"light_{i}"] = {
                'type': 'light',
                'position': pos,
                'intensity': 1.0
            }

    def add_patient(
        self,
        patient: Optional[PatientModel] = None,
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Add patient model to scene."""
        if patient is None:
            patient = PatientModel()

        if position is None and self.operating_table:
            # Place on operating table
            surface_height = self.operating_table.get_surface_height()
            position = np.array([0, 0, surface_height])

        if position is not None:
            patient.position = position

        self.patient = patient

        # Add default anatomy regions
        patient.add_region(
            "abdomen",
            center=patient.position + np.array([0, 0, 0.15]),
            size=np.array([0.3, 0.4, 0.2]),
            tissue_type="soft"
        )
        patient.add_region(
            "chest",
            center=patient.position + np.array([0, 0.3, 0.12]),
            size=np.array([0.35, 0.3, 0.25]),
            tissue_type="soft"
        )
        patient.add_region(
            "head",
            center=patient.position + np.array([0, 0.7, 0.1]),
            size=np.array([0.18, 0.22, 0.22]),
            tissue_type="hard"
        )

        # Create physics representation
        patient_obj = SceneObject(
            name="patient_body",
            object_type=ObjectType.ANATOMY,
            position=patient.position,
            color=np.array([0.9, 0.75, 0.65, 1.0]),
            mass=patient.weight,
            collision_enabled=True
        )
        self.add_object(patient_obj)

    def add_object(self, obj: SceneObject) -> bool:
        """Add object to scene."""
        if obj.name in self.objects:
            return False

        self.objects[obj.name] = obj

        # Create physics body
        if obj.collision_enabled:
            is_static = obj.object_type == ObjectType.STATIC

            # Determine collision shape
            if obj.mesh_path:
                collision_shape = CollisionShape(
                    shape_type='mesh',
                    dimensions=obj.scale,
                    mesh_path=obj.mesh_path
                )
            else:
                collision_shape = CollisionShape(
                    shape_type='box',
                    dimensions=obj.scale * 0.1
                )

            body = RigidBody(
                name=obj.name,
                mass=obj.mass,
                position=obj.position.copy(),
                orientation=obj.orientation.copy(),
                collision_shapes=[collision_shape],
                color=obj.color,
                is_static=is_static
            )

            self.physics.add_rigid_body(body)
            obj._physics_body = body

        return True

    def remove_object(self, name: str) -> bool:
        """Remove object from scene."""
        if name not in self.objects:
            return False

        self.physics.remove_body(name)
        del self.objects[name]
        return True

    def add_robot(
        self,
        robot: Any,
        name: str,
        base_position: NDArray[np.float64]
    ) -> None:
        """Add surgical robot to scene."""
        self.robots[name] = {
            'robot': robot,
            'base_position': base_position,
            'enabled': True
        }

    def add_tool(
        self,
        name: str,
        tool_type: str,
        position: NDArray[np.float64],
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add surgical tool to scene."""
        tool = SceneObject(
            name=name,
            object_type=ObjectType.TOOL,
            position=position,
            color=np.array([0.6, 0.6, 0.7, 1.0]),
            mass=0.1,
            metadata={
                'tool_type': tool_type,
                'properties': properties or {}
            }
        )
        self.tools[name] = tool
        self.add_object(tool)

    def add_camera(
        self,
        name: str,
        position: NDArray[np.float64],
        target: NDArray[np.float64],
        fov: float = 60.0
    ) -> None:
        """Add camera to scene."""
        direction = target - position
        direction = direction / np.linalg.norm(direction)

        self.cameras[name] = {
            'type': 'camera',
            'position': position.copy(),
            'target': target.copy(),
            'direction': direction,
            'fov': fov,
            'near': 0.01,
            'far': 10.0
        }

    def step(self, dt: float) -> None:
        """Step simulation forward."""
        self.physics.step(dt)

        # Update object positions from physics
        for name, obj in self.objects.items():
            if obj._physics_body is not None:
                try:
                    pos, orn = self.physics.get_body_state(name)
                    obj.position = pos
                    obj.orientation = orn
                except Exception:
                    pass

        # Check for collisions
        if self._on_collision:
            contacts = self.physics.get_contacts()
            for contact in contacts:
                self._on_collision(contact.body_a, contact.body_b)

    def get_object_pose(
        self,
        name: str
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        """Get object position and orientation."""
        if name in self.objects:
            obj = self.objects[name]
            return {
                'position': obj.position.copy(),
                'orientation': obj.orientation.copy()
            }
        return None

    def set_object_pose(
        self,
        name: str,
        position: NDArray[np.float64],
        orientation: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Set object position and orientation."""
        if name not in self.objects:
            return

        obj = self.objects[name]
        obj.position = position.copy()
        if orientation is not None:
            obj.orientation = orientation.copy()

        self.physics.set_body_state(
            name,
            obj.position,
            obj.orientation
        )

    def apply_force_to_object(
        self,
        name: str,
        force: NDArray[np.float64],
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Apply force to scene object."""
        self.physics.apply_force(name, force, position)

    def get_contacts(self) -> List[Dict[str, Any]]:
        """Get all current contacts."""
        contacts = self.physics.get_contacts()
        return [
            {
                'body_a': c.body_a,
                'body_b': c.body_b,
                'point': c.contact_point,
                'normal': c.contact_normal,
                'force': c.normal_force
            }
            for c in contacts
        ]

    def set_collision_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """Set callback for collision events."""
        self._on_collision = callback

    def reset(self) -> None:
        """Reset scene to initial state."""
        for name, obj in self.objects.items():
            if name in ['floor', 'operating_table']:
                continue
            self.physics.set_body_state(
                name,
                obj.position,
                obj.orientation
            )

    def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace information for robot setup."""
        return {
            'bounds': self.workspace_bounds,
            'table_height': (
                self.operating_table.get_surface_height()
                if self.operating_table else 0.9
            ),
            'patient_regions': (
                self.patient.anatomy_regions
                if self.patient else {}
            ),
            'robot_bases': {
                name: data['base_position']
                for name, data in self.robots.items()
            }
        }

    def shutdown(self) -> None:
        """Clean up scene resources."""
        self.physics.shutdown()
        self.objects.clear()
        self.robots.clear()
        self.tools.clear()


class SurgicalProcedureScene(SurgicalScene):
    """
    Specialized scene for specific surgical procedures.

    Pre-configured setups for common procedures.
    """

    @classmethod
    def create_laparoscopic_scene(cls) -> "SurgicalProcedureScene":
        """Create scene for laparoscopic surgery."""
        scene = cls()
        scene.setup_operating_room()

        # Add patient in supine position
        scene.add_patient()

        # Add trocar ports
        trocar_positions = [
            np.array([0, -0.05, 1.1]),  # Camera port (umbilicus)
            np.array([-0.1, 0.05, 1.1]),  # Left working port
            np.array([0.1, 0.05, 1.1]),  # Right working port
            np.array([-0.15, -0.1, 1.1]),  # Assistant port
        ]

        for i, pos in enumerate(trocar_positions):
            scene.add_tool(
                f"trocar_{i}",
                "trocar",
                pos,
                {'diameter': 12 if i == 0 else 5}  # mm
            )

        # Add endoscope camera
        scene.add_camera(
            "endoscope",
            position=trocar_positions[0] + np.array([0, 0, 0.1]),
            target=trocar_positions[0] + np.array([0, 0, -0.1]),
            fov=70.0
        )

        return scene

    @classmethod
    def create_orthopedic_scene(cls) -> "SurgicalProcedureScene":
        """Create scene for orthopedic surgery."""
        scene = cls()
        scene.setup_operating_room()

        # Add patient
        scene.add_patient()

        # Add bone model (e.g., knee)
        if scene.patient:
            scene.patient.add_region(
                "knee_joint",
                center=scene.patient.position + np.array([0, -0.4, 0]),
                size=np.array([0.15, 0.15, 0.2]),
                tissue_type="bone"
            )

        # Add cutting guide
        scene.add_tool(
            "cutting_guide",
            "orthopedic_guide",
            np.array([0, -0.4, 1.0]),
            {'procedure': 'tka'}
        )

        return scene

    @classmethod
    def create_neurosurgical_scene(cls) -> "SurgicalProcedureScene":
        """Create scene for neurosurgery."""
        scene = cls()
        scene.setup_operating_room()

        # Add patient with head positioned
        scene.add_patient()

        # Adjust for head surgery
        if scene.patient:
            scene.patient.add_region(
                "brain_target",
                center=scene.patient.position + np.array([0, 0.65, 0.12]),
                size=np.array([0.01, 0.01, 0.01]),
                tissue_type="brain"
            )

        # Add stereotactic frame reference
        scene.add_tool(
            "stereotactic_frame",
            "frame",
            np.array([0, 0.65, 1.05]),
            {'frame_type': 'leksell'}
        )

        # Add surgical microscope camera
        scene.add_camera(
            "microscope",
            position=np.array([0, 0.65, 1.5]),
            target=np.array([0, 0.65, 1.0]),
            fov=20.0  # Narrow for magnification
        )

        return scene
