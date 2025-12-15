"""Physics simulation backend for surgical robotics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray


class PhysicsBackend(Enum):
    """Available physics backends."""
    PYBULLET = auto()
    MUJOCO = auto()
    INTERNAL = auto()  # Simplified internal physics


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation."""
    backend: PhysicsBackend = PhysicsBackend.INTERNAL

    # Time stepping
    time_step: float = 0.001  # 1ms physics step
    substeps: int = 10  # Substeps per control step

    # Gravity
    gravity: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, -9.81])
    )

    # Solver parameters
    solver_iterations: int = 50
    contact_stiffness: float = 1e5
    contact_damping: float = 1e3
    friction_coefficient: float = 0.5

    # Rendering
    enable_rendering: bool = True
    render_rate: float = 60.0  # Hz


@dataclass
class CollisionShape:
    """Collision shape definition."""
    shape_type: str  # 'sphere', 'box', 'cylinder', 'mesh', 'capsule'
    dimensions: NDArray[np.float64]  # Shape-specific dimensions
    position: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    orientation: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1, 0, 0, 0])
    )
    mesh_path: Optional[str] = None


@dataclass
class RigidBody:
    """Rigid body definition."""
    name: str
    mass: float
    position: NDArray[np.float64]
    orientation: NDArray[np.float64]

    # Inertia (computed from shape if not provided)
    inertia: Optional[NDArray[np.float64]] = None

    # Collision
    collision_shapes: List[CollisionShape] = field(default_factory=list)

    # Visual (can differ from collision)
    visual_mesh: Optional[str] = None
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.7, 1.0])
    )

    # Dynamics
    linear_velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    angular_velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    is_static: bool = False

    # State (set by simulation)
    body_id: Optional[int] = None


@dataclass
class SoftBody:
    """Soft body definition for deformable objects."""
    name: str
    mesh_path: str
    position: NDArray[np.float64]
    orientation: NDArray[np.float64]

    # Material properties
    mass: float = 1.0
    stiffness: float = 100.0  # Young's modulus approximation
    damping: float = 0.1
    poisson_ratio: float = 0.3

    # Simulation parameters
    num_solver_iterations: int = 10
    collision_margin: float = 0.01

    # Visual
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.9, 0.6, 0.6, 1.0])
    )

    # State
    body_id: Optional[int] = None
    vertices: Optional[NDArray[np.float64]] = None


@dataclass
class ContactInfo:
    """Contact information between bodies."""
    body_a: str
    body_b: str
    contact_point: NDArray[np.float64]
    contact_normal: NDArray[np.float64]
    penetration_depth: float
    normal_force: float
    friction_force: NDArray[np.float64]


class PhysicsEngine(ABC):
    """
    Abstract physics engine interface.

    Provides unified interface for different physics backends.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        self.config = config or PhysicsConfig()
        self.bodies: Dict[str, RigidBody] = {}
        self.soft_bodies: Dict[str, SoftBody] = {}
        self.constraints: Dict[str, Any] = {}
        self._time = 0.0
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize physics engine."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown physics engine."""
        pass

    @abstractmethod
    def step(self, dt: Optional[float] = None) -> None:
        """Step simulation forward."""
        pass

    @abstractmethod
    def add_rigid_body(self, body: RigidBody) -> int:
        """Add rigid body to simulation."""
        pass

    @abstractmethod
    def add_soft_body(self, body: SoftBody) -> int:
        """Add soft body to simulation."""
        pass

    @abstractmethod
    def remove_body(self, name: str) -> bool:
        """Remove body from simulation."""
        pass

    @abstractmethod
    def get_body_state(
        self,
        name: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get body position and orientation."""
        pass

    @abstractmethod
    def set_body_state(
        self,
        name: str,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64]
    ) -> None:
        """Set body position and orientation."""
        pass

    @abstractmethod
    def apply_force(
        self,
        name: str,
        force: NDArray[np.float64],
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Apply force to body."""
        pass

    @abstractmethod
    def get_contacts(self) -> List[ContactInfo]:
        """Get all current contacts."""
        pass

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time


class InternalPhysicsEngine(PhysicsEngine):
    """
    Simplified internal physics engine.

    Provides basic physics without external dependencies.
    Suitable for testing and simple simulations.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        super().__init__(config)
        self._forces: Dict[str, NDArray[np.float64]] = {}
        self._torques: Dict[str, NDArray[np.float64]] = {}

    def initialize(self) -> bool:
        """Initialize physics engine."""
        self._initialized = True
        return True

    def shutdown(self) -> None:
        """Shutdown physics engine."""
        self.bodies.clear()
        self.soft_bodies.clear()
        self._initialized = False

    def step(self, dt: Optional[float] = None) -> None:
        """Step simulation forward."""
        if not self._initialized:
            return

        dt = dt or self.config.time_step

        for _ in range(self.config.substeps):
            sub_dt = dt / self.config.substeps
            self._step_bodies(sub_dt)

        self._time += dt

    def _step_bodies(self, dt: float) -> None:
        """Step all bodies forward."""
        for name, body in self.bodies.items():
            if body.is_static:
                continue

            # Get forces
            force = self._forces.get(name, np.zeros(3))
            torque = self._torques.get(name, np.zeros(3))

            # Add gravity
            force = force + body.mass * self.config.gravity

            # Integrate linear motion
            if body.mass > 0:
                acceleration = force / body.mass
                body.linear_velocity = body.linear_velocity + acceleration * dt
                body.position = body.position + body.linear_velocity * dt

            # Integrate angular motion (simplified)
            if body.inertia is not None:
                # Assuming diagonal inertia tensor
                angular_accel = torque / body.inertia
                body.angular_velocity = body.angular_velocity + angular_accel * dt

            # Update orientation (simplified quaternion integration)
            omega = body.angular_velocity
            omega_mag = np.linalg.norm(omega)
            if omega_mag > 1e-10:
                axis = omega / omega_mag
                angle = omega_mag * dt
                dq = np.array([
                    np.cos(angle / 2),
                    axis[0] * np.sin(angle / 2),
                    axis[1] * np.sin(angle / 2),
                    axis[2] * np.sin(angle / 2)
                ])
                body.orientation = self._quaternion_multiply(
                    body.orientation, dq
                )
                body.orientation = body.orientation / np.linalg.norm(body.orientation)

        # Clear applied forces
        self._forces.clear()
        self._torques.clear()

    def add_rigid_body(self, body: RigidBody) -> int:
        """Add rigid body to simulation."""
        body_id = len(self.bodies)
        body.body_id = body_id

        # Compute default inertia if not provided
        if body.inertia is None:
            body.inertia = np.array([1.0, 1.0, 1.0]) * body.mass / 6.0

        self.bodies[body.name] = body
        return body_id

    def add_soft_body(self, body: SoftBody) -> int:
        """Add soft body to simulation."""
        body_id = len(self.soft_bodies)
        body.body_id = body_id
        self.soft_bodies[body.name] = body
        return body_id

    def remove_body(self, name: str) -> bool:
        """Remove body from simulation."""
        if name in self.bodies:
            del self.bodies[name]
            return True
        if name in self.soft_bodies:
            del self.soft_bodies[name]
            return True
        return False

    def get_body_state(
        self,
        name: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get body position and orientation."""
        if name in self.bodies:
            body = self.bodies[name]
            return body.position.copy(), body.orientation.copy()
        raise ValueError(f"Body '{name}' not found")

    def set_body_state(
        self,
        name: str,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64]
    ) -> None:
        """Set body position and orientation."""
        if name in self.bodies:
            self.bodies[name].position = position.copy()
            self.bodies[name].orientation = orientation.copy()
        else:
            raise ValueError(f"Body '{name}' not found")

    def apply_force(
        self,
        name: str,
        force: NDArray[np.float64],
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Apply force to body."""
        if name not in self.bodies:
            return

        if name not in self._forces:
            self._forces[name] = np.zeros(3)
        self._forces[name] += force

        # Apply torque if force is applied at offset position
        if position is not None:
            body = self.bodies[name]
            r = position - body.position
            torque = np.cross(r, force)
            if name not in self._torques:
                self._torques[name] = np.zeros(3)
            self._torques[name] += torque

    def get_contacts(self) -> List[ContactInfo]:
        """Get all current contacts (simplified collision detection)."""
        contacts = []

        body_list = list(self.bodies.values())
        for i, body_a in enumerate(body_list):
            for body_b in body_list[i + 1:]:
                contact = self._check_sphere_collision(body_a, body_b)
                if contact is not None:
                    contacts.append(contact)

        return contacts

    def _check_sphere_collision(
        self,
        body_a: RigidBody,
        body_b: RigidBody
    ) -> Optional[ContactInfo]:
        """Simplified sphere-sphere collision check."""
        # Use bounding sphere approximation
        radius_a = 0.1  # Default radius
        radius_b = 0.1

        for shape in body_a.collision_shapes:
            if shape.shape_type == 'sphere':
                radius_a = shape.dimensions[0]
        for shape in body_b.collision_shapes:
            if shape.shape_type == 'sphere':
                radius_b = shape.dimensions[0]

        diff = body_b.position - body_a.position
        distance = float(np.linalg.norm(diff))

        if distance < radius_a + radius_b:
            normal = diff / distance if distance > 1e-10 else np.array([0, 0, 1])
            penetration = radius_a + radius_b - distance
            contact_point = body_a.position + normal * radius_a

            return ContactInfo(
                body_a=body_a.name,
                body_b=body_b.name,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=penetration,
                normal_force=penetration * self.config.contact_stiffness,
                friction_force=np.zeros(3)
            )

        return None

    @staticmethod
    def _quaternion_multiply(
        q1: NDArray[np.float64],
        q2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class PyBulletEngine(PhysicsEngine):
    """
    PyBullet physics engine wrapper.

    Provides high-fidelity physics simulation using PyBullet.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        super().__init__(config)
        self._client_id: Optional[int] = None
        self._body_ids: Dict[str, int] = {}

    def initialize(self) -> bool:
        """Initialize PyBullet."""
        try:
            import pybullet as p
            import pybullet_data

            if self.config.enable_rendering:
                self._client_id = p.connect(p.GUI)
            else:
                self._client_id = p.connect(p.DIRECT)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(*self.config.gravity, physicsClientId=self._client_id)
            p.setTimeStep(self.config.time_step, physicsClientId=self._client_id)

            self._initialized = True
            return True

        except ImportError:
            print("PyBullet not available, falling back to internal physics")
            return False

    def shutdown(self) -> None:
        """Shutdown PyBullet."""
        if self._client_id is not None:
            try:
                import pybullet as p
                p.disconnect(self._client_id)
            except Exception:
                pass
        self._initialized = False

    def step(self, dt: Optional[float] = None) -> None:
        """Step simulation."""
        if not self._initialized or self._client_id is None:
            return

        try:
            import pybullet as p

            steps = self.config.substeps if dt is None else int(dt / self.config.time_step)
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self._client_id)

            self._time += (dt or self.config.time_step * self.config.substeps)

        except Exception:
            pass

    def add_rigid_body(self, body: RigidBody) -> int:
        """Add rigid body to PyBullet."""
        if not self._initialized:
            return -1

        try:
            import pybullet as p

            # Create collision shape
            if body.collision_shapes:
                shape = body.collision_shapes[0]
                if shape.shape_type == 'sphere':
                    collision_id = p.createCollisionShape(
                        p.GEOM_SPHERE,
                        radius=shape.dimensions[0],
                        physicsClientId=self._client_id
                    )
                elif shape.shape_type == 'box':
                    collision_id = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=shape.dimensions / 2,
                        physicsClientId=self._client_id
                    )
                elif shape.shape_type == 'mesh' and shape.mesh_path:
                    collision_id = p.createCollisionShape(
                        p.GEOM_MESH,
                        fileName=shape.mesh_path,
                        physicsClientId=self._client_id
                    )
                else:
                    collision_id = p.createCollisionShape(
                        p.GEOM_SPHERE,
                        radius=0.1,
                        physicsClientId=self._client_id
                    )
            else:
                collision_id = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=0.1,
                    physicsClientId=self._client_id
                )

            # Create body
            mass = 0.0 if body.is_static else body.mass
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_id,
                basePosition=body.position.tolist(),
                baseOrientation=body.orientation[[1, 2, 3, 0]].tolist(),  # xyzw
                physicsClientId=self._client_id
            )

            body.body_id = body_id
            self.bodies[body.name] = body
            self._body_ids[body.name] = body_id

            return body_id

        except Exception:
            return -1

    def add_soft_body(self, body: SoftBody) -> int:
        """Add soft body to PyBullet."""
        if not self._initialized:
            return -1

        try:
            import pybullet as p

            body_id = p.loadSoftBody(
                body.mesh_path,
                basePosition=body.position.tolist(),
                baseOrientation=body.orientation[[1, 2, 3, 0]].tolist(),
                mass=body.mass,
                useNeoHookean=1,
                NeoHookeanMu=body.stiffness,
                NeoHookeanLambda=body.stiffness * body.poisson_ratio,
                NeoHookeanDamping=body.damping,
                useSelfCollision=1,
                physicsClientId=self._client_id
            )

            body.body_id = body_id
            self.soft_bodies[body.name] = body

            return body_id

        except Exception:
            return -1

    def remove_body(self, name: str) -> bool:
        """Remove body from PyBullet."""
        if not self._initialized:
            return False

        try:
            import pybullet as p

            if name in self._body_ids:
                p.removeBody(self._body_ids[name], physicsClientId=self._client_id)
                del self._body_ids[name]
                del self.bodies[name]
                return True

        except Exception:
            pass
        return False

    def get_body_state(
        self,
        name: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get body state from PyBullet."""
        if not self._initialized or name not in self._body_ids:
            raise ValueError(f"Body '{name}' not found")

        try:
            import pybullet as p

            pos, orn = p.getBasePositionAndOrientation(
                self._body_ids[name],
                physicsClientId=self._client_id
            )
            # Convert from xyzw to wxyz
            orientation = np.array([orn[3], orn[0], orn[1], orn[2]])
            return np.array(pos), orientation

        except Exception:
            raise

    def set_body_state(
        self,
        name: str,
        position: NDArray[np.float64],
        orientation: NDArray[np.float64]
    ) -> None:
        """Set body state in PyBullet."""
        if not self._initialized or name not in self._body_ids:
            return

        try:
            import pybullet as p

            # Convert from wxyz to xyzw
            orn = [orientation[1], orientation[2], orientation[3], orientation[0]]
            p.resetBasePositionAndOrientation(
                self._body_ids[name],
                position.tolist(),
                orn,
                physicsClientId=self._client_id
            )
        except Exception:
            pass

    def apply_force(
        self,
        name: str,
        force: NDArray[np.float64],
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Apply force in PyBullet."""
        if not self._initialized or name not in self._body_ids:
            return

        try:
            import pybullet as p

            if position is None:
                position = np.zeros(3)

            p.applyExternalForce(
                self._body_ids[name],
                -1,  # Base link
                force.tolist(),
                position.tolist(),
                p.WORLD_FRAME,
                physicsClientId=self._client_id
            )
        except Exception:
            pass

    def get_contacts(self) -> List[ContactInfo]:
        """Get contacts from PyBullet."""
        contacts = []

        if not self._initialized:
            return contacts

        try:
            import pybullet as p

            contact_points = p.getContactPoints(physicsClientId=self._client_id)

            # Reverse lookup for body names
            id_to_name = {v: k for k, v in self._body_ids.items()}

            for cp in contact_points:
                body_a = id_to_name.get(cp[1], f"body_{cp[1]}")
                body_b = id_to_name.get(cp[2], f"body_{cp[2]}")

                contacts.append(ContactInfo(
                    body_a=body_a,
                    body_b=body_b,
                    contact_point=np.array(cp[5]),
                    contact_normal=np.array(cp[7]),
                    penetration_depth=-cp[8],
                    normal_force=cp[9],
                    friction_force=np.array([cp[10], cp[12], 0])
                ))

        except Exception:
            pass

        return contacts


def create_physics_engine(config: Optional[PhysicsConfig] = None) -> PhysicsEngine:
    """
    Factory function to create appropriate physics engine.

    Tries PyBullet first, falls back to internal physics.
    """
    config = config or PhysicsConfig()

    if config.backend == PhysicsBackend.PYBULLET:
        engine = PyBulletEngine(config)
        if engine.initialize():
            return engine

    if config.backend == PhysicsBackend.MUJOCO:
        # MuJoCo implementation would go here
        pass

    # Fall back to internal physics
    config.backend = PhysicsBackend.INTERNAL
    engine = InternalPhysicsEngine(config)
    engine.initialize()
    return engine
