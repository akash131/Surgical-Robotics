"""Tissue simulation for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from numpy.typing import NDArray


class TissueType(Enum):
    """Types of biological tissue."""
    SOFT_TISSUE = auto()  # Muscle, fat, organs
    CONNECTIVE = auto()  # Fascia, ligaments
    VASCULAR = auto()  # Blood vessels
    NERVE = auto()  # Neural tissue
    BONE = auto()  # Hard tissue
    CARTILAGE = auto()  # Semi-rigid
    SKIN = auto()  # Dermis/epidermis


@dataclass
class TissueProperties:
    """Material properties for tissue simulation."""
    tissue_type: TissueType

    # Elastic properties
    youngs_modulus: float  # Pa (stiffness)
    poisson_ratio: float = 0.45  # Near-incompressible

    # Viscoelastic properties
    damping: float = 0.1
    relaxation_time: float = 0.5  # seconds

    # Cutting properties
    cutting_force_threshold: float = 5.0  # N
    tearing_force_threshold: float = 10.0  # N

    # Thermal properties (for cautery)
    thermal_conductivity: float = 0.5  # W/(m·K)
    coagulation_temperature: float = 60.0  # °C

    # Visual properties
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.9, 0.6, 0.6, 1.0])
    )
    bleeds: bool = True
    bleeding_rate: float = 1.0  # ml/s at standard vessel

    @classmethod
    def soft_tissue(cls) -> "TissueProperties":
        """Create soft tissue properties."""
        return cls(
            tissue_type=TissueType.SOFT_TISSUE,
            youngs_modulus=5000,  # 5 kPa
            poisson_ratio=0.45,
            damping=0.2,
            cutting_force_threshold=3.0,
            color=np.array([0.9, 0.6, 0.6, 1.0])
        )

    @classmethod
    def liver(cls) -> "TissueProperties":
        """Liver tissue properties."""
        return cls(
            tissue_type=TissueType.SOFT_TISSUE,
            youngs_modulus=3000,  # Very soft
            poisson_ratio=0.45,
            damping=0.15,
            cutting_force_threshold=2.0,
            bleeding_rate=2.0,
            color=np.array([0.6, 0.2, 0.2, 1.0])
        )

    @classmethod
    def bone(cls) -> "TissueProperties":
        """Cortical bone properties."""
        return cls(
            tissue_type=TissueType.BONE,
            youngs_modulus=17e9,  # 17 GPa
            poisson_ratio=0.3,
            damping=0.01,
            cutting_force_threshold=100.0,
            bleeds=True,
            bleeding_rate=0.5,
            color=np.array([0.95, 0.9, 0.8, 1.0])
        )

    @classmethod
    def vessel(cls, diameter: float = 3.0) -> "TissueProperties":
        """Blood vessel properties."""
        return cls(
            tissue_type=TissueType.VASCULAR,
            youngs_modulus=1e6,  # 1 MPa
            poisson_ratio=0.49,
            damping=0.05,
            cutting_force_threshold=1.0,
            bleeding_rate=diameter * 2.0,  # Proportional to diameter
            color=np.array([0.8, 0.2, 0.2, 1.0])
        )


@dataclass
class DeformableTissue:
    """
    Deformable tissue model using mass-spring or FEM.

    Supports:
    - Elastic deformation
    - Tool interaction
    - Cutting/tearing
    """
    name: str
    properties: TissueProperties
    position: NDArray[np.float64]

    # Mesh data
    vertices: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: NDArray[np.int32] = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))

    # Rest state
    rest_vertices: Optional[NDArray[np.float64]] = None

    # Simulation state
    velocities: Optional[NDArray[np.float64]] = None
    forces: Optional[NDArray[np.float64]] = None

    # Boundary conditions
    fixed_vertices: List[int] = field(default_factory=list)

    def initialize(self) -> None:
        """Initialize tissue simulation."""
        n_vertices = len(self.vertices)
        if n_vertices == 0:
            return

        self.rest_vertices = self.vertices.copy()
        self.velocities = np.zeros((n_vertices, 3))
        self.forces = np.zeros((n_vertices, 3))

    def apply_force(
        self,
        vertex_index: int,
        force: NDArray[np.float64]
    ) -> None:
        """Apply force to specific vertex."""
        if self.forces is not None and 0 <= vertex_index < len(self.forces):
            self.forces[vertex_index] += force

    def apply_tool_interaction(
        self,
        tool_position: NDArray[np.float64],
        tool_force: NDArray[np.float64],
        tool_radius: float = 0.01
    ) -> NDArray[np.float64]:
        """
        Apply tool force and compute reaction.

        Returns reaction force on tool.
        """
        if len(self.vertices) == 0:
            return np.zeros(3)

        # Find vertices within tool radius
        distances = np.linalg.norm(self.vertices - tool_position, axis=1)
        affected = distances < tool_radius

        if not np.any(affected):
            return np.zeros(3)

        # Distribute force to affected vertices
        n_affected = np.sum(affected)
        force_per_vertex = tool_force / n_affected

        affected_indices = np.where(affected)[0]
        for idx in affected_indices:
            self.apply_force(idx, force_per_vertex)

        # Compute reaction force (elastic response)
        displacement = np.zeros(3)
        for idx in affected_indices:
            if self.rest_vertices is not None:
                displacement += self.vertices[idx] - self.rest_vertices[idx]

        displacement /= n_affected
        reaction = -self.properties.youngs_modulus * displacement * 1e-6

        return reaction

    def step(self, dt: float) -> None:
        """Step tissue simulation forward."""
        if self.vertices.size == 0 or self.velocities is None or self.forces is None:
            return

        n_vertices = len(self.vertices)
        mass_per_vertex = 0.01  # Assume uniform mass

        # Compute internal forces (simplified spring model)
        internal_forces = self._compute_internal_forces()
        total_forces = self.forces + internal_forces

        # Add damping
        total_forces -= self.properties.damping * self.velocities

        # Integration (Verlet)
        acceleration = total_forces / mass_per_vertex

        for i in range(n_vertices):
            if i not in self.fixed_vertices:
                self.velocities[i] += acceleration[i] * dt
                self.vertices[i] += self.velocities[i] * dt

        # Clear applied forces
        self.forces = np.zeros((n_vertices, 3))

    def _compute_internal_forces(self) -> NDArray[np.float64]:
        """Compute internal elastic forces."""
        if self.rest_vertices is None:
            return np.zeros_like(self.vertices)

        forces = np.zeros_like(self.vertices)

        # Simple spring forces between triangle edges
        k = self.properties.youngs_modulus * 1e-9  # Scale for stability

        for tri in self.triangles:
            for i in range(3):
                v1 = tri[i]
                v2 = tri[(i + 1) % 3]

                # Current edge
                edge = self.vertices[v2] - self.vertices[v1]
                current_length = np.linalg.norm(edge)

                # Rest edge
                rest_edge = self.rest_vertices[v2] - self.rest_vertices[v1]
                rest_length = np.linalg.norm(rest_edge)

                if current_length < 1e-10:
                    continue

                # Spring force
                direction = edge / current_length
                strain = (current_length - rest_length) / rest_length
                force_magnitude = k * strain

                forces[v1] += force_magnitude * direction
                forces[v2] -= force_magnitude * direction

        return forces

    def get_deformation(self) -> float:
        """Get total deformation magnitude."""
        if self.rest_vertices is None or len(self.vertices) == 0:
            return 0.0

        displacement = self.vertices - self.rest_vertices
        return float(np.mean(np.linalg.norm(displacement, axis=1)))


@dataclass
class CuttableTissue(DeformableTissue):
    """
    Tissue that supports cutting operations.

    Extends DeformableTissue with:
    - Cut plane tracking
    - Mesh topology modification
    - Bleeding simulation
    """
    # Cut state
    cuts: List[Dict[str, Any]] = field(default_factory=list)
    is_cut: bool = False

    # Bleeding
    active_bleeding_sites: List[Dict[str, Any]] = field(default_factory=list)
    total_blood_loss: float = 0.0

    def apply_cut(
        self,
        cut_start: NDArray[np.float64],
        cut_end: NDArray[np.float64],
        cut_force: float
    ) -> bool:
        """
        Apply cutting operation.

        Returns True if cut was made.
        """
        if cut_force < self.properties.cutting_force_threshold:
            return False

        # Record cut
        cut_info = {
            'start': cut_start.copy(),
            'end': cut_end.copy(),
            'force': cut_force,
            'time': 0.0
        }
        self.cuts.append(cut_info)
        self.is_cut = True

        # Add bleeding site if tissue bleeds
        if self.properties.bleeds:
            center = (cut_start + cut_end) / 2
            self.active_bleeding_sites.append({
                'position': center,
                'rate': self.properties.bleeding_rate,
                'duration': 0.0,
                'coagulated': False
            })

        # Would modify mesh topology here
        # For now, just mark as cut

        return True

    def update_bleeding(self, dt: float) -> float:
        """
        Update bleeding simulation.

        Returns blood loss this step.
        """
        blood_loss = 0.0

        for site in self.active_bleeding_sites:
            if site['coagulated']:
                continue

            site['duration'] += dt

            # Natural coagulation over time
            if site['duration'] > 60.0:  # 1 minute
                site['coagulated'] = True
                continue

            # Bleeding rate decreases over time
            rate_factor = np.exp(-site['duration'] / 30.0)
            blood_loss += site['rate'] * rate_factor * dt

        self.total_blood_loss += blood_loss
        return blood_loss

    def apply_cautery(
        self,
        position: NDArray[np.float64],
        temperature: float,
        radius: float = 0.005
    ) -> bool:
        """
        Apply cautery/coagulation.

        Returns True if bleeding was stopped.
        """
        stopped_bleeding = False

        for site in self.active_bleeding_sites:
            if site['coagulated']:
                continue

            distance = np.linalg.norm(site['position'] - position)
            if distance < radius:
                if temperature >= self.properties.coagulation_temperature:
                    site['coagulated'] = True
                    stopped_bleeding = True

        return stopped_bleeding


class TissueSimulator:
    """
    High-level tissue simulation manager.

    Coordinates multiple tissue bodies with different properties.
    """

    def __init__(self) -> None:
        self.tissues: Dict[str, DeformableTissue] = {}
        self.cuttable_tissues: Dict[str, CuttableTissue] = {}

        # Global simulation parameters
        self.gravity = np.array([0, 0, -9.81])
        self.time_step = 0.001

        # Statistics
        self._total_blood_loss = 0.0

    def add_tissue(
        self,
        name: str,
        tissue: DeformableTissue
    ) -> None:
        """Add tissue to simulation."""
        tissue.initialize()
        self.tissues[name] = tissue

        if isinstance(tissue, CuttableTissue):
            self.cuttable_tissues[name] = tissue

    def create_box_tissue(
        self,
        name: str,
        center: NDArray[np.float64],
        size: NDArray[np.float64],
        properties: TissueProperties,
        resolution: int = 5
    ) -> DeformableTissue:
        """Create box-shaped tissue with tetrahedral mesh."""
        # Create grid of vertices
        nx, ny, nz = resolution, resolution, resolution
        vertices = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = center[0] + (i / (nx - 1) - 0.5) * size[0]
                    y = center[1] + (j / (ny - 1) - 0.5) * size[1]
                    z = center[2] + (k / (nz - 1) - 0.5) * size[2]
                    vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Create surface triangles (simplified - just outer faces)
        triangles = []
        # Would generate proper surface mesh here

        tissue = DeformableTissue(
            name=name,
            properties=properties,
            position=center,
            vertices=vertices,
            triangles=np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)
        )

        # Fix bottom vertices
        min_z = np.min(vertices[:, 2])
        tissue.fixed_vertices = [
            i for i, v in enumerate(vertices)
            if v[2] < min_z + 0.001
        ]

        self.add_tissue(name, tissue)
        return tissue

    def create_organ(
        self,
        name: str,
        organ_type: str,
        center: NDArray[np.float64],
        scale: float = 1.0
    ) -> CuttableTissue:
        """Create organ model with appropriate properties."""
        # Select properties based on organ type
        if organ_type == "liver":
            properties = TissueProperties.liver()
            size = np.array([0.15, 0.1, 0.08]) * scale
        elif organ_type == "kidney":
            properties = TissueProperties.soft_tissue()
            properties.youngs_modulus = 4000
            properties.color = np.array([0.7, 0.3, 0.3, 1.0])
            size = np.array([0.06, 0.03, 0.12]) * scale
        elif organ_type == "intestine":
            properties = TissueProperties.soft_tissue()
            properties.youngs_modulus = 2000
            properties.color = np.array([0.9, 0.7, 0.7, 1.0])
            size = np.array([0.03, 0.03, 0.1]) * scale
        else:
            properties = TissueProperties.soft_tissue()
            size = np.array([0.1, 0.1, 0.1]) * scale

        # Create vertices (simplified ellipsoid)
        n_points = 100
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)

        vertices = np.zeros((n_points, 3))
        vertices[:, 0] = center[0] + size[0] * np.sin(theta) * np.cos(phi)
        vertices[:, 1] = center[1] + size[1] * np.sin(theta) * np.sin(phi)
        vertices[:, 2] = center[2] + size[2] * np.cos(theta)

        tissue = CuttableTissue(
            name=name,
            properties=properties,
            position=center,
            vertices=vertices,
            triangles=np.zeros((0, 3), dtype=np.int32)
        )

        self.add_tissue(name, tissue)
        return tissue

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Step all tissues forward.

        Returns simulation statistics.
        """
        dt = dt or self.time_step

        # Apply gravity
        for tissue in self.tissues.values():
            if tissue.forces is not None:
                for i in range(len(tissue.vertices)):
                    if i not in tissue.fixed_vertices:
                        tissue.forces[i] += self.gravity * 0.01  # Mass ~10g

        # Step each tissue
        for tissue in self.tissues.values():
            tissue.step(dt)

        # Update bleeding
        blood_loss = 0.0
        for tissue in self.cuttable_tissues.values():
            blood_loss += tissue.update_bleeding(dt)
        self._total_blood_loss += blood_loss

        return {
            'blood_loss': blood_loss,
            'total_blood_loss': self._total_blood_loss,
            'active_bleeding_sites': sum(
                len([s for s in t.active_bleeding_sites if not s['coagulated']])
                for t in self.cuttable_tissues.values()
            )
        }

    def apply_tool_to_tissue(
        self,
        tissue_name: str,
        tool_position: NDArray[np.float64],
        tool_force: NDArray[np.float64],
        tool_type: str = "grasper"
    ) -> Tuple[bool, NDArray[np.float64]]:
        """
        Apply tool interaction to tissue.

        Returns (success, reaction_force).
        """
        if tissue_name not in self.tissues:
            return False, np.zeros(3)

        tissue = self.tissues[tissue_name]

        # Tool-specific behavior
        if tool_type == "grasper":
            reaction = tissue.apply_tool_interaction(
                tool_position, tool_force, tool_radius=0.015
            )
        elif tool_type == "scissors":
            if isinstance(tissue, CuttableTissue):
                # Apply cutting
                cut_force = float(np.linalg.norm(tool_force))
                tissue.apply_cut(
                    tool_position - np.array([0, 0, 0.005]),
                    tool_position + np.array([0, 0, 0.005]),
                    cut_force
                )
            reaction = np.zeros(3)
        elif tool_type == "cautery":
            if isinstance(tissue, CuttableTissue):
                tissue.apply_cautery(tool_position, temperature=80.0)
            reaction = np.zeros(3)
        else:
            reaction = tissue.apply_tool_interaction(tool_position, tool_force)

        return True, reaction

    def get_tissue_at_point(
        self,
        point: NDArray[np.float64],
        radius: float = 0.01
    ) -> Optional[str]:
        """Find tissue at given point."""
        for name, tissue in self.tissues.items():
            if len(tissue.vertices) == 0:
                continue

            distances = np.linalg.norm(tissue.vertices - point, axis=1)
            if np.any(distances < radius):
                return name

        return None

    def reset(self) -> None:
        """Reset all tissues to initial state."""
        for tissue in self.tissues.values():
            if tissue.rest_vertices is not None:
                tissue.vertices = tissue.rest_vertices.copy()
            if tissue.velocities is not None:
                tissue.velocities = np.zeros_like(tissue.velocities)

        for tissue in self.cuttable_tissues.values():
            tissue.cuts.clear()
            tissue.active_bleeding_sites.clear()
            tissue.total_blood_loss = 0.0
            tissue.is_cut = False

        self._total_blood_loss = 0.0
