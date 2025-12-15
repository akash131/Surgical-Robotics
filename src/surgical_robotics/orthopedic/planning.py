"""Patient-specific surgical planning for orthopedic robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class BoneType(Enum):
    """Types of bones in orthopedic surgery."""
    FEMUR = auto()
    TIBIA = auto()
    PATELLA = auto()
    PELVIS = auto()
    VERTEBRA = auto()
    HUMERUS = auto()


class ImplantType(Enum):
    """Types of orthopedic implants."""
    FEMORAL_COMPONENT_TKA = auto()  # Total knee - femoral
    TIBIAL_COMPONENT_TKA = auto()  # Total knee - tibial
    PATELLAR_COMPONENT = auto()
    ACETABULAR_CUP = auto()  # Hip - acetabulum
    FEMORAL_STEM = auto()  # Hip - femur
    PEDICLE_SCREW = auto()
    INTERBODY_CAGE = auto()


@dataclass
class PatientModel:
    """
    Patient-specific anatomical model.

    Derived from preoperative imaging (CT/MRI).
    """
    patient_id: str
    bone_type: BoneType

    # Surface mesh
    vertices: list[NDArray[np.float64]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)

    # Anatomical landmarks
    landmarks: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    # Anatomical axes
    mechanical_axis: Optional[NDArray[np.float64]] = None
    anatomical_axis: Optional[NDArray[np.float64]] = None
    transverse_axis: Optional[NDArray[np.float64]] = None

    # Measurements
    measurements: dict[str, float] = field(default_factory=dict)

    def add_landmark(self, name: str, position: NDArray[np.float64]) -> None:
        """Add anatomical landmark."""
        self.landmarks[name] = np.asarray(position)

    def compute_axes(self) -> bool:
        """Compute anatomical coordinate system."""
        if self.bone_type == BoneType.FEMUR:
            return self._compute_femur_axes()
        elif self.bone_type == BoneType.TIBIA:
            return self._compute_tibia_axes()
        return False

    def _compute_femur_axes(self) -> bool:
        """Compute femoral anatomical axes."""
        required = ["femoral_head_center", "medial_epicondyle", "lateral_epicondyle"]
        if not all(lm in self.landmarks for lm in required):
            return False

        # Mechanical axis: head center to knee center
        head_center = self.landmarks["femoral_head_center"]
        medial_epi = self.landmarks["medial_epicondyle"]
        lateral_epi = self.landmarks["lateral_epicondyle"]

        knee_center = (medial_epi + lateral_epi) / 2
        mechanical = head_center - knee_center
        self.mechanical_axis = mechanical / np.linalg.norm(mechanical)

        # Transverse axis: epicondylar axis
        trans = lateral_epi - medial_epi
        self.transverse_axis = trans / np.linalg.norm(trans)

        # AP axis (perpendicular)
        ap = np.cross(self.mechanical_axis, self.transverse_axis)
        self.anatomical_axis = ap / np.linalg.norm(ap)

        return True

    def _compute_tibia_axes(self) -> bool:
        """Compute tibial anatomical axes."""
        required = ["tibial_plateau_center", "ankle_center"]
        if not all(lm in self.landmarks for lm in required):
            return False

        plateau = self.landmarks["tibial_plateau_center"]
        ankle = self.landmarks["ankle_center"]

        # Mechanical axis
        mechanical = plateau - ankle
        self.mechanical_axis = mechanical / np.linalg.norm(mechanical)

        return True

    def get_surface_point(
        self,
        ray_origin: NDArray[np.float64],
        ray_direction: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Find intersection of ray with bone surface."""
        if not self.vertices or not self.triangles:
            return None

        closest_intersection: Optional[NDArray[np.float64]] = None
        min_distance = float('inf')

        ray_dir = ray_direction / np.linalg.norm(ray_direction)

        for tri in self.triangles:
            v0 = np.array(self.vertices[tri[0]])
            v1 = np.array(self.vertices[tri[1]])
            v2 = np.array(self.vertices[tri[2]])

            intersection = self._ray_triangle_intersection(
                ray_origin, ray_dir, v0, v1, v2
            )

            if intersection is not None:
                dist = float(np.linalg.norm(intersection - ray_origin))
                if dist < min_distance:
                    min_distance = dist
                    closest_intersection = intersection

        return closest_intersection

    @staticmethod
    def _ray_triangle_intersection(
        ray_origin: NDArray[np.float64],
        ray_direction: NDArray[np.float64],
        v0: NDArray[np.float64],
        v1: NDArray[np.float64],
        v2: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Möller–Trumbore ray-triangle intersection."""
        epsilon = 1e-8

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if abs(a) < epsilon:
            return None

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > epsilon:
            return ray_origin + ray_direction * t

        return None


@dataclass
class ImplantModel:
    """
    Implant model with geometry and placement parameters.
    """
    implant_type: ImplantType
    name: str
    size: str  # e.g., "Size 5", "Medium"

    # Implant geometry
    vertices: list[NDArray[np.float64]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)

    # Key points
    reference_points: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    # Planned pose (in patient coordinates)
    planned_position: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    planned_rotation: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3)
    )

    def get_transformation_matrix(self) -> NDArray[np.float64]:
        """Get 4x4 transformation matrix for implant placement."""
        transform = np.eye(4)
        transform[:3, :3] = self.planned_rotation
        transform[:3, 3] = self.planned_position
        return transform


@dataclass
class BoneCut:
    """Definition of a bone cut."""
    name: str
    plane_point: NDArray[np.float64]  # Point on cutting plane
    plane_normal: NDArray[np.float64]  # Normal to cutting plane
    depth: float  # Cut depth in meters
    width: float  # Cut width in meters
    height: float  # Cut height in meters

    # Tolerance
    position_tolerance: float = 0.001  # 1mm
    angle_tolerance: float = np.radians(1)  # 1 degree

    def get_corners(self) -> list[NDArray[np.float64]]:
        """Get corner points of rectangular cut."""
        normal = self.plane_normal / np.linalg.norm(self.plane_normal)

        # Create orthogonal vectors
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])

        x_axis = np.cross(up, normal)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)

        half_w = self.width / 2
        half_h = self.height / 2

        corners = [
            self.plane_point + x_axis * half_w + y_axis * half_h,
            self.plane_point - x_axis * half_w + y_axis * half_h,
            self.plane_point - x_axis * half_w - y_axis * half_h,
            self.plane_point + x_axis * half_w - y_axis * half_h,
        ]

        return corners


@dataclass
class SurgicalPlan:
    """
    Complete surgical plan for an orthopedic procedure.
    """
    plan_id: str
    patient_model: PatientModel
    implants: list[ImplantModel] = field(default_factory=list)
    bone_cuts: list[BoneCut] = field(default_factory=list)

    # Plan quality metrics
    leg_length_change: float = 0.0  # mm
    offset_change: float = 0.0  # mm
    alignment_angle: float = 0.0  # degrees

    # Approval status
    is_approved: bool = False
    surgeon_notes: str = ""

    def add_implant(self, implant: ImplantModel) -> None:
        """Add implant to plan."""
        self.implants.append(implant)

    def add_bone_cut(self, cut: BoneCut) -> None:
        """Add bone cut to plan."""
        self.bone_cuts.append(cut)

    def validate(self) -> Tuple[bool, list[str]]:
        """Validate surgical plan."""
        issues: list[str] = []

        if not self.patient_model.landmarks:
            issues.append("Patient model has no landmarks")

        if not self.bone_cuts:
            issues.append("No bone cuts defined")

        if not self.implants:
            issues.append("No implants selected")

        # Check cut-implant compatibility
        for implant in self.implants:
            if not self._check_implant_fit(implant):
                issues.append(f"Implant {implant.name} may not fit properly")

        return len(issues) == 0, issues

    def _check_implant_fit(self, implant: ImplantModel) -> bool:
        """Check if implant fits the prepared bone."""
        # Simplified check - would be more thorough in practice
        return True


class BoneCutPlanner:
    """
    Plan bone cuts for implant placement.
    """

    def __init__(self) -> None:
        self.patient_model: Optional[PatientModel] = None
        self.target_implant: Optional[ImplantModel] = None

    def set_patient_model(self, model: PatientModel) -> None:
        """Set patient-specific bone model."""
        self.patient_model = model

    def set_target_implant(self, implant: ImplantModel) -> None:
        """Set target implant for cut planning."""
        self.target_implant = implant

    def plan_tka_femoral_cuts(self) -> list[BoneCut]:
        """
        Plan femoral bone cuts for total knee arthroplasty.

        Returns 5 standard cuts: distal, anterior, posterior,
        anterior chamfer, posterior chamfer.
        """
        if self.patient_model is None:
            return []

        cuts: list[BoneCut] = []
        landmarks = self.patient_model.landmarks

        if "knee_center" not in landmarks:
            return []

        knee_center = landmarks["knee_center"]
        mechanical_axis = self.patient_model.mechanical_axis
        if mechanical_axis is None:
            mechanical_axis = np.array([0, 0, 1])

        # 1. Distal femoral cut (perpendicular to mechanical axis)
        distal_cut = BoneCut(
            name="distal_femoral",
            plane_point=knee_center,
            plane_normal=mechanical_axis,
            depth=0.009,  # 9mm resection
            width=0.070,
            height=0.070
        )
        cuts.append(distal_cut)

        # 2. Anterior cut
        anterior_normal = np.array([0, 1, 0])  # Simplified
        anterior_cut = BoneCut(
            name="anterior_femoral",
            plane_point=knee_center + np.array([0, 0.02, 0.02]),
            plane_normal=anterior_normal,
            depth=0.005,
            width=0.065,
            height=0.030
        )
        cuts.append(anterior_cut)

        # 3. Posterior cut
        posterior_cut = BoneCut(
            name="posterior_femoral",
            plane_point=knee_center + np.array([0, -0.02, 0]),
            plane_normal=-anterior_normal,
            depth=0.010,
            width=0.065,
            height=0.020
        )
        cuts.append(posterior_cut)

        # 4. Anterior chamfer (45 degrees)
        ant_chamfer_normal = (mechanical_axis + anterior_normal) / np.sqrt(2)
        ant_chamfer_cut = BoneCut(
            name="anterior_chamfer",
            plane_point=knee_center + np.array([0, 0.015, 0.015]),
            plane_normal=ant_chamfer_normal,
            depth=0.004,
            width=0.065,
            height=0.015
        )
        cuts.append(ant_chamfer_cut)

        # 5. Posterior chamfer (45 degrees)
        post_chamfer_normal = (mechanical_axis - anterior_normal) / np.sqrt(2)
        post_chamfer_cut = BoneCut(
            name="posterior_chamfer",
            plane_point=knee_center + np.array([0, -0.015, 0.005]),
            plane_normal=post_chamfer_normal,
            depth=0.004,
            width=0.065,
            height=0.015
        )
        cuts.append(post_chamfer_cut)

        return cuts

    def plan_tka_tibial_cuts(self) -> list[BoneCut]:
        """Plan tibial bone cuts for total knee arthroplasty."""
        if self.patient_model is None:
            return []

        cuts: list[BoneCut] = []
        landmarks = self.patient_model.landmarks

        if "tibial_plateau_center" not in landmarks:
            return []

        plateau_center = landmarks["tibial_plateau_center"]
        mechanical_axis = self.patient_model.mechanical_axis
        if mechanical_axis is None:
            mechanical_axis = np.array([0, 0, 1])

        # Tibial plateau cut (perpendicular to mechanical axis)
        # Typically 3-degree posterior slope
        slope_angle = np.radians(3)
        cut_normal = mechanical_axis.copy()
        # Apply posterior slope
        rotation = np.array([
            [1, 0, 0],
            [0, np.cos(slope_angle), -np.sin(slope_angle)],
            [0, np.sin(slope_angle), np.cos(slope_angle)]
        ])
        cut_normal = rotation @ cut_normal

        tibial_cut = BoneCut(
            name="tibial_plateau",
            plane_point=plateau_center,
            plane_normal=cut_normal,
            depth=0.010,  # 10mm resection
            width=0.075,
            height=0.050
        )
        cuts.append(tibial_cut)

        return cuts

    def plan_tha_acetabular_reaming(
        self,
        target_cup_size: float,
        anteversion: float = np.radians(20),
        inclination: float = np.radians(45)
    ) -> dict:
        """
        Plan acetabular reaming for total hip arthroplasty.

        Args:
            target_cup_size: Target cup diameter in meters
            anteversion: Anteversion angle in radians
            inclination: Inclination angle in radians

        Returns:
            Reaming parameters dictionary
        """
        if self.patient_model is None:
            return {}

        landmarks = self.patient_model.landmarks

        if "acetabulum_center" not in landmarks:
            return {}

        center = landmarks["acetabulum_center"]

        # Compute cup orientation
        # Start with lateral vector
        lateral = np.array([1, 0, 0])

        # Apply inclination (rotation about AP axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])

        # Apply anteversion (rotation about superior axis)
        Rz = np.array([
            [np.cos(anteversion), -np.sin(anteversion), 0],
            [np.sin(anteversion), np.cos(anteversion), 0],
            [0, 0, 1]
        ])

        cup_axis = Rz @ Rx @ lateral

        return {
            "center": center,
            "axis": cup_axis,
            "final_diameter": target_cup_size,
            "reaming_sequence": self._generate_reaming_sequence(target_cup_size),
            "anteversion_deg": np.degrees(anteversion),
            "inclination_deg": np.degrees(inclination),
        }

    def _generate_reaming_sequence(
        self,
        final_size: float,
        start_size: float = 0.044,  # 44mm starting reamer
        increment: float = 0.002  # 2mm increments
    ) -> list[float]:
        """Generate sequence of reamer sizes."""
        sizes: list[float] = []
        current = start_size

        while current <= final_size:
            sizes.append(current)
            current += increment

        return sizes

    def optimize_implant_position(
        self,
        implant: ImplantModel,
        objective: str = "minimize_bone_removal"
    ) -> ImplantModel:
        """
        Optimize implant position based on objective.

        Args:
            implant: Implant to optimize
            objective: Optimization objective

        Returns:
            Optimized implant with updated position
        """
        if self.patient_model is None:
            return implant

        # Simplified optimization - would use numerical optimization in practice
        optimized = ImplantModel(
            implant_type=implant.implant_type,
            name=implant.name,
            size=implant.size,
            vertices=implant.vertices.copy(),
            triangles=implant.triangles.copy(),
            planned_position=implant.planned_position.copy(),
            planned_rotation=implant.planned_rotation.copy()
        )

        return optimized
