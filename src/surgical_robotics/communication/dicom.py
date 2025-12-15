"""DICOM medical imaging protocol support."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray


class DICOMModality(Enum):
    """DICOM imaging modalities."""
    CT = auto()
    MR = auto()
    US = auto()  # Ultrasound
    XA = auto()  # X-ray Angiography
    CR = auto()  # Computed Radiography
    DX = auto()  # Digital Radiography
    PT = auto()  # PET
    NM = auto()  # Nuclear Medicine


@dataclass
class DICOMPatient:
    """DICOM patient information."""
    patient_id: str
    patient_name: str
    birth_date: str
    sex: str
    age: Optional[str] = None
    weight: Optional[float] = None


@dataclass
class DICOMStudy:
    """DICOM study information."""
    study_instance_uid: str
    study_date: str
    study_time: str
    study_description: str
    accession_number: str
    referring_physician: str


@dataclass
class DICOMSeries:
    """DICOM series information."""
    series_instance_uid: str
    series_number: int
    series_description: str
    modality: DICOMModality
    body_part_examined: str

    # Image properties
    rows: int = 512
    columns: int = 512
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    num_slices: int = 1

    # Image data
    pixel_data: Optional[NDArray] = None
    image_positions: List[NDArray[np.float64]] = field(default_factory=list)
    image_orientations: List[NDArray[np.float64]] = field(default_factory=list)

    @property
    def voxel_spacing(self) -> Tuple[float, float, float]:
        """Get 3D voxel spacing."""
        return (self.pixel_spacing[0], self.pixel_spacing[1], self.slice_thickness)


@dataclass
class DICOMImage:
    """Single DICOM image/slice."""
    sop_instance_uid: str
    instance_number: int
    image_position: NDArray[np.float64]
    image_orientation: NDArray[np.float64]
    pixel_data: NDArray
    window_center: float = 0.0
    window_width: float = 1000.0
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0


class DICOMReader:
    """
    DICOM file reader.

    Reads DICOM files and extracts image data and metadata.
    """

    def __init__(self) -> None:
        self.patient: Optional[DICOMPatient] = None
        self.study: Optional[DICOMStudy] = None
        self.series_list: List[DICOMSeries] = []

    def read_directory(self, directory: str) -> bool:
        """
        Read all DICOM files in a directory.

        Args:
            directory: Path to directory containing DICOM files
        """
        path = Path(directory)
        if not path.exists():
            return False

        # Would use pydicom or similar library to read files
        # Here we provide the interface structure

        dicom_files = list(path.glob("*.dcm")) + list(path.glob("*.DCM"))
        if not dicom_files:
            # Try reading files without extension (common in DICOM)
            dicom_files = [f for f in path.iterdir() if f.is_file()]

        return len(dicom_files) > 0

    def read_file(self, filepath: str) -> Optional[DICOMImage]:
        """Read a single DICOM file."""
        path = Path(filepath)
        if not path.exists():
            return None

        # Would parse DICOM file here
        # Return placeholder
        return DICOMImage(
            sop_instance_uid="1.2.3.4",
            instance_number=1,
            image_position=np.zeros(3),
            image_orientation=np.array([1, 0, 0, 0, 1, 0]),
            pixel_data=np.zeros((512, 512), dtype=np.int16)
        )

    def get_volume(self, series_uid: str) -> Optional[NDArray]:
        """
        Get 3D volume from series.

        Returns stacked slices as 3D array.
        """
        for series in self.series_list:
            if series.series_instance_uid == series_uid:
                return series.pixel_data

        return None

    def get_image_geometry(self, series_uid: str) -> Optional[Dict[str, Any]]:
        """Get image geometry information."""
        for series in self.series_list:
            if series.series_instance_uid == series_uid:
                return {
                    "rows": series.rows,
                    "columns": series.columns,
                    "num_slices": series.num_slices,
                    "pixel_spacing": series.pixel_spacing,
                    "slice_thickness": series.slice_thickness,
                    "origin": series.image_positions[0] if series.image_positions else np.zeros(3),
                    "orientation": series.image_orientations[0] if series.image_orientations else np.eye(3).flatten()[:6]
                }

        return None


class DICOMWriter:
    """
    DICOM file writer.

    Creates DICOM files from image data.
    """

    def __init__(self) -> None:
        self.patient: Optional[DICOMPatient] = None
        self.study: Optional[DICOMStudy] = None

    def set_patient(self, patient: DICOMPatient) -> None:
        """Set patient information."""
        self.patient = patient

    def set_study(self, study: DICOMStudy) -> None:
        """Set study information."""
        self.study = study

    def write_series(
        self,
        output_dir: str,
        pixel_data: NDArray,
        series_description: str,
        modality: DICOMModality = DICOMModality.CT,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        slice_thickness: float = 1.0,
        origin: NDArray[np.float64] = None
    ) -> bool:
        """
        Write image series to DICOM files.

        Args:
            output_dir: Output directory
            pixel_data: 3D image array (slices, rows, cols)
            series_description: Series description
            modality: Imaging modality
            pixel_spacing: Pixel spacing in mm
            slice_thickness: Slice thickness in mm
            origin: Origin point in patient coordinates
        """
        if self.patient is None or self.study is None:
            return False

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Would write DICOM files using pydicom
        # This is the interface structure
        return True

    def write_rt_struct(
        self,
        output_path: str,
        contours: Dict[str, List[NDArray[np.float64]]],
        reference_series_uid: str
    ) -> bool:
        """
        Write RT Structure Set (contours).

        Args:
            output_path: Output file path
            contours: Dictionary of structure name to list of contour points
            reference_series_uid: UID of referenced image series
        """
        # Would write DICOM RT Struct
        return True


class DICOMRegistration:
    """
    DICOM-based image registration.

    Handles registration between DICOM image sets and
    robot coordinate systems.
    """

    def __init__(self) -> None:
        self.source_series: Optional[DICOMSeries] = None
        self.registration_matrix: NDArray[np.float64] = np.eye(4)
        self.is_registered: bool = False

    def set_source_series(self, series: DICOMSeries) -> None:
        """Set source DICOM series."""
        self.source_series = series

    def register_with_fiducials(
        self,
        image_fiducials: List[NDArray[np.float64]],
        physical_fiducials: List[NDArray[np.float64]]
    ) -> bool:
        """
        Register using fiducial markers.

        Args:
            image_fiducials: Fiducial positions in image coordinates
            physical_fiducials: Corresponding positions in physical/robot coordinates
        """
        if len(image_fiducials) < 3 or len(physical_fiducials) < 3:
            return False

        if len(image_fiducials) != len(physical_fiducials):
            return False

        # Paired-point registration using SVD
        img_pts = np.array(image_fiducials)
        phys_pts = np.array(physical_fiducials)

        img_centroid = np.mean(img_pts, axis=0)
        phys_centroid = np.mean(phys_pts, axis=0)

        img_centered = img_pts - img_centroid
        phys_centered = phys_pts - phys_centroid

        H = img_centered.T @ phys_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = phys_centroid - R @ img_centroid

        self.registration_matrix[:3, :3] = R
        self.registration_matrix[:3, 3] = t
        self.is_registered = True

        return True

    def transform_point(
        self,
        image_point: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Transform point from image to physical coordinates."""
        if not self.is_registered:
            return None

        point_h = np.append(image_point, 1)
        result_h = self.registration_matrix @ point_h
        return result_h[:3]

    def inverse_transform_point(
        self,
        physical_point: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Transform point from physical to image coordinates."""
        if not self.is_registered:
            return None

        inv_matrix = np.linalg.inv(self.registration_matrix)
        point_h = np.append(physical_point, 1)
        result_h = inv_matrix @ point_h
        return result_h[:3]

    def get_fiducial_registration_error(
        self,
        image_fiducials: List[NDArray[np.float64]],
        physical_fiducials: List[NDArray[np.float64]]
    ) -> float:
        """Calculate fiducial registration error (FRE)."""
        if not self.is_registered:
            return float('inf')

        errors = []
        for img_pt, phys_pt in zip(image_fiducials, physical_fiducials):
            transformed = self.transform_point(img_pt)
            if transformed is not None:
                error = np.linalg.norm(transformed - phys_pt)
                errors.append(error)

        if not errors:
            return float('inf')

        return float(np.sqrt(np.mean(np.array(errors) ** 2)))


class DICOMRTDose:
    """DICOM RT Dose handling for radiation treatment planning."""

    def __init__(self) -> None:
        self.dose_grid: Optional[NDArray] = None
        self.dose_grid_scaling: float = 1.0
        self.grid_frame_offset_vector: List[float] = [0, 0, 0]

    def load(self, filepath: str) -> bool:
        """Load RT Dose file."""
        # Would parse DICOM RT Dose file
        return True

    def get_dose_at_point(
        self,
        point: NDArray[np.float64]
    ) -> Optional[float]:
        """Get dose value at a point."""
        if self.dose_grid is None:
            return None

        # Would interpolate dose at point
        return 0.0


class DICOMServer:
    """
    Simple DICOM server (SCP) implementation.

    Provides Query/Retrieve and Storage SCP services.
    """

    def __init__(self, ae_title: str = "SURGICAL_ROBOT", port: int = 11112) -> None:
        self.ae_title = ae_title
        self.port = port
        self.storage_path = "./dicom_storage"
        self._running = False

    def start(self) -> bool:
        """Start DICOM server."""
        # Would start actual DICOM SCP
        self._running = True
        return True

    def stop(self) -> None:
        """Stop DICOM server."""
        self._running = False

    def query(
        self,
        remote_ae: str,
        remote_host: str,
        remote_port: int,
        query_level: str = "STUDY",
        query_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query remote DICOM server (C-FIND).

        Args:
            remote_ae: Remote AE title
            remote_host: Remote hostname
            remote_port: Remote port
            query_level: Query level (PATIENT, STUDY, SERIES, IMAGE)
            query_params: Query parameters

        Returns:
            List of matching results
        """
        # Would perform C-FIND
        return []

    def retrieve(
        self,
        remote_ae: str,
        remote_host: str,
        remote_port: int,
        study_uid: str
    ) -> bool:
        """
        Retrieve study from remote server (C-MOVE).

        Args:
            remote_ae: Remote AE title
            remote_host: Remote hostname
            remote_port: Remote port
            study_uid: Study Instance UID to retrieve
        """
        # Would perform C-MOVE
        return True
