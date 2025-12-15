"""FDA 21 CFR Part 820 Quality System Regulation compliance helpers."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import hashlib
import uuid


class DeviceClass(Enum):
    """FDA Device Classifications."""
    CLASS_I = "I"  # General controls
    CLASS_II = "II"  # Special controls (510(k))
    CLASS_III = "III"  # Premarket approval (PMA)


class SubmissionType(Enum):
    """FDA Submission Types."""
    PREMARKET_510K = "510(k)"
    PMA = "PMA"
    DE_NOVO = "De Novo"
    HDE = "HDE"  # Humanitarian Device Exemption


@dataclass
class DeviceIdentifier:
    """Unique Device Identifier (UDI) components."""
    # Device Identifier (DI)
    labeler_code: str = ""
    product_code: str = ""
    version: str = ""

    # Production Identifier (PI)
    lot_number: str = ""
    serial_number: str = ""
    manufacturing_date: str = ""
    expiration_date: str = ""

    # GTIN (if applicable)
    gtin: str = ""

    def generate_udi(self) -> str:
        """Generate UDI string."""
        di = f"({self.labeler_code}){self.product_code}{self.version}"
        pi_parts = []
        if self.lot_number:
            pi_parts.append(f"(10){self.lot_number}")
        if self.serial_number:
            pi_parts.append(f"(21){self.serial_number}")
        if self.manufacturing_date:
            pi_parts.append(f"(11){self.manufacturing_date}")
        if self.expiration_date:
            pi_parts.append(f"(17){self.expiration_date}")

        pi = "".join(pi_parts)
        return f"{di}{pi}"


class UDIGenerator:
    """Generates Unique Device Identifiers."""

    def __init__(
        self,
        labeler_code: str,
        issuing_agency: str = "GS1"
    ) -> None:
        self.labeler_code = labeler_code
        self.issuing_agency = issuing_agency
        self._serial_counter = 0

    def generate(
        self,
        product_code: str,
        version: str,
        lot_number: Optional[str] = None
    ) -> DeviceIdentifier:
        """Generate new device identifier."""
        self._serial_counter += 1

        return DeviceIdentifier(
            labeler_code=self.labeler_code,
            product_code=product_code,
            version=version,
            lot_number=lot_number or "",
            serial_number=f"{self._serial_counter:08d}",
            manufacturing_date=datetime.now().strftime("%y%m%d")
        )


@dataclass
class DesignInput:
    """Design input record."""
    id: str
    title: str
    description: str
    source: str  # User needs, regulatory, risk, etc.

    # Classification
    input_type: str = "functional"  # functional, performance, safety, regulatory
    priority: str = "required"

    # Traceability
    derived_from: List[str] = field(default_factory=list)

    # Status
    approved: bool = False
    approved_by: str = ""
    approved_date: str = ""


@dataclass
class DesignOutput:
    """Design output record."""
    id: str
    title: str
    description: str

    # Links
    input_ids: List[str] = field(default_factory=list)  # Design inputs addressed

    # Documentation
    document_refs: List[str] = field(default_factory=list)
    drawing_refs: List[str] = field(default_factory=list)

    # Verification
    verified: bool = False
    verification_method: str = ""
    verification_record: str = ""


@dataclass
class DesignVerification:
    """Design verification record."""
    id: str
    output_id: str
    method: str  # test, inspection, analysis, demonstration

    # Test details
    procedure_ref: str = ""
    acceptance_criteria: str = ""

    # Results
    executed: bool = False
    executed_date: str = ""
    executed_by: str = ""
    result: str = ""  # pass, fail, incomplete
    deviations: List[str] = field(default_factory=list)


@dataclass
class DesignValidation:
    """Design validation record."""
    id: str
    description: str

    # User needs addressed
    user_need_refs: List[str] = field(default_factory=list)
    intended_use: str = ""

    # Validation approach
    protocol_ref: str = ""
    acceptance_criteria: str = ""

    # Results
    executed: bool = False
    executed_date: str = ""
    site: str = ""
    result: str = ""
    observations: List[str] = field(default_factory=list)


class DesignControl:
    """
    21 CFR 820.30 Design Controls management.

    Manages:
    - Design inputs and outputs
    - Design review
    - Verification and validation
    - Design transfer
    - Design history file
    """

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name

        self.inputs: Dict[str, DesignInput] = {}
        self.outputs: Dict[str, DesignOutput] = {}
        self.verifications: Dict[str, DesignVerification] = {}
        self.validations: Dict[str, DesignValidation] = {}

        # Design reviews
        self.reviews: List[Dict[str, Any]] = []

        # Design changes
        self.changes: List[Dict[str, Any]] = []

        # Design transfer records
        self.transfer_records: List[Dict[str, Any]] = []

    def add_design_input(
        self,
        title: str,
        description: str,
        source: str,
        input_type: str = "functional"
    ) -> str:
        """Add design input."""
        input_id = f"DI-{len(self.inputs) + 1:04d}"

        design_input = DesignInput(
            id=input_id,
            title=title,
            description=description,
            source=source,
            input_type=input_type
        )

        self.inputs[input_id] = design_input
        return input_id

    def add_design_output(
        self,
        title: str,
        description: str,
        input_ids: List[str]
    ) -> str:
        """Add design output."""
        output_id = f"DO-{len(self.outputs) + 1:04d}"

        design_output = DesignOutput(
            id=output_id,
            title=title,
            description=description,
            input_ids=input_ids
        )

        self.outputs[output_id] = design_output
        return output_id

    def add_verification(
        self,
        output_id: str,
        method: str,
        acceptance_criteria: str
    ) -> str:
        """Add verification record."""
        ver_id = f"DV-{len(self.verifications) + 1:04d}"

        verification = DesignVerification(
            id=ver_id,
            output_id=output_id,
            method=method,
            acceptance_criteria=acceptance_criteria
        )

        self.verifications[ver_id] = verification
        return ver_id

    def record_verification_result(
        self,
        ver_id: str,
        result: str,
        executed_by: str,
        deviations: Optional[List[str]] = None
    ) -> None:
        """Record verification execution."""
        if ver_id in self.verifications:
            self.verifications[ver_id].executed = True
            self.verifications[ver_id].executed_date = datetime.utcnow().isoformat()
            self.verifications[ver_id].executed_by = executed_by
            self.verifications[ver_id].result = result
            self.verifications[ver_id].deviations = deviations or []

    def add_validation(
        self,
        description: str,
        intended_use: str,
        user_need_refs: List[str]
    ) -> str:
        """Add validation record."""
        val_id = f"VAL-{len(self.validations) + 1:04d}"

        validation = DesignValidation(
            id=val_id,
            description=description,
            intended_use=intended_use,
            user_need_refs=user_need_refs
        )

        self.validations[val_id] = validation
        return val_id

    def conduct_design_review(
        self,
        phase: str,
        attendees: List[str],
        findings: List[str],
        actions: List[str]
    ) -> str:
        """Conduct and record design review."""
        review_id = f"DR-{len(self.reviews) + 1:04d}"

        review = {
            "id": review_id,
            "phase": phase,
            "date": datetime.utcnow().isoformat(),
            "attendees": attendees,
            "findings": findings,
            "actions": actions,
            "closed": False
        }

        self.reviews.append(review)
        return review_id

    def record_design_change(
        self,
        description: str,
        reason: str,
        affected_items: List[str],
        impact_assessment: str
    ) -> str:
        """Record design change."""
        change_id = f"DC-{len(self.changes) + 1:04d}"

        change = {
            "id": change_id,
            "description": description,
            "reason": reason,
            "affected_items": affected_items,
            "impact_assessment": impact_assessment,
            "date": datetime.utcnow().isoformat(),
            "approved": False,
            "approved_by": "",
            "implemented": False
        }

        self.changes.append(change)
        return change_id

    def get_traceability_matrix(self) -> List[Dict[str, Any]]:
        """Generate design traceability matrix."""
        matrix = []

        for input_id, di in self.inputs.items():
            # Find outputs linked to this input
            linked_outputs = [
                do for do in self.outputs.values()
                if input_id in do.input_ids
            ]

            # Find verifications for those outputs
            linked_verifications = []
            for output in linked_outputs:
                linked_verifications.extend([
                    v for v in self.verifications.values()
                    if v.output_id == output.id
                ])

            matrix.append({
                "input_id": input_id,
                "input_title": di.title,
                "input_type": di.input_type,
                "outputs": [o.id for o in linked_outputs],
                "verifications": [v.id for v in linked_verifications],
                "verification_status": all(v.result == "pass" for v in linked_verifications) if linked_verifications else "not_verified"
            })

        return matrix

    def generate_dhf_summary(self) -> Dict[str, Any]:
        """Generate Design History File summary."""
        return {
            "device_name": self.device_name,
            "date_generated": datetime.utcnow().isoformat(),
            "design_inputs": len(self.inputs),
            "design_outputs": len(self.outputs),
            "verifications": {
                "total": len(self.verifications),
                "passed": sum(1 for v in self.verifications.values() if v.result == "pass"),
                "failed": sum(1 for v in self.verifications.values() if v.result == "fail"),
                "pending": sum(1 for v in self.verifications.values() if not v.executed)
            },
            "validations": {
                "total": len(self.validations),
                "completed": sum(1 for v in self.validations.values() if v.executed)
            },
            "design_reviews": len(self.reviews),
            "design_changes": len(self.changes),
            "traceability_complete": len(self.get_traceability_matrix()) == len(self.inputs)
        }


@dataclass
class QualityRecord:
    """Generic quality system record."""
    id: str
    record_type: str
    title: str
    content: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = ""
    approved_by: str = ""
    approved_date: str = ""

    # Retention
    retention_years: int = 7  # Default per FDA requirements


class QualitySystemRecord:
    """
    21 CFR Part 820 Quality System Records management.

    Manages:
    - Device Master Record (DMR)
    - Device History Record (DHR)
    - Quality System Record (QSR)
    - Complaint files
    - CAPA records
    """

    def __init__(self, company_name: str) -> None:
        self.company_name = company_name

        # Records storage
        self.records: Dict[str, QualityRecord] = {}
        self.dmr: Dict[str, Any] = {}  # Device Master Record
        self.dhr: Dict[str, List[Dict]] = {}  # Device History Records

        # Complaints
        self.complaints: List[Dict[str, Any]] = []

        # CAPA (Corrective and Preventive Action)
        self.capas: Dict[str, Dict[str, Any]] = {}

    def create_dmr_entry(
        self,
        device_name: str,
        specifications: Dict[str, Any],
        procedures: List[str],
        drawings: List[str]
    ) -> None:
        """Create Device Master Record entry."""
        self.dmr[device_name] = {
            "device_name": device_name,
            "specifications": specifications,
            "production_procedures": procedures,
            "quality_procedures": [],
            "drawings": drawings,
            "labeling": [],
            "packaging": [],
            "created_date": datetime.utcnow().isoformat()
        }

    def create_dhr_entry(
        self,
        device_name: str,
        lot_number: str,
        serial_numbers: List[str],
        manufacturing_date: str,
        inspection_results: Dict[str, Any]
    ) -> str:
        """Create Device History Record entry."""
        dhr_id = f"DHR-{uuid.uuid4().hex[:8].upper()}"

        entry = {
            "id": dhr_id,
            "lot_number": lot_number,
            "serial_numbers": serial_numbers,
            "manufacturing_date": manufacturing_date,
            "inspection_results": inspection_results,
            "acceptance_status": "pending",
            "released_by": "",
            "release_date": ""
        }

        if device_name not in self.dhr:
            self.dhr[device_name] = []
        self.dhr[device_name].append(entry)

        return dhr_id

    def record_complaint(
        self,
        device_name: str,
        complaint_date: str,
        reporter: str,
        description: str,
        patient_involvement: bool = False,
        injury: bool = False
    ) -> str:
        """Record customer complaint."""
        complaint_id = f"CMP-{len(self.complaints) + 1:05d}"

        complaint = {
            "id": complaint_id,
            "device_name": device_name,
            "complaint_date": complaint_date,
            "received_date": datetime.utcnow().isoformat(),
            "reporter": reporter,
            "description": description,
            "patient_involvement": patient_involvement,
            "injury": injury,
            "mdr_reportable": injury,  # Simplified - real logic more complex
            "investigation_status": "open",
            "root_cause": "",
            "capa_id": ""
        }

        self.complaints.append(complaint)
        return complaint_id

    def create_capa(
        self,
        title: str,
        description: str,
        capa_type: str,  # corrective, preventive
        source: str,  # complaint, audit, ncr, etc.
        source_id: str = ""
    ) -> str:
        """Create CAPA record."""
        capa_id = f"CAPA-{len(self.capas) + 1:04d}"

        self.capas[capa_id] = {
            "id": capa_id,
            "title": title,
            "description": description,
            "type": capa_type,
            "source": source,
            "source_id": source_id,
            "created_date": datetime.utcnow().isoformat(),
            "root_cause_analysis": "",
            "corrective_actions": [],
            "preventive_actions": [],
            "verification": "",
            "status": "open",
            "effectiveness_check_date": "",
            "closed_date": ""
        }

        return capa_id

    def add_capa_action(
        self,
        capa_id: str,
        action_type: str,  # corrective, preventive
        description: str,
        responsible: str,
        due_date: str
    ) -> None:
        """Add action to CAPA."""
        if capa_id not in self.capas:
            return

        action = {
            "description": description,
            "responsible": responsible,
            "due_date": due_date,
            "status": "open",
            "completed_date": ""
        }

        if action_type == "corrective":
            self.capas[capa_id]["corrective_actions"].append(action)
        else:
            self.capas[capa_id]["preventive_actions"].append(action)

    def generate_quality_metrics(self) -> Dict[str, Any]:
        """Generate quality system metrics."""
        open_complaints = sum(1 for c in self.complaints if c["investigation_status"] == "open")
        open_capas = sum(1 for c in self.capas.values() if c["status"] == "open")

        return {
            "total_complaints": len(self.complaints),
            "open_complaints": open_complaints,
            "mdr_reportable": sum(1 for c in self.complaints if c["mdr_reportable"]),
            "total_capas": len(self.capas),
            "open_capas": open_capas,
            "devices_in_dmr": len(self.dmr),
            "total_dhr_records": sum(len(records) for records in self.dhr.values())
        }
