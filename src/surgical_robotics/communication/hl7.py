"""HL7 and FHIR healthcare interoperability protocols."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any
import json
import uuid


class HL7MessageType(Enum):
    """HL7 message types."""
    ADT = auto()  # Admit/Discharge/Transfer
    ORM = auto()  # Order Message
    ORU = auto()  # Observation Result
    SIU = auto()  # Scheduling Information
    MDM = auto()  # Medical Document Management
    ACK = auto()  # Acknowledgment


class HL7EventType(Enum):
    """HL7 event types."""
    A01 = auto()  # Patient Admit
    A02 = auto()  # Patient Transfer
    A03 = auto()  # Patient Discharge
    A08 = auto()  # Patient Update
    O01 = auto()  # Order
    R01 = auto()  # Observation Result
    S12 = auto()  # Schedule Notification


@dataclass
class HL7Segment:
    """HL7 message segment."""
    segment_id: str
    fields: List[str]

    def to_string(self) -> str:
        """Convert to HL7 string format."""
        return f"{self.segment_id}|{'|'.join(self.fields)}"

    @classmethod
    def from_string(cls, segment_str: str) -> "HL7Segment":
        """Parse from HL7 string."""
        parts = segment_str.split("|")
        return cls(segment_id=parts[0], fields=parts[1:] if len(parts) > 1 else [])


@dataclass
class HL7Message:
    """
    HL7 v2.x message.

    Provides creation and parsing of HL7 messages for
    healthcare system integration.
    """
    message_type: HL7MessageType
    event_type: HL7EventType
    segments: List[HL7Segment] = field(default_factory=list)
    message_control_id: str = ""
    sending_application: str = "SURGICAL_ROBOT"
    sending_facility: str = "SURGERY"
    receiving_application: str = ""
    receiving_facility: str = ""

    def __post_init__(self) -> None:
        if not self.message_control_id:
            self.message_control_id = str(uuid.uuid4())[:20]

    def to_string(self) -> str:
        """Convert to HL7 wire format."""
        # Build MSH segment
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        msh = HL7Segment(
            segment_id="MSH",
            fields=[
                "^~\\&",  # Encoding characters
                self.sending_application,
                self.sending_facility,
                self.receiving_application,
                self.receiving_facility,
                timestamp,
                "",
                f"{self.message_type.name}^{self.event_type.name}",
                self.message_control_id,
                "P",  # Processing ID
                "2.5"  # Version
            ]
        )

        lines = [msh.to_string()]
        lines.extend(seg.to_string() for seg in self.segments)

        return "\r".join(lines) + "\r"

    @classmethod
    def from_string(cls, message_str: str) -> "HL7Message":
        """Parse HL7 message from string."""
        lines = message_str.strip().split("\r")
        segments = [HL7Segment.from_string(line) for line in lines if line]

        # Parse MSH
        msh = segments[0] if segments else None
        if msh and msh.segment_id == "MSH":
            msg_type_field = msh.fields[7] if len(msh.fields) > 7 else "ACK^A01"
            parts = msg_type_field.split("^")
            msg_type = HL7MessageType[parts[0]] if parts[0] in HL7MessageType.__members__ else HL7MessageType.ACK
            event_type = HL7EventType[parts[1]] if len(parts) > 1 and parts[1] in HL7EventType.__members__ else HL7EventType.A01

            return cls(
                message_type=msg_type,
                event_type=event_type,
                segments=segments[1:],  # Exclude MSH
                sending_application=msh.fields[1] if len(msh.fields) > 1 else "",
                sending_facility=msh.fields[2] if len(msh.fields) > 2 else "",
                receiving_application=msh.fields[3] if len(msh.fields) > 3 else "",
                receiving_facility=msh.fields[4] if len(msh.fields) > 4 else "",
                message_control_id=msh.fields[8] if len(msh.fields) > 8 else ""
            )

        return cls(message_type=HL7MessageType.ACK, event_type=HL7EventType.A01)

    def add_patient_segment(
        self,
        patient_id: str,
        patient_name: str,
        birth_date: str,
        sex: str
    ) -> None:
        """Add PID (Patient Identification) segment."""
        # Name format: Last^First^Middle
        pid = HL7Segment(
            segment_id="PID",
            fields=[
                "1",  # Set ID
                "",   # External ID
                patient_id,  # Internal ID
                "",   # Alternate ID
                patient_name,  # Patient Name
                "",   # Mother's Maiden Name
                birth_date,  # Date of Birth
                sex,  # Sex
            ]
        )
        self.segments.append(pid)

    def add_observation_segment(
        self,
        observation_id: str,
        observation_value: str,
        units: str = "",
        observation_datetime: Optional[str] = None
    ) -> None:
        """Add OBX (Observation) segment."""
        if observation_datetime is None:
            observation_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

        obx = HL7Segment(
            segment_id="OBX",
            fields=[
                str(len([s for s in self.segments if s.segment_id == "OBX"]) + 1),
                "NM" if observation_value.replace(".", "").replace("-", "").isdigit() else "ST",
                observation_id,
                "",
                observation_value,
                units,
                "",  # Reference Range
                "",  # Abnormal Flags
                "",  # Probability
                "",  # Nature of Abnormal Test
                "F",  # Observation Result Status (Final)
                "",
                "",
                observation_datetime
            ]
        )
        self.segments.append(obx)


class HL7Client:
    """
    HL7 MLLP client for sending/receiving messages.

    Uses Minimal Lower Layer Protocol (MLLP) for transport.
    """

    def __init__(self, host: str = "localhost", port: int = 2575) -> None:
        self.host = host
        self.port = port
        self._connected = False

    def connect(self) -> bool:
        """Connect to HL7 server."""
        # Would establish TCP connection with MLLP framing
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False

    def send_message(self, message: HL7Message) -> Optional[HL7Message]:
        """
        Send HL7 message and wait for ACK.

        Args:
            message: HL7 message to send

        Returns:
            ACK message or None if failed
        """
        if not self._connected:
            return None

        # MLLP framing: <VT>message<FS><CR>
        # VT = 0x0B, FS = 0x1C, CR = 0x0D
        wire_message = f"\x0b{message.to_string()}\x1c\r"

        # Would send over socket and receive response
        # Return simulated ACK
        ack = HL7Message(
            message_type=HL7MessageType.ACK,
            event_type=message.event_type,
            receiving_application=message.sending_application
        )
        ack.segments.append(HL7Segment(
            segment_id="MSA",
            fields=["AA", message.message_control_id]  # AA = Application Accept
        ))

        return ack

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


@dataclass
class FHIRResource:
    """Base FHIR resource."""
    resource_type: str
    id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to FHIR JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resourceType": self.resource_type,
            "id": self.id,
            "meta": self.meta
        }


@dataclass
class FHIRPatient(FHIRResource):
    """FHIR Patient resource."""
    identifier: List[Dict] = field(default_factory=list)
    name: List[Dict] = field(default_factory=list)
    gender: str = ""
    birth_date: str = ""

    def __post_init__(self) -> None:
        self.resource_type = "Patient"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "identifier": self.identifier,
            "name": self.name,
            "gender": self.gender,
            "birthDate": self.birth_date
        })
        return result


@dataclass
class FHIRProcedure(FHIRResource):
    """FHIR Procedure resource."""
    status: str = "completed"
    code: Dict = field(default_factory=dict)
    subject: Dict = field(default_factory=dict)
    performed_date_time: str = ""
    performer: List[Dict] = field(default_factory=list)
    outcome: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resource_type = "Procedure"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "status": self.status,
            "code": self.code,
            "subject": self.subject,
            "performedDateTime": self.performed_date_time,
            "performer": self.performer,
            "outcome": self.outcome
        })
        return result


class FHIRClient:
    """
    FHIR R4 client for healthcare data exchange.

    Supports RESTful FHIR operations.
    """

    def __init__(self, base_url: str, auth_token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def create(self, resource: FHIRResource) -> Optional[str]:
        """
        Create a resource (POST).

        Returns resource ID if successful.
        """
        # Would POST to {base_url}/{resource_type}
        resource.id = str(uuid.uuid4())
        return resource.id

    def read(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        """
        Read a resource (GET).

        Returns resource as dictionary.
        """
        # Would GET from {base_url}/{resource_type}/{id}
        return None

    def update(self, resource: FHIRResource) -> bool:
        """Update a resource (PUT)."""
        # Would PUT to {base_url}/{resource_type}/{id}
        return True

    def delete(self, resource_type: str, resource_id: str) -> bool:
        """Delete a resource (DELETE)."""
        # Would DELETE {base_url}/{resource_type}/{id}
        return True

    def search(
        self,
        resource_type: str,
        params: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Search for resources.

        Returns list of matching resources.
        """
        # Would GET from {base_url}/{resource_type}?{params}
        return []


@dataclass
class SurgicalProcedureRecord:
    """
    Record of a surgical procedure for HL7/FHIR reporting.

    Captures all relevant data for healthcare system integration.
    """
    procedure_id: str
    patient_id: str
    procedure_type: str
    procedure_code: str  # CPT or ICD-10-PCS code

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Personnel
    surgeon_id: str = ""
    surgeon_name: str = ""
    assistants: List[str] = field(default_factory=list)

    # Robot data
    robot_system: str = ""
    robot_serial: str = ""

    # Procedure data
    implants_used: List[Dict] = field(default_factory=list)
    instruments_used: List[str] = field(default_factory=list)

    # Measurements
    intraop_measurements: Dict[str, float] = field(default_factory=dict)

    # Outcome
    complications: List[str] = field(default_factory=list)
    estimated_blood_loss: Optional[float] = None
    notes: str = ""

    def to_hl7_oru(self) -> HL7Message:
        """Convert to HL7 ORU message."""
        msg = HL7Message(
            message_type=HL7MessageType.ORU,
            event_type=HL7EventType.R01
        )

        # Add patient segment
        msg.add_patient_segment(
            patient_id=self.patient_id,
            patient_name="",  # Would be populated from patient record
            birth_date="",
            sex=""
        )

        # Add observations for measurements
        for name, value in self.intraop_measurements.items():
            msg.add_observation_segment(
                observation_id=name,
                observation_value=str(value),
                observation_datetime=self.start_time.strftime("%Y%m%d%H%M%S")
            )

        return msg

    def to_fhir_procedure(self) -> FHIRProcedure:
        """Convert to FHIR Procedure resource."""
        return FHIRProcedure(
            id=self.procedure_id,
            status="completed" if self.end_time else "in-progress",
            code={
                "coding": [{
                    "system": "http://www.ama-assn.org/go/cpt",
                    "code": self.procedure_code,
                    "display": self.procedure_type
                }]
            },
            subject={"reference": f"Patient/{self.patient_id}"},
            performed_date_time=self.start_time.isoformat(),
            performer=[{
                "actor": {"reference": f"Practitioner/{self.surgeon_id}"}
            }] if self.surgeon_id else [],
            outcome={
                "text": "Successful" if not self.complications else f"Complications: {', '.join(self.complications)}"
            }
        )

    def record_measurement(self, name: str, value: float) -> None:
        """Record intraoperative measurement."""
        self.intraop_measurements[name] = value

    def record_implant(
        self,
        implant_name: str,
        implant_id: str,
        lot_number: str
    ) -> None:
        """Record implant usage."""
        self.implants_used.append({
            "name": implant_name,
            "id": implant_id,
            "lot": lot_number,
            "timestamp": datetime.now().isoformat()
        })

    def complete(self) -> None:
        """Mark procedure as complete."""
        self.end_time = datetime.now()

    @property
    def duration_minutes(self) -> Optional[float]:
        """Get procedure duration in minutes."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 60
