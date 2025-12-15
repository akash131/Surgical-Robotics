"""IEC 60601 Medical Electrical Equipment compliance helpers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class SafetyClassification(Enum):
    """IEC 60601-1 Equipment Classifications."""
    # Protection against electric shock
    CLASS_I = "Class I"  # Earth grounded
    CLASS_II = "Class II"  # Double insulation
    INTERNALLY_POWERED = "Internally Powered"

    # Applied parts classification
    TYPE_B = "Type B"  # Body contact
    TYPE_BF = "Type BF"  # Body contact, floating
    TYPE_CF = "Type CF"  # Cardiac contact, floating


class OperatingMode(Enum):
    """Operating modes."""
    CONTINUOUS = "continuous"
    SHORT_TIME = "short_time"
    INTERMITTENT = "intermittent"


@dataclass
class AppliedPart:
    """Applied part definition (patient contact)."""
    name: str
    classification: SafetyClassification
    description: str = ""

    # Contact type
    body_contact: bool = True
    cardiac_contact: bool = False

    # Limits
    max_patient_current_ac: float = 0.0  # µA
    max_patient_current_dc: float = 0.0  # µA

    def __post_init__(self) -> None:
        """Set current limits based on classification."""
        if self.classification == SafetyClassification.TYPE_B:
            self.max_patient_current_ac = 100.0  # µA
            self.max_patient_current_dc = 10.0  # µA
        elif self.classification == SafetyClassification.TYPE_BF:
            self.max_patient_current_ac = 100.0
            self.max_patient_current_dc = 10.0
        elif self.classification == SafetyClassification.TYPE_CF:
            self.max_patient_current_ac = 10.0
            self.max_patient_current_dc = 10.0


@dataclass
class Hazard:
    """Identified hazard."""
    id: str
    name: str
    description: str

    # Classification
    hazard_type: str = ""  # electrical, mechanical, thermal, radiation, etc.
    source: str = ""
    harm: str = ""

    # Severity (ISO 14971)
    severity: int = 5  # 1-5, 5 being catastrophic
    probability: int = 5  # 1-5, 5 being frequent
    risk_level: str = "high"  # low, medium, high

    # Mitigation
    mitigations: List[str] = field(default_factory=list)
    residual_risk: str = "acceptable"

    def calculate_risk(self) -> int:
        """Calculate risk priority number."""
        return self.severity * self.probability

    def update_risk_level(self) -> None:
        """Update risk level based on severity and probability."""
        rpn = self.calculate_risk()
        if rpn <= 4:
            self.risk_level = "low"
        elif rpn <= 12:
            self.risk_level = "medium"
        else:
            self.risk_level = "high"


class RiskAnalysis:
    """
    Risk analysis per ISO 14971 and IEC 60601-1.

    Manages hazard identification, risk estimation,
    and risk control measures.
    """

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name
        self.hazards: Dict[str, Hazard] = {}
        self.risk_controls: Dict[str, Dict[str, Any]] = {}

        # Risk acceptability matrix
        self.risk_matrix = self._default_risk_matrix()

    def _default_risk_matrix(self) -> Dict[tuple, str]:
        """Default risk acceptability matrix."""
        # (severity, probability) -> acceptability
        matrix = {}
        for s in range(1, 6):
            for p in range(1, 6):
                rpn = s * p
                if rpn <= 4:
                    matrix[(s, p)] = "acceptable"
                elif rpn <= 8:
                    matrix[(s, p)] = "alarp"  # As Low As Reasonably Practicable
                elif rpn <= 15:
                    matrix[(s, p)] = "undesirable"
                else:
                    matrix[(s, p)] = "unacceptable"
        return matrix

    def identify_hazard(
        self,
        name: str,
        description: str,
        hazard_type: str,
        harm: str,
        severity: int,
        probability: int
    ) -> str:
        """Identify and record a hazard."""
        hazard_id = f"HAZ-{len(self.hazards) + 1:03d}"

        hazard = Hazard(
            id=hazard_id,
            name=name,
            description=description,
            hazard_type=hazard_type,
            harm=harm,
            severity=severity,
            probability=probability
        )
        hazard.update_risk_level()

        self.hazards[hazard_id] = hazard
        return hazard_id

    def add_risk_control(
        self,
        hazard_id: str,
        control_type: str,
        description: str,
        effectiveness: str = "effective"
    ) -> None:
        """Add risk control measure."""
        if hazard_id not in self.hazards:
            return

        control_id = f"RC-{hazard_id}-{len(self.risk_controls) + 1}"

        self.risk_controls[control_id] = {
            "hazard_id": hazard_id,
            "control_type": control_type,  # inherent_safety, protective, information
            "description": description,
            "effectiveness": effectiveness,
            "verified": False,
            "verification_method": ""
        }

        # Add to hazard mitigations
        self.hazards[hazard_id].mitigations.append(description)

    def evaluate_residual_risk(
        self,
        hazard_id: str,
        new_probability: int
    ) -> str:
        """Evaluate residual risk after controls."""
        if hazard_id not in self.hazards:
            return "unknown"

        hazard = self.hazards[hazard_id]
        residual = self.risk_matrix.get(
            (hazard.severity, new_probability),
            "unknown"
        )

        hazard.residual_risk = residual
        return residual

    def get_unacceptable_risks(self) -> List[Hazard]:
        """Get hazards with unacceptable risk."""
        return [
            h for h in self.hazards.values()
            if h.residual_risk in ["unacceptable", "undesirable"]
        ]

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate risk management report."""
        total_hazards = len(self.hazards)
        by_type = {}
        by_level = {"low": 0, "medium": 0, "high": 0}

        for hazard in self.hazards.values():
            htype = hazard.hazard_type
            by_type[htype] = by_type.get(htype, 0) + 1
            by_level[hazard.risk_level] = by_level.get(hazard.risk_level, 0) + 1

        return {
            "device_name": self.device_name,
            "total_hazards": total_hazards,
            "hazards_by_type": by_type,
            "hazards_by_risk_level": by_level,
            "total_controls": len(self.risk_controls),
            "unacceptable_risks": len(self.get_unacceptable_risks()),
            "hazards": [
                {
                    "id": h.id,
                    "name": h.name,
                    "risk_level": h.risk_level,
                    "residual_risk": h.residual_risk,
                    "mitigations": len(h.mitigations)
                }
                for h in self.hazards.values()
            ]
        }


class ElectricalSafetyChecker:
    """
    Checks IEC 60601-1 electrical safety requirements.

    Validates:
    - Leakage currents
    - Dielectric strength
    - Grounding
    - Applied part classification
    """

    # IEC 60601-1 limits (normal condition)
    EARTH_LEAKAGE_LIMIT = 500  # µA
    ENCLOSURE_LEAKAGE_LIMIT = 100  # µA
    PATIENT_LEAKAGE_B = 100  # µA
    PATIENT_LEAKAGE_BF = 100  # µA
    PATIENT_LEAKAGE_CF = 10  # µA

    # Single fault condition multipliers
    SFC_MULTIPLIER = 5

    def __init__(self) -> None:
        self.applied_parts: Dict[str, AppliedPart] = {}
        self.measurements: List[Dict[str, Any]] = []
        self.test_results: Dict[str, bool] = {}

    def add_applied_part(self, part: AppliedPart) -> None:
        """Register applied part."""
        self.applied_parts[part.name] = part

    def record_measurement(
        self,
        test_type: str,
        value: float,
        unit: str,
        condition: str = "normal",
        applied_part: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record safety measurement."""
        measurement = {
            "test_type": test_type,
            "value": value,
            "unit": unit,
            "condition": condition,
            "applied_part": applied_part,
            "timestamp": datetime.utcnow().isoformat(),
            "passed": self._evaluate_measurement(test_type, value, condition, applied_part)
        }

        self.measurements.append(measurement)
        return measurement

    def _evaluate_measurement(
        self,
        test_type: str,
        value: float,
        condition: str,
        applied_part: Optional[str]
    ) -> bool:
        """Evaluate if measurement passes."""
        multiplier = self.SFC_MULTIPLIER if condition == "single_fault" else 1

        if test_type == "earth_leakage":
            return value <= self.EARTH_LEAKAGE_LIMIT * multiplier

        elif test_type == "enclosure_leakage":
            return value <= self.ENCLOSURE_LEAKAGE_LIMIT * multiplier

        elif test_type == "patient_leakage":
            if applied_part and applied_part in self.applied_parts:
                part = self.applied_parts[applied_part]
                limit = part.max_patient_current_ac
            else:
                limit = self.PATIENT_LEAKAGE_B
            return value <= limit * multiplier

        return True

    def check_earth_continuity(
        self,
        resistance: float,
        max_resistance: float = 0.1
    ) -> bool:
        """Check protective earth continuity."""
        passed = resistance <= max_resistance
        self.test_results["earth_continuity"] = passed
        return passed

    def check_insulation_resistance(
        self,
        resistance_megohm: float,
        min_resistance: float = 2.0
    ) -> bool:
        """Check insulation resistance."""
        passed = resistance_megohm >= min_resistance
        self.test_results["insulation_resistance"] = passed
        return passed

    def check_dielectric_strength(
        self,
        test_voltage: float,
        breakdown: bool
    ) -> bool:
        """Check dielectric strength (no breakdown)."""
        passed = not breakdown
        self.test_results["dielectric_strength"] = passed
        return passed

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate electrical safety test report."""
        passed_measurements = sum(1 for m in self.measurements if m["passed"])

        return {
            "total_measurements": len(self.measurements),
            "passed_measurements": passed_measurements,
            "failed_measurements": len(self.measurements) - passed_measurements,
            "test_results": self.test_results,
            "applied_parts": [
                {
                    "name": p.name,
                    "classification": p.classification.value,
                    "max_current_ac": p.max_patient_current_ac,
                    "max_current_dc": p.max_patient_current_dc
                }
                for p in self.applied_parts.values()
            ],
            "measurements": self.measurements,
            "overall_pass": all(self.test_results.values()) and passed_measurements == len(self.measurements)
        }


# Common surgical robot hazards template
def get_surgical_robot_hazards() -> List[Dict[str, Any]]:
    """Get common hazards for surgical robots."""
    return [
        {
            "name": "Unintended robot motion",
            "type": "mechanical",
            "harm": "Patient injury, tissue damage",
            "severity": 5,
            "probability": 3
        },
        {
            "name": "Excessive force application",
            "type": "mechanical",
            "harm": "Tissue damage, organ perforation",
            "severity": 5,
            "probability": 3
        },
        {
            "name": "Instrument breakage",
            "type": "mechanical",
            "harm": "Foreign body retention, laceration",
            "severity": 4,
            "probability": 2
        },
        {
            "name": "Loss of position feedback",
            "type": "software",
            "harm": "Incorrect positioning, tissue damage",
            "severity": 4,
            "probability": 2
        },
        {
            "name": "Communication failure",
            "type": "software",
            "harm": "Loss of control, unintended motion",
            "severity": 4,
            "probability": 2
        },
        {
            "name": "Electrical shock to patient",
            "type": "electrical",
            "harm": "Burns, cardiac effects",
            "severity": 5,
            "probability": 1
        },
        {
            "name": "Thermal injury from motors",
            "type": "thermal",
            "harm": "Burns",
            "severity": 3,
            "probability": 2
        },
        {
            "name": "Infection from non-sterile components",
            "type": "biological",
            "harm": "Surgical site infection",
            "severity": 3,
            "probability": 2
        }
    ]
