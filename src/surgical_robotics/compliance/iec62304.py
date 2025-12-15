"""IEC 62304 Medical Device Software Lifecycle compliance helpers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set
from datetime import datetime
import hashlib


class SoftwareClass(Enum):
    """IEC 62304 Software Safety Classification."""
    CLASS_A = "A"  # No injury or damage to health possible
    CLASS_B = "B"  # Non-serious injury possible
    CLASS_C = "C"  # Death or serious injury possible


class LifecyclePhase(Enum):
    """Software development lifecycle phases."""
    PLANNING = "planning"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    DETAILED_DESIGN = "detailed_design"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    RELEASE = "release"
    MAINTENANCE = "maintenance"


@dataclass
class SoftwareItem:
    """
    Software item (unit, module, or system).

    IEC 62304 defines software items at various levels of decomposition.
    """
    id: str
    name: str
    description: str = ""
    software_class: SoftwareClass = SoftwareClass.CLASS_C

    # Version control
    version: str = "1.0.0"
    checksum: str = ""

    # Relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

    # Lifecycle
    phase: LifecyclePhase = LifecyclePhase.PLANNING

    # Traceability
    requirement_ids: List[str] = field(default_factory=list)
    test_ids: List[str] = field(default_factory=list)

    # Risk
    hazard_ids: List[str] = field(default_factory=list)
    risk_level: str = "high"

    # Metadata
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    author: str = ""
    reviewer: str = ""
    approver: str = ""


@dataclass
class Requirement:
    """Software requirement."""
    id: str
    title: str
    description: str
    requirement_type: str = "functional"  # functional, performance, safety, interface

    # Classification
    priority: str = "must"  # must, should, could
    safety_related: bool = False

    # Traceability
    source: str = ""  # Where requirement came from
    parent_id: Optional[str] = None
    derived_ids: List[str] = field(default_factory=list)

    # Verification
    verification_method: str = "test"  # test, inspection, analysis, demonstration
    test_ids: List[str] = field(default_factory=list)

    # Status
    status: str = "draft"  # draft, approved, implemented, verified


@dataclass
class TestCase:
    """Verification test case."""
    id: str
    title: str
    description: str

    # Links
    requirement_ids: List[str] = field(default_factory=list)
    software_item_id: Optional[str] = None

    # Test details
    preconditions: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)

    # Execution
    status: str = "pending"  # pending, passed, failed, blocked
    executed_date: Optional[str] = None
    executed_by: Optional[str] = None
    actual_results: str = ""
    defect_ids: List[str] = field(default_factory=list)


class RequirementTraceability:
    """
    Manages requirement traceability matrix.

    IEC 62304 requires bidirectional traceability:
    - Requirements → Design → Implementation → Tests
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, Requirement] = {}
        self.software_items: Dict[str, SoftwareItem] = {}
        self.tests: Dict[str, TestCase] = {}

        # Traceability links
        self._req_to_items: Dict[str, Set[str]] = {}
        self._item_to_reqs: Dict[str, Set[str]] = {}
        self._req_to_tests: Dict[str, Set[str]] = {}
        self._test_to_reqs: Dict[str, Set[str]] = {}

    def add_requirement(self, req: Requirement) -> None:
        """Add requirement."""
        self.requirements[req.id] = req
        self._req_to_items[req.id] = set()
        self._req_to_tests[req.id] = set()

    def add_software_item(self, item: SoftwareItem) -> None:
        """Add software item."""
        self.software_items[item.id] = item
        self._item_to_reqs[item.id] = set()

        # Link to requirements
        for req_id in item.requirement_ids:
            if req_id in self._req_to_items:
                self._req_to_items[req_id].add(item.id)
                self._item_to_reqs[item.id].add(req_id)

    def add_test(self, test: TestCase) -> None:
        """Add test case."""
        self.tests[test.id] = test
        self._test_to_reqs[test.id] = set()

        # Link to requirements
        for req_id in test.requirement_ids:
            if req_id in self._req_to_tests:
                self._req_to_tests[req_id].add(test.id)
                self._test_to_reqs[test.id].add(req_id)

    def link_requirement_to_item(self, req_id: str, item_id: str) -> None:
        """Create traceability link."""
        if req_id in self._req_to_items:
            self._req_to_items[req_id].add(item_id)
        if item_id in self._item_to_reqs:
            self._item_to_reqs[item_id].add(req_id)

    def link_requirement_to_test(self, req_id: str, test_id: str) -> None:
        """Create traceability link."""
        if req_id in self._req_to_tests:
            self._req_to_tests[req_id].add(test_id)
        if test_id in self._test_to_reqs:
            self._test_to_reqs[test_id].add(req_id)

    def get_unlinked_requirements(self) -> List[str]:
        """Get requirements without implementation links."""
        return [
            req_id for req_id, items in self._req_to_items.items()
            if not items
        ]

    def get_untested_requirements(self) -> List[str]:
        """Get requirements without test coverage."""
        return [
            req_id for req_id, tests in self._req_to_tests.items()
            if not tests
        ]

    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate traceability coverage report."""
        total_reqs = len(self.requirements)
        implemented = sum(1 for items in self._req_to_items.values() if items)
        tested = sum(1 for tests in self._req_to_tests.values() if tests)

        # Test results
        passed_tests = sum(1 for t in self.tests.values() if t.status == "passed")
        failed_tests = sum(1 for t in self.tests.values() if t.status == "failed")

        return {
            "total_requirements": total_reqs,
            "implemented_requirements": implemented,
            "implementation_coverage": implemented / total_reqs * 100 if total_reqs > 0 else 0,
            "tested_requirements": tested,
            "test_coverage": tested / total_reqs * 100 if total_reqs > 0 else 0,
            "total_tests": len(self.tests),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "unlinked_requirements": self.get_unlinked_requirements(),
            "untested_requirements": self.get_untested_requirements()
        }

    def export_matrix(self) -> List[Dict[str, Any]]:
        """Export full traceability matrix."""
        matrix = []

        for req_id, req in self.requirements.items():
            row = {
                "requirement_id": req_id,
                "requirement_title": req.title,
                "safety_related": req.safety_related,
                "software_items": list(self._req_to_items.get(req_id, [])),
                "tests": list(self._req_to_tests.get(req_id, [])),
                "test_status": self._get_req_test_status(req_id)
            }
            matrix.append(row)

        return matrix

    def _get_req_test_status(self, req_id: str) -> str:
        """Get aggregated test status for requirement."""
        test_ids = self._req_to_tests.get(req_id, set())
        if not test_ids:
            return "no_tests"

        statuses = [self.tests[tid].status for tid in test_ids if tid in self.tests]
        if all(s == "passed" for s in statuses):
            return "passed"
        if any(s == "failed" for s in statuses):
            return "failed"
        return "pending"


class SoftwareLifecycleManager:
    """
    Manages IEC 62304 software development lifecycle.

    Tracks:
    - Development phases
    - Configuration management
    - Problem resolution
    - Risk management integration
    """

    def __init__(
        self,
        product_name: str,
        software_class: SoftwareClass = SoftwareClass.CLASS_C
    ) -> None:
        self.product_name = product_name
        self.software_class = software_class

        # Traceability
        self.traceability = RequirementTraceability()

        # Configuration
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.current_baseline: Optional[str] = None

        # Problem reports
        self.problems: Dict[str, Dict[str, Any]] = {}

        # Phase records
        self.phase_records: Dict[LifecyclePhase, Dict[str, Any]] = {}

    def get_required_activities(self) -> List[str]:
        """Get required activities based on software class."""
        # IEC 62304 Table A.1 - Activities vs Software Safety Class
        class_c_activities = [
            "5.1 Software development planning",
            "5.2 Software requirements analysis",
            "5.3 Software architectural design",
            "5.4 Software detailed design",
            "5.5 Software unit implementation",
            "5.6 Software integration and integration testing",
            "5.7 Software system testing",
            "5.8 Software release",
            "6 Software maintenance",
            "7 Software risk management",
            "8 Software configuration management",
            "9 Software problem resolution"
        ]

        class_b_activities = [
            "5.1 Software development planning",
            "5.2 Software requirements analysis",
            "5.3 Software architectural design",
            "5.5 Software unit implementation",
            "5.6 Software integration and integration testing",
            "5.7 Software system testing",
            "5.8 Software release",
            "6 Software maintenance",
            "7 Software risk management",
            "8 Software configuration management",
            "9 Software problem resolution"
        ]

        class_a_activities = [
            "5.1 Software development planning",
            "5.2 Software requirements analysis",
            "5.5 Software unit implementation",
            "5.7 Software system testing",
            "5.8 Software release",
            "6 Software maintenance",
            "8 Software configuration management",
            "9 Software problem resolution"
        ]

        if self.software_class == SoftwareClass.CLASS_C:
            return class_c_activities
        elif self.software_class == SoftwareClass.CLASS_B:
            return class_b_activities
        else:
            return class_a_activities

    def create_baseline(
        self,
        version: str,
        description: str,
        items: List[SoftwareItem]
    ) -> str:
        """Create configuration baseline."""
        baseline_id = f"BL-{version}-{datetime.now().strftime('%Y%m%d')}"

        # Compute checksum of all items
        item_data = "".join(f"{i.id}:{i.version}:{i.checksum}" for i in items)
        baseline_checksum = hashlib.sha256(item_data.encode()).hexdigest()

        self.configurations[baseline_id] = {
            "version": version,
            "description": description,
            "created": datetime.utcnow().isoformat(),
            "checksum": baseline_checksum,
            "items": [
                {"id": i.id, "version": i.version, "checksum": i.checksum}
                for i in items
            ]
        }

        self.current_baseline = baseline_id
        return baseline_id

    def report_problem(
        self,
        title: str,
        description: str,
        severity: str,
        reporter: str,
        affected_items: Optional[List[str]] = None
    ) -> str:
        """Report software problem."""
        problem_id = f"PR-{len(self.problems) + 1:04d}"

        self.problems[problem_id] = {
            "title": title,
            "description": description,
            "severity": severity,  # critical, major, minor
            "reporter": reporter,
            "reported_date": datetime.utcnow().isoformat(),
            "affected_items": affected_items or [],
            "status": "open",
            "resolution": None,
            "verified": False
        }

        return problem_id

    def resolve_problem(
        self,
        problem_id: str,
        resolution: str,
        resolver: str
    ) -> None:
        """Resolve a problem report."""
        if problem_id in self.problems:
            self.problems[problem_id].update({
                "resolution": resolution,
                "resolver": resolver,
                "resolved_date": datetime.utcnow().isoformat(),
                "status": "resolved"
            })

    def verify_problem_resolution(
        self,
        problem_id: str,
        verifier: str
    ) -> None:
        """Verify problem resolution."""
        if problem_id in self.problems:
            self.problems[problem_id].update({
                "verified": True,
                "verifier": verifier,
                "verified_date": datetime.utcnow().isoformat(),
                "status": "closed"
            })

    def record_phase_completion(
        self,
        phase: LifecyclePhase,
        deliverables: List[str],
        approver: str
    ) -> None:
        """Record phase completion."""
        self.phase_records[phase] = {
            "completed_date": datetime.utcnow().isoformat(),
            "deliverables": deliverables,
            "approver": approver
        }

    def generate_compliance_checklist(self) -> List[Dict[str, Any]]:
        """Generate IEC 62304 compliance checklist."""
        required = self.get_required_activities()
        checklist = []

        for activity in required:
            section = activity.split()[0]
            status = "incomplete"

            # Check phase records
            phase_map = {
                "5.1": LifecyclePhase.PLANNING,
                "5.2": LifecyclePhase.REQUIREMENTS,
                "5.3": LifecyclePhase.ARCHITECTURE,
                "5.4": LifecyclePhase.DETAILED_DESIGN,
                "5.5": LifecyclePhase.IMPLEMENTATION,
                "5.6": LifecyclePhase.INTEGRATION,
                "5.7": LifecyclePhase.VERIFICATION,
                "5.8": LifecyclePhase.RELEASE,
            }

            if section in phase_map:
                phase = phase_map[section]
                if phase in self.phase_records:
                    status = "complete"

            checklist.append({
                "activity": activity,
                "status": status,
                "required_for_class": self.software_class.value
            })

        return checklist
