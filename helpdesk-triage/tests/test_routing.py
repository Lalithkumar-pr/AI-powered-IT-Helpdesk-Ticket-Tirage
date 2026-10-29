"""Unit and acceptance tests for Member 4 technician routing module.

Tests cover:
- Exact return dictionary schema and keys
- Deterministic skill-first matching
- Dynamic and static workload calculation
- Capacity limit enforcement
- Preventing assignment of unskilled technicians regardless of workload
- Graceful unassigned fallbacks (empty list, unknown category, all full)
- Priority-aware human-readable assignment rationale
- Offline/unavailable technician handling
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.routing import ROUTING_RETURN_KEYS, assign_technician
from src.storage import load_technicians


@pytest.fixture
def sample_technicians():
    """Return the four standard technician profiles from data/technicians.json."""
    return load_technicians()


class TestTechnicianRouting:
    """Test suite for assign_technician implementation."""

    def test_routing_return_schema(self, sample_technicians):
        """Verify the exact four return keys are always present."""
        result = assign_technician("Network", "Critical", sample_technicians)
        assert isinstance(result, dict)
        for key in ROUTING_RETURN_KEYS:
            assert key in result
            assert isinstance(result[key], str)

    def test_network_category_routes_to_network_technician(self, sample_technicians):
        """Network incident routes to Jordan Patel (Network Operations)."""
        result = assign_technician("Network", "Critical", sample_technicians)
        assert result["assigned_technician_id"] == "TECH-002"
        assert result["assigned_technician_name"] == "Jordan Patel"
        assert result["assigned_team"] == "Network Operations"
        assert "Network" in result["assignment_reason"]
        assert "Critical" in result["assignment_reason"]

    def test_hardware_category_routes_to_hardware_technician(self, sample_technicians):
        """Hardware incident routes to Alex Morgan (Field Hardware Support)."""
        result = assign_technician("Hardware", "Medium", sample_technicians)
        assert result["assigned_technician_id"] == "TECH-001"
        assert result["assigned_technician_name"] == "Alex Morgan"
        assert result["assigned_team"] == "Field Hardware Support"

    def test_software_category_routes_to_software_technician(self, sample_technicians):
        """Software incident routes to Taylor Reed (Application & Collaboration Support)."""
        result = assign_technician("Software", "Low", sample_technicians)
        assert result["assigned_technician_id"] == "TECH-003"
        assert result["assigned_technician_name"] == "Taylor Reed"

    def test_database_category_routes_to_identity_data_technician(self, sample_technicians):
        """Database incident routes to Casey Kim (Identity & Data Systems)."""
        result = assign_technician("Database", "High", sample_technicians)
        assert result["assigned_technician_id"] == "TECH-004"
        assert result["assigned_technician_name"] == "Casey Kim"

    def test_skill_aliases_matching(self, sample_technicians):
        """Sub-skills like 'Wireless' or 'Desktop' match the respective category technicians."""
        res_wireless = assign_technician("Wireless", "Critical", sample_technicians)
        assert res_wireless["assigned_technician_id"] == "TECH-002"

        res_desktop = assign_technician("Desktop", "Low", sample_technicians)
        assert res_desktop["assigned_technician_id"] == "TECH-001"

    def test_workload_tie_breaking(self):
        """When multiple technicians share a skill, the one with lower workload is chosen."""
        techs = [
            {
                "technician_id": "TECH-A",
                "name": "Alice Busy",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 5,
                "active_tickets_count": 3,
                "status": "available",
            },
            {
                "technician_id": "TECH-B",
                "name": "Bob Free",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 5,
                "active_tickets_count": 1,
                "status": "available",
            },
        ]
        result = assign_technician("Network", "High", techs)
        assert result["assigned_technician_id"] == "TECH-B"
        assert result["assigned_technician_name"] == "Bob Free"
        assert "1/5 active tickets" in result["assignment_reason"]

    def test_dynamic_active_tickets_override(self, sample_technicians):
        """Dynamic active_tickets list overrides static active_tickets_count."""
        # Jordan Patel (TECH-002) has static count 0.
        # Let's provide active_tickets showing Jordan Patel has 3 active tickets.
        active_tickets = [
            {"ticket_id": "TKT-1", "assigned_technician_id": "TECH-002", "status": "open"},
            {"ticket_id": "TKT-2", "assigned_technician_id": "TECH-002", "status": "in_progress"},
            {"ticket_id": "TKT-3", "assigned_technician_id": "TECH-002", "status": "open"},
            # Resolved tickets must NOT count towards active workload
            {"ticket_id": "TKT-4", "assigned_technician_id": "TECH-002", "status": "resolved"},
        ]
        result = assign_technician("Network", "Medium", sample_technicians, active_tickets=active_tickets)
        assert result["assigned_technician_id"] == "TECH-002"
        assert "3/4 active tickets" in result["assignment_reason"]

    def test_capacity_limit_skips_overloaded_technician(self):
        """An overloaded technician is avoided if an eligible technician with capacity exists."""
        techs = [
            {
                "technician_id": "TECH-FULL",
                "name": "Full Tech",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 3,
                "active_tickets_count": 3,
                "status": "available",
            },
            {
                "technician_id": "TECH-OPEN",
                "name": "Open Tech",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 3,
                "active_tickets_count": 2,
                "status": "available",
            },
        ]
        result = assign_technician("Network", "High", techs)
        assert result["assigned_technician_id"] == "TECH-OPEN"
        assert result["assigned_technician_name"] == "Open Tech"

    def test_all_technicians_overloaded_returns_unassigned(self):
        """If all skilled technicians are at maximum capacity, return Unassigned with reason."""
        techs = [
            {
                "technician_id": "TECH-FULL-1",
                "name": "Full Tech 1",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 2,
                "active_tickets_count": 2,
                "status": "available",
            },
            {
                "technician_id": "TECH-FULL-2",
                "name": "Full Tech 2",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 4,
                "active_tickets_count": 4,
                "status": "available",
            },
        ]
        result = assign_technician("Network", "Critical", techs)
        assert result["assigned_technician_id"] == ""
        assert result["assigned_technician_name"] == "Unassigned"
        assert "maximum capacity" in result["assignment_reason"]

    def test_unskilled_technician_never_assigned(self):
        """Technician without relevant skill is NEVER selected just because workload is 0."""
        techs = [
            {
                "technician_id": "TECH-HW",
                "name": "Hardware Specialist",
                "team": "Hardware",
                "skills": ["Hardware"],
                "max_workload": 5,
                "active_tickets_count": 0,
                "status": "available",
            }
        ]
        # Complaint is Network, but only Hardware tech is available
        result = assign_technician("Network", "Critical", techs)
        assert result["assigned_technician_id"] == ""
        assert result["assigned_technician_name"] == "Unassigned"
        assert "No technician with matching skill set" in result["assignment_reason"]

    def test_offline_technician_is_skipped(self):
        """Offline or on-leave technician is not assigned."""
        techs = [
            {
                "technician_id": "TECH-OFFLINE",
                "name": "Vacationing Tech",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 5,
                "active_tickets_count": 0,
                "status": "on_leave",
            },
            {
                "technician_id": "TECH-ONLINE",
                "name": "Working Tech",
                "team": "NetOps",
                "skills": ["Network"],
                "max_workload": 5,
                "active_tickets_count": 1,
                "status": "available",
            },
        ]
        result = assign_technician("Network", "High", techs)
        assert result["assigned_technician_id"] == "TECH-ONLINE"

    def test_empty_technicians_list_fallback(self):
        """Empty technician list returns safe Unassigned."""
        result = assign_technician("Network", "High", [])
        assert result["assigned_technician_id"] == ""
        assert result["assigned_technician_name"] == "Unassigned"
        assert "No technician profiles available" in result["assignment_reason"]

    def test_missing_category_fallback(self, sample_technicians):
        """Empty or blank category returns safe Unassigned."""
        result = assign_technician("", "Low", sample_technicians)
        assert result["assigned_technician_id"] == ""
        assert result["assigned_technician_name"] == "Unassigned"
        assert "Missing ticket category" in result["assignment_reason"]
