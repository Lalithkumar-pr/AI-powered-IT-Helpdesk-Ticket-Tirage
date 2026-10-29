"""Contract and interface validation tests for AI-Powered IT Helpdesk Ticket Triage.

These tests verify:
- Exact canonical enums and shared constants
- Canonical 21 ticket field specifications
- Function signatures and fallback return shapes
- Storage layer file-handling and resilience
- Absence of API key requirement during startup/import
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.retrieval import find_similar_tickets
from src.routing import ROUTING_RETURN_KEYS, assign_technician
from src.schemas import (
    CATEGORIES,
    IMPACT_SCOPES,
    PRIORITIES,
    TICKET_FIELDS,
    Category,
    ImpactScope,
    Priority,
    TicketStatus,
    get_default_ticket,
    normalize_ticket,
)
from src.storage import (
    create_ticket,
    load_historical_tickets,
    load_technicians,
    load_tickets,
    save_tickets,
    update_ticket,
)
from src.triage import TRIAGE_RETURN_KEYS, triage_ticket


class TestSharedContracts:
    """Verify frozen constants and canonical values."""

    def test_canonical_categories(self):
        expected_categories = {
            "Hardware",
            "Network",
            "Software",
            "Database",
            "Account & Access",
            "Email & Collaboration",
            "Other",
        }
        assert set(CATEGORIES) == expected_categories
        for cat in expected_categories:
            assert cat in Category._value2member_map_

    def test_canonical_priorities(self):
        expected_priorities = {"Critical", "High", "Medium", "Low"}
        assert set(PRIORITIES) == expected_priorities
        for prio in expected_priorities:
            assert prio in Priority._value2member_map_

    def test_canonical_impact_scopes(self):
        expected_scopes = {
            "single_user",
            "multiple_users",
            "building",
            "campus",
            "unknown",
        }
        assert set(IMPACT_SCOPES) == expected_scopes
        for scope in expected_scopes:
            assert scope in ImpactScope._value2member_map_

    def test_canonical_ticket_fields(self):
        expected_fields = {
            "ticket_id",
            "created_at",
            "raised_by",
            "building",
            "description",
            "category",
            "subcategory",
            "priority",
            "impact_scope",
            "triage_reason",
            "triage_confidence",
            "needs_human_review",
            "similar_tickets",
            "suggested_resolution",
            "assigned_technician_id",
            "assigned_technician_name",
            "assigned_team",
            "assignment_reason",
            "status",
            "resolution",
            "time_to_resolve_hrs",
        }
        assert set(TICKET_FIELDS) == expected_fields
        default_ticket = get_default_ticket()
        assert set(default_ticket.keys()) == expected_fields

    def test_normalize_ticket_safe_defaults(self):
        # Empty dict should be populated with all safe defaults
        normalized = normalize_ticket({})
        assert len(normalized) == len(TICKET_FIELDS)
        assert normalized["category"] == "Other"
        assert normalized["priority"] == "Medium"
        assert normalized["impact_scope"] == "unknown"
        assert normalized["assigned_technician_name"] == "Unassigned"
        assert normalized["needs_human_review"] is False
        assert isinstance(normalized["similar_tickets"], list)
        assert normalized["triage_confidence"] == 0.0
        assert normalized["time_to_resolve_hrs"] == 0.0


class TestModuleSignaturesAndFallbacks:
    """Verify function signatures and valid fallback return shapes."""

    def test_triage_signature_and_fallback(self):
        sig = inspect.signature(triage_ticket)
        params = list(sig.parameters.keys())
        assert params == ["description", "raised_by", "building", "context"]

        result = triage_ticket(description="Projector won't turn on")
        assert isinstance(result, dict)
        for key in TRIAGE_RETURN_KEYS:
            assert key in result, f"Missing key in triage_ticket output: {key}"

        assert result["category"] in CATEGORIES
        assert result["priority"] in PRIORITIES
        assert result["impact_scope"] in IMPACT_SCOPES
        assert isinstance(result["triage_confidence"], float)
        assert isinstance(result["needs_human_review"], bool)
        assert isinstance(result["triage_reason"], str)

    def test_retrieval_signature_and_fallback(self):
        sig = inspect.signature(find_similar_tickets)
        params = list(sig.parameters.keys())
        assert params == ["description", "historical_tickets", "top_k"]

        # Initial stub returns empty list
        matches = find_similar_tickets("Network is down", [], top_k=3)
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_routing_signature_and_fallback(self):
        sig = inspect.signature(assign_technician)
        params = list(sig.parameters.keys())
        assert params == ["category", "priority", "technicians", "active_tickets"]

        result = assign_technician("Hardware", "High", [])
        assert isinstance(result, dict)
        for key in ROUTING_RETURN_KEYS:
            assert key in result, f"Missing key in assign_technician output: {key}"

        assert result["assigned_technician_name"] == "Unassigned"
        assert result["assigned_technician_id"] == ""


class TestStorageLayer:
    """Verify persistence loading, saving, and edge-case handling."""

    def test_load_sample_historical_tickets(self):
        tickets = load_historical_tickets()
        assert isinstance(tickets, list)
        assert len(tickets) >= 10, "Historical sample dataset must contain at least 10 records"
        first = tickets[0]
        assert "ticket_id" in first
        assert "description" in first
        assert "category" in first
        assert "priority" in first
        assert "resolution" in first

    def test_load_sample_technicians(self):
        techs = load_technicians()
        assert isinstance(techs, list)
        assert len(techs) == 4, "Must contain exactly four fictional technicians"
        for tech in techs:
            assert "technician_id" in tech
            assert "name" in tech
            assert "team" in tech
            assert "skills" in tech
            assert isinstance(tech["skills"], list)

    def test_create_and_update_ticket(self, tmp_path, monkeypatch):
        # Redirect storage tickets path to tmp_path to keep tests isolated
        test_tickets_file = tmp_path / "test_tickets.json"
        monkeypatch.setattr("src.storage.TICKETS_PATH", test_tickets_file)

        # 1. Create a ticket
        created = create_ticket({
            "description": "Mouse right-click is physically stuck",
            "raised_by": "faculty",
            "building": "Science Hall",
        })
        assert created["ticket_id"].startswith("TICK-")
        assert created["description"] == "Mouse right-click is physically stuck"
        assert created["status"] == "open"
        assert created["assigned_technician_name"] == "Unassigned"

        # Verify persisted
        loaded = load_tickets()
        assert len(loaded) == 1
        assert loaded[0]["ticket_id"] == created["ticket_id"]

        # 2. Update the ticket
        updated = update_ticket(
            created["ticket_id"],
            {
                "status": "in_progress",
                "assigned_technician_name": "Alex Morgan",
                "assigned_technician_id": "TECH-001",
            },
        )
        assert updated is not None
        assert updated["status"] == "in_progress"
        assert updated["assigned_technician_name"] == "Alex Morgan"
        # Untouched fields must be preserved
        assert updated["description"] == "Mouse right-click is physically stuck"
        assert updated["building"] == "Science Hall"

        # 3. Nonexistent ticket update returns None
        assert update_ticket("NON-EXISTENT-ID", {"status": "closed"}) is None

    def test_storage_robustness_missing_files(self, tmp_path, monkeypatch):
        missing_csv = tmp_path / "missing.csv"
        missing_json = tmp_path / "missing.json"
        monkeypatch.setattr("src.storage.HISTORICAL_TICKETS_PATH", missing_csv)
        monkeypatch.setattr("src.storage.TECHNICIANS_PATH", missing_json)
        monkeypatch.setattr("src.storage.TICKETS_PATH", missing_json)

        assert load_historical_tickets() == []
        assert load_technicians() == []
        assert load_tickets() == []
