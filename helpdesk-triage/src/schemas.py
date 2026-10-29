"""Shared schemas, constants, and contracts for AI-Powered IT Helpdesk Ticket Triage.

All team members must use the canonical constants, enums, and field names defined
in this module to ensure compatibility across triage, retrieval, routing, UI, and storage.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, TypedDict


class Category(str, Enum):
    """Canonical ticket categories."""
    HARDWARE = "Hardware"
    NETWORK = "Network"
    SOFTWARE = "Software"
    DATABASE = "Database"
    ACCOUNT_AND_ACCESS = "Account & Access"
    EMAIL_AND_COLLABORATION = "Email & Collaboration"
    OTHER = "Other"


class Priority(str, Enum):
    """Canonical ticket priorities."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ImpactScope(str, Enum):
    """Canonical impact scopes."""
    SINGLE_USER = "single_user"
    MULTIPLE_USERS = "multiple_users"
    BUILDING = "building"
    CAMPUS = "campus"
    UNKNOWN = "unknown"


class TicketStatus(str, Enum):
    """Canonical ticket lifecycle statuses."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


# Canonical tuple constants for quick iteration, membership testing, and UI dropdowns
CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)
PRIORITIES: tuple[str, ...] = tuple(p.value for p in Priority)
IMPACT_SCOPES: tuple[str, ...] = tuple(s.value for s in ImpactScope)
TICKET_STATUSES: tuple[str, ...] = tuple(st.value for st in TicketStatus)

# Canonical list of all 21 ticket fields
TICKET_FIELDS: tuple[str, ...] = (
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
)


class TicketDict(TypedDict, total=False):
    """Typed dictionary representing a canonical ticket."""
    ticket_id: str
    created_at: str
    raised_by: str
    building: str
    description: str
    category: str
    subcategory: str
    priority: str
    impact_scope: str
    triage_reason: str
    triage_confidence: float
    needs_human_review: bool
    similar_tickets: List[Dict[str, Any]]
    suggested_resolution: str
    assigned_technician_id: str
    assigned_technician_name: str
    assigned_team: str
    assignment_reason: str
    status: str
    resolution: str
    time_to_resolve_hrs: float


def get_default_ticket() -> dict[str, Any]:
    """Return a dictionary containing all canonical ticket fields populated with safe defaults."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "ticket_id": "",
        "created_at": now_iso,
        "raised_by": "student",
        "building": "Unknown",
        "description": "",
        "category": Category.OTHER.value,
        "subcategory": "General",
        "priority": Priority.MEDIUM.value,
        "impact_scope": ImpactScope.UNKNOWN.value,
        "triage_reason": "",
        "triage_confidence": 0.0,
        "needs_human_review": False,
        "similar_tickets": [],
        "suggested_resolution": "",
        "assigned_technician_id": "",
        "assigned_technician_name": "Unassigned",
        "assigned_team": "",
        "assignment_reason": "",
        "status": TicketStatus.OPEN.value,
        "resolution": "",
        "time_to_resolve_hrs": 0.0,
    }


def normalize_ticket(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure a ticket dictionary contains all canonical fields with safe defaults.
    
    Any existing values are preserved. Missing fields are populated with safe defaults.
    """
    ticket = get_default_ticket()
    ticket.update(data)
    # Ensure numerical/boolean types are coerced safely if present
    if "triage_confidence" in ticket:
        try:
            ticket["triage_confidence"] = float(ticket["triage_confidence"])
        except (ValueError, TypeError):
            ticket["triage_confidence"] = 0.0

    if "time_to_resolve_hrs" in ticket:
        try:
            ticket["time_to_resolve_hrs"] = float(ticket["time_to_resolve_hrs"])
        except (ValueError, TypeError):
            ticket["time_to_resolve_hrs"] = 0.0

    if "needs_human_review" in ticket:
        ticket["needs_human_review"] = bool(ticket["needs_human_review"])

    if "similar_tickets" in ticket and not isinstance(ticket["similar_tickets"], list):
        ticket["similar_tickets"] = []

    return ticket
