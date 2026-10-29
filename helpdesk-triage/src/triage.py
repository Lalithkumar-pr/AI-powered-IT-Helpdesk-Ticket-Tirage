"""Ticket classification and triage module.

Owner: Member 2 (Groq Classification Specialist)
Responsibilities:
- Integrate Groq API with structured JSON output / prompt engineering
- Classify incident into canonical Category, subcategory, Priority, and ImpactScope
- Generate transparent triage_reason, triage_confidence (0.0 to 1.0), and needs_human_review flag

Important:
- Public function signature and return dictionary keys must NOT be changed.
- Use canonical values from src.schemas (Category, Priority, ImpactScope).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.schemas import Category, ImpactScope, Priority

# Canonical keys returned by triage_ticket
TRIAGE_RETURN_KEYS = (
    "category",
    "subcategory",
    "priority",
    "impact_scope",
    "triage_reason",
    "triage_confidence",
    "needs_human_review",
)


def triage_ticket(
    description: str,
    raised_by: str = "student",
    building: str = "Unknown",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Triage an incoming support ticket description using LLM-based classification.

    Parameters
    ----------
    description : str
        The user-submitted issue or problem description.
    raised_by : str, default "student"
        The role of the requester (e.g. 'student', 'faculty', 'staff').
    building : str, default "Unknown"
        Campus location or facility where the issue occurs.
    context : dict | None, optional
        Additional metadata (e.g., historical incidents or device details).

    Returns
    -------
    dict
        A dictionary containing:
        - category (str): One of canonical Category values.
        - subcategory (str): Specific subcategory identifier.
        - priority (str): One of canonical Priority values.
        - impact_scope (str): One of canonical ImpactScope values.
        - triage_reason (str): Natural language explanation for triage decisions.
        - triage_confidence (float): Model confidence between 0.0 and 1.0.
        - needs_human_review (bool): Flag indicating if manual verification is required.
    """
    # NOTE: Initial stub returns safe defaults pending Member 2's Groq LLM implementation.
    # We do NOT simulate or claim actual LLM classification here.
    return {
        "category": Category.OTHER.value,
        "subcategory": "Pending Classification",
        "priority": Priority.MEDIUM.value,
        "impact_scope": ImpactScope.UNKNOWN.value,
        "triage_reason": "Classification pending: Groq LLM module not yet connected.",
        "triage_confidence": 0.0,
        "needs_human_review": True,
    }
