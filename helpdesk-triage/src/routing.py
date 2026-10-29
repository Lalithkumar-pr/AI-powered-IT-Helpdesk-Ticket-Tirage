"""Technician routing and workload assignment module.

Owner: Member 4 (Technician Routing Specialist)
Responsibilities:
- Match ticket category, priority, and skills to available technicians
- Consider technician workload and availability balancing
- Provide human-readable assignment_reason for transparency
- Provide a graceful fallback to unassigned if no technician matches or all are overloaded

Important:
- Public function signature must NOT be changed.
- Return dictionary must contain: assigned_technician_id, assigned_technician_name, assigned_team, assignment_reason.
- Never fabricate a technician who does not exist in technicians.json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ROUTING_RETURN_KEYS = (
    "assigned_technician_id",
    "assigned_technician_name",
    "assigned_team",
    "assignment_reason",
)


def assign_technician(
    category: str,
    priority: str,
    technicians: List[Dict[str, Any]],
    active_tickets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assign an appropriate technician based on category, priority, and workload.

    Parameters
    ----------
    category : str
        Canonical ticket category.
    priority : str
        Canonical ticket priority.
    technicians : list[dict]
        List of technician profiles loaded from data/technicians.json.
    active_tickets : list[dict] | None, optional
        Currently open/in-progress tickets used to calculate live workload.

    Returns
    -------
    dict
        Assignment results containing:
        - assigned_technician_id (str): ID of assigned technician, or "" if unassigned.
        - assigned_technician_name (str): Name of technician, or "Unassigned".
        - assigned_team (str): Operating team/unit of assigned technician, or "".
        - assignment_reason (str): Explanation for the routing decision or fallback notice.
    """
    # NOTE: Initial stub returns a safe unassigned fallback pending Member 4's routing implementation.
    # No artificial technician is fabricated here.
    return {
        "assigned_technician_id": "",
        "assigned_technician_name": "Unassigned",
        "assigned_team": "",
        "assignment_reason": "Routing pending: Technician assignment module not yet implemented.",
    }
