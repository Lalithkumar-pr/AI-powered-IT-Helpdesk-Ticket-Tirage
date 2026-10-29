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
- Never assign an unskilled technician simply because their workload is low.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

ROUTING_RETURN_KEYS = (
    "assigned_technician_id",
    "assigned_technician_name",
    "assigned_team",
    "assignment_reason",
)

# Canonical mapping of categories and sub-skills for robust matching
CATEGORY_SKILL_MAP: Dict[str, Set[str]] = {
    "network": {
        "network",
        "wireless",
        "wifi",
        "dns",
        "dhcp",
        "vpn",
        "lan",
        "wan",
        "router",
        "switch",
        "firewall",
        "connectivity",
    },
    "hardware": {
        "hardware",
        "desktop",
        "laptop",
        "monitor",
        "display",
        "keyboard",
        "mouse",
        "printer",
        "scanner",
        "pc",
        "peripherals",
        "cables",
        "docking station",
    },
    "software": {
        "software",
        "operating system",
        "os",
        "windows",
        "linux",
        "macos",
        "application",
        "internal application",
        "browser",
        "installation",
        "driver",
    },
    "database": {
        "database",
        "sql",
        "postgres",
        "postgresql",
        "mysql",
        "oracle",
        "db",
        "mongodb",
        "data query",
    },
    "account & access": {
        "account & access",
        "account",
        "access",
        "password",
        "login",
        "identity",
        "active directory",
        "ldap",
        "sso",
        "permissions",
        "mfa",
        "2fa",
    },
    "email & collaboration": {
        "email & collaboration",
        "email",
        "collaboration",
        "outlook",
        "teams",
        "exchange",
        "gmail",
        "calendar",
        "onedrive",
        "sharepoint",
        "slack",
    },
    "other": {
        "other",
        "general",
        "misc",
        "miscellaneous",
        "inquiry",
    },
}

ACTIVE_TICKET_STATUSES = {
    "open",
    "in_progress",
    "new",
    "triaged",
    "assigned",
}


def _matches_skill(category: str, tech_skills: List[str]) -> bool:
    """Check whether a technician's skills match the given category.
    
    Checks direct equality (case-insensitive) as well as known category synonyms.
    """
    cat_norm = category.strip().lower()
    if not cat_norm:
        return False

    cat_synonyms = CATEGORY_SKILL_MAP.get(cat_norm, {cat_norm})

    for skill in tech_skills:
        skill_norm = str(skill).strip().lower()
        if not skill_norm:
            continue
        # Direct match
        if skill_norm == cat_norm:
            return True
        # Synonym match
        if skill_norm in cat_synonyms:
            return True
        # If the skill itself is a category key that maps to this category
        if cat_norm in CATEGORY_SKILL_MAP.get(skill_norm, set()):
            return True

    return False


def _calculate_workload(
    tech: Dict[str, Any],
    active_tickets: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Calculate the current active ticket workload for a technician.
    
    If active_tickets is provided, counts tickets assigned to this technician
    that are in an active lifecycle state. Otherwise, falls back to the profile count.
    """
    tech_id = str(tech.get("technician_id", "")).strip()
    tech_name = str(tech.get("name", "")).strip()

    if active_tickets is not None and isinstance(active_tickets, list):
        count = 0
        for ticket in active_tickets:
            if not isinstance(ticket, dict):
                continue
            status = str(ticket.get("status", "")).strip().lower()
            if status not in ACTIVE_TICKET_STATUSES:
                continue

            assigned_id = str(ticket.get("assigned_technician_id", "")).strip()
            assigned_name = str(ticket.get("assigned_technician_name", "")).strip()

            if (tech_id and assigned_id == tech_id) or (tech_name and assigned_name == tech_name):
                count += 1
        return count

    # Fallback to static count in technician profile
    raw_count = tech.get("active_tickets_count", tech.get("active_tickets", 0))
    try:
        return max(0, int(raw_count))
    except (ValueError, TypeError):
        return 0


def _get_max_workload(tech: Dict[str, Any]) -> int:
    """Get the maximum active workload capacity for a technician."""
    raw_max = tech.get("max_workload", tech.get("max_active_tickets", 5))
    try:
        val = int(raw_max)
        return val if val > 0 else 5
    except (ValueError, TypeError):
        return 5


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
    # Guard against empty technician repository
    if not technicians:
        return {
            "assigned_technician_id": "",
            "assigned_technician_name": "Unassigned",
            "assigned_team": "",
            "assignment_reason": "No technician profiles available in system for routing.",
        }

    cat_clean = str(category).strip() if category else ""
    prio_clean = str(priority).strip() if priority else "Medium"

    if not cat_clean:
        return {
            "assigned_technician_id": "",
            "assigned_technician_name": "Unassigned",
            "assigned_team": "",
            "assignment_reason": "Missing ticket category. Manual assignment required.",
        }

    # Step 1: Filter technicians by skill match and availability status
    skilled_technicians: List[Dict[str, Any]] = []
    for tech in technicians:
        if not isinstance(tech, dict):
            continue

        # Check availability status (default to available if not specified)
        status = str(tech.get("status", "available")).strip().lower()
        if status in {"offline", "on_leave", "inactive", "unavailable"}:
            continue

        skills = tech.get("skills", [])
        if isinstance(skills, list) and _matches_skill(cat_clean, skills):
            workload = _calculate_workload(tech, active_tickets)
            max_cap = _get_max_workload(tech)
            skilled_technicians.append({
                "profile": tech,
                "workload": workload,
                "max_capacity": max_cap,
                "utilization": workload / max_cap if max_cap > 0 else 1.0,
            })

    # Guard: No technician has matching skill
    if not skilled_technicians:
        return {
            "assigned_technician_id": "",
            "assigned_technician_name": "Unassigned",
            "assigned_team": "",
            "assignment_reason": (
                f"No technician with matching skill set for '{cat_clean}' found. "
                "Manual assignment required; expertise must not be bypassed for workload."
            ),
        }

    # Step 2: Separate available technicians from those at maximum capacity
    available_technicians = [
        t for t in skilled_technicians if t["workload"] < t["max_capacity"]
    ]

    # Guard: All skilled technicians are currently overloaded
    if not available_technicians:
        overloaded_names = ", ".join(
            f"{t['profile'].get('name', 'Unknown')} ({t['workload']}/{t['max_capacity']})"
            for t in skilled_technicians
        )
        return {
            "assigned_technician_id": "",
            "assigned_technician_name": "Unassigned",
            "assigned_team": "",
            "assignment_reason": (
                f"All technician(s) skilled in '{cat_clean}' are currently at maximum capacity: "
                f"{overloaded_names}. Ticket queued for workload rebalancing or manual assignment."
            ),
        }

    # Step 3: Sort available technicians by:
    # 1. Lowest active workload
    # 2. Lowest utilization ratio
    # 3. Technician ID (for deterministic tie-breaking)
    available_technicians.sort(
        key=lambda x: (
            x["workload"],
            x["utilization"],
            str(x["profile"].get("technician_id", "")),
        )
    )

    best = available_technicians[0]
    tech_profile = best["profile"]
    tech_id = str(tech_profile.get("technician_id", "")).strip()
    tech_name = str(tech_profile.get("name", "Unassigned")).strip()
    tech_team = str(tech_profile.get("team", "")).strip()
    workload = best["workload"]
    max_cap = best["max_capacity"]

    # Step 4: Build human-readable assignment reason incorporating priority and workload
    prio_lower = prio_clean.lower()
    if prio_lower == "critical":
        reason = (
            f"Critical priority incident: Assigned to {tech_name} ({tech_team}) due to matching "
            f"'{cat_clean}' skill set and lowest active workload ({workload}/{max_cap} active tickets). "
            "Immediate response requested."
        )
    elif prio_lower == "high":
        reason = (
            f"High priority ticket: Routed to {tech_name} ({tech_team}) based on '{cat_clean}' "
            f"expertise with lowest current workload ({workload}/{max_cap} active tickets)."
        )
    else:
        reason = (
            f"Matched '{cat_clean}' skill set; routed to {tech_name} ({tech_team}) "
            f"who currently has the lowest active workload ({workload}/{max_cap} active tickets)."
        )

    return {
        "assigned_technician_id": tech_id,
        "assigned_technician_name": tech_name,
        "assigned_team": tech_team,
        "assignment_reason": reason,
    }

