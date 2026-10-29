"""Lightweight file-based storage layer for tickets, technicians, and historical incidents.

Owner: Member 1 (Lead / Integration)
Persistence:
- Historical incidents: data/historical_tickets.csv
- Technicians: data/technicians.json
- Active/tracked tickets: data/tickets.json

All paths are resolved relative to the project root directory, guaranteeing
consistent behavior regardless of the current working directory in the terminal.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.schemas import normalize_ticket

logger = logging.getLogger(__name__)

# Base directory for the helpdesk-triage project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_TICKETS_PATH = DATA_DIR / "historical_tickets.csv"
TECHNICIANS_PATH = DATA_DIR / "technicians.json"
TICKETS_PATH = DATA_DIR / "tickets.json"


def _ensure_data_dir() -> Path:
    """Ensure data/ directory exists and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_historical_tickets() -> List[Dict[str, Any]]:
    """Load historical incident records from data/historical_tickets.csv.

    Returns
    -------
    list[dict]
        List of historical ticket dictionaries. Returns an empty list if
        the file is missing, empty, or unreadable.
    """
    if not HISTORICAL_TICKETS_PATH.exists() or HISTORICAL_TICKETS_PATH.stat().st_size == 0:
        return []

    tickets: List[Dict[str, Any]] = []
    try:
        with open(HISTORICAL_TICKETS_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row: Dict[str, Any] = {
                    (k.strip() if k else ""): (v.strip() if v else "")
                    for k, v in row.items()
                }
                # Parse numeric time_to_resolve_hrs if present
                if "time_to_resolve_hrs" in cleaned_row:
                    try:
                        cleaned_row["time_to_resolve_hrs"] = float(cleaned_row["time_to_resolve_hrs"])
                    except (ValueError, TypeError):
                        cleaned_row["time_to_resolve_hrs"] = 0.0
                tickets.append(cleaned_row)
    except Exception as e:
        logger.warning("Failed to load historical tickets from %s: %s", HISTORICAL_TICKETS_PATH, e)
        return []

    return tickets


def load_technicians() -> List[Dict[str, Any]]:
    """Load technician profiles from data/technicians.json.

    Returns
    -------
    list[dict]
        List of technician dictionaries. Returns an empty list if the file
        is missing, empty, or contains invalid JSON.
    """
    if not TECHNICIANS_PATH.exists() or TECHNICIANS_PATH.stat().st_size == 0:
        return []

    try:
        with open(TECHNICIANS_PATH, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("Technicians JSON root is not a list in %s", TECHNICIANS_PATH)
            return []
    except Exception as e:
        logger.warning("Failed to load technicians from %s: %s", TECHNICIANS_PATH, e)
        return []


def load_tickets() -> List[Dict[str, Any]]:
    """Load active tickets from data/tickets.json.

    Returns
    -------
    list[dict]
        List of normalized ticket dictionaries. Returns an empty list if
        the file does not exist or is empty.
    """
    if not TICKETS_PATH.exists() or TICKETS_PATH.stat().st_size == 0:
        return []

    try:
        with open(TICKETS_PATH, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [normalize_ticket(t) for t in data if isinstance(t, dict)]
            logger.warning("Tickets JSON root is not a list in %s", TICKETS_PATH)
            return []
    except Exception as e:
        logger.warning("Failed to load tickets from %s: %s", TICKETS_PATH, e)
        return []


def save_tickets(tickets: List[Dict[str, Any]]) -> None:
    """Save the list of active tickets to data/tickets.json atomically.

    Parameters
    ----------
    tickets : list[dict]
        List of ticket dictionaries to persist.
    """
    _ensure_data_dir()
    normalized = [normalize_ticket(t) for t in tickets if isinstance(t, dict)]
    temp_path = TICKETS_PATH.with_suffix(".tmp")
    with open(temp_path, mode="w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    temp_path.replace(TICKETS_PATH)


def create_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Create and persist a new ticket with safe default fields and an auto-generated ID.

    Parameters
    ----------
    ticket : dict
        Partial or complete ticket dictionary.

    Returns
    -------
    dict
        The complete, normalized, and persisted ticket record.
    """
    existing_tickets = load_tickets()

    normalized = normalize_ticket(ticket)
    # Generate ID if missing or empty
    if not normalized.get("ticket_id"):
        # Format: TICK-0001, TICK-0002, or unique hash fallback
        next_index = len(existing_tickets) + 1
        generated_id = f"TICK-{next_index:04d}"
        # Ensure collision resistance
        existing_ids = {t.get("ticket_id") for t in existing_tickets}
        while generated_id in existing_ids:
            next_index += 1
            generated_id = f"TICK-{next_index:04d}-{uuid.uuid4().hex[:4].upper()}"
        normalized["ticket_id"] = generated_id

    existing_tickets.append(normalized)
    save_tickets(existing_tickets)
    return normalized


def update_ticket(ticket_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update fields on an existing ticket while preserving untouched fields.

    Parameters
    ----------
    ticket_id : str
        The unique ID of the ticket to update.
    updates : dict
        Dictionary of fields and new values to merge into the ticket.

    Returns
    -------
    dict | None
        The updated ticket record, or None if the ticket_id was not found.
    """
    existing_tickets = load_tickets()
    target_index: Optional[int] = None

    for i, t in enumerate(existing_tickets):
        if t.get("ticket_id") == ticket_id:
            target_index = i
            break

    if target_index is None:
        return None

    current = existing_tickets[target_index]
    # Merge updates, preserving original ticket_id unless explicitly updated
    for key, val in updates.items():
        if key != "ticket_id":
            current[key] = val

    updated_normalized = normalize_ticket(current)
    existing_tickets[target_index] = updated_normalized
    save_tickets(existing_tickets)
    return updated_normalized
