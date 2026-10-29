"""Streamlit UI for AI-Powered IT Helpdesk Ticket Triage.

Owner: Member 5 (UI Specialist)
Responsibilities:
- Build an interactive, clean Streamlit dashboard
- Provide ticket submission form (description, requester type, building)
- Display live ticket list and historical reference data
- Interface with src.triage, src.retrieval, src.routing, and src.storage
- Clearly indicate module implementation status without fabricating AI outputs
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the project root directory is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.retrieval import find_similar_tickets
from src.routing import assign_technician
from src.schemas import (
    CATEGORIES,
    IMPACT_SCOPES,
    PRIORITIES,
    Category,
    ImpactScope,
    Priority,
)
from src.storage import (
    create_ticket,
    load_historical_tickets,
    load_technicians,
    load_tickets,
)
from src.triage import triage_ticket

# Configure page layout and visual identity
st.set_page_config(
    page_title="AI-Powered IT Helpdesk Ticket Triage",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.title("🎫 AI-Powered IT Helpdesk Ticket Triage")
st.caption(
    "Automated incident categorization, urgency scoring, historical resolution retrieval, "
    "and technician routing for campus IT support."
)

# Implementation status banner
st.info(
    "ℹ️ **Project Skeleton Status**: Core contracts, shared schemas, and persistence layer are active. "
    "AI Triage (Member 2), Historical Retrieval (Member 3), and Technician Routing (Member 4) "
    "are currently in development by their respective module owners. New tickets are safely persisted "
    "to local storage (`data/tickets.json`).",
    icon="🛠️",
)

# Sidebar with system status and roster preview
with st.sidebar:
    st.header("⚙️ System Overview")
    historical_tickets = load_historical_tickets()
    technicians = load_technicians()
    active_tickets = load_tickets()

    st.metric("Active Tickets", len(active_tickets))
    st.metric("Historical Incidents", len(historical_tickets))
    st.metric("Registered Technicians", len(technicians))

    st.divider()
    st.subheader("👥 Module Ownership")
    st.markdown(
        """
        - **Member 1**: Team Lead / Storage & Integration
        - **Member 2**: Groq LLM Classification (`src/triage.py`)
        - **Member 3**: TF-IDF Retrieval (`src/retrieval.py`)
        - **Member 4**: Technician Routing (`src/routing.py`)
        - **Member 5**: Streamlit UI (`app.py`)
        """
    )

    st.divider()
    st.subheader("📋 Canonical Enums")
    with st.expander("View Categories"):
        for c in CATEGORIES:
            st.write(f"- `{c}`")
    with st.expander("View Priorities"):
        for p in PRIORITIES:
            st.write(f"- `{p}`")
    with st.expander("View Impact Scopes"):
        for s in IMPACT_SCOPES:
            st.write(f"- `{s}`")

# Main content: Left column (Submission Form) & Right column (Ticket Queue & History)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Submit New Support Ticket")
    with st.form("new_ticket_form", clear_on_submit=True):
        description = st.text_area(
            "Issue Description *",
            placeholder="e.g., The Wi-Fi on the 3rd floor of the Library keeps disconnecting when multiple laptops connect.",
            height=130,
            help="Provide as much detail as possible about the symptoms, equipment, and location.",
        )

        col_req, col_bldg = st.columns(2)
        with col_req:
            raised_by = st.selectbox(
                "Requester Role",
                options=["student", "faculty", "staff", "other"],
                index=0,
                help="Role of the person experiencing the incident.",
            )
        with col_bldg:
            building = st.selectbox(
                "Campus Building",
                options=[
                    "Library",
                    "Science Hall",
                    "Engineering Complex",
                    "Administration",
                    "Student Union",
                    "Arts Building",
                    "Finance Center",
                    "Dormitory Quad A",
                    "Health Center",
                    "Other / Remote",
                ],
                index=0,
            )

        submit_btn = st.form_submit_button("Submit Ticket", use_container_width=True)

        if submit_btn:
            if not description.strip():
                st.error("Please enter an issue description before submitting.")
            else:
                # 1. Run Triage module (stub or live)
                triage_res = triage_ticket(
                    description=description,
                    raised_by=raised_by,
                    building=building,
                )

                # 2. Run Retrieval module (stub or live)
                similar = find_similar_tickets(
                    description=description,
                    historical_tickets=historical_tickets,
                    top_k=3,
                )

                # 3. Run Routing module (stub or live)
                routing_res = assign_technician(
                    category=triage_res.get("category", Category.OTHER.value),
                    priority=triage_res.get("priority", Priority.MEDIUM.value),
                    technicians=technicians,
                    active_tickets=active_tickets,
                )

                # Assemble complete ticket dictionary matching canonical schema
                new_ticket_payload = {
                    "raised_by": raised_by,
                    "building": building,
                    "description": description.strip(),
                    "category": triage_res.get("category", Category.OTHER.value),
                    "subcategory": triage_res.get("subcategory", "General"),
                    "priority": triage_res.get("priority", Priority.MEDIUM.value),
                    "impact_scope": triage_res.get("impact_scope", ImpactScope.UNKNOWN.value),
                    "triage_reason": triage_res.get("triage_reason", ""),
                    "triage_confidence": triage_res.get("triage_confidence", 0.0),
                    "needs_human_review": triage_res.get("needs_human_review", False),
                    "similar_tickets": similar,
                    "suggested_resolution": "",
                    "assigned_technician_id": routing_res.get("assigned_technician_id", ""),
                    "assigned_technician_name": routing_res.get("assigned_technician_name", "Unassigned"),
                    "assigned_team": routing_res.get("assigned_team", ""),
                    "assignment_reason": routing_res.get("assignment_reason", ""),
                    "status": "open",
                    "resolution": "",
                    "time_to_resolve_hrs": 0.0,
                }

                created = create_ticket(new_ticket_payload)
                st.success(
                    f"✅ Ticket **{created['ticket_id']}** created and saved to storage!",
                    icon="✅",
                )
                st.rerun()

    # Informational box explaining stub behavior
    st.caption(
        "💡 **Developer Note**: In this initial skeleton, tickets are created with safe fallback "
        "triage and routing fields. As teammates complete `src/triage.py`, `src/retrieval.py`, and "
        "`src/routing.py`, intelligent AI classification and automated routing will take effect here."
    )

with col_right:
    st.subheader("📋 Active Ticket Queue")
    tickets = load_tickets()

    if not tickets:
        st.info("No active tickets found in storage (`data/tickets.json`). Submit a ticket to see it appear here.")
    else:
        # Display as clean interactive table
        display_data = []
        for t in tickets:
            display_data.append({
                "ID": t.get("ticket_id"),
                "Created": t.get("created_at", "")[:16].replace("T", " "),
                "Requester": t.get("raised_by"),
                "Building": t.get("building"),
                "Description": t.get("description", "")[:50] + ("..." if len(t.get("description", "")) > 50 else ""),
                "Category": t.get("category"),
                "Priority": t.get("priority"),
                "Assigned To": t.get("assigned_technician_name"),
                "Status": t.get("status"),
                "Review Req.": "⚠️ Yes" if t.get("needs_human_review") else "No",
            })
        df_tickets = pd.DataFrame(display_data)
        st.dataframe(df_tickets, use_container_width=True, hide_index=True)

        # Allow inspecting latest ticket details
        with st.expander("🔍 Inspect Latest Ticket Details"):
            latest = tickets[-1]
            st.json(latest)

    st.divider()

    tab1, tab2 = st.tabs(["📚 Historical Knowledge Base", "👷 Technician Roster"])

    with tab1:
        if historical_tickets:
            df_hist = pd.DataFrame(historical_tickets)
            st.dataframe(
                df_hist[["ticket_id", "category", "priority", "description", "resolution"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No historical tickets found in `data/historical_tickets.csv`.")

    with tab2:
        if technicians:
            df_tech = pd.DataFrame(technicians)
            # Format skills list for table display
            df_tech_display = df_tech.copy()
            if "skills" in df_tech_display.columns:
                df_tech_display["skills"] = df_tech_display["skills"].apply(
                    lambda s: ", ".join(s) if isinstance(s, list) else str(s)
                )
            st.dataframe(
                df_tech_display[["technician_id", "name", "team", "skills", "active_tickets_count", "status"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No technicians found in `data/technicians.json`.")
