"""Unit tests for Member 3: Historical Ticket Retrieval and Duplicate Detection.

Tests cover:
- Exact contract signature and return fields
- TF-IDF and cosine similarity precision (Wi-Fi queries retrieve Wi-Fi incidents)
- Cross-category isolation (Python install does not retrieve database fixes)
- Threshold cutoff (Unrelated queries return empty list)
- Edge cases: empty description, whitespace, empty history, top_k limits
- Integrity: no fabricated resolutions when resolution is missing
- Duplicate detection helper
- Resolution suggestion helper
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

from src.retrieval import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
    SIMILAR_TICKET_FIELDS,
    detect_duplicate_tickets,
    find_similar_tickets,
    get_suggested_resolution,
)
from src.storage import load_historical_tickets


@pytest.fixture
def historical_tickets() -> list[dict]:
    """Load the canonical 50 synthetic historical tickets."""
    tickets = load_historical_tickets()
    assert len(tickets) >= 30, "Historical tickets dataset must contain at least 30 records"
    return tickets


class TestRetrievalContractAndSignatures:
    """Verify strict contract compliance."""

    def test_signature_preservation(self):
        sig = inspect.signature(find_similar_tickets)
        params = list(sig.parameters.keys())
        assert params == ["description", "historical_tickets", "top_k"]
        assert sig.parameters["top_k"].default == 3

    def test_return_fields_contract(self, historical_tickets):
        results = find_similar_tickets(
            description="Exam hall Wi-Fi connection failed",
            historical_tickets=historical_tickets,
            top_k=3,
        )
        assert len(results) > 0
        for item in results:
            assert set(item.keys()) == set(SIMILAR_TICKET_FIELDS)
            assert isinstance(item["ticket_id"], str)
            assert isinstance(item["description"], str)
            assert isinstance(item["category"], str)
            assert isinstance(item["priority"], str)
            assert isinstance(item["resolution"], str)
            assert isinstance(item["similarity"], float)
            assert 0.0 <= item["similarity"] <= 1.0


class TestSimilaritySearchScenarios:
    """Verify semantic matching and isolation across typical IT scenarios."""

    def test_wifi_retrieval(self, historical_tickets):
        """Wi-Fi outage query must retrieve relevant Wi-Fi historical incidents."""
        results = find_similar_tickets(
            description="Exam hall Wi-Fi outage affecting 120 students",
            historical_tickets=historical_tickets,
            top_k=3,
        )
        assert len(results) > 0
        top = results[0]
        assert top["category"] == "Network"
        assert "wi-fi" in top["description"].lower() or "wifi" in top["description"].lower() or "exam" in top["description"].lower()
        assert top["similarity"] >= DEFAULT_SIMILARITY_THRESHOLD
        assert len(top["resolution"]) > 0

    def test_python_install_no_database_mismatch(self, historical_tickets):
        """Python installation query must not retrieve database connection fixes."""
        results = find_similar_tickets(
            description="Routine request to install Python 3.11 scientific environment",
            historical_tickets=historical_tickets,
            top_k=3,
        )
        assert len(results) > 0
        top = results[0]
        assert top["category"] == "Software"
        assert "python" in top["description"].lower()
        # Verify database resolutions are not retrieved
        for match in results:
            assert match["category"] != "Database"
            assert "postgresql" not in match["resolution"].lower()
            assert "mysql" not in match["resolution"].lower()

    def test_database_connection_pool_retrieval(self, historical_tickets):
        """Database connection error retrieves database pool incident."""
        results = find_similar_tickets(
            description="PostgreSQL core database rejecting connections with pool exhaustion error",
            historical_tickets=historical_tickets,
            top_k=3,
        )
        assert len(results) > 0
        top = results[0]
        assert top["category"] == "Database"
        assert "postgresql" in top["description"].lower()
        assert "max_connections" in top["resolution"].lower() or "connection" in top["resolution"].lower()

    def test_unrelated_query_returns_empty_below_threshold(self, historical_tickets):
        """Queries unrelated to IT helpdesk incidents must return empty list."""
        unrelated_queries = [
            "What is the cafeteria lunch menu today?",
            "What is the best recipe for chocolate chip banana bread?",
            "xyzabc12345 nonexistingword foobarbaz",
            "What is the weather forecast for tomorrow morning?",
        ]
        for query in unrelated_queries:
            results = find_similar_tickets(
                description=query,
                historical_tickets=historical_tickets,
                top_k=3,
            )
            assert results == [], f"Expected [] for unrelated query: '{query}', got {results}"


class TestEdgeCasesAndSafety:
    """Verify robustness on edge conditions and anti-hallucination guardrails."""

    def test_empty_and_whitespace_descriptions(self, historical_tickets):
        assert find_similar_tickets("", historical_tickets) == []
        assert find_similar_tickets("    ", historical_tickets) == []
        assert find_similar_tickets("\n\t", historical_tickets) == []

    def test_empty_historical_corpus(self):
        assert find_similar_tickets("Wi-Fi is down", []) == []

    def test_top_k_limits(self, historical_tickets):
        res_1 = find_similar_tickets("Wi-Fi network connection issue", historical_tickets, top_k=1)
        assert len(res_1) <= 1

        res_2 = find_similar_tickets("Wi-Fi network connection issue", historical_tickets, top_k=2)
        assert len(res_2) <= 2

        res_0 = find_similar_tickets("Wi-Fi network connection issue", historical_tickets, top_k=0)
        assert res_0 == []

        res_neg = find_similar_tickets("Wi-Fi network connection issue", historical_tickets, top_k=-5)
        assert res_neg == []

    def test_ranking_order_descending(self, historical_tickets):
        results = find_similar_tickets(
            description="Wi-Fi network disconnection in Science Hall",
            historical_tickets=historical_tickets,
            top_k=5,
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_missing_resolution_does_not_fabricate_fix(self):
        """If a matched ticket has no resolution, do not fabricate or alter it."""
        mock_corpus = [
            {
                "ticket_id": "MOCK-001",
                "description": "Network switch in room 204 burned out",
                "category": "Network",
                "priority": "Critical",
                "resolution": "",  # Empty resolution
            }
        ]
        results = find_similar_tickets(
            description="Network switch in room 204 burned out",
            historical_tickets=mock_corpus,
            top_k=1,
        )
        assert len(results) == 1
        assert results[0]["resolution"] == "", "Resolution must remain empty and not be invented"


class TestHelperFunctions:
    """Verify suggested resolution and duplicate detection helpers."""

    def test_get_suggested_resolution(self, historical_tickets):
        similar = find_similar_tickets(
            description="Exam hall Wi-Fi outage affecting 120 students",
            historical_tickets=historical_tickets,
            top_k=3,
        )
        suggestion = get_suggested_resolution(similar)
        assert len(suggestion) > 0
        assert "cisco" in suggestion.lower() or "access point" in suggestion.lower() or "dhcp" in suggestion.lower()

        # No similar tickets returns empty string
        assert get_suggested_resolution([]) == ""

    def test_detect_duplicate_tickets(self, historical_tickets):
        # A near identical complaint in the same building
        duplicates = detect_duplicate_tickets(
            description="Exam hall Wi-Fi outage affecting 120 students during midterm examination",
            historical_tickets=historical_tickets,
            building="Science Hall",
            category="Network",
            duplicate_threshold=0.55,
        )
        assert len(duplicates) > 0
        top_dup = duplicates[0]
        assert top_dup["ticket_id"] == "TICK-HIST-015"
        assert top_dup["effective_duplicate_score"] >= 0.55
        assert top_dup["same_building"] is True
        assert top_dup["same_category"] is True
        assert "duplicate_reason" in top_dup
