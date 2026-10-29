"""Historical ticket retrieval and similarity search module.

Owner: Member 3 (Historical Retrieval Specialist)
Responsibilities:
- Build TF-IDF vectorizer and cosine similarity index over historical tickets
- Match incoming issue description against historical incidents
- Return top-k matching historical incidents with real resolutions and similarity scores
- Duplicate ticket detection and suggested resolution extraction

Important:
- Public function signature for find_similar_tickets must NOT be changed.
- Each returned dictionary must include: ticket_id, description, category, priority, resolution, similarity.
- Never invent or fabricate historical resolutions; only return verified resolutions from historical tickets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Minimum cosine similarity threshold to consider a historical incident relevant
# Cosine similarity is a ranking score, not a proof of identical root cause.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.20

# Threshold for classifying two tickets as potential duplicates
DUPLICATE_SIMILARITY_THRESHOLD: float = 0.55

# Expected fields for each item in the similar_tickets list (Canonical Contract)
SIMILAR_TICKET_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "description",
    "category",
    "priority",
    "resolution",
    "similarity",
)


def find_similar_tickets(
    description: str,
    historical_tickets: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve top-k similar historical tickets using text similarity (TF-IDF + Cosine Similarity).

    Parameters
    ----------
    description : str
        The problem description of the new ticket to find matches for.
    historical_tickets : list[dict]
        List of historical ticket dictionaries loaded from data/historical_tickets.csv.
    top_k : int, default 3
        Maximum number of similar tickets to retrieve.

    Returns
    -------
    list[dict]
        List of matching ticket summaries sorted descending by similarity, each containing:
        - ticket_id (str): Historical ticket identifier.
        - description (str): Historical problem summary.
        - category (str): Historical category.
        - priority (str): Historical priority.
        - resolution (str): Historical verified resolution steps.
        - similarity (float): Cosine similarity score between 0.0 and 1.0.
    """
    if not description or not description.strip():
        return []

    if not historical_tickets or top_k <= 0:
        return []

    clean_query = description.strip()

    # Extract corpus of descriptions from historical tickets
    corpus: List[str] = []
    valid_tickets: List[Dict[str, Any]] = []

    for ticket in historical_tickets:
        ticket_desc = str(ticket.get("description", "")).strip()
        if ticket_desc:
            corpus.append(ticket_desc)
            valid_tickets.append(ticket)

    if not corpus:
        return []

    try:
        # Build TF-IDF representation with unigram + bigram support and sublinear TF scaling
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([clean_query])
    except (ValueError, TypeError) as e:
        logger.debug("TF-IDF vectorizer failed (empty vocabulary or stop words only): %s", e)
        return []

    # Calculate cosine similarity between query and all historical ticket descriptions
    similarity_matrix = cosine_similarity(query_vector, tfidf_matrix)
    scores = similarity_matrix[0]

    # Collect matches meeting the similarity threshold
    matches: List[Dict[str, Any]] = []
    for idx, score in enumerate(scores):
        float_score = float(score)
        if float_score >= DEFAULT_SIMILARITY_THRESHOLD:
            original = valid_tickets[idx]
            match_record: Dict[str, Any] = {
                "ticket_id": str(original.get("ticket_id", "")),
                "description": str(original.get("description", "")),
                "category": str(original.get("category", "")),
                "priority": str(original.get("priority", "")),
                "resolution": str(original.get("resolution", "")),
                "similarity": round(float_score, 4),
            }
            matches.append(match_record)

    # Sort matches descending by similarity score
    matches.sort(key=lambda m: m["similarity"], reverse=True)

    # Return top_k matches
    return matches[:top_k]


def detect_duplicate_tickets(
    description: str,
    historical_tickets: List[Dict[str, Any]],
    building: Optional[str] = None,
    category: Optional[str] = None,
    duplicate_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Identify potential duplicate tickets based on high text similarity and matching metadata.

    Parameters
    ----------
    description : str
        The issue description of the incoming ticket.
    historical_tickets : list[dict]
        Active or historical tickets to check against.
    building : str, optional
        Campus building of the incoming ticket. If provided, incidents in the same
        building are boosted in duplicate scoring.
    category : str, optional
        Category of the incoming ticket.
    duplicate_threshold : float, default 0.55
        Higher threshold required to classify a match as a suspected duplicate.
    top_k : int, default 5
        Maximum duplicate candidates to return.

    Returns
    -------
    list[dict]
        List of candidate duplicate tickets with similarity score, reason, and metadata.
    """
    candidates = find_similar_tickets(
        description=description,
        historical_tickets=historical_tickets,
        top_k=len(historical_tickets),
    )

    duplicate_matches: List[Dict[str, Any]] = []
    for candidate in candidates:
        sim = candidate.get("similarity", 0.0)
        # Find matching full ticket from historical tickets to inspect building/category
        ticket_id = candidate.get("ticket_id", "")
        orig_ticket = next((t for t in historical_tickets if t.get("ticket_id") == ticket_id), {})

        same_building = (
            bool(building)
            and building.lower() != "unknown"
            and orig_ticket.get("building", "").lower() == building.lower()
        )
        same_category = (
            bool(category)
            and orig_ticket.get("category", "").lower() == category.lower()
        )

        effective_score = sim
        reasons: List[str] = [f"Text similarity score {sim:.2f}"]

        if same_building:
            effective_score += 0.05
            reasons.append(f"Same location ({building})")
        if same_category:
            reasons.append(f"Same category ({category})")

        if effective_score >= duplicate_threshold:
            duplicate_record = dict(candidate)
            duplicate_record["effective_duplicate_score"] = round(effective_score, 4)
            duplicate_record["duplicate_reason"] = "; ".join(reasons)
            duplicate_record["same_building"] = same_building
            duplicate_record["same_category"] = same_category
            duplicate_matches.append(duplicate_record)

    duplicate_matches.sort(key=lambda d: d["effective_duplicate_score"], reverse=True)
    return duplicate_matches[:top_k]


def get_suggested_resolution(
    similar_tickets: List[Dict[str, Any]],
    min_confidence: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> str:
    """Extract recommended resolution from top similar ticket if supported by evidence.

    Parameters
    ----------
    similar_tickets : list[dict]
        Output list from find_similar_tickets.
    min_confidence : float, default 0.20
        Minimum similarity score required to suggest a past resolution.

    Returns
    -------
    str
        Verified resolution text if available and confidence threshold met;
        otherwise an empty string (no fabricated resolution).
    """
    if not similar_tickets:
        return ""

    top_match = similar_tickets[0]
    score = top_match.get("similarity", 0.0)
    resolution = str(top_match.get("resolution", "")).strip()

    if score >= min_confidence and resolution:
        return resolution

    return ""

