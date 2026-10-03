"""A small, replaceable subject router for choosing book databases."""

from __future__ import annotations

import re


# This intentionally starts with transparent rules. A later LLM router can use
# the same ``subject`` output without changing the storage or retrieval layer.
SUBJECT_KEYWORDS = {
    "polity": {
        "constitution", "parliament", "president", "federal", "fundamental right",
        "article", "supreme court", "directive principle", "governor", "election",
    },
    "economics": {
        "economy", "economic", "gdp", "inflation", "fiscal", "monetary", "banking",
        "budget", "poverty", "employment", "agriculture", "trade", "rupee",
    },
    "history": {
        "history", "freedom struggle", "british", "mughal", "harappan", "revolt",
        "movement", "independence", "dynasty", "colonial",
    },
    "geography": {
        "geography", "monsoon", "climate", "river", "soil", "plate tectonic",
        "earthquake", "ocean", "mountain", "weather", "cyclone",
    },
    "environment": {
        "environment", "biodiversity", "ecosystem", "climate change", "pollution",
        "forest", "wildlife", "conservation", "species",
    },
}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def route_subject(question: str, available_subjects: list[str]) -> str | None:
    """Return the best available broad subject, or None when unsure.

    This performs only broad routing. The vector search inside the selected
    book database determines the actual chapter and section.
    """

    text = _normalise(question)
    scores: dict[str, int] = {}
    for subject in available_subjects:
        subject_key = _normalise(subject)
        keywords = SUBJECT_KEYWORDS.get(subject_key, set()) | {subject_key}
        scores[subject] = sum(1 for keyword in keywords if keyword in text)

    if not scores or max(scores.values()) == 0:
        return None
    highest_score = max(scores.values())
    winners = [subject for subject, score in scores.items() if score == highest_score]
    return winners[0] if len(winners) == 1 else None
