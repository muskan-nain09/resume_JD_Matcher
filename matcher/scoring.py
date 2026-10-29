"""Combine semantic similarity and keyword coverage into one 0-100 score.

Why two signals?
- Semantic similarity rewards a resume that is *about* the right things even
  when the wording differs, but it can't tell whether "Kubernetes" is there.
- Keyword coverage is what many ATS filters actually check, but on its own it
  rewards keyword stuffing.
Weighting them 60/40 favours overall relevance while still penalizing
missing hard requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

SIMILARITY_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# Score bands used by the UI for color coding.
GREEN_THRESHOLD = 75
AMBER_THRESHOLD = 50


@dataclass
class ScoreBreakdown:
    similarity: float               # 0..1
    keyword_coverage: float | None  # 0..1, or None if the JD had no keywords
    final_score: int                # 0..100


def keyword_coverage(matched: int, total: int) -> float | None:
    """Fraction of JD keywords found in the resume.

    Returns None when the JD has no keywords at all. Returning 0 would punish
    the resume for something wrong with the JD, and returning 1 would hand out
    free points, so "no data" is more honest.
    """
    if total <= 0:
        return None
    return min(max(matched / total, 0.0), 1.0)


def compute_score(similarity: float, matched: int, total: int) -> ScoreBreakdown:
    """Weighted final score. If coverage is unavailable, similarity alone is
    used so the score stays on the same 0-100 scale."""
    similarity = min(max(similarity, 0.0), 1.0)
    coverage = keyword_coverage(matched, total)
    if coverage is None:
        combined = similarity
    else:
        combined = SIMILARITY_WEIGHT * similarity + KEYWORD_WEIGHT * coverage
    return ScoreBreakdown(
        similarity=similarity,
        keyword_coverage=coverage,
        final_score=round(combined * 100),
    )


def score_band(score: int) -> str:
    """'green', 'amber' or 'red' for a 0-100 score."""
    if score >= GREEN_THRESHOLD:
        return "green"
    if score >= AMBER_THRESHOLD:
        return "amber"
    return "red"
