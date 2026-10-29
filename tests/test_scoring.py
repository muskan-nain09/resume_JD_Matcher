import pytest

from matcher import scoring
from matcher.scoring import compute_score, keyword_coverage, score_band


def test_weights_sum_to_one():
    assert scoring.SIMILARITY_WEIGHT + scoring.KEYWORD_WEIGHT == pytest.approx(1.0)


def test_formula_combines_inputs():
    # 0.6 * 0.8 + 0.4 * (3/4) = 0.48 + 0.30 = 0.78
    result = compute_score(similarity=0.8, matched=3, total=4)
    assert result.similarity == pytest.approx(0.8)
    assert result.keyword_coverage == pytest.approx(0.75)
    assert result.final_score == 78


@pytest.mark.parametrize("similarity, matched, total", [
    (0.0, 0, 10), (1.0, 10, 10), (1.5, 12, 10), (-0.3, 0, 5), (0.5, 0, 0),
])
def test_score_stays_within_0_and_100(similarity, matched, total):
    score = compute_score(similarity, matched, total).final_score
    assert 0 <= score <= 100
    assert isinstance(score, int)


def test_extremes():
    assert compute_score(1.0, 5, 5).final_score == 100
    assert compute_score(0.0, 0, 5).final_score == 0


def test_no_jd_keywords_falls_back_to_similarity():
    result = compute_score(similarity=0.65, matched=0, total=0)
    assert result.keyword_coverage is None
    assert result.final_score == 65


def test_keyword_coverage():
    assert keyword_coverage(2, 8) == pytest.approx(0.25)
    assert keyword_coverage(0, 0) is None


@pytest.mark.parametrize("score, band", [
    (0, "red"), (49, "red"), (50, "amber"), (74, "amber"), (75, "green"), (100, "green"),
])
def test_score_bands(score, band):
    assert score_band(score) == band
