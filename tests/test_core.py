import numpy as np
import pytest

from ginfinity_sw import (Alignment, EValueParameters, ScoringParameters,
                           align, align_multiple, align_scores,
                           collapse_alignments, pair_evalue, rank,
                           similarity_matrix)
from ginfinity_sw.formatting import format_alignment, format_alignment_set


PARAMETERS = ScoringParameters(
    mu=0.2, gamma=5.0, gap_open=6.0, gap_extend=1.0,
    score_min=-4.0, score_max=8.0)


def test_local_alignment_finds_the_only_shared_block():
    rng = np.random.default_rng(7)
    query = rng.standard_normal((20, 16))
    target = rng.standard_normal((20, 16))
    target[9:15] = query[7:13]
    result = align(query, target, params=PARAMETERS)
    assert result.score > 0
    assert result.query_span[0] >= 6
    assert result.query_span[1] <= 14
    assert result.target_span[0] >= 8
    assert result.target_span[1] <= 16
    assert result.match_count >= 6


def test_score_only_and_traceback_are_exactly_equal():
    rng = np.random.default_rng(11)
    query = rng.standard_normal((45, 24))
    target = rng.standard_normal((51, 24))
    score = align(query, target, params=PARAMETERS, traceback=False).score
    traced = align(query, target, params=PARAMETERS, traceback=True).score
    assert score == traced


def test_similarity_is_cosine_and_validates_dimensions():
    query = np.array([[3.0, 4.0]])
    target = np.array([[6.0, 8.0]])
    assert similarity_matrix(query, target)[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dimensions"):
        similarity_matrix(query, np.ones((2, 3)))


def test_cell_budget_is_checked_before_dynamic_programming():
    with pytest.raises(ValueError, match="maximum"):
        align_scores(np.ones((11, 10)), PARAMETERS, max_cells=100)


def test_rank_is_score_descending_with_stable_identifier_ties():
    query = np.eye(4)
    ranking = rank(query, [("z", query), ("a", query)], params=PARAMETERS)
    assert [identifier for identifier, _ in ranking] == ["a", "z"]


def test_multiple_hsps_are_collapsed_with_total_max_and_pair_evalue():
    query = np.eye(16)[:6]
    target = np.vstack([query[:3], np.eye(16)[6:16], query[3:]])
    result = align_multiple(query, target, params=PARAMETERS)

    assert result.alignment_count == 2
    assert [alignment.match_count for alignment in result.alignments] == [3, 3]
    assert result.total_score == pytest.approx(24.0)
    assert result.max_score == pytest.approx(12.0)
    assert result.evalue == pytest.approx(6 * 16 * np.exp(-24.0))
    assert result.to_dict()["alignment_count"] == 2
    assert result.to_dict()["total_score"] == pytest.approx(24.0)


def test_collapse_alignments_sorts_hsps_and_uses_aggregate_score():
    weaker = Alignment(3.0, (4, 6), (5, 7), ((4, 5), (5, 6)), 6)
    stronger = Alignment(5.0, (0, 2), (1, 3), ((0, 1), (1, 2)), 6)
    parameters = EValueParameters(lambda_=0.5, k=2.0)
    result = collapse_alignments(
        [stronger, weaker],
        query_length=10,
        target_length=11,
        evalue_parameters=parameters,
    )

    assert result.alignments[0] is stronger
    assert result.total_score == pytest.approx(8.0)
    assert result.max_score == pytest.approx(5.0)
    assert result.evalue == pytest.approx(2 * 110 * np.exp(-4.0))
    assert pair_evalue(8.0, 10, 11, parameters=parameters) == pytest.approx(
        result.evalue)


def test_alignment_set_formatter_keeps_pair_summary_and_hsps_together():
    query = np.eye(16)[:6]
    target = np.vstack([query[:3], np.eye(16)[6:16], query[3:]])
    result = align_multiple(query, target, params=PARAMETERS)
    rendered = format_alignment_set(
        result, "ACGUAC", "......", "ACG" + "A" * 10 + "UAC", "." * 16)

    assert "Total score = 24" in rendered
    assert "Max score = 12" in rendered
    assert "E-value =" in rendered
    assert "HSP 1" in rendered
    assert "HSP 2" in rendered


def test_alignment_formatter_explains_statistics_and_conserved_pairs():
    query = np.eye(4)
    result = align(query, query, params=PARAMETERS)
    rendered = format_alignment(result, "ACGU", "(())", "ACGU", "(())")
    assert "base identity = 100.0%" in rendered
    assert "structure identity = 100.0%" in rendered
    assert "conserved pairs = 2" in rendered
    assert "<<>>" in rendered


@pytest.mark.parametrize("changes", [
    {"sigma": 0}, {"gamma": 0}, {"score_min": 2, "score_max": 1},
    {"gap_open": -1},
])
def test_invalid_scoring_parameters_are_rejected(changes):
    values = {"mu": 0.2, **changes}
    with pytest.raises(ValueError):
        ScoringParameters(**values)
