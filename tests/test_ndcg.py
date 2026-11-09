"""Tests for the nDCG metrics added for the ICLR 2027 retrieval results."""

import math

import numpy as np
import pytest
from irmetrics.topk import ndcg as irmetrics_ndcg

from rtk.metrics import retrieval_graded_ndcg_at_k, retrieval_hit_at_k, retrieval_ndcg_at_k


def positive(y_true, retrieved, k):
    return retrieval_ndcg_at_k(y_true, retrieved, k)[0]


def test_perfect_ranking_scores_one():
    assert positive(1, [1, 1, 1, 0, 0], 3) == 1.0


def test_relevant_last_scores_less_than_relevant_first():
    assert positive(1, [1, 1, 0], 3) > positive(1, [0, 1, 1], 3)


def test_no_relevant_results_score_zero():
    assert positive(1, [0, 0, 0], 3) == 0.0


def test_at_k_one_matches_the_hit_rate():
    for retrieved in ([1, 0, 0], [0, 1, 1]):
        assert positive(1, retrieved, 1) == retrieval_hit_at_k(1, retrieved, 1)[0]


def test_negative_queries_fill_the_negative_slot():
    pos, neg = retrieval_ndcg_at_k(0, [0, 1, 0], 3)
    assert pos is None and neg is not None


def test_discounting_matches_the_closed_form():
    # one relevant case at rank 2, ideal is three relevant cases
    dcg = 1 / math.log2(3)
    idcg = sum(1 / math.log2(i + 2) for i in range(3))
    assert positive(1, [0, 1, 0], 3) == pytest.approx(dcg / idcg)


def test_agrees_with_irmetrics_when_the_top_k_is_saturated():
    """The two share a convention only when the ideal ranking is realised.

    ``irmetrics`` normalises against a single relevant item, because its ``rr`` and
    ``ap`` treat retrieval as having one correct answer. Here the corpus holds far more
    same-label cases than any reported k, so the ideal ranking fills the whole top-k and
    ``retrieval_ndcg_at_k`` normalises accordingly. The two therefore agree exactly when
    every retrieved case is relevant, and ours is the stricter of the two otherwise.
    """
    retrieved = [1, 1, 1]
    ours = positive(1, retrieved, 3)
    reference = float(np.mean(irmetrics_ndcg(np.array([1]), np.array([retrieved]), k=3)))
    assert ours == pytest.approx(reference)

    sparse = [0, 0, 1]
    assert positive(1, sparse, 3) < float(
        np.mean(irmetrics_ndcg(np.array([1]), np.array([sparse]), k=3))
    )


def test_graded_ndcg_rewards_sorted_gains():
    assert retrieval_graded_ndcg_at_k([0.9, 0.5, 0.1], 3) == 1.0
    assert retrieval_graded_ndcg_at_k([0.1, 0.5, 0.9], 3) < 1.0


def test_graded_ndcg_is_scale_invariant_without_an_ideal_gain():
    """Self-normalised graded nDCG measures ordering, not absolute similarity."""
    assert retrieval_graded_ndcg_at_k([0.4, 0.2, 0.1], 3) == pytest.approx(
        retrieval_graded_ndcg_at_k([0.8, 0.4, 0.2], 3)
    )


def test_graded_ndcg_against_a_fixed_ideal_tracks_absolute_similarity():
    low = retrieval_graded_ndcg_at_k([0.4, 0.2, 0.1], 3, ideal_gain=1.0)
    high = retrieval_graded_ndcg_at_k([0.8, 0.4, 0.2], 3, ideal_gain=1.0)
    assert high == pytest.approx(2 * low)


def test_all_zero_gains_score_zero_rather_than_dividing_by_zero():
    assert retrieval_graded_ndcg_at_k([0.0, 0.0], 2) == 0.0


def test_shorter_result_lists_are_not_padded_into_relevance():
    """A run that returned fewer than k candidates must not be credited for the gap."""
    assert positive(1, [1], 3) < positive(1, [1, 1, 1], 3)
