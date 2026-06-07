from __future__ import annotations

import math

import pytest

from deepresearch_agent.retrieval_eval import compute_retrieval_metrics


def test_compute_retrieval_metrics_at_k() -> None:
    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 1},
    }
    rankings = {
        "q1": [{"doc_id": "d1"}, {"doc_id": "d9"}],
        "q2": [{"doc_id": "d8"}, {"doc_id": "d3"}],
    }

    metrics = compute_retrieval_metrics(qrels, rankings, k=2)

    expected_q1_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3))
    expected_q2_ndcg = (1.0 / math.log2(3)) / 1.0
    assert metrics["query_count"] == 2
    assert metrics["recall@2"] == pytest.approx(0.75)
    assert metrics["ndcg@2"] == pytest.approx((expected_q1_ndcg + expected_q2_ndcg) / 2)
    assert metrics["mrr"] == pytest.approx(0.75)


def test_compute_retrieval_metrics_deduplicates_ranked_docs() -> None:
    qrels = {"q1": {"d1": 1, "d2": 1}}
    rankings = {"q1": [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}]}

    metrics = compute_retrieval_metrics(qrels, rankings, k=3)

    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["ndcg@3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_compute_retrieval_metrics_requires_overlapping_queries() -> None:
    with pytest.raises(ValueError, match="no overlapping query ids"):
        compute_retrieval_metrics({"q1": {"d1": 1}}, {"q2": [{"doc_id": "d1"}]}, k=10)
