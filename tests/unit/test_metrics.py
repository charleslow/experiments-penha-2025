"""Unit tests for evaluation metrics."""

import pytest
import torch
import numpy as np

from src.evaluation.metrics import (
    compute_ndcg,
    compute_recall,
    compute_hit_rate,
    compute_mrr,
    evaluate_retrieval,
    RetrievalMetrics,
)


class TestNDCG:
    """Tests for NDCG computation."""

    def test_perfect_ranking(self):
        # Perfect ranking should give NDCG = 1
        predictions = torch.tensor([[1.0, 0.5, 0.3, 0.1]])
        targets = torch.tensor([[1, 0, 0, 0]])

        ndcg = compute_ndcg(predictions, targets, k=4)
        assert ndcg == pytest.approx(1.0, rel=1e-4)

    def test_worst_ranking(self):
        # Relevant item at the end should give low NDCG
        predictions = torch.tensor([[0.1, 0.2, 0.3, 1.0]])
        targets = torch.tensor([[1, 0, 0, 0]])

        ndcg = compute_ndcg(predictions, targets, k=4)
        assert ndcg < 0.5

    def test_multiple_relevant(self):
        # Multiple relevant items
        predictions = torch.tensor([[1.0, 0.9, 0.1, 0.05]])
        targets = torch.tensor([[1, 1, 0, 0]])

        ndcg = compute_ndcg(predictions, targets, k=4)
        assert ndcg == pytest.approx(1.0, rel=1e-4)

    def test_no_relevant_items(self):
        # No relevant items should give NDCG = 0 (or handled gracefully)
        predictions = torch.tensor([[1.0, 0.5, 0.3, 0.1]])
        targets = torch.tensor([[0, 0, 0, 0]])

        ndcg = compute_ndcg(predictions, targets, k=4)
        assert ndcg == 0.0 or np.isfinite(ndcg)

    def test_batch_computation(self):
        predictions = torch.tensor([
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.0, 0.3, 0.1],
        ])
        targets = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        ndcg = compute_ndcg(predictions, targets, k=4)
        assert ndcg == pytest.approx(1.0, rel=1e-4)


class TestRecall:
    """Tests for Recall computation."""

    def test_perfect_recall(self):
        predictions = torch.tensor([[1.0, 0.9, 0.8, 0.1]])
        targets = torch.tensor([[1, 1, 1, 0]])

        recall = compute_recall(predictions, targets, k=3)
        assert recall == pytest.approx(1.0, rel=1e-4)

    def test_partial_recall(self):
        predictions = torch.tensor([[1.0, 0.1, 0.9, 0.8]])
        targets = torch.tensor([[1, 1, 0, 0]])

        recall = compute_recall(predictions, targets, k=2)
        # Only 1 of 2 relevant items in top-2
        assert recall == pytest.approx(0.5, rel=1e-4)

    def test_zero_recall(self):
        predictions = torch.tensor([[0.1, 0.2, 1.0, 0.9]])
        targets = torch.tensor([[1, 1, 0, 0]])

        recall = compute_recall(predictions, targets, k=2)
        assert recall == 0.0


class TestHitRate:
    """Tests for Hit Rate computation."""

    def test_hit(self):
        predictions = torch.tensor([[1.0, 0.5, 0.3, 0.1]])
        targets = torch.tensor([[1, 0, 0, 0]])

        hit_rate = compute_hit_rate(predictions, targets, k=1)
        assert hit_rate == 1.0

    def test_miss(self):
        predictions = torch.tensor([[0.1, 0.5, 0.3, 1.0]])
        targets = torch.tensor([[1, 0, 0, 0]])

        hit_rate = compute_hit_rate(predictions, targets, k=1)
        assert hit_rate == 0.0

    def test_batch_hit_rate(self):
        predictions = torch.tensor([
            [1.0, 0.5, 0.3, 0.1],
            [0.1, 0.5, 0.3, 1.0],
        ])
        targets = torch.tensor([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])

        hit_rate = compute_hit_rate(predictions, targets, k=1)
        assert hit_rate == 0.5


class TestMRR:
    """Tests for MRR computation."""

    def test_first_position(self):
        predictions = torch.tensor([[1.0, 0.5, 0.3, 0.1]])
        targets = torch.tensor([[1, 0, 0, 0]])

        mrr = compute_mrr(predictions, targets, k=4)
        assert mrr == pytest.approx(1.0, rel=1e-4)

    def test_second_position(self):
        predictions = torch.tensor([[0.5, 1.0, 0.3, 0.1]])
        targets = torch.tensor([[1, 0, 0, 0]])

        mrr = compute_mrr(predictions, targets, k=4)
        assert mrr == pytest.approx(0.5, rel=1e-4)

    def test_not_in_top_k(self):
        predictions = torch.tensor([[0.1, 0.2, 0.3, 1.0]])
        targets = torch.tensor([[1, 0, 0, 0]])

        mrr = compute_mrr(predictions, targets, k=2)
        assert mrr == 0.0


class TestEvaluateRetrieval:
    """Tests for the combined evaluation function."""

    def test_evaluate_retrieval(self):
        torch.manual_seed(42)

        query_emb = torch.randn(10, 64)
        item_emb = torch.randn(20, 64)
        relevance = torch.zeros(10, 20)
        # Set some relevant items
        for i in range(10):
            relevance[i, i % 20] = 1

        results = evaluate_retrieval(
            query_emb, item_emb, relevance,
            k_values=[1, 5, 10],
        )

        assert "NDCG@1" in results
        assert "NDCG@5" in results
        assert "NDCG@10" in results
        assert "Recall@10" in results
        assert "HitRate@10" in results
        assert "MRR@10" in results

        # All metrics should be between 0 and 1
        for key, value in results.items():
            assert 0 <= value <= 1


class TestRetrievalMetrics:
    """Tests for the RetrievalMetrics accumulator."""

    def test_accumulation(self):
        metrics = RetrievalMetrics(k_values=[5, 10])

        # Add some batches
        for _ in range(3):
            predictions = torch.rand(8, 20)
            targets = (torch.rand(8, 20) > 0.9).float()
            metrics.update(predictions, targets)

        results = metrics.compute()

        assert "NDCG@5" in results
        assert "NDCG@10" in results
        assert metrics.count == 24  # 3 batches * 8 samples

    def test_reset(self):
        metrics = RetrievalMetrics(k_values=[10])

        predictions = torch.rand(8, 20)
        targets = (torch.rand(8, 20) > 0.9).float()
        metrics.update(predictions, targets)

        metrics.reset()

        assert metrics.count == 0

    def test_empty_compute(self):
        metrics = RetrievalMetrics(k_values=[10])
        results = metrics.compute()

        assert results == {}
