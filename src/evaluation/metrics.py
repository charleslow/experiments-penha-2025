"""Evaluation metrics for retrieval tasks."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def compute_ndcg(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ k.

    Args:
        predictions: Predicted scores of shape (n_samples, n_candidates)
        targets: Binary relevance labels of shape (n_samples, n_candidates)
        k: Cutoff for NDCG calculation

    Returns:
        NDCG@k score
    """
    batch_size = predictions.shape[0]

    # Get top-k indices by predicted score
    topk_indices = torch.topk(predictions, k, dim=1).indices
    topk_true = targets.gather(1, topk_indices)

    # Compute DCG
    positions = torch.arange(2, k + 2, device=predictions.device).unsqueeze(0)
    dcg = torch.sum(topk_true / torch.log2(positions.float()), dim=1)

    # Compute IDCG (ideal DCG)
    ideal_topk = torch.topk(targets, min(k, targets.shape[1]), dim=1).values
    ideal_dcg = torch.sum(
        ideal_topk / torch.log2(positions[:, : ideal_topk.shape[1]].float()),
        dim=1,
    )

    # Handle cases where IDCG is zero
    ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)

    return ndcg.mean().item()


def compute_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute Recall @ k.

    Args:
        predictions: Predicted scores of shape (n_samples, n_candidates)
        targets: Binary relevance labels of shape (n_samples, n_candidates)
        k: Cutoff for recall calculation

    Returns:
        Recall@k score
    """
    # Get top-k indices by predicted score
    topk_indices = torch.topk(predictions, k, dim=1).indices
    topk_true = targets.gather(1, topk_indices)

    # Count true positives
    true_positives = topk_true.sum(dim=1)
    total_relevant = targets.sum(dim=1)

    # Compute recall (handle zero relevant case)
    recall = true_positives / total_relevant.clamp(min=1)

    return recall.mean().item()


def compute_hit_rate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute Hit Rate @ k (at least one relevant item in top-k).

    Args:
        predictions: Predicted scores of shape (n_samples, n_candidates)
        targets: Binary relevance labels of shape (n_samples, n_candidates)
        k: Cutoff for hit rate calculation

    Returns:
        Hit Rate@k score
    """
    # Get top-k indices by predicted score
    topk_indices = torch.topk(predictions, k, dim=1).indices
    topk_true = targets.gather(1, topk_indices)

    # Check if any relevant item is in top-k
    hits = (topk_true.sum(dim=1) > 0).float()

    return hits.mean().item()


def compute_mrr(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute Mean Reciprocal Rank @ k.

    Args:
        predictions: Predicted scores of shape (n_samples, n_candidates)
        targets: Binary relevance labels of shape (n_samples, n_candidates)
        k: Cutoff for MRR calculation

    Returns:
        MRR@k score
    """
    # Get top-k indices by predicted score
    topk_indices = torch.topk(predictions, k, dim=1).indices
    topk_true = targets.gather(1, topk_indices)

    # Find rank of first relevant item
    ranks = torch.arange(1, k + 1, device=predictions.device).unsqueeze(0)
    first_relevant_rank = torch.where(
        topk_true > 0,
        ranks,
        torch.full_like(ranks, k + 1),
    ).min(dim=1).values

    # Compute reciprocal rank (0 if no relevant item in top-k)
    rr = torch.where(
        first_relevant_rank <= k,
        1.0 / first_relevant_rank.float(),
        torch.zeros_like(first_relevant_rank.float()),
    )

    return rr.mean().item()


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    Args:
        query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
        item_embeddings: Item embeddings of shape (n_items, embedding_dim)
        relevance_labels: Binary relevance matrix of shape (n_queries, n_items)
        k_values: List of k values for metrics

    Returns:
        Dictionary of metric_name -> value
    """
    # Compute similarity scores
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
    item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)
    scores = torch.matmul(query_embeddings, item_embeddings.t())

    results = {}
    for k in k_values:
        results[f"NDCG@{k}"] = compute_ndcg(scores, relevance_labels, k)
        results[f"Recall@{k}"] = compute_recall(scores, relevance_labels, k)
        results[f"HitRate@{k}"] = compute_hit_rate(scores, relevance_labels, k)
        results[f"MRR@{k}"] = compute_mrr(scores, relevance_labels, k)

    return results


def evaluate_semantic_id_retrieval(
    generated_ids: torch.Tensor,
    target_ids: torch.Tensor,
    scores: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate semantic ID based retrieval.

    Args:
        generated_ids: Generated semantic IDs (batch_size, num_candidates, n_hierarchies)
        target_ids: Target semantic IDs (batch_size, n_hierarchies)
        scores: Generation scores (batch_size, num_candidates)
        k_values: List of k values for metrics

    Returns:
        Dictionary of metric_name -> value
    """
    batch_size, num_candidates, n_hierarchies = generated_ids.shape
    target_ids = target_ids.unsqueeze(1)  # (batch_size, 1, n_hierarchies)

    # Check for exact matches
    matches = torch.all(generated_ids == target_ids, dim=-1)  # (batch_size, num_candidates)

    results = {}
    for k in k_values:
        # Top-k candidates by score
        topk_indices = torch.topk(scores, min(k, num_candidates), dim=1).indices
        topk_matches = matches.gather(1, topk_indices)

        # Hit rate: at least one match in top-k
        hit_rate = (topk_matches.sum(dim=1) > 0).float().mean().item()
        results[f"HitRate@{k}"] = hit_rate

        # MRR: reciprocal rank of first match
        ranks = torch.arange(1, k + 1, device=scores.device).unsqueeze(0)
        first_match_rank = torch.where(
            topk_matches,
            ranks[:, : topk_matches.shape[1]],
            torch.full_like(topk_matches, k + 1, dtype=torch.long),
        ).min(dim=1).values

        mrr = torch.where(
            first_match_rank <= k,
            1.0 / first_match_rank.float(),
            torch.zeros_like(first_match_rank.float()),
        ).mean().item()
        results[f"MRR@{k}"] = mrr

    return results


class RetrievalMetrics:
    """Helper class to accumulate retrieval metrics across batches."""

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = k_values
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.ndcg_sums = {k: 0.0 for k in self.k_values}
        self.recall_sums = {k: 0.0 for k in self.k_values}
        self.hit_sums = {k: 0.0 for k in self.k_values}
        self.mrr_sums = {k: 0.0 for k in self.k_values}
        self.count = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update metrics with a batch.

        Args:
            predictions: Predicted scores (batch_size, n_candidates)
            targets: Binary relevance labels (batch_size, n_candidates)
        """
        batch_size = predictions.shape[0]
        self.count += batch_size

        for k in self.k_values:
            self.ndcg_sums[k] += compute_ndcg(predictions, targets, k) * batch_size
            self.recall_sums[k] += compute_recall(predictions, targets, k) * batch_size
            self.hit_sums[k] += compute_hit_rate(predictions, targets, k) * batch_size
            self.mrr_sums[k] += compute_mrr(predictions, targets, k) * batch_size

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.count == 0:
            return {}

        results = {}
        for k in self.k_values:
            results[f"NDCG@{k}"] = self.ndcg_sums[k] / self.count
            results[f"Recall@{k}"] = self.recall_sums[k] / self.count
            results[f"HitRate@{k}"] = self.hit_sums[k] / self.count
            results[f"MRR@{k}"] = self.mrr_sums[k] / self.count

        return results
