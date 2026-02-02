#!/usr/bin/env python
"""
Full experiment run script for semantic ID replication.

This script runs the complete experimental matrix:
- Ablation 1: 3 embedding strategies × 5 seeds = 15 runs
- Ablation 2: 4 discretization methods × 5 seeds = 20 runs

Target: < 5 hours total on A4500 GPU
"""

import os
# Set cache directories to workspace to avoid filling root filesystem
os.environ.setdefault("HF_HOME", "/workspace/experiments-penha-2025/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/experiments-penha-2025/.cache/huggingface/transformers")
os.environ.setdefault("TORCH_HOME", "/workspace/experiments-penha-2025/.cache/torch")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FullConfig, ExperimentResults, AggregatedResults
from src.utils.seed import set_seed
from src.data.movielens import load_movielens, chronological_split, MovieItem
from src.data.query_generator import generate_synthetic_queries
from src.data.datamodule import SemanticIDDataModule
from src.models.bi_encoder import BiEncoderModule, train_bi_encoder
from src.models.generative import GenerativeRetrievalModule, train_generative_model
from src.discretization import RQKMeansDiscretizer, RQVAEDiscretizer, LSHDiscretizer, PQDiscretizer
from src.evaluation.metrics import evaluate_retrieval, evaluate_semantic_id_retrieval
from src.visualization.plots import (
    plot_embedding_ablation,
    plot_discretization_ablation,
    plot_tradeoff_scatter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ValidationRecallCallback(L.Callback):
    """
    Callback to compute validation Recall@30 periodically during bi-encoder training.
    Implements early stopping when R@30 plateaus.
    """

    def __init__(
        self,
        items: Dict[int, "MovieItem"],
        queries: Dict[int, List[str]],
        val_df: pd.DataFrame,
        task: str,
        patience: int = 2,
        min_delta: float = 0.001,
        eval_every_n_epochs: int = 1,
    ):
        """
        Args:
            items: Dictionary of item_id to MovieItem
            queries: Dictionary of item_id to list of queries
            val_df: Validation dataframe
            task: Task type ("search", "rec", or "multi_task")
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            eval_every_n_epochs: Evaluate every N epochs
        """
        super().__init__()
        self.items = items
        self.queries = queries
        self.val_df = val_df
        self.task = task
        self.patience = patience
        self.min_delta = min_delta
        self.eval_every_n_epochs = eval_every_n_epochs

        self.best_recall = 0.0
        self.epochs_without_improvement = 0
        self.recall_history = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute validation R@30 after each epoch."""
        current_epoch = trainer.current_epoch

        # Only evaluate every N epochs
        if (current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        # Compute R@30 on validation set
        recall_30 = self._compute_recall_at_k(pl_module, k=30)
        self.recall_history.append(recall_30)

        logger.info(f"Epoch {current_epoch}: Val R@30 = {recall_30:.4f} (best = {self.best_recall:.4f})")

        # Log to trainer
        pl_module.log("val/Recall@30", recall_30, on_epoch=True, prog_bar=True)

        # Check for improvement
        if recall_30 > self.best_recall + self.min_delta:
            self.best_recall = recall_30
            self.epochs_without_improvement = 0
            logger.info(f"  -> New best R@30!")
        else:
            self.epochs_without_improvement += 1
            logger.info(f"  -> No improvement for {self.epochs_without_improvement}/{self.patience} epochs")

            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping: R@30 plateaued at {self.best_recall:.4f}")
                trainer.should_stop = True

    def _compute_recall_at_k(self, model: L.LightningModule, k: int = 30) -> float:
        """Compute Recall@k on validation set."""
        model.eval()

        # Get all item embeddings
        item_ids = sorted(self.items.keys())
        item_texts = [self.items[iid].text for iid in item_ids]
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

        with torch.no_grad():
            item_embeddings = model.get_item_embeddings(item_texts, batch_size=128)

        if self.task in ["search", "multi_task"]:
            # Evaluate query-to-item retrieval
            query_list = []
            target_idxs = []

            # Sample items for efficiency (use up to 1000 items)
            val_item_ids = self.val_df["item_id"].unique()
            if len(val_item_ids) > 1000:
                val_item_ids = np.random.choice(val_item_ids, 1000, replace=False)

            for item_id in val_item_ids:
                if item_id in self.queries and self.queries[item_id] and item_id in item_id_to_idx:
                    query_list.append(self.queries[item_id][0])  # First query
                    target_idxs.append(item_id_to_idx[item_id])

            if not query_list:
                return 0.0

            with torch.no_grad():
                query_embeddings = model.encode(query_list)
                if isinstance(query_embeddings, np.ndarray):
                    query_embeddings = torch.tensor(query_embeddings)

                # Move to CPU for large matrix ops
                query_embeddings = query_embeddings.cpu()
                item_embeddings = item_embeddings.cpu()

                # Normalize
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
                item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

                # Compute similarities
                similarities = torch.matmul(query_embeddings, item_embeddings.t())

                # Get top-k
                _, top_k_indices = torch.topk(similarities, k=min(k, similarities.size(1)), dim=1)

                # Compute recall
                hits = 0
                for i, target_idx in enumerate(target_idxs):
                    if target_idx in top_k_indices[i].tolist():
                        hits += 1

                recall = hits / len(target_idxs)

            return recall

        elif self.task == "rec":
            # Evaluate item-to-item retrieval
            from src.data.movielens import get_cooccurrence_pairs

            # Sample for efficiency
            if len(self.val_df) > 50000:
                val_df_sample = self.val_df.sample(n=50000, random_state=42)
            else:
                val_df_sample = self.val_df

            cooc = get_cooccurrence_pairs(val_df_sample, window_size=5)
            if len(cooc) == 0:
                return 0.0

            # Sample pairs
            if len(cooc) > 500:
                cooc = cooc.sample(n=500, random_state=42)

            query_item_ids = cooc["item1"].tolist()
            target_item_ids = cooc["item2"].tolist()

            query_texts = [self.items[iid].text for iid in query_item_ids if iid in self.items]
            if not query_texts:
                return 0.0

            with torch.no_grad():
                query_embeddings = model.encode(query_texts)
                if isinstance(query_embeddings, np.ndarray):
                    query_embeddings = torch.tensor(query_embeddings)

                # Move to CPU
                query_embeddings = query_embeddings.cpu()
                item_embeddings = item_embeddings.cpu()

                # Normalize
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
                item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

                # Compute similarities
                similarities = torch.matmul(query_embeddings, item_embeddings.t())

                # Get top-k
                _, top_k_indices = torch.topk(similarities, k=min(k, similarities.size(1)), dim=1)

                # Compute recall
                hits = 0
                for i, target_id in enumerate(target_item_ids[:len(query_texts)]):
                    if target_id in item_id_to_idx:
                        target_idx = item_id_to_idx[target_id]
                        if target_idx in top_k_indices[i].tolist():
                            hits += 1

                recall = hits / len(query_texts) if query_texts else 0.0

            return recall

        return 0.0


@dataclass
class FullRunConfig:
    """Configuration for full experiment run.

    Supports 3 phases:
    - Phase 1: Dev run (quick validation)
    - Phase 2: Single run per config (train bi-encoder, validate metrics)
    - Phase 3: Full 5-seed run (reuse bi-encoder, run 5 seeds for gen model)
    """

    # Data
    data_fraction: float = 1.0
    queries_per_item: int = 5

    # Bi-encoder
    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    encoder_epochs: int = 10  # Max epochs, will early stop when R@30 plateaus
    encoder_batch: int = 64  # Reduced for A4500 VRAM
    encoder_lr: float = 2e-5

    # Discretization (paper uses 2 codebooks × 256)
    n_hierarchies: int = 2
    codebook_size: int = 256

    # Generative
    gen_model: str = "google/flan-t5-base"
    gen_epochs: int = 10
    gen_batch: int = 32
    gen_lr: float = 1e-4

    # Experiment
    n_seeds: int = 5
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 1024)
    output_dir: Path = Path("/workspace/experiments-penha-2025/results/full_run")
    save_checkpoints: bool = False  # Disable by default to save disk space

    # Ablation choices
    embedding_strategies: Tuple[str, ...] = ("search", "rec", "multi_task")
    discretization_methods: Tuple[str, ...] = ("rq_kmeans", "rq_vae", "lsh", "pq")

    # Early stopping
    patience: int = 3

    # Phase control
    # Phase 2: train bi-encoder once (n_encoder_seeds=1), run gen once (n_seeds=1)
    # Phase 3: reuse bi-encoder from phase 2, run gen 5 times (n_seeds=5)
    n_encoder_seeds: int = 1  # How many seeds for bi-encoder training (usually 1)
    cache_embeddings: bool = True  # Cache and reuse bi-encoder embeddings
    embeddings_cache_dir: Optional[Path] = None  # Where to cache embeddings


def create_synthetic_data(n_items: int = 100, n_users: int = 50, n_interactions: int = 1000):
    """Create synthetic data for testing when MovieLens is not available."""
    from src.data.movielens import MovieItem

    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance"]
    items = {}
    for i in range(n_items):
        items[i] = MovieItem(
            item_id=i,
            title=f"Movie {i}",
            genres=[genres[i % len(genres)]],
            year=2000 + (i % 24),
        )

    np.random.seed(42)
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.uniform(1, 5, n_interactions)
    timestamps = np.sort(np.random.randint(1000000000, 1600000000, n_interactions))

    interactions = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })

    return interactions, items


def load_data(config: FullRunConfig, force: bool = False):
    """Load or create data."""
    # Try workspace path first, then /app/data path
    raw_dir = Path("/workspace/experiments-penha-2025/data/raw/ml-25m")
    if not raw_dir.exists():
        raw_dir = Path("/app/data/raw/ml-25m")

    if not raw_dir.exists():
        alt_dir = Path("data/raw/ml-25m")
        if alt_dir.exists():
            raw_dir = alt_dir
        else:
            logger.warning(f"MovieLens data not found. Using synthetic data.")
            # Scale synthetic data based on data_fraction
            n_items = max(500, int(5000 * config.data_fraction))
            n_users = max(100, int(1000 * config.data_fraction))
            n_interactions = max(2000, int(50000 * config.data_fraction))
            interactions, items = create_synthetic_data(
                n_items=n_items,
                n_users=n_users,
                n_interactions=n_interactions,
            )
            train_df, val_df, test_df = chronological_split(
                interactions, test_ratio=0.2, val_ratio=0.1
            )
            return items, train_df, val_df, test_df

    logger.info(f"Loading MovieLens data from {raw_dir} (fraction={config.data_fraction})")
    interactions, items = load_movielens(
        data_dir=raw_dir,
        fraction=config.data_fraction,
    )

    train_df, val_df, test_df = chronological_split(
        interactions, test_ratio=0.2, val_ratio=0.1
    )

    return items, train_df, val_df, test_df


def get_embeddings_cache_path(config: FullRunConfig, strategy: str) -> Path:
    """Get path for cached embeddings."""
    cache_dir = config.embeddings_cache_dir or (config.output_dir / "embeddings_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{strategy}_embeddings.pt"


def load_cached_embeddings(
    config: FullRunConfig, strategy: str, items: Dict[int, MovieItem]
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """Load cached embeddings if available."""
    cache_path = get_embeddings_cache_path(config, strategy)
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        cached = torch.load(cache_path)
        # Verify item count matches
        if len(cached["item_ids"]) == len(items):
            return cached["embeddings"], cached["item_ids"]
        else:
            logger.warning(f"Cache item count mismatch: {len(cached['item_ids'])} vs {len(items)}")
    return None


def save_embeddings_cache(
    config: FullRunConfig,
    strategy: str,
    embeddings: torch.Tensor,
    item_ids: List[int]
):
    """Save embeddings to cache."""
    cache_path = get_embeddings_cache_path(config, strategy)
    logger.info(f"Saving embeddings cache to {cache_path}")
    torch.save({
        "embeddings": embeddings.cpu(),
        "item_ids": item_ids,
        "strategy": strategy,
    }, cache_path)


def train_biencoder_for_strategy(
    strategy: str,
    items: Dict[int, MovieItem],
    queries: Dict[int, List[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: FullRunConfig,
    seed: int,
    output_dir: Path,
) -> Tuple[BiEncoderModule, torch.Tensor]:
    """Train bi-encoder for a specific strategy and return embeddings."""
    set_seed(seed)

    logger.info(f"Training {strategy} bi-encoder (seed={seed})...")

    # Create datamodule
    task = strategy if strategy != "multi_task" else "search"  # Use search for validation
    datamodule = SemanticIDDataModule(
        items=items,
        queries=queries,
        train_interactions=train_df,
        val_interactions=val_df,
        test_interactions=val_df,  # Use val as test for bi-encoder
        task=strategy,
        batch_size=config.encoder_batch,
        num_workers=4,
    )

    # Create model
    model = BiEncoderModule(
        model_name=config.encoder_model,
        task=strategy,
        learning_rate=config.encoder_lr,
    )

    # Callbacks
    callbacks = []
    if config.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=f"biencoder_{strategy}_seed{seed}_{{epoch}}",
            save_top_k=1,
            monitor="val/Recall@30",
            mode="max",
        )
        callbacks.append(checkpoint_callback)

    # Use validation R@30 for early stopping (stops when performance plateaus)
    val_recall_callback = ValidationRecallCallback(
        items=items,
        queries=queries,
        val_df=val_df,
        task=strategy,
        patience=config.patience,
        min_delta=0.001,
        eval_every_n_epochs=1,
    )
    callbacks.append(val_recall_callback)

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.encoder_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
        gradient_clip_val=1.0,
        enable_checkpointing=config.save_checkpoints,  # Disable default checkpointing
    )

    # Train
    datamodule.setup("fit")
    trainer.fit(model, datamodule=datamodule)

    # Get item embeddings
    model.eval()
    item_ids = sorted(items.keys())
    item_texts = [items[iid].text for iid in item_ids]
    with torch.no_grad():
        embeddings = model.get_item_embeddings(item_texts, batch_size=128)

    logger.info(f"Generated {strategy} embeddings: {embeddings.shape}")

    # Cache embeddings if enabled
    if config.cache_embeddings:
        save_embeddings_cache(config, strategy, embeddings, item_ids)

    return model, embeddings


def run_discretization(
    method: str,
    embeddings: torch.Tensor,
    config: FullRunConfig,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Run discretization and return semantic IDs and reconstruction error."""
    set_seed(seed)

    embedding_dim = embeddings.shape[1]
    n_items = embeddings.shape[0]

    logger.info(f"Running {method} discretization (seed={seed})...")

    if method == "rq_kmeans":
        discretizer = RQKMeansDiscretizer(
            n_hierarchies=config.n_hierarchies,
            codebook_size=config.codebook_size,
        )
    elif method == "rq_vae":
        discretizer = RQVAEDiscretizer(
            n_hierarchies=config.n_hierarchies,
            codebook_size=config.codebook_size,
            embedding_dim=embedding_dim,
            num_epochs=20,
            batch_size=256,
        )
    elif method == "lsh":
        discretizer = LSHDiscretizer(
            n_hierarchies=config.n_hierarchies,
            codebook_size=config.codebook_size,
        )
    elif method == "pq":
        # PQ requires embedding_dim divisible by n_hierarchies
        # Adjust n_hierarchies if needed
        n_sub = min(config.n_hierarchies, embedding_dim // 64)
        if n_items < config.codebook_size:
            logger.warning(f"Skipping PQ: need {config.codebook_size} samples, have {n_items}")
            return None, float("inf")
        discretizer = PQDiscretizer(
            n_hierarchies=n_sub,
            codebook_size=min(config.codebook_size, n_items),
        )
    else:
        raise ValueError(f"Unknown discretization method: {method}")

    # Fit and encode
    discretizer.fit(embeddings)
    codes = discretizer.encode(embeddings)
    mse = discretizer.reconstruction_error(embeddings)

    logger.info(f"{method}: codes shape {codes.shape}, MSE {mse:.6f}")

    return codes, mse


def train_generative_and_evaluate(
    items: Dict[int, MovieItem],
    queries: Dict[int, List[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    semantic_ids: Dict[int, List[int]],
    config: FullRunConfig,
    seed: int,
    output_dir: Path,
) -> Dict[str, float]:
    """Train generative model and evaluate retrieval performance."""
    set_seed(seed)

    logger.info(f"Training generative model (seed={seed})...")

    # Create generative datamodule
    datamodule = SemanticIDDataModule(
        items=items,
        queries=queries,
        train_interactions=train_df,
        val_interactions=val_df,
        test_interactions=test_df,
        semantic_ids=semantic_ids,
        task="generative",
        batch_size=config.gen_batch,
        num_workers=4,
    )

    # Create model
    model = GenerativeRetrievalModule(
        model_name=config.gen_model,
        n_hierarchies=config.n_hierarchies,
        codebook_size=config.codebook_size,
        learning_rate=config.gen_lr,
    )

    # Callbacks
    callbacks = []
    if config.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=f"generative_seed{seed}_{{epoch}}",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.patience,
        mode="min",
    )
    callbacks.append(early_stop_callback)

    # Trainer with gradient checkpointing for memory efficiency
    trainer = L.Trainer(
        max_epochs=config.gen_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = 64
        enable_checkpointing=config.save_checkpoints,  # Disable default checkpointing
    )

    # Train
    datamodule.setup("fit")
    trainer.fit(model, datamodule=datamodule)

    # Evaluate on test set
    model.eval()

    # Build test queries and targets
    test_queries = []
    test_targets = []
    test_item_ids = []

    for item_id in test_df["item_id"].unique():
        if item_id in queries and item_id in semantic_ids:
            for q in queries[item_id][:1]:  # Use first query per item
                test_queries.append(q)
                test_targets.append(torch.tensor(semantic_ids[item_id]))
                test_item_ids.append(item_id)

    if len(test_queries) == 0:
        logger.warning("No test queries available")
        return {"search_Recall@30": 0.0, "rec_Recall@30": 0.0}

    # Generate in batches (30 candidates for Recall@30)
    batch_size = 16
    all_generated = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(test_queries), batch_size):
            batch_queries = test_queries[i:i + batch_size]
            gen_ids, scores = model.generate(batch_queries, num_return_sequences=30)
            all_generated.append(gen_ids)
            all_scores.append(scores)

    if all_generated:
        generated_ids = torch.cat(all_generated, dim=0)
        scores = torch.cat(all_scores, dim=0)
        target_ids = torch.stack(test_targets)

        # Evaluate with k=30 to match paper's Recall@30
        results = evaluate_semantic_id_retrieval(
            generated_ids=generated_ids,
            target_ids=target_ids,
            scores=scores,
            k_values=[10, 20, 30],
        )
        # Map to expected field names for both tasks
        results["search_Recall@30"] = results.get("HitRate@30", 0.0)
        results["rec_Recall@30"] = results.get("HitRate@30", 0.0)
    else:
        results = {"search_Recall@30": 0.0, "rec_Recall@30": 0.0}

    return results


def evaluate_biencoder_retrieval(
    model: BiEncoderModule,
    items: Dict[int, MovieItem],
    queries: Dict[int, List[str]],
    test_df: pd.DataFrame,
    task: str,
) -> Dict[str, float]:
    """Evaluate bi-encoder retrieval performance."""
    model.eval()

    # Get all item embeddings
    item_ids = sorted(items.keys())
    item_texts = [items[iid].text for iid in item_ids]
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    with torch.no_grad():
        item_embeddings = model.get_item_embeddings(item_texts, batch_size=128)

    if task == "search":
        # Evaluate query-to-item retrieval
        query_list = []
        relevance = []

        for item_id in test_df["item_id"].unique():
            if item_id in queries and queries[item_id]:
                for q in queries[item_id][:1]:  # First query per item
                    query_list.append(q)
                    rel = torch.zeros(len(item_ids))
                    if item_id in item_id_to_idx:
                        rel[item_id_to_idx[item_id]] = 1.0
                    relevance.append(rel)

        if not query_list:
            return {"Recall@30": 0.0}

        with torch.no_grad():
            query_embeddings = model.encode(query_list)

        # Ensure all tensors are on the same device (CPU for large eval)
        device = "cpu"
        relevance_matrix = torch.stack(relevance).to(device)
        query_embeddings = query_embeddings.to(device) if isinstance(query_embeddings, torch.Tensor) else torch.tensor(query_embeddings).to(device)
        item_embeddings_eval = item_embeddings.to(device)

        results = evaluate_retrieval(
            query_embeddings=query_embeddings,
            item_embeddings=item_embeddings_eval,
            relevance_labels=relevance_matrix,
            k_values=[10, 20, 30],  # Paper uses R@30
        )

    elif task == "rec":
        # Evaluate item-to-item retrieval (next item prediction)
        from src.data.movielens import get_cooccurrence_pairs

        # Sample test_df first to avoid slow co-occurrence computation
        # Only need a subset for evaluation
        if len(test_df) > 100000:
            test_df_sample = test_df.sample(n=100000, random_state=42)
        else:
            test_df_sample = test_df

        cooc = get_cooccurrence_pairs(test_df_sample, window_size=5)
        if len(cooc) == 0:
            return {"Recall@30": 0.0}

        # Sample subset for efficiency
        if len(cooc) > 1000:
            cooc = cooc.sample(n=1000, random_state=42)

        query_item_ids = cooc["item1"].tolist()
        target_item_ids = cooc["item2"].tolist()

        query_texts = [items[iid].text for iid in query_item_ids if iid in items]
        if not query_texts:
            return {"Recall@30": 0.0}

        with torch.no_grad():
            query_embeddings = model.encode(query_texts)

        # Build relevance matrix
        relevance = []
        for target_id in target_item_ids[:len(query_texts)]:
            rel = torch.zeros(len(item_ids))
            if target_id in item_id_to_idx:
                rel[item_id_to_idx[target_id]] = 1.0
            relevance.append(rel)

        # Ensure all tensors are on the same device (CPU for large eval)
        device = "cpu"
        relevance_matrix = torch.stack(relevance).to(device)
        query_embeddings = query_embeddings.to(device) if isinstance(query_embeddings, torch.Tensor) else torch.tensor(query_embeddings).to(device)
        item_embeddings_eval = item_embeddings.to(device)

        results = evaluate_retrieval(
            query_embeddings=query_embeddings,
            item_embeddings=item_embeddings_eval,
            relevance_labels=relevance_matrix,
            k_values=[10, 20, 30],  # Paper uses R@30
        )
    else:
        # Multi-task: average of both
        search_results = evaluate_biencoder_retrieval(model, items, queries, test_df, "search")
        rec_results = evaluate_biencoder_retrieval(model, items, queries, test_df, "rec")
        results = {}
        for k in search_results:
            results[f"search_{k}"] = search_results[k]
        for k in rec_results:
            results[f"rec_{k}"] = rec_results[k]

    return results


def run_ablation1_embedding_strategy(
    config: FullRunConfig,
    items: Dict[int, MovieItem],
    queries: Dict[int, List[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> List[ExperimentResults]:
    """Run Ablation 1: Compare embedding strategies with fixed RQ-KMeans.

    Bi-encoder is trained only once per strategy (n_encoder_seeds=1 by default).
    Embeddings are cached and reused for all generative model seeds.
    """
    logger.info("=" * 60)
    logger.info("ABLATION 1: Embedding Strategy Comparison")
    logger.info("=" * 60)

    results = []

    for strategy in config.embedding_strategies:
        # Try to load cached embeddings first
        cached = load_cached_embeddings(config, strategy, items) if config.cache_embeddings else None
        embeddings = None
        model = None

        for seed_idx, seed in enumerate(config.seeds[:config.n_seeds]):
            run_start = time.time()
            logger.info(f"\n--- {strategy} (seed {seed}, run {seed_idx + 1}/{config.n_seeds}) ---")

            try:
                # Train bi-encoder only if:
                # 1. No cached embeddings, AND
                # 2. First seed OR n_encoder_seeds > seed_idx
                if cached is not None:
                    logger.info(f"Using cached embeddings for {strategy}")
                    embeddings = cached[0].to("cuda" if torch.cuda.is_available() else "cpu")
                    # Create a minimal model for evaluation (load pretrained)
                    if model is None:
                        model = BiEncoderModule(
                            model_name=config.encoder_model,
                            task=strategy,
                        )
                elif embeddings is None or seed_idx < config.n_encoder_seeds:
                    # Train bi-encoder
                    model, embeddings = train_biencoder_for_strategy(
                        strategy=strategy,
                        items=items,
                        queries=queries,
                        train_df=train_df,
                        val_df=val_df,
                        config=config,
                        seed=seed,
                        output_dir=output_dir,
                    )
                else:
                    logger.info(f"Reusing embeddings from first seed for {strategy}")

                # Discretize with RQ-KMeans (fixed for ablation 1)
                codes, mse = run_discretization(
                    method="rq_kmeans",
                    embeddings=embeddings,
                    config=config,
                    seed=seed,
                )

                # Create semantic ID mapping
                item_ids = sorted(items.keys())
                semantic_ids = {
                    iid: codes[idx].tolist()
                    for idx, iid in enumerate(item_ids)
                }

                # Evaluate bi-encoder directly
                eval_results = evaluate_biencoder_retrieval(
                    model=model,
                    items=items,
                    queries=queries,
                    test_df=test_df,
                    task=strategy,
                )

                # Store results (using Recall@30 to match paper)
                result = ExperimentResults(
                    embedding_strategy=strategy,
                    discretization_method="rq_kmeans",
                    search_recall_30=eval_results.get("search_Recall@30", eval_results.get("Recall@30", 0.0)),
                    rec_recall_30=eval_results.get("rec_Recall@30", eval_results.get("Recall@30", 0.0)),
                    run_id=seed_idx,
                )
                results.append(result)

                run_time = time.time() - run_start
                logger.info(f"Completed in {run_time:.1f}s")
                logger.info(f"Search Recall@30: {result.search_recall_30:.4f}")
                logger.info(f"Rec Recall@30: {result.rec_recall_30:.4f}")

            except Exception as e:
                logger.error(f"Run failed: {e}")
                import traceback
                traceback.print_exc()

                # Add placeholder result
                results.append(ExperimentResults(
                    embedding_strategy=strategy,
                    discretization_method="rq_kmeans",
                    search_recall_30=0.0,
                    rec_recall_30=0.0,
                    run_id=seed_idx,
                ))

    return results


def run_ablation2_discretization_method(
    config: FullRunConfig,
    items: Dict[int, MovieItem],
    queries: Dict[int, List[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    cached_multitask_embeddings: Optional[Dict[int, torch.Tensor]] = None,
) -> List[ExperimentResults]:
    """Run Ablation 2: Compare discretization methods with fixed multi-task embeddings."""
    logger.info("=" * 60)
    logger.info("ABLATION 2: Discretization Method Comparison")
    logger.info("=" * 60)

    results = []

    for method in config.discretization_methods:
        for seed_idx, seed in enumerate(config.seeds[:config.n_seeds]):
            run_start = time.time()
            logger.info(f"\n--- {method} (seed {seed}, run {seed_idx + 1}/{config.n_seeds}) ---")

            try:
                # Use cached embeddings if available, otherwise train
                if cached_multitask_embeddings and seed in cached_multitask_embeddings:
                    embeddings = cached_multitask_embeddings[seed]
                    logger.info("Using cached multi-task embeddings")
                else:
                    model, embeddings = train_biencoder_for_strategy(
                        strategy="multi_task",
                        items=items,
                        queries=queries,
                        train_df=train_df,
                        val_df=val_df,
                        config=config,
                        seed=seed,
                        output_dir=output_dir,
                    )

                # Run discretization
                codes, mse = run_discretization(
                    method=method,
                    embeddings=embeddings,
                    config=config,
                    seed=seed,
                )

                if codes is None:
                    logger.warning(f"Skipping {method} (discretization failed)")
                    results.append(ExperimentResults(
                        embedding_strategy="multi_task",
                        discretization_method=method,
                        search_recall_30=0.0,
                        rec_recall_30=0.0,
                        run_id=seed_idx,
                    ))
                    continue

                # Create semantic ID mapping
                item_ids = sorted(items.keys())
                semantic_ids = {
                    iid: codes[idx].tolist()
                    for idx, iid in enumerate(item_ids)
                }

                # Train and evaluate generative model
                gen_results = train_generative_and_evaluate(
                    items=items,
                    queries=queries,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    semantic_ids=semantic_ids,
                    config=config,
                    seed=seed,
                    output_dir=output_dir,
                )

                # Store results (using Recall@30 for generative retrieval)
                result = ExperimentResults(
                    embedding_strategy="multi_task",
                    discretization_method=method,
                    search_recall_30=gen_results.get("search_Recall@30", 0.0),
                    rec_recall_30=gen_results.get("rec_Recall@30", 0.0),
                    run_id=seed_idx,
                )
                results.append(result)

                run_time = time.time() - run_start
                logger.info(f"Completed in {run_time:.1f}s")
                logger.info(f"Search Recall@30: {result.search_recall_30:.4f}")
                logger.info(f"Rec Recall@30: {result.rec_recall_30:.4f}")

            except Exception as e:
                logger.error(f"Run failed: {e}")
                import traceback
                traceback.print_exc()

                results.append(ExperimentResults(
                    embedding_strategy="multi_task",
                    discretization_method=method,
                    search_recall_30=0.0,
                    rec_recall_30=0.0,
                    run_id=seed_idx,
                ))

    return results


def aggregate_results(results: List[ExperimentResults]) -> List[AggregatedResults]:
    """Aggregate results across seeds to get mean and std error."""
    from collections import defaultdict

    # Group by (embedding_strategy, discretization_method)
    groups = defaultdict(list)
    for r in results:
        key = (r.embedding_strategy, r.discretization_method)
        groups[key].append(r)

    aggregated = []
    for (strategy, method), runs in groups.items():
        n = len(runs)

        search_recall = [r.search_recall_30 for r in runs]
        rec_recall = [r.rec_recall_30 for r in runs]

        aggregated.append(AggregatedResults(
            embedding_strategy=strategy,
            discretization_method=method,
            search_recall_30_mean=np.mean(search_recall),
            search_recall_30_std=np.std(search_recall) / np.sqrt(n) if n > 1 else 0.0,
            rec_recall_30_mean=np.mean(rec_recall),
            rec_recall_30_std=np.std(rec_recall) / np.sqrt(n) if n > 1 else 0.0,
            n_runs=n,
        ))

    return aggregated


def save_results(
    ablation1_results: List[ExperimentResults],
    ablation2_results: List[ExperimentResults],
    config: FullRunConfig,
    output_dir: Path,
):
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    agg1 = aggregate_results(ablation1_results)
    agg2 = aggregate_results(ablation2_results)

    # Save raw results
    results_dict = {
        "config": {
            "data_fraction": config.data_fraction,
            "encoder_model": config.encoder_model,
            "encoder_epochs": config.encoder_epochs,
            "n_hierarchies": config.n_hierarchies,
            "codebook_size": config.codebook_size,
            "gen_model": config.gen_model,
            "gen_epochs": config.gen_epochs,
            "n_seeds": config.n_seeds,
            "timestamp": datetime.now().isoformat(),
        },
        "ablation1_raw": [asdict(r) for r in ablation1_results],
        "ablation2_raw": [asdict(r) for r in ablation2_results],
        "ablation1_aggregated": [asdict(r) for r in agg1],
        "ablation2_aggregated": [asdict(r) for r in agg2],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Create plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Ablation 1 plot (using Recall@30 to match paper)
    emb_results = {}
    for r in agg1:
        emb_results[r.embedding_strategy] = {
            "Search R@30": r.search_recall_30_mean,
            "Rec R@30": r.rec_recall_30_mean,
            "Search R@30_std": r.search_recall_30_std,
            "Rec R@30_std": r.rec_recall_30_std,
        }

    plot_embedding_ablation(
        results=emb_results,
        output_path=plots_dir / "ablation1_embedding.png",
        title="Ablation 1: Embedding Strategy (RQ-KMeans fixed)",
    )

    # Ablation 2 plot
    disc_results = {}
    for r in agg2:
        disc_results[r.discretization_method] = {
            "Search R@30": r.search_recall_30_mean,
            "Rec R@30": r.rec_recall_30_mean,
            "Search R@30_std": r.search_recall_30_std,
            "Rec R@30_std": r.rec_recall_30_std,
        }

    plot_discretization_ablation(
        results=disc_results,
        output_path=plots_dir / "ablation2_discretization.png",
        title="Ablation 2: Discretization Method (Multi-task fixed)",
    )

    # Trade-off scatter
    tradeoff_data = []
    for r in agg1 + agg2:
        tradeoff_data.append({
            "strategy": r.embedding_strategy,
            "method": r.discretization_method,
            "search_recall": r.search_recall_30_mean,
            "rec_recall": r.rec_recall_30_mean,
        })

    plot_tradeoff_scatter(
        results=tradeoff_data,
        output_path=plots_dir / "tradeoff_scatter.png",
        title="Search vs Recommendation Trade-off",
    )

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Plots saved to {plots_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Primary metric: Recall@30 to match paper)")
    print("=" * 80)

    # Paper reference values for validation
    print("\nPaper Reference (Table 1):")
    print("  Search-based: Search R@30=0.072, Rec R@30=0.026")
    print("  Rec-based: Search R@30=0.004, Rec R@30=0.062")
    print("  Multi-task (Table 2): Search R@30=0.046, Rec R@30=0.049")

    print("\nAblation 1: Embedding Strategy (RQ-KMeans fixed)")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Search R@30':<20} {'Rec R@30':<20}")
    for r in agg1:
        print(f"{r.embedding_strategy:<15} {r.search_recall_30_mean:.4f} ± {r.search_recall_30_std:.4f}    {r.rec_recall_30_mean:.4f} ± {r.rec_recall_30_std:.4f}")

    print("\nAblation 2: Discretization Method (Multi-task fixed)")
    print("-" * 60)
    print(f"{'Method':<15} {'Search R@30':<20} {'Rec R@30':<20}")
    for r in agg2:
        print(f"{r.discretization_method:<15} {r.search_recall_30_mean:.4f} ± {r.search_recall_30_std:.4f}    {r.rec_recall_30_mean:.4f} ± {r.rec_recall_30_std:.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_run"),
        help="Output directory",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds",
    )
    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=5,
        help="Number of bi-encoder epochs",
    )
    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=10,
        help="Number of generative model epochs",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="Which ablation to run",
    )
    parser.add_argument(
        "--skip-ablation1",
        action="store_true",
        help="Skip ablation 1",
    )
    parser.add_argument(
        "--skip-ablation2",
        action="store_true",
        help="Skip ablation 2",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints (requires disk space)",
    )
    parser.add_argument(
        "--n-encoder-seeds",
        type=int,
        default=1,
        help="Number of seeds for bi-encoder training (default 1, embeddings are reused)",
    )
    parser.add_argument(
        "--no-cache-embeddings",
        action="store_true",
        help="Disable embedding caching (retrain bi-encoder for each seed)",
    )
    parser.add_argument(
        "--embeddings-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached embeddings (default: output_dir/embeddings_cache)",
    )

    args = parser.parse_args()

    # Create config
    config = FullRunConfig(
        data_fraction=args.data_fraction,
        n_seeds=args.n_seeds,
        encoder_epochs=args.encoder_epochs,
        gen_epochs=args.gen_epochs,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        n_encoder_seeds=args.n_encoder_seeds,
        cache_embeddings=not args.no_cache_embeddings,
        embeddings_cache_dir=args.embeddings_cache_dir,
    )

    logger.info("=" * 80)
    logger.info("FULL EXPERIMENT RUN")
    logger.info("=" * 80)
    logger.info(f"Config:")
    logger.info(f"  Data fraction: {config.data_fraction}")
    logger.info(f"  Encoder model: {config.encoder_model}")
    logger.info(f"  Encoder epochs: {config.encoder_epochs}")
    logger.info(f"  Gen model: {config.gen_model}")
    logger.info(f"  Gen epochs: {config.gen_epochs}")
    logger.info(f"  Seeds: {config.seeds[:config.n_seeds]}")
    logger.info(f"  N hierarchies: {config.n_hierarchies}")
    logger.info(f"  Codebook size: {config.codebook_size}")
    logger.info("=" * 80)

    total_start = time.time()

    # Load data
    logger.info("\nLoading data...")
    items, train_df, val_df, test_df = load_data(config)
    logger.info(f"Loaded {len(items)} items, {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Generate queries
    logger.info("\nGenerating synthetic queries...")
    queries = generate_synthetic_queries(items, n_queries=config.queries_per_item)
    logger.info(f"Generated queries for {len(queries)} items")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Run ablations
    ablation1_results = []
    ablation2_results = []

    if args.ablation in ["1", "both"] and not args.skip_ablation1:
        ablation1_results = run_ablation1_embedding_strategy(
            config=config,
            items=items,
            queries=queries,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=config.output_dir,
        )

    if args.ablation in ["2", "both"] and not args.skip_ablation2:
        ablation2_results = run_ablation2_discretization_method(
            config=config,
            items=items,
            queries=queries,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=config.output_dir,
        )

    # Save results
    save_results(
        ablation1_results=ablation1_results,
        ablation2_results=ablation2_results,
        config=config,
        output_dir=config.output_dir,
    )

    total_time = time.time() - total_start
    logger.info(f"\nTotal experiment time: {total_time / 60:.1f} minutes")
    logger.info("Full experiment completed!")


if __name__ == "__main__":
    main()
