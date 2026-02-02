#!/usr/bin/env python
"""
Quick script to run generative training using cached embeddings.
This gives end-to-end results for the search task.
"""

import os
os.environ.setdefault("HF_HOME", "/workspace/experiments-penha-2025/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/experiments-penha-2025/.cache/huggingface/transformers")
os.environ.setdefault("TORCH_HOME", "/workspace/experiments-penha-2025/.cache/torch")

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.data.movielens import load_movielens, chronological_split, MovieItem
from src.data.query_generator import generate_synthetic_queries
from src.data.datamodule import SemanticIDDataModule
from src.models.generative import GenerativeRetrievalModule
from src.discretization import RQKMeansDiscretizer
from src.evaluation.metrics import evaluate_semantic_id_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str,
                        default="/workspace/experiments-penha-2025/results/phase2_v5/embeddings_cache/search_embeddings.pt")
    parser.add_argument("--gen-epochs", type=int, default=10)
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/workspace/experiments-penha-2025/results/generative_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Load data
    logger.info("Loading MovieLens data...")
    raw_dir = Path("/workspace/experiments-penha-2025/data/raw/ml-25m")
    interactions, items = load_movielens(data_dir=raw_dir, fraction=1.0)
    train_df, val_df, test_df = chronological_split(interactions, test_ratio=0.2, val_ratio=0.1)
    logger.info(f"Loaded {len(items)} items, {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Generate queries
    logger.info("Generating synthetic queries...")
    queries = generate_synthetic_queries(items, n_queries=5)

    # Load cached embeddings
    logger.info(f"Loading cached embeddings from {args.embeddings_path}")
    cached = torch.load(args.embeddings_path)
    embeddings = cached["embeddings"]
    item_ids = cached["item_ids"]
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Run RQ-KMeans discretization
    logger.info("Running RQ-KMeans discretization...")
    discretizer = RQKMeansDiscretizer(n_hierarchies=2, codebook_size=256)
    discretizer.fit(embeddings)
    codes = discretizer.encode(embeddings)
    mse = discretizer.reconstruction_error(embeddings)
    logger.info(f"RQ-KMeans: codes shape {codes.shape}, MSE {mse:.6f}")

    # Create semantic ID mapping
    semantic_ids = {}
    for idx, item_id in enumerate(item_ids):
        semantic_ids[item_id] = codes[idx].tolist()

    # Create datamodule for generative task
    logger.info("Creating generative datamodule...")
    datamodule = SemanticIDDataModule(
        items=items,
        queries=queries,
        train_interactions=train_df,
        val_interactions=val_df,
        test_interactions=test_df,
        semantic_ids=semantic_ids,
        task="generative",
        batch_size=args.gen_batch,
        num_workers=4,
    )

    # Create generative model
    logger.info("Creating generative model (flan-t5-base)...")
    model = GenerativeRetrievalModule(
        model_name="google/flan-t5-base",
        n_hierarchies=2,
        codebook_size=256,
        learning_rate=1e-4,
    )

    # Callbacks
    callbacks = []
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min",
    )
    callbacks.append(early_stop_callback)

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.gen_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        enable_checkpointing=False,
    )

    # Train
    logger.info("Starting generative model training...")
    start_time = time.time()
    datamodule.setup("fit")
    trainer.fit(model, datamodule=datamodule)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
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

    logger.info(f"Evaluating on {len(test_queries)} test queries...")

    if len(test_queries) == 0:
        logger.error("No test queries available")
        return

    # Generate in batches
    batch_size = 16
    all_generated = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, min(len(test_queries), 1000), batch_size):  # Cap at 1000 for speed
            batch_queries = test_queries[i:i + batch_size]
            gen_ids, scores = model.generate(batch_queries, num_return_sequences=30)
            all_generated.append(gen_ids)
            all_scores.append(scores)
            if i % 100 == 0:
                logger.info(f"  Generated {i}/{min(len(test_queries), 1000)}...")

    if all_generated:
        generated_ids = torch.cat(all_generated, dim=0)
        scores = torch.cat(all_scores, dim=0)
        target_ids = torch.stack(test_targets[:generated_ids.shape[0]])

        # Evaluate with k=30
        results = evaluate_semantic_id_retrieval(
            generated_ids=generated_ids,
            target_ids=target_ids,
            scores=scores,
            k_values=[10, 20, 30],
        )

        logger.info("=" * 60)
        logger.info("END-TO-END RESULTS (Search Task)")
        logger.info("=" * 60)
        logger.info(f"  NDCG@10: {results.get('NDCG@10', 0.0):.4f}")
        logger.info(f"  NDCG@30: {results.get('NDCG@30', 0.0):.4f}")
        logger.info(f"  Recall@10: {results.get('Recall@10', 0.0):.4f}")
        logger.info(f"  Recall@30: {results.get('Recall@30', 0.0):.4f}")
        logger.info(f"  HitRate@10: {results.get('HitRate@10', 0.0):.4f}")
        logger.info(f"  HitRate@30: {results.get('HitRate@30', 0.0):.4f}")
        logger.info("=" * 60)
        logger.info(f"Paper target: Search R@30 = 0.072 (±0.028)")
        logger.info(f"Our result:   Search R@30 = {results.get('Recall@30', 0.0):.4f}")
        logger.info("=" * 60)

        # Save results
        import json
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "task": "search",
                "metrics": results,
                "train_time": train_time,
                "n_test_queries": len(test_queries),
            }, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
