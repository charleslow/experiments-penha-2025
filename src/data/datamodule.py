"""Lightning DataModule for semantic ID training."""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np

from .movielens import MovieItem, get_cooccurrence_pairs

logger = logging.getLogger(__name__)


@dataclass
class SearchBatch:
    """Batch for search task (query -> item)."""

    queries: List[str]
    items: List[str]
    item_ids: torch.Tensor
    labels: torch.Tensor  # Positive item indices


@dataclass
class RecBatch:
    """Batch for recommendation task (item -> item)."""

    items1: List[str]
    items2: List[str]
    item_ids1: torch.Tensor
    item_ids2: torch.Tensor
    labels: torch.Tensor


@dataclass
class MultiTaskBatch:
    """Batch for multi-task training."""

    search_batch: Optional[SearchBatch]
    rec_batch: Optional[RecBatch]


class SearchDataset(Dataset):
    """Dataset for search task (query -> item matching)."""

    def __init__(
        self,
        items: Dict[int, MovieItem],
        queries: Dict[int, List[str]],
        interactions: pd.DataFrame,
    ):
        """
        Initialize the search dataset.

        Args:
            items: Dictionary of item_id to MovieItem
            queries: Dictionary of item_id to list of queries
            interactions: DataFrame of interactions to sample from
        """
        self.items = items
        self.queries = queries
        self.item_ids = list(items.keys())

        # Build (query, item_id) pairs from interactions
        self.pairs = []
        for item_id in interactions["item_id"].unique():
            if item_id in queries and queries[item_id]:
                for query in queries[item_id]:
                    self.pairs.append((query, item_id))

        logger.info(f"SearchDataset: {len(self.pairs)} query-item pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        query, item_id = self.pairs[idx]
        item = self.items[item_id]
        return query, item.text, item_id


class RecDataset(Dataset):
    """Dataset for recommendation task (item-item co-occurrence)."""

    def __init__(
        self,
        items: Dict[int, MovieItem],
        interactions: pd.DataFrame,
        window_size: int = 5,
    ):
        """
        Initialize the rec dataset.

        Args:
            items: Dictionary of item_id to MovieItem
            interactions: DataFrame with user_id, item_id, timestamp
            window_size: Window size for co-occurrence
        """
        self.items = items
        self.item_ids = list(items.keys())

        # Get co-occurrence pairs
        cooc = get_cooccurrence_pairs(interactions, window_size=window_size)
        self.pairs = cooc[["item1", "item2"]].values.tolist()

        # Filter to valid items
        valid_ids = set(items.keys())
        self.pairs = [
            (i1, i2) for i1, i2 in self.pairs if i1 in valid_ids and i2 in valid_ids
        ]

        logger.info(f"RecDataset: {len(self.pairs)} co-occurrence pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, int, int]:
        item_id1, item_id2 = self.pairs[idx]
        item1 = self.items[item_id1]
        item2 = self.items[item_id2]
        return item1.text, item2.text, item_id1, item_id2


class GenerativeDataset(Dataset):
    """Dataset for generative retrieval training."""

    def __init__(
        self,
        items: Dict[int, MovieItem],
        queries: Dict[int, List[str]],
        semantic_ids: Dict[int, List[int]],
        interactions: pd.DataFrame,
    ):
        """
        Initialize the generative dataset.

        Args:
            items: Dictionary of item_id to MovieItem
            queries: Dictionary of item_id to list of queries
            semantic_ids: Dictionary of item_id to semantic ID sequence
            interactions: DataFrame of interactions
        """
        self.items = items
        self.queries = queries
        self.semantic_ids = semantic_ids

        # Build (query, semantic_id) pairs
        self.pairs = []
        for item_id in interactions["item_id"].unique():
            if item_id in queries and item_id in semantic_ids:
                for query in queries[item_id]:
                    self.pairs.append((query, semantic_ids[item_id], item_id))

        logger.info(f"GenerativeDataset: {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, List[int], int]:
        query, sem_id, item_id = self.pairs[idx]
        return query, sem_id, item_id


def search_collate_fn(batch: List[Tuple]) -> SearchBatch:
    """Collate function for search batches."""
    queries, items, item_ids = zip(*batch)
    return SearchBatch(
        queries=list(queries),
        items=list(items),
        item_ids=torch.tensor(item_ids),
        labels=torch.arange(len(batch)),  # Diagonal positive
    )


def rec_collate_fn(batch: List[Tuple]) -> RecBatch:
    """Collate function for rec batches."""
    items1, items2, item_ids1, item_ids2 = zip(*batch)
    return RecBatch(
        items1=list(items1),
        items2=list(items2),
        item_ids1=torch.tensor(item_ids1),
        item_ids2=torch.tensor(item_ids2),
        labels=torch.arange(len(batch)),  # Diagonal positive
    )


def generative_collate_fn(batch: List[Tuple]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Collate function for generative batches."""
    queries, sem_ids, item_ids = zip(*batch)
    return (
        list(queries),
        torch.tensor(sem_ids),
        torch.tensor(item_ids),
    )


class SemanticIDDataModule(L.LightningDataModule):
    """Lightning DataModule for semantic ID training."""

    def __init__(
        self,
        items: Dict[int, MovieItem],
        queries: Dict[int, List[str]],
        train_interactions: pd.DataFrame,
        val_interactions: pd.DataFrame,
        test_interactions: pd.DataFrame,
        semantic_ids: Optional[Dict[int, List[int]]] = None,
        task: str = "multi_task",
        batch_size: int = 64,
        num_workers: int = 4,
        window_size: int = 5,
    ):
        """
        Initialize the DataModule.

        Args:
            items: Dictionary of item_id to MovieItem
            queries: Dictionary of item_id to list of queries
            train_interactions: Training interactions
            val_interactions: Validation interactions
            test_interactions: Test interactions
            semantic_ids: Optional semantic IDs for generative training
            task: Task type ("search", "rec", "multi_task", "generative")
            batch_size: Batch size
            num_workers: Number of data loading workers
            window_size: Window size for co-occurrence pairs
        """
        super().__init__()
        self.items = items
        self.queries = queries
        self.train_interactions = train_interactions
        self.val_interactions = val_interactions
        self.test_interactions = test_interactions
        self.semantic_ids = semantic_ids
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for the specified stage."""
        if stage in ("fit", None):
            if self.task in ("search", "multi_task"):
                self.train_search_dataset = SearchDataset(
                    items=self.items,
                    queries=self.queries,
                    interactions=self.train_interactions,
                )
                self.val_search_dataset = SearchDataset(
                    items=self.items,
                    queries=self.queries,
                    interactions=self.val_interactions,
                )

            if self.task in ("rec", "multi_task"):
                self.train_rec_dataset = RecDataset(
                    items=self.items,
                    interactions=self.train_interactions,
                    window_size=self.window_size,
                )
                self.val_rec_dataset = RecDataset(
                    items=self.items,
                    interactions=self.val_interactions,
                    window_size=self.window_size,
                )

            if self.task == "generative" and self.semantic_ids:
                self.train_gen_dataset = GenerativeDataset(
                    items=self.items,
                    queries=self.queries,
                    semantic_ids=self.semantic_ids,
                    interactions=self.train_interactions,
                )
                self.val_gen_dataset = GenerativeDataset(
                    items=self.items,
                    queries=self.queries,
                    semantic_ids=self.semantic_ids,
                    interactions=self.val_interactions,
                )

        if stage in ("test", None):
            if self.task in ("search", "multi_task"):
                self.test_search_dataset = SearchDataset(
                    items=self.items,
                    queries=self.queries,
                    interactions=self.test_interactions,
                )

            if self.task in ("rec", "multi_task"):
                self.test_rec_dataset = RecDataset(
                    items=self.items,
                    interactions=self.test_interactions,
                    window_size=self.window_size,
                )

            if self.task == "generative" and self.semantic_ids:
                self.test_gen_dataset = GenerativeDataset(
                    items=self.items,
                    queries=self.queries,
                    semantic_ids=self.semantic_ids,
                    interactions=self.test_interactions,
                )

    def train_dataloader(self):
        """Return training dataloader(s)."""
        if self.task == "search":
            return DataLoader(
                self.train_search_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=search_collate_fn,
                pin_memory=True,
            )
        elif self.task == "rec":
            return DataLoader(
                self.train_rec_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=rec_collate_fn,
                pin_memory=True,
            )
        elif self.task == "multi_task":
            # Return dict of dataloaders for multi-task
            return {
                "search": DataLoader(
                    self.train_search_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=search_collate_fn,
                    pin_memory=True,
                ),
                "rec": DataLoader(
                    self.train_rec_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=rec_collate_fn,
                    pin_memory=True,
                ),
            }
        elif self.task == "generative":
            return DataLoader(
                self.train_gen_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=generative_collate_fn,
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def val_dataloader(self):
        """Return validation dataloader(s)."""
        if self.task == "search":
            return DataLoader(
                self.val_search_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=search_collate_fn,
                pin_memory=True,
            )
        elif self.task == "rec":
            return DataLoader(
                self.val_rec_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=rec_collate_fn,
                pin_memory=True,
            )
        elif self.task == "multi_task":
            return {
                "search": DataLoader(
                    self.val_search_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=search_collate_fn,
                    pin_memory=True,
                ),
                "rec": DataLoader(
                    self.val_rec_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=rec_collate_fn,
                    pin_memory=True,
                ),
            }
        elif self.task == "generative":
            return DataLoader(
                self.val_gen_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=generative_collate_fn,
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def test_dataloader(self):
        """Return test dataloader(s)."""
        if self.task == "search":
            return DataLoader(
                self.test_search_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=search_collate_fn,
                pin_memory=True,
            )
        elif self.task == "rec":
            return DataLoader(
                self.test_rec_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=rec_collate_fn,
                pin_memory=True,
            )
        elif self.task == "multi_task":
            return {
                "search": DataLoader(
                    self.test_search_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=search_collate_fn,
                    pin_memory=True,
                ),
                "rec": DataLoader(
                    self.test_rec_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=rec_collate_fn,
                    pin_memory=True,
                ),
            }
        elif self.task == "generative":
            return DataLoader(
                self.test_gen_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=generative_collate_fn,
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
