"""MovieLens dataset loading and processing."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MovieItem:
    """Represents a movie item with metadata."""

    item_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None

    @property
    def text(self) -> str:
        """Get text representation for encoding."""
        genres_str = ", ".join(self.genres) if self.genres else "Unknown"
        if self.year:
            return f"{self.title} ({self.year}). Genres: {genres_str}"
        return f"{self.title}. Genres: {genres_str}"


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens interactions."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: Dict[int, MovieItem],
        queries: Optional[Dict[int, List[str]]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            interactions: DataFrame with columns [user_id, item_id, rating, timestamp]
            items: Dictionary mapping item_id to MovieItem
            queries: Optional dictionary mapping item_id to list of queries
        """
        self.interactions = interactions.reset_index(drop=True)
        self.items = items
        self.queries = queries or {}

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Dict:
        row = self.interactions.iloc[idx]
        item_id = int(row["item_id"])
        item = self.items.get(item_id)

        result = {
            "user_id": int(row["user_id"]),
            "item_id": item_id,
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
        }

        if item:
            result["item_text"] = item.text
            result["title"] = item.title
            result["genres"] = item.genres

        if item_id in self.queries and self.queries[item_id]:
            # Return a random query for this item
            result["query"] = np.random.choice(self.queries[item_id])

        return result

    @property
    def item_ids(self) -> List[int]:
        """Get unique item IDs in the dataset."""
        return self.interactions["item_id"].unique().tolist()

    @property
    def user_ids(self) -> List[int]:
        """Get unique user IDs in the dataset."""
        return self.interactions["user_id"].unique().tolist()


def parse_title_year(title: str) -> Tuple[str, Optional[int]]:
    """Parse movie title and year from MovieLens format."""
    import re

    match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", title.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


def load_movielens(
    data_dir: Path,
    fraction: float = 1.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, MovieItem]]:
    """
    Load and preprocess MovieLens dataset.

    Args:
        data_dir: Path to the MovieLens data directory
        fraction: Fraction of data to sample (for dev runs)
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        random_state: Random seed for sampling

    Returns:
        Tuple of (interactions DataFrame, items dictionary)
    """
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"MovieLens data not found at {data_dir}. "
            "Please run scripts/download_data.py first."
        )

    logger.info(f"Loading MovieLens data from {data_dir}")

    # Load ratings
    ratings = pd.read_csv(ratings_path)
    ratings.columns = ["user_id", "item_id", "rating", "timestamp"]
    logger.info(f"Loaded {len(ratings)} ratings")

    # Load movies
    movies = pd.read_csv(movies_path)
    movies.columns = ["item_id", "title", "genres"]

    # Parse items
    items = {}
    for _, row in movies.iterrows():
        title, year = parse_title_year(row["title"])
        genres = row["genres"].split("|") if row["genres"] != "(no genres listed)" else []
        items[row["item_id"]] = MovieItem(
            item_id=row["item_id"],
            title=title,
            genres=genres,
            year=year,
        )
    logger.info(f"Loaded {len(items)} items")

    # Sample if fraction < 1
    if fraction < 1.0:
        n_sample = int(len(ratings) * fraction)
        ratings = ratings.sample(n=n_sample, random_state=random_state)
        logger.info(f"Sampled {len(ratings)} ratings ({fraction*100:.1f}%)")

    # Filter by minimum interactions
    if min_user_interactions > 1 or min_item_interactions > 1:
        original_len = len(ratings)
        for _ in range(5):  # Iterative filtering
            user_counts = ratings["user_id"].value_counts()
            item_counts = ratings["item_id"].value_counts()

            valid_users = user_counts[user_counts >= min_user_interactions].index
            valid_items = item_counts[item_counts >= min_item_interactions].index

            ratings = ratings[
                ratings["user_id"].isin(valid_users) & ratings["item_id"].isin(valid_items)
            ]

            if len(ratings) == original_len:
                break
            original_len = len(ratings)

        logger.info(f"After filtering: {len(ratings)} ratings")

    # Filter items to only those in interactions
    valid_item_ids = set(ratings["item_id"].unique())
    items = {k: v for k, v in items.items() if k in valid_item_ids}
    logger.info(f"After filtering: {len(items)} items")

    return ratings, items


def chronological_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions chronologically.

    Args:
        interactions: DataFrame with timestamp column
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    # Sort by timestamp
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    n = len(interactions)
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))

    train = interactions.iloc[:train_end]
    val = interactions.iloc[train_end:val_end]
    test = interactions.iloc[val_end:]

    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
    )

    return train, val, test


def get_user_item_matrix(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Create user-item interaction matrix.

    Args:
        interactions: DataFrame with user_id, item_id columns

    Returns:
        Tuple of (matrix, user_id_map, item_id_map)
    """
    user_ids = interactions["user_id"].unique()
    item_ids = interactions["item_id"].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

    matrix = pd.DataFrame(
        0,
        index=range(len(user_ids)),
        columns=range(len(item_ids)),
    )

    for _, row in interactions.iterrows():
        u_idx = user_id_map[row["user_id"]]
        i_idx = item_id_map[row["item_id"]]
        matrix.iloc[u_idx, i_idx] = 1

    return matrix, user_id_map, item_id_map


def get_cooccurrence_pairs(
    interactions: pd.DataFrame,
    window_size: int = 5,
) -> pd.DataFrame:
    """
    Get item co-occurrence pairs within user sessions.

    Args:
        interactions: DataFrame sorted by timestamp within users
        window_size: Maximum distance between co-occurring items

    Returns:
        DataFrame with columns [item1, item2, count]
    """
    # Sort by user and timestamp
    interactions = interactions.sort_values(["user_id", "timestamp"])

    pairs = []
    for user_id, group in interactions.groupby("user_id"):
        items = group["item_id"].tolist()
        for i, item1 in enumerate(items):
            for j in range(i + 1, min(i + window_size + 1, len(items))):
                item2 = items[j]
                if item1 != item2:
                    pairs.append((min(item1, item2), max(item1, item2)))

    pair_counts = pd.DataFrame(pairs, columns=["item1", "item2"])
    pair_counts = pair_counts.groupby(["item1", "item2"]).size().reset_index(name="count")

    logger.info(f"Generated {len(pair_counts)} co-occurrence pairs")
    return pair_counts
