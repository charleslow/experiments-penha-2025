"""Stage 1: Data Preparation — download and process MovieLens-25M.

Usage:
    python -m src.data.prepare --mode mini
    python -m src.data.prepare --mode dev
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_FILENAME = "ml-25m.zip"


# ── Download ──────────────────────────────────────────────────────────────────

def download_movielens(raw_dir: Path) -> Path:
    """Download MovieLens-25M zip if not already present."""
    zip_path = raw_dir / ZIP_FILENAME
    if zip_path.exists():
        log.info("Zip already downloaded: %s", zip_path)
        return zip_path

    log.info("Downloading MovieLens-25M from %s ...", MOVIELENS_URL)
    raw_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(MOVIELENS_URL, stream=True, timeout=600)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    log.info("Download complete: %s", zip_path)
    return zip_path


def extract_movielens(zip_path: Path, raw_dir: Path) -> Path:
    """Extract the zip if the expected directory doesn't exist."""
    extract_dir = raw_dir / "ml-25m"
    if extract_dir.exists() and (extract_dir / "movies.csv").exists():
        log.info("Already extracted: %s", extract_dir)
        return extract_dir

    log.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    log.info("Extraction complete: %s", extract_dir)
    return extract_dir


# ── Parsing helpers ───────────────────────────────────────────────────────────

def load_movies(ml_dir: Path) -> pd.DataFrame:
    """Load movies.csv."""
    log.info("Loading movies.csv ...")
    df = pd.read_csv(ml_dir / "movies.csv")
    log.info("  %d movies loaded", len(df))
    return df


def load_ratings(ml_dir: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Load ratings.csv in chunks to limit peak memory."""
    log.info("Loading ratings.csv in chunks (%d rows each) ...", chunksize)
    chunks = []
    for chunk in pd.read_csv(ml_dir / "ratings.csv", chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    log.info("  %d ratings loaded", len(df))
    return df


def load_tags(ml_dir: Path) -> pd.DataFrame:
    """Load tags.csv."""
    log.info("Loading tags.csv ...")
    df = pd.read_csv(ml_dir / "tags.csv")
    log.info("  %d tags loaded", len(df))
    return df


def load_genome_scores(ml_dir: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Load genome-scores.csv in chunks (large file, ~500 MB)."""
    log.info("Loading genome-scores.csv in chunks (%d rows each) ...", chunksize)
    chunks = []
    for chunk in pd.read_csv(ml_dir / "genome-scores.csv", chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    log.info("  %d genome score rows loaded", len(df))
    return df


def load_genome_tags(ml_dir: Path) -> pd.DataFrame:
    """Load genome-tags.csv."""
    log.info("Loading genome-tags.csv ...")
    df = pd.read_csv(ml_dir / "genome-tags.csv")
    log.info("  %d genome tags loaded", len(df))
    return df


# ── Processing ────────────────────────────────────────────────────────────────

def filter_and_subsample(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    n_movies: int,
    min_interactions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to movies with enough interactions, then subsample to n_movies."""

    # Count interactions per movie
    movie_counts = ratings.groupby("movieId").size().reset_index(name="n_ratings")
    qualifying = movie_counts[movie_counts["n_ratings"] >= min_interactions]
    log.info(
        "  %d / %d movies have >= %d ratings",
        len(qualifying), len(movies), min_interactions,
    )

    qualifying_ids = set(qualifying["movieId"])

    # Subsample if needed
    if n_movies > 0 and len(qualifying_ids) > n_movies:
        rng = np.random.RandomState(seed)
        sampled_ids = set(rng.choice(sorted(qualifying_ids), size=n_movies, replace=False))
    else:
        sampled_ids = qualifying_ids

    movies_filtered = movies[movies["movieId"].isin(sampled_ids)].copy()
    ratings_filtered = ratings[ratings["movieId"].isin(sampled_ids)].copy()

    log.info(
        "  After subsampling: %d movies, %d ratings",
        len(movies_filtered), len(ratings_filtered),
    )
    return movies_filtered, ratings_filtered


def filter_users(
    ratings: pd.DataFrame,
    min_interactions: int,
) -> pd.DataFrame:
    """Remove users with fewer than min_interactions."""
    user_counts = ratings.groupby("userId").size()
    keep_users = user_counts[user_counts >= min_interactions].index
    filtered = ratings[ratings["userId"].isin(keep_users)].copy()
    log.info(
        "  After user filter (>= %d): %d ratings from %d users",
        min_interactions, len(filtered), filtered["userId"].nunique(),
    )
    return filtered


def chronological_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split: last interaction per user goes to test, rest to train."""
    ratings_sorted = ratings.sort_values(["userId", "timestamp"])

    # Index of last interaction per user
    last_idx = ratings_sorted.groupby("userId")["timestamp"].idxmax()
    test_mask = ratings_sorted.index.isin(last_idx)

    train = ratings_sorted[~test_mask].copy()
    test = ratings_sorted[test_mask].copy()

    log.info("  Train: %d ratings, Test: %d ratings", len(train), len(test))
    return train, test


def build_genome_features(
    genome_scores: pd.DataFrame,
    genome_tags: pd.DataFrame,
    movie_ids: set[int],
    top_k: int = 10,
) -> pd.DataFrame:
    """For each movie, get top-k genome tags by relevance score."""
    # Filter to our movies
    gs = genome_scores[genome_scores["movieId"].isin(movie_ids)].copy()

    # Merge tag names
    gs = gs.merge(genome_tags, on="tagId", how="left")

    # For each movie, keep top-k tags by relevance
    gs = gs.sort_values(["movieId", "relevance"], ascending=[True, False])
    top_tags = gs.groupby("movieId").head(top_k)

    # Aggregate into a single string per movie
    tag_strings = (
        top_tags
        .groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x))
        .reset_index()
        .rename(columns={"tag": "top_genome_tags"})
    )

    log.info("  Genome tag features built for %d movies", len(tag_strings))
    return tag_strings


# ── Main ──────────────────────────────────────────────────────────────────────

def main(mode: str = "dev"):
    cfg = get_config(mode)
    log.info("=== Stage 1: Data Preparation (mode=%s, n_movies=%d) ===", mode, cfg.n_movies)

    np.random.seed(cfg.seed)

    # Download and extract
    zip_path = download_movielens(cfg.data_raw_dir)
    ml_dir = extract_movielens(zip_path, cfg.data_raw_dir)

    # Load raw data
    movies = load_movies(ml_dir)
    ratings = load_ratings(ml_dir, chunksize=cfg.csv_chunksize)
    tags = load_tags(ml_dir)
    genome_tags = load_genome_tags(ml_dir)

    # Filter and subsample movies
    movies_f, ratings_f = filter_and_subsample(
        movies, ratings,
        n_movies=cfg.n_movies,
        min_interactions=cfg.min_interactions_per_movie,
        seed=cfg.seed,
    )

    # Filter users
    ratings_f = filter_users(ratings_f, min_interactions=cfg.min_interactions_per_user)

    # Re-filter movies to only those still in ratings after user filtering
    remaining_movie_ids = set(ratings_f["movieId"].unique())
    movies_f = movies_f[movies_f["movieId"].isin(remaining_movie_ids)].copy()
    log.info("  Movies remaining after user filter: %d", len(movies_f))

    # Chronological train/test split
    train_ratings, test_ratings = chronological_split(ratings_f)

    # Build genome tag features (load genome-scores only for our subset)
    log.info("Loading genome scores for selected movies ...")
    genome_scores_all = load_genome_scores(ml_dir, chunksize=cfg.csv_chunksize)
    genome_features = build_genome_features(
        genome_scores_all, genome_tags, remaining_movie_ids
    )
    # Free memory
    del genome_scores_all

    # Merge genome tags into movies
    movies_final = movies_f.merge(genome_features, on="movieId", how="left")
    movies_final["top_genome_tags"] = movies_final["top_genome_tags"].fillna("")

    # Also keep user tags (aggregate per movie)
    movie_tags = (
        tags[tags["movieId"].isin(remaining_movie_ids)]
        .groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.dropna().unique()[:20]))
        .reset_index()
        .rename(columns={"tag": "user_tags"})
    )
    movies_final = movies_final.merge(movie_tags, on="movieId", how="left")
    movies_final["user_tags"] = movies_final["user_tags"].fillna("")

    # Save processed data
    out_dir = cfg.data_processed_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    movies_final.to_parquet(out_dir / "movies.parquet", index=False)
    train_ratings.to_parquet(out_dir / "train_ratings.parquet", index=False)
    test_ratings.to_parquet(out_dir / "test_ratings.parquet", index=False)

    log.info("Saved processed data to %s", out_dir)
    log.info("  movies: %d rows", len(movies_final))
    log.info("  train_ratings: %d rows", len(train_ratings))
    log.info("  test_ratings: %d rows", len(test_ratings))
    log.info("=== Stage 1 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Data Preparation")
    parser.add_argument("--mode", type=str, default="dev", choices=["mini", "dev", "full"],
                        help="Run mode: mini, dev, or full")
    args = parser.parse_args()
    main(args.mode)
