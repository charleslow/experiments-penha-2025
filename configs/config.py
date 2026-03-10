"""Configuration system for the Penha 2025 replication experiments.

Supports three run modes:
  - mini: 200 movies, for pipeline validation
  - dev:  2000 movies, for local 16GB CPU runs
  - full: TBD, for modal deployment
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Run mode identifier
    mode: str = "dev"

    # Data sizing
    n_movies: int = 2000
    min_interactions_per_movie: int = 50
    min_interactions_per_user: int = 5

    # Query generation
    n_queries_per_movie: int = 6
    n_train_queries: int = 3
    n_test_queries: int = 3
    query_backend: str = "template"  # "template" (fast, no LLM) or "ollama"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_url: str = "http://localhost:11434"

    # Training
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Embeddings
    embedding_dim: int = 768
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # RQ-VAE / RQ-kmeans
    n_codebooks: int = 4
    codebook_size: int = 256

    # Generative model
    beam_size: int = 20
    max_seq_len: int = 512

    # Reproducibility
    seed: int = 42

    # Paths (derived from project root)
    data_raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    data_queries_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "queries")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    # Memory management
    csv_chunksize: int = 500_000

    def __post_init__(self):
        """Create directories if they don't exist."""
        for d in [self.data_raw_dir, self.data_processed_dir,
                  self.data_queries_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Preset configurations ────────────────────────────────────────────────────

CONFIGS = {
    "mini": Config(
        mode="mini",
        n_movies=200,
        n_queries_per_movie=6,
        n_train_queries=3,
        n_test_queries=3,
        batch_size=32,
        eval_batch_size=64,
        epochs=5,
    ),
    "dev": Config(
        mode="dev",
        n_movies=2000,
        n_queries_per_movie=6,
        n_train_queries=3,
        n_test_queries=3,
        batch_size=64,
        eval_batch_size=128,
        epochs=20,
    ),
    "full": Config(
        mode="full",
        n_movies=-1,  # TBD — use all qualifying movies
        n_queries_per_movie=6,
        n_train_queries=3,
        n_test_queries=3,
        query_backend="ollama",  # full run uses LLM for quality
        batch_size=128,
        eval_batch_size=256,
        epochs=50,
    ),
}


def get_config(mode: str = "dev") -> Config:
    """Return a Config for the given mode.

    Args:
        mode: One of 'mini', 'dev', 'full'.

    Returns:
        A Config dataclass instance.
    """
    if mode not in CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(CONFIGS.keys())}")
    return CONFIGS[mode]
