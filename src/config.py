"""Configuration dataclasses for the semantic ID replication."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    data_dir: Path = Path("/app/data")
    dataset_name: str = "ml-25m"
    data_fraction: float = 1.0
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    random_state: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / self.dataset_name

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def queries_dir(self) -> Path:
        return self.data_dir / "queries"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def semantic_ids_dir(self) -> Path:
        return self.data_dir / "semantic_ids"


@dataclass
class BiEncoderConfig:
    """Configuration for bi-encoder training."""

    model_name: str = "all-MiniLM-L6-v2"
    task: str = "multi_task"  # "search", "rec", or "multi_task"
    embedding_dim: int = 384
    max_seq_length: int = 128
    batch_size: int = 64
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    temperature: float = 0.07
    num_negatives: int = 7
    search_loss_weight: float = 1.0
    rec_loss_weight: float = 1.0


@dataclass
class DiscretizationConfig:
    """Configuration for embedding discretization."""

    method: str = "rq_kmeans"  # "rq_kmeans", "rq_vae", "lsh", "pq"
    n_hierarchies: int = 3
    codebook_size: int = 256
    n_bits: int = 8  # For LSH
    n_subquantizers: int = 8  # For PQ
    normalize_residuals: bool = True
    init_buffer_size: int = 1000


@dataclass
class GenerativeConfig:
    """Configuration for generative retrieval model."""

    model_name: str = "google/flan-t5-small"
    max_input_length: int = 128
    max_output_length: int = 32
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    num_beams: int = 10
    top_k: int = 10


@dataclass
class QueryGeneratorConfig:
    """Configuration for query generation."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    queries_per_item: int = 3
    max_new_tokens: int = 64
    temperature: float = 0.7
    batch_size: int = 8


@dataclass
class DevConfig:
    """Configuration for dev/debug runs."""

    # Data
    data_fraction: float = 0.01
    queries_per_item: int = 2

    # Bi-encoder
    encoder_model: str = "all-MiniLM-L6-v2"
    encoder_epochs: int = 1
    encoder_batch: int = 64

    # Discretization
    n_hierarchies: int = 2
    codebook_size: int = 64

    # Generative
    gen_model: str = "google/flan-t5-small"
    gen_epochs: int = 1
    gen_batch: int = 16

    # Misc
    seed: int = 42
    output_dir: Path = Path("results/dev_run")

    def to_data_config(self) -> DataConfig:
        return DataConfig(data_fraction=self.data_fraction)

    def to_bi_encoder_config(self, task: str = "multi_task") -> BiEncoderConfig:
        return BiEncoderConfig(
            model_name=self.encoder_model,
            task=task,
            batch_size=self.encoder_batch,
            num_epochs=self.encoder_epochs,
        )

    def to_discretization_config(self, method: str = "rq_kmeans") -> DiscretizationConfig:
        return DiscretizationConfig(
            method=method,
            n_hierarchies=self.n_hierarchies,
            codebook_size=self.codebook_size,
        )

    def to_generative_config(self) -> GenerativeConfig:
        return GenerativeConfig(
            model_name=self.gen_model,
            batch_size=self.gen_batch,
            num_epochs=self.gen_epochs,
        )

    def to_query_config(self) -> QueryGeneratorConfig:
        return QueryGeneratorConfig(queries_per_item=self.queries_per_item)


@dataclass
class FullConfig:
    """Configuration for full experiment runs."""

    # Data
    data_fraction: float = 1.0
    queries_per_item: int = 5

    # Bi-encoder
    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    encoder_epochs: int = 5
    encoder_batch: int = 128

    # Discretization
    n_hierarchies: int = 4
    codebook_size: int = 256

    # Generative
    gen_model: str = "google/flan-t5-base"
    gen_epochs: int = 5
    gen_batch: int = 32

    # Experiment
    n_runs: int = 5
    seed: int = 42
    output_dir: Path = Path("results/full_run")

    def to_data_config(self) -> DataConfig:
        return DataConfig(data_fraction=self.data_fraction)

    def to_bi_encoder_config(self, task: str = "multi_task") -> BiEncoderConfig:
        return BiEncoderConfig(
            model_name=self.encoder_model,
            task=task,
            batch_size=self.encoder_batch,
            num_epochs=self.encoder_epochs,
        )

    def to_discretization_config(self, method: str = "rq_kmeans") -> DiscretizationConfig:
        return DiscretizationConfig(
            method=method,
            n_hierarchies=self.n_hierarchies,
            codebook_size=self.codebook_size,
        )

    def to_generative_config(self) -> GenerativeConfig:
        return GenerativeConfig(
            model_name=self.gen_model,
            batch_size=self.gen_batch,
            num_epochs=self.gen_epochs,
        )

    def to_query_config(self) -> QueryGeneratorConfig:
        return QueryGeneratorConfig(queries_per_item=self.queries_per_item)


@dataclass
class ExperimentResults:
    """Container for experiment results.

    Primary metric is Recall@30 to match Penha et al. 2025 paper.
    """

    embedding_strategy: str
    discretization_method: str
    search_recall_30: float  # Primary metric (paper uses R@30)
    rec_recall_30: float  # Primary metric (paper uses R@30)
    run_id: int = 0


@dataclass
class AggregatedResults:
    """Container for aggregated results across runs.

    Primary metric is Recall@30 to match Penha et al. 2025 paper.
    """

    embedding_strategy: str
    discretization_method: str
    search_recall_30_mean: float
    search_recall_30_std: float
    rec_recall_30_mean: float
    rec_recall_30_std: float
    n_runs: int
