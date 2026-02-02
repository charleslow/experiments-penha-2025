"""Integration tests for the dev run pipeline."""

import pytest
import sys
import json
from pathlib import Path
import tempfile
import shutil

import torch
import pandas as pd


class TestDevRunPipeline:
    """Integration tests for the full dev run pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from src.data.movielens import MovieItem

        # Create items
        items = {}
        for i in range(50):
            items[i] = MovieItem(
                item_id=i,
                title=f"Test Movie {i}",
                genres=["Action", "Drama"][i % 2:i % 2 + 1],
                year=2000 + i,
            )

        # Create interactions
        interactions = pd.DataFrame({
            "user_id": list(range(100)) * 5,
            "item_id": [i % 50 for i in range(500)],
            "rating": [4.0] * 500,
            "timestamp": list(range(500)),
        })

        return items, interactions

    def test_data_loading(self, sample_data):
        """Test that data loading works."""
        from src.data.movielens import chronological_split

        items, interactions = sample_data

        train, val, test = chronological_split(
            interactions,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(interactions)

    def test_query_generation(self, sample_data):
        """Test synthetic query generation."""
        from src.data.query_generator import generate_synthetic_queries

        items, _ = sample_data
        queries = generate_synthetic_queries(items, n_queries=2)

        assert len(queries) == len(items)
        for item_id, item_queries in queries.items():
            assert len(item_queries) == 2

    def test_bi_encoder_embedding(self, sample_data):
        """Test bi-encoder creates valid embeddings."""
        from src.models.bi_encoder import BiEncoderModule

        items, _ = sample_data

        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="search",
        )

        item_texts = [items[i].text for i in list(items.keys())[:10]]
        embeddings = model.get_item_embeddings(item_texts, batch_size=4)

        assert embeddings.shape[0] == 10
        assert embeddings.shape[1] > 0
        assert torch.all(torch.isfinite(embeddings))

    def test_discretization_pipeline(self, sample_data):
        """Test discretization creates valid codes."""
        from src.models.bi_encoder import BiEncoderModule
        from src.discretization.rq_kmeans import RQKMeansDiscretizer

        items, _ = sample_data

        # Get embeddings
        model = BiEncoderModule(model_name="all-MiniLM-L6-v2", task="search")
        item_texts = [items[i].text for i in list(items.keys())[:20]]
        embeddings = model.get_item_embeddings(item_texts, batch_size=8)

        # Discretize
        discretizer = RQKMeansDiscretizer(
            n_hierarchies=2,
            codebook_size=8,
        )
        codes = discretizer.fit_encode(embeddings)

        assert codes.shape == (20, 2)
        assert codes.min() >= 0
        assert codes.max() < 8

    def test_metrics_computation(self):
        """Test that metrics can be computed."""
        from src.evaluation.metrics import evaluate_retrieval

        # Create dummy embeddings and relevance
        query_emb = torch.randn(5, 64)
        item_emb = torch.randn(10, 64)
        relevance = torch.zeros(5, 10)
        for i in range(5):
            relevance[i, i % 10] = 1

        results = evaluate_retrieval(
            query_emb, item_emb, relevance,
            k_values=[1, 5],
        )

        assert "NDCG@1" in results
        assert "NDCG@5" in results
        assert all(0 <= v <= 1 for v in results.values())

    def test_plot_generation(self, temp_output_dir):
        """Test that plots can be generated."""
        from src.visualization.plots import (
            plot_embedding_ablation,
            plot_discretization_ablation,
        )

        # Test embedding ablation plot
        embedding_results = {
            "search": {"NDCG@10": 0.3, "Recall@10": 0.4},
            "rec": {"NDCG@10": 0.35, "Recall@10": 0.45},
            "multi_task": {"NDCG@10": 0.38, "Recall@10": 0.48},
        }

        plot_path = temp_output_dir / "embedding_ablation.png"
        plot_embedding_ablation(embedding_results, plot_path)
        assert plot_path.exists()

        # Test discretization ablation plot
        disc_results = {
            "rq_kmeans": {"NDCG@10": 0.3, "Recall@10": 0.4},
            "lsh": {"NDCG@10": 0.25, "Recall@10": 0.35},
        }

        plot_path = temp_output_dir / "disc_ablation.png"
        plot_discretization_ablation(disc_results, plot_path)
        assert plot_path.exists()

    def test_end_to_end_mini_pipeline(self, sample_data, temp_output_dir):
        """Test a minimal end-to-end pipeline."""
        from src.data.movielens import chronological_split
        from src.data.query_generator import generate_synthetic_queries
        from src.models.bi_encoder import BiEncoderModule
        from src.discretization.rq_kmeans import RQKMeansDiscretizer
        from src.evaluation.metrics import evaluate_retrieval

        items, interactions = sample_data

        # 1. Split data
        train, val, test = chronological_split(interactions)

        # 2. Generate queries
        queries = generate_synthetic_queries(items, n_queries=2)

        # 3. Get embeddings
        model = BiEncoderModule(model_name="all-MiniLM-L6-v2", task="search")
        item_ids = list(items.keys())[:20]
        item_texts = [items[i].text for i in item_ids]
        embeddings = model.get_item_embeddings(item_texts, batch_size=8)

        # 4. Discretize
        discretizer = RQKMeansDiscretizer(n_hierarchies=2, codebook_size=8)
        codes = discretizer.fit_encode(embeddings)

        # 5. Evaluate (using embeddings as proxy)
        # Create dummy relevance matrix
        relevance = torch.zeros(5, 20)
        for i in range(5):
            relevance[i, i % 20] = 1

        results = evaluate_retrieval(
            embeddings[:5],  # queries
            embeddings,  # items
            relevance,
            k_values=[5, 10],
        )

        # Verify results
        assert "NDCG@10" in results
        assert results["NDCG@10"] >= 0

        # Save results
        results_path = temp_output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)
        assert results_path.exists()


class TestDataModuleIntegration:
    """Integration tests for the data module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from src.data.movielens import MovieItem

        items = {i: MovieItem(item_id=i, title=f"Movie {i}", genres=["Action"], year=2000)
                 for i in range(20)}

        queries = {i: [f"Find movie {i}", f"Looking for movie {i}"] for i in range(20)}

        interactions = pd.DataFrame({
            "user_id": list(range(50)) * 2,
            "item_id": [i % 20 for i in range(100)],
            "rating": [4.0] * 100,
            "timestamp": list(range(100)),
        })

        return items, queries, interactions

    def test_search_dataloader(self, sample_data):
        """Test search task dataloader."""
        from src.data.movielens import chronological_split
        from src.data.datamodule import SemanticIDDataModule

        items, queries, interactions = sample_data
        train, val, test = chronological_split(interactions)

        dm = SemanticIDDataModule(
            items=items,
            queries=queries,
            train_interactions=train,
            val_interactions=val,
            test_interactions=test,
            task="search",
            batch_size=8,
            num_workers=0,
        )

        dm.setup("fit")
        train_dl = dm.train_dataloader()

        batch = next(iter(train_dl))
        assert hasattr(batch, "queries")
        assert hasattr(batch, "items")
        assert len(batch.queries) <= 8

    def test_rec_dataloader(self, sample_data):
        """Test rec task dataloader."""
        from src.data.movielens import chronological_split
        from src.data.datamodule import SemanticIDDataModule

        items, queries, interactions = sample_data
        train, val, test = chronological_split(interactions)

        dm = SemanticIDDataModule(
            items=items,
            queries=queries,
            train_interactions=train,
            val_interactions=val,
            test_interactions=test,
            task="rec",
            batch_size=8,
            num_workers=0,
        )

        dm.setup("fit")
        train_dl = dm.train_dataloader()

        batch = next(iter(train_dl))
        assert hasattr(batch, "items1")
        assert hasattr(batch, "items2")


class TestConfigIntegration:
    """Test configuration classes."""

    def test_dev_config_conversions(self):
        """Test DevConfig conversion methods."""
        from src.config import DevConfig

        config = DevConfig()

        data_config = config.to_data_config()
        assert data_config.data_fraction == config.data_fraction

        bi_config = config.to_bi_encoder_config(task="search")
        assert bi_config.model_name == config.encoder_model
        assert bi_config.task == "search"

        disc_config = config.to_discretization_config(method="rq_kmeans")
        assert disc_config.method == "rq_kmeans"
        assert disc_config.n_hierarchies == config.n_hierarchies
