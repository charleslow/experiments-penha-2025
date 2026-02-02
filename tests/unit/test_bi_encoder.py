"""Unit tests for bi-encoder models."""

import pytest
import torch

from src.models.bi_encoder import BiEncoderModule, ContrastiveLoss
from src.data.datamodule import SearchBatch, RecBatch


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    def test_loss_computation(self):
        loss_fn = ContrastiveLoss(temperature=0.07)

        # Create dummy embeddings
        batch_size = 4
        embedding_dim = 64
        query_emb = torch.randn(batch_size, embedding_dim)
        key_emb = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(query_emb, key_emb)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)

    def test_loss_with_labels(self):
        loss_fn = ContrastiveLoss(temperature=0.07)

        batch_size = 4
        embedding_dim = 64
        query_emb = torch.randn(batch_size, embedding_dim)
        key_emb = torch.randn(batch_size, embedding_dim)
        labels = torch.arange(batch_size)

        loss = loss_fn(query_emb, key_emb, labels)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_loss_decreases_for_similar_embeddings(self):
        loss_fn = ContrastiveLoss(temperature=0.07)

        # Similar embeddings should have lower loss
        batch_size = 4
        embedding_dim = 64

        # Random embeddings
        query_random = torch.randn(batch_size, embedding_dim)
        key_random = torch.randn(batch_size, embedding_dim)
        loss_random = loss_fn(query_random, key_random)

        # Similar embeddings (query = key + small noise)
        query_similar = torch.randn(batch_size, embedding_dim)
        key_similar = query_similar + 0.01 * torch.randn(batch_size, embedding_dim)
        loss_similar = loss_fn(query_similar, key_similar)

        assert loss_similar < loss_random


class TestBiEncoderModule:
    """Tests for BiEncoderModule."""

    @pytest.fixture
    def model(self):
        return BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="search",
            temperature=0.07,
            learning_rate=2e-5,
        )

    def test_initialization(self, model):
        assert model.model_name == "all-MiniLM-L6-v2"
        assert model.task == "search"
        assert model.embedding_dim > 0

    def test_encode(self, model):
        texts = ["Hello world", "Test sentence"]
        embeddings = model.encode(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.embedding_dim

    def test_forward(self, model):
        texts = ["Hello world", "Test sentence"]
        embeddings = model(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.embedding_dim

    def test_compute_search_loss(self, model):
        batch = SearchBatch(
            queries=["Find action movie", "Comedy film"],
            items=["The Matrix - Action Sci-Fi", "Toy Story - Animation Comedy"],
            item_ids=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]),
        )

        loss = model.compute_search_loss(batch)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_get_item_embeddings(self, model):
        items = ["Movie 1 - Action", "Movie 2 - Drama", "Movie 3 - Comedy"]
        embeddings = model.get_item_embeddings(items, batch_size=2)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == model.embedding_dim


class TestBiEncoderTasks:
    """Tests for different bi-encoder tasks."""

    def test_search_task(self):
        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="search",
        )
        assert model.task == "search"

    def test_rec_task(self):
        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="rec",
        )
        assert model.task == "rec"

    def test_multi_task(self):
        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="multi_task",
        )
        assert model.task == "multi_task"

    def test_compute_rec_loss(self):
        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="rec",
        )

        batch = RecBatch(
            items1=["The Matrix - Action Sci-Fi", "Inception - Sci-Fi Thriller"],
            items2=["The Matrix Reloaded - Action Sci-Fi", "Interstellar - Sci-Fi Drama"],
            item_ids1=torch.tensor([1, 2]),
            item_ids2=torch.tensor([3, 4]),
            labels=torch.tensor([0, 1]),
        )

        loss = model.compute_rec_loss(batch)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestBiEncoderOptimizer:
    """Tests for optimizer configuration."""

    def test_configure_optimizers(self):
        model = BiEncoderModule(
            model_name="all-MiniLM-L6-v2",
            task="search",
            learning_rate=1e-4,
            warmup_steps=10,
        )

        config = model.configure_optimizers()

        assert "optimizer" in config
        assert "lr_scheduler" in config
