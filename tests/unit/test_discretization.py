"""Unit tests for discretization methods."""

import pytest
import torch
import numpy as np

from src.discretization.base import BaseDiscretizer
from src.discretization.rq_kmeans import RQKMeansDiscretizer
from src.discretization.lsh import LSHDiscretizer


class TestBaseDiscretizer:
    """Tests for BaseDiscretizer interface."""

    def test_is_abstract(self):
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseDiscretizer()


class TestRQKMeansDiscretizer:
    """Tests for RQKMeansDiscretizer."""

    @pytest.fixture
    def discretizer(self):
        return RQKMeansDiscretizer(
            n_hierarchies=2,
            codebook_size=16,
            normalize_residuals=True,
        )

    @pytest.fixture
    def sample_embeddings(self):
        # Create some clustered embeddings
        torch.manual_seed(42)
        n_samples = 100
        embedding_dim = 64
        return torch.randn(n_samples, embedding_dim)

    def test_initialization(self, discretizer):
        assert discretizer.n_hierarchies == 2
        assert discretizer.codebook_size == 16
        assert not discretizer.is_fitted

    def test_fit(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)

        assert discretizer.is_fitted
        assert len(discretizer.centroids) == 2
        assert discretizer.centroids[0].shape == (16, 64)

    def test_encode(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        codes = discretizer.encode(sample_embeddings)

        assert codes.shape == (100, 2)
        assert codes.dtype == torch.int64
        assert codes.min() >= 0
        assert codes.max() < 16

    def test_decode(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        codes = discretizer.encode(sample_embeddings)
        reconstructed = discretizer.decode(codes)

        assert reconstructed.shape == sample_embeddings.shape

    def test_reconstruction_error(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        mse = discretizer.reconstruction_error(sample_embeddings)

        assert mse >= 0
        assert np.isfinite(mse)

    def test_fit_encode(self, discretizer, sample_embeddings):
        codes = discretizer.fit_encode(sample_embeddings)

        assert discretizer.is_fitted
        assert codes.shape == (100, 2)

    def test_encode_single_sample(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        single = sample_embeddings[0]
        codes = discretizer.encode(single)

        assert codes.shape == (1, 2)

    def test_state_dict(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        state = discretizer.state_dict()

        assert "n_hierarchies" in state
        assert "codebook_size" in state
        assert "centroids" in state
        assert len(state["centroids"]) == 2

    def test_load_state_dict(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        state = discretizer.state_dict()

        # Create new discretizer and load state
        new_discretizer = RQKMeansDiscretizer(n_hierarchies=2, codebook_size=16)
        new_discretizer.load_state_dict(state)

        assert new_discretizer.is_fitted
        assert len(new_discretizer.centroids) == 2

        # Should produce same codes
        codes1 = discretizer.encode(sample_embeddings)
        codes2 = new_discretizer.encode(sample_embeddings)
        assert torch.allclose(codes1, codes2)


class TestLSHDiscretizer:
    """Tests for LSHDiscretizer."""

    @pytest.fixture
    def discretizer(self):
        return LSHDiscretizer(
            n_hierarchies=2,
            codebook_size=16,  # 4 bits per hierarchy
            random_state=42,
        )

    @pytest.fixture
    def sample_embeddings(self):
        torch.manual_seed(42)
        n_samples = 100
        embedding_dim = 64
        return torch.randn(n_samples, embedding_dim)

    def test_initialization(self, discretizer):
        assert discretizer.n_hierarchies == 2
        assert discretizer.codebook_size == 16
        assert discretizer.n_bits_per_hierarchy == 4
        assert not discretizer.is_fitted

    def test_fit(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)

        assert discretizer.is_fitted
        assert len(discretizer.projections) == 2
        assert discretizer.projections[0].shape == (64, 4)

    def test_encode(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        codes = discretizer.encode(sample_embeddings)

        assert codes.shape == (100, 2)
        assert codes.dtype == torch.int64
        assert codes.min() >= 0
        assert codes.max() < 16

    def test_decode(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        codes = discretizer.encode(sample_embeddings)
        reconstructed = discretizer.decode(codes)

        assert reconstructed.shape == sample_embeddings.shape
        # LSH reconstruction is approximate
        assert torch.all(torch.isfinite(reconstructed))

    def test_deterministic(self, sample_embeddings):
        # Same seed should give same projections
        d1 = LSHDiscretizer(n_hierarchies=2, codebook_size=16, random_state=42)
        d2 = LSHDiscretizer(n_hierarchies=2, codebook_size=16, random_state=42)

        d1.fit(sample_embeddings)
        d2.fit(sample_embeddings)

        codes1 = d1.encode(sample_embeddings)
        codes2 = d2.encode(sample_embeddings)

        assert torch.allclose(codes1, codes2)

    def test_state_dict(self, discretizer, sample_embeddings):
        discretizer.fit(sample_embeddings)
        state = discretizer.state_dict()

        assert "n_hierarchies" in state
        assert "projections" in state
        assert len(state["projections"]) == 2


class TestPQDiscretizer:
    """Tests for PQDiscretizer (requires FAISS)."""

    @pytest.fixture
    def sample_embeddings(self):
        torch.manual_seed(42)
        n_samples = 100
        # Use embedding dim divisible by n_hierarchies
        embedding_dim = 64
        return torch.randn(n_samples, embedding_dim)

    def test_pq_import(self):
        try:
            from src.discretization.pq import PQDiscretizer
            assert True
        except ImportError:
            pytest.skip("FAISS not available")

    def test_pq_fit_encode(self):
        """Test PQ with larger sample size to satisfy FAISS requirements."""
        try:
            import faiss
            from src.discretization.pq import PQDiscretizer

            # Create larger sample for FAISS
            torch.manual_seed(42)
            n_samples = 300  # Need >= 256 for 8-bit codes
            embedding_dim = 64
            embeddings = torch.randn(n_samples, embedding_dim)

            # n_hierarchies must divide embedding_dim (64)
            n_sub = 4  # 64 / 4 = 16 dims per subquantizer
            discretizer = PQDiscretizer(
                n_hierarchies=n_sub,
                codebook_size=256,  # 2^8 = 256
                n_bits=8,
            )

            discretizer.fit(embeddings)
            codes = discretizer.encode(embeddings)

            # With 8-bit codes, output should be (n_samples, n_hierarchies)
            assert codes.shape[0] == n_samples
            assert codes.shape[1] == n_sub
            assert codes.min() >= 0
            assert codes.max() < 256

        except ImportError:
            pytest.skip("FAISS not available")


class TestCodeToTokenConversion:
    """Tests for code to token ID conversion."""

    def test_codes_to_tokens(self):
        discretizer = RQKMeansDiscretizer(
            n_hierarchies=3,
            codebook_size=100,
        )

        codes = torch.tensor([[0, 1, 2], [50, 60, 70]])
        tokens = discretizer.codes_to_tokens(codes, offset=1000)

        # First hierarchy: 1000 + code
        # Second hierarchy: 1000 + 100 + code
        # Third hierarchy: 1000 + 200 + code
        expected = torch.tensor([
            [1000, 1101, 1202],
            [1050, 1160, 1270],
        ])

        assert torch.allclose(tokens, expected)

    def test_tokens_to_codes(self):
        discretizer = RQKMeansDiscretizer(
            n_hierarchies=3,
            codebook_size=100,
        )

        tokens = torch.tensor([
            [1000, 1101, 1202],
            [1050, 1160, 1270],
        ])
        codes = discretizer.tokens_to_codes(tokens, offset=1000)

        expected = torch.tensor([[0, 1, 2], [50, 60, 70]])
        assert torch.allclose(codes, expected)

    def test_roundtrip(self):
        discretizer = RQKMeansDiscretizer(
            n_hierarchies=4,
            codebook_size=256,
        )

        original_codes = torch.randint(0, 256, (10, 4))
        tokens = discretizer.codes_to_tokens(original_codes, offset=500)
        recovered_codes = discretizer.tokens_to_codes(tokens, offset=500)

        assert torch.allclose(original_codes, recovered_codes)
