"""Product Quantization discretizer using FAISS."""

import logging
from typing import Optional

import numpy as np
import torch

from .base import BaseDiscretizer

logger = logging.getLogger(__name__)


class PQDiscretizer(BaseDiscretizer):
    """
    Product Quantization for embedding discretization.

    Uses FAISS for efficient PQ training and encoding.
    The embedding is split into n_hierarchies subvectors,
    each quantized with codebook_size centroids.
    """

    def __init__(
        self,
        n_hierarchies: int = 3,
        codebook_size: int = 256,
        n_bits: int = 8,
    ):
        """
        Initialize the PQ discretizer.

        Args:
            n_hierarchies: Number of subquantizers (subspaces)
            codebook_size: Number of centroids per subquantizer
            n_bits: Number of bits per subcode (codebook_size = 2^n_bits)
        """
        super().__init__(n_hierarchies=n_hierarchies, codebook_size=codebook_size)

        self.n_bits = n_bits
        self.pq = None
        self.embedding_dim = None

    def fit(self, embeddings: torch.Tensor) -> "PQDiscretizer":
        """
        Fit the PQ discretizer.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            self
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required for PQ. Install with: pip install faiss-cpu")

        embeddings = embeddings.float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        self.embedding_dim = embeddings.shape[1]

        # Ensure embedding_dim is divisible by n_hierarchies
        if self.embedding_dim % self.n_hierarchies != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"n_hierarchies ({self.n_hierarchies})"
            )

        # Create and train PQ
        self.pq = faiss.ProductQuantizer(
            self.embedding_dim,
            self.n_hierarchies,
            self.n_bits,
        )

        # Convert to numpy and train
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        self.pq.train(embeddings_np)

        self._is_fitted = True
        logger.info(
            f"PQ fitted: {self.n_hierarchies} subquantizers, "
            f"{self.codebook_size} centroids each"
        )
        return self

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings using PQ.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            Tensor of shape (n_samples, n_hierarchies)
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before encoding")

        embeddings = embeddings.float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        # Encode with PQ
        codes = self.pq.compute_codes(embeddings_np)

        return torch.from_numpy(codes.astype(np.int64))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode PQ codes back to embeddings.

        Args:
            codes: Tensor of shape (n_samples, n_hierarchies)

        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before decoding")

        if codes.dim() == 1:
            codes = codes.unsqueeze(0)

        # Convert to numpy
        codes_np = codes.cpu().numpy().astype(np.uint8)

        # Decode with PQ
        reconstructed = self.pq.decode(codes_np)

        return torch.from_numpy(reconstructed)

    def state_dict(self) -> dict:
        """Get the state dictionary for saving."""
        # Serialize PQ centroids
        centroids = None
        if self.pq is not None:
            import faiss

            centroids = faiss.vector_to_array(self.pq.centroids).reshape(
                self.n_hierarchies, self.codebook_size, -1
            )

        return {
            "n_hierarchies": self.n_hierarchies,
            "codebook_size": self.codebook_size,
            "n_bits": self.n_bits,
            "embedding_dim": self.embedding_dim,
            "centroids": centroids,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from a dictionary."""
        import faiss

        self.n_hierarchies = state["n_hierarchies"]
        self.codebook_size = state["codebook_size"]
        self.n_bits = state["n_bits"]
        self.embedding_dim = state["embedding_dim"]
        self._is_fitted = state["is_fitted"]

        if state["centroids"] is not None and self._is_fitted:
            # Recreate PQ from centroids
            self.pq = faiss.ProductQuantizer(
                self.embedding_dim,
                self.n_hierarchies,
                self.n_bits,
            )
            centroids = state["centroids"].flatten().astype(np.float32)
            faiss.copy_array_to_vector(centroids, self.pq.centroids)
