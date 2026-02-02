"""Locality-Sensitive Hashing discretizer."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseDiscretizer

logger = logging.getLogger(__name__)


class LSHDiscretizer(BaseDiscretizer):
    """
    Locality-Sensitive Hashing for embedding discretization.

    Uses random hyperplane projections to create binary hash codes,
    which are then grouped into hierarchical buckets.
    """

    def __init__(
        self,
        n_hierarchies: int = 3,
        codebook_size: int = 256,
        n_bits_per_hierarchy: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the LSH discretizer.

        Args:
            n_hierarchies: Number of hierarchical levels
            codebook_size: Size of each codebook (must be power of 2)
            n_bits_per_hierarchy: Bits per hierarchy (derived from codebook_size if None)
            random_state: Random seed for reproducibility
        """
        super().__init__(n_hierarchies=n_hierarchies, codebook_size=codebook_size)

        # Compute bits per hierarchy from codebook size
        if n_bits_per_hierarchy is None:
            import math

            n_bits_per_hierarchy = int(math.log2(codebook_size))
            # Ensure codebook_size is power of 2
            assert 2**n_bits_per_hierarchy == codebook_size, (
                f"codebook_size must be power of 2, got {codebook_size}"
            )

        self.n_bits_per_hierarchy = n_bits_per_hierarchy
        self.total_bits = n_hierarchies * n_bits_per_hierarchy
        self.random_state = random_state

        # Random projection matrices
        self.projections = None
        self.embedding_dim = None

    def fit(self, embeddings: torch.Tensor) -> "LSHDiscretizer":
        """
        Fit the LSH discretizer by generating random projection matrices.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            self
        """
        embeddings = embeddings.float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        self.embedding_dim = embeddings.shape[1]

        # Generate random hyperplanes for each hierarchy
        generator = torch.Generator().manual_seed(self.random_state)
        self.projections = []

        for h in range(self.n_hierarchies):
            # Random Gaussian projection matrix
            proj = torch.randn(
                self.embedding_dim,
                self.n_bits_per_hierarchy,
                generator=generator,
            )
            # Normalize columns
            proj = F.normalize(proj, p=2, dim=0)
            self.projections.append(proj)

        self._is_fitted = True
        logger.info(
            f"LSH fitted: {self.n_hierarchies} hierarchies, "
            f"{self.n_bits_per_hierarchy} bits each"
        )
        return self

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings using LSH.

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

        device = embeddings.device
        codes = []

        for h, proj in enumerate(self.projections):
            proj = proj.to(device)

            # Project and binarize
            projected = torch.matmul(embeddings, proj)
            binary = (projected > 0).long()

            # Convert binary to integer code
            powers = 2 ** torch.arange(self.n_bits_per_hierarchy, device=device)
            code = torch.sum(binary * powers, dim=-1)
            codes.append(code)

        return torch.stack(codes, dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Approximate decode by using centroid of hash bucket.

        Note: LSH doesn't have perfect reconstruction. This returns
        an approximation based on the projection directions.

        Args:
            codes: Tensor of shape (n_samples, n_hierarchies)

        Returns:
            Approximate embeddings of shape (n_samples, embedding_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before decoding")

        if codes.dim() == 1:
            codes = codes.unsqueeze(0)

        device = codes.device
        batch_size = codes.shape[0]

        # Reconstruct by combining projection directions
        reconstructed = torch.zeros(batch_size, self.embedding_dim, device=device)

        for h, proj in enumerate(self.projections):
            proj = proj.to(device)
            layer_codes = codes[:, h]

            # Convert code to binary
            powers = 2 ** torch.arange(self.n_bits_per_hierarchy, device=device)
            binary = ((layer_codes.unsqueeze(-1) // powers) % 2).float()

            # Convert to +1/-1
            signs = 2 * binary - 1

            # Sum projection directions weighted by signs
            contribution = torch.matmul(signs, proj.t())
            reconstructed = reconstructed + contribution

        return F.normalize(reconstructed, p=2, dim=-1)

    def state_dict(self) -> dict:
        """Get the state dictionary for saving."""
        return {
            "n_hierarchies": self.n_hierarchies,
            "codebook_size": self.codebook_size,
            "n_bits_per_hierarchy": self.n_bits_per_hierarchy,
            "embedding_dim": self.embedding_dim,
            "projections": [p.clone() for p in self.projections] if self.projections else None,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from a dictionary."""
        self.n_hierarchies = state["n_hierarchies"]
        self.codebook_size = state["codebook_size"]
        self.n_bits_per_hierarchy = state["n_bits_per_hierarchy"]
        self.embedding_dim = state["embedding_dim"]
        self.projections = (
            [p.clone() for p in state["projections"]] if state["projections"] else None
        )
        self._is_fitted = state["is_fitted"]
