"""Residual Quantization with K-Means discretizer."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from .base import BaseDiscretizer

logger = logging.getLogger(__name__)


class RQKMeansDiscretizer(BaseDiscretizer):
    """
    Residual Quantization using K-Means clustering.

    This implements the hierarchical semantic ID generation from the paper,
    where embeddings are recursively quantized by subtracting centroids
    and quantizing the residual.
    """

    def __init__(
        self,
        n_hierarchies: int = 3,
        codebook_size: int = 256,
        normalize_residuals: bool = True,
        init: str = "k-means++",
        n_init: int = 3,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize the RQ-KMeans discretizer.

        Args:
            n_hierarchies: Number of quantization layers
            codebook_size: Number of centroids per layer
            normalize_residuals: Whether to normalize residuals before quantization
            init: Initialization method for k-means
            n_init: Number of k-means initializations
            max_iter: Maximum iterations for k-means
            random_state: Random seed
        """
        super().__init__(n_hierarchies=n_hierarchies, codebook_size=codebook_size)

        self.normalize_residuals = normalize_residuals
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        # Centroids for each layer
        self.centroids = []
        self.embedding_dim = None

    def fit(self, embeddings: torch.Tensor) -> "RQKMeansDiscretizer":
        """
        Fit the RQ-KMeans discretizer.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            self
        """
        embeddings = embeddings.float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        self.embedding_dim = embeddings.shape[1]
        self.centroids = []

        residuals = embeddings.clone()

        for layer_idx in range(self.n_hierarchies):
            logger.info(f"Fitting RQ-KMeans layer {layer_idx + 1}/{self.n_hierarchies}")

            # Normalize residuals if specified
            if self.normalize_residuals:
                residuals = F.normalize(residuals, p=2, dim=-1)

            # Fit k-means on residuals
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state + layer_idx,
                batch_size=min(1024, len(residuals)),
            )
            kmeans.fit(residuals.cpu().numpy())

            # Store centroids
            centroids = torch.from_numpy(kmeans.cluster_centers_).float()
            self.centroids.append(centroids)

            # Compute new residuals
            assignments = torch.from_numpy(kmeans.labels_).long()
            quantized = centroids[assignments]
            residuals = residuals - quantized

        self._is_fitted = True
        logger.info(f"RQ-KMeans fitted with {self.n_hierarchies} layers")
        return self

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings to discrete codes.

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

        codes = []
        residuals = embeddings.clone()

        for layer_idx, centroids in enumerate(self.centroids):
            # Normalize residuals if specified
            if self.normalize_residuals:
                residuals = F.normalize(residuals, p=2, dim=-1)

            # Find nearest centroid
            centroids = centroids.to(residuals.device)
            distances = torch.cdist(residuals, centroids)
            assignments = torch.argmin(distances, dim=-1)
            codes.append(assignments)

            # Compute new residuals
            quantized = centroids[assignments]
            residuals = residuals - quantized

        return torch.stack(codes, dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes back to embeddings.

        Args:
            codes: Tensor of shape (n_samples, n_hierarchies)

        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before decoding")

        if codes.dim() == 1:
            codes = codes.unsqueeze(0)

        device = codes.device
        batch_size = codes.shape[0]

        # Sum up quantized embeddings from each layer
        reconstructed = torch.zeros(batch_size, self.embedding_dim, device=device)

        for layer_idx, centroids in enumerate(self.centroids):
            centroids = centroids.to(device)
            layer_codes = codes[:, layer_idx]
            quantized = centroids[layer_codes]

            # If we normalized during encoding, we need to account for that
            # This is an approximation since we don't store the norms
            reconstructed = reconstructed + quantized

        return reconstructed

    def state_dict(self) -> dict:
        """Get the state dictionary for saving."""
        return {
            "n_hierarchies": self.n_hierarchies,
            "codebook_size": self.codebook_size,
            "normalize_residuals": self.normalize_residuals,
            "embedding_dim": self.embedding_dim,
            "centroids": [c.clone() for c in self.centroids],
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from a dictionary."""
        self.n_hierarchies = state["n_hierarchies"]
        self.codebook_size = state["codebook_size"]
        self.normalize_residuals = state["normalize_residuals"]
        self.embedding_dim = state["embedding_dim"]
        self.centroids = [c.clone() for c in state["centroids"]]
        self._is_fitted = state["is_fitted"]
