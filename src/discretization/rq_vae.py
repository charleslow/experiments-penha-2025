"""Residual Quantization VAE discretizer using vector-quantize-pytorch."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiscretizer

logger = logging.getLogger(__name__)


class RQVAEDiscretizer(BaseDiscretizer):
    """
    Residual Quantization VAE for embedding discretization.

    Uses vector-quantize-pytorch's ResidualVQ for learned quantization
    with commitment loss and codebook learning.
    """

    def __init__(
        self,
        n_hierarchies: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 384,
        commitment_weight: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        threshold_ema_dead_code: int = 2,
        num_epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the RQ-VAE discretizer.

        Args:
            n_hierarchies: Number of quantization layers (depth)
            codebook_size: Number of codes per hierarchy
            embedding_dim: Dimension of input embeddings
            commitment_weight: Weight for commitment loss
            decay: EMA decay for codebook update
            kmeans_init: Whether to use k-means for codebook initialization
            threshold_ema_dead_code: Threshold for dead code replacement
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
        """
        super().__init__(n_hierarchies=n_hierarchies, codebook_size=codebook_size)

        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.rq = None
        self.device = None

    def _build_model(self, device: torch.device):
        """Build the ResidualVQ model."""
        from vector_quantize_pytorch import ResidualVQ

        self.rq = ResidualVQ(
            dim=self.embedding_dim,
            codebook_size=self.codebook_size,
            num_quantizers=self.n_hierarchies,
            commitment_weight=self.commitment_weight,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        ).to(device)
        self.device = device

    def fit(self, embeddings: torch.Tensor) -> "RQVAEDiscretizer":
        """
        Fit the RQ-VAE discretizer by training on embeddings.

        ResidualVQ uses EMA (Exponential Moving Average) updates for codebooks,
        so we just need to run forward passes in training mode.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            self
        """
        embeddings = embeddings.float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        device = embeddings.device if embeddings.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embeddings.shape[1]

        # Build model
        self._build_model(device)
        embeddings = embeddings.to(device)

        # Training loop - ResidualVQ uses EMA updates during forward passes
        self.rq.train()

        n_samples = embeddings.shape[0]
        n_batches = max(1, n_samples // self.batch_size)

        logger.info(f"Training RQ-VAE: {self.num_epochs} epochs, {n_batches} batches/epoch")

        for epoch in range(self.num_epochs):
            # Shuffle
            perm = torch.randperm(n_samples, device=device)
            total_loss = 0.0

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)
                batch_idx = perm[start_idx:end_idx]
                batch = embeddings[batch_idx]

                # Forward pass - codebooks updated via EMA
                quantized, indices, commit_loss = self.rq(batch)

                # Compute reconstruction loss for logging
                with torch.no_grad():
                    recon_loss = F.mse_loss(quantized, batch)
                    total_loss += recon_loss.item()

            avg_loss = total_loss / n_batches
            if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, MSE: {avg_loss:.4f}")

        self.rq.eval()
        self._is_fitted = True
        logger.info(f"RQ-VAE fitted with {self.n_hierarchies} layers, {self.codebook_size} codes each")
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

        embeddings = embeddings.to(self.device)

        with torch.no_grad():
            self.rq.eval()
            _, indices, _ = self.rq(embeddings)

        # indices shape: (n_samples, n_hierarchies)
        return indices.cpu()

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

        codes = codes.to(self.device)

        with torch.no_grad():
            self.rq.eval()
            # Get quantized vectors from indices
            # ResidualVQ stores codebooks that we can index into
            quantized = self.rq.get_output_from_indices(codes)

        return quantized.cpu()

    def state_dict(self) -> dict:
        """Get the state dictionary for saving."""
        return {
            "n_hierarchies": self.n_hierarchies,
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "commitment_weight": self.commitment_weight,
            "decay": self.decay,
            "rq_state": self.rq.state_dict() if self.rq else None,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from a dictionary."""
        self.n_hierarchies = state["n_hierarchies"]
        self.codebook_size = state["codebook_size"]
        self.embedding_dim = state["embedding_dim"]
        self.commitment_weight = state["commitment_weight"]
        self.decay = state["decay"]
        self._is_fitted = state["is_fitted"]

        if state["rq_state"] is not None and self._is_fitted:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._build_model(device)
            self.rq.load_state_dict(state["rq_state"])
            self.rq.eval()
