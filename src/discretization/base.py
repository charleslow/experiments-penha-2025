"""Base class for discretization methods."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseDiscretizer(ABC):
    """Abstract base class for embedding discretization methods."""

    def __init__(
        self,
        n_hierarchies: int = 3,
        codebook_size: int = 256,
    ):
        """
        Initialize the discretizer.

        Args:
            n_hierarchies: Number of hierarchical levels
            codebook_size: Size of each codebook
        """
        self.n_hierarchies = n_hierarchies
        self.codebook_size = codebook_size
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if the discretizer has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(self, embeddings: torch.Tensor) -> "BaseDiscretizer":
        """
        Fit the discretizer to the embeddings.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            self
        """
        pass

    @abstractmethod
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings to discrete codes.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            Tensor of shape (n_samples, n_hierarchies) with integer codes
        """
        pass

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes back to embeddings.

        Args:
            codes: Tensor of shape (n_samples, n_hierarchies)

        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        pass

    def fit_encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fit the discretizer and encode embeddings in one step.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            Tensor of shape (n_samples, n_hierarchies)
        """
        self.fit(embeddings)
        return self.encode(embeddings)

    def reconstruction_error(self, embeddings: torch.Tensor) -> float:
        """
        Compute mean squared reconstruction error.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)

        Returns:
            Mean squared error
        """
        codes = self.encode(embeddings)
        reconstructed = self.decode(codes)
        mse = torch.mean((embeddings - reconstructed) ** 2).item()
        return mse

    def get_vocab_size(self) -> int:
        """Get the total vocabulary size for semantic IDs."""
        return self.n_hierarchies * self.codebook_size

    def codes_to_tokens(
        self,
        codes: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Convert codes to token IDs for vocabulary embedding.

        Each hierarchy level gets its own offset in the vocabulary.

        Args:
            codes: Tensor of shape (n_samples, n_hierarchies)
            offset: Base offset for token IDs

        Returns:
            Tensor of shape (n_samples, n_hierarchies) with token IDs
        """
        tokens = codes.clone()
        for h in range(self.n_hierarchies):
            tokens[:, h] = codes[:, h] + offset + h * self.codebook_size
        return tokens

    def tokens_to_codes(
        self,
        tokens: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Convert token IDs back to codes.

        Args:
            tokens: Tensor of shape (n_samples, n_hierarchies)
            offset: Base offset that was used

        Returns:
            Tensor of shape (n_samples, n_hierarchies) with codes
        """
        codes = tokens.clone()
        for h in range(self.n_hierarchies):
            codes[:, h] = tokens[:, h] - offset - h * self.codebook_size
        return codes

    def save(self, path: str) -> None:
        """Save the discretizer state."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> "BaseDiscretizer":
        """Load the discretizer state."""
        state = torch.load(path, weights_only=True)
        self.load_state_dict(state)
        return self

    @abstractmethod
    def state_dict(self) -> dict:
        """Get the state dictionary for saving."""
        pass

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Load state from a dictionary."""
        pass
