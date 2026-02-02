"""Caching utilities for artifact management."""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of artifacts to avoid recomputation."""

    def __init__(self, base_dir: Union[str, Path] = "/app/data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, *parts: str) -> Path:
        """Get a path relative to the base directory."""
        path = self.base_dir
        for part in parts:
            path = path / part
        return path

    def exists(self, *parts: str) -> bool:
        """Check if a cached artifact exists."""
        path = self.get_path(*parts)
        return path.exists()

    def save_tensor(
        self,
        tensor: torch.Tensor,
        *parts: str,
        force: bool = False,
    ) -> Path:
        """Save a tensor to cache."""
        path = self.get_path(*parts)
        if path.exists() and not force:
            logger.info(f"Skipping {path}, already exists")
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        logger.info(f"Saved tensor to {path}")
        return path

    def load_tensor(self, *parts: str) -> Optional[torch.Tensor]:
        """Load a tensor from cache."""
        path = self.get_path(*parts)
        if not path.exists():
            logger.warning(f"Cache miss: {path}")
            return None

        logger.info(f"Loading tensor from {path}")
        return torch.load(path, weights_only=True)

    def save_parquet(
        self,
        df: pd.DataFrame,
        *parts: str,
        force: bool = False,
    ) -> Path:
        """Save a DataFrame to Parquet format."""
        path = self.get_path(*parts)
        if path.exists() and not force:
            logger.info(f"Skipping {path}, already exists")
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved DataFrame to {path}")
        return path

    def load_parquet(self, *parts: str) -> Optional[pd.DataFrame]:
        """Load a DataFrame from Parquet format."""
        path = self.get_path(*parts)
        if not path.exists():
            logger.warning(f"Cache miss: {path}")
            return None

        logger.info(f"Loading DataFrame from {path}")
        return pd.read_parquet(path)

    def save_json(
        self,
        data: Any,
        *parts: str,
        force: bool = False,
    ) -> Path:
        """Save data to JSON format."""
        path = self.get_path(*parts)
        if path.exists() and not force:
            logger.info(f"Skipping {path}, already exists")
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON to {path}")
        return path

    def load_json(self, *parts: str) -> Optional[Any]:
        """Load data from JSON format."""
        path = self.get_path(*parts)
        if not path.exists():
            logger.warning(f"Cache miss: {path}")
            return None

        logger.info(f"Loading JSON from {path}")
        with open(path, "r") as f:
            return json.load(f)

    def save_checkpoint(
        self,
        state_dict: dict,
        *parts: str,
        force: bool = False,
    ) -> Path:
        """Save a model checkpoint."""
        path = self.get_path(*parts)
        if path.exists() and not force:
            logger.info(f"Skipping {path}, already exists")
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, *parts: str) -> Optional[dict]:
        """Load a model checkpoint."""
        path = self.get_path(*parts)
        if not path.exists():
            logger.warning(f"Cache miss: {path}")
            return None

        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, weights_only=False)

    def clear(self, *parts: str) -> None:
        """Clear a cached artifact or directory."""
        path = self.get_path(*parts)
        if path.is_file():
            path.unlink()
            logger.info(f"Removed {path}")
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
            logger.info(f"Removed directory {path}")
