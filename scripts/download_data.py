#!/usr/bin/env python
"""Download MovieLens-25M dataset."""

import argparse
import logging
import os
import zipfile
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(output_dir: Path, force: bool = False):
    """
    Download and extract MovieLens-25M dataset.

    Args:
        output_dir: Directory to save the dataset
        force: Force re-download even if exists
    """
    output_dir = Path(output_dir)
    ml_dir = output_dir / "ml-25m"

    # Check if already exists
    if ml_dir.exists() and not force:
        logger.info(f"MovieLens-25M already exists at {ml_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ml-25m.zip"

    # Download
    if not zip_path.exists() or force:
        logger.info(f"Downloading MovieLens-25M from {MOVIELENS_25M_URL}")
        urllib.request.urlretrieve(MOVIELENS_25M_URL, zip_path)
        logger.info(f"Downloaded to {zip_path}")

    # Extract
    logger.info(f"Extracting to {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    logger.info(f"MovieLens-25M extracted to {ml_dir}")

    # Cleanup zip file
    if zip_path.exists():
        zip_path.unlink()
        logger.info("Cleaned up zip file")


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens-25M dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/data/raw"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )

    args = parser.parse_args()
    download_movielens(args.output_dir, args.force)


if __name__ == "__main__":
    main()
