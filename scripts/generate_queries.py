#!/usr/bin/env python
"""Generate synthetic queries for items using LLM or templates."""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate queries for items")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/app/data"),
        help="Data directory",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev mode (synthetic queries, no LLM)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model for query generation",
    )
    parser.add_argument(
        "--queries-per-item",
        type=int,
        default=3,
        help="Number of queries per item",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration",
    )

    args = parser.parse_args()

    # Import after parsing to avoid slow imports if just showing help
    from src.data.movielens import load_movielens
    from src.data.query_generator import generate_queries_batch, generate_synthetic_queries

    # Determine output path
    mode = "dev" if args.dev else "full"
    output_path = args.data_dir / "queries" / mode / "queries.json"

    # Check cache
    if output_path.exists() and not args.force:
        logger.info(f"Queries already exist at {output_path}")
        return

    # Load data
    raw_dir = args.data_dir / "raw" / "ml-25m"
    fraction = 0.01 if args.dev else args.fraction

    logger.info(f"Loading MovieLens data (fraction={fraction})")
    interactions, items = load_movielens(
        data_dir=raw_dir,
        fraction=fraction,
    )

    # Generate queries
    if args.dev:
        logger.info("Generating synthetic queries (dev mode)")
        queries = generate_synthetic_queries(
            items=items,
            n_queries=args.queries_per_item,
        )
    else:
        logger.info(f"Generating queries using {args.model}")
        queries = generate_queries_batch(
            items=list(items.values()),
            model_name=args.model,
            n_queries=args.queries_per_item,
            cache_path=output_path,
            force=args.force,
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)

    logger.info(f"Saved {len(queries)} item queries to {output_path}")


if __name__ == "__main__":
    main()
