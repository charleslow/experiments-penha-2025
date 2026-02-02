"""Query generation using LLMs for search tasks."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
from tqdm import tqdm

from .movielens import MovieItem

logger = logging.getLogger(__name__)

# Prompt template for query generation
QUERY_PROMPT = """Generate {n} diverse search queries that a user might type to find this movie.
The queries should be natural language questions or search phrases.

Movie: {title}
Year: {year}
Genres: {genres}

Generate exactly {n} queries, one per line. Only output the queries, nothing else."""


def generate_queries_for_item(
    item: MovieItem,
    model,
    tokenizer,
    n_queries: int = 3,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate search queries for a single item.

    Args:
        item: MovieItem to generate queries for
        model: The language model
        tokenizer: The tokenizer
        n_queries: Number of queries to generate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated queries
    """
    prompt = QUERY_PROMPT.format(
        n=n_queries,
        title=item.title,
        year=item.year or "Unknown",
        genres=", ".join(item.genres) if item.genres else "Unknown",
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse queries from response
    queries = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Remove numbering if present (e.g., "1. ", "- ")
        if line and len(line) > 2:
            if line[0].isdigit() and line[1] in ".):":
                line = line[2:].strip()
            elif line[0] in "-*":
                line = line[1:].strip()
            if line:
                queries.append(line)

    return queries[:n_queries]


def generate_queries_batch(
    items: List[MovieItem],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_queries: int = 3,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[int, List[str]]:
    """
    Generate queries for a batch of items.

    Args:
        items: List of MovieItems
        model_name: Name of the model to use
        n_queries: Number of queries per item
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        cache_path: Optional path to cache results
        force: Force regeneration even if cache exists

    Returns:
        Dictionary mapping item_id to list of queries
    """
    # Check cache
    if cache_path and cache_path.exists() and not force:
        logger.info(f"Loading queries from cache: {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        # Convert string keys to int
        return {int(k): v for k, v in cached.items()}

    logger.info(f"Generating queries using {model_name}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    queries = {}
    for item in tqdm(items, desc="Generating queries"):
        try:
            item_queries = generate_queries_for_item(
                item=item,
                model=model,
                tokenizer=tokenizer,
                n_queries=n_queries,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            queries[item.item_id] = item_queries
        except Exception as e:
            logger.warning(f"Failed to generate queries for item {item.item_id}: {e}")
            queries[item.item_id] = [item.title]  # Fallback to title

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"Saved queries to cache: {cache_path}")

    return queries


def generate_synthetic_queries(
    items: Dict[int, MovieItem],
    n_queries: int = 3,
) -> Dict[int, List[str]]:
    """
    Generate simple synthetic queries without an LLM.
    Useful for testing and dev runs.

    Args:
        items: Dictionary of item_id to MovieItem
        n_queries: Number of queries per item

    Returns:
        Dictionary mapping item_id to list of queries
    """
    queries = {}
    templates = [
        "Find {title}",
        "{genre} movie {title}",
        "Looking for {title} from {year}",
        "Movie called {title}",
        "{genre} film {year}",
    ]

    for item_id, item in items.items():
        item_queries = []
        genre = item.genres[0] if item.genres else "drama"
        year = item.year or 2000

        for i in range(n_queries):
            template = templates[i % len(templates)]
            query = template.format(
                title=item.title,
                genre=genre,
                year=year,
            )
            item_queries.append(query)

        queries[item_id] = item_queries

    logger.info(f"Generated synthetic queries for {len(queries)} items")
    return queries
