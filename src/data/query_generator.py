"""Query generation using LLMs for search tasks."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
from tqdm import tqdm

from .movielens import MovieItem

logger = logging.getLogger(__name__)

# Prompt template for query generation (from Penha et al. 2025)
QUERY_PROMPT = """Your task is to return a list with {n} queries for a given movie (title of the movie, year and description and tags). After generating the initial set of queries, you should also generate a list of the same size with paraphrased versions of the first queries. The paraphrased queries should be similar to the original queries, but with different words, structure and slight variations in the meaning. The queries should be realistic things that a user would ask to find the movie. The queries should be diverse and cover different aspects of the movie. The queries should not include the title of the movie, but be broader descriptions of the movie and its content. The queries should also contain broad topics, themes and genres of the movie.

Movie:
Title: {title}
Year: {year}
Genres: {genres}
Description: {description}

Generate exactly {n} queries and {n} paraphrased versions. Output format:
Original queries (one per line):
1. [query]
...

Paraphrased queries (one per line):
1. [query]
..."""


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
    # Get description from item text (after title and genres)
    description = ""
    if hasattr(item, 'text') and item.text:
        # item.text is typically "Title (Year) - Genres"
        # Try to extract any additional description
        parts = item.text.split(" - ")
        if len(parts) > 1:
            description = parts[-1]  # Use genres/tags as description

    prompt = QUERY_PROMPT.format(
        n=n_queries,
        title=item.title,
        year=item.year or "Unknown",
        genres=", ".join(item.genres) if item.genres else "Unknown",
        description=description or ", ".join(item.genres) if item.genres else "A movie",
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

    IMPORTANT: Queries should NOT include the movie title to be meaningful.
    They should describe the movie's themes, genres, and content.

    Args:
        items: Dictionary of item_id to MovieItem
        n_queries: Number of queries per item

    Returns:
        Dictionary mapping item_id to list of queries
    """
    queries = {}
    # Templates that don't include the title - describe movie by content/genre/themes
    templates = [
        "{genre} movie from {year}",
        "{genre} film with {theme} themes",
        "Looking for a {genre} movie",
        "{decade}s {genre} film",
        "Movie about {theme}",
        "{genre} {genre2} film",
        "Classic {genre} from the {decade}s",
        "A {mood} {genre} movie",
    ]

    # Common themes/moods by genre for variety
    genre_themes = {
        "Action": ["adventure", "heroes", "explosions", "fighting"],
        "Comedy": ["funny", "humor", "laughs", "jokes"],
        "Drama": ["emotional", "relationships", "life", "struggles"],
        "Horror": ["scary", "supernatural", "terror", "suspense"],
        "Sci-Fi": ["future", "space", "technology", "aliens"],
        "Romance": ["love", "relationships", "passion", "heart"],
        "Thriller": ["suspense", "mystery", "tension", "danger"],
        "Animation": ["animated", "family", "colorful", "cartoon"],
        "Documentary": ["real", "informative", "educational", "factual"],
        "Fantasy": ["magic", "mythical", "adventure", "supernatural"],
    }

    moods = ["exciting", "touching", "thrilling", "fun", "intense", "moving", "gripping"]

    import random

    for item_id, item in items.items():
        item_queries = []
        genre = item.genres[0] if item.genres else "drama"
        genre2 = item.genres[1] if item.genres and len(item.genres) > 1 else "drama"
        year = item.year or 2000
        decade = (year // 10) * 10

        themes = genre_themes.get(genre, ["interesting", "compelling", "engaging"])

        for i in range(n_queries):
            template = templates[i % len(templates)]
            theme = themes[i % len(themes)]
            mood = moods[i % len(moods)]

            query = template.format(
                genre=genre.lower(),
                genre2=genre2.lower(),
                year=year,
                decade=decade,
                theme=theme,
                mood=mood,
            )
            item_queries.append(query)

        queries[item_id] = item_queries

    logger.info(f"Generated synthetic queries for {len(queries)} items (without titles)")
    return queries
