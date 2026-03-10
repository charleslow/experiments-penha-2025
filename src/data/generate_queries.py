"""Stage 2: Query Generation — generate search queries for movies.

Supports two backends:
  - "template": Fast, deterministic query generation from movie metadata (no LLM).
                Suitable for mini/dev pipeline validation on CPU.
  - "ollama":   LLM-based generation via Ollama for higher quality queries.
                Suitable for full runs with GPU or cloud API.

Usage:
    python -m src.data.generate_queries --mode mini
    python -m src.data.generate_queries --mode dev
"""

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Prompt template (for Ollama backend) ─────────────────────────────────────

PROMPT_TEMPLATE = """\
Your task is to return a list with {n_queries} queries for a given movie (title of the movie, year and description and tags). After generating the initial set of queries, you should also generate a list of the same size with paraphrased versions of the first queries. The paraphrased queries should be similar to the original queries, but with different words, structure and slight variations in the meaning. The queries should be realistic things that a user would ask to find the movie. The queries should be diverse and cover different aspects of the movie. The queries should not include the title of the movie, but be broader descriptions of the movie and its content. The queries should also contain broad topics, themes and genres of the movie.

Movie: {metadata}

Return ONLY a JSON object with two keys: "queries" (list of {n_queries} strings) and "paraphrases" (list of {n_queries} strings). No other text."""


# ── Shared helpers ───────────────────────────────────────────────────────────

def build_movie_metadata(row: pd.Series) -> str:
    """Build a metadata string for a movie from its row."""
    parts = [f"Title: {row['title']}"]

    # Extract year from title if present (format: "Movie Name (1999)")
    title = row["title"]
    year_match = re.search(r"\((\d{4})\)", title)
    if year_match:
        parts.append(f"Year: {year_match.group(1)}")

    if row.get("genres") and row["genres"] != "(no genres listed)":
        parts.append(f"Genres: {row['genres']}")

    if row.get("top_genome_tags") and str(row["top_genome_tags"]).strip():
        parts.append(f"Tags: {row['top_genome_tags']}")

    if row.get("user_tags") and str(row["user_tags"]).strip():
        # Limit user tags to avoid overly long prompts
        user_tags = str(row["user_tags"])
        if len(user_tags) > 300:
            user_tags = user_tags[:300] + "..."
        parts.append(f"User tags: {user_tags}")

    return "\n".join(parts)


def load_progress(output_path: Path) -> dict[int, dict]:
    """Load previously generated queries for resume capability."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r") as f:
            records = json.load(f)
        return {r["movieId"]: r for r in records}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_progress(records: dict[int, dict], output_path: Path):
    """Save all records to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(records.values()), f, indent=2)


# ── Template backend ────────────────────────────────────────────────────────

# Query templates that combine metadata in different ways.
# {genre_lower} = lowercased genre, {tag} = a genome/user tag, {decade} = "1990s" etc.
QUERY_TEMPLATES = [
    # Genre-focused
    "{genre_lower} movie from the {decade}",
    "a {genre_lower} film with themes of {tag}",
    "{decade} {genre_lower} film",
    # Tag-focused
    "movie about {tag} and {tag2}",
    "film featuring {tag}",
    "{tag} movie {decade}",
    # Genre + tag combos
    "{genre_lower} movie involving {tag}",
    "a {genre_lower} story about {tag}",
    # Broader descriptions
    "movie with {tag} themes set in {decade}",
    "film that explores {tag} and {tag2}",
    "{genre_lower} {genre_lower2} movie",
    "classic {genre_lower} about {tag}",
]

PARAPHRASE_TEMPLATES = [
    # Slight variations on the originals
    "{decade} film about {tag}",
    "{genre_lower} movie exploring {tag} and {tag2}",
    "a {genre_lower} picture from {decade}",
    "movie dealing with {tag}",
    "{tag} themed {genre_lower} film",
    "{genre_lower} {decade} movie about {tag}",
    "film with {tag} elements and {genre_lower} style",
    "a {decade} movie centered on {tag}",
    "{genre_lower} cinema involving {tag2}",
    "{tag} and {tag2} in a {genre_lower} setting",
    "movie from {decade} with {genre_lower} themes",
    "{genre_lower} film dealing with {tag} and {tag2}",
]


def _extract_year(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None


def _year_to_decade(year: int | None) -> str:
    if year is None:
        return "recent era"
    decade_start = (year // 10) * 10
    return f"{decade_start}s"


def _parse_tags(row: pd.Series) -> list[str]:
    """Extract a list of tags from genome + user tags."""
    tags = []
    genome = str(row.get("top_genome_tags", "")).strip()
    if genome:
        tags.extend([t.strip() for t in genome.split(",") if t.strip()])
    user = str(row.get("user_tags", "")).strip()
    if user:
        tags.extend([t.strip() for t in user.split(",") if t.strip()])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique.append(t.lower())
    return unique if unique else ["adventure", "drama"]


def _parse_genres(row: pd.Series) -> list[str]:
    """Extract genres as a list."""
    genres_str = str(row.get("genres", "")).strip()
    if not genres_str or genres_str == "(no genres listed)":
        return ["drama"]
    return [g.strip().lower() for g in genres_str.split("|") if g.strip()]


def generate_template_queries(
    row: pd.Series,
    n_queries: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Generate queries from templates using movie metadata.

    Returns (queries, paraphrases).
    """
    rng = random.Random(seed + int(row["movieId"]))

    year = _extract_year(row["title"])
    decade = _year_to_decade(year)
    genres = _parse_genres(row)
    tags = _parse_tags(row)

    def fill_template(template: str) -> str:
        genre = rng.choice(genres)
        genre2 = rng.choice(genres)
        tag = rng.choice(tags)
        # Pick a different tag if possible
        tag2 = rng.choice([t for t in tags if t != tag]) if len(tags) > 1 else tag
        return template.format(
            genre_lower=genre,
            genre_lower2=genre2,
            tag=tag,
            tag2=tag2,
            decade=decade,
        )

    # Shuffle templates and pick n_queries from each pool
    q_templates = list(QUERY_TEMPLATES)
    p_templates = list(PARAPHRASE_TEMPLATES)
    rng.shuffle(q_templates)
    rng.shuffle(p_templates)

    queries = [fill_template(q_templates[i % len(q_templates)]) for i in range(n_queries)]
    paraphrases = [fill_template(p_templates[i % len(p_templates)]) for i in range(n_queries)]

    return queries, paraphrases


# ── Ollama backend ──────────────────────────────────────────────────────────

def query_ollama(
    prompt: str,
    model: str,
    base_url: str,
    max_retries: int = 3,
) -> str | None:
    """Send a prompt to Ollama and return the response text."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 1024,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            log.warning("Ollama request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_queries_response(response_text: str, n_queries: int) -> tuple[list[str], list[str]] | None:
    """Parse the JSON response from the LLM.

    Returns (queries, paraphrases) or None if parsing fails.
    """
    if not response_text:
        return None

    # Try to find JSON in the response
    text = response_text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        queries = data.get("queries", [])
        paraphrases = data.get("paraphrases", [])
        if queries and paraphrases:
            return queries[:n_queries], paraphrases[:n_queries]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block from markdown code fence
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            queries = data.get("queries", [])
            paraphrases = data.get("paraphrases", [])
            if queries and paraphrases:
                return queries[:n_queries], paraphrases[:n_queries]
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object anywhere in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            queries = data.get("queries", [])
            paraphrases = data.get("paraphrases", [])
            if queries and paraphrases:
                return queries[:n_queries], paraphrases[:n_queries]
        except json.JSONDecodeError:
            pass

    return None


# ── Main ────────────────────────────────────────────────────────────────────

def main(mode: str = "dev"):
    cfg = get_config(mode)
    log.info("=== Stage 2: Query Generation (mode=%s, backend=%s) ===", mode, cfg.query_backend)

    # Load processed movies
    movies_path = cfg.data_processed_dir / mode / "movies.parquet"
    if not movies_path.exists():
        log.error("Movies file not found: %s. Run Stage 1 first.", movies_path)
        return

    movies = pd.read_parquet(movies_path)
    log.info("Loaded %d movies from %s", len(movies), movies_path)

    n_to_generate = cfg.n_queries_per_movie

    # Output file
    output_path = cfg.data_queries_dir / mode / "queries.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    progress = load_progress(output_path)
    log.info("Resuming: %d movies already processed", len(progress))

    # Ollama-specific: check reachability
    if cfg.query_backend == "ollama":
        try:
            resp = requests.get(f"{cfg.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            log.info("Ollama is reachable at %s", cfg.ollama_url)
        except requests.exceptions.RequestException as e:
            log.error("Cannot reach Ollama at %s: %s", cfg.ollama_url, e)
            log.error("Make sure Ollama is running: ollama serve")
            return

    # Generate queries for each movie
    n_success = 0
    n_fail = 0
    save_interval = 10  # save every N movies

    for idx, row in tqdm(movies.iterrows(), total=len(movies), desc="Generating queries"):
        movie_id = int(row["movieId"])

        # Skip if already done
        if movie_id in progress:
            n_success += 1
            continue

        if cfg.query_backend == "template":
            queries, paraphrases = generate_template_queries(row, n_to_generate, cfg.seed)
        elif cfg.query_backend == "ollama":
            metadata = build_movie_metadata(row)
            prompt = PROMPT_TEMPLATE.format(n_queries=n_to_generate, metadata=metadata)
            response_text = query_ollama(prompt, cfg.ollama_model, cfg.ollama_url)
            parsed = parse_queries_response(response_text, n_to_generate)
            if parsed is None:
                log.warning("Failed to parse queries for movie %d (%s)", movie_id, row["title"])
                n_fail += 1
                continue
            queries, paraphrases = parsed
        else:
            raise ValueError(f"Unknown query backend: {cfg.query_backend}")

        # Combine and split into train/test
        all_queries = queries + paraphrases
        train_queries = all_queries[:cfg.n_train_queries]
        test_queries = all_queries[cfg.n_train_queries:cfg.n_train_queries + cfg.n_test_queries]

        record = {
            "movieId": movie_id,
            "title": row["title"],
            "queries_original": queries,
            "queries_paraphrased": paraphrases,
            "train_queries": train_queries,
            "test_queries": test_queries,
        }
        progress[movie_id] = record
        n_success += 1

        # Periodic save
        if n_success % save_interval == 0:
            save_progress(progress, output_path)

    # Final save
    save_progress(progress, output_path)

    # Also save as a flat parquet for easier downstream use
    flat_records = []
    for movie_id, rec in progress.items():
        for split, key in [("train", "train_queries"), ("test", "test_queries")]:
            for qi, query in enumerate(rec.get(key, [])):
                flat_records.append({
                    "movieId": movie_id,
                    "title": rec["title"],
                    "split": split,
                    "query_idx": qi,
                    "query": query,
                })

    if flat_records:
        flat_df = pd.DataFrame(flat_records)
        flat_path = cfg.data_queries_dir / mode / "queries.parquet"
        flat_df.to_parquet(flat_path, index=False)
        log.info("Saved flat queries to %s (%d rows)", flat_path, len(flat_df))

    log.info("=== Stage 2 complete ===")
    log.info("  Success: %d, Failed: %d", n_success, n_fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Query Generation")
    parser.add_argument("--mode", type=str, default="dev", choices=["mini", "dev", "full"],
                        help="Run mode: mini, dev, or full")
    args = parser.parse_args()
    main(args.mode)
