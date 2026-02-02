"""Unit tests for data loading and processing."""

import pytest
import pandas as pd
import torch
import numpy as np

from src.data.movielens import (
    MovieItem,
    parse_title_year,
    chronological_split,
    get_cooccurrence_pairs,
)
from src.data.query_generator import generate_synthetic_queries
from src.data.datamodule import SearchDataset, RecDataset


class TestMovieItem:
    """Tests for MovieItem dataclass."""

    def test_text_with_year(self):
        item = MovieItem(
            item_id=1,
            title="The Matrix",
            genres=["Sci-Fi", "Action"],
            year=1999,
        )
        assert "The Matrix" in item.text
        assert "1999" in item.text
        assert "Sci-Fi" in item.text

    def test_text_without_year(self):
        item = MovieItem(
            item_id=1,
            title="Unknown Movie",
            genres=["Drama"],
            year=None,
        )
        assert "Unknown Movie" in item.text
        assert "Drama" in item.text

    def test_text_empty_genres(self):
        item = MovieItem(
            item_id=1,
            title="No Genre Movie",
            genres=[],
            year=2000,
        )
        assert "Unknown" in item.text or "No Genre Movie" in item.text


class TestParseTitleYear:
    """Tests for title/year parsing."""

    def test_standard_format(self):
        title, year = parse_title_year("The Matrix (1999)")
        assert title == "The Matrix"
        assert year == 1999

    def test_no_year(self):
        title, year = parse_title_year("Some Movie Without Year")
        assert title == "Some Movie Without Year"
        assert year is None

    def test_multiple_parentheses(self):
        title, year = parse_title_year("Movie (Part 1) (2020)")
        assert year == 2020

    def test_whitespace(self):
        title, year = parse_title_year("  Spaced Movie  (2015)  ")
        assert title == "Spaced Movie"
        assert year == 2015


class TestChronologicalSplit:
    """Tests for chronological splitting."""

    @pytest.fixture
    def sample_interactions(self):
        return pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rating": [4.0] * 10,
            "timestamp": list(range(10)),
        })

    def test_split_sizes(self, sample_interactions):
        train, val, test = chronological_split(
            sample_interactions,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        total = len(sample_interactions)
        assert len(train) == int(total * 0.7)
        assert len(val) == int(total * 0.1)
        assert len(test) == total - len(train) - len(val)

    def test_chronological_order(self, sample_interactions):
        train, val, test = chronological_split(
            sample_interactions,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        # Test data should have later timestamps
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()


class TestCooccurrencePairs:
    """Tests for co-occurrence pair generation."""

    @pytest.fixture
    def sample_interactions(self):
        return pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 2, 4],
            "rating": [4.0] * 5,
            "timestamp": [1, 2, 3, 1, 2],
        })

    def test_generates_pairs(self, sample_interactions):
        pairs = get_cooccurrence_pairs(sample_interactions, window_size=5)
        assert len(pairs) > 0
        assert "item1" in pairs.columns
        assert "item2" in pairs.columns
        assert "count" in pairs.columns

    def test_symmetric_pairs(self, sample_interactions):
        pairs = get_cooccurrence_pairs(sample_interactions, window_size=5)
        # item1 should always be <= item2 for consistency
        assert all(pairs["item1"] <= pairs["item2"])


class TestSyntheticQueries:
    """Tests for synthetic query generation."""

    @pytest.fixture
    def sample_items(self):
        return {
            1: MovieItem(item_id=1, title="The Matrix", genres=["Sci-Fi"], year=1999),
            2: MovieItem(item_id=2, title="Toy Story", genres=["Animation"], year=1995),
        }

    def test_generates_queries(self, sample_items):
        queries = generate_synthetic_queries(sample_items, n_queries=3)

        assert len(queries) == len(sample_items)
        assert 1 in queries
        assert 2 in queries
        assert len(queries[1]) == 3

    def test_query_content(self, sample_items):
        queries = generate_synthetic_queries(sample_items, n_queries=2)

        # Queries should contain title or genre references
        for item_id, item_queries in queries.items():
            for q in item_queries:
                assert len(q) > 0


class TestSearchDataset:
    """Tests for SearchDataset."""

    @pytest.fixture
    def sample_data(self):
        items = {
            1: MovieItem(item_id=1, title="Movie 1", genres=["Action"], year=2000),
            2: MovieItem(item_id=2, title="Movie 2", genres=["Drama"], year=2001),
        }
        queries = {
            1: ["Find Movie 1", "Action movie 2000"],
            2: ["Find Movie 2"],
        }
        interactions = pd.DataFrame({
            "user_id": [1, 1, 2],
            "item_id": [1, 2, 1],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": [1, 2, 3],
        })
        return items, queries, interactions

    def test_dataset_length(self, sample_data):
        items, queries, interactions = sample_data
        dataset = SearchDataset(items, queries, interactions)
        # Should have 3 pairs (2 queries for item 1, 1 for item 2)
        assert len(dataset) == 3

    def test_getitem(self, sample_data):
        items, queries, interactions = sample_data
        dataset = SearchDataset(items, queries, interactions)

        query, item_text, item_id = dataset[0]
        assert isinstance(query, str)
        assert isinstance(item_text, str)
        assert isinstance(item_id, (int, np.integer))


class TestRecDataset:
    """Tests for RecDataset."""

    @pytest.fixture
    def sample_data(self):
        items = {
            1: MovieItem(item_id=1, title="Movie 1", genres=["Action"], year=2000),
            2: MovieItem(item_id=2, title="Movie 2", genres=["Drama"], year=2001),
            3: MovieItem(item_id=3, title="Movie 3", genres=["Comedy"], year=2002),
        }
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 1, 2],
            "rating": [4.0, 3.5, 5.0, 4.0, 4.5],
            "timestamp": [1, 2, 3, 1, 2],
        })
        return items, interactions

    def test_dataset_length(self, sample_data):
        items, interactions = sample_data
        dataset = RecDataset(items, interactions, window_size=5)
        # Should have co-occurrence pairs
        assert len(dataset) > 0

    def test_getitem(self, sample_data):
        items, interactions = sample_data
        dataset = RecDataset(items, interactions, window_size=5)

        if len(dataset) > 0:
            item1_text, item2_text, item_id1, item_id2 = dataset[0]
            assert isinstance(item1_text, str)
            assert isinstance(item2_text, str)
            assert isinstance(item_id1, int)
            assert isinstance(item_id2, int)
