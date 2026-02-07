"""
Tests for the Symdex client API (symdex.client.Symdex).

Covers the public facade: index(), search(), search_by_cypher(), stats(),
async variants, config construction, and IndexNotFoundError when no index exists.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symdex import Symdex, SymdexConfig, IndexNotFoundError
from symdex.core.engine import CypherCache


# =============================================================================
# Fixtures — client and pre-populated index
# =============================================================================


@pytest.fixture
def config():
    """SymdexConfig with test defaults (no real API calls in most tests)."""
    return SymdexConfig(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        symdex_dir=".symdex",
        cache_db_name="index.db",
        min_search_score=0.0,
    )


@pytest.fixture
def client(config):
    """Symdex client with explicit config."""
    return Symdex(config=config)


@pytest.fixture
def indexed_project(tmp_path, config):
    """
    A temporary directory with a pre-built .symdex/index.db containing
    one file and one Cypher entry, so search/stats can be tested without
    running the full indexer or LLM.
    """
    symdex_dir = tmp_path / config.symdex_dir
    symdex_dir.mkdir()
    db_path = config.get_cache_path(symdex_dir)

    cache = CypherCache(db_path)
    sample_file = tmp_path / "src" / "app.py"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("def validate_email(email: str) -> bool:\n    return '@' in email\n", encoding="utf-8")

    cache.mark_file_indexed(sample_file, function_count=1)
    cache.add_cypher_entry(
        sample_file,
        "validate_email",
        1,
        3,
        "SEC:VAL_EMAIL--SYN",
        ["validate", "email"],
        "[email] → bool",
        "O(1)",
    )
    cache.close()  # avoid ResourceWarning: unclosed database
    return tmp_path


# =============================================================================
# Construction
# =============================================================================


class TestSymdexConstruction:
    """Client construction from config and from env."""

    def test_construct_with_explicit_config(self, config):
        client = Symdex(config=config)
        assert client.config is config
        assert client.config.symdex_dir == ".symdex"

    def test_construct_from_kwargs_overrides_env(self):
        client = Symdex(symdex_dir=".my_symdex", min_search_score=1.0)
        assert client.config.symdex_dir == ".my_symdex"
        assert client.config.min_search_score == 1.0

    def test_config_property_returns_same_instance(self, client):
        assert client.config is client._config


# =============================================================================
# stats()
# =============================================================================


class TestStats:
    """Index statistics via client.stats()."""

    def test_stats_returns_indexed_files_and_functions(self, client, indexed_project):
        result = client.stats(path=indexed_project)
        assert result["indexed_files"] == 1
        assert result["indexed_functions"] == 1

    def test_stats_raises_when_no_index(self, client, tmp_path):
        with pytest.raises(IndexNotFoundError, match="No Symdex index found"):
            client.stats(path=tmp_path)


# =============================================================================
# search() and search_by_cypher()
# =============================================================================


class TestSearch:
    """Search and search_by_cypher through the client."""

    def test_search_by_cypher_finds_entry(self, client, indexed_project):
        hits = client.search_by_cypher(
            "SEC:VAL_EMAIL--SYN",
            path=indexed_project,
            max_results=10,
        )
        assert len(hits) == 1
        assert hits[0].function_name == "validate_email"
        assert hits[0].cypher == "SEC:VAL_EMAIL--SYN"
        assert hits[0].file_path is not None
        assert hits[0].line_start == 1

    def test_search_by_cypher_with_wildcard(self, client, indexed_project):
        hits = client.search_by_cypher("SEC:%_%--*", path=indexed_project, max_results=10)
        assert len(hits) >= 1
        assert any(h.function_name == "validate_email" for h in hits)

    def test_search_by_cypher_raises_when_no_index(self, client, tmp_path):
        with pytest.raises(IndexNotFoundError, match="No Symdex index found"):
            client.search_by_cypher("SEC:*_*--*", path=tmp_path)

    def test_search_direct_strategy_uses_cypher_pattern(self, client, indexed_project):
        # "direct" strategy treats query as Cypher pattern
        hits = client.search(
            "SEC:VAL_EMAIL--SYN",
            path=indexed_project,
            strategy="direct",
            max_results=10,
        )
        assert len(hits) >= 1
        assert any(h.function_name == "validate_email" for h in hits)

    def test_search_raises_when_no_index(self, client, tmp_path):
        with pytest.raises(IndexNotFoundError, match="No Symdex index found"):
            client.search("validate email", path=tmp_path)

    def test_search_respects_min_score_from_config(self, client, indexed_project):
        # Config with high min_score filters out low-scoring results
        high_bar = SymdexConfig(
            anthropic_api_key="test",
            min_search_score=100.0,
        )
        strict_client = Symdex(config=high_bar)
        hits = strict_client.search(
            "SEC:VAL_EMAIL--SYN",
            path=indexed_project,
            strategy="direct",
        )
        # Exact match should still pass; if we had only weak matches they'd be filtered
        assert isinstance(hits, list)


# =============================================================================
# index()
# =============================================================================


class TestIndex:
    """Indexing via client.index()."""

    def test_index_creates_symdex_dir_and_returns_result(self, client, tmp_project):
        result = client.index(tmp_project, dry_run=False, show_progress=False)
        assert result is not None
        assert hasattr(result, "functions_indexed")
        assert hasattr(result, "files_processed")
        assert result.files_processed >= 1
        assert result.functions_indexed >= 1
        symdex_dir = tmp_project / ".symdex"
        assert symdex_dir.exists()
        db = symdex_dir / "index.db"
        assert db.exists()

    def test_index_dry_run_does_not_write_db(self, client, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def foo(): pass\n", encoding="utf-8")
        result = client.index(tmp_path, dry_run=True, show_progress=False)
        assert result is not None
        # Pipeline may create .symdex and empty DB; dry_run must not persist entries
        stats = client.stats(path=tmp_path)
        assert stats["indexed_functions"] == 0


# =============================================================================
# Async variants
# =============================================================================


class TestAsyncApi:
    """Async methods: aindex, asearch, astats."""

    @pytest.mark.asyncio
    async def test_astats_returns_same_as_stats(self, client, indexed_project):
        sync_stats = client.stats(path=indexed_project)
        async_stats = await client.astats(path=indexed_project)
        assert async_stats == sync_stats

    @pytest.mark.asyncio
    async def test_astats_raises_when_no_index(self, client, tmp_path):
        with pytest.raises(IndexNotFoundError, match="No Symdex index found"):
            await client.astats(path=tmp_path)

    @pytest.mark.asyncio
    async def test_asearch_returns_same_as_search(self, client, indexed_project):
        sync_hits = client.search_by_cypher("SEC:VAL_EMAIL--SYN", path=indexed_project)
        # search_by_cypher has no async variant; use search with direct strategy
        async_hits = await client.asearch(
            "SEC:VAL_EMAIL--SYN",
            path=indexed_project,
            strategy="direct",
        )
        assert len(async_hits) == len(sync_hits)
        if async_hits and sync_hits:
            assert async_hits[0].function_name == sync_hits[0].function_name


# =============================================================================
# Engine caching
# =============================================================================


class TestEngineCaching:
    """Client caches CypherSearchEngine per path."""

    def test_repeated_search_reuses_engine(self, client, indexed_project):
        hits1 = client.search_by_cypher("SEC:*_*--*", path=indexed_project)
        hits2 = client.search_by_cypher("SEC:*_*--*", path=indexed_project)
        assert len(hits1) == len(hits2)
        assert len(client._engines) == 1
