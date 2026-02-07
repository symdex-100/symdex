"""
Symdex-100 Client Facade

Single entry point for programmatic use of Symdex.  Wraps indexing,
search, and statistics behind a clean, instance-based API with optional
async support.

Usage::

    from symdex import Symdex

    # From environment variables
    client = Symdex()

    # With explicit configuration
    from symdex.core.config import SymdexConfig
    client = Symdex(config=SymdexConfig(
        llm_provider="openai",
        openai_api_key="sk-...",
    ))

    # Index a project
    result = client.index("./myproject")
    print(f"Indexed {result.functions_indexed} functions")

    # Search
    hits = client.search("validate user tokens", path="./myproject")
    for hit in hits:
        print(f"{hit.function_name} @ {hit.file_path}:{hit.line_start}")

    # Async variants (for FastAPI / Django async views)
    result = await client.aindex("./myproject")
    hits   = await client.asearch("validate user tokens")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from symdex.core.config import SymdexConfig
from symdex.core.engine import IndexResult, SearchResult
from symdex.exceptions import IndexNotFoundError

logger = logging.getLogger(__name__)


class Symdex:
    """
    High-level Symdex-100 client.

    Each instance carries its own :class:`SymdexConfig` and never
    touches global state, making it safe for multi-tenant services,
    testing, and embedding into third-party applications.

    Args:
        config: Explicit configuration object.  When *None*, a config
            is built from environment variables or keyword overrides.
        validate_on_init: If True, call :meth:`SymdexConfig.validate` in
            __init__ so missing API keys or invalid config surface immediately
            instead of on first :meth:`index` or :meth:`search`. Omit or False
            for embedding when using :attr:`cypher_fallback_only` or when
            validation is done separately.
        **kwargs: Forwarded to :class:`SymdexConfig` when *config* is
            ``None`` (e.g. ``llm_provider="openai"``).
    """

    def __init__(
        self,
        config: SymdexConfig | None = None,
        *,
        validate_on_init: bool = False,
        **kwargs,
    ):
        if config is not None:
            self._config = config
        elif kwargs:
            # Build a config from env, then overlay keyword overrides
            base = SymdexConfig.from_env()
            merged = {
                f.name: kwargs.get(f.name, getattr(base, f.name))
                for f in base.__dataclass_fields__.values()
            }
            self._config = SymdexConfig(**merged)
        else:
            self._config = SymdexConfig.from_env()

        if validate_on_init:
            self._config.validate()

        # Cache search engines per index path to avoid re-creating them
        self._engines: Dict[Path, object] = {}

    # ── Configuration ─────────────────────────────────────────────

    @property
    def config(self) -> SymdexConfig:
        """The active configuration for this client."""
        return self._config

    # ── Indexing ──────────────────────────────────────────────────

    def index(
        self,
        directory: str | Path,
        *,
        force: bool = False,
        dry_run: bool = False,
        show_progress: bool = False,
    ) -> IndexResult:
        """
        Index all supported source files in *directory*.

        Creates a ``.symdex/`` sidecar directory with the SQLite index.
        Source files are **never** modified.

        Args:
            directory: Root directory to index.
            force: Re-index even unchanged files.
            dry_run: Analyse without writing to disk.
            show_progress: Show a tqdm progress bar.

        Returns:
            :class:`IndexResult` with counts and statistics.
        """
        from symdex.core.indexer import IndexingPipeline

        root = Path(directory).resolve()
        pipeline = IndexingPipeline(
            root_dir=root,
            dry_run=dry_run,
            force_reindex=force,
            config=self._config,
            show_progress=show_progress,
        )
        return pipeline.run()

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        path: str | Path = ".",
        strategy: str = "auto",
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> List[SearchResult]:
        """
        Search the Symdex index for functions matching *query*.

        Args:
            query: Natural-language description or Cypher pattern.
            path: Root directory whose ``.symdex/`` index to search.
            strategy: ``'auto'`` | ``'llm'`` | ``'keyword'`` | ``'direct'``.
            max_results: Maximum hits to return.
            min_score: Minimum relevance score threshold.

        Returns:
            Ranked list of :class:`SearchResult` objects. Each result has
            :attr:`~SearchResult.path_root` set when available for
            relativizing :attr:`~SearchResult.file_path` (see ARCHITECTURE).

        Raises:
            IndexNotFoundError: If no index exists at *path*.
        """
        cache_dir = Path(path).resolve() / self._config.symdex_dir
        db = self._config.get_cache_path(cache_dir)
        if not db.exists():
            raise IndexNotFoundError(
                f"No Symdex index found at {db}. "
                "Run 'symdex index <dir>' or client.index() first."
            )

        engine = self._get_engine(cache_dir)
        results = engine.search(query, strategy=strategy, max_results=max_results)

        score_threshold = min_score if min_score is not None else self._config.min_search_score
        return [r for r in results if r.score >= score_threshold]

    def search_by_cypher(
        self,
        pattern: str,
        *,
        path: str | Path = ".",
        max_results: int = 10,
    ) -> List[SearchResult]:
        """
        Search by Cypher pattern directly (no LLM translation).

        Args:
            pattern: Cypher pattern with optional wildcards (``*``).
            path: Root directory whose index to search.
            max_results: Maximum hits to return.

        Returns:
            Ranked list of :class:`SearchResult` objects (each has
            :attr:`~SearchResult.path_root` set when available).

        Raises:
            IndexNotFoundError: If no index exists at *path*.
        """
        cache_dir = Path(path).resolve() / self._config.symdex_dir
        db = self._config.get_cache_path(cache_dir)
        if not db.exists():
            raise IndexNotFoundError(f"No Symdex index found at {db}.")

        engine = self._get_engine(cache_dir)
        return engine.search(pattern, strategy="direct", max_results=max_results)

    # ── Statistics ────────────────────────────────────────────────

    def stats(self, path: str | Path = ".") -> Dict[str, int]:
        """
        Return index statistics for the given directory.

        Args:
            path: Root directory whose index to inspect.

        Returns:
            Dict with ``indexed_files`` and ``indexed_functions`` counts.

        Raises:
            IndexNotFoundError: If no index exists at *path*.
        """
        from symdex.core.engine import CypherCache

        cache_dir = Path(path).resolve() / self._config.symdex_dir
        db = self._config.get_cache_path(cache_dir)
        if not db.exists():
            raise IndexNotFoundError(f"No Symdex index found at {db}.")

        cache = CypherCache(db)
        try:
            return cache.get_stats()
        finally:
            cache.close()

    # ── Async variants ────────────────────────────────────────────
    # These use asyncio.to_thread() to run sync operations off the
    # event loop. They raise the same exceptions as the sync methods
    # (e.g. IndexNotFoundError, ConfigError).

    async def aindex(
        self,
        directory: str | Path,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> IndexResult:
        """Async variant of :meth:`index`. Raises same exceptions as sync."""
        return await asyncio.to_thread(
            self.index, directory, force=force, dry_run=dry_run,
        )

    async def asearch(
        self,
        query: str,
        *,
        path: str | Path = ".",
        strategy: str = "auto",
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> List[SearchResult]:
        """Async variant of :meth:`search`. Raises same exceptions as sync."""
        return await asyncio.to_thread(
            self.search, query,
            path=path, strategy=strategy,
            max_results=max_results, min_score=min_score,
        )

    async def asearch_by_cypher(
        self,
        pattern: str,
        *,
        path: str | Path = ".",
        max_results: int = 10,
    ) -> List[SearchResult]:
        """Async variant of :meth:`search_by_cypher`. Raises same exceptions as sync."""
        return await asyncio.to_thread(
            self.search_by_cypher, pattern,
            path=path, max_results=max_results,
        )

    async def astats(self, path: str | Path = ".") -> Dict[str, int]:
        """Async variant of :meth:`stats`. Raises same exceptions as sync."""
        return await asyncio.to_thread(self.stats, path)

    # ── Health (for agents / status endpoints) ─────────────────────

    def health(self) -> Dict[str, object]:
        """
        Return a small status dict for agents or REST health checks.

        Does not require an index or network. Use for readiness probes
        or reporting which provider/version is active.
        """
        return {
            "version": __import__("symdex", fromlist=["__version__"]).__version__,
            "llm_provider": self._config.llm_provider,
            "cypher_fallback_only": getattr(
                self._config, "cypher_fallback_only", False
            ),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _get_engine(self, cache_dir: Path):
        """Return a cached CypherSearchEngine for *cache_dir*."""
        if cache_dir not in self._engines:
            from symdex.core.search import CypherSearchEngine
            self._engines[cache_dir] = CypherSearchEngine(
                cache_dir, config=self._config,
            )
        return self._engines[cache_dir]
