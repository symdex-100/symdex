"""
Symdex-100 MCP Server

Exposes Symdex indexing and search as tools that AI agents (Claude,
Cursor, Windsurf) can invoke natively via the Model Context Protocol.

Start with::

    symdex mcp                     # stdio transport (default for Cursor)
    symdex mcp --transport sse     # SSE transport for HTTP-based clients

Or programmatically::

    from symdex.mcp.server import create_server
    server = create_server()
    server.run()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from symdex.core.config import Config
from symdex.core.engine import CypherCache, scan_directory

logger = logging.getLogger(__name__)


def create_server():
    """
    Build and return a configured FastMCP server instance.

    Raises ``ImportError`` if ``fastmcp`` is not installed (install
    via ``pip install 'symdex-100[mcp]'``).
    """
    from fastmcp import FastMCP  # type: ignore[import-untyped]

    mcp = FastMCP(
        "Symdex-100",
        description=(
            "Inline semantic fingerprints for 100x faster code search. "
            "Indexes functions across 13+ languages and lets you search "
            "by intent using structured Cypher patterns."
        ),
    )

    # ------------------------------------------------------------------
    # Tool: search_codebase
    # ------------------------------------------------------------------

    @mcp.tool()
    def search_codebase(
        query: str,
        path: str = ".",
        strategy: str = "auto",
        max_results: int = 10,
    ) -> str:
        """Search the Symdex index for functions matching a natural-language
        query or a Cypher pattern (e.g. 'SEC:VAL_TOKEN--*').

        This is **much cheaper** than reading every file: a single call
        replaces thousands of tokens of raw code exploration.

        Args:
            query: Natural-language description or Cypher pattern.
            path: Root directory whose index to search (default: cwd).
            strategy: 'auto' | 'llm' | 'keyword' | 'direct'.
            max_results: Maximum hits to return (default 10).

        Returns:
            JSON array of matching functions with file, line, cypher,
            score, and a short code preview.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        cache_dir = Path(path).resolve() / Config.SYMDEX_DIR
        db = Config.get_cache_path(cache_dir)
        if not db.exists():
            return json.dumps({
                "error": f"No Symdex index found at {db}. "
                         "Run 'symdex index <dir>' first."
            })

        engine = CypherSearchEngine(cache_dir)
        results = engine.search(query, strategy=strategy, max_results=max_results)
        return ResultFormatter.format_json(results)

    # ------------------------------------------------------------------
    # Tool: search_by_cypher
    # ------------------------------------------------------------------

    @mcp.tool()
    def search_by_cypher(
        cypher_pattern: str,
        path: str = ".",
        max_results: int = 10,
    ) -> str:
        """Find code segments matching a Cypher-100 pattern directly.

        Use this when you already know the structured fingerprint, e.g.
        'SEC:VAL_*--SYN' to find all synchronous security validation
        functions.

        Args:
            cypher_pattern: A Cypher pattern with optional wildcards (*).
            path: Root directory whose index to search.
            max_results: Maximum hits to return.

        Returns:
            JSON array of matching functions.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        cache_dir = Path(path).resolve() / Config.SYMDEX_DIR
        db = Config.get_cache_path(cache_dir)
        if not db.exists():
            return json.dumps({"error": f"No index at {db}."})

        engine = CypherSearchEngine(cache_dir)
        results = engine.search(
            cypher_pattern, strategy="direct", max_results=max_results
        )
        return ResultFormatter.format_json(results)

    # ------------------------------------------------------------------
    # Tool: index_directory
    # ------------------------------------------------------------------

    @mcp.tool()
    def index_directory(
        path: str = ".",
        force: bool = False,
    ) -> str:
        """Index all supported source files in a directory and build a
        sidecar search index in ``.symdex/``.

        Source files are **never** modified.  Supports Python, JS/TS,
        Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, and Kotlin.

        Args:
            path: Directory to index.
            force: If True, re-index even unchanged files.

        Returns:
            JSON summary with counts of files scanned, functions indexed,
            and any errors.
        """
        from symdex.core.indexer import IndexingPipeline

        root = Path(path).resolve()
        if not root.is_dir():
            return json.dumps({"error": f"'{path}' is not a directory."})

        pipeline = IndexingPipeline(
            root_dir=root,
            dry_run=False,
            force_reindex=force,
        )
        pipeline.run()

        return json.dumps({
            "files_scanned": pipeline.stats["files_scanned"],
            "files_processed": pipeline.stats["files_processed"],
            "functions_indexed": pipeline.stats["functions_indexed"],
            "errors": pipeline.stats["errors"],
        })

    # ------------------------------------------------------------------
    # Tool: get_index_stats
    # ------------------------------------------------------------------

    @mcp.tool()
    def get_index_stats(path: str = ".") -> str:
        """Return statistics about the Symdex index for a directory.

        Args:
            path: Root directory whose index to inspect.

        Returns:
            JSON with indexed_files and indexed_functions counts.
        """
        cache_dir = Path(path).resolve() / Config.SYMDEX_DIR
        db = Config.get_cache_path(cache_dir)
        if not db.exists():
            return json.dumps({"error": f"No index at {db}."})

        cache = CypherCache(db)
        return json.dumps(cache.get_stats())

    return mcp
