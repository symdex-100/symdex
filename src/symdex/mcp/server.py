"""
Symdex-100 MCP Server

Exposes Symdex indexing and search as tools that AI agents (Claude,
Cursor, Windsurf) can invoke natively via the Model Context Protocol.

Also exposes **resources** (Cypher schema, index stats) and **prompt
templates** for common search workflows.

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

from symdex.core.config import Config, CypherSchema, SymdexConfig
from symdex.core.engine import CypherCache, scan_directory

logger = logging.getLogger(__name__)


def create_server(config: SymdexConfig | None = None):
    """
    Build and return a configured FastMCP server instance.

    Args:
        config: Instance-based configuration.  Defaults to
            ``SymdexConfig.from_env()`` so that the server respects
            the same environment variables as the CLI.

    Raises ``ImportError`` if ``fastmcp`` is not installed (install
    via ``pip install 'symdex-100[mcp]'``).
    """
    from fastmcp import FastMCP  # type: ignore[import-untyped]

    cfg = config or SymdexConfig.from_env()

    mcp = FastMCP(
        "Symdex-100",
        description=(
            "Inline semantic fingerprints for 100x faster code search. "
            "Indexes Python functions and lets you search by intent using "
            "structured Cypher patterns."
        ),
    )

    # ==================================================================
    # Helper — resolve cache path with proper error handling
    # ==================================================================

    def _resolve_cache(path: str) -> Path:
        """Resolve the .symdex cache directory and verify the DB exists.

        Raises ``FileNotFoundError`` (caught by FastMCP as a tool error)
        when no index is found.
        """
        cache_dir = Path(path).resolve() / cfg.symdex_dir
        db = cfg.get_cache_path(cache_dir)
        if not db.exists():
            raise FileNotFoundError(
                f"No Symdex index found at {db}. "
                "Run 'symdex index <dir>' first."
            )
        return cache_dir

    # ==================================================================
    # Tool: search_codebase
    # ==================================================================

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

        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.search(query, strategy=strategy, max_results=max_results)
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: search_by_cypher
    # ==================================================================

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

        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.search(
            cypher_pattern, strategy="direct", max_results=max_results,
        )
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: index_directory
    # ==================================================================

    @mcp.tool()
    def index_directory(
        path: str = ".",
        force: bool = False,
    ) -> str:
        """Index all supported source files in a directory and build a
        sidecar search index in ``.symdex/``.

        Source files are **never** modified.

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
            raise FileNotFoundError(f"'{path}' is not a directory.")

        pipeline = IndexingPipeline(
            root_dir=root,
            dry_run=False,
            force_reindex=force,
            config=cfg,
            show_progress=False,
        )
        result = pipeline.run()

        return json.dumps({
            "files_scanned": result.files_scanned,
            "files_processed": result.files_processed,
            "functions_indexed": result.functions_indexed,
            "errors": result.errors,
        })

    # ==================================================================
    # Tool: get_index_stats
    # ==================================================================

    @mcp.tool()
    def get_index_stats(path: str = ".") -> str:
        """Return statistics about the Symdex index for a directory.

        Args:
            path: Root directory whose index to inspect.

        Returns:
            JSON with indexed_files and indexed_functions counts.
        """
        cache_dir = _resolve_cache(path)
        cache = CypherCache(cfg.get_cache_path(cache_dir))
        return json.dumps(cache.get_stats())

    # ==================================================================
    # Tool: health (readiness check)
    # ==================================================================

    @mcp.tool()
    def health() -> str:
        """Check that the Symdex MCP server is running and responsive.

        Returns:
            JSON with status, provider, and model information.
        """
        return json.dumps({
            "status": "ok",
            "version": "1.0.0",
            "provider": cfg.llm_provider,
            "model": cfg.get_model(),
        })

    # ==================================================================
    # Resource: Cypher schema
    # ==================================================================

    @mcp.resource("symdex://schema/domains")
    def schema_domains() -> str:
        """Return the Cypher domain codes and their descriptions."""
        return json.dumps(CypherSchema.DOMAINS, indent=2)

    @mcp.resource("symdex://schema/actions")
    def schema_actions() -> str:
        """Return the Cypher action codes and their descriptions."""
        return json.dumps(CypherSchema.ACTIONS, indent=2)

    @mcp.resource("symdex://schema/patterns")
    def schema_patterns() -> str:
        """Return the Cypher pattern codes and their descriptions."""
        return json.dumps(CypherSchema.PATTERNS, indent=2)

    @mcp.resource("symdex://schema/full")
    def schema_full() -> str:
        """Return the complete Cypher-100 schema (domains, actions, patterns, common objects)."""
        return json.dumps({
            "format": "DOM:ACT_OBJ--PAT",
            "domains": CypherSchema.DOMAINS,
            "actions": CypherSchema.ACTIONS,
            "patterns": CypherSchema.PATTERNS,
            "common_objects": CypherSchema.COMMON_OBJECT_CODES,
        }, indent=2)

    # ==================================================================
    # Prompt templates
    # ==================================================================

    @mcp.prompt()
    def find_security_functions(path: str = ".") -> str:
        """Pre-built prompt: find all security-related functions."""
        return (
            f"Search the Symdex index at '{path}' for all security-related "
            "functions using the pattern SEC:*_*--*. List each function with "
            "its file, line number, and a brief description of what it does."
        )

    @mcp.prompt()
    def audit_domain(domain: str, path: str = ".") -> str:
        """Pre-built prompt: audit all functions in a specific domain."""
        domain_upper = domain.upper()[:3]
        return (
            f"Search the Symdex index at '{path}' using the pattern "
            f"{domain_upper}:*_*--*. For each result, describe the function's "
            "purpose and flag any potential issues (error handling, security, "
            "performance)."
        )

    @mcp.prompt()
    def explore_codebase(path: str = ".") -> str:
        """Pre-built prompt: get a high-level overview of an indexed codebase."""
        return (
            f"First, call get_index_stats for '{path}' to see the scale. "
            "Then search for each domain (SEC, DAT, NET, SYS, LOG, UI, BIZ, TST) "
            "using '*:*_*--*' patterns. Summarize the codebase architecture: "
            "which domains are most populated, key functions in each, and any "
            "gaps or areas that might need attention."
        )

    return mcp
