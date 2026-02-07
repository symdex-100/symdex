"""
Symdex-100 MCP Server

Exposes Symdex indexing and search as tools that AI agents (Claude,
Cursor, Windsurf) can invoke natively via the Model Context Protocol.

Also exposes **resources** (Cypher schema, index stats) and **prompt
templates** for common search workflows.

Start with::

    symdex mcp                      # stdio transport (default for Cursor)
    symdex mcp --transport http     # HTTP (Streamable) for Smithery/remote clients
    symdex mcp --transport sse      # SSE transport (legacy)

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

    Uses a **single config for the whole server**: all tool invocations
    (search, index, stats) share the same provider, API key, and
    settings. For multi-tenant or per-request config (e.g. different
    API keys per workspace), run separate server processes or a future
    callback-based config resolver.

    Args:
        config: Instance-based configuration.  Defaults to
            ``SymdexConfig.from_env()`` so that the server respects
            the same environment variables as the CLI.

    Raises ``ImportError`` if ``fastmcp`` is not installed (install
    via ``pip install 'symdex-100[mcp]'``).
    """
    from fastmcp import FastMCP  # type: ignore[import-untyped]

    cfg = config or SymdexConfig.from_env()

    mcp = FastMCP("Symdex-100")

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
            "version": "1.1.0",
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


# Server card JSON for Smithery scanning (https://smithery.ai/docs/build/external#server-scanning)
SERVER_CARD = {
    "serverInfo": {"name": "Symdex-100", "version": "1.1.0"},
    "authentication": {"required": False, "schemes": []},
    "tools": [
        {"name": "search_codebase", "description": "Search the Symdex index by natural-language or Cypher pattern.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "strategy": {"type": "string"}, "max_results": {"type": "integer"}}}},
        {"name": "search_by_cypher", "description": "Find code matching a Cypher-100 pattern (no LLM).", "inputSchema": {"type": "object", "properties": {"cypher_pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}}},
        {"name": "index_directory", "description": "Index source files into .symdex/ sidecar.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}}}},
        {"name": "get_index_stats", "description": "Return index statistics for a directory.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
        {"name": "health", "description": "Server readiness check.", "inputSchema": {"type": "object", "properties": {}}},
    ],
    "resources": [
        {"uri": "symdex://schema/domains", "description": "Cypher domain codes"},
        {"uri": "symdex://schema/actions", "description": "Cypher action codes"},
        {"uri": "symdex://schema/patterns", "description": "Cypher pattern codes"},
        {"uri": "symdex://schema/full", "description": "Complete Cypher-100 schema"},
    ],
    "prompts": [
        {"name": "find_security_functions", "description": "Find all security-related functions"},
        {"name": "audit_domain", "description": "Audit all functions in a domain"},
        {"name": "explore_codebase", "description": "High-level codebase overview"},
    ],
}


def run_with_server_card(mcp_server: Any, transport: str = "stdio", **kwargs: Any) -> None:
    """
    Run the MCP server, adding server-card route for HTTP/SSE transports.

    For HTTP or SSE transport, intercepts FastMCP's app creation via uvicorn patching
    to add /.well-known/mcp/server-card.json route for Smithery scanning.
    """
    if transport not in ("http", "sse"):
        mcp_server.run(transport=transport, **kwargs)
        return

    # For HTTP/SSE, patch uvicorn to intercept app creation and add server-card route
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    import uvicorn

    async def serve_server_card(request: Any) -> JSONResponse:
        """Serve the server-card JSON for Smithery scanning."""
        return JSONResponse(SERVER_CARD)

    # Patch uvicorn to intercept app creation and add server-card route
    # FastMCP may use uvicorn.run() or uvicorn.Server() directly
    original_uvicorn_run = uvicorn.run
    
    def add_server_card_route(app_obj: Any) -> None:
        """Helper to add server-card route to a Starlette/FastAPI app."""
        if hasattr(app_obj, "routes") and isinstance(app_obj.routes, list):
            route_exists = any(
                hasattr(route, "path") and route.path == "/.well-known/mcp/server-card.json"
                for route in app_obj.routes
            )
            if not route_exists:
                app_obj.routes.append(
                    Route("/.well-known/mcp/server-card.json", serve_server_card, methods=["GET"])
                )
                logger.info("Added server-card route: /.well-known/mcp/server-card.json")
                return True
        return False

    def patched_uvicorn_run(app_obj: Any, *args: Any, **uvicorn_kwargs: Any) -> None:
        """Patch uvicorn.run to add server-card route before running."""
        logger.debug(f"uvicorn.run called - app type: {type(app_obj)}")
        add_server_card_route(app_obj)
        return original_uvicorn_run(app_obj, *args, **uvicorn_kwargs)

    # Also patch uvicorn.Server.__init__ in case FastMCP uses Server directly
    try:
        import uvicorn.server
        original_server_init = uvicorn.server.Server.__init__
        
        def patched_server_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
            """Patch Server.__init__ to add server-card route."""
            original_server_init(self, config, *args, **kwargs)
            if hasattr(config, "app"):
                logger.debug(f"uvicorn.Server.__init__ - app type: {type(config.app)}")
                add_server_card_route(config.app)
        
        uvicorn.server.Server.__init__ = patched_server_init
        server_patched = True
    except (AttributeError, ImportError):
        server_patched = False
        logger.debug("Could not patch uvicorn.Server.__init__")

    # Apply uvicorn.run patch
    uvicorn.run = patched_uvicorn_run
    try:
        # For Docker/Smithery, bind to 0.0.0.0 so it's accessible from outside container
        host = kwargs.pop("host", "0.0.0.0")
        port = kwargs.pop("port", 8000)
        mcp_server.run(transport=transport, host=host, port=port, **kwargs)
    finally:
        # Restore original
        uvicorn.run = original_uvicorn_run
        if server_patched:
            try:
                import uvicorn.server
                uvicorn.server.Server.__init__ = original_server_init
            except (AttributeError, ImportError):
                pass
