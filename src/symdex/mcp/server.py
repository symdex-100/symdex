"""
Symdex-100 MCP Server

Exposes Symdex indexing and search as tools that AI agents (Claude,
Cursor, Windsurf) can invoke natively via the Model Context Protocol.

Also exposes **resources** (Cypher schema, index stats) and **prompt
templates** for common search workflows.

Start with::

    symdex mcp                              # stdio transport (default for Cursor)
    symdex mcp --transport streamable-http  # HTTP (Streamable) for Smithery/remote
    symdex mcp --transport sse              # SSE transport (legacy)

Or programmatically::

    from symdex.mcp.server import create_server
    server = create_server()
    server.run()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

# FastMCP uses pydantic for validation, so Field should be available
# If ImportError occurs, it indicates fastmcp[mcp] extra wasn't installed
from pydantic import Field  # type: ignore[import-untyped]

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
        query: Annotated[
            str,
            Field(description="Natural-language query (e.g., 'validate user tokens') or Cypher pattern (e.g., 'SEC:VAL_TOKEN--*'). Use natural language for intent-based search; use Cypher patterns when you know the exact semantic fingerprint structure.")
        ],
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to search. Defaults to current working directory ('.'). The index must exist at <path>/.symdex/index.db.")
        ] = ".",
        strategy: Annotated[
            str,
            Field(default="auto", description="Search strategy: 'auto' (try LLM translation first, fallback to keyword), 'llm' (always use LLM to translate query to Cypher), 'keyword' (direct keyword matching), 'direct' (treat query as Cypher pattern). Use 'auto' unless you need specific behavior.")
        ] = "auto",
        max_results: Annotated[
            int | None,
            Field(default=None, description="Maximum number of results to return. If None, uses default from config (typically 10). Increase for broader exploration, decrease for focused results.")
        ] = None,
        context_lines: Annotated[
            int | None,
            Field(default=None, description="Number of lines of code context to include per result. Default is 3 (good for exploration). Use 10-15 when you need to edit the found code (more context = more tokens but better editing capability).")
        ] = None,
        exclude_tests: Annotated[
            bool | None,
            Field(default=None, description="If True, exclude test functions from results. If None, uses config default (typically True, excluding tests). Set to False to include test code in search results.")
        ] = None,
    ) -> str:
        """Search the Symdex index for functions matching a natural-language
        query or a Cypher pattern (e.g. 'SEC:VAL_TOKEN--*').

        This is **much cheaper** than reading every file: a single call
        replaces thousands of tokens of raw code exploration.

        **When to use this tool:**
        - You need to find code by intent (e.g., "where do we validate passwords")
        - You want to reduce context window usage (instead of reading 10 files, get 1-3 precise hits)
        - You'd otherwise read 3+ files to find a function
        - The codebase has been indexed (check with get_index_stats first)

        **When NOT to use:**
        - You already know the exact file and line (just read it directly)
        - You're searching for a specific string/identifier (use grep/IDE search instead)
        - The project hasn't been indexed (call index_directory first)

        Returns:
            JSON array of matching functions with file, line, cypher,
            score, and a short code preview.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        n = max_results if max_results is not None else cfg.default_max_search_results
        ctx_lines = context_lines if context_lines is not None else cfg.default_context_lines
        no_tests = exclude_tests if exclude_tests is not None else cfg.default_exclude_tests

        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.search(query, strategy=strategy, max_results=n, context_lines=ctx_lines, exclude_tests=no_tests)
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: search_by_cypher
    # ==================================================================

    @mcp.tool()
    def search_by_cypher(
        cypher_pattern: Annotated[
            str,
            Field(description="A Cypher-100 pattern with optional wildcards (*). Format: DOM:ACT_OBJ--PAT. Examples: 'SEC:VAL_*--SYN' (all sync security validation), 'NET:SND_*--ASY' (all async network send), 'DAT:*_*--*' (all data-domain functions). Use wildcards (*) for any slot you're unsure about.")
        ],
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to search. Defaults to current working directory ('.'). The index must exist at <path>/.symdex/index.db.")
        ] = ".",
        max_results: Annotated[
            int | None,
            Field(default=None, description="Maximum number of results to return. If None, uses default from config (typically 10). This is faster than search_codebase since it skips LLM translation.")
        ] = None,
    ) -> str:
        """Find code segments matching a Cypher-100 pattern directly.

        Use this when you already know the structured fingerprint, e.g.
        'SEC:VAL_*--SYN' to find all synchronous security validation
        functions. This is faster than search_codebase since it skips
        LLM translation.

        **When to use this tool:**
        - You already know the Cypher pattern structure
        - You want deterministic, fast pattern matching (no LLM calls)
        - You're exploring a specific domain (e.g., all SEC functions)

        **Learn Cypher patterns:** Read the symdex://schema/full resource
        to understand valid domain/action/pattern codes.

        Returns:
            JSON array of matching functions.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        n = max_results if max_results is not None else cfg.default_max_search_results
        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.search(
            cypher_pattern, strategy="direct", max_results=n,
        )
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: index_directory
    # ==================================================================

    @mcp.tool()
    def index_directory(
        path: Annotated[
            str,
            Field(default=".", description="Directory path to index. Defaults to current working directory ('.'). All supported source files in this directory and subdirectories will be scanned and indexed.")
        ] = ".",
        force: Annotated[
            bool,
            Field(default=False, description="If True, re-index all files even if they haven't changed (ignores hash cache). Use when you suspect the index is stale or after major refactoring. Default False only indexes changed files.")
        ] = False,
    ) -> str:
        """Index all supported source files in a directory and build a
        sidecar search index in ``.symdex/``.

        Source files are **never** modified. The index is stored in
        ``<path>/.symdex/index.db``.

        **When to use this tool:**
        - get_index_stats shows 0 indexed files
        - You've added new source files
        - You've made significant code changes and want to refresh the index
        - This is the first time indexing this codebase

        **When NOT needed:**
        - The index already exists and is up-to-date
        - You're only searching (use search_codebase instead)

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
    def get_index_stats(
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to inspect. Defaults to current working directory ('.'). Checks for index at <path>/.symdex/index.db.")
        ] = ".",
    ) -> str:
        """Return statistics about the Symdex index for a directory.

        **Use this first** before searching to verify an index exists.
        If indexed_files is 0, call index_directory first.

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
            "version": "1.2.0",
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

    @mcp.prompt()
    def when_to_use_symdex() -> str:
        """Guidance prompt: explains when and how AI agents should use Symdex tools."""
        return (
            "**When to use Symdex tools:**\n"
            "1. Finding code by intent (e.g., 'where do we validate passwords') - use search_codebase\n"
            "2. Reducing context window usage - instead of reading 10 files, get 1-3 precise hits\n"
            "3. Understanding codebase structure - use get_index_stats, then search by domain\n"
            "4. You'd otherwise read 3+ files to find a function\n"
            "\n"
            "**When NOT to use Symdex:**\n"
            "1. You already know the exact file and line - just read it directly\n"
            "2. Searching for specific strings/identifiers - use grep/IDE search instead\n"
            "3. Project hasn't been indexed - call index_directory first\n"
            "4. Very small codebases (<50 functions) - indexing overhead outweighs benefits\n"
            "\n"
            "**Workflow:**\n"
            "1. Check: get_index_stats('.') - verify index exists\n"
            "2. If needed: index_directory('.') - create/update index\n"
            "3. Search: search_codebase('your query', context_lines=3) - exploration\n"
            "4. Edit: search_codebase('your query', context_lines=15) - more context for editing\n"
            "5. Read: Open the top result's file at the specific line\n"
        )

    # ==================================================================
    # Tool: get_callers (call graph)
    # ==================================================================

    @mcp.tool()
    def get_callers(
        function_name: Annotated[
            str,
            Field(description="Name of the function to find callers for (e.g., 'encrypt_file_content'). Searches the call graph built during indexing.")
        ],
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to search. Defaults to current working directory ('.').")
        ] = ".",
        context_lines: Annotated[
            int | None,
            Field(default=None, description="Number of lines of code context to include per result. Default uses config (typically 3).")
        ] = None,
    ) -> str:
        """Find all indexed functions that call the specified function.

        Use this to answer **"who calls X?"** — e.g., to trace which
        functions invoke ``encrypt_file_content``, or to understand where
        a utility is used across the codebase.

        Requires the codebase to have been indexed (call edges are
        extracted during ``index_directory``).

        Returns:
            JSON array of caller functions with file, line, cypher, and context.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        ctx_lines = context_lines if context_lines is not None else cfg.default_context_lines
        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.get_callers(function_name, context_lines=ctx_lines)
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: get_callees (call graph)
    # ==================================================================

    @mcp.tool()
    def get_callees(
        function_name: Annotated[
            str,
            Field(description="Name of the function to find callees for (e.g., 'process_files'). Searches the call graph built during indexing.")
        ],
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to search. Defaults to current working directory ('.').")
        ] = ".",
        file_path: Annotated[
            str | None,
            Field(default=None, description="Optional source file path to disambiguate when the function name exists in multiple files.")
        ] = None,
        context_lines: Annotated[
            int | None,
            Field(default=None, description="Number of lines of code context to include per result. Default uses config (typically 3).")
        ] = None,
    ) -> str:
        """Find all indexed functions called by the specified function.

        Use this to answer **"what does X call?"** — e.g., to trace the
        execution flow from ``process_files`` down to its dependencies.

        Only returns callees that are themselves indexed (external/built-in
        calls like ``print`` or ``len`` are excluded).

        Returns:
            JSON array of callee functions with file, line, cypher, and context.
        """
        from symdex.core.search import CypherSearchEngine, ResultFormatter

        ctx_lines = context_lines if context_lines is not None else cfg.default_context_lines
        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        results = engine.get_callees(function_name, file_path=file_path, context_lines=ctx_lines)
        return ResultFormatter.format_json(results)

    # ==================================================================
    # Tool: trace_call_chain (recursive call graph walk)
    # ==================================================================

    @mcp.tool()
    def trace_call_chain(
        function_name: Annotated[
            str,
            Field(description="Starting function name for the trace (e.g., 'encrypt_file_content').")
        ],
        path: Annotated[
            str,
            Field(default=".", description="Root directory path whose Symdex index to search.")
        ] = ".",
        direction: Annotated[
            str,
            Field(default="callers", description="Direction to trace: 'callers' (who calls this, walking up) or 'callees' (what this calls, walking down).")
        ] = "callers",
        max_depth: Annotated[
            int,
            Field(default=5, description="Maximum depth to trace (default 5). Higher values trace further but may return more results.")
        ] = 5,
        context_lines: Annotated[
            int | None,
            Field(default=None, description="Lines of code context per result.")
        ] = None,
    ) -> str:
        """Trace the call chain from a function, walking up (callers) or down (callees).

        This **recursively** follows the call graph to show the full execution flow.
        For example, tracing callers of ``encrypt_file_content`` might reveal::

            depth 1: encrypt_file_in_place   (calls encrypt_file_content)
            depth 2: process_files_batch     (calls encrypt_file_in_place)
            depth 3: process_files           (calls process_files_batch)

        Cycles are detected automatically and will not cause infinite recursion.

        Returns:
            JSON object with ``root``, ``direction``, ``max_depth``, and a
            ``chain`` array of nodes ordered by depth (nearest first).
        """
        from symdex.core.search import CypherSearchEngine

        ctx_lines = context_lines if context_lines is not None else cfg.default_context_lines
        cache_dir = _resolve_cache(path)
        engine = CypherSearchEngine(cache_dir, config=cfg)
        chain = engine.trace_call_chain(
            function_name, direction=direction,
            max_depth=max_depth, context_lines=ctx_lines,
        )

        return json.dumps({
            "root": function_name,
            "direction": direction,
            "max_depth": max_depth,
            "chain": chain,
        }, indent=2)

    return mcp


# Server card JSON for Smithery scanning (https://smithery.ai/docs/build/external#server-scanning)
SERVER_CARD = {
    "serverInfo": {"name": "Symdex-100", "version": "1.2.0"},
    "authentication": {"required": False, "schemes": []},
    "tools": [
        {"name": "search_codebase", "description": "Search the Symdex index by natural-language or Cypher pattern.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "strategy": {"type": "string"}, "max_results": {"type": "integer"}}}},
        {"name": "search_by_cypher", "description": "Find code matching a Cypher-100 pattern (no LLM).", "inputSchema": {"type": "object", "properties": {"cypher_pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}}},
        {"name": "index_directory", "description": "Index source files into .symdex/ sidecar.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}}}},
        {"name": "get_index_stats", "description": "Return index statistics for a directory.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
        {"name": "health", "description": "Server readiness check.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_callers", "description": "Find functions that call a given function (call graph).", "inputSchema": {"type": "object", "properties": {"function_name": {"type": "string"}, "path": {"type": "string"}, "context_lines": {"type": "integer"}}, "required": ["function_name"]}},
        {"name": "get_callees", "description": "Find functions called by a given function (call graph).", "inputSchema": {"type": "object", "properties": {"function_name": {"type": "string"}, "path": {"type": "string"}, "file_path": {"type": "string"}, "context_lines": {"type": "integer"}}, "required": ["function_name"]}},
        {"name": "trace_call_chain", "description": "Recursively trace the call chain (callers or callees) from a function.", "inputSchema": {"type": "object", "properties": {"function_name": {"type": "string"}, "path": {"type": "string"}, "direction": {"type": "string"}, "max_depth": {"type": "integer"}, "context_lines": {"type": "integer"}}, "required": ["function_name"]}},
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
        {"name": "when_to_use_symdex", "description": "Guidance on when and how AI agents should use Symdex tools"},
    ],
}


def run_with_server_card(mcp_server: Any, transport: str = "stdio", **kwargs: Any) -> None:
    """
    Run the MCP server, adding server-card route for HTTP transports.

    For streamable-http or SSE transport, intercepts FastMCP's app creation via uvicorn patching
    to add /.well-known/mcp/server-card.json route for Smithery scanning.
    """
    if transport not in ("streamable-http", "http", "sse"):
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

    def add_cors_for_mcp(app_obj: Any) -> None:
        """Add CORS middleware so gateways (e.g. Smithery) can read Mcp-Session-Id."""
        if not hasattr(app_obj, "add_middleware"):
            return
        try:
            from starlette.middleware.cors import CORSMiddleware
            # Expose Mcp-Session-Id so remote clients (Smithery Gateway) can complete init
            app_obj.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization", "Mcp-Session-Id", "mcp-session-id"],
                expose_headers=["Mcp-Session-Id", "mcp-session-id"],
            )
            logger.info("Added CORS middleware (expose Mcp-Session-Id for remote gateways)")
        except ImportError:
            logger.debug("CORSMiddleware not available, skipping CORS")

    def patched_uvicorn_run(app_obj: Any, *args: Any, **uvicorn_kwargs: Any) -> None:
        """Patch uvicorn.run: add server-card route, CORS, and respect PORT env (Fly.io/Railway)."""
        import os
        logger.debug(f"uvicorn.run called - app type: {type(app_obj)}")
        add_server_card_route(app_obj)
        add_cors_for_mcp(app_obj)
        # Fly.io/Railway set PORT; FastMCP may ignore run(port=...), so force it here
        env_port = os.environ.get("PORT")
        if env_port:
            try:
                uvicorn_kwargs["port"] = int(env_port)
                logger.info("Using port %s from PORT env", env_port)
            except ValueError:
                pass
        return original_uvicorn_run(app_obj, *args, **uvicorn_kwargs)

    # Also patch uvicorn.Server.__init__ in case FastMCP uses Server directly
    try:
        import uvicorn.server
        original_server_init = uvicorn.server.Server.__init__
        
        def patched_server_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
            """Patch Server.__init__ to add server-card route and CORS."""
            original_server_init(self, config, *args, **kwargs)
            if hasattr(config, "app"):
                logger.debug(f"uvicorn.Server.__init__ - app type: {type(config.app)}")
                add_server_card_route(config.app)
                add_cors_for_mcp(config.app)
        
        uvicorn.server.Server.__init__ = patched_server_init
        server_patched = True
    except (AttributeError, ImportError):
        server_patched = False
        logger.debug("Could not patch uvicorn.Server.__init__")

    # Apply uvicorn.run patch
    uvicorn.run = patched_uvicorn_run
    try:
        import os
        # For Docker/Smithery, bind to 0.0.0.0 so it's accessible from outside container
        host = kwargs.pop("host", "0.0.0.0")
        # Port: from kwargs (CLI passes from PORT env), else PORT env, else 8000
        port = kwargs.pop("port", None)
        if port is None:
            port = int(os.environ.get("PORT", "8000"))
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
