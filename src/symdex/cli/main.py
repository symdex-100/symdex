"""
Symdex-100 CLI

Unified command-line interface for indexing and searching code.

Usage::

    symdex index ./src              # Index a directory
    symdex search "validate user"   # Natural-language search
    symdex stats                    # Show index statistics
    symdex mcp                      # Start the MCP server
"""

import logging
import warnings
from pathlib import Path

import click

# Suppress noisy dependency warnings during index/search
warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*", category=UserWarning, module="anthropic")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="anyio")

from symdex.core.config import Config, SymdexConfig
from symdex.core.engine import CypherCache, scan_directory
from symdex.core.search import CypherSearchEngine, ResultFormatter


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool, config: SymdexConfig | None = None) -> None:
    """Set up logging for the CLI session."""
    cfg = config or SymdexConfig.from_env()
    level = logging.DEBUG if verbose else getattr(logging, cfg.log_level)
    logging.basicConfig(level=level, format=cfg.log_format)
    # Suppress noisy HTTP loggers and third-party debug output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def _build_config(provider: str | None = None) -> SymdexConfig:
    """Create a SymdexConfig from env vars with an optional provider override."""
    cfg = SymdexConfig.from_env()
    if provider:
        # Dataclass is not frozen, so we can override the provider
        cfg = SymdexConfig(
            **{
                **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
                "llm_provider": provider,
            }
        )
    return cfg


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="symdex-100")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "gemini"]),
    default=None,
    envvar="SYMDEX_LLM_PROVIDER",
    help="LLM provider override (default: $SYMDEX_LLM_PROVIDER or 'anthropic').",
)
@click.pass_context
def cli(ctx: click.Context, provider: str | None):
    """Symdex-100 — Semantic fingerprints for 100x faster code search.

    Commands:
      index   Build a sidecar index from Python source (uses LLM for Cyphers).
      search  Query the index by natural language or Cypher pattern.
      stats   Show index statistics (files, functions, call edges).
      callers Find functions that call a given function (call graph).
      callees Find functions called by a given function (call graph).
      trace   Recursively trace the call chain (callers or callees).
      watch   Watch a directory and auto-reindex on file changes.
      mcp     Start the MCP server for Cursor / Claude.

    Configuration is read from the environment (e.g. ANTHROPIC_API_KEY,
    SYMDEX_DEFAULT_MAX_RESULTS). See docs/CONFIGURATION.md for all options.
    """
    ctx.ensure_object(dict)
    # Build an instance-based config and attach it to the Click context.
    # All subcommands read it from ctx.obj["config"].
    ctx.obj["config"] = _build_config(provider)
    # Also update the legacy global Config for backward compat
    if provider:
        Config.LLM_PROVIDER = provider


# ---------------------------------------------------------------------------
# symdex index
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Override sidecar index directory (default: DIRECTORY/.symdex).")
@click.option("--force", is_flag=True,
              help="Force re-index all files, ignoring hash cache (slower, use after big refactors).")
@click.option("--dry-run", is_flag=True,
              help="Analyse code and show what would be indexed without writing to disk.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def index(ctx: click.Context, directory: str, cache_dir: str | None,
          force: bool, dry_run: bool, verbose: bool):
    """Index Python source files in DIRECTORY and build a sidecar search index.

    Metadata is stored in a .symdex/ directory — source files are never
    modified. Requires an LLM API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or
    GEMINI_API_KEY) unless SYMDEX_CYPHER_FALLBACK_ONLY=1. Python-only for v1.
    """
    cfg: SymdexConfig = ctx.obj["config"]
    _configure_logging(verbose, cfg)
    _validate_config(cfg)

    root_dir = Path(directory).resolve()
    cache_path = Path(cache_dir).resolve() if cache_dir else None

    # Lazy import so help text is instant even without the LLM SDK installed
    from symdex.core.indexer import IndexingPipeline  # noqa: E402

    try:
        pipeline = IndexingPipeline(
            root_dir=root_dir,
            cache_dir=cache_path,
            dry_run=dry_run,
            force_reindex=force,
            config=cfg,
            show_progress=True,
        )
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    pipeline.run()
    pipeline.print_statistics()


# ---------------------------------------------------------------------------
# symdex search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory containing the index (default: ./.symdex).")
@click.option("-f", "--format", "fmt",
              type=click.Choice(["console", "json", "compact", "ide"]),
              default="console",
              help="Output: console (rich), json, compact (grep-like), ide (file:line).")
@click.option("-n", "--max-results", type=int, default=None,
              help="Maximum number of results (default: from config, typically 5).")
@click.option("-p", "--page-size", type=int, default=None,
              help="Results per page; enables interactive pagination (Enter=next, q=quit).")
@click.option("--strategy",
              type=click.Choice(["auto", "llm", "keyword", "direct"]),
              default="auto",
              help="auto=LLM or keyword fallback, llm=force LLM, keyword=no LLM, direct=Cypher only.")
@click.option("--min-score", type=float, default=None,
              help="Minimum relevance score to show (default: CYPHER_MIN_SCORE or 5.0).")
@click.option("--context-lines", type=int, default=3,
              help="Lines of code preview per result (default: 3; use 10-15 for editing).")
@click.option("--include-tests", is_flag=True,
              help="Include test functions (by default tests are excluded).")
@click.option("--explain", is_flag=True,
              help="Print scoring breakdown per result (action/object/domain/name contributions).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def search(ctx: click.Context, query: str, cache_dir: str | None, fmt: str,
           max_results: int | None, page_size: int | None,
           strategy: str, min_score: float | None, context_lines: int,
           include_tests: bool, explain: bool, verbose: bool):
    """Search the Symdex index with a natural-language QUERY or Cypher pattern.

    QUERY can be a phrase (e.g. "validate user token") or a Cypher pattern
    (e.g. SEC:VAL_*--ASY). Use --explain to see why results rank as they do.
    """
    cfg: SymdexConfig = ctx.obj["config"]
    _configure_logging(verbose, cfg)
    _validate_config(cfg)

    cache_path = Path(cache_dir).resolve() if cache_dir else (Path.cwd() / cfg.symdex_dir)
    db = cfg.get_cache_path(cache_path)
    if not db.exists():
        click.echo(f"Error: Index database not found at {db}", err=True)
        click.echo("Run 'symdex index <dir>' first to build the index.", err=True)
        if cache_dir:
            click.echo(
                "Hint: In Docker, --cache-dir must be the path *inside* the container. "
                "Mount the indexed project, e.g.: docker compose run -v E:/CodeDD:/data "
                "symdex symdex search \"...\" --cache-dir /data/.symdex",
                err=True,
            )
        raise SystemExit(1)

    try:
        engine = CypherSearchEngine(cache_path, config=cfg)
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)
    exclude_tests = not include_tests  # Default: exclude tests (normal use)
    results = engine.search(query, strategy=strategy, max_results=max_results, context_lines=context_lines, exclude_tests=exclude_tests, explain=explain)

    # Apply minimum score filter
    score_threshold = min_score if min_score is not None else cfg.min_search_score
    results = [r for r in results if r.score >= score_threshold]

    # Elapsed time: DB + scoring + context only (excludes LLM translation)
    elapsed = engine.last_search_db_elapsed_seconds

    formatter = ResultFormatter()

    # ── Non-console formats: dump everything at once ─────────
    if fmt == "json":
        click.echo(formatter.format_json(results))
        timing_str = f"{elapsed:.3f}".replace(',', '.')
        click.echo(f"  Completed in {timing_str} seconds")
    elif fmt == "compact":
        click.echo(formatter.format_compact(results))
        timing_str = f"{elapsed:.3f}".replace(',', '.')
        click.echo(f"  Completed in {timing_str} seconds")
    elif fmt == "ide":
        click.echo(formatter.format_ide(results))
        timing_str = f"{elapsed:.3f}".replace(',', '.')
        click.echo(f"  Completed in {timing_str} seconds")
    else:
        # ── Console format with optional pagination (timing in header) ──
        _display_console_results(results, formatter, page_size, elapsed)


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _display_console_results(
    results: list,
    formatter: ResultFormatter,
    page_size: int | None,
    elapsed_time: float,
) -> None:
    """
    Print search results to the console, optionally paginated.

    When *page_size* is ``None`` (the default) all results are printed
    at once — identical to the previous behaviour.  When set, results
    are split into pages and the user can navigate interactively.

    Pagination commands (case-insensitive):
      Enter / n  — next page
      b / back   — previous page
      p / print  — reprint current page
      j / json   — dump current page as JSON
      q / quit   — stop
    """
    if not results or not page_size or page_size <= 0:
        click.echo(formatter.format_console(results, elapsed_time=elapsed_time))
        return

    import math
    total = len(results)
    total_pages = math.ceil(total / page_size)
    page_num = 0

    while page_num < total_pages:
        start = page_num * page_size
        end = min(start + page_size, total)
        page = results[start:end]

        click.echo(formatter.format_console(
            page,
            start_index=start + 1,
            total_count=total,
            elapsed_time=elapsed_time,
        ))

        if total_pages == 1 or page_num == total_pages - 1:
            if total_pages > 1:
                click.echo(f"  Page {page_num + 1}/{total_pages} (end)")
            break

        try:
            hint = (
                f"  Page {page_num + 1}/{total_pages}  "
                f"[Enter] next  [b] back  [p] print  [j] json  [q] quit"
            )
            answer = click.prompt(hint, default="", show_default=False)
            cmd = answer.strip().lower()
        except (KeyboardInterrupt, EOFError):
            click.echo("\n  Stopped.")
            return

        if cmd in ("q", "quit", "exit"):
            click.echo("  Stopped.")
            return
        elif cmd in ("b", "back"):
            if page_num > 0:
                page_num -= 1
            else:
                click.echo("  Already on the first page.")
            continue
        elif cmd in ("p", "print"):
            continue
        elif cmd in ("j", "json"):
            click.echo(formatter.format_json(page))
            continue
        else:
            page_num += 1


# ---------------------------------------------------------------------------
# symdex stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory containing the index (default: ./.symdex).")
@click.pass_context
def stats(ctx: click.Context, cache_dir: str | None):
    """Show Symdex index statistics (indexed files, functions, call edges)."""
    cfg: SymdexConfig = ctx.obj["config"]
    cache_path = Path(cache_dir).resolve() if cache_dir else (Path.cwd() / cfg.symdex_dir)
    db = cfg.get_cache_path(cache_path)
    if not db.exists():
        click.echo(f"No index found at {db}. Run 'symdex index' first.", err=True)
        raise SystemExit(1)

    cache = CypherCache(db)
    s = cache.get_stats()
    click.echo("─" * 50)
    click.echo("  SYMDEX — Index Statistics")
    click.echo("─" * 50)
    click.echo(f"  Index location : {cache_path}")
    click.echo()
    click.echo(f"  Indexed files     {s['indexed_files']:>8,}")
    click.echo(f"  Indexed functions {s['indexed_functions']:>8,}")
    click.echo(f"  Call edges        {s.get('call_edges', 0):>8,}  (for callers/callees/trace)")
    click.echo("─" * 50)


# ---------------------------------------------------------------------------
# symdex callers
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("function_name")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory containing the index (default: ./.symdex).")
@click.option("--context-lines", type=int, default=3,
              help="Lines of code preview per result (default: 3).")
@click.option("-f", "--format", "fmt",
              type=click.Choice(["console", "json", "compact", "ide"]),
              default="console",
              help="Output format (default: console).")
@click.pass_context
def callers(ctx: click.Context, function_name: str, cache_dir: str | None,
            context_lines: int, fmt: str):
    """Find functions that call FUNCTION_NAME (call graph).

    Requires an index built with 'symdex index'. Call edges are extracted
    at index time. Example: symdex callers add_cypher_entry
    """
    cfg: SymdexConfig = ctx.obj["config"]
    cache_path = Path(cache_dir).resolve() if cache_dir else (Path.cwd() / cfg.symdex_dir)
    db = cfg.get_cache_path(cache_path)
    if not db.exists():
        click.echo(f"Error: Index not found at {db}. Run 'symdex index' first.", err=True)
        raise SystemExit(1)
    engine = CypherSearchEngine(cache_path, config=cfg)
    results = engine.get_callers(function_name, context_lines=context_lines)
    _emit_call_graph_results(results, fmt, caller_or_callee=f"Callers of '{function_name}' (functions that call it):")


# ---------------------------------------------------------------------------
# symdex callees
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("function_name")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory containing the index (default: ./.symdex).")
@click.option("--file-path", type=click.Path(), default=None,
              help="Source file path to disambiguate when function name exists in multiple files.")
@click.option("--context-lines", type=int, default=3,
              help="Lines of code preview per result (default: 3).")
@click.option("-f", "--format", "fmt",
              type=click.Choice(["console", "json", "compact", "ide"]),
              default="console",
              help="Output format (default: console).")
@click.pass_context
def callees(ctx: click.Context, function_name: str, cache_dir: str | None,
            file_path: str | None, context_lines: int, fmt: str):
    """Find functions called by FUNCTION_NAME (call graph).

    Only returns callees that are themselves indexed. Example: symdex callees _process_function
    """
    cfg: SymdexConfig = ctx.obj["config"]
    cache_path = Path(cache_dir).resolve() if cache_dir else (Path.cwd() / cfg.symdex_dir)
    db = cfg.get_cache_path(cache_path)
    if not db.exists():
        click.echo(f"Error: Index not found at {db}. Run 'symdex index' first.", err=True)
        raise SystemExit(1)
    engine = CypherSearchEngine(cache_path, config=cfg)
    results = engine.get_callees(function_name, file_path=file_path, context_lines=context_lines)
    _emit_call_graph_results(results, fmt, caller_or_callee=f"Callees of '{function_name}' (functions it calls):")


# ---------------------------------------------------------------------------
# symdex trace
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("function_name")
@click.option("--direction",
              type=click.Choice(["callers", "callees"]),
              default="callers",
              help="callers = who calls this (walk up), callees = what this calls (walk down).")
@click.option("--depth", "max_depth", type=int, default=5,
              help="Maximum recursion depth (default: 5).")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory containing the index (default: ./.symdex).")
@click.option("--context-lines", type=int, default=2,
              help="Lines of code preview per node (default: 2).")
@click.option("-f", "--format", "fmt",
              type=click.Choice(["console", "json"]),
              default="console",
              help="Output format (default: console).")
@click.pass_context
def trace(ctx: click.Context, function_name: str, direction: str, max_depth: int,
          cache_dir: str | None, context_lines: int, fmt: str):
    """Trace the call chain from FUNCTION_NAME (call graph).

    Walk up (who calls this) or down (what this calls) recursively.
    Cycles are detected and will not cause infinite recursion.
    Example: symdex trace add_cypher_entry --direction callers --depth 4
    """
    cfg: SymdexConfig = ctx.obj["config"]
    cache_path = Path(cache_dir).resolve() if cache_dir else (Path.cwd() / cfg.symdex_dir)
    db = cfg.get_cache_path(cache_path)
    if not db.exists():
        click.echo(f"Error: Index not found at {db}. Run 'symdex index' first.", err=True)
        raise SystemExit(1)
    engine = CypherSearchEngine(cache_path, config=cfg)
    chain = engine.trace_call_chain(
        function_name, direction=direction, max_depth=max_depth, context_lines=context_lines,
    )
    if fmt == "json":
        import json
        click.echo(json.dumps({"root": function_name, "direction": direction, "max_depth": max_depth, "chain": chain}, indent=2))
    else:
        click.echo(f"  SYMDEX — Call chain from '{function_name}' ({direction}, max_depth={max_depth})")
        click.echo("─" * 60)
        for node in chain:
            indent = "  " * node.get("depth", 0)
            click.echo(f"  {indent}depth {node['depth']}: {node['function_name']} @ {node['file_path']}:{node['line_start']}")
        click.echo("─" * 60)


def _emit_call_graph_results(results: list, fmt: str, *, caller_or_callee: str = "") -> None:
    """Print get_callers/get_callees results in the chosen format.

    caller_or_callee: short label for the relationship, e.g. "Callees of get_stats"
    or "Callers of add_cypher_entry", used in console header when results are non-empty.
    """
    formatter = ResultFormatter()
    if fmt == "json":
        click.echo(formatter.format_json(results))
    elif fmt == "compact":
        click.echo(formatter.format_compact(results))
    elif fmt == "ide":
        click.echo(formatter.format_ide(results))
    else:
        # Console: prepend a one-line hint so the relationship is clear
        if results and caller_or_callee:
            click.echo(f"  {caller_or_callee}")
            click.echo("")
        click.echo(formatter.format_console(results, elapsed_time=None))


# ---------------------------------------------------------------------------
# symdex watch
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--interval", type=int, default=300,
              help="Minimum seconds between re-indexes (default: 300).")
@click.option("--debounce", type=int, default=5,
              help="Seconds to wait after last file change before re-indexing (default: 5).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def watch(ctx: click.Context, directory: str, interval: int, debounce: int, verbose: bool):
    """Watch DIRECTORY and auto-reindex when Python files change.

    Uses watchdog for real-time file events if installed (pip install watchdog),
    otherwise falls back to polling every --interval seconds. Press Ctrl+C to stop.
    """
    import signal
    import threading
    import time
    cfg: SymdexConfig = ctx.obj["config"]
    _configure_logging(verbose, cfg)
    _validate_config(cfg)

    root_dir = Path(directory).resolve()
    from symdex import Symdex
    from symdex.core.autoreindex import start_auto_reindex

    client = Symdex(config=cfg)
    # Initial index so the index exists before watching
    try:
        click.echo(f"Initial index of {root_dir}...")
        result = client.index(root_dir, show_progress=True)
        click.echo(f"Indexed {result.files_processed} files, {result.functions_indexed} functions.")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    reindexer = start_auto_reindex(
        root_dir, client,
        interval_seconds=interval,
        debounce_seconds=debounce,
    )
    shutdown = threading.Event()

    def _on_sig(_signum, _frame):
        shutdown.set()

    try:
        signal.signal(signal.SIGINT, _on_sig)
    except (ValueError, OSError):
        # Signal only valid in main thread; or unsupported on this platform
        pass

    click.echo(f"Watching for changes (interval={interval}s, debounce={debounce}s). Press Ctrl+C to stop.")
    try:
        while not shutdown.is_set():
            shutdown.wait(timeout=1)
    except KeyboardInterrupt:
        shutdown.set()
    finally:
        reindexer.stop()
        click.echo("Stopped.")


# ---------------------------------------------------------------------------
# symdex mcp
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--transport", type=click.Choice(["stdio", "streamable-http", "sse"]),
              default="stdio",
              help="stdio=local Cursor (default), streamable-http=Smithery/remote, sse=legacy.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def mcp(ctx: click.Context, transport: str, verbose: bool):
    """Start the Symdex MCP server for Cursor / Claude integration.

    Tool defaults (max_results, context_lines, exclude_tests) are read from
    config; set SYMDEX_DEFAULT_MAX_RESULTS, SYMDEX_DEFAULT_CONTEXT_LINES,
    SYMDEX_DEFAULT_EXCLUDE_TESTS to change them. See docs/CONFIGURATION.md.
    """
    cfg: SymdexConfig = ctx.obj["config"]
    _configure_logging(verbose, cfg)
    try:
        from symdex.mcp.server import create_server  # noqa: E402
    except ImportError:
        click.echo(
            "Error: MCP dependencies not installed.\n"
            "Install with:  pip install 'symdex-100[mcp]'",
            err=True,
        )
        raise SystemExit(1)

    from symdex.mcp.server import create_server, run_with_server_card

    server = create_server(config=cfg)
    # PORT env is respected inside run_with_server_card (Fly.io/Railway use 8080)
    run_with_server_card(server, transport=transport)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_config(cfg: SymdexConfig) -> None:
    """Ensure the API key for the active LLM provider is available."""
    try:
        cfg.validate()
    except (ValueError, Exception) as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
