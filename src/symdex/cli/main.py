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
import time
from pathlib import Path

import click

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
    # Suppress noisy HTTP loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


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
    """Symdex-100 — Semantic fingerprints for 100x faster code search."""
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
@click.option("--force", is_flag=True, help="Force re-index all files, even if unchanged.")
@click.option("--dry-run", is_flag=True, help="Analyse code without writing the index.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def index(ctx: click.Context, directory: str, cache_dir: str | None,
          force: bool, dry_run: bool, verbose: bool):
    """Index Python source files in DIRECTORY and build a sidecar search index.

    Metadata is stored in a .symdex/ directory — source files are never
    modified. Python-only for v1 (multi-language support planned).
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
              default="console", help="Output format.")
@click.option("-n", "--max-results", type=int, default=None,
              help="Maximum number of results.")
@click.option("-p", "--page-size", type=int, default=None,
              help="Results per page (enables interactive pagination).")
@click.option("--strategy",
              type=click.Choice(["auto", "llm", "keyword", "direct"]),
              default="auto", help="Query translation strategy.")
@click.option("--min-score", type=float, default=None,
              help="Minimum relevance score to display.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def search(ctx: click.Context, query: str, cache_dir: str | None, fmt: str,
           max_results: int | None, page_size: int | None,
           strategy: str, min_score: float | None, verbose: bool):
    """Search the Symdex index with a natural-language QUERY or Cypher pattern."""
    cfg: SymdexConfig = ctx.obj["config"]
    _configure_logging(verbose, cfg)
    _validate_config(cfg)
    t0 = time.perf_counter()

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
    results = engine.search(query, strategy=strategy, max_results=max_results)

    # Apply minimum score filter
    score_threshold = min_score if min_score is not None else cfg.min_search_score
    results = [r for r in results if r.score >= score_threshold]

    # Calculate elapsed time for display
    elapsed = time.perf_counter() - t0

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
    """Show Symdex index statistics."""
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
    click.echo("─" * 50)


# ---------------------------------------------------------------------------
# symdex mcp
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--transport", type=click.Choice(["stdio", "sse"]),
              default="stdio", help="MCP transport (default: stdio).")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def mcp(ctx: click.Context, transport: str, verbose: bool):
    """Start the Symdex MCP server for Cursor / Claude integration."""
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

    server = create_server(config=cfg)
    server.run(transport=transport)


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
