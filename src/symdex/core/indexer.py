"""
Symdex-100 Batch Indexer

Crawls directories, analyzes Python source code using AST,
generates Cypher metadata using LLM, and stores everything in a
sidecar SQLite index (``.symdex/index.db``).

**Source files are never modified.**  All metadata lives in the
``.symdex/`` directory so that the indexed codebase stays clean and
developers never see unwanted comment blocks in their diffs.

Production-ready with:
- Python AST-based function extraction (precise, robust)
- Incremental indexing (only processes changed files)
- Concurrent processing with rate limiting
- Comprehensive error handling and logging
- Progress tracking and statistics
- Dry-run mode for testing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from symdex.core.config import Config
from symdex.core.engine import (
    CodeAnalyzer, CypherCache, CypherGenerator,
    FunctionMetadata, scan_directory
)

logger = logging.getLogger(__name__)


# =============================================================================
# Indexing Pipeline
# =============================================================================

class IndexingPipeline:
    """
    Orchestrates the full indexing process.

    All generated metadata is stored in a sidecar SQLite database inside
    the ``.symdex/`` directory.  Source files are **never** modified.
    """

    def __init__(self, root_dir: Path, cache_dir: Path | None = None,
                 dry_run: bool = False, force_reindex: bool = False):
        """
        Initialize the indexing pipeline.

        Args:
            root_dir: Directory to index.
            cache_dir: Override directory for the sidecar index.
                       Defaults to ``root_dir / .symdex``.
            dry_run: If True, run analysis but don't persist to the index.
            force_reindex: If True, re-index even unchanged files.
        """
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.force_reindex = force_reindex

        # Resolve the sidecar directory
        self.cache_dir = cache_dir if cache_dir else (root_dir / Config.SYMDEX_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.cache = CypherCache(Config.get_cache_path(self.cache_dir))
        self.generator = CypherGenerator()
        self.analyzer = CodeAnalyzer()

        # Statistics
        self.stats = {
            "files_scanned": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "functions_found": 0,
            "functions_indexed": 0,
            "functions_skipped": 0,
            "errors": 0,
        }

    def run(self):
        """
        Execute the full indexing pipeline.

        Steps:
          1. Scan directories for source files
          2. Filter files (skip already-indexed unless --force)
          3. For each file: extract functions via AST / regex
          4. For each function: generate Cypher via LLM
          5. Store metadata in the sidecar SQLite index
        """
        logger.info("─" * 60)
        logger.info("  SYMDEX — Indexer")
        logger.info("─" * 60)
        logger.info(f"  Root : {self.root_dir}")
        logger.info(f"  Index: {self.cache_dir}")
        flags = []
        if self.dry_run:
            flags.append("dry-run")
        if self.force_reindex:
            flags.append("force")
        if flags:
            logger.info(f"  Flags: {', '.join(flags)}")
        logger.info("─" * 60)

        # ── Step 1: Scan for source files ────────────────────────
        logger.info("[1/4] Scanning for source files...")
        source_files = scan_directory(self.root_dir)
        self.stats["files_scanned"] = len(source_files)
        logger.info(f"  Found {len(source_files):,} source files")

        if not source_files:
            logger.warning("No source files found to index. Exiting.")
            return

        # ── Step 2: Filter files that need indexing ──────────────
        logger.info("[2/4] Checking index for changed files...")
        if not self.force_reindex:
            files_to_process = [
                f for f in source_files
                if not self.cache.is_file_indexed(f)
            ]
            self.stats["files_skipped"] = len(source_files) - len(files_to_process)
        else:
            files_to_process = source_files

        logger.info(
            f"  {len(files_to_process):,} to process, "
            f"{self.stats['files_skipped']:,} up-to-date"
        )

        if not files_to_process:
            logger.info("All files are up to date. Nothing to do.")
            self._print_statistics()
            return

        # ── Steps 3-4: Process files (parse → LLM → store) ──────
        logger.info(f"[3-4] Indexing functions ({Config.MAX_CONCURRENT_REQUESTS} workers)...")

        with tqdm(total=len(files_to_process), desc="Indexing files", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_REQUESTS) as executor:
                futures = {
                    executor.submit(self._process_file, file_path): file_path
                    for file_path in files_to_process
                }

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            self.stats["files_processed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        self.stats["errors"] += 1
                    finally:
                        pbar.update(1)

        # Print final statistics
        logger.info("=" * 60)
        self._print_statistics()

    def _process_file(self, file_path: Path) -> bool:
        """Process a single Python source file."""
        try:
            source_code = file_path.read_text(encoding='utf-8')

            # Extract functions
            functions = self.analyzer.extract_functions(source_code, str(file_path))
            self.stats["functions_found"] += len(functions)

            if not functions:
                logger.debug(f"No functions found in {file_path}")
                if not self.dry_run:
                    self.cache.mark_file_indexed(file_path, 0)
                return True

            # Clear existing index entries for this file
            if not self.dry_run:
                self.cache.clear_file_entries(file_path)

            indexed_count = 0
            for func_meta in functions:
                success = self._process_function(file_path, source_code, func_meta)
                if success:
                    indexed_count += 1
                    self.stats["functions_indexed"] += 1

            # Mark file as indexed
            if not self.dry_run:
                self.cache.mark_file_indexed(file_path, indexed_count)

            logger.info(f"  ✓ {file_path}  ({indexed_count}/{len(functions)} functions)")
            return True

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return False

    def _process_function(self, file_path: Path, source_code: str,
                          func_meta: FunctionMetadata) -> bool:
        """Process a single function: generate Cypher and store in index."""
        fn_label = f"{file_path.name}::{func_meta.name} (L{func_meta.start_line})"
        try:
            # Extract function source code
            func_source = self.analyzer.extract_function_source(
                source_code, func_meta.start_line, func_meta.end_line
            )

            # Generate Cypher using LLM (with retry + fallback)
            cypher = self.generator.generate_cypher(func_source, func_meta)

            # None means the LLM determined this is not a classifiable
            # function (e.g. code fragment, conditional, variable decl).
            if cypher is None:
                logger.debug(f"  [{fn_label}] Skipped (not a classifiable function)")
                self.stats["functions_skipped"] += 1
                return False

            logger.debug(f"  [{fn_label}] Cypher = {cypher}")

            # Build metadata components
            tags = self._generate_tags(func_meta)
            signature = self.analyzer.generate_signature(func_meta)
            complexity = self.analyzer.estimate_complexity_class(func_meta.complexity)

            # Persist to the sidecar SQLite index
            if not self.dry_run:
                self.cache.add_cypher_entry(
                    file_path, func_meta.name,
                    func_meta.start_line, func_meta.end_line,
                    cypher, tags, signature, complexity,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to process function {func_meta.name} in {file_path}: {e}")
            return False

    @staticmethod
    def _generate_tags(func_meta: FunctionMetadata) -> List[str]:
        """
        Generate semantic tags from function metadata.

        Sources (in priority order):
          1. Function name parts (split on _ / camelCase)
          2. Execution pattern (async, generator, etc.)
          3. Matched operations in function calls
          4. Docstring keyword extraction
        """
        tags: set = set()

        # ── 1. Function name parts ──────────────────────────────
        import re as _re
        name_parts = _re.findall(r'[a-z]+', func_meta.name.lower().replace('_', ' '))
        stop = {'get', 'set', 'the', 'and', 'for', 'def', 'self', 'cls', 'init', 'new'}
        for part in name_parts:
            if len(part) > 2 and part not in stop:
                tags.add(part)

        # ── 2. Execution pattern tags ───────────────────────────
        if func_meta.is_async:
            tags.add("async")

        # ── 3. Tags from function calls ─────────────────────────
        common_operations = {
            'read', 'write', 'open', 'close', 'send', 'receive',
            'get', 'set', 'update', 'delete', 'create', 'fetch',
            'validate', 'parse', 'serialize', 'deserialize',
            'encrypt', 'decrypt', 'hash', 'verify', 'log', 'logging',
            'connect', 'disconnect', 'listen', 'query', 'execute',
            'render', 'transform', 'filter', 'sort', 'merge',
            'upload', 'download', 'stream', 'emit', 'dispatch',
            'setup', 'configure', 'initialize', 'shutdown',
        }
        for call in func_meta.calls:
            call_lower = call.lower()
            if call_lower in common_operations:
                tags.add(call_lower)
            if '.' in call:
                leaf = call.rsplit('.', 1)[-1].lower()
                if leaf in common_operations:
                    tags.add(leaf)

        # ── 4. Docstring keyword extraction ─────────────────────
        if func_meta.docstring:
            doc_lower = func_meta.docstring.lower()
            keywords = [
                'security', 'performance', 'optimization', 'cache',
                'database', 'api', 'network', 'file', 'stream',
                'logging', 'config', 'configure', 'setup', 'initialize',
                'handler', 'middleware', 'authentication', 'authorization',
                'validation', 'serialization', 'encryption', 'compression',
                'batch', 'queue', 'worker', 'scheduler', 'retry',
                'error', 'exception', 'timeout', 'connection',
            ]
            for keyword in keywords:
                if keyword in doc_lower:
                    tags.add(keyword)

        # Return up to 12 unique tags (more signal = better search)
        return sorted(tags)[:12]

    def _print_statistics(self):
        """Print final indexing statistics in a human-readable format."""
        import shutil
        width = min(shutil.get_terminal_size().columns, 78)
        line = "─" * width

        cache_stats = self.cache.get_stats()
        s = self.stats

        print(f"\n{line}")
        print("  SYMDEX — Indexing Complete")
        print(line)
        print()
        print(f"  Root directory : {self.root_dir}")
        print(f"  Index location : {self.cache_dir}")
        print()
        print(f"  Files scanned      {s['files_scanned']:>7,}")
        print(f"  Files processed    {s['files_processed']:>7,}")
        print(f"  Files skipped      {s['files_skipped']:>7,}  (already up-to-date)")
        print()
        print(f"  Functions found    {s['functions_found']:>7,}")
        print(f"  Functions indexed  {s['functions_indexed']:>7,}")
        if s['functions_skipped']:
            print(f"  Functions skipped  {s['functions_skipped']:>7,}  (non-classifiable fragments)")
        if s['errors']:
            print(f"  Errors             {s['errors']:>7,}")
        print()
        print(f"  Total in index     {cache_stats['indexed_files']:>7,} files, "
              f"{cache_stats['indexed_functions']:,} functions")
        print(line)


# =============================================================================
# Legacy CLI Interface (prefer `symdex index` from cli/main.py)
# =============================================================================

def main():
    """Legacy entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Symdex-100 Batch Indexer — generate a sidecar search index for Python source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current directory (creates .symdex/ alongside your code)
  python -m symdex.core.indexer

  # Index specific directory
  python -m symdex.core.indexer --dir /path/to/project

  # Dry run (analyse but don't write the index)
  python -m symdex.core.indexer --dry-run

  # Force re-index all files
  python -m symdex.core.indexer --force
        """
    )

    parser.add_argument(
        'target',
        nargs='?',
        type=Path,
        help='Root directory to index (overrides --dir when provided)'
    )

    parser.add_argument(
        '--dir', '-d',
        type=Path,
        default=Path.cwd(),
        help='Root directory to index (default: current directory)'
    )

    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Override the sidecar index directory (default: <root>/.symdex)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyse code and generate Cyphers without writing the index'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-index all files, even if already indexed'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        Config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    root_dir: Path = args.target if args.target is not None else args.dir

    if not root_dir.exists():
        logger.error(f"Directory does not exist: {root_dir}")
        sys.exit(1)

    if not root_dir.is_dir():
        logger.error(f"Path is not a directory: {root_dir}")
        sys.exit(1)

    try:
        pipeline = IndexingPipeline(
            root_dir=root_dir,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
            force_reindex=args.force,
        )
        pipeline.run()
    except KeyboardInterrupt:
        logger.warning("\nIndexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
