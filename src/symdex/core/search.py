"""
Symdex-100 Search Engine

Natural language to Cypher query translation and high-speed code search.

Production-ready with:
- Multi-strategy search (exact, wildcard, semantic)
- Intelligent ranking and scoring
- Context extraction and preview
- Multiple output formats (console, JSON, IDE-friendly)
- Query caching and optimization
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import time

from symdex.core.config import Config, CypherSchema
from symdex.core.engine import (
    CypherCache, CypherGenerator, SearchResult,
    calculate_search_score
)

logger = logging.getLogger(__name__)


# =============================================================================
# Search Engine
# =============================================================================

class CypherSearchEngine:
    """
    Multi-lane search engine for Cypher-indexed code.

    Instead of relying on a single Cypher pattern (which can miss relevant
    results in adjacent domains), the engine always runs **five** parallel
    retrieval lanes and merges their output before a unified scoring pass:

    1. **Exact Cypher** — high-precision hit from the LLM-translated pattern
    2. **Domain-wildcarded Cypher** — same ACT_OBJ, any domain
    3. **Action-only Cypher** — same ACT, any OBJ/PAT/domain
    4. **Tag keywords** — query words matched against function tags
    5. **Function name** — query words matched inside function names

    Results are deduplicated, scored, and ranked so that exact matches
    still win, but cross-domain and name-based results are surfaced too.
    """

    # Words stripped from queries before keyword / name / tag matching
    # (imported from Config to avoid duplication)

    def __init__(self, cache_dir: Path):
        self.cache = CypherCache(Config.get_cache_path(cache_dir))
        self.generator = CypherGenerator()

    # ── Public API ────────────────────────────────────────────────

    def search(self, query: str, strategy: str = "auto",
               max_results: int = None) -> List[SearchResult]:
        """
        Execute a search query using the specified strategy.

        Args:
            query: Natural language search query or Cypher pattern.
            strategy: Search strategy ('auto', 'llm', 'keyword', 'direct').
            max_results: Maximum number of results to return.

        Returns:
            Ranked list of search results.
        """
        max_results = max_results or Config.MAX_SEARCH_RESULTS

        # Determine Cypher pattern from the query
        if self._is_cypher_pattern(query):
            cypher_pattern = query
            logger.info(f"Direct Cypher search: {cypher_pattern}")
        else:
            if strategy in ("llm", "auto"):
                cypher_pattern = self._translate_with_llm(query)
            else:
                cypher_pattern = self._translate_with_keywords(query)
            logger.info(f"Query: '{query}' → Cypher: '{cypher_pattern}'")

        # Always-on multi-lane retrieval
        raw_results = self._multi_lane_search(query, cypher_pattern, max_results)

        # Score, rank, limit
        search_results = self._process_results(raw_results, cypher_pattern, query)
        search_results.sort()
        return search_results[:max_results]

    def search_by_tag(self, tag: str, max_results: int = None) -> List[SearchResult]:
        """Search for functions by tag."""
        max_results = max_results or Config.MAX_SEARCH_RESULTS
        raw_results = self.cache.search_by_tags(tag, limit=max_results)
        return self._process_results(raw_results, None, tag)

    # ── Query analysis helpers ────────────────────────────────────

    def _is_cypher_pattern(self, query: str) -> bool:
        """Check if query is already in Cypher format."""
        pattern = r'^[A-Z*]{2,3}:[A-Z*]{3}_[A-Z0-9*]{2,20}--[A-Z*]{3}$'
        return bool(re.match(pattern, query.strip()))

    def _translate_with_llm(self, query: str) -> str:
        """Translate natural language query to Cypher using LLM."""
        try:
            return self.generator.translate_query(query)
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}. Falling back to keywords.")
            return self._translate_with_keywords(query)

    def _translate_with_keywords(self, query: str) -> str:
        """Translate query using keyword matching (fast fallback)."""
        query_lower = query.lower()

        dom = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_DOMAIN.items():
            if keyword in query_lower:
                dom = code
                break

        act = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_ACTION.items():
            if keyword in query_lower:
                act = code
                break

        pat = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_PATTERN.items():
            if keyword in query_lower:
                pat = code
                break

        obj = "*"
        obj_matches = re.findall(r'\b[A-Z][a-z]+\b|"([^"]+)"', query)
        if obj_matches:
            obj_word = (
                obj_matches[0]
                if isinstance(obj_matches[0], str)
                else obj_matches[0][0]
            )
            obj = obj_word[:4].upper().ljust(4, '*')

        return f"{dom}:{act}_{obj}--{pat}"

    @classmethod
    def _extract_keywords(cls, query: str) -> List[str]:
        """
        Extract meaningful keywords from a natural-language query.

        Filters stop words and very short tokens so that only
        semantically significant words remain for tag / name matching.
        """
        return [
            w for w in query.lower().split()
            if w not in Config.STOP_WORDS and len(w) > 2
        ]

    # ── Multi-lane retrieval ──────────────────────────────────────

    def _multi_lane_search(self, query: str, cypher_pattern: str,
                           max_results: int) -> List[Dict[str, Any]]:
        """
        Always-on multi-lane search.

        Runs several parallel retrieval strategies and merges unique
        results for unified ranking.

        Lanes:
          1. Exact Cypher pattern     — highest precision
          2. Domain-wildcarded Cypher  — cross-domain recall
          3. Action-only Cypher        — broadest Cypher recall
          4. Tag keyword search        — semantic tags
          5. Function name search      — literal name matching
        """
        fetch_limit = max(max_results * 5, 30)

        seen: set = set()
        merged: List[Dict[str, Any]] = []

        def _add(results: List[Dict[str, Any]]) -> None:
            """Append de-duplicated results to *merged*."""
            for r in results:
                key = (r["file_path"], r["function_name"],
                       r.get("line_start", 0))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)

        # ── Lane 1: Exact Cypher pattern ─────────────────────────
        _add(self.cache.search_by_cypher(cypher_pattern, limit=fetch_limit))

        # ── Lane 2: Domain-wildcarded (same ACT_OBJ--PAT) ───────
        parts = cypher_pattern.split(":")
        if len(parts) == 2:
            dom, rest_str = parts
            if dom != "*":
                _add(self.cache.search_by_cypher(
                    f"*:{rest_str}", limit=fetch_limit
                ))
        else:
            rest_str = cypher_pattern

        # ── Lane 3: Action-only (broadest Cypher) ────────────────
        rest = rest_str.split("--")
        if len(rest) == 2:
            act_obj = rest[0].split("_")
            if len(act_obj) == 2:
                act = act_obj[0]
                if act != "*":
                    _add(self.cache.search_by_cypher(
                        f"*:{act}_*--*", limit=fetch_limit
                    ))

        # ── Lane 4: Tag keyword search ───────────────────────────
        keywords = self._extract_keywords(query)
        for kw in keywords[:5]:
            _add(self.cache.search_by_tags(kw, limit=fetch_limit))

        # ── Lane 5: Function name search ─────────────────────────
        if keywords:
            _add(self.cache.search_by_name(keywords, limit=fetch_limit))

        logger.debug(
            f"Multi-lane: {len(merged)} unique candidates "
            f"(keywords: {keywords[:5]})"
        )
        return merged
    
    def _process_results(self, raw_results: List[Dict[str, Any]], 
                        cypher_pattern: Optional[str], query: str) -> List[SearchResult]:
        """Convert raw database results to SearchResult objects with scoring."""
        search_results = []
        
        # PERFORMANCE FIX: Cache file contents to avoid reading the same file multiple times
        # (common when multiple functions from the same file match)
        file_cache: Dict[str, List[str]] = {}
        
        for result in raw_results:
            # Extract tags
            tags = result.get('tags', '').split(',') if result.get('tags') else []
            func_name = result.get('function_name', '')
            
            # Calculate score
            if cypher_pattern:
                score = calculate_search_score(
                    cypher_pattern, result['cypher'], tags, query,
                    function_name=func_name
                )
            else:
                score = 1.0  # Base score for tag searches
            
            # Extract context (with file caching)
            context = self._extract_context_cached(
                result['file_path'],
                result['line_start'],
                result['line_end'],
                file_cache
            )
            
            search_results.append(SearchResult(
                file_path=result['file_path'],
                function_name=func_name,
                line_start=result['line_start'],
                line_end=result['line_end'],
                cypher=result['cypher'],
                score=score,
                context=context
            ))
        
        return search_results
    
    def _extract_context_cached(self, file_path: str, start_line: int, 
                                end_line: int, file_cache: Dict[str, List[str]], 
                                context_lines: int = 3) -> str:
        """
        Extract code context around a function (with file caching).
        
        PERFORMANCE: Caches file contents per search query to avoid reading
        the same file multiple times when multiple results come from it.
        """
        try:
            # Check cache first
            if file_path not in file_cache:
                path = Path(file_path)
                if not path.exists():
                    return "[File not found]"
                file_cache[file_path] = path.read_text(encoding='utf-8').splitlines()
            
            lines = file_cache[file_path]
            
            # Get function definition (first few lines)
            func_lines = lines[start_line - 1:min(start_line + context_lines, end_line)]
            return "\n".join(func_lines)
        except Exception as e:
            logger.warning(f"Could not extract context from {file_path}: {e}")
            return "[Context unavailable]"
    
    def get_statistics(self) -> Dict[str, int]:
        """Get search index statistics."""
        return self.cache.get_stats()


# =============================================================================
# Result Formatting
# =============================================================================

class ResultFormatter:
    """Format search results for different output modes."""

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _detect_language(file_path: str) -> str:
        """Return language label (Python-only for v1)."""
        return "Python" if file_path.endswith(".py") else ""

    @staticmethod
    def _numbered_preview(context: str, start_line: int) -> List[str]:
        """Return context lines prefixed with line numbers."""
        lines: List[str] = []
        for i, line in enumerate(context.splitlines()):
            lineno = start_line + i
            lines.append(f"    {lineno:>6} │ {line}")
        return lines

    # ── Console (human-friendly) ──────────────────────────────────

    @staticmethod
    def format_console(results: List[SearchResult], show_context: bool = True,
                       start_index: int = 1, total_count: int | None = None,
                       elapsed_time: float | None = None) -> str:
        """
        Rich console output with full paths, line numbers, language
        labels, and a numbered code preview.

        Args:
            results: The (page of) results to render.
            show_context: Include a numbered code preview.
            start_index: Global 1-based index for the first result
                         (used for paginated output so numbering is
                         continuous across pages).
            total_count: Total result count across all pages.  When
                         *None* the header shows ``len(results)``.
            elapsed_time: Optional search time in seconds to display in header.
        """
        if not results:
            return "\n  No results found.\n"

        import shutil
        width = min(shutil.get_terminal_size().columns, 78)
        thin = "─" * width

        display_total = total_count if total_count is not None else len(results)

        # Build header with optional timing
        header = f"  SYMDEX — {display_total} result{'s' if display_total != 1 else ''}"
        if elapsed_time is not None:
            # Format timing with explicit control (avoid locale comma/period confusion)
            timing_str = f"{elapsed_time:.4f}".replace(',', '.')
            header += f" in {timing_str} seconds"

        out: List[str] = []
        out.append(f"\n{thin}")
        out.append(header)
        out.append(thin)

        for offset, r in enumerate(results):
            idx = start_index + offset
            lang = ResultFormatter._detect_language(r.file_path)
            lang_label = f"  ({lang})" if lang else ""

            out.append("")
            out.append(f"  #{idx}  {r.function_name}{lang_label}")
            out.append(f"  {'─' * (width - 2)}")
            out.append(f"    File   : {r.file_path}")
            out.append(f"    Lines  : {r.line_start}–{r.line_end}")
            out.append(f"    Cypher : {r.cypher}")
            out.append(f"    Score  : {r.score:.1f}")

            if show_context and r.context:
                out.append("")
                out.extend(
                    ResultFormatter._numbered_preview(r.context, r.line_start)
                )

        out.append(f"\n{thin}")
        return "\n".join(out)

    # ── JSON ──────────────────────────────────────────────────────

    @staticmethod
    def format_json(results: List[SearchResult]) -> str:
        """Format results as JSON (full paths, all fields)."""
        json_results = [
            {
                "function_name": r.function_name,
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "cypher": r.cypher,
                "score": round(r.score, 2),
                "language": ResultFormatter._detect_language(r.file_path),
                "context": r.context,
            }
            for r in results
        ]
        return json.dumps(json_results, indent=2)

    # ── Compact (grep-like, one line per result) ──────────────────

    @staticmethod
    def format_compact(results: List[SearchResult]) -> str:
        """
        One-line-per-result format.  Full path with ``file:line``
        syntax so terminals can make it clickable.
        """
        if not results:
            return "No results found."

        lines: List[str] = []
        for r in results:
            lang = ResultFormatter._detect_language(r.file_path)
            lang_tag = f" ({lang})" if lang else ""
            lines.append(
                f"{r.file_path}:{r.line_start}  "
                f"{r.function_name}  [{r.cypher}]{lang_tag}"
            )
        return "\n".join(lines)

    # ── IDE (clickable file(line) format) ─────────────────────────

    @staticmethod
    def format_ide(results: List[SearchResult]) -> str:
        """
        ``file(line): message`` format recognised by most editors and
        CI tools for click-to-jump navigation.
        """
        if not results:
            return "No results found."

        lines: List[str] = []
        for r in results:
            lines.append(
                f"{r.file_path}({r.line_start}): "
                f"{r.function_name} [{r.cypher}]"
            )
        return "\n".join(lines)


# =============================================================================
# Interactive Search Mode
# =============================================================================

class InteractiveSearchSession:
    """Interactive search session with query refinement."""
    
    def __init__(self, search_engine: CypherSearchEngine):
        self.engine = search_engine
        self.history = []
    
    def run(self):
        """Run an interactive search session."""
        print("=" * 80)
        print("SYMDEX-100 INTERACTIVE SEARCH")
        print("=" * 80)
        print("Enter your search queries in natural language.")
        print("Special commands:")
        print("  /stats    - Show index statistics")
        print("  /history  - Show search history")
        print("  /help     - Show help")
        print("  /exit     - Exit interactive mode")
        print("=" * 80 + "\n")
        
        formatter = ResultFormatter()
        
        while True:
            try:
                query = input("Search> ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query == "/exit":
                    print("Goodbye!")
                    break
                elif query == "/help":
                    self._show_help()
                    continue
                elif query == "/stats":
                    self._show_stats()
                    continue
                elif query == "/history":
                    self._show_history()
                    continue
                
                # Execute search
                print(f"\nSearching for: {query}")
                results = self.engine.search(query)
                
                # Display results
                print(formatter.format_console(results, show_context=True))
                
                # Save to history
                self.history.append({
                    "query": query,
                    "result_count": len(results)
                })
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /exit to quit or continue searching.")
            except Exception as e:
                logger.error(f"Search error: {e}")
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\nCYPHER-100 SEARCH HELP")
        print("-" * 80)
        print("Search Syntax:")
        print("  - Natural language: 'find async email functions'")
        print("  - Direct Cypher: 'NET:SND_EMAL--ASY'")
        print("  - Tag search: 'tag:async' or '#security'")
        print("\nSearch Tips:")
        print("  - Be specific about the action (send, validate, transform, etc.)")
        print("  - Mention the domain if known (security, data, network, etc.)")
        print("  - Include patterns for better results (async, recursive, etc.)")
        print("-" * 80 + "\n")
    
    def _show_stats(self):
        """Show index statistics."""
        stats = self.engine.get_statistics()
        print("\nINDEX STATISTICS")
        print("-" * 80)
        print(f"Indexed files:     {stats['indexed_files']}")
        print(f"Indexed functions: {stats['indexed_functions']}")
        print("-" * 80 + "\n")
    
    def _show_history(self):
        """Show search history."""
        if not self.history:
            print("\nNo search history yet.\n")
            return
        
        print("\nSEARCH HISTORY")
        print("-" * 80)
        for idx, entry in enumerate(self.history[-10:], 1):  # Last 10 queries
            print(f"{idx}. '{entry['query']}' → {entry['result_count']} results")
        print("-" * 80 + "\n")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for the search tool."""
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Symdex-100 Search Engine - Natural language code search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Natural language search
  python cypher_search.py "find async email functions"

  # Direct Cypher search
  python cypher_search.py "NET:SND_EMAL--ASY"

  # Interactive mode
  python cypher_search.py --interactive

  # JSON output
  python cypher_search.py "security validation" --format json

  # Compact output (grep-like)
  python cypher_search.py "data processing" --format compact
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query in natural language or Cypher format'
    )
    
    default_cache_dir = Path(os.getenv("CYPHER_CACHE_DIR", Path.cwd()))
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=default_cache_dir,
        help='Directory containing the cache database (default: CYPHER_CACHE_DIR or current directory)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive search mode'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['console', 'json', 'compact', 'ide'],
        default='console',
        help='Output format (default: console)'
    )
    
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Do not show code context in results'
    )
    
    parser.add_argument(
        '--max-results', '-n',
        type=int,
        default=Config.MAX_SEARCH_RESULTS,
        help=f'Maximum number of results (default: {Config.MAX_SEARCH_RESULTS})'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['auto', 'llm', 'keyword', 'direct'],
        default='auto',
        help='Search strategy (default: auto)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show index statistics and exit'
    )
    
    parser.add_argument(
        '--min-score',
        type=float,
        default=Config.MIN_SEARCH_SCORE,
        help=f'Minimum relevance score to display a result (default: {Config.MIN_SEARCH_SCORE})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Check if cache exists
    cache_path = Config.get_cache_path(args.cache_dir)
    if not cache_path.exists():
        logger.error(f"Cache database not found: {cache_path}")
        logger.error("Run cypher_indexer.py first to build the index.")
        sys.exit(1)
    
    # Initialize search engine
    try:
        search_engine = CypherSearchEngine(args.cache_dir)
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        sys.exit(1)
    
    # Show statistics
    if args.stats:
        stats = search_engine.get_statistics()
        print("\nCYPHER INDEX STATISTICS")
        print("=" * 60)
        print(f"Indexed files:     {stats['indexed_files']}")
        print(f"Indexed functions: {stats['indexed_functions']}")
        print("=" * 60)
        return
    
    # Interactive mode
    if args.interactive:
        session = InteractiveSearchSession(search_engine)
        session.run()
        return
    
    # Single query mode
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Execute search
        results = search_engine.search(
            args.query,
            strategy=args.strategy,
            max_results=args.max_results
        )
        
        # Filter by minimum relevance score
        min_score = args.min_score
        total_before_filter = len(results)
        results = [r for r in results if r.score >= min_score]
        
        if not results and total_before_filter > 0:
            logger.info(
                f"Found {total_before_filter} results but none above "
                f"min score threshold ({min_score:.1f}). "
                f"Use --min-score 0 to see all."
            )
        
        # Format and display results
        formatter = ResultFormatter()
        
        if args.format == 'json':
            output = formatter.format_json(results)
        elif args.format == 'compact':
            output = formatter.format_compact(results)
        elif args.format == 'ide':
            output = formatter.format_ide(results)
        else:
            output = formatter.format_console(results, show_context=not args.no_context)
        
        print(output)

        elapsed = time.perf_counter() - start_time
        print(f"\n[INFO] Query completed in {elapsed:.3f}s")
        
        # Exit code based on results
        sys.exit(0 if results else 1)
    
    except KeyboardInterrupt:
        logger.warning("\nSearch interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
