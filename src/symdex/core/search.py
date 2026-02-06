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
    """High-performance search engine for Cypher-indexed code."""
    
    def __init__(self, cache_dir: Path):
        self.cache = CypherCache(Config.get_cache_path(cache_dir))
        self.generator = CypherGenerator()
    
    def search(self, query: str, strategy: str = "auto", 
               max_results: int = None) -> List[SearchResult]:
        """
        Execute a search query using the specified strategy.
        
        Args:
            query: Natural language search query or Cypher pattern
            strategy: Search strategy ('auto', 'llm', 'keyword', 'direct')
            max_results: Maximum number of results to return
        
        Returns:
            Ranked list of search results
        """
        max_results = max_results or Config.MAX_SEARCH_RESULTS
        
        # Determine if query is already a Cypher pattern
        if self._is_cypher_pattern(query):
            cypher_pattern = query
            logger.info(f"Direct Cypher search: {cypher_pattern}")
        else:
            # Translate natural language to Cypher
            if strategy == "llm" or strategy == "auto":
                cypher_pattern = self._translate_with_llm(query)
            else:
                cypher_pattern = self._translate_with_keywords(query)
            
            logger.info(f"Query: '{query}' → Cypher: '{cypher_pattern}'")
        
        # Execute search
        raw_results = self.cache.search_by_cypher(cypher_pattern, limit=max_results * 2)
        
        # If no results, try progressively broader searches
        if not raw_results:
            raw_results = self._fallback_search(query, cypher_pattern)
        
        # Convert to SearchResult objects and rank
        search_results = self._process_results(raw_results, cypher_pattern, query)
        
        # Sort by score and limit
        search_results.sort()
        return search_results[:max_results]
    
    def search_by_tag(self, tag: str, max_results: int = None) -> List[SearchResult]:
        """Search for functions by tag."""
        max_results = max_results or Config.MAX_SEARCH_RESULTS
        raw_results = self.cache.search_by_tags(tag, limit=max_results)
        return self._process_results(raw_results, None, tag)
    
    def _is_cypher_pattern(self, query: str) -> bool:
        """Check if query is already in Cypher format."""
        # 2-3 char DOM, 3 char ACT, 2-20 char OBJ (letters/digits), 3 char PAT
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
        
        # Find domain
        dom = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_DOMAIN.items():
            if keyword in query_lower:
                dom = code
                break
        
        # Find action
        act = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_ACTION.items():
            if keyword in query_lower:
                act = code
                break
        
        # Find pattern
        pat = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_PATTERN.items():
            if keyword in query_lower:
                pat = code
                break
        
        # Extract potential object
        obj = "*"
        # Look for capitalized words or quoted strings
        obj_matches = re.findall(r'\b[A-Z][a-z]+\b|"([^"]+)"', query)
        if obj_matches:
            obj_word = obj_matches[0] if isinstance(obj_matches[0], str) else obj_matches[0][0]
            obj = obj_word[:4].upper().ljust(4, '*')
        
        return f"{dom}:{act}_{obj}--{pat}"
    
    def _fallback_search(self, query: str, original_pattern: str) -> List[Dict[str, Any]]:
        """
        Perform progressively broader searches if initial search fails.
        Also runs a parallel tag-based search and merges unique results.
        """
        logger.info("No results found. Attempting fallback searches...")
        
        # ── Parallel tag search ──────────────────────────────────
        # Extract meaningful words from query to search by tags
        stop_words = {
            "i", "a", "the", "is", "it", "do", "we", "my", "me", "an", "in",
            "to", "for", "of", "and", "or", "on", "at", "by", "with", "from",
            "that", "this", "where", "what", "how", "which", "show", "find",
            "search", "look", "give", "list", "get", "see", "main", "function",
        }
        query_words = [
            w for w in query.lower().split()
            if w not in stop_words and len(w) > 1
        ]
        tag_results: List[Dict[str, Any]] = []
        for word in query_words:
            tag_results.extend(self.cache.search_by_tags(word, limit=20))
        
        # ── Cypher fallback cascade ──────────────────────────────
        parts = original_pattern.split(':')
        cypher_results: List[Dict[str, Any]] = []
        
        if len(parts) == 2:
            dom = parts[0]
            rest = parts[1].split('--')
            if len(rest) == 2:
                act_obj = rest[0].split('_')
                pat = rest[1]
                if len(act_obj) == 2:
                    act, obj = act_obj
                    
                    fallback_patterns = [
                        f"{dom}:{act}_*--{pat}",      # Wildcard object
                        f"{dom}:*_{obj}--{pat}",      # Wildcard action
                        f"{dom}:{act}_*--*",          # Wildcard object and pattern
                        f"{dom}:*_*--{pat}",          # Keep only domain and pattern
                        f"{dom}:*_*--*",              # Keep only domain
                        f"*:{act}_*--*",              # Keep only action
                    ]
                    
                    for pattern in fallback_patterns:
                        if pattern == original_pattern:
                            continue
                        logger.debug(f"Trying fallback: {pattern}")
                        results = self.cache.search_by_cypher(
                            pattern, limit=Config.MAX_SEARCH_RESULTS
                        )
                        if results:
                            logger.info(f"Found {len(results)} results with pattern: {pattern}")
                            cypher_results = results
                            break
        
        # ── Merge & deduplicate ──────────────────────────────────
        seen_keys = set()
        merged: List[Dict[str, Any]] = []
        for r in cypher_results + tag_results:
            key = (r['file_path'], r['function_name'], r['line_start'])
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(r)
        
        if merged:
            logger.info(
                f"Fallback total: {len(merged)} unique results "
                f"({len(cypher_results)} from Cypher, {len(tag_results)} from tags)"
            )
        
        return merged
    
    def _process_results(self, raw_results: List[Dict[str, Any]], 
                        cypher_pattern: Optional[str], query: str) -> List[SearchResult]:
        """Convert raw database results to SearchResult objects with scoring."""
        search_results = []
        
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
            
            # Extract context
            context = self._extract_context(
                Path(result['file_path']),
                result['line_start'],
                result['line_end']
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
    
    def _extract_context(self, file_path: Path, start_line: int, 
                        end_line: int, context_lines: int = 3) -> str:
        """Extract code context around a function."""
        try:
            if not file_path.exists():
                return "[File not found]"
            
            lines = file_path.read_text(encoding='utf-8').splitlines()
            
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
        """Infer a short language label from the file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
            ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
            ".java": "Java", ".go": "Go", ".rs": "Rust",
            ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++",
            ".cc": "C++", ".cxx": "C++", ".cs": "C#",
            ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
            ".kt": "Kotlin", ".kts": "Kotlin",
        }
        return lang_map.get(ext, "")

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
    def format_console(results: List[SearchResult], show_context: bool = True) -> str:
        """
        Rich console output with full paths, line numbers, language
        labels, and a numbered code preview.
        """
        if not results:
            return "\n  No results found.\n"

        import shutil
        width = min(shutil.get_terminal_size().columns, 78)
        thin = "─" * width

        out: List[str] = []
        out.append(f"\n{thin}")
        out.append(f"  SYMDEX — {len(results)} result{'s' if len(results) != 1 else ''}")
        out.append(thin)

        for idx, r in enumerate(results, 1):
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
