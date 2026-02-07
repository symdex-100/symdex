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

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from symdex.core.config import Config, CypherSchema, SymdexConfig
from symdex.core.engine import (
    CypherCache, CypherGenerator, SearchResult,
    calculate_search_score,
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

    def __init__(
        self,
        cache_dir: Path,
        config: SymdexConfig | None = None,
        generator: CypherGenerator | None = None,
    ):
        self._config = config or SymdexConfig.from_env()
        self._cache_dir = cache_dir
        self.cache = CypherCache(self._config.get_cache_path(cache_dir))
        # Lazy generator — only created when an LLM-based search is needed
        self._generator = generator
        # Time (seconds) of last search from DB query start to result list (excludes LLM translation)
        self._last_db_elapsed_seconds: float = 0.0

    @property
    def generator(self) -> CypherGenerator:
        """Lazy LLM generator — created on first use that requires it."""
        if self._generator is None:
            self._generator = CypherGenerator(config=self._config)
        return self._generator

    @generator.setter
    def generator(self, value: CypherGenerator):
        """Allow direct injection (used by tests and the facade)."""
        self._generator = value

    @property
    def last_search_db_elapsed_seconds(self) -> float:
        """
        Elapsed time (seconds) of the last search from DB query start to
        result list. Excludes LLM/keyword translation; use for reported
        "search time" so users see index lookup performance.
        """
        return self._last_db_elapsed_seconds

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
        max_results = max_results or self._config.max_search_results

        # Resolve to a list of Cypher patterns (1 = single, 3 = tiered tight/medium/broad)
        if self._is_cypher_pattern(query):
            patterns = [query]
            logger.info(f"Direct Cypher search: {query}")
        else:
            if strategy in ("llm", "auto"):
                patterns = self._translate_with_llm(query)
            else:
                patterns = [self._translate_with_keywords(query)]
            if len(patterns) > 1:
                logger.info(f"Query: '{query}' → Cypher (tiered): {patterns}")
            else:
                logger.info(f"Query: '{query}' → Cypher: '{patterns[0]}'")

        primary_pattern = patterns[0]

        # Time only DB + scoring + context (exclude LLM/keyword translation)
        t0 = time.perf_counter()
        if len(patterns) > 1:
            raw_results = self._tiered_multi_lane_search(query, patterns, max_results)
        else:
            # When query is a Cypher pattern (e.g. from search_by_cypher), only use Cypher lanes
            cypher_only = query.strip() == primary_pattern.strip()
            raw_results = self._multi_lane_search(
                query, primary_pattern, max_results, cypher_only=cypher_only
            )

        # Score against the tight (primary) pattern so precise matches rank highest
        search_results = self._process_results(raw_results, primary_pattern, query)
        search_results.sort()
        self._last_db_elapsed_seconds = time.perf_counter() - t0
        return search_results[:max_results]

    def search_by_tag(self, tag: str, max_results: int = None) -> List[SearchResult]:
        """Search for functions by tag."""
        max_results = max_results or self._config.max_search_results
        t0 = time.perf_counter()
        raw_results = self.cache.search_by_tags(tag, limit=max_results)
        search_results = self._process_results(raw_results, None, tag)
        self._last_db_elapsed_seconds = time.perf_counter() - t0
        return search_results

    # ── Query analysis helpers ────────────────────────────────────

    def _is_cypher_pattern(self, query: str) -> bool:
        """Check if query is already in Cypher format (allows * wildcards in any slot)."""
        # DOM:ACT_OBJ--PAT; allow 1–3 chars per slot so "*" is valid
        pattern = r'^[A-Z*]{1,3}:[A-Z*]{1,3}_[A-Z0-9*]{1,20}--[A-Z*]{1,3}$'
        return bool(re.match(pattern, query.strip()))

    def _translate_with_llm(self, query: str) -> List[str]:
        """Translate natural language query to 1–3 Cypher patterns (tiered) using LLM."""
        try:
            return self.generator.translate_query(query)
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}. Falling back to keywords.")
            return [self._translate_with_keywords(query)]

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

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from a natural-language query.

        Filters stop words and very short tokens so that only
        semantically significant words remain for tag / name matching.
        """
        return [
            w for w in query.lower().split()
            if w not in self._config.stop_words and len(w) > 2
        ]

    # ── Multi-lane retrieval ──────────────────────────────────────

    def _tiered_multi_lane_search(
        self, query: str, patterns: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Run multi-lane search for each pattern in order (tight → medium → broad),
        merging and deduplicating. Stops once we have enough candidates
        (max_results * 2) so we don't over-fetch from broad patterns.
        """
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        threshold = max_results * 2

        for cypher_pattern in patterns:
            raw = self._multi_lane_search(query, cypher_pattern, max_results)
            for r in raw:
                key = (r["file_path"], r["function_name"], r.get("line_start", 0))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            if len(merged) >= threshold:
                break

        return merged

    def _multi_lane_search(self, query: str, cypher_pattern: str,
                           max_results: int,
                           cypher_only: bool = False) -> List[Dict[str, Any]]:
        """
        Multi-lane search; merges unique results for unified ranking.

        Lanes:
          1. Exact Cypher pattern       — highest precision
          2. Domain-wildcarded Cypher   — cross-domain recall
          3. Action-only Cypher         — broadest Cypher recall
          4. Tag keyword search         — semantic tags (skipped if cypher_only)
          5. Function name search       — literal name matching (skipped if cypher_only)

        When cypher_only is True (e.g. from search_by_cypher), only lanes 1–3
        run so results are strictly Cypher-pattern matches.
        """
        fetch_limit = max(max_results * 5, 30)
        fetch_limit_tag = min(fetch_limit, 50)

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
            # Skip when rest is fully wildcard (*_*--*) or we'd match the whole index
            if dom != "*" and rest_str.strip() != "*_*--*":
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
                act, obj = act_obj[0], act_obj[1]
                pat = rest[1]
                if act != "*" and (obj != "*" or pat != "*"):
                    _add(self.cache.search_by_cypher(
                        f"*:{act}_*--*", limit=fetch_limit
                    ))

        if cypher_only:
            keywords = []
            logger.debug("Direct Cypher search: tag/name lanes skipped")
        else:
            # ── Lane 4: Tag keyword search ───────────────────────────
            keywords = self._extract_keywords(query)
            for kw in keywords[:5]:
                _add(self.cache.search_by_tags(kw, limit=fetch_limit_tag))

            # ── Lane 5: Function name search ─────────────────────────
            if keywords:
                _add(self.cache.search_by_name(keywords, limit=fetch_limit_tag))

        # Cap total candidates to avoid scoring huge sets
        cap = self._config.max_search_candidates or (max_results * 10)
        if cap > 0 and len(merged) > cap:
            merged = merged[:cap]

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
                    function_name=func_name,
                    config=self._config,
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
            
            path_root = str(self._cache_dir.parent) if self._cache_dir else ""
            search_results.append(SearchResult(
                file_path=result['file_path'],
                function_name=func_name,
                line_start=result['line_start'],
                line_end=result['line_end'],
                cypher=result['cypher'],
                score=score,
                context=context,
                path_root=path_root,
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
        """Format results as JSON (full paths, all fields). Includes path_root when set."""
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
                **({"path_root": r.path_root} if r.path_root else {}),
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



# NOTE: InteractiveSearchSession and legacy CLI main() removed in v1.1.
# Use the ``symdex`` CLI (symdex.cli.main) or the Symdex facade
# (symdex.client.Symdex) instead.
