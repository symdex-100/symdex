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
from typing import List, Dict, Any, Optional, Union

import re

from symdex.core.config import Config, CypherSchema, SymdexConfig
from symdex.core.engine import (
    CypherCache, CypherGenerator, SearchResult,
    calculate_search_score,
)

logger = logging.getLogger(__name__)


def _cypher_domain(cypher: str) -> str:
    """Extract domain (DOM) from Cypher string, e.g. 'BIZ:SYN_TASK--SYN' -> 'BIZ'."""
    if ":" in cypher:
        return cypher.split(":")[0].strip()
    return ""


def _cypher_action(cypher: str) -> str:
    """Extract action (ACT) from Cypher string, e.g. 'BIZ:SYN_TASK--SYN' -> 'SYN'."""
    if ":" in cypher:
        rest = cypher.split(":", 1)[1]
        if "--" in rest:
            act_obj = rest.split("--", 1)[0]
            if "_" in act_obj:
                return act_obj.split("_", 1)[0].strip()
    return ""


def _filter_by_domain_action(
    results: List[SearchResult],
    domain_filter: Optional[List[str]] = None,
    action_filter: Optional[List[str]] = None,
) -> List[SearchResult]:
    """Keep only results whose Cypher domain/action is in the given sets (if provided)."""
    if not domain_filter and not action_filter:
        return results
    domains = set(d.upper() for d in (domain_filter or []))
    actions = set(a.upper() for a in (action_filter or []))
    out: List[SearchResult] = []
    for r in results:
        if domains and _cypher_domain(r.cypher) not in domains:
            continue
        if actions and _cypher_action(r.cypher) not in actions:
            continue
        out.append(r)
    return out


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
               max_results: int = None, context_lines: int = 3,
               explain: bool = False, exclude_tests: bool = True,
               directory_scope: Optional[str] = None,
               domain_filter: Optional[List[str]] = None,
               action_filter: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Execute a search query using the specified strategy.

        Args:
            query: Natural language search query or Cypher pattern.
            strategy: Search strategy ('auto', 'llm', 'keyword', 'direct').
            max_results: Maximum number of results to return.
            context_lines: Lines of code context per result (default 3; increase for editing).
            explain: Include scoring breakdown in results (for debugging).
            exclude_tests: If True, filter out test functions (default True for normal use).
            directory_scope: If set, restrict results to functions under this path (relative to index root).
            domain_filter: If set, keep only results whose Cypher domain is in this list (e.g. ['BIZ', 'NET']).
            action_filter: If set, keep only results whose Cypher action is in this list (e.g. ['FET', 'SND']).

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
            raw_results = self._tiered_multi_lane_search(query, patterns, max_results, directory_scope=directory_scope)
        else:
            # When query is a Cypher pattern (e.g. from search_by_cypher), only use Cypher lanes
            cypher_only = query.strip() == primary_pattern.strip()
            raw_results = self._multi_lane_search(
                query, primary_pattern, max_results, cypher_only=cypher_only,
                directory_scope=directory_scope,
            )

        # Score against the tight (primary) pattern so precise matches rank highest
        search_results = self._process_results(raw_results, primary_pattern, query, context_lines, explain)
        
        # Filter tests if requested
        if exclude_tests:
            search_results = [r for r in search_results if not r.is_test]

        # Apply domain/action filters
        search_results = _filter_by_domain_action(search_results, domain_filter=domain_filter, action_filter=action_filter)
        
        search_results.sort()
        self._last_db_elapsed_seconds = time.perf_counter() - t0
        return search_results[:max_results]

    def search_by_tag(self, tag: str, max_results: int = None) -> List[SearchResult]:
        """Search for functions by tag."""
        max_results = max_results or self._config.max_search_results
        t0 = time.perf_counter()
        raw_results = self.cache.search_by_tags(tag, limit=max_results)
        search_results = self._process_results(raw_results, None, tag, context_lines=3)
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
        self, query: str, patterns: List[str], max_results: int,
        directory_scope: Optional[str] = None,
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
            raw = self._multi_lane_search(
                query, cypher_pattern, max_results,
                directory_scope=directory_scope,
            )
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
                           cypher_only: bool = False,
                           directory_scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Multi-lane search; merges unique results for unified ranking.
        When directory_scope is set, only functions under that path are returned.
        """
        fetch_limit = max(max_results * 5, 30)
        fetch_limit_tag = min(fetch_limit, 50)
        path_prefix = (directory_scope or "").strip() or None
        keywords: List[str] = []

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
        _add(self.cache.search_by_cypher(cypher_pattern, limit=fetch_limit, path_prefix=path_prefix))

        # ── Lane 2: Domain-wildcarded (same ACT_OBJ--PAT) ───────
        parts = cypher_pattern.split(":")
        if len(parts) == 2:
            dom, rest_str = parts
            # Skip when rest is fully wildcard (*_*--*) or we'd match the whole index
            if dom != "*" and rest_str.strip() != "*_*--*":
                _add(self.cache.search_by_cypher(
                    f"*:{rest_str}", limit=fetch_limit, path_prefix=path_prefix
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
                        f"*:{act}_*--*", limit=fetch_limit, path_prefix=path_prefix
                    ))

        if cypher_only:
            logger.debug("Direct Cypher search: tag/name lanes skipped")
        else:
            # ── Lane 4: Tag keyword search ───────────────────────────
            keywords = self._extract_keywords(query)
            for kw in keywords[:5]:
                _add(self.cache.search_by_tags(kw, limit=fetch_limit_tag, path_prefix=path_prefix))

            # ── Lane 5: Function name search ─────────────────────────
            if keywords:
                _add(self.cache.search_by_name(keywords, limit=fetch_limit_tag, path_prefix=path_prefix))

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
                        cypher_pattern: Optional[str], query: str,
                        context_lines: int = 3, explain: bool = False) -> List[SearchResult]:
        """Convert raw database results to SearchResult objects with scoring."""
        search_results = []
        
        # PERFORMANCE FIX: Cache file contents to avoid reading the same file multiple times
        # (common when multiple functions from the same file match)
        file_cache: Dict[str, List[str]] = {}
        
        for result in raw_results:
            # Extract tags
            tags = result.get('tags', '').split(',') if result.get('tags') else []
            func_name = result.get('function_name', '')
            file_path = result['file_path']
            cypher = result['cypher']
            
            # Calculate score
            explanation_dict = None
            if cypher_pattern:
                score_result = calculate_search_score(
                    cypher_pattern, cypher, tags, query,
                    function_name=func_name,
                    config=self._config,
                    explain=explain,
                )
                if explain:
                    score, explanation_dict = score_result
                else:
                    score = score_result
            else:
                score = 1.0  # Base score for tag searches

            path_root = str(self._cache_dir.parent) if self._cache_dir else ""
            # Resolve path for portability: use path_root + relative_path when stored path missing (e.g. index from Docker, search on Windows)
            resolved_path = self._resolve_file_path(file_path, result.get("relative_path"), path_root)
            
            # Extract context using resolved path so we read from the correct location
            context = self._extract_context_with_signature(
                resolved_path,
                result['line_start'],
                result['line_end'],
                file_cache,
                context_lines
            )
            
            # Derive module path from resolved path
            module_path = self._derive_module_path(resolved_path)
            # Detect if this is a test function
            is_test = cypher.startswith('TST:') or self._is_test_file(resolved_path)

            search_results.append(SearchResult(
                file_path=resolved_path,
                function_name=func_name,
                line_start=result['line_start'],
                line_end=result['line_end'],
                cypher=cypher,
                score=score,
                context=context,
                path_root=path_root,
                explanation=explanation_dict,
                module_path=module_path,
                is_test=is_test,
            ))
        
        return search_results
    
    def _resolve_file_path(self, stored_path: str, relative_path: str | None, path_root: str) -> str:
        """Resolve to a path that exists: use path_root + relative_path when stored path is missing (e.g. index in Docker, search on Windows)."""
        p = Path(stored_path)
        if p.exists():
            return str(p.resolve())
        if path_root and relative_path:
            resolved = Path(path_root) / relative_path
            if resolved.exists():
                return str(resolved.resolve())
        return stored_path

    def _extract_context_with_signature(self, file_path: str, start_line: int, 
                                        end_line: int, file_cache: Dict[str, List[str]], 
                                        context_lines: int = 3) -> str:
        """
        Extract code context including full function signature + body preview.
        
        IMPROVEMENT (v1.2): Always includes full function signature, even if multi-line.
        """
        try:
            # Check cache first
            if file_path not in file_cache:
                path = Path(file_path)
                if not path.exists():
                    return "[File not found]"
                file_cache[file_path] = path.read_text(encoding='utf-8').splitlines()
            
            lines = file_cache[file_path]
            function_length = end_line - start_line + 1
            
            # Auto-adjust for small functions
            if function_length <= context_lines * 2:
                # Function is small, return full function
                func_lines = lines[start_line - 1:end_line]
                return "\n".join(func_lines)
            
            # For larger functions: full signature + context_lines of body + extra (docstring/first statement)
            # Find end of signature (line with '):' or ':' at the end)
            signature_end = start_line
            for i in range(start_line - 1, min(start_line + 5, end_line)):
                line = lines[i].rstrip()
                if line.endswith(':') or line.endswith('):'):
                    signature_end = i + 1
                    break
            
            # Add 2 extra lines so first line of docstring or first insert/match often appears (improves quick confirmation)
            extra_lines = 2
            end_preview = min(signature_end + context_lines + extra_lines, end_line)
            func_lines = lines[start_line - 1:end_preview]
            return "\n".join(func_lines)
        except Exception as e:
            logger.warning(f"Could not extract context from {file_path}: {e}")
            return "[Context unavailable]"
    
    def _derive_module_path(self, file_path: str) -> str:
        """Derive Python module path from file path (e.g., 'auth.tokens')."""
        try:
            path = Path(file_path)
            # Find the root (look for common project indicators)
            parts = list(path.parts)
            
            # Try to find src/ or project root
            for i, part in enumerate(parts):
                if part in ('src', 'lib', 'app'):
                    parts = parts[i+1:]
                    break
            
            # Convert to module path
            if path.stem == '__init__':
                # __init__.py → parent package name
                module_parts = parts[:-1]
            else:
                # file.py → file
                module_parts = parts[:-1] + [path.stem]
            
            return '.'.join(module_parts) if module_parts else ""
        except Exception:
            return ""
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file based on path patterns."""
        path_lower = file_path.lower()
        return any([
            '/test/' in path_lower,
            '/tests/' in path_lower,
            '\\test\\' in path_lower,
            '\\tests\\' in path_lower,
            path_lower.endswith('_test.py'),
            path_lower.endswith('test_.py'),
            path_lower.startswith('test_'),
            'conftest.py' in path_lower,
        ])
    
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
    
    # ── Call Graph ────────────────────────────────────────────────

    def get_callers(self, function_name: str, context_lines: int = 3,
                    directory_scope: Optional[str] = None,
                    domain_filter: Optional[List[str]] = None,
                    action_filter: Optional[List[str]] = None) -> List[SearchResult]:
        """Find indexed functions that call *function_name*.

        Args:
            function_name: Name of the target function.
            context_lines: Lines of code context per result.
            directory_scope: If set, restrict to callers under this path.
            domain_filter: If set, keep only callers whose Cypher domain is in this list.
            action_filter: If set, keep only callers whose Cypher action is in this list.

        Returns:
            List of :class:`SearchResult` objects for each caller, with context.
        """
        path_prefix = (directory_scope or "").strip() or None
        raw_results = self.cache.get_callers(function_name, path_prefix=path_prefix)
        results = self._process_call_graph_results(raw_results, context_lines)
        return _filter_by_domain_action(results, domain_filter=domain_filter, action_filter=action_filter)

    def get_callees(self, function_name: str, file_path: str | None = None,
                    context_lines: int = 3,
                    directory_scope: Optional[str] = None,
                    domain_filter: Optional[List[str]] = None,
                    action_filter: Optional[List[str]] = None) -> List[SearchResult]:
        """Find indexed functions called by *function_name*.

        Args:
            function_name: Name of the caller function.
            file_path: Optional source file path to disambiguate when the
                function name exists in multiple files.
            context_lines: Lines of code context per result.
            directory_scope: If set, restrict to callees under this path.
            domain_filter: If set, keep only callees whose Cypher domain is in this list.
            action_filter: If set, keep only callees whose Cypher action is in this list.

        Returns:
            List of :class:`SearchResult` objects for each callee, with context.
        """
        path_prefix = (directory_scope or "").strip() or None
        raw_results = self.cache.get_callees(function_name, caller_file=file_path, path_prefix=path_prefix)
        results = self._process_call_graph_results(raw_results, context_lines)
        return _filter_by_domain_action(results, domain_filter=domain_filter, action_filter=action_filter)

    def trace_call_chain(self, function_name: str, direction: str = "callers",
                         max_depth: int = 5, context_lines: int = 3,
                         directory_scope: Optional[str] = None,
                         domain_filter: Optional[List[str]] = None,
                         action_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Walk the call graph recursively from a starting function.

        Args:
            function_name: Starting function name.
            direction: ``'callers'`` (who calls this — walk up) or
                ``'callees'`` (what this calls — walk down).
            max_depth: Maximum recursion depth (default 5).
            context_lines: Lines of code context per node.
            directory_scope: If set, restrict to nodes under this path.
            domain_filter: If set, keep only nodes whose Cypher domain is in this list.
            action_filter: If set, keep only nodes whose Cypher action is in this list.

        Returns:
            Flat list of dicts, each with function info and a ``depth`` field.
            Ordered by discovery (nearest first). Cycles are detected and
            will not cause infinite recursion.
        """
        visited: set = set()
        chain: List[Dict[str, Any]] = []

        self._walk_call_graph(
            function_name, direction, 1, max_depth,
            visited, chain, context_lines,
            directory_scope=directory_scope,
            domain_filter=domain_filter,
            action_filter=action_filter,
        )

        return chain

    def _walk_call_graph(self, function_name: str, direction: str,
                         current_depth: int, max_depth: int,
                         visited: set, chain: List[Dict[str, Any]],
                         context_lines: int,
                         directory_scope: Optional[str] = None,
                         domain_filter: Optional[List[str]] = None,
                         action_filter: Optional[List[str]] = None) -> None:
        """Recursive DFS helper for :meth:`trace_call_chain`."""
        if current_depth > max_depth:
            return

        path_prefix = (directory_scope or "").strip() or None
        if direction == "callers":
            raw = self.cache.get_callers(function_name, path_prefix=path_prefix)
        else:
            raw = self.cache.get_callees(function_name, path_prefix=path_prefix)

        domains = set(d.upper() for d in (domain_filter or []))
        actions = set(a.upper() for a in (action_filter or []))

        for row in raw:
            fname = row.get("function_name", "")
            fpath = row.get("file_path", "")
            key = (fpath, fname)
            if key in visited:
                continue
            visited.add(key)

            # Build a SearchResult with context
            results = self._process_call_graph_results([row], context_lines)
            if not results:
                continue
            r = results[0]
            # Apply domain/action filter: skip adding to chain if filters set and node doesn't pass
            if domains and _cypher_domain(r.cypher) not in domains:
                self._walk_call_graph(
                    fname, direction, current_depth + 1, max_depth,
                    visited, chain, context_lines,
                    directory_scope=directory_scope,
                    domain_filter=domain_filter,
                    action_filter=action_filter,
                )
                continue
            if actions and _cypher_action(r.cypher) not in actions:
                self._walk_call_graph(
                    fname, direction, current_depth + 1, max_depth,
                    visited, chain, context_lines,
                    directory_scope=directory_scope,
                    domain_filter=domain_filter,
                    action_filter=action_filter,
                )
                continue

            chain.append({
                "function_name": r.function_name,
                "file_path": ResultFormatter._sanitize_for_json(r.file_path),
                "line_start": r.line_start,
                "line_end": r.line_end,
                "cypher": r.cypher,
                "depth": current_depth,
                "context": ResultFormatter._sanitize_for_json(r.context or ""),
                "path_root": ResultFormatter._sanitize_for_json(r.path_root or ""),
            })

            # Recurse into next level
            self._walk_call_graph(
                fname, direction, current_depth + 1, max_depth,
                visited, chain, context_lines,
                directory_scope=directory_scope,
                domain_filter=domain_filter,
                action_filter=action_filter,
            )

    def _process_call_graph_results(self, raw_results: List[Dict[str, Any]],
                                     context_lines: int) -> List[SearchResult]:
        """Convert raw call-graph DB rows to :class:`SearchResult` objects with context."""
        search_results = []
        file_cache: Dict[str, List[str]] = {}

        for result in raw_results:
            func_name = result.get("function_name", "")
            file_path = result.get("file_path", "")
            cypher = result.get("cypher", "")

            if not file_path or not func_name:
                continue

            path_root = str(self._cache_dir.parent) if self._cache_dir else ""
            resolved_path = self._resolve_file_path(
                file_path, result.get("relative_path"), path_root,
            )

            line_start = result.get("line_start", 0)
            line_end = result.get("line_end", 0)

            context = ""
            if line_start and line_end:
                context = self._extract_context_with_signature(
                    resolved_path, line_start, line_end,
                    file_cache, context_lines,
                )

            module_path = self._derive_module_path(resolved_path)
            is_test = cypher.startswith("TST:") or self._is_test_file(resolved_path)

            search_results.append(SearchResult(
                file_path=resolved_path,
                function_name=func_name,
                line_start=line_start or 0,
                line_end=line_end or 0,
                cypher=cypher,
                score=0.0,  # Call-graph results are not scored by relevance
                context=context,
                path_root=path_root,
                module_path=module_path,
                is_test=is_test,
            ))

        return search_results

    # ── Statistics ─────────────────────────────────────────────────

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
            
            # Show scoring explanation if available
            if r.explanation:
                parts = []
                if 'action_match' in r.explanation:
                    parts.append(f"action(+{r.explanation['action_match']:.0f})")
                if 'object_match' in r.explanation:
                    parts.append(f"object(+{r.explanation['object_match']:.0f})")
                if 'domain_match' in r.explanation:
                    parts.append(f"domain(+{r.explanation['domain_match']:.0f})")
                if 'name_matches' in r.explanation:
                    parts.append(f"name(+{r.explanation['name_matches']['score']:.1f})")
                if parts:
                    out.append(f"    Explain: {' '.join(parts)}")

            if show_context and r.context:
                out.append("")
                out.extend(
                    ResultFormatter._numbered_preview(r.context, r.line_start)
                )

        out.append(f"\n{thin}")
        return "\n".join(out)

    # ── JSON ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_for_json(s: str) -> str:
        """Ensure string is safe for JSON (no control chars, normalise path separators for portability)."""
        if not s:
            return s
        # Normalise backslashes to forward slashes so emitted JSON has no backslash in paths
        s = s.replace("\\", "/")
        # Remove control characters that can break strict JSON parsers
        return "".join(c for c in s if (ord(c) >= 32 and ord(c) != 127) or c in "\n\r\t")

    @staticmethod
    def format_json(results: List[SearchResult], group_by: Optional[str] = None) -> str:
        """Format results as JSON (full paths, all fields). Includes path_root when set.
        When group_by is 'domain' or 'action', returns a dict grouped by Cypher domain or action.
        Paths are normalised to forward slashes; control chars in context are stripped so output is valid JSON.
        """
        def _to_obj(r: SearchResult) -> dict:
            obj = {
                "function_name": r.function_name,
                "file_path": ResultFormatter._sanitize_for_json(r.file_path),
                "line_start": r.line_start,
                "line_end": r.line_end,
                "cypher": r.cypher,
                "score": round(r.score, 2),
                "language": ResultFormatter._detect_language(r.file_path),
                "context": ResultFormatter._sanitize_for_json(r.context or ""),
            }
            if r.path_root:
                obj["path_root"] = ResultFormatter._sanitize_for_json(r.path_root)
            return obj

        if group_by == "domain":
            by_domain: Dict[str, List[dict]] = {}
            for r in results:
                dom = _cypher_domain(r.cypher) or "_"
                by_domain.setdefault(dom, []).append(_to_obj(r))
            return json.dumps({"by_domain": by_domain}, indent=2, allow_nan=False)
        if group_by == "action":
            by_action: Dict[str, List[dict]] = {}
            for r in results:
                act = _cypher_action(r.cypher) or "_"
                by_action.setdefault(act, []).append(_to_obj(r))
            return json.dumps({"by_action": by_action}, indent=2, allow_nan=False)
        json_results = [_to_obj(r) for r in results]
        return json.dumps(json_results, indent=2, allow_nan=False)

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
