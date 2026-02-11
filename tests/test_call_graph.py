"""
Tests for the call graph feature — call-site extraction, edge storage,
caller/callee queries, and recursive call-chain tracing.
"""

from pathlib import Path
from typing import List

import pytest
from symdex.core.config import SymdexConfig
from symdex.core.engine import (
    CallSite,
    CodeAnalyzer,
    CypherCache,
    FunctionMetadata,
)
from symdex.core.search import CypherSearchEngine


# =============================================================================
# CallSite extraction via CodeAnalyzer
# =============================================================================

class TestCallSiteExtraction:
    """Verify that CodeAnalyzer populates call_sites with line numbers."""

    def test_extracts_simple_calls(self):
        code = (
            "def process(data):\n"
            "    cleaned = clean(data)\n"
            "    result = transform(cleaned)\n"
            "    save(result)\n"
        )
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        assert len(funcs) == 1
        callee_names = [cs.callee_name for cs in funcs[0].call_sites]
        assert "clean" in callee_names
        assert "transform" in callee_names
        assert "save" in callee_names

    def test_extracts_method_calls(self):
        code = (
            "def process(obj):\n"
            "    obj.validate()\n"
            "    result = obj.transform()\n"
            "    return result\n"
        )
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        callee_names = [cs.callee_name for cs in funcs[0].call_sites]
        assert "validate" in callee_names
        assert "transform" in callee_names

    def test_celery_delay_treated_as_task_call(self):
        """Celery task.delay() and task.apply_async() record the task as callee, not 'delay'/'apply_async'."""
        code = (
            "def run_validation():\n"
            "    validate_vulnerabilities_task.delay()\n"
            "    other_task.apply_async()\n"
        )
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        callee_names = [cs.callee_name for cs in funcs[0].call_sites]
        assert "validate_vulnerabilities_task" in callee_names
        assert "other_task" in callee_names
        assert "delay" not in callee_names
        assert "apply_async" not in callee_names

    def test_call_sites_have_line_numbers(self):
        code = (
            "def process(data):\n"       # line 1
            "    cleaned = clean(data)\n"  # line 2
            "    save(cleaned)\n"          # line 3
        )
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        sites = funcs[0].call_sites
        clean_site = next(cs for cs in sites if cs.callee_name == "clean")
        assert clean_site.line == 2
        save_site = next(cs for cs in sites if cs.callee_name == "save")
        assert save_site.line == 3

    def test_backward_compat_calls_field(self):
        """The legacy 'calls' field should still be populated (deduped, max 10)."""
        code = (
            "def process(data):\n"
            "    clean(data)\n"
            "    clean(data)\n"        # duplicate
            "    transform(data)\n"
        )
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        func = funcs[0]
        assert "clean" in func.calls
        assert "transform" in func.calls

    def test_no_calls_empty_lists(self):
        code = "def noop():\n    pass\n"
        funcs = CodeAnalyzer.extract_functions(code, "example.py")
        func = funcs[0]
        assert func.call_sites == []
        assert func.calls == []

    def test_call_site_to_dict(self):
        cs = CallSite(callee_name="foo", line=42)
        d = cs.to_dict()
        assert d == {"callee_name": "foo", "line": 42}

    def test_function_metadata_default_call_sites(self):
        """FunctionMetadata created without call_sites should default to []."""
        meta = FunctionMetadata(
            name="f", start_line=1, end_line=2, is_async=False,
            args=[], calls=[], imports=[], docstring=None,
            complexity=1, language="Python",
        )
        assert meta.call_sites == []


# =============================================================================
# CypherCache — call_edges table
# =============================================================================

class TestCypherCacheCallEdges:
    """Test call edge storage and retrieval in CypherCache."""

    @pytest.fixture
    def cache(self, tmp_path) -> CypherCache:
        c = CypherCache(tmp_path / "test_cache.db")
        yield c
        c.close()

    @pytest.fixture
    def sample_files(self, tmp_path) -> dict:
        """Create sample Python files for realistic testing."""
        f1 = tmp_path / "file_utils.py"
        f1.write_text(
            "def encrypt_file_in_place(fp): ...\n"
            "def encrypt_file_content(data): ...\n",
            encoding="utf-8",
        )
        f2 = tmp_path / "tasks.py"
        f2.write_text(
            "def process_files(files): ...\n"
            "def process_files_batch(files): ...\n",
            encoding="utf-8",
        )
        return {"file_utils": f1, "tasks": f2}

    def _populate_index(self, cache, sample_files):
        """Add cypher entries and call edges for a realistic call chain:
        process_files → process_files_batch → encrypt_file_in_place → encrypt_file_content
        """
        f1 = sample_files["file_utils"]
        f2 = sample_files["tasks"]

        # Cypher entries
        cache.add_cypher_entry(
            f1, "encrypt_file_content", 2, 5,
            "SEC:TRN_FILE--SYN", ["encrypt"], "→ bytes", "O(1)",
        )
        cache.add_cypher_entry(
            f1, "encrypt_file_in_place", 7, 15,
            "SEC:TRN_FILE--SYN", ["encrypt", "file"], "→ None", "O(N)",
        )
        cache.add_cypher_entry(
            f2, "process_files_batch", 5, 20,
            "DAT:TRN_FILE--SYN", ["process", "batch"], "→ list", "O(N)",
        )
        cache.add_cypher_entry(
            f2, "process_files", 22, 30,
            "DAT:TRN_FILE--SYN", ["process"], "→ None", "O(N)",
        )

        # Call edges
        cache.add_call_edges(f2, "process_files", 22, [
            CallSite("process_files_batch", 25),
        ])
        cache.add_call_edges(f2, "process_files_batch", 5, [
            CallSite("encrypt_file_in_place", 12),
        ])
        cache.add_call_edges(f1, "encrypt_file_in_place", 7, [
            CallSite("encrypt_file_content", 10),
        ])

    def test_add_and_get_callers(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        callers = cache.get_callers("encrypt_file_content")
        assert len(callers) == 1
        assert callers[0]["function_name"] == "encrypt_file_in_place"

    def test_get_callers_returns_metadata(self, cache, sample_files):
        """Callers should include full cypher_index metadata via JOIN."""
        self._populate_index(cache, sample_files)
        callers = cache.get_callers("encrypt_file_content")
        caller = callers[0]
        assert caller["cypher"] == "SEC:TRN_FILE--SYN"
        assert caller["line_start"] == 7
        assert caller["line_end"] == 15

    def test_get_callers_empty_for_top_level(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        # process_files is the top-level entry — no one calls it
        callers = cache.get_callers("process_files")
        assert callers == []

    def test_get_callees(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        callees = cache.get_callees("encrypt_file_in_place")
        assert len(callees) == 1
        assert callees[0]["function_name"] == "encrypt_file_content"

    def test_get_callees_with_file_filter(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        f2 = sample_files["tasks"]
        callees = cache.get_callees("process_files", caller_file=str(f2))
        assert len(callees) == 1
        assert callees[0]["function_name"] == "process_files_batch"

    def test_get_callees_empty_for_leaf(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        # encrypt_file_content is a leaf — it doesn't call indexed functions
        callees = cache.get_callees("encrypt_file_content")
        assert callees == []

    def test_clear_file_entries_removes_call_edges(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        f2 = sample_files["tasks"]
        cache.clear_file_entries(f2)
        # Call edges from tasks.py should be gone
        callees = cache.get_callees("process_files")
        assert callees == []
        # But call edges from file_utils.py should remain
        callers = cache.get_callers("encrypt_file_content")
        assert len(callers) == 1

    def test_stats_include_call_edges(self, cache, sample_files):
        self._populate_index(cache, sample_files)
        stats = cache.get_stats()
        assert stats["call_edges"] == 3

    def test_deduplicate_call_edges(self, cache, sample_files):
        """Multiple calls to the same function from one caller should be deduplicated."""
        f1 = sample_files["file_utils"]
        cache.add_cypher_entry(
            f1, "helper", 1, 3, "DAT:TRN_DATA--SYN", [], "→ Any", "O(1)",
        )
        cache.add_call_edges(f1, "helper", 1, [
            CallSite("encrypt_file_content", 2),
            CallSite("encrypt_file_content", 3),  # same callee, different line
        ])
        stats = cache.get_stats()
        assert stats["call_edges"] == 1

    def test_add_call_edges_empty_list(self, cache, sample_files):
        """Passing an empty list should be a no-op."""
        cache.add_call_edges(sample_files["file_utils"], "noop", 1, [])
        stats = cache.get_stats()
        assert stats["call_edges"] == 0


# =============================================================================
# CypherSearchEngine — call graph queries
# =============================================================================

class TestSearchEngineCallGraph:
    """Test get_callers, get_callees, and trace_call_chain on CypherSearchEngine."""

    @pytest.fixture
    def engine_with_call_graph(self, tmp_path):
        """Create an engine with a populated index including call edges."""
        cache_dir = tmp_path / ".symdex"
        cache_dir.mkdir()

        cfg = SymdexConfig.from_env()

        # Create source files (needed for context extraction)
        f1 = tmp_path / "encryption.py"
        f1.write_text(
            "def encrypt_file_content(data: bytes) -> bytes:\n"
            "    \"\"\"Encrypt raw file content.\"\"\"\n"
            "    return aes_encrypt(data)\n"
            "\n"
            "def encrypt_file_in_place(file_path: str) -> None:\n"
            "    \"\"\"Read, encrypt, and write back a file.\"\"\"\n"
            "    data = read_file(file_path)\n"
            "    encrypted = encrypt_file_content(data)\n"
            "    write_file(file_path, encrypted)\n",
            encoding="utf-8",
        )
        f2 = tmp_path / "tasks.py"
        f2.write_text(
            "def process_files_batch(files: list) -> list:\n"
            "    \"\"\"Process a batch of files.\"\"\"\n"
            "    results = []\n"
            "    for f in files:\n"
            "        encrypt_file_in_place(f)\n"
            "        results.append(f)\n"
            "    return results\n"
            "\n"
            "def process_files(directory: str) -> None:\n"
            "    \"\"\"Entry point: process all files in directory.\"\"\"\n"
            "    files = list_files(directory)\n"
            "    process_files_batch(files)\n",
            encoding="utf-8",
        )

        # Populate the cache
        cache = CypherCache(cfg.get_cache_path(cache_dir))

        cache.add_cypher_entry(
            f1, "encrypt_file_content", 1, 3,
            "SEC:TRN_FILE--SYN", ["encrypt", "file"], "[data] → bytes", "O(1)",
        )
        cache.add_cypher_entry(
            f1, "encrypt_file_in_place", 5, 9,
            "SEC:TRN_FILE--SYN", ["encrypt", "file"], "[file_path] → None", "O(N)",
        )
        cache.add_cypher_entry(
            f2, "process_files_batch", 1, 7,
            "DAT:TRN_FILE--SYN", ["process", "batch"], "[files] → list", "O(N)",
        )
        cache.add_cypher_entry(
            f2, "process_files", 9, 12,
            "DAT:TRN_FILE--SYN", ["process"], "[directory] → None", "O(N)",
        )

        # Call edges mirroring the actual call relationships
        cache.add_call_edges(f2, "process_files", 9, [
            CallSite("list_files", 11),
            CallSite("process_files_batch", 12),
        ])
        cache.add_call_edges(f2, "process_files_batch", 1, [
            CallSite("encrypt_file_in_place", 5),
        ])
        cache.add_call_edges(f1, "encrypt_file_in_place", 5, [
            CallSite("read_file", 7),
            CallSite("encrypt_file_content", 8),
            CallSite("write_file", 9),
        ])
        cache.add_call_edges(f1, "encrypt_file_content", 1, [
            CallSite("aes_encrypt", 3),
        ])

        cache.close()

        engine = CypherSearchEngine(cache_dir, config=cfg)
        return engine

    # ── get_callers ───────────────────────────────────────────────

    def test_get_callers_returns_results(self, engine_with_call_graph):
        callers = engine_with_call_graph.get_callers("encrypt_file_content")
        assert len(callers) == 1
        assert callers[0].function_name == "encrypt_file_in_place"

    def test_get_callers_includes_context(self, engine_with_call_graph):
        callers = engine_with_call_graph.get_callers(
            "encrypt_file_content", context_lines=5,
        )
        assert len(callers) == 1
        assert "encrypt_file_in_place" in callers[0].context

    def test_get_callers_empty_for_top_level(self, engine_with_call_graph):
        callers = engine_with_call_graph.get_callers("process_files")
        assert callers == []

    # ── get_callees ───────────────────────────────────────────────

    def test_get_callees_returns_indexed_only(self, engine_with_call_graph):
        """get_callees should only return functions that are in the index."""
        callees = engine_with_call_graph.get_callees("encrypt_file_in_place")
        callee_names = [c.function_name for c in callees]
        assert "encrypt_file_content" in callee_names
        # read_file and write_file are NOT indexed, so they shouldn't appear
        assert "read_file" not in callee_names
        assert "write_file" not in callee_names

    def test_get_callees_from_entry_point(self, engine_with_call_graph):
        callees = engine_with_call_graph.get_callees("process_files")
        callee_names = [c.function_name for c in callees]
        assert "process_files_batch" in callee_names

    def test_get_callees_empty_for_leaf(self, engine_with_call_graph):
        """encrypt_file_content only calls aes_encrypt which is not indexed."""
        callees = engine_with_call_graph.get_callees("encrypt_file_content")
        assert callees == []

    # ── trace_call_chain ──────────────────────────────────────────

    def test_trace_callers_chain(self, engine_with_call_graph):
        """Trace callers from encrypt_file_content should walk up the full chain."""
        chain = engine_with_call_graph.trace_call_chain(
            "encrypt_file_content", direction="callers", max_depth=5,
        )
        names = [node["function_name"] for node in chain]
        assert "encrypt_file_in_place" in names
        assert "process_files_batch" in names
        assert "process_files" in names

        # Verify depth ordering
        depths = {node["function_name"]: node["depth"] for node in chain}
        assert depths["encrypt_file_in_place"] == 1
        assert depths["process_files_batch"] == 2
        assert depths["process_files"] == 3

    def test_trace_callees_chain(self, engine_with_call_graph):
        """Trace callees from process_files should walk down the chain."""
        chain = engine_with_call_graph.trace_call_chain(
            "process_files", direction="callees", max_depth=5,
        )
        names = [node["function_name"] for node in chain]
        assert "process_files_batch" in names
        assert "encrypt_file_in_place" in names
        assert "encrypt_file_content" in names

    def test_trace_respects_max_depth(self, engine_with_call_graph):
        chain = engine_with_call_graph.trace_call_chain(
            "encrypt_file_content", direction="callers", max_depth=1,
        )
        # Should only get direct callers (depth 1)
        assert all(node["depth"] == 1 for node in chain)
        assert len(chain) == 1
        assert chain[0]["function_name"] == "encrypt_file_in_place"

    def test_trace_includes_context(self, engine_with_call_graph):
        chain = engine_with_call_graph.trace_call_chain(
            "encrypt_file_content", direction="callers",
            max_depth=1, context_lines=5,
        )
        assert len(chain) == 1
        assert chain[0]["context"]  # should have code preview

    def test_trace_handles_cycles(self, tmp_path):
        """Trace should not infinite-loop on cycles (A calls B, B calls A)."""
        cache_dir = tmp_path / ".symdex"
        cache_dir.mkdir()
        cfg = SymdexConfig.from_env()

        f1 = tmp_path / "cycle.py"
        f1.write_text("def a(): b()\ndef b(): a()\n", encoding="utf-8")

        cache = CypherCache(cfg.get_cache_path(cache_dir))
        cache.add_cypher_entry(
            f1, "a", 1, 1, "DAT:TRN_DATA--SYN", [], "→ Any", "O(1)",
        )
        cache.add_cypher_entry(
            f1, "b", 2, 2, "DAT:TRN_DATA--SYN", [], "→ Any", "O(1)",
        )
        cache.add_call_edges(f1, "a", 1, [CallSite("b", 1)])
        cache.add_call_edges(f1, "b", 2, [CallSite("a", 2)])
        cache.close()

        engine = CypherSearchEngine(cache_dir, config=cfg)
        chain = engine.trace_call_chain("a", direction="callees", max_depth=10)
        # Should not infinite loop; b calls a but a is already visited
        assert len(chain) <= 2
        names = [n["function_name"] for n in chain]
        assert "b" in names

    def test_trace_empty_for_leaf_callers(self, engine_with_call_graph):
        """A function with no callers should return empty chain."""
        chain = engine_with_call_graph.trace_call_chain(
            "process_files", direction="callers", max_depth=5,
        )
        assert chain == []

    def test_trace_empty_for_leaf_callees(self, engine_with_call_graph):
        """A function whose callees are all unindexed should return empty chain."""
        chain = engine_with_call_graph.trace_call_chain(
            "encrypt_file_content", direction="callees", max_depth=5,
        )
        assert chain == []
