"""
Tests for symdex.core.engine — CodeAnalyzer, CypherMeta, CypherCache,
CypherGenerator helpers, LLM providers, scan_directory, and search scoring.
"""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from symdex.core.engine import (
    CodeAnalyzer,
    CypherCache,
    CypherGenerator,
    CypherMeta,
    FunctionMetadata,
    IndexResult,
    LLMProvider,
    _create_provider,
    _PROVIDER_REGISTRY,
    calculate_search_score,
    scan_directory,
)


# =============================================================================
# CodeAnalyzer — Python AST extraction
# =============================================================================

class TestCodeAnalyzerPython:
    """Python-specific AST extraction via CodeAnalyzer."""

    def test_extract_sync_function(self, python_source):
        funcs = CodeAnalyzer.extract_functions(python_source, "example.py")
        names = [f.name for f in funcs]
        assert "fetch_user" in names

    def test_extract_async_function(self, python_source):
        funcs = CodeAnalyzer.extract_functions(python_source, "example.py")
        async_funcs = [f for f in funcs if f.is_async]
        assert any(f.name == "send_email" for f in async_funcs)

    def test_function_metadata_fields(self, python_source):
        funcs = CodeAnalyzer.extract_functions(python_source, "example.py")
        fetch = next(f for f in funcs if f.name == "fetch_user")
        assert fetch.start_line > 0
        assert fetch.end_line >= fetch.start_line
        assert fetch.language == "Python"
        assert "user_id" in fetch.args

    def test_docstring_extracted(self, python_source):
        funcs = CodeAnalyzer.extract_functions(python_source, "example.py")
        fetch = next(f for f in funcs if f.name == "fetch_user")
        assert fetch.docstring is not None
        assert "user" in fetch.docstring.lower()

    def test_complexity_calculated(self, python_source):
        funcs = CodeAnalyzer.extract_functions(python_source, "example.py")
        fetch = next(f for f in funcs if f.name == "fetch_user")
        # if-branch → complexity >= 2
        assert fetch.complexity >= 2

    def test_syntax_error_returns_empty(self):
        bad_code = "def broken(\n"
        funcs = CodeAnalyzer.extract_functions(bad_code, "bad.py")
        assert funcs == []


# =============================================================================
# CodeAnalyzer — Shared helpers
# =============================================================================

class TestCodeAnalyzerHelpers:
    """Shared utility methods on CodeAnalyzer."""

    def test_extract_function_source(self):
        code = "line1\nline2\nline3\nline4\n"
        result = CodeAnalyzer.extract_function_source(code, 2, 3)
        assert result == "line2\nline3"

    def test_generate_signature_with_args(self):
        meta = FunctionMetadata(
            name="foo", start_line=1, end_line=5, is_async=False,
            args=["a", "b"], calls=[], imports=[], docstring=None,
            complexity=1, language="Python",
        )
        sig = CodeAnalyzer.generate_signature(meta)
        assert "a, b" in sig

    def test_generate_signature_no_args(self):
        meta = FunctionMetadata(
            name="foo", start_line=1, end_line=5, is_async=False,
            args=[], calls=[], imports=[], docstring=None,
            complexity=1, language="Python",
        )
        sig = CodeAnalyzer.generate_signature(meta)
        assert "void" in sig

    @pytest.mark.parametrize("complexity,expected", [
        (1, "O(1)"),
        (2, "O(1)"),
        (3, "O(N)"),
        (5, "O(N)"),
        (8, "O(N²)"),
        (15, "O(N³+)"),
    ])
    def test_estimate_complexity_class(self, complexity, expected):
        assert CodeAnalyzer.estimate_complexity_class(complexity) == expected


# =============================================================================
# CypherMeta — comment block generation and parsing
# =============================================================================

class TestCypherMeta:
    """SEARCH_META comment block generation and parsing."""

    def _make_meta(self) -> CypherMeta:
        return CypherMeta(
            cypher="SEC:VAL_TOKEN--SYN",
            tags=["validate", "token", "security"],
            signature="[token] → bool",
            complexity="O(1)",
        )

    def test_to_comment_block_python(self):
        block = self._make_meta().to_comment_block()
        assert block.startswith("# <SEARCH_META")
        assert "# CYPHER: SEC:VAL_TOKEN--SYN" in block
        assert "# TAGS:" in block
        assert "# </SEARCH_META>" in block

    def test_parse_from_python_code(self):
        code = (
            "# <SEARCH_META v1.0>\n"
            "# CYPHER: DAT:FET_USER--SYN\n"
            "# SIG: [user_id] → Any\n"
            "# TAGS: #data, #fetch\n"
            "# COMPLEXITY: O(N)\n"
            "# </SEARCH_META>\n"
            "def fetch_user(user_id):\n"
            "    pass\n"
        )
        meta = CypherMeta.parse_from_code(code)
        assert meta is not None
        assert meta.cypher == "DAT:FET_USER--SYN"
        assert "data" in meta.tags
        assert "fetch" in meta.tags

    def test_parse_returns_none_when_no_meta(self):
        code = "def plain_function():\n    pass\n"
        assert CypherMeta.parse_from_code(code) is None


# =============================================================================
# CypherCache — SQLite cache operations
# =============================================================================

class TestCypherCache:
    """Cache CRUD operations."""

    @pytest.fixture
    def cache(self, tmp_path) -> CypherCache:
        c = CypherCache(tmp_path / "test_cache.db")
        yield c
        c.close()  # avoid ResourceWarning: unclosed database

    @pytest.fixture
    def sample_file(self, tmp_path) -> Path:
        f = tmp_path / "sample.py"
        f.write_text("def foo(): pass\n", encoding="utf-8")
        return f

    def test_new_file_not_indexed(self, cache, sample_file):
        assert cache.is_file_indexed(sample_file) is False

    def test_mark_file_indexed(self, cache, sample_file):
        cache.mark_file_indexed(sample_file, function_count=1)
        assert cache.is_file_indexed(sample_file) is True

    def test_modified_file_invalidates_cache(self, cache, sample_file):
        cache.mark_file_indexed(sample_file, function_count=1)
        # Modify the file
        sample_file.write_text("def bar(): pass\n", encoding="utf-8")
        assert cache.is_file_indexed(sample_file) is False

    def test_add_and_search_cypher_entry(self, cache, sample_file):
        cache.add_cypher_entry(
            sample_file, "foo", 1, 3, "DAT:FET_DATA--SYN",
            ["data", "fetch"], "[void] → Any", "O(1)"
        )
        results = cache.search_by_cypher("DAT:FET_DATA--SYN")
        assert len(results) == 1
        assert results[0]["function_name"] == "foo"

    def test_search_with_wildcard(self, cache, sample_file):
        cache.add_cypher_entry(
            sample_file, "foo", 1, 3, "DAT:FET_DATA--SYN",
            ["data"], "[void] → Any", "O(1)"
        )
        cache.add_cypher_entry(
            sample_file, "bar", 5, 8, "DAT:TRN_DATA--SYN",
            ["data"], "[void] → Any", "O(N)"
        )
        results = cache.search_by_cypher("DAT:%_DATA--SYN")
        assert len(results) == 2

    def test_search_by_tags(self, cache, sample_file):
        cache.add_cypher_entry(
            sample_file, "foo", 1, 3, "SEC:VAL_TOKEN--SYN",
            ["security", "validate"], "[token] → bool", "O(1)"
        )
        results = cache.search_by_tags("security")
        assert len(results) == 1

    def test_get_stats(self, cache, sample_file):
        cache.mark_file_indexed(sample_file, 1)
        cache.add_cypher_entry(
            sample_file, "foo", 1, 3, "DAT:FET_DATA--SYN",
            [], "", "O(1)"
        )
        stats = cache.get_stats()
        assert stats["indexed_files"] == 1
        assert stats["indexed_functions"] == 1

    def test_clear_file_entries(self, cache, sample_file):
        cache.mark_file_indexed(sample_file, 1)
        cache.add_cypher_entry(
            sample_file, "foo", 1, 3, "DAT:FET_DATA--SYN",
            [], "", "O(1)"
        )
        cache.clear_file_entries(sample_file)
        assert cache.get_stats()["indexed_functions"] == 0


# =============================================================================
# CypherGenerator — validation & fallback (no API calls)
# =============================================================================

class TestCypherGeneratorValidation:
    """Test Cypher validation and fallback logic (offline, no API key needed).
    
    These methods are static — no CypherGenerator instance (and therefore
    no ``anthropic`` import) is required.
    """

    @pytest.mark.parametrize("cypher,valid", [
        ("SEC:VAL_TOKEN--SYN", True),
        ("DAT:FET_USER--ASY", True),
        ("NET:SND_EMAL--SYN", True),
        ("UI:TRN_NODE--REC", True),
        ("LOG:SCR_LOGS--GEN", True),
        # Valid: longer OBJ tokens (up to 20 chars) are allowed
        ("BIZ:FET_RELATIONSHIPS--SYN", True),
        ("SEC:VAL_AUTHENTICATION--SYN", True),
        ("BIZ:CRT_RECOMMENDATIONS--SYN", True),
        ("BIZ:FLT_RECOMMENDATIONS--SYN", True),
        # Invalid: wrong domain
        ("XXX:VAL_TOKEN--SYN", False),
        # Invalid: missing parts
        ("SEC:VAL--SYN", False),
        # Invalid: lowercase
        ("sec:val_token--syn", False),
        # Invalid: no double-dash
        ("SEC:VAL_TOKEN-SYN", False),
    ])
    def test_validate_cypher(self, cypher, valid):
        assert CypherGenerator._validate_cypher(cypher) == valid

    @pytest.mark.parametrize("pattern,valid", [
        ("SEC:*_*--*", True),
        ("*:VAL_*--SYN", True),
        ("*:*_*--*", True),
        ("DAT:FET_USER--*", True),
    ])
    def test_validate_cypher_with_wildcards(self, pattern, valid):
        assert CypherGenerator._validate_cypher(pattern, allow_wildcards=True) == valid

    def test_fallback_cypher_generation(self):
        meta = FunctionMetadata(
            name="validate_password", start_line=1, end_line=5,
            is_async=False, args=["pwd"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        cypher = CypherGenerator._generate_fallback_cypher(meta)
        # Should return a valid Cypher string
        assert CypherGenerator._validate_cypher(cypher)
        # "validate" should map to VAL action
        assert ":VAL_" in cypher

    def test_fallback_async_pattern(self):
        meta = FunctionMetadata(
            name="send_message", start_line=1, end_line=5,
            is_async=True, args=["msg"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        cypher = CypherGenerator._generate_fallback_cypher(meta)
        assert cypher.endswith("--ASY")

    def test_keyword_based_translation(self):
        result = CypherGenerator._keyword_based_translation("validate security tokens")
        assert "SEC" in result or "VAL" in result


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class TestLLMProviderAbstraction:
    """Verify the provider factory and polymorphic interface."""

    def test_registry_has_all_providers(self):
        assert "anthropic" in _PROVIDER_REGISTRY
        assert "openai" in _PROVIDER_REGISTRY
        assert "gemini" in _PROVIDER_REGISTRY

    def test_create_provider_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _create_provider(provider="foo_llm", api_key="key")

    def test_base_provider_not_implemented(self):
        base = LLMProvider()
        with pytest.raises(NotImplementedError):
            base.complete(system="s", user_message="u")

    def test_cypher_generator_with_mock_provider(self):
        """CypherGenerator delegates to the underlying provider."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.return_value = "SEC:VAL_TOKEN--SYN"

        gen = CypherGenerator.__new__(CypherGenerator)
        gen.llm = mock_provider

        meta = FunctionMetadata(
            name="validate_token", start_line=1, end_line=5,
            is_async=False, args=["token"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        result = gen.generate_cypher("def validate_token(token): ...", meta)
        assert result == "SEC:VAL_TOKEN--SYN"
        mock_provider.complete.assert_called_once()

    def test_cypher_generator_translate_query_with_mock(self):
        """translate_query returns a list of 1–3 Cypher patterns (tiered)."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.return_value = "SEC:VAL_*--*"

        gen = CypherGenerator.__new__(CypherGenerator)
        gen.llm = mock_provider

        result = gen.translate_query("find security validation functions")
        assert result == ["SEC:VAL_*--*"]
        mock_provider.complete.assert_called_once()

    def test_cypher_generator_falls_back_on_provider_error(self):
        """When all LLM attempts fail, the rule-based fallback is used."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = RuntimeError("API down")

        gen = CypherGenerator.__new__(CypherGenerator)
        gen.llm = mock_provider

        meta = FunctionMetadata(
            name="validate_password", start_line=1, end_line=5,
            is_async=False, args=["pwd"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        result = gen.generate_cypher("def validate_password(pwd): ...", meta)
        # Should be a valid fallback cypher
        assert CypherGenerator._validate_cypher(result)
        assert ":VAL_" in result

    def test_cypher_generator_retries_invalid_cypher(self):
        """When LLM returns garbage, it retries before falling back."""
        mock_provider = MagicMock(spec=LLMProvider)
        # First two calls return invalid, third returns valid
        mock_provider.complete.side_effect = [
            "not-a-cypher",
            "also-bad",
            "DAT:FET_USER--ASY",
        ]

        gen = CypherGenerator.__new__(CypherGenerator)
        gen.llm = mock_provider

        meta = FunctionMetadata(
            name="fetch_user", start_line=1, end_line=5,
            is_async=True, args=["uid"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        result = gen.generate_cypher("async function fetchUser(uid) {...}", meta)
        assert result == "DAT:FET_USER--ASY"
        assert mock_provider.complete.call_count == 3


# =============================================================================
# scan_directory — Python file scanning
# =============================================================================

class TestScanDirectory:
    """Directory scanning respects extensions and exclusions."""

    def test_finds_python_files(self, tmp_project):
        files = scan_directory(tmp_project)
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) >= 1

    def test_excludes_pycache(self, tmp_project):
        files = scan_directory(tmp_project)
        for f in files:
            assert "__pycache__" not in f.parts

    def test_returns_sorted_paths(self, tmp_project):
        files = scan_directory(tmp_project)
        assert files == sorted(files)

    def test_empty_directory(self, tmp_path):
        files = scan_directory(tmp_path)
        assert files == []


# =============================================================================
# calculate_search_score
# =============================================================================

class TestSearchScoring:
    """Verify relevance scoring logic."""

    def test_exact_match_gets_highest_score(self):
        score = calculate_search_score(
            cypher_pattern="SEC:VAL_TOKEN--SYN",
            result_cypher="SEC:VAL_TOKEN--SYN",
            tags=["validate", "token"],
            query="validate token",
        )
        # Exact match bonus + domain + action + object + pattern
        assert score >= 20.0

    def test_wildcard_domain_still_scores(self):
        score = calculate_search_score(
            cypher_pattern="*:VAL_TOKEN--SYN",
            result_cypher="SEC:VAL_TOKEN--SYN",
            tags=[],
            query="validate token",
        )
        assert score > 0

    def test_completely_different_cypher_low_score(self):
        score = calculate_search_score(
            cypher_pattern="SEC:VAL_TOKEN--SYN",
            result_cypher="LOG:AGG_METRIC--ASY",
            tags=[],
            query="validate token",
        )
        # No slot matches — score should be minimal
        assert score < 5.0

    def test_function_name_boosts_score(self):
        score_with_name = calculate_search_score(
            cypher_pattern="*:*_*--*",
            result_cypher="DAT:FET_USER--SYN",
            tags=[],
            query="fetch user",
            function_name="fetch_user",
        )
        score_without_name = calculate_search_score(
            cypher_pattern="*:*_*--*",
            result_cypher="DAT:FET_USER--SYN",
            tags=[],
            query="fetch user",
            function_name="some_other_func",
        )
        assert score_with_name > score_without_name

    def test_tag_match_adds_score(self):
        score_with_tags = calculate_search_score(
            cypher_pattern="*:*_*--*",
            result_cypher="DAT:FET_DATA--SYN",
            tags=["database", "fetch", "cache"],
            query="database fetch",
        )
        score_no_tags = calculate_search_score(
            cypher_pattern="*:*_*--*",
            result_cypher="DAT:FET_DATA--SYN",
            tags=[],
            query="database fetch",
        )
        assert score_with_tags > score_no_tags
