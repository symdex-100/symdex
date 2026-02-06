"""
Tests for symdex.core.config — CypherSchema, LanguageRegistry, Config, Prompts.
"""

import pytest
from symdex.core.config import Config, CypherSchema, LanguageRegistry, Prompts


# =============================================================================
# CypherSchema tests
# =============================================================================

class TestCypherSchema:
    """Verify the Cypher translation tables and helpers."""

    def test_domains_are_non_empty(self):
        assert len(CypherSchema.DOMAINS) >= 8

    def test_actions_are_non_empty(self):
        assert len(CypherSchema.ACTIONS) >= 10

    def test_patterns_are_non_empty(self):
        assert len(CypherSchema.PATTERNS) >= 5

    def test_get_all_codes_structure(self):
        codes = CypherSchema.get_all_codes()
        assert "domains" in codes
        assert "actions" in codes
        assert "patterns" in codes
        assert "SEC" in codes["domains"]
        assert "VAL" in codes["actions"]
        assert "ASY" in codes["patterns"]

    def test_format_for_llm_includes_all_sections(self):
        formatted = CypherSchema.format_for_llm()
        assert "DOMAIN CODES:" in formatted
        assert "ACTION CODES:" in formatted
        assert "PATTERN CODES:" in formatted
        assert "COMMON OBJECT CODES" in formatted

    def test_keyword_to_domain_maps_security(self):
        assert CypherSchema.KEYWORD_TO_DOMAIN["security"] == "SEC"
        assert CypherSchema.KEYWORD_TO_DOMAIN["login"] == "SEC"

    def test_keyword_to_action_maps_validate(self):
        assert CypherSchema.KEYWORD_TO_ACTION["validate"] == "VAL"
        assert CypherSchema.KEYWORD_TO_ACTION["check"] == "VAL"


# =============================================================================
# LanguageRegistry tests
# =============================================================================

class TestLanguageRegistry:
    """Verify language detection and registry completeness."""

    @pytest.mark.parametrize("ext,expected_name", [
        (".py", "Python"),
        (".js", "JavaScript"),
        (".jsx", "JavaScript"),
        (".ts", "TypeScript"),
        (".tsx", "TypeScript"),
        (".java", "Java"),
        (".go", "Go"),
        (".rs", "Rust"),
        (".c", "C"),
        (".h", "C"),
        (".cpp", "C++"),
        (".cs", "C#"),
        (".rb", "Ruby"),
        (".php", "PHP"),
        (".swift", "Swift"),
        (".kt", "Kotlin"),
    ])
    def test_detect_language_by_extension(self, ext, expected_name):
        # Fake a file path with the target extension
        lang = LanguageRegistry.detect_language(f"project/src/file{ext}")
        assert lang is not None, f"No language detected for {ext}"
        assert lang["name"] == expected_name

    def test_detect_language_unknown_extension(self):
        assert LanguageRegistry.detect_language("data.csv") is None

    def test_supported_extensions_covers_config(self):
        """Every extension in Config.TARGET_EXTENSIONS should be registered."""
        registered = LanguageRegistry.supported_extensions()
        for ext in Config.TARGET_EXTENSIONS:
            assert ext in registered, f"{ext} is in Config but not registered"

    @pytest.mark.parametrize("lang_key", [
        "python", "javascript", "typescript", "java", "go", "rust",
        "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin",
    ])
    def test_registered_language_has_required_fields(self, lang_key):
        lang = LanguageRegistry.get_language(lang_key)
        assert lang is not None, f"Language '{lang_key}' not registered"
        assert "name" in lang
        assert "comment_single" in lang
        assert "extensions" in lang
        assert "function_patterns" in lang
        assert len(lang["function_patterns"]) >= 1, f"{lang_key} has no function patterns"

    def test_python_uses_hash_comments(self):
        lang = LanguageRegistry.get_language("python")
        assert lang["comment_single"] == "#"

    def test_javascript_uses_slash_comments(self):
        lang = LanguageRegistry.get_language("javascript")
        assert lang["comment_single"] == "//"

    def test_ruby_uses_hash_comments(self):
        lang = LanguageRegistry.get_language("ruby")
        assert lang["comment_single"] == "#"


# =============================================================================
# Config tests
# =============================================================================

class TestConfig:
    """Verify Config class defaults and validation."""

    def test_target_extensions_includes_python(self):
        assert ".py" in Config.TARGET_EXTENSIONS

    def test_target_extensions_includes_javascript(self):
        assert ".js" in Config.TARGET_EXTENSIONS

    def test_target_extensions_includes_typescript(self):
        assert ".ts" in Config.TARGET_EXTENSIONS

    def test_target_extensions_includes_go(self):
        assert ".go" in Config.TARGET_EXTENSIONS

    def test_target_extensions_includes_rust(self):
        assert ".rs" in Config.TARGET_EXTENSIONS

    def test_exclude_dirs_has_common_entries(self):
        for d in ("__pycache__", ".git", "node_modules", "dist", "build"):
            assert d in Config.EXCLUDE_DIRS

    def test_cypher_version_is_set(self):
        assert Config.CYPHER_VERSION == "1.0"

    def test_search_ranking_weights_has_required_keys(self):
        required = {
            "exact_match", "domain_match", "action_match",
            "object_match", "pattern_match", "tag_match",
        }
        assert required.issubset(set(Config.SEARCH_RANKING_WEIGHTS.keys()))


class TestConfigLLMProvider:
    """Verify multi-provider configuration and validation."""

    def test_default_provider_is_anthropic(self):
        # Unless overridden by env var, default should be 'anthropic'
        assert Config.LLM_PROVIDER in ("anthropic", "openai", "gemini")

    def test_all_provider_models_have_defaults(self):
        assert isinstance(Config.ANTHROPIC_MODEL, str) and Config.ANTHROPIC_MODEL
        assert isinstance(Config.OPENAI_MODEL, str) and Config.OPENAI_MODEL
        assert isinstance(Config.GEMINI_MODEL, str) and Config.GEMINI_MODEL

    def test_shared_llm_params_exist(self):
        assert Config.LLM_MAX_TOKENS > 0
        assert isinstance(Config.LLM_TEMPERATURE, float)

    def test_get_api_key_returns_string(self):
        key = Config.get_api_key()
        assert key is not None  # set by conftest

    def test_get_model_returns_string(self):
        model = Config.get_model()
        assert isinstance(model, str) and model

    def test_validate_rejects_unknown_provider(self):
        original = Config.LLM_PROVIDER
        try:
            Config.LLM_PROVIDER = "nonexistent"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                Config.validate()
        finally:
            Config.LLM_PROVIDER = original

    def test_validate_rejects_missing_key(self):
        import os
        original_provider = Config.LLM_PROVIDER
        original_key = Config.OPENAI_API_KEY
        try:
            Config.LLM_PROVIDER = "openai"
            Config.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                Config.validate()
        finally:
            Config.LLM_PROVIDER = original_provider
            Config.OPENAI_API_KEY = original_key

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_validate_succeeds_with_key_set(self, provider):
        original = Config.LLM_PROVIDER
        try:
            Config.LLM_PROVIDER = provider
            assert Config.validate() is True
        finally:
            Config.LLM_PROVIDER = original


# =============================================================================
# Prompts tests
# =============================================================================

class TestPrompts:
    """Verify prompt templates are language-agnostic."""

    def test_generation_system_mentions_any_language(self):
        prompt = Prompts.CYPHER_GENERATION_SYSTEM
        assert "ANY programming language" in prompt

    def test_generation_system_has_multi_language_examples(self):
        prompt = Prompts.CYPHER_GENERATION_SYSTEM
        # Should mention at least Python, JavaScript, Go, Rust
        for lang in ("Python", "JavaScript", "Go", "Rust"):
            assert lang in prompt, f"Missing {lang} example in system prompt"

    def test_generation_user_has_language_placeholder(self):
        tpl = Prompts.CYPHER_GENERATION_USER
        assert "{language}" in tpl
        assert "{language_lower}" in tpl
        assert "{code}" in tpl

    def test_generation_user_renders_correctly(self):
        rendered = Prompts.CYPHER_GENERATION_USER.format(
            language="Go", language_lower="go", code="func Foo() {}"
        )
        assert "Go" in rendered
        assert "func Foo()" in rendered
