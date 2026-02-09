"""
Tests for symdex.core.config — SymdexConfig, CypherSchema, Config, Prompts.
"""

import pytest
from symdex.core.config import Config, CypherSchema, Prompts, SymdexConfig
from symdex.exceptions import ConfigError


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
# Config tests
# =============================================================================

class TestConfig:
    """Verify Config class defaults and validation."""

    def test_target_extensions_includes_python(self):
        assert ".py" in Config.TARGET_EXTENSIONS

    def test_exclude_dirs_has_common_entries(self):
        for d in ("__pycache__", ".git", "dist", "build"):
            assert d in Config.EXCLUDE_DIRS

    def test_cypher_version_is_set(self):
        assert Config.CYPHER_VERSION == "1.0"

    def test_search_ranking_weights_has_required_keys(self):
        required = {
            "exact_match", "domain_match", "action_match",
            "object_match", "object_semantic_match", "multi_object_match",
            "semantic_pair_match", "pattern_match", "tag_match", "name_match",
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
    """Verify prompt templates for Python."""

    def test_generation_system_mentions_python(self):
        prompt = Prompts.CYPHER_GENERATION_SYSTEM
        assert "Python" in prompt

    def test_generation_system_has_python_examples(self):
        prompt = Prompts.CYPHER_GENERATION_SYSTEM
        assert "Python:" in prompt or "python" in prompt.lower()

    def test_generation_user_has_code_placeholder(self):
        tpl = Prompts.CYPHER_GENERATION_USER
        assert "{code}" in tpl

    def test_generation_user_renders_correctly(self):
        rendered = Prompts.CYPHER_GENERATION_USER.format(
            code="def foo(): pass"
        )
        assert "Python" in rendered
        assert "def foo()" in rendered


# =============================================================================
# SymdexConfig (instance-based) tests
# =============================================================================

class TestSymdexConfig:
    """Verify SymdexConfig dataclass and its methods."""

    def test_defaults_match_config(self):
        cfg = SymdexConfig()
        assert cfg.llm_provider == "anthropic"
        assert cfg.cypher_version == "1.0"
        assert ".py" in cfg.target_extensions
        assert "__pycache__" in cfg.exclude_dirs

    def test_from_env_creates_valid_config(self):
        cfg = SymdexConfig.from_env()
        assert isinstance(cfg, SymdexConfig)
        assert cfg.llm_provider in ("anthropic", "openai", "gemini")

    def test_validate_rejects_unknown_provider(self):
        cfg = SymdexConfig(llm_provider="nonexistent")
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            cfg.validate()

    def test_validate_rejects_missing_key(self):
        cfg = SymdexConfig(llm_provider="openai", openai_api_key=None)
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            cfg.validate()

    def test_validate_succeeds_with_key(self):
        cfg = SymdexConfig(llm_provider="anthropic", anthropic_api_key="test-key")
        assert cfg.validate() is True

    def test_get_api_key_returns_correct_key(self):
        cfg = SymdexConfig(
            llm_provider="openai",
            openai_api_key="sk-test",
        )
        assert cfg.get_api_key() == "sk-test"

    def test_get_model_returns_correct_model(self):
        cfg = SymdexConfig(llm_provider="gemini", gemini_model="custom-model")
        assert cfg.get_model() == "custom-model"

    def test_get_cache_path(self):
        from pathlib import Path
        cfg = SymdexConfig(cache_db_name="myindex.db")
        result = cfg.get_cache_path(Path("/tmp/test"))
        assert result == Path("/tmp/test/myindex.db")

    def test_search_ranking_weights_are_independent(self):
        """Each SymdexConfig instance should have its own weights dict."""
        cfg1 = SymdexConfig()
        cfg2 = SymdexConfig()
        cfg1.search_ranking_weights["exact_match"] = 999.0
        assert cfg2.search_ranking_weights["exact_match"] == 12.0  # Current default


class TestConfigToInstance:
    """Verify Config.to_instance() snapshots global state correctly."""

    def test_to_instance_returns_symdex_config(self):
        cfg = Config.to_instance()
        assert isinstance(cfg, SymdexConfig)

    def test_to_instance_captures_provider(self):
        original = Config.LLM_PROVIDER
        try:
            Config.LLM_PROVIDER = "gemini"
            cfg = Config.to_instance()
            assert cfg.llm_provider == "gemini"
        finally:
            Config.LLM_PROVIDER = original

    def test_to_instance_captures_model(self):
        cfg = Config.to_instance()
        assert cfg.anthropic_model == Config.ANTHROPIC_MODEL
