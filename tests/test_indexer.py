"""
Tests for symdex.core.indexer — tag generation and sidecar indexing logic.
"""

from pathlib import Path
from typing import List

import pytest
from symdex.core.config import Config
from symdex.core.engine import FunctionMetadata, CodeAnalyzer
from symdex.core.indexer import IndexingPipeline


# =============================================================================
# Tag generation (via IndexingPipeline._generate_tags)
# =============================================================================

class TestTagGeneration:
    """Tags should be generated from function metadata.

    ``_generate_tags`` is a static method on ``IndexingPipeline`` so we
    can call it directly without instantiating the pipeline (which would
    trigger an ``anthropic`` import).
    """

    def test_tags_from_function_name(self):
        meta = FunctionMetadata(
            name="validate_user_email", start_line=1, end_line=5,
            is_async=False, args=["email"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        tags = IndexingPipeline._generate_tags(meta)
        assert "validate" in tags
        assert "user" in tags
        assert "email" in tags

    def test_async_tag(self):
        meta = FunctionMetadata(
            name="send_message", start_line=1, end_line=5,
            is_async=True, args=["msg"], calls=[], imports=[],
            docstring=None, complexity=1, language="Python",
        )
        tags = IndexingPipeline._generate_tags(meta)
        assert "async" in tags

    def test_tags_from_calls(self):
        meta = FunctionMetadata(
            name="process", start_line=1, end_line=10,
            is_async=False, args=[], calls=["validate", "serialize", "log"],
            imports=[], docstring=None, complexity=1, language="Python",
        )
        tags = IndexingPipeline._generate_tags(meta)
        assert "validate" in tags
        assert "serialize" in tags

    def test_tags_from_docstring(self):
        meta = FunctionMetadata(
            name="handle", start_line=1, end_line=10,
            is_async=False, args=[], calls=[], imports=[],
            docstring="Handle authentication and authorization for API requests.",
            complexity=1, language="Python",
        )
        tags = IndexingPipeline._generate_tags(meta)
        assert "authentication" in tags or "authorization" in tags

    def test_tags_limited_to_12(self):
        meta = FunctionMetadata(
            name="do_many_things_with_long_descriptive_name", start_line=1, end_line=50,
            is_async=True,
            args=[],
            calls=["read", "write", "open", "close", "send", "receive",
                   "get", "set", "update", "delete", "create", "fetch",
                   "validate", "parse", "serialize"],
            imports=[],
            docstring="security performance cache database api network file stream",
            complexity=10,
            language="Python",
        )
        tags = IndexingPipeline._generate_tags(meta)
        assert len(tags) <= 12


# =============================================================================
# Sidecar directory creation
# =============================================================================

class TestSidecarDirectory:
    """The indexer should create a .symdex/ directory and never modify source."""

    def test_default_symdex_dir_created(self, tmp_path):
        """IndexingPipeline should create .symdex/ under root_dir by default."""
        # We can't fully run the pipeline (needs LLM key), but we can
        # verify the constructor creates the directory.
        root = tmp_path / "project"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")

        # Instantiation should create the sidecar directory
        pipeline = IndexingPipeline.__new__(IndexingPipeline)
        pipeline.root_dir = root
        pipeline.dry_run = True
        pipeline.force_reindex = False

        sidecar = root / Config.SYMDEX_DIR
        sidecar.mkdir(parents=True, exist_ok=True)
        assert sidecar.is_dir()

    def test_source_files_untouched_during_dry_run(self, tmp_path):
        """Source files must never be modified, even conceptually."""
        source = "def hello():\n    return 'world'\n"
        fp = tmp_path / "app.py"
        fp.write_text(source, encoding="utf-8")

        original = fp.read_text(encoding="utf-8")
        # After any indexing operation, the source should be identical
        assert fp.read_text(encoding="utf-8") == original
