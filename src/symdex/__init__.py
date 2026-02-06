"""
Symdex-100 — Inline Semantic Fingerprints for 100x Faster Code Search.

The ``symdex`` package provides language-agnostic code indexing and search
powered by LLM-generated structured metadata ("Cyphers") embedded directly
into source files.

Quick start::

    from symdex.core.config import Config
    from symdex.core.engine import CodeAnalyzer, CypherCache, CypherGenerator

Or via the CLI::

    symdex index ./src
    symdex search "where do we validate user tokens"
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
