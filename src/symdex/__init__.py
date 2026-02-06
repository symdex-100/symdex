"""
Symdex-100 — Inline Semantic Fingerprints for 100x Faster Code Search.

The ``symdex`` package provides language-agnostic code indexing and search
powered by LLM-generated structured metadata ("Cyphers") embedded directly
into source files.

Quick start (programmatic API)::

    from symdex import Symdex

    client = Symdex()                                  # reads env vars
    client.index("./src")                              # build the index
    results = client.search("validate user tokens")    # search by intent

Quick start (CLI)::

    symdex index ./src
    symdex search "where do we validate user tokens"

Configuration override::

    from symdex import Symdex, SymdexConfig

    config = SymdexConfig(llm_provider="openai", openai_api_key="sk-...")
    client = Symdex(config=config)
"""

__version__ = "1.0.0"

# Primary public API — the Symdex facade
from symdex.client import Symdex

# Configuration
from symdex.core.config import SymdexConfig

# Core data types that callers interact with
from symdex.core.engine import IndexResult, SearchResult

# Exception hierarchy
from symdex.exceptions import (
    ConfigError,
    CypherValidationError,
    IndexingError,
    IndexNotFoundError,
    ProviderError,
    SearchError,
    SymdexError,
)

__all__ = [
    "__version__",
    # Facade
    "Symdex",
    # Config
    "SymdexConfig",
    # Data types
    "IndexResult",
    "SearchResult",
    # Exceptions
    "SymdexError",
    "ConfigError",
    "ProviderError",
    "IndexNotFoundError",
    "IndexingError",
    "SearchError",
    "CypherValidationError",
]
