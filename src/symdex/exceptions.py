"""
Symdex-100 Exception Hierarchy

Structured exceptions for clear error handling across CLI, API, and MCP
consumers.  Each exception type maps to a specific failure mode so that
callers can handle errors precisely without parsing message strings.

Usage::

    from symdex.exceptions import SymdexError, IndexNotFoundError

    try:
        results = client.search("validate user")
    except IndexNotFoundError:
        print("Run 'symdex index' first.")
    except SymdexError as exc:
        print(f"Symdex error: {exc}")
"""


class SymdexError(Exception):
    """Base exception for all Symdex-100 errors."""


class ConfigError(SymdexError, ValueError):
    """Configuration is invalid or incomplete (e.g. missing API key).

    Inherits from ``ValueError`` for backward compatibility with code
    that already catches ``ValueError`` from ``Config.validate()``.
    """


class ProviderError(SymdexError):
    """LLM provider failure — API error, authentication, or rate limiting."""


class IndexNotFoundError(SymdexError, FileNotFoundError):
    """No Symdex index exists at the expected path.

    Inherits from ``FileNotFoundError`` for intuitive exception handling.
    """


class IndexingError(SymdexError):
    """Fatal error during the indexing pipeline."""


class SearchError(SymdexError):
    """Error during search execution."""


class CypherValidationError(SymdexError):
    """A generated or parsed Cypher string does not conform to the schema."""
