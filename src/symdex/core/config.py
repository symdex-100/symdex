"""
Symdex-100 Configuration Module

Centralized configuration for the Symdex indexing and search system.
The Cypher string format (DOM:ACT_OBJ--PAT) is the internal fingerprint
notation — "Symdex" is the product, "Cypher" is the fingerprint.

Production-ready with validation and security considerations.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# Instance-Based Configuration (preferred for new code)
# =============================================================================

@dataclass
class SymdexConfig:
    """
    Instance-based configuration for Symdex-100.

    Unlike the legacy :class:`Config` (global class attributes), each
    ``SymdexConfig`` instance is self-contained and can be passed through
    the call stack — enabling multi-tenant usage, testing, and embedding
    into third-party products.

    Create from environment variables::

        config = SymdexConfig.from_env()

    Or with explicit values::

        config = SymdexConfig(llm_provider="openai", openai_api_key="sk-...")
    """

    # ── LLM Provider ──────────────────────────────────────────────
    llm_provider: str = "anthropic"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    llm_max_tokens: int = 300
    llm_temperature: float = 0.0

    # ── Rate Limiting ─────────────────────────────────────────────
    max_requests_per_minute: int = 50
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_backoff_base: float = 2.0

    # ── File Processing ───────────────────────────────────────────
    target_extensions: frozenset = frozenset((".py",))
    exclude_dirs: frozenset = frozenset((
        "__pycache__", ".git", ".venv", "venv",
        ".pytest_cache", "dist", "build", ".symdex",
    ))
    max_file_size_mb: int = 5

    # ── Cypher Schema ─────────────────────────────────────────────
    cypher_version: str = "1.0"

    # ── Sidecar Index ─────────────────────────────────────────────
    symdex_dir: str = ".symdex"
    cache_db_name: str = "index.db"
    cache_expiry_days: int = 30

    # ── Cypher generation (skip LLM) ──────────────────────────────
    cypher_fallback_only: bool = False
    """If True, never call the LLM; use rule-based Cypher and query translation only."""

    # ── Search ────────────────────────────────────────────────────
    max_search_results: int = 5
    max_search_candidates: int = 200  # Cap merged candidates before scoring (0 = no cap)
    min_search_score: float = 5.0
    stop_words: frozenset = frozenset({
        "i", "a", "the", "is", "it", "do", "we", "my", "me", "an", "in",
        "to", "for", "of", "and", "or", "on", "at", "by", "with", "from",
        "that", "this", "where", "what", "how", "which", "show", "find",
        "search", "look", "give", "list", "get", "see", "main", "function",
        "code", "define", "does", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "can", "could", "should", "would",
    })
    # Weights: ACT and OBJ dominate (what it does, on what), then DOM, then PAT.
    search_ranking_weights: dict = field(default_factory=lambda: {
        "exact_match": 10.0,
        "domain_match": 4.0,
        "action_match": 6.0,
        "object_match": 5.0,
        "object_similarity": 2.0,
        "pattern_match": 2.0,
        "domain_mismatch_penalty": -3.0,
        "tag_match": 1.5,
        "name_match": 3.0,
    })

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "SymdexConfig":
        """Build a config snapshot from current environment variables.

        Reads :envvar:`SYMDEX_CYPHER_FALLBACK_ONLY` (1/true/yes) to enable
        rule-only mode without an LLM API key.
        """
        fallback_raw = os.getenv("SYMDEX_CYPHER_FALLBACK_ONLY", "").lower()
        cypher_fallback_only = fallback_raw in ("1", "true", "yes", "on")
        return cls(
            llm_provider=os.getenv("SYMDEX_LLM_PROVIDER", "anthropic").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            llm_max_tokens=int(os.getenv("SYMDEX_MAX_TOKENS", "300")),
            min_search_score=float(os.getenv("CYPHER_MIN_SCORE", "5.0")),
            log_level=os.getenv("CYPHER_LOG_LEVEL", "INFO"),
            cypher_fallback_only=cypher_fallback_only,
        )

    # ── Validation & Accessors ────────────────────────────────────

    def validate(self) -> bool:
        """
        Validate that the active LLM provider has an API key set.

        When :attr:`cypher_fallback_only` is True, skips key check (no LLM used).
        Raises :class:`~symdex.exceptions.ConfigError` on failure.
        """
        from symdex.exceptions import ConfigError

        if getattr(self, "cypher_fallback_only", False):
            return True

        key_map = {
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "openai":    ("OPENAI_API_KEY",    self.openai_api_key),
            "gemini":    ("GEMINI_API_KEY",    self.gemini_api_key),
        }

        if self.llm_provider not in key_map:
            raise ConfigError(
                f"Unknown LLM provider '{self.llm_provider}'. "
                f"Supported: {', '.join(key_map.keys())}.\n"
                "  Set via: export SYMDEX_LLM_PROVIDER=anthropic"
            )

        env_name, value = key_map[self.llm_provider]
        if not value:
            raise ConfigError(
                f"{env_name} not found (required by provider '{self.llm_provider}').\n"
                f"  Windows (PowerShell): $env:{env_name}='your-key-here'\n"
                f"  Linux/Mac: export {env_name}='your-key-here'"
            )
        return True

    def get_api_key(self) -> str:
        """Return the API key for the currently active provider."""
        return {
            "anthropic": self.anthropic_api_key,
            "openai":    self.openai_api_key,
            "gemini":    self.gemini_api_key,
        }[self.llm_provider]

    def get_model(self) -> str:
        """Return the model name for the currently active provider."""
        return {
            "anthropic": self.anthropic_model,
            "openai":    self.openai_model,
            "gemini":    self.gemini_model,
        }[self.llm_provider]

    def get_cache_path(self, base_dir: Path) -> Path:
        """Get the path to the cache database."""
        return base_dir / self.cache_db_name


# =============================================================================
# Legacy Global Configuration (backward-compatible)
# =============================================================================

class Config:
    """Central configuration class for the Symdex-100 system."""

    # ── LLM Provider Selection ───────────────────────────────────
    # Supported values: "anthropic", "openai", "gemini"
    # Override via environment:
    #   export SYMDEX_LLM_PROVIDER=openai
    LLM_PROVIDER: str = os.getenv("SYMDEX_LLM_PROVIDER", "anthropic").lower()

    # ── Anthropic Configuration ──────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # ── OpenAI Configuration ─────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Google Gemini Configuration ──────────────────────────────
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Shared LLM Parameters ────────────────────────────────────
    LLM_MAX_TOKENS: int = int(os.getenv("SYMDEX_MAX_TOKENS", "300"))
    LLM_TEMPERATURE: float = 0.0  # Deterministic for reproducibility

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 50
    MAX_CONCURRENT_REQUESTS: int = 5
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE: float = 2.0
    
    # File Processing — Python only (v1)
    # frozenset for O(1) membership tests during directory scanning
    TARGET_EXTENSIONS: frozenset = frozenset((".py",))
    EXCLUDE_DIRS: frozenset = frozenset((
        "__pycache__", ".git", ".venv", "venv",
        ".pytest_cache", "dist", "build",
        ".symdex",  # Symdex sidecar index directory
    ))
    MAX_FILE_SIZE_MB: int = 5
    
    # Cypher Schema Version
    CYPHER_VERSION: str = "1.0"
    
    # Sidecar Index Directory
    # All metadata is stored in this hidden directory; source files are
    # never modified.  The directory is created automatically next to the
    # indexed root.
    SYMDEX_DIR: str = ".symdex"
    
    # Cache Configuration
    CACHE_DB_NAME: str = "index.db"
    CACHE_EXPIRY_DAYS: int = 30
    
    # Search Configuration
    MAX_SEARCH_RESULTS: int = 5
    MAX_SEARCH_CANDIDATES: int = 200
    MIN_SEARCH_SCORE: float = float(os.getenv("CYPHER_MIN_SCORE", "5.0"))
    
    # Stop words for query/tag/name matching (centralized to avoid duplication)
    STOP_WORDS: frozenset = frozenset({
        "i", "a", "the", "is", "it", "do", "we", "my", "me", "an", "in",
        "to", "for", "of", "and", "or", "on", "at", "by", "with", "from",
        "that", "this", "where", "what", "how", "which", "show", "find",
        "search", "look", "give", "list", "get", "see", "main", "function",
        "code", "define", "does", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "can", "could", "should", "would",
    })
    
    SEARCH_RANKING_WEIGHTS: dict = {
        "exact_match": 10.0,
        "domain_match": 4.0,
        "action_match": 6.0,
        "object_match": 5.0,
        "object_similarity": 2.0,
        "pattern_match": 2.0,
        "domain_mismatch_penalty": -3.0,
        "tag_match": 1.5,
        "name_match": 3.0,
    }
    
    # Logging
    LOG_LEVEL: str = os.getenv("CYPHER_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical configuration for the active LLM provider.
        
        Checks that the API key for the selected provider
        (``SYMDEX_LLM_PROVIDER``) is set.
        """
        provider = cls.LLM_PROVIDER
        key_map = {
            "anthropic": ("ANTHROPIC_API_KEY", cls.ANTHROPIC_API_KEY),
            "openai":    ("OPENAI_API_KEY",    cls.OPENAI_API_KEY),
            "gemini":    ("GEMINI_API_KEY",    cls.GEMINI_API_KEY),
        }

        if provider not in key_map:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Supported: {', '.join(key_map.keys())}.\n"
                "  Set via: export SYMDEX_LLM_PROVIDER=anthropic"
            )

        env_name, value = key_map[provider]
        if not value:
            raise ValueError(
                f"{env_name} not found (required by provider '{provider}').\n"
                f"  Windows (PowerShell): $env:{env_name}='your-key-here'\n"
                f"  Linux/Mac: export {env_name}='your-key-here'"
            )
        return True

    @classmethod
    def get_api_key(cls) -> str:
        """Return the API key for the currently active provider."""
        return {
            "anthropic": cls.ANTHROPIC_API_KEY,
            "openai":    cls.OPENAI_API_KEY,
            "gemini":    cls.GEMINI_API_KEY,
        }[cls.LLM_PROVIDER]

    @classmethod
    def get_model(cls) -> str:
        """Return the model name for the currently active provider."""
        return {
            "anthropic": cls.ANTHROPIC_MODEL,
            "openai":    cls.OPENAI_MODEL,
            "gemini":    cls.GEMINI_MODEL,
        }[cls.LLM_PROVIDER]
    
    @classmethod
    def get_cache_path(cls, base_dir: Path) -> Path:
        """Get the path to the cache database."""
        return base_dir / cls.CACHE_DB_NAME

    @classmethod
    def to_instance(cls) -> "SymdexConfig":
        """Snapshot current global Config state into a SymdexConfig instance.

        Useful when you need to capture the current (possibly mutated)
        global state and pass it to instance-based APIs.
        """
        return SymdexConfig(
            llm_provider=cls.LLM_PROVIDER,
            anthropic_api_key=cls.ANTHROPIC_API_KEY,
            anthropic_model=cls.ANTHROPIC_MODEL,
            openai_api_key=cls.OPENAI_API_KEY,
            openai_model=cls.OPENAI_MODEL,
            gemini_api_key=cls.GEMINI_API_KEY,
            gemini_model=cls.GEMINI_MODEL,
            llm_max_tokens=cls.LLM_MAX_TOKENS,
            llm_temperature=cls.LLM_TEMPERATURE,
            max_requests_per_minute=cls.MAX_REQUESTS_PER_MINUTE,
            max_concurrent_requests=cls.MAX_CONCURRENT_REQUESTS,
            retry_attempts=cls.RETRY_ATTEMPTS,
            retry_backoff_base=cls.RETRY_BACKOFF_BASE,
            target_extensions=cls.TARGET_EXTENSIONS,
            exclude_dirs=cls.EXCLUDE_DIRS,
            max_file_size_mb=cls.MAX_FILE_SIZE_MB,
            cypher_version=cls.CYPHER_VERSION,
            symdex_dir=cls.SYMDEX_DIR,
            cache_db_name=cls.CACHE_DB_NAME,
            cache_expiry_days=cls.CACHE_EXPIRY_DAYS,
            max_search_results=cls.MAX_SEARCH_RESULTS,
            max_search_candidates=cls.MAX_SEARCH_CANDIDATES,
            min_search_score=cls.MIN_SEARCH_SCORE,
            stop_words=cls.STOP_WORDS,
            search_ranking_weights=cls.SEARCH_RANKING_WEIGHTS.copy(),
            log_level=cls.LOG_LEVEL,
            log_format=cls.LOG_FORMAT,
        )


# =============================================================================
# Translation Tables (The "Cypher-100 Dictionary")
# =============================================================================

class CypherSchema:
    """
    The standardized translation table for Cypher-100.
    These dictionaries ensure reproducibility across LLM calls.
    """
    
    # Domain codes and their descriptions
    DOMAINS = {
        "SEC": "Security / Authentication / Encryption",
        "DAT": "Data Processing / Storage / CRUD",
        "NET": "Networking / API / External Communication",
        "SYS": "OS / Filesystem / Memory / System",
        "LOG": "Logging / Observability / Monitoring",
        "UI":  "User Interface / Rendering / Display",
        "BIZ": "Business Logic / Domain Rules",
        "TST": "Testing / Validation / Quality Assurance"
    }
    
    # Action codes and their descriptions
    ACTIONS = {
        "VAL": "Validate / Check / Verify",
        "TRN": "Transform / Map / Convert",
        "SND": "Send / Dispatch / Notify",
        "FET": "Fetch / Retrieve / Read",
        "SCR": "Scrub / Redact / Delete / Clean",
        "CRT": "Create / Initialize / Build",
        "UPD": "Update / Modify / Patch",
        "AGG": "Aggregate / Combine / Merge",
        "FLT": "Filter / Select / Query",
        "SYN": "Synchronize / Coordinate"
    }
    
    # Pattern codes and their descriptions
    PATTERNS = {
        "ASY": "Asynchronous (async/await)",
        "SYN": "Synchronous (blocking)",
        "REC": "Recursive (self-calling)",
        "GEN": "Generator / Stream (yield)",
        "DEC": "Decorator / Wrapper",
        "CTX": "Context Manager (with statement)",
        "CLS": "Class Method / Static Method"
    }
    
    # Natural language keywords mapped to Cypher components
    KEYWORD_TO_DOMAIN = {
        "security": "SEC", "auth": "SEC", "login": "SEC", "encrypt": "SEC",
        "password": "SEC", "token": "SEC", "permission": "SEC",
        "data": "DAT", "database": "DAT", "sql": "DAT", "store": "DAT",
        "save": "DAT", "load": "DAT", "parse": "DAT",
        "network": "NET", "api": "NET", "request": "NET", "http": "NET",
        "fetch": "NET", "download": "NET", "upload": "NET",
        "file": "SYS", "filesystem": "SYS", "process": "SYS", "thread": "SYS",
        "log": "LOG", "debug": "LOG", "error": "LOG", "monitor": "LOG",
        "ui": "UI", "render": "UI", "display": "UI", "view": "UI",
        "business": "BIZ", "domain": "BIZ", "rule": "BIZ",
        "test": "TST", "validate": "TST", "check": "TST"
    }
    
    KEYWORD_TO_ACTION = {
        "validate": "VAL", "verify": "VAL", "check": "VAL",
        "transform": "TRN", "convert": "TRN", "map": "TRN", "serialize": "TRN",
        "send": "SND", "notify": "SND", "dispatch": "SND", "emit": "SND",
        "fetch": "FET", "get": "FET", "retrieve": "FET", "load": "FET",
        "delete": "SCR", "remove": "SCR", "clean": "SCR", "redact": "SCR",
        "create": "CRT", "initialize": "CRT", "build": "CRT",
        "update": "UPD", "modify": "UPD", "edit": "UPD", "patch": "UPD",
        "aggregate": "AGG", "combine": "AGG", "merge": "AGG", "sum": "AGG",
        "filter": "FLT", "select": "FLT", "query": "FLT", "search": "FLT",
        "sync": "SYN", "coordinate": "SYN"
    }
    
    KEYWORD_TO_PATTERN = {
        "async": "ASY", "await": "ASY", "asynchronous": "ASY",
        "recursive": "REC", "recurse": "REC",
        "generator": "GEN", "yield": "GEN", "stream": "GEN",
        "decorator": "DEC", "wrapper": "DEC",
        "context": "CTX", "with": "CTX"
    }

    # Common / preferred OBJ codes used across domains.
    # This list is what we expect the LLM to choose from
    # for at least ~95% of typical application objects.
    COMMON_OBJECT_CODES = [
        # Data structures & collections
        "DATA", "DATASET", "ROW", "ROWS", "COL", "COLS", "TABLE", "RECORD",
        "LIST", "DICT", "MAP", "QUEUE", "STACK", "BATCH", "WINDOW",
        # Users, auth, security
        "USER", "USERS", "ACCOUNT", "SESSION", "TOKEN", "SECRET", "PASS", "PWD",
        "PERM", "ROLE", "AUTH",
        # Files, paths, storage
        "FILE", "FILES", "PATH", "DIR", "FOLDER", "TMP", "CACHE", "CACH",
        "CONFIG", "CFG", "SETTINGS", "ENV",
        # Network, HTTP, API
        "REQUEST", "RESPONSE", "HTTPREQ", "HTTPRESP", "URL", "URI", "HEADER",
        "BODY", "PAYLOAD",
        # Formats & content
        "JSON", "CSV", "XML", "HTML", "TEXT", "BINARY", "IMAGE", "AUDIO", "VIDEO",
        # Logging & observability
        "LOG", "LOGS", "METRIC", "EVENT", "TRACE",
        # Models & ML
        "MODEL", "EMBED", "VECTOR", "FEATURE", "LABEL",
        # Misc domain objects
        "NODE", "EDGE", "GRAPH", "JOB", "TASK", "MSG", "MESSAGE",
        "ORDER", "ITEM", "PRODUCT",
        # Relationship / association objects
        "RELATIONSHIP", "RELATIONSHIPS", "LINK", "REF",
        # Advisory / decision objects
        "RECOMMENDATION", "RECOMMENDATIONS", "SUGGESTION", "ALERT",
    ]
    
    @classmethod
    def get_all_codes(cls) -> dict:
        """Return all valid codes for validation."""
        return {
            "domains": list(cls.DOMAINS.keys()),
            "actions": list(cls.ACTIONS.keys()),
            "patterns": list(cls.PATTERNS.keys())
        }
    
    @classmethod
    def format_for_llm(cls) -> str:
        """Format the schema as a prompt for the LLM."""
        return f"""
DOMAIN CODES:
{chr(10).join(f"- {code}: {desc}" for code, desc in cls.DOMAINS.items())}

ACTION CODES:
{chr(10).join(f"- {code}: {desc}" for code, desc in cls.ACTIONS.items())}

PATTERN CODES:
{chr(10).join(f"- {code}: {desc}" for code, desc in cls.PATTERNS.items())}

COMMON OBJECT CODES (preferred OBJ tokens):
{chr(10).join(f"- {code}" for code in cls.COMMON_OBJECT_CODES)}
"""


# =============================================================================
# System Prompts for LLM
# =============================================================================

class Prompts:
    """Standardized prompts for consistent LLM behavior."""
    
    CYPHER_GENERATION_SYSTEM = f"""You are a Cypher-100 code classifier. Your task is to analyze Python functions and generate a standardized metadata string called a "Cypher".

The Cypher format is: DOM:ACT_OBJ--PAT

Where:
- DOM (Domain, 2-3 uppercase letters): Pick from {list(CypherSchema.DOMAINS.keys())}
- ACT (Action, 3 uppercase letters): Pick from {list(CypherSchema.ACTIONS.keys())}
- OBJ (Object, 2-20 uppercase letters/digits): Choose a token from COMMON OBJECT CODES whenever possible (e.g. DATASET, USER, REQUEST, JSON, CSV, CACHE, RELATIONSHIPS, RECOMMENDATIONS). Prefer full, readable words like DATASET over cryptic abbreviations like DSET.
- PAT (Pattern, 3 uppercase letters): Pick from {list(CypherSchema.PATTERNS.keys())}

{CypherSchema.format_for_llm()}

RULES:
1. Choose exactly ONE code from each category (DOM, ACT, PAT must be from the lists above)
2. OBJ is 2-20 uppercase letters or digits. When the object matches a COMMON OBJECT CODE, use that exact token (e.g. DATASET, USER, REQUEST, RELATIONSHIPS, RECOMMENDATIONS). Avoid inventing new abbreviations (do NOT use DSET if DATASET is available).
3. If uncertain, choose the most specific applicable code
4. Output ONLY the Cypher string, nothing else — no explanations, no caveats
5. Be consistent: same code logic should always produce the same Cypher
6. **CRITICAL — Non-classifiable code:** If the code is NOT a complete function or method — for example it is a code fragment, a conditional branch (if/else), a loop body, a variable declaration, a single statement, or any incomplete/unidentifiable snippet — respond with exactly the word: SKIP
   Do NOT explain why. Do NOT describe what is wrong. Just output: SKIP

EXAMPLES:
- Python: "async def send_email(to, subject): ..." → NET:SND_EMAL--ASY
- Python: "def validate_password(pwd): ..." → SEC:VAL_PASS--SYN
- Python: "def fetch_user_data(user_id): ..." → DAT:FET_USER--SYN
- Python: "async def scrub_sensitive_logs(stream): ..." → LOG:SCR_LOGS--ASY
- Python: "def transform_csv_to_json(csv_data): ..." → DAT:TRN_JSON--SYN
"""
    
    CYPHER_GENERATION_USER = """Analyze this Python function and generate its Cypher:

```python
{code}
```

Output only the Cypher string in format DOM:ACT_OBJ--PAT"""
    
    QUERY_TRANSLATION_SYSTEM = f"""You are a natural language to Cypher-100 query translator.

The user is SEARCHING for code. Your job is to describe WHAT THE TARGET CODE DOES, not the user's act of searching.

IMPORTANT: Ignore verbs like "find", "search", "show me", "where is", "look for" — they describe the USER's intent to locate code. Instead, focus on the SUBJECT of the search: what the target code creates, fetches, validates, transforms, etc.

{CypherSchema.format_for_llm()}

Output exactly THREE Cypher patterns, one per line, in order of precision:
1. TIGHT: No wildcards (or at most one * only where the query is truly ambiguous). Use specific DOM, ACT, OBJ, PAT from the schema.
2. MEDIUM: One or two wildcards where reasonable (e.g. specific ACT_OBJ but PAT=* or DOM=*).
3. BROAD: Fallback with wildcards (e.g. *:ACT_*--* or DOM:*_*--*) to catch related code.

RULES:
- For OBJ, use COMMON OBJECT CODES that match the query noun (e.g. "dependencies" → DEPS or DEPENDENCY, "dataset" → DATASET).
- Prefer specific ACT when the query has a clear verb (e.g. "analyze" → AGG or FLT, "validate" → VAL).
- When the query mentions "setup", "configure", "initialize" → ACT = CRT.
- When the query mentions "logging", "log" as main subject → DOM = LOG, OBJ = LOGS.
- Output ONLY three lines, each line is a single Cypher in format DOM:ACT_OBJ--PAT. No numbering, no explanation.

EXAMPLES (each block is one query → three lines):
"where do we validate users"
SEC:VAL_USER--SYN
SEC:VAL_USER--*
*:VAL_USER--*

"where does the AI model analyze for dependencies"
BIZ:AGG_DEPS--SYN
BIZ:AGG_DEPS--*
*:AGG_*--*

"find async email functions"
NET:SND_EMAL--ASY
NET:SND_EMAL--*
*:SND_*--ASY
"""
    
    QUERY_TRANSLATION_USER = """The user wants to FIND code. Output exactly THREE Cypher patterns (tight, medium, broad), one per line.

User's search query: "{query}"

Output only three Cypher strings, one per line, in format DOM:ACT_OBJ--PAT. Line 1 = tight (no/minimal wildcards), Line 2 = medium, Line 3 = broad."""


# NOTE: Import-time validation removed (v1.1).  Callers that need
# validation should call ``Config.validate()`` or
# ``SymdexConfig.from_env().validate()`` explicitly.
