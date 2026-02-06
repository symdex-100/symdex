"""
Symdex-100 Configuration Module

Centralized configuration for the Symdex indexing and search system.
The Cypher string format (DOM:ACT_OBJ--PAT) is the internal fingerprint
notation — "Symdex" is the product, "Cypher" is the fingerprint.

Production-ready with validation and security considerations.
"""

import os
from pathlib import Path
from typing import Optional

# =============================================================================
# API Configuration
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
    
    # File Processing — all supported languages
    # frozenset for O(1) membership tests during directory scanning
    TARGET_EXTENSIONS: frozenset = frozenset((
        ".py",                          # Python
        ".js", ".jsx", ".mjs",          # JavaScript
        ".ts", ".tsx",                  # TypeScript
        ".java",                        # Java
        ".go",                          # Go
        ".rs",                          # Rust
        ".c", ".h",                     # C
        ".cpp", ".hpp", ".cc", ".cxx",  # C++
        ".cs",                          # C#
        ".rb",                          # Ruby
        ".php",                         # PHP
        ".swift",                       # Swift
        ".kt", ".kts",                  # Kotlin
    ))
    EXCLUDE_DIRS: frozenset = frozenset((
        "__pycache__", ".git", ".venv", "venv",
        "node_modules", ".pytest_cache", "dist", "build",
        "target", "bin", "obj", ".gradle", ".idea",
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
    MIN_SEARCH_SCORE: float = float(os.getenv("CYPHER_MIN_SCORE", "7.0"))
    SEARCH_RANKING_WEIGHTS: dict = {
        "exact_match": 10.0,
        "domain_match": 5.0,
        "action_match": 5.0,
        "object_match": 3.0,
        "object_similarity": 2.0,
        "pattern_match": 2.0,
        "tag_match": 1.5,
        "name_match": 2.0,      # Boost when query words appear in function name
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
# Language Registry — maps file extensions to language metadata
# =============================================================================

class LanguageRegistry:
    """
    Central registry of supported programming languages.
    
    Provides comment styles, function-detection regex patterns, and
    brace-counting hints for each language so the rest of the system
    stays language-agnostic.
    """

    # Each language entry contains:
    #   name            – human-readable language name
    #   comment_single  – single-line comment prefix (used for SEARCH_META blocks)
    #   comment_block   – (open, close) for block comments, or None
    #   extensions      – set of file extensions that map to this language
    #   function_patterns – list of compiled regex patterns that match function
    #                       definitions.  Each pattern MUST define named groups:
    #                         - 'name'   (the function/method name)
    #                         - 'args'   (the argument list, raw text)
    #                       And may optionally define:
    #                         - 'async'  (if present, the function is async)
    #   uses_braces     – whether the language uses { } to delimit bodies
    #   uses_indent     – whether the language uses indentation (Python)

    _LANGUAGES: dict = {}  # populated by register() calls below

    @classmethod
    def register(cls, key: str, *, name: str, comment_single: str,
                 comment_block: tuple = None, extensions: tuple = (),
                 function_patterns: list = None, uses_braces: bool = True,
                 uses_indent: bool = False):
        """Register a language definition."""
        import re as _re
        compiled_patterns = []
        for pat_str in (function_patterns or []):
            compiled_patterns.append(_re.compile(pat_str, _re.MULTILINE))

        cls._LANGUAGES[key] = {
            "name": name,
            "comment_single": comment_single,
            "comment_block": comment_block,
            "extensions": set(extensions),
            "function_patterns": compiled_patterns,
            "uses_braces": uses_braces,
            "uses_indent": uses_indent,
        }

    @classmethod
    def detect_language(cls, file_path) -> dict | None:
        """
        Detect the language for a given file path based on its extension.
        
        Returns the language dict or None if unsupported.
        """
        ext = str(file_path).rsplit(".", 1)[-1] if "." in str(file_path) else ""
        ext = f".{ext}"
        for lang in cls._LANGUAGES.values():
            if ext in lang["extensions"]:
                return lang
        return None

    @classmethod
    def get_language(cls, key: str) -> dict | None:
        """Retrieve a language definition by key."""
        return cls._LANGUAGES.get(key)

    @classmethod
    def supported_extensions(cls) -> set:
        """Return the union of all registered extensions."""
        exts: set = set()
        for lang in cls._LANGUAGES.values():
            exts |= lang["extensions"]
        return exts


# ── Register all supported languages ─────────────────────────────────────────

# Python
LanguageRegistry.register(
    "python",
    name="Python",
    comment_single="#",
    comment_block=('"""', '"""'),
    extensions=(".py",),
    uses_braces=False,
    uses_indent=True,
    function_patterns=[
        # async def / def  — captures name, args, optional async keyword
        r'(?P<async>async\s+)?def\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# JavaScript / TypeScript — keyword guard for class-method patterns.
# Prevents control-flow statements like `if(cond) {`, `for(…) {`,
# `while(cond) {` from being misidentified as method definitions.
_JS_KEYWORD_GUARD = (
    r'(?!if\b|else\b|for\b|while\b|do\b|switch\b|catch\b|return\b|throw\b'
    r'|new\b|delete\b|typeof\b|instanceof\b|void\b|await\b|case\b)'
)

# JavaScript
LanguageRegistry.register(
    "javascript",
    name="JavaScript",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".js", ".jsx", ".mjs"),
    function_patterns=[
        # function declarations: [async] [export] function name(args)
        r'(?:export\s+)?(?P<async>async\s+)?function\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
        # arrow / const fn: [export] const name = [async] (args) =>
        r'(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?P<async>async\s+)?\((?P<args>[^)]*)\)\s*=>',
        # class method: [async] name(args) {  — with keyword guard
        r'^\s+(?P<async>async\s+)?' + _JS_KEYWORD_GUARD +
        r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*\{',
    ],
)

# TypeScript
LanguageRegistry.register(
    "typescript",
    name="TypeScript",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".ts", ".tsx"),
    function_patterns=[
        # function declarations with optional type params
        r'(?:export\s+)?(?P<async>async\s+)?function\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<args>[^)]*)\)',
        # arrow / const fn with optional type
        r'(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=]+)?\s*=\s*(?P<async>async\s+)?\((?P<args>[^)]*)\)\s*(?::\s*\w+)?\s*=>',
        # class method — with keyword guard
        # Return type slot accepts simple types (void) and generics (Promise<void>)
        r'^\s+(?:public|private|protected)?\s*(?:static\s+)?' +
        r'(?P<async>async\s+)?' + _JS_KEYWORD_GUARD +
        r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*(?::\s*\w+(?:<[^>]*>)?)?\s*\{',
    ],
)

# Java
LanguageRegistry.register(
    "java",
    name="Java",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".java",),
    function_patterns=[
        # [modifiers] ReturnType name(args) { — excludes constructors-like matches via return type
        r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]*>)?)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# Go
LanguageRegistry.register(
    "go",
    name="Go",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".go",),
    function_patterns=[
        # func (receiver) name(args) [returns] {
        r'func\s+(?:\([^)]*\)\s+)?(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# Rust
LanguageRegistry.register(
    "rust",
    name="Rust",
    comment_single="//",
    comment_block=None,
    extensions=(".rs",),
    function_patterns=[
        # [pub] [async] fn name(args) [-> Type] {
        r'(?:pub\s+)?(?P<async>async\s+)?fn\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<args>[^)]*)\)',
    ],
)

# C
# Keyword guard prevents matching `if(cond) {`, `for(...)`, etc. as functions.
_C_KEYWORD_GUARD = (
    r'(?!if\b|else\b|for\b|while\b|do\b|switch\b|return\b|sizeof\b'
    r'|typeof\b|goto\b|case\b)'
)
LanguageRegistry.register(
    "c",
    name="C",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".c", ".h"),
    function_patterns=[
        # ReturnType [*] name(args) {  — top-level definitions
        r'^(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?'
        r'\w+[\s*]+'
        + _C_KEYWORD_GUARD +
        r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*\{',
    ],
)

# C++
# Negative lookahead excludes C++ keywords that look like function calls
# (e.g. `else if(cond) {` where `else` is parsed as return type and `if`
# as function name).  Pattern 2 no longer matches `;` so variable
# declarations like `std::lock_guard lock(m)` are excluded.
_CPP_KEYWORD_GUARD = (
    r'(?!if\b|else\b|for\b|while\b|do\b|switch\b|catch\b|return\b|throw\b'
    r'|sizeof\b|alignof\b|decltype\b|typeid\b|static_cast\b|dynamic_cast\b'
    r'|const_cast\b|reinterpret_cast\b|new\b|delete\b|case\b|goto\b)'
)
LanguageRegistry.register(
    "cpp",
    name="C++",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".cpp", ".hpp", ".cc", ".cxx"),
    function_patterns=[
        # Pattern 1: Standard function/method definition (brace on same line)
        # ReturnType [Class::]name(args) [const] [override] [noexcept] {
        r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:const\s+)?'
        r'[\w:]+(?:<[^>]*>)?[\s*&]+(?:[\w:]+::)?'
        + _CPP_KEYWORD_GUARD +
        r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*'
        r'(?:const\s*)?(?:override\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?\{',
        # Pattern 2: Function with brace on next line + attribute / trailing return
        # Does NOT match `;` (declarations / variable inits are excluded)
        r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:const\s+)?'
        r'(?:\[\[[^\]]+\]\]\s+)?[\w:]+(?:<[^>]*>)?[\s*&]+'
        r'(?:[\w:]+::)?'
        + _CPP_KEYWORD_GUARD +
        r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*'
        r'(?:const\s*)?(?:\s*->\s*[\w:]+(?:<[^>]*>)?)?\s*\{',
    ],
)

# C#
LanguageRegistry.register(
    "csharp",
    name="C#",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".cs",),
    function_patterns=[
        # [modifiers] ReturnType name(args) {
        r'(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:async\s+)?(?:\w+(?:<[^>]*>)?)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# Ruby
LanguageRegistry.register(
    "ruby",
    name="Ruby",
    comment_single="#",
    comment_block=("=begin", "=end"),
    extensions=(".rb",),
    uses_braces=False,
    uses_indent=False,  # uses end keyword
    function_patterns=[
        # def [self.]name(args)
        r'def\s+(?:self\.)?(?P<name>\w+[?!=]?)\s*(?:\((?P<args>[^)]*)\))?',
    ],
)

# PHP
LanguageRegistry.register(
    "php",
    name="PHP",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".php",),
    function_patterns=[
        # [modifiers] function name(args)
        r'(?:public|private|protected)?\s*(?:static\s+)?function\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# Swift
LanguageRegistry.register(
    "swift",
    name="Swift",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".swift",),
    function_patterns=[
        # [modifiers] func name(args) [-> Type] {
        r'(?:public|private|internal|fileprivate|open)?\s*(?:static\s+)?(?:class\s+)?func\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)

# Kotlin
LanguageRegistry.register(
    "kotlin",
    name="Kotlin",
    comment_single="//",
    comment_block=("/*", "*/"),
    extensions=(".kt", ".kts"),
    function_patterns=[
        # [modifiers] fun name(args) [: Type] {
        r'(?:public|private|protected|internal)?\s*(?:suspend\s+)?fun\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)',
    ],
)


# =============================================================================
# System Prompts for LLM
# =============================================================================

class Prompts:
    """Standardized prompts for consistent LLM behavior."""
    
    CYPHER_GENERATION_SYSTEM = f"""You are a Cypher-100 code classifier. Your task is to analyze functions/methods in ANY programming language and generate a standardized metadata string called a "Cypher".

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
6. The language does NOT affect classification — the same logic produces the same Cypher regardless of language
7. **CRITICAL — Non-classifiable code:** If the code is NOT a complete function or method — for example it is a code fragment, a conditional branch (if/else), a loop body, a variable declaration, a single statement, or any incomplete/unidentifiable snippet — respond with exactly the word: SKIP
   Do NOT explain why. Do NOT describe what is wrong. Just output: SKIP

EXAMPLES (multiple languages):
- Python: "async def send_email(to, subject): ..." → NET:SND_EMAL--ASY
- Python: "def validate_password(pwd): ..." → SEC:VAL_PASS--SYN
- JavaScript: "async function fetchUserData(id) {{...}}" → DAT:FET_USER--ASY
- Go: "func ScrubLogs(stream io.Reader) error {{...}}" → LOG:SCR_LOGS--SYN
- Rust: "pub async fn render_node(node: &Node) -> Html {{...}}" → UI:TRN_NODE--ASY
- Java: "public static List<Row> toCsv(Data data) {{...}}" → DAT:TRN_CSV--SYN
- TypeScript: "export const validateToken = (token: string): boolean => {{...}}" → SEC:VAL_TOKEN--SYN
"""
    
    CYPHER_GENERATION_USER = """Analyze this {language} function and generate its Cypher:

```{language_lower}
{code}
```

Output only the Cypher string in format DOM:ACT_OBJ--PAT"""
    
    QUERY_TRANSLATION_SYSTEM = f"""You are a natural language to Cypher-100 query translator.

The user is SEARCHING for code. Your job is to describe WHAT THE TARGET CODE DOES, not the user's act of searching.

IMPORTANT: Ignore verbs like "find", "search", "show me", "where is", "look for" — they describe the USER's intent to locate code. Instead, focus on the SUBJECT of the search: what the target code creates, fetches, validates, transforms, etc.

Convert user search queries into Cypher patterns using wildcards (*) for unknown components.

{CypherSchema.format_for_llm()}

RULES:
1. Use * for unknown components
2. For OBJ, prefer tokens from COMMON OBJECT CODES that best match the noun in the query (e.g. "dataset" → DATASET, "user" → USER, "request" → REQUEST, "logs" → LOGS).
3. Output ONLY the Cypher pattern
4. Be liberal with wildcards - it's better to match too much than too little
5. Consider synonyms and related terms
6. When the query mentions "setup", "configure", "initialize" → ACT = CRT
7. When the query mentions "logging", "log" as the main subject → DOM = LOG, OBJ = LOGS
8. When unsure about ACT, use * instead of guessing wrong

EXAMPLES:
- "find async email functions" → NET:SND_EMAL--ASY
- "where do we validate users" → SEC:VAL_USER--*
- "show me data transformations" → DAT:TRN_*--*
- "where do we fetch the dataset" → DAT:FET_DATASET--*
- "I search the main logging function for file logs" → LOG:CRT_LOGS--*
- "security functions" → SEC:*_*--*
- "find the setup logging function" → LOG:CRT_LOGS--*
- "where is the config initialized" → SYS:CRT_CONFIG--*
"""
    
    QUERY_TRANSLATION_USER = """The user wants to FIND code. Describe what the TARGET CODE does as a Cypher pattern.

User's search query: "{query}"

Output only the Cypher pattern in format DOM:ACT_OBJ--PAT (use * for wildcards)"""


# Validate on import if API key is required
if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError:
        # Allow import without key for documentation purposes
        pass
