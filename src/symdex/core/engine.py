"""
Symdex-100 Core Engine

Python-focused code analysis, caching, Cypher generation, and search scoring.
Uses Python's AST module for precise function extraction.
Production-grade with comprehensive error handling and logging.
"""

import ast
import hashlib
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from symdex.core.config import Config, CypherSchema, Prompts, SymdexConfig

# NOTE: Module-level logging.basicConfig() removed (v1.1).
# Application code (CLI, MCP server) is responsible for configuring logging.
# This avoids side effects when symdex is imported as a library.
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FunctionMetadata:
    """Structured representation of a function in any supported language."""
    name: str
    start_line: int
    end_line: int
    is_async: bool
    args: List[str]
    calls: List[str]
    imports: List[str]
    docstring: Optional[str]
    complexity: int  # Cyclomatic complexity approximation
    language: str = "Python"  # Human-readable language name
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CypherMeta:
    """Structured Cypher metadata (Python comment format)."""
    cypher: str
    tags: List[str]
    signature: str
    complexity: str
    version: str = Config.CYPHER_VERSION
    
    def to_comment_block(self) -> str:
        """Generate the <SEARCH_META> comment block (Python format)."""
        tags_str = ", ".join(f"#{tag}" for tag in self.tags)
        return (
            f"# <SEARCH_META v{self.version}>\n"
            f"# CYPHER: {self.cypher}\n"
            f"# SIG: {self.signature}\n"
            f"# TAGS: {tags_str}\n"
            f"# COMPLEXITY: {self.complexity}\n"
            f"# </SEARCH_META>\n"
        )
    
    @staticmethod
    def parse_from_code(code: str) -> Optional['CypherMeta']:
        """Extract existing SEARCH_META from Python code."""
        pattern = r'#\s*<SEARCH_META.*?>(.*?)#\s*</SEARCH_META>'
        match = re.search(pattern, code, re.DOTALL)
        if not match:
            return None
        
        content = match.group(1)
        cypher = re.search(r'#\s*CYPHER:\s*(.*)', content)
        tags = re.search(r'#\s*TAGS:\s*(.*)', content)
        sig = re.search(r'#\s*SIG:\s*(.*)', content)
        comp = re.search(r'#\s*COMPLEXITY:\s*(.*)', content)
        
        if cypher:
            return CypherMeta(
                cypher=cypher.group(1).strip(),
                tags=[t.strip().replace('#', '') for t in tags.group(1).split(',')] if tags else [],
                signature=sig.group(1).strip() if sig else "",
                complexity=comp.group(1).strip() if comp else "O(?)"
            )
        return None


@dataclass
class SearchResult:
    """Search result with ranking."""
    file_path: str
    function_name: str
    line_start: int
    line_end: int
    cypher: str
    score: float
    context: str = ""
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score = better


@dataclass
class IndexResult:
    """Typed result returned by :meth:`IndexingPipeline.run`.

    Provides structured access to indexing statistics without requiring
    callers to inspect internal pipeline state.
    """
    files_scanned: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    functions_found: int = 0
    functions_indexed: int = 0
    functions_skipped: int = 0
    errors: int = 0
    root_dir: str = ""
    index_dir: str = ""


# =============================================================================
# Code Analysis — language-agnostic with specialised Python AST path
# =============================================================================

class CodeAnalyzer:
    """
    Python code analyzer using AST-based extraction.
    
    Uses Python's built-in ``ast`` module for precise function metadata
    extraction with full support for nested functions, decorators, and edge cases.
    """

    # ── Public API ───────────────────────────────────────────────

    @staticmethod
    def extract_functions(source_code: str, file_path: str = "<string>") -> List[FunctionMetadata]:
        """
        Extract all function / method definitions from Python source code.
        
        Uses Python's AST module for accurate, robust extraction.
        """
        return CodeAnalyzer._extract_python_ast(source_code, file_path)

    @staticmethod
    def extract_function_source(source_code: str, start_line: int, end_line: int) -> str:
        """Extract source code for a specific function (1-indexed lines)."""
        lines = source_code.splitlines()
        return "\n".join(lines[start_line - 1:end_line])

    @staticmethod
    def generate_signature(metadata: FunctionMetadata) -> str:
        """Generate a concise function signature."""
        args_str = ", ".join(metadata.args) if metadata.args else "void"
        return_hint = "→ ?" if not metadata.docstring else "→ Any"
        return f"[{args_str}] {return_hint}"

    @staticmethod
    def estimate_complexity_class(complexity: int) -> str:
        """Convert cyclomatic complexity to Big-O notation."""
        if complexity <= 2:
            return "O(1)"
        elif complexity <= 5:
            return "O(N)"
        elif complexity <= 10:
            return "O(N²)"
        else:
            return "O(N³+)"

    # ── Python AST extraction (precise) ──────────────────────────

    @staticmethod
    def _extract_python_ast(source_code: str, file_path: str) -> List[FunctionMetadata]:
        """Extract functions from Python source using the ``ast`` module."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metadata = CodeAnalyzer._analyze_python_function(node, source_code)
                functions.append(metadata)
        return functions

    @staticmethod
    def _analyze_python_function(node: ast.FunctionDef, source_code: str) -> FunctionMetadata:
        """Analyse a single Python AST function node."""
        name = node.name
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        is_async = isinstance(node, ast.AsyncFunctionDef)

        args = [arg.arg for arg in node.args.args]

        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        calls = list(set(calls))[:10]

        imports: List[str] = []
        docstring = ast.get_docstring(node)

        complexity = sum(
            1 for child in ast.walk(node)
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With))
        ) + 1

        return FunctionMetadata(
            name=name,
            start_line=start_line,
            end_line=end_line,
            is_async=is_async,
            args=args,
            calls=calls,
            imports=imports,
            docstring=docstring,
            complexity=complexity,
            language="Python",
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_args(raw_args: str) -> List[str]:
        """
        Parse a raw Python argument string into a list of parameter names.
        
        Strips type annotations and default values.
        """
        if not raw_args or not raw_args.strip():
            return []

        args: List[str] = []
        for part in raw_args.split(","):
            part = part.strip()
            if not part:
                continue
            # Remove default value (= ...)
            part = part.split("=")[0].strip()
            # Remove type annotation after ':'  (Python)
            part = part.split(":")[0].strip()
            if part and part.isidentifier():
                args.append(part)
        return args


# =============================================================================
# Cache Management (SQLite)
# =============================================================================

class CypherCache:
    """SQLite-based cache for indexed files and Cypher metadata.

    Uses thread-local connections so that each thread reuses a single
    connection instead of opening/closing one per method call.  This is
    both more efficient and thread-safe (each thread has its own conn).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating it on first use."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                last_indexed TIMESTAMP NOT NULL,
                function_count INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cypher_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                function_name TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                cypher TEXT NOT NULL,
                tags TEXT,
                signature TEXT,
                complexity TEXT,
                indexed_at TIMESTAMP NOT NULL,
                FOREIGN KEY (file_path) REFERENCES indexed_files (file_path) ON DELETE CASCADE
            )
        """)

        # Create indexes for fast searching
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cypher ON cypher_index(cypher)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON cypher_index(file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_function_name ON cypher_index(function_name)")
        conn.commit()

    def get_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def is_file_indexed(self, file_path: Path) -> bool:
        """Check if file is already indexed and up-to-date."""
        current_hash = self.get_file_hash(file_path)
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT file_hash FROM indexed_files WHERE file_path = ?",
            (str(file_path),),
        )
        row = cursor.fetchone()
        return bool(row and row[0] == current_hash)

    def mark_file_indexed(self, file_path: Path, function_count: int):
        """Mark a file as indexed."""
        file_hash = self.get_file_hash(file_path)
        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO indexed_files (file_path, file_hash, last_indexed, function_count)
            VALUES (?, ?, ?, ?)
        """, (str(file_path), file_hash, datetime.now(), function_count))
        conn.commit()

    def add_cypher_entry(self, file_path: Path, function_name: str, line_start: int,
                         line_end: int, cypher: str, tags: List[str],
                         signature: str, complexity: str):
        """Add a Cypher entry to the index."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO cypher_index
            (file_path, function_name, line_start, line_end, cypher, tags, signature, complexity, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(file_path), function_name, line_start, line_end, cypher,
              ",".join(tags), signature, complexity, datetime.now()))
        conn.commit()

    def search_by_cypher(self, pattern: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for functions matching a Cypher pattern (supports wildcards)."""
        sql_pattern = pattern.replace("*", "%")
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT file_path, function_name, line_start, line_end, cypher, tags, signature
            FROM cypher_index
            WHERE cypher LIKE ?
            ORDER BY indexed_at DESC
            LIMIT ?
        """, (sql_pattern, limit))
        results = [dict(row) for row in cursor.fetchall()]
        conn.row_factory = None  # Reset for other queries
        return results

    def search_by_tags(self, tag: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for functions by tag."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT file_path, function_name, line_start, line_end, cypher, tags
            FROM cypher_index
            WHERE tags LIKE ?
            LIMIT ?
        """, (f"%{tag}%", limit))
        results = [dict(row) for row in cursor.fetchall()]
        conn.row_factory = None
        return results

    def search_by_name(self, keywords: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for functions whose name contains any of the given keywords.

        Uses SQL LIKE with wildcard wrapping so ``delete`` matches
        ``_delete_directory_tree_and_relations``.  The function_name column
        is already indexed (``idx_function_name``) for fast lookups.
        """
        if not keywords:
            return []

        conditions = " OR ".join(["function_name LIKE ?"] * len(keywords))
        params: list = [f"%{kw}%" for kw in keywords]

        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(f"""
            SELECT file_path, function_name, line_start, line_end,
                   cypher, tags, signature
            FROM cypher_index
            WHERE {conditions}
            ORDER BY indexed_at DESC
            LIMIT ?
        """, params + [limit])
        results = [dict(row) for row in cursor.fetchall()]
        conn.row_factory = None
        return results

    def get_stats(self) -> Dict[str, int]:
        """Get indexing statistics."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM indexed_files")
        file_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM cypher_index")
        function_count = cursor.fetchone()[0]

        return {
            "indexed_files": file_count,
            "indexed_functions": function_count,
        }

    def clear_file_entries(self, file_path: Path):
        """Remove all entries for a specific file."""
        conn = self._get_connection()
        conn.execute("DELETE FROM cypher_index WHERE file_path = ?", (str(file_path),))
        conn.execute("DELETE FROM indexed_files WHERE file_path = ?", (str(file_path),))
        conn.commit()


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class LLMProvider:
    """
    Abstract base for LLM API providers.

    Each subclass wraps a single vendor SDK and exposes a uniform
    ``complete(system, user_message, ...)`` interface so that the rest
    of the codebase never imports a vendor SDK directly.
    """

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's text response for the given prompts."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider — uses the ``anthropic`` SDK."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        try:
            import anthropic  # Lazy import
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The 'anthropic' SDK is not installed or broken in this Python environment.\n"
                "  Install:  pip install 'symdex-100[anthropic]'\n"
                "  Or switch provider:  export SYMDEX_LLM_PROVIDER=openai"
            ) from exc
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user_message: str,
                 max_tokens: int = 300,
                 temperature: float = 0.0) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI (GPT) provider — uses the ``openai`` SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            import openai  # Lazy import
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The 'openai' SDK is not installed.\n"
                "  Install:  pip install 'symdex-100[openai]'\n"
                "  Or switch provider:  export SYMDEX_LLM_PROVIDER=anthropic"
            ) from exc
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user_message: str,
                 max_tokens: int = 300,
                 temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini provider — uses the ``google-genai`` SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        try:
            from google import genai  # Lazy import
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The 'google-genai' SDK is not installed.\n"
                "  Install:  pip install 'symdex-100[gemini]'\n"
                "  Or switch provider:  export SYMDEX_LLM_PROVIDER=anthropic"
            ) from exc
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, system: str, user_message: str,
                 max_tokens: int = 300,
                 temperature: float = 0.0) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text.strip()


# Provider registry — maps config name → class
_PROVIDER_REGISTRY: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def _create_provider(
    provider: str | None = None,
    api_key: str | None = None,
    config: SymdexConfig | None = None,
) -> LLMProvider:
    """
    Factory that instantiates the correct :class:`LLMProvider`.

    When *config* is provided, provider name, API key, and model are
    read from the instance.  Falls back to ``SymdexConfig.from_env()``
    when no explicit config is given.
    """
    cfg = config or SymdexConfig.from_env()
    provider = (provider or cfg.llm_provider).lower()
    if provider not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDER_REGISTRY)}"
        )
    api_key = api_key or cfg.get_api_key()
    model_map = {
        "anthropic": cfg.anthropic_model,
        "openai": cfg.openai_model,
        "gemini": cfg.gemini_model,
    }
    return _PROVIDER_REGISTRY[provider](api_key=api_key, model=model_map[provider])


# =============================================================================
# LLM Integration
# =============================================================================

class CypherGenerator:
    """
    LLM-powered Cypher fingerprint generator.

    Works with any configured provider (Anthropic, OpenAI, Gemini).
    The LLM provider is created **lazily** on first use, so constructing
    a ``CypherGenerator`` does not require an API key until an LLM call
    is actually made.

    Args:
        api_key: Explicit API key override.
        provider: Explicit provider name override.
        config: Instance-based configuration.  Falls back to
            ``SymdexConfig.from_env()`` when *None*.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str | None = None,
        config: SymdexConfig | None = None,
    ):
        self._config = config or SymdexConfig.from_env()
        self._api_key = api_key
        self._provider_name = provider
        # Lazy — created on first actual LLM call via _ensure_llm()
        self.llm: LLMProvider | None = None

    # Maximum length of a valid Cypher string (e.g. "SEC:VAL_HTTPREQUEST--ASY").
    # Responses longer than this that don't validate are treated as
    # LLM explanations and cause an immediate skip (no retry).
    _MAX_CYPHER_LENGTH = 40

    @property
    def _effective_config(self) -> SymdexConfig:
        """Config accessor safe for instances created via ``__new__``."""
        return getattr(self, "_config", None) or SymdexConfig.from_env()

    def _ensure_llm(self) -> LLMProvider:
        """Create the LLM provider on first use (lazy initialization)."""
        if self.llm is None:
            self.llm = _create_provider(
                provider=getattr(self, "_provider_name", None),
                api_key=getattr(self, "_api_key", None),
                config=self._effective_config,
            )
        return self.llm

    def generate_cypher(self, function_code: str, metadata: FunctionMetadata) -> Optional[str]:
        """
        Generate a Cypher string for a Python function.

        Returns ``None`` when the LLM determines the code is not a
        classifiable function (SKIP response) or consistently returns
        explanation text instead of a valid Cypher.

        Retries on both API errors AND invalid LLM responses (up to
        ``retry_attempts`` total).  Only falls back to the rule-based
        generator after all attempts are exhausted.
        """
        import time

        cfg = self._effective_config
        for attempt in range(1, cfg.retry_attempts + 1):
            try:
                cypher = self._ensure_llm().complete(
                    system=Prompts.CYPHER_GENERATION_SYSTEM,
                    user_message=Prompts.CYPHER_GENERATION_USER.format(
                        code=function_code,
                    ),
                    max_tokens=cfg.llm_max_tokens,
                    temperature=cfg.llm_temperature,
                )

                stripped = cypher.strip()

                # ── Explicit SKIP: LLM says the code is not classifiable ──
                if stripped.upper().startswith("SKIP"):
                    logger.info(
                        f"LLM returned SKIP for '{metadata.name}' — "
                        "code fragment is not a classifiable function."
                    )
                    return None

                # ── Valid Cypher — accept immediately ──
                if self._validate_cypher(stripped):
                    return stripped

                # ── Explanation guard: if the response is clearly a
                #    natural-language explanation (too long, contains
                #    spaces) rather than a malformed Cypher, skip
                #    immediately instead of wasting retries. ──
                if len(stripped) > self._MAX_CYPHER_LENGTH and " " in stripped:
                    logger.info(
                        f"LLM returned an explanation instead of a Cypher "
                        f"for '{metadata.name}' — treating as "
                        f"non-classifiable. Response (truncated): "
                        f"'{stripped[:100]}...'"
                    )
                    return None

                logger.warning(
                    f"Invalid Cypher on attempt {attempt}/{cfg.retry_attempts}: "
                    f"'{stripped}' for {metadata.name}"
                )
            except Exception as e:
                logger.warning(
                    f"API error on attempt {attempt}/{cfg.retry_attempts} "
                    f"for {metadata.name}: {e}"
                )

            # Back off before next attempt (skip sleep on last attempt)
            if attempt < cfg.retry_attempts:
                backoff = cfg.retry_backoff_base ** (attempt - 1)
                time.sleep(backoff)

        # All attempts exhausted — deterministic fallback
        logger.error(
            f"All {cfg.retry_attempts} attempts failed for {metadata.name}. "
            "Using rule-based fallback."
        )
        return self._generate_fallback_cypher(metadata)

    def translate_query(self, natural_query: str) -> str:
        """Translate a natural-language query to a Cypher pattern."""
        import time

        cfg = self._effective_config
        for attempt in range(1, cfg.retry_attempts + 1):
            try:
                pattern = self._ensure_llm().complete(
                    system=Prompts.QUERY_TRANSLATION_SYSTEM,
                    user_message=Prompts.QUERY_TRANSLATION_USER.format(
                        query=natural_query,
                    ),
                    max_tokens=cfg.llm_max_tokens,
                    temperature=0.0,
                )
                return (
                    pattern
                    if self._validate_cypher(pattern, allow_wildcards=True)
                    else "*:*_*--*"
                )
            except Exception as e:
                if attempt >= cfg.retry_attempts:
                    logger.error(
                        f"Query translation error after {attempt} attempts: {e}"
                    )
                    break
                backoff = cfg.retry_backoff_base ** (attempt - 1)
                logger.warning(
                    f"Query translation error on attempt {attempt}: {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)

        # Fallback to keyword-based translation if LLM fails
        return self._keyword_based_translation(natural_query)
    
    @staticmethod
    def _validate_cypher(cypher: str, allow_wildcards: bool = False) -> bool:
        """
        Validate Cypher format: DOM:ACT_OBJ--PAT
        
        Lengths are flexible to accommodate real schema codes:
          DOM: 2-3 chars   (e.g. UI, SEC, DAT)
          ACT: 3 chars     (e.g. TRN, VAL, FET)
          OBJ: 2-20 chars  (e.g. QS, CSV, USER, BINOP, CLSDEF, DICTCOMP, B64, K8OBJ, HTTPREQ)
          PAT: 3 chars     (e.g. SYN, ASY, REC)
        
        With allow_wildcards=True each slot may also be a single '*'.
        OBJ allows A–Z and 0–9 to support compact encodings like B64, K8OBJ, HTTPREQ.
        """
        # Slot patterns: exact code OR single wildcard '*'
        dom_p = r'(?:\*|[A-Z]{2,3})'      if allow_wildcards else r'[A-Z]{2,3}'
        act_p = r'(?:\*|[A-Z]{3})'        if allow_wildcards else r'[A-Z]{3}'
        obj_p = r'(?:\*|[A-Z0-9]{2,20})'  if allow_wildcards else r'[A-Z0-9]{2,20}'
        pat_p = r'(?:\*|[A-Z]{3})'        if allow_wildcards else r'[A-Z]{3}'
        
        full_pattern = rf'^({dom_p}):({act_p})_({obj_p})--({pat_p})$'
        match = re.match(full_pattern, cypher)
        if not match:
            return False
        
        if not allow_wildcards:
            dom, act, obj, pat = match.groups()
            codes = CypherSchema.get_all_codes()
            return (dom in codes["domains"] and 
                    act in codes["actions"] and 
                    pat in codes["patterns"])
        
        return True
    
    @staticmethod
    def _generate_fallback_cypher(metadata: FunctionMetadata) -> str:
        """Generate a rule-based Cypher when LLM fails."""
        # Determine domain and action from function name
        name_lower = metadata.name.lower()
        
        dom = "DAT"  # Default
        for keyword, code in CypherSchema.KEYWORD_TO_DOMAIN.items():
            if keyword in name_lower:
                dom = code
                break
        
        act = "TRN"  # Default
        for keyword, code in CypherSchema.KEYWORD_TO_ACTION.items():
            if keyword in name_lower:
                act = code
                break
        
        pat = "ASY" if metadata.is_async else "SYN"
        
        # Extract object from function name (crude heuristic)
        obj = "DATA"
        name_parts = re.findall(r'[A-Z][a-z]+', metadata.name.replace('_', ' ').title())
        if name_parts:
            obj = "".join(name_parts[0][:4].upper()).ljust(4, 'X')
        
        return f"{dom}:{act}_{obj}--{pat}"
    
    @staticmethod
    def _keyword_based_translation(query: str) -> str:
        """Fallback keyword-based query translation."""
        query_lower = query.lower()
        
        dom = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_DOMAIN.items():
            if keyword in query_lower:
                dom = code
                break
        
        act = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_ACTION.items():
            if keyword in query_lower:
                act = code
                break
        
        pat = "*"
        for keyword, code in CypherSchema.KEYWORD_TO_PATTERN.items():
            if keyword in query_lower:
                pat = code
                break
        
        return f"{dom}:{act}_*--{pat}"


# =============================================================================
# Utility Functions
# =============================================================================

def scan_directory(root_path: Path, config: SymdexConfig | None = None) -> List[Path]:
    """
    Recursively scan for Python source files (.py).

    Uses :func:`os.walk` with **early directory pruning** so that
    excluded subtrees (e.g. ``.git/``, ``__pycache__/``) are never
    entered — a significant speedup on large repositories.

    Respects ``config.target_extensions`` and ``config.exclude_dirs``.
    """
    import os

    cfg = config or SymdexConfig.from_env()
    source_files: List[Path] = []
    exclude = cfg.exclude_dirs            # frozenset — O(1) lookups
    extensions = cfg.target_extensions     # frozenset — O(1) lookups
    max_bytes = cfg.max_file_size_mb * 1024 * 1024

    for dirpath, dirnames, filenames in os.walk(root_path):
        # ── Prune excluded directories IN-PLACE so os.walk never
        #    descends into them.  Modifying dirnames[:] is the
        #    documented way to control os.walk traversal.
        dirnames[:] = [d for d in dirnames if d not in exclude]

        for fname in filenames:
            # Fast extension check (splitext is C-level, frozenset lookup is O(1))
            _, ext = os.path.splitext(fname)
            if ext not in extensions:
                continue

            full = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue

            if size <= max_bytes:
                source_files.append(Path(full))
            else:
                logger.warning(
                    f"Skipping large file: {full} ({size / (1024 * 1024):.1f}MB)"
                )

    source_files.sort()
    return source_files


def _object_similarity(obj_pattern: str, obj_result: str) -> float:
    """
    Compute a soft similarity between two OBJ codes in [0, 1].
    
    - Treat '*' as unknown (similarity 0).
    - Boost when one is a substring of the other (e.g. DATASET vs DSET).
    - Base score is Jaccard similarity over character sets.
    """
    obj_pattern = obj_pattern.upper()
    obj_result = obj_result.upper()
    
    if obj_pattern == "*" or obj_result == "*":
        return 0.0
    if not obj_pattern or not obj_result:
        return 0.0
    
    set_p = set(obj_pattern)
    set_r = set(obj_result)
    intersection = len(set_p & set_r)
    union = len(set_p | set_r)
    if union == 0:
        base = 0.0
    else:
        base = intersection / union
    
    # Extra boost if one token clearly contains the other
    boost = 0.0
    if obj_pattern in obj_result or obj_result in obj_pattern:
        boost = 0.3
    
    return min(1.0, base + boost)


def calculate_search_score(
    cypher_pattern: str,
    result_cypher: str,
    tags: List[str],
    query: str,
    function_name: str = "",
    config: SymdexConfig | None = None,
) -> float:
    """
    Calculate relevance score for a search result.
    
    Scoring components:
      - Cypher slot matches (domain, action, object, pattern)
      - Object similarity (substring / Jaccard)
      - Exact Cypher match bonus
      - Tag overlap with query words
      - Function name overlap with query words
    """
    cfg = config or SymdexConfig.from_env()
    score = 0.0
    weights = cfg.search_ranking_weights
    
    # Use centralized stop words from config
    stop_words = cfg.stop_words
    
    pattern_parts = cypher_pattern.split(':')
    result_parts = result_cypher.split(':')
    
    if len(pattern_parts) == 2 and len(result_parts) == 2:
        # Domain match
        if pattern_parts[0] == result_parts[0]:
            score += weights["domain_match"]
        
        # Action and Object
        pattern_aop = pattern_parts[1].split('--')
        result_aop = result_parts[1].split('--')
        
        if len(pattern_aop) == 2 and len(result_aop) == 2:
            pattern_ao = pattern_aop[0].split('_')
            result_ao = result_aop[0].split('_')
            
            if len(pattern_ao) == 2 and len(result_ao) == 2:
                pat_act, pat_obj = pattern_ao
                res_act, res_obj = result_ao
                
                # Action match
                if pat_act == res_act or pat_act == '*':
                    score += weights["action_match"]
                
                # Object exact / wildcard match
                if pat_obj == res_obj or pat_obj == '*':
                    score += weights["object_match"]
                else:
                    # Soft similarity between object codes, e.g. DATASET vs DSET
                    sim = _object_similarity(pat_obj, res_obj)
                    score += sim * weights.get("object_similarity", 0.0)
            
            # Pattern match
            if pattern_aop[1] == result_aop[1] or pattern_aop[1] == '*':
                score += weights["pattern_match"]
    
    # Exact match bonus
    if cypher_pattern == result_cypher:
        score += weights["exact_match"]
    
    # ── Query word matching ──
    query_words = {
        w for w in query.lower().split()
        if w not in stop_words and len(w) > 1
    }
    
    # Tag relevance
    matching_tags = sum(
        1 for tag in tags
        if any(word in tag.lower() for word in query_words)
    )
    score += matching_tags * weights["tag_match"]
    
    # Function name relevance — exact word overlap + substring matching.
    # Exact:     "delete" in query matches "delete" in snake_case parts.
    # Substring: "deletion" in query → "delete" found as substring inside
    #            the full function name, scoring at 70 % of exact weight.
    if function_name and query_words:
        name_lower = function_name.lower()
        name_parts = set(name_lower.replace("_", " ").split())

        # Exact word overlap (highest confidence)
        exact_hits = len(query_words & name_parts)

        # Substring overlap for remaining words
        remaining = query_words - name_parts
        substr_hits = sum(
            1 for qw in remaining if qw in name_lower
        )

        score += (exact_hits + substr_hits * 0.7) * weights.get("name_match", 3.0)

    return score
