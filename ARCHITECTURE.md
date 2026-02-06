# Symdex-100 Architecture Documentation

## System Overview

Symdex-100 is a semantic code indexing and search system that achieves 100x faster code search through LLM-generated structured metadata ("Cyphers"). The system consists of three layers: **Core** (analysis, caching, generation), **Pipelines** (indexing, search), and **Interfaces** (CLI, Python API, MCP server).

## Core Design Principles

### 1. **Speed Through Reduction**
Instead of searching 2KB of code per function, we search 20 bytes of metadata. This 100:1 reduction is the primary source of performance gains.

### 2. **Reproducibility**
By using temperature=0.0 for LLM calls and providing a strict schema, we ensure the same code always generates the same Cypher. This is critical for consistent search results.

### 3. **Semantic + Structural**
We combine rule-based AST parsing (deterministic) with LLM-based semantic understanding (contextual) to get the best of both approaches.

### 4. **Graceful Degradation**
Every component has a fallback strategy:
- LLM fails → Rule-based Cypher generation
- No exact match → Multi-lane search with progressive broadening
- API rate limit → Automatic retry with exponential backoff

### 5. **No Global State**
All core classes accept an instance-based `SymdexConfig` so that multiple clients can run in the same process with different providers, keys, and settings. No import-time side effects.

### 6. **Source Files Are Never Modified**
All metadata is stored in a `.symdex/` sidecar directory. The indexed codebase stays pristine — no unwanted comment blocks in diffs.

## Package Layout

```
src/symdex/
├── __init__.py            # Public API: Symdex, SymdexConfig, SearchResult, exceptions
├── client.py              # Symdex facade — single entry point for programmatic use
├── exceptions.py          # Custom exception hierarchy (SymdexError, ConfigError, ...)
│
├── core/
│   ├── __init__.py        # Re-exports: Config, SymdexConfig, CodeAnalyzer, CypherCache, ...
│   ├── config.py          # SymdexConfig (instance), Config (legacy global), CypherSchema, Prompts
│   ├── engine.py          # CodeAnalyzer, CypherCache, CypherGenerator, LLM providers, scoring
│   ├── indexer.py          # IndexingPipeline → IndexResult
│   └── search.py          # CypherSearchEngine, ResultFormatter
│
├── cli/
│   ├── __init__.py
│   └── main.py            # Click CLI: index, search, stats, mcp
│
└── mcp/
    ├── __init__.py
    └── server.py           # FastMCP server: tools, resources, prompt templates
```

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      SYMDEX-100 SYSTEM                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INTERFACES                                                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐       │
│  │   CLI    │  │  Python API  │  │     MCP Server       │       │
│  │ (Click)  │  │  (Symdex)    │  │ (tools + resources   │       │
│  │          │  │              │  │  + prompts + health)  │       │
│  └────┬─────┘  └──────┬───────┘  └──────────┬───────────┘       │
│       │               │                      │                    │
│       └───────────────┼──────────────────────┘                   │
│                       │                                           │
│  PIPELINES            ▼                                          │
│  ┌──────────────┐  ┌──────────────────┐                         │
│  │   Indexing   │  │     Search       │                         │
│  │   Pipeline   │  │     Engine       │                         │
│  │ → IndexResult│  │ → SearchResult[] │                         │
│  └──────┬───────┘  └────────┬─────────┘                         │
│         │                   │                                     │
│         └────────┬──────────┘                                    │
│                  │                                                │
│  CORE            ▼                                               │
│  ┌─────────────────────────────────┐                            │
│  │  CodeAnalyzer   (Python AST)    │                            │
│  │  CypherCache    (SQLite, thread-local conns)                 │
│  │  CypherGenerator(multi-provider LLM, lazy init)              │
│  │  LLMProvider    (Anthropic | OpenAI | Gemini)                │
│  └──────────┬──────────────────────┘                            │
│             │                                                     │
│  CONFIG     ▼                                                    │
│  ┌─────────────────────────────────┐                            │
│  │  SymdexConfig  (instance-based) │                            │
│  │  Config        (legacy global)  │                            │
│  │  CypherSchema  (translation tables)                          │
│  │  Prompts       (LLM templates)  │                            │
│  └─────────────────────────────────┘                            │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Configuration Layer (`core/config.py`)

**Purpose**: Centralized configuration and schema definitions.

**Key Components**:

- `SymdexConfig` *(dataclass, instance-based)*: The primary config object. Created via `SymdexConfig.from_env()` or with explicit values. Passed through all core classes. Supports multi-tenant usage — multiple clients with different providers/keys in the same process.

- `Config` *(class attributes, global)*: Legacy configuration for backward compatibility with CLI and tests. Mutable class attributes read from environment at import time. `Config.to_instance()` snapshots current state into a `SymdexConfig`.

- `CypherSchema`: The translation tables (domains, actions, patterns, keyword mappings, common object codes). Stateless.

- `Prompts`: LLM prompt templates. Reference `CypherSchema` for the schema section.

**Design Decisions**:
- **Instance-based config as primary**: `SymdexConfig` carries all settings per-client. No global mutation.
- **Environment variables for secrets**: API keys via `os.getenv()`, never hardcoded.
- **Closed vocabulary**: Fixed lists for DOM/ACT/PAT ensure reproducibility.
- **Open vocabulary for OBJ**: 2–20 uppercase letters/digits. Common objects defined in `COMMON_OBJECT_CODES` (preferred), but the LLM can generate project-specific tokens.
- **No import-time side effects**: Importing `config.py` does not validate keys or configure logging.

### 2. Core Engine (`core/engine.py`)

**Purpose**: Shared functionality used by both indexer and search engine.

#### 2.1 CodeAnalyzer

**What it does**: Python-specific function extraction using the built-in `ast` module.

**Extracts**:
- Function/method name, line numbers, arguments
- Async/sync detection
- Called functions (for tag generation)
- Docstrings
- Cyclomatic complexity approximation (branch counting)

**Why AST (not regex)**:
- Handles decorators, nested functions, multiline signatures, edge cases
- Python's `ast` module is built-in and battle-tested
- Precise `end_lineno` for exact source extraction

#### 2.2 CypherCache (SQLite)

**What it does**: Fast persistent storage for indexed metadata.

**Schema**:

```sql
indexed_files:
  - file_path (PRIMARY KEY)
  - file_hash (SHA256 for change detection)
  - last_indexed (timestamp)
  - function_count

cypher_index:
  - file_path (FOREIGN KEY)
  - function_name
  - line_start, line_end
  - cypher (INDEXED for fast LIKE queries)
  - tags, signature, complexity
```

**Thread-local connections**: Each thread reuses a single `sqlite3.Connection` via `threading.local()`. This avoids per-method connection overhead while remaining thread-safe (each thread gets its own connection).

**Why SQLite**:
- Zero configuration — no separate database server
- B-tree indexes for O(log N) lookups
- Portable — single file, works everywhere
- ACID transactions for data integrity

#### 2.3 CypherGenerator

**What it does**: Interfaces with any configured LLM provider (Anthropic, OpenAI, Gemini) to generate semantic Cyphers.

**Key Features**:

1. **Lazy LLM initialization**: The provider SDK is imported and the client created only on first actual LLM call. This means constructing a `CypherGenerator` or `CypherSearchEngine` does **not** require an API key — you only need a key when you actually call `generate_cypher()` or `translate_query()`.

2. **Multi-provider support**: `_create_provider(provider, api_key, config)` factory instantiates the correct `LLMProvider` subclass. Provider name, API key, and model are all read from the `SymdexConfig` instance.

3. **Deterministic output**: Temperature = 0.0 for consistency. Strict format validation via regex.

4. **Retry with backoff**: Configurable `retry_attempts` and `retry_backoff_base` (exponential). Retries on both API errors and invalid LLM responses.

5. **Explanation guard**: If the LLM returns a long natural-language explanation instead of a Cypher string, it's detected immediately (no wasted retries).

6. **Fallback strategy**: When all LLM attempts fail, a deterministic rule-based generator produces a valid Cypher from function name keywords and AST metadata.

**Validation**:
```
Pattern: ^[A-Z]{2,3}:[A-Z]{3}_[A-Z0-9]{2,20}--[A-Z]{3}$

DOM: 2-3 uppercase letters (SEC, DAT, UI, ...)
ACT: 3 uppercase letters   (VAL, FET, TRN, ...)
OBJ: 2-20 uppercase letters/digits (USER, TOKEN, HTTPREQ, B64, ...)
PAT: 3 uppercase letters   (ASY, SYN, REC, ...)
```

#### 2.4 LLM Provider Abstraction

```
LLMProvider (abstract base)
├── AnthropicProvider  (anthropic SDK)
├── OpenAIProvider     (openai SDK)
└── GeminiProvider     (google-genai SDK)
```

Each provider accepts `api_key` and `model` at construction. The `complete(system, user_message, max_tokens, temperature)` interface is uniform. SDK imports are lazy (only when a provider is instantiated).

#### 2.5 Data Models

| Class | Purpose |
|-------|---------|
| `FunctionMetadata` | AST-extracted function info (name, lines, args, calls, docstring, complexity) |
| `CypherMeta` | Parsed SEARCH_META block (cypher, tags, signature, complexity) |
| `SearchResult` | Ranked search hit (file, function, lines, cypher, score, context) |
| `IndexResult` | Typed return from `IndexingPipeline.run()` (counts for files/functions/errors) |

### 3. Indexing Pipeline (`core/indexer.py`)

**Purpose**: Crawl a directory, analyze Python source, generate Cyphers via LLM, store in sidecar SQLite index.

#### 3.1 Workflow

```
1. Directory Scan (os.walk with early pruning of excluded dirs)
   ↓
2. File Filtering (extensions, size limit, hash-based skip)
   ↓
3. Python AST Parsing (extract FunctionMetadata for each function)
   ↓
4. LLM Cypher Generation (with retry + rule-based fallback)
   ↓
5. Tag Generation (from name, calls, docstring, patterns)
   ↓
6. SQLite Insert (into .symdex/index.db — source files untouched)
   ↓
7. Return IndexResult (typed dataclass with all statistics)
```

#### 3.2 Concurrency Strategy

`ThreadPoolExecutor` with configurable `max_concurrent_requests` (default 5).

- 5 workers × ~6 sec per LLM request ≈ 50 req/min
- Balances speed and API rate limits
- `tqdm` progress bar (disabled when `show_progress=False` for API use)

#### 3.3 Incremental Indexing

SHA256 hash tracking skips unchanged files. On re-run, 90%+ of files are skipped. The `--force` flag bypasses hash checks.

### 4. Search Pipeline (`core/search.py`)

**Purpose**: Translate natural language to Cypher patterns and find matching functions.

#### 4.1 Multi-Lane Search Architecture

Instead of relying on a single Cypher pattern, the engine always runs **five** parallel retrieval lanes:

```
Query: "delete directory"
    ↓
┌──────────────────────────────────────────────────────────┐
│ LANE 1: Exact Cypher      │ SYS:SCR_DIR--SYN             │
│ LANE 2: Domain wildcard   │ *:SCR_DIR--SYN               │
│ LANE 3: Action-only       │ *:SCR_*--*                   │
│ LANE 4: Tag keywords      │ delete, directory             │
│ LANE 5: Function name     │ _delete_directory_tree        │
└──────────────────────────────────────────────────────────┘
    ↓
Merge + Deduplicate + Unified Scoring → Ranked Results
```

#### 4.2 Query Translation Strategies

| Strategy | Mechanism | Latency | Accuracy |
|----------|-----------|---------|----------|
| `auto` (default) | Try LLM, fall back to keyword | ~500ms | High |
| `llm` | Force LLM translation | ~500ms | Highest |
| `keyword` | Keyword mapping only | ~1ms | Medium |
| `direct` | Query is already a Cypher pattern | ~1ms | Exact |

#### 4.3 Ranking Algorithm

```python
WEIGHTS = {
    "exact_match":      10.0,   # Full Cypher match
    "domain_match":      5.0,   # SEC = SEC
    "action_match":      5.0,   # VAL = VAL
    "object_match":      3.0,   # TOKEN = TOKEN
    "object_similarity": 2.0,   # DATASET ≈ DSET (Jaccard + substring)
    "pattern_match":     2.0,   # ASY = ASY
    "tag_match":         1.5,   # Query words in function tags
    "name_match":        3.0,   # Query words in function name
}
```

Scoring includes exact word overlap and substring matching against function names, with stop-word filtering.

#### 4.4 Result Formatting

`ResultFormatter` supports four output modes:

| Format | Use Case |
|--------|----------|
| `console` | Human-friendly with line-numbered code preview |
| `json` | Scripting, piping, MCP tool responses |
| `compact` | grep-like one-line-per-result |
| `ide` | `file(line): message` for editor click-to-jump |

### 5. Exception Hierarchy (`exceptions.py`)

```
SymdexError
├── ConfigError          (ValueError)   — invalid/missing config
├── ProviderError                       — LLM API failure
├── IndexNotFoundError   (FileNotFoundError) — no .symdex/ index
├── IndexingError                       — fatal indexing failure
├── SearchError                         — search execution error
└── CypherValidationError               — malformed Cypher string
```

Inherits from stdlib types where appropriate so `except ValueError` and `except FileNotFoundError` still work.

### 6. Client Facade (`client.py`)

**Purpose**: Single entry point for programmatic use.

```python
from symdex import Symdex, SymdexConfig

client = Symdex(config=SymdexConfig(llm_provider="openai", openai_api_key="sk-..."))
result = client.index("./project")
hits   = client.search("validate tokens", path="./project")
stats  = client.stats("./project")
```

**Key properties**:
- Instance-based — each `Symdex` client has its own config, no global state
- Caches `CypherSearchEngine` per index path
- Async variants via `asyncio.to_thread()`: `aindex()`, `asearch()`, `astats()`
- Raises typed exceptions (`IndexNotFoundError`, `ConfigError`)

### 7. MCP Server (`mcp/server.py`)

**Purpose**: Expose Symdex as an MCP server for AI agents.

Built on [FastMCP](https://github.com/jlowin/fastmcp). Accepts a `SymdexConfig` at creation. Provides:

| Primitive | Items |
|-----------|-------|
| **Tools** | `search_codebase`, `search_by_cypher`, `index_directory`, `get_index_stats`, `health` |
| **Resources** | `symdex://schema/domains`, `symdex://schema/actions`, `symdex://schema/patterns`, `symdex://schema/full` |
| **Prompts** | `find_security_functions`, `audit_domain`, `explore_codebase` |

Tools raise `FileNotFoundError` on missing index (FastMCP translates this to a proper MCP error response).

### 8. CLI (`cli/main.py`)

Click-based CLI. Builds a `SymdexConfig` from env vars + `--provider` override, passes it via Click context to all subcommands.

Commands: `index`, `search`, `stats`, `mcp`.

## The Cypher-100 Schema

### Slot Specification

| Slot | Length | Charset | Vocabulary |
|------|--------|---------|------------|
| DOM (Domain) | 2–3 chars | A–Z | Closed: 8 codes (SEC, DAT, NET, SYS, LOG, UI, BIZ, TST) |
| ACT (Action) | 3 chars | A–Z | Closed: 10 codes (VAL, TRN, SND, FET, SCR, CRT, UPD, AGG, FLT, SYN) |
| OBJ (Object) | 2–20 chars | A–Z, 0–9 | Open: prefer `COMMON_OBJECT_CODES`, LLM can generate new |
| PAT (Pattern) | 3 chars | A–Z | Closed: 7 codes (ASY, SYN, REC, GEN, DEC, CTX, CLS) |

### Why 2–3 Letter Codes?
- **Readability**: Short enough to scan quickly
- **Uniqueness**: Sufficient combinations for each slot
- **Consistency**: Fixed width for pattern matching and LIKE queries
- **Mnemonic**: SEC, NET, VAL are easy to remember

### OBJ Flexibility
Objects are codebase-specific (User, Order, Email, Token). The `COMMON_OBJECT_CODES` list (~70 tokens) covers typical objects. The LLM is instructed to prefer these tokens for consistency, but can generate project-specific codes (2–20 uppercase letters/digits).

## Design Trade-offs

### 1. LLM vs Rule-Based
**Decision**: Hybrid — LLM primary, rule-based fallback.

- LLM: Better semantic understanding (recognizes PII scrubbing, business logic)
- Rules: Faster, deterministic, zero-cost
- Hybrid: Best of both, resilient to API failures

### 2. SQLite vs Vector DB
**Decision**: SQLite for primary storage.

- SQLite: Simple, fast for exact/wildcard matches, zero-config
- Vector DB: Better for semantic "find similar" queries (future enhancement)
- Current bottleneck is accuracy of Cypher generation, not search mechanism

### 3. Sidecar Index vs In-File Metadata
**Decision**: Sidecar-only (`.symdex/index.db`).

- Source files are never modified — no unwanted diffs
- Index can be rebuilt from source at any time via `symdex index --force`
- `.symdex/` can be gitignored or committed (team preference)

### 4. Instance Config vs Global Config
**Decision**: Both — `SymdexConfig` (primary) + `Config` (legacy).

- Instance-based: Multi-tenant safe, testable, no side effects
- Global: Backward-compatible for CLI and existing tests
- `Config.to_instance()` bridges the two

### 5. Temperature = 0.0
**Decision**: Zero temperature for deterministic output.

- Same code always generates the same Cypher
- Critical for reproducible search results
- Trade-off: Less creative outputs, but that's desired for classification

## Security Considerations

1. **API key storage**: Environment variables only, never in code or config files
2. **SQL injection**: Parameterized queries throughout `CypherCache`
3. **Path traversal**: `Path.resolve()` used for all path operations
4. **Rate limiting**: Configurable backoff prevents accidental API abuse
5. **No import-time validation**: Importing symdex never triggers network calls

## Extensibility

### Adding a New LLM Provider

1. Subclass `LLMProvider` in `engine.py`:
   ```python
   class OllamaProvider(LLMProvider):
       def __init__(self, api_key: str, model: str = "llama3"):
           ...
       def complete(self, system, user_message, max_tokens=300, temperature=0.0):
           ...
   ```

2. Register in `_PROVIDER_REGISTRY`:
   ```python
   _PROVIDER_REGISTRY["ollama"] = OllamaProvider
   ```

3. Add config fields to `SymdexConfig` and `Config` if needed.

### Adding a New Domain

1. Add to `CypherSchema.DOMAINS` in `config.py`
2. Add keyword mappings to `KEYWORD_TO_DOMAIN`
3. Prompts auto-update via `CypherSchema.format_for_llm()`
4. Re-index with `symdex index --force`

### Adding a New Language (future)

1. Extend `CodeAnalyzer` with a language-specific extraction path
2. Add the extension to `SymdexConfig.target_extensions`
3. Adjust prompt templates if needed
4. Re-index

---

**Built for production. Designed for speed. Optimized for accuracy.**
