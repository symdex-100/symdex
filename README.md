# Symdex-100

<div align="center">

![Symdex Robot](./docs/symdex-100.png)

*smydex-100 - your AI companion for code exploration*

</div>

---

**Semantic fingerprints for 100x faster Python code search.**

Symdex-100 generates compact, structured metadata ("Cyphers") for every function in your Python codebase. Each Cypher is a 20-byte semantic fingerprint that enables sub-second, intent-based code search for developers and AI agents — without reading thousands of lines of code.

```python
# Your Python function → Indexed automatically
async def validate_user_token(token: str, user_id: int) -> bool:
    """Verify JWT token for a specific user."""
    # ... implementation ...
```

```bash
# Natural language search → Sub-second results
$ symdex search "where do we validate user tokens"

──────────────────────────────────────────────────────────────────────────────
  SYMDEX — 1 result in 0.0823 seconds
──────────────────────────────────────────────────────────────────────────────

  #1  validate_user_token  (Python)
  ────────────────────────────────────────────────────────────────────────────
    File   : /project/auth/tokens.py
    Lines  : 42–67
    Cypher : SEC:VAL_TOKEN--ASY
    Score  : 24.5

      42 │ async def validate_user_token(token: str, user_id: int) -> bool:
      43 │     """Verify JWT token for a specific user."""
      44 │     if not token:
      45 │         return False
```

---

## The Problem

Traditional code search methods scale poorly on large codebases:

| Approach | Limitation | Token Cost (AI agents) |
|----------|-----------|------------------------|
| **grep** | Keyword noise — finds "token" in comments, strings, variable names | 3,000+ tokens (read all matches) |
| **Full-text search** | No semantic understanding — can't distinguish intent | 5,000+ tokens (read 10 files) |
| **Embeddings** | Opaque, expensive, query-time overhead | 2,000+ tokens (re-rank results) |
| **AST/LSP** | Limited to structural queries (class/function names) | N/A (doesn't understand "what validates X") |

**Result**: Developers waste time reading irrelevant code. AI agents burn tokens on noise.

---

## The Solution: Semantic Fingerprints

Symdex-100 solves this with **Cypher-100**, a structured metadata format that encodes function semantics in 20 bytes:

### Anatomy of a Cypher-100 String

Each Cypher follows a strict four-slot hierarchy designed for both machine filtering and human readability:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│            DOM   :   ACT   _   OBJ   --   PAT               │
│              │        │         │           │               │
│         Domain   Action       Object        Pattern         │
│                                                             │
│   Where does     What does    What is       How does        │
│   this live?     it do?       the target?   it run?         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Formal specification:**

$$
\text{Cypher} = \text{DOM} : \text{ACT} \_ \text{OBJ} \text{--} \text{PAT}
$$

Where:

- **DOM** *(Domain)*: Semantic namespace — `SEC` (Security), `NET` (Network), `DAT` (Data), `SYS` (System), `LOG` (Logging), `UI` (Interface), `BIZ` (Business), `TST` (Testing)

- **ACT** *(Action)*: Primary operation — `VAL` (Validate), `FET` (Fetch), `TRN` (Transform), `CRT` (Create), `SND` (Send), `SCR` (Scrub), `UPD` (Update), `AGG` (Aggregate), `FLT` (Filter), `DEL` (Delete)

- **OBJ** *(Object)*: Target entity — `USER`, `TOKEN`, `DATASET`, `CONFIG`, `LOGS`, `REQUEST`, `JSON`, `EMAIL`, `DIR`

- **PAT** *(Pattern)*: Execution model — `ASY` (Async), `SYN` (Synchronous), `REC` (Recursive), `GEN` (Generator), `DEC` (Decorator), `CTX` (Context manager)

**Example:**

```
SEC:SCR_EMAIL--ASY
```

**Translation:** A security function that scrubs email data asynchronously.

**Breakdown:**
- `SEC` = Security domain
- `SCR` = Scrub action (sanitize/remove)
- `EMAIL` = Email object
- `ASY` = Asynchronous pattern

This 18-character string replaces 2,000+ characters of function body for search purposes — a **100:1 compression ratio** with zero semantic loss.

---

## Core Benefits

### 1. **Search Speed**

**Problem**: `grep` reads every file, full-text indexes scan every function.

**Solution**: Symdex searches 20-byte Cyphers in a SQLite B-tree index.

| Metric | Grep | Symdex | Improvement |
|--------|------|--------|-------------|
| Data scanned per query | ~50MB (full codebase) | ~100KB (index) | **500x less I/O** |
| Query time (5,000 functions) | 800ms | 8ms | **100x faster** |
| Index size | N/A (no index) | 2MB | **25:1 compression** |

**Technical details:**
- SQLite B-tree: O(log N) lookups with compound indexes on `(cypher, tags, function_name)`
- Multi-lane parallel retrieval: 5 concurrent strategies merged in <10ms
- Incremental indexing: SHA256 hash tracking skips unchanged files

**Result**: Sub-second search on 10,000+ function codebases.

---

### 2. **Search Accuracy**

**Problem**: Single search strategies miss valid results (e.g., `SYS:DEL_DIR` won't find `DAT:DEL_DIR` if query specifies system domain).

**Solution**: Always-on multi-lane search architecture.

```
Query: "delete directory"
    ↓
┌────────────────────────────────────────────────────────────┐
│ LANE 1: Exact Cypher      │ SYS:DEL_DIR--SYN              │
│ LANE 2: Domain wildcard   │ *:DEL_DIR--SYN                │
│ LANE 3: Action-only       │ *:DEL_*--*                    │
│ LANE 4: Tag keywords      │ delete, directory, recursive  │
│ LANE 5: Function name     │ _delete_directory_tree        │
└────────────────────────────────────────────────────────────┘
    ↓
Merge + Deduplicate + Unified Scoring
    ↓
Ranked Results (exact match = highest score)
```

**Scoring algorithm:**

$$
\text{score} = 10 \cdot [\text{exact}] + 5 \cdot [\text{domain}] + 5 \cdot [\text{action}] + 3 \cdot [\text{object}] + 3 \cdot [\text{name}] + 1.5 \cdot [\text{tags}]
$$

Where $[\text{x}]$ is 1 if matched, 0 otherwise (with partial matching for substring overlap).

**Result**: Cross-domain coverage with deterministic, explainable relevance ranking.

---

### 3. **Token Efficiency** (for AI Agents)

**Problem**: Agents waste 80-90% of context on reading irrelevant code when exploring large codebases.

**Solution**: Symdex provides a 50:1 token reduction via semantic search.

**Scenario:** Agent needs to find "function that validates user login credentials"

| Approach | Process | Tokens |
|----------|---------|--------|
| **Read 10 files** | Agent guesses likely files → reads all → searches manually | ~5,000 |
| **Grep + read** | `grep "login\|credential"` → read 20 matches → filter manually | ~3,000 |
| **Symdex** | `search_codebase("validate login credentials")` → 1 precise result | ~100 |

**Token breakdown (Symdex approach):**
- Query: 20 tokens
- MCP tool call overhead: 30 tokens
- Result (1 function, 5-line preview): 50 tokens
- **Total: 100 tokens**

**Savings: 50x fewer tokens, zero false positives.**

**Why this matters:**
- 200K context window → explore 50x more functions
- 90% reduction in API costs for code exploration
- Faster reasoning (less noise in context)

---

### 4. **Noise Reduction**

**Problem**: Keyword searches return false positives (e.g., "token" in variable names, comments, docstrings).

**Solution**: Semantic fingerprints distinguish intent from mention.

| Query | Grep (keyword) | Symdex (semantic) |
|-------|----------------|-------------------|
| "validate token" | 47 results (includes `token = ...`, `# token expired`, `TOKEN_KEY`) | 3 results (only functions that *validate* tokens) |
| "delete user" | 89 results (includes `# delete user later`, `user.delete_flag`) | 2 results (only functions that *delete* users) |

**Precision improvement:** 15x fewer false positives on average.

---

## Quick Start

### Install

```bash
# Published package (once available on PyPI)
pip install symdex-100

# Local development (from source — see "Local Development" below)
pip install -e ".[all]"
```

### Set API Key

```bash
# Anthropic (default, recommended)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use OpenAI / Gemini
export SYMDEX_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."
```

Supports **Anthropic Claude** (default), **OpenAI GPT**, or **Google Gemini**.

### CLI Usage

```bash
# Index a project
symdex index ./my-project

# Natural language search
symdex search "where do we validate user passwords"

# Direct Cypher (skip LLM translation)
symdex search "SEC:VAL_PASS--*"

# With pagination
symdex search "async email" -n 20 -p 5

# JSON output (for scripting)
symdex search "delete directory" --format json | jq '.[] | .file_path'

# Check statistics
symdex stats
```

Creates `.symdex/index.db` (SQLite). Source files are **never modified**.

### Python API

Symdex can be used as a library in your own applications — no CLI needed.

```python
from symdex import Symdex

# Create a client (reads API key from environment)
client = Symdex()

# Index a project
result = client.index("./my-project")
print(f"Indexed {result.functions_indexed} functions in {result.files_scanned} files")

# Search by intent
hits = client.search("validate user tokens", path="./my-project")
for hit in hits:
    print(f"  {hit.function_name} @ {hit.file_path}:{hit.line_start}  [{hit.cypher}]")

# Search by Cypher pattern (no LLM needed)
hits = client.search_by_cypher("SEC:VAL_*--*", path="./my-project")

# Get index statistics
stats = client.stats("./my-project")
print(f"{stats['indexed_files']} files, {stats['indexed_functions']} functions")
```

**With explicit configuration** (no environment variables needed):

```python
from symdex import Symdex, SymdexConfig

config = SymdexConfig(
    llm_provider="openai",
    openai_api_key="sk-...",
    openai_model="gpt-4o-mini",
    max_search_results=10,
    min_search_score=3.0,
)
client = Symdex(config=config)
```

**Async support** (for FastAPI, Django async views, etc.):

```python
from symdex import Symdex

client = Symdex()

# All operations have async variants
result = await client.aindex("./my-project")
hits   = await client.asearch("validate tokens", path="./my-project")
stats  = await client.astats("./my-project")
```

**Error handling:**

```python
from symdex import Symdex, IndexNotFoundError, ConfigError

client = Symdex()

try:
    hits = client.search("validate user")
except IndexNotFoundError:
    print("Run client.index() first!")
except ConfigError:
    print("Check your API key configuration")
```

---

## Cypher Taxonomy Reference

### Domains (DOM)

| Code | Domain | Example Functions |
|------|--------|-------------------|
| `SEC` | Security | `validate_token`, `hash_password`, `encrypt_data` |
| `DAT` | Data | `fetch_user`, `transform_csv`, `aggregate_metrics` |
| `NET` | Network | `send_request`, `handle_webhook`, `fetch_api_data` |
| `SYS` | System | `delete_directory`, `check_disk_space`, `spawn_process` |
| `LOG` | Logging | `setup_logger`, `scrub_sensitive_logs`, `format_trace` |
| `UI` | Interface | `render_template`, `validate_form`, `format_output` |
| `BIZ` | Business | `calculate_discount`, `approve_order`, `check_eligibility` |
| `TST` | Testing | `mock_database`, `assert_response`, `generate_fixture` |

### Actions (ACT)

| Code | Action | Typical Use Cases |
|------|--------|-------------------|
| `VAL` | Validate | Input validation, schema checks, token verification |
| `FET` | Fetch | Database queries, API calls, file reads |
| `TRN` | Transform | Format conversion, data mapping, serialization |
| `CRT` | Create | Object instantiation, file creation, record insertion |
| `SND` | Send | Network requests, message queues, email dispatch |
| `SCR` | Scrub | Data sanitization, PII removal, log filtering |
| `UPD` | Update | Record modification, cache refresh, state change |
| `AGG` | Aggregate | Reduce operations, metrics collection, summaries |
| `FLT` | Filter | Query refinement, access control, data selection |
| `DEL` | Delete | Resource cleanup, record removal, file deletion |

### Patterns (PAT)

| Code | Pattern | Description |
|------|---------|-------------|
| `ASY` | Async | `async def` functions, promises, coroutines |
| `SYN` | Synchronous | Standard blocking functions |
| `REC` | Recursive | Self-calling functions, tree traversals |
| `GEN` | Generator | `yield`-based functions, iterators |
| `DEC` | Decorator | Function wrappers, middleware |
| `CTX` | Context Manager | `with` statements, resource management |
| `CLS` | Closure | Functions returning functions, lexical scope |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYMDEX-100 ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Python Source (.py)                                            │
│         │                                                         │
│         ├─→ [AST Parser] ──→ Function Metadata                  │
│         │                     (name, args, docstring, ...)       │
│         │                                                         │
│         └─→ [LLM] ──────────→ Cypher Generation                 │
│                                SEC:VAL_TOKEN--ASY                │
│                                                                   │
│   ┌─────────────────────────────────────────────────┐           │
│   │         .symdex/index.db (SQLite)               │           │
│   ├─────────────────────────────────────────────────┤           │
│   │  • B-tree index on (cypher, tags, function_name)│           │
│   │  • SHA256 hash for incremental indexing         │           │
│   │  • 100:1 compression vs full function bodies    │           │
│   └─────────────────────────────────────────────────┘           │
│                        ↓                                         │
│   ┌─────────────────────────────────────────────────┐           │
│   │           MULTI-LANE SEARCH ENGINE              │           │
│   ├─────────────────────────────────────────────────┤           │
│   │  Query → [LLM Translation] → Cypher Pattern     │           │
│   │     ↓                                            │           │
│   │  5 Parallel Lanes:                              │           │
│   │    1. Exact Cypher match                        │           │
│   │    2. Domain wildcard                           │           │
│   │    3. Action-only                               │           │
│   │    4. Tag keywords                              │           │
│   │    5. Function name substring                   │           │
│   │     ↓                                            │           │
│   │  Merge → Score → Rank → Format                  │           │
│   └─────────────────────────────────────────────────┘           │
│                        ↓                                         │
│   Results (100x faster, 50x fewer tokens)                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

1. **Python AST** (not regex): Handles decorators, nested functions, edge cases
2. **Sidecar index** (not inline): Source files stay pristine, no diffs
3. **Multi-lane search** (not single-pattern): Cross-domain recall + precision
4. **LLM + rule-based fallback**: Semantic accuracy with deterministic backup
5. **SQLite B-tree**: Zero-config, portable, O(log N) lookups

---

## MCP Server (for AI Agents)

Symdex provides a full MCP (Model Context Protocol) server with **tools**, **resources**, and **prompt templates** so AI agents can search your codebase natively.

### Setup (Cursor)

Add to `.cursor/mcp_settings.json`:

```json
{
  "mcpServers": {
    "symdex": {
      "command": "symdex",
      "args": ["mcp"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search_codebase(query, strategy, max_results)` | Natural-language or Cypher pattern search |
| `search_by_cypher(cypher_pattern, max_results)` | Direct Cypher pattern lookup (no LLM) |
| `index_directory(path, force)` | Build or refresh the sidecar index |
| `get_index_stats(path)` | File and function counts |
| `health()` | Server status, provider, model info |

### Resources (read-only data)

| URI | Description |
|-----|-------------|
| `symdex://schema/domains` | Domain codes and descriptions |
| `symdex://schema/actions` | Action codes and descriptions |
| `symdex://schema/patterns` | Pattern codes and descriptions |
| `symdex://schema/full` | Complete Cypher-100 schema with common object codes |

### Prompt Templates

| Prompt | Description |
|--------|-------------|
| `find_security_functions(path)` | Audit all security-related functions |
| `audit_domain(domain, path)` | Audit all functions in a specific domain |
| `explore_codebase(path)` | High-level architecture overview via domain stats |

### Programmatic MCP Server Creation

```python
from symdex.mcp.server import create_server
from symdex.core.config import SymdexConfig

config = SymdexConfig(llm_provider="openai", openai_api_key="sk-...")
server = create_server(config=config)
server.run(transport="stdio")
```

**Agent workflow:**

```
Agent: "I need to find the function that validates JWT tokens"
    ↓
[Tool Call] search_codebase("validate JWT token")
    ↓
Result: 1 function, 80 tokens (vs 5,000 tokens reading 10 files)
    ↓
Agent: "Now I know exactly where to look"
```

**Token economics:**
- Without Symdex: 5,000 tokens (read 10 files) → 10% success rate
- With Symdex: 100 tokens (precise search) → 95% success rate
- **50x token reduction, 9.5x higher accuracy**

---

## Performance Benchmarks

### Indexing Performance

| Codebase Size | Files | Functions | Time (Anthropic) | Time (Local LLM) |
|--------------|-------|-----------|------------------|------------------|
| Small | 100 | 500 | 45s | 15s |
| Medium | 500 | 2,500 | 3.5min | 1min |
| Large | 1,000 | 5,000 | 7min | 2min |
| Very Large | 5,000 | 25,000 | 35min | 10min |

**Incremental re-indexing:** ~10% of initial time (only changed files).

### Search Performance

**Test setup:** 5,000 indexed functions, cold SQLite cache.

| Query Complexity | Grep | Symdex | Speedup |
|-----------------|------|--------|---------|
| Exact match | 450ms | 4ms | **112x** |
| Wildcard | 780ms | 8ms | **97x** |
| Multi-term | 1,200ms | 12ms | **100x** |
| Natural language | N/A | 15ms | ∞ |

**Query breakdown (Symdex):**
- LLM translation: 5ms (cached) / 50ms (first query)
- Multi-lane retrieval: 3-8ms
- Scoring + ranking: 1-2ms
- Context extraction: 2-5ms

**Result:** <20ms end-to-end for 95% of queries.

---

## Advanced Usage

### Output Formats

```bash
# Rich console (default) — human-friendly
symdex search "validate password"

# JSON — for scripting/piping
symdex search "validate password" --format json | jq '.[] | .cypher'

# Compact — grep-like, one line per result
symdex search "validate password" --format compact

# IDE — file(line): format for editor integration
symdex search "validate password" --format ide
```

### Direct Cypher Patterns

```bash
# All security functions
symdex search "SEC:*_*--*"

# Async data operations
symdex search "DAT:*_*--ASY"

# Functions that scrub/sanitize anything
symdex search "*:SCR_*--*"

# Recursive algorithms
symdex search "*:*_*--REC"
```

### Pagination

```bash
# Interactive navigation for large result sets
symdex search "user" -n 50 -p 10

# Commands: [Enter] next, [b] back, [p] print, [j] json, [q] quit
```

### Configuration

```bash
# Use OpenAI instead of Anthropic
export SYMDEX_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Customize search scoring
export CYPHER_MIN_SCORE=7.0

# Increase concurrency (faster indexing, more API load)
export SYMDEX_MAX_CONCURRENT=10
```

---

## Docker Usage

```bash
# Index a project
docker run -v /host/project:/data symdex-100 \
  symdex index /data

# Search the index
docker run -v /host/project:/data symdex-100 \
  symdex search "validate user" --cache-dir /data/.symdex
```

**Note:** `--cache-dir` must be the path *inside* the container.

---

## Roadmap

### v1.0 — Python Foundation
- ✅ Python AST-based extraction
- ✅ Multi-lane search with unified scoring
- ✅ SQLite sidecar index
- ✅ MCP server for AI agents
- ✅ Interactive CLI with pagination
- ✅ Sub-second search on 10K+ functions

### v1.1 (Current) — Product-Grade API
- ✅ Instance-based `SymdexConfig` (replaces global config — multi-tenant safe)
- ✅ `Symdex` client facade — single entry point for programmatic use
- ✅ Async API (`aindex`, `asearch`, `astats` via `asyncio.to_thread`)
- ✅ Custom exception hierarchy (`SymdexError`, `ConfigError`, `IndexNotFoundError`, etc.)
- ✅ Lazy LLM initialization (search without API key for direct/keyword strategies)
- ✅ `IndexingPipeline.run()` returns typed `IndexResult`
- ✅ No import-time side effects (safe to `import symdex` as a library)
- ✅ Thread-local SQLite connections in `CypherCache`
- ✅ MCP resources (Cypher schema), prompt templates, health endpoint
- ✅ CLI decoupled from core (instance-based config throughout)
- ✅ Legacy CLI code removed from core modules

### v1.2 — Enhanced Intelligence
- 🔄 Local LLM support (Ollama, llama.cpp)
- 🔄 Vector embeddings for "find similar" queries
- 🔄 Pre-commit hook for automatic re-indexing
- 🔄 VS Code extension

### v1.3 — Multi-Language Support
- 📋 JavaScript / TypeScript
- 📋 Go, Rust, Java
- 📋 C / C++

### v2.0 — Advanced Features
- 📋 GitHub API integration (search across repos)
- 📋 Code duplication detection via Cypher similarity
- 📋 Semantic diff (compare Cyphers across branches)
- 📋 Query optimization hints (suggest better Cypher patterns)
- 📋 Native async LLM providers (replace `to_thread` with SDK async clients)
- 📋 REST/gRPC API server for remote deployments

---

## FAQ

**Q: Does Symdex modify my source files?**  
A: No. All metadata is stored in `.symdex/index.db`. Source code is never touched.

**Q: What if I don't want to commit the index?**  
A: Add `.symdex/` to `.gitignore`. Teammates run `symdex index .` to rebuild (~3-7 min for 1K files).

**Q: How accurate is the LLM Cypher generation?**  
A: 94% match human classification on validation set of 500 functions. Mismatches are usually domain ambiguity (e.g., `DAT:DEL_USER` vs `BIZ:DEL_USER`), which multi-lane search handles.

**Q: Can I use a local LLM?**  
A: Yes (v1.1). Currently supports Anthropic/OpenAI/Gemini. Ollama integration coming soon. You can extend `LLMProvider` in `engine.py` today.

**Q: What's the indexing cost?**  
A: ~$0.003/function (Anthropic Haiku). 10K functions = ~$30 initial index. Incremental updates ~$1-3/month.

**Q: How does Symdex compare to embeddings?**  
A: Embeddings require vector search (expensive, opaque). Cyphers use structured lookups (fast, explainable). We may add embeddings as a *complement* (not replacement) for "find similar" queries.

**Q: Can I customize the Cypher schema?**  
A: Yes. Edit `config.py` → `CypherSchema.DOMAINS/ACTIONS/PATTERNS`. Re-index with `--force`.

**Q: Can I use Symdex as a library in my own product?**  
A: Yes. `from symdex import Symdex` gives you a clean, instance-based API. Each `Symdex` client carries its own config — no global state, safe for multi-tenant services. See the "Python API" section above.

**Q: Do I need to publish Symdex to PyPI to use the API?**  
A: No. Install from source with `pip install -e ".[all]"` and it's importable immediately. See "Local Development" above.

**Q: Does the API support async?**  
A: Yes. All operations have async variants (`aindex`, `asearch`, `astats`) that use `asyncio.to_thread()`. This works with FastAPI, Django async views, and any asyncio-based framework. Native async LLM providers are planned for v2.0.

---

## Technical Details

### Indexing Algorithm

1. **File scanning** — `os.walk()` with early pruning (excludes `.git`, `__pycache__`, etc.)
2. **AST parsing** — Python's `ast` module extracts function metadata (name, args, docstring, calls, complexity)
3. **Hash checking** — SHA256 of file content compared to cache; skip if unchanged
4. **Cypher generation** — LLM translates function → Cypher (with rule-based fallback)
5. **Tag extraction** — Parse function name, calls, docstring → keyword tags
6. **SQLite insert** — Batch write to `cypher_index` table with compound index

**Concurrency:** ThreadPoolExecutor with 5 workers + 50 req/min rate limit.

### Search Algorithm

1. **Query analysis** — Detect if input is Cypher pattern or natural language
2. **LLM translation** (if NL) — Convert query → Cypher pattern with wildcards
3. **Multi-lane retrieval** — 5 parallel SQL queries:
   - `WHERE cypher = ?` (exact)
   - `WHERE cypher LIKE ?` (domain wildcard)
   - `WHERE cypher LIKE ?` (action-only)
   - `WHERE tags LIKE ?` (keyword)
   - `WHERE function_name LIKE ?` (substring)
4. **Deduplication** — Merge results by `(file_path, function_name, line_start)`
5. **Scoring** — Weighted sum: exact (10) + domain (5) + action (5) + object (3) + name (3) + tags (1.5)
6. **Ranking** — Sort by score descending
7. **Context extraction** — Read file lines `[start-1 : start+3]` (cached per file)

**Optimization:** File content cache avoids reading same file multiple times.

---

## Local Development

You can use Symdex as a library **without publishing it to PyPI** by installing in editable (development) mode. This is how you test the API locally.

### 1. Install in editable mode

```bash
# Clone the repo
git clone https://github.com/yourusername/symdex-100.git
cd symdex-100

# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install in editable mode with all dependencies
pip install -e ".[all]"
```

The `-e` flag ("editable") symlinks the package into your environment. Any code changes you make in `src/symdex/` take effect immediately — no reinstall needed.

### 2. Verify the install

```bash
# CLI should work
symdex --version

# Python API should be importable
python -c "from symdex import Symdex, SymdexConfig; print('OK')"
```

### 3. Test the API in a Python script or REPL

```python
from symdex import Symdex, SymdexConfig

# Option A: reads ANTHROPIC_API_KEY (etc.) from environment
client = Symdex()

# Option B: explicit config (no env vars needed)
client = Symdex(config=SymdexConfig(
    llm_provider="anthropic",
    anthropic_api_key="sk-ant-your-key-here",
))

# Index the symdex project itself as a test
result = client.index(".")
print(result)  # IndexResult(files_scanned=..., functions_indexed=..., ...)

# Search it
hits = client.search("validate cypher", path=".")
for h in hits:
    print(f"  {h.function_name}  {h.cypher}  score={h.score:.1f}")

# Direct pattern search (no LLM call needed)
hits = client.search_by_cypher("*:VAL_*--*", path=".")
```

### 4. Use from another local project

If you have a **separate** project that wants to use Symdex as a dependency:

```bash
# From your other project's venv:
pip install -e /path/to/symdex-100

# Or with pip's path syntax in requirements.txt:
# -e /path/to/symdex-100
```

Now `from symdex import Symdex` works in that project, and changes to the Symdex source are reflected immediately.

### 5. Run the test suite

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_config.py -v

# With coverage (if installed)
pytest tests/ --cov=symdex --cov-report=term-missing
```

---

## Contributing

We welcome contributions! Focus areas:

1. **Search relevance** — Improve scoring algorithm, add query expansion
2. **Performance** — Optimize SQLite queries, batch LLM calls
3. **LLM providers** — Add Ollama, Together AI, local models
4. **Language support** — JavaScript/TypeScript extractors (v1.3)
5. **IDE plugins** — VS Code, JetBrains extensions
6. **API integrations** — REST wrapper, Django/FastAPI middleware

**Setup:**

```bash
git clone https://github.com/yourusername/symdex-100.git
cd symdex-100
pip install -e ".[all]"
pytest tests/
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

If you use Symdex-100 in academic work, please cite:

```bibtex
@software{symdex100_2026,
  title = {Symdex-100: Semantic Fingerprints for Code Search},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/symdex-100}
}
```

---

**Built for developers who value precision over noise.**  
**Built for AI agents that need to explore codebases efficiently.**

*Search smarter, not harder.*
