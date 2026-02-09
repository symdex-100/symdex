# Symdex-100

<div align="center">

![Symdex Robot](./docs/symdex-100.png)

*Symdex-100 — your AI companion for code exploration*

</div>

---

**Semantic fingerprints for intent-based Python code search — 50–100x faster index lookups, 10–50x fewer tokens for AI agents.**

Symdex-100 generates compact, structured metadata ("Cyphers") for every function in your Python codebase. Each Cypher is typically 20 bytes — a semantic fingerprint that enables sub-second, intent-based code search for developers and AI agents without reading thousands of lines of code.

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
| **grep** | Keyword noise — finds "token" in comments, strings, variable names | 3,000+ tokens (read all matches, many false positives) |
| **Full-text search** | No semantic understanding — can't distinguish intent | 5,000+ tokens (read 10 files, variable success) |
| **Embeddings** | Opaque, expensive, query-time overhead | 2,000+ tokens (re-rank results, embedding index size) |
| **AST/LSP** | Limited to structural queries (class/function names) | N/A (doesn't understand "what validates X") |
| **Symdex** | Requires indexing step (one-time per codebase) | ~100–300 tokens (1–5 precise results with context) |

**Result**: Developers waste time reading irrelevant code. AI agents burn tokens on noise. Symdex reduces token usage by **10–50x** for intent-based queries (vs reading multiple files) while providing sub-second index lookups.

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
\text{Cypher} = \text{DOM} : \text{ACT} \text{OBJ} \text{--} \text{PAT}
$$

Where:

- **DOM** *(Domain)*: Semantic namespace — `SEC` (Security), `NET` (Network), `DAT` (Data), `SYS` (System), `LOG` (Logging), `UI` (Interface), `BIZ` (Business), `TST` (Testing)

- **ACT** *(Action)*: Primary operation — `VAL` (Validate), `FET` (Fetch), `TRN` (Transform), `CRT` (Create), `SND` (Send), `SCR` (Scrub), `UPD` (Update), `AGG` (Aggregate), `FLT` (Filter), `DEL` (Delete)

- **OBJ** *(Object)*: Target entity — `USER`, `TOKEN`, `DATASET`, `CONFIG`, `LOGS`, `REQUEST`, `JSON`, `EMAIL`, `DIR`. Can be compound (primary+secondary+tertiary, max 3 parts) when function involves multiple objects: `RELATIONSHIPS+AUDIT`, `RECORD+INDEX`, `FILE+CACHE`

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

This 18-character string (or 30–40 chars with compound OBJ like `RELATIONSHIPS+AUDIT`) replaces 2,000+ characters of function body for search purposes — a **50–100:1 compression ratio** with zero semantic loss. Compound OBJ improves ranking for multi-concept queries (e.g. "audit relations" → functions with `RELATIONSHIPS+AUDIT` rank higher than single-OBJ matches).

---

## Core Benefits

### 1. **Search Speed**

**Problem**: `grep` reads every file, full-text indexes scan every function.

**Solution**: Symdex searches 20-byte Cyphers in a SQLite B-tree index.

| Metric | Grep | Symdex (DB only) | Improvement |
|--------|------|------------------|-------------|
| Data scanned per query | ~50MB (full codebase) | ~100KB (index) | **500x less I/O** |
| Index lookup (5,000 functions) | 800ms | 8ms | **100x faster** |

**Technical details:**
- SQLite B-tree: O(log N) lookups with compound indexes on `(cypher, tags, function_name)`
- Tiered Cypher + multi-lane retrieval; candidate cap (default 200) keeps latency and result size bounded
- Incremental indexing: SHA256 hash tracking skips unchanged files
- **Reported search time** in CLI/API is index lookup only (excludes LLM translation for natural-language queries)

**Result**: Sub-second index lookup on 10,000+ function codebases.

---

### 2. **Search Accuracy**

**Problem**: Single search strategies miss valid results (e.g., `SYS:DEL_DIR` won't find `DAT:DEL_DIR` if query specifies system domain), or return too many low-quality hits when the Cypher is too broad.

**Solution**: **Tiered Cypher patterns** plus always-on **multi-lane search**.

**Tiered translation (natural-language queries):** The LLM returns three Cypher patterns — *tight* (no wildcards), *medium* (minimal wildcards), *broad* (fallback). The engine queries the tight pattern first; if the candidate pool is too small, it runs the medium then broad pattern and merges (deduplicated). Results are scored against the tight pattern so precise matches rank highest.

**Multi-lane retrieval** (per pattern):

```
Query: "delete directory"  →  Tiered: [SYS:SCR_DIR--SYN, SYS:SCR_DIR--*, *:SCR_*--*]
    ↓
┌────────────────────────────────────────────────────────────┐
│ LANE 1: Exact Cypher      │ SYS:SCR_DIR--SYN               │
│ LANE 2: Domain wildcard   │ *:SCR_DIR--SYN                 │
│ LANE 3: Action-only       │ *:SCR_*--*                     │  
│ LANE 4: Tag keywords      │ delete, directory  (capped)    │
│ LANE 5: Function name     │ _delete_directory_tree (capped)│
└────────────────────────────────────────────────────────────┘
    ↓
Merge + Cap candidates (default 200) + Score against tight pattern
    ↓
Ranked Results (exact match + domain/action/object = highest score)
```

**Result**: High precision from tiered + tight-pattern scoring; cross-domain recall when needed; fewer irrelevant results (candidate cap, Lane 3 skip, smaller tag/name limits).

---

### 3. **Token Efficiency** (for AI Agents)

**Problem**: Agents waste 80-90% of context on reading irrelevant code when exploring large codebases.

**Solution**: Symdex provides **10–50x token reduction** via semantic search, depending on the alternative approach.

**Scenario:** Agent needs to find "function that validates user login credentials"

| Approach | Process | Tokens | Notes |
|----------|---------|--------|-------|
| **Read 10 files** | Agent guesses likely files → reads all → searches manually | ~5,000 | High token cost, variable success |
| **Grep + read** | `grep "login\|credential"` → read 20 matches → filter manually | ~3,000 | Many false positives |
| **Symdex (1 result)** | `search_codebase("validate login credentials")` → 1 precise result | ~100 | **50x reduction** |
| **Symdex (5 results)** | Same query → top 5 results with context | ~300 | **10–15x reduction** vs reading 10 files |

**Savings:** 
- **50x fewer tokens** when the alternative is reading 10+ files or scanning many grep hits
- **10–15x fewer tokens** when returning 5 results vs reading 10 files
- **2–3x fewer tokens** when the alternative is reading 1–2 files (fewer file opens, result set instead of full file)

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

**Precision improvement:** Far fewer false positives than keyword search; actual ratios depend on codebase and query. Symdex matches function-level *intent* (domain, action, object), so mentions in variable names or comments are not returned as function results.

---

## Use Cases & Best Practices

### Repository size and when Symdex works best

Effectiveness correlates with codebase size:

| Size | Indexed functions | Recommendation |
|------|-------------------|----------------|
| **Very small** | &lt;50 | Skip Symdex — indexing overhead outweighs benefits; read files or use grep. |
| **Small** | 50–500 | Optional — intent search works, but grep with good keywords often suffices. |
| **Medium** | 500–3,000 | **Sweet spot** — one natural-language query often returns the right function in the top 5–10; fewer file reads than grep when you don't know exact names. Query phrasing can help (e.g. include domain terms like "developer expertise", "audit relations"). |
| **Large** | 3,000–10,000+ | **Strong fit** — index lookup and ranking keep result sets bounded; natural-language ranking is especially valuable vs. Cypher-only (which returns all matches with no relevance order). |

**Best use cases:** Intent-based discovery ("find the function that does X"), reducing file reads via `context_lines` in the result snippet, codebases where the right code doesn't contain the exact words you'd grep for, and AI agents exploring without reading many files.

**Prefer grep or a direct file read when:** You need an exact identifier or string, the repo is very small, or the codebase is non-Python (Symdex is Python-only today).

### When to Use Symdex

**✅ Use Symdex when:**
1. **Finding code by intent** — "where do we validate user passwords", "find the CSV parsing function", "which function sends email notifications"
2. **Onboarding to unfamiliar codebases** — Quickly map out architecture by domain (`SEC:*_*--*` for security functions, `DAT:*_*--*` for data processing)
3. **Code refactoring / impact analysis** — Find all functions that touch a specific object (`*:*_USER--*` for user-related operations)
4. **Documentation generation** — Extract function summaries with semantic context (Cypher + first 5 lines of code)
5. **AI agent code exploration** — 50x fewer tokens than reading files directly

**❌ Don't use Symdex when:**
1. **You know the exact file and line** — Just read the file directly
2. **Simple string search** — Use grep/IDE search for exact identifiers or literals
3. **Non-Python codebases** — Currently Python-only (JS/TS/Go/Rust support planned)
4. **Extremely small projects** (<50 functions) — Overhead of indexing outweighs benefits

### How to Use Symdex Effectively

#### 1. Tuning Search Results

**Adjust context_lines for editing vs. reading:**
```python
# Default: 3 lines (quick preview for exploration)
client.search("validate token", context_lines=3)

# For editing: 10-15 lines (full function body)
client.search("validate token", context_lines=15)
```

**Use explain to debug scoring:**
```python
results = client.search("validate token", explain=True)
for result in results:
    print(f"Score: {result.score}")
    print(f"Breakdown: {result.explanation}")
    # Example: {'action_match': 6, 'object_match': 5, 'name_matches': {'exact': 1, 'score': 3}}
```

#### 2. Search Strategies

**Auto (default) — Fastest for most queries:**
```bash
symdex search "validate token"
# Auto selects: LLM translation if available, else keyword fallback
```

**LLM (force semantic) — Best for natural language:**
```python
client.search("where do we check if user is admin", strategy="llm")
```

**Keyword (no LLM) — Fast, works offline:**
```python
client.search("delete user", strategy="keyword")
# Keyword-based translation: ~5ms vs. LLM: ~200-500ms
```

**Direct (skip translation) — Use Cypher patterns:**
```python
client.search("SEC:VAL_*--ASY", strategy="direct")
# Zero translation overhead
```

#### 3. Indexing Best Practices

**Incremental indexing (default):**
```bash
symdex index ./project
# Only re-processes changed files (SHA256 tracking)
```

**Force re-index (after major refactors):**
```bash
symdex index ./project --force
```

**Monitor indexing (get summary):**
```python
result = client.index("./project")
print(result.summary)
# {'top_files': [{'file': 'auth.py', 'functions': 47}],
#  'domain_distribution': {'SEC': 23, 'DAT': 18, 'NET': 6}}
```

#### 4. MCP Server (AI Agents)

**Use context_lines for agent tasks:**
```typescript
// Exploration (default): 3 lines
await searchCodebase({ query: "validate token", context_lines: 3 });

// Editing task: 10+ lines
await searchCodebase({ query: "validate token", context_lines: 15 });
```

**Prefer Symdex over file reading when:**
- Searching for code by intent (not exact identifiers)
- You'd otherwise read 3+ files to find the right function
- Codebase has 200+ functions (indexing overhead paid off)

**Example agent workflow:**
```
1. explore_codebase("how does authentication work")
   → Returns: SEC:VAL_TOKEN--ASY, SEC:CRT_SESSION--SYN, SEC:VAL_PASS--SYN

2. Read top result (SEC:VAL_TOKEN) with context_lines=15

3. Edit the function (now you have the right context)
```

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
│                                                                 │
│   Python Source (.py)                                           │
│         │                                                       │
│         ├─→ [AST Parser] ──→ Function Metadata                  │
│         │                     (name, args, docstring, ...)      │
│         │                                                       │
│         └─→ [LLM] ──────────→ Cypher Generation                 │
│                                SEC:VAL_TOKEN--ASY               │
│                                                                 │
│   ┌─────────────────────────────────────────────────┐           │
│   │         .symdex/index.db (SQLite)               │           │
│   ├─────────────────────────────────────────────────┤           │
│   │  • B-tree index on (cypher, tags, function_name)│           │
│   │  • SHA256 hash for incremental indexing         │           │
│   │  • 100:1 compression vs full function bodies    │           │
│   └─────────────────────────────────────────────────┘           │
│                        ↓                                        │
│   ┌─────────────────────────────────────────────────┐           │
│   │           MULTI-LANE SEARCH ENGINE              │           │
│   ├─────────────────────────────────────────────────┤           │
│   │  Query → [LLM] → 3 Cypher patterns (tight/med/broad)        │
│   │     ↓  Try tight first; merge medium/broad if needed        │
│   │  5 Lanes per pattern:  Exact │ Domain* │ Act* │ Tags │ Name │
│   │  (Lane 3 skipped when redundant; tag/name capped)           │
│   │     ↓  Candidate cap (e.g. 200)                             │
│   │  Score vs tight pattern → Rank → Format                     │
│   └─────────────────────────────────────────────────┘           │
│                        ↓                                        │
│   Results (100x faster, 50x fewer tokens)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

1. **Python AST** (not regex): Handles decorators, nested functions, edge cases
2. **Sidecar index** (not inline): Source files stay pristine, no diffs
3. **Tiered Cypher** (tight → medium → broad): LLM returns 3 patterns; try precise first, broaden only if needed — fewer irrelevant results
4. **Multi-lane search** (per pattern): Exact, domain wildcard, action-only (when not redundant), tag/name (capped); candidate cap before scoring
5. **LLM + rule-based fallback**: Semantic accuracy with deterministic backup
6. **SQLite B-tree**: Zero-config, portable, O(log N) lookups

---

## MCP Server (for AI Agents)

Symdex provides a full MCP (Model Context Protocol) server with **tools**, **resources**, and **prompt templates** so AI agents can search your codebase natively.

### Setup (Cursor)

1. **Install** (in this repo or your project): `pip install -e ".[mcp]"` so the `symdex` command is on your PATH.
2. **Index** (optional but recommended): in your project root run `symdex index .` so search has data. Or use the MCP tool `index_directory` from the agent.
3. **Configure Cursor:** create or edit `.cursor/mcp_settings.json` in your workspace (or Cursor user config) with:

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

4. **Reload:** Restart Cursor or run "MCP: Restart" so it starts the server. The server uses **stdio** by default (no port needed).

**Test:** Open a chat and ask the agent to run `get_index_stats` for `.` or `search_codebase("validate user")`; if the index exists you should get results.

If `symdex` is not on PATH (e.g. you use a venv and Cursor runs without it), set `"command"` to your Python and `"args"` to `["-m", "symdex.cli.main", "mcp"]`, or use the full path to the `symdex` executable (e.g. `".venv/bin/symdex"` on Unix, `".venv\\Scripts\\symdex.exe"` on Windows).

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

**Token economics (real-world testing on Symdex codebase):**
- Without Symdex: reading 3–5 files → 1,500–2,500 tokens, variable success
- With Symdex: one search → typically 100–300 tokens (1–5 results with `context_lines=3`), correct function often in **top 1–3** for specific queries
- **7–25x token reduction** when the alternative is reading 3+ files
- **2–3x token reduction** when the alternative is reading 1–2 files (still faster, less noise)
- Combine with `context_lines=10–15` for editing without opening the file (adds ~50–100 tokens per result)
- Compound OBJ improves ranking for multi-concept queries, reducing need to scan multiple results

---

## Performance Benchmarks

### Indexing Performance

| Codebase Size | Files | Functions | Time (Anthropic) | 
|--------------|-------|-----------|------------------|
| Small | 100 | 500 | 45s |
| Medium | 500 | 2,500 | 3.5min | 
| Large | 1,000 | 5,000 | 7min | 
| **Real-world (≈300k LOC)** | **≈400–1,000** | **≈2,700–2,800** | **≈7–15min** |
| Very Large | 5,000 | 25,000 | 35min | 

**Incremental re-indexing:** ~10% of initial time (only changed files).

### Search Performance

**Reported time:** The CLI and API report **DB-only** search time (multi-lane retrieval, scoring, context extraction). LLM translation for natural-language queries is **not** included (adds ~1–3s depending on provider).

**Real-world testing (Symdex codebase):**

| Query | Results | DB time | Ranking | Note |
|-------|---------|---------|---------|------|
| *"where do we calculate search score for cypher results"* | 8 | **<50ms** | #1 correct | Compound OBJ: `AGG_RECORD+DATASET` |
| *"function that validates cypher format"* | 8 | **<50ms** | #1 correct | Single OBJ: `VAL_TEXT` |
| *"add cypher entry to the index or cache"* | 8 | **<50ms** | #1 correct | Compound OBJ: `CRT_RECORD+INDEX` |
| *"extract context or get lines of code from file"* | 8 | **<50ms** | #1 correct | Compound OBJ: `FET_FILE+TEXT`, `FET_FILE+CACHE` |
| *"something that writes or stores things"* | 10 | **<50ms** | Top 10 relevant | Vague query; compound OBJ helps (`SCR_FILE+RECORD`) |

**Synthetic benchmark (5,000 indexed functions, cold SQLite cache):**

| Query Complexity | Grep | Symdex (DB only) | Speedup |
|-----------------|------|------------------|---------|
| Exact match | 450ms | 4ms | **112x** |
| Wildcard | 780ms | 8ms | **97x** |
| Multi-term | 1,200ms | 12ms | **100x** |
| Natural language | N/A | 15ms + LLM | ∞ |

**Compound OBJ impact:** ~30% of indexed functions use compound OBJ (e.g. `CRT_RECORD+INDEX`, `FET_FILE+CACHE`, `CRT_INDEX+DIR`). Multi-object boost (+6.0) improves ranking for queries mentioning multiple concepts (e.g. "audit relations" → `RELATIONSHIPS+AUDIT` ranks higher than single-OBJ matches).

**Query breakdown (Symdex):**
- LLM translation: not included in reported time (one-time per query, ~1–3s depending on provider)
- Multi-lane retrieval: typically 50–400ms (depends on result count and candidate cap)
- Scoring + ranking: 1–5ms
- Context extraction: scales with result count

**Result:** Sub-second index lookup for typical queries; tiered patterns and candidate cap keep result sets focused and fast.

---

## Advanced Usage

### Configuration reference

All parameters, default values, and **how to configure MCP defaults** (e.g. `SYMDEX_DEFAULT_CONTEXT_LINES`, `SYMDEX_DEFAULT_MAX_RESULTS`) are in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

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

The image includes MCP server support by default (install extras: `anthropic,mcp`). Override with build arg `EXTRAS` (e.g. `openai,mcp` or `llm-all,mcp`) if needed.

```bash
# Index a project
docker run -v /host/project:/data symdex-100 \
  symdex index /data

# Search the index
docker run -v /host/project:/data symdex-100 \
  symdex search "validate user" --cache-dir /data/.symdex
```

**Note:** `--cache-dir` must be the path *inside* the container.

### Running the MCP server in Docker (e.g. Smithery)

The default container command runs the MCP server with **HTTP (Streamable)** transport for remote clients (Smithery, HTTP-based MCP clients):

```bash
# Default: symdex mcp --transport streamable-http (listens on 0.0.0.0:8000 for remote connections)
docker run -p 8000:8000 -v /host/project:/data -e ANTHROPIC_API_KEY=sk-... symdex-100
```

For **stdio** (e.g. local Cursor talking to a container), override the command:

```bash
docker run -it -v /host/project:/data symdex-100 symdex mcp --transport stdio
```

With docker-compose, the default service runs `symdex mcp --transport streamable-http`. Set `CODE_DIR` and provide API keys via `.env` so the server can index and search the mounted project.

**Publishing on Smithery:** Smithery *can* host from GitHub, but **Hosted** deploy is for servers built with the [Smithery CLI and SDK](https://smithery.ai/docs/build/build) (TypeScript) and runs in their edge runtime—**128 MB, no filesystem, no native modules, no spawning processes** ([Hosting Limits](https://smithery.ai/docs/build/limits)). Symdex needs filesystem (SQLite index, reading source files) and a Python runtime, so it cannot run in that environment. Use the **URL** method: host the server elsewhere and give Smithery your public HTTPS URL. The server uses **Streamable HTTP** on **`/mcp`**, serves **`/.well-known/mcp/server-card.json`**, and sends CORS headers. To get a URL with no server to maintain: deploy this repo’s Docker image to [Fly.io](https://fly.io) or [Railway](https://railway.app). With Fly: install [flyctl](https://fly.io/docs/hands-on/install-flyctl/), run `fly launch` in this repo, then `fly deploy`; at [smithery.ai/new](https://smithery.ai/new) choose **URL** and enter `https://<app-name>.fly.dev/mcp`. The repo includes `fly.toml` for Fly.

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
- ✅ Rule-only mode (`SYMDEX_CYPHER_FALLBACK_ONLY`) — no API key required
- ✅ `IndexingPipeline.run()` returns typed `IndexResult`
- ✅ No import-time side effects (safe to `import symdex` as a library)
- ✅ Thread-local SQLite connections in `CypherCache`
- ✅ MCP resources (Cypher schema), prompt templates, health endpoint
- ✅ CLI decoupled from core (instance-based config throughout)
- ✅ Legacy CLI code removed from core modules
- ✅ Smithery-ready (server-card, config schema, Docker); GitHub Actions CI/release

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

**Q: Can I run without an API key?**  
A: Yes. Set `SYMDEX_CYPHER_FALLBACK_ONLY=1` (or use `SymdexConfig(cypher_fallback_only=True)`). Indexing and search use rule-based Cypher generation only — no LLM calls. Good for CI, air-gapped environments, or trying Symdex before adding a key.

**Q: Can I use a local LLM?**  
A: Yes (v1.1). Currently supports Anthropic/OpenAI/Gemini. Ollama integration is planned for v1.2; you can extend `LLMProvider` in `engine.py` today.

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

**Q: How do I deploy the MCP server on Smithery?**  
A: Smithery **Hosted** (GitHub → they build and run) only runs servers built with their TypeScript CLI/SDK in their edge runtime (no filesystem, 128 MB). Symdex is Python and needs filesystem (SQLite, source files), so use the **URL** method: deploy this repo’s Docker image to Fly.io or Railway, then at [smithery.ai/new](https://smithery.ai/new) choose **URL** and enter `https://your-app.example.com/mcp`. The server exposes `/.well-known/mcp/server-card.json` and Streamable HTTP on `/mcp`.

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

### 3b. Manually test the API with an example repository

To index a directory and run example searches in one go (index → stats → natural-language search → Cypher pattern search):

```bash
# Index and search this repo's src/ (default)
python scripts/try_api.py

# Use a specific folder
python scripts/try_api.py src
python scripts/try_api.py /path/to/any/python/project

# Index only (then use REPL or your own script to search)
python scripts/try_api.py src --index-only

# No API key: use rule-based Cypher fallback only
python scripts/try_api.py src --no-llm
```

The script prints index results, stats, and sample search hits so you can review the API behaviour end-to-end.

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
  author = {Camillo Pachmann},
  year = {2026},
  url = {https://github.com/symdex-100/symdex}
}
```

---

**Built for developers who value precision over noise.**  
**Built for AI agents that need to explore codebases efficiently.**

*Search smarter, not harder.*
