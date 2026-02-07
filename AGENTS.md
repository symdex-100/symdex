# AGENTS.md — How AI Agents Should Use Symdex-100

This file provides opinionated guidance for AI agents (Claude, Cursor,
Windsurf, Copilot, etc.) on **when and how** to use the Symdex-100
tools for code navigation and understanding.

---

## What is Symdex-100?

Symdex-100 indexes source code into a **sidecar SQLite database**
(`.symdex/index.db`).  Each function gets a compact semantic
fingerprint called a **Cypher** — a structured `DOM:ACT_OBJ--PAT`
string that encodes **what the function does**, not just what it's
called.  Source files are **never modified**.

**The Cypher format:**

```
DOM:ACT_OBJ--PAT

DOM = Domain    (SEC, DAT, NET, SYS, LOG, UI, BIZ, TST)
ACT = Action    (VAL, TRN, SND, FET, SCR, CRT, UPD, AGG, FLT, SYN)
OBJ = Object    (USER, TOKEN, CONFIG, DATASET, LOGS, REQUEST, JSON, ...)
PAT = Pattern   (ASY, SYN, REC, GEN, DEC, CTX, CLS)
```

**Example:** `SEC:VAL_TOKEN--ASY` means "Security domain, validates a
token, async pattern."

---

## When to use Symdex

### USE Symdex when:

1. **You need to find code by intent** — "where do we validate user
   passwords", "find the logging setup function", "which function
   transforms the CSV data".

2. **You want to reduce context window usage** — instead of reading 10
   files (5,000+ tokens), call `search_codebase` and get 1–3 precise
   hits (100–300 tokens).

3. **You need to understand the codebase structure** — call
   `get_index_stats` to know the scale, then `search_codebase` with
   broad patterns like `SEC:*_*--*` to map out all security functions.

4. **The codebase has been indexed** — check with `get_index_stats`
   first.  If the index exists, prefer Symdex over raw file reading.

### DO NOT use Symdex when:

1. **You already know the exact file and line** — just read it directly.
2. **You're searching for a specific string/identifier** — use grep or
   IDE search instead.
3. **The project has not been indexed** — call `index_directory` first,
   or fall back to normal code exploration.

---

## Available MCP tools

If Symdex is configured as an MCP server, you have these tools:

### `search_codebase(query, path=".", strategy="auto", max_results=10)`

**Your primary tool.** Pass a natural-language query or Cypher pattern.

- **Natural language:** The LLM returns **three** Cypher patterns (tight, medium, broad). The engine tries the tight pattern first and only broadens if needed, so you get fewer irrelevant results and faster, more focused hits. Results are ranked against the tight pattern.
- **Reported time** in tool output is **DB-only** (index lookup, scoring, context); LLM translation time is not included.

```
# Natural language → tiered Cypher (tight first)
search_codebase("where do we validate user tokens")

# Direct Cypher pattern (if you already know the shape)
search_codebase("SEC:VAL_TOKEN--*")

# Broad exploration
search_codebase("DAT:*_*--*")  → all Data-domain functions
```

**Returns:** JSON array of `{function_name, file_path, line_start, line_end, cypher, score, context}`.

### `search_by_cypher(cypher_pattern, path=".", max_results=10)`

Direct pattern search — no LLM translation.  Faster, deterministic.

```
search_by_cypher("SEC:VAL_*--SYN")   → all sync security validation
search_by_cypher("NET:SND_*--ASY")   → all async network send functions
search_by_cypher("LOG:CRT_LOGS--*")  → logging setup functions
```

### `index_directory(path=".", force=False)`

Index a directory.  Call this if `get_index_stats` returns an error or
shows 0 indexed files.

### `get_index_stats(path=".")`

Returns `{indexed_files, indexed_functions}`.  Use to check if indexing
has been done.

### `health()`

Server readiness check.  Returns version, status, and loaded
configuration details.  Call this to verify the MCP server is alive
before other operations.

---

## MCP Resources

The server exposes read-only resources that describe the Cypher schema.
Fetch these to understand what domains/actions/patterns exist **before**
constructing manual Cypher patterns.

| URI | Description |
|-----|-------------|
| `symdex://schema/domains` | All domain codes with descriptions |
| `symdex://schema/actions` | All action codes with descriptions |
| `symdex://schema/patterns` | All pattern codes with descriptions |
| `symdex://schema/full` | Complete schema (domains + actions + objects + patterns) |

**Usage:** Read `symdex://schema/full` once at the start of a session
to populate your context with the valid Cypher vocabulary.

---

## MCP Prompt Templates

Pre-built prompt templates for common agent tasks:

| Template | Arguments | What it does |
|----------|-----------|-------------|
| `find_security_functions` | `path` | Searches for all security-related functions and formats a summary |
| `audit_domain` | `domain`, `path` | Explores all functions in a given domain (e.g. `NET`, `DAT`) |
| `explore_codebase` | `path` | Generates a high-level overview of the codebase structure by domain |

---

## Python API (alternative to MCP)

If you're running inside a Python process (e.g., a custom agent or
script), you can use the Symdex client directly instead of going
through the MCP server:

```python
from symdex import Symdex, SymdexConfig

client = Symdex(config=SymdexConfig.from_env())

# Index a project
result = client.index("./project")

# Search by intent
hits = client.search("validate user tokens", path="./project")
for hit in hits:
    print(f"{hit['file_path']}:{hit['line_start']}  {hit['cypher']}")

# Get stats
stats = client.stats("./project")
```

Async variants are available: `client.aindex()`, `client.asearch()`,
`client.astats()`.

---

## Recommended workflow for agents

```
1. Health:  health()
             → Verify the MCP server is alive and configured

2. Schema:  Read symdex://schema/full
             → Learn the valid Cypher vocabulary

3. Check:   get_index_stats(".")
             → If no index exists: index_directory(".")
             → If index exists: proceed to search

4. Search:  search_codebase("your intent query")
             → Returns ranked results with file, line, score, preview

5. Read:    Open the top result's file at the specific line
             → You now have precise context, not a haystack

6. Act:     Make your edit / answer the question / generate the code
```

---

## Constructing Cypher patterns manually

If you understand the schema, you can build patterns directly for faster,
more precise results:

| You want to find... | Pattern |
|---------------------|---------|
| All security functions | `SEC:*_*--*` |
| Functions that validate anything | `*:VAL_*--*` |
| Async functions in any domain | `*:*_*--ASY` |
| Functions that fetch users | `*:FET_USER--*` |
| Data transformation functions | `DAT:TRN_*--*` |
| Logging setup / initialization | `LOG:CRT_*--*` |
| Network send functions (async) | `NET:SND_*--ASY` |
| All functions touching config | `SYS:*_CONFIG--*` |

Use `*` for any slot you're unsure about.  The search engine will rank
results by relevance.

---

## Token economics

| Approach | Tokens used | Accuracy |
|----------|------------|----------|
| Read 10 files to find a function | ~5,000 | Low (might miss it) |
| `search_codebase("validate password")` | ~100 | High (ranked, tiered precision) |
| **Savings** | **50x fewer tokens** | **Higher accuracy** |

Natural-language search uses **tiered Cypher** (tight → medium → broad) and a **candidate cap** so result sets stay focused (e.g. dozens of high-quality hits instead of hundreds of noisy ones).

**Rule of thumb:** If you would otherwise read more than 3 files to
answer a question, try Symdex first.

---

## Common domain codes (reference)

| Code | Domain |
|------|--------|
| `SEC` | Security / Authentication / Encryption |
| `DAT` | Data Processing / Storage / CRUD |
| `NET` | Networking / API / External Communication |
| `SYS` | OS / Filesystem / Memory / System |
| `LOG` | Logging / Observability / Monitoring |
| `UI`  | User Interface / Rendering / Display |
| `BIZ` | Business Logic / Domain Rules |
| `TST` | Testing / Validation / Quality Assurance |

---

**Remember:** Symdex doesn't replace your ability to read code.  It
tells you **which code to read**.  Use it as a compass, not a map.
