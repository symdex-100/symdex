# AGENTS.md — How AI Agents Should Use Symdex-100

This file provides opinionated guidance for AI agents (Claude, Cursor,
Windsurf, Copilot, etc.) on **when and how** to use the Symdex-100
tools for code navigation and understanding.

---

## What is Symdex-100?

Symdex-100 embeds **semantic fingerprints** ("Cyphers") into source files
as `SEARCH_META` comment blocks.  Each function gets a compact
`DOM:ACT_OBJ--PAT` string that encodes **what the function does**, not
just what it's called.

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
   first. If the index exists, prefer Symdex over raw file reading.

### DO NOT use Symdex when:

1. **You already know the exact file and line** — just read it directly.
2. **You're searching for a specific string/identifier** — use grep or
   IDE search instead.
3. **The project has not been indexed** — call `index_directory` first,
   or fall back to normal code exploration.

---

## Available tools (MCP)

If Symdex is configured as an MCP server, you have these tools:

### `search_codebase(query, path=".", strategy="auto", max_results=10)`

**Your primary tool.** Pass a natural-language query or Cypher pattern.

```
# Natural language → translated to Cypher automatically
search_codebase("where do we validate user tokens")

# Direct Cypher pattern (if you already know the shape)
search_codebase("SEC:VAL_TOKEN--*")

# Broad exploration
search_codebase("DAT:*_*--*")  → all Data-domain functions
```

**Returns:** JSON array of `{function_name, file_path, line_start, line_end, cypher, score, context}`.

### `search_by_cypher(cypher_pattern, path=".", max_results=10)`

Direct pattern search — no LLM translation. Faster, deterministic.

```
search_by_cypher("SEC:VAL_*--SYN")   → all sync security validation
search_by_cypher("NET:SND_*--ASY")   → all async network send functions
search_by_cypher("LOG:CRT_LOGS--*")  → logging setup functions
```

### `index_directory(path=".", force=False)`

Index a directory. Call this if `get_index_stats` returns an error or
shows 0 indexed files.

### `get_index_stats(path=".")`

Returns `{indexed_files, indexed_functions}`. Use to check if indexing
has been done.

---

## Recommended workflow for agents

```
1. Check:   get_index_stats(".")
             → If no index exists: index_directory(".")
             → If index exists: proceed to search

2. Search:  search_codebase("your intent query")
             → Returns ranked results with file, line, score, preview

3. Read:    Open the top result's file at the specific line
             → You now have precise context, not a haystack

4. Act:     Make your edit / answer the question / generate the code
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

Use `*` for any slot you're unsure about. The search engine will rank
results by relevance.

---

## Token economics

| Approach | Tokens used | Accuracy |
|----------|------------|----------|
| Read 10 files to find a function | ~5,000 | Low (might miss it) |
| `search_codebase("validate password")` | ~100 | High (ranked results) |
| **Savings** | **50x fewer tokens** | **Higher accuracy** |

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

**Remember:** Symdex doesn't replace your ability to read code. It tells
you **which code to read**. Use it as a compass, not a map.
