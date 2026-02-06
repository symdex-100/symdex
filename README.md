# Symdex-100: Inline Semantic Fingerprints for 100x Faster Code Search

Symdex-100 is a **language-agnostic** system that embeds compact, structured **semantic fingerprints** ("Cyphers") directly into source code, enabling **100x faster, intent-based code search** for humans and AI agents alike.

```
# <SEARCH_META v1.0>
# CYPHER: SEC:VAL_TOKEN--SYN
# SIG: [token] → bool
# TAGS: #validate, #token, #security
# COMPLEXITY: O(1)
# </SEARCH_META>
def validate_token(token: str) -> bool: ...
```

Instead of reading 2 KB of code per function, agents (and developers) search **20 bytes of metadata**.

---

## Why this exists

| Tool | Weakness |
|------|----------|
| **grep / IDE search** | No understanding of *intent* |
| **Sourcegraph / LSIF / ctags** | Searches the **full code graph** |
| **Embeddings / vector DBs** | Opaque, infra-heavy, query-time expensive |

Symdex-100 takes a different route:

- **LLM-generated, structured taxonomy embedded inline** — each function gets a `DOM:ACT_OBJ--PAT` fingerprint in a `SEARCH_META` comment.
- **100:1 search target reduction** — search 20–40 bytes of metadata, not the raw source.
- **Natural language → structured query → SQLite** — "where do we validate tokens" becomes `SEC:VAL_TOKEN--*`, resolved in < 10 ms.
- **Metadata travels with the code** — clone the repo, get the index for free. No external service required.

---

## The Cypher format: `DOM:ACT_OBJ--PAT`

| Slot | Len | Meaning | Examples |
|------|-----|---------|----------|
| **DOM** | 2–3 | Domain | `SEC` (Security), `DAT` (Data), `NET` (Network), `LOG` (Logging), `UI`, `BIZ`, `SYS`, `TST` |
| **ACT** | 3 | Action | `VAL` (Validate), `FET` (Fetch), `TRN` (Transform), `CRT` (Create), `SND` (Send), `SCR` (Scrub), `UPD`, `AGG`, `FLT`, `SYN` |
| **OBJ** | 2–12 | Object | `USER`, `TOKEN`, `DATASET`, `CONFIG`, `LOGS`, `REQUEST`, `JSON`, `CSV`, `CACHE` … |
| **PAT** | 3 | Pattern | `ASY` (async), `SYN` (sync), `REC` (recursive), `GEN` (generator), `DEC`, `CTX`, `CLS` |

**Example mappings:**

| Code | Cypher | Explanation |
|------|--------|-------------|
| `async def send_email(...)` | `NET:SND_EMAL--ASY` | Network, send, email, async |
| `function validateToken(t)` | `SEC:VAL_TOKEN--SYN` | Security, validate, token, sync |
| `func FetchRecords(limit)` | `DAT:FET_RECORD--SYN` | Data, fetch, records, sync |
| `pub async fn render(node)` | `UI:TRN_NODE--ASY` | UI, transform, node, async |

The taxonomy is **finite and interpretable** — every result is explainable and debuggable.

---

## Multi-language comment styles

The `SEARCH_META` block uses the **native comment syntax** of each language:

```python
# <SEARCH_META v1.0>        ← Python / Ruby
# CYPHER: DAT:FET_DATASET--SYN
# </SEARCH_META>
```

```js
// <SEARCH_META v1.0>       ← JS / TS / Java / Go / Rust / C / C++ / C# / Swift / Kotlin / PHP
// CYPHER: SEC:VAL_TOKEN--ASY
// </SEARCH_META>
```

Supported languages: **Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#, Ruby, PHP, Swift, Kotlin**.

---

## Quick start

### Install

```bash
pip install symdex-100
```

Or with MCP server support:

```bash
pip install 'symdex-100[mcp]'
```

### Set your API key

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### Index a project

```bash
symdex index ./my-project
```

### Search

```bash
# Natural language
symdex search "where do we validate user tokens"

# Direct Cypher pattern
symdex search "SEC:VAL_TOKEN--*"

# JSON output (for piping / scripting)
symdex search "async email functions" --format json
```

### Docker

The image expects the project to index (or the directory that contains `.symdex`) to be **mounted**; paths are resolved **inside** the container.

**Index** a project on the host (e.g. `E:\CodeDD`):

```bash
docker compose run -v E:/CodeDD:/data symdex symdex index /data --cache-dir /data/.symdex
```

**Search** that index from the same host:

```bash
docker compose run -v E:/CodeDD:/data symdex symdex search "where do we define deletion of source code data" --cache-dir /data/.symdex
```

`--cache-dir` must be the path **inside the container** (e.g. `/data/.symdex`), not the host path (`E:/CodeDD/.symdex`). Use the same `-v` mount so the container can read the index.

### Check stats

```bash
symdex stats --cache-dir ./my-project
```

---

## MCP server (for Cursor / Claude / Windsurf)

Start the MCP server so AI agents can call Symdex natively:

```bash
symdex mcp
```

This exposes four tools via the Model Context Protocol:

| Tool | What it does |
|------|-------------|
| `search_codebase` | NL or Cypher search — returns matching functions with file, line, score, preview |
| `search_by_cypher` | Direct Cypher pattern search (no LLM translation) |
| `index_directory` | Index all supported source files in a directory |
| `get_index_stats` | Return index size and counts |

### Cursor integration

Add to your Cursor MCP settings:

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

**Why agents love this:** Instead of reading 10 files to find a login function (5,000 tokens), the agent calls `search_codebase("SEC:VAL_*--*")` and gets the 1 exact match (100 tokens). That's a **50x context window savings**.

---

## Project structure

```
symdex-100/
├── src/
│   └── symdex/
│       ├── core/              # The Logic
│       │   ├── config.py      # Config, CypherSchema, LanguageRegistry, Prompts
│       │   ├── engine.py      # CodeAnalyzer, CypherCache, CypherGenerator, scoring
│       │   ├── indexer.py     # IndexingPipeline, FileModifier
│       │   └── search.py     # SearchEngine, ResultFormatter, interactive mode
│       ├── cli/               # The Tool
│       │   └── main.py        # Click CLI: symdex index | search | stats | mcp
│       └── mcp/               # The Bridge
│           └── server.py      # FastMCP server with 4 agent-callable tools
├── tests/                     # 150+ tests across config, core, indexer
├── pyproject.toml             # Unified build — pip install symdex-100
├── ARCHITECTURE.md            # Detailed design & trade-offs
├── AGENTS.md                  # How AI agents should use Symdex
└── README.md                  # ← You are here
```

---

## How it works

### Indexer (`symdex index`)

For each supported source file:

1. **Language-aware parse** — AST for Python; regex patterns via `LanguageRegistry` for 12 other languages.
2. **LLM call** — Claude generates the `DOM:ACT_OBJ--PAT` Cypher using a strict schema prompt.
3. **Tag generation** — from function name parts, calls, docstrings, async pattern.
4. **Complexity estimation** — `O(1)`, `O(N)`, `O(N²)`, `O(N³+)`.
5. **`SEARCH_META` injection** — inserted above each function with the correct comment prefix.
6. **SQLite cache update** — file hash, cypher, tags, signature, complexity.

Incremental by default: SHA256 hashes skip unchanged files. Concurrent: configurable worker count.

### Search (`symdex search`)

1. **NL → Cypher translation** (LLM with rule-based fallback).
2. **SQLite lookup** — exact or wildcard pattern match.
3. **Progressive fallback cascade** — broaden pattern until results found.
4. **Parallel tag search** — merge unique hits from tag index.
5. **Ranking** — domain/action/object/pattern match, object similarity, tag overlap, function name overlap.
6. **Context extraction** — read only the relevant lines for the top N hits.

---

## Configuration

Key settings in `symdex.core.config.Config`:

```python
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
TARGET_EXTENSIONS = (".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".cs", ".rb", ".php", ".swift", ".kt")
EXCLUDE_DIRS = ("__pycache__", ".git", "node_modules", "dist", "build", "target", ...)
SEARCH_RANKING_WEIGHTS = {
    "exact_match": 10.0, "domain_match": 5.0, "action_match": 5.0,
    "object_match": 3.0, "object_similarity": 2.0, "pattern_match": 2.0,
    "tag_match": 1.5, "name_match": 2.0,
}
```

The `LanguageRegistry` maps every supported extension to its comment style, function-detection regexes, and body-boundary strategy.

---

## Testing efficiency

To measure the 100x speedup claim:

```bash
# 1. Index a large project
symdex index /path/to/large-project --force

# 2. Benchmark: grep vs symdex
time grep -rn "validate.*password" /path/to/large-project
time symdex search "validate password" --cache-dir /path/to/large-project --format compact

# 3. Measure token savings (for agents)
# Without Symdex: agent reads N files × avg 200 lines = N×200 lines of context
# With Symdex:    agent gets K results × 5 lines each  = K×5 lines of context
# Token reduction: (N×200) / (K×5) ≈ 50-100x
```

See the **Efficiency Testing** section in `ARCHITECTURE.md` for detailed benchmarks.

---

## Contributing

1. Extend the `CypherSchema` with new domains / actions / objects.
2. Register new languages in `LanguageRegistry` (data-only, no code changes needed).
3. Improve prompts or ranking for edge cases.
4. Integrate Symdex-100 with your IDE, CI pipeline, or agent framework.

---

## License

MIT License — see `LICENSE` for details.

---

**Built for developers who spend too much time searching and not enough time shipping.**
**Built for agents who spend too many tokens reading and not enough tokens solving.**
