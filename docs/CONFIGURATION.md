# Symdex-100 — Parameters & Configuration Reference

All configurable parameters, their default values, and where they apply (Python API, CLI, MCP). Use this to tune behaviour and to set **default values for the MCP server** via environment variables.

---

## 1. Where parameters live

| Layer | Source of defaults |
|-------|---------------------|
| **SymdexConfig** | `src/symdex/core/config.py` — single source of truth for engine/indexer |
| **CLI** | `symdex index`, `symdex search`, etc. — options override config when passed |
| **MCP** | Tools use **SymdexConfig** (from env) when a tool param is omitted |
| **Python API** | `Symdex(config=SymdexConfig.from_env())` — same config, explicit args override |

So: **to configure MCP defaults**, set the environment variables below before starting the MCP server (e.g. before starting Cursor, or in the shell that runs `symdex mcp`).

---

## 2. SymdexConfig — all fields and defaults

### LLM provider

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `llm_provider` | `"anthropic"` | `SYMDEX_LLM_PROVIDER` | `anthropic`, `openai`, or `gemini` |
| `anthropic_api_key` | `None` | `ANTHROPIC_API_KEY` | Required if provider is anthropic |
| `anthropic_model` | `"claude-haiku-4-5"` | `ANTHROPIC_MODEL` | Model name |
| `openai_api_key` | `None` | `OPENAI_API_KEY` | Required if provider is openai |
| `openai_model` | `"gpt-4o-mini"` | `OPENAI_MODEL` | Model name |
| `gemini_api_key` | `None` | `GEMINI_API_KEY` | Required if provider is gemini |
| `gemini_model` | `"gemini-2.0-flash"` | `GEMINI_MODEL` | Model name |
| `llm_max_tokens` | `300` | `SYMDEX_MAX_TOKENS` | Max tokens per LLM response |
| `llm_temperature` | `0.0` | — | Not overridable via env (reproducibility) |

### Rate limiting

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `max_requests_per_minute` | `50` | — | LLM rate limit |
| `max_concurrent_requests` | `5` | — | Indexing concurrency |
| `retry_attempts` | `3` | — | LLM retries |
| `retry_backoff_base` | `2.0` | — | Exponential backoff base |

### File processing (indexing)

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `target_extensions` | `frozenset((".py",))` | — | File extensions to index |
| `exclude_dirs` | `__pycache__`, `.git`, `.venv`, … | — | Directories to skip |
| `max_file_size_mb` | `5` | — | Skip files larger than this |

### Sidecar index

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `symdex_dir` | `".symdex"` | — | Index directory name under project root |
| `cache_db_name` | `"index.db"` | — | SQLite filename |
| `cache_expiry_days` | `30` | — | Unused for now |

### Search (engine)

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `max_search_results` | `5` | — | Used when API/CLI does not pass `max_results` (engine fallback) |
| `max_search_candidates` | `200` | — | Cap merged candidates before scoring |
| `min_search_score` | `5.0` | `CYPHER_MIN_SCORE` | Minimum score to include a result |
| `default_context_lines` | `3` | `SYMDEX_DEFAULT_CONTEXT_LINES` | **MCP/API default** context lines per result |
| `default_max_search_results` | `10` | `SYMDEX_DEFAULT_MAX_RESULTS` | **MCP/API default** max hits |
| `default_exclude_tests` | `True` | `SYMDEX_DEFAULT_EXCLUDE_TESTS` | **MCP/API default** exclude test functions (0/false/no = include tests) |
| `stop_words` | (long set) | — | Words ignored in query/name scoring |
| `search_ranking_weights` | (dict) | — | Scoring weights (exact_match, action_match, …) |

### Other

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| `cypher_fallback_only` | `False` | `SYMDEX_CYPHER_FALLBACK_ONLY` | 1/true/yes = no LLM, rule-only |
| `cypher_version` | `"1.0"` | — | Schema version |
| `log_level` | `"INFO"` | `CYPHER_LOG_LEVEL` | Logging level |

---

## 3. CLI options (override config when passed)

### symdex index

| Option | Default | Description |
|--------|---------|-------------|
| `DIRECTORY` | `.` | Root to index |
| `--force` | `False` | Re-index all files (ignore hash cache) |
| `--cache-dir` | `./.symdex` | Override index location |

### symdex search

| Option | Default | Description |
|--------|---------|-------------|
| `QUERY` | (required) | Natural-language or Cypher pattern |
| `--cache-dir` | `./.symdex` | Override index location |
| `-f, --format` | `console` | `console`, `json`, `compact`, `ide` |
| `-n, --max-results` | from config | Max hits |
| `--strategy` | `auto` | `auto`, `llm`, `keyword`, `direct` |
| `--min-score` | from config | Min relevance score |
| `--context-lines` | `3` | Lines of code preview per result |
| `--include-tests` | `False` | Include test functions (by default tests are excluded) |
| `--explain` | `False` | Show scoring breakdown |

### symdex watch

| Option | Default | Description |
|--------|---------|-------------|
| `DIRECTORY` | `.` | Root to watch |
| `--interval` | `300` | Min seconds between re-indexes |
| `--debounce` | `5` | Seconds to wait after last change |

### symdex mcp

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | `stdio`, `streamable-http`, `sse` |
| `-v, --verbose` | `False` | Debug logging |

---

## 4. MCP tool parameters (defaults from config when omitted)

When a tool param is **not** sent by the client, the server uses **SymdexConfig** (and thus env) for that value.

### search_codebase

| Param | Config field | Default | Env override |
|-------|----------------|---------|----------------|
| `query` | — | (required) | — |
| `path` | — | `"."` | — |
| `strategy` | — | `"auto"` | — |
| `max_results` | `default_max_search_results` | `10` | `SYMDEX_DEFAULT_MAX_RESULTS` |
| `context_lines` | `default_context_lines` | `3` | `SYMDEX_DEFAULT_CONTEXT_LINES` |
| `exclude_tests` | `default_exclude_tests` | `True` | `SYMDEX_DEFAULT_EXCLUDE_TESTS` |

### search_by_cypher

| Param | Config field | Default | Env override |
|-------|----------------|---------|----------------|
| `cypher_pattern` | — | (required) | — |
| `path` | — | `"."` | — |
| `max_results` | `default_max_search_results` | `10` | `SYMDEX_DEFAULT_MAX_RESULTS` |

### index_directory / get_index_stats / health

No search-default overrides; other options (e.g. `path`, `force`) are fixed defaults.

### get_callers / get_callees / trace_call_chain (call graph)

| Param | Config field | Default | Env override |
|-------|----------------|---------|----------------|
| `function_name` | — | (required) | — |
| `path` | — | `"."` | — |
| `context_lines` | `default_context_lines` | `3` | `SYMDEX_DEFAULT_CONTEXT_LINES` |
| `file_path` (get_callees only) | — | `None` | — |
| `direction` (trace_call_chain only) | — | `"callers"` | — |
| `max_depth` (trace_call_chain only) | — | `5` | — |

---

## 5. How to configure MCP with default values

1. **Set environment variables** before starting the process that runs the MCP server (Cursor, or `symdex mcp`):

   ```bash
   # Examples (Linux/macOS)
   export SYMDEX_DEFAULT_MAX_RESULTS=15
   export SYMDEX_DEFAULT_CONTEXT_LINES=8
   export SYMDEX_DEFAULT_EXCLUDE_TESTS=true

   # Windows PowerShell
   $env:SYMDEX_DEFAULT_MAX_RESULTS = 15
   $env:SYMDEX_DEFAULT_CONTEXT_LINES = 8
   $env:SYMDEX_DEFAULT_EXCLUDE_TESTS = "true"
   ```

2. **Then start Cursor** (or run `symdex mcp`). The server is created with `SymdexConfig.from_env()`, so these defaults apply to every `search_codebase` / `search_by_cypher` call when the client omits the corresponding parameter.

3. **Optional:** If you run the MCP server yourself (e.g. for debugging), you can also pass a **config object**:

   ```python
   from symdex.mcp.server import create_server
   from symdex.core.config import SymdexConfig

   cfg = SymdexConfig.from_env()
   cfg.default_context_lines = 10
   cfg.default_exclude_tests = True
   server = create_server(config=cfg)
   server.run()
   ```

---

## 6. Quick reference — env vars for MCP defaults

| Env var | Default | Effect |
|---------|---------|--------|
| `SYMDEX_DEFAULT_MAX_RESULTS` | `10` | Default max hits for search_codebase / search_by_cypher |
| `SYMDEX_DEFAULT_CONTEXT_LINES` | `3` | Default context lines per result |
| `SYMDEX_DEFAULT_EXCLUDE_TESTS` | `true` | Default exclude_tests (set to 0/false/no to include tests by default) |
| `CYPHER_MIN_SCORE` | `5.0` | Minimum score for a result to be included (API/engine) |
| `SYMDEX_LLM_PROVIDER` | `anthropic` | LLM for indexing/query translation |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | — | Required for the chosen provider |

All other SymdexConfig fields use the defaults in code unless you build a custom `SymdexConfig` and pass it into `create_server(config=...)` or `Symdex(config=...)`.
