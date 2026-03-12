# Docker

The image includes MCP server support by default (install extras: `anthropic,mcp`). Override with build arg `EXTRAS` (e.g. `openai,mcp` or `llm-all,mcp`) if needed.

## CLI: index and search

```bash
# Index a project
docker run -v /host/project:/data symdex-100 \
  symdex index /data

# Search the index
docker run -v /host/project:/data symdex-100 \
  symdex search "validate user" --cache-dir /data/.symdex
```

**Note:** `--cache-dir` must be the path *inside* the container.

## Running the MCP server in Docker (e.g. Smithery)

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

## Index on host vs remote server URL

The MCP server always resolves the `path` parameter (default `"."`) on the **machine where the server process runs**. So:

- **Smithery (or Fly/Railway) URL** — The server runs in the cloud and has no access to your laptop. Indexing and search happen only on the codebase that is deployed/mounted there (e.g. a clone of the repo on the server). You cannot have that remote server index the code on your machine.
- **Index strictly on your machine** — Use one of:
  - **Local MCP (recommended for one workspace):** In Cursor, point MCP to the local server (e.g. `"command": "symdex", "args": ["mcp"]`). The server runs as a subprocess with your workspace as CWD; index and search use your repo on disk.
  - **Local Docker with mount:** Run the server in Docker on your machine, mount your project, and set `SYMDEX_DEFAULT_PATH=/data` so that when the client sends `path="."`, the server uses the mounted directory. Then point Cursor to `http://localhost:8000/mcp` (or use stdio with the container). Index and search run inside the container but read/write the mounted host path (your project and `.symdex` live on your machine).

### Example — local Docker, index on host

```bash
# Mount your project at /data; server will use /data when path is "."
docker run -p 8000:8000 -v E:\Cypher:/data -e SYMDEX_DEFAULT_PATH=/data -e ANTHROPIC_API_KEY=sk-... symdex-100
```

In Cursor, add an MCP server with URL `http://localhost:8000/mcp`. The agent’s `index_directory(".")` and `search_codebase(..., path=".")` will use `/data` (your repo). The index is written to `E:\Cypher\.symdex` on the host.

## Publishing on Smithery

Smithery *can* host from GitHub, but **Hosted** deploy is for servers built with the [Smithery CLI and SDK](https://smithery.ai/docs/build/build) (TypeScript) and runs in their edge runtime—**128 MB, no filesystem, no native modules, no spawning processes** ([Hosting Limits](https://smithery.ai/docs/build/limits)). Symdex needs filesystem (SQLite index, reading source files) and a Python runtime, so it cannot run in that environment.

Use the **URL** method: host the server elsewhere and give Smithery your public HTTPS URL. The server uses **Streamable HTTP** on **`/mcp`**, serves **`/.well-known/mcp/server-card.json`**, and sends CORS headers.

To get a URL with no server to maintain: deploy this repo’s Docker image to [Fly.io](https://fly.io) or [Railway](https://railway.app). With Fly: install [flyctl](https://fly.io/docs/hands-on/install-flyctl/), run `fly launch` in this repo, then `fly deploy`; at [smithery.ai/new](https://smithery.ai/new) choose **URL** and enter `https://<app-name>.fly.dev/mcp`. The repo includes `fly.toml` for Fly.
