FROM python:3.11-slim

# Minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package with LLM provider + MCP server support.
# Build arg: EXTRAS e.g. "anthropic,mcp" or "openai,mcp" or "llm-all,mcp"
ARG EXTRAS=anthropic,mcp
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Default: run MCP server with HTTP (Streamable HTTP for Smithery / remote clients).
# Override for CLI: docker run ... symdex --help   or   symdex index /data
# For stdio (local Cursor): docker run -it ... symdex mcp --transport stdio
EXPOSE 8000
ENV MCP_TRANSPORT=http
CMD ["sh", "-c", "symdex mcp --transport ${MCP_TRANSPORT}"]