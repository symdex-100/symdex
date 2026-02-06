FROM python:3.11-slim

# Minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package with the default (Anthropic) provider
# Switch to [openai] or [gemini] or [llm-all] as needed.
RUN pip install --no-cache-dir ".[anthropic]"

# Default: show CLI help (overridden by docker-compose)
CMD ["symdex", "--help"]