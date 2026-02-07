"""
Shared fixtures for the Symdex-100 test suite.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

# Filter deprecation warnings from pytest-asyncio (Python 3.16 prep); we cannot fix the library.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="pytest_asyncio",
)

# Ensure the src/ directory is on the import path so that
# symdex.core.config / symdex.core.engine / etc. can be imported.
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# Also keep the old project root for backward compat during migration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set dummy API keys so that Config / SymdexConfig have valid keys
# for tests that call validate(), get_api_key(), or create providers.
# Import-time validation was removed in v1.1 but keys are still needed
# for functional tests.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")


# =============================================================================
# Fixtures — sample source code snippets for each language
# =============================================================================

@pytest.fixture
def python_source() -> str:
    """Minimal Python module with two functions."""
    return (
        "import os\n"
        "\n"
        "def fetch_user(user_id: int) -> dict:\n"
        "    \"\"\"Retrieve a user from the database.\"\"\"\n"
        "    if user_id <= 0:\n"
        "        raise ValueError('bad id')\n"
        "    return {'id': user_id}\n"
        "\n"
        "\n"
        "async def send_email(to: str, subject: str, body: str) -> bool:\n"
        "    \"\"\"Send an email asynchronously.\"\"\"\n"
        "    return True\n"
    )


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """
    Create a temporary project directory containing Python source files
    for integration tests.
    """
    # Python files
    py_dir = tmp_path / "src"
    py_dir.mkdir()
    (py_dir / "app.py").write_text(
        "def validate_email(email: str) -> bool:\n"
        "    \"\"\"Check email format.\"\"\"\n"
        "    return '@' in email\n"
        "\n"
        "async def process_order(order_id: int) -> dict:\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    # Excluded directory (should be ignored)
    excluded = tmp_path / "__pycache__"
    excluded.mkdir()
    (excluded / "ignore.pyc").write_bytes(b"fake bytecode")

    return tmp_path
