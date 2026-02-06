"""
Shared fixtures for the Symdex-100 test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the src/ directory is on the import path so that
# symdex.core.config / symdex.core.engine / etc. can be imported.
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# Also keep the old project root for backward compat during migration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set dummy API keys so Config.validate() does not blow up during
# import of symdex.core.engine (the validation runs at import time when
# __name__ != "__main__").
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
def javascript_source() -> str:
    """JavaScript module with different function styles."""
    return (
        "import axios from 'axios';\n"
        "\n"
        "/**\n"
        " * Validates a JWT token.\n"
        " */\n"
        "function validateToken(token) {\n"
        "    if (!token) {\n"
        "        throw new Error('missing token');\n"
        "    }\n"
        "    return true;\n"
        "}\n"
        "\n"
        "export const fetchData = async (url) => {\n"
        "    const resp = await axios.get(url);\n"
        "    return resp.data;\n"
        "};\n"
    )


@pytest.fixture
def typescript_source() -> str:
    """TypeScript snippet with typed function and class method."""
    return (
        "export async function getUserById(id: number): Promise<User> {\n"
        "    const user = await db.find(id);\n"
        "    return user;\n"
        "}\n"
        "\n"
        "export const transformPayload = (data: RawData): CleanData => {\n"
        "    return { ...data, clean: true };\n"
        "};\n"
    )


@pytest.fixture
def java_source() -> str:
    """Minimal Java class with a method."""
    return (
        "package com.example;\n"
        "\n"
        "public class UserService {\n"
        "\n"
        "    public static User fetchUser(int id) {\n"
        "        if (id <= 0) {\n"
        "            throw new IllegalArgumentException(\"bad id\");\n"
        "        }\n"
        "        return db.findById(id);\n"
        "    }\n"
        "\n"
        "    private void deleteUser(int id) {\n"
        "        db.delete(id);\n"
        "    }\n"
        "}\n"
    )


@pytest.fixture
def go_source() -> str:
    """Go functions including a receiver method."""
    return (
        "package main\n"
        "\n"
        "import \"fmt\"\n"
        "\n"
        "// FetchRecords retrieves all records from storage.\n"
        "func FetchRecords(limit int) ([]Record, error) {\n"
        "    if limit <= 0 {\n"
        "        return nil, fmt.Errorf(\"bad limit\")\n"
        "    }\n"
        "    return store.GetAll(limit)\n"
        "}\n"
        "\n"
        "func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {\n"
        "    fmt.Fprintf(w, \"ok\")\n"
        "}\n"
    )


@pytest.fixture
def rust_source() -> str:
    """Rust functions including pub async."""
    return (
        "use std::io;\n"
        "\n"
        "/// Validates the input data.\n"
        "pub fn validate_input(data: &str) -> Result<(), io::Error> {\n"
        "    if data.is_empty() {\n"
        "        return Err(io::Error::new(io::ErrorKind::InvalidInput, \"empty\"));\n"
        "    }\n"
        "    Ok(())\n"
        "}\n"
        "\n"
        "pub async fn send_notification(msg: &str) -> bool {\n"
        "    println!(\"{}\", msg);\n"
        "    true\n"
        "}\n"
    )


@pytest.fixture
def ruby_source() -> str:
    """Ruby module with methods."""
    return (
        "class UserService\n"
        "  # Create a new user record.\n"
        "  def create_user(name, email)\n"
        "    user = User.new(name: name, email: email)\n"
        "    user.save!\n"
        "    user\n"
        "  end\n"
        "\n"
        "  def self.find_user(id)\n"
        "    User.find(id)\n"
        "  end\n"
        "end\n"
    )


@pytest.fixture
def cpp_source() -> str:
    """C++ with a class method."""
    return (
        "#include <string>\n"
        "#include <vector>\n"
        "\n"
        "// Parses CSV data into rows.\n"
        "std::vector<Row> parseCsv(const std::string& data) {\n"
        "    std::vector<Row> rows;\n"
        "    // parsing logic ...\n"
        "    return rows;\n"
        "}\n"
    )


@pytest.fixture
def csharp_source() -> str:
    """C# controller method."""
    return (
        "using System;\n"
        "\n"
        "public class OrderController\n"
        "{\n"
        "    public async Task<Order> CreateOrder(OrderRequest request)\n"
        "    {\n"
        "        var order = new Order(request);\n"
        "        await _db.SaveAsync(order);\n"
        "        return order;\n"
        "    }\n"
        "}\n"
    )


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """
    Create a temporary project directory containing sample source files
    in several languages for integration tests.
    """
    # Python file
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

    # JavaScript file
    (py_dir / "utils.js").write_text(
        "function formatCurrency(amount) {\n"
        "    return '$' + amount.toFixed(2);\n"
        "}\n"
        "\n"
        "export const parseJson = (raw) => {\n"
        "    return JSON.parse(raw);\n"
        "};\n",
        encoding="utf-8",
    )

    # Go file
    (py_dir / "handler.go").write_text(
        "package main\n"
        "\n"
        "func ServeHTTP(w http.ResponseWriter, r *http.Request) {\n"
        "    w.Write([]byte(\"ok\"))\n"
        "}\n",
        encoding="utf-8",
    )

    # Excluded directory (should be ignored)
    excluded = tmp_path / "node_modules"
    excluded.mkdir()
    (excluded / "ignore.js").write_text("function ignored() {}\n", encoding="utf-8")

    return tmp_path
