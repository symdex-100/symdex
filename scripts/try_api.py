#!/usr/bin/env python3
"""
Manually test and review the Symdex Python API using an example repository.

This script indexes a directory, prints stats, and runs example searches
so you can see index(), stats(), search(), and search_by_cypher() in action.

Usage:
  # From project root (index and search this repo's src/)
  python scripts/try_api.py
  python scripts/try_api.py src

  # Use another repo as the example
  python scripts/try_api.py /path/to/your/project

  # Index only (no searches) — useful to build index then use REPL
  python scripts/try_api.py src --index-only

Requirements:
  - Symdex installed (pip install -e . from project root)
  - ANTHROPIC_API_KEY or OPENAI_API_KEY set (or pass --no-llm to use fallback only)
"""

import sys
from pathlib import Path

# Project root on path for symdex import when run from repo
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Optional: use src layout so "symdex" is the package
_src = _project_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def main() -> None:
    import argparse
    from symdex import Symdex, SymdexConfig, IndexNotFoundError

    parser = argparse.ArgumentParser(
        description="Manually test Symdex API: index a directory and run example searches.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="src",
        type=Path,
        help="Directory to index and search (default: src)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only run index(); skip search examples (handy before using REPL)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based Cypher fallback only (no API key needed)",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    if not root.exists():
        # Default "src" may be missing when run from another repo; fall back to cwd
        if args.path == Path("src"):
            root = Path(".").resolve()
        else:
            print(f"Error: path does not exist: {args.path}")
            sys.exit(1)
    if not root.is_dir():
        print(f"Error: not a directory: {root}")
        sys.exit(1)

    # Config: optional no-LLM mode (fallback only; no API key needed)
    if args.no_llm:
        config = SymdexConfig(cypher_fallback_only=True)
        client = Symdex(config=config)
        print("Using rule-based fallback only (no LLM calls).\n")
    else:
        client = Symdex()
        print("Using configured LLM provider from environment.\n")

    # ── Index ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Index")
    print("=" * 60)
    print(f"  Path: {root}\n")

    result = client.index(root, show_progress=True)
    print(f"\n  Result: {result.files_scanned} files scanned, "
          f"{result.functions_indexed} functions indexed, "
          f"{result.errors} errors.")

    if args.index_only:
        print("\n  (--index-only: skipping search examples)")
        return

    # ── Stats ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Stats")
    print("=" * 60)
    try:
        stats = client.stats(path=root)
        print(f"  indexed_files: {stats['indexed_files']}")
        print(f"  indexed_functions: {stats['indexed_functions']}")
    except IndexNotFoundError as e:
        print(f"  Error: {e}")
        return

    # ── Search (natural language) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Search (natural language)")
    print("=" * 60)

    example_queries = [
        "validate or check something",
        "fetch or load data",
        "search or query the index",
    ]
    for q in example_queries:
        print(f"\n  Query: \"{q}\"")
        try:
            hits = client.search(q, path=root, max_results=3)
            for i, h in enumerate(hits, 1):
                print(f"    {i}. {h.function_name} @ {h.file_path}:{h.line_start}  "
                      f"[{h.cypher}]  score={h.score:.1f}")
            if not hits:
                print("    (no hits)")
        except IndexNotFoundError:
            print("    (no index)")
            break

    # ── Search by Cypher pattern ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Search by Cypher pattern")
    print("=" * 60)

    patterns = [
        "*:VAL_*--*",
        "*:FET_*--*",
        "SEC:*_*--*",
    ]
    for pat in patterns:
        print(f"\n  Pattern: \"{pat}\"")
        try:
            hits = client.search_by_cypher(pat, path=root, max_results=3)
            for i, h in enumerate(hits, 1):
                print(f"    {i}. {h.function_name} @ {h.file_path}:{h.line_start}  [{h.cypher}]")
            if not hits:
                print("    (no hits)")
        except IndexNotFoundError:
            print("    (no index)")
            break

    print("\n" + "=" * 60)
    print("  Done. Try your own queries in Python:")
    print("    from symdex import Symdex")
    print("    client = Symdex()")
    print(f"    client.search('your query', path=r'{root}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
