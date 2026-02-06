#!/usr/bin/env python3
"""
Compare grep-style text search vs Symdex/Cypher search on a folder.

Grep behaviour:
  - 1 word:  lines containing that word (substring match)
  - n words (phrase):  lines containing the exact phrase "word1 word2 ..."
  - n words (AND):     lines containing all words, in any order

Symdex: semantic search over indexed functions (run `symdex index <dir>` first).

Reports hard metrics: elapsed time, unique (file, line) hits, file count,
and estimated token count of the result set (chars ÷ 4, LLM-approximate).

Usage:
  python scripts/compare_grep_symdex.py <folder> <query>
  python scripts/compare_grep_symdex.py . "validate user"
  python scripts/compare_grep_symdex.py src "token" --grep-only
"""

import sys
import time
from pathlib import Path

# Approximate tokens for LLM context (typical ~4 chars per token for code/English)
CHARS_PER_TOKEN = 4

# Ensure project root is on path so we can import symdex
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# File extensions to search (match Symdex default)
PY_EXT = (".py",)
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".symdex", ".pytest_cache", "dist", "build"}


def _format_tokens(chars: int) -> str:
    """Estimated token count from character count; formatted with commas."""
    n = max(0, round(chars / CHARS_PER_TOKEN))
    return f"{n:,}"


def _grep_stats(results: list[tuple[Path, int, str]]) -> dict:
    """Hard facts for grep result set: hits, unique lines, files, chars, tokens."""
    if not results:
        return {"hits": 0, "unique_lines": 0, "files": 0, "chars": 0, "tokens_est": "0"}
    paths = [p for p, _, _ in results]
    unique_files = len(set(paths))
    unique_lines = len(set((p, ln) for p, ln, _ in results))
    total_chars = sum(len(line) for _, _, line in results)
    return {
        "hits": len(results),
        "unique_lines": unique_lines,
        "files": unique_files,
        "chars": total_chars,
        "tokens_est": _format_tokens(total_chars),
    }


def grep_folder(
    folder: Path,
    query: str,
    *,
    mode: str = "all",
    ext: tuple = PY_EXT,
    max_results: int = 50,
) -> list[tuple[Path, int, str]]:
    """
    Grep-style search over files in folder.

    mode:
      - "word":   single word (substring) — 1 word
      - "phrase": exact phrase — n words as "word1 word2 ..."
      - "and":    all words must appear (any order) — n words AND

    Returns list of (file_path, line_number, line_text).
    """
    folder = folder.resolve()
    words = query.strip().split()
    results: list[tuple[Path, int, str]] = []

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix not in ext:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if mode == "word":
                if words and words[0].lower() in line.lower():
                    results.append((path, i, line.strip()))
            elif mode == "phrase":
                if query.strip().lower() in line.lower():
                    results.append((path, i, line.strip()))
            else:  # and
                lower_line = line.lower()
                if all(w.lower() in lower_line for w in words):
                    results.append((path, i, line.strip()))

            if len(results) >= max_results:
                return results

    return results


def _symdex_stats(results: list[dict]) -> dict:
    """Hard facts for Symdex result set: hits, unique (file, line), files, chars, tokens."""
    if not results:
        return {"hits": 0, "unique_lines": 0, "files": 0, "chars": 0, "tokens_est": "0"}
    unique_files = len(set(r["file"] for r in results))
    unique_lines = len(set((r["file"], r["line"]) for r in results))
    total_chars = sum(len(r.get("context", "")) for r in results)
    return {
        "hits": len(results),
        "unique_lines": unique_lines,
        "files": unique_files,
        "chars": total_chars,
        "tokens_est": _format_tokens(total_chars),
    }


def symdex_search(folder: Path, query: str, max_results: int = 20) -> tuple[list[dict], float]:
    """
    Run Symdex/Cypher search. Requires folder to be indexed (symdex index <folder>).
    Returns (list of result dicts including 'context', elapsed_seconds).
    """
    cache_dir = folder / ".symdex"
    if not (cache_dir / "index.db").exists():
        return [], 0.0

    from symdex.core.config import Config
    from symdex.core.search import CypherSearchEngine

    db_path = Config.get_cache_path(cache_dir)
    if not db_path.exists():
        return [], 0.0

    t0 = time.perf_counter()
    engine = CypherSearchEngine(cache_dir)
    raw = engine.search(query, strategy="auto", max_results=max_results)
    elapsed = time.perf_counter() - t0

    results = [
        {
            "file": r.file_path,
            "function": r.function_name,
            "line": r.line_start,
            "cypher": r.cypher,
            "score": r.score,
            "context": getattr(r, "context", "") or "",
        }
        for r in raw
    ]
    return results, elapsed


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Compare grep (1 word / n-word phrase / n-word AND) vs Symdex search on a folder.",
    )
    p.add_argument("folder", type=Path, default=Path("."), nargs="?",
                   help="Folder to search (default: current dir)")
    p.add_argument("query", type=str, nargs="?", default="validate",
                   help="Search query: one word or several (default: validate)")
    p.add_argument("--grep-only", action="store_true", help="Run only grep-style search")
    p.add_argument("--symdex-only", action="store_true", help="Run only Symdex search")
    p.add_argument("-n", "--max", type=int, default=15, help="Max results per method (default 15)")
    args = p.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    query = args.query.strip()
    if not query:
        print("Error: empty query", file=sys.stderr)
        sys.exit(1)

    words = query.split()
    n_words = len(words)

    def _print_stats(label: str, stats: dict, elapsed_sec: float) -> None:
        """Print one line of hard facts: hits, files, unique lines, tokens, time."""
        print(
            f"  {label}\n"
            f"    {stats['hits']} hits in {stats['files']} file(s) "
            f"({stats['unique_lines']} unique lines), "
            f"~{stats['tokens_est']} tokens (est.), "
            f"{elapsed_sec:.3f}s"
        )

    # ─── Grep modes ─────────────────────────────────────────────
    if not args.symdex_only:
        print("=" * 60)
        print("GREP-STYLE (text in file lines)")
        print("  (tokens est. = total chars in result set ÷ 4, LLM-approximate)")
        print("=" * 60)

        if n_words == 1:
            t0 = time.perf_counter()
            one = grep_folder(folder, query, mode="word", max_results=args.max)
            elapsed = time.perf_counter() - t0
            st = _grep_stats(one)
            _print_stats(f"1 word (substring): '{query}'", st, elapsed)
            for path, line_no, line in one[:args.max]:
                rel = path.relative_to(folder) if folder in path.parents else path
                print(f"    {rel}:{line_no}  {line[:80]}{'...' if len(line) > 80 else ''}")
        else:
            t0 = time.perf_counter()
            phrase = grep_folder(folder, query, mode="phrase", max_results=args.max)
            e1 = time.perf_counter() - t0
            t0 = time.perf_counter()
            and_ = grep_folder(folder, query, mode="and", max_results=args.max)
            e2 = time.perf_counter() - t0
            st_phrase = _grep_stats(phrase)
            st_and = _grep_stats(and_)
            _print_stats(f"Phrase (exact): '{query}'", st_phrase, e1)
            for path, line_no, line in phrase[:args.max]:
                rel = path.relative_to(folder) if folder in path.parents else path
                print(f"      {rel}:{line_no}  {line[:70]}{'...' if len(line) > 70 else ''}")
            _print_stats(f"AND (all words, any order): '{query}'", st_and, e2)
            for path, line_no, line in and_[:args.max]:
                rel = path.relative_to(folder) if folder in path.parents else path
                print(f"      {rel}:{line_no}  {line[:70]}{'...' if len(line) > 70 else ''}")

    # ─── Symdex ────────────────────────────────────────────────
    if not args.grep_only:
        print("\n" + "=" * 60)
        print("SYMDEX / CYPHER (semantic function search)")
        print("  (tokens est. = context snippet chars ÷ 4)")
        print("=" * 60)
        symdex_results, symdex_elapsed = symdex_search(folder, query, max_results=args.max)
        if not symdex_results:
            cache = folder / ".symdex" / "index.db"
            print(f"\n  No index at {cache}. Run: symdex index {folder}")
        else:
            st = _symdex_stats(symdex_results)
            _print_stats(f"Query: '{query}' → ranked by intent (Cypher)", st, symdex_elapsed)
            for r in symdex_results:
                print(f"  {r['file']}:{r['line']}  {r['function']}  [{r['cypher']}] score={r['score']:.1f}")

    print()


if __name__ == "__main__":
    main()
