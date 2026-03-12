"""
JavaScript/TypeScript function extraction using tree-sitter.

Uses tree-sitter-language-pack (required dependency). Parses .js, .jsx, .ts, .tsx
and produces FunctionMetadata for the indexing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

from symdex.core.engine import CallSite, FunctionMetadata, JS_TS_EXTENSIONS

logger = logging.getLogger(__name__)

# Node types that define a function/method in tree-sitter JS/TS grammars.
_FUNCTION_NODE_TYPES = frozenset({
    "function_declaration",
    "method_definition",
    "function_expression",
    "arrow_function",
})

# Node types that represent call expressions (for call graph).
_CALL_NODE_TYPE = "call_expression"

# Node types that increase cyclomatic complexity (approximation).
_COMPLEXITY_NODE_TYPES = frozenset({
    "if_statement",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "catch_clause",
    "conditional_expression",
})


def _get_parser_for_extension(ext: str):
    """Return a tree-sitter Parser for the given file extension, or None."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        try:
            from tree_sitter_languages import get_parser
        except ImportError:
            return None
    ext = ext.lower()
    if ext in (".js", ".jsx"):
        return get_parser("javascript")
    if ext == ".ts":
        return get_parser("typescript")
    if ext == ".tsx":
        try:
            return get_parser("tsx")
        except Exception:
            return get_parser("typescript")
    return None


def _node_text(source_bytes: bytes, node) -> str:
    """Return the source slice for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_start_line(node) -> int:
    """Return 1-based start line for a tree-sitter node."""
    return node.start_point[0] + 1


def _node_end_line(node) -> int:
    """Return 1-based end line for a tree-sitter node."""
    return node.end_point[0] + 1


def _is_async_from_source(source_bytes: bytes, node) -> bool:
    """Heuristic: check if function node is preceded by 'async' in source."""
    start = node.start_byte
    # Look back at most 10 bytes for "async"
    chunk_start = max(0, start - 10)
    chunk = source_bytes[chunk_start:start].decode("utf-8", errors="replace")
    return "async" in chunk


def _get_function_name(node, source_bytes: bytes, parent_node=None) -> str:
    """
    Extract a display name for the function from the AST.
    For arrow_function/function_expression, the name may come from the parent (variable declarator).
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(source_bytes, name_node).strip()
    # Arrow function or anonymous function expression: try parent variable declarator
    if parent_node is not None and parent_node.type == "variable_declarator":
        decl_name = parent_node.child_by_field_name("name")
        if decl_name is not None:
            return _node_text(source_bytes, decl_name).strip()
    return "<anonymous>"


def _collect_parameter_names(params_node: Any, source_bytes: bytes) -> List[str]:
    """Extract parameter names from formal_parameters node."""
    if params_node is None:
        return []
    args: List[str] = []
    for child in getattr(params_node, "named_children", params_node.children):
        if child.type == "identifier":
            text = _node_text(source_bytes, child).strip()
            if text and text != "this":
                args.append(text)
        elif child.type == "rest_parameter":
            rest = child.child_by_field_name("parameter") or child.child_by_field_name("name")
            if rest is not None:
                args.append("..." + _node_text(source_bytes, rest).strip())
        elif child.type == "optional_parameter":
            name = child.child_by_field_name("parameter")
            if name is not None:
                args.append(_node_text(source_bytes, name).strip())
    return args


def _callee_name_from_call(call_node, source_bytes: bytes) -> Optional[str]:
    """
    Get the callee name for a call_expression (for call graph).
    function child can be: identifier, member_expression, etc.
    """
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _node_text(source_bytes, func_node).strip()
    if func_node.type == "member_expression":
        # Use the property (method name) as the callee for simplicity
        prop = func_node.child_by_field_name("property")
        if prop is not None:
            return _node_text(source_bytes, prop).strip()
        return _node_text(source_bytes, func_node).strip()
    return _node_text(source_bytes, func_node).strip() or None


def _walk_descendants(node, predicate=None):
    """Yield all descendants of node (depth-first). Optionally filter by type predicate."""
    for child in node.children:
        if predicate is None or predicate(child):
            yield child
        for d in _walk_descendants(child, predicate):
            yield d


def _collect_calls_and_complexity(body_node, source_bytes: bytes) -> Tuple[List[CallSite], int]:
    """Walk function body for call_expressions and complexity nodes."""
    if body_node is None:
        return [], 1
    call_sites: List[CallSite] = []
    calls_seen: Set[str] = set()
    complexity = 1
    for desc in _walk_descendants(body_node):
        if desc.type == _CALL_NODE_TYPE:
            name = _callee_name_from_call(desc, source_bytes)
            if name and name not in calls_seen:
                calls_seen.add(name)
                call_sites.append(CallSite(callee_name=name, line=_node_start_line(desc)))
        elif desc.type in _COMPLEXITY_NODE_TYPES:
            complexity += 1
    calls_list = list(calls_seen)[:10]
    return call_sites, complexity


def _extract_one_function(
    node,
    source_bytes: bytes,
    file_path: str,
    parent_node=None,
) -> Optional[FunctionMetadata]:
    """Build FunctionMetadata from a single function/method/arrow node."""
    start_line = _node_start_line(node)
    end_line = _node_end_line(node)
    is_async = _is_async_from_source(source_bytes, node)
    name = _get_function_name(node, source_bytes, parent_node)
    params_node = node.child_by_field_name("parameters")
    args = _collect_parameter_names(params_node, source_bytes)
    body_node = node.child_by_field_name("body")
    call_sites, complexity = _collect_calls_and_complexity(body_node, source_bytes)
    calls = [cs.callee_name for cs in call_sites]
    # JSDoc could be extracted from preceding comment; leave empty for now
    docstring: Optional[str] = None
    lang = "JavaScript" if Path(file_path).suffix.lower() in (".js", ".jsx") else "TypeScript"
    return FunctionMetadata(
        name=name,
        start_line=start_line,
        end_line=end_line,
        is_async=is_async,
        args=args,
        calls=calls,
        imports=[],
        docstring=docstring,
        complexity=complexity,
        language=lang,
        call_sites=call_sites,
    )


def _collect_function_nodes(root: Any, into: List[Tuple[Any, Any]]) -> None:
    """Recursively collect (node, parent) for all function-like nodes."""
    def walk(n: Any, parent: Any = None) -> None:
        if n.type in _FUNCTION_NODE_TYPES:
            into.append((n, parent))
        for child in n.children:
            walk(child, n)

    walk(root, None)


def parse_js_ts_functions(source_code: str, file_path: str) -> List[FunctionMetadata]:
    """
    Parse JavaScript or TypeScript source and return a list of FunctionMetadata.

    Uses tree-sitter-language-pack (built-in). Supports .js, .jsx, .ts, .tsx.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in JS_TS_EXTENSIONS:
        return []
    parser = _get_parser_for_extension(ext)
    if parser is None:
        logger.debug("No tree-sitter parser for %s", ext)
        return []
    source_bytes = source_code.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.error("Parse error in %s: %s", file_path, e)
        return []
    root = tree.root_node
    if root is None or root.has_error:
        logger.debug("Parse produced no root or had errors in %s", file_path)
        return []
    pairs: List[Tuple[Any, Any]] = []
    _collect_function_nodes(root, pairs)
    results: List[FunctionMetadata] = []
    for node, parent in pairs:
        try:
            meta = _extract_one_function(node, source_bytes, file_path, parent)
            if meta is not None:
                results.append(meta)
        except Exception as e:
            logger.warning("Skipping function in %s: %s", file_path, e)
    return results
