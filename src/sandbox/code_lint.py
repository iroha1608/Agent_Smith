"""The main checker for code. Checks AI code for imports and syntax."""
from __future__ import annotations
from src.models.sandbox_config import SandboxConfig
from typing import Any, Callable
from pathlib import Path
import builtins
import os
import ast


_BLOCKED = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "breakpoint", "input", "exit", "quit",
})


# 許可モジュールが内部で遅延 import する補助モジュール。C実装の関数(datetime.strptime など)は
# 呼び出し元(=サンドボックスのコード)の __import__ を使って import するため、ここで通さないと
# 許可モジュールの通常の機能が ImportError になる。静的チェックでは許可しないので、
# コードに直接 `import _strptime` と書いた場合は従来どおり拒否される。
_IMPLICIT_IMPORTS = frozenset({"_strptime"})


def _allowed(name: str, authorized: frozenset[str]) -> bool:
    """Match a module name against the allowlist."""
    return name in authorized or any(
        e.endswith(".*") and (name == e[:-2] or name.startswith(e[:-1]))
        for e in authorized
    )


class _Checker(ast.NodeVisitor):
    """Static pass: cheap rejection before anything happens."""
    def __init__(
        self,
        authorized: frozenset[str],
        tool_names: frozenset[str] = frozenset()
    ) -> None:
        """Record the allowlist, the tool nams, and any rejections."""
        self.authorized: frozenset[str] = authorized
        self.tool_names = tool_names
        self.errors: list[str] = []
        # 許可モジュールの名前(とこのスニペットでの別名)。これらの `_` 始まりの属性
        # (random._os, string._re など)は内部実装で、許可外のモジュールへの抜け道になる。
        self.module_names: set[str] = {entry.split(".")[0] for entry in authorized}

    def _check_module(self, name: str) -> None:
        """Check if importing named module is allowed."""
        if not _allowed(name, self.authorized):
            self.errors.append(f"import of {name!r} is not allowed")

    #  Visit methods so need the capital letter to match the AST node names.
    #  Or else the generic_visit() will not call them.
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Visit all nodes."""
        for alias in node.names:
            self._check_module(alias.name)
            self.module_names.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Visit from-imports and reject relative ones."""
        if node.level:
            self.errors.append("relative imports are not allowed")
        elif node.module:
            self._check_module(node.module)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Blocks dunder access, which reaches globals and subclasses.

        Also blocks private attributes of modules (random._os ...), a static first line of
        defence; the audit hook in the worker is what actually enforces the restrictions.
        """
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"access to dunder attribute {node.attr!r}")
        elif node.attr.startswith("_") and isinstance(node.value, ast.Name) \
                and node.value.id in self.module_names:
            self.errors.append(f"access to private module attribute {node.value.id}.{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        """Check names, letting injectted tools shadow blocked bultins.

        An unknown MCP server may expose a tools that aren't allowed.
        """
        if node.id in _BLOCKED and node.id not in self.tool_names:
            self.errors.append(f"use of {node.id!r} is not allowed")
        self.generic_visit(node)


def strip_provided_imports(code: str, provided: frozenset[str]) -> tuple[str, list[str]]:
    """Drop `import <name>` of names the sandbox already provides (tools, final_answer...).

    Models trained on other frameworks write `import json, run_tests, final_answer`. The tools
    are not modules, so the import would fail and the whole snippet would not run (実測:
    gpt-oss, MBPP 444). Removing just those names keeps the rest of the statement and leaves
    the provided objects untouched. `from x import name` is left alone (it is a real import).

    Args:
        code (str): the snippet.
        provided (frozenset[str]): names already bound in the sandbox namespace.

    Returns:
        tuple[str, list[str]]: (code to run, names whose import was dropped).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []
    dropped: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        kept = []
        for alias in node.names:
            if alias.name in provided and alias.asname in (None, alias.name):
                dropped.append(alias.name)
            else:
                kept.append(alias)
        node.names = kept
    if not dropped:
        return code, []

    class _DropEmptyImports(ast.NodeTransformer):
        def visit_Import(self, node: ast.Import) -> ast.AST:  # noqa: N802
            return node if node.names else ast.Pass()

    tree = ast.fix_missing_locations(_DropEmptyImports().visit(tree))
    return ast.unparse(tree), dropped


def check_code(
    code: str,
    config: SandboxConfig,
    tool_names: frozenset[str] = frozenset()
) -> str | None:
    """Return a rejection reason, or None if the code is okay."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    checker = _Checker(frozenset(config.authorized_imports), tool_names)
    checker.visit(tree)
    return "; ".join(checker.errors) or None


def _guarded_open(config: SandboxConfig) -> Callable[..., Any]:
    """Build an open() restricted to allowed_directories."""
    roots = [Path(d).resolve() for d in config.allowed_directories]
    real = builtins.open

    def guard(file: Any, mode: str = "r", *args: Any, **kw: Any) -> Any:
        """Resolve the path, then allow only inside allowlist."""
        if isinstance(file, int):
            raise PermissionError("Raw file descriptors are not allowed")
        target = Path(os.fsdecode(file)).resolve()
        if not any(target == r or target.is_relative_to(r) for r in roots):
            raise PermissionError(f"path outside allowlist: {target}")
        return real(target, mode, *args, **kw)

    return guard


def _safe_builtins(config: SandboxConfig) -> dict[str, Any]:
    """Bultins minus the dangerous ones, plus guarded __import__.

    This must be set explicitly: exec() given a globals dict with no
    '__bultins__' key silently inserts the real bultins module.
    """
    safe: dict[str, Any] = {
        n: getattr(builtins, n)
        for n in dir(builtins) if not n.startswith("_") and n not in _BLOCKED
    }
    authorized = frozenset(config.authorized_imports)
    real = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        """Enforcement point: every import compiles to this."""
        if not _allowed(name, authorized) and name not in _IMPLICIT_IMPORTS:
            raise ImportError(f"import of {name!r} not allowed")
        return real(name, *args, **kwargs)

    safe["__import__"] = guarded
    safe["open"] = _guarded_open(config)
    # `class` 文は __build_class__ を呼ぶ。無いとクラスが一切定義できない(MBPPの解答で使う)
    safe["__build_class__"] = builtins.__build_class__
    return safe
