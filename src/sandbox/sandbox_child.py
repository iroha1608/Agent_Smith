"""The child sandbox process that runs student code in a safe environment and its helpers."""

from __future__ import annotations
from src.models.sandbox_config import SandboxConfig
from src.sandbox.code_lint import _safe_builtins
from src.sandbox.sandbox_result import FinalAnswer
from src.sandbox.sandbox_helper import (
    _apply_limits,
    _catalog_relays,
    _disable_network,
    _ensure_scratch_dirs,
    _install_audit_hook,
    _scrub_environment,
    _tool_proxy,
    _recv,
    _send,
    _on_alarm,
    _truncate,
    _Timeout
)
from typing import Any
from contextlib import redirect_stdout, redirect_stderr
import os
import signal
import io
import traceback
import ast


def _exit_code(code: Any) -> int:
    """Map SystemExit.code to the integer exit status Python itself would use.

    None -> 0, an int -> itself, anything else (a message) -> 1.
    """
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _execute(
    code: str, namespace: dict[str, Any], config: SandboxConfig
) -> dict[str, Any]:
    """Run one snippet and describe what happened, as a dict."""
    timeout = config.max_execution_time_seconds
    tree = ast.parse(code)
    # 最後の文が式なら、REPLのようにその値を表示するため切り出して別に評価する
    last = tree.body[-1] if tree.body else None
    tail = last if isinstance(last, ast.Expr) else None
    if tail is not None:
        tree.body.pop()
    body = compile(tree, "<sandbox>", "exec")

    out, err = io.StringIO(), io.StringIO()
    result: dict[str, Any] = {"kind": "ok", "value": None, "timeout": False}
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(body, namespace)
            if tail is not None:
                expr = compile(
                    ast.Expression(tail.value), "<sandbox>", "eval"
                )
                value = eval(expr, namespace)
                if value is not None:
                    result["value"] = repr(value)
    except FinalAnswer as answer:
        result = {"kind": "final_answer", "value": str(answer.value)}
    except _Timeout:
        result = {"kind": "error", "timeout": True, "error": (
            f"timeout after {timeout}s; any output above is partial"
        )}
    except MemoryError:
        result = {"kind": "error", "error": "memory limit exceeded"}
    except KeyboardInterrupt:
        result = {"kind": "interrupt"}
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) else None
        result = {"kind": "exit", "code": _exit_code(exc.code), "message": message}
    except BaseException:  # noqa: B036 - KeyboardInterrupt/SystemExit
        # are caught above; this is needed to catch LLM code that can
        # raise any exception.
        result = {"kind": "error", "error": traceback.format_exc(limit=5)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)

    limit = config.max_output_chars
    out_text, out_cut = _truncate(out.getvalue(), limit)
    err_text, err_cut = _truncate(err.getvalue(), limit)
    result["stdout"] = out_text
    result["stderr"] = err_text
    result["truncated"] = out_cut or err_cut
    return result


def _child_main(
        config: SandboxConfig,
        cmd_fd: int,
        res_fd: int,
        tool_params: dict[str, list[str]],
        with_catalog: bool = False,
) -> None:
    """Worker entry point. Never returns.

    Args:
        config (SandboxConfig): the sandbox limits.
        cmd_fd (int): pipe from the parent (commands and tool replies).
        res_fd (int): pipe to the parent (results and tool requests).
        tool_params (dict[str, list[str]]): MCP tool name -> parameter names in schema order.
        with_catalog (bool): whether the server offers resources / prompts (adds read_resource / get_prompt).
    """
    os.setsid()
    signal.signal(signal.SIGALRM, _on_alarm)
    _disable_network()

    def final_answer(value: Any = None) -> Any:
        """Flag for final answer."""
        raise FinalAnswer(value)

    namespace: dict[str, Any] = {
        "__name__": "__sandbox__",
        "__builtins__": _safe_builtins(config),
        "final_answer": final_answer,
    }
    for name, params in tool_params.items():
        namespace[name] = _tool_proxy(name, params, res_fd, cmd_fd)
    if with_catalog:
        for name, relay in _catalog_relays(res_fd, cmd_fd).items():
            namespace.setdefault(name, relay)

    # 順番が大事: 書き込み先の用意 -> 秘密(環境変数)の消去 -> 資源の上限 -> 監査フック。
    # 監査フックを入れた後は、この子プロセス自身も許可リスト外へは書き込めない。
    _ensure_scratch_dirs(config)
    _scrub_environment()
    _apply_limits(config)
    _install_audit_hook(config)

    while True:
        try:
            message = _recv(cmd_fd)
        except (EOFError, OSError):
            os._exit(0)
        if message.get("kind") != "run":
            os._exit(0)
        outcome = _execute(message["code"], namespace, config)
        try:
            _send(res_fd, outcome)
        except OSError:
            os._exit(1)
