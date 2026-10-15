"""The main entry for the sandbox."""

from __future__ import annotations
from src.models.sandbox_config import SandboxConfig
from src.sandbox.code_lint import check_code, strip_provided_imports, _BLOCKED
from src.sandbox.sandbox_helper import _send, _recv, _truncate
from src.sandbox.sandbox_child import _child_main
from src.models.output import ExecuteResult
from src.mcp_core.client.mcp_tool_function import MCPToolFunction
from src.mcp_core.client.client import MCPClient
from src.mcp_core.client.stdio import StdioClientTransport
from src.mcp_core.client.http import HttpClientTransport
from typing import Any, Callable
from pathlib import Path
import os
import ast
import difflib
import re
import sys
import select
import signal
import argparse

# The last line of a traceback for a name that does not exist
_UNDEFINED_NAME = re.compile(r"NameError: name '(\w+)' is not defined")


class Sandbox:
    """Runs snippets in an isolated child with a persistent namespace."""
    def __init__(
        self,
        config: SandboxConfig | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        mcp_stdio: str | None = None,
        mcp_server: str | None = None
    ) -> None:
        """Init the Sandbox and fork its worker."""
        self.config = config or SandboxConfig()
        self.tools: dict[str, Callable[..., Any]] = tools or {}
        self.client: MCPClient | None = None
        self._pid: int | None = None
        self._cmd_w = -1
        self._res_r = -1
        self._tool_truncated = False
        # resources / prompts の一覧(マニュアル用)。あれば read_resource / get_prompt を名前空間に置く
        self._catalog: list[tuple[str, list[Any]]] = []
        if mcp_stdio:
            self.client = MCPClient(StdioClientTransport(mcp_stdio))
        elif mcp_server:
            self.client = MCPClient(HttpClientTransport(mcp_server))
        if self.client is not None:
            try:
                self.client.connect()
                # 接続中サーバのツールを動的に発見し、サンドボックス名前空間へ公開する。
                # resources / prompts も同じクライアント経由で参照できる。
                self.tools = {
                    tool.name: MCPToolFunction(self.client, tool)
                    for tool in self.client.list_tools()
                }
                self._catalog = self._load_catalog()
            except Exception:
                self.client.close()
                self.client = None
                raise
        self._spawn()

    def __enter__(self) -> "Sandbox":
        """Enter a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave a context manager, killing the worker."""
        self.close()

    def _spawn(self) -> None:
        """Fork a worker and keep the parent endsd of both pipes."""
        sys.stdout.flush()
        sys.stderr.flush()
        cmd_r, cmd_w = os.pipe()
        res_r, res_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(cmd_w)
            os.close(res_r)
            try:
                _child_main(self.config, cmd_r, res_w, self._tool_params(), bool(self._catalog))
            finally:
                os._exit(1)
        os.close(cmd_r)
        os.close(res_w)
        self._pid, self._cmd_w, self._res_r = pid, cmd_w, res_r

    def reset(self) -> None:
        """Restart the worker, keeping the MCP client connected.

        Only the sandboxed namespace is lost. The cline belongs to
        the session.
        """
        self._close_worker()
        self._spawn()

    def close(self) -> None:
        """Shut down the worker and the MCP client.

        Safe to call twice; the orchestrator should call it in a finaly
        block.
        """
        self._close_worker()
        if self.client is not None:
            self.client.close()
            self.client = None

    def _close_worker(self) -> None:
        """Kill the worker's process group, reap it, and close pipes.

        Each descriptor is closed exactly once then blanked:
        a freed fd number can be reused.
        """
        if self._pid is None:
            return
        pid = self._pid
        self._pid = None

        for fd in (self._cmd_w, self._res_r):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

        self._cmd_w = self._res_r = -1

        for kill in (
            lambda: os.killpg(pid, signal.SIGKILL),
            lambda: os.kill(pid, signal.SIGKILL)
        ):
            try:
                kill()
                break
            except (ProcessLookupError, PermissionError):
                continue
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    def _tool_params(self) -> dict[str, list[str]]:
        """Tool name -> parameter names in input-schema order (for positional calls)."""
        params: dict[str, list[str]] = {}
        for name, tool in sorted(self.tools.items()):
            schema: dict[str, Any] = getattr(tool, "input_schema", None) or {}
            params[name] = list((schema.get("properties") or {}).keys())
        return params

    def _provided_names(self) -> frozenset[str]:
        """Names the sandbox binds itself (tools, final_answer, resource / prompt relays)."""
        extra = {"read_resource", "get_prompt"} if self._catalog else set()
        return frozenset(self.tools) | {"final_answer"} | extra

    def _load_catalog(self) -> list[tuple[str, list[Any]]]:
        """Fetch the connected server's resources and prompts (empty if it offers none)."""
        if self.client is None:
            return []
        catalog: list[tuple[str, list[Any]]] = []
        for heading, fetch in (("Resources", self.client.list_resources),
                               ("Prompts", self.client.list_prompts)):
            try:
                entries = list(fetch() or [])
            except Exception:
                continue
            if entries:
                catalog.append((heading, entries))
        return catalog

    def run(self, code: str) -> ExecuteResult:
        """Check, exectue, and service tool calls untill the snippet ends."""
        code, dropped = strip_provided_imports(code, self._provided_names())
        rejection = check_code(code, self.config, frozenset(self.tools))
        if rejection:
            return ExecuteResult(error=rejection)
        result = self._explain_unknown_call(code, self._run_checked(code))
        if dropped:
            note = (f"[NOTE] {', '.join(dropped)} {'is' if len(dropped) == 1 else 'are'} already available "
                    "in the sandbox; the import was skipped (no import is needed).\n")
            result = result.model_copy(update={"stdout": note + (result.stdout or "")})
        return result

    def _explain_unknown_call(self, code: str, result: ExecuteResult) -> ExecuteResult:
        """Add the tool list (and the closest names) to a NameError raised by calling an unknown function.

        Measured: nemotron-3-super wrote search_file(...) for search_code(...) on sympy-14711 and, seeing
        only "name 'search_file' is not defined", spent two more iterations guessing other names.
        """
        match = _UNDEFINED_NAME.search(result.error or "")
        if match is None or not self.tools:
            return result
        name = match.group(1)
        try:
            called = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
                         for node in ast.walk(ast.parse(code)))
        except SyntaxError:
            return result
        if not called:
            return result
        names = sorted(self._provided_names())
        hint = f"\n{name}() is not an available tool. Available tools: {', '.join(names)}."
        close = difflib.get_close_matches(name, names, n=2, cutoff=0.6)
        if close:
            hint += f" Did you mean {' or '.join(close)}?"
        return result.model_copy(update={"error": (result.error or "") + hint})

    def _run_checked(self, code: str) -> ExecuteResult:
        """Execute already-checked code and service tool calls until the snippet ends."""

        self._tool_truncated = False
        try:
            _send(self._cmd_w, {"kind": "run", "code": code})
        except OSError:
            self.reset()
            return ExecuteResult(error="worker was dead; restarted")

        grace = self.config.max_execution_time_seconds + 5
        while True:
            ready, _, _ = select.select([self._res_r], [], [], grace)
            if not ready:
                self.reset()
                return ExecuteResult(
                    timeout=True,
                    error=(f"timeout: the code ran past {self.config.max_execution_time_seconds}s and ignored "
                           "the time limit, so the worker was killed and restarted. All names defined in "
                           "earlier steps are gone; redefine what you need.")
                )
            try:
                message = _recv(self._res_r)
            except (EOFError, OSError, ValueError):
                self.reset()
                return ExecuteResult(
                    error=("the worker died (likely the memory limit) and was restarted. All names defined "
                           "in earlier steps are gone; redefine what you need.")
                )
            if message.get("kind") in ("tool_call", "resource_read", "prompt_get"):
                self._dispatch(message)
                continue
            return self._to_result(message)

    def _tool_doc(self, name: str, tool: Any) -> str:
        """Render one tool as a Python-looking signature plus its docs."""
        input_schema: dict[str, Any] = getattr(tool, "input_schema", None) or {}
        props: dict[str, Any] = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])
        types = {"string": "str", "integer": "int", "number": "float",
                 "boolean": "bool", "array": "list", "object": "dict"}

        params = []
        for arg, spec in props.items():
            hint = types.get(spec.get("type", ""), "Any")
            params.append(f"{arg}: {hint}" if arg in required
                          else f"{arg}: {hint} = ...")

        out = [f"{name}({', '.join(params)})"]
        description = getattr(tool, "description", "") or ""
        if description:
            out.append(f"   {description.strip()}")
        for arg, spec in props.items():
            if spec.get("description"):
                out.append(f"   - {arg}: {spec['description']}")
        return "\n".join(out)

    def manual(self) -> str:
        """Render the sandbox manual for the LLM's first prompt.

        Made from connected server.
        """
        lines = [
            "# Sandbox",
            "",
            "Your code runs in a restricted Python namespace.",
            "State persists between steps: names you define stay defined.",
            "",
            "## Rules",
            f"- Imports allowed: "
            f"{', '.join(sorted(self.config.authorized_imports))}",
            f"- File access limited to: "
            f"{', '.join(sorted(self.config.allowed_directories))}",
            "- No network access.",
            f"- Execution time limit: {self.config.max_execution_time_seconds}s "
            f"(tool calls are not counted against this limit).",
            f"- Memory limit: {self.config.max_memory_mb}MB.",
            "- Unavaiable: " + ", ".join(sorted(_BLOCKED)),
            "- Dunder attribute access (__class__, __globals__) and private module attributes "
            "(random._os) are blocked.",
            "",
            "## final_answer",
            ""
            "final_answer(value) ends the task. It is always available and",
            "is not an MCP tool. Call it once you have a verified answer.",
            "",
            "## Tools",
            "",
            "Call these as normal Python functions (positional or keyword arguments). They are "
            "already defined: do not import them.",
            "",
        ]
        if not self.tools:
            lines.append("No MCP server connected.")
            return "\n".join(lines)
        for name in sorted(self.tools):
            lines.append(self._tool_doc(name, self.tools[name]))
            lines.append("")
        lines.extend(self._catalog_doc())
        return "\n".join(lines)

    def _catalog_doc(self) -> list[str]:
        """List the connected server's resources and prompts for the manual.

        The task requires tools, resources and prompts to be exposed (課題 V.2): they are
        listed here and readable from the sandbox with read_resource() / get_prompt().
        A server that offers none simply contributes nothing.
        """
        if not self._catalog:
            return []
        lines: list[str] = []
        for heading, entries in self._catalog:
            lines.append(f"## {heading}")
            lines.append("")
            if heading == "Resources":
                lines.append("Read one with read_resource(uri) -> str.")
            else:
                lines.append("Render one with get_prompt(name, **arguments) -> str.")
            for entry in entries:
                name = getattr(entry, "uri", None) or getattr(entry, "name", "") or str(entry)
                description = (getattr(entry, "description", "") or "").strip()
                lines.append(f"- {name}: {description}" if description else f"- {name}")
            lines.append("")
        return lines

    def _catalog_reply(self, message: dict[str, Any]) -> dict[str, Any]:
        """Serve read_resource / get_prompt from the worker through the MCP client."""
        reply: dict[str, Any] = {"result": None, "error": None}
        if self.client is None:
            reply["error"] = "no MCP server connected"
            return reply
        try:
            if message.get("kind") == "resource_read":
                contents = self.client.read_resource(str(message.get("uri", ""))).contents
                reply["result"] = "\n".join(c.text or "" for c in contents)
            else:
                prompt = self.client.get_prompt(str(message.get("name", "")),
                                                {str(k): str(v) for k, v in (message.get("arguments") or {}).items()})
                reply["result"] = "\n".join(f"{m.role}: {m.content.text or ''}" for m in prompt.messages)
        except Exception as exc:
            reply["error"] = f"{type(exc).__name__}: {exc}"
        return reply

    def _dispatch(self, message: dict[str, Any]) -> None:
        """Run one tool in the parent and send the reply back."""
        if message.get("kind") in ("resource_read", "prompt_get"):
            _send(self._cmd_w, self._catalog_reply(message))
            return
        name = str(message.get("name", ""))
        handler = self.tools.get(name)
        reply: dict[str, Any] = {"result": None, "error": None}
        if handler is None:
            reply["error"] = f"unknown tool {name!r}"
        else:
            try:
                raw = handler(**(message.get("arguments") or {}))
                if isinstance(raw, str):
                    raw, was_cut = _truncate(
                        raw, self.config.max_output_chars
                    )
                    self._tool_truncated = self._tool_truncated or was_cut
                reply["result"] = raw
            except Exception as exc:
                reply["error"] = f"{type(exc).__name__}: {exc}"
        _send(self._cmd_w, reply)

    def _to_result(self, message: dict[str, Any]) -> ExecuteResult:
        """Turn a terminal worker message into an ExecuteResult."""
        base = {
            "stdout": message.get("stdout", ""),
            "stderr": message.get("stderr", ""),
            "timeout": bool(message.get("timeout", False)),
            "truncated": bool(message.get("truncated", False) or self._tool_truncated),
        }
        kind = message.get("kind")
        if kind == "final_answer":
            return ExecuteResult(
                final_answer=message.get("value"),
                final_answer_bool=True,  # This is for cases where final answer is ""
                **base
            )
        if kind == "error":
            return ExecuteResult(error=message.get("error"), **base)
        if kind == "interrupt":
            raise KeyboardInterrupt
        if kind == "exit":
            code = message.get("code", 0)
            raise SystemExit(code if isinstance(code, int) else 1)
        return ExecuteResult(value=message.get("value"), **base)


def _show(result: ExecuteResult) -> None:
    """Shows the result of the executed code."""
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.error:
        print(result.error)
    elif result.final_answer is not None:
        print(f"[final_answer] {result.final_answer}")
    elif result.value is not None:
        print(result.value)


def _repl(sandbox: Sandbox) -> int:
    """REPL-style mode: one entry per iteration, shared namespace."""
    print("sandbox - 'exit' or Ctrl-D to quit")
    while True:
        try:
            line = input(">>> ")
        except EOFError:
            print()
            return 0
        if line.strip() in {"exit", "quit"}:
            return 0
        if not line.strip():
            continue
        buffer = [line]
        while True:
            try:
                ast.parse("\n".join(buffer))
                break
            except SyntaxError:
                try:
                    more = input("... ")
                except EOFError:
                    return 0
                if not more.strip():
                    break
                buffer.append(more)
        try:
            _show(sandbox.run("\n".join(buffer)))
        except KeyboardInterrupt:
            print("\n[interrupted]")


def main() -> int:
    """Parse arguments and run a snippet, a file, or the REPL."""
    parser = argparse.ArgumentParser(prog="sandbox")
    parser.add_argument("config", nargs="?", help="JSON config file")
    parser.add_argument("--mcp-stdio", metavar="CMD",
                        help="command launching an MCP server over stdio")
    parser.add_argument("--mcp-server", metavar="URL",
                        help="streamable HTTP MCP server endpoint")
    parser.add_argument("-c", "--command", help="run one snippet and exit")
    parser.add_argument("-f", "--file", help="run a file and exit")
    args = parser.parse_args()

    if args.mcp_stdio and args.mcp_server:
        print("sandbox: use either --mcp-stdio or --mcp-server, not both", file=sys.stderr)
        return 2
    try:
        config = (
            SandboxConfig.model_validate_json(
                Path(args.config).read_text(encoding="utf-8")
            )
            if args.config else SandboxConfig()
        )
    except (OSError, ValueError) as exc:
        print(f"sandbox: cannot load config {args.config!r}: {exc}".splitlines()[0], file=sys.stderr)
        return 1
    try:
        sandbox = Sandbox(config=config, mcp_stdio=args.mcp_stdio, mcp_server=args.mcp_server)
    except Exception as exc:  # noqa: BLE001 - 接続失敗はトレースバックでなく1行で伝える
        print(f"sandbox: cannot connect to the MCP server: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    with sandbox:
        if args.command is not None:
            _show(sandbox.run(args.command))
        elif args.file is not None:
            _show(sandbox.run(Path(args.file).read_text(encoding="utf-8")))
        else:
            return _repl(sandbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
