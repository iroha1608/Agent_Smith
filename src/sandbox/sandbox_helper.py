"""Sandbox helper functions."""
from __future__ import annotations
from src.models.sandbox_config import SandboxConfig
from typing import Any, Callable
import os
import signal
import resource
import json
import struct
import sys


def _tool_proxy(name: str, params: list[str], out_fd: int, in_fd: int) -> Callable[..., Any]:
    """Build a stub that forwards a call to the parent's MCP client.

    The interval timer is cancelled across the round trip and the
    remainder restored afterwards, so a tool spawning a container does
    not burn the snippet's exectuion budget.
    Positional arguments are mapped to parameter names in the order of the
    tool's input schema (課題 III.1 の例 read_file("models.py", 1, 50) のように呼べる).
    """
    def proxy(*args: Any, **kwargs: Any) -> Any:
        """Proxy the tools."""
        if len(args) > len(params):
            raise TypeError(f"{name}() takes at most {len(params)} positional arguments "
                            f"({', '.join(params) or 'none'}), {len(args)} given")
        for param, value in zip(params, args):
            if param in kwargs:
                raise TypeError(f"{name}() got multiple values for argument {param!r}")
            kwargs[param] = value
        # MCPサーバーは未知の引数を黙って捨てるので、捨てたことを伝える(実測: search_code(filepath=...) と
        # 書いたモデルが、1ファイルに絞れたと思い込んでリポジトリ全体の検索結果を読んでいた)
        unknown = [key for key in kwargs if params and key not in params]
        if unknown:
            for key in unknown:
                kwargs.pop(key)
            print(f"[NOTE] {name}() has no parameter {', '.join(map(repr, unknown))}, so "
                  f"{'it was' if len(unknown) == 1 else 'they were'} ignored. Parameters: {', '.join(params)}.")
        return _relay(out_fd, in_fd, {"kind": "tool_call", "name": name, "arguments": kwargs},
                      f"tool {name!r}")
    proxy.__name__ = name
    proxy.__doc__ = f"MCP tool {name!r}, executed outside the sandbox."
    return proxy


def _relay(out_fd: int, in_fd: int, message: dict[str, Any], what: str) -> Any:
    """Send a request to the parent (which owns the MCP client) and return its result.

    Args:
        out_fd (int): pipe to the parent.
        in_fd (int): pipe from the parent.
        message (dict[str, Any]): the request.
        what (str): label used in the error message.

    Returns:
        Any: the result returned by the parent.

    Raises:
        RuntimeError: when the parent reports an error.
    """
    remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0.0)
    try:
        _send(out_fd, message)
        reply = _recv(in_fd)
    finally:
        if remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining)
    if reply.get("error") is not None:
        raise RuntimeError(f"{what} failed: {reply['error']}")
    return reply.get("result")


def _catalog_relays(out_fd: int, in_fd: int) -> dict[str, Callable[..., Any]]:
    """Build read_resource() / get_prompt(), which relay MCP resources and prompts to the parent.

    Like final_answer they are sandbox built-ins, not MCP tools (課題 V.2: resources and
    prompts must be exposed).
    """
    def read_resource(uri: str) -> Any:
        """Return the text of an MCP resource of the connected server."""
        return _relay(out_fd, in_fd, {"kind": "resource_read", "uri": uri}, f"resource {uri!r}")

    def get_prompt(name: str, **arguments: str) -> Any:
        """Return the rendered text of an MCP prompt of the connected server."""
        return _relay(out_fd, in_fd, {"kind": "prompt_get", "name": name, "arguments": arguments},
                      f"prompt {name!r}")

    return {"read_resource": read_resource, "get_prompt": get_prompt}


def _apply_limits(config: SandboxConfig) -> None:
    """Drop privileges, Hard limits only go down.

    RLIMIT_AS caps virtual address space, which is what makes a runaway
    allocation raise MemoryError instead of swapping the machine.
    """
    max_bytes = config.max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _disable_network() -> None:
    """Neuter sockets in the child, in case a module reaches one."""
    try:
        import socket
    except ImportError:
        return

    def blocked(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("network access is disabled")

    socket.socket = blocked  # type: ignore[assignment,misc]
    socket.create_connection = blocked
    socket.getaddrinfo = blocked


class _Timeout(BaseException):
    """Raised for timeout when too long. SIGALRM handler in child.

    BaseException (not Exception) so that `except Exception: pass` in the
    sandboxed code cannot swallow the timeout.
    """


def _on_alarm(*_args: Any) -> None:
    """Raise timeout."""
    raise _Timeout()


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cap text, stating what was dropped.

    The marker matters: without it the LLM treats a cut-off result as
    complete and reasons from a false premise.
    """
    if len(text) <= limit:
        return text, False
    dropped = len(text) - limit
    marked = (
        f"{text[:limit]}\n[output truncated: {dropped} of {len(text)} "
        f"characters omitted; narrow your request to see the rest]"
    )
    return marked, True


# send to child/parent
def _send(fd: int, message: dict[str, Any]) -> None:
    """Write one length-prefixed JSON message.

    fd is the legnth of the message.
    """
    payload = json.dumps(message).encode("utf-8")
    data = memoryview(struct.pack("!I", len(payload)) + payload)
    while data:
        data = data[os.write(fd, data):]


# receieve from child/parent
def _recv(fd: int) -> dict[str, Any]:
    """Read one length-prefixed JSON message."""
    def read_exactly(size: int) -> bytes:
        """Read exactly amount of bytes."""
        chunks: list[bytes] = []
        while size:
            chunk = os.read(fd, size)
            if not chunk:
                raise EOFError("pipe closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    (length,) = struct.unpack("!I", read_exactly(4))
    message: dict[str, Any] = json.loads(read_exactly(length).decode("utf-8"))
    return message


# Events denied outright once user code may run (process creation, network, native code,
# heap inspection, raising resource limits).
_DENIED_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "os.kill", "os.killpg", "pty.spawn", "resource.setrlimit", "resource.prlimit",
    "gc.get_objects", "gc.get_referrers", "gc.get_referents", "sys.setprofile",
})
_DENIED_PREFIXES = ("socket.", "ctypes.", "winreg.", "msvcrt.")
# Filesystem events whose path arguments must stay inside the allowlist.
_PATH_EVENTS = frozenset({
    "os.listdir", "os.scandir", "os.chdir", "os.mkdir", "os.remove", "os.rmdir", "os.rename",
    "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate", "os.utime", "os.chflags",
    "os.lchmod", "os.lchown", "os.mkfifo", "os.mknod", "os.setxattr", "os.removexattr",
    "os.getxattr", "os.listxattr", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.chown", "shutil.make_archive",
    "shutil.unpack_archive",
})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _under(path: str, roots: list[str]) -> bool:
    """Whether the resolved path is one of the roots or inside one."""
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _install_audit_hook(config: SandboxConfig) -> None:
    """Enforce the filesystem / network / process restrictions at the interpreter level.

    The static check and the guarded builtins can be bypassed by reaching os/sys/codecs
    through attributes of allowed modules (e.g. random._os, typing.sys.modules['os'],
    json.codecs.open). A PEP 578 audit hook sees every such operation no matter how the
    module was reached, and it cannot be removed once installed (標準ライブラリだけで実装).
    - open(): writes only inside allowed_directories; reads also from the interpreter's
      own install (stdlib / site-packages) so lazy imports keep working.
    - directory listing / file mutation events: only inside allowed_directories.
    - process creation, sockets, ctypes, heap inspection, rlimit changes: denied.
    """
    allowed = [os.path.realpath(d) for d in config.allowed_directories]
    readable = allowed + sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
    busy = [False]

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if busy[0]:
            return
        if event in _DENIED_EVENTS or event.startswith(_DENIED_PREFIXES):
            raise PermissionError(f"{event} is not allowed in the sandbox")
        if event == "open":
            target, mode, flags = (tuple(args) + (None, None, None))[:3]
            if isinstance(target, int):
                raise PermissionError("opening raw file descriptors is not allowed in the sandbox")
            busy[0] = True
            try:
                path = os.path.realpath(os.fsdecode(target))
            finally:
                busy[0] = False
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            if not _under(path, allowed if writing else readable):
                raise PermissionError(f"path outside allowlist: {path}")
        elif event in _PATH_EVENTS:
            for arg in args:
                if isinstance(arg, (str, bytes, os.PathLike)):
                    busy[0] = True
                    try:
                        path = os.path.realpath(os.fsdecode(arg))
                    finally:
                        busy[0] = False
                    if not _under(path, allowed):
                        raise PermissionError(f"{event} outside allowlist: {path}")

    sys.addaudithook(hook)


def _scrub_environment() -> None:
    """Drop the parent's environment (it holds API keys loaded from .env).

    The worker is forked from the agent process; without this, code that reaches os.environ
    through an allowed module could print the keys into the LLM transcript.
    """
    os.environ.clear()


def _ensure_scratch_dirs(config: SandboxConfig) -> None:
    """Create the allowed directories that can be created (e.g. /tmp/agent), ignoring failures."""
    for directory in config.allowed_directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            continue
