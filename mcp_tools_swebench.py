"""mcp_tools_swebench.py - SWE-bench 用の MCP ツールサーバー(課題 V.5 の必須ツール).

起動時に環境変数を読み、ツールの実行場所を決める。
- SWE_DOCKER_IMAGE がある : そのイメージからコンテナを1つ起動し(--network none)、中で実行する。
- 無い                    : ホストの TESTBED_PATH で実行する(採点側がツールを単体で試すときの形)。
その他の環境変数: TESTBED_PATH(既定 /testbed) / SWE_CONTAINER_NAME / SWE_EVAL_SCRIPT_FILE /
SWE_COMMAND_TIMEOUT(既定120) / SWE_TEST_TIMEOUT(既定300) / SWE_DEADLINE_EPOCH(締め切りのUNIX時刻) /
SWE_RAW_TEST_OUTPUT(1 なら run_tests の出力を要約しない。BENCHMARK_REPORT の ablation 用)。

stdio(既定)または streamable HTTP(--transport http)で待ち受ける。終了時(stdinのEOF / SIGTERM /
SIGINT / 例外)にはコンテナを必ず削除する。
"""
import argparse
import atexit
import os
import signal
import sys
import uuid
from types import FrameType

from src.mcp_core.server.http import HttpServerTransport
from src.mcp_core.server.server import MCPServer
from src.mcp_core.swe_bench.executor import DockerExecutor, Executor, LocalExecutor
from src.mcp_core.swe_bench.workspace import SweWorkspace

server = MCPServer(name="agent-smith-swebench", version="0.1.0")
_workspace: SweWorkspace | None = None


def _float_env(name: str, default: float | None) -> float | None:
    """数値の環境変数を読む。無い・壊れていれば既定値.

    Args:
        name (str): 環境変数名。
        default (float | None): 既定値。

    Returns:
        float | None: 読んだ値。
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_workspace() -> SweWorkspace:
    """環境変数から作業場を組み立てる。Dockerモードならここでコンテナを起動する.

    Returns:
        SweWorkspace: ツールの実装。
    """
    image = os.environ.get("SWE_DOCKER_IMAGE")
    executor: Executor
    if image:
        docker = DockerExecutor(image, os.environ.get("SWE_CONTAINER_NAME") or f"agentsmith-{uuid.uuid4().hex[:12]}")
        docker.start()
        executor = docker
    else:
        executor = LocalExecutor()
    return SweWorkspace(
        executor=executor,
        testbed=os.environ.get("TESTBED_PATH", "/testbed"),
        eval_script_file=os.environ.get("SWE_EVAL_SCRIPT_FILE") or None,
        command_timeout=_float_env("SWE_COMMAND_TIMEOUT", 120.0) or 120.0,
        test_timeout=_float_env("SWE_TEST_TIMEOUT", 300.0) or 300.0,
        deadline_epoch=_float_env("SWE_DEADLINE_EPOCH", None),
        summarize_tests=os.environ.get("SWE_RAW_TEST_OUTPUT", "") not in ("1", "true", "yes"),
    )


def _ws() -> SweWorkspace:
    """作業場を返す(サーバー起動前に呼ばれたら、その場で組み立てる).

    Returns:
        SweWorkspace: ツールの実装。
    """
    global _workspace
    if _workspace is None:
        _workspace = build_workspace()
    return _workspace


# ====== File System Tools ======

@server.tool()
def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file with line numbers, one line per row as "<line_number>: <line_content>".

    start_line / end_line are 1-based and inclusive; omit them to read from the start / to the end.
    Relative paths are resolved against the repository root.
    """
    return _ws().read_file(filepath, start_line, end_line)


@server.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """Replace an exact string in a file with a new string.

    old_str must occur exactly once (copy it from read_file output without the "N: " prefixes).
    For .py files an edit that breaks the syntax is reverted and reported as SYNTAX ERROR.
    """
    return _ws().edit_file(filepath, old_str, new_str)


@server.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files under a directory (recursively, skipping .git) whose name matches a glob pattern."""
    return _ws().list_files(directory, pattern)


# ====== Code Search Tools ======

@server.tool()
def search_code(pattern: str, file_pattern: str = "*") -> str:
    """Grep-like regex search over the repository.

    Output lines: "/absolute/path/to/file.py:<line_number> <line_content>". file_pattern is a glob such as "*.py".
    """
    return _ws().search_code(pattern, file_pattern)


@server.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """Find where a function or class is defined (def / async def / class), same format as search_code."""
    return _ws().search_function_or_class_definition_in_code(name)


@server.tool()
def find_references(name: str, filepath: str | None = None, line: int | None = None) -> str:
    """Find all usages of a symbol (whole word, .py files), same format as search_code.

    filepath / line optionally point at the definition; matches in that file are listed first.
    """
    return _ws().find_references(name, filepath, line)


# ====== Execution Tools ======

@server.tool()
def run_tests() -> str:
    """Run the task's evaluation script and summarize the result (exit code, failing tests, test output)."""
    return _ws().run_tests()


@server.tool()
def get_patch() -> str:
    """Return the unified git diff of all changes made to the repository (test files excluded)."""
    return _ws().get_patch()


@server.tool()
def run_command(command: str, workdir: str = "") -> str:
    """Execute a shell command in the given working directory (default: the repository root).

    Returns the exit code, stdout and stderr.
    """
    return _ws().run_command(command, workdir or None)


def _close_workspace() -> None:
    """作業場(コンテナ)を片付ける。何度呼んでもよい."""
    if _workspace is not None:
        _workspace.executor.close()


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """SIGTERM / SIGINT で SystemExit を送出し、finally / atexit の後片付けを走らせる.

    Args:
        signum (int): 受けたシグナル番号。
        frame (FrameType | None): 受けた時点のフレーム(未使用)。
    """
    raise SystemExit(128 + signum)


def main() -> None:
    """引数を読み、作業場を用意してから待ち受ける."""
    parser = argparse.ArgumentParser(description="SWE-bench MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)
    atexit.register(_close_workspace)
    try:
        _ws()
    except RuntimeError as error:
        print(f"mcp_tools_swebench: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    try:
        if args.transport == "http":
            server.run(HttpServerTransport(host=args.host, port=args.port))
        else:
            server.run()
    finally:
        _close_workspace()


if __name__ == "__main__":
    main()
