"""mcp_tools_mbpp.py"""

import sys
import json
import subprocess
import tempfile
import argparse

from src.mcp_core.server.server import MCPServer
from src.mcp_core.server.http import HttpServerTransport

server = MCPServer(name="agent-smith-mbpp", version="0.1.0")
# 1回の検証に許す秒数。MBPPは1タスク120秒なので、無限ループの解答で長く止まらないよう短めにする
TIMEOUT = 10
# LLMへ返す出力の上限文字数(長すぎるとObservationが切り詰められ、肝心の失敗理由が落ちる)
OUTPUT_LIMIT = 1500
# 1件のテスト結果の上限文字数
DETAIL_LIMIT = 300
# 子プロセスが結果JSONの前に付ける印(解答コードのprint出力と区別するため)
RESULT_MARKER = "__RUN_TESTS_RESULT__"

# 子プロセスで実行する検証スクリプト。解答コードを実行したあと、テストを1件ずつ実行し、
# 落ちたassertは「左辺の実際の値」まで調べて返す。
# 1本に連結して `python -c` で実行すると、失敗時に「<string>の4行目でAssertionError」としか
# 分からず、LLMがどのテストで何を返したのか知る手段が無かった(実測: 同じ誤答を5回繰り返した)。
_RUNNER = r"""
import ast, json, sys, traceback
source, tests, marker = json.loads(sys.stdin.read())
namespace = {"__name__": "__main__"}
results = []
try:
    exec(compile(source, "<solution>", "exec"), namespace)
except BaseException as error:
    lines = traceback.format_exception_only(type(error), error)
    results.append({"test": "<your code>", "passed": False, "detail": "".join(lines).strip()})
else:
    for test in tests:
        is_assert = test.lstrip().startswith("assert")
        try:
            exec(compile(test, "<test>", "exec"), namespace)
            if is_assert:
                results.append({"test": test, "passed": True})
        except AssertionError:
            detail = "assertion failed"
            try:
                node = ast.parse(test.strip()).body[0]
                if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
                    left = node.test.left
                    value = eval(compile(ast.Expression(left), "<test>", "eval"), namespace)
                    detail = f"{ast.unparse(left)} returned {value!r}"
            except BaseException as inner:
                detail = f"assertion failed (could not re-evaluate: {type(inner).__name__})"
            results.append({"test": test, "passed": False, "detail": detail})
        except BaseException as error:
            lines = traceback.format_exception_only(type(error), error)
            results.append({"test": test, "passed": False, "detail": "".join(lines).strip()})
sys.stdout.flush()
print(marker + json.dumps(results))
"""


def _clip(text: str, limit: int) -> str:
    """長い文字列を上限文字数で切り詰める."""
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _format_report(results: list[dict[str, object]], stdout: str, stderr: str) -> tuple[bool, str]:
    """子プロセスの結果を、LLMが読める1つの文字列にまとめる.

    Args:
        results (list[dict[str, object]]): テストごとの結果。
        stdout (str): 解答コードやテストがprintした内容。
        stderr (str): 標準エラー出力。

    Returns:
        tuple[bool, str]: (全テスト合格か, output文字列)。
    """
    asserts = [result for result in results if result["test"] != "<your code>"]
    failed = [result for result in results if not result["passed"]]
    passed_count = len(asserts) - len([result for result in failed if result["test"] != "<your code>"])
    lines = [f"{passed_count}/{len(asserts)} tests passed."] if asserts else []
    for result in failed:
        lines.append(_clip(f"FAILED: {result['test']}  ->  {result.get('detail', '')}", DETAIL_LIMIT))
    if stdout.strip():
        lines.append("--- stdout ---\n" + stdout.strip())
    if stderr.strip():
        lines.append("--- stderr ---\n" + stderr.strip())
    return not failed, _clip("\n".join(lines), OUTPUT_LIMIT)


@server.tool()
def run_tests(code: str, test_list: list[str]) -> str:
    """Run candidate solution code against a list of assert statements.

    Returns a JSON string with "success" (True if all tests passed),
    "message" and "output". On failure, "output" names each failing assert and
    what its left-hand side actually returned, plus the stdout/stderr of the run.
    """
    payload = json.dumps([code, list(test_list), RESULT_MARKER])
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 自動で削除される temp_dir 内で、環境変数を空にした隔離モードの子プロセスで実行
            result = subprocess.run([sys.executable, '-I', '-c', _RUNNER], input=payload, cwd=temp_dir, env={},
                                    capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "message": "Timeout",
                           "output": f"Execution did not finish within {TIMEOUT} seconds "
                                     "(infinite loop or a very slow algorithm?)."})
    except Exception as e:
        return json.dumps({"success": False, "message": type(e).__name__, "output": str(e)})

    # 結果の印より前は解答コードのprint出力。印が無ければ子プロセス自体が異常終了している。
    stdout, marker, tail = result.stdout.rpartition(RESULT_MARKER)
    if not marker:
        output = _clip((result.stdout + result.stderr).strip() or f"exit code {result.returncode}", OUTPUT_LIMIT)
        return json.dumps({"success": False, "message": "Crashed", "output": output})
    try:
        results = json.loads(tail)
    except json.JSONDecodeError:
        return json.dumps({"success": False, "message": "Crashed", "output": _clip(result.stdout, OUTPUT_LIMIT)})

    success, output = _format_report(results, stdout, result.stderr)
    return json.dumps({"success": success, "message": "passed" if success else "Tests failed", "output": output})


def main() -> None:
    parser = argparse.ArgumentParser(description="MBPP MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        server.run(HttpServerTransport(host=args.host, port=args.port))
        pass
    else:
        server.run()


if __name__ == "__main__":
    main()
