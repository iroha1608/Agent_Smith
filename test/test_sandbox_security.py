"""実サンドボックス(Sandbox)のセキュリティと互換性のテスト.

課題 VI の exam_sandbox(import / builtin / network / path / timeout / memory / MCP protocol)を
想定し、ギャップ監査で実証された脱出経路(許可モジュールの属性経由で os / sys / codecs / _socket に
届く)が塞がっていること、通常の MBPP 風のコードが今までどおり動くことを、モックではなく
本物のサンドボックス(fork した子プロセス + 監査フック)で確かめる。
"""
import os
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.models.sandbox_config import SandboxConfig
from src.sandbox.sandbox_main import Sandbox

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sandbox() -> Iterator[Sandbox]:
    """MCPサーバー無しのサンドボックス(実行時間の上限は短め)."""
    with Sandbox(config=SandboxConfig(max_execution_time_seconds=2)) as box:
        yield box


def _blocked(result_error: str | None, stdout: str = "") -> bool:
    text = (result_error or "") + stdout
    return any(word in text for word in ("not allowed", "outside allowlist", "PermissionError", "private module"))


@pytest.mark.parametrize("code", [
    "import os",
    "import socket",
    "import subprocess",
    "from os import path",
    "import random\nprint(random._os.listdir('/'))",
    "import string\nprint(string._re)",
])
def test_static_blocks(sandbox: Sandbox, code: str) -> None:
    """許可外の import と、許可モジュールの private 属性は実行前に拒否する."""
    result = sandbox.run(code)
    assert _blocked(result.error), result


@pytest.mark.parametrize("code", [
    # 別名にして静的チェックをすり抜けても、監査フックが操作そのものを止める
    "print(getattr is None)",
    "import typing\nprint(typing.sys.modules['os'].listdir('/'))",
    "import typing\nprint(typing.sys.modules['builtins'].open('/etc/hostname').read())",
    "import json\nprint(json.codecs.open('/etc/hostname').read())",
    "import typing\nprint(typing.sys.modules['_socket'].socket())",
    "import typing\nprint(typing.sys.modules['os'].system('echo pwned'))",
    "import typing\nprint(typing.sys.modules['os'].popen('echo pwned').read())",
    "import typing\nprint(typing.sys.modules['os'].open('/etc/passwd', 0))",
    "open('/etc/passwd').read()",
    "open('/tmp/outside_allowlist.txt', 'w').write('x')",
])
def test_runtime_escapes_are_denied(sandbox: Sandbox, code: str) -> None:
    """許可モジュールの属性経由で届いた os / builtins / codecs / _socket でも、危険な操作は失敗する."""
    result = sandbox.run(code)
    assert "pwned" not in (result.stdout or "")
    if "getattr is None" in code:
        assert result.error is not None  # getattr は組み込みから外してある
        return
    assert _blocked(result.error, result.stdout or ""), result


def test_environment_secrets_are_not_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """親プロセスの環境変数(.env から読んだAPIキー)は、サンドボックスの子から見えない."""
    monkeypatch.setenv("AGENT_SMITH_TEST_SECRET", "sk-should-not-leak")
    with Sandbox(config=SandboxConfig(max_execution_time_seconds=2)) as box:
        result = box.run("import typing\nprint(dict(typing.sys.modules['os'].environ))")
    assert "sk-should-not-leak" not in (result.stdout or "")


def test_allowed_scratch_directory_is_writable(sandbox: Sandbox) -> None:
    """許可ディレクトリ(/tmp/agent)は読み書きできる(起動時に作られる)."""
    code = "open('/tmp/agent/note.txt', 'w').write('ok')\nprint(open('/tmp/agent/note.txt').read())"
    result = sandbox.run(code)
    assert result.error is None and result.stdout.strip() == "ok", result


def test_normal_mbpp_style_code_still_works(sandbox: Sandbox) -> None:
    """許可モジュールの通常利用・クラス定義・遅延 import(datetime.strptime)は今までどおり動く."""
    code = textwrap.dedent("""
        import re, json, math, collections, itertools, heapq, functools, random, string, typing
        import copy, bisect, operator, array, cmath, datetime
        from collections import Counter, abc

        class Node:
            def __init__(self, value):
                self._value = value

            def double(self):
                return self._value * 2

        print(Node(21).double(), math.gcd(12, 18), Counter('aab')['a'], re.sub('a', 'b', 'aa'))
        print(datetime.datetime.strptime('2026-10-15', '%Y-%m-%d').year, json.dumps({'k': 1}))
        print(sorted(itertools.permutations([1, 2]))[0], heapq.nsmallest(1, [3, 1]), random.Random(1).random() < 1)
    """)
    result = sandbox.run(code)
    assert result.error is None, result.error
    assert result.stdout.splitlines() == ["42 6 2 bb", '2026 {"k": 1}', "(1, 2) [1] True"]


def test_timeout_returns_partial_output(sandbox: Sandbox) -> None:
    """無限ループは制限時間で止まり、それまでの出力と「部分的」である旨が返る."""
    result = sandbox.run("print('started')\nx = 0\nwhile True:\n    x += 1")
    assert result.timeout is True
    assert "started" in (result.stdout or "")
    assert "partial" in (result.error or "")


def test_timeout_cannot_be_swallowed(sandbox: Sandbox) -> None:
    """`except Exception: pass` で囲んだループでもタイムアウトは効き、作り直しを明示する.

    CPython は try の直前ではシグナルの確認を飛ばすため、この形のループには SIGALRM が
    届かない。その場合も親が制限時間+5秒で子を止め、名前空間が消えたことを伝える。
    """
    code = "print('started')\nwhile True:\n    try:\n        pass\n    except Exception:\n        pass"
    result = sandbox.run(code)
    assert result.timeout is True
    assert "names defined in earlier steps are gone" in (result.error or "")
    assert sandbox.run("print(1 + 1)").stdout.strip() == "2"


def test_memory_limit(sandbox: Sandbox) -> None:
    """メモリ上限を超える確保は失敗する."""
    result = sandbox.run("x = bytearray(2 * 1024 ** 3)")
    assert result.error is not None and "memory" in result.error.lower()


def test_system_exit_propagates_with_integer_code(sandbox: Sandbox) -> None:
    """SystemExit は握りつぶさず、整数の終了コードのまま親へ伝わる(課題 V.2)."""
    with pytest.raises(SystemExit) as info:
        sandbox.run("raise SystemExit(3)")
    assert info.value.code == 3


def test_mcp_tools_accept_positional_args_and_skip_provided_imports() -> None:
    """MCPツールは位置引数でも呼べ、ツール名の import は no-op として注記付きで実行される."""
    command = f"{sys.executable} {_REPO / 'mcp_tools_mbpp.py'}"
    with Sandbox(config=SandboxConfig(max_execution_time_seconds=10), mcp_stdio=command) as box:
        result = box.run("import json, run_tests\nprint(json.loads(run_tests('def f():\\n    return 1', "
                         "['assert f() == 1']))['success'])")
    assert result.error is None, result.error
    assert result.stdout.startswith("[NOTE] run_tests is already available")
    assert result.stdout.strip().endswith("True")


def test_unknown_server_resources_prompts_and_tools(tmp_path: Path) -> None:
    """未知のMCPサーバーでも、ツール・resources・prompts を動的に発見して使える."""
    server = tmp_path / "other_server.py"
    server.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_REPO)!r})
        from src.mcp_core.server.server import MCPServer
        server = MCPServer(name="other", version="9.9")

        @server.tool()
        def shout(text: str, times: int = 1) -> str:
            \"\"\"Repeat the text in upper case.\"\"\"
            return " ".join([text.upper()] * times)

        @server.resource("memo://greeting", description="A greeting")
        def greeting() -> str:
            return "hello from a resource"

        @server.prompt()
        def ask(topic: str) -> str:
            \"\"\"Ask about a topic.\"\"\"
            return "Tell me about " + topic

        server.run()
    """), encoding="utf-8")
    with Sandbox(config=SandboxConfig(max_execution_time_seconds=10), mcp_stdio=f"{sys.executable} {server}") as box:
        manual = box.manual()
        result = box.run("print(shout('hi', 2))\nprint(read_resource('memo://greeting'))\nprint(get_prompt('ask', topic='MCP'))")
    assert "shout(text: str, times: int = ...)" in manual
    assert "memo://greeting" in manual and "read_resource(uri)" in manual
    assert result.error is None, result.error
    assert result.stdout.splitlines() == ["HI HI", "hello from a resource", "user: Tell me about MCP"]
    assert os.path.isdir("/tmp/agent")
