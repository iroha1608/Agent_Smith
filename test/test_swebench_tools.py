"""SWE-bench の必須ツール(SweWorkspace)のテスト.

ローカルモード(ホスト上の一時的なgitリポジトリ)で全ツールを実際に動かし、
課題 V.5 の出力形式・編集の構文チェック・パッチの中身・評価スクリプトの扱いを確かめる。
"""
import subprocess
from pathlib import Path

import pytest

from src.mcp_core.swe_bench.executor import LocalExecutor
from src.mcp_core.swe_bench.workspace import SweWorkspace, eval_test_files, summarize_test_output


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """calc/ops.py と tests/test_ops.py を持つ小さなリポジトリ."""
    root = tmp_path / "testbed"
    (root / "calc").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "calc" / "ops.py").write_text(
        "def divide(a, b):\n    return a * b\n\n\nclass Calculator:\n    def run(self):\n        return divide(4, 2)\n",
        encoding="utf-8")
    (root / "tests" / "test_ops.py").write_text("from calc.ops import divide\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "-c", "user.email=a@b", "-c", "user.name=t", "add", ".")
    _git(root, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init")
    return root


def _workspace(repo: Path, eval_script: Path | None = None, **kwargs: object) -> SweWorkspace:
    return SweWorkspace(LocalExecutor(), testbed=str(repo),
                        eval_script_file=str(eval_script) if eval_script else None, **kwargs)  # type: ignore[arg-type]


def test_read_file_formats_lines_and_ranges(repo: Path) -> None:
    ws = _workspace(repo)
    assert ws.read_file("calc/ops.py", 1, 2) == "1: def divide(a, b):\n2:     return a * b"
    assert ws.read_file(str(repo / "calc" / "ops.py")).startswith("1: def divide(a, b):")
    assert "file not found" in ws.read_file("nope.py")
    assert "invalid range" in ws.read_file("calc/ops.py", 3, 1)
    assert "past the end" in ws.read_file("calc/ops.py", 99, 100)
    assert "is a directory" in ws.read_file("calc")


def test_edit_file_replaces_once_and_reverts_syntax_errors(repo: Path) -> None:
    ws = _workspace(repo)
    reply = ws.edit_file("calc/ops.py", "    return a * b", "    return a / b")
    assert reply.startswith(f"Edited {repo}/calc/ops.py (replaced 1 occurrence starting at line 2).")
    assert "2:     return a / b" in reply
    broken = ws.edit_file("calc/ops.py", "    return a / b", "    return (a / b")
    assert broken.startswith("SYNTAX ERROR") and "reverted" in broken
    assert "return a / b" in (repo / "calc" / "ops.py").read_text(encoding="utf-8")
    assert "not found" in ws.edit_file("calc/ops.py", "missing text", "x")
    assert "found 2 times" in ws.edit_file("calc/ops.py", "divide", "div")


def test_list_files_returns_absolute_paths(repo: Path) -> None:
    listing = _workspace(repo).list_files(".", "*.py").splitlines()
    assert listing == [f"{repo}/calc/ops.py", f"{repo}/tests/test_ops.py"]
    assert "No files matching" in _workspace(repo).list_files("calc", "*.txt")


def test_search_tools_use_the_required_format(repo: Path) -> None:
    ws = _workspace(repo)
    assert ws.search_code("return a \\* b", "*.py") == f"{repo}/calc/ops.py:2     return a * b"
    assert ws.search_function_or_class_definition_in_code("divide") == f"{repo}/calc/ops.py:1 def divide(a, b):"
    assert ws.search_function_or_class_definition_in_code("Calculator.run") == f"{repo}/calc/ops.py:6     def run(self):"
    refs = ws.find_references("divide", "tests/test_ops.py").splitlines()
    assert refs[0] == f"{repo}/tests/test_ops.py:1 from calc.ops import divide"
    assert len(refs) == 3
    assert ws.search_code("no_such_symbol_xyz").startswith("No matches found")
    assert "not a valid Python identifier" in ws.find_references("1bad")


@pytest.mark.parametrize("file_pattern", [
    "calc/ops.py",          # リポジトリからの相対パス
    "{repo}/calc/ops.py",   # 絶対パス
    "calc/*.py",            # ディレクトリ付きの glob
    "**/*.py",              # 再帰 glob
    "*/calc/ops.py",        # 先頭がワイルドカードの部分パス
    "calc",                 # ディレクトリ
    "ops.py",               # ワイルドカード無しのファイル名
    "./calc/",              # ./ 付き・末尾スラッシュ付きのディレクトリ
])
def test_search_code_accepts_path_patterns(repo: Path, file_pattern: str) -> None:
    """パスで絞り込んでも見つかる(実測: モデルの多くがパスを渡し、ファイル名だけを見る grep では毎回0件だった)."""
    ws = _workspace(repo)
    found = ws.search_code("return a \\* b", file_pattern.format(repo=repo))
    assert found == f"{repo}/calc/ops.py:2     return a * b"


def test_search_code_path_patterns_report_why_nothing_was_found(repo: Path) -> None:
    """対象ファイルが無いのか、ファイルはあるが一致しないのかを区別して返す."""
    ws = _workspace(repo)
    assert ws.search_code("divide", "calc/nope.py").startswith("No files match 'calc/nope.py'")
    assert ws.search_code("no_such_symbol_xyz", "calc/*.py") == \
        "No matches found for pattern 'no_such_symbol_xyz' in calc/*.py."
    assert ws.search_code("(unclosed", "calc/*.py").startswith("Error: ")
    # tests/ は calc/*.py に含まれない
    assert ws.search_code("import divide", "calc/*.py").startswith("No matches found")


def test_get_patch_includes_new_files_and_excludes_eval_test_files(repo: Path, tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.sh"
    base = _git(repo, "rev-parse", "HEAD").strip()
    eval_script.write_text(f"git checkout {base} tests/test_ops.py\n", encoding="utf-8")
    ws = _workspace(repo, eval_script)
    ws.edit_file("calc/ops.py", "    return a * b", "    return a / b")
    (repo / "calc" / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "tests" / "test_ops.py").write_text("changed\n", encoding="utf-8")
    patch = ws.get_patch()
    assert "diff --git a/calc/ops.py b/calc/ops.py" in patch
    assert "+    return a / b" in patch
    assert "diff --git a/calc/helpers.py b/calc/helpers.py" in patch
    assert "tests/test_ops.py" not in patch


def test_run_tests_summarizes_and_restores_test_files(repo: Path, tmp_path: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    eval_script = tmp_path / "eval.sh"
    eval_script.write_text(
        "set -uxo pipefail\n"
        f"git checkout {base} tests/test_ops.py\n"
        "echo 'patched by eval' >> tests/test_ops.py\n"
        ": '>>>>> Start Test Output'\n"
        "echo 'tests/test_ops.py::test_divide FAILED'\n"
        "echo 'FAILED tests/test_ops.py::test_divide - AssertionError'\n"
        "cat <<'TXT'\nx.py:1: DeprecationWarning: old\n  import old\nTXT\n"
        "echo '=== 1 failed in 0.01s ==='\n"
        ": '>>>>> End Test Output'\n"
        f"git checkout {base} tests/test_ops.py\n", encoding="utf-8")
    report = _workspace(repo, eval_script).run_tests()
    assert report.startswith("exit_code: 0")
    assert "Failing tests (1): tests/test_ops.py::test_divide" in report
    assert "x.py:1: DeprecationWarning: old" not in report and "import old" not in report
    assert "=== 1 failed in 0.01s ===" in report
    assert (repo / "tests" / "test_ops.py").read_text(encoding="utf-8") == "from calc.ops import divide\n"


def test_run_tests_timeout_restores_and_marks_partial(repo: Path, tmp_path: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    eval_script = tmp_path / "eval.sh"
    eval_script.write_text(
        f"git checkout {base} tests/test_ops.py\necho changed > tests/test_ops.py\n"
        ": '>>>>> Start Test Output'\necho started\nsleep 30\n", encoding="utf-8")
    report = _workspace(repo, eval_script, test_timeout=1.0).run_tests()
    assert "[TIMEOUT]" in report
    assert (repo / "tests" / "test_ops.py").read_text(encoding="utf-8") == "from calc.ops import divide\n"


def test_run_tests_without_eval_script_is_explicit(repo: Path) -> None:
    assert "no evaluation script is configured" in _workspace(repo).run_tests()


def test_deadline_stops_commands_before_they_start(repo: Path) -> None:
    ws = _workspace(repo, deadline_epoch=0.0)
    assert ws.run_command("echo hi").startswith("Error: time budget exhausted")
    assert ws.read_file("calc/ops.py").startswith("Error: time budget exhausted")


def test_run_command_reports_exit_code_and_streams(repo: Path) -> None:
    out = _workspace(repo).run_command("echo out; echo err >&2; exit 3")
    assert out == "exit_code: 3\n--- stdout ---\nout\n\n--- stderr ---\nerr\n"
    timed = _workspace(repo, command_timeout=1.0).run_command("sleep 5")
    assert "[TIMEOUT]" in timed


def test_eval_helpers_parse_real_script_shapes() -> None:
    script = ("git checkout c675 sympy/a/test_x.py sympy/b/test_y.py\ngit apply -v - <<'EOF'\nEOF\n"
              "git checkout c675 sympy/a/test_x.py sympy/b/test_y.py\n")
    assert eval_test_files(script) == ["sympy/a/test_x.py", "sympy/b/test_y.py"]
    body, failing = summarize_test_output(
        "noise\n+ : '>>>>> Start Test Output'\ntest_Vector E\ntest_ok ok\n"
        "FAIL: test_x (pkg.tests.T)\n+ : '>>>>> End Test Output'\ntrailer\n")
    assert failing == ["test_Vector", "test_x (pkg.tests.T)"]
    assert "noise" not in body and "trailer" not in body
