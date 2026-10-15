"""SWE-bench の必須ツール(課題 V.5)の中身を実装するモジュール.

MCPサーバー(mcp_tools_swebench.py)はここの SweWorkspace のメソッドを呼ぶだけの薄い層にする。
ツールの中身をMCPから切り離しておくと、単体テストで直接呼べる。

- 返り値はすべてLLMがそのまま読む英語の文字列。失敗も例外ではなく "Error: ..." で返す
  (LLMが「何が起きたか」を推測しなくて済むように。課題 V.1)。
- 長い出力は切り詰め、切り詰めたことを明示する(課題 V.1)。
- ファイルの読み書きは、コンテナ内のPythonで補助スクリプトを実行して行う。データは
  標準入力のJSONで渡し、シェルの引用符の問題を避ける。補助スクリプトは古いPython
  (SWE-benchの環境には3.5〜3.6もある)でも動くよう、f-stringを使わずに書く。
"""
import json
import os
import re
import shlex
import time
from collections.abc import Callable

from src.mcp_core.swe_bench.executor import CommandResult, Executor

# LLMへ返す出力の上限文字数
OUTPUT_LIMIT = 6000
# 一覧系ツールの上限件数
LIST_LIMIT = 200
MATCH_LIMIT = 100
# 1行の上限文字数(圧縮されたJS等の巨大な1行を避ける)
LINE_LIMIT = 300
# 範囲指定なしの read_file で返す最大行数
READ_DEFAULT_LINES = 400
# 締め切りの何秒前でコマンドを打ち切るか(後片付けとパッチ回収の時間を残す)
DEADLINE_RESERVE_SEC = 5.0
# 評価スクリプトをコンテナ内に置く場所
EVAL_SCRIPT_TARGET = "/tmp/agent_eval.sh"
# get_patch に含めない未追跡ファイル(ビルド生成物・パッチの残骸)
_JUNK_PATTERN = re.compile(r"(\.pyc$|__pycache__/|\.orig$|\.rej$|\.egg-info/|^build/|^dist/)")
# 評価スクリプト中の「テストファイルを基準コミットへ戻す」行
_CHECKOUT_PATTERN = re.compile(r"^git checkout ([0-9a-f]{4,40}) (.+)$", re.MULTILINE)
# 識別子(関数名・クラス名)。Class.method 形式も受け付けて最後の部分で探す
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
# grep の出力 "path:line:content"
_GREP_LINE = re.compile(r"^(.*?):(\d+):(.*)$")
# 失敗したテスト名の拾い方(pytest -rA / unittest・Django / sympy の bin/test)
_FAILING_PATTERNS = (
    re.compile(r"^(?:FAILED|ERROR) (\S+)"),
    re.compile(r"^(?:FAIL|ERROR): (\S+ \([^)]*\))"),
    re.compile(r"^(test_\w+) +[EF]\s*$"),
)


def _path_glob(file_pattern: str) -> str:
    """パスや素の名前の file_pattern を find -path 用の glob にする.

    find -path の `*` は `/` もまたぐので、`**` は `*` と同じ意味に畳む。相対パスは先頭に `*/` を付け、
    リポジトリのルートからのパスでも途中からの部分パス("physics/vector/vector.py")や素の名前("ops.py")でも
    合うようにする。

    Args:
        file_pattern (str): パスの glob(例: "sympy/core/*.py"、"/testbed/a.py"、"**/*.py"、"calc")。

    Returns:
        str: find -path に渡す glob。
    """
    glob = file_pattern.rstrip("/") or "/"
    if not glob.startswith("/"):
        glob = "*/" + (glob[2:] if glob.startswith("./") else glob)
    return glob.replace("**/", "*").replace("**", "*")


_READ_SCRIPT = r'''
import io, json, os, sys
a = json.loads(sys.stdin.read())
path, start, end, limit = a["path"], a["start"], a["end"], a["limit"]
if os.path.isdir(path):
    print("Error: {} is a directory; use list_files to see its contents.".format(path)); sys.exit(1)
if not os.path.isfile(path):
    print("Error: file not found: {}".format(path)); sys.exit(1)
with io.open(path, encoding="utf-8", errors="replace") as f:
    lines = f.read().splitlines()
total = len(lines)
first = 1 if start is None else start
last = total if end is None else end
if first < 1 or last < first:
    print("Error: invalid range start_line={}, end_line={} (the file has {} lines).".format(start, end, total))
    sys.exit(1)
if total == 0:
    print("(the file is empty)"); sys.exit(0)
if first > total:
    print("Error: start_line {} is past the end of the file ({} lines).".format(first, total)); sys.exit(1)
last = min(last, total)
note = ""
if start is None and end is None and total > limit:
    last = limit
    note = "[showing lines 1-{} of {}; pass start_line/end_line to read the rest]".format(limit, total)
for number in range(first, last + 1):
    print("{}: {}".format(number, lines[number - 1]))
if note:
    print(note)
'''

_EDIT_SCRIPT = r'''
import io, json, os, sys
a = json.loads(sys.stdin.read())
path, old, new = a["path"], a["old"], a["new"]
if not old:
    print("Error: old_str must not be empty."); sys.exit(1)
if not os.path.isfile(path):
    print("Error: file not found: {}".format(path)); sys.exit(1)
with io.open(path, encoding="utf-8", newline="") as f:
    content = f.read()
count = content.count(old)
if count == 0:
    print("Error: old_str was not found in {}. It must match the file exactly, including indentation and "
          "whitespace. Copy it from read_file output without the 'N: ' line-number prefixes.".format(path))
    sys.exit(1)
if count > 1:
    print("Error: old_str was found {} times in {}. Include more surrounding lines so it matches exactly "
          "once.".format(count, path))
    sys.exit(1)
start_line = content.count("\n", 0, content.index(old)) + 1
updated = content.replace(old, new, 1)
if path.endswith(".py"):
    try:
        compile(updated, path, "exec")
    except SyntaxError as error:
        print("SYNTAX ERROR: this edit would make {} invalid Python: {} (line {}). The edit was reverted; "
              "the file is unchanged.".format(path, error.msg, error.lineno))
        if error.text:
            print("    " + error.text.rstrip())
        sys.exit(1)
with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(updated)
print("Edited {} (replaced 1 occurrence starting at line {}).".format(path, start_line))
if path.endswith(".py"):
    try:
        import ast
        from pyflakes import checker
        def messages(source):
            return set(m.message % m.message_args for m in checker.Checker(ast.parse(source), path).messages)
        for text in sorted(messages(updated) - messages(content))[:5]:
            print("LINT: " + text)
    except Exception:
        pass
lines = updated.splitlines()
last = min(len(lines), start_line + new.count("\n") + 2)
for number in range(max(1, start_line - 2), last + 1):
    print("{}: {}".format(number, lines[number - 1]))
'''


def clip(text: str, limit: int = OUTPUT_LIMIT, tail_ratio: float = 0.34) -> str:
    """長い文字列の中央を省き、省いたことを明示して返す.

    Args:
        text (str): 対象の文字列。
        limit (int): 残す最大文字数。
        tail_ratio (float): 残す分のうち末尾に割り当てる割合(テスト結果のように末尾が大事なら大きく)。

    Returns:
        str: 切り詰めた文字列。
    """
    if len(text) <= limit:
        return text
    tail = int(limit * tail_ratio)
    head = limit - tail
    omitted = len(text) - limit
    return (f"{text[:head]}\n[output truncated: {omitted} characters omitted. Narrow the command, "
            f"pattern or line range to see less at once.]\n{text[-tail:] if tail else ''}")


def eval_test_files(eval_script: str) -> list[str]:
    """評価スクリプトがテスト前後で基準コミットへ戻すテストファイルの一覧を返す.

    Args:
        eval_script (str): 評価スクリプトの本文。

    Returns:
        list[str]: リポジトリからの相対パス(重複なし・出現順)。
    """
    files: list[str] = []
    for match in _CHECKOUT_PATTERN.finditer(eval_script):
        for path in shlex.split(match.group(2)):
            if path not in files:
                files.append(path)
    return files


def summarize_test_output(output: str) -> tuple[str, list[str]]:
    """評価スクリプトの出力から、テスト部分と失敗したテスト名を取り出す.

    `>>>>> Start Test Output` と `>>>>> End Test Output` の間だけを残し、
    大量に出る DeprecationWarning(とその次の行のソース表示)は落とす。

    Args:
        output (str): 評価スクリプトの標準出力+標準エラー。

    Returns:
        tuple[str, list[str]]: (整えたテスト出力, 失敗したテスト名の一覧)。
    """
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if ">>>>> Start Test Output" in line), None)
    end = next((i for i, line in enumerate(lines) if ">>>>> End Test Output" in line), None)
    if start is not None:
        lines = lines[start + 1:end] if end is not None and end > start else lines[start + 1:]
    kept: list[str] = []
    skip_next = False
    for line in lines:
        if skip_next and line.startswith((" ", "\t")):
            skip_next = False
            continue
        skip_next = False
        if "DeprecationWarning" in line or "PendingDeprecationWarning" in line:
            skip_next = True
            continue
        kept.append(line)
    failing: list[str] = []
    for line in kept:
        for pattern in _FAILING_PATTERNS:
            found = pattern.match(line.strip())
            if found and found.group(1) not in failing:
                failing.append(found.group(1))
    return "\n".join(kept), failing


class SweWorkspace:
    """SWE-bench のリポジトリ1つ分の作業場(必須ツールの実装).

    Attributes:
        executor (Executor): コマンドの実行場所。
        testbed (str): リポジトリのルート(TESTBED_PATH)。
        eval_script_file (str | None): ホスト上の評価スクリプト(run_tests 用)。
        command_timeout (float): 1コマンドの既定の制限時間。
        test_timeout (float): run_tests の既定の制限時間。
        deadline_epoch (float | None): タスク全体の締め切り(UNIX時刻)。各コマンドの制限時間をこれに合わせて縮める。
    """

    def __init__(
        self,
        executor: Executor,
        testbed: str = "/testbed",
        eval_script_file: str | None = None,
        command_timeout: float = 120.0,
        test_timeout: float = 300.0,
        deadline_epoch: float | None = None,
        clock: Callable[[], float] = time.time,
        summarize_tests: bool = True,
    ) -> None:
        """設定を保持する.

        Args:
            executor (Executor): コマンドの実行場所。
            testbed (str): リポジトリのルート。
            eval_script_file (str | None): ホスト上の評価スクリプト。
            command_timeout (float): 1コマンドの既定の制限時間(秒)。
            test_timeout (float): run_tests の既定の制限時間(秒)。
            deadline_epoch (float | None): タスク全体の締め切り(UNIX時刻)。
            clock (Callable[[], float]): 現在時刻を返す関数(テスト用に差し替え可能)。
            summarize_tests (bool): run_tests の出力を要約するか(False なら生の出力の末尾を返す。
                BENCHMARK_REPORT の ablation 用)。
        """
        self.executor = executor
        self.testbed = testbed.rstrip("/") or "/"
        self.eval_script_file = eval_script_file
        self.command_timeout = command_timeout
        self.test_timeout = test_timeout
        self.deadline_epoch = deadline_epoch
        self._clock = clock
        self.summarize_tests = summarize_tests

    # ---- 共通 -----------------------------------------------------------------
    def _budget(self, wanted: float) -> float | None:
        """締め切りを越えないよう、このコマンドに使ってよい秒数を返す。もう無ければNone.

        Args:
            wanted (float): 既定の制限時間。

        Returns:
            float | None: 使ってよい秒数。
        """
        if self.deadline_epoch is None:
            return wanted
        remaining = self.deadline_epoch - self._clock() - DEADLINE_RESERVE_SEC
        if remaining < 1.0:
            return None
        return min(wanted, remaining)

    def _path(self, path: str) -> str:
        """相対パスをリポジトリのルート基準の絶対パスにする.

        Args:
            path (str): ツールに渡されたパス。

        Returns:
            str: 絶対パス。
        """
        return path if path.startswith("/") else os.path.normpath(os.path.join(self.testbed, path))

    def _run(self, command: str, workdir: str | None = None, timeout: float | None = None,
             stdin: str | None = None) -> CommandResult | str:
        """締め切りを考慮してコマンドを実行する。時間が残っていなければエラー文字列を返す.

        Args:
            command (str): bashに渡すコマンド。
            workdir (str | None): 作業ディレクトリ(既定はリポジトリのルート)。
            timeout (float | None): 既定の制限時間(既定は command_timeout)。
            stdin (str | None): 標準入力に渡す文字列。

        Returns:
            CommandResult | str: 実行結果。時間切れなら "Error: time budget exhausted ..."。
        """
        budget = self._budget(self.command_timeout if timeout is None else timeout)
        if budget is None:
            return "Error: time budget exhausted for this task; no more commands can run. Submit your patch now."
        return self.executor.run(command, workdir or self.testbed, budget, stdin)

    def _python(self, script: str, payload: dict[str, object]) -> str:
        """補助スクリプトを実行場所のPythonで動かし、その出力を返す.

        Args:
            script (str): 実行するPythonコード。
            payload (dict[str, object]): 標準入力にJSONで渡すデータ。

        Returns:
            str: スクリプトの出力(失敗時はエラーの説明)。
        """
        result = self._run(f"{shlex.quote(self.executor.python)} -c {shlex.quote(script)}",
                           stdin=json.dumps(payload))
        if isinstance(result, str):
            return result
        if result.timed_out:
            return "Error: the operation timed out."
        text = result.stdout.rstrip("\n")
        if result.exit_code not in (0, 1) and not text:
            return f"Error: {result.stderr.strip()[-500:] or f'exit code {result.exit_code}'}"
        return text

    @staticmethod
    def _grep_lines(output: str) -> list[str]:
        """grep の "path:line:content" を課題の形式 "path:line content" に直す.

        Args:
            output (str): grep -n の出力。

        Returns:
            list[str]: 整形した行。
        """
        formatted: list[str] = []
        for raw in output.splitlines():
            match = _GREP_LINE.match(raw)
            if match is None:
                continue
            path, number, content = match.groups()
            if len(content) > LINE_LIMIT:
                content = content[:LINE_LIMIT] + "..."
            formatted.append(f"{path}:{number} {content}")
        return formatted

    def _limited(self, lines: list[str], what: str) -> str:
        """一覧を上限件数で切り、省いた件数を明示する.

        Args:
            lines (list[str]): 一覧。
            what (str): 何の一覧か(注記に使う)。

        Returns:
            str: 整形した一覧。
        """
        shown = lines[:MATCH_LIMIT]
        text = "\n".join(shown)
        if len(lines) > MATCH_LIMIT:
            text += f"\n[{len(lines) - MATCH_LIMIT} more {what} not shown; narrow the search]"
        return clip(text)

    # ---- File System Tools ----------------------------------------------------
    def read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """ファイルを行番号付き(`<行番号>: <内容>`)で読む.

        Args:
            filepath (str): 読むファイル。
            start_line (int | None): 最初の行(1始まり・含む)。
            end_line (int | None): 最後の行(含む)。

        Returns:
            str: 行番号付きの内容、またはエラー。
        """
        return clip(self._python(_READ_SCRIPT, {
            "path": self._path(filepath), "start": start_line, "end": end_line, "limit": READ_DEFAULT_LINES,
        }))

    def edit_file(self, filepath: str, old_str: str, new_str: str) -> str:
        """ファイル中のちょうど1か所の文字列を置き換える。Pythonの構文を壊す編集は元に戻す.

        Args:
            filepath (str): 編集するファイル。
            old_str (str): 置き換える文字列(ファイル中にちょうど1回現れること)。
            new_str (str): 置き換え後の文字列。

        Returns:
            str: 結果(置き換えた行・前後の行・構文エラーやlintの警告)。
        """
        return clip(self._python(_EDIT_SCRIPT, {"path": self._path(filepath), "old": old_str, "new": new_str}))

    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """ディレクトリ以下でパターンに合うファイルを絶対パスで列挙する(.git は除く).

        Args:
            directory (str): 探すディレクトリ。
            pattern (str): ファイル名のglob(`/` を含めばパス全体に対するglob)。

        Returns:
            str: 1行1ファイルの一覧、またはエラー。
        """
        root = self._path(directory)
        test = "-path" if "/" in pattern else "-name"
        command = (f"if [ ! -d {shlex.quote(root)} ]; then echo 'Error: directory not found: {root}'; exit 1; fi; "
                   f"find {shlex.quote(root)} -name .git -prune -o -type f {test} {shlex.quote(pattern)} -print")
        result = self._run(command)
        if isinstance(result, str):
            return result
        if result.exit_code != 0:
            return (result.stdout.strip() or result.stderr.strip())[:500]
        files = sorted(line for line in result.stdout.splitlines() if line)
        if not files:
            return f"No files matching {pattern!r} under {root}."
        text = "\n".join(files[:LIST_LIMIT])
        if len(files) > LIST_LIMIT:
            text += f"\n[{len(files) - LIST_LIMIT} more files not shown; use a narrower directory or pattern]"
        return clip(text)

    # ---- Code Search Tools ----------------------------------------------------
    def _grep(self, arguments: str, what: str) -> str:
        """リポジトリ全体に grep をかけ、課題の形式で返す.

        Args:
            arguments (str): grep のオプションとパターン(引用済み)。
            what (str): 該当なしのときの説明に使う語。

        Returns:
            str: `/abs/path:<行番号> <内容>` の一覧、または該当なし / エラーの説明。
        """
        result = self._run(f"grep -rnI --exclude-dir=.git {arguments} {shlex.quote(self.testbed)}")
        if isinstance(result, str):
            return result
        if result.exit_code == 1:
            return f"No matches found for {what}."
        if result.exit_code != 0 and not result.stdout:
            return f"Error: {result.stderr.strip()[:500]}"
        return self._limited(self._grep_lines(result.stdout), "matches")

    def search_code(self, pattern: str, file_pattern: str = "*") -> str:
        """正規表現(grep -E)でコードを検索する.

        file_pattern がワイルドカード入りのファイル名(例: "*.py")なら、その名前のファイルをリポジトリ全体から探す。
        それ以外(パスや素の名前)は、その名前のファイルか、その名前のディレクトリの中を探す。パスは絶対パス・
        リポジトリからの相対パス・途中からの部分パスのどれでもよい。実測では多くのモデルが
        "sympy/physics/vector/*.py" や "/testbed/a/b.py"、"**/*.py" のようにパスで絞り込み、ファイル名しか
        見ない grep --include では毎回0件になっていた(gpt-oss-20b は sympy-14711 で11回空振りし、
        修正箇所に辿り着けないまま上限に達した)。

        Args:
            pattern (str): 検索する正規表現。
            file_pattern (str): 対象ファイルの glob(ファイル名またはパス)。

        Returns:
            str: `/abs/path:<行番号> <内容>` の一覧。
        """
        file_pattern = (file_pattern or "*").strip()
        what = f"pattern {pattern!r} in {file_pattern}"
        if "/" not in file_pattern and any(char in file_pattern for char in "*?["):
            return self._grep(f"-E --include={shlex.quote(file_pattern)} -e {shlex.quote(pattern)}", what)
        path_glob = _path_glob(file_pattern)
        # 対象ファイルが1つも無いときは終了コード3で知らせる(「ファイルはあるが一致なし」と区別する)
        command = (f"files=$(find {shlex.quote(self.testbed)} -name .git -prune -o -type f "
                   f"\\( -path {shlex.quote(path_glob)} -o -path {shlex.quote(path_glob + '/*')} \\) -print); "
                   "if [ -z \"$files\" ]; then exit 3; fi; "
                   f"printf '%s\\n' \"$files\" | xargs -d '\\n' grep -nIH -E -e {shlex.quote(pattern)}")
        result = self._run(command)
        if isinstance(result, str):
            return result
        if result.exit_code == 3 and not result.stdout:
            return (f"No files match {file_pattern!r}. Check the path with list_files, "
                    "or pass a file name glob such as '*.py'.")
        if not result.stdout:
            # xargs は一致なしでも grep のエラーでも 123 を返すので、標準エラーの有無で見分ける
            error = result.stderr.strip()
            return f"Error: {error[:500]}" if error else f"No matches found for {what}."
        return self._limited(self._grep_lines(result.stdout), "matches")

    def search_function_or_class_definition_in_code(self, name: str) -> str:
        """関数・クラスの定義(def / async def / class)を .py ファイルから探す.

        Args:
            name (str): 関数名・クラス名(`Class.method` なら method を探す)。

        Returns:
            str: search_code と同じ形式の一覧。
        """
        if not _NAME_PATTERN.match(name):
            return f"Error: {name!r} is not a valid Python identifier."
        target = name.split(".")[-1]
        regex = rf"^[[:space:]]*(async[[:space:]]+)?(def|class)[[:space:]]+{target}[[:space:]]*[(:]"
        return self._grep(f"-E --include='*.py' -e {shlex.quote(regex)}", f"a definition of {target!r}")

    def find_references(self, name: str, filepath: str | None = None, line: int | None = None) -> str:
        """シンボルが単語として現れる箇所(定義を含む)を .py ファイルから探す.

        Args:
            name (str): 関数名・クラス名(`Class.method` なら method を探す)。
            filepath (str | None): 定義のあるファイル(指定するとそのファイルの結果を先頭に並べる)。
            line (int | None): 定義の行(参考情報。結果の絞り込みには使わない)。

        Returns:
            str: search_code と同じ形式の一覧。
        """
        if not _NAME_PATTERN.match(name):
            return f"Error: {name!r} is not a valid Python identifier."
        target = name.split(".")[-1]
        result = self._grep(f"-w -F --include='*.py' -e {shlex.quote(target)}", f"references to {target!r}")
        if filepath is None or result.startswith(("No matches", "Error")):
            return result
        own = self._path(filepath)
        lines = result.splitlines()
        first = [entry for entry in lines if entry.startswith(own + ":")]
        return "\n".join(first + [entry for entry in lines if entry not in first])

    # ---- Execution Tools ------------------------------------------------------
    def _test_files(self) -> list[str]:
        """評価スクリプトが触るテストファイル(get_patch から除くもの)を返す.

        Returns:
            list[str]: リポジトリからの相対パス。
        """
        if not self.eval_script_file or not os.path.isfile(self.eval_script_file):
            return []
        with open(self.eval_script_file, encoding="utf-8") as handle:
            return eval_test_files(handle.read())

    def run_tests(self) -> str:
        """評価スクリプトを実行し、テスト結果を要約して返す。実行後はテストファイルを元に戻す.

        Returns:
            str: 終了コード・所要時間・失敗したテスト名・テスト出力(切り詰めあり)。
        """
        if not self.eval_script_file or not os.path.isfile(self.eval_script_file):
            return "Error: no evaluation script is configured for this task (SWE_EVAL_SCRIPT_FILE is not set)."
        target = EVAL_SCRIPT_TARGET if getattr(self.executor, "name", None) else self.eval_script_file
        try:
            self.executor.copy_in(self.eval_script_file, target)
        except (OSError, RuntimeError) as error:
            return f"Error: could not prepare the evaluation script: {error}"
        started = time.monotonic()
        result = self._run(f"bash {shlex.quote(target)} 2>&1", timeout=self.test_timeout)
        elapsed = time.monotonic() - started
        restored = self._restore_test_files()
        if isinstance(result, str):
            return result
        if not self.summarize_tests:
            # ablation 用: 要約せず、評価スクリプトの生の出力の末尾をそのまま返す
            return f"exit_code: {result.exit_code}\n" + clip(result.stdout + result.stderr, OUTPUT_LIMIT, tail_ratio=0.7)
        body, failing = summarize_test_output(result.stdout + result.stderr)
        header = [f"exit_code: {result.exit_code}  elapsed: {elapsed:.0f}s"]
        if result.timed_out:
            header.append("[TIMEOUT] the evaluation did not finish in time; the output below is partial.")
        if failing:
            header.append(f"Failing tests ({len(failing)}): " + ", ".join(failing[:20]))
        if restored:
            header.append(restored)
        header.append("--- test output (between the Start/End markers; DeprecationWarnings removed) ---")
        return "\n".join(header) + "\n" + clip(body, OUTPUT_LIMIT - 600, tail_ratio=0.7)

    def _restore_test_files(self) -> str:
        """評価スクリプトが当てたテストの変更を取り消す(get_patch に混ざらないように).

        評価スクリプトは最後に `git checkout <base> <テストファイル>` で元に戻すが、時間切れで
        途中終了すると変更が残る。同じ行をもう一度実行して確実に戻す。

        Returns:
            str: 戻せなかった場合の注記。問題なければ空文字。
        """
        if not self.eval_script_file:
            return ""
        with open(self.eval_script_file, encoding="utf-8") as handle:
            checkouts = _CHECKOUT_PATTERN.findall(handle.read())
        if not checkouts:
            return ""
        commit, files = checkouts[-1]
        result = self._run(f"git checkout {shlex.quote(commit)} {files}", timeout=60)
        if isinstance(result, str) or result.exit_code != 0:
            return "[warning] could not restore the test files touched by the evaluation script."
        return ""

    def get_patch(self) -> str:
        """リポジトリの変更を unified diff で返す(`git -c core.fileMode=false diff`).

        新しく作ったソースファイルも含め(intent-to-add)、評価スクリプトが触るテストファイルと
        ビルド生成物は除く。

        Returns:
            str: パッチ本文。変更が無ければ空文字。失敗時は "Error: ..."。
        """
        excluded = self._test_files()
        untracked = self._run("git ls-files --others --exclude-standard", timeout=60)
        if isinstance(untracked, str):
            return untracked
        new_files = [path for path in untracked.stdout.splitlines()
                     if path and not _JUNK_PATTERN.search(path) and path not in excluded]
        if new_files:
            self._run("git add -N -- " + " ".join(shlex.quote(path) for path in new_files), timeout=60)
        excludes = " ".join(shlex.quote(f":(exclude){path}") for path in excluded)
        result = self._run(f"git -c core.fileMode=false diff -- . {excludes}", timeout=60)
        if isinstance(result, str):
            return result
        if result.exit_code != 0:
            return f"Error: git diff failed: {result.stderr.strip()[:500]}"
        return result.stdout

    def run_command(self, command: str, workdir: str | None = None) -> str:
        """シェルコマンドを実行し、終了コード・標準出力・標準エラー出力を返す.

        Args:
            command (str): 実行するコマンド。
            workdir (str | None): 作業ディレクトリ(既定はリポジトリのルート)。

        Returns:
            str: `exit_code: N` / `--- stdout ---` / `--- stderr ---` の3部構成の文字列。
        """
        result = self._run(command, self._path(workdir) if workdir else None)
        if isinstance(result, str):
            return result
        note = f"\n[TIMEOUT] the command was stopped after {self.command_timeout:.0f}s; output is partial." \
            if result.timed_out else ""
        half = OUTPUT_LIMIT // 2
        return (f"exit_code: {result.exit_code}{note}\n--- stdout ---\n{clip(result.stdout, half)}\n"
                f"--- stderr ---\n{clip(result.stderr, half)}")
