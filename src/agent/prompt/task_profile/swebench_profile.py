"""SWE-benchタスク用のTaskProfileを定義するモジュール.

SWE-benchの提出物は「gitパッチ(unified diff)」。リポジトリはDockerコンテナの中にあり、
エージェントはMCPツール(search_code / read_file / edit_file / run_tests / get_patch ...)で
外から操作する。パッチは手書きさせず、get_patch() の戻り値をそのまま提出させる
(課題 V.2: SWE-bench は final_answer(get_patch()) で提出する)。
"""
from src.agent.prompt.task_profile.task_profile import TaskProfile
from src.models.sandbox_base import SandboxBase

# moulinette の検証と同じ「パッチらしさ」の判定に使う目印
_PATCH_MARKERS = ("diff --git", "--- a/", "+++ b/", "@@")

# 打ち切り時に、編集済みの変更をパッチとして回収するためにサンドボックスで実行するコード。
# 出力はサイズ上限で切り詰められるので print ではなく final_answer で値ごと受け取る。
_COLLECT_PATCH_CODE = "final_answer(get_patch())"

_SWEBENCH_FORMAT_RULES = """- Start with a short Thought (at most 3 lines: what you learned, what you do next), then exactly
  ONE block fenced as ```python ... ```, then <end_code>.
- Tools are plain Python functions that RETURN strings: print() what you need to see, e.g.
  print(search_code(pattern="def foo", file_pattern="*.py")). Only printed output comes back.
- The tools and final_answer are already defined globals: call them directly, never import them.
- Print only the relevant part of large results (slice long strings, read files by line ranges).
- The namespace persists between turns: keep results in variables instead of re-running tools.
- Limits on iterations, tokens and time are strict: one focused step per turn."""

_SWEBENCH_FEW_SHOT_EXAMPLE = '''Thought: Find where the failing function is defined.
```python
print(search_function_or_class_definition_in_code(name="divide"))
```
<end_code>
Observation:
[OK] Executed successfully.
--- stdout ---
/testbed/calc/ops.py:10 def divide(a, b):
Thought: Read the function body.
```python
print(read_file(filepath="/testbed/calc/ops.py", start_line=8, end_line=14))
```
<end_code>
Observation:
[OK] Executed successfully.
--- stdout ---
10: def divide(a, b):
11:     return a * b
Thought: It multiplies instead of dividing. Fix it, run the evaluation tests, and submit the patch.
```python
print(edit_file(filepath="/testbed/calc/ops.py", old_str="    return a * b", new_str="    return a / b"))
print(run_tests()[-1500:])
patch = get_patch()
print(patch)
```
<end_code>
Observation:
[OK] Executed successfully.
--- stdout ---
Edited /testbed/calc/ops.py (replaced 1 occurrence starting at line 11).
exit_code: 0 ... all tests passed
diff --git a/calc/ops.py b/calc/ops.py
...
Thought: The tests pass and the patch is correct. Submit it.
```python
final_answer(patch)
```
<end_code>'''


class SWEBenchProfile(TaskProfile):
    """SWE-bench(実リポジトリのバグ修正)用のTaskProfileを定義するクラス.

    ファイルを読むためObservationを長めに残し、探索で提出しない反復が続くのは正常なので
    停滞によるモデル切り替えは行わない。累積入力の上限(300,000トークン)に収まるよう、
    再送する履歴は直近の数往復に絞る。
    """
    name = "swebench"

    history_window = 6
    observation_max_chars = 3000
    latest_observation_max_chars = 8000
    assistant_max_chars = 2000
    rotate_after_misses = None
    # [REPEATED] の指摘のあとも同じコードを出し直したら、別プロバイダのモデルに替える。実測(codestral,
    # sympy-14711): 失敗する同じ edit_file を22反復出し続け、空のパッチのまま上限に達した
    rotate_after_repeats = 2
    # 残り4反復で促す(テストの実行・パッチの確認・提出に数反復かかるため)
    iteration_warn_at = 4

    def submit_hint(self) -> str:
        """SWE-bench の提出の段取りを返す.

        Returns:
            str: 提出の促し。
        """
        return ("Stop exploring: if your fix is in place, run run_tests() once and submit with "
                "final_answer(get_patch()). If no fix is in place yet, make the most likely minimal edit now.")

    def format_rules(self) -> str:
        """SWE-bench用の出力形式のルールを返す.

        Returns:
            str: ## Format 節に入れるルール
        """
        return _SWEBENCH_FORMAT_RULES

    def task_instructions(self) -> str:
        """SWE-benchタスクの進め方を返す.

        Returns:
            str: タスクの指示
        """
        return (
            "SWE-bench: fix the GitHub issue given in the next message in the repository checked out "
            "at /testbed (inside a container you reach ONLY through the tools above).\n"
            "1. Locate: search_function_or_class_definition_in_code / search_code / list_files.\n"
            "2. Understand: read_file with start_line/end_line around the relevant code. Optionally "
            "reproduce the bug. Put the script in a triple-quoted string and pass it to python through "
            "a quoted heredoc, so that no quote needs escaping (python -c \"...\" with nested quotes "
            "breaks easily):\n"
            "print(run_command(command=\"\"\"cd /testbed && python - <<'EOF'\n"
            "from pkg.module import func\n"
            "print(func(0))\n"
            "EOF\"\"\"))\n"
            "3. Fix: edit_file(filepath=..., old_str=..., new_str=...) with a MINIMAL change in the "
            "library source. old_str must match the file exactly (copy indentation from read_file "
            "output, without the 'N: ' prefixes). Read the tool's reply: it reports syntax errors.\n"
            "4. Verify: run_tests() runs the official evaluation script (it adds the hidden tests). "
            "Read the failing test names and fix until they pass.\n"
            "5. Submit: patch = get_patch(); if patch: final_answer(patch).\n"
            "Rules: never edit or add test files; do not try to fetch fixes from the internet, pull "
            "requests or issues; base every change on the code you read in this repository."
        )

    def final_answer_specification(self) -> str:
        """final_answer()に何を渡すべきか、最終的な回答の仕方を返す.

        Returns:
            str: 最終的な回答の仕様
        """
        return (
            "Pass the git diff returned by get_patch(): final_answer(get_patch()). Never write a "
            "diff by hand, and do not submit before your edit is in place."
        )

    def few_shot_example(self) -> str:
        """出力形式を教えるためのFew-shotの例を返す.

        探索 -> 読解 -> 修正 -> 検証 -> get_patch() で提出、の流れを示す。実タスクとは無関係。

        Returns:
            str: Few-shotの例
        """
        return _SWEBENCH_FEW_SHOT_EXAMPLE

    def manual(self, sandbox: SandboxBase) -> str:
        """接続中MCPサーバーのツールスキーマから動的生成した、LLM向けツールマニュアルを返す.

        Args:
            sandbox (SandboxBase): サンドボックスのインスタンス

        Returns:
            str: LLM向けツールマニュアル
        """
        return sandbox.manual()

    def sanitize_submission(self, raw: str) -> str:
        """提出されたパッチを git apply できる形に整える.

        前後の空行とコードフェンスを除き、末尾の改行を保証する(末尾改行の無いパッチは
        git apply で「corrupt patch」になる)。

        Args:
            raw (str): final_answer()が捕捉した生の文字列。

        Returns:
            str: 整えたパッチ。空なら空文字。
        """
        text = raw.strip("\n")
        if text.lstrip().startswith("```"):
            lines = text.strip().splitlines()[1:]
            while lines and lines[-1].strip().startswith("```"):
                lines.pop()
            text = "\n".join(lines)
        if not text.strip():
            return ""
        return text if text.endswith("\n") else text + "\n"

    def submission_rejection(self, candidate: str) -> str | None:
        """提出物がgitパッチとして受理できるかを検査する.

        moulinette と同じく、diffの目印(diff --git / --- a/ / +++ b/ / @@)を要求する。

        Args:
            candidate (str): 整えた提出物。

        Returns:
            str | None: 受理できるならNone。だめなら理由。
        """
        if not candidate.strip():
            return (
                "final_answer() received an empty patch: get_patch() returns nothing until you "
                "edit a file with edit_file"
            )
        if not any(marker in candidate for marker in _PATCH_MARKERS):
            return "the submission is not a git diff: pass the value returned by get_patch()"
        return None

    def fallback_solution(self, sandbox: SandboxBase) -> str:
        """打ち切り時に、コンテナ内の編集済みの変更をパッチとして回収する.

        修正が途中でも、空の提出より採点される可能性がある。get_patch() の値は
        final_answer() 経由で受け取る(stdout はサイズ上限で切り詰められるため)。

        Args:
            sandbox (SandboxBase): 接続中のサンドボックス。

        Returns:
            str: 回収したパッチ。変更が無いか回収できなければ空文字。
        """
        result = sandbox.run(_COLLECT_PATCH_CODE)
        if not result.final_answer_bool or not result.final_answer:
            return ""
        patch = self.sanitize_submission(result.final_answer)
        return patch if self.submission_rejection(patch) is None else ""
