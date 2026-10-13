"""MBPPタスク用のTaskProfileを定義するモジュール."""
from src.agent.prompt.task_profile.task_profile import TaskProfile
from src.models.sandbox_base import SandboxBase

# 1ステップで「実装 -> run_testsで検証 -> 合格なら即submit」まで到す手本。
# 反復数 / 累積トークンを最小化するため、検証と提出を同じコードブロックで行う形を示す。
_MBPP_FEW_SHOT_EXAMPLE = '''Thought: Implement the function, verify it with the given tests, submit if they pass.
```python
solution = """def add(a, b):
    return a + b
"""
import json
report = json.loads(run_tests(code=solution, test_list=["assert add(2, 3) == 5"]))
print(report)
if report["success"]:
    final_answer(solution)
```
<end_code>
Observation:
[OK] Executed successfully.
--- stdout ---
{'success': True, 'message': 'passed', 'output': ''}
'''


class MBPPProfile(TaskProfile):
    """MBPP(単一の関数の実装課題)用のTaskProfileを定義するクラス."""
    name = "mbpp"

    def task_instructions(self) -> str:
        """MBPPタスクの進め方を返す.

        Returns:
            str: タスクの指示
        """
        return (
            "This is an MBPP task: implement ONE Python function with the exact name and signature "
            "given by the task's function signature.\n"
            "- Build the full source as a string in a variable (triple-quoted), then verify it with "
            "run_tests(code=..., test_list=[...]) using the asserts from the task message.\n"
            "- As soon as run_tests reports success, call final_answer(<that same string>) in the "
            "SAME code block. Do not spend another turn.\n"
            "- The submitted string must contain ONLY imports plus the function (and any helper "
            "functions it needs). No prints, no asserts, no example calls: the grader appends its "
            "own test lines to your code and runs the file, so any extra statement that raises "
            "makes the whole submission fail.\n"
            "- HIDDEN TESTS: you are shown only part of the test suite. Write a general, correct "
            "implementation that handles edge cases; never hardcode the shown expected values.\n"
            "- If run_tests fails, read its output, fix the implementation and try again. "
            "Keep each turn small: iterations, tokens and time are strictly limited."
        )

    def final_answer_specification(self) -> str:
        """final_answer()に何を渡すべきか、最終的な回答の仕方を返す.

        Returns:
            str: 最終的な回答の仕様
        """
        return (
            "Pass the function's full source code as a string, e.g. final_answer(solution) where "
            'solution = """def solve(n):\n    return n * 2\n""". '
            "Never pass a computed value, a function object, or markdown fences."
        )

    def few_shot_example(self) -> str:
        """出力形式を教えるためのFew-shotの例を返す.

        Thought -> Code -> Observation -> final_answer
        実タスクとは無関係な例。形式を示すためだけの固定のテキスト。

        Returns:
            str: Few-shotの例
        """
        return _MBPP_FEW_SHOT_EXAMPLE

    def manual(self, sandbox: SandboxBase) -> str:
        """接続中MCPサーバーのツールスキーマから動的生成した、LLM向けツールマニュアルを返す.

        このタスクで使えるツール / 環境の説明を返す。
        ツールはハードコードせずに、sandbox.manual()から動的に取得する。

        Args:
            sandbox (SandboxBase): サンドボックスのインスタンス

        Returns:
            str: LLM向けツールマニュアル
        """
        return sandbox.manual()
