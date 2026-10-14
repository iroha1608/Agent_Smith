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
report = json.loads(run_tests(code=solution, test_list=['assert add(2, 3) == 5']))
print(report)
if report['success']:
    final_answer(solution)
```
<end_code>
Observation:
[OK] Executed successfully.
--- stdout ---
{'success': True, 'message': 'passed', 'output': '1/1 tests passed.'}
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
            "MBPP: implement ONE function with the exact name and signature given by the task.\n"
            "- Build its source as a triple-quoted string, verify it with "
            "run_tests(code=..., test_list=[...]) using the task's asserts, and when it passes call "
            "final_answer(<that string>) in the SAME block.\n"
            "- The submitted string must contain ONLY imports and the function(s). No prints, "
            "asserts or demo calls: the grader appends its own tests to your code and runs it, so "
            "any extra statement that raises fails the submission.\n"
            "- HIDDEN TESTS: you see only part of the suite. Write a general, correct "
            "implementation; never hardcode the shown values. Infer the intended rule from ALL the "
            "given examples (they often encode a convention the wording leaves out).\n"
            "- Gate final_answer on the task's own asserts only. Do not add asserts you invented: "
            "if they are wrong they block a correct submission. To explore edge cases, print values.\n"
            "- If run_tests fails, its output names each failing assert and what your function "
            "actually returned. Compare that with the expected value, fix the rule, retry.\n"
            "- When the expected outputs follow a numeric or bit pattern you cannot see at once, "
            "compute it in code from the given examples (differences, ratios, binary forms) and "
            "print it, instead of guessing a formula."
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
