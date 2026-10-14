"""MBPPタスク用のTaskProfileを定義するモジュール.

MBPPの提出物は「関数のソースコード」。採点側は提出コードに隠しテストを連結して
ファイル全体を実行するため、その形で例外を投げないコードだけを受理する。
"""
import ast

from src.agent.prompt.task_profile.task_profile import TaskProfile
from src.models.sandbox_base import SandboxBase

# 採点環境に存在しないサンドボックス専用の名前。これらを参照するコードを提出すると
# NameErrorで全テストが落ちる。実測(MBPPタスク160)では、打ち切り時のフォールバックが
# `report = json.loads(run_tests(code="""..."""))` という検証用ラッパーをそのまま提出していた。
_SANDBOX_ONLY_NAMES = frozenset({"run_tests", "final_answer"})
# トップレベルに置いても採点時に例外を投げない文。採点側は提出コードにテスト行を
# 連結してファイル全体を実行するため、これ以外の文(デモ呼び出し/assert/print等)は
# 1つでも例外を投げると提出全体が不合格になる。
_ALLOWED_TOP_LEVEL = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.Assign, ast.AnnAssign,
)


def strip_code_fences(raw: str) -> str:
    """final_answer()に渡された文字列から前後の空白とmarkdownのコードフェンスを除く.

    LLMはしばしばmarkdownのコードフェンス付きで提出する。採点側は提出文字列に
    テスト行を連結して実行するため、フェンスが残っていると構文エラーになる。

    Args:
        raw (str): final_answer()が捕捉した生の文字列。

    Returns:
        str: 前後の空白とコードフェンスを除去した文字列。
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # ```python 等の開始フェンスを落とす
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    return text


def mbpp_rejection(solution: str) -> str | None:
    """MBPPの提出コードを受理してよいか検査し、だめな理由を返す.

    採点側は提出コードに隠しテストを連結してファイル全体を実行し、exit 0のみを合格とする。
    そのため「構文が通る」だけでは足りず、(1)関数定義がある (2)トップレベルに例外を
    投げ得る文が無い (3)サンドボックス専用の名前を参照していない の3点を確かめる。

    Args:
        solution (str): サニタイズ済みの提出コード。

    Returns:
        str | None: 受理可能ならNone。不可なら理由の文字列。
    """
    if not solution:
        return "final_answer() received an empty value"
    try:
        tree = ast.parse(solution)
    except SyntaxError as error:
        return f"the submitted code is not valid Python ({error.msg} on line {error.lineno})"

    # サンドボックス専用名の検査を先に行う。検証ラッパーをそのまま提出した場合、
    # 関数定義が文字列リテラルの中にあるため「関数定義なし」が先に当たってしまい、
    # 本当の原因(ラッパーごと提出した)が伝わらない。
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    blocked = sorted(referenced & _SANDBOX_ONLY_NAMES)
    if blocked:
        return (
            f"the submitted code calls {', '.join(blocked)}, which exist only inside this "
            "sandbox. Submit just the function source, with no test harness around it"
        )

    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
        return "the submitted code contains no function definition"

    for node in tree.body:
        if isinstance(node, _ALLOWED_TOP_LEVEL):
            continue
        # モジュールdocstringだけは例外を投げないので許す
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        return (
            f"the submitted code has a top-level {type(node).__name__} statement. Submit only "
            "imports and the function definition(s): the grader appends its own tests to your "
            "code and runs the whole file, so any extra statement that raises fails it"
        )
    return None


# 「答えを直書きした」と判定するときに無視する、ありふれた戻り値。
# None / True / 0 / 空文字 などは正しい実装でも普通に return するので、一致しても証拠にならない。
_TRIVIAL_LITERALS: tuple[object, ...] = (None, True, False, 0, 1, -1, "", (), [], {})


def _literal_value(node: ast.AST) -> tuple[bool, object]:
    """ASTノードがリテラルなら (True, 値)、そうでなければ (False, None) を返す.

    Args:
        node (ast.AST): 評価するノード。

    Returns:
        tuple[bool, object]: (リテラルだったか, その値)。
    """
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _is_trivial(value: object) -> bool:
    """ありふれた戻り値(None, 0, True, 空文字など)かどうかを返す.

    Args:
        value (object): 判定する値。

    Returns:
        bool: ありふれた値ならTrue。
    """
    return any(type(value) is type(trivial) and value == trivial for trivial in _TRIVIAL_LITERALS)


def hardcoded_answers(solution: str, test_list: list[str]) -> list[str]:
    """提出コードが、可視テストの期待値をリテラルのまま return しているかを調べる.

    実測(codestral, MBPPタスク268「n番目の星数」): `if n == 4: return 73 / elif n == 5: return 121`
    と可視テストの答えを直書きし、可視テストだけの提出前検証を通過して、隠しテスト(n=3)で落ちた。
    `assert f(...) == 期待値` の期待値のうち、ありふれた値を除いたものが2つ以上、そのまま
    return のリテラルとして現れたら直書きとみなす(1つだけの一致は偶然でも起こるので見逃す)。

    Args:
        solution (str): 提出コード。
        test_list (list[str]): 可視テスト(assert文)のリスト。

    Returns:
        list[str]: 直書きされていた期待値の repr。直書きでなければ空リスト。
    """
    expected: list[object] = []
    for test in test_list:
        try:
            tree = ast.parse(test.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) \
                    and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
                is_literal, value = _literal_value(node.test.comparators[0])
                if is_literal and not _is_trivial(value):
                    expected.append(value)
    if len(expected) < 2:
        return []
    try:
        solution_tree = ast.parse(solution)
    except SyntaxError:
        return []
    returned: list[object] = []
    for node in ast.walk(solution_tree):
        if isinstance(node, ast.Return) and node.value is not None:
            is_literal, value = _literal_value(node.value)
            if is_literal:
                returned.append(value)
    matched: list[str] = []
    for value in expected:
        if any(type(value) is type(candidate) and value == candidate for candidate in returned):
            if repr(value) not in matched:
                matched.append(repr(value))
    return matched if len(matched) >= 2 else []


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

    def sanitize_submission(self, raw: str) -> str:
        """提出文字列からコードフェンスを除く.

        Args:
            raw (str): final_answer()が捕捉した生の文字列。

        Returns:
            str: 整えた提出コード。
        """
        return strip_code_fences(raw)

    def submission_rejection(self, candidate: str) -> str | None:
        """採点側の実行方式で例外を投げない関数ソースかを検査する.

        Args:
            candidate (str): 整えた提出コード。

        Returns:
            str | None: 受理できるならNone。だめなら理由。
        """
        return mbpp_rejection(candidate)

    def is_fallback_candidate(self, code: str) -> bool:
        """そのまま提出できる形(関数定義を含み、トップレベルに危険な文が無い)のコードかを返す.

        Args:
            code (str): その反復で実行したコード。

        Returns:
            bool: 打ち切り時の提出候補として残すならTrue。
        """
        return mbpp_rejection(code) is None

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
