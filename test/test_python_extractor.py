"""PythonCodeExtractor の単体テスト."""
import pytest

from src.agent.extractor.python_extractor import (
    SALVAGE_UNCLOSED_FENCE,
    SALVAGE_UNFENCED,
    PythonCodeExtractor,
    salvage_kind,
)


@pytest.fixture
def extractor() -> PythonCodeExtractor:
    """テスト対象の抽出器を返す."""
    return PythonCodeExtractor()


def test_extracts_single_block(extractor: PythonCodeExtractor) -> None:
    """前後に地の文がある単一ブロックからコードだけ取れる."""
    text = "説明文\n```python\nprint('hi')\n```\n続き"
    assert extractor.extract(text) == "print('hi')"


def test_returns_last_block_when_multiple(extractor: PythonCodeExtractor) -> None:
    """複数ブロックがあるとき最後のブロックを返す(D-b)."""
    text = (
        "例:\n```python\nx = 1\n```\n"
        "本命:\n```python\nx = 2\nprint(x)\n```\n"
    )
    assert extractor.extract(text) == "x = 2\nprint(x)"


def test_returns_none_when_no_block(extractor: PythonCodeExtractor) -> None:
    """コードブロックが無ければ None(D-a)."""
    assert extractor.extract("コードはありません") is None


def test_ignores_non_python_fence(extractor: PythonCodeExtractor) -> None:
    """python/py 以外の言語ラベルは拾わない."""
    assert extractor.extract("```js\nconsole.log(1)\n```") is None


def test_accepts_py_and_uppercase_label(extractor: PythonCodeExtractor) -> None:
    """py / 大文字ラベルも受け付ける(点2)."""
    assert extractor.extract("```py\nprint(1)\n```") == "print(1)"
    assert extractor.extract("```Python\nprint(2)\n```") == "print(2)"


def test_rstrips_trailing_blank_lines(extractor: PythonCodeExtractor) -> None:
    """末尾の空行/空白は除去する(点3)."""
    assert extractor.extract("```python\nx = 1\n\n\n```") == "x = 1"


def test_multiline_code_preserved(extractor: PythonCodeExtractor) -> None:
    """複数行コードのインデント・改行(末尾以外)が保たれる."""
    text = "```python\ndef f():\n    return 1\n```"
    assert extractor.extract(text) == "def f():\n    return 1"


def test_closing_fence_without_trailing_newline(extractor: PythonCodeExtractor) -> None:
    """閉じフェンスが独立行なら、その後に改行が無くても取れる."""
    assert extractor.extract("```python\nprint('x')\n```") == "print('x')"


def test_inline_fence_is_ignored(extractor: PythonCodeExtractor) -> None:
    """行頭でないフェンス(地の文の途中)は拾わない(点4)."""
    assert extractor.extract("見て ```python x``` ね") is None


def test_nested_backticks_in_string(extractor: PythonCodeExtractor) -> None:
    """コード中の行頭でない ``` では閉じない(点1)."""
    text = "```python\ns = '```'\nprint(s)\n```"
    assert extractor.extract(text) == "s = '```'\nprint(s)"


def test_unfenced_final_answer_is_salvaged(extractor: PythonCodeExtractor) -> None:
    """フェンス無しで素のPythonを返した応答も救済して実行する.

    実測(MBPPタスク235 step1)でgpt-ossが final_answer(三連引用符の関数ソース) をフェンス無しで
    出力し、抽出できず1反復(600トークン)を丸損した。A13(崩れた形式の解釈)が要求する救済。
    """
    text = 'final_answer("""def f(n):\n    return n * 2\n""")'
    assert extractor.extract(text) == text
    assert salvage_kind(text) == SALVAGE_UNFENCED


def test_unfenced_code_after_prose_is_salvaged(extractor: PythonCodeExtractor) -> None:
    """地の文のあとにフェンス無しのコードが続く場合、コード部分から拾う."""
    text = "Thought: just define it.\ndef f(n):\n    return n + 1\nprint(f(1))"
    assert extractor.extract(text) == "def f(n):\n    return n + 1\nprint(f(1))"
    assert salvage_kind(text) == SALVAGE_UNFENCED


def test_prose_only_is_not_mistaken_for_code(extractor: PythonCodeExtractor) -> None:
    """純粋な散文をコードと誤認しない(構文が通らないものは採用しない)."""
    text = "I think the answer is 42 but I will not write any code block."
    assert extractor.extract(text) is None
    assert salvage_kind(text) is None


def test_salvage_kind_distinguishes_unclosed_fence(extractor: PythonCodeExtractor) -> None:
    """閉じフェンス欠落とフェンス無しを区別する(LLMに返す助言が異なるため)."""
    assert salvage_kind("```python\nprint('x')\n") == SALVAGE_UNCLOSED_FENCE
    assert salvage_kind("```python\nprint('x')\n```") is None
