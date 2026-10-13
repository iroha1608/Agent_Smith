"""Pythonコード用コード抽出機能を提供するモジュール.

LLM出力中の```python```フェンスブロックを抽出する具象クラス。
"""
import re

from src.agent.extractor.code_extractor import CodeExtractor

# ```python``` Pythonコードブロック
# [ \t]* ラベルのあとの空白、タブを許可
# \r?\n コードブロックあとの改行を許可
# (.*?) コード本体。re.DOTALLで改行を含む任意の文字列を許可
# \r?\n? コードブロックの終わりの直前に改行を許可
_PYTHON_CODE_BLOCK_PATTERN = re.compile(
    r"^[ \t]*```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)\r?\n[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE | re.IGNORECASE
)

# 閉じフェンスが無いブロックの救済用。出力トークン上限で応答が途中で切れると
# ```python のあとが閉じられずに終わる。捨てると1反復が丸ごと無駄になるので、
# 開始フェンス以降を本体として拾う(課題 V.1「崩れたブロックを解釈したなら明示する」)。
_UNCLOSED_CODE_BLOCK_PATTERN = re.compile(
    r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*)\Z", re.DOTALL | re.IGNORECASE
)


def was_salvaged(llm_output: str) -> bool:
    """閉じフェンス無しのブロックを救済して抽出したかどうかを返す.

    Args:
        llm_output (str): LLMが生成した出力テキスト全体。

    Returns:
        bool: 正規のブロックが無く、未閉鎖ブロックから拾った場合True。
    """
    if _PYTHON_CODE_BLOCK_PATTERN.findall(llm_output):
        return False
    match = _UNCLOSED_CODE_BLOCK_PATTERN.search(llm_output)
    return bool(match and match.group(1).strip())


class PythonCodeExtractor(CodeExtractor):
    """LLMの出力テキストからPythonブロックのコードを抽出する具象クラス."""

    def extract(self, llm_output: str) -> str | None:
        """LLMの出力テキストからPythonコードを抽出する.

        複数のPythonコードブロックが存在する場合は、最後のブロックを抽出する。

        Args:
            llm_output (str): LLMが生成した出力テキスト全体。

        Returns:
            str | None: 抽出されたコード
            または抽出できなかった場合(コードブロックが見つからない)はNone
        """
        # テキスト全体からPythonコードブロックを抽出
        code_blocks = _PYTHON_CODE_BLOCK_PATTERN.findall(llm_output)
        if code_blocks:
            # 末尾の改行を削除し最後のコードブロックを返す
            return str(code_blocks[-1].rstrip())

        # 正規のブロックが無い場合は、閉じフェンスが欠けたブロックを救済する
        match = _UNCLOSED_CODE_BLOCK_PATTERN.search(llm_output)
        if match and match.group(1).strip():
            return str(match.group(1).rstrip())
        return None
