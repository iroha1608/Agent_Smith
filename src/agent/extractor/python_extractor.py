"""Pythonコード用コード抽出機能を提供するモジュール.

LLM出力中の```python```フェンスブロックを抽出する具象クラス。
"""
import ast
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


# フェンスを一切書かず、素のPythonを本文に返すモデルがある
# (実測: gpt-ossがタスク235のstep1で `final_answer("""def ...""")` をフェンス無しで出力し、
#  抽出できず600トークンを丸損した)。行頭が明らかにコードの位置から末尾までを候補とし、
# ast.parse()が通ったときだけ採用する(散文はほぼ構文エラーになるため誤爆しにくい)。
_BARE_CODE_START_PATTERN = re.compile(
    r"^[ \t]*(?:final_answer\(|print\(|def |class |import |from |[A-Za-z_]\w*\s*=[^=])",
    re.MULTILINE,
)
# 救済したコードが「ただの名前や文字列」でないことを確かめるために要求するノード。
_CODE_LIKE_NODES = (
    ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
)

SALVAGE_UNCLOSED_FENCE = "unclosed_fence"
SALVAGE_UNFENCED = "unfenced"


def _extract_bare_code(llm_output: str) -> str | None:
    """フェンスの無い素のPythonコードを本文から救済する.

    Args:
        llm_output (str): LLMが生成した出力テキスト全体。

    Returns:
        str | None: 構文が通り、呼び出し/定義/importを含む候補。見つからなければNone。
    """
    for match in _BARE_CODE_START_PATTERN.finditer(llm_output):
        candidate = llm_output[match.start():].strip()
        if not candidate:
            continue
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(isinstance(node, _CODE_LIKE_NODES) for node in ast.walk(tree)):
            return candidate
    return None


def salvage_kind(llm_output: str) -> str | None:
    """崩れた出力をどの方法で救済したかを返す.

    Args:
        llm_output (str): LLMが生成した出力テキスト全体。

    Returns:
        str | None: 正規のブロックがあればNone。未閉鎖フェンスから拾ったなら
            SALVAGE_UNCLOSED_FENCE、フェンス無しの素のコードから拾ったなら SALVAGE_UNFENCED。
    """
    if _PYTHON_CODE_BLOCK_PATTERN.findall(llm_output):
        return None
    match = _UNCLOSED_CODE_BLOCK_PATTERN.search(llm_output)
    if match and match.group(1).strip():
        return SALVAGE_UNCLOSED_FENCE
    if _extract_bare_code(llm_output) is not None:
        return SALVAGE_UNFENCED
    return None


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

        # フェンスを一切書かず素のPythonを返した場合も救済する
        return _extract_bare_code(llm_output)
