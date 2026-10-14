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
SALVAGE_ESCAPED_QUOTES = "escaped_quotes"

# JSON形式のツール呼び出しで学習したモデルは、コードブロック全体をJSON文字列のように
# エスケープして返すことがある。実測(gpt-oss, MBPPタスク305): few-shotの
# `if report["success"]:` を `if report[\"success\"]:` と書き、文字列の外のバックスラッシュで
# SyntaxError(unexpected character after line continuation character)になり4反復を溶かした。
# 三連引用符の中の \" は正当なので一括置換はできない。構文エラーが出た行だけを戻す。
_MAX_UNESCAPE_PASSES = 30


def _repair_escaped_quotes(code: str) -> str | None:
    """構文エラーの行に限ってクォートのエスケープを戻し、構文が通る形を探す.

    Args:
        code (str): 抽出したコード。

    Returns:
        str | None: 修復できたコード。もともと構文が通る場合や修復できない場合はNone。
    """
    try:
        ast.parse(code)
        return None
    except SyntaxError:
        pass

    lines = code.splitlines()
    for _ in range(_MAX_UNESCAPE_PASSES):
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except SyntaxError as error:
            index = (error.lineno or 0) - 1
            if not 0 <= index < len(lines):
                return None
            repaired = lines[index].replace('\\"', '"').replace("\\'", "'")
            if repaired == lines[index]:
                return None  # その行にエスケープが無い = 別の原因の構文エラー
            lines[index] = repaired
    return None


def _extract_with_kind(llm_output: str) -> tuple[str | None, str | None]:
    """コードを抽出し、崩れを救済した場合はその種類も返す.

    Args:
        llm_output (str): LLMが生成した出力テキスト全体。

    Returns:
        tuple[str | None, str | None]: (抽出したコード, 救済の種類)。
            抽出できなければ(None, None)。正規のブロックをそのまま使えたなら種類はNone。
    """
    blocks = _PYTHON_CODE_BLOCK_PATTERN.findall(llm_output)
    kind: str | None = None
    code: str | None = None
    if blocks:
        # 複数ある場合は最後のブロックを使う
        code = str(blocks[-1].rstrip())
    else:
        match = _UNCLOSED_CODE_BLOCK_PATTERN.search(llm_output)
        if match and match.group(1).strip():
            code = str(match.group(1).rstrip())
            kind = SALVAGE_UNCLOSED_FENCE
        else:
            code = _extract_bare_code(llm_output)
            if code is not None:
                kind = SALVAGE_UNFENCED
    if code is None:
        return None, None

    # 過剰エスケープは実行自体を止めてしまうので、他の救済より優先して報告する
    repaired = _repair_escaped_quotes(code)
    if repaired is not None:
        return repaired, SALVAGE_ESCAPED_QUOTES
    return code, kind


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
        str | None: そのまま使えたならNone。救済した場合は SALVAGE_* のいずれか。
    """
    return _extract_with_kind(llm_output)[1]


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
        return _extract_with_kind(llm_output)[0]
