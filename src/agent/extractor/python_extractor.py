"""Pythonコード用コード抽出機能を提供するモジュール.

LLM出力中の```python```フェンスブロックを抽出する具象クラス。
"""
import ast
import re

from src.agent.extractor.code_extractor import CodeExtractor

# ```python``` Pythonコードブロック
# [ \t]* ラベルのあとの空白、タブを許可
# \r?\n コードブロックあとの改行を許可(ここが開始フェンスと地の文の境目)
# (.*?) コード本体。re.DOTALLで改行を含む任意の文字列を許可
# \r?\n[ \t]*``` 閉じフェンスは行頭(直前が改行)に限る
#
# 開始フェンスに行頭アンカー(^[ \t]*)は付けない。実測(MBPPタスク251/395/108)でモデルが
# `Thought: ...すること。```python\n` のようにThoughtと同じ行にフェンスを書き、行頭アンカーの
# せいで厳格パターンが0件マッチ -> 未閉鎖フェンス救済に落ちて末尾の``` まで取り込み構文エラー、
# さらに「出力上限で切られた」という誤った原因をLLMへ伝えていた。
# 行中のフェンス(例: "見て ```python x``` ね")はラベル直後に改行が無いので依然マッチしない。
_PYTHON_CODE_BLOCK_PATTERN = re.compile(
    r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)\r?\n[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE | re.IGNORECASE
)
# 未閉鎖ブロックの救済で本体末尾に閉じフェンスが残っていたら落とす(保険)。
_TRAILING_FENCE_PATTERN = re.compile(r"\r?\n[ \t]*```[ \t]*\Z")

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
SALVAGE_JSON_WRAPPED = "json_wrapped"
SALVAGE_MULTIPLE_BLOCKS = "multiple_blocks"
SALVAGE_UNMATCHED_CLOSERS = "unmatched_closers"

# 停止シーケンスの「前半だけ」が本文末尾に漏れたものを取り除くときの最短長。
# 1文字("<"など)は正当なコードの末尾でもあり得るので対象にしない。
_MIN_LEAKED_STOP_PREFIX = 2


def strip_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    """LLM出力から停止シーケンスとその残骸を取り除く.

    停止シーケンスはAPI側で生成を止めるための印で、コードの一部ではない。完全な形で
    混ざっていれば全て除き、本文の末尾に前半だけが残っていればそれも除く。
    実測(NVIDIA NIMのnemotron-3-super): 22ステップ中17回、本文が ```` ```<end_code ```` で
    終わっていた(`>`の手前で止まり、そこまでが本文に含まれる)。閉じフェンスと同じ行に
    残骸があるため閉じフェンスと認識されず、未閉鎖ブロックの救済に回って残骸ごと実行され
    SyntaxErrorになっていた。

    Args:
        text (str): LLMの生の出力。
        stop_sequences (list[str]): リクエストに指定した停止シーケンス。

    Returns:
        str: 停止シーケンスとその残骸を除いた文字列。
    """
    cleaned = text
    for stop in stop_sequences:
        if stop:
            cleaned = cleaned.replace(stop, "")
    stripped = cleaned.rstrip()
    for stop in stop_sequences:
        for length in range(len(stop) - 1, _MIN_LEAKED_STOP_PREFIX - 1, -1):
            if stripped.endswith(stop[:length]):
                return stripped[:-length].rstrip() + "\n"
    return cleaned


def _unwrap_json_code(code: str) -> str | None:
    """```python ブロックの中身が {"code": "..."} 形式の包みなら、中のコードを取り出す.

    ツール呼び出しのJSONで学習したモデルは、Pythonブロックの中に
    `{"code": "print(1)"}` のようなJSONを書くことがある(実測: NVIDIAのnemotronがMBPP
    タスク305で4反復連続)。dictリテラルとしては正しいPythonなので実行しても何も起きず、
    反復だけが消費される。中身が「codeキーに文字列を持つdict」だけのときに限り展開する。

    Args:
        code (str): フェンスから取り出したコード。

    Returns:
        str | None: 展開したコード。包みでなければNone。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    node = tree.body[0].value
    if not isinstance(node, ast.Dict):
        return None
    try:
        payload = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    if isinstance(payload.get("arguments"), dict):
        payload = payload["arguments"]
    inner = payload.get("code")
    if isinstance(inner, str) and inner.strip():
        return inner
    return None


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


# 文字列の中の括弧まで数えて、閉じ括弧を多く書くモデルがある。実測(nemotron-3-super, sympy-14711):
# print(run_command(command="python -c \"print(sum([a, (0 * b)]))\"")))) のように `)` を1〜2個多く書き、
# 「unmatched ')'」を8反復続けた(1個だけ消しては、もう1個残っていて失敗する、を繰り返した)。
# Pythonの構文解析器は余った閉じ括弧の位置を正確に返すので、その1文字だけを消して解析し直す。
_UNMATCHED_CLOSER = re.compile(r"^unmatched '([)\]}])'")
_MAX_CLOSER_REMOVALS = 5


def _repair_unmatched_closers(code: str) -> str | None:
    """構文解析器が「対応しない閉じ括弧」と報告した文字だけを取り除き、構文が通る形を探す.

    Args:
        code (str): 抽出したコード。

    Returns:
        str | None: 修復できたコード。もともと構文が通る場合や、別の原因の構文エラーならNone。
    """
    lines = code.split("\n")
    for attempt in range(_MAX_CLOSER_REMOVALS + 1):
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines) if attempt else None
        except SyntaxError as error:
            match = _UNMATCHED_CLOSER.match(error.msg or "")
            index, column = (error.lineno or 0) - 1, (error.offset or 0) - 1
            if match is None or not 0 <= index < len(lines) or not 0 <= column < len(lines[index]) \
                    or lines[index][column] != match.group(1):
                return None
            lines[index] = lines[index][:column] + lines[index][column + 1:]
    return None


def _extract_with_kind(llm_output: str) -> tuple[str | None, str | None]:
    """コードを抽出し、崩れを救済した場合はその種類も返す.

    Args:
        llm_output (str): LLMが生成した出力テキスト全体。

    Returns:
        tuple[str | None, str | None]: (抽出したコード, 救済の種類)。
            抽出できなければ(None, None)。正規のブロックをそのまま使えたなら種類はNone。
    """
    blocks = [str(block).rstrip() for block in _PYTHON_CODE_BLOCK_PATTERN.findall(llm_output)]
    kind: str | None = None
    code: str | None = None
    if len(blocks) > 1:
        # 「関数定義のブロック」と「それを検証するブロック」に分けて書くモデルがある
        # (実測: nemotron。最後のブロックだけを実行すると定義が無くNameErrorになり、
        # 打ち切り時のフォールバック候補も残らない)。連結して構文が通れば全部を順に実行し、
        # 通らなければ従来どおり最後のブロックを使う。
        joined = "\n\n".join(blocks)
        try:
            ast.parse(joined)
            code, kind = joined, SALVAGE_MULTIPLE_BLOCKS
        except SyntaxError:
            code = blocks[-1]
    elif blocks:
        code = blocks[0]
    else:
        match = _UNCLOSED_CODE_BLOCK_PATTERN.search(llm_output)
        if match and match.group(1).strip():
            body = _TRAILING_FENCE_PATTERN.sub("", str(match.group(1)))
            code = body.rstrip()
            kind = SALVAGE_UNCLOSED_FENCE
        else:
            code = _extract_bare_code(llm_output)
            if code is not None:
                kind = SALVAGE_UNFENCED
    if code is None:
        return None, None

    unwrapped = _unwrap_json_code(code)
    if unwrapped is not None:
        return unwrapped, SALVAGE_JSON_WRAPPED

    # 過剰エスケープは実行自体を止めてしまうので、他の救済より優先して報告する
    repaired = _repair_escaped_quotes(code)
    if repaired is not None:
        return repaired, SALVAGE_ESCAPED_QUOTES
    repaired = _repair_unmatched_closers(code)
    if repaired is not None:
        return repaired, SALVAGE_UNMATCHED_CLOSERS
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
