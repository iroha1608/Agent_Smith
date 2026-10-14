"""トークン使用量の抽出を担うモジュール.

1レスポンスから input / output トークン数を取り出す。
ステップ横断での合計(SolutionOutput.total_input_tokens 等)の積み上げはOrchestratorが担う。

課題の制限(VI.1)は「思考トークンも出力トークンとして数える」ため、プロバイダの報告が
実際より小さい場合は大きい側に倒す(過少計上は制限違反を見逃すが、過大計上は安全側)。
"""
from typing import Any

# usageが無い / 信用できない場合に文字数から概算するときの係数(1トークン≒4文字)
_CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens_from_text(text: str) -> int:
    """文字数からトークン数を概算する.

    Args:
        text (str): 概算対象の文字列。

    Returns:
        int: 概算トークン数(空文字なら0)。
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _as_int(value: Any) -> int | None:
    """usageの値をintとして取り出す。数値でなければNone.

    Args:
        value (Any): usage内の値。

    Returns:
        int | None: 整数値。boolや非数値はNone。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def extract_token_usage(response_json: dict[str, Any], text: str) -> tuple[int, int]:
    """レスポンスJSONから input_tokens / output_tokens トークン数を抽出する.

    出力トークンは次の最大値を採る(どれも「少なくともこれだけは生成した」という下限):
      - completion_tokens
      - total_tokens - prompt_tokens(思考トークンを completion_tokens に含めない実装への対策)
      - 本文があるのに上記が0なら、本文の文字数からの概算
        (実測: Gemini経由のgemma-4は本文1200字でも completion_tokens=0 を返した)

    Args:
        response_json (dict[str, Any]): LLMプロバイダのレスポンスJSON。
        text (str): レスポンスのテキスト部分。

    Returns:
        tuple[int, int]: input_tokensとoutput_tokensのタプル。
        usageが無い場合 input_tokensは0、output_tokensは本文からの概算値を返す。
    """
    usage = response_json.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    total_tokens = _as_int(usage.get("total_tokens"))

    input_tokens = prompt_tokens if prompt_tokens is not None else 0
    candidates = [estimate_tokens_from_text(text)] if text else [0]
    if completion_tokens is not None:
        candidates.append(completion_tokens)
        # 報告値がある場合、本文からの概算は「報告が0なのに本文がある」ときだけ使う
        if completion_tokens > 0:
            candidates = [completion_tokens]
    if total_tokens is not None and prompt_tokens is not None:
        candidates.append(total_tokens - prompt_tokens)
    return input_tokens, max(0, *candidates)
