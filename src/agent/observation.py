"""LLMに返すObservationの文字列の体裁をExecuteResultから成形するモジュール.

先頭にstatusラベル(全列挙)、1行要約、stdout, stderr / traceback, 任意でnoteを付与する。
巨大なstdout /stderrのトークン切り詰めは履歴の構築側で行う。
"""
from enum import Enum

from src.agent.extractor.python_extractor import (
    SALVAGE_ESCAPED_QUOTES,
    SALVAGE_JSON_WRAPPED,
    SALVAGE_MULTIPLE_BLOCKS,
    SALVAGE_UNCLOSED_FENCE,
    SALVAGE_UNFENCED,
)
from src.models.output import ExecuteResult


class Status(str, Enum):
    """Observationの先頭に付与するステータスラベル."""

    OK = "OK"
    EXCEPTION = "EXCEPTION"
    TIMEOUT = "TIMEOUT"
    TRUNCATED = "TRUNCATED"
    FINAL_ANSWER_INVALID = "FINAL_ANSWER_INVALID"
    NO_CODE = "NO_CODE"


_SUMMARY: dict[Status, str] = {
    Status.OK: "Executed successfully.",
    Status.EXCEPTION: "An exception was raised during execution.",
    Status.TIMEOUT: "Execution exceeded the time limit.",
    Status.TRUNCATED: "Output was truncated by the sandbox size limit.",
    Status.FINAL_ANSWER_INVALID: "final_answer() was called with an empty value.",
    Status.NO_CODE: "No executable Python code block was found in your message."
}

_NOTE: dict[Status, str] = {
    Status.EXCEPTION: "Fix the error shown above, then re-run.",
    Status.TIMEOUT: "Add a termination condition and avoid unbounded loops.",
    Status.TRUNCATED: "Print less: show only the values you need.",
    Status.FINAL_ANSWER_INVALID: "Call final_answer() with the actual solution as its argument.",
    Status.NO_CODE: "Write exactly one ```python ... ``` code block."
}


def collect_statuses(result: ExecuteResult) -> list[Status]:
    """ExecuteResultのフラグからObservationのステータスラベルを収集する.

    Args:
        result (ExecuteResult): sandbox.run()の実行結果。

    Returns:
        list[Status]: Observationに付与するステータスラベルのリスト。何もなければOK。
    """
    statuses: list[Status] = []
    if result.error is not None:
        statuses.append(Status.EXCEPTION)
    if result.timeout:
        statuses.append(Status.TIMEOUT)
    if result.truncated:
        statuses.append(Status.TRUNCATED)
    if result.final_answer_bool and not (result.final_answer or "").strip():
        statuses.append(Status.FINAL_ANSWER_INVALID)
    if not statuses:
        statuses.append(Status.OK)
    return statuses


def _render_observation(statuses: list[Status], stdout: str, detail: str, extra_note: str = "",
                        value: str | None = None) -> str:
    """ラベル、stdout、詳細からObservation文字列を構築する.

    Args:
        statuses (list[Status]): Observationに付与するステータスラベルのリスト。
        stdout (str): sandbox.run()の標準出力。(空ならno output)
        detail (str): sandbox.run()の標準エラーまたは例外トレースバック。
        extra_note (str): ステータス固定文の後ろに足す追加の助言。空なら足さない。
        value (str | None): 最後の式の値(REPLの表示に相当)。無ければ None。

    Returns:
        str: 構築したObservation文字列。
    """
    label_prefix = "".join(f"[{status.value}]" for status in statuses)
    summary = " ".join(_SUMMARY[status] for status in statuses)
    lines: list[str] = [f"{label_prefix} {summary}", "--- stdout ---", stdout if stdout else "(no output)"]
    if value is not None:
        # print し忘れたツール呼び出しの結果も届くようにする(実測: nemotron-3-ultra が
        # run_command(...) を print せず15反復「出力が無い」と誤解し続けた)
        lines.append("--- value of the last expression ---")
        lines.append(value)
    if detail:
        lines.append("--- stderr / traceback ---")
        lines.append(detail)
    notes = [_NOTE[status] for status in statuses if status in _NOTE]
    if extra_note:
        notes.append(extra_note)
    if notes:
        lines.append("--- note ---")
        lines.append(" ".join(notes))
    return "\n".join(lines)


def build_observation(result: ExecuteResult) -> str:
    """ExecuteResultからLLMへ返すObservation文字列を構築する.

    Args:
        result (ExecuteResult): sandbox.run()の実行結果。

    Returns:
        str: LLMに返すObservation文字列。
            [LABELS] 1行要約 \n\n --- stdout --- \n\n (stderr / traceback) --- \n\n --- note ---
    """
    statuses = collect_statuses(result)
    return _render_observation(statuses, stdout=result.stdout, detail=result.error or result.stderr,
                               value=result.value)


# 「散文で考え込んで出力上限に達し、コードフェンスへ到達しなかった」場合の助言。
# 実測(MBPPタスク264)では3反復すべてがこれで、累積出力1500を使い切って不合格になった。
# 原因(prose)と対処(コードで試して観測に判定させる)を名指ししないと同じ失敗を繰り返す。
_CUT_OFF_NOTE = (
    "Your reply spent its whole {spent}-token allowance on prose and was cut off before any "
    "```python fence, so nothing ran. Do NOT derive formulas or trace arithmetic in words: "
    "write a candidate implementation in code and let run_tests tell you if it is right."
)


def observation_no_code(cut_off_at: int | None = None) -> str:
    """LLMに返すObservation文字列を構築する（コードがない場合）.

    Args:
        cut_off_at (int | None): 1反復の出力上限に達して応答が切れた場合、その消費トークン数。
            自然に書き終えた(＝単にコードを書かなかった)場合はNone。

    Returns:
        str: LLMに返すObservation文字列。
            [NO_CODE] No executable Python code block was found in your message. \n\n --- note ---
    """
    extra_note = "" if cut_off_at is None else _CUT_OFF_NOTE.format(spent=cut_off_at)
    return _render_observation([Status.NO_CODE], stdout="", detail="", extra_note=extra_note)


def budget_note(remaining_output_tokens: int) -> str:
    """残り出力トークンが乏しいことを伝え、即時提出を促す一文を返す.

    予算切れでの打ち切りは不合格が確定するため、残りを使って「現時点の最善」を
    提出させる方が期待値が高い。

    Args:
        remaining_output_tokens (int): このタスクに残っている累積出力トークン数。

    Returns:
        str: Observationの末尾に付ける警告文。
    """
    return (
        f"[BUDGET] Only {remaining_output_tokens} output tokens remain for this whole task. "
        "Submit now: one short Thought line, then a code block that calls final_answer() with "
        "your best implementation."
    )


def iteration_note(iterations_left: int, submit_hint: str) -> str:
    """反復回数の残りが少ないことを伝え、提出の段取りに入るよう促す一文を返す.

    Args:
        iterations_left (int): このタスクに残っている反復回数。
        submit_hint (str): タスク種別ごとの提出のしかた(プロファイルが決める)。

    Returns:
        str: Observationの末尾に付ける警告文。
    """
    plural = "iteration remains" if iterations_left == 1 else "iterations remain"
    return f"[BUDGET] Only {iterations_left} {plural} for this task. {submit_hint}"


_MALFORMED_NOTES: dict[str, str] = {
    SALVAGE_UNCLOSED_FENCE: (
        "[MALFORMED_CODE] Your ```python block was never closed, so your response was cut off "
        "by the per-turn output limit. I executed everything after the opening fence anyway. "
        "Write shorter turns so the block closes."
    ),
    SALVAGE_UNFENCED: (
        "[MALFORMED_CODE] You wrote bare Python with no ```python fence. I detected it and "
        "executed it anyway, but wrap your code in ```python ... ``` so it is read reliably."
    ),
    SALVAGE_ESCAPED_QUOTES: (
        "[MALFORMED_CODE] You backslash-escaped quotes outside a string literal (for example "
        'if report[\\"success\\"]: instead of if report["success"]:). That is a SyntaxError. '
        "I unescaped the broken lines and ran your code. Write plain Python quotes: this is a "
        "code block, not a JSON string."
    ),
    SALVAGE_JSON_WRAPPED: (
        "[MALFORMED_CODE] Your ```python block contained a JSON object like {\"code\": \"...\"} "
        "instead of Python. I took the string under \"code\" and executed it as Python. Write the "
        "Python statements directly inside the block."
    ),
    SALVAGE_MULTIPLE_BLOCKS: (
        "[NOTE] Your message had several ```python blocks. I joined them in order and executed "
        "them as one program. Prefer exactly one block per turn."
    ),
}


def malformed_code_note(kind: str) -> str:
    """崩れたコードを救済実行したことをLLMへ明示する一文を返す.

    課題 V.1 は「崩れたコードブロックを解釈して実行したなら、それをLLMに説明する」ことを要求する。
    黙って実行すると、LLMは自分の出力が崩れていたことに気付かず同じ失敗を繰り返す。

    Args:
        kind (str): 救済の種類(python_extractorのSALVAGE_*定数)。

    Returns:
        str: Observationの先頭に付ける説明文。未知の種類なら空文字。
    """
    return _MALFORMED_NOTES.get(kind, "")


def repeated_code_note(first_step: int) -> str:
    """以前と同一のコードを実行したことをLLMへ伝える一文を返す.

    実測(codestral, MBPP 160/235): 失敗したコードを一字一句同じまま3〜4回再提出し、
    同じ失敗で予算を使い切った。同じ入力は同じ結果になるので、明示して考え直させる。

    Args:
        first_step (int): 同じコードを最初に実行したstep番号。

    Returns:
        str: Observationの先頭に付ける説明文。
    """
    return (
        f"[REPEATED] This code is identical to what you ran in step {first_step}, so it fails the "
        "same way. Do not resubmit it: re-read the failing values and change the logic."
    )
