"""Orchestrator(orchestrator.py)のループ制御テスト。LLMはStubProviderで決定的に差替."""
from src.agent.llm.resilient_client import AllProvidersExhausted
from src.agent.extractor.python_extractor import PythonCodeExtractor
from src.agent.llm.provider.provider import LLMProvider, LLMResponse
from src.agent.orchestrator import Orchestrator
from src.agent.prompt.task_profile.mbpp_profile import MBPPProfile
from src.config.settings import Limits
from src.sandbox.mock import MockSandbox

_RESP_PRINT = "Thought: test it.\n```python\nprint('hello')\n```\n<end_code>"
_RESP_DEF = "Thought: draft it.\n```python\ndef f():\n    return 1\n```\n<end_code>"
_RESP_FINAL = "Thought: submit.\n```python\nfinal_answer('def f():\\n    return 1')\n```\n<end_code>"
_RESP_NO_CODE = "I think the answer is 42 but I will not write any code block."
_RESP_ENDCODE_INSIDE = "Thought: oops.\n```python\nprint('x')<end_code>\n```"


class StubProvider(LLMProvider):
    """台本どおりに応答を返す決定的プロバイダ(ネットワーク不要)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        text = self._responses[self._index]
        self._index += 1
        return LLMResponse(
            text=text, input_tokens=10, output_tokens=5, request_time_ms=1.0,
            api_url="stub", model_name="stub", retries=0,
        )

    def _build_request(  # type: ignore[no-untyped-def]
        self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int
    ):
        raise NotImplementedError

    def _parse_response(self, body: str, request_time_ms: float) -> LLMResponse:
        raise NotImplementedError


def _limits(**kw: object) -> Limits:
    base: dict[str, object] = dict(
        max_iterations=5, max_input_tokens=10**9, max_output_tokens=10**9,
        total_time_sec=10**9, time_margin_sec=1.0,
    )
    base.update(kw)
    return Limits(**base)  # type: ignore[arg-type]


def _orchestrator(responses: list[str], **limit_kw: object) -> Orchestrator:
    return Orchestrator(
        provider=StubProvider(responses),
        extractor=PythonCodeExtractor(),
        sandbox=MockSandbox(),
        profile=MBPPProfile(),
        limits=_limits(**limit_kw),
    )


def test_final_answer_success() -> None:
    """1反復で final_answer 提出 -> success / solution / iterations."""
    solution = _orchestrator([_RESP_FINAL]).run("82", "mbpp", "task")
    assert solution.success is True
    assert solution.solution == "def f():\n    return 1"
    assert solution.iterations == 1
    assert len(solution.steps) == 1
    assert solution.error is None


def test_no_code_then_success() -> None:
    """コード無し -> 再促し -> 次反復で提出成功(2反復)."""
    solution = _orchestrator([_RESP_NO_CODE, _RESP_FINAL]).run("82", "mbpp", "task")
    assert solution.success is True
    assert solution.iterations == 2
    assert solution.steps[0].sandbox_input == ""
    assert solution.steps[0].sandbox_output.startswith("[NO_CODE]")


def test_limit_stop_max_iterations() -> None:
    """final_answer を出さないと max_iterations で打切り(提出可能な best-so-far を提出)."""
    solution = _orchestrator([_RESP_DEF, _RESP_DEF], max_iterations=2).run("82", "mbpp", "task")
    assert solution.success is False
    assert solution.error == "MAX_ITERATIONS"
    assert solution.iterations == 2
    assert solution.solution == "def f():\n    return 1"


def test_fallback_skips_non_submittable_code() -> None:
    """打切り時のフォールバックに、関数定義を含まない作業コードは採用しない.

    採点側は提出コードにテスト行を連結して実行するため、printだけの断片を提出すると
    「関数が無い」で確実に不合格になる。空で返す方が誤提出より正しい。
    """
    solution = _orchestrator([_RESP_PRINT, _RESP_PRINT], max_iterations=2).run("82", "mbpp", "task")
    assert solution.success is False
    assert solution.error == "MAX_ITERATIONS"
    assert solution.solution == ""


class _TruncatingStub(LLMProvider):
    """1反復の出力上限ぶんを使い切って切られた応答を模すスタブ(output_tokens=max_tokens)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        text = self._responses[self._index]
        self._index += 1
        return LLMResponse(
            text=text, input_tokens=10, output_tokens=max_tokens, request_time_ms=1.0,
            api_url="stub", model_name="stub", retries=0,
        )

    def _build_request(  # type: ignore[no-untyped-def]
        self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int
    ):
        raise NotImplementedError

    def _parse_response(self, body: str, request_time_ms: float) -> LLMResponse:
        raise NotImplementedError


def test_no_code_cut_off_names_prose_as_the_cause() -> None:
    """出力上限で切られてコードに到達しなかった反復は、原因(散文で考え込んだ)を名指しする.

    実測(MBPPタスク264)では3反復すべてがこれで、既定の再促しでは同じ失敗を繰り返して
    累積出力1500を使い切って不合格になった。原因を指摘できているかを検証する。
    """
    orchestrator = Orchestrator(
        provider=_TruncatingStub([_RESP_NO_CODE, _RESP_FINAL]),
        extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(max_output_tokens_per_iteration=50),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    observation = solution.steps[0].sandbox_output
    assert observation.startswith("[NO_CODE]")
    assert "50-token allowance" in observation
    assert "cut off before any" in observation
    assert solution.success is True


def test_no_code_without_truncation_omits_cut_off_note() -> None:
    """上限に達せず自然に書き終えた場合は、切り詰めの助言を混ぜない(誤った原因を教えない)."""
    solution = _orchestrator([_RESP_NO_CODE, _RESP_FINAL]).run("82", "mbpp", "task")
    observation = solution.steps[0].sandbox_output
    assert observation.startswith("[NO_CODE]")
    assert "cut off before any" not in observation


def test_budget_note_appended_when_one_turn_of_output_remains() -> None:
    """残り出力トークンが1反復ぶんまで減ったら、Observationに提出を促す警告を足す."""
    solution = _orchestrator(
        [_RESP_PRINT, _RESP_FINAL], max_output_tokens=55, max_output_tokens_per_iteration=50,
    ).run("82", "mbpp", "task")
    assert "[BUDGET] Only 50 output tokens remain" in solution.steps[0].sandbox_output


def test_budget_note_absent_while_two_turns_of_output_remain() -> None:
    """残り2反復ぶんある間は促さない.

    本番設定(総1500 / 1反復600)で早期に促すと、3〜5反復かけて正解に至るタスクを
    未完成のまま提出させてしまう。促すのは「次が最後の1手」になってから。
    """
    solution = _orchestrator(
        [_RESP_PRINT, _RESP_FINAL], max_output_tokens=105, max_output_tokens_per_iteration=50,
    ).run("82", "mbpp", "task")
    assert "[BUDGET]" not in solution.steps[0].sandbox_output


def test_budget_note_absent_when_budget_is_ample() -> None:
    """予算に余裕があるうちは警告を足さない(毎反復の入力を無駄に膨らませない)."""
    solution = _orchestrator([_RESP_PRINT, _RESP_FINAL]).run("82", "mbpp", "task")
    assert "[BUDGET]" not in solution.steps[0].sandbox_output


_SUBMITTABLE = "import math\n\n\ndef f(n):\n    return math.isqrt(n)\n"
# 実測(MBPPタスク160)で打ち切り時のフォールバックが提出してしまった検証用ラッパー。
# 構文は通り `def ` も含むため旧実装では受理されたが、採点環境に run_tests は無い。
_HARNESS_WRAPPER = (
    'import json\n'
    'report = json.loads(run_tests(code="""\ndef f(n):\n    return n\n""", test_list=["assert f(1)==1"]))\n'
)


def test_reject_reason_accepts_imports_and_definitions() -> None:
    """import + 関数定義だけの提出は受理する(モジュールdocstring/定数も可)."""
    assert Orchestrator._reject_reason(_SUBMITTABLE) is None
    assert Orchestrator._reject_reason('"""doc."""\nLIMIT = 3\n\n\ndef f():\n    return LIMIT\n') is None


def test_reject_reason_rejects_sandbox_only_names() -> None:
    """run_tests / final_answer を参照する提出は拒否する.

    採点側はこれらを定義しないため、提出すると NameError で全テストが落ちる。
    """
    reason = Orchestrator._reject_reason(_HARNESS_WRAPPER)
    assert reason is not None
    assert "run_tests" in reason


def test_reject_reason_rejects_top_level_statements() -> None:
    """トップレベルのデモ呼び出しやassertは拒否する(採点時に例外を投げ得る)."""
    for bad in ("def f():\n    return 1\n\n\nprint(f())\n",
                "def f():\n    return 1\n\n\nassert f() == 1\n"):
        reason = Orchestrator._reject_reason(bad)
        assert reason is not None
        assert "top-level" in reason


def test_reject_reason_rejects_empty_and_broken_code() -> None:
    """空 / 構文エラー / 関数定義なしはこれまでどおり拒否する."""
    assert Orchestrator._reject_reason("") is not None
    assert Orchestrator._reject_reason("def f(:\n    pass\n") is not None
    assert Orchestrator._reject_reason("x = 1\n") is not None


def test_window_truncation_keeps_the_trailing_note() -> None:
    """履歴の切り詰めで末尾の助言が落ちない(中央を省いて両端を残す).

    Observationは末尾に `--- note ---` が来る。先頭だけ残す切り詰めだと、切り詰め警告や
    予算警告や提出の却下理由といった「次に何をすべきか」が必ず消えてLLMに届かない。
    """
    observation = "[OK] head marker\n" + ("x" * 5000) + "\n--- note ---\nTAIL MARKER"
    windowed = Orchestrator._window_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
         {"role": "assistant", "content": "a"}, {"role": "user", "content": observation}]
    )
    sent = windowed[-1]["content"]
    assert len(sent) <= Orchestrator._OBS_MAX_CHARS + len("\n...[truncated]...\n")
    assert "head marker" in sent
    assert "TAIL MARKER" in sent


def test_endcode_inside_block_is_stripped() -> None:
    """コードブロック内の <end_code> を抽出前に除去して実行できる(回収#2)."""
    solution = _orchestrator([_RESP_ENDCODE_INSIDE, _RESP_FINAL]).run("82", "mbpp", "task")
    assert solution.steps[0].sandbox_output.startswith("[OK]")
    assert "x" in solution.steps[0].sandbox_output
    assert solution.success is True


def test_solution_metrics_totals() -> None:
    """トークン累計 / リクエスト数 / 反復数が集計される."""
    solution = _orchestrator([_RESP_PRINT, _RESP_FINAL]).run("82", "mbpp", "task")
    assert solution.total_input_tokens == 20
    assert solution.total_output_tokens == 10
    assert solution.total_requests == 2
    assert solution.iterations == 2


class _RaisingProvider:
    """generate が必ず AllProvidersExhausted を送出する(全滅の再現)."""

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        raise AllProvidersExhausted("all providers/keys exhausted")


def test_llm_unavailable_is_graceful() -> None:
    """generateが全滅例外でもクラッシュせず success=False / error=LLM_UNAVAILABLE を返す."""
    orchestrator = Orchestrator(
        provider=_RaisingProvider(), extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    assert solution.success is False
    assert solution.error == "LLM_UNAVAILABLE"
    assert solution.steps == []
    assert solution.solution == ""
    assert solution.total_requests == 0  # attempts未指定(=0)の全滅なら試行加算も0


class _RetryingStub(LLMProvider):
    """各応答に retries 回数を持たせて返すスタブ(total_requests計上の検証用)."""

    def __init__(self, responses: list[tuple[str, int]]) -> None:
        self._responses = responses
        self._index = 0

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        text, retries = self._responses[self._index]
        self._index += 1
        return LLMResponse(
            text=text, input_tokens=10, output_tokens=5, request_time_ms=1.0,
            api_url="stub", model_name="stub", retries=retries,
        )

    def _build_request(self, messages, stop_sequences, max_tokens):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def _parse_response(self, body: str, request_time_ms: float) -> LLMResponse:
        raise NotImplementedError


def test_total_requests_includes_retries() -> None:
    """total_requests は各stepの (retries+1) 合計(HTTP試行数、リトライ込み)."""
    provider = _RetryingStub([(_RESP_PRINT, 2), (_RESP_FINAL, 0)])
    orchestrator = Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    # step1: retries=2 -> 3試行, step2: retries=0 -> 1試行 => 合計4
    assert solution.total_requests == 4
    assert solution.steps[0].retries == 2


class _ExhaustAfterFirstProvider:
    """1回目のgenerateは成功、2回目で attempts を消費して全滅する(途中全滅の再現)."""

    def __init__(self, first: LLMResponse, attempts: int) -> None:
        self._first = first
        self._attempts = attempts
        self._called = 0

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        self._called += 1
        if self._called == 1:
            return self._first
        raise AllProvidersExhausted("exhausted", attempts=self._attempts)


def test_total_requests_includes_failed_attempts() -> None:
    """途中で全滅した場合、成功step分 + 全滅時の試行数が total_requests に入る."""
    first = LLMResponse(
        text=_RESP_PRINT, input_tokens=10, output_tokens=5,
        request_time_ms=1.0, api_url="stub", model_name="stub", retries=0,
    )
    provider = _ExhaustAfterFirstProvider(first, attempts=3)
    orchestrator = Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    assert solution.success is False
    assert solution.error == "LLM_UNAVAILABLE"
    assert len(solution.steps) == 1          # 1回目のstepだけ積まれる
    assert solution.total_requests == 1 + 3  # step1の1試行 + 全滅時の3試行
