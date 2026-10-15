"""Orchestrator(orchestrator.py)のループ制御テスト。LLMはStubProviderで決定的に差替."""
from src.agent.llm.resilient_client import AllProvidersExhausted
from src.agent.extractor.python_extractor import PythonCodeExtractor
from src.agent.llm.provider.provider import LLMProvider, LLMResponse
from src.agent.orchestrator import Orchestrator
from src.agent.prompt.task_profile.mbpp_profile import MBPPProfile, mbpp_rejection
from src.config.settings import Limits
from src.sandbox.mock import MockSandbox

_RESP_PRINT = "Thought: test it.\n```python\nprint('hello')\n```\n<end_code>"
_RESP_DEF = "Thought: draft it.\n```python\ndef f():\n    return 1\n```\n<end_code>"
_RESP_FINAL = "Thought: submit.\n```python\nfinal_answer('def f():\\n    return 1')\n```\n<end_code>"
_RESP_NO_CODE = "I think the answer is 42 but I will not write any code block."
_RESP_ENDCODE_INSIDE = "Thought: oops.\n```python\nprint('x')<end_code>\n```"


class StubProvider(LLMProvider):
    """台本どおりに応答を返す決定的プロバイダ(ネットワーク不要)。受け取ったメッセージも記録する."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self.received: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        self.received.append(messages)
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


def _feedback_after_first_step(provider: "StubProvider | _TruncatingStub") -> str:
    """2回目のリクエストで送られた最後のuserメッセージ(=1反復目へのフィードバック)を返す."""
    return provider.received[1][-1]["content"]


def test_no_code_then_success() -> None:
    """コード無し -> 再促し -> 次反復で提出成功(2反復)。実行していないstepの入出力は空."""
    provider = StubProvider([_RESP_NO_CODE, _RESP_FINAL])
    orchestrator = Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    assert solution.success is True
    assert solution.iterations == 2
    assert solution.steps[0].sandbox_input == ""
    assert solution.steps[0].sandbox_output == ""
    assert _feedback_after_first_step(provider).startswith("[NO_CODE]")


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
        self.received: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        self.received.append(messages)
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
    provider = _TruncatingStub([_RESP_NO_CODE, _RESP_FINAL])
    orchestrator = Orchestrator(
        provider=provider,
        extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(max_output_tokens_per_iteration=50),
    )
    solution = orchestrator.run("82", "mbpp", "task")
    observation = _feedback_after_first_step(provider)
    assert observation.startswith("[NO_CODE]")
    assert "50-token allowance" in observation
    assert "cut off before any" in observation
    assert solution.success is True


def test_no_code_without_truncation_omits_cut_off_note() -> None:
    """上限に達せず自然に書き終えた場合は、切り詰めの助言を混ぜない(誤った原因を教えない)."""
    provider = StubProvider([_RESP_NO_CODE, _RESP_FINAL])
    Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    ).run("82", "mbpp", "task")
    observation = _feedback_after_first_step(provider)
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
    assert mbpp_rejection(_SUBMITTABLE) is None
    assert mbpp_rejection('"""doc."""\nLIMIT = 3\n\n\ndef f():\n    return LIMIT\n') is None


def test_reject_reason_rejects_sandbox_only_names() -> None:
    """run_tests / final_answer を参照する提出は拒否する.

    採点側はこれらを定義しないため、提出すると NameError で全テストが落ちる。
    """
    reason = mbpp_rejection(_HARNESS_WRAPPER)
    assert reason is not None
    assert "run_tests" in reason


def test_reject_reason_rejects_top_level_statements() -> None:
    """トップレベルのデモ呼び出しやassertは拒否する(採点時に例外を投げ得る)."""
    for bad in ("def f():\n    return 1\n\n\nprint(f())\n",
                "def f():\n    return 1\n\n\nassert f() == 1\n"):
        reason = mbpp_rejection(bad)
        assert reason is not None
        assert "top-level" in reason


def test_reject_reason_rejects_empty_and_broken_code() -> None:
    """空 / 構文エラー / 関数定義なしはこれまでどおり拒否する."""
    assert mbpp_rejection("") is not None
    assert mbpp_rejection("def f(:\n    pass\n") is not None
    assert mbpp_rejection("x = 1\n") is not None


def test_window_truncation_keeps_the_trailing_note() -> None:
    """履歴の切り詰めで末尾の助言が落ちない(中央を省いて両端を残す).

    Observationは末尾に `--- note ---` が来る。先頭だけ残す切り詰めだと、切り詰め警告や
    予算警告や提出の却下理由といった「次に何をすべきか」が必ず消えてLLMに届かない。
    """
    observation = "[OK] head marker\n" + ("x" * 5000) + "\n--- note ---\nTAIL MARKER"
    windowed = _orchestrator([])._window_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
         {"role": "assistant", "content": "a"}, {"role": "user", "content": observation}]
    )
    sent = windowed[-1]["content"]
    assert len(sent) <= MBPPProfile.latest_observation_max_chars + len("\n...[truncated]...\n")
    assert "head marker" in sent
    assert "TAIL MARKER" in sent


def test_older_observations_are_trimmed_harder_than_the_latest() -> None:
    """直前のObservationは長めに残し、それより前のものは短く切り詰める."""
    old_obs = "OLD " + "y" * 5000
    new_obs = "NEW " + "z" * 5000
    windowed = _orchestrator([])._window_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
         {"role": "assistant", "content": "a1"}, {"role": "user", "content": old_obs},
         {"role": "assistant", "content": "a2"}, {"role": "user", "content": new_obs}]
    )
    marker = len("\n...[truncated]...\n")
    assert len(windowed[3]["content"]) <= MBPPProfile.observation_max_chars + marker
    assert len(windowed[5]["content"]) <= MBPPProfile.latest_observation_max_chars + marker
    assert len(windowed[5]["content"]) > len(windowed[3]["content"])


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
    # LLMの応答が無い反復はstepにも反復数にも数えない(llm_outputを捏造しない)
    assert len(solution.steps) == 1
    assert solution.iterations == 1
    assert solution.total_requests == 1 + 3  # step1の1試行 + 全滅時の3試行


class _RotatingStub(StubProvider):
    """rotate()の呼び出し回数を記録するスタブ."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.rotations = 0

    def rotate(self) -> None:
        self.rotations += 1


def test_repeated_code_is_flagged_and_provider_rotates() -> None:
    """同じコードの再実行は[REPEATED]で指摘し、提出に至らない反復が2回続いたら切り替える."""
    provider = _RotatingStub([_RESP_PRINT, _RESP_PRINT, _RESP_FINAL])
    orchestrator = Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=MBPPProfile(), limits=_limits(),
    )
    solution = orchestrator.run("1", "mbpp", "task")
    assert solution.success is True
    assert "[REPEATED]" not in solution.steps[0].sandbox_output
    assert solution.steps[1].sandbox_output.startswith("[REPEATED] This code is identical to what you ran in step 1")
    assert provider.rotations == 1


def test_swebench_rotates_after_repeating_the_same_code() -> None:
    """SWE-bench では [REPEATED] の指摘のあとも同じコードを出し直したら、別プロバイダのモデルへ切り替える.

    探索で提出しない反復が続くのは正常なので、提出の有無では切り替えない(実測: codestral が
    sympy-14711 で同じ edit_file を22反復出し続けた)。
    """
    from src.agent.prompt.task_profile.swebench_profile import SWEBenchProfile
    patch = "diff --git a/x.py b/x.py\\n--- a/x.py\\n+++ b/x.py\\n@@ -1 +1 @@\\n-a\\n+b\\n"
    submit = f'Thought: submit.\n```python\nfinal_answer("{patch}")\n```'
    explore = "Thought: look.\n```python\nprint('reading')\n```"
    other = "Thought: look elsewhere.\n```python\nprint('elsewhere')\n```"
    provider = _RotatingStub([explore, other, explore, explore, submit])
    orchestrator = Orchestrator(
        provider=provider, extractor=PythonCodeExtractor(), sandbox=MockSandbox(),
        profile=SWEBenchProfile(), limits=_limits(),
    )
    solution = orchestrator.run("1", "swebench", "task")
    assert solution.success is True
    # 新しいコード2つは数えず、繰り返しが2回続いた4反復目で1回だけ切り替える
    assert provider.rotations == 1
    assert solution.steps[3].sandbox_output.startswith("[REPEATED]")


def test_input_guard_shrinks_window_before_giving_up() -> None:
    """累積入力の残りが少ないときは、履歴の窓を縮めて送り、上限を踏み越えない."""
    sent_sizes: list[int] = []

    class _Recording(StubProvider):
        def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
            sent_sizes.append(len(messages))
            response = super().generate(messages, stop_sequences, max_tokens)
            # 送った文字数に見合う入力トークンを報告する(1トークン=4文字)
            tokens = sum(len(m["content"]) for m in messages) // 4
            return response.model_copy(update={"input_tokens": tokens})

    long_print = "Thought: t.\n```python\nprint('" + "x" * 3000 + "')\n```"
    orchestrator = Orchestrator(
        provider=_Recording([long_print, long_print, long_print, long_print, long_print]),
        extractor=PythonCodeExtractor(), sandbox=MockSandbox(), profile=MBPPProfile(),
        limits=_limits(max_input_tokens=6000, max_iterations=10),
    )
    solution = orchestrator.run("1", "mbpp", "task")
    assert solution.total_input_tokens <= 6000
    assert solution.error == "MAX_INPUT_TOKENS"
    assert min(sent_sizes) < max(sent_sizes)   # 途中で窓を縮めて送った


def test_hardcoded_answers_are_detected() -> None:
    """可視テストの答えをreturnに直書きした提出を検出する(MBPP 268 実測)."""
    from src.agent.prompt.task_profile.mbpp_profile import hardcoded_answers
    tests = ["assert find_star_num(4) == 73", "assert find_star_num(5) == 121"]
    hardcoded = "def find_star_num(n):\n    if n == 4:\n        return 73\n    elif n == 5:\n        return 121\n"
    general = "def find_star_num(n):\n    return 6 * n * (n - 1) + 1\n"
    assert hardcoded_answers(hardcoded, tests) == ["73", "121"]
    assert hardcoded_answers(general, tests) == []


def test_trivial_return_values_are_not_hardcoding() -> None:
    """None / True / False のようなありふれた戻り値の一致は直書きとみなさない."""
    from src.agent.prompt.task_profile.mbpp_profile import hardcoded_answers
    tests = ["assert is_even(2) == True", "assert is_even(3) == False"]
    solution = "def is_even(n):\n    if n % 2 == 0:\n        return True\n    return False\n"
    assert hardcoded_answers(solution, tests) == []
