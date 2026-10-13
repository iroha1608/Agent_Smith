"""Thought -> Code -> Observationのループを制御するOrchestratorモジュール.

1タスク分のループ本体。
system(共通のテンプレート + TaskProfile) + user(問題分)から開始し、
毎反復で
LimitTrackerの先頭ガード -> LLM生成 -> <end_code>除去 -> コード抽出 -> sandbox実行
 -> Observation生成 -> 履歴 / StepMetric追加
 のループを回す。
 final_answer()で成功終了、ハード制限で打ち切り。
"""
from collections.abc import Callable

from src.agent.extractor.code_extractor import CodeExtractor
from src.agent.limits import LimitTracker
from src.agent.llm.provider.provider import LLMGenerator, LLMResponse, LLMError
from src.agent.extractor.python_extractor import was_salvaged
from src.agent.observation import build_observation, malformed_code_note, observation_no_code
from src.agent.llm.transport.transport import TransportError
from src.agent.prompt.task_profile.task_profile import TaskProfile
from src.agent.prompt.template import build_system_prompt
from src.config.settings import Limits
from src.models.output import SolutionOutput, StepMetrics
from src.models.sandbox_base import SandboxBase
from src.models.output import ExecuteResult

_DEFAULT_STOP_SEQUENCE = ["<end_code>"]


class Orchestrator:
    """1タスク分の Thought -> Code -> Observationのループを制御するOrchestratorクラス."""

    def __init__(
        self,
        provider: LLMGenerator,
        extractor: CodeExtractor,
        sandbox: SandboxBase,
        profile: TaskProfile,
        limits: Limits,
        stop_sequence: list[str] | None = None,
        solution_verifier: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        """依存部品を注入してOrchestratorを初期化する.

        Args:
            provider (LLMGenerator): 1回の生成を行うLLMクライアント(単体 / 粘り付き)。
            extractor (CodeExtractor): LLM出力から実行コードを取り出す抽出器。
            sandbox (SandboxBase): コードを実行するサンドボックス。
            profile (TaskProfile): タスク種別ごとのプロンプト差分。
            limits (Limits): 反復 / トークン / 時間のハード制限。
            stop_sequence (list[str] | None): 生成停止文字列。Noneなら既定の ["<end_code>"]。
            solution_verifier (Callable[[str], tuple[bool, str]] | None): 提出コードを受理前に
                検証する関数。(合否, 詳細)を返す。Noneなら構文検査のみで受理する。
        """
        self.provider = provider
        self.extractor = extractor
        self.sandbox = sandbox
        self.profile = profile
        self.limits = limits
        self.stop_sequences = stop_sequence if stop_sequence is not None else list(_DEFAULT_STOP_SEQUENCE)
        self.solution_verifier = solution_verifier

    def run(self, task_id: str, benchmark: str, task_message: str) -> SolutionOutput:
        """1タスク分のループを回しSolutionOutputを生成する.

        Args:
            task_id (str): タスク識別子(MBPPのtask_idまたはSWE-benchのinstance_id)
            benchmark (str): ベンチマークの種類: 'mbpp' または 'swebench'
            task_message (str): LLMに渡す問題分

        Returns:
            SolutionOutput:
                成功ならエージェントの最終的な解答とメトリクスを含むSolutionOutputオブジェクト。
                打ち切りなら errorに打ち切り理由を含むSolutionOutputオブジェクト。
        """
        # システムプロンプトとユーザープロンプトを組み合わせて、最初のメッセージを作成
        system_prompt = build_system_prompt(self.profile, self.sandbox)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_message}
        ]

        # ループ前の下準備
        tracker = LimitTracker(self.limits)
        steps: list[StepMetrics] = []
        best_solution: str = ""

        # ループ本体
        while True:
            # ループの先頭で制限をチェックし、打ち切りならSolutionOutputを返す
            stop_reason = tracker.start_iteration()
            if stop_reason is not None:
                return self._build_solution(
                    task_id=task_id,
                    benchmark=benchmark,
                    success=False,
                    solution=best_solution,
                    tracker=tracker,
                    steps=steps,
                    system_prompt=system_prompt,
                    error=stop_reason.value
                )

            # LLMにリクエストを送信し、レスポンスを取得する
            max_tokens = min(
                self.limits.max_output_tokens_per_iteration,
                tracker.remaining_output_tokens()
            )
            try:
                response: LLMResponse = self.provider.generate(
                    messages=self._window_messages(messages),
                    stop_sequences=self.stop_sequences,
                    max_tokens=max_tokens
                )
            # AllProvidersExhausted / BadRequestErrorをLLMErrorで一括で補足
            except (LLMError, TransportError) as error:
                failed_requests = getattr(error, "attempts", 1)
                return self._build_solution(
                    task_id=task_id,
                    benchmark=benchmark,
                    success=False,
                    solution=best_solution,
                    tracker=tracker,
                    steps=steps,
                    system_prompt=system_prompt,
                    error="LLM_UNAVAILABLE",
                    failed_requests=failed_requests
                )
            tracker.record_usage(
                response.input_tokens,
                response.output_tokens,
            )
            messages.append(
                {"role": "assistant", "content": response.text}
            )

            # レスポンスからコードを抽出する
            cleaned_code = response.text.replace("<end_code>", "")
            code = self.extractor.extract(cleaned_code)

            # コードが抽出できなかった場合は、Observationを生成して次のループに進む
            if code is None:
                observation = observation_no_code()
                messages.append(
                    {"role": "user", "content": observation}
                )
                steps.append(
                    self._build_step(len(steps) + 1, response, "", observation)
                )
                continue

            # コードが抽出できた場合は、サンドボックスで実行し、Observationを生成する
            result: ExecuteResult = self.sandbox.run(code)
            observation = build_observation(result)
            # 閉じフェンスが欠けたブロックを救済実行した場合は、その事実をLLMに明示する
            if was_salvaged(response.text):
                observation = f"{malformed_code_note()}\n{observation}"

            # final_answer()が呼ばれていれば、提出物を検査する。
            # 採点側は提出コードに隠しテストを連結して実行するため、
            # 構文エラーや余計な文が混ざった提出はそれだけで不合格になる。
            # 検査に落ちた提出は受理せず、理由をObservationに足してLLMに直させる。
            submission: str | None = None
            if result.final_answer_bool:
                candidate = self._sanitize_solution(result.final_answer or "")
                rejection = self._reject_reason(candidate)
                if rejection is None and self.solution_verifier is not None:
                    verified, detail = self.solution_verifier(candidate)
                    if not verified:
                        rejection = f"the submitted code did not pass the task tests: {detail}"
                if rejection is None:
                    submission = candidate
                else:
                    best_solution = candidate or best_solution
                    observation = (
                        f"{observation}\n--- submission rejected ---\n{rejection}\n"
                        "Fix the solution string and call final_answer again."
                    )
            elif result.error is None:
                # 提出前の作業コードは、打ち切り時のフォールバック候補として残す。
                best_solution = code

            messages.append(
                {"role": "user", "content": observation}
            )
            steps.append(
                self._build_step(len(steps) + 1, response, code, observation)
            )

            # 検査を通った提出があれば、成功としてSolutionOutputを返す
            if submission is not None:
                return self._build_solution(
                    task_id=task_id,
                    benchmark=benchmark,
                    success=True,
                    solution=submission,
                    tracker=tracker,
                    steps=steps,
                    system_prompt=system_prompt,
                    error=None
                )

    # 送信履歴の窓(直近N往復)と1メッセージの最大文字数。累積入力トークンを抑えるための切り詰め設定。
    _HISTORY_WINDOW = 2
    _OBS_MAX_CHARS = 800

    @staticmethod
    def _sanitize_solution(raw: str) -> str:
        """final_answer()に渡された文字列を、そのまま採点に出せる形へ整える.

        LLMはしばしばmarkdownのコードフェンス付きで提出する。採点側は提出文字列に
        テスト行を連結して実行するため、フェンスが残っていると構文エラーになる。

        Args:
            raw (str): final_answer()が捕捉した生の文字列。

        Returns:
            str: 前後の空白とコードフェンスを除去した提出コード。
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]  # ```python 等の開始フェンスを落とす
            while lines and lines[-1].strip().startswith("```"):
                lines.pop()
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _reject_reason(solution: str) -> str | None:
        """提出コードを受理してよいか検査し、だめな理由を返す.

        Args:
            solution (str): サニタイズ済みの提出コード。

        Returns:
            str | None: 受理可能ならNone。不可なら理由の文字列。
        """
        if not solution:
            return "final_answer() received an empty value"
        try:
            compile(solution, "<solution>", "exec")
        except SyntaxError as error:
            return f"the submitted code is not valid Python ({error.msg} on line {error.lineno})"
        if "def " not in solution:
            return "the submitted code contains no function definition"
        return None

    @staticmethod
    def _window_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """LLMへ毎回再送する履歴を切り詰めて返す(累積入力トークン対策).

        Chat Completions はステートレスで毎回全履歴を送るため、反復が増えると累積入力が
        ハード上限(max_input_tokens)を食い潰す。system と最初のタスク文は常に残し、
        それ以降は直近 _HISTORY_WINDOW 往復(assistant+user)だけを再送する。
        長い Observation(role=user)は _OBS_MAX_CHARS で切り詰める(assistantのコードは保持)。

        Args:
            messages (list[dict[str, str]]): [system, user(task), (assistant, user)...] の全履歴。

        Returns:
            list[dict[str, str]]: 切り詰めた送信用メッセージ列。
        """
        head = messages[:2]
        tail = messages[2:]
        if len(tail) > Orchestrator._HISTORY_WINDOW * 2:
            tail = tail[-Orchestrator._HISTORY_WINDOW * 2:]

        trimmed: list[dict[str, str]] = []
        for message in tail:
            content = message["content"]
            if message["role"] == "user" and len(content) > Orchestrator._OBS_MAX_CHARS:
                content = content[:Orchestrator._OBS_MAX_CHARS] + "\n...[truncated]"
            trimmed.append({"role": message["role"], "content": content})
        return head + trimmed

    def _build_step(
        self,
        step: int,
        response: LLMResponse,
        sandbox_input: str,
        sandbox_output: str
    ) -> StepMetrics:
        """1反復分のStepMetricsを構築するヘルパー関数.

        Args:
            step (int): 1から始まるイテレーション番号
            response (LLMResponse): LLMの生成結果
            sandbox_input (str): サンドボックスに送信したコード
            sandbox_output (str): LLMへ返すObservationの文字列

        Returns:
            StepMetrics: 1反復分のStepMetricsオブジェクト
        """
        return StepMetrics(
            step=step,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            request_time_ms=response.request_time_ms,
            api_url=response.api_url,
            model_name=response.model_name,
            llm_output=response.text,
            sandbox_input=sandbox_input,
            sandbox_output=sandbox_output,
            retries=response.retries
        )

    def _build_solution(
        self,
        task_id: str,
        benchmark: str,
        success: bool,
        solution: str,
        tracker: LimitTracker,
        steps: list[StepMetrics],
        system_prompt: str,
        error: str | None = None,
        failed_requests: int = 0
    ) -> SolutionOutput:
        """ループの結果からSolutionOutputを構築するヘルパー関数.(成功 / 打ち切り共通)

        Args:
            task_id (str): タスク識別子(MBPPのtask_idまたはSWE-benchのinstance_id)
            benchmark (str): ベンチマークの種類: 'mbpp' または 'swebench'
            success (bool): エージェントがタスクを解決したと判断したかどうか
            solution (str):
                MBPPの場合: Pythonの関数コード。SWE-benchの場合: gitパッチ（diff）
            tracker (LimitTracker): ループの制限を追跡するLimitTrackerオブジェクト
            steps (list[StepMetrics]): ループの各反復のStepMetricsのリスト
            system_prompt (str): LLMに送信された完全なシステムプロンプト（出所確認用）
            error (str | None): エージェントが失敗した場合のエラーメッセージ
                （成功した場合はNone）
            failed_requests (int): LLM呼び出しが失敗した場合の総試行回数（リトライを含む / 既定0）
        """
        return SolutionOutput(
            task_id=task_id,
            benchmark=benchmark,
            success=success,
            solution=solution,
            iterations=tracker.iterations,
            total_requests=sum(step.retries + 1 for step in steps) + failed_requests,
            total_input_tokens=tracker.total_input_tokens,
            total_output_tokens=tracker.total_output_tokens,
            total_time_seconds=tracker.elapsed_seconds(),
            steps=steps,
            system_prompt=system_prompt,
            error=error
        )
