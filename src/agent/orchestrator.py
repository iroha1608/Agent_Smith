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
from src.agent.limits import LimitTracker, StopReason
from src.agent.llm.provider.provider import LLMGenerator, LLMResponse, LLMError
from src.agent.extractor.python_extractor import salvage_kind, strip_stop_sequences
from src.agent.observation import (
    budget_note,
    build_observation,
    iteration_note,
    malformed_code_note,
    observation_no_code,
    repeated_code_note,
)
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
        started_at: float | None = None,
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
            started_at (float | None): 制限時間の起点(time.monotonic()の値)。Noneならrun()の開始時。
        """
        self.provider = provider
        self.extractor = extractor
        self.sandbox = sandbox
        self.profile = profile
        self.limits = limits
        self.stop_sequences = stop_sequence if stop_sequence is not None else list(_DEFAULT_STOP_SEQUENCE)
        self.solution_verifier = solution_verifier
        # 入力トークン見積もりの係数。実測に合わせて安全側(小さい側)へだけ補正する。
        self._chars_per_token = self._CHARS_PER_TOKEN
        self.started_at = started_at

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
        tracker = LimitTracker(self.limits, start_time=self.started_at)
        steps: list[StepMetrics] = []
        best_solution: str = ""
        # 実行したコード(空白を正規化) -> 最初に実行したstep番号。同じコードの再実行を指摘する。
        executed_codes: dict[str, int] = {}
        # 提出に至らなかった反復の連続数。停滞したら別プロバイダのモデルへ切り替える。
        misses_in_a_row = 0

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

            # 送る履歴を決め、累積入力トークンの上限を踏み越えないか事前に確かめる。
            # 判定に使うのは「これまでの実績(プロバイダ報告値の累計) + 次の1回の見積もり」。
            # 次の1回の入力トークン数は送る前には分からないため文字数から見積もる。
            # 踏み越えると即不合格なので、窓を段階的に縮め、それでも収まらないなら
            # リクエストせずに打ち切る。
            windowed = self._fit_window(messages, tracker)
            if windowed is None:
                return self._build_solution(
                    task_id=task_id,
                    benchmark=benchmark,
                    success=False,
                    solution=best_solution,
                    tracker=tracker,
                    steps=steps,
                    system_prompt=system_prompt,
                    error=StopReason.MAX_INPUT_TOKENS.value
                )

            # LLMにリクエストを送信し、レスポンスを取得する
            max_tokens = min(
                self.limits.max_output_tokens_per_iteration,
                tracker.remaining_output_tokens()
            )
            try:
                response: LLMResponse = self.provider.generate(
                    messages=windowed,
                    stop_sequences=self.stop_sequences,
                    max_tokens=max_tokens
                )
            # AllProvidersExhausted / BadRequestErrorをLLMErrorで一括で補足
            except (LLMError, TransportError) as error:
                failed_requests = getattr(error, "attempts", 1)
                # LLMの応答が無い反復はstepとして記録しない(llm_outputを捏造しないため)。
                # 反復にも数えず、失敗した試行は total_requests に、プロバイダ側で消費された
                # トークン(空応答など)は合計に計上する(制限の判定を甘くしないため)。
                tracker.iterations -= 1
                tracker.record_usage(
                    int(getattr(error, "input_tokens", 0) or 0),
                    int(getattr(error, "output_tokens", 0) or 0),
                )
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
            self._learn_chars_per_token(windowed, response)
            messages.append(
                {"role": "assistant", "content": response.text}
            )

            # レスポンスからコードを抽出する(停止シーケンスの残骸を取り除いてから)
            cleaned_code = strip_stop_sequences(response.text, self.stop_sequences)
            code = self.extractor.extract(cleaned_code)

            # コードが抽出できなかった場合は、Observationを生成して次のループに進む。
            # 1反復の上限ぶんを使い切っているなら「散文で考え込んで切られた」ことが原因なので、
            # それを名指しして伝える(黙って再促しすると同じ失敗を繰り返して予算が尽きる)。
            if code is None:
                cut_off_at = response.output_tokens if response.output_tokens >= max_tokens else None
                observation = self._with_budget_note(observation_no_code(cut_off_at), tracker)
                messages.append(
                    {"role": "user", "content": observation}
                )
                # 何も実行していないので sandbox_input / sandbox_output は空(フィードバックは次の
                # userメッセージとしてLLMに届く)。採点側の display は「入力が空なのに出力がある」
                # stepを不自然とみなすため、実行結果が無いことをそのまま記録する。
                steps.append(
                    self._build_step(len(steps) + 1, response, "", "")
                )
                misses_in_a_row = self._count_miss(misses_in_a_row, submitted=False)
                continue

            # コードが抽出できた場合は、サンドボックスで実行し、Observationを生成する
            result: ExecuteResult = self.sandbox.run(code)
            observation = build_observation(result)
            # 以前と全く同じコードなら、そのことを明示する(同じ誤答の再提出で反復を溶かすのを防ぐ)
            fingerprint = " ".join(code.split())
            if fingerprint in executed_codes:
                observation = f"{repeated_code_note(executed_codes[fingerprint])}\n{observation}"
            else:
                executed_codes[fingerprint] = len(steps) + 1
            # 閉じフェンスが欠けたブロックを救済実行した場合は、その事実をLLMに明示する
            salvaged = salvage_kind(cleaned_code)
            if salvaged is not None:
                observation = f"{malformed_code_note(salvaged)}\n{observation}"

            # final_answer()が呼ばれていれば、提出物を検査する。
            # 採点側は提出コードに隠しテストを連結して実行するため、
            # 構文エラーや余計な文が混ざった提出はそれだけで不合格になる。
            # 検査に落ちた提出は受理せず、理由をObservationに足してLLMに直させる。
            submission: str | None = None
            if result.final_answer_bool:
                candidate = self.profile.sanitize_submission(result.final_answer or "")
                rejection = self.profile.submission_rejection(candidate)
                if rejection is None and self.solution_verifier is not None:
                    verified, detail = self.solution_verifier(candidate)
                    if not verified:
                        rejection = f"the submission did not pass the task's checks: {detail}"
                if rejection is None:
                    submission = candidate
                else:
                    best_solution = candidate or best_solution
                    observation = (
                        f"{observation}\n--- submission rejected ---\n{rejection}\n"
                        "Fix it and call final_answer again."
                    )
            elif result.error is None and self.profile.is_fallback_candidate(code):
                # 打ち切り時のフォールバック候補。提出可能な形のコードだけを残す
                # (MBPPなら構文が通り関数定義を含むもの。printだけの断片は確実に不合格になる)。
                best_solution = code

            observation = self._with_budget_note(observation, tracker)
            messages.append(
                {"role": "user", "content": observation}
            )
            steps.append(
                self._build_step(len(steps) + 1, response, code, observation)
            )

            # 提出に至らない反復が続いたら、次から別プロバイダのモデルを優先させる
            misses_in_a_row = self._count_miss(misses_in_a_row, submitted=submission is not None)

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

    def _count_miss(self, misses_in_a_row: int, submitted: bool) -> int:
        """提出に至らなかった反復の連続数を更新し、閾値に達したらモデルを切り替える.

        閾値はプロファイルが決める(MBPPは2回。SWE-benchは探索で提出しない反復が続くのが
        正常なので切り替えない)。

        Args:
            misses_in_a_row (int): これまでの連続数。
            submitted (bool): この反復で検査を通った提出があったか。

        Returns:
            int: 更新後の連続数(切り替えたら0に戻す)。
        """
        if submitted:
            return 0
        misses_in_a_row += 1
        threshold = self.profile.rotate_after_misses
        if threshold is not None and misses_in_a_row >= threshold:
            self._rotate_provider()
            return 0
        return misses_in_a_row

    def _rotate_provider(self) -> None:
        """LLMクライアントが切り替えに対応していれば、別プロバイダのモデルを優先させる.

        単一プロバイダのLLMProviderやテスト用のスタブは切り替えを持たないので何もしない。
        """
        rotate = getattr(self.provider, "rotate", None)
        if callable(rotate):
            rotate()

    # トークン見積もりの係数(1トークンあたりの文字数)の初期値。事前ガードで使うため安全側に
    # 小さめに取る。実測(2026-10-15・過去runの全ステップを再構成して比較): 中央値は3.7〜4.1だが
    # qwenは最小3.03・command-aは3.49。旧値3.6では見積もりが甘く、geminiで6309/6000を踏み越えた。
    _CHARS_PER_TOKEN = 3.0
    # 実測の比から係数を下げるときの安全率(実測値よりさらに5%小さく見る)
    _CHARS_PER_TOKEN_SAFETY = 0.95
    # 残り出力トークンが「1反復ぶん × この倍率」以下になったら提出を促す。
    # 1(=残り1反復ぶん)にしているのは、本番設定(総1500 / 1反復600)で2にすると
    # 1反復目の直後(残り1200)から警告が出てしまい、3〜5反復かけて正解に至るタスクを
    # 早期提出へ追い込むため。「次が最後の1手」になった時点でだけ促す。
    _BUDGET_WARN_FACTOR = 1

    def _with_budget_note(self, observation: str, tracker: LimitTracker) -> str:
        """残り出力トークンが乏しければ、提出を促す一文をObservationの末尾に足す.

        出力予算切れでの打ち切りは不合格が確定する。残り1反復ぶんになった時点で
        「最善を提出せよ」と明示した方が、黙って探索を続けさせるより期待値が高い。

        Args:
            observation (str): 組み立て済みのObservation文字列。
            tracker (LimitTracker): 現在の消費状況を持つトラッカー。

        Returns:
            str: 必要なら警告文を足したObservation文字列。
        """
        remaining = tracker.remaining_output_tokens()
        if 0 < remaining <= self.limits.max_output_tokens_per_iteration * self._BUDGET_WARN_FACTOR:
            # (0以下なら次の反復は制限チェックで打ち切られるため、促しても送られない)
            return f"{observation}\n{budget_note(remaining)}"
        # 反復回数の残りが少なくなったら、提出の段取りに入るよう促す。実測(nemotron-3-ultra,
        # SWE 13480): 3反復目で正しい修正を入れたのに、確認の run_command を27回繰り返して
        # run_tests も提出もしないまま上限に達した。
        iterations_left = self.limits.max_iterations - tracker.iterations
        if 0 < iterations_left <= self.profile.iteration_warn_at:
            return f"{observation}\n{iteration_note(iterations_left, self.profile.submit_hint())}"
        return observation

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """これから送るメッセージ列の入力トークン数を概算する.

        ハード制限(累積入力トークン)を踏み越えないための事前ガードに使う。
        踏み越えるとその時点でタスクが不合格になるため、安全側に多めへ見積もる。

        Args:
            messages (list[dict[str, str]]): 送信予定のメッセージ列。

        Returns:
            int: 概算の入力トークン数。
        """
        characters = sum(len(message["content"]) for message in messages)
        # メッセージごとのrole等のオーバーヘッドを4トークンずつ加算する
        return int(characters / self._chars_per_token) + 1 + 4 * len(messages)

    def _learn_chars_per_token(self, sent: list[dict[str, str]], response: LLMResponse) -> None:
        """実際に送った文字数と報告された入力トークン数から、見積もり係数を安全側へ補正する.

        トークナイザはモデルごとに違い、固定係数では甘くなることがある。実測の比が
        今の係数より小さければ(=1トークンが短い)、係数をそこまで下げる。上げはしない。
        失敗試行のトークンが足されたレスポンス(retries>0)は比が歪むので使わない。

        Args:
            sent (list[dict[str, str]]): このリクエストで送ったメッセージ列。
            response (LLMResponse): そのレスポンス。
        """
        if response.retries or response.input_tokens <= 0:
            return
        characters = sum(len(message["content"]) for message in sent)
        observed = characters / response.input_tokens * self._CHARS_PER_TOKEN_SAFETY
        if 0 < observed < self._chars_per_token:
            self._chars_per_token = observed

    def _fit_window(
        self, messages: list[dict[str, str]], tracker: LimitTracker
    ) -> list[dict[str, str]] | None:
        """累積入力トークンの上限に収まる送信履歴を返す。どう縮めても収まらなければNone.

        直近 profile.history_window 往復 -> ... -> 1往復 -> system+タスク文だけ、の順に縮める。
        いきなり履歴を全部捨てると、直前の失敗とその原因(Observation)をLLMが見られず
        同じ失敗を繰り返すため、まず1往復だけ残す段階を挟む。

        Args:
            messages (list[dict[str, str]]): 全履歴。
            tracker (LimitTracker): 現在の消費状況。

        Returns:
            list[dict[str, str]] | None: 送信用メッセージ列。収まらなければNone。
        """
        budget = self.limits.max_input_tokens - tracker.total_input_tokens
        for window in range(self.profile.history_window, -1, -1):
            candidate = self._window_messages(messages, window)
            if self._estimate_tokens(candidate) <= budget:
                return candidate
        return None

    def _window_messages(self, messages: list[dict[str, str]], window: int | None = None) -> list[dict[str, str]]:
        """LLMへ毎回再送する履歴を切り詰めて返す(累積入力トークン対策).

        Chat Completions はステートレスで毎回全履歴を送るため、反復が増えると累積入力が
        ハード上限(max_input_tokens)を食い潰す。system と最初のタスク文は常に残し、
        それ以降は直近 window 往復(assistant+user)だけを再送する。
        長い Observation(role=user)と再送するassistant発話は、プロファイルの上限文字数で
        切り詰める(自分の書いたコード全文の再送が最大の膨張要因)。上限はタスク種別で違う
        (MBPPは累積6000トークンと厳しく、SWE-benchはファイルを読むので長いObservationが要る)。

        Args:
            messages (list[dict[str, str]]): [system, user(task), (assistant, user)...] の全履歴。
            window (int | None): 残す往復数。Noneならプロファイルの既定。0ならsystemとタスク文だけ。

        Returns:
            list[dict[str, str]]: 切り詰めた送信用メッセージ列。
        """
        keep = self.profile.history_window if window is None else window
        head = messages[:2]
        tail = messages[2:]
        tail = tail[-keep * 2:] if keep > 0 else []

        limits = {"user": self.profile.observation_max_chars, "assistant": self.profile.assistant_max_chars}
        trimmed: list[dict[str, str]] = []
        for index, message in enumerate(tail):
            content = message["content"]
            budget = limits.get(message["role"])
            # 直前の反復の結果(LLMがこれから読むObservation)は、過去分より長く残す
            if index == len(tail) - 1 and message["role"] == "user":
                budget = max(budget or 0, self.profile.latest_observation_max_chars)
            if budget is not None:
                content = Orchestrator._truncate_middle(content, budget)
            trimmed.append({"role": message["role"], "content": content})
        return head + trimmed

    @staticmethod
    def _truncate_middle(content: str, budget: int) -> str:
        """中央を省き、前後の両端を残す形で切り詰める.

        Observationは先頭にstatus/stdout、末尾に `--- note ---` の助言が来る。
        先頭だけ残す切り詰めだと、最も行動を促す助言(切り詰め警告・予算警告・提出の却下理由)が
        必ず落ちてしまい、LLMに伝わらない。

        Args:
            content (str): 切り詰め対象の本文。
            budget (int): 残してよい最大文字数。

        Returns:
            str: budget以内に収めた本文(超過時は中央を省略マーカーに置換)。
        """
        if len(content) <= budget:
            return content
        head_size = budget * 2 // 3
        tail_size = budget - head_size
        return f"{content[:head_size]}\n...[truncated]...\n{content[-tail_size:]}"

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
        if not success and not solution:
            # 提出が無いまま打ち切られた。環境から回収できる最善の提出物があれば出す
            # (SWE-benchなら編集済みのパッチ。空提出は確実に不合格なので、出さない理由が無い)。
            try:
                solution = self.profile.fallback_solution(self.sandbox)
            except Exception:  # noqa: BLE001 - 回収の失敗で結果の書き出しまで失わない
                solution = ""
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
