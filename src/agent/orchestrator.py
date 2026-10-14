"""Thought -> Code -> Observationのループを制御するOrchestratorモジュール.

1タスク分のループ本体。
system(共通のテンプレート + TaskProfile) + user(問題分)から開始し、
毎反復で
LimitTrackerの先頭ガード -> LLM生成 -> <end_code>除去 -> コード抽出 -> sandbox実行
 -> Observation生成 -> 履歴 / StepMetric追加
 のループを回す。
 final_answer()で成功終了、ハード制限で打ち切り。
"""
import ast
from collections.abc import Callable

from src.agent.extractor.code_extractor import CodeExtractor
from src.agent.limits import LimitTracker, StopReason
from src.agent.llm.provider.provider import LLMGenerator, LLMResponse, LLMError
from src.agent.extractor.python_extractor import salvage_kind, strip_stop_sequences
from src.agent.observation import (
    budget_note,
    build_observation,
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
        # 入力トークン見積もりの係数。実測に合わせて安全側(小さい側)へだけ補正する。
        self._chars_per_token = self._CHARS_PER_TOKEN

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
                # 失敗した試行でもプロバイダ側で消費したトークンは計上し、
                # 反復数とstepsの数が食い違わないよう、この反復の記録も残す。
                wasted_input = int(getattr(error, "input_tokens", 0) or 0)
                wasted_output = int(getattr(error, "output_tokens", 0) or 0)
                tracker.record_usage(wasted_input, wasted_output)
                if failed_requests <= 0:
                    # 残り時間が無く1回も送っていない。LLMを呼んでいないので反復に数えない。
                    tracker.iterations -= 1
                else:
                    steps.append(StepMetrics(
                        step=len(steps) + 1,
                        input_tokens=wasted_input,
                        output_tokens=wasted_output,
                        request_time_ms=0.0,
                        sandbox_output=f"[LLM_UNAVAILABLE] {error}"[:500],
                        retries=failed_requests - 1,
                    ))
                return self._build_solution(
                    task_id=task_id,
                    benchmark=benchmark,
                    success=False,
                    solution=best_solution,
                    tracker=tracker,
                    steps=steps,
                    system_prompt=system_prompt,
                    error="LLM_UNAVAILABLE"
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
                steps.append(
                    self._build_step(len(steps) + 1, response, "", observation)
                )
                misses_in_a_row += 1
                if misses_in_a_row >= self._ROTATE_AFTER_MISSES:
                    self._rotate_provider()
                    misses_in_a_row = 0
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
            elif result.error is None and self._reject_reason(code) is None:
                # 打ち切り時のフォールバック候補。提出可能な形(構文が通り関数定義を含む)の
                # コードだけを残す。デバッグ用のprintだけの断片を出すと確実に不合格になる。
                best_solution = code

            observation = self._with_budget_note(observation, tracker)
            messages.append(
                {"role": "user", "content": observation}
            )
            steps.append(
                self._build_step(len(steps) + 1, response, code, observation)
            )

            # 提出に至らない反復が続いたら、次から別プロバイダのモデルを優先させる
            misses_in_a_row = 0 if submission is not None else misses_in_a_row + 1
            if misses_in_a_row >= self._ROTATE_AFTER_MISSES:
                self._rotate_provider()
                misses_in_a_row = 0

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

    # 提出に至らない反復がこの回数続いたら、別プロバイダのモデルへ切り替える
    _ROTATE_AFTER_MISSES = 2

    def _rotate_provider(self) -> None:
        """LLMクライアントが切り替えに対応していれば、別プロバイダのモデルを優先させる.

        単一プロバイダのLLMProviderやテスト用のスタブは切り替えを持たないので何もしない。
        """
        rotate = getattr(self.provider, "rotate", None)
        if callable(rotate):
            rotate()

    # 送信履歴の窓(直近N往復)と1メッセージの最大文字数。累積入力トークンを抑えるための切り詰め設定。
    _HISTORY_WINDOW = 2
    _OBS_MAX_CHARS = 800
    # 履歴として再送するassistant発話の上限。自分が書いたコード全文を毎回送り直すと
    # 反復2回目以降の入力が一気に膨らみ、累積入力の上限(6000)を食い潰す。
    _ASSISTANT_MAX_CHARS = 1200
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
        if remaining > self.limits.max_output_tokens_per_iteration * self._BUDGET_WARN_FACTOR:
            return observation
        if remaining <= 0:
            # 次の反復は制限チェックで打ち切られるため、促しても送られない。
            return observation
        return f"{observation}\n{budget_note(remaining)}"

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

        直近 _HISTORY_WINDOW 往復 -> 1往復 -> system+タスク文だけ、の順に縮める。
        いきなり履歴を全部捨てると、直前の失敗とその原因(Observation)をLLMが見られず
        同じ失敗を繰り返すため、まず1往復だけ残す段階を挟む。

        Args:
            messages (list[dict[str, str]]): 全履歴。
            tracker (LimitTracker): 現在の消費状況。

        Returns:
            list[dict[str, str]] | None: 送信用メッセージ列。収まらなければNone。
        """
        budget = self.limits.max_input_tokens - tracker.total_input_tokens
        for window in range(self._HISTORY_WINDOW, -1, -1):
            candidate = self._window_messages(messages, window)
            if self._estimate_tokens(candidate) <= budget:
                return candidate
        return None

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

    # 採点環境に存在しないサンドボックス専用の名前。これらを参照するコードを提出すると
    # NameErrorで全テストが落ちる。実測(MBPPタスク160)では、打ち切り時のフォールバックが
    # `report = json.loads(run_tests(code="""..."""))` という検証用ラッパーをそのまま提出していた。
    _SANDBOX_ONLY_NAMES = frozenset({"run_tests", "final_answer"})
    # トップレベルに置いても採点時に例外を投げない文。採点側は提出コードにテスト行を
    # 連結してファイル全体を実行するため、これ以外の文(デモ呼び出し/assert/print等)は
    # 1つでも例外を投げると提出全体が不合格になる。
    _ALLOWED_TOP_LEVEL = (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign,
    )

    @classmethod
    def _reject_reason(cls, solution: str) -> str | None:
        """提出コードを受理してよいか検査し、だめな理由を返す.

        採点側は提出コードに隠しテストを連結してファイル全体を実行し、exit 0のみを合格とする。
        そのため「構文が通る」だけでは足りず、(1)関数定義がある (2)トップレベルに例外を
        投げ得る文が無い (3)サンドボックス専用の名前を参照していない の3点を確かめる。

        Args:
            solution (str): サニタイズ済みの提出コード。

        Returns:
            str | None: 受理可能ならNone。不可なら理由の文字列。
        """
        if not solution:
            return "final_answer() received an empty value"
        try:
            tree = ast.parse(solution)
        except SyntaxError as error:
            return f"the submitted code is not valid Python ({error.msg} on line {error.lineno})"

        # サンドボックス専用名の検査を先に行う。検証ラッパーをそのまま提出した場合、
        # 関数定義が文字列リテラルの中にあるため「関数定義なし」が先に当たってしまい、
        # 本当の原因(ラッパーごと提出した)が伝わらない。
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        blocked = sorted(referenced & cls._SANDBOX_ONLY_NAMES)
        if blocked:
            return (
                f"the submitted code calls {', '.join(blocked)}, which exist only inside this "
                "sandbox. Submit just the function source, with no test harness around it"
            )

        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
            return "the submitted code contains no function definition"

        for node in tree.body:
            if isinstance(node, cls._ALLOWED_TOP_LEVEL):
                continue
            # モジュールdocstringだけは例外を投げないので許す
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                continue
            return (
                f"the submitted code has a top-level {type(node).__name__} statement. Submit only "
                "imports and the function definition(s): the grader appends its own tests to your "
                "code and runs the whole file, so any extra statement that raises fails it"
            )
        return None

    @staticmethod
    def _window_messages(messages: list[dict[str, str]], window: int | None = None) -> list[dict[str, str]]:
        """LLMへ毎回再送する履歴を切り詰めて返す(累積入力トークン対策).

        Chat Completions はステートレスで毎回全履歴を送るため、反復が増えると累積入力が
        ハード上限(max_input_tokens)を食い潰す。system と最初のタスク文は常に残し、
        それ以降は直近 _HISTORY_WINDOW 往復(assistant+user)だけを再送する。
        長い Observation(role=user)は _OBS_MAX_CHARS、再送するassistant発話は
        _ASSISTANT_MAX_CHARS で切り詰める(自分の書いたコード全文の再送が最大の膨張要因)。

        Args:
            messages (list[dict[str, str]]): [system, user(task), (assistant, user)...] の全履歴。
            window (int | None): 残す往復数。Noneなら _HISTORY_WINDOW。0ならsystemとタスク文だけ。

        Returns:
            list[dict[str, str]]: 切り詰めた送信用メッセージ列。
        """
        keep = Orchestrator._HISTORY_WINDOW if window is None else window
        head = messages[:2]
        tail = messages[2:]
        tail = tail[-keep * 2:] if keep > 0 else []

        limits = {"user": Orchestrator._OBS_MAX_CHARS, "assistant": Orchestrator._ASSISTANT_MAX_CHARS}
        trimmed: list[dict[str, str]] = []
        for message in tail:
            content = message["content"]
            budget = limits.get(message["role"])
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
