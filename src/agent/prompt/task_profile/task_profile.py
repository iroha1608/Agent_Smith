"""TaskProfileのベースクラスを定義するモジュール.

タスク種別ごとのプロンプト差分の注入口。
共通テンプレートに差し込む、タスク種別で変わる要素を抽象メソッドとして持つ。
新しいタスク種別を足すときは、このクラスを継承したプロファイルを1つ実装する。

プロンプト以外にも、タスク種別で変わる「提出物の扱い」と「履歴の窓幅」をここに置く。
Orchestrator(ループ本体)はタスク種別を知らずに済み、MBPP(関数のソース)と
SWE-bench(gitパッチ)の違いはプロファイルの差し替えだけで表現できる。
"""
from abc import ABC, abstractmethod

from src.models.sandbox_base import SandboxBase

# 全タスク共通の出力形式のルール(システムプロンプトの ## Format 節)
DEFAULT_FORMAT_RULES = """- One brief Thought line (20 words max), then exactly ONE block fenced as ```python ... ```,
  then <end_code>.
- NEVER derive formulas or trace arithmetic in prose. Your turn is cut off at a hard output
  limit, and a reply that never closes a ```python block runs nothing and burns the budget.
  Write the candidate in code and let the Observation correct you.
- Only what you print() (plus errors and tracebacks) comes back in the Observation.
- The tools and final_answer are already defined globals: call them directly, never import them.
- The namespace persists between turns, but your FINAL answer must be SELF-CONTAINED.
- Limits on iterations, tokens and time are strict: keep turns short, never print large output."""


class TaskProfile(ABC):
    """タスク種別ごとのプロンプト差分と提出物の扱いを提供する抽象基底クラス.

    具象クラスがプロンプトの4メソッドを実装する。提出物の検査・整形、打ち切り時の
    フォールバック、履歴の窓幅は既定の実装があり、必要なプロファイルだけが上書きする。

    Attributes:
        history_window (int): LLMへ再送する直近の往復数。
        observation_max_chars (int): 再送する過去のObservation(role=user)1件の最大文字数。
        latest_observation_max_chars (int): 直前の反復のObservation(LLMがこれから読む結果)の最大文字数。
            過去分より長く残す(run_testsの失敗詳細や読んだファイルの中身を途中で切らないため)。
        assistant_max_chars (int): 再送するassistant発話1件の最大文字数。
        rotate_after_misses (int | None): 提出に至らない反復がこの回数続いたら別プロバイダの
            モデルへ切り替える。Noneなら切り替えない(探索が本分で、提出しない反復が続くのが
            正常なタスク向け)。
    """

    history_window: int = 2
    observation_max_chars: int = 800
    latest_observation_max_chars: int = 1500
    assistant_max_chars: int = 1200
    rotate_after_misses: int | None = 2

    def format_rules(self) -> str:
        """システムプロンプトの出力形式のルールを返す.

        Returns:
            str: ## Format 節に入れるルール
        """
        return DEFAULT_FORMAT_RULES

    def sanitize_submission(self, raw: str) -> str:
        """final_answer()に渡された値を、採点に出せる形へ整える.

        Args:
            raw (str): final_answer()が捕捉した生の文字列。

        Returns:
            str: 整えた提出物。
        """
        return raw.strip()

    def submission_rejection(self, candidate: str) -> str | None:
        """提出物を受理してよいか検査し、だめならLLMへ返す理由を返す.

        Args:
            candidate (str): sanitize_submission()済みの提出物。

        Returns:
            str | None: 受理できるならNone。だめなら理由。
        """
        return None if candidate else "final_answer() received an empty value"

    def is_fallback_candidate(self, code: str) -> bool:
        """エラー無く実行できたコードを、打ち切り時に提出する候補として残すかを返す.

        Args:
            code (str): その反復で実行したコード。

        Returns:
            bool: 候補として残すならTrue。
        """
        return False

    def fallback_solution(self, sandbox: SandboxBase) -> str:
        """提出が無いまま打ち切られたとき、環境から回収できる最善の提出物を返す.

        Args:
            sandbox (SandboxBase): 接続中のサンドボックス。

        Returns:
            str: 回収した提出物。無ければ空文字。
        """
        return ""

    @abstractmethod
    def task_instructions(self) -> str:
        """このタスクの進め方を返す.

        具体的な問題分は、orchestratorがuserメッセージとして別途与えるため含めない。
        MBPP -> 「関数を書き、printでテストを出力し、final_answer(関数のソース)で提出」。

        Returns:
            str: タスクの指示
        """

    @abstractmethod
    def final_answer_specification(self) -> str:
        """final_answer()に何を渡すべきか、最終的な回答の仕方を返す.

        Returns:
            str: 最終的な回答の仕様
        """

    @abstractmethod
    def few_shot_example(self) -> str:
        """出力形式を教えるためのFew-shotの例を返す.

        Thought -> Code -> Observation -> final_answer
        実タスクとは無関係な例。形式を示すためだけの固定のテキスト。

        Returns:
            str: Few-shotの例
        """

    @abstractmethod
    def manual(self, sandbox: SandboxBase) -> str:
        """接続中MCPサーバーのツールスキーマから動的生成した、LLM向けツールマニュアルを返す.

        このタスクで使えるツール / 環境の説明を返す。
        ツールはハードコードせずに、sandbox.manual()から動的に取得する。

        Args:
            sandbox (SandboxBase): サンドボックスのインスタンス

        Returns:
            str: LLM向けツールマニュアル
        """
