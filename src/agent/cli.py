"""MBPP エージェントの組み立て、実行を担う配線.

タスクファイル読み込み -> 設定 / プロバイダ解決 ->
Provider / Sandbox / Orchestrator 構築 ->
実行 -> SolutionOutput 書き出し
fireはagent_mbpp.py / agent_swebench.py から呼ばれ、
検証済みの引数だけを受け取る関数郡

- [ ] BaseAgentCliArgsモデルを作成して検証を拡充
"""
import logging
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from dotenv import load_dotenv

from src.agent.extractor.composite_extractor import CompositeCodeExtractor
from src.agent.llm.resilient_client import ResilientLLMClient
# from src.agent.llm.key_manager import resolve_api_key
# from src.agent.llm.provider.openai_compat import OpenAICompatProvider
from src.agent.llm.transport.requests_transport import RequestsTransport
from src.agent.orchestrator import Orchestrator
from src.agent.prompt.task_profile.mbpp_profile import MBPPProfile, hardcoded_answers
from src.config.settings import load_providers, load_settings, order_providers
from src.models.input import MBPPTaskInput
from src.models.output import SolutionOutput
from src.sandbox.sandbox_main import Sandbox
from src.models.sandbox_config import SandboxConfig
# from src.sandbox.mock import MockSandbox

_LOGGER = logging.getLogger("agent.cli")
# リポジトリ直下(このファイルは src/agent/cli.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def format_mbpp_task(task_input: MBPPTaskInput) -> str:
    """MBPPTaskInputのタスク入力をLLMに渡すuser messages文字列に整形する.

    Args:
        task_input (MBPPTaskInput): タスクファイルから読み込んだMBPPTaskInputオブジェクト

    Returns:
        str: LLMに渡す問題分、関数シグネチャ、テストコードを含む文字列
    """
    lines = [
        f"Problem:\n{task_input.task_definition}\n",
        f"Function signature:\n{task_input.function_definition}\n",
    ]
    if task_input.test_imports:
        # 採点側は「解答 + test_imports + テスト」の順に連結して実行する
        lines.append("\nThe tests run after these imports:\n")
        lines.extend(task_input.test_imports)
    if task_input.test_list:
        lines.append("\nYour solution must pass these tests:\n")
        lines.extend(task_input.test_list)
    return "\n".join(lines)


def make_mbpp_verifier(
    sandbox: Sandbox,
    test_list: list[str]
) -> Callable[[str], tuple[bool, str]]:
    """提出コードを受理前に可視テストで検証する関数を作る.

    採点側は提出コードに隠しテストを連結して実行するため、構文エラーや
    テスト不合格のコードを提出するとその時点で不合格になる。提出前に
    MCPツール run_tests で同じ形の検証を行い、落ちた提出は受理させない。

    Args:
        sandbox (Sandbox): run_tests ツールを公開しているサンドボックス。
        test_list (list[str]): タスクが与えた assert 文のリスト。

    Returns:
        Callable[[str], tuple[bool, str]]: (合否, 詳細)を返す検証関数。
    """
    def verify(solution: str) -> tuple[bool, str]:
        """提出コードを run_tests に通し、(合否, 詳細)を返す."""
        if not test_list:
            return True, ""
        # 可視テストの答えを直書きした提出は、可視テストを通っても隠しテストで落ちる
        hardcoded = hardcoded_answers(solution, test_list)
        if hardcoded:
            return False, (
                f"it returns the visible tests' expected values as literals ({', '.join(hardcoded)}). "
                "The grader calls it with other inputs: implement the general rule instead of "
                "special-casing the examples"
            )
        snippet = (
            "import json\n"
            f"_report = json.loads(run_tests(code={solution!r}, test_list={list(test_list)!r}))\n"
            "print('VERIFY_OK' if _report['success'] else "
            "'VERIFY_FAIL ' + str(_report.get('output', ''))[:300])\n"
        )
        stdout = (sandbox.run(snippet).stdout or "").strip()
        if "VERIFY_FAIL" in stdout:
            return False, stdout.split("VERIFY_FAIL", 1)[1].strip()[:300]
        # 検証そのものが動かなかった場合は受理側に倒す(正答を取り逃がさない)
        return True, ""

    return verify


def run_mbpp_agent(
    task_file_path: str,
    output_file_path: str,
    model_name: str | None = None,
    provider_url: str | None = None,
    providers_file_path: str = "providers/providers.json"
) -> None:
    """MBPPタスクを1問解くエージェントを実行し、SolutionOutputを必ず書き出す.

    課題 IV.1「評価中のクラッシュは失格」のため、想定外の例外が起きても
    success=False と原因を入れた solution.json を書いて正常終了する
    (KeyboardInterrupt / SystemExit は握りつぶさずに伝える)。

    Args:
        task_file_path (str): MBPPタスクファイルのパス
        output_file_path (str): SolutionOutputを書き出すJSONファイルのパス
        model_name (str | None): 使用するLLMモデル名。Noneなら設定ファイルのデフォルトを使用
        provider_url (str | None): 使用するLLMプロバイダのURL。Noneなら設定ファイルのデフォルトを使用
        providers_file_path (str): プロバイダ設定ファイルのパス
    """
    # 採点側の制限時間はプロセス起動から数えるので、できるだけ早く起点を取る
    started_at = time.monotonic()
    task_id = "unknown"
    try:
        task_input = MBPPTaskInput.model_validate_json(
            Path(task_file_path).read_text(encoding="utf-8")
        )
        task_id = str(task_input.task_id)
        solution_output = _solve_mbpp(task_input, started_at, model_name, provider_url, providers_file_path)
    except Exception as error:  # noqa: BLE001 - 評価中は落ちずに失敗を記録して返す
        _LOGGER.exception("agent crashed; writing a failed solution.json")
        solution_output = SolutionOutput(
            task_id=task_id,
            benchmark="mbpp",
            success=False,
            solution="",
            iterations=0,
            total_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_time_seconds=time.monotonic() - started_at,
            error=f"AGENT_CRASH: {type(error).__name__}: {error}"[:500],
        )
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(solution_output.model_dump_json(indent=2), encoding="utf-8")


def _solve_mbpp(
    task_input: MBPPTaskInput,
    started_at: float,
    model_name: str | None,
    provider_url: str | None,
    providers_file_path: str,
) -> SolutionOutput:
    """設定・プロバイダ・サンドボックスを組み立て、1問を解いてSolutionOutputを返す.

    Args:
        task_input (MBPPTaskInput): 読み込んだタスク。
        started_at (float): プロセス起動直後の time.monotonic()。
        model_name (str | None): CLIで指定されたモデル名。
        provider_url (str | None): CLIで指定されたプロバイダURL。
        providers_file_path (str): プロバイダ設定ファイルのパス。

    Returns:
        SolutionOutput: エージェントの実行結果。
    """
    # 設定読み込み
    settings = load_settings()
    # .envから環境変数を読み込む(既に環境変数にあるキーは上書きしない)
    load_dotenv()

    # CLIで指定されたモデル / URLのプロバイダを先頭に、残りをフォールバックとして並べる
    providers = order_providers(load_providers(providers_file_path), model_name, provider_url)

    # 時間制限を使い果たさないようにデッドラインを計算(起点はプロセス起動時)
    # バックオフ待機とHTTPタイムアウトはこの残予算にクランプされる
    dead_line = started_at + settings.limits.total_time_sec - settings.limits.time_margin_sec
    client = ResilientLLMClient(
        providers=providers,
        transport=RequestsTransport(),
        retry=settings.retry,
        time_remaining=lambda: max(0, dead_line - time.monotonic()),
    )

    # Sandbox構築。MCPサーバーは実行中のPythonで、リポジトリ直下のファイルを起動する
    # (カレントディレクトリやPATH上のpythonに依存しない)
    sandbox_config = SandboxConfig(
        max_execution_time_seconds=int(settings.sandbox.exec_timeout_sec),
        max_memory_mb=settings.sandbox.memory_limit_mb
    )
    with Sandbox(
        config=sandbox_config,
        mcp_stdio=f"{shlex.quote(sys.executable)} {shlex.quote(str(_REPO_ROOT / 'mcp_tools_mbpp.py'))}"
    ) as sandbox:

        # Provider(ResilientLLMClient) / Sandbox / Orchestrator 構築
        orchestrator = Orchestrator(
            provider=client,
            extractor=CompositeCodeExtractor(),
            sandbox=sandbox,
            profile=MBPPProfile(),
            limits=settings.limits,
            solution_verifier=make_mbpp_verifier(sandbox, [*task_input.test_imports, *task_input.test_list]),
            started_at=started_at,
        )

        # 実行
        return orchestrator.run(
            str(task_input.task_id), "mbpp", format_mbpp_task(task_input)
        )
