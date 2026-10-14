"""MBPP / SWE-bench エージェントの組み立て、実行を担う配線.

タスクファイル読み込み -> 設定 / プロバイダ解決 ->
Provider / Sandbox / Orchestrator 構築 ->
実行 -> SolutionOutput 書き出し
fireはagent_mbpp.py / agent_swebench.py から呼ばれ、
検証済みの引数だけを受け取る関数郡。

課題 IV.1「評価中のクラッシュは失格」のため、どの経路(正常終了 / 想定外の例外 /
Ctrl+C / 採点側のSIGTERM)でも solution.json を必ず書く。SWE-benchではさらに、
どの経路でもDockerコンテナを片付ける(課題 V.4 / VI.1)。
"""
import logging
import os
import shlex
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from src.agent.docker_env import container_name_for, ensure_image, remove_container
from src.agent.extractor.composite_extractor import CompositeCodeExtractor
from src.agent.llm.resilient_client import ResilientLLMClient
from src.agent.llm.transport.requests_transport import RequestsTransport
from src.agent.orchestrator import Orchestrator
from src.agent.prompt.task_profile.mbpp_profile import MBPPProfile, hardcoded_answers
from src.agent.prompt.task_profile.swebench_profile import SWEBenchProfile
from src.config.settings import Limits, Settings, load_providers, load_settings, order_providers
from src.models.input import MBPPTaskInput, SWEBenchTaskInput
from src.models.output import SolutionOutput
from src.models.sandbox_config import SandboxConfig
from src.sandbox.sandbox_main import Sandbox

_LOGGER = logging.getLogger("agent.cli")
# リポジトリ直下(このファイルは src/agent/cli.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
# サンドボックスの設定ファイル(課題 V.2: 制限はPydanticモデルとJSONで設定する)
_SANDBOX_TEMPLATE = _REPO_ROOT / "sandbox_template.json"
# SWE-benchのissueに付くヒント(議論のコメント)の最大文字数。長いと毎反復の入力を圧迫する
_HINTS_MAX_CHARS = 4000
# Dockerイメージのpullに使ってよいのは、残り時間からこの秒数を引いた分まで(解く時間を残す)
_TIME_RESERVED_FOR_SOLVING = 300.0

_Task = TypeVar("_Task", bound=BaseModel)


class _Terminated(BaseException):
    """採点側からSIGTERMで止められたことを表す例外クラス.

    Exceptionではなく BaseException を継承し、途中の `except Exception` に握りつぶされずに
    最上位まで届くようにする(届いたところで solution.json を書き、コンテナを片付ける)。
    """


def _raise_terminated(signum: int, frame: FrameType | None) -> None:
    """SIGTERMを受けたら _Terminated を送出するシグナルハンドラ.

    Args:
        signum (int): 受けたシグナル番号。
        frame (FrameType | None): 受けた時点のフレーム(未使用)。
    """
    raise _Terminated(f"received signal {signum}")


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


def format_swebench_task(task_input: SWEBenchTaskInput) -> str:
    """SWEBenchTaskInputをLLMに渡すuser messages文字列に整形する.

    issue本文に加え、moulinetteがタスクとして渡すヒント(issueの議論)も含める。
    ヒントはタスク入力の一部であり、外部からの取得ではない(課題 VI.4.1)。

    Args:
        task_input (SWEBenchTaskInput): タスクファイルから読み込んだSWEBenchTaskInputオブジェクト

    Returns:
        str: LLMに渡すリポジトリ名、issue、ヒントを含む文字列
    """
    lines = [
        f"Repository: {task_input.repo or task_input.instance_id} (checked out at /testbed)",
        "",
        "Issue:",
        task_input.problem_statement.strip(),
    ]
    hints = task_input.hints_text.strip()
    if hints:
        if len(hints) > _HINTS_MAX_CHARS:
            hints = hints[:_HINTS_MAX_CHARS] + "\n...[hints truncated]"
        lines += ["", "Hints from the issue discussion:", hints]
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

    Args:
        task_file_path (str): MBPPタスクファイルのパス
        output_file_path (str): SolutionOutputを書き出すJSONファイルのパス
        model_name (str | None): 使用するLLMモデル名。Noneなら設定ファイルのデフォルトを使用
        provider_url (str | None): 使用するLLMプロバイダのURL。Noneなら設定ファイルのデフォルトを使用
        providers_file_path (str): プロバイダ設定ファイルのパス
    """
    # 採点側の制限時間はプロセス起動から数えるので、できるだけ早く起点を取る
    started_at = time.monotonic()
    _run_safely(
        benchmark="mbpp",
        started_at=started_at,
        task_file_path=task_file_path,
        output_file_path=output_file_path,
        task_model=MBPPTaskInput,
        task_id_of=lambda task: str(task.task_id),
        solve=lambda task: _solve_mbpp(task, started_at, model_name, provider_url, providers_file_path),
    )


def run_swebench_agent(
    task_file_path: str,
    output_file_path: str,
    model_name: str | None = None,
    provider_url: str | None = None,
    providers_file_path: str = "providers/providers.json"
) -> None:
    """SWE-benchタスクを1問解くエージェントを実行し、SolutionOutputを必ず書き出す.

    Args:
        task_file_path (str): SWE-benchタスクファイルのパス
        output_file_path (str): SolutionOutputを書き出すJSONファイルのパス
        model_name (str | None): 使用するLLMモデル名。Noneなら設定ファイルのデフォルトを使用
        provider_url (str | None): 使用するLLMプロバイダのURL。Noneなら設定ファイルのデフォルトを使用
        providers_file_path (str): プロバイダ設定ファイルのパス
    """
    started_at = time.monotonic()
    _run_safely(
        benchmark="swebench",
        started_at=started_at,
        task_file_path=task_file_path,
        output_file_path=output_file_path,
        task_model=SWEBenchTaskInput,
        task_id_of=lambda task: task.instance_id,
        solve=lambda task: _solve_swebench(task, started_at, model_name, provider_url, providers_file_path),
    )


def _run_safely(
    benchmark: str,
    started_at: float,
    task_file_path: str,
    output_file_path: str,
    task_model: type[_Task],
    task_id_of: Callable[[_Task], str],
    solve: Callable[[_Task], SolutionOutput],
) -> None:
    """タスクを読み込んで解き、どの経路で終わっても solution.json を書く.

    - 想定外の例外: success=False / error=AGENT_CRASH を書いて正常終了する。
    - Ctrl+C / SIGTERM: success=False / error=INTERRUPTED を書いてから、終了コードで知らせる
      (KeyboardInterrupt / SystemExit 相当は握りつぶさない。課題 V.2)。
    後片付け(コンテナ削除など)は solve の中の finally が担う。

    Args:
        benchmark (str): "mbpp" または "swebench"。
        started_at (float): プロセス起動直後の time.monotonic()。
        task_file_path (str): タスクファイルのパス。
        output_file_path (str): solution.json の書き出し先。
        task_model (type[_Task]): タスク入力のPydanticモデル。
        task_id_of (Callable[[_Task], str]): タスクからtask_idを取り出す関数。
        solve (Callable[[_Task], SolutionOutput]): タスクを解く関数。

    Raises:
        SystemExit: Ctrl+C(130) / SIGTERM(143)で止められた場合、solution.jsonを書いた後に送出する。
    """
    task_id = "unknown"
    interrupted: BaseException | None = None
    previous_handler = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        task = task_model.model_validate_json(Path(task_file_path).read_text(encoding="utf-8"))
        task_id = task_id_of(task)
        solution_output = solve(task)
    except (KeyboardInterrupt, _Terminated) as error:
        interrupted = error
        solution_output = _failed_solution(task_id, benchmark, started_at, f"INTERRUPTED: {type(error).__name__}")
    except Exception as error:  # noqa: BLE001 - 評価中は落ちずに失敗を記録して返す
        _LOGGER.exception("agent crashed; writing a failed solution.json")
        solution_output = _failed_solution(
            task_id, benchmark, started_at, f"AGENT_CRASH: {type(error).__name__}: {error}"
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(solution_output.model_dump_json(indent=2), encoding="utf-8")
    if interrupted is not None:
        raise SystemExit(130 if isinstance(interrupted, KeyboardInterrupt) else 143)


def _failed_solution(task_id: str, benchmark: str, started_at: float, error: str) -> SolutionOutput:
    """ループに入る前 / 途中で止まった場合の、失敗を表すSolutionOutputを作る.

    Args:
        task_id (str): タスクID(読み込み前に失敗したら "unknown")。
        benchmark (str): "mbpp" または "swebench"。
        started_at (float): プロセス起動直後の time.monotonic()。
        error (str): 失敗の理由。

    Returns:
        SolutionOutput: success=False のSolutionOutput。
    """
    return SolutionOutput(
        task_id=task_id,
        benchmark=benchmark,
        success=False,
        solution="",
        iterations=0,
        total_requests=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_time_seconds=time.monotonic() - started_at,
        error=error[:500],
    )


def _build_client(
    settings: Settings,
    limits: Limits,
    started_at: float,
    model_name: str | None,
    provider_url: str | None,
    providers_file_path: str,
) -> ResilientLLMClient:
    """CLI指定を反映したプロバイダ列から、残り時間を意識するLLMクライアントを作る.

    Args:
        settings (Settings): 読み込んだ設定。
        limits (Limits): このベンチマークのハード制限。
        started_at (float): プロセス起動直後の time.monotonic()。
        model_name (str | None): CLIで指定されたモデル名。
        provider_url (str | None): CLIで指定されたプロバイダURL。
        providers_file_path (str): プロバイダ設定ファイルのパス。

    Returns:
        ResilientLLMClient: 複数プロバイダ / 複数キーで粘るクライアント。
    """
    # CLIで指定されたモデル / URLのプロバイダを先頭に、残りをフォールバックとして並べる
    providers = order_providers(load_providers(providers_file_path), model_name, provider_url)
    # 時間制限を使い果たさないようにデッドラインを計算(起点はプロセス起動時)。
    # バックオフ待機とHTTPタイムアウトはこの残予算にクランプされる
    dead_line = started_at + limits.total_time_sec - limits.time_margin_sec
    return ResilientLLMClient(
        providers=providers,
        transport=RequestsTransport(),
        retry=settings.retry,
        time_remaining=lambda: max(0.0, dead_line - time.monotonic()),
    )


def _sandbox_config(settings: Settings) -> SandboxConfig:
    """サンドボックスの設定を作る.

    import / ディレクトリの許可リストとメモリ上限は sandbox_template.json(あれば)から、
    1回の実行時間の上限はエージェントの config.json から取る(MBPPは120秒の中で反復するので
    sandbox_template.json の既定30秒より短くしたい)。

    Args:
        settings (Settings): 読み込んだ設定。

    Returns:
        SandboxConfig: サンドボックスの設定。
    """
    base = SandboxConfig()
    if _SANDBOX_TEMPLATE.exists():
        base = SandboxConfig.model_validate_json(_SANDBOX_TEMPLATE.read_text(encoding="utf-8"))
    return base.model_copy(update={
        "max_execution_time_seconds": int(settings.sandbox.exec_timeout_sec),
        "max_memory_mb": settings.sandbox.memory_limit_mb,
    })


def _mcp_command(script_name: str) -> str:
    """MCPサーバーを起動するコマンドを返す.

    実行中のPythonで、リポジトリ直下のファイルを起動する(カレントディレクトリやPATH上の
    pythonに依存しない)。

    Args:
        script_name (str): リポジトリ直下のサーバースクリプト名。

    Returns:
        str: stdioで起動するコマンド文字列。
    """
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(_REPO_ROOT / script_name))}"


def _solve_mbpp(
    task_input: MBPPTaskInput,
    started_at: float,
    model_name: str | None,
    provider_url: str | None,
    providers_file_path: str,
) -> SolutionOutput:
    """設定・プロバイダ・サンドボックスを組み立て、MBPPの1問を解いてSolutionOutputを返す.

    Args:
        task_input (MBPPTaskInput): 読み込んだタスク。
        started_at (float): プロセス起動直後の time.monotonic()。
        model_name (str | None): CLIで指定されたモデル名。
        provider_url (str | None): CLIで指定されたプロバイダURL。
        providers_file_path (str): プロバイダ設定ファイルのパス。

    Returns:
        SolutionOutput: エージェントの実行結果。
    """
    settings = load_settings()
    # .envから環境変数を読み込む(既に環境変数にあるキーは上書きしない)
    load_dotenv()
    client = _build_client(settings, settings.limits, started_at, model_name, provider_url, providers_file_path)

    with Sandbox(config=_sandbox_config(settings), mcp_stdio=_mcp_command("mcp_tools_mbpp.py")) as sandbox:
        orchestrator = Orchestrator(
            provider=client,
            extractor=CompositeCodeExtractor(),
            sandbox=sandbox,
            profile=MBPPProfile(),
            limits=settings.limits,
            solution_verifier=make_mbpp_verifier(sandbox, [*task_input.test_imports, *task_input.test_list]),
            started_at=started_at,
        )
        return orchestrator.run(str(task_input.task_id), "mbpp", format_mbpp_task(task_input))


def _solve_swebench(
    task_input: SWEBenchTaskInput,
    started_at: float,
    model_name: str | None,
    provider_url: str | None,
    providers_file_path: str,
) -> SolutionOutput:
    """Docker環境とMCPサーバーを用意し、SWE-benchの1問を解いてSolutionOutputを返す.

    コンテナはMCPサーバー(mcp_tools_swebench.py)が起動・所有する。エージェントは
    環境変数で「どのイメージを・どの名前で・どの評価スクリプトで・いつまでに」を渡し、
    最後に同じ名前のコンテナを必ず消す(サーバーがSIGKILLされても残さないため)。

    Args:
        task_input (SWEBenchTaskInput): 読み込んだタスク。
        started_at (float): プロセス起動直後の time.monotonic()。
        model_name (str | None): CLIで指定されたモデル名。
        provider_url (str | None): CLIで指定されたプロバイダURL。
        providers_file_path (str): プロバイダ設定ファイルのパス。

    Returns:
        SolutionOutput: エージェントの実行結果。
    """
    settings = load_settings()
    load_dotenv()
    limits = settings.swebench_limits
    client = _build_client(settings, limits, started_at, model_name, provider_url, providers_file_path)
    dead_line = started_at + limits.total_time_sec - limits.time_margin_sec

    # MCPサーバーは --pull never でコンテナを起動するので、イメージは先に用意する
    ensure_image(task_input.docker_image, timeout=dead_line - time.monotonic() - _TIME_RESERVED_FOR_SOLVING)

    container_name = container_name_for(task_input.instance_id)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="agentsmith_eval_", suffix=".sh", delete=False
    ) as eval_file:
        eval_file.write(task_input.eval_script)
    try:
        # MCPサーバー(stdioの子プロセス)は環境変数を引き継ぐ。締め切りは壁時計で渡す
        os.environ.update({
            "TESTBED_PATH": "/testbed",
            "SWE_DOCKER_IMAGE": task_input.docker_image,
            "SWE_CONTAINER_NAME": container_name,
            "SWE_EVAL_SCRIPT_FILE": eval_file.name,
            "SWE_DEADLINE_EPOCH": str(time.time() + (dead_line - time.monotonic())),
        })
        with Sandbox(config=_sandbox_config(settings), mcp_stdio=_mcp_command("mcp_tools_swebench.py")) as sandbox:
            orchestrator = Orchestrator(
                provider=client,
                extractor=CompositeCodeExtractor(),
                sandbox=sandbox,
                profile=SWEBenchProfile(),
                limits=limits,
                started_at=started_at,
            )
            return orchestrator.run(task_input.instance_id, "swebench", format_swebench_task(task_input))
    finally:
        remove_container(container_name)
        Path(eval_file.name).unlink(missing_ok=True)
