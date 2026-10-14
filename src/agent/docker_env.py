"""SWE-bench用のDocker環境を、エージェント側から準備・後片付けするモジュール.

コンテナ自体はMCPサーバー(mcp_tools_swebench.py)が起動して所有する。ここでは
(1) 起動前にイメージを用意する(サーバーは --pull never で起動するため)
(2) サーバーが強制終了(SIGKILL)されてコンテナが残った場合に備え、名前を決めて最後に必ず消す
の2点だけを担う(課題 V.4: 実行後にコンテナを片付ける責任はエージェント側にある)。
"""
import logging
import re
import subprocess
import uuid

_LOGGER = logging.getLogger("agent.docker")
# コンテナ名に使えない文字を置き換えるパターン(Dockerの名前は [a-zA-Z0-9][a-zA-Z0-9_.-]*)
_UNSAFE_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")


class DockerEnvironmentError(RuntimeError):
    """Dockerイメージを用意できなかったことを表す例外クラス."""


def container_name_for(instance_id: str) -> str:
    """タスクごとに一意なコンテナ名を作る.

    後から `docker ps` で見て、どのタスクの残骸か分かるよう instance_id を含める。

    Args:
        instance_id (str): SWE-benchのインスタンスID(例: sympy__sympy-14711)。

    Returns:
        str: コンテナ名(例: agentsmith-sympy__sympy-14711-1a2b3c4d)。
    """
    safe = _UNSAFE_NAME_CHARS.sub("-", instance_id)[:60]
    return f"agentsmith-{safe}-{uuid.uuid4().hex[:8]}"


def ensure_image(image: str, timeout: float) -> None:
    """Dockerイメージが手元に無ければpullする.

    Args:
        image (str): イメージ名(例: swebench/sweb.eval.x86_64.sympy_1776_sympy-14711:latest)。
        timeout (float): pullに許す秒数(タスクの残り時間から決める)。

    Raises:
        DockerEnvironmentError: dockerが無い / pullに失敗した / 時間切れの場合。
    """
    try:
        inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DockerEnvironmentError(f"docker is not available: {error}") from error
    if inspected.returncode == 0:
        return
    _LOGGER.info("pulling docker image %s (timeout %.0fs)", image, timeout)
    try:
        pulled = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=max(1.0, timeout))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DockerEnvironmentError(f"docker pull {image} did not finish: {error}") from error
    if pulled.returncode != 0:
        raise DockerEnvironmentError(f"docker pull {image} failed: {pulled.stderr.strip()[-300:]}")


def remove_container(name: str) -> None:
    """コンテナを(あれば)強制削除する。無ければ何もしない。失敗しても例外は投げない.

    MCPサーバーが自分で片付けるのが基本だが、サーバーがSIGKILLされた場合は
    後片付けが走らないので、エージェント側の finally で必ずこれを呼ぶ。

    Args:
        name (str): コンテナ名。
    """
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        _LOGGER.warning("could not remove container %s: %s", name, error)
