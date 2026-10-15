"""SWE-bench のMCPツールがコマンドを実行する場所(ホスト / Dockerコンテナ)を抽象化するモジュール.

ツールの中身(workspace.py)は「どこで実行するか」を知らずに Executor.run() を呼ぶだけにする。
- LocalExecutor : ホストの TESTBED_PATH で実行する。採点側がツールを単体で試すとき
                  (TESTBED_PATH だけを設定してサーバーを起動する。課題 V.4)はこちらになる。
- DockerExecutor: タスクのイメージからコンテナを1つ起動し、docker exec で実行する。
                  ネットワークは切る(--network none。外部から解答を取ってこられないように。課題 VI.4.1)。
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Protocol

# docker exec の外側で待つ余裕(コンテナ内の timeout コマンドが先に止めるはずなので、その保険)
_DOCKER_EXEC_GRACE_SEC = 10.0


@dataclass
class CommandResult:
    """1回のコマンド実行の結果.

    Attributes:
        stdout (str): 標準出力。
        stderr (str): 標準エラー出力。
        exit_code (int): 終了コード。
        timed_out (bool): 制限時間で打ち切ったか。
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class Executor(Protocol):
    """コマンドの実行場所のインターフェース.

    Attributes:
        python (str): ツールの補助スクリプトを実行するPythonのコマンド。
    """

    python: str

    def run(self, command: str, workdir: str, timeout: float, stdin: str | None = None) -> CommandResult:
        """bashでコマンドを実行する."""
        ...

    def copy_in(self, host_path: str, target_path: str) -> None:
        """ホストのファイルを実行場所へコピーする."""
        ...

    def close(self) -> None:
        """後片付けをする(何度呼んでもよい)."""
        ...


def _run_process(argv: list[str], cwd: str | None, stdin: str | None, timeout: float) -> CommandResult:
    """子プロセスを実行し、制限時間を過ぎたらプロセスグループごと止める.

    subprocess.run(timeout=...) は直接の子しか止めないので、孫プロセスが残らないように
    新しいセッションで起動し、時間切れならグループ全体に SIGKILL を送る。

    Args:
        argv (list[str]): 実行するコマンド。
        cwd (str | None): 作業ディレクトリ。
        stdin (str | None): 標準入力に渡す文字列。
        timeout (float): 制限時間(秒)。

    Returns:
        CommandResult: 実行結果。
    """
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", start_new_session=True,
        )
    except OSError as error:
        return CommandResult("", f"cannot start {argv[0]}: {error}", 127)
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=max(0.1, timeout))
        return CommandResult(stdout, stderr, process.returncode)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return CommandResult(stdout or "", stderr or "", 124, timed_out=True)


class LocalExecutor:
    """ホスト上でコマンドを実行する(ツールの単体試験・ローカル開発用)."""

    def __init__(self) -> None:
        """補助スクリプト用のPythonには、サーバー自身を動かしているPythonを使う."""
        self.python = sys.executable

    def run(self, command: str, workdir: str, timeout: float, stdin: str | None = None) -> CommandResult:
        """ホストのbashでコマンドを実行する.

        Args:
            command (str): bashに渡すコマンド。
            workdir (str): 作業ディレクトリ。
            timeout (float): 制限時間(秒)。
            stdin (str | None): 標準入力に渡す文字列。

        Returns:
            CommandResult: 実行結果。
        """
        if not os.path.isdir(workdir):
            return CommandResult("", f"working directory does not exist: {workdir}", 2)
        return _run_process(["bash", "-c", command], workdir, stdin, timeout)

    def copy_in(self, host_path: str, target_path: str) -> None:
        """ホスト内でファイルをコピーする.

        Args:
            host_path (str): コピー元。
            target_path (str): コピー先。
        """
        if os.path.abspath(host_path) != os.path.abspath(target_path):
            shutil.copyfile(host_path, target_path)

    def close(self) -> None:
        """片付けるものは無い."""


class DockerExecutor:
    """タスクのイメージからコンテナを1つ起動し、その中でコマンドを実行する.

    コンテナ内のシェルは `bash -lc`(ログインシェル)で起動する。SWE-bench のイメージは
    ~/.bashrc で conda の testbed 環境を有効にしており、`bash -c` だとリポジトリ用ではない
    Python が使われてしまうため(実測: bash -c は Python 3.11、bash -lc は testbed の 3.9)。

    Attributes:
        image (str): 起動するイメージ。
        name (str): コンテナ名。
    """

    def __init__(self, image: str, name: str) -> None:
        """設定を保持する(起動は start() で行う).

        Args:
            image (str): 起動するイメージ。
            name (str): コンテナ名。
        """
        self.image = image
        self.name = name
        self.python = "python"
        self._started = False

    def start(self, timeout: float = 120.0) -> None:
        """ネットワークを切ったコンテナを起動する。イメージはpullしない(エージェントが用意済みの前提).

        Args:
            timeout (float): 起動を待つ秒数。

        Raises:
            RuntimeError: 起動に失敗した場合。
        """
        result = _run_process(
            ["docker", "run", "-d", "--name", self.name, "--label", "agentsmith=1", "--network", "none",
             "--pull", "never", self.image, "sleep", "infinity"],
            None, None, timeout,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"could not start container from {self.image}: {result.stderr.strip()[-400:]}")
        self._started = True

    def run(self, command: str, workdir: str, timeout: float, stdin: str | None = None) -> CommandResult:
        """コンテナ内のbash(ログインシェル)でコマンドを実行する.

        制限時間はコンテナ内の `timeout -s KILL` で守らせる(docker exec のクライアントを止めても
        コンテナ内のプロセスは止まらないため)。外側でも少し長めに待って打ち切る。

        Args:
            command (str): bashに渡すコマンド。
            workdir (str): コンテナ内の作業ディレクトリ。
            timeout (float): 制限時間(秒)。
            stdin (str | None): 標準入力に渡す文字列。

        Returns:
            CommandResult: 実行結果。
        """
        seconds = max(1, int(timeout))
        started = time.monotonic()
        result = _run_process(
            ["docker", "exec", "-i", "-w", workdir, self.name, "timeout", "-s", "KILL", str(seconds),
             "bash", "-lc", command],
            None, stdin, seconds + _DOCKER_EXEC_GRACE_SEC,
        )
        # timeout -s KILL で止められたコマンドは 137(128+9) で終わる
        if result.exit_code == 137 and time.monotonic() - started >= seconds - 0.5:
            result.timed_out = True
        return result

    def copy_in(self, host_path: str, target_path: str) -> None:
        """ホストのファイルをコンテナへコピーする.

        Args:
            host_path (str): ホスト上のコピー元。
            target_path (str): コンテナ内のコピー先。

        Raises:
            RuntimeError: コピーに失敗した場合。
        """
        result = _run_process(["docker", "cp", host_path, f"{self.name}:{target_path}"], None, None, 60)
        if result.exit_code != 0:
            raise RuntimeError(f"docker cp failed: {result.stderr.strip()[-300:]}")

    def close(self) -> None:
        """コンテナを削除する。何度呼んでもよい."""
        if not self._started:
            return
        self._started = False
        _run_process(["docker", "rm", "-f", self.name], None, None, 60)
