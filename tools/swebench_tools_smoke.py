"""SWE-bench ツールサーバーの実機スモークテスト(手動実行・pytest対象外).

エージェントと同じ経路(環境変数を設定 -> Sandbox が mcp_tools_swebench.py を stdio で起動 ->
サンドボックス内のPythonからツールを呼ぶ)で、実際のSWE-benchイメージを使って9ツールを通しで動かす。
終了後に agentsmith ラベルのコンテナが残っていないことも確かめる。

使い方:
    uv run python tools/swebench_tools_smoke.py <dumpしたタスクJSON>
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.cli import _mcp_command  # noqa: E402
from src.agent.docker_env import container_name_for, ensure_image  # noqa: E402
from src.models.sandbox_config import SandboxConfig  # noqa: E402
from src.sandbox.sandbox_main import Sandbox  # noqa: E402

_SNIPPETS = [
    "print(sorted(t for t in ['read_file','edit_file','list_files','search_code',"
    "'search_function_or_class_definition_in_code','find_references','run_tests','get_patch','run_command']))",
    "print(search_function_or_class_definition_in_code(name='_check_vector'))",
    "print(search_code(pattern='def __add__', file_pattern='vector.py'))",
    "print(read_file(filepath='sympy/physics/vector/vector.py', start_line=58, end_line=62))",
    "print(list_files(directory='sympy/physics/vector', pattern='test_*.py'))",
    "print(find_references(name='_check_vector', filepath='sympy/physics/vector/vector.py')[:600])",
    "print(edit_file(filepath='sympy/physics/vector/vector.py', old_str='        other = _check_vector(other)\\n"
    "        return Vector(self.args + other.args)', new_str='        if other == 0:\\n            return self\\n"
    "        other = _check_vector(other)\\n        return Vector(self.args + other.args)'))",
    "print(edit_file(filepath='sympy/physics/vector/vector.py', old_str='        if other == 0:', new_str='        if other == 0'))",
    "patch = get_patch(); print(patch)",
    "report = run_tests(); print(report[-1500:])",
    "print(run_command(command='git status --short'))",
]


def main(task_file: str) -> None:
    """タスクのイメージでツールサーバーを起動し、全ツールを呼んで結果を表示する.

    Args:
        task_file (str): moulinette dump swebench で書き出したタスクJSON。
    """
    task = json.loads(Path(task_file).read_text(encoding="utf-8"))
    ensure_image(task["docker_image"], timeout=1800)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
        handle.write(task["eval_script"])
    os.environ.update({
        "TESTBED_PATH": "/testbed",
        "SWE_DOCKER_IMAGE": task["docker_image"],
        "SWE_CONTAINER_NAME": container_name_for(task["instance_id"]),
        "SWE_EVAL_SCRIPT_FILE": handle.name,
        "SWE_DEADLINE_EPOCH": str(time.time() + 600),
    })
    started = time.monotonic()
    try:
        with Sandbox(config=SandboxConfig(max_execution_time_seconds=30),
                     mcp_stdio=_mcp_command("mcp_tools_swebench.py")) as sandbox:
            print(f"[smoke] server + container ready in {time.monotonic() - started:.1f}s")
            for snippet in _SNIPPETS:
                began = time.monotonic()
                result = sandbox.run(snippet)
                print(f"\n>>> {snippet[:90]}  ({time.monotonic() - began:.1f}s)")
                print((result.stdout or "") + (result.error or ""))
    finally:
        Path(handle.name).unlink(missing_ok=True)
    time.sleep(1)
    left = subprocess.run(["docker", "ps", "-a", "--filter", "label=agentsmith=1", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    print(f"\n[smoke] containers left after exit: {left or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1])
