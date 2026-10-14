"""SWE-bench エージェント側(プロファイル / CLI の安全網)のテスト.

実LLM・実Docker・実MCPサーバーは使わない。サンドボックスは結果を台本どおり返すスタブ。
"""
import json
import os
import signal
from pathlib import Path

import pytest

from src.agent.cli import _run_safely, format_swebench_task
from src.agent.extractor.python_extractor import PythonCodeExtractor
from src.agent.llm.provider.provider import LLMResponse
from src.agent.orchestrator import Orchestrator
from src.agent.prompt.task_profile.swebench_profile import SWEBenchProfile
from src.config.settings import Limits
from src.models.input import SWEBenchTaskInput
from src.models.output import ExecuteResult, SolutionOutput

_PATCH = "diff --git a/calc/ops.py b/calc/ops.py\n--- a/calc/ops.py\n+++ b/calc/ops.py\n@@ -1 +1 @@\n-x\n+y\n"


class _PatchSandbox:
    """get_patch() の結果だけを返すスタブ(final_answer経由で値を返す)."""

    def __init__(self, patch: str) -> None:
        self.patch = patch
        self.codes: list[str] = []

    def run(self, code: str) -> ExecuteResult:
        self.codes.append(code)
        if "get_patch()" in code and "final_answer" in code:
            return ExecuteResult(final_answer_bool=True, final_answer=self.patch)
        return ExecuteResult(stdout="explored\n")

    def manual(self) -> str:
        return "## Tools\nget_patch() -> str"


class _ScriptedLLM:
    """毎回同じ探索コードを返すLLMスタブ."""

    def __init__(self) -> None:
        self.calls = 0
        self.rotations = 0

    def generate(self, messages: list[dict[str, str]], stop_sequences: list[str], max_tokens: int) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"Thought: look.\n```python\nprint('step {self.calls}')\n```", input_tokens=10,
            output_tokens=5, request_time_ms=1.0, api_url="stub", model_name="stub", retries=0,
        )

    def rotate(self) -> None:
        self.rotations += 1


def test_sanitize_keeps_patch_and_adds_trailing_newline() -> None:
    """末尾改行の無いパッチには改行を足し、フェンスは外す(git apply の corrupt patch 対策)."""
    profile = SWEBenchProfile()
    assert profile.sanitize_submission(_PATCH.rstrip("\n")) == _PATCH
    assert profile.sanitize_submission("```diff\n" + _PATCH + "```") == _PATCH
    assert profile.sanitize_submission("   \n") == ""


def test_rejects_empty_and_non_diff_submissions() -> None:
    """空の提出とdiffでない提出は理由付きで差し戻し、diffは受理する."""
    profile = SWEBenchProfile()
    assert "empty patch" in (profile.submission_rejection("") or "")
    assert "not a git diff" in (profile.submission_rejection("I fixed it") or "")
    assert profile.submission_rejection(_PATCH) is None


def test_fallback_collects_patch_via_final_answer() -> None:
    """打ち切り時は get_patch() の値を final_answer 経由で丸ごと回収する(printだと切り詰められる)."""
    sandbox = _PatchSandbox(_PATCH)
    assert SWEBenchProfile().fallback_solution(sandbox) == _PATCH
    assert sandbox.codes == ["final_answer(get_patch())"]
    assert SWEBenchProfile().fallback_solution(_PatchSandbox("")) == ""


def test_orchestrator_submits_collected_patch_when_iterations_run_out() -> None:
    """提出が無いまま反復上限に達したら、編集済みのパッチを提出物にする。探索中はモデルを切り替えない."""
    llm = _ScriptedLLM()
    sandbox = _PatchSandbox(_PATCH)
    orchestrator = Orchestrator(
        provider=llm, extractor=PythonCodeExtractor(), sandbox=sandbox, profile=SWEBenchProfile(),
        limits=Limits(max_iterations=4, max_input_tokens=10**6, max_output_tokens=10**6,
                      max_output_tokens_per_iteration=100, total_time_sec=10**6, time_margin_sec=1.0),
    )
    solution = orchestrator.run("demo__demo-1", "swebench", "Issue: fix it")
    assert solution.success is False
    assert solution.error == "MAX_ITERATIONS"
    assert solution.solution == _PATCH
    assert llm.rotations == 0


def test_format_swebench_task_includes_issue_and_truncated_hints() -> None:
    """issue本文とヒントを含め、長いヒントは切り詰める."""
    task = SWEBenchTaskInput(
        instance_id="demo__demo-1", problem_statement="Bug: x is wrong", docker_image="img",
        eval_script="true", hints_text="h" * 10_000, repo="demo/demo",
    )
    text = format_swebench_task(task)
    assert "demo/demo (checked out at /testbed)" in text
    assert "Bug: x is wrong" in text
    assert "[hints truncated]" in text
    assert len(text) < 5_000


def _task_file(tmp_path: Path) -> Path:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({
        "instance_id": "demo__demo-1", "problem_statement": "p", "docker_image": "img", "eval_script": "true",
    }), encoding="utf-8")
    return path


def test_run_safely_writes_failed_solution_on_crash(tmp_path: Path) -> None:
    """解く途中で想定外の例外が起きても、success=False の solution.json を書いて正常終了する."""
    def solve(task: SWEBenchTaskInput) -> SolutionOutput:
        raise RuntimeError("docker exploded")

    output = tmp_path / "out" / "solution.json"
    _run_safely("swebench", 0.0, str(_task_file(tmp_path)), str(output), SWEBenchTaskInput,
                lambda task: task.instance_id, solve)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["task_id"] == "demo__demo-1"
    assert written["success"] is False
    assert written["error"].startswith("AGENT_CRASH: RuntimeError: docker exploded")


def test_run_safely_handles_sigterm(tmp_path: Path) -> None:
    """SIGTERMで止められても後片付け(finally)が走り、solution.json を書いてから終了コード143で終わる."""
    cleaned: list[str] = []

    def solve(task: SWEBenchTaskInput) -> SolutionOutput:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM should interrupt before this line")
        finally:
            cleaned.append("container removed")

    output = tmp_path / "solution.json"
    with pytest.raises(SystemExit) as info:
        _run_safely("swebench", 0.0, str(_task_file(tmp_path)), str(output), SWEBenchTaskInput,
                    lambda task: task.instance_id, solve)
    assert info.value.code == 143
    assert cleaned == ["container removed"]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["error"] == "INTERRUPTED: _Terminated"
    assert signal.getsignal(signal.SIGTERM) is not None
