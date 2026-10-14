"""MBPP診断ハーネス(test/mbpp-tuning ブランチ・ローカル専用・push禁止).

MBPPタスクを実LLMで解かせ moulinette で検証し、evaluations/<ts>/<task_id>/ に成果物を保存する。
再実行:
    uv run python3 tools/harness_mbpp.py --count 50 --seed 1 \
        --providers_file tools/providers_mbpp.json --workers 3
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOULINETTE = REPO_ROOT.parent / "moulinette" / ".venv" / "bin" / "moulinette_eval"
_DEFAULT_MOULINETTE_PATH = str(DEFAULT_MOULINETTE)
AGENT_HARD_TIMEOUT_SEC = 180
# groq無料枠はモデル毎に 8000 tokens/min。連続実行で枠を自分で枯らさないよう、
# 直近60秒に投入したトークン量を見て次のタスク開始を待つ(ハーネス側のペーシング)。
# 実測(2026-10-14の50問run・枠が生きていた42件): 1タスクの総トークンは中央値2680/平均3199/最大7150。
# 2500では平均の1.3倍過小で、枠を自分で枯らして末尾8タスクがLLM_UNAVAILABLEになった。
TOKENS_PER_TASK_ESTIMATE = 4000


@dataclass
class TaskResult:
    """1タスク分のハーネス結果."""

    task_id: str
    crashed: bool = False
    passed: bool = False
    detail: str = ""
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    models: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """REPO_ROOT を cwd にコマンドを実行する."""
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def _select(moulinette: str, count: int, seed: int, run_root: Path,
            task_ids: tuple[str, ...] = ()) -> list[tuple[str, Path]]:
    """moulinette dump を連番seedで回し、重複しないcount個のタスクを選ぶ.

    task_idsを渡した場合は、それらが揃うまでseedを回して該当タスクだけを選ぶ
    (前回落ちたタスクの再測に使う)。
    """
    selected: dict[str, Path] = {}
    wanted = set(task_ids)
    target = len(wanted) if wanted else count
    offset = 0
    while len(selected) < target and offset < target * 200:
        current = seed + offset
        offset += 1
        tmp = run_root / f"_seed{current}.json"
        completed = _run([moulinette, "dump", "mbpp", "--seed", str(current), "--output", str(tmp)])
        if completed.returncode != 0 or not tmp.exists():
            continue
        task_id = str(json.loads(tmp.read_text())["task_id"])
        if task_id in selected or (wanted and task_id not in wanted):
            tmp.unlink(missing_ok=True)
            continue
        task_dir = run_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / "task.json"
        tmp.rename(task_path)
        selected[task_id] = task_path
    if len(selected) < target:
        raise SystemExit(f"only {len(selected)}/{target} tasks selected")
    return list(selected.items())


def _solve(moulinette: str, task_id: str, task_path: Path, providers_file: str | None) -> TaskResult:
    """1タスクをエージェントに解かせ、moulinetteで検証して結果を返す."""
    task_dir = task_path.parent
    solution_path = task_dir / "solution.json"
    cmd = [sys.executable, "-m", "agent_mbpp", "--task-file", str(task_path), "--output", str(solution_path)]
    if providers_file:
        cmd += ["--providers_file_path", providers_file]

    try:
        agent = _run(cmd, timeout=AGENT_HARD_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        (task_dir / "stderr.log").write_text("[harness] agent timeout\n", encoding="utf-8")
        return TaskResult(task_id, crashed=True, detail="agent timeout")

    (task_dir / "stdout.log").write_text(agent.stdout, encoding="utf-8")
    (task_dir / "stderr.log").write_text(agent.stderr, encoding="utf-8")
    if agent.returncode != 0 or not solution_path.exists():
        return TaskResult(task_id, crashed=True, detail=f"agent exited {agent.returncode} / no solution")

    solution = json.loads(solution_path.read_text())
    result = TaskResult(
        task_id,
        iterations=solution.get("iterations", 0),
        input_tokens=solution.get("total_input_tokens", 0),
        output_tokens=solution.get("total_output_tokens", 0),
        error=solution.get("error"),
        models=sorted({step.get("model_name", "") for step in solution.get("steps") or []}),
    )
    validate = _run([moulinette, "validate", "mbpp", "--task-file", str(task_path),
                     "--solution-file", str(solution_path)])
    (task_dir / "validate.log").write_text(validate.stdout + "\n[stderr]\n" + validate.stderr, encoding="utf-8")
    result.passed = validate.returncode == 0
    if result.passed:
        result.detail = "PASSED"
    elif "Correctness: FAILED" in validate.stdout:
        result.detail = "wrong answer"
    elif "Metrics: INVALID" in validate.stdout:
        result.detail = "metrics invalid"
    else:
        result.detail = "validate FAILED"
    return result


def main(count: int = 5, seed: int = 1, eval_type: str = "mbpp", providers_file: str | None = None,
         workers: int = 1, tpm_budget: int = 7000, task_ids: str | tuple[str, ...] = "",
         moulinette: str = _DEFAULT_MOULINETTE_PATH) -> None:
    """ハーネスを実行し、PASS数と失敗の内訳を表示する.

    Args:
        count (int): 解かせるタスク数。
        seed (int): タスク選択の連番seed開始値。
        eval_type (str): 保存先 evaluations/<eval_type>/。
        providers_file (str | None): agentに渡す providers 設定ファイル。
        workers (int): 並列実行数。レート制限が出るなら 1 に落とす。
        tpm_budget (int): 直近60秒に投入してよいトークン数の上限(0で無効)。
        task_ids (str | tuple[str, ...]): カンマ区切りのtask_id。指定するとそのタスクだけを再測する。
            fireはカンマ区切りをtupleに変換するため、strとtupleの両方を受ける。
        moulinette (str): moulinette_eval 実行ファイルのパス。
    """
    raw_ids = task_ids.split(",") if isinstance(task_ids, str) else task_ids
    wanted = tuple(str(part).strip() for part in raw_ids if str(part).strip())
    if not Path(moulinette).exists():
        raise SystemExit(f"moulinette_eval not found: {moulinette}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = REPO_ROOT / "evaluations" / eval_type / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[harness] providers={providers_file or 'default'} workers={workers} run={run_root}", flush=True)

    tasks = _select(moulinette, count, seed, run_root, wanted)
    print(f"[harness] selected {len(tasks)} tasks", flush=True)

    spent: list[tuple[float, int]] = []

    def pace() -> None:
        """直近60秒のトークン投入量がTPM予算に収まるまで待つ."""
        if tpm_budget <= 0:
            return
        while True:
            now = time.monotonic()
            recent = sum(tokens for stamp, tokens in spent if now - stamp < 60.0)
            if recent + TOKENS_PER_TASK_ESTIMATE <= tpm_budget or not spent:
                return
            time.sleep(3.0)

    results: list[TaskResult] = []

    def report(result: TaskResult) -> None:
        """1タスク分の結果を即座に1行表示して集計へ加える(長時間runの進捗を見るため)."""
        mark = "PASS" if result.passed else ("CRASH" if result.crashed else "FAIL")
        print(f"  {result.task_id:>6}: {mark:5} iter={result.iterations} "
              f"in={result.input_tokens} out={result.output_tokens} "
              f"err={result.error} ({result.detail})", flush=True)
        results.append(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for task_id, path in tasks:
            pace()
            future = pool.submit(_solve, moulinette, task_id, path, providers_file)
            # 逐次実行なら1件ごとに待って即表示する。並列なら投入後にまとめて回収する。
            if workers == 1:
                result = future.result()
                spent.append((time.monotonic(), result.input_tokens + result.output_tokens))
                report(result)
            else:
                pending.append(future)
        for future in pending:
            report(future.result())

    passed = sum(1 for r in results if r.passed)
    summary = {"timestamp": timestamp, "providers_file": providers_file, "count": count, "seed": seed,
               "workers": workers, "passed": passed, "results": [r.__dict__ for r in results]}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPO_ROOT / "tools" / "results" / f"{timestamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    buckets: dict[str, int] = {}
    for result in results:
        if not result.passed:
            buckets[result.detail] = buckets.get(result.detail, 0) + 1
    print(f"\n[harness] {passed}/{len(results)} passed  failures={buckets}")
    print(f"[harness] summary: tools/results/{timestamp}.json")


if __name__ == "__main__":
    fire.Fire(main)
