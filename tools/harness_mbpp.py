"""MBPP診断ハーネス(test/mbpp-tuning ブランチ・ローカル専用・push禁止).

MBPPタスクを実LLMで解かせ moulinette で検証、evaluations/<ts>/<task_id>/ に成果物を保存する。
再実行: uv run python3 tools/harness_mbpp.py --count 12 --seed 1 --providers_file tools/providers_qwen.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOULINETTE = REPO_ROOT.parent / "moulinette" / ".venv" / "bin" / "moulinette_eval"
AGENT_HARD_TIMEOUT_SEC = 180


@dataclass
class TaskResult:
    task_id: str
    crashed: bool
    passed: bool
    detail: str


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def _select(moulinette: str, count: int, seed: int, run_root: Path) -> list[tuple[str, Path]]:
    sel: dict[str, Path] = {}
    off = 0
    while len(sel) < count and off < count * 50:
        s = seed + off
        off += 1
        tmp = run_root / f"_seed{s}.json"
        c = _run([moulinette, "dump", "mbpp", "--seed", str(s), "--output", str(tmp)])
        if c.returncode != 0 or not tmp.exists():
            continue
        tid = str(json.loads(tmp.read_text())["task_id"])
        if tid in sel:
            tmp.unlink(missing_ok=True)
            continue
        td = run_root / tid
        td.mkdir(parents=True, exist_ok=True)
        tp = td / "task.json"
        tmp.rename(tp)
        sel[tid] = tp
    if len(sel) < count:
        raise SystemExit(f"only {len(sel)}/{count} tasks")
    return list(sel.items())


def _solve(moulinette: str, tid: str, tp: Path, providers_file: str | None) -> TaskResult:
    td = tp.parent
    sp = td / "solution.json"
    cmd = [sys.executable, "-m", "agent_mbpp", "--task-file", str(tp), "--output", str(sp)]
    if providers_file:
        cmd += ["--providers_file_path", providers_file]
    try:
        a = _run(cmd, timeout=AGENT_HARD_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        (td / "stderr.log").write_text("[harness] timeout\n", encoding="utf-8")
        return TaskResult(tid, True, False, "agent timeout")
    (td / "stdout.log").write_text(a.stdout, encoding="utf-8")
    (td / "stderr.log").write_text(a.stderr, encoding="utf-8")
    if a.returncode != 0 or not sp.exists():
        return TaskResult(tid, True, False, f"agent exited {a.returncode} / no solution")
    v = _run([moulinette, "validate", "mbpp", "--task-file", str(tp), "--solution-file", str(sp)])
    (td / "validate.log").write_text(v.stdout + "\n[stderr]\n" + v.stderr, encoding="utf-8")
    return TaskResult(tid, False, v.returncode == 0, "PASSED" if v.returncode == 0 else "validate FAILED")


def main(count: int = 5, seed: int = 1, eval_type: str = "mbpp",
         providers_file: str | None = None, moulinette: str = str(DEFAULT_MOULINETTE)) -> None:
    if not Path(moulinette).exists():
        raise SystemExit(f"no moulinette: {moulinette}")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = REPO_ROOT / "evaluations" / eval_type / ts
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[harness] providers={providers_file or 'default'} run={run_root}")
    tasks = _select(moulinette, count, seed, run_root)
    print(f"[harness] selected: {[t for t, _ in tasks]}", flush=True)
    results = []
    for tid, tp in tasks:
        r = _solve(moulinette, tid, tp, providers_file)
        print(f"  {tid:>6}: {'PASS' if r.passed else ('CRASH' if r.crashed else 'FAIL'):5} ({r.detail})", flush=True)
        results.append(r)
    passed = sum(1 for r in results if r.passed)
    summary = {"timestamp": ts, "providers_file": providers_file, "count": count, "seed": seed,
               "passed": passed, "results": [r.__dict__ for r in results]}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPO_ROOT / "tools" / "results" / f"{ts}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[harness] {passed}/{count} passed  (summary copied to tools/results/{ts}.json)")


if __name__ == "__main__":
    fire.Fire(main)
