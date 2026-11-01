"""BENCHMARK_REPORT 用の表（課題 V.7 の 2〜4）を、bench_swe/runs の solution.json から作る.

10-15 のテストブランチの tools/benchmark_summary.py と同じ定義で、置き場の形
（runs/<ラベル>/r<回>/<タスク>/）に合わせたもの。

使い方: python3 report_tables.py <ラベル>[,<ラベル>...] [--round r1] [--runs <dir>]
"""
import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

_PATCH_FILE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)
_PASSED_SUMMARY = re.compile(r"\b\d+ passed\b")
_FAILED_SUMMARY = re.compile(r"\b\d+ (failed|exceptions?|errors?)\b")


def _passed(task_dir: Path) -> str:
    """validate.log から合否を返す."""
    log = task_dir / "validate.log"
    if not log.exists():
        return "n/a"
    return "PASS" if "Overall: PASSED" in log.read_text(encoding="utf-8", errors="replace") else "FAIL"


def _first_step(steps: list[dict[str, Any]], tool: str, files: list[str]) -> int | None:
    """指定ツールを、最終パッチのファイルに対して初めて呼んだステップ番号."""
    for step in steps:
        code = str(step.get("sandbox_input") or "")
        if tool in code and any(name in code for name in files):
            return int(step.get("step") or 0)
    return None


def _tests_first_green(steps: list[dict[str, Any]]) -> int | None:
    """run_tests() の結果が全部通った最初のステップ番号（テストランナーの要約行で判定）."""
    for step in steps:
        code = str(step.get("sandbox_input") or "")
        output = str(step.get("sandbox_output") or "")
        if "run_tests(" in code and "[TIMEOUT]" not in output \
                and _PASSED_SUMMARY.search(output) and not _FAILED_SUMMARY.search(output):
            return int(step.get("step") or 0)
    return None


def collect(runs: Path, labels: list[str], round_name: str | None) -> list[dict[str, Any]]:
    """ラベル（と回）ごとの全タスクを1行ずつ読む."""
    rows = []
    for label in labels:
        for solution_path in sorted(runs.glob(f"{label}/r*/*/solution.json")):
            task_dir = solution_path.parent
            if round_name and task_dir.parent.name != round_name:
                continue
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            steps = solution.get("steps") or []
            patch_files = _PATCH_FILE.findall(solution.get("solution") or "")
            first_green = _tests_first_green(steps)
            rows.append({
                "label": label, "round": task_dir.parent.name, "task": task_dir.name,
                "result": _passed(task_dir),
                "iterations": solution.get("iterations"),
                "input": solution.get("total_input_tokens") or 0,
                "output": solution.get("total_output_tokens") or 0,
                "seconds": round(float(solution.get("total_time_seconds") or 0.0)),
                "requests": solution.get("total_requests") or 0,
                "retries": sum(int(step.get("retries") or 0) for step in steps),
                "responses": len(steps),
                "latency": [float(step["request_time_ms"]) for step in steps if step.get("request_time_ms")],
                "error": solution.get("error"),
                "models": sorted({str(step.get("model_name")) for step in steps}),
                "first_read": _first_step(steps, "read_file", patch_files) if patch_files else None,
                "first_edit": _first_step(steps, "edit_file", patch_files) if patch_files else None,
                "green_to_submit": (int(solution.get("iterations") or 0) - first_green)
                if first_green and solution.get("success") else None,
            })
    return rows


def main() -> None:
    """表を3つ出す."""
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("--round", default=None)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent / "runs")
    arguments = parser.parse_args()
    rows = collect(arguments.runs, arguments.labels.split(","), arguments.round)

    print("| Configuration | Round | Task | Result | Iterations | Input tokens | Output tokens | Time (s) | Stop reason |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['label']} | {row['round']} | {row['task']} | {row['result']} | {row['iterations']} | "
              f"{row['input']:,} | {row['output']:,} | {row['seconds']} | {row['error'] or 'final_answer'} |")

    print("\n| Configuration | Pass | Avg response time per request (s) | Requests | Retries | Failed requests "
          "(retried or run-ending) | Request success rate | Runs not ended by provider failure |")
    print("|---|---|---|---|---|---|---|---|")
    for label in dict.fromkeys(row["label"] for row in rows):
        mine = [row for row in rows if row["label"] == label]
        latencies = [value for row in mine for value in row["latency"]]
        available = sum(1 for row in mine if row["error"] != "LLM_UNAVAILABLE")
        requests = sum(row["requests"] for row in mine)
        responses = sum(row["responses"] for row in mine)
        print(f"| {label} | {sum(row['result'] == 'PASS' for row in mine)}/{len(mine)} | "
              f"{statistics.mean(latencies) / 1000 if latencies else 0:.1f} | {requests} | "
              f"{sum(row['retries'] for row in mine)} | {requests - responses} | "
              f"{responses / requests if requests else 0:.0%} | {available}/{len(mine)} |")

    print("\n| Configuration | Round | Task | First read of a patched file (step) | First edit (step) | "
          "Iterations from tests passing to final_answer |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['label']} | {row['round']} | {row['task']} | {row['first_read'] or '-'} | "
              f"{row['first_edit'] or '-'} | {row['green_to_submit'] if row['green_to_submit'] is not None else '-'} |")


if __name__ == "__main__":
    main()
