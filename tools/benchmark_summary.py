"""SWE-bench ベンチマークの solution.json を集計し、BENCHMARK_REPORT 用のマークダウン表を出す.

集計する指標(課題 V.7):
- 結果表: モデル x タスクごとの Pass/Fail・反復数・入出力トークン・時間
- プロバイダの信頼性: 1リクエストの平均応答時間・リトライ回数・可用性(LLMが全滅せずに終わった割合)
- 中間指標:
  * 最終パッチに含まれるファイルを初めて読んだ / 編集したステップ(探索の効率)
  * run_tests() が初めて全部通ったステップから final_answer までの反復数(提出の規律。0が理想)

使い方:
    uv run python tools/benchmark_summary.py <ベンチマークのディレクトリ> [<ディレクトリ> ...]
"""
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

_PATCH_FILE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)


def _passed(task_dir: Path) -> str:
    """validate.log(リポジトリに取り込んだ要約なら validate_summary.txt)から合否を返す."""
    for name in ("validate.log", "validate_summary.txt"):
        log = task_dir / name
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            return "PASS" if "Overall: PASSED" in text else "FAIL"
    return "n/a"


def _first_step(steps: list[dict[str, Any]], tool: str, files: list[str]) -> int | None:
    """指定ツールを、最終パッチのファイルに対して初めて呼んだステップ番号."""
    for step in steps:
        code = str(step.get("sandbox_input") or "")
        if tool in code and any(name in code for name in files):
            return int(step.get("step") or 0)
    return None


_PASSED_SUMMARY = re.compile(r"\b\d+ passed\b")
_FAILED_SUMMARY = re.compile(r"\b\d+ (failed|exceptions?|errors?)\b")


def _tests_first_green(steps: list[dict[str, Any]]) -> int | None:
    """run_tests() の結果が全部通った最初のステップ番号.

    モデルは run_tests()[-1500:] のように末尾だけを表示することがあるので、先頭の見出しではなく
    テストランナーの要約行(pytest の "N passed" / sympy の "tests finished: N passed")で判定する。
    """
    for step in steps:
        code = str(step.get("sandbox_input") or "")
        output = str(step.get("sandbox_output") or "")
        if "run_tests(" in code and "Failing tests" not in output and "[TIMEOUT]" not in output \
                and _PASSED_SUMMARY.search(output) and not _FAILED_SUMMARY.search(output):
            return int(step.get("step") or 0)
    return None


def summarize(run_dirs: list[Path]) -> None:
    """ベンチマークのディレクトリ群を集計して表を表示する."""
    rows = []
    for run_dir in run_dirs:
        for solution_path in sorted(run_dir.glob("*/*/solution.json")):
            task_dir = solution_path.parent
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            steps = solution.get("steps") or []
            patch_files = _PATCH_FILE.findall(solution.get("solution") or "")
            first_green = _tests_first_green(steps)
            rows.append({
                "model": task_dir.parent.name.replace("__", " / ", 1),
                "task": task_dir.name,
                "result": _passed(task_dir),
                "iterations": solution.get("iterations"),
                "input": solution.get("total_input_tokens"),
                "output": solution.get("total_output_tokens"),
                "seconds": round(float(solution.get("total_time_seconds") or 0.0)),
                "requests": solution.get("total_requests") or 0,
                "retries": sum(int(step.get("retries") or 0) for step in steps),
                "latency": [float(step.get("request_time_ms") or 0.0) for step in steps if step.get("request_time_ms")],
                "error": solution.get("error"),
                "first_read": _first_step(steps, "read_file", patch_files) if patch_files else None,
                "first_edit": _first_step(steps, "edit_file", patch_files) if patch_files else None,
                "green_to_submit": (int(solution.get("iterations") or 0) - first_green)
                if first_green and solution.get("success") else None,
            })

    print("| Model / provider | Task | Result | Iterations | Input tokens | Output tokens | Time (s) | Stop reason |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['model']} | {row['task']} | {row['result']} | {row['iterations']} | {row['input']:,} | "
              f"{row['output']:,} | {row['seconds']} | {row['error'] or 'final_answer'} |")

    print("\n| Model / provider | Pass | Avg response time per request (s) | Requests | Retries | "
          "Availability (runs not ended by provider failure) |")
    print("|---|---|---|---|---|---|")
    for model in sorted({row["model"] for row in rows}):
        mine = [row for row in rows if row["model"] == model]
        latencies = [value for row in mine for value in row["latency"]]
        available = sum(1 for row in mine if row["error"] != "LLM_UNAVAILABLE")
        print(f"| {model} | {sum(row['result'] == 'PASS' for row in mine)}/{len(mine)} | "
              f"{statistics.mean(latencies) / 1000 if latencies else 0:.1f} | {sum(row['requests'] for row in mine)} | "
              f"{sum(row['retries'] for row in mine)} | {available}/{len(mine)} |")

    print("\n| Model / provider | Task | First read of a patched file (step) | First edit (step) | "
          "Iterations from tests passing to final_answer |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['model']} | {row['task']} | {row['first_read'] or '-'} | {row['first_edit'] or '-'} | "
              f"{row['green_to_submit'] if row['green_to_submit'] is not None else '-'} |")


if __name__ == "__main__":
    summarize([Path(argument) for argument in sys.argv[1:]])
