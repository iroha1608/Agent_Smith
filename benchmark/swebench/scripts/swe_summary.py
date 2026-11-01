"""SWE の計測結果を1回1行の表にする（10-30/02 §7 の集計に、使ったモデルと打ち切りの理由を足したもの）.

使い方: python3 swe_summary.py [<OUT>] [--by-label]
  <OUT>/<ラベル>/r<回>/<タスクID>/ の solution.json と validate_rc を読む
  --by-label: ラベルごとの PASS 数・平均の反復・トークン・秒にまとめる
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_run(run_dir: Path) -> dict:
    """1回分を読んで辞書にする."""
    validate_rc_file = run_dir / "validate_rc"
    validate_rc = validate_rc_file.read_text().strip() if validate_rc_file.exists() else "running"
    result = {"0": "PASS", "running": "running"}.get(validate_rc, "FAIL")
    row = {"label": run_dir.parent.parent.name, "round": run_dir.parent.name, "task": run_dir.name, "result": result}
    solution_file = run_dir / "solution.json"
    if not solution_file.exists():
        row["error"] = "no solution.json"
        return row
    solution = json.loads(solution_file.read_text())
    steps = solution.get("steps", [])
    models = Counter(step.get("model_name", "?") for step in steps)
    row.update({
        "iter": solution["iterations"],
        "req": solution["total_requests"],
        "in_tok": solution["total_input_tokens"],
        "out_tok": solution["total_output_tokens"],
        "sec": round(solution["total_time_seconds"]),
        "retries": sum(step.get("retries", 0) for step in steps),
        "avg_req_s": round(sum(step.get("request_time_ms", 0) for step in steps) / len(steps) / 1000, 1) if steps else 0,
        "submitted": solution.get("success"),
        "error": (solution.get("error") or "")[:40],
        "models": ",".join(f"{name.split('/')[-1]}x{count}" for name, count in models.most_common()),
    })
    return row


def main() -> None:
    """表を出す."""
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    output_root = Path(arguments[0]) if arguments else Path(__file__).resolve().parent / "runs"
    rows = [read_run(run_dir) for run_dir in sorted(output_root.glob("*/r*/*"))
            if run_dir.is_dir() and run_dir.parent.parent.name != "tasks"]
    if "--by-label" in sys.argv:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["label"], row["round"])].append(row)
        print("\t".join(["label", "round", "pass", "done", "avg_iter", "avg_in_tok", "avg_out_tok", "avg_sec", "errors"]))
        for (label, round_name), group in sorted(grouped.items()):
            done = [row for row in group if row["result"] != "running"]
            measured = [row for row in done if "iter" in row]

            def average(key: str, items: list[dict] = measured) -> str:
                return str(round(sum(row[key] for row in items) / len(items))) if items else "-"
            errors = Counter(row.get("error") or "-" for row in done)
            print("\t".join([label, round_name, str(sum(row["result"] == "PASS" for row in done)), str(len(done)),
                             average("iter"), average("in_tok"), average("out_tok"), average("sec"),
                             " ".join(f"{name}:{count}" for name, count in errors.most_common())]))
        return
    header = ["label", "round", "task", "result", "iter", "req", "in_tok", "out_tok", "sec", "retries", "avg_req_s",
              "submitted", "error", "models"]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(key, "-")) for key in header))


if __name__ == "__main__":
    main()
