"""レポートの裏付け（課題 V.7: backing solution.json）を、提出リポジトリの benchmark/swebench/ へ写す.

写すもの（1回ごと）: solution.json・task.json・validate_summary.txt（validate.log の判定の行）・providers.json
使い方: python3 export_benchmark.py <提出リポジトリ>
"""
import shutil
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
PROVIDERS = Path(__file__).resolve().parent / "providers"

SELECTION = ["mistral_codestral", "nvidia_glm53_flash", "ollama_gptoss120_low", "ollama_gptoss120_medium",
             "cohere_command_a", "cohere_command_a_plus", "cohere_reasoning_off", "cohere_reasoning_on",
             "kilo_ultra_off", "kilo_ultra_on", "nvidia_ultra_off", "nvidia_ultra_on", "ollama_ultra_none",
             "ollama_ultra_medium", "gemini_31lite_medium", "gemini_35flash"]

# (書き出し先のまとまり, ラベル, 回の一覧 or None=全部, タスクの絞り込み or None)
GROUPS = [
    ("selection_easy3", label, ["r1"], None) for label in SELECTION
] + [
    ("task14711_repeats", label, [f"r{n}" for n in range(2, 10)], "sympy__sympy-14711")
    for label in ["mistral_codestral", "nvidia_glm53_flash", "ollama_gptoss120_low", "ollama_gptoss120_medium",
                  "gemini_31lite_medium"]
] + [
    ("task14711_repeats", "swe_final", [f"r{n}" for n in range(5, 10)], "sympy__sympy-14711"),
    ("production_codestral_first", "swe_final", ["r1", "r2", "r3", "r4"], None),
    ("production_cohere_first", "swe_cohere", ["r1", "r2", "r3"], None),
    ("top3_6tasks", "s6_glm53", None, None),
    ("top3_6tasks", "s6_gptoss_low", None, None),
    ("top3_6tasks", "s6_gptoss_medium", None, None),
    ("production_final", "swe_final2", None, None),
]


def validate_summary(log: Path) -> str:
    """validate.log から判定の行だけを抜き出す."""
    if not log.exists():
        return "no validate.log\n"
    keep = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(key in line for key in ("Correctness", "Metrics:", "Overall", "Iterations:", "tokens:",
                                           "Time", "EXCEEDED"))]
    return "\n".join(keep) + "\n"


def main() -> None:
    """全部のまとまりを書き出す."""
    destination_root = Path(sys.argv[1]) / "benchmark" / "swebench"
    copied = 0
    for group, label, rounds, task_filter in GROUPS:
        for round_dir in sorted((RUNS / label).glob("r*")):
            if rounds is not None and round_dir.name not in rounds:
                continue
            for task_dir in sorted(round_dir.iterdir()):
                if task_filter and task_dir.name != task_filter:
                    continue
                if not (task_dir / "solution.json").exists():
                    continue
                target = destination_root / group / label / round_dir.name / task_dir.name
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(task_dir / "solution.json", target / "solution.json")
                shutil.copy2(RUNS / "tasks" / f"{task_dir.name}.json", target / "task.json")
                (target / "validate_summary.txt").write_text(validate_summary(task_dir / "validate.log"))
                copied += 1
        providers_file = PROVIDERS / f"{label}.json"
        if providers_file.exists():
            shutil.copy2(providers_file, destination_root / group / label / "providers.json")
    print(f"{copied} 回を書き出した: {destination_root}")


if __name__ == "__main__":
    main()
