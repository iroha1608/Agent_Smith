"""SWE-bench 計測ハーネス(ベンチマークレポート用).

指定したSWE-benchタスクを、指定したモデル1つ(フォールバック無し)で解かせ、moulinetteで検証し、
evaluations/swebench/<ts>/<model>/<instance_id>/ に task.json / solution.json / stdout.log /
stderr.log / validate.log を保存する(課題 VI.5 の評価ログ構造に合わせる)。

フォールバック無しにするのは、結果を「どのモデルの成績か」に一意に帰属させるため
(本番の providers.json はプロバイダをまたいで粘るので、ベンチマークには使わない)。

例:
    uv run python3 tools/harness_swebench.py \
        --task_ids sympy__sympy-14711,sympy__sympy-13480,pydata__xarray-4629 \
        --models mistral:codestral-latest,cohere:command-a-03-2025 --workers 2
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parent.parent
MOULINETTE_DIR = REPO_ROOT.parent / "moulinette"
MOULINETTE = MOULINETTE_DIR / ".venv" / "bin" / "moulinette_eval"
PROVIDERS_FILE = REPO_ROOT / "providers" / "providers.json"
# 採点側の制限(900秒)より少し長く待ってから強制終了する(エージェント自身の打ち切りを観察するため)
AGENT_HARD_TIMEOUT_SEC = 960
# 検証(テスト実行)に許す秒数
VALIDATE_TIMEOUT_SEC = 1800


def _single_model_providers(provider_name: str, model: str, out_path: Path) -> Path:
    """本番の providers.json から、指定プロバイダの指定モデルだけを含む設定を作る.

    Args:
        provider_name (str): プロバイダ名(providers.json の name)。
        model (str): モデル名。
        out_path (Path): 書き出し先。

    Returns:
        Path: 書き出した設定ファイルのパス。
    """
    providers = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))["providers"]
    matched = [entry for entry in providers if entry["name"] == provider_name]
    if not matched:
        raise SystemExit(f"provider {provider_name!r} not found in {PROVIDERS_FILE}")
    entry = dict(matched[0])
    entry["models"] = [model]
    entry["model_params"] = {model: entry.get("model_params", {}).get(model, {})} if entry.get("model_params") else {}
    entry["priority"] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"providers": [entry]}, indent=2), encoding="utf-8")
    return out_path


def _dump(instance_id: str, task_path: Path) -> None:
    """moulinette で SWE-bench タスクを書き出す.

    Args:
        instance_id (str): インスタンスID。
        task_path (Path): 書き出し先。
    """
    task_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(MOULINETTE), "dump", "swebench", "--task-id", instance_id, "--output", str(task_path)],
        cwd=MOULINETTE_DIR, capture_output=True, text=True, timeout=600,
    )
    if completed.returncode != 0 or not task_path.exists():
        raise SystemExit(f"dump failed for {instance_id}: {completed.stderr[-400:]}")


def _solve_and_validate(instance_id: str, task_path: Path, providers_file: Path) -> dict[str, object]:
    """1タスクをエージェントに解かせ、moulinette で検証して結果を返す.

    Args:
        instance_id (str): インスタンスID。
        task_path (Path): タスクファイル。
        providers_file (Path): このモデル専用のプロバイダ設定。

    Returns:
        dict[str, object]: 1タスク分の結果(合否・反復・トークン・時間など)。
    """
    task_dir = task_path.parent
    solution_path = task_dir / "solution.json"
    command = [sys.executable, "-m", "agent_swebench", "--task-file", str(task_path),
               "--output", str(solution_path), "--providers_file_path", str(providers_file)]
    started = time.monotonic()
    try:
        agent = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=AGENT_HARD_TIMEOUT_SEC)
        (task_dir / "stdout.log").write_text(agent.stdout, encoding="utf-8")
        (task_dir / "stderr.log").write_text(agent.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired:
        (task_dir / "stderr.log").write_text("[harness] agent timeout\n", encoding="utf-8")
    wall = time.monotonic() - started
    result: dict[str, object] = {"instance_id": instance_id, "wall_seconds": round(wall, 1), "passed": False}
    if not solution_path.exists():
        result["detail"] = "no solution.json"
        return result

    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    steps = solution.get("steps") or []
    result.update({
        "iterations": solution.get("iterations"),
        "input_tokens": solution.get("total_input_tokens"),
        "output_tokens": solution.get("total_output_tokens"),
        "time_seconds": round(float(solution.get("total_time_seconds") or 0.0), 1),
        "total_requests": solution.get("total_requests"),
        "retries": sum(int(step.get("retries") or 0) for step in steps),
        "error": solution.get("error"),
        "patch_chars": len(solution.get("solution") or ""),
        "models": sorted({step.get("model_name", "") for step in steps}),
    })
    validate = subprocess.run(
        [str(MOULINETTE), "validate", "swebench", str(task_path), str(solution_path)],
        cwd=MOULINETTE_DIR, capture_output=True, text=True, timeout=VALIDATE_TIMEOUT_SEC,
    )
    (task_dir / "validate.log").write_text(validate.stdout + "\n[stderr]\n" + validate.stderr, encoding="utf-8")
    result["passed"] = validate.returncode == 0
    result["detail"] = "PASSED" if result["passed"] else (
        "metrics invalid" if "Metrics: INVALID" in validate.stdout else "tests failed")
    return result


def main(task_ids: str | tuple[str, ...], models: str | tuple[str, ...], workers: int = 1, label: str = "") -> None:
    """モデル x タスクの組合せを計測し、結果の表を表示する.

    Args:
        task_ids (str | tuple[str, ...]): カンマ区切りのインスタンスID。
        models (str | tuple[str, ...]): カンマ区切りの "プロバイダ名:モデル名"。"production" なら本番の
            providers_swebench.json(フォールバックあり)で走らせる。
        workers (int): 並列数(Dockerとレート制限を見て決める)。
        label (str): 保存ディレクトリ名に付ける目印(ablation の before/after 等)。
    """
    def split(value: str | tuple[str, ...]) -> list[str]:
        parts = value.split(",") if isinstance(value, str) else list(value)
        return [str(part).strip() for part in parts if str(part).strip()]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = REPO_ROOT / "evaluations" / "swebench" / (f"{timestamp}_{label}" if label else timestamp)
    jobs: list[tuple[str, str, Path, Path]] = []
    # タスクごとにモデルを交互に並べる(同じプロバイダのモデルが同時に走ってレート制限に当たりにくくする)
    for instance_id in split(task_ids):
        for spec in split(models):
            if spec == "production":
                # 本番の設定(providers_swebench.json の順でフォールバックあり)で走らせる。試験の再現用
                model_dir = run_root / "production"
                providers_file = REPO_ROOT / "providers" / "providers_swebench.json"
            else:
                provider_name, _, model = spec.partition(":")
                model_dir = run_root / f"{provider_name}__{model.replace('/', '_')}"
                providers_file = _single_model_providers(provider_name, model, model_dir / "providers.json")
            task_path = model_dir / instance_id / "task.json"
            _dump(instance_id, task_path)
            jobs.append((spec, instance_id, task_path, providers_file))
    print(f"[harness] {len(jobs)} job(s) -> {run_root}", flush=True)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_solve_and_validate, instance_id, task_path, providers_file): spec
                   for spec, instance_id, task_path, providers_file in jobs}
        for future, spec in futures.items():
            result = {"model": spec, **future.result()}
            results.append(result)
            print(f"  {spec:45} {result['instance_id']:24} {'PASS' if result['passed'] else 'FAIL':4} "
                  f"iter={result.get('iterations')} in={result.get('input_tokens')} out={result.get('output_tokens')} "
                  f"t={result.get('time_seconds')}s err={result.get('error')} ({result.get('detail')})", flush=True)

    (run_root / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[harness] summary: {run_root / 'summary.json'}")


if __name__ == "__main__":
    fire.Fire(main)
