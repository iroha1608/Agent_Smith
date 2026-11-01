#!/usr/bin/env bash
# SWE-bench のタスクを、1つの providers ファイルで順に回す（試験の exam_swebench.sh と同じ: dump → run-agent 900 秒 → validate）
# 使い方: bash swe_run.sh <回> <ラベル...>     （providers/<ラベル>.json を使う）
#   例) bash swe_run.sh 1 cohere_command_a nvidia_ultra_on
#   6問で回す: TASKS="$EXAM6" bash swe_run.sh 1 production
# 10-30/03 §5 の swe_run.sh を元に、並べて回せるようにした（後片付けは自分のコンテナだけ・タスクファイルは先に作る）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"   # Agent_Smith_Project
REPO="${REPO:-$ROOT/wt_swe}"                       # 計測する commit の作業ツリー（.env を置く）
MOUL="${MOUL:-$ROOT/moulinette}"
PROV="${PROV:-$SCRIPT_DIR/providers}"
OUT="${OUT:-$SCRIPT_DIR/runs}"
EASY3="sympy__sympy-14711 sympy__sympy-13480 pydata__xarray-4629"   # 課題 PDF V.4 の「the simplest task first」
TASKS="${TASKS:-$EASY3}"
TIME_LIMIT=900
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"

run_one() {
    local label="$1" task_id="$2" round="$3"
    local task_file="$OUT/tasks/$task_id.json"
    local run_dir="$OUT/$label/r$round/$task_id"
    mkdir -p "$run_dir"
    rm -f "$run_dir/solution.json" "$run_dir/validate_rc"
    echo "$REPO" >"$run_dir/repo"

    # 走っている間に自分が作ったコンテナだけを後で片付けるため、始める前のコンテナを覚えておく
    local before
    before=$(docker ps -aq --filter "label=agentsmith=1" | sort)

    local agent_command="uv run python -m agent_swebench --task-file $task_file --output $run_dir/solution.json --providers_file_path $PROV/$label.json"
    echo ">>> $label r$round $task_id  ($(date '+%m-%d %H:%M'))"
    local started=$SECONDS
    ( cd "$REPO" && uv run --project "$MOUL" moulinette_eval run-agent "$TIME_LIMIT" "$agent_command" ) \
        >"$run_dir/agent_stdout.log" 2>"$run_dir/agent_stderr.log"
    echo $? >"$run_dir/run_rc"
    echo $((SECONDS - started)) >"$run_dir/wall_seconds"

    # 残ったコンテナのうち、始める前に無かったもの＝この回のもの（ほかの列のコンテナは触らない）
    local after leaked
    after=$(docker ps -aq --filter "label=agentsmith=1" | sort)
    leaked=$(comm -13 <(echo "$before") <(echo "$after") | grep -v '^$' || true)
    if [ -n "$leaked" ]; then
        # 並べて回しているときは、ほかの列が同じ間に作ったものも混ざりうる。走っているものは消さず記録だけにする
        echo "$leaked" >"$run_dir/leaked_containers"
        for container in $leaked; do
            if [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" = "false" ]; then
                docker rm -f "$container" >/dev/null 2>&1
            fi
        done
    fi

    if [ -f "$run_dir/solution.json" ]; then
        uv run --project "$MOUL" moulinette_eval validate swebench "$task_file" "$run_dir/solution.json" \
            >"$run_dir/validate.log" 2>&1
        echo $? >"$run_dir/validate_rc"
    else
        echo "no-solution" >"$run_dir/validate_rc"
    fi
    echo "    run_rc=$(cat "$run_dir/run_rc") validate_rc=$(cat "$run_dir/validate_rc") $(cat "$run_dir/wall_seconds")s"
}

if [ $# -lt 2 ]; then
    echo "usage: bash $0 <回> <ラベル...>" >&2
    exit 1
fi
round="$1"; shift
for label in "$@"; do
    [ -f "$PROV/$label.json" ] || { echo "providers ファイルが無い: $PROV/$label.json" >&2; exit 1; }
done
[ -d "$REPO" ] || { echo "作業ツリーが無い: $REPO" >&2; exit 1; }
for task_id in $TASKS; do
    [ -f "$OUT/tasks/$task_id.json" ] || { echo "タスクファイルが無い（先に dump する）: $OUT/tasks/$task_id.json" >&2; exit 1; }
done

for label in "$@"; do
    for task_id in $TASKS; do
        [ -f "$OUT/$label/r$round/$task_id/validate_rc" ] && continue
        run_one "$label" "$task_id" "$round"
    done
    echo "=== $label r$round 終わり ($(date '+%m-%d %H:%M'))"
done
