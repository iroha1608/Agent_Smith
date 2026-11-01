#!/usr/bin/env bash
# 14711 だけを、上位の候補（1モデルだけ・フォールバックなし）で r2〜r9 の8回ずつ回す（r1 は選定の回）。
# 済んだ回は飛ばすので、止まっても同じコマンドで続きから回る。
# mistral（codestral・本番の並び）は、MBPP の A/C と swe_final の3・4周目が終わってから回す（枠を取り合わないため）。
cd "$(dirname "$0")"
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
T=sympy__sympy-14711
lane() {  # lane <名前> <ラベル...>
    local name="$1"; shift
    for r in 2 3 4 5 6 7 8 9; do TASKS=$T bash swe_run.sh $r "$@"; done > "logs/c14711_$name.log" 2>&1
}
lane nvidia nvidia_glm53_flash &
lane ollama ollama_gptoss120_low ollama_gptoss120_medium &
lane gemini gemini_31lite_medium &
# mistral を使うものは、ほかの計測が終わってから
while pgrep -f '[a]b_interleave[.]sh|[a]fter_mbpp_final[.]sh' >/dev/null; do sleep 60; done
lane mistral mistral_codestral &
# 本番の並び（フォールバックあり）も 14711 だけ r5〜r9 を足す（r1〜r4 は6問の回）
( for r in 5 6 7 8 9; do TASKS=$T bash swe_run.sh $r swe_final; done > logs/c14711_swe_final.log 2>&1 ) &
wait
echo "=== compare_14711 終わり ($(date '+%m-%d %H:%M'))"
