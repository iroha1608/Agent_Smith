#!/usr/bin/env bash
# 14711 で 9/9 だった3モデルを、1モデルだけ（フォールバックなし）で6問 × 3周。済んだ回は飛ばす（同じコマンドで再開できる）
cd "$(dirname "$0")"
EXAM6="sympy__sympy-14711 sympy__sympy-13480 pydata__xarray-4629 django__django-11066 scikit-learn__scikit-learn-13439 sympy__sympy-18189"
lane() { local name="$1"; shift; for r in 1 2 3; do TASKS="$EXAM6" bash swe_run.sh $r "$@"; done > "logs/s6_$name.log" 2>&1; }
lane glm53 s6_glm53 &
lane gptoss_low s6_gptoss_low &
lane gptoss_medium s6_gptoss_medium &
wait
echo "=== s6_top3 終わり ($(date '+%m-%d %H:%M'))"
