#!/usr/bin/env bash
# MBPP の A/C 比べ（codestral が1番手）が終わってから、swe_final の3・4周目を回す（mistral の枠を取り合わないため）
cd "$(dirname "$0")"
while pgrep -f '[a]b_interleave[.]sh' >/dev/null; do sleep 60; done
EXAM6="sympy__sympy-14711 sympy__sympy-13480 pydata__xarray-4629 django__django-11066 scikit-learn__scikit-learn-13439 sympy__sympy-18189"
for r in 3 4; do TASKS="$EXAM6" bash swe_run.sh $r swe_final; done
