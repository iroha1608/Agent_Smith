# test/mbpp-tuning （ローカル専用・push禁止）

`origin/Agent/main` から切った**検証用ブランチ**。MBPPチューニングの①③を実装して実測した参照。
確定実装は本人が `Agent/main` に移植する（ここはリファレンス）。`origin` には push しない。

## 入っている変更（本実装への移植候補）
- **① content=null の graceful 化**
  - `src/agent/llm/provider/openai_compat.py`: `_parse_response` で `content` が str でなければ `LLMError` を送出。
  - `src/agent/llm/resilient_client.py`: ループ末尾に `except LLMError`（想定外のLLM層失敗＝parse失敗/null）を追加し、クラッシュさせず次の組合せへ advance。
- **③ 履歴の切り詰め**（累積入力トークン対策）
  - `src/agent/orchestrator.py`: `_window_messages()` を追加。system＋最初のタスク文は常に残し、以降は直近 `_HISTORY_WINDOW=3` 往復だけ再送、role=user の長い Observation は `_OBS_MAX_CHARS=1200` で truncate。`generate(messages=self._window_messages(messages))`。

## 再実行
```bash
# 前提: Docker起動 / ../moulinette で uv sync 済 / .env に GROQ_API_KEY・OPENROUTER_API_KEY
git checkout test/mbpp-tuning
uv run python3 tools/harness_mbpp.py --count 12 --seed 1 --providers_file tools/providers_qwen.json
# 結果サマリは tools/results/<ts>.json に自動コピー（evaluations/ はgitignore）
```
providers ファイルを差し替えれば別モデルも試せる（例: nemotron fallback を足した mbpp.json）。

## 実測の要点（2026-10-13）
- **③で累積入力が激減**: task252 は before `Input tokens 7268 > 6000 → INVALID`、after `total_in=1707 → PASSED`。「正答なのにトークン超過で不合格」が解消。
- **①でcrash消滅**: task462 は before `content=null ValidationError → exit1/no solution`、after `sol✓・LLM_UNAVAILABLE で graceful`。
- 残る失敗の多くは**単一groqキー連打による429**（`retry_after=26` 等）で `LLM_UNAVAILABLE`→空solution。exam本番(間隔あり)やfallback(qwen+nemotron)で緩和見込み。
- lint(flake8/mypy)・pytest(78) 緑。

## 注意
- `origin/Agent/main` が進んだら `git rebase origin/Agent/main` で追従。
- これはローカル参照。コミットは**このマシンのこのブランチのみ**（push しない）。
