# モデルのベンチマークのレポート（SWE-bench）

> **下書き（2026-11-01・日本語）。** ユーザーが添削したあと、英語に訳す前提です。
> 取捨選択しやすいように、アブレーションの候補などを広めに入れています。
> - `【要判断】` の印がある節は、残すかどうかを決めてもらうところです。
>
> 数字はすべて実際の計測から取ったものです。各回の `solution.json` は `benchmark/` に置きます（§9）。

## 0. まとめ

- **比べたもの（課題の要件「同じ3問以上・5モデル以上・2社以上」）**:
  - SWE-bench の3問（課題 PDF が「最初に試す、いちばん簡単な問題」として挙げるもの）
  - 6社・11モデル・16設定。同じモデルの推論 ON／OFF は、別の設定として数えています。
  - 1つの設定につき1モデルだけで回しました（フォールバックなし）。制限は試験と同じです。
- **3問とも PASS したのは7設定**です。そのうち4つは、自分から `final_answer` で提出して通しました: Mistral codestral、NVIDIA glm-5.3-flash、Ollama gpt-oss:120b（推論 low）、Cohere command-a。
  - 残りの3つは、反復の上限（30）で止まったときに回収したパッチで1問通っています。
- **推論を強くしても、点は上がりませんでした**（§5.1）。
  - gpt-oss:120b は、推論 medium で出力の上限（10,000）を使い切りました。
  - nemotron-3-ultra は、推論 medium で反復が16回から27回に増えました。
- **本番の並び**: **Ollama gpt-oss:120b（推論 low）を1番手**にします（§6）。
  - 試験の候補の6問 × 3周を、1モデルだけで回して **18/18** でした。
  - いちばん差の出た 14711 は 9/9 でした。
  - 2番手は NVIDIA glm-5.3-flash（17/18。落ちた1回は NVIDIA の通信エラー）です。
  - 今の1番手の cohere command-a と、案にしていた codestral は、14711 と scikit-learn で落ちる回がありました。
- **使わないもの**:
  - Cohere の command-a-plus と command-a-reasoning（推論 ON）。毎回 HTTP 422 でした。
  - Gemini 3.5-flash。1日の枠が小さすぎました。
  - Groq。1分あたり 8,000 トークンの枠が、SWE の1回の入力より小さいためです。

## 1. 準備（Setup）

### 1.1 1回の測り方

各（設定, タスク）について、試験と同じ3段で回しました。

1. `moulinette_eval dump swebench --task-id <id>` で `task.json` を書き出す
2. `moulinette_eval run-agent 900 "uv run python -m agent_swebench --task-file … --output … --providers_file_path <その設定だけの providers ファイル>"`
3. `moulinette_eval validate swebench task.json solution.json`（パッチを当ててテストを回し、上限も確かめる）

- **1回に1モデルだけ・フォールバックなし**。どの結果も、そのモデル1つのものです。
- 本番の設定（フォールバックあり）は、別に §7 で測りました。
- 制限は試験と同じです（30反復、入力 300,000、出力 10,000、900秒。どれもタスク全体の合計）。
  - エージェント自身が制限を守り、validate がもう一度確かめます。
  - この節の回は、すべて `Metrics: VALID` でした。
- エージェントの版: `da749ed`（提出する版）。全部の設定で、プロンプト・ツール・ループは同じです。変えたのは、モデル名とリクエストの設定（`model_params`）だけです。
- 日時: 2026-11-01 22:16〜22:47。4つの列を並べて回しました（プロバイダが違うので、枠はぶつかりません）。
- 使ったスクリプト: `benchmark/swebench/scripts/` の `swe_run.sh`・`swe_summary.py`・`report_tables.py`（§9）。

### 1.2 比べたモデルとプロバイダ（6社・11モデル・16設定）

| プロバイダ（無料枠） | モデル | リクエストの設定 |
|---|---|---|
| Mistral | `codestral-latest` | なし |
| Cohere（試用キー） | `command-a-03-2025` | なし |
| Cohere | `command-a-plus-05-2026` | なし |
| Cohere | `command-a-reasoning-08-2025` | `reasoning_effort: none`／既定（ON） |
| NVIDIA NIM | `nvidia/nemotron-3-ultra-550b-a55b` | `enable_thinking: false`／`true` |
| NVIDIA NIM | `z-ai/glm-5.3-flash` | なし（推論する） |
| Ollama Cloud | `gpt-oss:120b` | `reasoning_effort: low`／`medium` |
| Ollama Cloud | `nemotron-3-ultra` | `reasoning_effort: none`／`medium` |
| Kilo Code（OpenRouter 互換） | `nvidia/nemotron-3-ultra-550b-a55b:free` | `reasoning.enabled: false`／`true` |
| Google AI Studio | `gemini-3.1-flash-lite` | `reasoning_effort: medium` |
| Google AI Studio | `gemini-3.5-flash` | なし（推論する） |

**選び方**:

- 9社の無料モデルを、1回ずつ呼んで確かめました。
  - 203件中93件が使えました。
  - MBPP の10問 × 2回と、難しい10問 × 2回で絞りました。
- SWE では「推論の多いモデルのほうが、入出力の上限が大きい分、点が上がるのでは」という仮説がありました。そこで、推論 ON／OFF を同じモデルで並べました。
- 推論 ON の設定は、先に「エージェントと同じ形のリクエスト」（`stop=["<end_code>"]`・`max_tokens=1200`）で1回ずつ確かめ、使えないものを外しました（`benchmark/swebench/scripts/reasoning_probe.json`）。

| 外したもの | 理由 |
|---|---|
| NVIDIA `minimax-m3.1-flash` | 1回の応答に 81秒。900秒の中で約10回しか呼べない |
| NVIDIA `deepseek-v4.1-flash` | 1回の応答に 100秒 |
| Mistral `mistral-large-4` | `content` が文字列ではなくリストで返る。エージェントは空の応答として扱う |
| Mistral `magistral-medium` | 確かめたときに 429 |
| Ollama `gpt-oss:120b` 推論 high | `finish_reason=tool_calls` で本文が空（自分の形式のツール呼び出しを返す） |
| Kilo `laguna-s-2.1:free` | 計測の日に Kilo の一覧から消えていた（404）。無料モデルは予告なく消える |
| Groq（全モデル） | 1分あたり 8,000 トークンがアカウント単位。SWE の1回は 11,000〜130,000 トークンの入力なので、900秒に収まらない |
| Cloudflare Workers AI | 1日の枠（10,000 neurons）が、MBPP の計測でも約40分で尽きた |

### 1.3 タスク

`sympy__sympy-14711`・`sympy__sympy-13480`・`pydata__xarray-4629` を使いました。理由は次のとおりです。

- 課題 PDF V.4 が「最初に試す、いちばん簡単な問題」として挙げている3問です。
- 3問とも moulinette の試験の候補（`EXAM_POOL` の6問）に入っています。そのため、試験の結果の見込みにそのまま使えます。
- 2つのリポジトリ（sympy・xarray）、2つのテストの回し方（sympy の `bin/test`・pytest）、3種類の直し方にまたがります。
  - 14711: `Vector.__add__` で 0 の場合が抜けている
  - 13480: 打ち間違い（`cotm` → `cothm`）
  - xarray-4629: `merge` が最初のオブジェクトの `attrs` を、コピーせずにそのまま返す
- 14711 は、モデルの差がいちばん出た問題です。30反復まで使ったモデルが多くありました。

### 1.4 指標の定義

| 指標 | 定義 |
|---|---|
| 反復・入出力トークン・時間 | `solution.json` の `iterations`・`total_input_tokens`・`total_output_tokens`・`total_time_seconds`。トークンはプロバイダの `usage` の合計で、隠れた推論のトークンも含む |
| 1リクエストの平均応答時間 | 各ステップの `request_time_ms` の平均（成功した応答について。中のリトライの時間も含む） |
| リクエスト数・失敗したリクエスト | `total_requests`（リトライ込みの全試行）と、そのうち応答が得られなかった試行 |
| 提供側の失敗で終わらなかった回 | 止まった理由が `LLM_UNAVAILABLE` でない回 |
| 最終パッチのファイルを初めて読んだ／編集したステップ | 最終パッチ（`diff --git a/<file>`）に出てくるファイルについて、`read_file`／`edit_file` を初めて呼んだステップ。探索の効率を表す |
| テストが全部通ってから `final_answer` までの反復 | 最後の反復 − `run_tests()` の出力が初めて全部通った（`N passed` があり、failed／error が無い）ステップ。提出の規律を表す。0 が理想 |

## 2. 結果（モデル × タスク）

| 設定 | タスク | 結果 | 反復 | 入力トークン | 出力トークン | 時間(秒) | 止まった理由 |
|---|---|---|---|---|---|---|---|
| Mistral / codestral-latest | xarray-4629 | PASS | 4 | 11,288 | 244 | 15 | final_answer |
| Mistral / codestral-latest | sympy-13480 | PASS | 3 | 5,331 | 222 | 19 | final_answer |
| Mistral / codestral-latest | sympy-14711 | PASS | 4 | 7,747 | 454 | 18 | final_answer |
| NVIDIA / glm-5.3-flash（推論あり） | xarray-4629 | PASS | 6 | 18,759 | 592 | 232 | final_answer |
| NVIDIA / glm-5.3-flash（推論あり） | sympy-13480 | PASS | 5 | 11,710 | 536 | 168 | final_answer |
| NVIDIA / glm-5.3-flash（推論あり） | sympy-14711 | PASS | 5 | 14,950 | 1,021 | 75 | final_answer |
| Ollama / gpt-oss:120b（推論 low） | xarray-4629 | PASS | 14 | 50,148 | 1,106 | 26 | final_answer |
| Ollama / gpt-oss:120b（推論 low） | sympy-13480 | PASS | 6 | 11,567 | 352 | 23 | final_answer |
| Ollama / gpt-oss:120b（推論 low） | sympy-14711 | PASS | 20 | 65,169 | 3,178 | 39 | final_answer |
| Ollama / gpt-oss:120b（推論 medium） | xarray-4629 | FAIL | 17 | 65,782 | 10,000 | 55 | MAX_OUTPUT_TOKENS |
| Ollama / gpt-oss:120b（推論 medium） | sympy-13480 | PASS | 9 | 27,311 | 3,371 | 41 | final_answer |
| Ollama / gpt-oss:120b（推論 medium） | sympy-14711 | PASS | 13 | 47,192 | 5,982 | 68 | final_answer |
| Cohere / command-a-03-2025 | xarray-4629 | PASS | 6 | 25,934 | 529 | 88 | final_answer |
| Cohere / command-a-03-2025 | sympy-13480 | PASS | 5 | 13,817 | 443 | 56 | final_answer |
| Cohere / command-a-03-2025 | sympy-14711 | PASS | 9 | 27,276 | 1,000 | 168 | final_answer |
| Cohere / command-a-plus-05-2026 | xarray-4629 | FAIL | 0 | 0 | 0 | 4 | LLM_UNAVAILABLE |
| Cohere / command-a-plus-05-2026 | sympy-13480 | FAIL | 0 | 0 | 0 | 7 | LLM_UNAVAILABLE |
| Cohere / command-a-plus-05-2026 | sympy-14711 | FAIL | 0 | 0 | 0 | 10 | LLM_UNAVAILABLE |
| Cohere / command-a-reasoning（推論 none） | xarray-4629 | PASS | 2 | 5,611 | 300 | 9 | LLM_UNAVAILABLE |
| Cohere / command-a-reasoning（推論 none） | sympy-13480 | PASS | 4 | 8,877 | 188 | 27 | final_answer |
| Cohere / command-a-reasoning（推論 none） | sympy-14711 | FAIL | 1 | 1,930 | 64 | 15 | LLM_UNAVAILABLE |
| Cohere / command-a-reasoning（推論 ON） | xarray-4629 | FAIL | 1 | 2,654 | 534 | 15 | LLM_UNAVAILABLE |
| Cohere / command-a-reasoning（推論 ON） | sympy-13480 | FAIL | 0 | 0 | 0 | 6 | LLM_UNAVAILABLE |
| Cohere / command-a-reasoning（推論 ON） | sympy-14711 | FAIL | 1 | 1,930 | 1,199 | 15 | LLM_UNAVAILABLE |
| Kilo / nemotron-3-ultra:free（推論 OFF） | xarray-4629 | PASS | 9 | 58,964 | 9,423 | 425 | final_answer |
| Kilo / nemotron-3-ultra:free（推論 OFF） | sympy-13480 | PASS | 7 | 19,104 | 572 | 36 | final_answer |
| Kilo / nemotron-3-ultra:free（推論 OFF） | sympy-14711 | PASS* | 30 | 122,730 | 3,245 | 235 | MAX_ITERATIONS |
| Kilo / nemotron-3-ultra:free（推論 ON） | xarray-4629 | PASS | 10 | 46,892 | 1,403 | 82 | final_answer |
| Kilo / nemotron-3-ultra:free（推論 ON） | sympy-13480 | PASS | 8 | 21,252 | 764 | 92 | final_answer |
| Kilo / nemotron-3-ultra:free（推論 ON） | sympy-14711 | FAIL | 27 | 105,197 | 4,097 | 180 | LLM_UNAVAILABLE |
| NVIDIA / nemotron-3-ultra（推論 OFF） | xarray-4629 | PASS | 15 | 65,633 | 1,216 | 99 | final_answer |
| NVIDIA / nemotron-3-ultra（推論 OFF） | sympy-13480 | PASS | 7 | 19,376 | 617 | 42 | final_answer |
| NVIDIA / nemotron-3-ultra（推論 OFF） | sympy-14711 | FAIL | 30 | 131,210 | 2,562 | 132 | MAX_ITERATIONS |
| NVIDIA / nemotron-3-ultra（推論 ON） | xarray-4629 | PASS | 9 | 34,193 | 1,160 | 82 | final_answer |
| NVIDIA / nemotron-3-ultra（推論 ON） | sympy-13480 | PASS | 6 | 13,774 | 475 | 57 | final_answer |
| NVIDIA / nemotron-3-ultra（推論 ON） | sympy-14711 | PASS* | 30 | 108,468 | 3,991 | 192 | MAX_ITERATIONS |
| Ollama / nemotron-3-ultra（推論 none） | xarray-4629 | PASS | 11 | 54,305 | 1,468 | 85 | final_answer |
| Ollama / nemotron-3-ultra（推論 none） | sympy-13480 | PASS | 6 | 14,411 | 488 | 39 | final_answer |
| Ollama / nemotron-3-ultra（推論 none） | sympy-14711 | PASS* | 30 | 110,291 | 2,517 | 175 | MAX_ITERATIONS |
| Ollama / nemotron-3-ultra（推論 medium） | xarray-4629 | PASS | 21 | 124,476 | 2,639 | 220 | final_answer |
| Ollama / nemotron-3-ultra（推論 medium） | sympy-13480 | PASS* | 30 | 106,696 | 3,702 | 278 | MAX_ITERATIONS |
| Ollama / nemotron-3-ultra（推論 medium） | sympy-14711 | FAIL | 30 | 125,707 | 4,726 | 282 | MAX_ITERATIONS |
| Google / gemini-3.1-flash-lite（推論 medium） | xarray-4629 | PASS | 3 | 10,093 | 878 | 36 | final_answer |
| Google / gemini-3.1-flash-lite（推論 medium） | sympy-13480 | PASS | 3 | 9,096 | 670 | 48 | final_answer |
| Google / gemini-3.1-flash-lite（推論 medium） | sympy-14711 | FAIL | 6 | 27,368 | 2,715 | 115 | final_answer |
| Google / gemini-3.5-flash | xarray-4629 | FAIL | 0 | 2,647 | 0 | 42 | LLM_UNAVAILABLE |
| Google / gemini-3.5-flash | sympy-13480 | FAIL | 0 | 0 | 0 | 10 | LLM_UNAVAILABLE |
| Google / gemini-3.5-flash | sympy-14711 | FAIL | 6 | 22,734 | 496 | 30 | LLM_UNAVAILABLE |

\* 反復の上限で止まり、エージェントが最後に回収したパッチ（`get_patch()`）が採点で通った回。
`LLM_UNAVAILABLE` で PASS の回（Cohere reasoning none の xarray）も同じで、途中で応答が止まったときの回収パッチが通っています。

**3問の合計**:

| 設定 | 結果 | うち自分で提出 | 平均反復 | 平均入力 | 平均出力 | 平均秒 |
|---|---|---|---|---|---|---|
| Mistral / codestral-latest | **3/3** | 3 | 4 | 8,122 | 307 | **17** |
| NVIDIA / glm-5.3-flash | **3/3** | 3 | 5 | 15,140 | 716 | 158 |
| Ollama / gpt-oss:120b（low） | **3/3** | 3 | 13 | 42,295 | 1,545 | 29 |
| Cohere / command-a-03-2025 | **3/3** | 3 | 7 | 22,342 | 657 | 104 |
| Kilo / nemotron-3-ultra（OFF） | 3/3 | 2 | 15 | 66,933 | 4,413 | 232 |
| NVIDIA / nemotron-3-ultra（ON） | 3/3 | 2 | 15 | 52,145 | 1,875 | 110 |
| Ollama / nemotron-3-ultra（none） | 3/3 | 2 | 16 | 59,669 | 1,491 | 100 |
| NVIDIA / nemotron-3-ultra（OFF） | 2/3 | 2 | 17 | 72,073 | 1,465 | 91 |
| Kilo / nemotron-3-ultra（ON） | 2/3 | 2 | 15 | 57,780 | 2,088 | 118 |
| Ollama / gpt-oss:120b（medium） | 2/3 | 2 | 13 | 46,762 | 6,451 | 55 |
| Ollama / nemotron-3-ultra（medium） | 2/3 | 1 | 27 | 118,960 | 3,689 | 260 |
| Google / gemini-3.1-flash-lite（medium） | 2/3 | 2 | 4 | 15,519 | 1,421 | 66 |
| Cohere / command-a-reasoning（none） | 2/3 | 1 | 2 | 5,473 | 184 | 17 |
| Cohere / command-a-reasoning（ON） | 0/3 | 0 | - | - | - | - |
| Cohere / command-a-plus | 0/3 | 0 | - | - | - | - |
| Google / gemini-3.5-flash | 0/3 | 0 | - | - | - | - |

### 2.1 失敗の中身

| 設定 | タスク | 何が起きたか（`llm_output`・`sandbox_output` から） |
|---|---|---|
| gpt-oss:120b（medium） | xarray-4629 | 17反復で、出力の合計 10,000 を使い切った。1回あたり約600トークンを推論に使っている |
| nemotron-3-ultra（各社） | 14711 | 正しいファイルは2ステップ目で読めている。しかし、最初の編集が15〜27ステップ目と遅く、30反復で止まった |
| Cohere command-a-plus・reasoning（ON） | 全部 | 1回目のリクエストから HTTP 422 `NO_VALID_RESPONSE_GENERATED`（"No valid response generated. Try updating messages"）。`stop` を外しても同じ |
| Cohere command-a-reasoning（none） | 14711・xarray | 1〜2反復目のあと 422 |
| gemini-3.1-flash-lite | 14711 | 6反復目に自分で提出したが、隠れたテストで落ちた |
| gemini-3.5-flash | 全部 | 1日の枠切れ（429） |

## 3. プロバイダの信頼性

| 設定 | 結果 | 1リクエストの平均応答時間(秒) | リクエスト数 | リトライ | 失敗したリクエスト | 成功率 | 提供側の失敗で終わらなかった回 |
|---|---|---|---|---|---|---|---|
| Mistral / codestral-latest | 3/3 | **1.4** | 11 | 0 | 0 | 100% | 3/3 |
| NVIDIA / glm-5.3-flash | 3/3 | 27.3 | 16 | 0 | 0 | 100% | 3/3 |
| Ollama / gpt-oss:120b（low） | 3/3 | **1.2** | 40 | 0 | 0 | 100% | 3/3 |
| Ollama / gpt-oss:120b（medium） | 2/3 | 2.3 | 39 | 0 | 0 | 100% | 3/3 |
| Cohere / command-a-03-2025 | 3/3 | 8.7 | 21 | 1 | 1 | 95% | 3/3 |
| Cohere / command-a-plus-05-2026 | 0/3 | - | 3 | 0 | 3 | 0% | 0/3 |
| Cohere / command-a-reasoning（none） | 2/3 | 2.7 | 9 | 0 | 2 | 78% | 1/3 |
| Cohere / command-a-reasoning（ON） | 0/3 | 10.5 | 5 | 0 | 3 | 40% | 0/3 |
| Kilo / nemotron-3-ultra（OFF） | 3/3 | 12.2 | 47 | 1 | 1 | 98% | 3/3 |
| Kilo / nemotron-3-ultra（ON） | 2/3 | 6.4 | 46 | 0 | 1 | 98% | 2/3 |
| NVIDIA / nemotron-3-ultra（OFF） | 2/3 | 4.2 | 52 | 0 | 0 | 100% | 3/3 |
| NVIDIA / nemotron-3-ultra（ON） | 3/3 | 5.6 | 46 | 1 | 1 | 98% | 3/3 |
| Ollama / nemotron-3-ultra（none） | 3/3 | 4.8 | 47 | 0 | 0 | 100% | 3/3 |
| Ollama / nemotron-3-ultra（medium） | 2/3 | 7.3 | 82 | 1 | 1 | 99% | 3/3 |
| Google / gemini-3.1-flash-lite（medium） | 2/3 | 13.5 | 12 | 0 | 0 | 100% | 3/3 |
| Google / gemini-3.5-flash | 0/3 | 3.0 | 12 | 0 | 6 | 50% | 0/3 |

- **Mistral・Ollama は、失敗したリクエストが0件**です。応答も 1.2〜4.8秒と速いです。
- **Cohere は、モデルによって大きく違います**。
  - command-a は 95%（1件の失敗は、リトライで回復）でした。
  - plus と reasoning は、422 でほとんど使えませんでした。
  - 10-30 の試しでも、command-a が途中から 422 になる回がありました。
  - 試用キーは月 1,000回までです。
  - 1回の応答が 8.7秒と遅めです。選定では 93.9秒かかった応答もありました。
- **NVIDIA は、落ちませんが遅いことがあります**。glm-5.3-flash は1回 27秒、nemotron-3-ultra は 4〜6秒でした。確かめのとき、推論 ON で1回だけ 503 が出ました。
- **Kilo** は、OpenRouter と同じ形のゲートウェイです。`:free` のモデルは、予告なく一覧から消えます（laguna-s が計測の日に消えた）。
- **Google** は、gemini-3.5-flash の1日の枠が小さすぎます。3問を回し切れませんでした。

## 4. 中間指標

| 設定 | タスク | 最終パッチのファイルを初めて読んだステップ | 初めて編集したステップ | テストが全部通ってから final_answer までの反復 |
|---|---|---|---|---|
| Mistral / codestral-latest | xarray-4629 | 2 | 3 | 1 |
| Mistral / codestral-latest | sympy-13480 | 2 | 3 | 0 |
| Mistral / codestral-latest | sympy-14711 | 2 | 4 | 0 |
| NVIDIA / glm-5.3-flash | xarray-4629 | 2 | 3 | 1 |
| NVIDIA / glm-5.3-flash | sympy-13480 | 2 | 3 | 1 |
| NVIDIA / glm-5.3-flash | sympy-14711 | 1 | 3 | 1 |
| Ollama / gpt-oss:120b（low） | xarray-4629 | 12 | 13 | - |
| Ollama / gpt-oss:120b（low） | sympy-13480 | - | 3 | 1 |
| Ollama / gpt-oss:120b（low） | sympy-14711 | 3 | 11 | 3 |
| Ollama / gpt-oss:120b（medium） | sympy-13480 | 4 | 5 | 1 |
| Ollama / gpt-oss:120b（medium） | sympy-14711 | 3 | 5 | 1 |
| Cohere / command-a-03-2025 | xarray-4629 | 2 | 3 | 1 |
| Cohere / command-a-03-2025 | sympy-13480 | 2 | 3 | 1 |
| Cohere / command-a-03-2025 | sympy-14711 | 2 | 7 | 2 |
| Kilo / nemotron-3-ultra（OFF） | xarray-4629 | 1 | 2 | 6 |
| Kilo / nemotron-3-ultra（OFF） | sympy-13480 | 1 | 2 | 3 |
| Kilo / nemotron-3-ultra（OFF） | sympy-14711 | 2 | 27 | - |
| Kilo / nemotron-3-ultra（ON） | xarray-4629 | 3 | 5 | 3 |
| Kilo / nemotron-3-ultra（ON） | sympy-13480 | 2 | 3 | 2 |
| NVIDIA / nemotron-3-ultra（OFF） | xarray-4629 | 4 | 11 | 2 |
| NVIDIA / nemotron-3-ultra（OFF） | sympy-13480 | 1 | 2 | 3 |
| NVIDIA / nemotron-3-ultra（ON） | xarray-4629 | 2 | 4 | 3 |
| NVIDIA / nemotron-3-ultra（ON） | sympy-13480 | 2 | 3 | 1 |
| NVIDIA / nemotron-3-ultra（ON） | sympy-14711 | 2 | 22 | - |
| Ollama / nemotron-3-ultra（none） | xarray-4629 | 2 | 4 | 5 |
| Ollama / nemotron-3-ultra（none） | sympy-13480 | 1 | 2 | 2 |
| Ollama / nemotron-3-ultra（none） | sympy-14711 | 2 | 15 | - |
| Ollama / nemotron-3-ultra（medium） | xarray-4629 | 3 | 5 | 14 |
| Ollama / nemotron-3-ultra（medium） | sympy-13480 | 2 | 4 | - |
| Google / gemini-3.1-flash-lite（medium） | xarray-4629 | 1 | 1 | - |
| Google / gemini-3.1-flash-lite（medium） | sympy-13480 | 1 | 1 | - |
| Google / gemini-3.1-flash-lite（medium） | sympy-14711 | 1 | 5 | - |

- パッチを作らなかった回（422 で止まった回など）は省きました。
- `-` の意味:
  - 「読んだ」が `-`: `read_file` を使わずに、`run_command`（`sed -n`・`cat`）や `search_code` で中身を見た回です。
  - 最後の列が `-`: `run_tests()` が全部通ったのを見ないまま終わった回です。テストを回さずに提出した回と、反復の上限で止まった回があります。

**探索の効率**:

- 3問とも PASS した上位4設定は、正しいファイルを1〜2ステップ目で読み、3〜4ステップ目（14711 の Cohere は7）で編集しています。課題文が直す関数を名指ししているので、強いモデルはまっすぐそこへ行きます。
- 差が出たのは 14711 です。nemotron-3-ultra は、どの社でも2ステップ目には正しいファイルを読んでいます。それでも、最初の編集は15・22・27ステップ目でした。読んではいるのに、直す方針を決めきれず、確かめをくり返しています。
- gpt-oss:120b（low）の xarray は、最終パッチのファイル（`merge.py`）を12ステップ目に初めて読んでいます。それまでは `search_code` と `run_command` で、まわりを探していました。

**提出の規律**:

- codestral は、14711・13480 で **0**（テストを回して、通ったらそのまま同じステップで `final_answer(get_patch())`）でした。xarray は 1 です。
- Cohere command-a と glm-5.3-flash は、ほぼ 1 です（結果を読んでから提出）。
- nemotron-3-ultra は、通ってからも 2〜6回、medium では14回も確かめを続けました。
- SWE では、残りが4反復になると「探すのをやめて提出せよ」とエージェントが伝えます（`[BUDGET]`・`iteration_warn_at=4`）。それでも、反復の上限に当たった回がありました。
- gemini-3.1-flash-lite は、3問とも `run_tests()` が全部通るのを見ないまま提出しました（列が `-`）。14711 は、それで隠れたテストに落ちています。

## 5. アブレーション（候補）【要判断：どれを残すか】

課題の要件は「エージェントへの変更（プロンプト・ツール・パラメータ）を、同じタスク・同じモデルで、前後で比べたもの」を1つ以上です。使える候補を並べます。

### 5.1 推論の強さ（パラメータ）: 同じモデル・同じ3問 ★勧め

変えたのは、リクエストの推論の設定（`model_params`）だけです。

| モデル | 弱い／OFF | 強い／ON | 差 |
|---|---|---|---|
| Ollama gpt-oss:120b（low → medium） | 3/3・平均13反復・出力 1,545・29秒 | **2/3**・13反復・出力 **6,451**・55秒 | medium は出力が4倍になり、xarray で上限 10,000 に当たって FAIL |
| Ollama nemotron-3-ultra（none → medium） | 3/3・16反復・入力 59,669・100秒 | **2/3**・**27反復**・入力 **118,960**・260秒 | 反復と入力が約2倍。2問が反復の上限 |
| NVIDIA nemotron-3-ultra（OFF → ON） | 2/3・17反復・入力 72,073 | 3/3・15反復・入力 52,145 | ON がやや良い。ただし 14711 の PASS は回収パッチ |
| Kilo nemotron-3-ultra（OFF → ON） | 3/3・15反復・出力 4,413 | 2/3・15反復・出力 2,088 | OFF がやや良い。ON の 14711 は応答が止まって FAIL |
| Cohere command-a-reasoning（none → ON） | 2/3 | 0/3 | ON は 422 で使えない |

**分かったこと**:

- **推論を強くしても、合格は増えませんでした**。5組の比べのうち、強いほうが良かったのは NVIDIA の1組だけです。それも回収パッチの PASS でした。
- **出力の上限（10,000）が、推論には狭い**:
  - SWE は1問に10〜30回呼びます。1回あたりに使える出力は、330〜1,000 トークンです。
  - gpt-oss medium は1回平均約500トークンを使い、xarray では17反復で上限に当たりました。
  - 課題 PDF VI.1.2 の「推論で上限が厳しくなるなら、推論しないモデルも検討せよ」のとおりです。
- **推論すると、反復が増えることもあります**（ollama ultra: 16 → 27）。考えた分、試す回数も増えています。
- 推論するモデルで良かったのは glm-5.3-flash（3/3・平均5反復）です。1回で考えて少ない反復で決める型ですが、1回 27秒と遅いです。
- **注意**: どれも1回ずつ（n=1）です。同じ設定でも、反復の数はかなり揺れます（§7 の codestral の 14711 は 4・21・6反復）。

### 5.2 本番の並び（設定ファイル）: cohere が1番手 と codestral が1番手

- 同じ6問・同じ版で、`providers_swebench.json` の並びだけを変えて比べます。
  - 前: cohere command-a が1番手（今の本番）
  - 後: codestral が1番手
- 回数は、cohere が1番手の並びが3周、codestral が1番手の並びが4周です。
- モデルの並びは「エージェントの設定」の変更ですが、1回の中で複数のモデルが答えるので、「同じモデル」の条件からは少し外れます。結論（§6）の裏付けとして使う案です。
- **注意（このまま載せるなら断りが要ります）**: cohere の試用キーが、1周目の途中で月 1,000回の上限に達しました（429 "Trial key … limited to 1000 API calls / month"）。
  - cohere が答えたのは、1周目の最初の4問だけです。
  - それ以降は、2番手の NVIDIA nemotron-3-ultra が答えています（同じ NVIDIA の nemotron-3-super は、このアカウントでは 404 で飛ばされる）。
  - したがって、この比べは実質「codestral が1番手 対 nemotron-3-ultra が1番手（cohere は枠切れ）」です。
  - 結果は §7 にあります。

### 5.2b 14711 だけを9回ずつ（1モデルだけ・フォールバックなし）★勧め（モデル選びの決め手）

- 3問の選定では、多くのモデルが通り、差が出たのは 14711 だけでした。そこで、14711 だけを上位の候補で9回ずつ回しました（選定の1回＋8回）。

| 設定 | 14711 の PASS |
|---|---|
| NVIDIA glm-5.3-flash | **9/9** |
| Ollama gpt-oss:120b（推論 low） | **9/9** |
| Ollama gpt-oss:120b（推論 medium） | **9/9** |
| Google gemini-3.1-flash-lite（推論 medium） | 6/9 |
| Mistral codestral | 3/9 |

- codestral は選定の1回では通しましたが、9回では 3/9 でした。**1回だけの計測では、モデルの強さを見誤る**例です。
- codestral の落ち方の多くは、3〜5反復で自信をもって間違ったパッチを提出するものでした。ほかに、出力の上限に当たる回もありました。

### 5.3 MBPP: エージェントの直し①〜③（プロンプト・提出の検査・Observation）

- 版: `da749ed`（前）と、ローカルのブランチ `5cf0984`（後）。後のブランチは計測用で、提出には入れていません。
- 変えたこと:
  1. プロンプトの例と指示を、生の三重引用符 `r"""…"""` にした（正規表現の `\1`・`\b` が壊れていた）
  2. 直書きの検査を、引数が見えているテストの入力と同じときだけにした（誤検知を減らす）
  3. `import` を断ったとき、Observation に直し方を書く
- 同じ MBPP の10問（set10）× 2回 × 45モデルで比べました。

| | 前（`da749ed`） | 後（`5cf0984`） |
|---|---|---|
| 応答があった回 | 738 | 731 |
| PASS | 572（**77.5%**） | 566（**77.4%**） |
| 入力上限で落ちた回 | 115 | **93** |
| 出力上限で落ちた回 | 10 | 9 |

- **合格率は変わりませんでした**。入力上限で落ちる回は、約2割減りました。
- 直しが狙ったのは、全257問のうち4問（396・454・286・468）だけです。10問の中には入っていないので、効果は見えにくい比べです。
- 結論: 提出には入れていません（チームのコードを変えないため）。

### 5.4 MBPP: プロンプトの短縮＋送り直す履歴を2往復から1往復に（プロンプト・パラメータ）

- 版: `5cf0984`（前）と、ローカルのブランチ `41c982d`（後）。どちらも計測用です。
- 変えたこと:
  - system プロンプトを約4,210文字から約3,230文字に短くした
  - MBPP で送り直す履歴を、直近2往復から1往復にした
- 比べた対象: これまでに入力上限（6,000）で一度でも落ちた（モデル, 問題）の組、227組です。

| | 前（同じ組の、それまでの全部の回） | 後（各組1回） |
|---|---|---|
| 応答があった回 | 630 | 174 |
| PASS | 42% | 46% |
| 入力上限で落ちた | **51%** | **34%** |
| 出力上限で落ちた | 3% | 8% |
| 入力上限で落ちた回の反復 | 3〜4回 | 4〜5回 |

- 1回目の入力は、1,132 から 930 トークンに減りました。
- 入力上限は減りましたが、その分だけ反復が増え、**出力の合計（1,500）で落ちる回が増えました**。
- 前の数字は「入力上限で落ちた組」を選んで数えているので、前のほうが悪く出る偏りがあります。
- 結論: **入れませんでした**。続けて直すには1回の出力（600）を下げることになり、推論するモデルが途中で切れやすくなるからです。

### 5.5 MBPP: プロバイダの並び（設定ファイル）

- 同じ257問（試験の範囲の全問）・同じ版で、並びだけを変えて比べます。2つの並びを、1問ごとに交互に回しています。

| 並び | 1番手 → 2番手 → … |
|---|---|
| A（今の本番） | codestral → groq（gpt-oss-120b, qwen3.8, gpt-oss-20b）→ ollama → cohere → gemini → nvidia → openrouter |
| C（案） | codestral → groq qwen3.8 → ollama（gpt-oss:120b, nemotron-3-ultra）→ gemini → kilo → nvidia → cloudflare |
| B（参考。別の時間に1回） | groq qwen3.8 → kilo nemotron-ultra → … → **224/257（87.2%）**。groq が途中で1日の枠に当たり、半分の問題は kilo が1番手になった |

| 並び | PASS（257問） |
|---|---|
| A（今の本番） | **234**（91.1%） |
| C（案） | **237**（92.2%） |
| 参考: 10-30 にメンバーが今の本番で回した1周 | 233（90.7%） |

**結果が分かれた9問の中身**:

| 分かれ方 | 問題 | A だけ PASS | C だけ PASS |
|---|---|---|---|
| 3反復目から2番手が答え、そこで分かれた（**並びの影響**） | 72・101・396・260・448 | 72・101・396 | 260・448 |
| 1・2反復目の codestral の答えが違った（**生成の揺れ**。並びとは無関係） | 286・306・422・440 | - | 286・306・422・440 |

- 並びの影響と言えるのは5問で、**A 3・C 2 とほぼ互角**です。差の3問は、生成の揺れの範囲です。
- MBPP は、1・2反復目は必ず1番手（codestral）が答えます。3反復目以降に2番手へ切り替わります（`rotate_after_missed=2`）。そのため、2番手を替えて効くのは、codestral が2回で解けない問題だけです。
- 結論: 正答率では A と C を区別できませんでした。C を選ぶ理由は次の3つです。
  - cohere の試用の枠に頼らない
  - 2番手が hard10 で1位の qwen3.8
  - 少なくとも悪くはない

### 5.6 参考: 10-15 のテストブランチでのアブレーション

- 提出の版とは別の版（`test/mbpp-tuning`）での計測です。
- 載せるなら「同じ考え方の変更を、別の版で測った」と断る必要があります。

| 変更 | 結果 |
|---|---|
| `run_tests()` の出力を要約する | 3問では差なし |
| ツールの出力の誤り（パスを指定した検索が常に "No matches" になる、など）を直す | ログを再生して確かめた。本番の回で効果を見るには、件数が足りなかった |
| 再現スクリプトを、入れ子の引用符なしで書く例をプロンプトに足す | 引用符の失敗 13 → 3、反復 −18%、入力トークン −24% |

## 6. 結論

**決め方**: 速さより正答率を優先しました。手順は次の3段です。

1. 簡単な3問で、16設定から候補を絞る（§2）
2. 差の出た 14711 を9回ずつ回す（§5.2b）
3. 9/9 だった3モデルで、6問 × 3周を回す（§7）

**SWE-bench の本番の並び**（`providers/providers_swebench.json`。この順に試す）:

1. **Ollama `gpt-oss:120b`（推論 low）** → 同じ Ollama の `nemotron-3-ultra`（推論 none）
   - 6問 × 3周を1モデルだけで **18/18**、14711 は **9/9** でした。
   - 18回とも自分で提出し、上限にも当たっていません。
   - 1問平均 32秒・1回 1.2秒と、速さも上位です。
   - `nemotron-3-ultra` は、gpt-oss が応答しなかったときの同じ社の予備です。
2. **NVIDIA `glm-5.3-flash`** → `nemotron-3-ultra`（推論 ON）
   - 17/18（落ちた1回は通信エラー）、14711 は 9/9 でした。
   - 1回 27秒と遅いので2番手です。
   - 1番手と別の社にしたのは、同じコードのくり返しで切り替わったときに、違う間違い方をするモデルに渡すためです。
3. **Mistral `codestral-latest`**
   - 速くて枠も大きいですが、14711 は 3/9、scikit-learn は 1/4 でした。自信をもって間違ったパッチを出すことがあるので、3番手です。
4. Google `gemini-3.1-flash-lite`（推論 medium）: 3問で 2/3、14711 は 6/9。
5. Kilo `nemotron-3-ultra:free`（推論 OFF）: `:free` は消えることがあるので後ろに置きます。
6. Cohere `command-a-03-2025`
   - 3問では 3/3 でした。しかし、試用キーの月 1,000回の上限があり、計測中にその上限に達しました。
   - 1回の応答も遅いので、最後の保険にします。

- 確認: この並びのファイルで、6問 × 2周を回して **12/12** でした（§7）。

**比べた並びのまとめ**（6問）:

| 並び（1番手） | 結果 |
|---|---|
| cohere command-a（今の本番。実質は nemotron-ultra） | 15/18（14711 は 0/3） |
| codestral | 19/24（14711 は 5/9、scikit-learn は 1/4） |
| **gpt-oss:120b（low）単体** | **18/18** |

**使わないもの**:

| モデル | 理由 |
|---|---|
| Cohere command-a-plus・command-a-reasoning（推論 ON） | 毎回 422 |
| Google gemini-3.5-flash | 1日の枠が小さく、3問を回し切れない |
| 推論を強くした設定（gpt-oss medium・nemotron medium） | 合格が増えず、出力・反復が増える。gpt-oss は出力の上限に当たった |
| Groq | 1分あたり 8,000 トークンでは、SWE の1回を 900秒に収められない |
| minimax・deepseek（NVIDIA） | 1回の応答が 80〜100秒 |

**データから言えること**:

- SWE の上限（出力 10,000）の中では、**推論を強くしても合格は増えませんでした**。同じ gpt-oss:120b でも、low の 18/18 に対して medium は 16/18 です。
- **1回だけの計測では、モデルの強さを見誤ります**。
  - codestral は、選定の1回では 14711 を通しましたが、9回では 3/9 でした。
  - 3問の選定（n=1）は候補を絞るのに使い、最後は回数を重ねて決めました。
- 簡単な問題（13480・django など）は、上位のどのモデルでも解けます。並びの差は 14711・scikit-learn のような、迷いやすい問題で出ます。
- 「提出が早いこと」は、良いことばかりではありません。codestral は少ない反復で提出しますが、間違ったパッチを出すと、そこで回が終わります。フォールバックは、それを救えません。

## 7. 本番の設定での確認（全6問）

- moulinette の試験の候補（`EXAM_POOL`）の6問を、本番の並び（フォールバックあり）で回しました。

| 並び | 回数 | 結果 |
|---|---|---|
| codestral が1番手（mistral → ollama → nvidia → cohere → kilo → gemini） | 4周（r1〜r4） | 6・5・4・4 → **19/24（79%）** |
| cohere が1番手（今の `providers_swebench.json`。cohere は1周目の途中から枠切れで、実質 nemotron-3-ultra が1番手） | 3周（r1〜r3） | 5・5・5 → **15/18（83%）** |
| 上位3モデル（14711 で 9/9）を1モデルだけで × 6問 × 3周 | 下の表 | gpt-oss low **18/18**・glm 17/18・gpt-oss medium 16/18 |
| **最終の並び（gpt-oss が1番手。§6）** | 2周（11-02 07:12〜07:26） | **12/12**（平均 9.5反復・入力約30,500・出力約1,260・35秒）。答えたのは gpt-oss:120b だけで、xarray の1回だけ途中から glm-5.3-flash に切り替わって通った |

**上位3モデル × 6問 × 3周（1モデルだけ・フォールバックなし・11-02 02:15〜02:58）**:

| タスク | gpt-oss:120b（low） | glm-5.3-flash | gpt-oss:120b（medium） |
|---|---|---|---|
| django-11066 | 3/3（5〜7反復・15〜25秒） | 3/3（4〜5・17〜46秒） | 3/3（5〜18・16〜36秒） |
| xarray-4629 | 3/3（7〜8・20〜22秒） | 3/3（4〜6・52〜66秒） | 3/3（4〜7・17〜31秒） |
| scikit-learn-13439 | 3/3（7〜9・22〜24秒） | 3/3（5〜10・44〜93秒） | 2/3（1回は反復の上限で FAIL、1回は上限の回収パッチで PASS） |
| sympy-13480 | 3/3（3〜6・31〜40秒） | 3/3（4〜5・43〜270秒） | 3/3（4〜9・11〜55秒） |
| sympy-14711 | 3/3（7〜10・19〜42秒） | 2/3（1回は NVIDIA の通信エラー〔TransportError〕で応答が止まる） | 2/3（出力の上限 10,000 に2回当たり、1回は回収パッチで PASS） |
| sympy-18189 | 3/3（3〜11・59〜61秒） | 3/3（4〜5・59〜111秒） | 3/3（5〜28・51〜195秒） |
| **合計** | **18/18** | **17/18** | **16/18** |
| 平均（1問） | 7反復・入力約21,000・出力約740・32秒 | 6反復・入力約21,000・出力約1,240・105秒 | 14反復・入力約49,000・出力約4,030・51秒 |

- **gpt-oss:120b（low）は、18回とも自分で `final_answer` を出して通しました**。反復の上限・出力の上限に一度も当たっていません。
- 同じ gpt-oss:120b でも、推論 medium にすると、反復は約2倍、出力は約5倍に増えます。出力の上限（10,000）や反復の上限（30）に当たって、2回落ちました（§5.1 と同じ傾向）。
- glm-5.3-flash は、モデルとしては全部解けています。落ちた1回は、NVIDIA の API の通信エラーで応答が止まったものです。1問あたり平均105秒と遅めです。

**問題ごと（PASS の回数）**:

| タスク | codestral が1番手（4周） | cohere が1番手＝実質 nemotron-ultra（3周） |
|---|---|---|
| django-11066 | 4/4 | 3/3 |
| xarray-4629 | 3/4（r4 が出力の上限） | 3/3 |
| scikit-learn-13439 | 1/4 | 3/3 |
| sympy-13480 | 4/4 | 3/3 |
| sympy-14711 | 3/4（14711 だけの追加5回を含めると 5/9） | 0/3 |
| sympy-18189 | 4/4 | 3/3 |

- **codestral が1番手の並びの弱点**:
  - scikit-learn（1/4）: 3〜4反復で間違ったパッチを提出する回が2回、`edit_file` の失敗をくり返して出力の上限に当たる回が1回ありました。
  - 14711（5/9）: 間違ったパッチを提出する回と、出力の上限に当たる回がありました。
  - codestral は提出が早い（速くて安い）一方で、自信をもって間違えることがあります。間違ったパッチを提出すると、そこで回が終わるので、フォールバックが働く前に負けます。
- **nemotron-ultra が1番手のとき**は、14711 以外は全部通りました（15/15）。ただし 14711 は3回とも、30反復の上限で止まりました。

**codestral が1番手の r1・r2**:

| タスク | r1 | r2 |
|---|---|---|
| sympy-14711 | PASS（21反復・57秒。codestral 7回 → 同じコードの繰り返しで切り替え → gpt-oss 14回） | PASS（6・37秒） |
| sympy-13480 | PASS（4・23秒） | PASS（4・29秒） |
| xarray-4629 | PASS（4・15秒） | PASS（4・15秒） |
| django-11066 | PASS（4・17秒） | PASS（4・19秒） |
| scikit-learn-13439 | PASS（5・28秒） | **FAIL**（出力の上限） |
| sympy-18189 | PASS（4・57秒） | PASS（3・54秒） |

- r2 の scikit-learn の FAIL:
  - codestral が `edit_file` の置き換え元の文字列を合わせられず、メソッドを丸ごと書き写す応答（1回 1,200トークンの上限まで）を6回出しました。
  - 毎回少しずつ違うので、「同じコードが2回続くと切り替え」がすぐには働きませんでした。
  - gpt-oss に切り替わったときには、出力の残りがほとんどありませんでした。
  - エージェント側の改善の候補です（今回は入れていません）。

## 8. MBPP について（参考。課題のレポートの要件ではない）

- 試験の範囲（test split の257問）を全部回した結果です。
  - 10-30（メンバー、今の本番の並び）: 233/257（90.7%）
  - 11-01（並び B）: 224/257（87.2%）
  - 11-02（A と C を1問ずつ交互）: A 234（91.1%）・C **237**（92.2%）→ **本番は C**（理由は §5.5）
- 試験は「5問中4問」です。1問の成功率が 0.92 なら、合格の見込みは約95%です。

## 9. データの置き場所

`benchmark/swebench/` に、201回分を置きました（約 39MB）。

- 1回ごとに、次の6つがあります。
  - `solution.json`
  - `task.json`
  - `validate.log`（採点の全文）
  - `validate_summary.txt`（`validate.log` から判定の行だけを抜き出したもの）
  - `agent_stdout.log`・`agent_stderr.log`（エージェントの標準出力・標準エラー）
- 設定ごとに、その回に使った `providers.json` があります。

| フォルダ | 中身 | この文書の節 |
|---|---|---|
| `selection_easy3/<設定>/r1/<タスク>/` | 簡単な3問 × 16設定 | §2〜§5.1 |
| `task14711_repeats/<設定>/r<回>/sympy__sympy-14711/` | 14711 だけの追加の回（r2〜r9。本番の並びの案は r5〜r9） | §5.2b |
| `top3_6tasks/<設定>/r<回>/<タスク>/` | 上位3モデル × 6問 × 3周 | §7 |
| `production_codestral_first/swe_final/r<回>/<タスク>/` | codestral が1番手の並び × 6問 × 4周 | §5.2・§7 |
| `production_cohere_first/swe_cohere/r<回>/<タスク>/` | cohere が1番手の並び × 6問 × 3周 | §5.2・§7 |
| `production_final/swe_final2/r<回>/<タスク>/` | 最終の並び × 6問 × 2周 | §6・§7 |

- 計測のスクリプトは `benchmark/swebench/scripts/` にあります。
  - `swe_run.sh`（1回を回して採点する）・`swe_summary.py`（集計）・`report_tables.py`（この文書の表）・`reasoning_probe.py`／`reasoning_probe.json`（推論 ON の事前の確かめ）・`export_benchmark.py`（このフォルダへの書き出し）
  - `compare_14711.sh`・`s6_top3.sh`・`after_mbpp_final.sh`（§5.2b・§7 の回を順に回したもの）
  - 計測は別の作業ツリー（`da749ed` を取り出したもの）で回しました。スクリプトの中のパス（`runs/` など）は、その作業ツリーの置き方のままです。
- MBPP の生データ（257問 × 何周か）は、大きいので置いていません。集計の表だけを §5.5・§8 に載せています。
