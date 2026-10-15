# Model Benchmark Report — SWE-bench

*Agent Smith — measured on 2026-10-15. Every number below comes from a real run. The backing
`solution.json` files are in [`benchmark/swebench/`](benchmark/swebench/), next to each task's
`task.json`, the validator's verdict (`validate_summary.txt`) and the single-model `providers.json` used.
The tables were generated with `uv run python tools/benchmark_summary.py <directory>`.*

## Summary

- **Compared:** 6 models from 3 providers on the three tasks the subject recommends, one model per run
  and no fallback (§1–§4). Later the same day we added 3 candidates from 2 more providers: Mistral and
  Ollama Cloud (§8).
- **Selected for the final pipeline** (`providers/providers_swebench.json`):
  1. Cohere `command-a-03-2025`
  2. NVIDIA `nemotron-3-super-120b-a12b`
  3. Ollama Cloud (`nemotron-3-super`, `gemma4:31b`)
  4. Mistral and Gemini as last resorts

  command-a and nemotron-3-super solved all three tasks without a failed request. command-a used the
  fewest iterations and tokens and answered fastest (2.2 s per request). nemotron-3-super passed 26 of
  its 27 runs, and none of its 555 requests failed.
- **Disregarded:**
  - `gemma-4-31b-it` on Google AI Studio: the API failed 30 of 57 requests. The same model on Ollama
    passed 3/3.
  - `openai/gpt-oss-20b` on NVIDIA: it often answers in its own tool-call format instead of Python, and
    the endpoint ended one run with HTTP 400.
  - Groq: 8,000 tokens per minute per account cannot carry one SWE-bench run within 900 s.
- **Exam reproduction:** we ran all six tasks of the moulinette's exam pool through its own
  `quickstart.sh`, with the production configuration, before and after the fixes. Both runs passed
  6/6 within the limits, and the final code needed 34% fewer input and 76% fewer output tokens (§7).
- **Ablations** (§5):
  - Summarizing the `run_tests()` output: no measurable difference on these tasks.
  - Fixing misleading tool feedback (for example, a search limited to a path always returned "No
    matches"): proven by replaying the logged calls, too rare to show in end-to-end runs.
  - Showing how to write a reproduction script without nested quotes: quoting failures fell from 13 to
    3, iterations by 18% and input tokens by 24%.

## 1. Setup

### 1.1 How a run is made

For each (model, task) pair, `tools/harness_swebench.py` runs three steps:

1. `moulinette_eval dump swebench --task-id <id>` writes `task.json`.
2. `python -m agent_swebench --task-file task.json --output solution.json --providers_file_path <file>`.
   The providers file contains **only the model under test**.
3. `moulinette_eval validate swebench task.json solution.json` applies the patch, runs the tests and
   checks the limits.

**One model per run, no fallback.** Each result therefore belongs to exactly one model. The production
configuration falls back across providers; it is measured separately in §7.

The limits are the exam's: 30 iterations, 300,000 input tokens, 10,000 output tokens and 900 s, all
cumulative per task. The agent enforces them itself, and the validator checks them again. Every run in
this report is `Metrics: VALID`.

| Run | Agent commit | Content |
|---|---|---|
| v1 | `1163b6d` | all 6 models × 3 tasks |
| v2 | `9f1ebef` | the 3 models that failed a task in v1, after two fixes (§5.4) |
| Ablation 1 | `73de8c7` | `run_tests()` output summary on / off |
| Ablation 2 | `8e54ca5` → `0c080b4` | tool feedback fixes, before / after (2 runs each) |
| Ablation 3 | `0c080b4` → `5e2f7cd` | prompt: how to write a reproduction script (2 runs each) |
| Exam reproduction | `8e54ca5`, `1ac6023` | production configuration, all 6 exam-pool tasks, before and after the fixes |

### 1.2 Models and providers: 6 models from 3 providers

| Provider (free tier) | Model | Extra request parameters |
|---|---|---|
| Cohere (trial key) | `command-a-03-2025` | — |
| NVIDIA NIM (build.nvidia.com) | `nvidia/nemotron-3-super-120b-a12b` | `chat_template_kwargs: {enable_thinking: false}` |
| NVIDIA NIM | `nvidia/nemotron-3-ultra-550b-a55b` | — |
| NVIDIA NIM | `openai/gpt-oss-20b` | — |
| Google AI Studio (Gemini API) | `gemini-3.5-flash-lite` | `reasoning_effort: low` (v1), `minimal` (v2) |
| Google AI Studio (Gemini API) | `gemma-4-31b-it` | — |

All models are called through the same OpenAI-compatible client (`src/agent/llm/`). The prompt, the tools
and the loop are therefore identical for every model; only the model name and its request parameters
change. Thinking is switched off or minimized for the two models that think by default
(nemotron-3-super, gemini-3.5-flash-lite). The reason: the limits count reasoning tokens, and a thinking
model can hit the `<end_code>` stop sequence inside its thoughts and return an empty answer.

We have keys for two more providers that are not in the table:

- **Groq.** The free tier allows 8,000 tokens per minute, and the limit is per *account*: we measured
  that adding keys does not add capacity. A successful SWE-bench run here used 11,000–127,000 input
  tokens. At 8,000 tokens per minute, the larger runs would need 13–16 minutes of quota, more than the
  900 s limit. Groq is used for MBPP only.
- **Mistral.** It is the MBPP primary (`codestral-latest`), but its API key was rejected (HTTP 401)
  during the v1/v2 runs. After the key was renewed the same day, codestral was measured with the
  later candidates (§8).

### 1.3 Tasks

We used `sympy__sympy-14711`, `sympy__sympy-13480` and `pydata__xarray-4629`, for three reasons:

- The subject recommends them as the simplest tasks for initial testing.
- All three are in the moulinette's exam pool (`EXAM_POOL`). Their results therefore estimate the exam
  directly.
- They cover two repositories, two test runners (sympy's `bin/test` and pytest) and three kinds of fix:
  - 14711: a missing zero case in `Vector.__add__`
  - 13480: a typo, `cotm` instead of `cothm`
  - xarray-4629: `merge` returns the first object's `attrs` dict instead of a copy

### 1.4 Metrics

| Metric | Definition |
|---|---|
| Iterations, input / output tokens | from `solution.json`: provider-reported usage, summed over the run |
| Time (s) | `total_time_seconds` in `solution.json`. It is within 0.6 s of the process wall-clock time measured by the harness, for all 33 runs where both exist |
| Avg response time per request | mean `request_time_ms` over the steps, i.e. over the successful LLM responses. It includes the time of retries inside a step |
| Requests / failed requests | all HTTP attempts, including retries / the attempts that returned no usable response: retried ones, plus those of a call that ended the run |
| Request success rate | successful responses ÷ requests |
| Runs not ended by provider failure | runs whose stop reason is not `LLM_UNAVAILABLE` |
| First read / first edit (step) | first step whose code calls `read_file` / `edit_file` on a file that appears in the final patch. Measures exploration efficiency |
| Iterations from tests passing to `final_answer` | final iteration minus the first step in which `run_tests()` reported every test passing. Measures submission discipline: 0 is ideal, and `-` means no fully green `run_tests()` was seen before the run ended |

## 2. Results

### 2.1 v1: all models (commit `1163b6d`)

| Model / provider | Task | Result | Iterations | Input tokens | Output tokens | Time (s) | Stop reason |
|---|---|---|---|---|---|---|---|
| cohere / command-a-03-2025 | pydata__xarray-4629 | PASS | 5 | 15,737 | 388 | 16 | final_answer |
| cohere / command-a-03-2025 | sympy__sympy-13480 | PASS | 5 | 11,315 | 308 | 19 | final_answer |
| cohere / command-a-03-2025 | sympy__sympy-14711 | PASS | 15 | 52,497 | 2,507 | 51 | final_answer |
| gemini / gemini-3.5-flash-lite | pydata__xarray-4629 | PASS | 1 | 2,674 | 519 | 19 | final_answer |
| gemini / gemini-3.5-flash-lite | sympy__sympy-13480 | FAIL | 1 | 6,837 | 600 | 7 | LLM_UNAVAILABLE |
| gemini / gemini-3.5-flash-lite | sympy__sympy-14711 | FAIL | 2 | 10,742 | 1,355 | 19 | LLM_UNAVAILABLE |
| gemini / gemma-4-31b-it | pydata__xarray-4629 | FAIL | 3 | 9,616 | 1,231 | 293 | LLM_UNAVAILABLE |
| gemini / gemma-4-31b-it | sympy__sympy-13480 | FAIL | 6 | 17,268 | 1,882 | 241 | LLM_UNAVAILABLE |
| gemini / gemma-4-31b-it | sympy__sympy-14711 | FAIL | 2 | 4,510 | 912 | 169 | LLM_UNAVAILABLE |
| nvidia / nemotron-3-super-120b-a12b | pydata__xarray-4629 | PASS | 11 | 41,636 | 1,420 | 49 | final_answer |
| nvidia / nemotron-3-super-120b-a12b | sympy__sympy-13480 | PASS | 14 | 52,331 | 3,816 | 122 | final_answer |
| nvidia / nemotron-3-super-120b-a12b | sympy__sympy-14711 | PASS | 14 | 58,513 | 1,618 | 53 | final_answer |
| nvidia / nemotron-3-ultra-550b-a55b | pydata__xarray-4629 | PASS | 17 | 89,057 | 1,706 | 130 | final_answer |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-13480 | PASS | 6 | 14,859 | 443 | 36 | final_answer |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-14711 | FAIL | 30 | 97,289 | 3,193 | 151 | MAX_ITERATIONS |
| nvidia / openai/gpt-oss-20b | pydata__xarray-4629 | PASS | 11 | 42,443 | 1,174 | 25 | final_answer |
| nvidia / openai/gpt-oss-20b | sympy__sympy-13480 | PASS | 9 | 22,932 | 843 | 23 | final_answer |
| nvidia / openai/gpt-oss-20b | sympy__sympy-14711 | FAIL | 30 | 108,218 | 2,419 | 37 | MAX_ITERATIONS |

| Model | 14711 | 13480 | xarray-4629 | Solved |
|---|---|---|---|---|
| command-a-03-2025 | ✅ | ✅ | ✅ | **3/3** |
| nemotron-3-super-120b-a12b | ✅ | ✅ | ✅ | **3/3** |
| nemotron-3-ultra-550b-a55b | ❌ | ✅ | ✅ | 2/3 |
| gpt-oss-20b | ❌ | ✅ | ✅ | 2/3 |
| gemini-3.5-flash-lite | ❌ | ❌ | ✅ | 1/3 |
| gemma-4-31b-it | ❌ | ❌ | ❌ | 0/3 |

### 2.2 v2: the models that failed a task, re-run after fixes (commit `9f1ebef`)

| Model / provider | Task | Result | Iterations | Input tokens | Output tokens | Time (s) | Stop reason |
|---|---|---|---|---|---|---|---|
| gemini / gemini-3.5-flash-lite | pydata__xarray-4629 | PASS | 6 | 25,673 | 2,037 | 46 | final_answer |
| gemini / gemini-3.5-flash-lite | sympy__sympy-13480 | PASS | 8 | 33,585 | 291 | 23 | final_answer |
| gemini / gemini-3.5-flash-lite | sympy__sympy-14711 | FAIL | 2 | 4,877 | 540 | 13 | final_answer |
| gemini / gemma-4-31b-it | pydata__xarray-4629 | FAIL | 7 | 37,899 | 5,741 | 350 | LLM_UNAVAILABLE |
| gemini / gemma-4-31b-it | sympy__sympy-13480 | FAIL | 2 | 4,258 | 766 | 159 | LLM_UNAVAILABLE |
| gemini / gemma-4-31b-it | sympy__sympy-14711 | FAIL | 7 | 24,688 | 2,174 | 150 | LLM_UNAVAILABLE |
| nvidia / nemotron-3-ultra-550b-a55b | pydata__xarray-4629 | PASS | 15 | 74,145 | 1,779 | 87 | final_answer |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-13480 | PASS | 30 | 103,335 | 3,256 | 179 | MAX_ITERATIONS |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-14711 | FAIL | 30 | 125,581 | 5,604 | 202 | MAX_ITERATIONS |

`PASS` with `MAX_ITERATIONS` means the agent ran out of iterations, and the patch that the loop collects
on the way out (`final_answer(get_patch())`) passed the tests anyway.

### 2.3 What the failures were

| Model | Task | Why it failed (from `llm_output` / `sandbox_output`) |
|---|---|---|
| nemotron-3-ultra | 14711 (v1, v2) | v1: no edit at all. Four times (steps 14, 15, 19 and 22) it called a tool without `print`, saw `(no output)` and drew no information from the step. v2: it printed its calls, but its first edit came at step 22, and it ran out of iterations |
| gpt-oss-20b | 14711 | its `search_code` calls named a path, such as `sympy/physics/vector/*.py`, which our tool then did not support. It got "No matches" 8 times for code that exists, never found `Vector.__add__`, and produced no patch (§5.2) |
| gemini-3.5-flash-lite | 13480, 14711 (v1) | with `reasoning_effort: low`, the model spent the whole response on reasoning and returned empty answers. Fixed in v2 by `minimal` |
| gemini-3.5-flash-lite | 14711 (v2) | it submitted at step 2 without running the tests. The patch made `_check_vector` return `Vector(0)` for `0`, which the hidden tests reject |
| gemma-4-31b-it | all | the API returned 5xx / overloaded errors; see §3 |

## 3. Provider reliability

v1:

| Model / provider | Pass | Avg response time per request (s) | Requests | Failed requests (retried or run-ending) | Request success rate | Runs not ended by provider failure |
|---|---|---|---|---|---|---|
| cohere / command-a-03-2025 | 3/3 | 2.2 | 25 | 0 | 100% | 3/3 |
| gemini / gemini-3.5-flash-lite | 1/3 | 5.1 | 6 | 2 | 67% | 1/3 |
| gemini / gemma-4-31b-it | 0/3 | 32.9 | 26 | 15 | 42% | 0/3 |
| nvidia / nemotron-3-super-120b-a12b | 3/3 | 4.8 | 39 | 0 | 100% | 3/3 |
| nvidia / nemotron-3-ultra-550b-a55b | 2/3 | 4.8 | 53 | 0 | 100% | 3/3 |
| nvidia / openai/gpt-oss-20b | 2/3 | 1.2 | 50 | 0 | 100% | 3/3 |

v2:

| Model / provider | Pass | Avg response time per request (s) | Requests | Failed requests (retried or run-ending) | Request success rate | Runs not ended by provider failure |
|---|---|---|---|---|---|---|
| gemini / gemini-3.5-flash-lite | 2/3 | 1.7 | 16 | 0 | 100% | 3/3 |
| gemini / gemma-4-31b-it | 0/3 | 29.0 | 31 | 15 | 52% | 0/3 |
| nvidia / nemotron-3-ultra-550b-a55b | 2/3 | 5.6 | 75 | 0 | 100% | 3/3 |

- **Cohere** answered every request, in 2.2 s on average. The trial key is limited to 1,000 calls per
  month; a task takes 4–15 calls. This is the main reason the production configuration has a fallback.
- **NVIDIA NIM** answered all 217 requests of v1 and v2, in 1.2–5.6 s depending on the model. The
  ablation runs (§5) brought the nemotron total to 555 requests, none failed. There is one exception: in the
  gpt-oss-20b check of §5.2, its endpoint returned HTTP 400 `Unknown role: assistant<|channel|>analysis`.
  That is a server-side parsing error of the model's own output, and it ended the run.
- **Google AI Studio** is two different stories:
  - `gemma-4-31b-it` failed 30 of 57 requests (5xx and overloaded responses). A successful request took
    29–33 s including retries, and every one of its six runs ended with `LLM_UNAVAILABLE`.
  - `gemini-3.5-flash-lite` with `minimal` reasoning answered all its requests, in 1.7 s on average. Its
    v1 failures were empty answers caused by the reasoning setting, not by the service. The free tier
    has daily quotas: another Gemini model, `gemini-3-flash-preview`, ran out of its daily quota during
    our MBPP measurements.

## 4. Intermediary metrics

v1:

| Model / provider | Task | First read of a patched file (step) | First edit (step) | Iterations from tests passing to final_answer |
|---|---|---|---|---|
| cohere / command-a-03-2025 | pydata__xarray-4629 | 2 | 3 | 1 |
| cohere / command-a-03-2025 | sympy__sympy-13480 | 2 | 3 | 1 |
| cohere / command-a-03-2025 | sympy__sympy-14711 | 2 | 8 | 1 |
| gemini / gemini-3.5-flash-lite | pydata__xarray-4629 | 1 | 1 | - |
| nvidia / nemotron-3-super-120b-a12b | pydata__xarray-4629 | 2 | 4 | 3 |
| nvidia / nemotron-3-super-120b-a12b | sympy__sympy-13480 | 1 | 2 | 2 |
| nvidia / nemotron-3-super-120b-a12b | sympy__sympy-14711 | 2 | 8 | 2 |
| nvidia / nemotron-3-ultra-550b-a55b | pydata__xarray-4629 | 3 | 5 | 10 |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-13480 | 1 | 2 | 2 |
| nvidia / openai/gpt-oss-20b | pydata__xarray-4629 | 4 | 5 | - |
| nvidia / openai/gpt-oss-20b | sympy__sympy-13480 | 2 | 3 | - |

The table leaves out runs that never read or edited a patched file: the failed runs of 2.3. The full
table is printed by the tool.

v2:

| Model / provider | Task | First read of a patched file (step) | First edit (step) | Iterations from tests passing to final_answer |
|---|---|---|---|---|
| gemini / gemini-3.5-flash-lite | pydata__xarray-4629 | 2 | 4 | 1 |
| gemini / gemini-3.5-flash-lite | sympy__sympy-13480 | 3 | 4 | 3 |
| gemini / gemini-3.5-flash-lite | sympy__sympy-14711 | 2 | 2 | - |
| nvidia / nemotron-3-ultra-550b-a55b | pydata__xarray-4629 | 3 | 5 | 8 |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-13480 | 2 | 3 | - |
| nvidia / nemotron-3-ultra-550b-a55b | sympy__sympy-14711 | 2 | 22 | - |

**Exploration efficiency.** The models that pass read the right file at step 1–3, and most edit it at
step 2–5. In every case the issue text names the failing function, and the strong models go straight to
it. The late edits tell where the time went:

- 14711 (step 8 for command-a and nemotron-3-super): before editing `__add__`, both looked for
  `_check_vector`, the helper that raises the error. nemotron-3-super lost two of those steps (4 and 5) to
  searches that named a path and got "No matches" (§5.2). It then read lines 600–726 to find the
  helper by hand.
- nemotron-3-ultra on 14711 in v2 (step 22): it kept re-reading and re-running before it committed to a
  fix.

**Submission discipline.**

- command-a submits one iteration after the first green `run_tests()`, on every task. It reads the
  result and then calls `final_answer(get_patch())`.
- nemotron-3-super takes 2–3 iterations.
- nemotron-3-ultra took 10 iterations on xarray in v1 and 8 in v2, verifying an already passing fix.
  These numbers are why we added an iteration-budget note: when 4 iterations are left, the agent is
  told to stop exploring and submit (commit `73de8c7`).

In the exam reproduction (§7), command-a ran `run_tests()` and `final_answer(get_patch())` in the same
step on 14711, which scores the ideal 0.

## 5. Ablation studies

### 5.1 `run_tests()` output summary (nemotron-3-super, commit `73de8c7`)

**Change.** By default, `run_tests()` returns a summary of the evaluation script's run:

- the exit code
- the names of the failing tests
- only the test section of the output, with DeprecationWarnings removed

With `SWE_RAW_TEST_OUTPUT=1` it returns the raw output instead, clipped at 6,000 characters like every
tool output. Everything else stays the same: same model, same commit, same tasks.

| Task | Summary ON | Summary OFF | Size of the `run_tests()` observation, ON → OFF |
|---|---|---|---|
| sympy__sympy-14711 | PASS*, 30 it., 126,751 in, 6,652 out, 186 s | PASS*, 30 it., 123,696 in, 7,412 out, 220 s | 1,119 → 6,167 chars |
| sympy__sympy-13480 | PASS, 25 it., 107,110 in, 9,931 out, 302 s | PASS, 9 it., 25,874 in, 567 out, 29 s | 1,305 → 6,168 chars |
| pydata__xarray-4629 | PASS, 11 it., 56,557 in, 1,172 out, 62 s | PASS, 8 it., 29,222 in, 864 out, 33 s | 4,396 → 6,167 chars |

\* Out of iterations; the patch collected at the limit passed.

**Results.**

- Both variants solved 3/3.
- The summary shrank each `run_tests()` observation by 29–82%. Each run called `run_tests()` only once,
  so the saving is at most about 5,000 characters (about 1,300 tokens) each time that observation is
  sent. That is small next to the 26,000–127,000 input tokens of a run.
- The large differences in iterations are not caused by the summary. In the 13480 ON run:
  1. `edit_file` failed twice because `old_str` had one space more indentation than the file.
  2. `search_code` then answered "No matches" for the line that was still there. The search named a
     path, which the tool did not support (§5.2).
  3. Convinced that its view of the file was wrong, the model spent 12 iterations reading the line
     character by character.

  The run ended with `final_answer` only because the output-token guard (`[BUDGET] Only 246 output
  tokens remain … Submit now`) made the agent submit. It did so with 69 tokens to spare, and the
  patch passed.
- With n = 1 per variant, the iteration counts vary more than the change could explain: the same model
  and task took 14 iterations in v1.

**Decision.** We keep the summary, but not because of these numbers. On these three tasks the test
output is short, so there was nothing to gain. The summary always names the failing tests, which a
clipped raw log does not guarantee; it costs one extra parsing step.

### 5.2 Tool feedback fixes (nemotron-3-super, commits `8e54ca5` → `0c080b4`)

**Why.** The ablation 1 run above, and a scan of every log in `benchmark/swebench/`, showed tool
feedback that sent models in the wrong direction:

| Problem in the logs | Fix |
|---|---|
| `search_code(pattern, file_pattern)` passed `file_pattern` to `grep --include`, which only matches file *names*. Any path returned "No matches": `sympy/physics/vector/*.py`, `/testbed/…/vector.py`, `**/*.py`. 23 of the 25 empty searches on 14711 and 13480 named a path. Replaying them on the real containers, 17 of those 23 find matches with the fixed tool; one of them may have been a true negative, because the agent had already edited that line | a `file_pattern` with a `/`, or with no wildcard, is matched as a path: a file, a directory, `dir/*.py` or `**/*.py`. "No files match" is now reported apart from "No matches" |
| 8 iterations of `SyntaxError: unmatched ')'` (nemotron-3-super, 14711): the model counted the brackets inside a string literal | the parser reports the position of the unmatched closing bracket; only that character is removed, the code is run, and `[MALFORMED_CODE]` says so |
| `edit_file` syntax errors showed only `error.text`, which can be a blank line. command-a spent 8 turns on an edit whose quote escapes Python had consumed; 7 of them hit the per-turn cap, and the run used 9,715 of its 10,000 output tokens (§7) | the error shows the edited lines with `>>` on the failing one |
| `edit_file` "old_str not found" when only the indentation differed (13480, above) | the tool lists the lines that match apart from indentation, with their leading-whitespace count |
| `NameError: name 'search_file' is not defined`; unknown keyword arguments such as `search_code(filepath=…)` were silently dropped by the MCP server | the error lists the available tools and the closest names; dropped arguments are reported with a `[NOTE]` |

**First attempt: gpt-oss-20b.** This was the model most affected
(`benchmark/swebench/ablation_tools_gptoss20b_{before,after}`).

- Before: 2/3. 14711 failed at 30 iterations; 13480 passed only with the patch collected at the limit;
  xarray passed in 19 iterations.
- After: 1/3. 14711 failed again; 13480 passed in 8 iterations instead of 30.

The comparison is not usable, for two reasons:

- The after-run of xarray ended at step 3 with an NVIDIA HTTP 400.
- On 14711, 12 of the model's 30 turns contained no Python at all, only its native tool-call JSON
  (`{"cmd": ["bash", "-lc", …]}`), so the model's format, not our tools, decided the result.

We therefore repeated the comparison with the production fallback model.

**Measurement.** nemotron-3-super, the same three tasks, two runs per variant.
`benchmark/swebench/ablation_tools_{before,after}_r{1,2}`:

| Task | Before (`8e54ca5`), iterations / input tokens | After (`0c080b4`), iterations / input tokens |
|---|---|---|
| sympy__sympy-14711 | 16 / 47,151 · 21 / 83,102 | 22 / 81,232 · 30* / 118,075 |
| sympy__sympy-13480 | 11 / 34,288 · 6 / 14,692 | 6 / 15,383 · 24 / 119,683 |
| pydata__xarray-4629 | 25 / 121,062 · 7 / 23,358 | 10 / 37,295 · 12 / 48,674 |
| **Total** | **6/6 PASS**, 86 iterations, 323,653 input tokens | **6/6 PASS**, 104 iterations, 420,342 input tokens |

\* Out of iterations; the patch collected at the limit passed.

**Results.**

- **No measurable effect with this model.** Both variants passed 6/6. The iteration counts overlap
  completely: 13480 took 6–24 iterations, and xarray 7–25, with the *same* code.
- **The failure modes the fixes target were rare in these 12 runs**: one unknown tool name before, and
  one `edit_file` syntax error after. nemotron-3-super seldom goes down these paths, so a 2 × 3 sample
  cannot show their effect.
- **The long runs were caused by something else, on both sides.** The model wrote reproduction scripts
  as `run_command("python -c \"...\"")` with nested quotes and needed 5–9 attempts to get the quoting
  right (xarray before-r1, steps 5–13; 14711 after-r2, steps 5–13). Or it verified with ad-hoc
  commands for 20 iterations before calling `run_tests()` once (13480 after-r2). This led to §5.3.
- **The direct evidence for the fixes is deterministic.** We replayed the logged calls against the real
  containers: 17 of the 23 empty path searches now return the code, and the 8 logged
  `unmatched ')'` turns of 14711 now run. The fixes matter for the models and runs that hit those paths:
  gpt-oss-20b's eight empty searches on 14711, and command-a's 9,715 output tokens in §7. Measuring that
  end to end would need more runs than we had quota for.

### 5.3 Prompt: how to write a reproduction script (nemotron-3-super, `0c080b4` → `5e2f7cd`)

**Change.** The SWE-bench instructions used to suggest `run_command(command="python -c '...'")`, which
invites nested quotes. They now show a triple-quoted string with a quoted heredoc, where nothing needs
escaping:

```python
print(run_command(command="""cd /testbed && python - <<'EOF'
from pkg.module import func
print(func(0))
EOF"""))
```

**Measurement.** nemotron-3-super, the same three tasks, two runs per variant. "Before" is the pair of
"after" runs of §5.2 (same commit `0c080b4`); "after" is `benchmark/swebench/ablation_prompt_after_r{1,2}`.
The counts below were taken from the `sandbox_input` / `sandbox_output` of every step.

| | Before (`0c080b4`) | After (`5e2f7cd`) |
|---|---|---|
| `run_command` calls using `python -c` | 31 | **3** |
| `run_command` calls using a heredoc | 9 | 11 |
| `run_command` steps that failed on quoting (SyntaxError, unterminated string, shell syntax error) | 13 | **3** |
| Iterations, 6 runs | 104 | **85** (−18%) |
| Input tokens, 6 runs | 420,342 | **320,436** (−24%) |
| Passed | 6/6 | 5/6 |

| Task | Before: iterations per run | After: iterations per run |
|---|---|---|
| sympy__sympy-14711 | 22, 30* | 18, 12 |
| sympy__sympy-13480 | 6, 24 | 8, 6 |
| pydata__xarray-4629 | 10, 12 | 30 (FAIL), 11 |

\* Out of iterations; the patch collected at the limit passed.

**Results.**

- The change did what it targets. The model almost stopped writing `python -c` one-liners, quoting
  failures fell from 13 to 3, and the runs became shorter and cheaper.
- The one failure is unrelated to the change. In xarray after-r1:
  1. `run_tests()` showed the failing hidden test, its code and the failing assertion (`assert ds1.x == 0`).
  2. The model then spent 18 iterations searching the repository for that test, which the evaluation
     script had already removed again.
  3. It never edited `merge.py`.

  Since commit `5c833a4`, a failing `run_tests()` says so explicitly. It names the test files that the
  evaluation script swaps in and restores, and points to the output for the test's code.

### 5.4 Other before/after observations (v1 → v2)

These are not clean ablations: the agent changed in several places between `1163b6d` and `9f1ebef`.
Two changes target specific failures, though, and the effect on those failures is direct:

- **`gemini-3.5-flash-lite`, `reasoning_effort` `low` → `minimal`.** Empty answers disappeared: the
  request success rate went from 67% to 100% and the runs not ended by provider failure from 1/3 to
  3/3. 13480 went from FAIL to PASS, and the score from 1/3 to 2/3.
- **The value of the last expression is shown in the observation** (like a REPL). In v1, nemotron-3-ultra
  called tools without `print` four times on 14711 and saw `(no output)`. In v2 it printed every call, so
  the new display was never needed. It still failed 14711, for another reason (first edit at step 22). We
  therefore have no measurement of this change; it is kept because it cannot hurt.

## 6. Conclusions

**Final SWE-bench pipeline** (`providers/providers_swebench.json`, tried in this order):

1. **Cohere `command-a-03-2025`.** 3/3 in v1 and 6/6 in both exam reproductions.
   - It used the fewest iterations (4–15) and input tokens (11,000–55,000 per task) of all models.
   - It answered fastest (2.2 s per request) and never failed a request.
   - It submitted one iteration after the tests passed.
   - Its trial quota (1,000 calls per month) is the reason for the fallbacks below.
2. **NVIDIA `nemotron-3-super-120b-a12b`**, then **`nemotron-3-ultra-550b-a55b`**.
   - nemotron-3-super passed 3/3 in v1 and 26 of the 27 runs it made in total (v1 and the ablations).
     Not one of the 555 nemotron requests on NVIDIA failed.
   - It is slower and needs more iterations than command-a.
   - nemotron-3-ultra shares NVIDIA's reliability but scored 2/3 twice and is the slowest to commit to a
     fix, so it is only the second NVIDIA model.
3. **Ollama Cloud: `nemotron-3-super`, then `gemma4:31b`** (§8). Both passed 3/3, with no failed
   request. Ollama serves the same nemotron model through a separate quota, which protects the chain
   against NVIDIA being unavailable. gemma4 is the fastest model we measured on SWE-bench (1.7 s per
   request).
4. **Mistral `codestral-latest`**, then **`gemini-3.5-flash-lite`** (`minimal`), as the last resorts.
   - codestral passed 2/3 (§8). It is fast, but on 14711 it repeated the same failing edit 22 times.
   - flash-lite is fast and solved 2/3 in v2, but it can submit without testing (14711 in v2), and the
     Gemini free tier has daily quotas.

**Disregarded:**

- **`gemma-4-31b-it` on Google AI Studio.** 0/6 runs, every one ended by provider failure, with 30 of
  57 requests failed and about 30 s per successful request. The same model through Ollama passed 3/3
  (§8), so the problem was the free API, not the model.
- **`openai/gpt-oss-20b` on NVIDIA.** It is fast (1.2 s per request), but:
  - it fails 14711 consistently (v1, and both runs of the gpt-oss check in
    `benchmark/swebench/ablation_tools_gptoss20b_*`);
  - in 12 of 30 turns of one run, it answered with its native tool-call JSON
    (`{"cmd": ["bash", "-lc", …]}`) instead of Python;
  - the endpoint ended one run with HTTP 400.
- **Groq.** Its per-account limit of 8,000 tokens per minute cannot carry a SWE-bench run within 900 s.

**What the data says beyond model choice.**

- Where the logs showed iterations being lost, the cause was usually what the agent showed or
  suggested to the model, not the model's skill:
  - a search tool that ignored paths;
  - error messages that hid the broken line;
  - a prompt that suggested nested quotes;
  - a test that disappears after `run_tests()`.
- The prompt fix had the clearest measured effect: quoting failures fell from 13 to 3, and runs needed
  18% fewer iterations and 24% fewer input tokens (§5.3).
- The tool fixes are proven by replaying the logged calls. Their end-to-end effect was too rare to
  measure with nemotron-3-super in 12 runs (§5.2).
- Run-to-run variance is large: the same model, code and task took 6–24 iterations. A single run per
  configuration (n = 1) cannot rank close alternatives, and we report it as such.

## 7. Exam reproduction (production configuration)

All six `EXAM_POOL` tasks were run through the moulinette's own `quickstart.sh`:
dump → `run-agent` with its 900 s SIGTERM/SIGKILL → validate. No model was given, so the agent used its
default `providers/providers_swebench.json`. We did this twice:

- before the fixes of §5.2–§5.3, at `8e54ca5`;
- at the final agent code, `1ac6023`.

The data is in `benchmark/swebench/exam_reproduction_{8e54ca5,1ac6023}`.

| Task | `8e54ca5`: result, iterations, input, output | `1ac6023` (final): result, iterations, input, output |
|---|---|---|
| django__django-11066 | PASS, 4, 12,653, 391 | PASS, 5, 16,246, 376 |
| pydata__xarray-4629 | PASS, 7, 30,023, 505 | PASS, 5, 16,299, 385 |
| scikit-learn__scikit-learn-13439 | PASS, 13, 54,528, 2,138 | PASS, 6, 21,589, 723 |
| sympy__sympy-13480 | PASS, 6, 16,864, 389 | PASS, 6, 18,305, 466 |
| sympy__sympy-14711 | PASS, 12, 49,076, **9,715** | PASS, 9, 28,176, 915 |
| sympy__sympy-18189 | PASS, 7, 25,695, 569 | PASS, 6, 23,568, 458 |
| **Total** | **6/6**, 49 it., 188,839 in, 13,707 out | **6/6**, 37 it., 124,183 in, 3,323 out |

- Both runs passed 6/6 (the exam needs 2/3), within every limit.
- Every step was answered by command-a (Cohere): 49 and 37 requests, none failed, no retries.

**The warning in the first run** was 14711's **9,715 of 10,000 output tokens**:

- command-a wrote several long `edit_file` attempts per turn, and seven turns were cut off at the
  per-turn cap of 1,200 tokens.
- Its edits put `\'` inside a `"""…"""` argument. Python consumed the escapes, so the edited file
  was invalid, and `edit_file` reverted it.
- The error message showed only a blank `error.text` line, so the model could not see what was wrong.

That message is one of the fixes in §5.2. With the final code, the same task took 915 output tokens.
Over the six tasks, the final run used 25% fewer iterations, 34% fewer input tokens and 76% fewer
output tokens. That is one run each, but it is consistent with the ablations.

The output-token budget is enforced in any case:

- the cap of each request is lowered to what remains;
- a `[BUDGET]` note asks for a submission;
- the patch collected at the end is still submitted.

## 8. Later candidates: Mistral (key renewed) and Ollama Cloud

After the main runs, the Mistral key was renewed and an Ollama Cloud key was added. We measured them the
same way: one model per run, no fallback, the same three tasks (commit `4b8172b`). The data is in
`benchmark/swebench/candidate_*`.

**Smoke test first** (one request per model: `tools/smoke/2026-10-15_smoke_ollama_mistral.json` and
`tools/smoke/2026-10-15_variants_ollama.json`):

- Ollama's free tier serves 4 of the 18 models its API lists: `gpt-oss:120b`, `gpt-oss:20b`,
  `nemotron-3-super` and `gemma4:31b`. The others return HTTP 402 ("not included in your free usage")
  or 403.
- Two of those models need a request parameter:
  - gpt-oss needs `reasoning_effort: low`;
  - nemotron-3-super needs `reasoning_effort: none`.

  Otherwise their thinking reaches the `<end_code>` stop sequence and the answer is empty.

**SWE-bench:**

| Model / provider | Task | Result | Iterations | Input tokens | Output tokens | Time (s) | Stop reason |
|---|---|---|---|---|---|---|---|
| mistral / codestral-latest | pydata__xarray-4629 | PASS | 6 | 19,100 | 435 | 23 | final_answer |
| mistral / codestral-latest | sympy__sympy-13480 | PASS | 4 | 8,827 | 246 | 15 | final_answer |
| mistral / codestral-latest | sympy__sympy-14711 | FAIL | 30 | 177,412 | 5,171 | 190 | MAX_ITERATIONS |
| ollama / nemotron-3-super | pydata__xarray-4629 | PASS | 15 | 61,229 | 1,282 | 67 | final_answer |
| ollama / nemotron-3-super | sympy__sympy-13480 | PASS | 6 | 15,620 | 491 | 30 | final_answer |
| ollama / nemotron-3-super | sympy__sympy-14711 | PASS | 11 | 39,430 | 1,402 | 67 | final_answer |
| ollama / gemma4:31b | pydata__xarray-4629 | PASS | 7 | 27,949 | 1,149 | 21 | final_answer |
| ollama / gemma4:31b | sympy__sympy-13480 | PASS | 5 | 15,432 | 492 | 19 | final_answer |
| ollama / gemma4:31b | sympy__sympy-14711 | PASS | 15 | 73,582 | 4,622 | 42 | final_answer |

| Model / provider | Pass | Avg response time per request (s) | Requests | Failed requests (retried or run-ending) | Request success rate | Runs not ended by provider failure |
|---|---|---|---|---|---|---|
| mistral / codestral-latest | 2/3 | 1.7 | 40 | 0 | 100% | 3/3 |
| ollama / gemma4:31b | 3/3 | 1.7 | 27 | 0 | 100% | 3/3 |
| ollama / nemotron-3-super | 3/3 | 4.0 | 32 | 0 | 100% | 3/3 |

- **Ollama passed 6/6 without a failed request.**
  - nemotron-3-super behaved as it does on NVIDIA: 11, 6 and 15 iterations here, against 14, 14 and
    11 in v1.
  - `gemma4:31b` is the same model as `gemma-4-31b-it`, which failed every run on Google AI Studio
    because of API errors. On Ollama it passed 3/3 at 1.7 s per request. This is the clearest case in
    our data of a provider, not a model, deciding the result.
- **codestral passed 2/3.** It is fast on the simple tasks. On 14711 it opened the wrong module
  (`sympy/vector/vector.py` instead of `sympy/physics/vector/vector.py`) and sent the same failing
  `edit_file` 22 times. SWE-bench never switched models, because unsuccessful exploration is normal
  there. Since commit `9180487`, two repetitions of identical code in a row switch the agent to a model
  of another provider.

**MBPP, for completeness** (not part of the subject's report): the same 10 tasks of our MBPP regression set, one model per run
(`benchmark/mbpp_candidates/`).

| Model (provider) | Passed | Notes |
|---|---|---|
| `gemma4:31b` (Ollama) | 9/10 | 9 tasks solved in a single iteration |
| `nemotron-3-super` (Ollama) | 9/10 | |
| `gpt-oss:120b` (Ollama) | 8/10 | 245 passed the visible tests but not the hidden one |
| `codestral-latest` (Mistral) | 7/10 | on the three failures it resent identical code after `[REPEATED]` until the input limit |

- Every model failed task 430, whose formula has to be inferred from the visible tests.
- The first gpt-oss and codestral runs started in the same second and shared one output directory, so
  they were discarded and re-run. The MBPP harness now picks a free directory name (commit `8c933d8`).

**Decision.**

- Ollama is now the third provider in both chains:
  - SWE-bench: after Cohere and NVIDIA, with `nemotron-3-super` then `gemma4:31b`;
  - MBPP: after Mistral and Groq.
- codestral stays as MBPP primary because it is fast on easy tasks, and the MBPP loop switches provider
  after two unsuccessful turns. On SWE-bench it is only a late fallback.
