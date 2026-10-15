*This project has been created as part of the 42 curriculum by ayhirose, nsato, joseo.*

# Agent Smith — an autonomous code agent for MBPP and SWE-bench

## Description

Agent Smith is a **code agent**: instead of answering in one shot, an LLM works in a
**Thought → Code → Observation** loop. In every turn the model writes a short thought and one
Python code block; the code runs inside a **sandbox**, and what it printed (or the error it raised)
is sent back as the next observation. Tools such as `run_tests()`, `read_file()` or `get_patch()` are
plain Python functions inside the sandbox, provided by our own **MCP (Model Context Protocol)**
servers, so the model can combine them with variables, loops and conditions.

The same framework solves two benchmarks:

| Benchmark | What the agent submits | Limits (per task) |
|---|---|---|
| **MBPP** (algorithmic Python problems) | the source code of one function | 10 iterations, 6,000 input / 1,500 output tokens, 120 s |
| **SWE-bench Verified** (real bugs in real repositories, inside Docker) | a git patch (`git -c core.fileMode=false diff`) | 30 iterations, 300,000 input / 10,000 output tokens, 900 s |

Everything — the agent loop, the sandbox, the MCP client and servers, the provider layer — is our
own code (no agent framework, no MCP SDK). The project only uses free LLM tiers.

## Instructions

### Requirements
- Python **3.10** and [uv](https://docs.astral.sh/uv/) (`make uv-install` installs uv).
- **Docker** (SWE-bench only; the SWE-bench images are pulled on first use).
- API keys of free-tier, OpenAI-compatible providers (see `.env.example`).

### Setup
```bash
make install                 # = uv sync (do this before an evaluation: the first `uv run` is slow)
cp .env.example .env         # then fill in the keys you have
```
Several keys per provider are supported: `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, …
Keys are read from the environment / `.env` only; `providers/providers.json` contains only the
*names* of the variables.

### Run the agents
```bash
# MBPP
cd ../moulinette && uv run moulinette_eval dump mbpp --output ../cache/mbpp_task.json
cd ../My_AgentSmith
uv run python -m agent_mbpp --task-file ../cache/mbpp_task.json --output ../cache/mbpp_solution.json \
    [--model-name codestral-latest --provider-url https://api.mistral.ai/v1]
cd ../moulinette && uv run moulinette_eval validate mbpp ../cache/mbpp_task.json ../cache/mbpp_solution.json

# SWE-bench (Docker must be running)
uv run python -m agent_swebench --task-file ../cache/swebench_task.json --output ../cache/swebench_solution.json
```
- `--model-name` / `--provider-url` are optional. If the URL (or the model) belongs to a provider of
  the providers file, that provider's keys and settings are used and the other providers stay as
  fallbacks; an unknown URL uses the key `<DOMAIN>_API_KEY` (e.g. `TOGETHER_API_KEY`).
  `--providers_file_path` selects another providers file.
- `--mcp-stdio "<command>"` / `--mcp-server <url>` (optional, mutually exclusive) connect the agent to
  another MCP server. By default each run starts its own server (`mcp_tools_mbpp.py` /
  `mcp_tools_swebench.py`) as a private child process over stdio.
- The agent **always writes `solution.json`**, even on an unexpected error, Ctrl+C or SIGTERM
  (then `success=false` and `error` says why), and always removes its Docker container.

### Sandbox CLI
```bash
uv run sandbox                                                     # interactive REPL (exit / Ctrl+D)
uv run sandbox sandbox_template.json                               # with a configuration file
uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
uv run sandbox --mcp-stdio "python mcp_tools_swebench.py" sandbox_template.json   # uses $TESTBED_PATH
uv run python mcp_tools_mbpp.py --transport http --port 8000       # terminal 1: start an HTTP server
uv run sandbox --mcp-server http://127.0.0.1:8000/mcp              # terminal 2: connect to it
uv run sandbox -c "print(1 + 1)"                                   # run one snippet (or -f file.py)
```

### Configuration files
| File | What it configures |
|---|---|
| `config.json` | hard limits per benchmark (`limits` = MBPP, `swebench_limits`), retry / backoff, the per-snippet execution timeout of the agent's sandbox |
| `providers/providers.json` | MBPP: providers in fallback order — base URL, models, the env-var name of the keys, per-model request parameters (e.g. turning "thinking" off) |
| `providers/providers_swebench.json` | SWE-bench: the same format, ordered by our SWE-bench benchmark (providers whose per-minute token quota is smaller than one SWE-bench request are left out) |
| `sandbox_template.json` | `SandboxConfig`: authorized imports (`x.*` allows submodules), allowed directories, time and memory limits |
| `.env` | API keys (not committed) |

### Tests, lint and measurement tools
```bash
make test                    # uv run pytest (unit tests + real-sandbox security tests)
make lint                    # flake8 + mypy
uv run python3 tools/harness_mbpp.py --count 10 --seed 1                     # MBPP: dump, solve, validate
uv run python3 tools/harness_swebench.py --task_ids sympy__sympy-14711 \
    --models nvidia:nvidia/nemotron-3-super-120b-a12b                        # SWE-bench benchmark runs
```

## System architecture

```
agent_mbpp.py / agent_swebench.py          CLI (fire): parse arguments only
        │
src/agent/cli.py                           wiring: settings, providers, sandbox, Docker image,
        │                                  solution.json on every exit path, container cleanup
        ▼
Orchestrator (src/agent/orchestrator.py)   the Thought → Code → Observation loop
   ├─ LimitTracker (limits.py)                    iterations / cumulative tokens / wall-clock time
   ├─ TaskProfile (prompt/task_profile/*)         MBPPProfile, SWEBenchProfile: prompt parts,
   │                                              submission checks, history window, fallbacks
   ├─ ResilientLLMClient (llm/resilient_client.py) providers × keys × models, retries, fallback
   │     └─ OpenAICompatProvider → HttpTransport (requests)   one HTTP request each
   ├─ CompositeCodeExtractor (extractor/*)        ```python blocks, XML, JSON/Hermes, ReAct
   └─ Sandbox (src/sandbox/*)                     forked worker + audit hook, persistent namespace
         └─ MCPClient (src/mcp_core/client/*) ── stdio / streamable HTTP ──► MCP servers
                mcp_tools_mbpp.py      run_tests(code, test_list)
                mcp_tools_swebench.py  the 9 mandatory tools; commands run in the task's Docker
                                        container (src/mcp_core/swe_bench/*)
```
The orchestrator only depends on interfaces (`LLMGenerator`, `CodeExtractor`, `SandboxBase`,
`TaskProfile`), so a provider, an extraction format, a sandbox or a benchmark can be swapped without
touching the loop.

## Agent loop explanation

Each iteration of `Orchestrator.run()`:

1. **Limits first.** Stop if the iteration, cumulative-token or time budget is used up.
2. **Fit the history.** Chat APIs are stateless, so the whole history is re-sent every turn. The
   system prompt and the task are always kept; only the last few exchanges follow (2 for MBPP, 6 for
   SWE-bench), each trimmed in the middle so that the advice at the end of an observation survives.
   Before sending, the input tokens of the request are **estimated** (characters ÷ 3.0, corrected
   downwards with the ratio actually observed) and the window shrinks — 2 → 1 → 0 exchanges — so the
   cumulative input limit can never be crossed by one request.
3. **Generate** with the stop sequence `<end_code>` (otherwise models invent their own
   "Observation:"). The output budget of the request is `min(per-turn cap, remaining tokens)`.
4. **Extract the code** (Python block; XML / JSON-Hermes / ReAct tool calls are converted to Python
   calls). Leaked stop-sequence fragments are removed; an unclosed block, bare code, over-escaped
   quotes, extra closing brackets (removed one by one where the parser reports them), a
   `{"code": ...}` JSON wrapper or several blocks are still executed, and the model is told exactly how
   its output was interpreted.
5. **Execute** in the sandbox and build the observation: status labels (`[NO_CODE]`, `[TIMEOUT]`,
   `[TRUNCATED]`, `[MALFORMED_CODE]`, `[REPEATED]` when the same code is run twice, `[BUDGET]` when
   only one turn of output is left), stdout, the error, and a hint. Calling a tool that does not exist
   lists the available tools and the closest names; keyword arguments a tool does not have are
   reported as ignored.
6. **Submission.** When the code calls `final_answer(...)`, the profile checks the value — MBPP: it
   must be importable function source with no harness code, it must pass the visible tests, and it
   must not return the visible answers as literals; SWE-bench: it must be a git diff (use
   `final_answer(get_patch())`). A rejected submission goes back to the model with the reason.
7. **Stagnation (MBPP).** After two turns without an accepted submission the next request prefers a
   model of *another provider* (different models fail differently).

When the loop stops without a submission, the best candidate is submitted anyway: MBPP — the last
function that ran without error; SWE-bench — the current `get_patch()` of the container.

**Tokens and providers.** Token counts come from the provider's `usage`; hidden reasoning tokens
(`total − prompt > completion`) and tokens burnt by empty responses are counted too, because the
limits include reasoning tokens. `ResilientLLMClient` tries providers × keys × models in order:
429 → next key, 5xx / network → backoff (clamped to the remaining time), 401/403/400/empty → next
combination, and that combination is demoted for the rest of the task. Every HTTP timeout is clamped
to the remaining task time, so the agent finishes before the grader's SIGTERM.

## Sandbox design

The sandbox executes LLM-generated code in a **separate forked process** that keeps its namespace
between snippets; the parent talks to it over two pipes (length-prefixed JSON).
- **Imports**: only `authorized_imports` (`x.*` allows submodules), checked statically (AST) and at
  run time (guarded `__import__`).
- **Builtins**: `eval`, `exec`, `compile`, `getattr`, `globals`, `open` (replaced by a guarded one),
  … are removed or wrapped; dunder attributes and private attributes of modules are rejected statically.
- **Audit hook (PEP 578)**: the real enforcement point. However a module was reached (e.g.
  `typing.sys.modules["os"]`), opening files outside `allowed_directories` (reading is also allowed
  inside the Python installation for lazy imports), listing / changing files outside them, sockets,
  subprocesses, `ctypes`, heap inspection and raising resource limits all raise `PermissionError`.
- **Secrets**: the worker's environment is cleared, so API keys are not reachable from the code.
- **Time**: `SIGALRM` stops a snippet after `max_execution_time_seconds` and returns the partial
  output; if the code ignores it, the parent kills the worker and says that the namespace was lost.
  Tool calls do not count against this timer.
- **Memory / processes**: `RLIMIT_AS` (MemoryError instead of swapping), `RLIMIT_NPROC = 0`.
- **final_answer** is a sandbox built-in (not an MCP tool). `KeyboardInterrupt` and `SystemExit`
  are propagated to the caller with their exit code.
- **MCP integration**: tools are discovered dynamically (`tools/list`, with pagination and protocol
  version negotiation) and exposed as Python functions accepting positional or keyword arguments;
  resources and prompts are listed in the manual and readable with `read_resource(uri)` /
  `get_prompt(name, **arguments)`. The manual given to the LLM is generated from the connected
  server's schemas, so an unknown server works without any code change.
- For SWE-bench we use approach **(b)**: the sandbox runs on the host and the MCP tools bridge into
  the task's Docker container.

## Tool implementation details

**MBPP — `run_tests(code, test_list)`** runs the candidate in a fresh, isolated interpreter
(empty environment, temporary directory, resource limits, the same audit hook) and executes the
asserts **one by one**. A failure reports the assert, what the function returned and what was
expected (`got [(3, 2, 1), …], expected '[(3, 2, 1), …]' (you returned a list, the test expects a
str)`), instead of a bare `AssertionError` on line 4. Returns `{"success", "message", "output"}`.

**SWE-bench — `mcp_tools_swebench.py`** (all outputs are English text; long outputs are truncated
with an explicit note). With `SWE_DOCKER_IMAGE` set, the server starts one container
(`--network none`, label `agentsmith=1`) and runs every command with `docker exec … bash -lc`
(the task's conda environment); otherwise it works on the host directory `TESTBED_PATH`, which is how
the tools are tested in isolation.

| Tool | Behaviour |
|---|---|
| `read_file(filepath, start_line, end_line)` | `<line>: <content>` lines; clear errors for missing files / bad ranges |
| `edit_file(filepath, old_str, new_str)` | replaces exactly one occurrence; for `.py` files a syntax error **reverts the edit** and shows the lines the edit would have produced, with `>>` on the error line (plus pyflakes warnings when available); when `old_str` is not found, lines that differ only in indentation are listed with their leading-whitespace count |
| `list_files(directory, pattern)` | absolute paths, `.git` skipped, capped |
| `search_code(pattern, file_pattern)` | `grep -E`; lines formatted `/abs/path.py:<line> <content>`. `file_pattern` is a file-name glob (`*.py`) or a path: a file, a directory, `pkg/sub/*.py`, `**/*.py`, absolute or relative. "No files match" and "No matches found" are reported separately |
| `search_function_or_class_definition_in_code(name)` | `def` / `async def` / `class` definitions, same format |
| `find_references(name, filepath, line)` | whole-word usages in `.py` files, same format |
| `run_tests()` | runs the task's evaluation script, returns the exit code, the failing test names and the test section of the output, then restores the test files it touched |
| `get_patch()` | `git -c core.fileMode=false diff`, including new files and excluding the evaluation's test files |
| `run_command(command, workdir)` | exit code, stdout and stderr |

`SWE_DEADLINE_EPOCH` shortens every command's timeout so that no tool call can outlive the task's
900 s; the server removes its container on exit (EOF, SIGTERM, SIGINT), and the agent removes it
again in a `finally` as a safety net.

## Benchmark results and analysis

The full report — setup, per-task results, provider reliability, intermediary metrics, ablation and
conclusions — is in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md); the backing `solution.json` files
are under [`benchmark/`](benchmark/).

## Resources

- SWE-bench: Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
  (ICLR 2024) — https://www.swebench.com ; MBPP: Austin et al., *Program Synthesis with Large
  Language Models* (2021).
- Wang et al., *Executable Code Actions Elicit Better LLM Agents* (CodeAct, ICML 2024) — the idea of
  acting through code instead of JSON tool calls.
- Model Context Protocol specification — https://modelcontextprotocol.io/specification
- PEP 578 — Python Runtime Audit Hooks; Python docs for `resource`, `signal`, `subprocess`.
- OpenAI-compatible Chat Completions APIs of Groq, Mistral, Cohere, Google AI Studio, NVIDIA NIM, Ollama Cloud and
  OpenRouter (official documentation of each provider).

### How AI was used
We used AI assistants (Claude Code) as a pair programmer, and reviewed every change as a team:
- **Analysis**: reading failing runs (`solution.json`) to find root causes (token accounting, stop
  sequence leaks, uninformative test output, sandbox escapes) and comparing them with the subject.
- **Implementation help**: robustness fixes of the agent loop and provider layer, the SWE-bench tool
  server and the sandbox hardening were drafted with AI and then reviewed, tested and adjusted by us.
- **Tests and measurement**: unit tests, the real-sandbox security tests and the benchmark harnesses.
- **Documentation**: drafting this README and the benchmark report from our measured data.
The design decisions (architecture, interfaces, limits, provider choice) were made and validated by
the team, and every number in the report comes from real runs stored in this repository.
