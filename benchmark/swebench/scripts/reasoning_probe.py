"""SWE の候補（推論 ON/OFF の組）を、エージェントと同じリクエストの形で1回ずつ確かめる.

stop=["<end_code>"]・max_tokens=1200（SWE の1反復の上限）で送り、
本文が返るか・本文にコードがあるか・出力トークン（思考を含む）・秒数を記録する。
キーは load_dotenv ＋ resolve_api_keys で読む（.env を直接は読まない）。

使い方: (wt_swe で) uv run python bench_swe/reasoning_probe.py [--match 正規表現]
"""
import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agent.llm.key_manager import resolve_api_keys
from src.agent.llm.transport.requests_transport import RequestsTransport
from src.agent.llm.usage import extract_token_usage

SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_DIR = SCRIPT_DIR.parents[1] / "cache" / "models" / "providers"
OUTPUT = SCRIPT_DIR / "reasoning_probe.json"

SYSTEM = """You are Agent Smith, an autonomous coding agent fixing a bug in a git repository.
Loop: Thought (one line) -> ONE Python code block -> read the Observation.
## Format
- One short Thought line, then exactly ONE ```python block, then <end_code>.
- Tools are Python functions: read_file(path), search_code(pattern, file_pattern), edit_file(path, old, new),
  run_tests(test_path), get_patch(). Print their results; only printed output comes back.
## Submitting
Call final_answer(get_patch()) when the fix is in place and tests pass."""

USER = """Repository: /testbed (calc)
Issue: `calc.ops.divide(a, b)` raises ZeroDivisionError when b == 0. It should return None instead.
Start by locating the function."""

# (ラベル, 単体ファイル（cache/models/providers）か None, 手で書く設定, model_params の上書き。None は model_params を空にする)
MANUAL = {
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
}
CONFIGS = [
    ("kilo_ultra_off", "kilo__nvidia_nemotron-3-ultra-550b-a55b_free", None, {"reasoning": {"enabled": False}}),
    ("kilo_ultra_on", "kilo__nvidia_nemotron-3-ultra-550b-a55b_free", None, {"reasoning": {"enabled": True}}),
    ("kilo_laguna_s_off", "kilo__poolside_laguna-s-2.1_free", None, {"reasoning": {"enabled": False}}),
    ("kilo_laguna_s_on", "kilo__poolside_laguna-s-2.1_free", None, {"reasoning": {"enabled": True}}),
    ("nvidia_ultra_off", "nvidia__nvidia_nemotron-3-ultra-550b-a55b", None,
     {"chat_template_kwargs": {"enable_thinking": False}}),
    ("nvidia_ultra_on", "nvidia__nvidia_nemotron-3-ultra-550b-a55b", None,
     {"chat_template_kwargs": {"enable_thinking": True}}),
    ("nvidia_minimax", "nvidia__minimaxai_minimax-m3.1-flash", None, None),
    ("nvidia_glm53_flash", "nvidia__z-ai_glm-5.3-flash", None, None),
    ("nvidia_deepseek_v41_flash", None, ("nvidia", "deepseek-ai/deepseek-v4.1-flash"), None),
    ("ollama_gptoss120_low", "ollama__gpt-oss_120b", None, {"reasoning_effort": "low"}),
    ("ollama_gptoss120_medium", "ollama__gpt-oss_120b", None, {"reasoning_effort": "medium"}),
    ("ollama_gptoss120_high", "ollama__gpt-oss_120b", None, {"reasoning_effort": "high"}),
    ("ollama_ultra_none", "ollama__nemotron-3-ultra", None, {"reasoning_effort": "none"}),
    ("ollama_ultra_medium", "ollama__nemotron-3-ultra", None, {"reasoning_effort": "medium"}),
    ("cohere_command_a", "cohere__command-a-03-2025", None, None),
    ("cohere_command_a_plus", "cohere__command-a-plus-05-2026", None, None),
    ("cohere_reasoning_off", "cohere__command-a-reasoning-08-2025", None, {"reasoning_effort": "none"}),
    ("cohere_reasoning_on", "cohere__command-a-reasoning-08-2025", None, None),
    ("gemini_35flash", "gemini__gemini-3.5-flash", None, None),
    ("gemini_31lite_minimal", "gemini__gemini-3.1-flash-lite", None, {"reasoning_effort": "minimal"}),
    ("gemini_31lite_medium", "gemini__gemini-3.1-flash-lite", None, {"reasoning_effort": "medium"}),
    ("gemini_35lite_medium", "gemini__gemini-3.5-flash-lite", None, {"reasoning_effort": "medium"}),
    ("mistral_codestral", "mistral__codestral-latest", None, None),
    ("mistral_large4", None, ("mistral", "mistral-large-4"), None),
    ("mistral_magistral_medium", None, ("mistral", "magistral-medium-latest"), None),
]


def provider_entry(single: str | None, manual: tuple[str, str] | None, params: dict | None) -> dict:
    """1モデルだけの providers の1要素を作る."""
    if single is not None:
        entry = json.loads((SINGLE_DIR / f"{single}.json").read_text())["providers"][0]
    else:
        name, model = manual
        url, keys_env = MANUAL[name]
        entry = {"name": name, "provider_url": url, "models": [model], "keys_env": keys_env}
    model = entry["models"][0]
    entry["model_params"] = {} if params is None else {model: params}
    entry["priority"] = 1
    return entry


def probe(entry: dict) -> dict:
    """1回送って結果をまとめる."""
    keys = resolve_api_keys(entry["keys_env"])
    if not keys:
        return {"error": "no key"}
    model = entry["models"][0]
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
               "max_tokens": 1200, "stop": ["<end_code>"]}
    payload.update(entry.get("model_params", {}).get(model, {}))
    url = entry["provider_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"}
    started = time.monotonic()
    try:
        response = RequestsTransport().post(url, headers, json.dumps(payload), timeout=120)
    except Exception as error:  # 計測用なので種類を問わず記録する
        return {"error": f"{type(error).__name__}: {error}"[:200], "seconds": round(time.monotonic() - started, 1)}
    seconds = round(time.monotonic() - started, 1)
    if response.status != 200:
        return {"status": response.status, "error": response.body[:200], "seconds": seconds}
    data = json.loads(response.body)
    message = data["choices"][0].get("message") or {}
    content = message.get("content") or ""
    content_is_list = isinstance(content, list)
    if content_is_list:
        # エージェントは str 以外の content を空として扱う（openai_compat._parse_response）
        content = ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    # エージェントは本文が空なら思考の文を代わりに使う（openai_compat._parse_response）
    used = content if content.strip() else reasoning
    _, output_tokens = extract_token_usage(data, used)
    return {
        "status": 200, "seconds": seconds, "content_is_list": content_is_list,
        "content_chars": len(content), "reasoning_chars": len(reasoning),
        "output_tokens": output_tokens,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "content_has_code": bool(re.search(r"```python", content)),
        "used_has_code": bool(re.search(r"```python", used)),
        "content_head": content[:160],
    }


def main() -> None:
    """全部の組を1回ずつ確かめて JSON に書く."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="")
    arguments = parser.parse_args()
    load_dotenv()
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    for label, single, manual, params in CONFIGS:
        if arguments.match and not re.search(arguments.match, label):
            continue
        result = probe(provider_entry(single, manual, params))
        results[label] = result
        print(label, json.dumps({k: v for k, v in result.items() if k != "content_head"}, ensure_ascii=False), flush=True)
        OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        time.sleep(2)


if __name__ == "__main__":
    main()
