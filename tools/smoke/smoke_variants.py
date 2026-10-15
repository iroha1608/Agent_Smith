"""推論系モデルについて「推論OFFパラメータ」「stop有無」の組合せで本文が出るかを確かめる.

生のHTTPボディ(content / reasoning_content の長さと先頭)を記録する。キーは表示しない。
リポジトリ直下を cwd にし、PYTHONPATH=. で実行する。
"""
import json
from typing import Any
import sys
import time

import requests
from dotenv import load_dotenv

from src.agent.llm.key_manager import resolve_api_keys

OUT_PATH = sys.argv[1]

SYSTEM_PROMPT = (
    "You are a Python coding agent. Reply with a short line starting with 'Thought:' and then exactly one "
    "fenced block ```python ... ``` that defines the requested function and checks it with the given asserts. "
    "End the block with <end_code>."
)
USER_PROMPT = (
    "Task: Write a function to find the shared elements from the given two lists.\n"
    "Tests:\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"
    "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))"
)

BASE = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY"),
}

# (provider, model, 追加パラメータ, stopを付けるか)
VARIANTS: list[tuple[str, str, dict[str, Any], bool]] = [
    ("nvidia", "deepseek-ai/deepseek-v4.1-flash", {"chat_template_kwargs": {"thinking": False}}, True),
    ("nvidia", "deepseek-ai/deepseek-v4.1-flash", {}, False),
    ("nvidia", "z-ai/glm-5.3-flash", {"chat_template_kwargs": {"enable_thinking": False}}, True),
    ("nvidia", "z-ai/glm-5.3-flash", {}, False),
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b", {"chat_template_kwargs": {"enable_thinking": False}}, True),
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b", {}, False),
    ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", {"chat_template_kwargs": {"enable_thinking": False}}, True),
    ("nvidia", "openai/gpt-oss-20b", {"reasoning_effort": "low"}, True),
    ("cohere", "command-a-plus-05-2026", {"reasoning_effort": "none"}, True),
    ("cohere", "command-a-plus-05-2026", {}, False),
    ("cohere", "command-a-reasoning-08-2025", {"reasoning_effort": "none"}, True),
    ("gemini", "gemini-2.5-flash", {}, True),
    ("gemini", "gemini-3.1-flash-lite", {}, True),
    ("gemini", "gemini-3.5-flash", {}, True),
    ("gemini", "gemini-3.8-flash", {}, True),
    ("mistral", "mistral-small-latest", {}, True),
]


def main() -> None:
    """各バリアントを1回ずつ叩き、生のレスポンス要約を保存する."""
    load_dotenv(dotenv_path=".env")  # cwd(リポジトリ直下)の.envを環境変数へ読み込む
    results = []
    for provider_name, model, extra_params, use_stop in VARIANTS:
        base_url, keys_env = BASE[provider_name]
        keys = resolve_api_keys(keys_env)
        payload: dict[str, Any] = {
            "model": model, "max_tokens": 600,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}],
        }
        if use_stop:
            payload["stop"] = ["<end_code>"]
        payload.update(dict(extra_params))
        record: dict[str, Any] = {"provider": provider_name, "model": model, "params": extra_params, "stop": use_stop}
        started = time.monotonic()
        try:
            response = requests.post(f"{base_url}/chat/completions", timeout=45,
                                     headers={"Authorization": f"Bearer {keys[0]}"}, json=payload)
            record["status"] = response.status_code
            try:
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                record.update(finish=choice.get("finish_reason"), content_len=len(content),
                              reasoning_len=len(reasoning), has_fence="```python" in content,
                              content_head=content[:500], usage=body.get("usage"))
            except (ValueError, KeyError, IndexError, TypeError):
                record["body"] = response.text[:400]
        except requests.RequestException as error:
            record["error"] = str(error)[:200]
        record["latency_sec"] = round(time.monotonic() - started, 1)
        print(f"{provider_name:7s} {model:38s} stop={use_stop!s:5s} {json.dumps(extra_params):48s} "
              f"st={record.get('status')} fin={record.get('finish')} content={record.get('content_len')} "
              f"reason={record.get('reasoning_len')} fence={record.get('has_fence')} {record['latency_sec']}s "
              f"{record.get('body', '')[:150]}{record.get('error', '')[:80]}", flush=True)
        results.append(record)
        with open(OUT_PATH, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        time.sleep(3)


if __name__ == "__main__":
    main()
