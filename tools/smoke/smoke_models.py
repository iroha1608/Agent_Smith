"""新規プロバイダのモデルを、エージェントと同じ OpenAICompatProvider 経路で1リクエストずつ疎通確認する.

キーは表示しない。結果は JSON で保存する。リポジトリのルートを cwd にして uv run で実行する。
"""
import json
import sys
import time

from dotenv import load_dotenv

from src.agent.llm.key_manager import resolve_api_keys
from src.agent.llm.provider.openai_compat import OpenAICompatProvider
from src.agent.llm.provider.provider import LLMError, LLMHTTPError, ProviderConfig
from src.agent.llm.transport.requests_transport import RequestsTransport
from src.agent.llm.transport.transport import HttpResponse, TransportError

OUT_PATH = sys.argv[1]
ONLY = set(sys.argv[2:])

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

CANDIDATES = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", [
        "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite",
        "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemma-4-31b-it",
    ]),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", [
        "deepseek-ai/deepseek-v4.1-flash", "moonshotai/kimi-k2.6", "moonshotai/kimi-k3", "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash", "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-ultra-550b-a55b", "openai/gpt-oss-20b", "mistralai/codestral-22b-instruct-v0.1",
        "google/gemma-4-31b-it", "mistralai/mistral-large-2-instruct", "poolside/laguna-xs-2.1",
    ]),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", [
        "codestral-latest", "mistral-small-latest", "mistral-medium-latest", "magistral-small-latest",
        "ministral-14b-latest", "mistral-code-latest",
    ]),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY", [
        "command-a-03-2025", "command-a-plus-05-2026", "command-a-reasoning-08-2025", "north-mini-code-1-0",
    ]),
}
SLEEP_BETWEEN_SEC = {"gemini": 4.0, "nvidia": 2.0, "mistral": 2.0, "cohere": 4.0}


class RecordingTransport(RequestsTransport):
    """最後の応答(ヘッダ・本文)を覚えておく Transport."""

    last: HttpResponse | None = None

    def post(self, url: str, headers: dict[str, str], data: str, timeout: float = 60.0) -> HttpResponse:
        """送信して応答を記録する."""
        response = super().post(url, headers, data, timeout)
        self.last = response
        return response


def main() -> None:
    """全候補を順に叩いて結果を保存する."""
    load_dotenv(dotenv_path=".env")  # cwd(リポジトリ直下)の.envをエージェントと同じく環境変数へ読み込む
    results = []
    for provider_name, (base_url, keys_env, models) in CANDIDATES.items():
        if ONLY and provider_name not in ONLY:
            continue
        keys = resolve_api_keys(keys_env)
        if not keys:
            print(f"{provider_name}: no key")
            continue
        for model in models:
            config = ProviderConfig(name=provider_name, provider_url=base_url, models=[model], keys_env=keys_env,
                                    request_timeout_sec=45.0)
            transport = RecordingTransport()
            provider = OpenAICompatProvider(config, transport, keys[0], model)
            record: dict = {"provider": provider_name, "model": model}
            started = time.monotonic()
            try:
                response = provider.generate(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}],
                    stop_sequences=["<end_code>"], max_tokens=600)
                text = response.text
                record.update(ok=True, in_tok=response.input_tokens, out_tok=response.output_tokens,
                              has_fence="```python" in text, chars=len(text), text_head=text[:400])
            except LLMHTTPError as error:
                record.update(ok=False, error=type(error).__name__, status=error.status, message=str(error)[:300])
            except (LLMError, TransportError) as error:
                record.update(ok=False, error=type(error).__name__, message=str(error)[:300])
            record["latency_sec"] = round(time.monotonic() - started, 2)
            if transport.last is not None:
                record["ratelimit_headers"] = {key: value for key, value in transport.last.headers.items()
                                               if "ratelimit" in key or key == "retry-after"}
                try:
                    message = json.loads(transport.last.body)["choices"][0]["message"]
                    record["message_keys"] = sorted(message.keys())
                    record["finish_reason"] = json.loads(transport.last.body)["choices"][0].get("finish_reason")
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
            status = "OK " if record.get("ok") else "NG "
            print(f"{status}{provider_name:8s} {model:42s} {record['latency_sec']:6.1f}s "
                  f"out={record.get('out_tok', '-')} fence={record.get('has_fence', '-')} "
                  f"fin={record.get('finish_reason', '-')} {record.get('error', '')} {record.get('status', '')}",
                  flush=True)
            results.append(record)
            with open(OUT_PATH, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
            time.sleep(SLEEP_BETWEEN_SEC[provider_name])
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
