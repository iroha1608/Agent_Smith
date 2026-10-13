"""JSON / Hermes形式のツール呼び出しをPythonコードに変換して抽出するモジュール.

Hermes系のモデルが出す
    <tool_call>{"name": "read_file", "arguments": {"filepath": "/testbed/x.py"}}</tool_call>
形式を、サンドボックスがそのまま実行できる
    print(read_file(filepath='/testbed/x.py'))
へ変換する(課題 V.1.2)。<tool_call>タグが無い素のJSONオブジェクトにも対応する。
"""
import json
import re
from typing import Any

from src.agent.extractor.code_extractor import CodeExtractor

# <tool_call>{...}</tool_call>
_TOOL_CALL_TAG_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
# タグ無しで "name" と "arguments" を持つJSONオブジェクト
_BARE_JSON_PATTERN = re.compile(r"\{[^{}]*\"name\"\s*:.*?\"arguments\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)


def _call_source(payload: dict[str, Any]) -> str | None:
    """ツール呼び出しJSONを1行のPython呼び出し文に変換する.

    Args:
        payload (dict[str, Any]): {"name": ..., "arguments": {...}} 形式の辞書。

    Returns:
        str | None: Python呼び出し文。name が無ければNone。
    """
    tool_name = payload.get("name")
    if not isinstance(tool_name, str) or not tool_name.isidentifier():
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    rendered = [f"{key}={value!r}" for key, value in arguments.items() if str(key).isidentifier()]
    return f"print({tool_name}({', '.join(rendered)}))"


class JSONHermesCodeExtractor(CodeExtractor):
    """JSON / Hermes形式のツール呼び出しをPython呼び出しに変換する具象クラス."""

    def extract(self, llm_output: str) -> str | None:
        """LLM出力中のJSONツール呼び出しをPythonコードに変換して返す.

        Args:
            llm_output (str): LLMが生成した出力テキスト全体。

        Returns:
            str | None: 変換したPythonコード。JSON呼び出しが無ければNone。
        """
        candidates = _TOOL_CALL_TAG_PATTERN.findall(llm_output) or _BARE_JSON_PATTERN.findall(llm_output)
        lines: list[str] = []
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            source = _call_source(payload)
            if source is not None:
                lines.append(source)
        return "\n".join(lines) if lines else None
