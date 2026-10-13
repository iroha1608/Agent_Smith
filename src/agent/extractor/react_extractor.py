"""ReAct形式のツール呼び出しをPythonコードに変換して抽出するモジュール.

ReActで訓練されたモデルが出す
    Action: read_file
    Action Input: {"filepath": "/testbed/x.py"}
形式を、サンドボックスがそのまま実行できる
    print(read_file(filepath='/testbed/x.py'))
へ変換する(課題 V.1.2)。Action Input はJSONでも `key=value` の素形式でも受ける。
"""
import json
import re
from typing import Any

from src.agent.extractor.code_extractor import CodeExtractor
from src.agent.extractor.xml_extractor import literal_for

# Action: <tool>  と、続く Action Input: <rest of line / JSON>
_ACTION_PATTERN = re.compile(
    r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n+\s*Action\s*Input\s*:\s*(.+?)(?=\n\s*(?:Action|Thought|Observation)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE
)


def _arguments_source(raw: str) -> str:
    """Action Input をPythonのキーワード引数列に変換する.

    JSONオブジェクトならキーごとに展開し、そうでなければ
    `key=value` のカンマ区切り、いずれでもなければ単一の位置引数として扱う。

    Args:
        raw (str): Action Input の本文。

    Returns:
        str: 呼び出しに埋め込む引数列。
    """
    text = raw.strip()
    if text.startswith("{"):
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return ", ".join(
                f"{key}={value!r}" for key, value in payload.items() if str(key).isidentifier()
            )
    if "=" in text:
        pairs = []
        for chunk in text.split(","):
            key, _, value = chunk.partition("=")
            if key.strip().isidentifier():
                pairs.append(f"{key.strip()}={literal_for(value)}")
        if pairs:
            return ", ".join(pairs)
    return literal_for(text) if text else ""


class ReactCodeExtractor(CodeExtractor):
    """ReAct形式のツール呼び出しをPython呼び出しに変換する具象クラス."""

    def extract(self, llm_output: str) -> str | None:
        """LLM出力中のReActアクションをPythonコードに変換して返す.

        Args:
            llm_output (str): LLMが生成した出力テキスト全体。

        Returns:
            str | None: 変換したPythonコード。Actionが無ければNone。
        """
        actions = _ACTION_PATTERN.findall(llm_output)
        if not actions:
            return None
        lines = [
            f"print({tool_name}({_arguments_source(action_input)}))"
            for tool_name, action_input in actions
        ]
        return "\n".join(lines)
