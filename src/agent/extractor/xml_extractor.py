"""XML形式のツール呼び出しをPythonコードに変換して抽出するモジュール.

Anthropic系のモデルが出す
    <invoke name="read_file"><parameter name="filepath">/testbed/x.py</parameter></invoke>
形式を、サンドボックスがそのまま実行できる
    print(read_file(filepath='/testbed/x.py'))
へ変換する。サンドボックスは形式を知らないまま済む(課題 V.1.2)。
"""
import re

from src.agent.extractor.code_extractor import CodeExtractor

# <invoke name="tool"> ... </invoke> を1件ずつ取り出す
_INVOKE_PATTERN = re.compile(
    r"<invoke\s+name\s*=\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE
)
# <parameter name="key">value</parameter> を1件ずつ取り出す
_PARAMETER_PATTERN = re.compile(
    r"<parameter\s+name\s*=\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE
)


def literal_for(raw: str) -> str:
    """XML / ReAct由来の素の値を、Pythonリテラルの文字列表現にする.

    数値 / 真偽値 / Noneは型を保ち、それ以外は文字列リテラルにする。

    Args:
        raw (str): タグに囲まれていた素の値。

    Returns:
        str: Pythonソースに埋め込めるリテラル表現。
    """
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "True" if lowered == "true" else "False"
    if lowered in {"none", "null"}:
        return "None"
    try:
        return repr(int(value))
    except ValueError:
        pass
    try:
        return repr(float(value))
    except ValueError:
        pass
    return repr(value)


class XMLCodeExtractor(CodeExtractor):
    """XML形式のツール呼び出しをPython呼び出しに変換する具象クラス."""

    def extract(self, llm_output: str) -> str | None:
        """LLM出力中のXMLツール呼び出しをPythonコードに変換して返す.

        Args:
            llm_output (str): LLMが生成した出力テキスト全体。

        Returns:
            str | None: 変換したPythonコード。XML呼び出しが無ければNone。
        """
        invocations = _INVOKE_PATTERN.findall(llm_output)
        if not invocations:
            return None

        lines: list[str] = []
        for tool_name, inner in invocations:
            arguments = [
                f"{key}={literal_for(value)}"
                for key, value in _PARAMETER_PATTERN.findall(inner)
            ]
            lines.append(f"print({tool_name}({', '.join(arguments)}))")
        return "\n".join(lines)
