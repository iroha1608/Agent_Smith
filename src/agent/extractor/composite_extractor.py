"""複数フォーマットを順に試すコード抽出器を提供するモジュール.

課題 V.1.2 は「Pythonコードブロックだけでなく、モデルが実際に出す形式
(XMLツール呼び出し / JSON-Hermes / ReAct)も扱い、非Python形式は等価な
Python関数呼び出しに変換してからサンドボックスへ渡す」ことを求める。
このクラスは優先度順に具象抽出器を試し、最初に取れた結果を返すだけに留め、
形式ごとの知識は各具象クラスに閉じ込める。
"""
from src.agent.extractor.code_extractor import CodeExtractor
from src.agent.extractor.json_hermes_extractor import JSONHermesCodeExtractor
from src.agent.extractor.python_extractor import PythonCodeExtractor
from src.agent.extractor.react_extractor import ReactCodeExtractor
from src.agent.extractor.xml_extractor import XMLCodeExtractor


class CompositeCodeExtractor(CodeExtractor):
    """登録順に抽出を試し、最初に成功した結果を返す抽出器."""

    def __init__(self, extractors: list[CodeExtractor] | None = None) -> None:
        """抽出器の優先順を決めて初期化する.

        Args:
            extractors (list[CodeExtractor] | None): 試す順に並べた抽出器。
                Noneなら Python -> JSON/Hermes -> XML -> ReAct の既定順。
                Pythonブロックが主系なので最優先に置く。
        """
        self.extractors = extractors if extractors is not None else [
            PythonCodeExtractor(),
            JSONHermesCodeExtractor(),
            XMLCodeExtractor(),
            ReactCodeExtractor(),
        ]

    def extract(self, llm_output: str) -> str | None:
        """登録順に抽出を試し、最初に得られたコードを返す.

        Args:
            llm_output (str): LLMが生成した出力テキスト全体。

        Returns:
            str | None: 実行可能なPythonコード。どの形式でも取れなければNone。
        """
        for extractor in self.extractors:
            code = extractor.extract(llm_output)
            if code is not None and code.strip():
                return code
        return None
