"""複数フォーマットのコード抽出(課題 V.1.2)のテスト.

非Python形式は「等価なPython関数呼び出しに変換されてからサンドボックスへ渡る」ことが要件。
変換後の文字列が実際にPythonとしてコンパイルでき、ツール名と引数が保たれることを検証する。
"""
from src.agent.extractor.composite_extractor import CompositeCodeExtractor
from src.agent.extractor.json_hermes_extractor import JSONHermesCodeExtractor
from src.agent.extractor.python_extractor import PythonCodeExtractor
from src.agent.extractor.react_extractor import ReactCodeExtractor
from src.agent.extractor.xml_extractor import XMLCodeExtractor


def _compiles(code: str) -> bool:
    """変換結果が構文として成立するか(サンドボックスに渡せるか)を返す."""
    try:
        compile(code, "<extracted>", "exec")
    except SyntaxError:
        return False
    return True


def test_python_block_is_primary() -> None:
    """```python ブロックはそのまま取り出される(主系)."""
    output = "Thought: go\n```python\nprint(1 + 1)\n```\n<end_code>"
    assert PythonCodeExtractor().extract(output) == "print(1 + 1)"


def test_xml_tool_call_becomes_python_call() -> None:
    """XMLツール呼び出しがPython関数呼び出しに変換される."""
    output = (
        'I will read it.\n<invoke name="read_file">'
        '<parameter name="filepath">/testbed/x.py</parameter>'
        '<parameter name="start_line">1</parameter></invoke>'
    )
    code = XMLCodeExtractor().extract(output)
    assert code is not None
    assert _compiles(code)
    assert "read_file(" in code
    assert "filepath='/testbed/x.py'" in code
    assert "start_line=1" in code  # 数値は数値のまま


def test_json_hermes_tool_call_becomes_python_call() -> None:
    """JSON/Hermesツール呼び出しがPython関数呼び出しに変換される."""
    output = '<tool_call>{"name": "search_code", "arguments": {"pattern": "validate_email"}}</tool_call>'
    code = JSONHermesCodeExtractor().extract(output)
    assert code is not None
    assert _compiles(code)
    assert "search_code(" in code
    assert "pattern='validate_email'" in code


def test_react_action_becomes_python_call() -> None:
    """ReActのAction / Action InputがPython関数呼び出しに変換される."""
    output = 'Thought: inspect\nAction: list_files\nAction Input: {"directory": "/testbed", "pattern": "*.py"}'
    code = ReactCodeExtractor().extract(output)
    assert code is not None
    assert _compiles(code)
    assert "list_files(" in code
    assert "directory='/testbed'" in code
    assert "pattern='*.py'" in code


def test_no_recognisable_format_returns_none() -> None:
    """どの形式でもない素の文章はNone(再促し対象)になる."""
    assert CompositeCodeExtractor().extract("I think the answer is 42, but I will not run code.") is None


def test_composite_prefers_python_block() -> None:
    """Pythonブロックと他形式が混在する場合はPythonブロックを優先する."""
    output = (
        '<tool_call>{"name": "read_file", "arguments": {"filepath": "/a"}}</tool_call>\n'
        "```python\nprint('real code')\n```"
    )
    assert CompositeCodeExtractor().extract(output) == "print('real code')"


def test_composite_falls_back_to_tool_call_format() -> None:
    """Pythonブロックが無ければ他形式にフォールバックして変換する."""
    output = 'Action: run_tests\nAction Input: {"code": "def f(): pass", "test_list": ["assert True"]}'
    code = CompositeCodeExtractor().extract(output)
    assert code is not None
    assert _compiles(code)
    assert "run_tests(" in code
