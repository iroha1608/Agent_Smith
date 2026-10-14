"""プロバイダごとの複数のAPIキーを環境変数から解決するモジュール.

キーは環境変数(.env)にだけ置き、providers.json には変数名(keys_env)だけを書く。
1プロバイダに複数のキーを持てる(課題 V.6.1「multiple API tokens per provider」):
    GROQ_API_KEY=...      # 1本目(基準名)
    GROQ_API_KEY_2=...    # 2本目
    GROQ_API_KEY_3=...    # 以降、連番が途切れるまで
キーの切り替え(429で次のキー)・バックオフ・別プロバイダへのフォールバックは
ResilientLLMClient が担い、このモジュールはキーの一覧を返すことだけを担う。
"""
import os


def resolve_api_key(keys_env: str) -> str:
    """env変数名 keys_env から実APIキーを取得する.

    Args:
        keys_env (str): APIキーを格納した環境変数名。

    Returns:
        str: APIキー文字列。

    Raises:
        KeyError: 指定された環境変数が存在しない / 未設定の場合。
    """
    # 環境変数からAPIキーを取得
    key = os.environ.get(keys_env)
    # 環境変数が存在しない / 未設定の場合は KeyError を送出
    if not key:
        raise KeyError(f"Environment variable '{keys_env}' is not set.")

    return key


def resolve_api_keys(keys_env: str) -> list[str]:
    """連番サフィックス付きの環境変数から複数のAPIキーを取得する.

    基準名 keys_env ('GROQ_API_KEY')を先頭に
    '{keys_env}_2, '_3', ...を連番が途切れるまで取得する。
    各値はstripし、空文字 / 未設定は除外する。重複は順序を保って除去。

    Args:
        keys_env (str): APIキーを格納した環境変数名。

    Returns:
        list[str]: APIキー文字列のリスト(優先度順)。一本もなければ空リストを返す。
    """
    collected_keys: list[str] = []

    # 基準名の環境変数を取得
    base_key = os.environ.get(keys_env)
    if base_key:
        collected_keys.append(base_key.strip())

    # 連番サフィックス付きの環境変数を取得
    index = 2
    while True:
        raw_key = os.environ.get(f"{keys_env}_{index}")
        if not raw_key or not raw_key.strip():
            break
        collected_keys.append(raw_key.strip())
        index += 1

    # 重複を除去し、順序を保つ
    seen_keys: set[str] = set()
    unique_keys: list[str] = []
    for key in collected_keys:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_keys.append(key)

    return unique_keys
