"""config/settings の読み込み・選択・階層の検証(pydantic-settings)."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import load_settings
# load_providers/select_provider のテストは 07 の §4 のまま(同居なら同一import)
from src.config.settings import load_providers, order_providers, select_provider


def _write(path: Path, obj: object) -> str:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """config.jsonが無ければ既定値(CWDにファイル無し)."""
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.limits.max_input_tokens == 6000
    assert settings.manual.verbosity == "concise"


def test_load_settings_reads_config_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """config.jsonの値が既定を上書き、未指定フィールドは既定のまま."""
    (tmp_path / "config.json").write_text(
        json.dumps({"limits": {"max_iterations": 3}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.limits.max_iterations == 3        # ファイル値
    assert settings.limits.max_input_tokens == 6000   # 未指定は既定


def test_cli_override_is_highest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI(init) が config.json より優先される(D29最優先)."""
    (tmp_path / "config.json").write_text(
        json.dumps({"limits": {"max_iterations": 3}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    settings = load_settings(limits={"max_iterations": 7})
    assert settings.limits.max_iterations == 7


def test_settings_rejects_unknown_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """未知キーは extra=forbid で ValidationError."""
    (tmp_path / "config.json").write_text(
        json.dumps({"limits": {"max_iteration": 3}}), encoding="utf-8"  # sが無い
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        load_settings()


# --- providers 系(07 §4のまま。パス引数で渡すので chdir 不要) ---
_PROVIDERS = {
    "providers": [
        {"name": "groq", "provider_url": "https://api.groq.com/openai/v1",
         "models": ["llama-3.3-70b-versatile"], "keys_env": "GROQ_API_KEY", "priority": 2},
        {"name": "openrouter", "provider_url": "https://openrouter.ai/api/v1",
         "models": ["qwen/qwen3-235b-a22b-2507"], "keys_env": "OPENROUTER_API_KEY", "priority": 1},
    ]
}


def test_load_providers_sorted_by_priority(tmp_path: Path) -> None:
    """priority昇順で先頭が主(openrouter)."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    assert [p.name for p in providers] == ["openrouter", "groq"]


def test_select_provider_cli_overrides_primary(tmp_path: Path) -> None:
    """CLIが主のmodel/urlを上書きし keys_env は維持."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    primary = select_provider(providers, model_name="x/y", provider_url="https://eval.test/v1")
# test_select_provider_cli_overrides_primary:
    assert primary.models == ["x/y"]          # model -> models
    assert primary.provider_url == "https://eval.test/v1"
    assert primary.keys_env == "OPENROUTER_API_KEY"


def test_shipped_config_files_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """リポジトリ直下の実 config.json / providers.json がそのままロードできる(出荷物の回帰)."""
    repo_root = Path(__file__).resolve().parents[1]   # test/ の1つ上 = リポジトリ直下
    monkeypatch.chdir(repo_root)                        # Settings は CWD の config.json を読む
    settings = load_settings()
    assert settings.limits.max_input_tokens == 6000
    assert settings.logging.level == "INFO"
# test_shipped_config_files_parse:
    providers = load_providers(str(repo_root / "providers" / "providers.json"))
    assert providers[0].priority == 1                   # priority昇順で先頭が主


def test_order_providers_uses_matching_provider_keys(tmp_path: Path) -> None:
    """--provider-url が既知のプロバイダなら、そのキーで指定モデルを先頭に試す(主エントリのキーを流用しない)."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    ordered = order_providers(providers, model_name="x/y", provider_url="https://api.groq.com/openai/v1/")
    assert ordered[0].name == "groq"
    assert ordered[0].keys_env == "GROQ_API_KEY"
    assert ordered[0].models == ["x/y", "llama-3.3-70b-versatile"]
    assert [p.name for p in ordered[1:]] == ["openrouter"]


def test_order_providers_matches_by_model_name(tmp_path: Path) -> None:
    """URLの指定が無くても、モデル名を持つプロバイダを先頭にする."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    ordered = order_providers(providers, model_name="llama-3.3-70b-versatile")
    assert ordered[0].name == "groq"
    assert len(ordered) == 2


def test_order_providers_unknown_url_guesses_keys_env(tmp_path: Path) -> None:
    """未知のURLはホスト名からキーの環境変数名を推測し、既存プロバイダはフォールバックに残す."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    ordered = order_providers(providers, model_name="m", provider_url="https://api.together.xyz/v1")
    assert ordered[0].keys_env == "TOGETHER_API_KEY"
    assert ordered[0].models == ["m"]
    assert [p.name for p in ordered[1:]] == ["openrouter", "groq"]


def test_order_providers_without_cli_keeps_priority(tmp_path: Path) -> None:
    """指定が無ければpriority順のまま."""
    providers = load_providers(_write(tmp_path / "providers.json", _PROVIDERS))
    assert [p.name for p in order_providers(providers)] == ["openrouter", "groq"]


def test_shipped_swebench_providers_parse() -> None:
    """SWE-bench用の providers_swebench.json も読み込め、1分あたりの枠が小さい groq を含まない."""
    repo_root = Path(__file__).resolve().parents[1]
    providers = load_providers(str(repo_root / "providers" / "providers_swebench.json"))
    names = [provider.name for provider in providers]
    assert names[0] == "cohere"
    assert "groq" not in names
