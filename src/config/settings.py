"""既定 -> ファイル(providers.json / config.json) -> CLIの順で設定を読み込むモジュール.

config.jsonはpydantic-settingsのBaseSettingsで読む。
優先度: CLI(init) > config.json > 環境変数 > .env > コード既定値。
APIキーは.envからのみ。
providers.jsonにはキーの値ではなく、keys_env(env変数名)だけ書く。
providers.jsonは構造化データなので、pydanticでProviderConfigモデルに変換して扱う。
pydantic-settingsを使うと、環境変数やJSON/YAMLファイルから設定を読み込むことができる。
ハード制限はベンチマークごとに持つ(limits: MBPP / swebench_limits: SWE-bench)。
"""
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict
)

from src.agent.llm.provider.provider import ProviderConfig


class Limits(BaseModel):
    """反復 / トークン / 時間のハード制限値.監視、打ち切りはlimits.pyで行う."""
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = 10
    max_input_tokens: int = 6000
    max_output_tokens: int = 1500
    max_output_tokens_per_iteration: int = 500
    total_time_sec: float = 120.0
    time_margin_sec: float = 10.0


def _swebench_limits() -> "Limits":
    """SWE-benchの既定のハード制限(課題 VI.1.2: 30反復 / 入力30万 / 出力1万 / 900秒).

    time_margin_sec はDockerコンテナの停止・削除とパッチ回収の時間を見込んで長めに取る。
    """
    return Limits(
        max_iterations=30,
        max_input_tokens=300_000,
        max_output_tokens=10_000,
        max_output_tokens_per_iteration=1_200,
        total_time_sec=900.0,
        time_margin_sec=60.0,
    )


class SandboxSettings(BaseModel):
    """サンドボックス実行の設定."""
    model_config = ConfigDict(extra="forbid")

    exec_timeout_sec: float = 10.0
    memory_limit_mb: int = 512


class LoggingSettings(BaseModel):
    """ログ設定(レベル / 出力パス / トレース有無)."""
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    path: str = "logs/agent.log"
    trace: bool = False


class ManualSettings(BaseModel):
    """ツールマニュアルの詳細度"""
    model_config = ConfigDict(extra="forbid")

    verbosity: str = "concise"


class RetrySettings(BaseModel):
    """ResilientLLMClientのリトライ / バックオフ設定.

    Attributes:
        max_retries_per_key (int): 1つのAPIキーでの最大リトライ回数。初回除くデフォルト2回。
        backoff_base_seconds (float): バックオフの基本秒数。指数バックオフの基準値。デフォルト1秒。
        backoff_max_seconds (float): バックオフの1回あたりの上限、最大秒数。デフォルト8秒。
        safety_margin_seconds (float): 時間予算からこの分を引いた範囲でしか持たない。
    """
    model_config = ConfigDict(extra="forbid")

    max_retries_per_key: int = 2
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 8.0
    safety_margin_seconds: float = 2.0


class Settings(BaseSettings):
    """config.json + env + CLIの階層設定.

    sourceの優先度:
    init(CLI) > config.json > 環境変数 > .env > コード既定値
    """
    model_config = SettingsConfigDict(
        json_file="config.json",
        json_file_encoding="utf-8",
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    limits: Limits = Limits()
    swebench_limits: Limits = Field(default_factory=_swebench_limits)
    sandbox: SandboxSettings = SandboxSettings()
    logging: LoggingSettings = LoggingSettings()
    manual: ManualSettings = ManualSettings()
    retry: RetrySettings = RetrySettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """設定ソースの優先度を定義する.

        Args:
            settings_cls (type[BaseSettings]): 設定クラスの型。
            init_settings (PydanticBaseSettingsSource): CLI引数などの初期化設定。
            env_settings (PydanticBaseSettingsSource): 環境変数からの設定。
            dotenv_settings (PydanticBaseSettingsSource): .envファイルからの設定。
            file_secret_settings (PydanticBaseSettingsSource): ファイルシークレットからの設定。
        Returns:
            tuple[PydanticBaseSettingsSource, ...]: 設定ソースの優先度。
        """
        return (
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_providers(providers_path: str) -> list[ProviderConfig]:
    """providers.jsonを読み込んでProviderConfigのリストに変換する.

    Args:
        providers_path (str): providers.jsonのパス。デフォルトは"providers.json"。

    Returns:
        list[ProviderConfig]: 読み込んだプロバイダ設定のリスト。

    Raises:
        FileNotFoundError: providers.jsonが存在しない、または空の場合。
        ValueError: providers.jsonに'providers'エントリがない場合。
    """
    path = Path(providers_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"providers.json not found or empty: {providers_path}")

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = raw.get("providers", [])
    if not entries:
        raise ValueError(f"providers.json has no 'providers' entries: {providers_path}")

    providers: list[ProviderConfig] = [ProviderConfig(**entry) for entry in entries]
    return sorted(providers, key=lambda provider: provider.priority)  # priorityが小さい順にソートして返す


def select_provider(
    providers: list[ProviderConfig],
    model_name: str | None = None,
    provider_url: str | None = None
) -> ProviderConfig:
    """primaryプロバイダを1つ選ぶ。CLI引数で指定された場合はそれを優先する。

    priorityが最小を最優先し、--model_name / --provider_urlで指定された場合は上書きする。
    keys_envは主エントリのまま。
    残りのプロバイダはフォールバック用に保持する。
    """
    if not providers:
        raise ValueError("No providers available to select from.")

    primary = providers[0]  # priorityが最小のものをprimaryとする
    updates: dict[str, Any] = {}
    if provider_url:
        updates["provider_url"] = provider_url
    if model_name:
        updates["models"] = [model_name]

    return primary.model_copy(update=updates) if updates else primary


def _same_url(left: str, right: str) -> bool:
    """末尾のスラッシュと大文字小文字の違いを無視してURLを比べる."""
    return left.rstrip("/").lower() == right.rstrip("/").lower()


def _guess_keys_env(provider_url: str) -> str | None:
    """未知のプロバイダURLから、APIキーの環境変数名を推測する(例: api.mistral.ai -> MISTRAL_API_KEY).

    Args:
        provider_url (str): プロバイダのURL。

    Returns:
        str | None: 推測した環境変数名。推測できなければNone。
    """
    host = urlsplit(provider_url).hostname or ""
    labels = [label for label in host.split(".") if label not in ("api", "www", "")]
    if len(labels) < 2:
        return None
    return f"{labels[-2].upper().replace('-', '_')}_API_KEY"


def order_providers(
    providers: list[ProviderConfig],
    model_name: str | None = None,
    provider_url: str | None = None
) -> list[ProviderConfig]:
    """CLI引数(--model-name / --provider-url)を反映した、試行順のプロバイダ列を返す.

    指定が無ければpriority順のまま。指定があれば、URL(無ければモデル名)が一致する設定を
    探して先頭に置き、そのプロバイダのキー(keys_env)とモデル別パラメータをそのまま使う。
    指定モデルを先頭にし、同じプロバイダの他モデルと他プロバイダはフォールバックとして残す。
    どれにも一致しない未知のプロバイダは、URLからキーの環境変数名を推測して新しい設定を作る
    (主エントリのキーを流用すると、別会社にキーを送ってしまうため)。

    Args:
        providers (list[ProviderConfig]): priority順のプロバイダ設定。
        model_name (str | None): CLIで指定されたモデル名。
        provider_url (str | None): CLIで指定されたプロバイダURL。

    Returns:
        list[ProviderConfig]: 先頭が主、残りがフォールバックのプロバイダ列。

    Raises:
        ValueError: providersが空の場合。
    """
    if not providers:
        raise ValueError("No providers available to select from.")
    if not model_name and not provider_url:
        return list(providers)

    matched: int | None = None
    if provider_url:
        matched = next((index for index, provider in enumerate(providers)
                        if _same_url(provider.provider_url, provider_url)), None)
    if matched is None and model_name:
        matched = next((index for index, provider in enumerate(providers)
                        if model_name in provider.models), None)

    if matched is None:
        primary = select_provider(providers, model_name, provider_url)
        guessed = _guess_keys_env(provider_url) if provider_url else None
        if guessed and guessed != primary.keys_env:
            primary = primary.model_copy(update={
                "name": guessed.removesuffix("_API_KEY").lower(),
                "keys_env": guessed,
                "model_params": {},
                "additional_params": {},
            })
        return [primary, *providers]

    base = providers[matched]
    updates: dict[str, Any] = {}
    if provider_url:
        updates["provider_url"] = provider_url
    if model_name:
        updates["models"] = [model_name, *[model for model in base.models if model != model_name]]
    first = base.model_copy(update=updates) if updates else base
    rest = [provider for index, provider in enumerate(providers) if index != matched]
    return [first, *rest]


def load_settings(**cli_overrides: Any) -> Settings:
    """Settingsを構築して返す.

    config.json + env + CLI overridesを階層的に読み込む。

    Args:
        **cli_overrides: CLIからの上書き
    """
    return Settings(**cli_overrides)
