"""複数プロバイダ、複数キーを束ね、冗長性と耐障害性を提供するLLMクライアント

provider(1キーで1リクエスト)の上位レイヤ。
priorityに応じてprovider内で複数のキーを回し、
型付きエラーに応じて
バックオフ再送 -> 次キー -> 次プロバイダ
の順にフォールバックする。
すべて尽きたらAllProvidersExhausted返す。
上位Orchestratorはクラッシュさせず、success=Falseで返す。

- RateLimitError / ServerError / TransportError
    一過性。バックオフして同一キーで再送。
- AuthenticationError
    同一キー再送は無意味。即次のキーへ。
- BadRequestError
    リクエスト不正。他キー / 他プロバイダでも直らないので即送出。

time_remainingを渡すと、待機がMBPPのハード制限を食いつぶさないように
残予算にクランプする。
残予算がなければそのキーを諦めて次のキーへ。
backoff / retry の設定はRetrySettings(config.json)。
"""
import time
import logging
from collections.abc import Callable

from src.agent.llm.key_manager import resolve_api_keys
from src.agent.llm.provider.openai_compat import OpenAICompatProvider
from src.agent.llm.provider.provider import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    RateLimitError,
    ServerError
)
from src.agent.llm.transport.transport import HttpTransport, TransportError
from src.config.settings import RetrySettings

_LOGGER = logging.getLogger("agent.llm")

# (config, transport, api_key, model) -> LLMProvider を生成するファクトリ型。
# 既定はOpenAICompatProvider
ProviderFactory = Callable[[ProviderConfig, HttpTransport, str, str], LLMProvider]

# 全プロバイダ / 全キー / 全モデルを巡る「周回」の最大数。
# 1周目は待たずに全組合せを試し、全滅かつ全てレート制限だった場合のみ、
# 残り時間に収まる待機を1度だけ入れて2周目を試す(待てば解ける429だけを救う)。
_MAX_SWEEPS = 2


class AllProvidersExhausted(LLMError):
    """全プロバイダ・全キーを試しても成功しなかったことを表す例外クラス.

    Attributes:
        attempts (int): 試行回数。成功までのリトライを含む。
            上位のOrchestratorはこの例外を捕まえて、total_requestsに加算する。
        retry_after (float | None): レート制限が解除されるまでの最短待機秒数の推定。
            Noneは「レート制限以外の理由で全滅した(待っても無駄)」を意味する。
    """
    def __init__(self, message: str, attempts: int = 0, retry_after: float | None = None) -> None:
        super().__init__(message, attempts, retry_after)
        self.attempts = attempts
        self.retry_after = retry_after


class ResilientLLMClient:
    """複数プロバイダ / 複数キーで冗長性と耐障害性を提供するProviderの上位クライアント.

    OrchestratorはProviderと同じ
    gengerate(messages, stop_sequences, max_tokens)を呼び出すだけでよい。
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        transport: HttpTransport,
        retry: RetrySettings,
        provider_factory: ProviderFactory = OpenAICompatProvider,
        sleep: Callable[[float], None] = time.sleep,
        time_remaining: Callable[[], float] | None = None,
        logger: logging.Logger | None = None
    ) -> None:
        """依存を注入して初期化。

        Args:
            providers (list[ProviderConfig]): プロバイダ設定のリスト。優先度順に並べる。
            transport (HttpTransport): HTTP POST で送信を担当するTransportクラス
            retry (RetrySettings): バックオフ / リトライの設定
            provider_factory (ProviderFactory): ProviderConfigからLLMProviderを生成するファクトリ関数
            sleep (Callable[[float], None]): 待機関数。time.sleepの代替を注入可能。
            time_remaining (Callable[[], float] | None): 残り時間を返す関数。Noneなら無制限。
            logger (logging.Logger | None): ロガー。Noneならデフォルトロガーを使用。
        """
        self._providers = providers
        self._transport = transport
        self._retry = retry
        self._provider_factory = provider_factory
        self._sleep = sleep
        self._time_remaining = time_remaining
        self._log = logger if logger is not None else _LOGGER

    def _wait_seconds(self, error: BaseException, attempt_on_key: int) -> float | None:
        """次の再送までの待機秒数を返す。残予算がなければNone.

        429でretry_afterがあればそれを優先する。
        無ければ指数バックオフ(base * 2^n をmaxまでクランプ)。
        time_remainingがあれば残予算 - safety_margin_secondsにクランプする。
        0以下ならNoneを返す。
        """
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            wait = error.retry_after
        else:
            wait = min(self._retry.backoff_base_seconds * (2 ** attempt_on_key), self._retry.backoff_max_seconds)

        if self._time_remaining is not None:
            budget = self._time_remaining() - self._retry.safety_margin_seconds
            if budget <= 0:
                return None
            wait = min(wait, budget)

        return wait

    def _sweep_wait(self, retry_after: float | None) -> float | None:
        """全組合せ全滅のあと、もう1周するために待つ秒数を返す。待っても無駄ならNone.

        待って解決するのはレート制限(429)だけなので、周回中に429を観測していない
        (retry_afterがNone)場合は待たない。観測していれば、その最短待機秒数が
        残り時間に収まるときに限って待つ。収まらない(日次枠切れ等)なら諦める。

        Args:
            retry_after (float | None): 周回中に観測した最短の待機秒数。Noneなら429なし。

        Returns:
            float | None: 待機秒数。待つ意味がなければNone。
        """
        if retry_after is None:
            return None
        if self._time_remaining is None:
            return retry_after
        budget = self._time_remaining() - self._retry.safety_margin_seconds
        # 待っても制限解除が残り時間に収まらないなら、待たずに諦める
        if budget <= 0 or retry_after > budget:
            return None
        return retry_after

    def generate(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str],
        max_tokens: int
    ) -> LLMResponse:
        """粘り付きで1回の生成を行い、成功したLLMResponseを返す.

        まず全プロバイダ / 全キー / 全モデルを待たずに1周する。全滅かつ原因が
        「残り時間内に解除されるレート制限」のときだけ、その秒数を待ってもう1周する。

        Args:
            messages (list[dict[str, str]]): LLMに送信するメッセージ。OpenAI互換の形式で、roleとcontentを持つ辞書のリスト。
            stop_sequences (list[str]): 生成を停止する文字列郡(既定は ["<end_code>"])
            max_tokens (int): 生成する最大トークン数

        Returns:
            LLMResponse: 生成テキストと使用量、計測値をまとめた結果。

        Raises:
            AllProvidersExhausted: すべてのプロバイダ / キー / モデルを試しても成功しなかった場合。
            BadRequestError: リクエスト不正。上位のOrchestratorはこの例外を捕まえて、success=Falseで返す。
        """
        attempts = 0
        last_error: BaseException | None = None

        for sweep in range(_MAX_SWEEPS):
            try:
                return self._sweep(messages, stop_sequences, max_tokens, attempts)
            except AllProvidersExhausted as exhausted:
                attempts = exhausted.attempts
                last_error = exhausted.__cause__ or exhausted
                wait = self._sweep_wait(exhausted.retry_after)
                if sweep + 1 >= _MAX_SWEEPS or wait is None:
                    break
                self._log.warning(
                    "all providers/keys/models exhausted after %d attempt(s); "
                    "waiting %.1fs for the rate limit to clear, then retrying once",
                    attempts,
                    wait
                )
                self._sleep(wait)

        raise AllProvidersExhausted(
            f"All providers/keys/models exhausted after {attempts} attempt(s).",
            attempts=attempts
        ) from last_error

    def _sweep(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str],
        max_tokens: int,
        attempts_so_far: int
    ) -> LLMResponse:
        """全プロバイダ / 全キー / 全モデルを優先度順に1周だけ試す.

        Args:
            messages (list[dict[str, str]]): LLMに送信するメッセージ。
            stop_sequences (list[str]): 生成を停止する文字列郡。
            max_tokens (int): 生成する最大トークン数。
            attempts_so_far (int): 前の周回までに送信した回数(通算の試行回数を引き継ぐ)。

        Returns:
            LLMResponse: 生成テキストと使用量、計測値をまとめた結果。

        Raises:
            AllProvidersExhausted: この周回で全組合せが失敗した場合。
            BadRequestError: リクエスト不正。
        """
        attempts = attempts_so_far
        last_error: BaseException | None = None
        # 周回中に観測した429の待機秒数。retry_afterが無い429は上限秒で代用し、
        # 最短のものを「待てば解けるかもしれない秒数」として上位へ返す。
        rate_limit_waits: list[float] = []

        for config in self._providers:
            # config.keys_envからAPIキーを取得する。もしkeysが空なら、次のプロバイダへ。
            keys = resolve_api_keys(config.keys_env)
            if not keys:
                self._log.warning(
                    "provider=%s skipped: no API keys in %s",
                    config.name,
                    config.keys_env
                )
                continue

            # すべてのキーを順番に試す。1キーで1リクエスト。
            for key_number, key in enumerate(keys, start=1):
                for model in config.models:
                    # Providerを生成して、generate()を呼ぶ。
                    provider = self._provider_factory(config, self._transport, key, model)

                    # 1キーでのリトライループ。バックオフ再送 -> 次キー -> 次プロバイダ の順にフォールバックする。
                    for attempt_on_key in range(self._retry.max_retries_per_key + 1):
                        attempts += 1

                        try:
                            response = provider.generate(messages, stop_sequences, max_tokens)

                        # 1. RateLimitErrorは次のキーへ。バックオフはしない。
                        except RateLimitError as error:
                            last_error = error
                            rate_limit_waits.append(
                                error.retry_after if error.retry_after is not None
                                else self._retry.backoff_max_seconds
                            )
                            self._log.warning(
                                "provider=%s key#%d attempts=%d rate limited (retry_after=%s); advance, no backoff.",
                                config.name,
                                key_number,
                                attempts,
                                error.retry_after
                            )
                            break

                        # 2. ServerError / 3. TransportError は一過性(500系 / ネットワークエラー)。バックオフして同一キーで再送。
                        except (ServerError, TransportError) as error:
                            last_error = error
                            # 1つのキーでのリトライ回数が上限に達したら、次のキーへ。
                            if attempt_on_key >= self._retry.max_retries_per_key:
                                self._log.warning(
                                    "provider=%s key#%d attempts=%d %s; give up key",
                                    config.name,
                                    key_number,
                                    attempts,
                                    type(error).__name__
                                )
                                break

                            # バックオフ秒数を計算し、残予算がなければ次のキーへ。
                            wait = self._wait_seconds(error, attempt_on_key)
                            if wait is None:
                                self._log.warning(
                                    "provider=%s key#%d attempts=%d %s; no time budget, give up key.",
                                    config.name,
                                    key_number,
                                    attempts,
                                    type(error).__name__
                                )
                                break

                            # バックオフ秒数待機し、同一キーで再送する。
                            self._log.warning(
                                "provider=%s key#%d attempts=%d %s; backoff %.1fs.",
                                config.name,
                                key_number,
                                attempts,
                                type(error).__name__,
                                wait
                            )
                            self._sleep(wait)

                        # 4. 認証エラー(401 / 403)は次のキーへ。バックオフはしない。
                        except AuthenticationError as error:
                            last_error = error
                            self._log.warning(
                                "provider=%s key#%d attempts=%d auth/quota; next key.",
                                config.name,
                                key_number,
                                attempts
                            )
                            break

                        # 5. BadRequestErrorはリクエスト不正。
                        except BadRequestError as error:
                            last_error = error
                            self._log.error(
                                "provider=%s key#%d bad request; skip key: %s.",
                                config.name,
                                key_number,
                                error
                            )
                            break

                        # 6. その他の想定外LLM層失敗(JSONパース失敗 / content=null 等)。
                        #    クラッシュさせず、この組合せを飛ばして次へ(全滅ならAllProvidersExhausted)。
                        except LLMError as error:
                            last_error = error
                            self._log.warning(
                                "provider=%s key#%d attempts=%d unexpected LLM error; advance: %s",
                                config.name,
                                key_number,
                                attempts,
                                error
                            )
                            break

                        # エラーが発生しなければ、成功。LLMResponseを返す。
                        else:
                            if attempts > 1:
                                response = response.model_copy(update={"retries": attempts - 1})

                            self._log.info(
                                "provider=%s key#%d success after %d attempt(s)",
                                config.name,
                                key_number,
                                attempts
                            )
                            return response

        # この周回で全組合せが失敗した。429を観測していればその最短待機秒数を添えて送出し、
        # generate()が「待てば解けるか」を判断する。attemptsはリトライを含む総試行回数。
        raise AllProvidersExhausted(
            f"All providers/keys exhausted after {attempts} attempt(s).",
            attempts=attempts,
            retry_after=min(rate_limit_waits) if rate_limit_waits else None
        ) from last_error
