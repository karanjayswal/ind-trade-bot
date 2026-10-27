"""Typed, rate-limited HTTP client for the INDstocks (INDmoney) trading API.

This is the module every rupee flows through, so it is deliberately paranoid:

* The ``Authorization`` header carries the raw token with **no** ``Bearer `` prefix.
* A real token bucket per rate-limit category paces every call, shared across
  client instances in this process because the limit is per *account*, not per
  object — the engine and a backtest running together must not double the rate.
* ``429`` and ``5xx`` are retried with exponential backoff and jitter, but
  ``POST /order`` is **never** retried: a retried order is a second position.
* Nothing on an order path is swallowed. A timeout on a placement raises
  :class:`OrderError` telling the caller to reconcile against the order book,
  because "we do not know whether that order exists" is information the engine
  needs, not an error to hide.
* Prices are never fabricated. A missing LTP is absent from the result rather
  than ``0.0``; a position marked at zero would read as an instant stop-out.

Request windows (``start_time``/``end_time``) are epoch **milliseconds**; the
``ts`` on every returned candle is epoch **seconds**. The conversion happens at
this boundary and nowhere else.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Sequence

import httpx

from . import clock, token_store
from .errors import (
    ApiError,
    AuthError,
    ConfigError,
    DataError,
    IndbotError,
    OrderError,
    RateLimitError,
)
from .types import Candle

log = logging.getLogger(__name__)

BASE_URL = "https://api.indstocks.com"

#: Where the access token lives when it is not passed to the constructor.
TOKEN_ENV = "INDSTOCKS_ACCESS_TOKEN"

#: Env vars naming a PEM bundle of trusted roots, in the order they are honoured.
#: INDBOT_CA_BUNDLE is ours; the rest are the de-facto standards other tools read,
#: so a machine already configured for a TLS-inspecting proxy needs nothing new.
CA_BUNDLE_ENVS = ("INDBOT_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")

#: Looked for next to the project, for the common case of a bundle dropped in the
#: folder. Checked in order; the first readable one wins.
CA_BUNDLE_FILENAMES = ("cacert.pem", "ca-bundle.pem", "corporate-ca.pem")


def resolve_ca_bundle(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find the CA bundle to verify TLS against, or None for the system store.

    Order: the explicit argument, then CA_BUNDLE_ENVS, then CA_BUNDLE_FILENAMES in
    the package parent and its parent. Returns None when nothing is found, which
    means "use the default trust store" — not "skip verification".

    This exists because a corporate TLS-inspecting proxy presents a self-signed
    root that is not in certifi's store, which surfaces as
    CERTIFICATE_VERIFY_FAILED on the first call. The fix is to trust that root,
    never to pass verify=False.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env in CA_BUNDLE_ENVS:
        raw = os.environ.get(env, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    here = Path(__file__).resolve().parent.parent
    for folder in (here, here.parent):
        for name in CA_BUNDLE_FILENAMES:
            candidates.append(folder / name)

    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    if explicit:
        # An explicit path that does not exist is a configuration error, not a
        # reason to silently fall back to a store that is already known to fail.
        raise ConfigError(
            f"ca_bundle {explicit!r} does not exist or is not readable",
            key="ca_bundle",
        )
    return None

# --------------------------------------------------------------------------- #
# Documented limits, enums and quirks — one place, so a doc change is one edit.
# --------------------------------------------------------------------------- #

#: Requests per second, per category (docs §API Conventions → Rate Limiting).
CATEGORY_LIMITS: dict[str, float] = {
    "order": 10.0,
    "data": 5.0,
    "quote": 5.0,
    "nontrading": 15.0,
}

#: `GET /market/historical/{interval}` takes at most five `scrip-codes`.
MAX_SCRIP_CODES_HISTORICAL = 5
#: `GET /market/historical/expired/{interval}` takes 1..50 `trading-symbols`.
MAX_SYMBOLS_HISTORICAL_EXPIRED = 50
#: The quote endpoints accept up to 1000 instruments in one call.
MAX_SCRIP_CODES_QUOTE = 1000

#: The complete set of interval labels the API accepts.
INTERVALS: frozenset[str] = frozenset(
    {
        "1minute", "2minute", "3minute", "4minute", "5minute", "10minute",
        "15minute", "30minute", "60minute", "120minute", "180minute",
        "240minute", "1day", "1week", "1month",
    }
)

INSTRUMENT_SOURCES: frozenset[str] = frozenset({"equity", "fno", "index", "mcx"})
DERIVATIVE_SEGMENTS: frozenset[str] = frozenset({"NFO", "BFO"})
CHAIN_EXCHANGES: frozenset[str] = frozenset({"NSE", "BSE"})
CHAIN_SEGMENTS: frozenset[str] = frozenset({"INDEX", "EQUITY"})
ORDER_SEGMENTS: frozenset[str] = frozenset({"DERIVATIVE", "EQUITY", "COMMODITY"})

#: `remarks` values reserved for INDstocks' own channel tags. Matching is done on
#: the STORED value — after truncation to 100 chars — and ignores case and
#: surrounding whitespace.
RESERVED_REMARKS: frozenset[str] = frozenset({"tv-terminal"})
MAX_REMARK_CHARS = 100

#: Retry budget for idempotent reads. Four attempts total, not four retries.
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 0.5
RETRY_MAX_DELAY_S = 8.0

#: `GET /order` answers 512 — not 404 — for an order it cannot fetch, including
#: one that does not exist. It is a terminal lookup failure, so it is excluded
#: from the "any 5xx is retryable" rule.
NON_RETRYABLE_SERVER_STATUSES: frozenset[int] = frozenset({512})

#: TOTP token generation: 1 per 60s, 5 wrong codes in a rolling 15 minutes locks
#: the account out for 15 minutes, and 3 lockouts in an hour for a full hour.
TOKEN_GEN_MIN_GAP_S = 60.0
TOTP_FAILURE_WINDOW_S = 900.0
#: We refuse locally on the 5th attempt in the window — the one that would trip
#: the server-side lockout. Four consecutive failures mean a wrong MPIN, a wrong
#: secret or a skewed clock; a fifth will not succeed, it will only lock the
#: account out of token generation for 15 minutes.
TOTP_MAX_LOCAL_FAILURES = 4

#: Documented `error_type` values mapped onto the indbot exception hierarchy.
#: Anything not listed falls back to ApiError (or OrderError on an order path).
ERROR_TYPE_MAP: dict[str, type[IndbotError]] = {
    "InputException": ApiError,                 # 400 malformed request
    "RequestValidationException": ApiError,     # 400 business-rule failure
    "TokenException": AuthError,                # 403/401 token invalid or revoked
    "UserException": AuthError,                 # 403 segment not activated
    "NotFoundException": ApiError,              # 404 bad path
    "MethodNotAllowedException": ApiError,      # 405 wrong verb
    "DataException": DataError,                 # 400 bad scrip/interval/window
    "OrderException": OrderError,               # 400 RMS rejection
    "NetworkException": ApiError,               # 503 upstream blip, retryable
    "GeneralException": ApiError,               # 500 server error, retryable once
    "ServiceUnavailableException": ApiError,    # 503 maintenance, retryable
    "GatewayTimeoutException": ApiError,        # 504 upstream timeout, retryable
    "InternalServerException": OrderError,      # 512 order lookup failed, terminal
}

#: `error_type` values the docs call safe to retry after a backoff.
RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "NetworkException",
        "GeneralException",
        "ServiceUnavailableException",
        "GatewayTimeoutException",
    }
)

_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Any plausible trading timestamp in milliseconds is well above 1e11, while any
#: timestamp in seconds is around 1.7e9. Anything between is a unit mix-up.
_MS_SANITY_FLOOR = 100_000_000_000


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class TokenBucket:
    """A thread-safe token bucket: `rate` permits per second, burst `capacity`.

    The engine thread, the UI's on-demand calls and a backtest worker all share
    one account's quota, so the bucket is locked rather than per-thread.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ConfigError(f"rate limit must be positive, got {rate!r}")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else rate)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, *, max_wait: float = 30.0, label: str = "") -> float:
        """Block until `tokens` permits are free; return the seconds spent waiting.

        Raises RateLimitError rather than waiting forever: an engine tick that
        blocks indefinitely on a bucket is indistinguishable from a hung bot.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._updated)
                self._updated = now
                # rate is validated > 0 in __init__, so this cannot divide by zero.
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                sleep_for = (tokens - self._tokens) / self.rate
            if waited + sleep_for > max_wait:
                raise RateLimitError(
                    f"local rate limiter for {label or 'api'} would block "
                    f"{waited + sleep_for:.1f}s, over the {max_wait:.0f}s ceiling",
                    status=None,
                    endpoint=label,
                )
            # Sleep OUTSIDE the lock, or one waiter would stall every other caller.
            time.sleep(sleep_for)
            waited += sleep_for


_BUCKET_REGISTRY: dict[str, dict[str, TokenBucket]] = {}
_BUCKET_REGISTRY_LOCK = threading.Lock()


def _buckets_for(key: str) -> dict[str, TokenBucket]:
    """Per-account bucket set, created once and shared by every client in-process."""
    with _BUCKET_REGISTRY_LOCK:
        buckets = _BUCKET_REGISTRY.get(key)
        if buckets is None:
            buckets = {cat: TokenBucket(rate) for cat, rate in CATEGORY_LIMITS.items()}
            _BUCKET_REGISTRY[key] = buckets
        return buckets


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _batched(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield `items` in chunks of at most `size` (itertools.batched is 3.12+)."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _dedupe(items: Sequence[str]) -> list[str]:
    """Order-preserving de-duplication; the API counts a repeated code once."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _mask(secret: str) -> str:
    """A loggable fingerprint of a token — never the token itself."""
    if not secret:
        return "<empty>"
    return f"…{secret[-4:]} (len {len(secret)})"


def _build_error(
    exc_cls: type[IndbotError],
    message: str,
    *,
    status: int | None,
    payload: Any,
    endpoint: str,
) -> IndbotError:
    """Instantiate the mapped exception, since the subclasses differ in signature."""
    if issubclass(exc_cls, ApiError):
        return exc_cls(message, status=status, payload=payload, endpoint=endpoint)
    if issubclass(exc_cls, OrderError):
        return exc_cls(message, payload=payload, status=status, endpoint=endpoint)
    return exc_cls(message, status=status, payload=payload, endpoint=endpoint)


def _check_ms(value: int, name: str) -> int:
    """Validate an epoch-MILLISECONDS argument, catching the classic seconds mix-up.

    A seconds value passed here would silently request a window in 1970 and the
    caller would read the empty result as "no trades that day".
    """
    try:
        ms = int(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{name} must be epoch milliseconds, got {value!r}") from exc
    if ms <= 0:
        raise DataError(f"{name} must be a positive epoch-millisecond value, got {ms}")
    if ms < _MS_SANITY_FLOOR:
        raise DataError(
            f"{name}={ms} looks like epoch SECONDS, not milliseconds "
            f"(read as seconds that is {clock.to_ist(ms).isoformat()}). Multiply by 1000."
        )
    return ms


def _check_expiry(expiry: str, *, field: str = "expiry") -> str:
    """Validate a YYYY-MM-DD date string and return it stripped.

    The option chain returns the same opaque 400 for a bad date FORMAT as for an
    unknown underlying, so the format is checked here to give a usable message.
    """
    text = str(expiry or "").strip()
    if not _EXPIRY_RE.match(text):
        raise DataError(
            f"{field} must be formatted YYYY-MM-DD (e.g. 2026-08-18), got {expiry!r}"
        )
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise DataError(f"{field}={text!r} is not a real calendar date") from exc
    return text


def _parse_candle(raw: Any, *, code: str, index: int) -> Candle:
    """Coerce one API candle object into a Candle, or raise DataError.

    A malformed bar is not dropped: a hole in the series silently corrupts every
    indicator computed over it, and the safe response is for the caller to skip
    this instrument on this pass rather than trade off a distorted VWAP.
    """
    if not isinstance(raw, dict):
        raise DataError(f"{code}: candle {index} is {type(raw).__name__}, expected an object")
    try:
        candle: Candle = {
            "ts": int(raw["ts"]),
            "o": float(raw["o"]),
            "h": float(raw["h"]),
            "l": float(raw["l"]),
            "c": float(raw["c"]),
            "v": int(float(raw["v"])),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(
            f"{code}: candle {index} is missing or has a non-numeric ts/o/h/l/c/v field"
        ) from exc
    if min(candle["o"], candle["h"], candle["l"], candle["c"]) < 0 or candle["h"] < candle["l"]:
        raise DataError(
            f"{code}: candle {index} at {candle['ts']} is corrupt "
            f"(o={candle['o']} h={candle['h']} l={candle['l']} c={candle['c']})"
        )
    if candle["c"] == 0:
        log.warning("%s: candle %d at %s closed at 0", code, index, candle["ts"])
    return candle


def _normalise_candles(raw: Any, *, code: str) -> list[Candle]:
    """Turn one instrument's `candles` value into an ordered list.

    `candles` is null for a zero-width window, so null becomes []. The series is
    sorted and de-duplicated by `ts` because every indicator downstream assumes
    strictly ascending bars.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DataError(f"{code}: 'candles' is {type(raw).__name__}, expected a list or null")
    parsed = [_parse_candle(item, code=code, index=i) for i, item in enumerate(raw)]
    if not parsed:
        return []
    ordered = sorted(parsed, key=lambda c: c["ts"])
    if any(a["ts"] >= b["ts"] for a, b in zip(parsed, parsed[1:])):
        log.warning("%s: candles arrived out of order or with duplicate ts; re-sorted", code)
    deduped: list[Candle] = []
    for candle in ordered:
        if deduped and deduped[-1]["ts"] == candle["ts"]:
            deduped[-1] = candle  # keep the last value seen for a repeated ts
            continue
        deduped.append(candle)
    if len(deduped) != len(ordered):
        log.warning("%s: dropped %d duplicate candle timestamps", code, len(ordered) - len(deduped))
    return deduped


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


class INDstocksClient:
    """Synchronous REST client for https://api.indstocks.com.

    Every method returns the useful part of the response — the `data` block, a
    list, or a normalised mapping — and raises an indbot error on anything else.
    Use it as a context manager, or call :meth:`close` when done.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 15.0,
        share_rate_limiter: bool = True,
        ca_bundle: str | os.PathLike[str] | None = None,
    ) -> None:
        """Build a client; the token defaults to $INDSTOCKS_ACCESS_TOKEN.

        `share_rate_limiter` keeps the token buckets process-wide (keyed by
        account) because the limits are enforced per account, not per object —
        two clients on one token must still add up to 5 data calls a second.

        `ca_bundle` is a PEM file of trusted roots, for networks behind a TLS
        inspecting proxy whose root CA is not in the system store. Defaults to
        `resolve_ca_bundle()`. Verification is never disabled: on a network that
        rewrites TLS, turning verification off would hand your access token and
        every order to whatever is doing the rewriting.
        """
        if token is not None:
            raw, self.token_source = token_store.clean(token), "argument"
        else:
            # $INDSTOCKS_ACCESS_TOKEN first, then the file the dashboard writes,
            # so pasting a token into the UI works without restarting the process.
            raw, self.token_source = token_store.load()
        if not raw:
            raise AuthError(
                f"no INDstocks access token: set ${TOKEN_ENV}, paste one into the "
                f"dashboard's Settings tab, or pass token=. Tokens expire at 07:00 "
                f"IST, so a token from yesterday will not work."
            )
        self.token = raw
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

        key = (
            hashlib.sha256(f"{self.base_url}|{self.token}".encode()).hexdigest()[:16]
            if share_rate_limiter
            else f"private-{id(self):x}"
        )
        self._buckets = _buckets_for(key)
        self.ca_bundle = resolve_ca_bundle(ca_bundle)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            # str path = verify against that PEM; True = the system/certifi store.
            verify=str(self.ca_bundle) if self.ca_bundle else True,
            headers={
                # No "Bearer " prefix — the API takes the raw token.
                "Authorization": self.token,
                "Accept": "application/json",
                "User-Agent": "indbot/1.0",
            },
        )
        if self.ca_bundle:
            log.info("TLS verified against %s", self.ca_bundle)
        log.debug("INDstocksClient ready for %s with token %s", self.base_url, _mask(self.token))

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "INDstocksClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- transport ---------------------------------------------------------- #

    def _request(
        self,
        method: str,
        path: str,
        *,
        category: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        retry: bool = True,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Issue one rate-limited, optionally retried request and return the response.

        `retry=False` is the order-placement path: a retried POST /order is a
        second live position, so a timeout there is reported, never resent.
        """
        if category not in CATEGORY_LIMITS:
            raise ConfigError(f"unknown rate-limit category {category!r}")
        bucket = self._buckets[category]
        attempt = 0
        last_exc: Exception | None = None

        while attempt < MAX_ATTEMPTS:
            attempt += 1
            bucket.acquire(label=f"{category}:{path}")
            started = time.perf_counter()
            try:
                response = self._client.request(
                    method, path, params=params, json=json_body, headers=headers
                )
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                log.debug("%s %s timed out after %.0f ms (attempt %d)", method, path, elapsed_ms, attempt)
                last_exc = exc
                if not retry or attempt >= MAX_ATTEMPTS:
                    raise ApiError(
                        f"{method} {path} timed out after {self.timeout:.0f}s",
                        status=None,
                        endpoint=path,
                    ) from exc
                self._sleep_backoff(attempt, None, path)
                continue
            except httpx.HTTPError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                log.debug("%s %s transport error after %.0f ms: %s", method, path, elapsed_ms, exc)
                last_exc = exc
                if not retry or attempt >= MAX_ATTEMPTS:
                    raise ApiError(
                        f"{method} {path} failed to reach the API: {exc}",
                        status=None,
                        endpoint=path,
                    ) from exc
                self._sleep_backoff(attempt, None, path)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            log.debug(
                "%s %s -> %d in %.0f ms (attempt %d/%d)",
                method, path, response.status_code, elapsed_ms, attempt, MAX_ATTEMPTS,
            )
            if retry and attempt < MAX_ATTEMPTS and self._should_retry(response):
                self._sleep_backoff(attempt, response, path)
                continue
            return response

        # Unreachable in practice: every branch above either returns or raises.
        raise ApiError(f"{method} {path} exhausted {MAX_ATTEMPTS} attempts", endpoint=path) from last_exc

    @staticmethod
    def _should_retry(response: httpx.Response) -> bool:
        """True for 429 and transient 5xx; 512 is a terminal order lookup failure."""
        status = response.status_code
        if status == 429:
            return True
        if 500 <= status < 600 and status not in NON_RETRYABLE_SERVER_STATUSES:
            return True
        return False

    def _sleep_backoff(self, attempt: int, response: httpx.Response | None, path: str) -> None:
        """Exponential backoff with jitter, honouring Retry-After when present."""
        delay = min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
        if response is not None:
            header = response.headers.get("Retry-After", "")
            try:
                delay = max(delay, float(header)) if header else delay
            except ValueError:
                pass  # a date-formatted Retry-After; the exponential delay stands
        # Jitter spreads retries so concurrent callers do not re-collide.
        delay *= 0.5 + random.random() * 0.5
        log.debug("retrying %s in %.2fs (attempt %d)", path, delay, attempt)
        time.sleep(delay)

    # -- response parsing --------------------------------------------------- #

    @staticmethod
    def _json(response: httpx.Response, endpoint: str) -> Any:
        """Parse a JSON body, or raise ApiError carrying the first 300 chars of text."""
        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text[:300]
            raise ApiError(
                f"{endpoint} returned a non-JSON body: {snippet!r}",
                status=response.status_code,
                endpoint=endpoint,
            ) from exc

    def _raise_for_error(
        self, response: httpx.Response, endpoint: str, *, order_path: bool = False
    ) -> None:
        """Map a non-2xx response onto the right indbot exception.

        The API uses four different error shapes, so the HTTP status is checked
        first and then whichever of error_type / debug_info / message / error is
        present is read for the human-readable reason.
        """
        if response.is_success:
            return
        status = response.status_code
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text[:500]

        error_type = ""
        detail = ""
        if isinstance(payload, dict):
            error_type = str(payload.get("error_type") or "")
            parts = [
                str(payload.get(key))
                for key in ("debug_info", "message", "error")
                if payload.get(key)
            ]
            # debug_info leads: on the option chain and the contracts endpoints
            # `message` is the generic category and debug_info holds the cause.
            detail = " | ".join(dict.fromkeys(parts))
        if not detail:
            detail = str(payload)[:300] or response.reason_phrase

        if status == 429:
            raise RateLimitError(
                f"{endpoint} rate limited (429): {detail}",
                status=status,
                payload=payload,
                endpoint=endpoint,
            )
        if status in (401, 403):
            raise AuthError(
                f"{endpoint} rejected the access token ({status}): {detail}. "
                "Generate a fresh token — another process calling /generate/token "
                "invalidates the one you are holding.",
                status=status,
                payload=payload,
                endpoint=endpoint,
            )

        exc_cls = ERROR_TYPE_MAP.get(error_type) or (OrderError if order_path else ApiError)
        if order_path and not issubclass(exc_cls, (AuthError, OrderError)):
            # Anything that fails on an order path is an OrderError whatever the
            # API called it. The engine routes OrderError to a loud alert and a
            # reconcile; a failed order downgraded to a generic ApiError could be
            # logged and forgotten while a real position sits open.
            exc_cls = OrderError
        prefix = f"{error_type}: " if error_type else ""
        raise _build_error(
            exc_cls,
            f"{endpoint} failed ({status}): {prefix}{detail}",
            status=status,
            payload=payload,
            endpoint=endpoint,
        )

    def _envelope(
        self, response: httpx.Response, endpoint: str, *, order_path: bool = False
    ) -> Any:
        """Validate the standard {"status": "success", "data": …} envelope, return `data`."""
        self._raise_for_error(response, endpoint, order_path=order_path)
        payload = self._json(response, endpoint)
        if not isinstance(payload, dict):
            raise ApiError(
                f"{endpoint} returned {type(payload).__name__}, expected an object",
                status=response.status_code,
                endpoint=endpoint,
            )
        if "data" not in payload:
            # A 200 with no data block: report the body rather than pretending.
            detail = payload.get("message") or payload.get("debug_info") or payload
            exc_cls = OrderError if order_path else ApiError
            raise _build_error(
                exc_cls,
                f"{endpoint} returned 200 with no 'data' block: {detail}",
                status=response.status_code,
                payload=payload,
                endpoint=endpoint,
            )
        return payload["data"]

    def _success_envelope(self, response: httpx.Response, endpoint: str) -> dict[str, Any]:
        """Validate the historical endpoints' {"success": bool, "data": {…}} envelope.

        These two endpoints use `success`, not `status`, which is the single most
        common way to misread a historical response as empty.
        """
        self._raise_for_error(response, endpoint)
        payload = self._json(response, endpoint)
        if not isinstance(payload, dict):
            raise DataError(f"{endpoint} returned {type(payload).__name__}, expected an object")
        if payload.get("success") is not True:
            detail = payload.get("debug_info") or payload.get("message") or payload
            raise DataError(f"{endpoint} returned success=false: {detail}")
        data = payload.get("data")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise DataError(f"{endpoint}: 'data' is {type(data).__name__}, expected an object")
        return data

    # ------------------------------------------------------------------ #
    # User / funds                                            nontrading
    # ------------------------------------------------------------------ #

    def profile(self) -> dict[str, Any]:
        """Account profile. The cheapest way to prove the access token is alive."""
        response = self._request("GET", "/user/profile", category="nontrading")
        data = self._envelope(response, "GET /user/profile")
        if not isinstance(data, dict):
            raise ApiError("GET /user/profile returned a non-object data block", endpoint="/user/profile")
        return data

    def funds(self) -> dict[str, Any]:
        """Funds snapshot, including `detailed_avl_balance.option_buy` the gate sizes against."""
        response = self._request("GET", "/funds", category="nontrading")
        data = self._envelope(response, "GET /funds")
        if not isinstance(data, dict):
            raise ApiError("GET /funds returned a non-object data block", endpoint="/funds")
        return data

    # ------------------------------------------------------------------ #
    # Instruments and contracts                                      data
    # ------------------------------------------------------------------ #

    def instruments_csv(self, source: str) -> str:
        """Download an instrument master as raw CSV text (equity|fno|index|mcx).

        Returns text, not JSON — `source=index` is a three-column file whose
        second column holds the index NAME despite its `SEGMENT` header.
        """
        key = str(source).strip().lower()
        if key not in INSTRUMENT_SOURCES:
            raise ConfigError(
                f"instrument source must be one of {sorted(INSTRUMENT_SOURCES)}, got {source!r}"
            )
        response = self._request(
            "GET", "/market/instruments", category="data", params={"source": key}
        )
        self._raise_for_error(response, f"GET /market/instruments?source={key}")
        text = response.text
        if not text.strip() or "," not in text.splitlines()[0]:
            raise DataError(
                f"instrument master for source={key} does not look like CSV "
                f"(got {len(text)} bytes starting {text[:60]!r})"
            )
        log.debug("instruments source=%s: %d bytes, %d lines", key, len(text), text.count("\n"))
        return text

    def expiries(self, underlying: str, *, segment: str = "DERIVATIVE") -> list[str]:
        """Upcoming expiry dates for an underlying, ascending, as YYYY-MM-DD strings."""
        response = self._request(
            "GET",
            "/market/instruments/expiries",
            category="data",
            params={"underlying": underlying, "segment": segment},
        )
        data = self._envelope(response, "GET /market/instruments/expiries")
        return self._as_date_list(data, "GET /market/instruments/expiries")

    def expired_expiries(
        self,
        underlying: str,
        expiry_from: str,
        expiry_to: str,
        *,
        segment: str = "DERIVATIVE",
    ) -> list[str]:
        """Past expiry dates in a window, DESCENDING (newest first) as the API returns them.

        The ordering is the opposite of :meth:`expiries`; it is left as-is so the
        caller sees exactly what the API said. The window may not span a year.
        """
        start = _check_expiry(expiry_from, field="expiry_from")
        end = _check_expiry(expiry_to, field="expiry_to")
        if start > end:
            raise DataError(f"expiry_from {start} is after expiry_to {end}")
        span_days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        if span_days > 366:
            raise DataError(
                f"expired expiry window {start}..{end} spans {span_days} days; "
                "the API caps this endpoint at 1 year — page it in yearly windows"
            )
        response = self._request(
            "GET",
            "/market/instruments/expired/expiries",
            category="data",
            params={
                "underlying": underlying,
                "segment": segment,
                "expiry_from": start,
                "expiry_to": end,
            },
        )
        data = self._envelope(response, "GET /market/instruments/expired/expiries")
        return self._as_date_list(data, "GET /market/instruments/expired/expiries")

    @staticmethod
    def _as_date_list(data: Any, endpoint: str) -> list[str]:
        """Coerce a `data` block that should be a flat array of YYYY-MM-DD dates."""
        if not isinstance(data, list):
            raise DataError(f"{endpoint}: 'data' is {type(data).__name__}, expected a list of dates")
        return [str(item).strip() for item in data if str(item).strip()]

    def expired_contracts(
        self,
        underlying: str,
        expiry: str,
        instrument_type: str = "OPTIDX",
        *,
        segment: str = "DERIVATIVE",
    ) -> list[dict[str, Any]]:
        """Full contract chain for one past expiry — a flat array, no pagination.

        A NIFTY expiry is 460–480 rows (a CE and a PE per strike, plus the future
        on monthly expiries). Rows carry `trading_symbol` and `lot_size` but no
        `security_id`: expired contracts are addressed by symbol.
        """
        date = _check_expiry(expiry)
        params: dict[str, Any] = {
            "underlying": underlying,
            "segment": segment,
            "expiry": date,
        }
        if instrument_type:
            params["instrument_type"] = instrument_type
        response = self._request(
            "GET", "/market/instruments/expired/contracts", category="data", params=params
        )
        data = self._envelope(response, "GET /market/instruments/expired/contracts")
        if not isinstance(data, list):
            raise DataError(
                "GET /market/instruments/expired/contracts: 'data' is "
                f"{type(data).__name__}, expected a flat list of contracts"
            )
        rows = [row for row in data if isinstance(row, dict)]
        if len(rows) != len(data):
            log.warning(
                "expired_contracts(%s, %s): dropped %d non-object rows",
                underlying, date, len(data) - len(rows),
            )
        log.debug("expired_contracts(%s, %s, %s): %d rows", underlying, date, instrument_type, len(rows))
        return rows

    # ------------------------------------------------------------------ #
    # Market data                                              data/quote
    # ------------------------------------------------------------------ #

    def historical(
        self, interval: str, scrip_codes: list[str], start_ms: int, end_ms: int
    ) -> dict[str, list[Candle]]:
        """OHLCV candles for live instruments, batching five `scrip-codes` per call.

        Returns one entry per requested code, `[]` where the API returned nothing
        — a code with no data is absent from `data` rather than empty, and
        `candles` itself can be null, so both become an empty list here.
        """
        return self._historical_impl(
            interval=interval,
            keys=scrip_codes,
            start_ms=start_ms,
            end_ms=end_ms,
            path=f"/market/historical/{interval}",
            param_name="scrip-codes",
            batch_size=MAX_SCRIP_CODES_HISTORICAL,
            extra_params=None,
            what="scrip code",
        )

    def historical_expired(
        self,
        interval: str,
        trading_symbols: list[str],
        start_ms: int,
        end_ms: int,
        segment: str = "NFO",
    ) -> dict[str, list[Candle]]:
        """OHLCV candles for expired F&O contracts, batching fifty symbols per call.

        BSE contracts need `segment="BFO"`: sent without it they are looked up
        against NSE, match nothing, and come back absent rather than as an error.
        """
        seg = str(segment or "NFO").strip().upper()
        if seg not in DERIVATIVE_SEGMENTS:
            raise ConfigError(f"segment must be NFO or BFO, got {segment!r}")
        return self._historical_impl(
            interval=interval,
            keys=trading_symbols,
            start_ms=start_ms,
            end_ms=end_ms,
            path=f"/market/historical/expired/{interval}",
            param_name="trading-symbols",
            batch_size=MAX_SYMBOLS_HISTORICAL_EXPIRED,
            extra_params={"segment": seg},
            what="trading symbol",
        )

    def _historical_impl(
        self,
        *,
        interval: str,
        keys: Sequence[str],
        start_ms: int,
        end_ms: int,
        path: str,
        param_name: str,
        batch_size: int,
        extra_params: dict[str, Any] | None,
        what: str,
    ) -> dict[str, list[Candle]]:
        """Shared batching and normalisation for the two historical endpoints."""
        if interval not in INTERVALS:
            raise ConfigError(f"interval must be one of {sorted(INTERVALS)}, got {interval!r}")
        start = _check_ms(start_ms, "start_time")
        end = _check_ms(end_ms, "end_time")
        if start > end:
            raise DataError(
                f"start_time {start} is after end_time {end} "
                f"({clock.to_ist(start / 1000).isoformat()} .. {clock.to_ist(end / 1000).isoformat()})"
            )
        max_days = clock.INTERVAL_MAX_DAYS.get(interval)
        span_days = (end - start) / 86_400_000.0
        if max_days is not None and span_days > max_days:
            # A wider window returns 200 with the MAXIMUM window's data, not an
            # error, so an unpaged caller silently receives a truncated series.
            log.warning(
                "%s: requested %.1f days but %s caps a single call at %d days — "
                "the response will be truncated; page by date range",
                path, span_days, interval, max_days,
            )

        unique = _dedupe(keys)
        out: dict[str, list[Candle]] = {key: [] for key in unique}
        if not unique:
            return out

        for batch in _batched(unique, batch_size):
            params: dict[str, Any] = {
                param_name: ",".join(batch),
                "start_time": start,
                "end_time": end,
            }
            if extra_params:
                params.update(extra_params)
            response = self._request("GET", path, category="data", params=params)
            data = self._success_envelope(response, f"GET {path}")
            requested = set(batch)
            for key, block in data.items():
                if key not in requested:
                    # Keys we did not ask for are dropped rather than returned.
                    log.debug("%s: ignoring unrequested key %r", path, key)
                    continue
                candles = block.get("candles") if isinstance(block, dict) else None
                out[key] = _normalise_candles(candles, code=key)
            missing = [key for key in batch if key not in data]
            if missing:
                log.debug(
                    "%s: no data for %d %s(s) in this window: %s",
                    path, len(missing), what, ", ".join(missing[:5]),
                )
        total = sum(len(v) for v in out.values())
        log.debug("%s: %d %s(s) -> %d candles", path, len(unique), what, total)
        return out

    def ltp(self, scrip_codes: list[str]) -> dict[str, float]:
        """Last traded price per `SEGMENT_TOKEN` code.

        A code the API did not price is ABSENT from the result, never 0.0: a
        position marked at zero would read as an instant stop-out and be sold.
        """
        out: dict[str, float] = {}
        for code, block in self._quote_impl("/market/quotes/ltp", scrip_codes).items():
            price = block.get("live_price") if isinstance(block, dict) else None
            if price is None:
                log.warning("ltp: %s came back without a live_price; omitting it", code)
                continue
            try:
                out[code] = float(price)
            except (TypeError, ValueError):
                log.warning("ltp: %s has a non-numeric live_price %r; omitting it", code, price)
        return out

    def quotes(self, scrip_codes: list[str]) -> dict[str, dict[str, Any]]:
        """Full market quote per code: OHLC, change, volume, circuit limits, depth."""
        return self._quote_impl("/market/quotes/full", scrip_codes)

    def depth(self, scrip_codes: list[str]) -> dict[str, dict[str, Any]]:
        """Five-level market depth per code (not in CONTRACTS.md; added for completeness)."""
        return self._quote_impl("/market/quotes/mkt", scrip_codes)

    def _quote_impl(self, path: str, scrip_codes: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Batch the quote endpoints at 1000 codes and merge their `data` blocks.

        One unknown code fails the whole request with 400, so a batch that fails
        is not partially recoverable — the error names the batch.
        """
        unique = _dedupe(scrip_codes)
        out: dict[str, dict[str, Any]] = {}
        if not unique:
            return out
        for batch in _batched(unique, MAX_SCRIP_CODES_QUOTE):
            response = self._request(
                "GET", path, category="quote", params={"scrip-codes": ",".join(batch)}
            )
            data = self._envelope(response, f"GET {path}")
            if not isinstance(data, dict):
                raise DataError(f"GET {path}: 'data' is {type(data).__name__}, expected an object")
            for code, block in data.items():
                if isinstance(block, dict):
                    out[code] = block
            missing = [code for code in batch if code not in out]
            if missing:
                log.warning("GET %s: no quote returned for %s", path, ", ".join(missing[:5]))
        return out

    def option_chain(
        self,
        exchange: str,
        segment: str,
        underlying_scrip: str,
        expiry: str,
        strike_count: int = 10,
    ) -> dict[str, Any]:
        """The live chain for one underlying and expiry: `{underlying_ltp, expiry, strikes}`.

        `strikes` is an object keyed by strike price, not an array, and carries a
        `ce` and a `pe` leg with top-of-book, OI, IV and greeks. `underlying_scrip`
        is the SECURITY_ID of the UNDERLYING (from source=index or source=equity),
        never a contract's id from source=fno.
        """
        exch = str(exchange or "").strip().upper()
        seg = str(segment or "").strip().upper()
        if exch not in CHAIN_EXCHANGES:
            raise ConfigError(f"option chain exchange must be NSE or BSE, got {exchange!r}")
        if seg not in CHAIN_SEGMENTS:
            raise ConfigError(f"option chain segment must be INDEX or EQUITY, got {segment!r}")
        scrip = str(underlying_scrip or "").strip()
        if not scrip:
            raise DataError("option_chain needs the underlying's SECURITY_ID")
        # Validated here because a malformed date returns the SAME opaque 400 as
        # an unknown underlying, and "2026-08-18, not 20260818" is a better message.
        date = _check_expiry(expiry)
        count = int(strike_count)
        if count < 1:
            raise ConfigError(f"strike_count must be at least 1, got {strike_count!r}")

        endpoint = "GET /market/option-chain"
        response = self._request(
            "GET",
            "/market/option-chain",
            category="data",
            params={
                "exchange": exch,
                "segment": seg,
                "underlying-scrip": scrip,
                "expiry": date,
                "strike_count": count,
            },
        )
        # This endpoint does NOT use the standard envelope on failure: check the
        # HTTP status first, then read debug_info / message / error.
        self._raise_for_error(response, endpoint)
        payload = self._json(response, endpoint)
        if not isinstance(payload, dict):
            raise DataError(f"{endpoint} returned {type(payload).__name__}, expected an object")
        data = payload.get("data")
        if not isinstance(data, dict):
            detail = payload.get("debug_info") or payload.get("message") or payload.get("error")
            raise DataError(
                f"{endpoint} for {exch}/{seg}/{scrip} expiry {date} returned no chain: {detail}"
            )
        strikes = data.get("strikes")
        if not isinstance(strikes, dict) or not strikes:
            raise DataError(
                f"{endpoint} for {exch}/{seg}/{scrip} expiry {date} returned an empty strike ladder "
                "— check that the expiry is one the underlying actually lists"
            )
        log.debug(
            "%s %s/%s/%s expiry=%s: %d strikes, underlying_ltp=%s",
            endpoint, exch, seg, scrip, date, len(strikes), data.get("underlying_ltp"),
        )
        return data

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def place_order(self, **kw: Any) -> dict[str, Any]:
        """POST /order — place one live order. NEVER retried.

        A retried placement is a second position: if the call times out we do not
        know whether the exchange saw it, so this raises and tells the caller to
        reconcile against the order book instead of resending. Returns the `data`
        block, `{"order_id": …, "order_status": …}`.
        """
        body = self._validate_order_body(kw)
        log.info(
            "ORDER place: %s %s qty=%s %s @ %s remarks=%r",
            body.get("txn_type"), body.get("security_id"), body.get("qty"),
            body.get("order_type"), body.get("limit_price"), body.get("remarks", ""),
        )
        started = time.perf_counter()
        try:
            # retry=False is load-bearing. Exponential backoff on a POST /order
            # would buy a second lot at a worse price with no way to tell.
            response = self._request("POST", "/order", category="order", json_body=body, retry=False)
        except RateLimitError as exc:
            if exc.status is None:
                # The local token bucket refused before anything left this process,
                # so the order definitively does not exist. Say so plainly — the
                # ambiguous wording below would send the engine reconciling for
                # an order that was never sent.
                log.error("ORDER place REFUSED locally by the rate limiter: %s", exc)
                raise OrderError(
                    f"POST /order was not sent: {exc}. No order was placed.",
                    security_id=body.get("security_id"),
                    qty=body.get("qty"),
                    txn_type=body.get("txn_type"),
                ) from exc
            raise
        except IndbotError as exc:
            log.error("ORDER place FAILED to complete: %s", exc)
            raise OrderError(
                f"POST /order did not complete ({exc}). The order may or may not have "
                "reached the exchange — reconcile with order_book() before retrying.",
                payload=getattr(exc, "payload", None),
                security_id=body.get("security_id"),
                qty=body.get("qty"),
                txn_type=body.get("txn_type"),
            ) from exc

        if response.status_code >= 500:
            # A 5xx is the ambiguous case: the exchange may well have the order
            # and only the reply was lost. Never retried, always escalated.
            try:
                payload: Any = response.json()
            except ValueError:
                payload = response.text[:500]
            log.error("ORDER place returned %d: %s", response.status_code, payload)
            raise OrderError(
                f"POST /order returned {response.status_code}: {payload}. The order may or "
                "may not have reached the exchange — reconcile with order_book() before "
                "placing another.",
                payload=payload,
                security_id=body.get("security_id"),
                qty=body.get("qty"),
                txn_type=body.get("txn_type"),
            )
        data = self._envelope(response, "POST /order", order_path=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(data, dict):
            raise OrderError(f"POST /order returned a non-object data block: {data!r}", payload=data)
        order_id = data.get("order_id")
        if not order_id:
            raise OrderError(f"POST /order returned no order_id: {data!r}", payload=data)
        log.info(
            "ORDER placed: id=%s status=%s in %.0f ms",
            order_id, data.get("order_status"), elapsed_ms,
        )
        return data

    def _validate_order_body(self, kw: dict[str, Any]) -> dict[str, Any]:
        """Check an order body before it costs money, and normalise `remarks`."""
        body = dict(kw)
        required = ("txn_type", "exchange", "segment", "product", "order_type", "validity", "security_id", "qty")
        missing = [field for field in required if body.get(field) in (None, "")]
        if missing:
            raise OrderError(f"order body is missing required field(s): {', '.join(missing)}", payload=body)

        txn = str(body["txn_type"]).upper()
        if txn not in ("BUY", "SELL"):
            raise OrderError(f"txn_type must be BUY or SELL, got {body['txn_type']!r}", payload=body)
        body["txn_type"] = txn

        segment = str(body["segment"]).upper()
        if segment not in ORDER_SEGMENTS:
            raise OrderError(f"segment must be one of {sorted(ORDER_SEGMENTS)}, got {body['segment']!r}", payload=body)
        body["segment"] = segment

        if str(body["validity"]).upper() != "DAY":
            raise OrderError(f"validity must be DAY (IOC is not accepted), got {body['validity']!r}", payload=body)
        body["validity"] = "DAY"

        try:
            qty = int(body["qty"])
        except (TypeError, ValueError) as exc:
            raise OrderError(f"qty must be an integer number of UNITS, got {body['qty']!r}", payload=body) from exc
        if qty <= 0:
            raise OrderError(f"qty must be greater than zero, got {qty}", payload=body)
        body["qty"] = qty

        order_type = str(body["order_type"]).upper()
        if order_type not in ("LIMIT", "MARKET"):
            raise OrderError(f"order_type must be LIMIT or MARKET, got {body['order_type']!r}", payload=body)
        body["order_type"] = order_type
        if order_type == "LIMIT":
            try:
                price = float(body.get("limit_price"))
            except (TypeError, ValueError) as exc:
                raise OrderError(
                    f"a LIMIT order needs a numeric limit_price, got {body.get('limit_price')!r}", payload=body
                ) from exc
            if price <= 0:
                raise OrderError(f"limit_price must be greater than zero, got {price}", payload=body)
            body["limit_price"] = price
        elif "limit_price" in body:
            body.pop("limit_price")  # a MARKET order must not carry one

        body["security_id"] = str(body["security_id"])
        body.setdefault("is_amo", False)

        remark = body.get("remarks")
        if remark is not None:
            text = str(remark)
            if len(text) > MAX_REMARK_CHARS:
                # The API truncates silently; truncating here keeps what we send
                # identical to what gets stored and matched against reserved tags.
                log.warning("remarks truncated from %d to %d chars", len(text), MAX_REMARK_CHARS)
                text = text[:MAX_REMARK_CHARS]
            if text.strip().lower() in RESERVED_REMARKS:
                raise OrderError(
                    f"remarks {text!r} is reserved by INDstocks and would be rejected; "
                    "use your own strategy/signal tag",
                    payload=body,
                )
            body["remarks"] = text
        return body

    def modify_order(self, **kw: Any) -> dict[str, Any]:
        """POST /order/modify — change a pending order's qty and limit price.

        Retryable: modify sets absolute values, so applying it twice leaves the
        same state. Only 25 modifications are allowed per order.
        """
        body = dict(kw)
        for field in ("order_id", "segment", "qty", "limit_price"):
            if body.get(field) in (None, ""):
                raise OrderError(f"modify_order is missing required field {field!r}", payload=body)
        body["order_id"] = str(body["order_id"])
        body["segment"] = str(body["segment"]).upper()
        try:
            body["qty"] = int(body["qty"])
            body["limit_price"] = float(body["limit_price"])
        except (TypeError, ValueError) as exc:
            raise OrderError(f"modify_order needs a numeric qty and limit_price: {body!r}", payload=body) from exc
        if body["qty"] <= 0 or body["limit_price"] <= 0:
            raise OrderError(f"modify_order qty and limit_price must be positive: {body!r}", payload=body)
        body.pop("remarks", None)  # the endpoint rejects it; the order keeps its original tag

        log.info("ORDER modify: id=%s qty=%s price=%s", body["order_id"], body["qty"], body["limit_price"])
        response = self._request("POST", "/order/modify", category="order", json_body=body)
        data = self._envelope(response, "POST /order/modify", order_path=True)
        if not isinstance(data, dict):
            raise OrderError(f"POST /order/modify returned a non-object data block: {data!r}", payload=data)
        log.info("ORDER modified: id=%s status=%s", data.get("order_id"), data.get("order_status"))
        return data

    def cancel_order(self, order_id: str, *, segment: str = "DERIVATIVE") -> dict[str, Any]:
        """POST /order/cancel — cancel a pending order. Idempotent enough to retry."""
        oid = str(order_id or "").strip()
        if not oid:
            raise OrderError("cancel_order needs an order_id")
        seg = str(segment).upper()
        if seg not in ORDER_SEGMENTS:
            raise OrderError(f"segment must be one of {sorted(ORDER_SEGMENTS)}, got {segment!r}")
        log.info("ORDER cancel: id=%s segment=%s", oid, seg)
        response = self._request(
            "POST", "/order/cancel", category="order", json_body={"order_id": oid, "segment": seg}
        )
        data = self._envelope(response, "POST /order/cancel", order_path=True)
        if not isinstance(data, dict):
            raise OrderError(f"POST /order/cancel returned a non-object data block: {data!r}", payload=data)
        log.info("ORDER cancelled: id=%s status=%s", data.get("order_id"), data.get("order_status"))
        return data

    def order_book(self) -> list[dict[str, Any]]:
        """Today's full order book: every order placed, in any state."""
        response = self._request("GET", "/order-book", category="nontrading")
        data = self._envelope(response, "GET /order-book")
        if not isinstance(data, list):
            raise ApiError(
                f"GET /order-book: 'data' is {type(data).__name__}, expected a list",
                endpoint="/order-book",
            )
        return [row for row in data if isinstance(row, dict)]

    def order_details(self, order_id: str, *, segment: str = "DERIVATIVE") -> dict[str, Any]:
        """GET /order — one order's current state. Takes a JSON BODY, not a query string.

        A 512 here means the lookup failed, including for an order_id that does
        not exist; it is terminal and deliberately not retried.
        """
        oid = str(order_id or "").strip()
        if not oid:
            raise OrderError("order_details needs an order_id")
        seg = str(segment).upper()
        response = self._request(
            "GET", "/order", category="nontrading", json_body={"order_id": oid, "segment": seg}
        )
        if response.status_code == 512:
            raise OrderError(
                f"GET /order could not fetch {oid} (512). The API answers 512, not 404, "
                "for an order it cannot look up — including one that does not exist. "
                "Check order_book() rather than retrying.",
                payload=response.text[:500],
                order_id=oid,
            )
        data = self._envelope(response, "GET /order", order_path=True)
        if not isinstance(data, dict):
            raise OrderError(f"GET /order returned a non-object data block: {data!r}", payload=data)
        return data

    def order_trades(self, order_id: str, *, segment: str = "DERIVATIVE") -> list[dict[str, Any]]:
        """GET /order/trades — the individual fills of one order, for an average price.

        Not in CONTRACTS.md; added because a partially filled order's true average
        price can only be computed from its fills.
        """
        oid = str(order_id or "").strip()
        if not oid:
            raise OrderError("order_trades needs an order_id")
        response = self._request(
            "GET",
            "/order/trades",
            category="nontrading",
            json_body={"order_id": oid, "segment": str(segment).upper()},
        )
        data = self._envelope(response, "GET /order/trades", order_path=True)
        if not isinstance(data, list):
            raise OrderError(f"GET /order/trades: 'data' is {type(data).__name__}, expected a list", payload=data)
        return [row for row in data if isinstance(row, dict)]

    def trade_book(self, *, segment: str = "DERIVATIVE") -> list[dict[str, Any]]:
        """Today's executed fills for one segment, with the `remarks` tag on each."""
        seg = str(segment).upper()
        if seg not in ORDER_SEGMENTS:
            raise ConfigError(f"segment must be one of {sorted(ORDER_SEGMENTS)}, got {segment!r}")
        response = self._request("GET", "/trade-book", category="nontrading", params={"segment": seg})
        data = self._envelope(response, "GET /trade-book")
        if not isinstance(data, list):
            raise ApiError(
                f"GET /trade-book: 'data' is {type(data).__name__}, expected a list",
                endpoint="/trade-book",
            )
        return [row for row in data if isinstance(row, dict)]

    def positions(self, segment: str | None = None, product: str | None = None) -> list[dict[str, Any]]:
        """Open positions, optionally filtered. Both filter values are lower-case.

        `product` is only accepted alongside `segment`; sent on its own the API
        returns 400, so that combination is rejected here with a clearer message.
        """
        params: dict[str, Any] = {}
        if product and not segment:
            raise ConfigError("positions(product=…) also needs segment= — the API rejects product alone")
        if segment:
            params["segment"] = str(segment).lower()
        if product:
            params["product"] = str(product).lower()
        response = self._request("GET", "/portfolio/positions", category="nontrading", params=params or None)
        data = self._envelope(response, "GET /portfolio/positions")
        if not isinstance(data, list):
            raise ApiError(
                f"GET /portfolio/positions: 'data' is {type(data).__name__}, expected a list",
                endpoint="/portfolio/positions",
            )
        return [row for row in data if isinstance(row, dict)]

    def margin(self, **kw: Any) -> dict[str, Any]:
        """GET /margin — required margin and charges for an order, before placing it.

        Takes a JSON body with camelCase keys (`securityID`, `txnType`) and string
        `quantity`/`price`, unlike the snake_case order endpoints.
        """
        body = dict(kw)
        required = ("segment", "exchange", "securityID", "txnType", "quantity", "price", "product")
        missing = [field for field in required if field not in body]
        if missing:
            raise ConfigError(f"margin() is missing required field(s): {', '.join(missing)}")
        for field in ("quantity", "price"):
            body[field] = "" if body[field] is None else str(body[field])
        body["securityID"] = str(body["securityID"])
        response = self._request("GET", "/margin", category="nontrading", json_body=body)
        data = self._envelope(response, "GET /margin")
        if not isinstance(data, dict):
            raise ApiError("GET /margin returned a non-object data block", endpoint="/margin")
        return data


# --------------------------------------------------------------------------- #
# TOTP token generation
# --------------------------------------------------------------------------- #

_TOKEN_GEN_LOCK = threading.Lock()
_TOKEN_GEN_LAST_ATTEMPT: float | None = None
_TOKEN_GEN_FAILURES: list[float] = []


def _extract_token(payload: Any) -> str | None:
    """Pull the access token out of a /generate/token response.

    The docs name the field `token` and do not show the surrounding envelope, so
    both the bare and the `data`-wrapped shapes are accepted.
    """
    if not isinstance(payload, dict):
        return None
    for container in (payload, payload.get("data")):
        if isinstance(container, dict):
            for key in ("token", "access_token"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def generate_token(
    client_id: str,
    totp_secret: str,
    pin: str,
    *,
    base_url: str = BASE_URL,
    timeout: float = 15.0,
) -> str:
    """Mint a fresh 24-hour access token from a TOTP code. One call per session.

    Limited to one request per MINUTE, and five wrong codes in fifteen minutes
    locks token generation for fifteen more — so this throttles locally and
    raises AuthError with the lockout guidance instead of ever retrying. Each
    success invalidates the previous token, so one process should call this and
    share the result; calling it per request kills the token you are using.
    """
    global _TOKEN_GEN_LAST_ATTEMPT

    cid = str(client_id or "").strip()
    secret = str(totp_secret or "").strip().replace(" ", "")
    mpin = str(pin or "").strip()
    if not cid or not secret or not mpin:
        raise AuthError("generate_token needs a client_id (x-api-key), a TOTP secret and an MPIN")

    try:
        import pyotp
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise ConfigError("pyotp is required for TOTP token generation: pip install pyotp") from exc

    with _TOKEN_GEN_LOCK:
        now_mono = time.monotonic()
        # Stricter than the documented throttle: the gap is measured from the last
        # ATTEMPT, not the last success, so a failing loop cannot hammer the endpoint.
        if _TOKEN_GEN_LAST_ATTEMPT is not None:
            gap = now_mono - _TOKEN_GEN_LAST_ATTEMPT
            if gap < TOKEN_GEN_MIN_GAP_S:
                raise AuthError(
                    f"/generate/token was called {gap:.0f}s ago and allows one request per "
                    f"{TOKEN_GEN_MIN_GAP_S:.0f}s. Reuse the token you already hold — it is "
                    "valid for 24 hours and a new one would invalidate it."
                )
        _TOKEN_GEN_FAILURES[:] = [t for t in _TOKEN_GEN_FAILURES if now_mono - t < TOTP_FAILURE_WINDOW_S]
        if len(_TOKEN_GEN_FAILURES) >= TOTP_MAX_LOCAL_FAILURES:
            # Refusing the attempt that would trip the server-side lockout. Four
            # failures in a row mean a wrong MPIN, a wrong secret or a skewed
            # clock; a fifth locks token generation for 15 minutes.
            raise AuthError(
                f"{len(_TOKEN_GEN_FAILURES)} TOTP attempts have failed in the last "
                f"{TOTP_FAILURE_WINDOW_S / 60:.0f} minutes. Refusing to try again: the next "
                "failure triggers a 15-minute lockout (and three lockouts in an hour, a "
                "one-hour one). Check the MPIN, the TOTP secret, and the host clock (NTP) "
                "before trying again. Any token already issued keeps working."
            )
        _TOKEN_GEN_LAST_ATTEMPT = now_mono

    try:
        totp = pyotp.TOTP(secret)
        # TOTP is derived from real UNIX time, so this deliberately does not go
        # through indbot.clock: a faked clock would produce codes the server
        # rejects, and every rejection counts toward the lockout.
        remaining = totp.interval - (int(time.time()) % totp.interval)
        if remaining <= 2:
            # Never send a code in its last two seconds — it may expire in flight
            # and an expired code counts as a failed attempt.
            log.debug("TOTP code expires in %ds; waiting for the next window", remaining)
            time.sleep(remaining + 0.5)
        code = totp.now()
    except AuthError:
        raise
    except Exception as exc:  # pyotp raises bare ValueError/binascii.Error on a bad secret
        with _TOKEN_GEN_LOCK:
            _TOKEN_GEN_FAILURES.append(time.monotonic())
        raise AuthError(f"could not derive a TOTP code from the stored secret: {exc}") from exc

    log.info("requesting a new access token for client_id %s", _mask(cid))
    started = time.perf_counter()
    try:
        # Same trust store as the main client: this call carries the MPIN and the
        # TOTP code, so it is the last request that should go out unverified.
        _ca = resolve_ca_bundle()
        with httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            verify=str(_ca) if _ca else True,
        ) as client:
            response = client.post(
                "/generate/token",
                headers={"x-api-key": cid, "Content-Type": "application/json"},
                json={"mpin": mpin, "totp": code},
            )
    except httpx.HTTPError as exc:
        with _TOKEN_GEN_LOCK:
            _TOKEN_GEN_FAILURES.append(time.monotonic())
        raise AuthError(
            f"POST /generate/token could not be reached: {exc}. Not retrying — wait for the "
            "next TOTP code, since a repeated or expired code counts toward the lockout."
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    log.debug("POST /generate/token -> %d in %.0f ms", response.status_code, elapsed_ms)

    if not response.is_success:
        with _TOKEN_GEN_LOCK:
            _TOKEN_GEN_FAILURES.append(time.monotonic())
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text[:500]
        detail = payload
        if isinstance(payload, dict):
            detail = payload.get("message") or payload.get("debug_info") or payload.get("error") or payload
        # The docs say the exact status/error_type values here are not yet pinned,
        # so this reads the status and the message rather than branching on a code.
        raise AuthError(
            f"POST /generate/token failed ({response.status_code}): {detail}. Back off for the "
            "full window rather than retrying — wrong codes accumulate toward a 15-minute "
            "lockout. Check the MPIN, and sync the host clock if every code is rejected.",
            status=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthError("POST /generate/token returned a non-JSON body") from exc
    token = _extract_token(payload)
    if not token:
        raise AuthError(
            f"POST /generate/token succeeded but returned no 'token' field: "
            f"{str(payload)[:200]}"
        )
    with _TOKEN_GEN_LOCK:
        _TOKEN_GEN_FAILURES.clear()
    log.info("new access token issued: %s (valid 24h; it replaced any previous one)", _mask(token))
    return token
