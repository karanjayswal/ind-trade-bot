"""Backtester — replays the live strategy over REAL historical option premiums.

CONTRACTS.md section L. The user asked to backtest first, then paper, then live, so
this module is the one that decides whether anything else gets trusted. It is written
to be *honest* before it is written to be flattering.

WHAT IS REAL HERE
-----------------
* Option premiums come from ``GET /market/historical/expired/{interval}`` — the
  exchange's own OHLCV for the actual contract that was trading that day. There is no
  Black-Scholes anywhere in this file.
* Contracts are resolved the way the API intends: ``expired_expiries`` -> pick the
  expiry the live bot would have traded on that day -> ``expired_contracts`` (which
  carries ``lot_size`` for that expiry, so a lot-size change in the past cannot
  mis-size a historical trade) -> ``historical_expired`` for the premiums.
* The strategy, the risk gate, the journal and the day limits are the *same objects*
  the live engine uses, against an in-memory journal in ``mode="backtest"`` so nothing
  here can pollute paper or live statistics or trip the live day-halt.

WHAT IS MODELLED, AND THEREFORE WRONG
-------------------------------------
The expired-candle endpoint returns OHLCV only. There is **no order book, no open
interest and no greeks** in the historical data, so three of the live chain filters
cannot be reproduced:

======================  ==============================================================
live filter             what the backtest does instead
======================  ==============================================================
delta band 0.35-0.60    a **moneyness band** around spot (see ``DEFAULT_MONEYNESS_BAND``)
bid/ask spread          a synthetic spread of ``backtest_spread_pct`` around the close
open interest floor     **not applied at all** — OI is not in the data
======================  ==============================================================

Every one of those substitutions is stated in ``BacktestResult.caveats`` and in
``BacktestResult.assumptions``. A backtest that hides its assumptions is worse than no
backtest, because it is believed.

FILLS
-----
A stop fills at the bar's **close**, not its low. A target fills at the bar's
**close**, not its high. Assuming the best price inside a bar is the classic backtest
lie and it is the single easiest way to invent a profitable bot that does not exist.
Because that cuts both ways, the result also reports how many bars *touched* the stop
or the target intrabar without closing through it, so the size of the assumption is
visible rather than merely disclosed.

LOOK-AHEAD
----------
``Strategy.evaluate`` is called on ``series[:idx + 1]`` — up to **and including** the
current bar, never one element further — with ``now_ts`` set to that bar's close time,
so the strategy's own forming-candle guard runs exactly as it does live.

The process clock is deliberately **not** faked here (no ``INDBOT_FAKE_NOW``): the
server can be replaying a backtest in one thread while a live engine ticks in another,
and a process-global fake clock would corrupt the live side. Every time-dependent call
is passed its timestamp explicitly instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import date, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TYPE_CHECKING

from . import clock
from .config import UNDERLYINGS
from .errors import ConfigError, DataError, IndbotError
from .journal import Journal
from .learner import Learner
from .risk import RiskGate
from .strategy import Strategy
from .types import Candle, OptionLeg, Position, Signal

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .api import INDstocksClient
    from .config import Config
    from .instruments import Instruments

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #

#: Where every API response is cached. A year of NIFTY is thousands of calls against a
#: 5 requests/second data budget, so a re-run must not re-fetch.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest_cache"

#: Trading days of history prepended to each replayed day so EMA50/RSI/ATR/ADX are
#: already warm at 09:15, exactly as they are for the live bot, which fetches a
#: multi-day window. VWAP and the opening range reset per IST day inside the
#: indicators, so a multi-day prefix is safe.
DEFAULT_WARMUP_DAYS = 3

#: Substitute for the delta band. Moneyness is signed so that POSITIVE is out of the
#: money for both sides: ``(K - S)/S`` for a CE, ``(S - K)/S`` for a PE. The default
#: band runs from 0.4% in the money to 1.0% out of it, which is roughly where a weekly
#: index option's delta sits between 0.60 and 0.35 — "roughly" is the whole problem,
#: and it is why this is a caveat and not a feature.
DEFAULT_MONEYNESS_BAND = (-0.004, 0.010)

#: How far either side of the day's spot range to pull premiums for. Wide enough that
#: the strikes the premium budget *can* afford are priced too, so the budget rejection
#: can quote real numbers instead of guessing.
DEFAULT_LADDER_PCT = 0.02

#: Hard cap on symbols priced per day, to bound the call count. 50 symbols per request,
#: so 150 is three requests a day.
MAX_LADDER_SYMBOLS = 150

#: Index options tick in 5 paise. Expired contracts have no ``security_id``, so
#: ``Instruments.tick_size`` cannot be consulted for them.
DEFAULT_TICK_SIZE = 0.05

#: Fewer bars than this in a day means a data gap, not a session. Skip the day loudly.
MIN_CANDLES_PER_DAY = 6

#: Expired expiries are capped at a one-year window per call; page conservatively.
EXPIRY_WINDOW_DAYS = 350

#: Days of expiry lookahead past ``end``, so the last replayed day still finds the
#: expiry that was nearest to it.
EXPIRY_LOOKAHEAD_DAYS = 45

#: Symbols per ``historical_expired`` request, per the API docs.
EXPIRED_BATCH = 50

#: Floating-point slack for "<=" comparisons on money and ratios.
EPS = 1e-9
MONEY_EPS = 0.01

#: Full block reasons kept verbatim for the UI, per blocking rule. The counts are
#: complete; only the examples are sampled, because a year can produce thousands.
MAX_BLOCK_EXAMPLES_PER_RULE = 3


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, falling back rather than raising on broker junk."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return out


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce to int, tolerating "75", 75.0 and None."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Read a config attribute tolerantly; a stub Config must not break a backtest."""
    value = getattr(cfg, name, None)
    return default if value is None else value


def _group_inr(whole: int) -> str:
    """Indian digit grouping: 1234567 -> '12,34,567'."""
    sign = "-" if whole < 0 else ""
    digits = str(abs(whole))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return sign + ",".join(parts + [tail])


def _inr(amount: float, dp: int = 0) -> str:
    """Rupees for a reason sentence, grouped the Indian way."""
    if dp <= 0:
        return "Rs." + _group_inr(int(round(amount)))
    whole = int(abs(amount))
    frac = abs(amount) - whole
    sign = "-" if amount < 0 else ""
    return f"{sign}Rs.{_group_inr(whole)}.{int(round(frac * (10 ** dp))):0{dp}d}"


def _a_pct(fraction: float, dp: int = 0) -> str:
    """A ratio as a percentage: 0.30 -> '30%'."""
    return f"{fraction * 100:.{dp}f}%"


def _floor_tick(value: float, tick: float) -> float:
    """Round DOWN to the tick. Used for any price we RECEIVE."""
    if tick <= 0:
        return round(value, 2)
    return round((value // tick) * tick, 4)


def _ceil_tick(value: float, tick: float) -> float:
    """Round UP to the tick. Used for any price we PAY."""
    if tick <= 0:
        return round(value, 2)
    steps = value / tick
    whole = int(steps)
    if steps - whole > EPS:
        whole += 1
    return round(whole * tick, 4)


def _batched(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size chunks; `size` is clamped to at least 1."""
    step = max(1, int(size))
    for start in range(0, len(items), step):
        yield list(items[start:start + step])


def _synth_book(mid: float, spread_pct: float, tick: float) -> tuple[float, float]:
    """Synthesise a two-sided book around a candle close.

    The expired-candle endpoint has no order book, so `spread_pct` of the mid is split
    either side and each side is rounded AWAY from the mid — the conservative direction.
    Returns ``(0.0, 0.0)`` when the result would be non-positive, which makes the leg
    fail the tradability filter rather than quoting a free option.

    Rounding away from the mid widens the spread in percentage terms on cheap strikes
    (a Rs.2 option lands at 1.95/2.05, a 5% spread). That is realistic, and it is a
    second, honest reason the far-out-of-the-money strikes get rejected.
    """
    if mid <= 0 or spread_pct < 0:
        return 0.0, 0.0
    half = mid * (spread_pct / 200.0)
    bid = _floor_tick(mid - half, tick)
    ask = _ceil_tick(mid + half, tick)
    if bid <= 0 or ask <= 0 or ask < bid:
        return 0.0, 0.0
    return bid, ask


def _moneyness(strike: float, spot: float, side: str) -> float:
    """Signed distance from the money, POSITIVE when out of the money, for both sides.

    Returns 0.0 when spot is unusable rather than dividing by zero — a zero moneyness
    reads as "at the money", and the caller's band check then behaves as if the strike
    were ATM, which is the branch that gets filtered on premium anyway.
    """
    if spot <= 0:
        return 0.0
    return ((strike - spot) / spot) if side == "CE" else ((spot - strike) / spot)


# --------------------------------------------------------------------------- #
# response cache
# --------------------------------------------------------------------------- #

class _Cache:
    """On-disk JSON cache for every API response, keyed by symbol + window + interval.

    A year of NIFTY is thousands of calls against a 5/s data budget, so a second run
    must be served from disk. Writes are atomic (temp file + ``os.replace``) so a
    cancelled run cannot leave a half-written entry that reads as valid data, and an
    unreadable entry counts as a miss rather than as an error.
    """

    def __init__(self, root: Path, *, enabled: bool = True, refresh: bool = False) -> None:
        """Open (and create) the cache directory; a directory it cannot create disables it."""
        self.root = Path(root)
        self.enabled = bool(enabled)
        #: When True, existing entries are ignored on read but still rewritten — how a
        #: user re-pulls a window they believe was cached from a bad day.
        self.refresh = bool(refresh)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("backtest cache disabled: cannot create %s (%s)", self.root, exc)
                self.enabled = False

    def _path(self, kind: str, key: str) -> Path:
        """Shard by the first two hex characters so no directory holds 10k files."""
        digest = hashlib.sha1(f"{kind}|{key}".encode("utf-8")).hexdigest()
        return self.root / kind / digest[:2] / f"{digest}.json"

    def get(self, kind: str, key: str) -> Any | None:
        """The cached payload, or None on a miss. Never raises."""
        if not self.enabled:
            self.misses += 1
            return None
        path = self._path(kind, key)
        if self.refresh:
            self.misses += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, ValueError) as exc:
            # A corrupt entry is a miss, not a failure: re-fetching costs a call, while
            # raising here would abort a run that is otherwise fine.
            logger.warning("backtest cache entry %s unreadable (%s); re-fetching", path, exc)
            self.errors += 1
            self.misses += 1
            return None
        self.hits += 1
        return blob.get("value")

    def put(self, kind: str, key: str, value: Any, *, empty: bool = False) -> None:
        """Store a payload atomically. `empty` marks a genuinely empty API answer."""
        if not self.enabled:
            return
        path = self._path(kind, key)
        payload = {
            "kind": kind,
            "key": key,
            "empty": bool(empty),
            "written_at": clock.now().isoformat(),
            "value": value,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, separators=(",", ":"))
                os.replace(tmp, path)
            except BaseException:
                # Never leave a temp file behind, and never leave a partial entry in
                # place of a good one.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("could not cache %s/%s (%s); continuing uncached", kind, key, exc)
            self.errors += 1
            return
        self.writes += 1

    def as_dict(self) -> dict[str, Any]:
        """Counters for BacktestResult.stats."""
        return {
            "dir": str(self.root),
            "enabled": self.enabled,
            "refresh": self.refresh,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# result shapes
# --------------------------------------------------------------------------- #

@dataclass
class BacktestTrade:
    """One simulated round trip, in plain JSON types so the UI can render it directly."""

    id: str
    day: str
    underlying: str
    setup: str
    bucket: str
    side: str
    strike: float
    expiry: str
    trading_symbol: str
    lot_size: int
    qty: int
    lots: int
    score: float
    spot_at_entry: float
    moneyness_pct: float
    stop_pct: float
    target_pct: float
    entry_price: float
    entry_ts: int
    entry_at: str
    exit_price: float
    exit_ts: int
    exit_at: str
    exit_reason: str
    stop_price: float
    target_price: float
    high_water: float
    costs: float
    pnl: float
    pnl_pct: float
    bars_held: int
    hold_minutes: float
    is_win: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Everything a run produced, including everything it could not produce.

    Fully ``asdict``-able and JSON-serialisable: ``bucket_stats`` is included verbatim
    from the in-memory journal so the user can seed the learner from a backtest, and
    ``caveats``/``assumptions``/``skipped_days`` are what stop the summary being read
    as a promise.
    """

    underlying: str
    start: str
    end: str
    interval: str
    mode: str
    generated_at: str
    summary: dict[str, Any] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    bucket_stats: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    blocks: dict[str, int] = field(default_factory=dict)
    block_examples: list[dict[str, Any]] = field(default_factory=list)
    skipped_days: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The whole result as plain JSON types, for the API and for the UI."""
        return asdict(self)

    @property
    def net_pnl(self) -> float:
        """Realised rupees across the window, after modelled costs."""
        return _as_float(self.summary.get("net_pnl"), 0.0)


@dataclass(frozen=True)
class _Contract:
    """One row of an expired expiry's chain, normalised.

    Expired contracts carry no ``security_id`` — exchanges reuse numeric tokens after
    expiry — so ``trading_symbol`` is the only stable handle.
    """

    trading_symbol: str
    strike: float
    side: str
    lot_size: int
    expiry: str


@dataclass
class _DayOutcome:
    """What one replayed day produced; folded into the result by `run`."""

    trades: list[BacktestTrade] = field(default_factory=list)
    signals: int = 0
    attempted: int = 0
    pick_failures: int = 0
    gate_blocks: int = 0
    premium_blocks: int = 0
    bars: int = 0
    premium_gaps: int = 0
    stop_touched_not_closed: int = 0
    target_touched_not_closed: int = 0
    forced_data_end_exits: int = 0
    halted: bool = False
    halt_reason: str = ""


# --------------------------------------------------------------------------- #
# the backtester
# --------------------------------------------------------------------------- #

class Backtester:
    """Replays the live strategy, chain selection and risk gate over past premiums."""

    def __init__(
        self,
        client: "INDstocksClient",
        instruments: "Instruments",
        cfg: "Config",
        *,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Bind the API client, the instrument master and the config.

        Nothing is fetched here. The config is cloned into ``mode="backtest"`` on every
        ``run`` so a backtest can never be journalled, gated or halted as if it were
        paper or live.
        """
        self.client = client
        self.instruments = instruments
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        #: Counted per HTTP request we cause, including the batches api.py splits for us.
        self._api_calls = 0

    # ------------------------------------------------------------------ run --

    def run(
        self,
        underlying: str,
        start: date,
        end: date,
        *,
        progress: Callable[[dict], None] | None = None,
        warmup_days: int = DEFAULT_WARMUP_DAYS,
        spread_pct: float | None = None,
        moneyness_band: tuple[float, float] | None = None,
        ladder_pct: float = DEFAULT_LADDER_PCT,
        cost_per_order: float | None = None,
        slippage_ticks: int | None = None,
        tick_size: float = DEFAULT_TICK_SIZE,
        learn: bool = False,
        use_cache: bool = True,
        refresh_cache: bool = False,
        cancel: Callable[[], bool] | None = None,
    ) -> BacktestResult:
        """Replay `underlying` over [start, end] and return an auditable result.

        `progress` is called once per trading day with
        ``{"day", "done", "total", "trades", "pnl", ...}`` so the UI can stream it; an
        exception raised by the callback is logged and ignored, because a broken UI
        must not abort a two-minute data pull.

        `learn=False` by default: the headline numbers are the RAW strategy, with no
        bucket gating. ``bucket_stats`` is always reported so the learner can be seeded
        from the run. Pass ``learn=True`` for a genuine walk-forward replay in which the
        learner gates buckets as the backtest's own history accumulates.
        """
        started = _time.monotonic()
        self._api_calls = 0

        name = str(underlying or "").strip().upper()
        if name not in UNDERLYINGS:
            raise ConfigError(
                f"{name!r} is not one of the configured underlyings "
                f"({', '.join(sorted(UNDERLYINGS))})",
                key="underlying",
            )
        if start > end:
            raise ConfigError(
                f"start {start.isoformat()} is after end {end.isoformat()}", key="start"
            )
        today = clock.today()
        if end >= today:
            # Only EXPIRED contracts have history on this endpoint. Clamp rather than
            # fail: a user who types "to today" wants everything that is available.
            logger.warning(
                "backtest end %s is not in the past; clamping to %s — only expired "
                "contracts have candles on /market/historical/expired",
                end.isoformat(), (today - timedelta(days=1)).isoformat(),
            )
            end = today - timedelta(days=1)
            if start > end:
                raise ConfigError(
                    "the whole backtest window is in the future or today; expired "
                    "contract history starts yesterday at the earliest",
                    key="end",
                )

        meta = UNDERLYINGS[name]
        segment = str(meta.get("fno_segment") or "NFO").upper()
        exchange = str(meta.get("exchange") or "NSE").upper()

        cfg = self._backtest_config()
        interval = str(_cfg_get(cfg, "interval", "5minute"))
        interval_s = clock.interval_minutes(interval) * 60

        spread = _as_float(
            spread_pct if spread_pct is not None else _cfg_get(cfg, "backtest_spread_pct", 1.5), 1.5
        )
        if spread < 0:
            logger.warning("backtest_spread_pct %.2f is negative; using 0", spread)
            spread = 0.0
        costs_per_order = _as_float(
            cost_per_order if cost_per_order is not None else _cfg_get(cfg, "cost_per_order", 20.0),
            20.0,
        )
        slip_ticks = max(0, _as_int(
            slippage_ticks if slippage_ticks is not None else _cfg_get(cfg, "slippage_ticks", 1), 1
        ))
        tick = _as_float(tick_size, DEFAULT_TICK_SIZE)
        if tick <= 0:
            logger.warning("tick_size %r is not positive; using %s", tick_size, DEFAULT_TICK_SIZE)
            tick = DEFAULT_TICK_SIZE
        band = tuple(moneyness_band) if moneyness_band else tuple(
            _cfg_get(cfg, "backtest_moneyness_band", DEFAULT_MONEYNESS_BAND)
        )
        lo_m, hi_m = _as_float(band[0], DEFAULT_MONEYNESS_BAND[0]), _as_float(
            band[1], DEFAULT_MONEYNESS_BAND[1]
        )
        if lo_m > hi_m:
            lo_m, hi_m = hi_m, lo_m
        ladder = max(0.002, _as_float(ladder_pct, DEFAULT_LADDER_PCT))
        warm_days = max(0, _as_int(warmup_days, DEFAULT_WARMUP_DAYS))

        cache = _Cache(self.cache_dir, enabled=bool(use_cache), refresh=bool(refresh_cache))

        # The journal is in memory and tagged "backtest": the day-halt, the trade cap
        # and the bucket statistics are all real, and none of them touch data/indbot.db.
        journal = Journal(":memory:", default_mode="backtest")
        errors: list[dict[str, Any]] = []
        try:
            return self._run_locked(
                name=name,
                exchange=exchange,
                segment=segment,
                start=start,
                end=end,
                cfg=cfg,
                journal=journal,
                cache=cache,
                interval=interval,
                interval_s=interval_s,
                spread=spread,
                costs_per_order=costs_per_order,
                slip_ticks=slip_ticks,
                tick=tick,
                band=(lo_m, hi_m),
                ladder=ladder,
                warm_days=warm_days,
                learn=bool(learn),
                progress=progress,
                cancel=cancel,
                started=started,
                errors=errors,
            )
        finally:
            journal.close()

    # --------------------------------------------------------------- config --

    def _backtest_config(self) -> "Config":
        """A throwaway clone of the config pinned to ``mode="backtest"``, never armed.

        Mutating the live config would be a real-money bug (``armed`` is one field
        away), so the clone is mandatory and failing to produce one is fatal.
        """
        try:
            return self.cfg.copy_with({"mode": "backtest", "armed": False})
        except AttributeError:
            pass
        except IndbotError as exc:
            logger.warning("Config.copy_with rejected mode=backtest (%s); cloning by hand", exc)
        # Fallback for a stub Config: copy the dataclass and force the mode. The
        # clone must never be the caller's object, because the gate reads cfg.mode.
        try:
            import copy as _copy

            clone = _copy.copy(self.cfg)
            clone.mode = "backtest"       # type: ignore[attr-defined]
            clone.armed = False           # type: ignore[attr-defined]
            return clone
        except Exception as exc:  # noqa: BLE001 - never silently reuse the live config
            raise ConfigError(
                f"could not clone the config for a backtest ({exc}); refusing to run a "
                "backtest against the live config object",
            ) from exc

    # ----------------------------------------------------------- resolution --

    def _scrip_code(self, underlying: str, exchange: str) -> str:
        """The underlying index's REST ``scrip-code``, as ``SEGMENT_TOKEN``.

        The constants in config are guesses for everything but NIFTY (CONTRACTS.md A),
        so the live index instrument file wins and a disagreement is logged.
        """
        meta = UNDERLYINGS[underlying]
        constant = str(meta.get("scrip") or "").strip()
        label = str(meta.get("label") or underlying)
        resolved = ""
        try:
            resolved = str(self.instruments.index_scrip(label) or "").strip()
        except IndbotError as exc:
            logger.warning(
                "index_scrip(%r) failed (%s); falling back to the configured scrip %r",
                label, exc, constant,
            )
        except Exception:  # noqa: BLE001 - logged with a traceback, never swallowed
            logger.exception("index_scrip(%r) raised; falling back to %r", label, constant)
        if resolved and constant and resolved != constant:
            logger.warning(
                "scrip for %s: config says %s, the instruments file says %s — using the file",
                underlying, constant, resolved,
            )
        scrip = resolved or constant
        if not scrip:
            raise DataError(
                f"could not resolve a scrip id for {underlying} (label {label!r} is not in "
                "the index instruments file and no constant is set)",
                underlying=underlying,
            )
        return f"{exchange}_{scrip}"

    def _expiries(self, underlying: str, frm: date, to: date, cache: _Cache) -> list[str]:
        """Every expired expiry in [frm, to], ascending, paged under the 1-year cap.

        The API returns these DESCENDING; they are re-sorted ascending here because the
        caller wants "the first expiry on or after day D", which is a forward scan.
        """
        out: set[str] = set()
        window_start = frm
        while window_start <= to:
            window_end = min(to, window_start + timedelta(days=EXPIRY_WINDOW_DAYS))
            key = f"{underlying}|{window_start.isoformat()}|{window_end.isoformat()}"
            cached = cache.get("expiries", key)
            if cached is None:
                dates = self.client.expired_expiries(
                    underlying, window_start.isoformat(), window_end.isoformat()
                )
                self._api_calls += 1
                cache.put("expiries", key, dates, empty=not dates)
            else:
                dates = list(cached)
            out.update(str(d).strip() for d in dates if str(d).strip())
            window_start = window_end + timedelta(days=1)
        return sorted(out)

    @staticmethod
    def _expiry_for_day(day: date, expiries: Sequence[str]) -> str | None:
        """The expiry the live bot would have traded on `day`: the first one on or after it.

        On expiry day itself that is the expiring contract, which is exactly what
        ``ChainPicker.nearest_expiry`` returns live.
        """
        iso = day.isoformat()
        for value in expiries:  # ascending
            if value >= iso:
                return value
        return None

    def _chain(self, underlying: str, expiry: str, cache: _Cache) -> list[_Contract]:
        """The full option chain for one past expiry, normalised and de-duplicated.

        ``lot_size`` is taken from the API row, not from today's instrument master: lot
        sizes change by circular, and sizing a 2024 trade with a 2026 lot size would
        mis-state every rupee figure in the result.
        """
        key = f"{underlying}|{expiry}|OPTIDX"
        rows = cache.get("contracts", key)
        if rows is None:
            rows = self.client.expired_contracts(underlying, expiry, "OPTIDX")
            self._api_calls += 1
            cache.put("contracts", key, rows, empty=not rows)

        fallback_lot = 0
        out: dict[str, _Contract] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("trading_symbol") or "").strip()
            side = str(row.get("option_type") or "").strip().upper()
            if not symbol or side not in ("CE", "PE"):
                continue  # futures rows on a monthly expiry, and anything malformed
            strike = _as_float(row.get("strike_price"), 0.0)
            if strike <= 0:
                logger.debug("contract %s has no usable strike_price; dropped", symbol)
                continue
            lot = _as_int(row.get("lot_size"), 0)
            if lot <= 0:
                if fallback_lot <= 0:
                    fallback_lot = self._fallback_lot(underlying, expiry)
                lot = fallback_lot
                if lot <= 0:
                    # Never guess a lot size: every rupee figure downstream is a
                    # multiple of it, so an unknown lot size drops the contract.
                    logger.warning("contract %s has no lot_size and no fallback; dropped", symbol)
                    continue
            out[symbol] = _Contract(
                trading_symbol=symbol,
                strike=strike,
                side=side,
                lot_size=lot,
                expiry=str(row.get("expiry") or expiry),
            )
        if not out:
            raise DataError(
                f"the {underlying} {expiry} expired chain returned no usable OPTIDX rows",
                underlying=underlying, expiry=expiry,
            )
        return sorted(out.values(), key=lambda c: (c.strike, c.side))

    def _fallback_lot(self, underlying: str, expiry: str) -> int:
        """Lot size from the instrument master, used only when the API row omits it."""
        try:
            return max(0, _as_int(self.instruments.lot_size(underlying, expiry), 0))
        except IndbotError as exc:
            logger.warning("lot_size(%s, %s) failed: %s", underlying, expiry, exc)
            return 0
        except Exception:  # noqa: BLE001 - logged with a traceback
            logger.exception("lot_size(%s, %s) raised", underlying, expiry)
            return 0

    # -------------------------------------------------------------- candles --

    def _underlying_candles(
        self, scrip_code: str, interval: str, days: Sequence[date], cache: _Cache
    ) -> dict[date, list[Candle]]:
        """Index candles for every day in `days`, bucketed by IST date.

        Fetched in windows of the interval's documented maximum (7 days on 5minute), so
        a year costs ~52 calls instead of ~250. A wider request would silently return
        only the maximum window's data, so the window width is the coverage.
        """
        out: dict[date, list[Candle]] = {}
        if not days:
            return out
        max_days = max(1, _as_int(clock.INTERVAL_MAX_DAYS.get(interval, 7), 7))
        ordered = sorted(set(days))
        first, last = ordered[0], ordered[-1]

        window_start = first
        while window_start <= last:
            window_end = min(last, window_start + timedelta(days=max_days - 1))
            start_ms = clock.epoch_ms(clock.session_open(window_start))
            # end_time is exclusive; +1 minute so the 15:30 candle is included.
            end_ms = clock.epoch_ms(clock.session_close(window_end) + timedelta(minutes=1))
            key = f"{scrip_code}|{interval}|{start_ms}|{end_ms}"
            candles = cache.get("index_candles", key)
            if candles is None:
                fetched = self.client.historical(interval, [scrip_code], start_ms, end_ms)
                self._api_calls += 1
                candles = fetched.get(scrip_code) or []
                cache.put("index_candles", key, candles, empty=not candles)
            for raw in candles:
                try:
                    ts = int(raw["ts"])
                except (KeyError, TypeError, ValueError):
                    continue
                out.setdefault(clock.to_ist(ts).date(), []).append(raw)  # type: ignore[arg-type]
            window_start = window_end + timedelta(days=1)

        for day_candles in out.values():
            day_candles.sort(key=lambda c: int(c["ts"]))
        return out

    def _ladder(
        self,
        contracts: Sequence[_Contract],
        low: float,
        high: float,
        ladder_pct: float,
    ) -> tuple[list[_Contract], bool]:
        """The strikes worth pricing for a day, and whether the set had to be trimmed.

        The band spans the day's own spot range widened by `ladder_pct`, so it covers
        both the near-the-money strikes the moneyness filter wants and the cheaper
        out-of-the-money strikes the premium budget can actually afford — the budget
        rejection has to be able to quote real prices, not guesses.
        """
        if low <= 0 or high <= 0:
            return [], False
        lo = low * (1.0 - ladder_pct)
        hi = high * (1.0 + ladder_pct)
        mid = (low + high) / 2.0
        inside = [c for c in contracts if lo <= c.strike <= hi]
        if len(inside) <= MAX_LADDER_SYMBOLS:
            return inside, False
        # Trim to the strikes nearest the day's midpoint; a trimmed ladder is reported.
        inside.sort(key=lambda c: (abs(c.strike - mid), c.strike, c.side))
        trimmed = inside[:MAX_LADDER_SYMBOLS]
        trimmed.sort(key=lambda c: (c.strike, c.side))
        return trimmed, True

    def _expired_candles(
        self,
        symbols: Sequence[str],
        interval: str,
        start_ms: int,
        end_ms: int,
        segment: str,
        cache: _Cache,
    ) -> dict[str, dict[int, Candle]]:
        """Premiums for `symbols` over one window, as ``{symbol: {ts: candle}}``.

        Cached per symbol per window per interval, so a re-run with a slightly different
        strike ladder still hits on the overlap. Only the misses are requested, batched
        fifty symbols a call as the endpoint requires.
        """
        out: dict[str, dict[int, Candle]] = {}
        misses: list[str] = []
        for symbol in symbols:
            key = f"{segment}|{interval}|{symbol}|{start_ms}|{end_ms}"
            cached = cache.get("expired_candles", key)
            if cached is None:
                misses.append(symbol)
            else:
                out[symbol] = self._index_by_ts(cached)

        for batch in _batched(misses, EXPIRED_BATCH):
            fetched = self.client.historical_expired(interval, batch, start_ms, end_ms, segment)
            self._api_calls += 1
            for symbol in batch:
                candles = fetched.get(symbol) or []
                key = f"{segment}|{interval}|{symbol}|{start_ms}|{end_ms}"
                cache.put("expired_candles", key, candles, empty=not candles)
                out[symbol] = self._index_by_ts(candles)
        return out

    @staticmethod
    def _index_by_ts(candles: Iterable[Any]) -> dict[int, Candle]:
        """Index a candle list by its epoch-SECONDS open time for O(1) bar lookup."""
        out: dict[int, Candle] = {}
        for raw in candles or []:
            try:
                out[int(raw["ts"])] = raw  # type: ignore[index]
            except (KeyError, TypeError, ValueError):
                continue
        return out

    # ---------------------------------------------------------- strike pick --

    def _premium_cap(self, cfg: "Config", lot_size: int, stop_pct: float) -> tuple[float, float, str]:
        """The per-unit and per-lot premium budget, and which rule set it.

        Mirrors CONTRACTS.md section F filter 2 and the RISK ARITHMETIC note. Two
        readings are reconciled and the TIGHTER wins, deliberately:

        * ``cfg.effective_max_premium(lot_size)`` — rupees PER UNIT (section A).
        * the same constraint recomputed from ``max_trade_loss`` and THIS signal's
          ``stop_pct``, which the strategy may have widened for a low-ADX regime.

        Taking the minimum means a misread unit or a widened stop can only ever make the
        budget tighter, never let a trade risk more than ``max_trade_loss``. The live
        ``ChainPicker`` performs the same reconciliation; if the two ever disagree the
        backtest is lying about which strikes the live bot could buy.
        """
        max_trade_loss = _as_float(_cfg_get(cfg, "max_trade_loss", 1000.0), 1000.0)
        denominator = stop_pct * lot_size
        risk_cap = (max_trade_loss / denominator) if denominator > 0 else 0.0

        cfg_cap = 0.0
        try:
            cfg_cap = _as_float(cfg.effective_max_premium(lot_size), 0.0)
        except AttributeError:
            logger.warning("cfg has no effective_max_premium(); using the risk-derived cap")
        except IndbotError as exc:
            logger.warning("effective_max_premium(%d) failed (%s); using the risk cap", lot_size, exc)

        usable = [value for value in (cfg_cap, risk_cap) if value > 0]
        if not usable:
            # Both unusable. Zero blocks every strike and the reason says why, which is
            # the safe direction when the budget itself cannot be computed.
            logger.error(
                "no usable premium cap for lot %d (stop %.3f, max_trade_loss %.2f)",
                lot_size, stop_pct, max_trade_loss,
            )
            return 0.0, 0.0, "unknown"
        cap_unit = min(usable)
        source = "config" if (0 < cfg_cap < risk_cap) else "risk"
        return cap_unit, cap_unit * lot_size, source

    def _pick(
        self,
        underlying: str,
        side: str,
        *,
        cfg: "Config",
        exchange: str,
        expiry: str,
        spot: float,
        stop_pct: float,
        contracts: Sequence[_Contract],
        bars: dict[str, dict[int, Candle]],
        bar_ts: int,
        day_volume: dict[str, int],
        spread_pct: float,
        tick: float,
        band: tuple[float, float],
    ) -> tuple[OptionLeg | None, list[str], str]:
        """The leg the live ChainPicker would have bought, or None plus why not.

        Returns ``(leg, reasons, rule)``. `rule` is "" on success and otherwise names
        which filter emptied the list, so the caller can count rejections by cause
        instead of parsing the sentences it prints.

        Runs CONTRACTS.md section F's filters in the same ORDER — side, premium budget,
        "delta", spread, liquidity, tradability — because the order is what makes the
        rejection honest: the budget removes the near-the-money strikes BEFORE the
        moneyness check sees them, which is exactly why a Rs.1,000-per-trade account
        cannot buy an at-the-money index option.

        Two filters are substituted, and the reasons say so in the sentence the UI
        prints: delta is not in the historical data, so a moneyness band stands in for
        it, and open interest is not in the data at all, so that floor is NOT applied.
        """
        want = str(side or "").strip().upper()
        if want not in ("CE", "PE"):
            raise DataError(f"side must be CE or PE, got {side!r}", underlying=underlying)
        reasons: list[str] = []
        lo_m, hi_m = band

        # --- filter 1: correct side, and a premium actually printed on this bar ---
        candidates: list[tuple[_Contract, Candle]] = []
        unpriced = 0
        for contract in contracts:
            if contract.side != want:
                continue  # a partition, not a rejection; 20 "wrong side" lines help nobody
            bar = bars.get(contract.trading_symbol, {}).get(bar_ts)
            if bar is None:
                unpriced += 1
                continue
            if _as_float(bar.get("c"), 0.0) <= 0:
                unpriced += 1
                continue
            candidates.append((contract, bar))
        if unpriced:
            # Not noise: a strike with no print in this five-minute bar genuinely did
            # not trade, which is the illiquidity the live oi/volume filters catch.
            reasons.append(
                f"{unpriced} {underlying} {want} strike(s) printed no premium in the "
                f"{clock.to_ist(bar_ts).strftime('%H:%M')} candle, so they were not priced "
                "and could not be considered."
            )
        if not candidates:
            reasons.insert(
                0,
                f"No {underlying} {expiry} {want} strike had a premium in the "
                f"{clock.to_ist(bar_ts).strftime('%H:%M')} candle — nothing to choose from.",
            )
            return None, reasons, "chain_no_premium"

        lot_size = candidates[0][0].lot_size
        stop = stop_pct if stop_pct > 0 else 0.30
        if stop_pct <= 0:
            logger.warning("stop_pct was %r for %s; using 0.30 for the premium cap", stop_pct, underlying)
        cap_unit, cap_lot, cap_source = self._premium_cap(cfg, lot_size, stop)

        legs: list[tuple[OptionLeg, float]] = []  # (leg, signed moneyness)
        for contract, bar in candidates:
            close = _as_float(bar.get("c"), 0.0)
            bid, ask = _synth_book(close, spread_pct, tick)
            leg = OptionLeg(
                underlying=underlying,
                expiry=contract.expiry,
                strike=contract.strike,
                side=contract.side,
                # Expired contracts have no orderable security_id. The prefix makes that
                # obvious in the journal and would be rejected outright by POST /order.
                security_id=f"EXPIRED:{contract.trading_symbol}",
                trading_symbol=contract.trading_symbol,
                lot_size=contract.lot_size,
                exchange=exchange,
                ltp=close,
                bid=bid,
                ask=ask,
                oi=0,                                   # unknown: not in the historical data
                volume=max(0, _as_int(day_volume.get(contract.trading_symbol), 0)),
                iv=0.0,                                 # unknown
                delta=0.0,                              # unknown
                tick_size=tick,
            )
            legs.append((leg, _moneyness(contract.strike, spot, want)))

        # --- filter 2: the per-trade premium budget (the binding constraint) ---
        affordable: list[tuple[OptionLeg, float]] = []
        premium_reasons: list[str] = []
        for leg, money in legs:
            if leg.lot_cost <= cap_lot + MONEY_EPS:
                affordable.append((leg, money))
            else:
                premium_reasons.append(
                    f"{leg.label} at {_inr(leg.ltp, 2)}: one lot ({leg.lot_size} units) costs "
                    f"{_inr(leg.lot_cost)} and a {_a_pct(stop)} stop risks "
                    f"{_inr(leg.lot_cost * stop)} — over the {_inr(cap_lot)} a lot this trade "
                    "is allowed to cost."
                )
        premium_dropped = len(legs) - len(affordable)
        if not affordable:
            reasons.insert(0, self._premium_arithmetic(
                underlying, want, legs, lot_size, cap_unit, cap_lot, stop, cap_source
            ))
            reasons.extend(premium_reasons[:MAX_BLOCK_EXAMPLES_PER_RULE])
            return None, reasons, "chain_premium_budget"
        reasons.extend(premium_reasons[:MAX_BLOCK_EXAMPLES_PER_RULE])

        # --- filter 3 (SUBSTITUTED): moneyness band in place of the delta band ---
        in_band: list[tuple[OptionLeg, float]] = []
        for leg, money in affordable:
            if lo_m - EPS <= money <= hi_m + EPS:
                in_band.append((leg, money))
            else:
                where = "too far out of the money" if money > hi_m else "too deep in the money"
                reasons.append(
                    f"{leg.label}: {money * 100:+.2f}% from spot {_inr(spot, 2)} is outside the "
                    f"{lo_m * 100:+.2f}%..{hi_m * 100:+.2f}% moneyness band — {where}. "
                    "(The backtest has no delta: the expired-candle endpoint returns OHLCV "
                    "only, so moneyness stands in for the 0.35-0.60 delta band.)"
                )

        # --- filter 4: bid-ask spread (synthetic, so this honours the config setting) ---
        max_spread = _as_float(_cfg_get(cfg, "max_spread_pct", 2.0), 2.0)
        tight: list[tuple[OptionLeg, float]] = []
        for leg, money in in_band:
            if leg.bid <= 0 or leg.ask <= 0:
                reasons.append(
                    f"{leg.label}: the modelled book is one-sided at {_inr(leg.ltp, 2)} — a "
                    "premium this small cannot carry the assumed spread, so it is not tradable."
                )
            elif leg.spread_pct <= max_spread + EPS:
                tight.append((leg, money))
            else:
                reasons.append(
                    f"{leg.label}: modelled spread {leg.spread_pct:.1f}% ({_inr(leg.spread, 2)} on a "
                    f"{_inr(leg.mid, 2)} mid) is over the {max_spread:.1f}% limit — the round trip "
                    f"costs {_inr(leg.spread * leg.lot_size)} a lot before the trade moves."
                )

        # --- filter 5: liquidity. Volume only; OI is NOT in the historical data. ---
        min_volume = _as_int(_cfg_get(cfg, "min_volume", 10000), 10000)
        liquid: list[tuple[OptionLeg, float]] = []
        for leg, money in tight:
            if leg.volume < min_volume:
                reasons.append(
                    f"{leg.label}: {leg.volume:,} units traded so far today against a floor of "
                    f"{min_volume:,} — too thin to get out of. (Open interest is not available "
                    "for expired contracts, so the OI floor was not applied.)"
                )
            else:
                liquid.append((leg, money))

        # --- filter 6: a real price and a real two-sided book ---
        survivors = [(leg, money) for leg, money in liquid if leg.ltp > 0 and leg.bid > 0 and leg.ask > 0]

        if not survivors:
            # Name the stage that emptied the list and show the funnel, because "nothing
            # qualified" with twenty per-strike lines underneath is unreadable.
            stage = (
                "the moneyness band" if not in_band
                else "the spread limit" if not tight
                else "the volume floor" if not liquid
                else "the two-sided-book check"
            )
            head = (
                f"No {underlying} {expiry} {want} strike survived {stage}. Of {len(legs)} strike(s) "
                f"priced in this candle: {len(affordable)} were inside the {_inr(cap_lot)} per-lot "
                f"budget, {len(in_band)} of those inside the moneyness band, {len(tight)} inside the "
                f"spread limit, {len(liquid)} over the volume floor."
            )
            if premium_dropped:
                # The honest root cause: the budget took the near-the-money strikes off the
                # table before the later filters ever saw them.
                head += (
                    f" The per-trade budget had already removed {premium_dropped} of them — "
                    f"everything above {_inr(cap_unit, 2)} a unit — so only far-out-of-the-money "
                    "strikes reached the moneyness, spread and volume checks."
                )
            reasons.insert(0, head)
            # When the budget had already taken the near-the-money strikes away and the
            # moneyness band is what came up empty, the budget IS the root cause — report
            # it as such, because that is the number the user has to act on.
            rule = ("chain_budget_vs_moneyness"
                    if (premium_dropped and not in_band) else "chain_filters")
            return None, reasons, rule

        # Rank: nearest the money, then the cheaper premium. The live rule is "nearest a
        # 0.5 delta, then tightest spread, then deepest OI"; none of those three numbers
        # exist in the historical data, and |moneyness| is the closest honest proxy for
        # the first. The substitution is recorded in BacktestResult.assumptions.
        survivors.sort(key=lambda pair: (abs(pair[1]), pair[0].ltp, pair[0].strike))
        best, best_money = survivors[0]
        chosen = (
            f"Chose {best.label} at {_inr(best.ltp, 2)} (best of {len(survivors)} tradable "
            f"{'strike' if len(survivors) == 1 else 'strikes'}): {best_money * 100:+.2f}% from "
            f"spot {_inr(spot, 2)}, modelled spread {best.spread_pct:.1f}%, {best.volume:,} units "
            f"traded today. One lot is {best.lot_size} units, so it costs {_inr(best.lot_cost)} "
            f"and a {_a_pct(stop)} stop risks {_inr(best.lot_cost * stop)}."
        )
        return best, [chosen] + reasons, ""

    def _premium_arithmetic(
        self,
        underlying: str,
        side: str,
        legs: Sequence[tuple[OptionLeg, float]],
        lot_size: int,
        cap_unit: float,
        cap_lot: float,
        stop: float,
        cap_source: str,
    ) -> str:
        """The one sentence the user most needs when every strike is over budget.

        CONTRACTS.md's RISK ARITHMETIC section is explicit that this must NOT be papered
        over: the bot does not widen the stop, does not raise the loss limit and does not
        drift to lottery strikes. It states the arithmetic and offers the two honest
        choices instead.
        """
        cheapest = min((leg for leg, _ in legs), key=lambda leg: leg.ltp, default=None)
        head = (
            f"Every {underlying} {side} strike priced in this candle is over the per-trade "
            f"premium budget. One lot is {lot_size} units, a {_a_pct(stop)} stop is the risk, "
            f"and the {cap_source} limit caps the position at {_inr(cap_lot)} a lot "
            f"({_inr(cap_unit, 2)} a unit)."
        )
        if cheapest is not None:
            head += (
                f" The cheapest strike priced here is {cheapest.label} at "
                f"{_inr(cheapest.ltp, 2)}, which is {_inr(cheapest.lot_cost)} a lot and risks "
                f"{_inr(cheapest.lot_cost * stop)} on the stop."
            )
        head += (
            " Two honest choices: raise max_trade_loss so one lot of a near-the-money strike "
            f"fits (about {_inr(max(0.0, (cheapest.lot_cost * stop) if cheapest else 0.0))} "
            "for the cheapest strike above), or accept a tighter stop and its whipsaw cost "
            "(min_stop_pct exists to stop that becoming a 5% stop the spread alone triggers)."
        )
        return head

    # ------------------------------------------------------------ the replay --

    def _run_locked(
        self,
        *,
        name: str,
        exchange: str,
        segment: str,
        start: date,
        end: date,
        cfg: "Config",
        journal: Journal,
        cache: _Cache,
        interval: str,
        interval_s: int,
        spread: float,
        costs_per_order: float,
        slip_ticks: int,
        tick: float,
        band: tuple[float, float],
        ladder: float,
        warm_days: int,
        learn: bool,
        progress: Callable[[dict], None] | None,
        cancel: Callable[[], bool] | None,
        started: float,
        errors: list[dict[str, Any]],
    ) -> BacktestResult:
        """Day-by-day orchestration. Everything that can be pre-fetched is, once."""
        strategy = Strategy(cfg)
        learner: Learner | None = Learner(journal, cfg) if learn else None
        gate = RiskGate(journal, cfg, learner=learner)

        scrip_code = self._scrip_code(name, exchange)
        days = clock.trading_days(start, end)
        if not days:
            raise DataError(
                f"there is no trading day between {start.isoformat()} and {end.isoformat()}",
                underlying=name,
            )

        # Warm-up days sit BEFORE the window so EMA50/RSI/ATR/ADX are already warm at
        # 09:15 on the first replayed day, exactly as they are for the live bot.
        warm_pool: list[date] = []
        cursor = start
        for _ in range(warm_days):
            cursor = clock.prev_trading_day(cursor)
            warm_pool.insert(0, cursor)
        index_candles = self._underlying_candles(scrip_code, interval, warm_pool + days, cache)
        expiries = self._expiries(
            name, start, end + timedelta(days=EXPIRY_LOOKAHEAD_DAYS), cache
        )
        if not expiries:
            raise DataError(
                f"no expired {name} expiries between {start.isoformat()} and "
                f"{end.isoformat()} — the window may be too recent for the expired-contract "
                "endpoints, which only know expiries that have already passed",
                underlying=name,
            )

        trades: list[BacktestTrade] = []
        equity: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        blocks: dict[str, int] = {}
        block_examples: dict[str, list[str]] = {}
        notes: set[str] = set()
        totals = _DayOutcome()
        cum_pnl = 0.0
        total = len(days)
        cancelled = False
        halted_days = 0
        replayed = 0

        self._emit(progress, {
            "day": days[0].isoformat(), "done": 0, "total": total,
            "trades": 0, "pnl": 0.0, "phase": "starting",
        })

        for done, day in enumerate(days, start=1):
            if cancel is not None and self._cancelled(cancel):
                cancelled = True
                errors.append({
                    "type": "Cancelled",
                    "message": f"the run was cancelled after {done - 1} of {total} days",
                })
                break

            why = ""
            outcome = _DayOutcome()
            expiry = self._expiry_for_day(day, expiries)
            day_candles = index_candles.get(day) or []

            if expiry is None:
                why = (
                    "no expired expiry falls on or after this day in the window, so the "
                    "contract the live bot would have traded cannot be identified"
                )
            elif len(day_candles) < MIN_CANDLES_PER_DAY:
                why = (
                    f"only {len(day_candles)} {interval} candle(s) came back for the index "
                    f"(need at least {MIN_CANDLES_PER_DAY}) — an exchange holiday the clock's "
                    "holiday table does not list, or a data gap"
                )

            if not why:
                try:
                    outcome, why, day_notes = self._prepare_and_replay(
                        day=day, name=name, exchange=exchange, segment=segment,
                        expiry=str(expiry), cfg=cfg, journal=journal, gate=gate,
                        strategy=strategy, learner=learner, cache=cache,
                        interval=interval, interval_s=interval_s,
                        day_candles=day_candles, index_candles=index_candles,
                        warm_days=warm_days, contracts_ladder=ladder, spread=spread,
                        tick=tick, slip_ticks=slip_ticks, costs_per_order=costs_per_order,
                        band=band, blocks=blocks, block_examples=block_examples,
                        errors=errors,
                    )
                    notes.update(day_notes)
                except IndbotError as exc:
                    # One bad day must not lose a year of work, but it must be visible:
                    # it is counted as skipped AND reported in errors.
                    logger.warning("backtest: %s %s skipped — %s", name, day.isoformat(), exc)
                    errors.append(exc.as_dict())
                    why = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
                except Exception as exc:  # noqa: BLE001 - logged with a traceback, never silent
                    # Anything the modules below did not wrap (a sqlite3 error, a malformed
                    # candle). Reported in errors AND as a skipped day, so it can never read
                    # as a flat, uneventful session.
                    logger.exception("backtest: %s %s raised", name, day.isoformat())
                    errors.append({"type": type(exc).__name__, "message": str(exc),
                                   "context": {"day": day.isoformat()}})
                    why = f"{type(exc).__name__}: {exc}"

            if why:
                skipped.append({"day": day.isoformat(), "why": why})
            else:
                replayed += 1
                trades.extend(outcome.trades)
                totals = self._fold(totals, outcome)
                if outcome.halted:
                    halted_days += 1

            day_pnl = sum(t.pnl for t in outcome.trades)
            cum_pnl += day_pnl
            equity.append({
                "day": day.isoformat(),
                "trades": len(outcome.trades),
                "realized_pnl": round(day_pnl, 2),
                "cumulative_pnl": round(cum_pnl, 2),
                "skipped": bool(why),
                "halted": outcome.halted,
            })
            self._emit(progress, {
                "day": day.isoformat(), "done": done, "total": total,
                "trades": len(trades), "pnl": round(cum_pnl, 2),
                "day_trades": len(outcome.trades), "day_pnl": round(day_pnl, 2),
                "skipped": len(skipped), "api_calls": self._api_calls,
                "cache_hits": cache.hits, "elapsed_s": round(_time.monotonic() - started, 1),
                "phase": "running",
            })

        bucket_stats = self._bucket_stats(journal, errors)
        elapsed = _time.monotonic() - started
        assumptions = self._assumptions(
            cfg=cfg, interval=interval, spread=spread, band=band, ladder=ladder,
            tick=tick, slip_ticks=slip_ticks, costs_per_order=costs_per_order,
            warm_days=warm_days, learn=learn, segment=segment, scrip_code=scrip_code,
        )
        result = BacktestResult(
            underlying=name,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            mode="backtest",
            generated_at=clock.now().isoformat(),
            summary=self._summary(
                trades=trades, equity=equity, totals=totals, days_total=total,
                days_replayed=replayed, days_skipped=len(skipped), halted_days=halted_days,
                cancelled=cancelled, blocks=blocks,
            ),
            trades=trades,
            equity_curve=equity,
            bucket_stats=bucket_stats,
            caveats=self._caveats(
                cfg=cfg, spread=spread, band=band, costs_per_order=costs_per_order,
                slip_ticks=slip_ticks, tick=tick, totals=totals, skipped=len(skipped),
                trades=len(trades), learn=learn, notes=sorted(notes), errors=errors,
                blocks=blocks, warm_days=warm_days,
            ),
            assumptions=assumptions,
            blocks=dict(sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0]))),
            block_examples=[
                {"rule": rule, "examples": examples}
                for rule, examples in sorted(block_examples.items())
            ],
            skipped_days=skipped,
            stats={
                "api_calls": self._api_calls,
                "cache": cache.as_dict(),
                "elapsed_s": round(elapsed, 2),
                "days_total": total,
                "days_replayed": replayed,
                "days_skipped": len(skipped),
                "days_halted": halted_days,
                "bars_walked": totals.bars,
                "signals_generated": totals.signals,
                "signals_attempted": totals.attempted,
                "premium_gaps": totals.premium_gaps,
                "forced_data_end_exits": totals.forced_data_end_exits,
                "stop_touched_not_closed": totals.stop_touched_not_closed,
                "target_touched_not_closed": totals.target_touched_not_closed,
                "cancelled": cancelled,
            },
            config=self._config_dict(cfg),
            errors=errors,
        )
        logger.info(
            "backtest %s %s..%s: %d trade(s), net %s, %d day(s) skipped, %d API call(s), "
            "%d cache hit(s) in %.1fs",
            name, start.isoformat(), end.isoformat(), len(trades),
            _inr(result.net_pnl), len(skipped), self._api_calls, cache.hits, elapsed,
        )
        self._emit(progress, {
            "day": end.isoformat(), "done": total, "total": total,
            "trades": len(trades), "pnl": round(cum_pnl, 2), "phase": "done",
        })
        return result

    @staticmethod
    def _cancelled(cancel: Callable[[], bool]) -> bool:
        """Ask the caller whether to stop, treating a broken callback as "keep going"."""
        try:
            return bool(cancel())
        except Exception:  # noqa: BLE001 - a caller's predicate must not abort a run
            logger.exception("backtest cancel predicate raised; continuing")
            return False

    @staticmethod
    def _emit(progress: Callable[[dict], None] | None, payload: dict[str, Any]) -> None:
        """Push one progress frame. A broken UI callback must not abort a data pull."""
        if progress is None:
            return
        try:
            progress(payload)
        except Exception:  # noqa: BLE001 - deliberately contained: not an order path
            logger.exception("backtest progress callback raised; continuing")

    @staticmethod
    def _fold(total: _DayOutcome, day: _DayOutcome) -> _DayOutcome:
        """Accumulate a day's counters into the run totals."""
        return _DayOutcome(
            trades=[],
            signals=total.signals + day.signals,
            attempted=total.attempted + day.attempted,
            pick_failures=total.pick_failures + day.pick_failures,
            gate_blocks=total.gate_blocks + day.gate_blocks,
            premium_blocks=total.premium_blocks + day.premium_blocks,
            bars=total.bars + day.bars,
            premium_gaps=total.premium_gaps + day.premium_gaps,
            stop_touched_not_closed=total.stop_touched_not_closed + day.stop_touched_not_closed,
            target_touched_not_closed=total.target_touched_not_closed + day.target_touched_not_closed,
            forced_data_end_exits=total.forced_data_end_exits + day.forced_data_end_exits,
        )

    # ------------------------------------------------------------- one day --

    def _prepare_and_replay(
        self,
        *,
        day: date,
        name: str,
        exchange: str,
        segment: str,
        expiry: str,
        cfg: "Config",
        journal: Journal,
        gate: RiskGate,
        strategy: Strategy,
        learner: Learner | None,
        cache: _Cache,
        interval: str,
        interval_s: int,
        day_candles: list[Candle],
        index_candles: dict[date, list[Candle]],
        warm_days: int,
        contracts_ladder: float,
        spread: float,
        tick: float,
        slip_ticks: int,
        costs_per_order: float,
        band: tuple[float, float],
        blocks: dict[str, int],
        block_examples: dict[str, list[str]],
        errors: list[dict[str, Any]],
    ) -> tuple[_DayOutcome, str, set[str]]:
        """Price one day's strike ladder, then walk the session bar by bar.

        Returns ``(outcome, skip_reason, notes)``. A non-empty skip reason means the day
        produced nothing and must be reported as skipped, not as a flat day — a silent
        gap reads as a clean backtest when it is not.
        """
        notes: set[str] = set()
        contracts = self._chain(name, expiry, cache)

        lows = [v for v in (_as_float(c.get("l"), 0.0) for c in day_candles) if v > 0]
        highs = [v for v in (_as_float(c.get("h"), 0.0) for c in day_candles) if v > 0]
        if not lows or not highs:
            return _DayOutcome(), "the index candles for this day carry no usable high/low", notes
        low, high = min(lows), max(highs)

        ladder_contracts, trimmed = self._ladder(contracts, low, high, contracts_ladder)
        if not ladder_contracts:
            return _DayOutcome(), (
                f"no strike in the {expiry} chain falls within {_a_pct(contracts_ladder, 1)} of "
                f"the day's {low:,.2f}-{high:,.2f} index range"
            ), notes
        if trimmed:
            notes.add(
                f"At least one day's strike ladder was trimmed to the {MAX_LADDER_SYMBOLS} "
                "strikes nearest the money to bound the number of API calls; strikes further "
                "out were never priced and so were never considered."
            )

        start_ms = clock.epoch_ms(clock.session_open(day))
        # end_time is exclusive, so +1 minute keeps the 15:30 candle in the window.
        end_ms = clock.epoch_ms(clock.session_close(day) + timedelta(minutes=1))
        bars = self._expired_candles(
            [c.trading_symbol for c in ladder_contracts], interval, start_ms, end_ms, segment, cache
        )
        priced = [c for c in ladder_contracts if bars.get(c.trading_symbol)]
        if not priced:
            return _DayOutcome(), (
                f"/market/historical/expired/{interval} returned no candles for any of the "
                f"{len(ladder_contracts)} {expiry} strikes in this day's ladder"
            ), notes
        if len(priced) < len(ladder_contracts):
            notes.add(
                "Some strikes in the ladder had no candles at all on some days and were "
                "excluded from selection on those days."
            )

        if sum(_as_int(c.get("v"), 0) for c in day_candles) <= 0:
            # Index candles routinely report v=0. VWAP is volume-weighted, so it comes
            # back None and the vwap_reclaim / ema_cross setups cannot fire at all.
            notes.add(
                "The index candles reported zero volume on at least one day. VWAP is "
                "volume-weighted, so on those days it is undefined and the vwap_reclaim and "
                "ema_cross setups could not fire, and the volume-surge score term was skipped."
            )

        warm: list[Candle] = []
        cursor = day
        for _ in range(warm_days):
            cursor = clock.prev_trading_day(cursor)
            warm = list(index_candles.get(cursor) or []) + warm

        outcome = self._replay_day(
            day=day, name=name, expiry=expiry, exchange=exchange, cfg=cfg, journal=journal,
            gate=gate, strategy=strategy, learner=learner, day_candles=day_candles,
            warm_candles=warm, contracts=priced, bars=bars, interval_s=interval_s,
            spread=spread, tick=tick, slip_ticks=slip_ticks, costs_per_order=costs_per_order,
            band=band, blocks=blocks, block_examples=block_examples, errors=errors,
        )
        return outcome, "", notes

    def _replay_day(
        self,
        *,
        day: date,
        name: str,
        expiry: str,
        exchange: str,
        cfg: "Config",
        journal: Journal,
        gate: RiskGate,
        strategy: Strategy,
        learner: Learner | None,
        day_candles: list[Candle],
        warm_candles: list[Candle],
        contracts: Sequence[_Contract],
        bars: dict[str, dict[int, Candle]],
        interval_s: int,
        spread: float,
        tick: float,
        slip_ticks: int,
        costs_per_order: float,
        band: tuple[float, float],
        blocks: dict[str, int],
        block_examples: dict[str, list[str]],
        errors: list[dict[str, Any]],
    ) -> _DayOutcome:
        """Walk one session bar by bar: manage the position, then look for an entry.

        The order inside a bar is the live engine's tick order (CONTRACTS.md K): the
        open position is managed FIRST, then the day-halt and the position cap are
        checked, and only then is the strategy evaluated. Reversing that would let a new
        entry be taken on the same bar an exit happened, which the live bot cannot do.
        """
        out = _DayOutcome()
        series = list(warm_candles) + list(day_candles)
        n_warm = len(warm_candles)
        position: Position | None = None
        open_meta: dict[str, Any] = {}
        day_volume: dict[str, int] = {}
        last_mark: float = 0.0
        seq = 0
        flat_by = self._flat_by(cfg)

        for i, bar in enumerate(day_candles):
            bar_ts = int(bar["ts"])
            bar_close_ts = bar_ts + interval_s
            bar_close_dt = clock.to_ist(bar_close_ts)
            out.bars += 1

            # Day-cumulative traded volume per symbol, up to AND INCLUDING this bar —
            # that is what the live option chain's `volume` field means. Comparing a
            # single bar's volume against min_volume would reject almost everything.
            for symbol, by_ts in bars.items():
                candle = by_ts.get(bar_ts)
                if candle is not None:
                    day_volume[symbol] = day_volume.get(symbol, 0) + max(0, _as_int(candle.get("v"), 0))

            # ---- 1. manage the open position against THIS bar -------------------
            if position is not None:
                opt = bars.get(position.leg.trading_symbol, {}).get(bar_ts)
                mark = _as_float(opt.get("c"), 0.0) if opt else 0.0
                if mark > 0:
                    low_px = _as_float(opt.get("l"), mark)
                    high_px = _as_float(opt.get("h"), mark)
                    # Count how often a live stop/target order would have fired inside
                    # this bar while the close-only rule kept holding. This is the size
                    # of the fill assumption, not a correction to it.
                    stop_in_force = gate.effective_stop(position)
                    if low_px <= stop_in_force and mark > stop_in_force:
                        out.stop_touched_not_closed += 1
                    target = _as_float(position.target_price, 0.0)
                    if target > 0 and high_px >= target and mark < target:
                        out.target_touched_not_closed += 1
                    last_mark = mark
                    # High water advances on CLOSES only. Using the bar's high would arm
                    # the trail earlier and invent protection the close-only rule cannot
                    # deliver; arming late is the conservative direction.
                    gate.update_trail(position, mark)
                else:
                    out.premium_gaps += 1

                # A zero mark is passed through deliberately: check_exit_required skips
                # the price-based exits on an unusable LTP but still enforces flat_by and
                # the time stop, so a data gap cannot carry a position past 15:10.
                reason = gate.check_exit_required(position, mark, now=bar_close_dt)
                if reason:
                    exit_mark = mark if mark > 0 else last_mark
                    trade = self._close_position(
                        position, open_meta, exit_mark=exit_mark, exit_ts=bar_close_ts,
                        reason=reason, spread=spread, tick=tick, slip_ticks=slip_ticks,
                        journal=journal, learner=learner, errors=errors, bar_index=i,
                    )
                    out.trades.append(trade)
                    position = None
                    open_meta = {}

            # ---- 2. entries: only when flat, and only inside the day's budget ----
            if position is not None:
                continue
            halted, halt_reason = gate.day_halted("backtest", day=day)
            if halted:
                out.halted = True
                out.halt_reason = halt_reason
                self._note_block(blocks, block_examples, "day_halt", halt_reason)
                break

            # UP TO AND INCLUDING this bar. Slicing one element further would hand the
            # strategy a candle that had not closed yet and invent a profitable bot.
            prefix = series[: n_warm + i + 1]
            try:
                signals = strategy.evaluate(name, prefix, now_ts=bar_close_ts)
            except DataError as exc:
                logger.warning("%s %s: strategy rejected the candle series — %s",
                               name, day.isoformat(), exc)
                errors.append(exc.as_dict())
                continue
            if not signals:
                continue
            out.signals += len(signals)

            # The live engine takes THE SINGLE BEST signal per tick and records the rest
            # (CONTRACTS.md K step 5). Trying each in turn until one passed would be a
            # cherry-picking bot that does not exist.
            best = signals[0]
            for other in signals[1:]:
                self._record_signal(
                    journal, other, taken=False, errors=errors,
                    reason=(
                        f"Not attempted: {best.setup} scored higher ({best.score:.2f} vs "
                        f"{other.score:.2f}) on the same candle and the bot evaluates one "
                        "signal per tick."
                    ),
                )

            sig = best
            if learner is not None:
                sig, _verdict = learner.adjust(sig)
            out.attempted += 1

            leg, pick_reasons, pick_rule = self._pick(
                name, sig.side, cfg=cfg, exchange=exchange, expiry=expiry, spot=sig.spot,
                stop_pct=sig.stop_pct, contracts=contracts, bars=bars, bar_ts=bar_ts,
                day_volume=day_volume, spread_pct=spread, tick=tick, band=band,
            )
            if leg is None:
                out.pick_failures += 1
                if pick_rule in ("chain_premium_budget", "chain_budget_vs_moneyness"):
                    out.premium_blocks += 1
                self._note_block(blocks, block_examples, pick_rule or "chain_filters",
                                 " ".join(pick_reasons[:2]))
                self._record_signal(journal, sig, taken=False, errors=errors,
                                    reason=" ".join(pick_reasons[:4]))
                continue

            verdict = gate.check_entry(
                sig, leg, open_positions=[], mode="backtest", funds=None,
                blocked_buckets=(learner.blocked_buckets() if learner is not None else None),
                now=bar_close_dt,
            )
            if not verdict.ok:
                out.gate_blocks += 1
                rule = verdict.blocking_rule or "unknown"
                if rule == "per_trade_risk":
                    out.premium_blocks += 1
                self._note_block(blocks, block_examples, rule, "; ".join(verdict.reasons[:2]))
                self._record_signal(journal, sig, taken=False, errors=errors,
                                    reason="; ".join(verdict.reasons))
                continue

            seq += 1
            position, open_meta = self._open_position(
                sig=sig, leg=leg, day=day, name=name, seq=seq, bar_index=i,
                bar_close_ts=bar_close_ts, tick=tick, slip_ticks=slip_ticks,
                costs_per_order=costs_per_order, pick_reasons=pick_reasons,
                journal=journal, errors=errors,
            )
            last_mark = position.entry_price

        # ---- 3. never carry a position overnight ------------------------------
        if position is not None:
            last_ts = int(day_candles[-1]["ts"]) + interval_s
            opt = bars.get(position.leg.trading_symbol, {}).get(int(day_candles[-1]["ts"]))
            exit_mark = _as_float(opt.get("c"), 0.0) if opt else 0.0
            if exit_mark <= 0:
                exit_mark = last_mark
            at_or_past_flat = flat_by is not None and clock.to_ist(last_ts).time() >= flat_by
            reason = "flat_by" if at_or_past_flat else "data_end"
            if not at_or_past_flat:
                # The session's candles stopped before flat_by. The live bot would have
                # kept managing the trade; this run had no more prices for it.
                out.forced_data_end_exits += 1
                logger.warning(
                    "%s %s: the candle series ends at %s, before flat_by — position %s was "
                    "force-closed at the last printed premium",
                    name, day.isoformat(), clock.to_ist(last_ts).strftime("%H:%M"), position.id,
                )
            trade = self._close_position(
                position, open_meta, exit_mark=exit_mark, exit_ts=last_ts, reason=reason,
                spread=spread, tick=tick, slip_ticks=slip_ticks, journal=journal,
                learner=learner, errors=errors, bar_index=len(day_candles) - 1,
            )
            out.trades.append(trade)

        return out

    @staticmethod
    def _flat_by(cfg: "Config") -> dtime | None:
        """`flat_by` as a time, or None when it is malformed (then only the gate enforces it)."""
        raw = str(_cfg_get(cfg, "flat_by", "15:10"))
        try:
            return clock.parse_hhmm(raw)
        except (ValueError, AttributeError):
            logger.error("config flat_by=%r is not HH:MM; the end-of-day exit is labelled "
                         "data_end instead", raw)
            return None

    # ------------------------------------------------------------ positions --

    def _open_position(
        self,
        *,
        sig: Signal,
        leg: OptionLeg,
        day: date,
        name: str,
        seq: int,
        bar_index: int,
        bar_close_ts: int,
        tick: float,
        slip_ticks: int,
        costs_per_order: float,
        pick_reasons: list[str],
        journal: Journal,
        errors: list[dict[str, Any]],
    ) -> tuple[Position, dict[str, Any]]:
        """Open one lot at the signal bar's close, and journal it.

        The fill price is the synthesised ASK plus `slippage_ticks`, rounded UP to the
        tick — every rounding on a price we pay goes against us. `entry_ts` is the bar's
        CLOSE, not its open, because that is the instant the live engine could have acted.
        """
        entry_px = _ceil_tick(leg.ask + slip_ticks * tick, tick)
        if entry_px <= 0:
            # Cannot happen after filter 6, which requires ask > 0. Guarded anyway,
            # because a zero entry price would make every ratio below meaningless.
            raise DataError(
                f"refusing to open {leg.label}: the modelled entry price came out at "
                f"{entry_px} from ask {leg.ask}",
                underlying=name, trading_symbol=leg.trading_symbol,
            )
        stop_price = max(0.0, _floor_tick(entry_px * (1.0 - sig.stop_pct), tick))
        target_price = _ceil_tick(entry_px * (1.0 + sig.target_pct), tick)
        trade_id = f"bt-{name}-{day.isoformat()}-{seq}"

        position = Position(
            id=trade_id,
            leg=leg,
            signal_setup=sig.setup,
            bucket=sig.bucket,
            # ONE lot, expressed in UNITS, as the order endpoint and the journal expect.
            qty=leg.lot_size,
            entry_price=entry_px,
            entry_ts=bar_close_ts,
            entry_order_id=trade_id,
            stop_price=stop_price,
            target_price=target_price,
            mode="backtest",
            high_water=entry_px,
            # Both legs of the round trip are charged up front so the marked P&L never
            # flatters an open trade by a cost it is certain to pay.
            costs=round(2.0 * max(0.0, costs_per_order), 2),
        )
        try:
            journal.open_trade(position, "backtest")
        except IndbotError as exc:
            # The journal drives the day-halt and the trade cap, so a failed write means
            # the rest of the day is gated on wrong numbers. Report it, never swallow it.
            logger.error("backtest journal.open_trade(%s) failed: %s", trade_id, exc)
            errors.append(exc.as_dict())
        self._record_signal(journal, sig, taken=True, errors=errors, reason="", trade_id=trade_id)

        meta = {
            "day": day.isoformat(),
            "score": round(float(sig.score), 4),
            "spot": round(float(sig.spot), 2),
            "moneyness": round(_moneyness(leg.strike, float(sig.spot), leg.side) * 100.0, 3),
            "stop_pct": round(float(sig.stop_pct), 4),
            "target_pct": round(float(sig.target_pct), 4),
            "entry_bar": bar_index,
            # The chosen-strike sentence leads, then the signal's own confluence terms.
            "reasons": ([pick_reasons[0]] if pick_reasons else []) + list(sig.reasons),
        }
        logger.debug(
            "backtest %s: opened %s at %.2f (stop %.2f, target %.2f) on %s",
            trade_id, leg.label, entry_px, stop_price, target_price, sig.setup,
        )
        return position, meta

    def _close_position(
        self,
        position: Position,
        meta: dict[str, Any],
        *,
        exit_mark: float,
        exit_ts: int,
        reason: str,
        spread: float,
        tick: float,
        slip_ticks: int,
        journal: Journal,
        learner: Learner | None,
        errors: list[dict[str, Any]],
        bar_index: int,
    ) -> BacktestTrade:
        """Close at the bar's CLOSE, journal it, and return the trade record.

        The exit price is the synthesised BID at that close minus `slippage_ticks`,
        rounded DOWN to the tick: a stop fills at the close and not at the bar's low, a
        target fills at the close and not at the bar's high. Assuming the best price
        inside a bar is the classic backtest lie, and it is the one that most often
        invents a profitable bot.
        """
        bid, _ask = _synth_book(exit_mark, spread, tick)
        # A premium too small to carry the assumed spread exits at zero, not at a floor:
        # an option that has gone to nothing really is worth nothing.
        fill = max(0.0, _floor_tick(bid - slip_ticks * tick, tick)) if bid > 0 else 0.0

        position.exit_price = fill
        position.exit_ts = int(exit_ts)
        position.exit_reason = str(reason)
        position.exit_order_id = f"{position.id}-x"
        try:
            journal.close_trade(position)
        except IndbotError as exc:
            # A realised loss that never reaches the journal leaves the day-halt switched
            # off for the rest of that day, so the backtest would take trades the live bot
            # could not. Surfaced in errors and in the caveats, never swallowed.
            logger.error("backtest journal.close_trade(%s) failed: %s", position.id, exc)
            errors.append(exc.as_dict())
        if learner is not None:
            # Walk-forward: the next signal is gated on everything closed so far.
            learner.refresh()

        entry_bar = _as_int(meta.get("entry_bar"), bar_index)
        hold_minutes = max(0.0, (int(exit_ts) - int(position.entry_ts)) / 60.0)
        trade = BacktestTrade(
            id=position.id,
            day=str(meta.get("day") or clock.to_ist(position.entry_ts).date().isoformat()),
            underlying=position.leg.underlying,
            setup=position.signal_setup,
            bucket=position.bucket,
            side=position.leg.side,
            strike=float(position.leg.strike),
            expiry=position.leg.expiry,
            trading_symbol=position.leg.trading_symbol,
            lot_size=int(position.leg.lot_size),
            qty=int(position.qty),
            lots=int(position.lots),
            score=_as_float(meta.get("score"), 0.0),
            spot_at_entry=_as_float(meta.get("spot"), 0.0),
            moneyness_pct=_as_float(meta.get("moneyness"), 0.0),
            stop_pct=_as_float(meta.get("stop_pct"), 0.0),
            target_pct=_as_float(meta.get("target_pct"), 0.0),
            entry_price=round(float(position.entry_price), 2),
            entry_ts=int(position.entry_ts),
            entry_at=clock.to_ist(position.entry_ts).isoformat(),
            exit_price=round(fill, 2),
            exit_ts=int(exit_ts),
            exit_at=clock.to_ist(exit_ts).isoformat(),
            exit_reason=str(reason),
            stop_price=round(float(position.stop_price), 2),
            target_price=round(float(position.target_price), 2),
            high_water=round(float(position.high_water), 2),
            costs=round(float(position.costs), 2),
            pnl=round(position.pnl(), 2),
            pnl_pct=round(position.pnl_pct(), 2),
            bars_held=max(0, bar_index - entry_bar),
            hold_minutes=round(hold_minutes, 1),
            is_win=bool(position.is_win),
            reasons=list(meta.get("reasons") or []),
        )
        logger.debug(
            "backtest %s: closed %s at %.2f (%s) for %s",
            trade.id, trade.trading_symbol, fill, reason, _inr(trade.pnl, 2),
        )
        return trade

    @staticmethod
    def _record_signal(
        journal: Journal,
        sig: Signal,
        *,
        taken: bool,
        errors: list[dict[str, Any]],
        reason: str = "",
        trade_id: str | None = None,
    ) -> None:
        """Journal a signal and whether it was traded; a write failure is reported."""
        try:
            journal.record_signal(sig, taken=taken, reason=reason, mode="backtest", trade_id=trade_id)
        except IndbotError as exc:
            logger.warning("backtest journal.record_signal failed: %s", exc)
            errors.append(exc.as_dict())

    @staticmethod
    def _note_block(
        blocks: dict[str, int],
        examples: dict[str, list[str]],
        rule: str,
        sentence: str,
    ) -> None:
        """Count a rejection by rule, keeping the first few sentences verbatim.

        The counts are complete; only the examples are sampled, because a year of
        five-minute bars can reject the same way thousands of times.
        """
        key = str(rule or "unknown")
        blocks[key] = blocks.get(key, 0) + 1
        bucket = examples.setdefault(key, [])
        text = (sentence or "").strip()
        if text and len(bucket) < MAX_BLOCK_EXAMPLES_PER_RULE and text not in bucket:
            bucket.append(text)

    @staticmethod
    def _bucket_stats(journal: Journal, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-bucket outcome stats for mode="backtest", for seeding the learner."""
        try:
            return list(journal.bucket_stats(mode="backtest"))
        except IndbotError as exc:
            logger.warning("backtest bucket_stats failed: %s", exc)
            errors.append(exc.as_dict())
            return []

    @staticmethod
    def _config_dict(cfg: "Config") -> dict[str, Any]:
        """The config the run used, as plain JSON types."""
        try:
            return dict(cfg.as_dict())
        except AttributeError:
            return {}

    # -------------------------------------------------------------- summary --

    def _summary(
        self,
        *,
        trades: list[BacktestTrade],
        equity: list[dict[str, Any]],
        totals: _DayOutcome,
        days_total: int,
        days_replayed: int,
        days_skipped: int,
        halted_days: int,
        cancelled: bool,
        blocks: dict[str, int],
    ) -> dict[str, Any]:
        """Headline numbers, plus a plain sentence that reads correctly at zero trades."""
        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)          # a POSITIVE magnitude
        net = sum(t.pnl for t in trades)
        costs = sum(t.costs for t in trades)
        curve = [0.0] + [_as_float(row.get("cumulative_pnl"), 0.0) for row in equity]
        peak = curve[0]
        drawdown = 0.0
        for value in curve:
            peak = max(peak, value)
            drawdown = max(drawdown, peak - value)

        exits: dict[str, int] = {}
        setups: dict[str, int] = {}
        for trade in trades:
            exits[trade.exit_reason] = exits.get(trade.exit_reason, 0) + 1
            setups[trade.setup] = setups.get(trade.setup, 0) + 1

        return {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "scratches": n - len(wins) - len(losses),
            "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "gross_win": round(gross_win, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net, 2),
            "costs": round(costs, 2),
            "expectancy": round(net / n, 2) if n else 0.0,
            # None, not 0 or inf, when there is nothing to divide by — a profit factor of
            # "infinity" reads as a perfect system when it only means no losses yet.
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
            "max_drawdown": round(drawdown, 2),
            "best_trade": round(max((t.pnl for t in trades), default=0.0), 2),
            "worst_trade": round(min((t.pnl for t in trades), default=0.0), 2),
            "avg_hold_minutes": round(sum(t.hold_minutes for t in trades) / n, 1) if n else 0.0,
            "days_total": days_total,
            "days_replayed": days_replayed,
            "days_with_trades": sum(1 for row in equity if _as_int(row.get("trades"), 0) > 0),
            "days_skipped": days_skipped,
            "days_halted": halted_days,
            "signals_generated": totals.signals,
            "signals_attempted": totals.attempted,
            "blocked_by_premium_budget": totals.premium_blocks,
            "blocked_by_chain": totals.pick_failures,
            "blocked_by_risk_gate": totals.gate_blocks,
            "cancelled": cancelled,
            "exits_by_reason": dict(sorted(exits.items())),
            "trades_by_setup": dict(sorted(setups.items())),
            "headline": self._headline(
                n=n, net=net, wins=len(wins), totals=totals, days_replayed=days_replayed,
                blocks=blocks, cancelled=cancelled,
            ),
        }

    @staticmethod
    def _headline(
        *,
        n: int,
        net: float,
        wins: int,
        totals: _DayOutcome,
        days_replayed: int,
        blocks: dict[str, int],
        cancelled: bool,
    ) -> str:
        """One sentence the UI can print above the chart.

        A zero-trade run is the expected outcome at the default limits (CONTRACTS.md
        RISK ARITHMETIC: a Rs.1,000 per-trade cap with a 30% stop allows about Rs.44 of
        NIFTY premium against a Rs.150-200 at-the-money option), so it says that
        plainly instead of rendering as a flat, successful-looking equity curve.
        """
        if cancelled:
            return "The run was cancelled before it finished; these numbers cover part of the window only."
        if n == 0:
            if totals.signals == 0:
                return (
                    f"No setup fired in {days_replayed} replayed day(s). "
                    "Nothing was blocked — the strategy simply had no trigger."
                )
            budget = totals.premium_blocks
            if budget and budget >= max(1, totals.attempted // 2):
                return (
                    f"{totals.signals} signal(s) fired and none became a trade: {budget} were "
                    "stopped by the per-trade premium budget. At these limits one lot of a "
                    "near-the-money option costs more than the per-trade loss cap allows. "
                    "Raise max_trade_loss, or accept a tighter stop and its whipsaw cost — "
                    "the bot will not widen the stop or drift to lottery strikes on its own."
                )
            return (
                f"{totals.signals} signal(s) fired and none became a trade: "
                f"{totals.pick_failures} were stopped by the strike filters and "
                f"{totals.gate_blocks} by the risk gate. Read `blocks` and `block_examples` "
                "for which rule and why."
            )
        sign = "made" if net >= 0 else "lost"
        return (
            f"{n} trade(s) over {days_replayed} replayed day(s): {wins} winner(s), "
            f"{sign} {_inr(abs(net))} after modelled costs. Read `caveats` before believing it."
        )

    # ----------------------------------------------------------- assumptions --

    def _assumptions(
        self,
        *,
        cfg: "Config",
        interval: str,
        spread: float,
        band: tuple[float, float],
        ladder: float,
        tick: float,
        slip_ticks: int,
        costs_per_order: float,
        warm_days: int,
        learn: bool,
        segment: str,
        scrip_code: str,
    ) -> dict[str, Any]:
        """Every modelling choice, as machine-readable numbers next to the caveats text."""
        return {
            "price_source": f"GET /market/historical/expired/{interval} (real exchange OHLCV)",
            "segment": segment,
            "underlying_scrip_code": scrip_code,
            "spread_model": "constant percentage of the candle close, split either side",
            "spread_pct": round(spread, 4),
            "tick_size": tick,
            "slippage_ticks_each_way": slip_ticks,
            "cost_per_order": round(max(0.0, costs_per_order), 2),
            "cost_per_round_trip": round(2.0 * max(0.0, costs_per_order), 2),
            "entry_fill": "synthesised ask at the signal bar's close, rounded up to the tick",
            "exit_fill": "synthesised bid at the exit bar's close, rounded down to the tick",
            "stop_fill_rule": "the bar's CLOSE, never its low",
            "target_fill_rule": "the bar's CLOSE, never its high",
            "high_water_rule": "advances on bar closes only, never on bar highs",
            "lots_per_trade": 1,
            "warmup_trading_days": warm_days,
            "learner_applied": bool(learn),
            "filters_substituted": {
                "delta_band": {
                    "live": [
                        _as_float(_cfg_get(cfg, "min_delta", 0.35), 0.35),
                        _as_float(_cfg_get(cfg, "max_delta", 0.60), 0.60),
                    ],
                    "backtest": "moneyness band, fraction of spot, positive = out of the money",
                    "moneyness_band": [round(band[0], 5), round(band[1], 5)],
                    "why": "the expired-candle endpoint returns OHLCV only — no greeks",
                },
                "ranking": {
                    "live": "|delta - 0.5| asc, then spread_pct asc, then oi desc",
                    "backtest": "|moneyness| asc, then premium asc, then strike asc",
                },
            },
            "filters_not_applied": {
                "min_oi": {
                    "value": _as_int(_cfg_get(cfg, "min_oi", 50000), 50000),
                    "why": "open interest is not returned for expired contracts at all",
                },
            },
            "volume_basis": "day-cumulative traded volume per contract, summed from the candles "
                            "up to and including the signal bar (matches the live chain's field)",
            "ladder_pct_around_day_range": round(ladder, 4),
            "max_ladder_symbols_per_day": MAX_LADDER_SYMBOLS,
            "expiry_rule": "the first expiry on or after the replayed day, i.e. what "
                           "ChainPicker.nearest_expiry would have returned that morning",
            "one_signal_per_bar": True,
            "journal": "in-memory sqlite, mode=\"backtest\" — paper and live stats are untouched",
        }

    # --------------------------------------------------------------- caveats --

    def _caveats(
        self,
        *,
        cfg: "Config",
        spread: float,
        band: tuple[float, float],
        costs_per_order: float,
        slip_ticks: int,
        tick: float,
        totals: _DayOutcome,
        skipped: int,
        trades: int,
        learn: bool,
        notes: list[str],
        errors: list[dict[str, Any]],
        blocks: dict[str, int],
        warm_days: int,
    ) -> list[str]:
        """Everything that makes this result an estimate rather than a record.

        CONTRACTS.md section L requires the spread model, the stop-fill rule, slippage,
        brokerage and expiry survivorship to be stated. The rest are the substitutions
        the historical data forced, and the gaps this particular run actually hit — a
        caveat list that does not change with the run is decoration.
        """
        max_spread = _as_float(_cfg_get(cfg, "max_spread_pct", 2.0), 2.0)
        out: list[str] = [
            "Premiums are REAL: every fill is priced from the exchange's own OHLCV for the "
            "contract that was actually trading, via /market/historical/expired. Nothing here "
            "is a Black-Scholes approximation.",

            f"SPREAD IS MODELLED. The historical endpoint returns no order book, so a flat "
            f"{spread:.2f}% of the candle close is split either side as bid/ask and each side is "
            f"rounded away from the mid. Real spreads widen on gaps, in the first and last "
            f"minutes, and on every strike away from the money, so a day's worst fills are "
            f"better here than they were in reality. Rounding away from the mid also means a "
            f"cheap strike carries a much wider percentage spread (a Rs.2 option lands at "
            f"1.95/2.05, about 5%), which is realistic and is a second reason cheap "
            f"out-of-the-money strikes fail the {max_spread:.1f}% spread filter.",

            "STOPS FILL AT THE BAR'S CLOSE, NOT ITS LOW, and targets fill at the bar's close, "
            "not its high. Assuming the best price inside a bar is the classic backtest lie. "
            "The cost of that choice cuts both ways and is measured rather than merely "
            f"disclaimed: on {totals.stop_touched_not_closed} bar(s) the premium traded through "
            f"the stop intrabar and closed back above it (a live stop order would have fired and "
            f"this run kept holding), and on {totals.target_touched_not_closed} bar(s) it traded "
            f"through the target and closed back below it (a live limit order would have filled "
            f"and this run kept holding). Treat those two counts as the error bars on the P&L.",

            f"SLIPPAGE beyond the modelled spread is {slip_ticks} tick(s) each way "
            f"({slip_ticks * tick:.2f} rupees a unit at a {tick:.2f} tick), taken from "
            f"cfg.slippage_ticks. There is no modelling of impact, of a partial fill, of a "
            f"queue, or of the order being rejected — every order is assumed to fill in full, "
            f"immediately, at the bar's close.",

            f"BROKERAGE AND TAXES are a flat {_inr(max(0.0, costs_per_order), 2)} an order "
            f"({_inr(2.0 * max(0.0, costs_per_order), 2)} a round trip) from cfg.cost_per_order. "
            "That is a stand-in for brokerage, exchange fees, STT, GST, stamp duty and SEBI "
            "charges combined; the real figure on an index-option round trip is usually higher "
            "and scales with the premium, so net P&L here is optimistic by that difference.",

            "EXPIRY SURVIVORSHIP: each day trades the expiry that was nearest to it, which is "
            "what the live bot picks. Nothing searches for the expiry that would have worked "
            "best, and no result from a later expiry is substituted — but the window itself is "
            "chosen by the user, and a window picked because it looked good is survivorship "
            "bias no backtester can remove.",

            "NO GREEKS, NO BOOK, NO OPEN INTEREST. /market/historical/expired returns "
            f"{{ts,o,h,l,c,v}} and nothing else, so the live delta band "
            f"({_as_float(_cfg_get(cfg, 'min_delta', 0.35), 0.35):.2f}-"
            f"{_as_float(_cfg_get(cfg, 'max_delta', 0.60), 0.60):.2f}) CANNOT be applied. "
            f"A moneyness band of {band[0] * 100:+.2f}%..{band[1] * 100:+.2f}% of spot stands in "
            "for it (positive is out of the money, for both CE and PE), and the strike ranking "
            "falls back from \"nearest a 0.5 delta\" to \"nearest the money\". A moneyness band "
            "is not a delta band: the same distance from spot is a very different delta one day "
            "before expiry than five days before, so the backtest trades somewhat different "
            "strikes than the live bot would.",

            f"THE OPEN-INTEREST FLOOR ({_as_int(_cfg_get(cfg, 'min_oi', 50000), 50000):,}) IS NOT "
            "APPLIED AT ALL, because open interest is not in the historical data. The live bot "
            "applies it, so live will reject some strikes this backtest happily traded. Volume "
            "IS applied, as the contract's day-cumulative traded volume up to the signal bar.",

            "Entries fill at the signal bar's close plus half the modelled spread, with no "
            "allowance for the latency between the candle closing and the live engine's next "
            "poll (default 20 seconds), during which the premium moves.",

            "The trailing stop's high-water mark advances on bar CLOSES only, never on bar "
            "highs, so the trail arms later here than it would live. That makes trail exits "
            "rarer and the result more conservative, not less.",

            "One signal is attempted per bar — the highest-scoring one — exactly as the live "
            "engine does. The others are journalled as not attempted. A backtest that tried "
            "each signal in turn until one passed would be a bot that does not exist.",

            f"Indicators are warmed up with {warm_days} prior trading day(s) of index candles, "
            "as the live bot's multi-day fetch does. VWAP and the opening range reset per IST "
            "day inside the indicators, so the warm-up cannot leak across the session boundary.",

            "Risk is enforced by the SAME RiskGate the live bot uses, against an in-memory "
            "journal tagged mode=\"backtest\": the per-trade loss cap, the daily loss halt (which "
            "is sticky for the rest of that historical day, as live), the trades-per-day cap, "
            "the one-position cap and the entry window all applied. Nothing written here can "
            "reach data/indbot.db or affect paper or live statistics.",
        ]

        if learn:
            out.append(
                "WALK-FORWARD LEARNING WAS ON: the learner gated buckets using only the trades "
                "this backtest had already closed at that point, so the headline numbers "
                "include its effect. It starts from zero knowledge, so early trades are "
                "ungated."
            )
        else:
            out.append(
                "The learner was NOT applied: these are the RAW strategy's numbers, with no "
                "bucket promotion, demotion or blocking. bucket_stats is reported so the "
                "learner can be seeded from this run; re-run with learn=True for a genuine "
                "walk-forward replay."
            )

        if skipped:
            out.append(
                f"{skipped} day(s) in this window produced NOTHING and were skipped — see "
                "skipped_days for the reason on each. A missing day is not a flat day, and the "
                "equity curve marks them."
            )
        if totals.premium_gaps:
            out.append(
                f"On {totals.premium_gaps} bar(s) an open position's contract printed no "
                "premium. No price was invented: the price-based exits were skipped on those "
                "bars and only flat_by and the time stop could fire, so an exit may be later "
                "and at a worse price than it would have been live."
            )
        if totals.forced_data_end_exits:
            out.append(
                f"{totals.forced_data_end_exits} position(s) were force-closed at the last "
                "printed premium because that day's candles stopped before flat_by. Those exits "
                "carry the reason \"data_end\" and are a data artefact, not a strategy decision."
            )
        if blocks.get("day_halt"):
            out.append(
                f"The daily loss halt fired on {blocks['day_halt']} occasion(s) and stopped "
                "further entries for the rest of that historical day, exactly as it would live."
            )
        if trades and trades < 20:
            out.append(
                f"Only {trades} trade(s) closed. That is far too small a sample to conclude "
                "anything: the win rate, the expectancy and every bucket statistic below are "
                "noise at this count."
            )
        if errors:
            out.append(
                f"{len(errors)} error(s) occurred during the run (see `errors`). Any journal "
                "write that failed means the day-halt and the trade cap were evaluated on "
                "incomplete numbers for the rest of that day."
            )
        out.extend(notes)
        return out


__all__ = ["BacktestResult", "BacktestTrade", "Backtester"]
