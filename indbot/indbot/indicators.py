"""Hand-rolled technical indicators — pure functions over lists, no state, no numpy.

Every series function returns a list the **same length as its input**, left-padded with
`None` where the indicator is not yet defined. Callers index by bar (`vals[-1]` is the
last closed candle), so a short list would be a silent off-by-n and a wrong trade.

Smoothing conventions, because they change the numbers a chart shows:

* `rsi`, `atr` and `adx` use **Wilder's** smoothing (alpha = 1/n), seeded with the first-n
  simple average — exactly as Wilder defined it, and as TradingView's `ta.rma` does it.
  A plain-SMA RSI is a different indicator and will not match the user's chart.
* `ema` uses alpha = 2/(n+1), seeded with the first-n simple average (TradingView `ta.ema`).
* `bollinger` uses the **population** standard deviation (divide by n), as charts do.
* `vwap` is **session-anchored**: it resets at every new IST trading day in the series.

| function        | input         | first defined at | leading `None`s | notes |
|-----------------|---------------|------------------|-----------------|-------|
| `sma(v, n)`     | `list[float]` | index `n-1`      | `n-1`           | exact window sum, not a rolling one |
| `ema(v, n)`     | `list[float]` | index `n-1`      | `n-1`           | SMA seed, then alpha = 2/(n+1) |
| `rsi(v, n)`     | `list[float]` | index `n`        | `n`             | needs `n+1` prices for `n` changes |
| `atr(c, n)`     | `list[Candle]`| index `n-1`      | `n-1`           | TR[0] = h-l (no prior close) |
| `vwap(c)`       | `list[Candle]`| first bar of each IST day with volume | 0 (or 1 per zero-volume bar) | resets daily |
| `supertrend(c, n, mult)` | `list[Candle]` | index `n-1` | `n-1`      | first defined bar is a SEED, never a flip |
| `adx(c, n)`     | `list[Candle]`| index `2n-1`     | `2n-1`          | DI seeded over bars 1..n, ADX over DX[n..2n-1] |
| `bollinger(v, n, k)` | `list[float]` | index `n-1` | `n-1`           | `(lower, mid, upper)` |
| `opening_range(c, minutes)` | `list[Candle]` | n/a | n/a            | scalar `(high, low)` or `None` |
| `body_ratio(c)` | `Candle`      | n/a              | n/a             | scalar `0.0..1.0` |
| `regime(c)`     | `list[Candle]`| needs 50 bars    | n/a             | scalar; `"range"` until warmed up |

Failure policy: every function guards every division and every empty/short input, and
returns all-`None` (or the neutral scalar) rather than raising. A malformed candle — a
missing OHLCV key, a non-numeric, a NaN — is logged at WARNING and treated the same way.
This module never raises, because an indicator that throws mid-tick would strand an open
position; the engine reads `None` as "no opinion" and does not trade.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import TYPE_CHECKING, Any

from . import clock

if TYPE_CHECKING:  # type-only; keeps this module's runtime deps to stdlib + clock
    from .types import Candle

logger = logging.getLogger(__name__)

__all__ = [
    "sma",
    "ema",
    "rsi",
    "atr",
    "vwap",
    "supertrend",
    "adx",
    "bollinger",
    "opening_range",
    "body_ratio",
    "regime",
    "REGIMES",
    "ADX_TREND_THRESHOLD",
    "EMA_FAST",
    "EMA_MID",
    "EMA_SLOW",
]

#: The three values `regime()` may return. Also a dimension of `Signal.make_bucket`.
REGIMES: tuple[str, str, str] = ("trend_up", "trend_down", "range")

#: Wilder's own threshold: below 20 the market is not trending, whatever the EMAs say.
ADX_TREND_THRESHOLD: float = 20.0

EMA_FAST: int = 9
EMA_MID: int = 21
EMA_SLOW: int = 50


# ---------------------------------------------------------------------------
# input hygiene
# ---------------------------------------------------------------------------


def _floats(values: list[float]) -> list[float] | None:
    """Coerce a price series to finite floats, or None if any element is unusable.

    Returning None (and logging) rather than raising keeps the "never raise" promise:
    the caller gets an all-None series and simply does not trade that bar.
    """
    out: list[float] = []
    for i, v in enumerate(values):
        try:
            f = float(v)
        except (TypeError, ValueError):
            logger.warning("value %d is not numeric (%r); indicator returns all-None", i, v)
            return None
        if not math.isfinite(f):
            logger.warning("value %d is not finite (%r); indicator returns all-None", i, v)
            return None
        out.append(f)
    return out


def _col(candles: list["Candle"], key: str) -> list[float] | None:
    """Extract one OHLCV column as finite floats, or None if any candle is malformed."""
    out: list[float] = []
    for i, c in enumerate(candles):
        try:
            f = float(c[key])  # type: ignore[literal-required]
        except (KeyError, IndexError, TypeError, ValueError):
            logger.warning(
                "candle %d has a missing or non-numeric %r; indicator returns all-None", i, key
            )
            return None
        if not math.isfinite(f):
            logger.warning("candle %d has a non-finite %r (%r); indicator returns all-None", i, key, f)
            return None
        out.append(f)
    return out


def _hlc(candles: list["Candle"]) -> tuple[list[float], list[float], list[float]] | None:
    """High, low and close columns together, or None if any of them is unusable."""
    h = _col(candles, "h")
    low = _col(candles, "l")
    c = _col(candles, "c")
    if h is None or low is None or c is None:
        return None
    return h, low, c


def _true_ranges(h: list[float], low: list[float], c: list[float]) -> list[float]:
    """Wilder's true range per bar; bar 0 falls back to high-low (it has no prior close)."""
    n = len(h)
    tr: list[float] = [0.0] * n
    if n == 0:
        return tr
    tr[0] = max(0.0, h[0] - low[0])
    for i in range(1, n):
        prev_close = c[i - 1]
        tr[i] = max(h[i] - low[i], abs(h[i] - prev_close), abs(low[i] - prev_close))
    return tr


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------


def sma(values: list[float], n: int) -> list[float | None]:
    """Simple moving average of the last `n` values, left-padded with None.

    Each window is summed exactly with `math.fsum` instead of carried as a rolling sum:
    a rolling sum drifts over thousands of bars and correctness beats speed here.
    """
    length = len(values)
    out: list[float | None] = [None] * length
    if n <= 0 or length < n:
        return out
    vals = _floats(values)
    if vals is None:
        return out
    for i in range(n - 1, length):
        out[i] = math.fsum(vals[i - n + 1 : i + 1]) / n
    return out


def ema(values: list[float], n: int) -> list[float | None]:
    """Exponential moving average, alpha = 2/(n+1), seeded with the first-n simple average.

    The SMA seed is what TradingView's `ta.ema` uses; seeding with values[0] instead would
    leave the first few dozen bars visibly off the user's chart.
    """
    length = len(values)
    out: list[float | None] = [None] * length
    if n <= 0 or length < n:
        return out
    vals = _floats(values)
    if vals is None:
        return out
    alpha = 2.0 / (n + 1.0)  # n >= 1 here, so the denominator can never be zero
    prev = math.fsum(vals[:n]) / n
    out[n - 1] = prev
    for i in range(n, length):
        prev = (vals[i] - prev) * alpha + prev
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    """RSI from Wilder's smoothed average gain and loss, with both edge cases named."""
    if avg_loss <= 0.0:
        # No losses at all. If there were no gains either the series is flat, which is a
        # dead market, not a screaming overbought — report the neutral 50 so an RSI
        # setup cannot fire on a stale feed of identical prints.
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi(values: list[float], n: int = 14) -> list[float | None]:
    """Wilder's RSI over `n` periods; first value at index `n`, left-padded with None.

    Seeded with the simple average of the first `n` changes, then smoothed at alpha = 1/n.
    """
    length = len(values)
    out: list[float | None] = [None] * length
    if n <= 0 or length < n + 1:  # n changes need n+1 prices
        return out
    vals = _floats(values)
    if vals is None:
        return out

    # gains[j] / losses[j] describe the move INTO values[j + 1].
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, length):
        change = vals[i] - vals[i - 1]
        gains.append(change if change > 0.0 else 0.0)
        losses.append(-change if change < 0.0 else 0.0)

    avg_gain = math.fsum(gains[:n]) / n
    avg_loss = math.fsum(losses[:n]) / n
    out[n] = _rsi_value(avg_gain, avg_loss)
    for i in range(n + 1, length):
        avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


def atr(candles: list["Candle"], n: int = 14) -> list[float | None]:
    """Wilder's Average True Range; first value at index `n-1`, left-padded with None.

    Bar 0's true range is high-low because it has no prior close, matching TradingView's
    `ta.atr`. `adx` deliberately starts its own true range at bar 1 instead — see there.
    """
    length = len(candles)
    out: list[float | None] = [None] * length
    if n <= 0 or length < n:
        return out
    cols = _hlc(candles)
    if cols is None:
        return out
    h, low, c = cols
    tr = _true_ranges(h, low, c)
    prev = math.fsum(tr[:n]) / n
    out[n - 1] = prev
    for i in range(n, length):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def bollinger(
    values: list[float], n: int = 20, k: float = 2.0
) -> list[tuple[float, float, float] | None]:
    """Bollinger bands as `(lower, mid, upper)`; mid is SMA(n), width is k population sigma."""
    length = len(values)
    out: list[tuple[float, float, float] | None] = [None] * length
    if n <= 0 or length < n:
        return out
    vals = _floats(values)
    if vals is None:
        return out
    for i in range(n - 1, length):
        window = vals[i - n + 1 : i + 1]
        mean = math.fsum(window) / n
        variance = math.fsum((x - mean) ** 2 for x in window) / n  # population, as charts use
        sigma = math.sqrt(variance) if variance > 0.0 else 0.0
        out[i] = (mean - k * sigma, mean, mean + k * sigma)
    return out


# ---------------------------------------------------------------------------
# session-anchored volume
# ---------------------------------------------------------------------------


def vwap(candles: list["Candle"]) -> list[float | None]:
    """Session-anchored VWAP of the typical price, reset at every new IST trading day.

    Typical price is (h+l+c)/3 weighted by volume, accumulated from the session open.
    The reset is the whole point: a VWAP carried across days is an average of yesterday's
    prices and is meaningless as an intraday reference, so the day boundary is detected
    from `clock.to_ist(ts).date()` rather than assumed from the list's start.

    A bar with no cumulative volume yet (a zero-volume open) yields None, not a bare
    typical price — an unweighted "VWAP" would be a different line than the chart's.
    """
    length = len(candles)
    out: list[float | None] = [None] * length
    if length == 0:
        return out
    cols = _hlc(candles)
    ts = _col(candles, "ts")
    vols = _col(candles, "v")
    if cols is None or ts is None or vols is None:
        return out
    h, low, c = cols

    cum_pv = 0.0
    cum_v = 0.0
    current_day: date | None = None
    for i in range(length):
        day = clock.to_ist(ts[i]).date()
        if day != current_day:
            current_day = day
            cum_pv = 0.0
            cum_v = 0.0
        typical = (h[i] + low[i] + c[i]) / 3.0
        # A negative volume is nonsense from a bad print; treat it as zero rather than
        # letting it unwind the day's accumulator and move VWAP the wrong way.
        volume = vols[i] if vols[i] > 0.0 else 0.0
        cum_pv += typical * volume
        cum_v += volume
        out[i] = (cum_pv / cum_v) if cum_v > 0.0 else None
    return out


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


def supertrend(
    candles: list["Candle"], n: int = 10, mult: float = 2.0
) -> list[tuple[float, int] | None]:
    """ATR Supertrend as `(level, direction)`, direction +1 up / -1 down, padded with None.

    The real algorithm, not a band comparison:

    1. basic bands = hl2 +/- mult * ATR(n);
    2. the **final-band carry-forward** rule — while the trend holds, the band only ever
       ratchets in the favourable direction (the upper band may fall, the lower band may
       rise), and it only resets when the previous close broke through it. Without this
       carry-forward the level jitters with ATR and the direction flips on almost every
       bar, firing a trade each time;
    3. the flip — an uptrend ends when the close drops under the carried lower band, a
       downtrend ends when the close rises above the carried upper band.

    `level` is the active stop band: the lower band in an uptrend, the upper in a downtrend.

    The first defined bar (index `n-1`) is a SEED, taken from the close's position within
    its own bar rather than from a flip. A caller that detects a trade on `direction[i-1]
    != direction[i]` must therefore ignore `i == n-1`; there is no prior direction to flip
    from, and treating the seed as a flip would open a position on warm-up noise.
    """
    length = len(candles)
    out: list[tuple[float, int] | None] = [None] * length
    if n <= 0 or length < n:
        return out
    cols = _hlc(candles)
    if cols is None:
        return out
    h, low, c = cols
    atr_vals = atr(candles, n)

    prev_upper: float | None = None
    prev_lower: float | None = None
    prev_dir = 0
    for i in range(length):
        a = atr_vals[i]
        if a is None:
            continue
        hl2 = (h[i] + low[i]) / 2.0
        basic_upper = hl2 + mult * a
        basic_lower = hl2 - mult * a

        if prev_upper is None or prev_lower is None or prev_dir == 0:
            # Seed bar: no prior band to carry and no prior direction to flip from.
            final_upper, final_lower = basic_upper, basic_lower
            direction = 1 if c[i] >= hl2 else -1
        else:
            prev_close = c[i - 1]
            # Carry forward unless the band tightened or the previous close broke it.
            final_upper = basic_upper if (basic_upper < prev_upper or prev_close > prev_upper) else prev_upper
            final_lower = basic_lower if (basic_lower > prev_lower or prev_close < prev_lower) else prev_lower
            if prev_dir == 1:
                direction = -1 if c[i] < final_lower else 1
            else:
                direction = 1 if c[i] > final_upper else -1

        out[i] = (final_lower if direction == 1 else final_upper, direction)
        prev_upper, prev_lower, prev_dir = final_upper, final_lower, direction
    return out


def _dx_value(plus_sm: float, minus_sm: float, tr_sm: float) -> float:
    """Wilder's DX from smoothed +DM, -DM and TR, with both divisions guarded."""
    if tr_sm <= 0.0:
        return 0.0  # no range at all => no measurable directional movement
    plus_di = 100.0 * plus_sm / tr_sm
    minus_di = 100.0 * minus_sm / tr_sm
    total = plus_di + minus_di
    if total <= 0.0:
        return 0.0  # an inside-bar run: no directional movement either way
    return 100.0 * abs(plus_di - minus_di) / total


def adx(candles: list["Candle"], n: int = 14) -> list[float | None]:
    """Wilder's ADX; first value at index `2n-1` (28 bars at n=14), left-padded with None.

    Two Wilder passes, as Wilder specified: +DM/-DM/TR are smoothed over bars 1..n to give
    the first DI pair and DX at index `n`, then DX itself is smoothed over `n` periods, so
    the first ADX lands at index `2n-1`.

    The true range here starts at bar 1, not bar 0 as in `atr`: +DM and -DM need a previous
    bar too, and seeding TR over one more bar than DM would bias +DI/-DI downward.
    """
    length = len(candles)
    out: list[float | None] = [None] * length
    if n <= 0 or length < 2 * n:
        return out
    cols = _hlc(candles)
    if cols is None:
        return out
    h, low, c = cols

    plus_dm: list[float] = [0.0] * length
    minus_dm: list[float] = [0.0] * length
    tr: list[float] = [0.0] * length
    for i in range(1, length):
        up_move = h[i] - h[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        prev_close = c[i - 1]
        tr[i] = max(h[i] - low[i], abs(h[i] - prev_close), abs(low[i] - prev_close))

    # Pass one: Wilder averages seeded over bars 1..n.
    tr_sm = math.fsum(tr[1 : n + 1]) / n
    plus_sm = math.fsum(plus_dm[1 : n + 1]) / n
    minus_sm = math.fsum(minus_dm[1 : n + 1]) / n
    dx: list[float] = [0.0] * length
    dx[n] = _dx_value(plus_sm, minus_sm, tr_sm)
    for i in range(n + 1, length):
        tr_sm = (tr_sm * (n - 1) + tr[i]) / n
        plus_sm = (plus_sm * (n - 1) + plus_dm[i]) / n
        minus_sm = (minus_sm * (n - 1) + minus_dm[i]) / n
        dx[i] = _dx_value(plus_sm, minus_sm, tr_sm)

    # Pass two: smooth DX. Seed over DX[n .. 2n-1], which the length guard guarantees.
    first = 2 * n - 1
    prev = math.fsum(dx[n : 2 * n]) / n
    out[first] = prev
    for i in range(first + 1, length):
        prev = (prev * (n - 1) + dx[i]) / n
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# session structure
# ---------------------------------------------------------------------------


def opening_range(
    candles: list["Candle"], minutes: int = 15, *, session_date: date | None = None
) -> tuple[float, float] | None:
    """(high, low) of the first `minutes` of the SESSION, or None when that window is absent.

    The window is `[clock.session_open(d), + minutes)` for the session date `d`, NOT the
    first `minutes` worth of candles in the list — the list routinely starts mid-session or
    spans several days, and anchoring to its first element would silently measure the wrong
    range and break every ORB trade.

    `d` defaults to the IST date of the LAST candle, i.e. the session currently being
    traded. Pass `session_date` to measure an earlier day in a multi-day series.

    A candle belongs to the range when its OPEN time `ts` falls in the window (candles are
    half-open `[ts, ts+interval)`), so `minutes` should be a whole multiple of the candle
    interval. A candle that straddles the window's end is included whole, which can only
    widen the range and so can only make an ORB trigger later — the safe direction.

    Returns None when the list is empty, malformed, or holds no candle inside the window.
    A None here means ORB simply does not fire; it never means "use a guessed range".
    """
    if not candles or minutes <= 0:
        return None
    ts = _col(candles, "ts")
    h = _col(candles, "h")
    low = _col(candles, "l")
    if ts is None or h is None or low is None:
        return None

    day = session_date or clock.to_ist(ts[-1]).date()
    start_s = clock.epoch_s(clock.session_open(day))
    end_s = start_s + minutes * 60

    hi: float | None = None
    lo: float | None = None
    for i in range(len(ts)):
        if start_s <= ts[i] < end_s:
            hi = h[i] if hi is None else max(hi, h[i])
            lo = low[i] if lo is None else min(lo, low[i])
    if hi is None or lo is None:
        return None
    return (hi, lo)


def body_ratio(c: "Candle") -> float:
    """|close-open| / (high-low) for one candle: 1.0 is a marubozu, 0.0 a doji.

    Returns 0.0 for a zero-range or malformed candle. 0.0 is the safe branch: every setup
    uses this as a *confirmation*, so a candle we cannot measure adds no confidence and
    cannot talk the bot into a trade.
    """
    try:
        o = float(c["o"])  # type: ignore[literal-required]
        high = float(c["h"])  # type: ignore[literal-required]
        low = float(c["l"])  # type: ignore[literal-required]
        close = float(c["c"])  # type: ignore[literal-required]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("body_ratio got a malformed candle (%r); returning 0.0", c)
        return 0.0
    if not all(math.isfinite(x) for x in (o, high, low, close)):
        logger.warning("body_ratio got a non-finite candle (%r); returning 0.0", c)
        return 0.0
    bar_range = high - low
    if bar_range <= 0.0:
        return 0.0
    # Clamp: bad data where |close-open| exceeds the bar's range must not score above 1.
    return min(1.0, abs(close - o) / bar_range)


def regime(
    candles: list["Candle"], *, adx_n: int = 14, adx_threshold: float = ADX_TREND_THRESHOLD
) -> str:
    """Classify the last bar as "trend_up", "trend_down" or "range".

    Thresholds, measured on the LAST candle in the list:

    * `ADX(14) >= 20` is required for any trend verdict. Below 20 the market is not
       trending by Wilder's own definition, whatever the EMAs happen to be doing, and the
       verdict is "range".
    * With ADX at or above the threshold, the EMA stack decides the side:
       `EMA9 > EMA21 > EMA50` is "trend_up", `EMA9 < EMA21 < EMA50` is "trend_down".
    * A mixed stack (a pullback inside a trend, or a crossover in progress) is "range".

    Returns "range" when the series is too short to warm up EMA50 (50 bars) or ADX
    (2*adx_n bars), when a candle is malformed, or when anything is undefined. "range" is
    the deliberate safe default: it is the regime for which the strategy widens its stops
    and the learner's neutral bucket, so an unwarmed series cannot be mistaken for
    momentum. It is also one of the three values `Signal.make_bucket` accepts.
    """
    if not candles:
        return "range"
    closes = _col(candles, "c")
    if closes is None:
        return "range"
    if len(closes) < EMA_SLOW or len(closes) < 2 * max(adx_n, 1):
        return "range"

    fast = ema(closes, EMA_FAST)[-1]
    mid = ema(closes, EMA_MID)[-1]
    slow = ema(closes, EMA_SLOW)[-1]
    strength = adx(candles, adx_n)[-1]
    if fast is None or mid is None or slow is None or strength is None:
        return "range"
    if strength < adx_threshold:
        return "range"
    if fast > mid > slow:
        return "trend_up"
    if fast < mid < slow:
        return "trend_down"
    return "range"
