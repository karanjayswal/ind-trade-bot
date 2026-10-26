"""Offline unit tests for `indbot.indicators`.

Every expected value in this file is **hand-computed and shown in a comment next to the
assertion**. Nothing here was produced by running `indicators.py` and recording its
output: a test that asserts whatever the code does cannot catch the code being wrong.
Where the arithmetic is long (RSI, ADX) the derivation is written out step by step so a
future reader can re-check it with a calculator.

No network, no clock dependency: none of the functions under test call `clock.now()`, and
every timestamp here is built from an explicit IST date, so these tests pass in any local
timezone and with or without `INDBOT_FAKE_NOW` set.

Run with any of::

    python3 tests/test_indicators.py
    python3 -m unittest discover -s tests -t tests
    python3 -m pytest tests/test_indicators.py

Stdlib `unittest` is used deliberately — `requirements.txt` carries no test dependency —
and the repo root is put on `sys.path` below so no `tests/__init__.py` is needed.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Make `import indbot` work when this file is run directly or discovered from tests/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indbot import indicators  # noqa: E402
from indbot.indicators import (  # noqa: E402
    adx,
    atr,
    body_ratio,
    bollinger,
    ema,
    opening_range,
    regime,
    rsi,
    sma,
    supertrend,
    vwap,
)

# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

#: Two consecutive trading days, neither an exchange holiday.
#: 2026-01-05 is a Monday, 2026-01-06 a Tuesday (verified in test_fixture_dates_are_sane).
DAY_A = date(2026, 1, 5)
DAY_B = date(2026, 1, 6)


def ts_at(d: date, hh: int, mm: int) -> int:
    """Epoch SECONDS for `hh:mm` IST on `d` — the unit of a Candle's `ts`."""
    return int(datetime.combine(d, time(hh, mm), tzinfo=IST).timestamp())


def candle(
    *, ts: int = 0, o: float = 0.0, h: float = 0.0, l: float = 0.0, c: float = 0.0, v: int = 1000
) -> dict[str, Any]:
    """Build one Candle dict. Keyword-only so a transposed h/l cannot slip through."""
    return {"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v}


def hlc_bars(rows: Sequence[tuple[float, float, float]], *, start_ts: int = 0) -> list[dict]:
    """Candles from (high, low, close) triples; `o` mirrors the close, volume is constant.

    Most indicators here read only h/l/c, so `o` is set to the close rather than left at
    0.0 — a zero open would make `body_ratio` meaningless on these same fixtures.
    """
    return [
        candle(ts=start_ts + i * 300, o=row[2], h=row[0], l=row[1], c=row[2], v=1000)
        for i, row in enumerate(rows)
    ]


def closes_only(values: Sequence[float], *, start_ts: int = 0) -> list[dict]:
    """Candles whose h == l == c == the given value: a zero-range series."""
    return [
        candle(ts=start_ts + i * 300, o=v, h=v, l=v, c=v, v=1000) for i, v in enumerate(values)
    ]


def walk(nbars: int, *, start_ts: int = 0) -> list[dict]:
    """A deterministic non-degenerate OHLCV series, for the padding/length contract tests.

    Deterministic (no RNG) so a failure is always reproducible, and deliberately never
    flat — a flat series would make several indicators legitimately return 0.0 and hide a
    padding bug behind a plausible-looking number.
    """
    bars: list[dict] = []
    for i in range(nbars):
        close = 100.0 + 5.0 * math.sin(i / 3.0) + 0.1 * i
        high = close + 1.0 + 0.5 * abs(math.cos(i / 2.0))
        low = close - 1.0 - 0.5 * abs(math.sin(i / 2.0))
        bars.append(
            candle(
                ts=start_ts + i * 300,
                o=close - 0.2,
                h=high,
                l=low,
                c=close,
                v=1000 + 7 * i,
            )
        )
    return bars


def up_staircase(nbars: int, *, start_ts: int = 0) -> list[dict]:
    """Bar i = (h=101+i, l=99+i, c=100+i): +DM == 1, -DM == 0, TR == 2 on every bar.

    Derivation (used by the ADX and regime tests):
      up_move   = h[i] - h[i-1] = 1
      down_move = l[i-1] - l[i] = -1          -> +DM = 1, -DM = 0
      TR        = max(h-l = 2, |h[i]-c[i-1]| = |101+i - (99+i)| = 2,
                      |l[i]-c[i-1]| = |99+i - (99+i)| = 0) = 2
    """
    return hlc_bars([(101.0 + i, 99.0 + i, 100.0 + i) for i in range(nbars)], start_ts=start_ts)


def down_staircase(nbars: int, *, start_ts: int = 0) -> list[dict]:
    """The mirror of `up_staircase`: +DM == 0, -DM == 1, TR == 2 on every bar."""
    return hlc_bars([(101.0 - i, 99.0 - i, 100.0 - i) for i in range(nbars)], start_ts=start_ts)


def zigzag(nbars: int, *, start_ts: int = 0) -> list[dict]:
    """Alternating bars with equal up and down movement — a market going nowhere.

    even i: (h=102, l=100, c=101); odd i: (h=103, l=101, c=102).
      odd  i: up=+1, down=-1 -> +DM=1, -DM=0, TR = max(2, |103-101|, 0) = 2
      even i: up=-1, down=+1 -> +DM=0, -DM=1, TR = max(2, 0, |100-102|) = 2
    So +DM and -DM arrive in equal numbers and TR is constant at 2.
    """
    rows = [((103.0, 101.0, 102.0) if i % 2 else (102.0, 100.0, 101.0)) for i in range(nbars)]
    return hlc_bars(rows, start_ts=start_ts)


class SeriesAssertions(unittest.TestCase):
    """Comparison helpers that treat `None` as a value, not as a missing one."""

    PLACES = 9

    def assertSeries(
        self,
        actual: Sequence[Any],
        expected: Sequence[Any],
        *,
        places: int | None = None,
        msg: str = "",
    ) -> None:
        """Assert two padded series match element-wise, including where each is None."""
        places = self.PLACES if places is None else places
        self.assertEqual(
            len(actual), len(expected), f"{msg}: length {len(actual)} != {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            if e is None:
                self.assertIsNone(a, f"{msg}: index {i} should be None, got {a!r}")
                continue
            self.assertIsNotNone(a, f"{msg}: index {i} should be {e!r}, got None")
            if isinstance(e, tuple):
                self.assertEqual(
                    len(a), len(e), f"{msg}: index {i} tuple arity {len(a)} != {len(e)}"
                )
                for j, (aj, ej) in enumerate(zip(a, e)):
                    if isinstance(ej, int) and not isinstance(ej, bool):
                        self.assertEqual(aj, ej, f"{msg}: index {i} field {j}")
                    else:
                        self.assertAlmostEqual(aj, ej, places=places, msg=f"{msg}: [{i}][{j}]")
            else:
                self.assertAlmostEqual(a, e, places=places, msg=f"{msg}: index {i}")

    def assertPadding(
        self,
        out: Sequence[Any],
        *,
        length: int,
        first_defined: int,
        msg: str = "",
    ) -> None:
        """Assert the length contract: same length in, None before `first_defined`, set after."""
        self.assertEqual(len(out), length, f"{msg}: must return len(input)")
        for i in range(min(first_defined, length)):
            self.assertIsNone(out[i], f"{msg}: index {i} must be None (first={first_defined})")
        for i in range(max(first_defined, 0), length):
            self.assertIsNotNone(out[i], f"{msg}: index {i} must be defined (first={first_defined})")


# ---------------------------------------------------------------------------
# fixture sanity
# ---------------------------------------------------------------------------


class TestFixtures(SeriesAssertions):
    """The fixtures themselves, so a wrong fixture is not mistaken for a wrong indicator."""

    def test_fixture_dates_are_sane(self) -> None:
        """DAY_A/DAY_B are consecutive weekdays and not exchange holidays."""
        from indbot import clock

        self.assertEqual(DAY_A.weekday(), 0, "DAY_A should be a Monday")
        self.assertEqual(DAY_B.weekday(), 1, "DAY_B should be a Tuesday")
        self.assertEqual((DAY_B - DAY_A).days, 1)
        self.assertTrue(clock.is_trading_day(DAY_A))
        self.assertTrue(clock.is_trading_day(DAY_B))

    def test_ts_at_matches_clock(self) -> None:
        """This file's timestamp helper agrees with `clock.session_open`, independently built."""
        from indbot import clock

        self.assertEqual(ts_at(DAY_A, 9, 15), clock.epoch_s(clock.session_open(DAY_A)))
        # 1767584700 = 2026-01-05 09:15:00+05:30, computed from the UTC epoch by hand.
        self.assertEqual(ts_at(DAY_A, 9, 15), 1767584700)
        self.assertEqual(ts_at(DAY_B, 9, 15), 1767584700 + 86400)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSma(SeriesAssertions):
    """Simple moving average: exact window sums, padded with n-1 Nones."""

    def test_sma_hand_computed(self) -> None:
        """sma([1..5], 3): windows (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4."""
        self.assertSeries(sma([1, 2, 3, 4, 5], 3), [None, None, 2.0, 3.0, 4.0])

    def test_sma_first_value_is_at_index_n_minus_1(self) -> None:
        """The seed lands at n-1, not at 0 and not at n."""
        out = sma([10.0, 20.0, 30.0, 40.0], 2)
        # (10+20)/2 = 15, (20+30)/2 = 25, (30+40)/2 = 35
        self.assertSeries(out, [None, 15.0, 25.0, 35.0])

    def test_sma_n_equal_to_length(self) -> None:
        """A window exactly as long as the series yields one value, at the last index."""
        # (2+4+6+8)/4 = 5.0
        self.assertSeries(sma([2.0, 4.0, 6.0, 8.0], 4), [None, None, None, 5.0])

    def test_sma_too_short_and_non_positive_n(self) -> None:
        """Too-short input and n <= 0 both give an all-None series of the input's length."""
        self.assertSeries(sma([1.0, 2.0], 3), [None, None])
        self.assertSeries(sma([1.0, 2.0, 3.0], 0), [None, None, None])
        self.assertSeries(sma([1.0, 2.0, 3.0], -4), [None, None, None])
        self.assertSeries(sma([], 3), [])

    def test_sma_all_equal(self) -> None:
        """A flat series averages to its own level at every defined index."""
        self.assertSeries(sma([7.5] * 5, 3), [None, None, 7.5, 7.5, 7.5])

    def test_sma_single_value_n1(self) -> None:
        """sma(v, 1) is v itself."""
        self.assertSeries(sma([42.0], 1), [42.0])


# ---------------------------------------------------------------------------
# EMA — seeding is the thing worth pinning
# ---------------------------------------------------------------------------


class TestEma(SeriesAssertions):
    """EMA seeding (SMA of the first n) and the alpha = 2/(n+1) recursion."""

    def test_ema_seed_is_the_sma_of_the_first_n(self) -> None:
        """The value at index n-1 is the simple average of the first n, not values[0] or values[-1].

        ema([5, 10, 15], 3) seeds at (5+10+15)/3 = 10.0. A values[0] seed would give
        5 -> 7.5 -> 11.25 (i.e. 11.25 at index 2) and a "last value" seed would give 15.
        """
        out = ema([5.0, 10.0, 15.0], 3)
        self.assertSeries(out, [None, None, 10.0])
        self.assertNotAlmostEqual(out[2], 11.25, places=6, msg="values[0] seed leaked through")
        self.assertNotAlmostEqual(out[2], 15.0, places=6, msg="last-value seed leaked through")

    def test_ema_hand_computed_alpha_half(self) -> None:
        """ema([1..8], 3): alpha = 2/4 = 0.5, seed = (1+2+3)/3 = 2.

        i=3: 2 + 0.5*(4-2) = 3      i=4: 3 + 0.5*(5-3) = 4
        i=5: 4 + 0.5*(6-4) = 5      i=6: 5 + 0.5*(7-5) = 6
        i=7: 6 + 0.5*(8-6) = 7
        """
        self.assertSeries(
            ema([1, 2, 3, 4, 5, 6, 7, 8], 3),
            [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )

    def test_ema_hand_computed_alpha_point_four(self) -> None:
        """ema([10..14], 4): alpha = 2/5 = 0.4, seed = (10+11+12+13)/4 = 11.5.

        i=4: 11.5 + 0.4*(14 - 11.5) = 11.5 + 1.0 = 12.5
        """
        self.assertSeries(ema([10, 11, 12, 13, 14], 4), [None, None, None, 11.5, 12.5])

    def test_ema_n1_is_the_identity(self) -> None:
        """alpha = 2/2 = 1, so ema(v, 1) reproduces v exactly."""
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        self.assertSeries(ema(values, 1), values)

    def test_ema_of_a_unit_ramp_lags_by_exactly_half_the_period(self) -> None:
        """On an arithmetic ramp of step 1 the EMA settles at `last - (n-1)/2`, exactly.

        Derivation: the seed at index n-1 is the mean of the first n, which is
        v[n-1] - (n-1)/2. If EMA[i-1] = v[i-1] - (n-1)/2 then
            v[i] - EMA[i-1] = 1 + (n-1)/2 = (n+1)/2,
            EMA[i] = EMA[i-1] + (2/(n+1)) * (n+1)/2 = EMA[i-1] + 1 = v[i] - (n-1)/2.
        So the invariant holds for every bar after the seed. This is a property of the
        seeding AND the alpha together — get either wrong and it breaks.
        """
        values = [100.0 + i for i in range(60)]  # last = 159.0
        for n, expected in ((9, 159.0 - 4.0), (21, 159.0 - 10.0), (50, 159.0 - 24.5)):
            with self.subTest(n=n):
                self.assertAlmostEqual(ema(values, n)[-1], expected, places=9)
        # Spelled out: ema9 = 155.0, ema21 = 149.0, ema50 = 134.5 -> strictly stacked.
        self.assertAlmostEqual(ema(values, 9)[-1], 155.0, places=9)
        self.assertAlmostEqual(ema(values, 21)[-1], 149.0, places=9)
        self.assertAlmostEqual(ema(values, 50)[-1], 134.5, places=9)

    def test_ema_degenerate(self) -> None:
        """Empty, too-short, n <= 0 and all-equal inputs."""
        self.assertSeries(ema([], 9), [])
        self.assertSeries(ema([1.0], 9), [None])
        self.assertSeries(ema([1.0, 2.0, 3.0], 0), [None, None, None])
        self.assertSeries(ema([4.0] * 5, 3), [None, None, 4.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# RSI — Wilder, against a hand-computed 14-period series
# ---------------------------------------------------------------------------

#: 17 closes whose 16 changes are +2, -1, +2, -1, ... — chosen so the seed window holds
#: exactly seven +2 gains and seven -1 losses, which makes the Wilder arithmetic exact.
RSI_SERIES: list[float] = [
    100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0,
    106.0, 105.0, 107.0, 106.0, 108.0, 107.0, 109.0, 108.0,
]


class TestRsi(SeriesAssertions):
    """Wilder's RSI: SMA-seeded average gain/loss, then alpha = 1/n."""

    def test_rsi_14_hand_computed(self) -> None:
        """RSI(14) over RSI_SERIES, worked out in exact fractions.

        Changes (16 of them): +2, -1 repeating.
        Seed window = the first 14 changes = seven +2 and seven -1, so
            avg_gain = 14/14 = 1.0,  avg_loss = 7/14 = 0.5
            RS = 1.0/0.5 = 2        RSI[14] = 100 - 100/3 = 200/3 = 66.666666...

        Bar 15 (change into values[15] = 109 is +2):
            avg_gain = (1.0*13 + 2)/14 = 15/14
            avg_loss = (0.5*13 + 0)/14 = 6.5/14
            RS = 15/6.5 = 30/13     RSI[15] = 100 - 100*6.5/21.5 = 3000/43 = 69.767441...

        Bar 16 (change into values[16] = 108 is -1):
            avg_gain = (15/14*13 + 0)/14 = 195/196
            avg_loss = (6.5/14*13 + 1)/14 = 98.5/196
            RS = 195/98.5 = 390/197
            RSI[16] = 100 - 100*197/587 = 39000/587 = 66.439522...
        """
        out = rsi(RSI_SERIES, 14)
        expected: list[float | None] = [None] * 14 + [
            200.0 / 3.0,       # 66.66666666666667
            3000.0 / 43.0,     # 69.76744186046511
            39000.0 / 587.0,   # 66.43952299829643
        ]
        self.assertSeries(out, expected)
        # Restated as decimals so a reader can check with a calculator.
        self.assertAlmostEqual(out[14], 66.666666667, places=8)
        self.assertAlmostEqual(out[15], 69.767441860, places=8)
        self.assertAlmostEqual(out[16], 66.439522998, places=8)

    def test_rsi_uses_wilder_not_a_plain_sma(self) -> None:
        """Bar 15 must be the Wilder update, not a re-averaged 14-change SMA.

        The SMA-of-last-14-changes alternative at bar 15 would be
            gains  = changes[1:15] -> seven +2  = 14 -> avg 1.0
            losses = changes[1:15] -> seven -1  =  7 -> avg 0.5
        i.e. RS = 2 and RSI = 66.6667 again. Wilder's gives 69.7674, so the two differ by
        more than 3 points here and this assertion distinguishes them.
        """
        out = rsi(RSI_SERIES, 14)
        self.assertAlmostEqual(out[15], 3000.0 / 43.0, places=9)
        self.assertNotAlmostEqual(out[15], 200.0 / 3.0, places=3)

    def test_rsi_first_value_is_at_index_n(self) -> None:
        """n changes need n+1 prices, so the first RSI is at index n, not n-1."""
        out = rsi(RSI_SERIES, 14)
        self.assertPadding(out, length=len(RSI_SERIES), first_defined=14, msg="rsi(n=14)")
        self.assertEqual(len(rsi(RSI_SERIES[:15], 14)), 15)
        self.assertIsNotNone(rsi(RSI_SERIES[:15], 14)[14])
        # Exactly n prices is one short: n changes cannot be formed.
        self.assertSeries(rsi(RSI_SERIES[:14], 14), [None] * 14)

    def test_rsi_all_gains_is_100(self) -> None:
        """A monotone rise has no losses: avg_loss == 0 and RSI pins at 100."""
        out = rsi([float(i) for i in range(1, 17)], 14)
        self.assertSeries(out, [None] * 14 + [100.0, 100.0])

    def test_rsi_all_losses_is_zero(self) -> None:
        """A monotone fall has no gains: RS = 0, RSI = 100 - 100/1 = 0."""
        out = rsi([float(100 - i) for i in range(16)], 14)
        self.assertSeries(out, [None] * 14 + [0.0, 0.0])

    def test_rsi_flat_series_is_neutral_50_not_100(self) -> None:
        """A dead feed of identical prints reports 50, the documented safe branch.

        avg_gain == avg_loss == 0. The naive `avg_loss == 0 -> 100` shortcut would read a
        frozen feed as screaming overbought, and `rsi_reversal` would fire on it. This
        assertion exists to stop that regression costing real money.
        """
        out = rsi([100.0] * 20, 14)
        self.assertSeries(out, [None] * 14 + [50.0] * 6)

    def test_rsi_degenerate(self) -> None:
        """Empty, single value, and n <= 0."""
        self.assertSeries(rsi([], 14), [])
        self.assertSeries(rsi([100.0], 14), [None])
        self.assertSeries(rsi([1.0, 2.0, 3.0], 0), [None, None, None])
        self.assertSeries(rsi([1.0, 2.0, 3.0], -1), [None, None, None])

    def test_rsi_small_n_hand_computed(self) -> None:
        """RSI(2) on [10, 11, 10, 12]: a short window checked by hand.

        changes: +1, -1, +2
        seed over the first 2: avg_gain = 1/2 = 0.5, avg_loss = 1/2 = 0.5
            RSI[2] = 100 - 100/2 = 50
        bar 3 (change +2): avg_gain = (0.5*1 + 2)/2 = 1.25, avg_loss = (0.5*1 + 0)/2 = 0.25
            RS = 5, RSI[3] = 100 - 100/6 = 500/6 = 83.333333...
        """
        self.assertSeries(rsi([10.0, 11.0, 10.0, 12.0], 2), [None, None, 50.0, 500.0 / 6.0])


# ---------------------------------------------------------------------------
# ATR — Wilder, TR[0] = h - l
# ---------------------------------------------------------------------------

#: Five bars with deliberately varied true ranges, so a wrong TR term shows up.
ATR_BARS: list[tuple[float, float, float]] = [
    (105.0, 95.0, 100.0),
    (110.0, 102.0, 108.0),
    (109.0, 100.0, 101.0),
    (104.0, 99.0, 103.0),
    (120.0, 103.0, 118.0),
]


class TestAtr(SeriesAssertions):
    """Wilder's ATR, including the bar-0 true-range convention."""

    def test_atr_hand_computed(self) -> None:
        """ATR(3) over ATR_BARS, worked out in exact fractions.

        True ranges (bar 0 has no prior close, so TR = h - l):
          TR0 = 105 - 95 = 10
          TR1 = max(110-102=8,  |110-100|=10, |102-100|=2)  = 10
          TR2 = max(109-100=9,  |109-108|=1,  |100-108|=8)  =  9
          TR3 = max(104-99=5,   |104-101|=3,  |99-101|=2)   =  5
          TR4 = max(120-103=17, |120-103|=17, |103-103|=0)  = 17

        Wilder, n = 3:
          ATR[2] = (10 + 10 + 9)/3 = 29/3  = 9.666666...
          ATR[3] = (29/3 * 2 + 5)/3 = 73/9 = 8.111111...
          ATR[4] = (73/9 * 2 + 17)/3 = 299/27 = 11.074074...
        """
        out = atr(hlc_bars(ATR_BARS), 3)
        self.assertSeries(out, [None, None, 29.0 / 3.0, 73.0 / 9.0, 299.0 / 27.0])
        self.assertAlmostEqual(out[2], 9.666666667, places=8)
        self.assertAlmostEqual(out[3], 8.111111111, places=8)
        self.assertAlmostEqual(out[4], 11.074074074, places=8)

    def test_atr_bar_zero_true_range_is_high_minus_low(self) -> None:
        """With n == 1 the first ATR is bar 0's own range, proving TR[0] = h - l."""
        out = atr(hlc_bars(ATR_BARS), 1)
        self.assertAlmostEqual(out[0], 10.0, places=9)  # 105 - 95
        # And with n == 1 Wilder's update is the identity, so ATR == TR bar by bar.
        self.assertSeries(out, [10.0, 10.0, 9.0, 5.0, 17.0])

    def test_atr_uses_wilder_not_an_sma(self) -> None:
        """ATR[3] must be the Wilder carry, not the simple mean of TR[1..3].

        SMA of TR[1:4] = (10 + 9 + 5)/3 = 8.0, while Wilder's gives 73/9 = 8.111...
        """
        out = atr(hlc_bars(ATR_BARS), 3)
        self.assertAlmostEqual(out[3], 73.0 / 9.0, places=9)
        self.assertNotAlmostEqual(out[3], 8.0, places=4)

    def test_atr_padding(self) -> None:
        """First value at n-1; a series shorter than n is all None."""
        bars = walk(20)
        self.assertPadding(atr(bars, 14), length=20, first_defined=13, msg="atr(n=14)")
        self.assertSeries(atr(bars[:5], 14), [None] * 5)
        self.assertSeries(atr(bars, 0), [None] * 20)

    def test_atr_degenerate(self) -> None:
        """Empty, single candle, all-equal prices (zero range)."""
        self.assertSeries(atr([], 14), [])
        self.assertSeries(atr([candle(h=5.0, l=5.0, c=5.0)], 14), [None])
        self.assertSeries(atr([candle(h=12.0, l=8.0, c=10.0)], 1), [4.0])
        # A flat market has zero true range: ATR is 0.0, not None and not a crash.
        self.assertSeries(atr(closes_only([100.0] * 6), 3), [None, None, 0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# VWAP — session-anchored, resets at the day boundary
# ---------------------------------------------------------------------------


class TestVwap(SeriesAssertions):
    """Volume-weighted typical price, re-anchored at every new IST trading day."""

    @staticmethod
    def _two_day_series() -> list[dict]:
        """Three bars on DAY_A then two on DAY_B, with typical prices that are whole numbers."""
        return [
            candle(ts=ts_at(DAY_A, 9, 15), o=100, h=102, l=98, c=100, v=100),   # tp = 100
            candle(ts=ts_at(DAY_A, 9, 20), o=100, h=105, l=101, c=103, v=300),  # tp = 103
            candle(ts=ts_at(DAY_A, 9, 25), o=103, h=109, l=104, c=108, v=100),  # tp = 107
            candle(ts=ts_at(DAY_B, 9, 15), o=115, h=120, l=114, c=117, v=200),  # tp = 117
            candle(ts=ts_at(DAY_B, 9, 20), o=117, h=123, l=117, c=120, v=200),  # tp = 120
        ]

    def test_vwap_hand_computed_within_a_session(self) -> None:
        """Day A accumulates (h+l+c)/3 weighted by volume.

          bar0 tp = (102+98+100)/3  = 100; cum_pv = 10000,  cum_v = 100 -> 100.00
          bar1 tp = (105+101+103)/3 = 103; cum_pv = 40900,  cum_v = 400 -> 102.25
          bar2 tp = (109+104+108)/3 = 107; cum_pv = 51600,  cum_v = 500 -> 103.20
        """
        out = vwap(self._two_day_series())
        self.assertAlmostEqual(out[0], 100.0, places=9)
        self.assertAlmostEqual(out[1], 102.25, places=9)
        self.assertAlmostEqual(out[2], 103.2, places=9)

    def test_vwap_resets_across_the_day_boundary(self) -> None:
        """The first bar of DAY_B equals its own typical price — yesterday is discarded.

          bar3 tp = (120+114+117)/3 = 117; after the reset cum_pv = 23400, cum_v = 200
                 -> 117.00
          bar4 tp = (123+117+120)/3 = 120; cum_pv = 47400, cum_v = 400 -> 118.50

        Without the reset bar3 would read (51600 + 23400)/(500 + 200) = 107.142857, so
        the two implementations are 10 rupees apart on a NIFTY-sized number and every
        vwap_reclaim signal on the first bars of a day would be wrong.
        """
        out = vwap(self._two_day_series())
        self.assertSeries(out, [100.0, 102.25, 103.2, 117.0, 118.5])
        self.assertAlmostEqual(out[3], 117.0, places=9)
        self.assertNotAlmostEqual(out[3], 75000.0 / 700.0, places=4)  # 107.142857...

    def test_vwap_day_boundary_is_detected_from_ts_not_from_list_position(self) -> None:
        """A list that starts mid-session still anchors to that session's own bars.

        Here the list begins at 11:00 on DAY_A, so VWAP[0] is 11:00's own typical price.
        The function cannot know about the morning's volume and must not pretend to.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 11, 0), o=100, h=102, l=98, c=100, v=100),  # tp = 100
            candle(ts=ts_at(DAY_A, 11, 5), o=100, h=105, l=101, c=103, v=300),  # tp = 103
        ]
        # cum: 10000/100 = 100.0, then (10000 + 30900)/400 = 102.25
        self.assertSeries(vwap(bars), [100.0, 102.25])

    def test_vwap_three_days_resets_twice(self) -> None:
        """Every new IST date re-anchors, not just the first boundary."""
        day_c = date(2026, 1, 7)
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=102, l=98, c=100, v=100),   # tp = 100
            candle(ts=ts_at(DAY_B, 9, 15), h=202, l=198, c=200, v=100),  # tp = 200
            candle(ts=ts_at(day_c, 9, 15), h=302, l=298, c=300, v=100),  # tp = 300
        ]
        self.assertSeries(vwap(bars), [100.0, 200.0, 300.0])

    def test_vwap_zero_volume_bar_is_none(self) -> None:
        """A bar with no cumulative volume yields None, not a bare typical price.

        An unweighted "VWAP" is a different line from the chart's, and treating it as the
        real one would put vwap_reclaim on the wrong side of the level.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=102, l=98, c=100, v=0),     # tp = 100, no volume
            candle(ts=ts_at(DAY_A, 9, 20), h=105, l=101, c=103, v=300),  # tp = 103
        ]
        self.assertSeries(vwap(bars), [None, 103.0])

    def test_vwap_all_zero_volume_is_all_none(self) -> None:
        """No volume anywhere in the series: every bar is None, length preserved."""
        bars = [candle(ts=ts_at(DAY_A, 9, 15 + i), h=102, l=98, c=100, v=0) for i in range(4)]
        self.assertSeries(vwap(bars), [None] * 4)

    def test_vwap_none_can_appear_mid_series_at_a_day_boundary(self) -> None:
        """The reset means a None is possible *after* a defined value, not only before it.

        Callers that assume "once defined, always defined" would index a None here. This
        pins the real shape of the contract rather than the convenient one.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=102, l=98, c=100, v=100),   # tp = 100 -> 100.0
            candle(ts=ts_at(DAY_B, 9, 15), h=120, l=114, c=117, v=0),    # reset, no volume
            candle(ts=ts_at(DAY_B, 9, 20), h=123, l=117, c=120, v=200),  # tp = 120 -> 120.0
        ]
        self.assertSeries(vwap(bars), [100.0, None, 120.0])

    def test_vwap_negative_volume_is_treated_as_zero(self) -> None:
        """A bad print with negative volume must not unwind the day's accumulator."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=102, l=98, c=100, v=100),     # tp = 100 -> 100.0
            candle(ts=ts_at(DAY_A, 9, 20), h=202, l=198, c=200, v=-500),   # ignored
        ]
        # cum stays 10000/100, so VWAP holds at 100.0 instead of swinging to 125.0.
        self.assertSeries(vwap(bars), [100.0, 100.0])

    def test_vwap_all_equal_prices(self) -> None:
        """A flat market's VWAP is that flat price at every bar."""
        bars = closes_only([100.0] * 5, start_ts=ts_at(DAY_A, 9, 15))
        self.assertSeries(vwap(bars), [100.0] * 5)

    def test_vwap_degenerate_lengths(self) -> None:
        """Empty and single-candle inputs."""
        self.assertSeries(vwap([]), [])
        self.assertSeries(
            vwap([candle(ts=ts_at(DAY_A, 9, 15), h=102, l=98, c=100, v=10)]), [100.0]
        )

    def test_vwap_has_no_leading_padding_when_volume_is_present(self) -> None:
        """VWAP is defined from bar 0: it needs no warm-up window."""
        bars = walk(10, start_ts=ts_at(DAY_A, 9, 15))
        self.assertPadding(vwap(bars), length=10, first_defined=0, msg="vwap")


# ---------------------------------------------------------------------------
# Supertrend — band carry-forward and the flip
# ---------------------------------------------------------------------------

#: Seven bars each of range exactly 2, with every close inside the next bar's range, so
#: every TR is exactly 2 and ATR(3) is exactly 2.0 from index 2 onward. That holds the
#: ATR constant and isolates the band carry-forward logic, which is the part that breaks.
ST_BARS: list[tuple[float, float, float]] = [
    (102.0, 100.0, 101.0),    # 0
    (103.0, 101.0, 102.0),    # 1
    (104.0, 102.0, 103.0),    # 2  seed
    (105.0, 103.0, 104.0),    # 3  lower band ratchets up, upper band carries
    (104.0, 102.0, 102.5),    # 4  lower band must NOT loosen back down
    (103.0, 101.0, 101.0),    # 5  close < carried lower band -> FLIP to -1
    (102.0, 100.0, 100.0),    # 6  downtrend holds
]


class TestSupertrend(SeriesAssertions):
    """The final-band carry-forward rule and the direction flip."""

    def test_supertrend_fixture_has_constant_atr(self) -> None:
        """Sanity-check the fixture: every TR is 2, so ATR(3) is exactly 2.0.

          TR0 = 102 - 100 = 2 (no prior close)
          TR1 = max(2, |103-101|=2, |101-101|=0) = 2
          TR2 = max(2, |104-102|=2, |102-102|=0) = 2
          TR3 = max(2, |105-103|=2, |103-103|=0) = 2
          TR4 = max(2, |104-104|=0, |102-104|=2) = 2
          TR5 = max(2, |103-102.5|=0.5, |101-102.5|=1.5) = 2
          TR6 = max(2, |102-101|=1, |100-101|=1) = 2
        """
        self.assertSeries(atr(hlc_bars(ST_BARS), 3), [None, None, 2.0, 2.0, 2.0, 2.0, 2.0])

    def test_supertrend_hand_computed(self) -> None:
        """Supertrend(n=3, mult=1.0) over ST_BARS, bar by bar.

        basic_upper = hl2 + 1*ATR, basic_lower = hl2 - 1*ATR, ATR = 2 throughout.

          i=2 SEED: hl2 = 103, basic = (101, 105). No prior band, so final = basic.
                    direction = +1 because close 103 >= hl2 103.   -> (101.0, +1)
          i=3: hl2 = 104, basic = (102, 106), prev_close = 103.
               upper: 106 < 105? no; 103 > 105? no  -> CARRY 105
               lower: 102 > 101? yes                -> RATCHET to 102
               dir +1 held: close 104 < 102? no     -> (102.0, +1)
          i=4: hl2 = 103, basic = (101, 105), prev_close = 104.
               upper: 105 < 105? no; 104 > 105? no  -> CARRY 105
               lower: 101 > 102? no; 104 < 102? no  -> CARRY 102 (does not loosen to 101)
               dir +1 held: close 102.5 < 102? no   -> (102.0, +1)
          i=5: hl2 = 102, basic = (100, 104), prev_close = 102.5.
               upper: 104 < 105? yes                -> TIGHTEN to 104
               lower: 100 > 102? no; 102.5 < 102? no-> CARRY 102
               dir +1: close 101 < 102? YES         -> FLIP to -1, level = upper
                                                    -> (104.0, -1)
          i=6: hl2 = 101, basic = (99, 103), prev_close = 101.
               upper: 103 < 104? yes                -> TIGHTEN to 103
               lower: 99 > 102? no; 101 < 102? yes  -> RESET to 99
               dir -1: close 100 > 103? no          -> (103.0, -1)
        """
        out = supertrend(hlc_bars(ST_BARS), n=3, mult=1.0)
        self.assertSeries(
            out,
            [
                None,
                None,
                (101.0, 1),
                (102.0, 1),
                (102.0, 1),
                (104.0, -1),
                (103.0, -1),
            ],
        )

    def test_supertrend_band_carries_instead_of_tracking_atr(self) -> None:
        """At i=3 the upper band stays at 105 even though the basic band rose to 106.

        Without the carry-forward the level would jitter with hl2 every bar and the
        direction would flip constantly — one spurious trade per bar.
        """
        out = supertrend(hlc_bars(ST_BARS), n=3, mult=1.0)
        # i=3's lower band is the active level in an uptrend: 102, the ratcheted value.
        self.assertAlmostEqual(out[3][0], 102.0, places=9)
        # i=4: the basic lower band fell back to 101 but the carried band holds at 102.
        self.assertAlmostEqual(out[4][0], 102.0, places=9)
        self.assertNotAlmostEqual(out[4][0], 101.0, places=6)

    def test_supertrend_flip_happens_exactly_once_and_at_bar_5(self) -> None:
        """Direction is +1 through bar 4 and -1 from bar 5: one flip, where it belongs."""
        out = supertrend(hlc_bars(ST_BARS), n=3, mult=1.0)
        dirs = [None if v is None else v[1] for v in out]
        self.assertEqual(dirs, [None, None, 1, 1, 1, -1, -1])
        flips = [
            i
            for i in range(1, len(out))
            if out[i] is not None and out[i - 1] is not None and out[i][1] != out[i - 1][1]
        ]
        self.assertEqual(flips, [5], "exactly one flip, at index 5")

    def test_supertrend_level_switches_band_with_direction(self) -> None:
        """`level` is the lower band in an uptrend and the upper band in a downtrend."""
        out = supertrend(hlc_bars(ST_BARS), n=3, mult=1.0)
        self.assertAlmostEqual(out[2][0], 101.0, places=9)   # uptrend -> lower band
        self.assertAlmostEqual(out[5][0], 104.0, places=9)   # downtrend -> upper band
        self.assertGreater(out[5][0], out[4][0], "the stop band jumps above price on a flip")

    def test_supertrend_seed_bar_is_not_a_flip(self) -> None:
        """Index n-1 is a seed taken from the close's place in its own bar, not a flip.

        Here the seed bar's close (102.5) sits below its hl2 (103), so the seed direction
        is -1 even though nothing crossed anything. Index n-2 is None, so a caller
        comparing `out[i-1][1] != out[i][1]` must skip i == n-1 or it will read the seed
        as a signal and open a position on warm-up noise.
        """
        bars = hlc_bars([(102.0, 100.0, 101.0), (103.0, 101.0, 102.0), (104.0, 102.0, 102.5)])
        # TRs are 2, 2, 2 -> ATR(3)[2] = 2.0; hl2 = 103; basic_upper = 105.
        out = supertrend(bars, n=3, mult=1.0)
        self.assertSeries(out, [None, None, (105.0, -1)])
        self.assertIsNone(out[1], "no prior direction exists to flip from")

    def test_supertrend_seed_direction_on_a_close_equal_to_hl2(self) -> None:
        """A close exactly at hl2 seeds +1 (the `>=` branch)."""
        bars = hlc_bars([(102.0, 100.0, 101.0), (103.0, 101.0, 102.0), (104.0, 102.0, 103.0)])
        out = supertrend(bars, n=3, mult=1.0)
        self.assertEqual(out[2][1], 1)

    def test_supertrend_padding(self) -> None:
        """First value at n-1, all None when shorter than n, length always preserved."""
        bars = walk(30)
        self.assertPadding(supertrend(bars, n=10), length=30, first_defined=9, msg="supertrend")
        self.assertSeries(supertrend(bars[:5], n=10), [None] * 5)
        self.assertSeries(supertrend(bars, n=0), [None] * 30)

    def test_supertrend_degenerate(self) -> None:
        """Empty, single candle, and an all-equal series (zero ATR)."""
        self.assertSeries(supertrend([], n=10), [])
        self.assertSeries(supertrend([candle(h=1.0, l=1.0, c=1.0)], n=10), [None])

        # All-equal: ATR == 0, so both bands collapse onto hl2 == close. The seed takes
        # the `close >= hl2` branch (+1) and nothing can ever cross a band it sits on, so
        # the direction must hold at +1 with no flips. A flip here would mean the bot
        # trades a market that has not moved at all.
        flat = closes_only([100.0] * 8)
        out = supertrend(flat, n=3, mult=2.0)
        self.assertSeries(out, [None, None] + [(100.0, 1)] * 6)

    def test_supertrend_single_bar_with_n1(self) -> None:
        """n == 1 defines the indicator on bar 0 from that bar's own range.

        ATR(1)[0] = h - l = 12 - 8 = 4; hl2 = (12 + 8)/2 = 10;
        mult 1.0 -> basic bands = (10 - 4, 10 + 4) = (6, 14);
        close 11 >= hl2 10 -> direction +1, so level = the lower band = 6.
        """
        out = supertrend([candle(h=12.0, l=8.0, c=11.0)], n=1, mult=1.0)
        self.assertSeries(out, [(6.0, 1)])


# ---------------------------------------------------------------------------
# ADX — two Wilder passes
# ---------------------------------------------------------------------------

#: Five bars chosen so +DM, -DM and TR are small whole numbers and n = 2 brings the
#: first ADX down to index 3, where the whole two-pass pipeline fits on one page.
ADX_BARS: list[tuple[float, float, float]] = [
    (10.0, 8.0, 9.0),
    (12.0, 9.0, 11.0),
    (13.0, 11.0, 12.0),
    (12.0, 9.0, 10.0),
    (11.0, 8.0, 8.5),
]


class TestAdx(SeriesAssertions):
    """Wilder's ADX: DI pass then DX pass, first value at index 2n-1."""

    def test_adx_hand_computed_n2(self) -> None:
        """ADX(2) over ADX_BARS, derived in full.

        Directional movement and true range (bar 0 contributes nothing — ADX starts TR
        at bar 1, because +DM/-DM also need a previous bar):
          i=1: up = 12-10 = +2, down = 8-9  = -1 -> +DM 2, -DM 0
               TR = max(12-9=3,  |12-9|=3,  |9-9|=0)   = 3
          i=2: up = 13-12 = +1, down = 9-11 = -2 -> +DM 1, -DM 0
               TR = max(13-11=2, |13-11|=2, |11-11|=0) = 2
          i=3: up = 12-13 = -1, down = 11-9 = +2 -> +DM 0, -DM 2
               TR = max(12-9=3,  |12-12|=0, |9-12|=3)  = 3
          i=4: up = 11-12 = -1, down = 9-8  = +1 -> +DM 0, -DM 1
               TR = max(11-8=3,  |11-10|=1, |8-10|=2)  = 3

        Pass one (seed over bars 1..n = 1..2, then Wilder at alpha = 1/2). Note the TR
        term cancels out of DX, so DX = 100*|+SM - -SM| / (+SM + -SM):
          seed: TR_sm = 5/2, +SM = 3/2, -SM = 0
                DX[2] = 100 * |1.5 - 0| / 1.5 = 100
          i=3:  TR_sm = (2.5 + 3)/2 = 2.75, +SM = (1.5 + 0)/2 = 0.75, -SM = (0 + 2)/2 = 1
                DX[3] = 100 * 0.25 / 1.75 = 100/7  = 14.285714...
          i=4:  TR_sm = (2.75 + 3)/2 = 2.875, +SM = 0.375, -SM = (1 + 1)/2 = 1
                DX[4] = 100 * 0.625 / 1.375 = 500/11 = 45.454545...

        Pass two (smooth DX, seeded over DX[n .. 2n-1] = DX[2..3]):
          ADX[3] = (100 + 100/7)/2 = 400/7 = 57.142857...
          ADX[4] = (400/7 + 500/11)/2 = 3950/77 = 51.298701...
        """
        out = adx(hlc_bars(ADX_BARS), 2)
        self.assertSeries(out, [None, None, None, 400.0 / 7.0, 3950.0 / 77.0])
        self.assertAlmostEqual(out[3], 57.142857143, places=8)
        self.assertAlmostEqual(out[4], 51.298701299, places=8)

    def test_adx_outside_bar_records_only_the_larger_directional_move(self) -> None:
        """On an outside bar only the bigger of +DM/-DM counts — Wilder's `>` comparison.

        Bars: (10,8,9) (12,9,11) (15,7,8) (14,10,12) (11,6,7), n = 2.
          i=1: up = +2, down = -1              -> +DM 2, -DM 0
               TR = max(12-9=3, |12-9|=3, |9-9|=0) = 3
          i=2: OUTSIDE bar. up = 15-12 = +3 and down = 9-7 = +2 are BOTH positive.
               up > down, so +DM = 3 and -DM = 0 (not 2) — a bar cannot push the index
               both ways at once.
               TR = max(15-7=8, |15-11|=4, |7-11|=4) = 8
          i=3: INSIDE bar. up = -1, down = -3, both negative -> +DM 0, -DM 0
               TR = max(14-10=4, |14-8|=6, |10-8|=2) = 6
          i=4: up = -3, down = +4              -> +DM 0, -DM 4
               TR = max(11-6=5, |11-12|=1, |6-12|=6) = 6

        Pass one (DX = 100*|+SM - -SM|/(+SM + -SM), the TR term cancelling):
          seed: TR_sm = 11/2, +SM = 5/2, -SM = 0   -> DX[2] = 100
          i=3:  +SM = 5/4,  -SM = 0                -> DX[3] = 100
          i=4:  +SM = 5/8,  -SM = 2                -> DX[4] = 100*1.375/2.625
                                                             = 1100/21 = 52.380952...
        Pass two:
          ADX[3] = (100 + 100)/2 = 100
          ADX[4] = (100 + 1100/21)/2 = 1600/21 = 76.190476...

        Dropping the comparison and crediting -DM = 2 on the outside bar gives
        ADX[3] = 300/7 = 42.857 and ADX[4] = 8000/161 = 49.689 — a trend reading less
        than half the real one, which would mute every "ADX > 20" confirmation.
        """
        bars = hlc_bars([(10.0, 8.0, 9.0), (12.0, 9.0, 11.0), (15.0, 7.0, 8.0),
                         (14.0, 10.0, 12.0), (11.0, 6.0, 7.0)])
        out = adx(bars, 2)
        self.assertSeries(out, [None, None, None, 100.0, 1600.0 / 21.0])
        self.assertAlmostEqual(out[4], 76.190476190, places=8)
        self.assertNotAlmostEqual(out[3], 300.0 / 7.0, places=3)
        self.assertNotAlmostEqual(out[4], 8000.0 / 161.0, places=3)

    def test_adx_first_value_is_at_index_2n_minus_1(self) -> None:
        """Two Wilder passes push the first ADX to 2n-1 — 27 at the default n = 14."""
        bars = up_staircase(40)
        self.assertPadding(adx(bars, 14), length=40, first_defined=27, msg="adx(n=14)")
        # Exactly 2n bars is the minimum; one fewer is all None.
        self.assertIsNotNone(adx(up_staircase(28), 14)[27])
        self.assertSeries(adx(up_staircase(27), 14), [None] * 27)
        # Same boundary at n = 2: 4 bars define index 3, 3 bars define nothing.
        self.assertSeries(
            adx(hlc_bars(ADX_BARS[:4]), 2), [None, None, None, 400.0 / 7.0]
        )
        self.assertSeries(adx(hlc_bars(ADX_BARS[:3]), 2), [None, None, None])

    def test_adx_pure_uptrend_is_100(self) -> None:
        """A one-directional staircase pins ADX at exactly 100.

        Every bar has +DM 1, -DM 0, TR 2 (see `up_staircase`). Wilder's average of a
        constant is that constant, so +SM = 1, -SM = 0 forever, giving
        -DI = 0 and DX = 100*|+DI - 0|/(+DI) = 100 on every bar from index n. Smoothing a
        constant 100 leaves 100, so ADX[2n-1:] == 100 exactly.
        """
        out = adx(up_staircase(32), 14)
        for i in range(27, 32):
            with self.subTest(i=i):
                self.assertAlmostEqual(out[i], 100.0, places=9)

    def test_adx_pure_downtrend_is_also_100(self) -> None:
        """ADX measures trend strength, not direction: the mirror series also gives 100.

        The down staircase has +DM 0, -DM 1, TR 2, so +DI = 0 and DX = 100 again.
        """
        out = adx(down_staircase(32), 14)
        for i in range(27, 32):
            with self.subTest(i=i):
                self.assertAlmostEqual(out[i], 100.0, places=9)

    def test_adx_balanced_zigzag_is_near_zero(self) -> None:
        """Equal up and down movement cannot register as a trend.

        For `zigzag`, TR is 2 on every bar and exactly one of +DM/-DM is 1, so
        (+SM + -SM) is invariant at 1.0 under Wilder smoothing and +DI + -DI = 50 always.
        Hence DX = 2*|+DI - -DI| = 100*|+SM - -SM|, and the difference d = +SM - -SM
        obeys d <- (13/14)d +/- 1/14 with the sign alternating every bar: starting from
        d = 0 at the seed it never exceeds 1/14 in magnitude, so DX stays under ~7.2 and
        the 14-bar average of DX stays well under 10.

        The bound, not an observed number, is the assertion: a broken ADX reports a big
        number for a market that has gone nowhere, and the strategy's "ADX > 20"
        confirmation would then fire on chop.
        """
        out = adx(zigzag(60), 14)
        for i in range(27, 60):
            with self.subTest(i=i):
                self.assertIsNotNone(out[i])
                self.assertGreaterEqual(out[i], 0.0)
                self.assertLess(out[i], 10.0)

    def test_adx_all_equal_prices_is_zero(self) -> None:
        """A zero-range market has no true range at all: the guarded division returns 0.0.

        TR == 0 on every bar, so TR_sm == 0 and the `tr_sm <= 0` guard must yield 0.0
        rather than dividing by zero.
        """
        out = adx(closes_only([100.0] * 32), 14)
        for i in range(27, 32):
            with self.subTest(i=i):
                self.assertAlmostEqual(out[i], 0.0, places=9)

    def test_adx_degenerate(self) -> None:
        """Empty, single candle, and n <= 0."""
        self.assertSeries(adx([], 14), [])
        self.assertSeries(adx([candle(h=1.0, l=1.0, c=1.0)], 14), [None])
        self.assertSeries(adx(hlc_bars(ADX_BARS), 0), [None] * 5)
        self.assertSeries(adx(hlc_bars(ADX_BARS), -3), [None] * 5)

    def test_adx_depends_only_on_directional_movement_not_on_true_range(self) -> None:
        """ADX is invariant to the true range: `tr_sm` cancels out of DX algebraically.

            DX = 100 * |+DI - -DI| / (+DI + -DI)
               = 100 * |100*(+SM/TR_sm) - 100*(-SM/TR_sm)| / (100*(+SM/TR_sm) + 100*(-SM/TR_sm))
               = 100 * |+SM - -SM| / (+SM + -SM)

        So TR affects the returned ADX only through the `tr_sm <= 0` guard. This test
        moves every close down to its own bar's low — which changes TR on two bars but
        leaves every high and low, and therefore every +DM/-DM, untouched — and asserts
        the ADX series is identical.

        Recorded here because it is genuinely surprising: `adx()` computes a TR average
        that cannot change its own output, and its docstring justifies the bar-1 TR seed
        in terms of +DI/-DI bias, which this function never returns. A future change that
        exposes +DI/-DI will make that seeding observable, and this test is the note
        saying so.
        """
        base = hlc_bars(ADX_BARS)
        closes_at_lows = hlc_bars([(h, l, l) for h, l, _c in ADX_BARS])
        self.assertSeries(adx(closes_at_lows, 2), [None, None, None, 400.0 / 7.0, 3950.0 / 77.0])
        self.assertSeries(adx(closes_at_lows, 2), adx(base, 2))

    def test_adx_is_bounded(self) -> None:
        """ADX is a percentage: it must stay inside [0, 100] on a varied series."""
        for value in adx(walk(80), 14):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------


class TestBollinger(SeriesAssertions):
    """(lower, mid, upper) with a population standard deviation, as charts draw it."""

    def test_bollinger_hand_computed(self) -> None:
        """[2,4,4,4,5,5,7,9] with n=8, k=2: the textbook sigma = 2 series.

        mean = 40/8 = 5
        deviations: -3,-1,-1,-1, 0, 0, 2, 4
        squares:      9, 1, 1, 1, 0, 0, 4,16 -> sum 32
        population variance = 32/8 = 4 -> sigma = 2
        bands = (5 - 2*2, 5, 5 + 2*2) = (1, 5, 9)
        """
        out = bollinger([2, 4, 4, 4, 5, 5, 7, 9], 8, 2.0)
        self.assertSeries(out, [None] * 7 + [(1.0, 5.0, 9.0)])

    def test_bollinger_uses_population_sigma_not_sample_sigma(self) -> None:
        """Dividing by n-1 instead of n would widen the bands measurably.

        For the same window, the sample variance is 32/7 = 4.571428..., sigma = 2.13809,
        and the upper band would be 9.27619 rather than 9.
        """
        out = bollinger([2, 4, 4, 4, 5, 5, 7, 9], 8, 2.0)
        self.assertAlmostEqual(out[7][2], 9.0, places=9)
        self.assertNotAlmostEqual(out[7][2], 5.0 + 2.0 * math.sqrt(32.0 / 7.0), places=4)

    def test_bollinger_hand_computed_irrational_sigma(self) -> None:
        """[1,2,3,4] with n=4, k=2: mean 2.5, variance 5/4, sigma = sqrt(1.25).

        sqrt(1.25) = 1.1180339887498949, so the bands are
          (2.5 - 2.2360679774997898, 2.5, 2.5 + 2.2360679774997898)
          = (0.2639320225002102, 2.5, 4.7360679774997898)
        """
        out = bollinger([1, 2, 3, 4], 4, 2.0)
        sigma = math.sqrt(1.25)
        self.assertSeries(out, [None, None, None, (2.5 - 2 * sigma, 2.5, 2.5 + 2 * sigma)])
        self.assertAlmostEqual(out[3][0], 0.263932023, places=8)
        self.assertAlmostEqual(out[3][2], 4.736067977, places=8)

    def test_bollinger_mid_band_equals_sma(self) -> None:
        """The mid band is SMA(n) by definition — cross-checked against `sma`."""
        values = [float(v) for v in (11, 13, 12, 15, 14, 18, 17, 16, 19, 21, 20, 22)]
        mids = bollinger(values, 5, 2.0)
        means = sma(values, 5)
        for i, (band, mean) in enumerate(zip(mids, means)):
            with self.subTest(i=i):
                if band is None:
                    self.assertIsNone(mean)
                else:
                    self.assertAlmostEqual(band[1], mean, places=9)

    def test_bollinger_all_equal_collapses_the_bands(self) -> None:
        """Zero variance means all three bands sit on the mean; sqrt(0) must not blow up."""
        self.assertSeries(
            bollinger([50.0] * 4, 3, 2.0), [None, None, (50.0, 50.0, 50.0), (50.0, 50.0, 50.0)]
        )

    def test_bollinger_padding_and_degenerate(self) -> None:
        """First value at n-1; empty, too-short and n <= 0 give padded Nones."""
        self.assertPadding(bollinger(list(range(30)), 20, 2.0), length=30, first_defined=19,
                           msg="bollinger(n=20)")
        self.assertSeries(bollinger([], 20), [])
        self.assertSeries(bollinger([1.0], 20), [None])
        self.assertSeries(bollinger([1.0, 2.0, 3.0], 0), [None, None, None])

    def test_bollinger_k_zero(self) -> None:
        """k = 0 degenerates to three copies of the SMA."""
        out = bollinger([1, 2, 3, 4], 4, 0.0)
        self.assertSeries(out, [None, None, None, (2.5, 2.5, 2.5)])


# ---------------------------------------------------------------------------
# opening_range — anchored to the session, not to the list
# ---------------------------------------------------------------------------


class TestOpeningRange(SeriesAssertions):
    """The ORB window is [09:15, 09:15 + minutes) of the session date, not the list's head."""

    def test_opening_range_full_session_from_the_open(self) -> None:
        """Candles from 09:15: the first three 5-minute bars make the 15-minute range."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0),
            candle(ts=ts_at(DAY_A, 9, 20), h=112.0, l=103.0),
            candle(ts=ts_at(DAY_A, 9, 25), h=108.0, l=104.0),
            candle(ts=ts_at(DAY_A, 9, 30), h=130.0, l=90.0),   # outside the window
        ]
        # high = max(110, 112, 108) = 112; low = min(105, 103, 104) = 103
        self.assertEqual(opening_range(bars, 15), (112.0, 103.0))

    def test_opening_range_when_the_list_starts_mid_session(self) -> None:
        """A list beginning at 09:25 measures only the bars genuinely inside the window.

        Window = [09:15, 09:30). Only the 09:25 bar qualifies, so the range is its own
        high/low. Taking "the first three candles of the list" instead would return
        (120, 100) — a range 16 points wider, built from bars after the opening range
        closed, and every ORB trigger computed from it would be wrong.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 9, 25), h=112.0, l=108.0),
            candle(ts=ts_at(DAY_A, 9, 30), h=120.0, l=100.0),
            candle(ts=ts_at(DAY_A, 9, 35), h=118.0, l=109.0),
        ]
        self.assertEqual(opening_range(bars, 15), (112.0, 108.0))
        self.assertNotEqual(opening_range(bars, 15), (120.0, 100.0))

    def test_opening_range_is_none_when_the_window_is_absent(self) -> None:
        """A list that starts after the window returns None, never a guessed range.

        This is the case that matters operationally: the engine restarts at 10:00, fetches
        only today's recent candles, and must get None (ORB does not fire) rather than a
        fabricated range built from 10:00-10:15.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 10, 0), h=120.0, l=110.0),
            candle(ts=ts_at(DAY_A, 10, 5), h=125.0, l=115.0),
            candle(ts=ts_at(DAY_A, 10, 10), h=122.0, l=112.0),
        ]
        self.assertIsNone(opening_range(bars, 15))

    def test_opening_range_excludes_the_bar_at_the_window_end(self) -> None:
        """The window is half-open: a candle whose ts == start + minutes is excluded."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0),
            candle(ts=ts_at(DAY_A, 9, 30), h=200.0, l=10.0),
        ]
        self.assertEqual(opening_range(bars, 15), (110.0, 105.0))

    def test_opening_range_excludes_pre_open_candles(self) -> None:
        """A pre-market print at 09:00 is before the window's start and must not widen it."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 0), h=999.0, l=1.0),     # pre-open, ignored
            candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0),
            candle(ts=ts_at(DAY_A, 9, 20), h=112.0, l=103.0),
        ]
        self.assertEqual(opening_range(bars, 15), (112.0, 103.0))

    def test_opening_range_includes_a_straddling_candle_whole(self) -> None:
        """A 15-minute candle inside a 20-minute window counts whole, widening the range.

        Window = [09:15, 09:35). The 09:30 candle starts inside it and is taken entire,
        which can only make the range wider and so can only make an ORB trigger LATER —
        the safe direction for a breakout rule.
        """
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=100.0, l=95.0),
            candle(ts=ts_at(DAY_A, 9, 30), h=130.0, l=94.0),   # straddles 09:35
            candle(ts=ts_at(DAY_A, 9, 45), h=140.0, l=80.0),   # fully outside
        ]
        self.assertEqual(opening_range(bars, 20), (130.0, 94.0))

    def test_opening_range_defaults_to_the_last_candle_s_session(self) -> None:
        """In a multi-day series the default window is the day of the LAST candle."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=100.0, l=90.0),
            candle(ts=ts_at(DAY_A, 9, 20), h=101.0, l=95.0),
            candle(ts=ts_at(DAY_B, 9, 15), h=200.0, l=190.0),
            candle(ts=ts_at(DAY_B, 9, 20), h=205.0, l=195.0),
            candle(ts=ts_at(DAY_B, 10, 0), h=210.0, l=180.0),
        ]
        # DAY_B window: max(200, 205) = 205, min(190, 195) = 190
        self.assertEqual(opening_range(bars, 15), (205.0, 190.0))

    def test_opening_range_honours_an_explicit_session_date(self) -> None:
        """`session_date` measures an earlier day in the same multi-day series."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=100.0, l=90.0),
            candle(ts=ts_at(DAY_A, 9, 20), h=101.0, l=95.0),
            candle(ts=ts_at(DAY_B, 9, 15), h=200.0, l=190.0),
        ]
        # DAY_A window: max(100, 101) = 101, min(90, 95) = 90
        self.assertEqual(opening_range(bars, 15, session_date=DAY_A), (101.0, 90.0))

    def test_opening_range_window_length_changes_the_answer(self) -> None:
        """A 5-minute window takes one bar; a 10-minute window takes two."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=200.0, l=190.0),
            candle(ts=ts_at(DAY_A, 9, 20), h=205.0, l=195.0),
            candle(ts=ts_at(DAY_A, 9, 25), h=260.0, l=150.0),
        ]
        self.assertEqual(opening_range(bars, 5), (200.0, 190.0))
        self.assertEqual(opening_range(bars, 10), (205.0, 190.0))
        self.assertEqual(opening_range(bars, 15), (260.0, 150.0))

    def test_opening_range_degenerate(self) -> None:
        """Empty list, non-positive minutes, and a single in-window candle."""
        self.assertIsNone(opening_range([], 15))
        self.assertIsNone(opening_range([candle(ts=ts_at(DAY_A, 9, 15), h=1.0, l=0.0)], 0))
        self.assertIsNone(opening_range([candle(ts=ts_at(DAY_A, 9, 15), h=1.0, l=0.0)], -5))
        self.assertEqual(
            opening_range([candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0)], 15),
            (110.0, 105.0),
        )

    def test_opening_range_all_equal_prices(self) -> None:
        """A zero-range opening gives high == low, which is a valid (if untradable) range."""
        bars = closes_only([100.0] * 3, start_ts=ts_at(DAY_A, 9, 15))
        self.assertEqual(opening_range(bars, 15), (100.0, 100.0))

    def test_opening_range_returns_high_first(self) -> None:
        """The tuple order is (high, low) — reversing it would invert every ORB trade."""
        bars = [candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0)]
        high, low = opening_range(bars, 15)
        self.assertGreater(high, low)
        self.assertEqual((high, low), (110.0, 105.0))


# ---------------------------------------------------------------------------
# body_ratio
# ---------------------------------------------------------------------------


class TestBodyRatio(SeriesAssertions):
    """|close - open| / (high - low), clamped to [0, 1]."""

    def test_body_ratio_hand_computed(self) -> None:
        """A doji, a half-body bar and a marubozu."""
        self.assertAlmostEqual(body_ratio(candle(o=100, h=105, l=95, c=100)), 0.0, places=9)
        # |105 - 100| / (110 - 100) = 5/10 = 0.5
        self.assertAlmostEqual(body_ratio(candle(o=100, h=110, l=100, c=105)), 0.5, places=9)
        # |110 - 100| / (110 - 100) = 1.0
        self.assertAlmostEqual(body_ratio(candle(o=100, h=110, l=100, c=110)), 1.0, places=9)
        # A bearish bar is measured on the absolute body: |95 - 100| / (102 - 94) = 5/8
        self.assertAlmostEqual(body_ratio(candle(o=100, h=102, l=94, c=95)), 0.625, places=9)

    def test_body_ratio_zero_range_is_zero(self) -> None:
        """h == l would divide by zero; the safe answer is 0.0 (adds no confidence)."""
        self.assertEqual(body_ratio(candle(o=100, h=100, l=100, c=100)), 0.0)

    def test_body_ratio_clamps_contradictory_data(self) -> None:
        """A body wider than its own bar is bad data and must not score above 1.

        |120 - 100| / (110 - 100) = 2.0 raw. Left unclamped it would inflate a confluence
        score past its documented cap and talk the bot into a trade on a broken print.
        """
        self.assertEqual(body_ratio(candle(o=100, h=110, l=100, c=120)), 1.0)

    def test_body_ratio_malformed_candle_is_zero(self) -> None:
        """A missing key, a non-numeric and a NaN all return 0.0 rather than raising."""
        with self.assertLogs(indicators.logger, level="WARNING"):
            self.assertEqual(body_ratio({"o": 1.0, "h": 2.0, "l": 0.0}), 0.0)  # no "c"
        with self.assertLogs(indicators.logger, level="WARNING"):
            self.assertEqual(body_ratio(candle(o="x", h=2.0, l=0.0, c=1.0)), 0.0)
        with self.assertLogs(indicators.logger, level="WARNING"):
            self.assertEqual(
                body_ratio(candle(o=float("nan"), h=2.0, l=0.0, c=1.0)), 0.0
            )

    def test_body_ratio_is_bounded(self) -> None:
        """Every bar of a varied series scores inside [0, 1]."""
        for bar in walk(40):
            value = body_ratio(bar)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


class TestRegime(SeriesAssertions):
    """ADX gate plus EMA stack, defaulting to "range" whenever anything is undefined."""

    def test_regime_trend_up(self) -> None:
        """60 bars of a rising staircase: ADX 100 and EMA9 > EMA21 > EMA50.

        From the EMA ramp identity, the last EMAs are 159-4 = 155, 159-10 = 149 and
        159-24.5 = 134.5, which is a strict stack; ADX is exactly 100 (see
        test_adx_pure_uptrend_is_100), comfortably past the 20 threshold.
        """
        self.assertEqual(regime(up_staircase(60)), "trend_up")

    def test_regime_trend_down(self) -> None:
        """The mirror series: ADX 100 and EMA9 < EMA21 < EMA50 (45 < 51 < 65.5)."""
        self.assertEqual(regime(down_staircase(60)), "trend_down")

    def test_regime_range_when_adx_is_below_the_threshold(self) -> None:
        """A balanced zigzag has an ADX under 4, so no trend verdict is allowed."""
        self.assertEqual(regime(zigzag(60)), "range")

    def test_regime_adx_gate_outranks_the_ema_stack(self) -> None:
        """Raising the threshold above the measured ADX must turn a trend into a range.

        The up staircase is a textbook trend, but with the threshold pushed past 100 the
        ADX gate has to veto it — the gate is not decorative.
        """
        bars = up_staircase(60)
        self.assertEqual(regime(bars, adx_threshold=0.0), "trend_up")
        self.assertEqual(regime(bars, adx_threshold=100.5), "range")

    def test_regime_range_when_too_short_to_warm_up(self) -> None:
        """Fewer than 50 bars cannot define EMA50, so the verdict is the safe "range".

        This is the live-trading case worth knowing: on 5-minute candles a single day
        reaches 50 bars only at 13:25, so a candle list covering just today reads as
        "range" all morning.
        """
        self.assertEqual(regime(up_staircase(49)), "range")
        self.assertEqual(regime(up_staircase(20)), "range")
        self.assertEqual(regime(up_staircase(1)), "range")

    def test_regime_range_when_adx_cannot_warm_up(self) -> None:
        """50 bars satisfy EMA50 but a large adx_n still leaves ADX undefined -> "range"."""
        self.assertEqual(regime(up_staircase(50), adx_n=40), "range")  # needs 80 bars

    def test_regime_all_equal_prices_is_range(self) -> None:
        """A flat market has ADX 0 and a flat EMA stack: "range", not a trend."""
        self.assertEqual(regime(closes_only([100.0] * 60)), "range")

    def test_regime_mixed_ema_stack_is_range(self) -> None:
        """A trend that has just turned down leaves a mixed stack, which is "range".

        50 bars up then 10 sharply down: EMA9 has crossed under EMA21 while EMA50 is
        still below both, so the stack is neither strictly rising nor strictly falling.
        """
        rows = [(101.0 + i, 99.0 + i, 100.0 + i) for i in range(50)]
        rows += [(150.0 - 3 * i, 146.0 - 3 * i, 148.0 - 3 * i) for i in range(10)]
        bars = hlc_bars(rows)
        closes = [r[2] for r in rows]
        fast, mid, slow = (ema(closes, k)[-1] for k in (9, 21, 50))
        self.assertFalse(fast > mid > slow, "fixture should not be a clean up stack")
        self.assertFalse(fast < mid < slow, "fixture should not be a clean down stack")
        self.assertEqual(regime(bars), "range")

    def test_regime_returns_one_of_the_declared_values(self) -> None:
        """The verdict is always a member of REGIMES, so `Signal.make_bucket` stays valid."""
        for bars in (
            [],
            up_staircase(60),
            down_staircase(60),
            zigzag(60),
            closes_only([100.0] * 60),
            walk(80),
        ):
            with self.subTest(n=len(bars)):
                self.assertIn(regime(bars), indicators.REGIMES)

    def test_regime_degenerate(self) -> None:
        """Empty and malformed inputs both fall back to "range"."""
        self.assertEqual(regime([]), "range")
        bad = up_staircase(60)
        bad[30] = dict(bad[30])
        del bad[30]["c"]
        with self.assertLogs(indicators.logger, level="WARNING"):
            self.assertEqual(regime(bad), "range")


# ---------------------------------------------------------------------------
# the padding / length contract, across every series function
# ---------------------------------------------------------------------------

#: (name, callable taking the input, first defined index) for every padded series
#: function. The contract in CONTRACTS.md section D is "a list of the same length,
#: left-padded with None" — this table is what enforces it.
_SERIES_CASES: list[tuple[str, Callable[[list], list], int, bool]] = [
    ("sma(n=5)", lambda xs: sma(xs, 5), 4, False),
    ("ema(n=5)", lambda xs: ema(xs, 5), 4, False),
    ("rsi(n=5)", lambda xs: rsi(xs, 5), 5, False),
    ("bollinger(n=5)", lambda xs: bollinger(xs, 5, 2.0), 4, False),
    ("atr(n=5)", lambda xs: atr(xs, 5), 4, True),
    ("supertrend(n=5)", lambda xs: supertrend(xs, 5, 2.0), 4, True),
    ("adx(n=3)", lambda xs: adx(xs, 3), 5, True),
    ("vwap", vwap, 0, True),
]


class TestPaddingContract(SeriesAssertions):
    """Length and padding, checked for every function at every input length 0..24."""

    def test_length_and_padding_at_every_input_length(self) -> None:
        """len(out) == len(input), None strictly before the first defined index, set after."""
        bars = walk(24, start_ts=ts_at(DAY_A, 9, 15))
        prices = [b["c"] for b in bars]
        for name, fn, first, wants_candles in _SERIES_CASES:
            source: list = bars if wants_candles else prices
            for length in range(0, len(source) + 1):
                with self.subTest(fn=name, length=length):
                    out = fn(source[:length])
                    self.assertPadding(out, length=length, first_defined=first, msg=name)

    def test_all_none_when_shorter_than_the_warm_up(self) -> None:
        """A series shorter than the warm-up returns Nones, not a truncated list."""
        bars = walk(24, start_ts=ts_at(DAY_A, 9, 15))
        prices = [b["c"] for b in bars]
        for name, fn, first, wants_candles in _SERIES_CASES:
            if first == 0:
                continue
            source: list = bars if wants_candles else prices
            with self.subTest(fn=name):
                out = fn(source[:first])  # one bar short of the first defined index
                self.assertEqual(len(out), first, f"{name}: length must survive")
                self.assertTrue(all(v is None for v in out), f"{name}: must be all None")

    def test_empty_input_returns_empty_list(self) -> None:
        """Nothing in, nothing out — and an empty list, never None."""
        for name, fn, _first, wants_candles in _SERIES_CASES:
            with self.subTest(fn=name):
                out = fn([])
                self.assertIsInstance(out, list, f"{name}: must return a list")
                self.assertEqual(out, [], f"{name}: must return []")

    def test_single_element_input(self) -> None:
        """One bar in, one slot out, for every function."""
        bar = walk(1, start_ts=ts_at(DAY_A, 9, 15))
        for name, fn, _first, wants_candles in _SERIES_CASES:
            source: list = bar if wants_candles else [bar[0]["c"]]
            with self.subTest(fn=name):
                self.assertEqual(len(fn(source)), 1, f"{name}: must return one slot")

    def test_functions_do_not_mutate_their_input(self) -> None:
        """Indicators are pure: the caller's candle list and prices come back untouched.

        The engine reuses one cached candle list across every setup on every tick, so a
        mutation here would corrupt every later indicator on that bar.
        """
        bars = walk(30, start_ts=ts_at(DAY_A, 9, 15))
        prices = [b["c"] for b in bars]
        bars_before = [dict(b) for b in bars]
        prices_before = list(prices)
        for _name, fn, _first, wants_candles in _SERIES_CASES:
            fn(bars if wants_candles else prices)
        opening_range(bars, 15)
        regime(bars)
        body_ratio(bars[0])
        self.assertEqual(bars, bars_before)
        self.assertEqual(prices, prices_before)

    def test_series_functions_accept_ints(self) -> None:
        """Integer prices are coerced to float rather than rejected."""
        self.assertSeries(sma([1, 2, 3], 3), [None, None, 2.0])
        self.assertSeries(ema([1, 2, 3], 3), [None, None, 2.0])
        int_bars = [candle(ts=ts_at(DAY_A, 9, 15) + 300 * i, o=1, h=2, l=0, c=1, v=10)
                    for i in range(6)]
        self.assertSeries(atr(int_bars, 3), [None, None, 2.0, 2.0, 2.0, 2.0])


# ---------------------------------------------------------------------------
# the never-raise policy
# ---------------------------------------------------------------------------


class TestMalformedInput(SeriesAssertions):
    """Bad data yields padded Nones and a WARNING — an indicator must never throw mid-tick."""

    def _candle_variants(self) -> list[tuple[str, list[dict]]]:
        """Six-bar candle series, each corrupted a different way."""
        base = walk(6, start_ts=ts_at(DAY_A, 9, 15))

        def corrupt(index: int, **patch: Any) -> list[dict]:
            rows = [dict(b) for b in base]
            rows[index].update(patch)
            return rows

        missing = [dict(b) for b in base]
        del missing[3]["h"]

        return [
            ("missing high", missing),
            ("non-numeric close", corrupt(2, c="n/a")),
            ("NaN low", corrupt(4, l=float("nan"))),
            ("inf high", corrupt(1, h=float("inf"))),
            ("None volume", corrupt(0, v=None)),
        ]

    def test_candle_functions_return_all_none_on_bad_data(self) -> None:
        """atr, supertrend, adx and vwap degrade to all-None instead of raising."""
        for label, bars in self._candle_variants():
            for name, fn, first in (
                ("atr", lambda xs: atr(xs, 3), 2),
                ("supertrend", lambda xs: supertrend(xs, 3, 2.0), 2),
                ("adx", lambda xs: adx(xs, 3), 5),
                ("vwap", vwap, 0),
            ):
                with self.subTest(data=label, fn=name):
                    # "None volume" only breaks the functions that read volume.
                    if label == "None volume" and name != "vwap":
                        continue
                    out = fn(bars)
                    self.assertEqual(len(out), len(bars), f"{name}: length must survive")
                    self.assertTrue(
                        all(v is None for v in out), f"{name} on {label}: expected all None"
                    )

    def test_price_functions_return_all_none_on_bad_data(self) -> None:
        """sma, ema, rsi and bollinger degrade the same way on a bad price series."""
        for label, values in (
            ("non-numeric", [1.0, 2.0, "x", 4.0, 5.0, 6.0]),
            ("None", [1.0, 2.0, None, 4.0, 5.0, 6.0]),
            ("NaN", [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]),
            ("inf", [1.0, 2.0, float("inf"), 4.0, 5.0, 6.0]),
        ):
            for name, fn in (
                ("sma", lambda xs: sma(xs, 3)),
                ("ema", lambda xs: ema(xs, 3)),
                ("rsi", lambda xs: rsi(xs, 3)),
                ("bollinger", lambda xs: bollinger(xs, 3, 2.0)),
            ):
                with self.subTest(data=label, fn=name):
                    out = fn(list(values))
                    self.assertEqual(len(out), len(values))
                    self.assertTrue(all(v is None for v in out), f"{name} on {label}")

    def test_bad_data_is_logged_not_swallowed(self) -> None:
        """The all-None fallback comes with a WARNING, so a silent feed problem is visible."""
        bars = walk(6, start_ts=ts_at(DAY_A, 9, 15))
        broken = [dict(b) for b in bars]
        del broken[2]["c"]
        with self.assertLogs(indicators.logger, level="WARNING") as captured:
            atr(broken, 3)
        self.assertTrue(any("candle 2" in line for line in captured.output), captured.output)

    def test_opening_range_returns_none_on_bad_data(self) -> None:
        """A malformed candle makes ORB silent, never a guessed range."""
        bars = [
            candle(ts=ts_at(DAY_A, 9, 15), h=110.0, l=105.0),
            candle(ts=ts_at(DAY_A, 9, 20), h="bad", l=103.0),
        ]
        with self.assertLogs(indicators.logger, level="WARNING"):
            self.assertIsNone(opening_range(bars, 15))

    def test_nothing_raises_on_any_fixture(self) -> None:
        """A blunt sweep: no fixture in this file may make any public function throw."""
        fixtures: list[list[dict]] = [
            [],
            walk(1, start_ts=ts_at(DAY_A, 9, 15)),
            walk(3, start_ts=ts_at(DAY_A, 9, 15)),
            walk(60, start_ts=ts_at(DAY_A, 9, 15)),
            closes_only([100.0] * 60, start_ts=ts_at(DAY_A, 9, 15)),
            up_staircase(60, start_ts=ts_at(DAY_A, 9, 15)),
            zigzag(60, start_ts=ts_at(DAY_A, 9, 15)),
            [b for label, bars in self._candle_variants() for b in bars],
        ]
        for i, bars in enumerate(fixtures):
            prices = [b.get("c", 0.0) for b in bars]
            with self.subTest(fixture=i):
                for n in (0, 1, 3, 14, 200):
                    sma(prices, n)
                    ema(prices, n)
                    rsi(prices, n)
                    bollinger(prices, n, 2.0)
                    atr(bars, n)
                    supertrend(bars, n, 2.0)
                    adx(bars, n)
                vwap(bars)
                regime(bars)
                for minutes in (0, 15, 375):
                    opening_range(bars, minutes)
                for bar in bars:
                    body_ratio(bar)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
