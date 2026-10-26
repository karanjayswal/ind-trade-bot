"""Signal engine: six setups, one confluence score, one exit rule.

This module decides *what the bot trades*. It is deliberately stateless — the same
`Strategy` object, given the same candles, returns the same signals in the backtest
and in the live loop. Any state here (a "have I already fired today?" flag, a cached
last price) would make the two diverge, and a backtest that disagrees with live is
worse than no backtest at all.

LOOK-AHEAD
----------
`evaluate` treats the LAST element of `candles` as the last CLOSED candle and never
reads a forming one. A candle stamped `ts` covers the half-open window
`[ts, ts + interval)` (see the API docs, "Candle timestamps"), so it is closed only
once `now_ts >= ts + interval`. When `now_ts` is supplied, trailing candles whose
window has not elapsed are dropped with a warning. Look-ahead is the one bug that
turns a backtest into a lie, so it is checked rather than assumed.

WARM-UP
-------
The caller should pass roughly 60+ bars of history, not just today's. `ema(50)` needs
50 closes; on `5minute` that is over four hours. With less history the EMA-stack term
contributes 0.00 and the `pullback` setup cannot fire at all — both say so in
`reasons` rather than failing silently.

The series must also still contain **today's 09:15 candles**, because the opening range
is measured from them. A rolling "last N bars" window that has scrolled past the open
leaves `or_high`/`or_low` as None and `orb` simply never fires. Pass the day from 09:15
plus whatever prior history you have.

CONFLUENCE SCORE
----------------
`score = clamp(base_weight(setup) + sum(six confirmations), 0.0, 1.0)`.

Base weight, per setup:

| setup           | base  | family    |
|-----------------|-------|-----------|
| `orb`           | +0.50 | trend     |
| `pullback`      | +0.50 | trend     |
| `vwap_reclaim`  | +0.45 | reversal  |
| `ema_cross`     | +0.45 | trend     |
| `supertrend`    | +0.42 | trend     |
| `rsi_reversal`  | +0.38 | reversal  |

The six confirmations, evaluated on the trigger candle:

| # | confirmation      | condition (as seen from the trade's direction)        | delta |
|---|-------------------|-------------------------------------------------------|-------|
| 1 | EMA stack         | EMA9>EMA21>EMA50 with the trade                       | +0.10 |
|   |                   | fully inverted against the trade                      | -0.10 |
|   |                   | mixed, or EMA50 not warmed up yet                     |  0.00 |
| 2 | VWAP side         | close on the trade's side of session VWAP             | +0.08 |
|   |                   | close on the wrong side                               | -0.08 |
|   |                   | VWAP unavailable                                      |  0.00 |
| 3 | RSI not extreme   | RSI past 50 towards the trade, not yet extreme        | +0.05 |
|   |                   | RSI > 75 on a CE, or RSI < 25 on a PE (chasing)       | -0.07 |
|   |                   | otherwise                                             |  0.00 |
| 4 | Volume vs 20-bar  | >= 1.5x the mean of the 20 bars BEFORE this one       | +0.10 |
|   |                   | >= 1.0x                                               | +0.05 |
|   |                   | <  0.6x (dry)                                         | -0.08 |
|   |                   | index reports no volume, or < 5 prior bars            |  0.00 |
| 5 | ADX > 20          | trend setup, ADX >= 25                                | +0.08 |
|   |                   | trend setup, ADX > 20                                 | +0.04 |
|   |                   | trend setup, ADX <= 20 (breakouts fail in a range)    | -0.05 |
|   |                   | reversal setup, ADX <= 20 (chop is its habitat)       | +0.04 |
|   |                   | reversal setup, ADX >= 25 (fading a strong trend)     | -0.05 |
|   |                   | reversal setup, 20 < ADX < 25, or ADX unavailable     |  0.00 |
| 6 | Body ratio        | |c-o| / (h-l) >= 0.60 (conviction close)              | +0.07 |
|   |                   | >= 0.40                                               | +0.03 |
|   |                   | <  0.25 (doji — no one committed)                     | -0.06 |
|   |                   | otherwise                                             |  0.00 |

Best case is 0.50 + 0.48 = 0.98; worst is 0.38 - 0.44, clamped to 0.00. Every term
that contributes lands in `Signal.reasons` as a short English phrase with its delta,
e.g. `"EMA9>EMA21>EMA50, stacked with the trade (+0.10)"`. The UI prints those
verbatim, so they are written as sentences a person can audit.

STOP AND TARGET
---------------
Both start at `cfg.stop_pct` / `cfg.target_pct` and are then adjusted:

* **range regime (low ADX)** — both widened by 1.25x. Range premiums chop, and a
  tight stop in a range is a donation to the spread. Both legs move by the same
  factor on purpose: that keeps `stop/(stop+target)`, the break-even win rate the
  learner measures a bucket against, constant inside a bucket.
* **last 30 minutes before `cfg.flat_by`** — stop x0.8, target x0.6. There is no time
  left to recover, so take what the move gives and cut faster.
* The stop is then floored at `cfg.min_stop_pct` and capped at 0.90 (a long option
  cannot lose more than its premium), and the target is raised to at least the stop
  so the bot never knowingly risks more than it is trying to win.

Every adjustment appends its own sentence to `reasons`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

from . import clock
from . import indicators as ind
from .errors import ConfigError, DataError
from .types import Candle, Position, Signal

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .config import Config

logger = logging.getLogger(__name__)

__all__ = ["SETUPS", "BASE_WEIGHT", "TREND_SETUPS", "Strategy"]

#: Canonical order. `evaluate` iterates this, so the output order of equally scored
#: signals is deterministic (and therefore so is the backtest).
SETUPS: list[str] = [
    "orb",
    "vwap_reclaim",
    "ema_cross",
    "supertrend",
    "rsi_reversal",
    "pullback",
]

BASE_WEIGHT: dict[str, float] = {
    "orb": 0.50,
    "pullback": 0.50,
    "vwap_reclaim": 0.45,
    "ema_cross": 0.45,
    "supertrend": 0.42,
    "rsi_reversal": 0.38,
}

#: Trend-following setups want a high ADX; the other two want chop. Confirmation 5
#: flips sign on this split.
TREND_SETUPS: frozenset[str] = frozenset({"orb", "ema_cross", "supertrend", "pullback"})

# --- confirmation weights (the table in the module docstring) ---------------------
W_EMA_STACK_WITH = 0.10
W_EMA_STACK_AGAINST = -0.10
W_VWAP_WITH = 0.08
W_VWAP_AGAINST = -0.08
W_RSI_SUPPORTIVE = 0.05
W_RSI_EXTREME = -0.07
W_VOL_SURGE = 0.10
W_VOL_ABOVE = 0.05
W_VOL_DRY = -0.08
W_ADX_STRONG = 0.08
W_ADX_OK = 0.04
W_ADX_WEAK = -0.05
W_BODY_STRONG = 0.07
W_BODY_OK = 0.03
W_BODY_DOJI = -0.06

# --- confirmation thresholds -----------------------------------------------------
RSI_OVERBOUGHT = 75.0
RSI_OVERSOLD = 25.0
VOL_SURGE_MULT = 1.5
VOL_DRY_MULT = 0.6
VOL_LOOKBACK = 20
VOL_MIN_SAMPLES = 5
ADX_TREND = 20.0
ADX_STRONG = 25.0
BODY_STRONG = 0.60
BODY_OK = 0.40
BODY_DOJI = 0.25
VWAP_RECLAIM_MIN_BODY = 0.5
RSI_OVERSOLD_CROSS = 30.0
RSI_OVERBOUGHT_CROSS = 70.0

# --- stop / target adjustment ----------------------------------------------------
RANGE_WIDEN = 1.25
EOD_TAPER_MINUTES = 30
EOD_STOP_MULT = 0.8
EOD_TARGET_MULT = 0.6
#: A long option's worst case is a total loss of the premium. A "stop" wider than
#: this is not a stop, so cap it rather than letting a config typo through.
MAX_STOP_PCT = 0.90
MAX_TARGET_PCT = 3.00

# --- exit urgency ----------------------------------------------------------------
EXIT_SCORE_SUPERTREND = 0.85
EXIT_SCORE_ORB_FAIL = 0.80
EXIT_SCORE_OPPOSING_MIN = 0.60

#: Defaults mirror CONTRACTS.md section A exactly. Read through `_cfg` so a stub or a
#: partially-populated Config cannot raise AttributeError halfway down a money path.
_CFG_DEFAULTS: dict[str, object] = {
    "interval": "5minute",
    "or_minutes": 15,
    "stop_pct": 0.30,
    "target_pct": 0.50,
    "min_stop_pct": 0.12,
    "min_score": 0.55,
    "flat_by": "15:10",
    "enabled_setups": None,
}


def _term(text: str, delta: float) -> str:
    """Format one scoring term for the UI: `"ADX 27.4, trending (+0.08)"`."""
    return f"{text} ({delta:+.2f})"


def _px(value: float) -> str:
    """Format a price for a reason string."""
    return f"{value:,.2f}"


class _Trigger(NamedTuple):
    """A setup that fired: which option side to buy, and the sentence explaining it."""

    side: str
    reason: str


@dataclass(frozen=True)
class _Ctx:
    """Everything the triggers and the scorer need, computed once per `evaluate`.

    Indicator lists are index-aligned with `candles`, so `ema9[i]` is the EMA at
    `candles[i]`. All of them may hold `None` during warm-up.
    """

    underlying: str
    candles: list[Candle]
    i: int
    cur: Candle
    prev: Candle
    ema9: list[float | None]
    ema21: list[float | None]
    ema50: list[float | None]
    rsi: list[float | None]
    vwap: list[float | None]
    supertrend: list[tuple[float, int] | None]
    adx: list[float | None]
    or_high: float | None
    or_low: float | None
    or_label: str
    or_start_ts: int
    vol_avg: float | None
    regime: str

    @property
    def close(self) -> float:
        return float(self.cur["c"])

    @property
    def adx_now(self) -> float | None:
        return self.adx[self.i]

    @property
    def vwap_now(self) -> float | None:
        return self.vwap[self.i]

    @property
    def rsi_now(self) -> float | None:
        return self.rsi[self.i]

    def st_dir(self, idx: int) -> int | None:
        """Supertrend direction at `idx`: +1 up, -1 down, None while warming up."""
        v = self.supertrend[idx]
        return None if v is None else int(v[1])


class Strategy:
    """Turns closed candles into scored, auditable option signals.

    Holds no market state: construct once, call `evaluate` as often as you like.
    """

    def __init__(self, cfg: "Config") -> None:
        """Bind the config and resolve the candle interval once.

        Raises ConfigError on an interval the clock does not know, because every
        look-ahead check downstream is measured in that interval's seconds — getting
        it wrong would silently re-admit forming candles.
        """
        self.cfg = cfg
        interval = str(self._cfg("interval"))
        try:
            self.interval_minutes: int = clock.interval_minutes(interval)
        except ValueError as exc:
            raise ConfigError(
                f"config.interval {interval!r} is not a supported candle interval"
            ) from exc
        self.interval: str = interval
        self.interval_s: int = self.interval_minutes * 60

        raw = self._cfg("enabled_setups")
        wanted = list(raw) if raw else list(SETUPS)
        unknown = [s for s in wanted if s not in BASE_WEIGHT]
        if unknown:
            logger.warning(
                "ignoring unknown setup(s) in config.enabled_setups: %s (known: %s)",
                ", ".join(sorted(unknown)),
                ", ".join(SETUPS),
            )
        #: Canonical order, so the signal list is deterministic.
        self.enabled: list[str] = [s for s in SETUPS if s in set(wanted)]
        if not self.enabled:
            logger.warning("no known setups enabled — evaluate() will return nothing")

        self._triggers: dict[str, Callable[[_Ctx], _Trigger | None]] = {
            "orb": self._t_orb,
            "vwap_reclaim": self._t_vwap_reclaim,
            "ema_cross": self._t_ema_cross,
            "supertrend": self._t_supertrend,
            "rsi_reversal": self._t_rsi_reversal,
            "pullback": self._t_pullback,
        }

    # ---------------------------------------------------------------- config ----

    def _cfg(self, key: str) -> object:
        """Read a config field, falling back to the CONTRACTS.md default."""
        return getattr(self.cfg, key, _CFG_DEFAULTS[key])

    def _cfg_float(self, key: str) -> float:
        return float(self._cfg(key))  # type: ignore[arg-type]

    # ------------------------------------------------------------ public API ----

    def evaluate(
        self,
        underlying: str,
        candles: list[Candle],
        *,
        now_ts: int | None = None,
    ) -> list[Signal]:
        """Evaluate every enabled setup against the LAST CLOSED candle.

        Returns 0..n BUY signals, highest score first, at most one per
        `(underlying, setup)` for this candle. Never looks at a forming candle: when
        `now_ts` is given, any trailing candle whose `[ts, ts+interval)` window has
        not yet elapsed is dropped first.

        Raises DataError if the series is out of order or self-contradictory —
        trading on scrambled candles is worse than not trading.
        """
        closed = self._closed_only(candles, now_ts)
        if len(closed) < 2:
            # Every trigger compares the current candle with the previous one.
            return []

        ctx = self._context(underlying, closed)
        out: list[Signal] = []
        for setup in self.enabled:
            trig = self._triggers[setup](ctx)
            if trig is None:
                continue
            score, reasons = self._score(setup, trig, ctx)
            stop_pct, target_pct, risk_reasons = self._risk_pcts(ctx)
            reasons.extend(risk_reasons)
            out.append(
                Signal(
                    setup=setup,
                    side=trig.side,
                    action="BUY",
                    underlying=underlying,
                    ts=int(ctx.cur["ts"]),
                    spot=ctx.close,
                    score=score,
                    reasons=reasons,
                    bucket=Signal.make_bucket(setup, trig.side, int(ctx.cur["ts"]), ctx.regime),
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                )
            )

        # Highest confluence first; setup name breaks ties so the order is stable.
        out.sort(key=lambda s: (-s.score, s.setup))
        if out:
            logger.info(
                "%s %s: %d signal(s) on the %s candle — %s",
                underlying,
                self.interval,
                len(out),
                clock.to_ist(int(ctx.cur["ts"])).strftime("%H:%M"),
                ", ".join(f"{s.setup}/{s.side} {s.score:.2f}" for s in out),
            )
        return out

    def exit_signal(
        self,
        position: Position,
        candles: list[Candle],
        *,
        now_ts: int | None = None,
    ) -> Signal | None:
        """An invalidation on the underlying that should close `position` now.

        Returns a `SELL` Signal when the thesis is dead: the supertrend flips against
        the position, an ORB entry closes back inside the opening range, or an
        opposing setup fires. `side` names the contract being sold (the CE or PE we
        hold), not a new directional view, and `bucket` is the entry's bucket so the
        learner credits the exit to the trade that opened it. `score` is urgency, not
        confluence.

        This is the "if a sell signal is generated, sell that 1 lot" half of the bot.
        It is advisory: the risk gate's stop, target, trail and `flat_by` checks run
        independently and are not replaced by this.
        """
        if not position.is_open:
            return None

        closed = self._closed_only(candles, now_ts)
        if len(closed) < 2:
            return None

        last_ts = int(closed[-1]["ts"])
        # A candle that closed before we entered cannot invalidate the entry. Without
        # this guard an entry taken mid-bar would be exited by its own trigger bar.
        if last_ts + self.interval_s <= int(position.entry_ts):
            return None

        held = position.leg.side
        if held not in ("CE", "PE"):
            raise DataError(f"position {position.id} has an unknown side {held!r}")
        opposite = "PE" if held == "CE" else "CE"
        bullish = held == "CE"

        ctx = self._context(position.leg.underlying, closed)
        reasons: list[str] = []
        urgency = 0.0

        # 1. Supertrend flipped against the position on this candle.
        cur_dir = ctx.st_dir(ctx.i)
        prev_dir = ctx.st_dir(ctx.i - 1)
        adverse = -1 if bullish else 1
        if cur_dir is not None and prev_dir is not None and prev_dir != cur_dir and cur_dir == adverse:
            level = ctx.supertrend[ctx.i]
            level_txt = _px(level[0]) if level is not None else "n/a"
            reasons.append(
                f"Supertrend flipped {'up to down' if bullish else 'down to up'} at "
                f"{level_txt} on the {clock.to_ist(last_ts).strftime('%H:%M')} candle — "
                f"the trend that justified this {held} is gone."
            )
            urgency = max(urgency, EXIT_SCORE_SUPERTREND)

        # 2. An ORB entry that closed back inside the opening range: the breakout failed.
        if position.signal_setup == "orb":
            if bullish and ctx.or_high is not None and ctx.close <= ctx.or_high:
                reasons.append(
                    f"Price closed back inside the {ctx.or_label} opening range "
                    f"({_px(ctx.close)} is at or below the range high {_px(ctx.or_high)}) — "
                    f"the breakout failed."
                )
                urgency = max(urgency, EXIT_SCORE_ORB_FAIL)
            elif (not bullish) and ctx.or_low is not None and ctx.close >= ctx.or_low:
                reasons.append(
                    f"Price closed back inside the {ctx.or_label} opening range "
                    f"({_px(ctx.close)} is at or above the range low {_px(ctx.or_low)}) — "
                    f"the breakdown failed."
                )
                urgency = max(urgency, EXIT_SCORE_ORB_FAIL)

        # 3. An enabled setup fired in the opposite direction.
        for setup in self.enabled:
            trig = self._triggers[setup](ctx)
            if trig is None or trig.side != opposite:
                continue
            score, _ = self._score(setup, trig, ctx)
            reasons.append(
                f"Opposing {setup} setup fired for a {opposite} at confluence "
                f"{score:.2f}: {trig.reason}"
            )
            urgency = max(urgency, EXIT_SCORE_OPPOSING_MIN, score)

        if not reasons:
            return None

        stop_pct, target_pct = self._position_pcts(position)
        reasons.insert(
            0,
            f"SELL {position.leg.label} (1 lot): the underlying invalidated the "
            f"{position.signal_setup or 'entry'} setup.",
        )
        logger.info(
            "exit signal for %s (%s): %s",
            position.leg.label,
            position.id,
            reasons[1] if len(reasons) > 1 else "",
        )
        return Signal(
            setup=position.signal_setup or "exit",
            side=held,
            action="SELL",
            underlying=position.leg.underlying,
            ts=last_ts,
            spot=ctx.close,
            score=max(0.0, min(1.0, urgency)),
            reasons=reasons,
            bucket=position.bucket,
            stop_pct=stop_pct,
            target_pct=target_pct,
        )

    # ------------------------------------------------------------- internals ----

    def _closed_only(self, candles: Sequence[Candle], now_ts: int | None) -> list[Candle]:
        """Validate the series and strip any candle that has not finished forming.

        A candle stamped `ts` is closed only once `now_ts >= ts + interval`. Anything
        still inside its window is dropped with a warning: reading it would be
        look-ahead bias, and the same function runs live and in the backtest.
        """
        if not candles:
            return []
        self._check_series(candles)
        out = list(candles)
        if now_ts is None:
            # The caller guarantees the last element is closed (CONTRACTS.md E).
            return out

        now = int(now_ts)
        while out:
            ts = int(out[-1]["ts"])
            window_end = ts + self.interval_s
            if window_end <= now:
                break
            dropped = out.pop()
            logger.warning(
                "dropping the %s candle at %s: its window ends at %s but now is %s — "
                "it is still forming, and reading it would be look-ahead bias",
                self.interval,
                clock.to_ist(int(dropped["ts"])).strftime("%H:%M:%S"),
                clock.to_ist(window_end).strftime("%H:%M:%S"),
                clock.to_ist(now).strftime("%H:%M:%S"),
            )
        return out

    def _check_series(self, candles: Sequence[Candle]) -> None:
        """Reject a candle series that is out of order, duplicated or impossible.

        Scrambled candles would quietly corrupt every indicator and every trigger, so
        this raises instead of trading on them.
        """
        prev_ts: int | None = None
        for idx, c in enumerate(candles):
            try:
                ts = int(c["ts"])
                high = float(c["h"])
                low = float(c["l"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError(f"candle {idx} is malformed: {c!r}") from exc
            if prev_ts is not None and ts <= prev_ts:
                raise DataError(
                    f"candles are not strictly ascending: index {idx} is stamped "
                    f"{clock.to_ist(ts).isoformat()} but index {idx - 1} is "
                    f"{clock.to_ist(prev_ts).isoformat()}"
                )
            if high < low:
                raise DataError(
                    f"candle {idx} at {clock.to_ist(ts).isoformat()} has high {high} "
                    f"below low {low}"
                )
            prev_ts = ts

    def _context(self, underlying: str, candles: list[Candle]) -> _Ctx:
        """Compute every indicator once for this candle series.

        Indicators run over the whole series so they stay warmed up across days;
        the opening range is taken from the last day's candles only, because "the
        first 15 minutes" is only meaningful within one session.
        """
        closes = [float(c["c"]) for c in candles]
        i = len(candles) - 1

        or_minutes = int(self._cfg("or_minutes"))  # type: ignore[arg-type]
        day = clock.to_ist(int(candles[-1]["ts"])).date()
        day_candles = [c for c in candles if clock.to_ist(int(c["ts"])).date() == day]
        opening = ind.opening_range(day_candles, or_minutes)
        or_open = clock.session_open(day)
        or_label = (
            f"{or_open.strftime('%H:%M')}–"
            f"{clock.to_ist(clock.epoch_s(or_open) + or_minutes * 60).strftime('%H:%M')}"
        )

        if opening is None and int(candles[-1]["ts"]) >= clock.epoch_s(or_open) + or_minutes * 60:
            # Not fatal, but orb can never fire without it — say so once per pass so a
            # truncated candle window is debuggable instead of looking like a quiet market.
            logger.debug(
                "%s: no opening range for %s — the candle series does not reach back to "
                "the %s session open, so the orb setup is disabled for this pass",
                underlying,
                day.isoformat(),
                or_open.strftime("%H:%M"),
            )

        return _Ctx(
            underlying=underlying,
            candles=candles,
            i=i,
            cur=candles[i],
            prev=candles[i - 1],
            ema9=ind.ema(closes, 9),
            ema21=ind.ema(closes, 21),
            ema50=ind.ema(closes, 50),
            rsi=ind.rsi(closes, 14),
            vwap=ind.vwap(candles),
            supertrend=ind.supertrend(candles, 10, 2.0),
            adx=ind.adx(candles, 14),
            or_high=opening[0] if opening else None,
            or_low=opening[1] if opening else None,
            or_label=or_label,
            or_start_ts=clock.epoch_s(or_open),
            vol_avg=self._volume_baseline(candles),
            regime=ind.regime(candles),
        )

    def _volume_baseline(self, candles: list[Candle]) -> float | None:
        """Mean volume of the up-to-20 bars BEFORE the trigger bar.

        Excluding the trigger bar is the point: including its own (often large)
        volume in the baseline it is compared against would wash the surge out.
        Returns None when there is no usable baseline — index candles frequently
        report v=0, and a zero baseline must read as "unknown", not as "dry".
        """
        prior = [float(c.get("v", 0) or 0) for c in candles[:-1]][-VOL_LOOKBACK:]
        if len(prior) < VOL_MIN_SAMPLES:
            return None
        total = sum(prior)
        if total <= 0:
            return None
        return total / len(prior)  # len >= VOL_MIN_SAMPLES, so never a zero divide

    # --------------------------------------------------------------- triggers ---
    # Each returns a _Trigger when the LAST CLOSED candle satisfies the setup, else
    # None. CE and PE conditions are mutually exclusive in every one of them.

    def _t_orb(self, ctx: _Ctx) -> _Trigger | None:
        """First close outside the opening range, once the range is complete."""
        if ctx.or_high is None or ctx.or_low is None:
            return None
        or_minutes = int(self._cfg("or_minutes"))  # type: ignore[arg-type]
        breakout_from = ctx.or_start_ts + or_minutes * 60
        if int(ctx.cur["ts"]) < breakout_from:
            return None  # still inside the range-building window

        # "First" is derived from the series, not from remembered state, so the live
        # loop and the backtest agree. Only the last closed candle may be the first.
        first_up: int | None = None
        first_dn: int | None = None
        for idx, c in enumerate(ctx.candles):
            if int(c["ts"]) < breakout_from:
                continue
            if clock.to_ist(int(c["ts"])).date() != clock.to_ist(int(ctx.cur["ts"])).date():
                continue
            close = float(c["c"])
            if first_up is None and close > ctx.or_high:
                first_up = idx
            if first_dn is None and close < ctx.or_low:
                first_dn = idx

        if first_up == ctx.i:
            return _Trigger(
                "CE",
                f"ORB long: the {self.interval} candle closed at {_px(ctx.close)}, "
                f"the first close above the {ctx.or_label} range high "
                f"{_px(ctx.or_high)}",
            )
        if first_dn == ctx.i:
            return _Trigger(
                "PE",
                f"ORB short: the {self.interval} candle closed at {_px(ctx.close)}, "
                f"the first close below the {ctx.or_label} range low "
                f"{_px(ctx.or_low)}",
            )
        return None

    def _t_vwap_reclaim(self, ctx: _Ctx) -> _Trigger | None:
        """Close back across session VWAP on a decisive candle."""
        vw_now = ctx.vwap[ctx.i]
        vw_prev = ctx.vwap[ctx.i - 1]
        if vw_now is None or vw_prev is None:
            return None
        body = ind.body_ratio(ctx.cur)
        if body < VWAP_RECLAIM_MIN_BODY:
            return None
        prev_close = float(ctx.prev["c"])
        if prev_close < vw_prev and ctx.close > vw_now:
            return _Trigger(
                "CE",
                f"VWAP reclaim: the previous candle closed {_px(prev_close)} below VWAP "
                f"{_px(vw_prev)} and this one closed {_px(ctx.close)} above VWAP "
                f"{_px(vw_now)} with a {body:.0%} body",
            )
        if prev_close > vw_prev and ctx.close < vw_now:
            return _Trigger(
                "PE",
                f"VWAP loss: the previous candle closed {_px(prev_close)} above VWAP "
                f"{_px(vw_prev)} and this one closed {_px(ctx.close)} below VWAP "
                f"{_px(vw_now)} with a {body:.0%} body",
            )
        return None

    def _t_ema_cross(self, ctx: _Ctx) -> _Trigger | None:
        """EMA9 crossing EMA21, confirmed by the correct side of VWAP."""
        e9, e9p = ctx.ema9[ctx.i], ctx.ema9[ctx.i - 1]
        e21, e21p = ctx.ema21[ctx.i], ctx.ema21[ctx.i - 1]
        vw = ctx.vwap[ctx.i]
        if None in (e9, e9p, e21, e21p) or vw is None:
            return None
        assert e9 is not None and e9p is not None and e21 is not None and e21p is not None
        if e9p <= e21p and e9 > e21 and ctx.close > vw:
            return _Trigger(
                "CE",
                f"EMA9 crossed above EMA21 ({_px(e9)} over {_px(e21)}) with the close "
                f"{_px(ctx.close)} above VWAP {_px(vw)}",
            )
        if e9p >= e21p and e9 < e21 and ctx.close < vw:
            return _Trigger(
                "PE",
                f"EMA9 crossed below EMA21 ({_px(e9)} under {_px(e21)}) with the close "
                f"{_px(ctx.close)} below VWAP {_px(vw)}",
            )
        return None

    def _t_supertrend(self, ctx: _Ctx) -> _Trigger | None:
        """Supertrend direction flipping on this candle."""
        cur = ctx.st_dir(ctx.i)
        prev = ctx.st_dir(ctx.i - 1)
        if cur is None or prev is None or cur == prev:
            return None
        band = ctx.supertrend[ctx.i]
        level = _px(band[0]) if band is not None else "n/a"
        if prev == -1 and cur == 1:
            return _Trigger(
                "CE",
                f"Supertrend flipped down to up, band now at {level} under the close "
                f"{_px(ctx.close)}",
            )
        if prev == 1 and cur == -1:
            return _Trigger(
                "PE",
                f"Supertrend flipped up to down, band now at {level} over the close "
                f"{_px(ctx.close)}",
            )
        return None

    def _t_rsi_reversal(self, ctx: _Ctx) -> _Trigger | None:
        """RSI(14) crossing back out of oversold (30) or overbought (70)."""
        r, rp = ctx.rsi[ctx.i], ctx.rsi[ctx.i - 1]
        if r is None or rp is None:
            return None
        if rp <= RSI_OVERSOLD_CROSS < r:
            return _Trigger(
                "CE",
                f"RSI(14) crossed up through 30, from {rp:.1f} to {r:.1f} — oversold "
                f"pressure releasing",
            )
        if rp >= RSI_OVERBOUGHT_CROSS > r:
            return _Trigger(
                "PE",
                f"RSI(14) crossed down through 70, from {rp:.1f} to {r:.1f} — "
                f"overbought pressure releasing",
            )
        return None

    def _t_pullback(self, ctx: _Ctx) -> _Trigger | None:
        """A touch of EMA21 inside a stacked trend, then a break of the pullback bar."""
        e9, e21, e50 = ctx.ema9[ctx.i], ctx.ema21[ctx.i], ctx.ema50[ctx.i]
        e21p = ctx.ema21[ctx.i - 1]
        if None in (e9, e21, e50) or e21p is None:
            return None  # needs 50 bars of warm-up; says so via the EMA-stack reason
        assert e9 is not None and e21 is not None and e50 is not None
        prev_low = float(ctx.prev["l"])
        prev_high = float(ctx.prev["h"])
        if e9 > e21 > e50 and prev_low <= e21p and ctx.close > prev_high:
            return _Trigger(
                "CE",
                f"Pullback long: EMA9 {_px(e9)} > EMA21 {_px(e21)} > EMA50 {_px(e50)}, "
                f"the previous candle dipped to {_px(prev_low)} at EMA21 {_px(e21p)}, "
                f"and this close {_px(ctx.close)} took out its high {_px(prev_high)}",
            )
        if e9 < e21 < e50 and prev_high >= e21p and ctx.close < prev_low:
            return _Trigger(
                "PE",
                f"Pullback short: EMA9 {_px(e9)} < EMA21 {_px(e21)} < EMA50 {_px(e50)}, "
                f"the previous candle rallied to {_px(prev_high)} at EMA21 {_px(e21p)}, "
                f"and this close {_px(ctx.close)} broke its low {_px(prev_low)}",
            )
        return None

    # ------------------------------------------------------------------ score ---

    def _score(self, setup: str, trig: _Trigger, ctx: _Ctx) -> tuple[float, list[str]]:
        """Base weight plus the six confirmations, clamped to [0, 1].

        Returns the score and the reasons list. Every term that moved the score is in
        the list, phrased for a human, because the UI shows these verbatim and the
        user is being asked to trust them with real money.
        """
        bullish = trig.side == "CE"
        base = BASE_WEIGHT[setup]
        score = base
        reasons: list[str] = [_term(trig.reason, base)]

        for delta, text in (
            self._c_ema_stack(ctx, bullish),
            self._c_vwap_side(ctx, bullish),
            self._c_rsi(ctx, bullish),
            self._c_volume(ctx),
            self._c_adx(ctx, setup),
            self._c_body(ctx),
        ):
            if text:
                reasons.append(_term(text, delta))
            score += delta

        clamped = max(0.0, min(1.0, score))
        if clamped != score:
            reasons.append(f"Raw score {score:.2f} clamped to {clamped:.2f}.")
        reasons.append(
            f"Confluence score {clamped:.2f}; this bot only trades "
            f"{self._cfg_float('min_score'):.2f} and above."
        )
        return clamped, reasons

    def _c_ema_stack(self, ctx: _Ctx, bullish: bool) -> tuple[float, str]:
        """Confirmation 1: is the 9/21/50 EMA stack lined up with the trade?"""
        e9, e21, e50 = ctx.ema9[ctx.i], ctx.ema21[ctx.i], ctx.ema50[ctx.i]
        if e9 is None or e21 is None or e50 is None:
            return 0.0, "EMA50 needs 50 candles of history and is not warmed up yet"
        if bullish:
            if e9 > e21 > e50:
                return W_EMA_STACK_WITH, "EMA9>EMA21>EMA50, stacked with the trade"
            if e9 < e21 < e50:
                return W_EMA_STACK_AGAINST, "EMA9<EMA21<EMA50, stacked against this call"
        else:
            if e9 < e21 < e50:
                return W_EMA_STACK_WITH, "EMA9<EMA21<EMA50, stacked with the trade"
            if e9 > e21 > e50:
                return W_EMA_STACK_AGAINST, "EMA9>EMA21>EMA50, stacked against this put"
        return 0.0, "EMA stack is mixed, neither confirming nor denying"

    def _c_vwap_side(self, ctx: _Ctx, bullish: bool) -> tuple[float, str]:
        """Confirmation 2: which side of session VWAP did the candle close on?"""
        vw = ctx.vwap_now
        if vw is None:
            return 0.0, "VWAP is not available for this session yet"
        if (ctx.close > vw) == bullish and ctx.close != vw:
            return W_VWAP_WITH, (
                f"Close {_px(ctx.close)} is {'above' if bullish else 'below'} VWAP "
                f"{_px(vw)}, the right side for this trade"
            )
        if ctx.close == vw:
            return 0.0, f"Close {_px(ctx.close)} is sitting exactly on VWAP"
        return W_VWAP_AGAINST, (
            f"Close {_px(ctx.close)} is on the wrong side of VWAP {_px(vw)}"
        )

    def _c_rsi(self, ctx: _Ctx, bullish: bool) -> tuple[float, str]:
        """Confirmation 3: RSI leaning the right way, and not already exhausted."""
        r = ctx.rsi_now
        if r is None:
            return 0.0, "RSI(14) is not warmed up yet"
        if bullish and r > RSI_OVERBOUGHT:
            return W_RSI_EXTREME, f"RSI(14) at {r:.1f} is overbought — this call is chasing"
        if (not bullish) and r < RSI_OVERSOLD:
            return W_RSI_EXTREME, f"RSI(14) at {r:.1f} is oversold — this put is chasing"
        if bullish and 50.0 < r <= RSI_OVERBOUGHT:
            return W_RSI_SUPPORTIVE, f"RSI(14) at {r:.1f} is bullish without being extreme"
        if (not bullish) and RSI_OVERSOLD <= r < 50.0:
            return W_RSI_SUPPORTIVE, f"RSI(14) at {r:.1f} is bearish without being extreme"
        return 0.0, f"RSI(14) at {r:.1f} is neutral for this trade"

    def _c_volume(self, ctx: _Ctx) -> tuple[float, str]:
        """Confirmation 4: this candle's volume against the prior 20 bars."""
        avg = ctx.vol_avg
        vol = float(ctx.cur.get("v", 0) or 0)
        if avg is None or avg <= 0:
            # Index candles often carry v=0. Scoring that as "dry" would dock every
            # index signal 0.08 for a data quirk, so it reads as unknown instead.
            return 0.0, "No usable volume baseline for this index — volume ignored"
        ratio = vol / avg  # avg > 0 checked above
        if ratio >= VOL_SURGE_MULT:
            return W_VOL_SURGE, f"Volume {ratio:.1f}x the 20-bar average — a real surge"
        if ratio >= 1.0:
            return W_VOL_ABOVE, f"Volume {ratio:.1f}x the 20-bar average"
        if ratio < VOL_DRY_MULT:
            return W_VOL_DRY, f"Volume only {ratio:.1f}x the 20-bar average — nobody is here"
        return 0.0, f"Volume {ratio:.1f}x the 20-bar average, unremarkable"

    def _c_adx(self, ctx: _Ctx, setup: str) -> tuple[float, str]:
        """Confirmation 5: ADX > 20, signed by whether the setup needs a trend.

        A breakout wants trend; a VWAP or RSI reversal wants chop. The same ADX
        reading therefore helps one family and hurts the other.
        """
        a = ctx.adx_now
        if a is None:
            return 0.0, "ADX(14) is not warmed up yet"
        trend_setup = setup in TREND_SETUPS
        if trend_setup:
            if a >= ADX_STRONG:
                return W_ADX_STRONG, f"ADX(14) at {a:.1f} — a genuine trend to ride"
            if a > ADX_TREND:
                return W_ADX_OK, f"ADX(14) at {a:.1f} is above 20, trend forming"
            return W_ADX_WEAK, (
                f"ADX(14) at {a:.1f} is below 20 — this is a range, and breakouts fail "
                f"in a range"
            )
        if a <= ADX_TREND:
            return W_ADX_OK, f"ADX(14) at {a:.1f} is below 20 — a range is where a reversal works"
        if a >= ADX_STRONG:
            return W_ADX_WEAK, (
                f"ADX(14) at {a:.1f} is a strong trend — fading it is the expensive way "
                f"to be right"
            )
        return 0.0, f"ADX(14) at {a:.1f} is between 20 and 25, no edge either way"

    def _c_body(self, ctx: _Ctx) -> tuple[float, str]:
        """Confirmation 6: how much of the candle's range is body, not wick."""
        body = ind.body_ratio(ctx.cur)
        if body >= BODY_STRONG:
            return W_BODY_STRONG, f"Candle body is {body:.0%} of its range — decisive close"
        if body >= BODY_OK:
            return W_BODY_OK, f"Candle body is {body:.0%} of its range"
        if body < BODY_DOJI:
            return W_BODY_DOJI, f"Candle body is only {body:.0%} of its range — a doji, no conviction"
        return 0.0, f"Candle body is {body:.0%} of its range, unconvincing but not a doji"

    # --------------------------------------------------- stop / target sizing ---

    def _risk_pcts(self, ctx: _Ctx) -> tuple[float, float, list[str]]:
        """Stop and target as fractions of premium, adjusted for regime and clock.

        Widened in a range (chop eats tight stops) and tightened in the run-up to
        `flat_by` (no time left to recover). The stop is floored at `min_stop_pct`
        because a stop inside the bid-ask spread is triggered by the spread alone.
        """
        stop = self._cfg_float("stop_pct")
        target = self._cfg_float("target_pct")
        reasons: list[str] = []

        if ctx.regime == "range":
            stop *= RANGE_WIDEN
            target *= RANGE_WIDEN
            adx_txt = f"{ctx.adx_now:.1f}" if ctx.adx_now is not None else "low"
            reasons.append(
                f"Range regime (ADX {adx_txt}): stop widened to {stop:.0%} and target to "
                f"{target:.0%} of premium, both by the same 1.25x so the break-even win "
                f"rate stays {stop / (stop + target) if (stop + target) > 0 else 0.5:.0%}."
            )

        minutes_left = self._minutes_to_flat(int(ctx.cur["ts"]))
        if minutes_left is not None and minutes_left <= EOD_TAPER_MINUTES:
            stop *= EOD_STOP_MULT
            target *= EOD_TARGET_MULT
            flat_by = str(self._cfg("flat_by"))
            if minutes_left >= 0:
                when = f"{minutes_left} minutes before the {flat_by} flatten"
            else:
                when = f"already past the {flat_by} flatten"
            reasons.append(
                f"{when.capitalize()}: stop tightened to {stop:.0%} and target to "
                f"{target:.0%} — there is no time left for this to come back."
            )

        min_stop = self._cfg_float("min_stop_pct")
        if stop < min_stop:
            reasons.append(
                f"Stop floored at the {min_stop:.0%} minimum (a stop tighter than that "
                f"is triggered by the bid-ask spread alone, not by the market)."
            )
            stop = min_stop
        if stop > MAX_STOP_PCT:
            reasons.append(
                f"Stop capped at {MAX_STOP_PCT:.0%}: a bought option cannot lose more "
                f"than its premium."
            )
            stop = MAX_STOP_PCT

        target = min(target, MAX_TARGET_PCT)
        if target < stop:
            # Prefer the safe branch: never knowingly risk more than the trade aims
            # to win. Raising the target keeps the break-even win rate at or under 50%.
            reasons.append(
                f"Target raised from {target:.0%} to the {stop:.0%} stop so the trade "
                f"is not risking more than it is trying to make."
            )
            target = stop

        return round(stop, 4), round(target, 4), reasons

    def _minutes_to_flat(self, ts: int) -> int | None:
        """Minutes from this candle's close to `cfg.flat_by`, on the candle's own date.

        Uses the candle's date, not today's, so a backtest taper fires on the right
        bars. Returns None when `flat_by` is unparseable, which leaves the stop and
        target untapered rather than silently mangled.
        """
        raw = str(self._cfg("flat_by"))
        try:
            flat_t = clock.parse_hhmm(raw)
        except (ValueError, TypeError):
            logger.warning("config.flat_by %r is not HH:MM — skipping the end-of-day taper", raw)
            return None
        candle_close = clock.to_ist(ts + self.interval_s)
        flat_at = clock.at(candle_close.date(), flat_t)
        return int((flat_at - candle_close).total_seconds() // 60)

    def _position_pcts(self, position: Position) -> tuple[float, float]:
        """Recover the stop/target fractions a position was opened with.

        Derived from the stored prices so an exit signal reports the trade's own
        risk profile, not whatever the config says now. Falls back to the config when
        the entry price is unusable.
        """
        entry = float(position.entry_price)
        if entry <= 0:
            return self._cfg_float("stop_pct"), self._cfg_float("target_pct")
        stop = (entry - float(position.stop_price)) / entry
        target = (float(position.target_price) - entry) / entry
        if not 0.0 < stop <= 1.0:
            stop = self._cfg_float("stop_pct")
        if not 0.0 < target <= MAX_TARGET_PCT:
            target = self._cfg_float("target_pct")
        return round(stop, 4), round(target, 4)
