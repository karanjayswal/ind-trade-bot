"""Outcome-gated setups — the bot learning from its own mistakes.

No machine learning. Every judgement in this module is a statistic the UI can print
as one plain sentence, because a gate the user cannot understand is a gate they will
switch off.

The mechanism: group closed trades by learning bucket (setup x side x time-of-day x
regime), take the LOWER BOUND of the Wilson score interval on each bucket's win rate,
and compare that honest worst case against the bucket's own break-even win rate. A
bucket that cannot clear break-even gets its signal score multiplied down; a bucket
with enough evidence of losing money gets multiplied to zero and stops firing.

Why Wilson and not the textbook `p +/- z*sqrt(p(1-p)/n)`: the normal approximation
falls apart exactly where this bot lives. At n=12 it is already unreliable, it happily
returns a negative lower bound, and at p=0 or p=1 it returns a zero-width interval
that claims certainty from a handful of trades. Wilson stays inside [0, 1] and stays
honest at n=12, which is the whole point of gating on it.

Why the break-even reference comes from the trades themselves and not from the current
config: the stop and target in force when a trade was taken are what decided the win
rate it needed. Re-judging old trades with today's numbers would let a config tweak
silently unblock a setup that lost money under different rules.

Modes are kept apart by default (`learn_across_modes=False`): a losing streak in paper
must not stop a live setup from firing unless the user explicitly asks for that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from . import clock
from .errors import ConfigError, DataError, IndbotError
from .types import Signal

if TYPE_CHECKING:  # pragma: no cover - type-checker only, no runtime import cycle
    from .config import Config
    from .journal import Journal

logger = logging.getLogger(__name__)

#: The four states and the score multiplier each one applies. Fixed by CONTRACTS.md H.
STATES: tuple[str, ...] = ("promoted", "neutral", "demoted", "blocked")
STATE_MULTIPLIERS: dict[str, float] = {
    "promoted": 1.15,
    "neutral": 1.0,
    "demoted": 0.75,
    "blocked": 0.0,
}

#: How far clear of break-even the worst case must be to promote, and how far below to block.
PROMOTE_MARGIN: float = 0.10
BLOCK_MARGIN: float = 0.10

DEFAULT_CONFIDENCE: float = 0.90
#: A "confidence" under 0.5 gives a NEGATIVE z, which turns the lower bound into an
#: upper bound and would promote losing setups. Clamp rather than trade on that.
MIN_CONFIDENCE: float = 0.50
MAX_CONFIDENCE: float = 0.999
#: 30% stop / 50% target — the config default, used only when nothing better is known.
DEFAULT_BREAK_EVEN: float = 0.375
DEFAULT_STALE_AFTER_S: float = 300.0
#: After a journal failure, wait this long before an automatic retry, so a broken
#: database is not re-queried once per bucket lookup inside a single tick.
_RETRY_AFTER_S: float = 30.0

#: Bounds a derived stop/target fraction must fall inside to be believed.
_PLAUSIBLE_STOP: tuple[float, float] = (0.02, 0.95)
_PLAUSIBLE_TARGET: tuple[float, float] = (0.01, 5.0)
#: How many journal rows to scan when deriving per-bucket stop/target averages.
_TRADE_SCAN_LIMIT: int = 5000

# --------------------------------------------------------------------------------
# Statistics — pure functions, unit-testable without a journal or a config
# --------------------------------------------------------------------------------

#: Exact z for the tail probabilities the common confidence levels need, one-sided:
#: 0.80 / 0.90 / 0.95 / 0.99, plus the two-sided equivalents (0.975, 0.995, 0.999).
_Z_BY_TAIL: dict[float, float] = {
    0.80: 0.8416212336,
    0.90: 1.2815515655,
    0.95: 1.6448536270,
    0.975: 1.9599639845,
    0.99: 2.3263478740,
    0.995: 2.5758293035,
    0.999: 3.0902323062,
}

# Acklam's rational approximation to the inverse normal CDF (|error| < 1.2e-9 before
# the refinement step below). Used for any confidence level not in the table.
_ACK_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACK_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACK_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACK_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACK_P_LOW = 0.02425


def inverse_normal_cdf(p: float) -> float:
    """The standard-normal quantile z such that P(Z <= z) == p, for 0 < p < 1.

    Acklam's approximation plus one Halley refinement against math.erfc, which takes
    it to full double precision. Written out so the Wilson bound has no dependency
    on scipy and can be checked against a table in a unit test.
    """
    if not 0.0 < p < 1.0:
        raise ConfigError(f"inverse_normal_cdf needs 0 < p < 1, got {p!r}", p=p)
    if p < _ACK_P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]) / (
            (((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3]) * q + 1.0
        )
    elif p <= 1.0 - _ACK_P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r + _ACK_A[3]) * r + _ACK_A[4]) * r + _ACK_A[5]) * q / (
            ((((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r + _ACK_B[3]) * r + _ACK_B[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]) / (
            (((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3]) * q + 1.0
        )
    # One Halley step on the residual of the true CDF.
    err = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    denom = 1.0 + x * u / 2.0
    return x - (u / denom if denom != 0 else 0.0)  # guard: denom is ~1 in practice


def z_for_confidence(confidence: float, *, two_sided: bool = False) -> float:
    """The z multiplier for a ONE-SIDED lower confidence bound at `confidence`.

    One-sided is the right reading of "the win rate is at least X with 90%
    confidence", which is the only question this gate asks. `two_sided=True` uses
    z at (1+confidence)/2 instead, which is strictly more conservative (it both
    promotes less and blocks more).
    """
    if not 0.0 < confidence < 1.0:
        raise ConfigError(
            f"gate_confidence must be strictly between 0 and 1, got {confidence!r}",
            gate_confidence=confidence,
        )
    tail = (1.0 + confidence) / 2.0 if two_sided else confidence
    exact = _Z_BY_TAIL.get(round(tail, 6))
    if exact is not None:
        return exact
    return inverse_normal_cdf(tail)


def wilson_lower_bound(
    wins: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    two_sided: bool = False,
) -> float:
    """Lower bound of the Wilson score interval on a bucket's win rate, in [0, 1].

    Reads as: "with `confidence` certainty the true win rate of this bucket is at
    least this much". An empty bucket returns 0.0 — no evidence is not evidence of
    skill, and the sample-size rule above keeps that from blocking anything.
    """
    n = int(n)
    if n <= 0:
        return 0.0
    wins = max(0, min(int(wins), n))  # a bucket cannot win more trades than it took
    z = z_for_confidence(confidence, two_sided=two_sided)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    if denom <= 0:  # unreachable for z >= 0; guarded because it is a divisor
        return 0.0
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(max(0.0, (p * (1.0 - p) / n) + z2 / (4.0 * n * n)))
    return max(0.0, min(1.0, (centre - margin) / denom))


def break_even_win_rate(stop_pct: float, target_pct: float) -> float:
    """The win rate a stop/target pair needs just to break even: stop/(stop+target).

    A 30% stop with a 50% target needs 37.5%. A degenerate pair returns 0.5 rather
    than dividing by zero: 0.5 is the neutral reference that neither promotes nor
    blocks, which is the safe branch when the trade's own risk plan is unknown.
    """
    stop = float(stop_pct)
    target = float(target_pct)
    total = stop + target
    if total <= 0 or stop < 0 or target < 0:
        return 0.5
    return min(1.0, max(0.0, stop / total))


def classify(
    n: int,
    expectancy: float,
    lower_bound: float,
    break_even: float,
    *,
    min_samples: int,
) -> tuple[str, float]:
    """Apply the four gate rules and return (state, score multiplier).

    Tested most-specific first: `blocked` is a strict subset of `demoted`
    (lower_bound < be - 0.10 implies lower_bound < be, and both want expectancy < 0),
    so evaluating `demoted` first — the order CONTRACTS.md happens to list them in —
    would mean nothing ever reached `blocked`.
    """
    min_samples = max(1, int(min_samples))
    if n < min_samples:
        return "neutral", STATE_MULTIPLIERS["neutral"]
    if lower_bound > break_even + PROMOTE_MARGIN:
        return "promoted", STATE_MULTIPLIERS["promoted"]
    if n >= 2 * min_samples and expectancy < 0 and lower_bound < break_even - BLOCK_MARGIN:
        return "blocked", STATE_MULTIPLIERS["blocked"]
    if lower_bound < break_even and expectancy < 0:
        return "demoted", STATE_MULTIPLIERS["demoted"]
    return "neutral", STATE_MULTIPLIERS["neutral"]


# --------------------------------------------------------------------------------
# Formatting — the note is the product, so it gets real care
# --------------------------------------------------------------------------------

_MINUS = "−"  # true minus sign, as in the CONTRACTS.md example "average -Rs.212"


def _pct(fraction: float) -> str:
    """0.375 -> "37.5%", 0.25 -> "25%". One decimal, trailing zero dropped."""
    return f"{fraction * 100.0:.1f}".rstrip("0").rstrip(".") + "%"


def _group_inr(whole: int) -> str:
    """Indian digit grouping: 1234567 -> "12,34,567"."""
    s = str(whole)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def _inr(amount: float, *, signed: bool = True) -> str:
    """Rupees for a sentence: -212.4 -> "-Rs.212", 310.0 -> "+Rs.310", 0.2 -> "Rs.0"."""
    whole = int(round(abs(amount)))
    body = "₹" + _group_inr(whole)
    if not signed or whole == 0:
        return body
    return (_MINUS if amount < 0 else "+") + body


def _plural(count: int, word: str) -> str:
    """"1 trade" / "2 trades"."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def build_note(
    state: str,
    *,
    n: int,
    wins: int,
    win_rate: float,
    expectancy: float,
    lower_bound: float,
    break_even: float,
    confidence: float,
    min_samples: int,
    mode: str | None = None,
) -> str:
    """One plain sentence, always carrying the numbers, shown verbatim in the UI.

    This text is the only explanation the user gets for why a setup stopped firing,
    so it names the sample size, the win rate, the break-even it is measured against
    and the money, in that order.
    """
    mode_word = f"{mode} " if mode else ""
    if state == "neutral" and n < max(1, int(min_samples)):
        missing = max(0, max(1, int(min_samples)) - n)
        if n <= 0:
            return (
                f"No trades yet: this setup needs {_plural(max(1, int(min_samples)), f'{mode_word}trade'.strip())} "
                f"before it is judged."
            )
        return (
            f"Still learning: {wins} wins in {_plural(n, f'{mode_word}trade'.strip())} ({_pct(win_rate)}), "
            f"{missing} more before this setup is judged."
        )
    if state == "promoted":
        return (
            f"Promoted: {wins} wins in {_plural(n, 'trade')} ({_pct(win_rate)}), at least {_pct(lower_bound)} "
            f"at {_pct(confidence)} confidence against a {_pct(break_even)} break-even, "
            f"average {_inr(expectancy)} a trade."
        )
    if state == "blocked":
        return (
            f"Blocked: {wins} wins in {_plural(n, 'trade')} ({_pct(win_rate)}), "
            f"break-even needs {_pct(break_even)}, average {_inr(expectancy)} a trade."
        )
    if state == "demoted":
        return (
            f"Demoted: {wins} wins in {_plural(n, 'trade')} ({_pct(win_rate)}), worst case {_pct(lower_bound)} "
            f"is under the {_pct(break_even)} break-even, average {_inr(expectancy)} a trade."
        )
    return (
        f"Neutral: {wins} wins in {_plural(n, 'trade')} ({_pct(win_rate)}), worst case {_pct(lower_bound)} "
        f"against a {_pct(break_even)} break-even, average {_inr(expectancy)} a trade."
    )


# --------------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketVerdict:
    """What the learner has concluded about one bucket, and why.

    Frozen: a verdict is a statement about trades that have already closed. The
    fields after `note` are additions on top of the CONTRACTS.md shape, all
    defaulted, so positional construction of the contract fields still works.
    """

    bucket: str
    n: int
    wins: int
    win_rate: float
    expectancy: float
    lower_bound: float
    state: str
    multiplier: float
    note: str
    # --- additions, keyword/defaulted ---
    losses: int = 0
    break_even: float = DEFAULT_BREAK_EVEN
    be_source: str = "config"
    avg_stop_pct: float = 0.0
    avg_target_pct: float = 0.0
    confidence: float = DEFAULT_CONFIDENCE
    z: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    trades_needed: int = 0
    mode: str | None = None
    applied: bool = True

    @property
    def is_blocked(self) -> bool:
        """True when this bucket's signals are multiplied to zero."""
        return self.state == "blocked"

    @property
    def setup(self) -> str:
        """The setup name out of the bucket key, for grouping in the UI."""
        return self.bucket.split(":", 1)[0] if self.bucket else ""

    @property
    def side(self) -> str:
        """"CE" or "PE" out of the bucket key."""
        parts = self.bucket.split(":")
        return parts[1] if len(parts) > 1 else ""

    @property
    def tod(self) -> str:
        """The time-of-day slice out of the bucket key."""
        parts = self.bucket.split(":")
        return parts[2] if len(parts) > 2 else ""

    @property
    def regime(self) -> str:
        """The market regime out of the bucket key."""
        parts = self.bucket.split(":")
        return parts[3] if len(parts) > 3 else ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for /api/leaderboard and the snapshot."""
        return {
            "bucket": self.bucket,
            "setup": self.setup,
            "side": self.side,
            "tod": self.tod,
            "regime": self.regime,
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "win_rate_pct": round(self.win_rate * 100.0, 1),
            "expectancy": round(self.expectancy, 2),
            "lower_bound": round(self.lower_bound, 4),
            "lower_bound_pct": round(self.lower_bound * 100.0, 1),
            "break_even": round(self.break_even, 4),
            "break_even_pct": round(self.break_even * 100.0, 1),
            "be_source": self.be_source,
            "avg_stop_pct": round(self.avg_stop_pct, 4),
            "avg_target_pct": round(self.avg_target_pct, 4),
            "state": self.state,
            "multiplier": self.multiplier,
            "note": self.note,
            "confidence": self.confidence,
            "z": round(self.z, 6),
            "gross_win": round(self.gross_win, 2),
            "gross_loss": round(self.gross_loss, 2),
            "trades_needed": self.trades_needed,
            "mode": self.mode,
            "applied": self.applied,
        }


# --------------------------------------------------------------------------------
# Small readers — journals and configs are written by other agents, so read defensively
# --------------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any) -> Any:
    """Read an attribute, falling back to `default` when it is absent or None."""
    value = getattr(obj, name, None)
    return default if value is None else value


def _as_float(value: Any) -> float | None:
    """Coerce a journal cell to float, or None when it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a journal cell to int, or `default` when it is not a number."""
    out = _as_float(value)
    return default if out is None else int(out)


def _first_float(row: dict[str, Any], *names: str) -> float | None:
    """The first of `names` present in `row` as a float."""
    for name in names:
        out = _as_float(row.get(name))
        if out is not None:
            return out
    return None


def _pcts_from_row(row: dict[str, Any]) -> tuple[float, float] | None:
    """Stop/target fractions a row states outright, if it states them."""
    stop = _first_float(row, "avg_stop_pct", "stop_pct")
    target = _first_float(row, "avg_target_pct", "target_pct")
    if stop is None or target is None:
        return None
    if not (_PLAUSIBLE_STOP[0] <= stop <= _PLAUSIBLE_STOP[1]):
        return None
    if not (_PLAUSIBLE_TARGET[0] <= target <= _PLAUSIBLE_TARGET[1]):
        return None
    return stop, target


def _pcts_from_prices(row: dict[str, Any], *, stop_floor: float) -> tuple[float, float] | None:
    """Stop/target fractions implied by a trade's entry, stop and target prices.

    Returns None rather than a guess whenever the arithmetic is not trustworthy.
    A stop under half of `min_stop_pct` almost certainly means the row holds a
    TRAILED stop written over the original, which would understate the risk the
    trade was taken with and make break-even look easier to clear than it was —
    the unsafe direction, so such a row is dropped from the average.
    """
    entry = _as_float(row.get("entry_price"))
    stop = _as_float(row.get("stop_price"))
    target = _as_float(row.get("target_price"))
    if entry is None or entry <= 0 or stop is None or target is None:
        return None  # guard: entry is the divisor
    stop_pct = (entry - stop) / entry
    target_pct = (target - entry) / entry
    if not (_PLAUSIBLE_STOP[0] <= stop_pct <= _PLAUSIBLE_STOP[1]):
        return None
    if not (_PLAUSIBLE_TARGET[0] <= target_pct <= _PLAUSIBLE_TARGET[1]):
        return None
    if stop_floor > 0 and stop_pct < stop_floor * 0.5:
        return None
    return stop_pct, target_pct


def _expectancy_from_row(row: dict[str, Any], n: int) -> float:
    """Mean rupees per trade, preferring the journal's own `expectancy` column.

    Falls back to (gross_win net of gross_loss) / n, detecting by sign whether
    `gross_loss` is stored signed (negative) or as a magnitude.
    """
    stated = _as_float(row.get("expectancy"))
    if stated is not None:
        return stated
    gross_win = _as_float(row.get("gross_win")) or 0.0
    gross_loss = _as_float(row.get("gross_loss")) or 0.0
    net = gross_win + gross_loss if gross_loss < 0 else gross_win - gross_loss
    return net / n if n > 0 else 0.0  # guard: empty bucket


# --------------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------------


class Learner:
    """Scores each learning bucket from closed trades and gates the ones that lose.

    `refresh()` recomputes every verdict from `journal.bucket_stats()` and caches
    them with the time of the refresh, so the UI can show how stale the numbers are
    rather than implying they are live.
    """

    def __init__(
        self,
        journal: "Journal",
        cfg: "Config",
        *,
        learn_across_modes: bool | None = None,
        two_sided: bool = False,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> None:
        """Hold the journal and config; nothing is read until the first refresh.

        `learn_across_modes` defaults to the config flag of the same name if it has
        one, else False: paper losses do not gate live setups unless asked.
        """
        self.journal = journal
        self.cfg = cfg
        self.two_sided = bool(two_sided)
        self.stale_after_s = float(stale_after_s)
        self.learn_across_modes = (
            bool(_get(cfg, "learn_across_modes", False))
            if learn_across_modes is None
            else bool(learn_across_modes)
        )
        self._verdicts: dict[str, BucketVerdict] | None = None
        self._cached_mode: str | None = None
        self._cached_mode_set: bool = False
        self._last_refresh_ts: int | None = None
        self._failed_at_ts: int | None = None
        self._stats_error: str | None = None

    # -- configuration readers ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether verdicts are actually applied to signal scores."""
        return bool(_get(self.cfg, "learning_enabled", True))

    @property
    def min_samples(self) -> int:
        """Trades a bucket needs before it is judged at all. Never below 1."""
        return max(1, _as_int(_get(self.cfg, "min_samples_to_gate", 10), 10))

    @property
    def confidence(self) -> float:
        """Confidence level for the Wilson lower bound, repaired if it is nonsense.

        Repaired rather than raised on, because a bad value here would otherwise take
        down the tick loop. Below MIN_CONFIDENCE it falls back to the 0.90 default
        instead of clamping to 0.50: a z of zero would make the "lower bound" the raw
        win rate itself, which promotes a lucky 7-for-10 bucket — the expensive
        direction. Above the ceiling it clamps, which only makes the gate stricter.
        """
        raw = _as_float(_get(self.cfg, "gate_confidence", DEFAULT_CONFIDENCE))
        if raw is None:
            return DEFAULT_CONFIDENCE
        if raw < MIN_CONFIDENCE:
            logger.warning(
                "gate_confidence %.4f is too low to be a confidence level; using the "
                "%.2f default instead", raw, DEFAULT_CONFIDENCE,
            )
            return DEFAULT_CONFIDENCE
        if raw > MAX_CONFIDENCE:
            logger.warning("gate_confidence %.4f is too high; using %.4f", raw, MAX_CONFIDENCE)
            return MAX_CONFIDENCE
        return raw

    @property
    def z(self) -> float:
        """The z multiplier currently in force, for the UI and for tests."""
        return z_for_confidence(self.confidence, two_sided=self.two_sided)

    @property
    def config_break_even(self) -> float:
        """Break-even implied by the CURRENT config, used only as a last resort."""
        return break_even_win_rate(
            _as_float(_get(self.cfg, "stop_pct", 0.30)) or 0.30,
            _as_float(_get(self.cfg, "target_pct", 0.50)) or 0.50,
        )

    @property
    def stats_mode(self) -> str | None:
        """Which mode's trades the learner reads: the live mode, or None for all."""
        if self.learn_across_modes:
            return None
        mode = _get(self.cfg, "mode", "paper")
        return str(mode) if mode else "paper"

    # -- staleness ----------------------------------------------------------------

    @property
    def last_refresh_ts(self) -> int | None:
        """Epoch SECONDS of the last successful refresh, or None."""
        return self._last_refresh_ts

    @property
    def last_refresh_at(self) -> str | None:
        """ISO-8601 IST timestamp of the last successful refresh, for the UI."""
        if self._last_refresh_ts is None:
            return None
        return clock.to_ist(self._last_refresh_ts).isoformat()

    @property
    def age_seconds(self) -> float | None:
        """Seconds since the last refresh, or None if it has never run."""
        if self._last_refresh_ts is None:
            return None
        return max(0.0, clock.epoch_s(clock.now()) - float(self._last_refresh_ts))

    @property
    def is_stale(self) -> bool:
        """True when the cached verdicts are older than `stale_after_s`."""
        age = self.age_seconds
        return True if age is None else age > self.stale_after_s

    # -- the main entry points ------------------------------------------------------

    def refresh(self) -> None:
        """Recompute every bucket verdict from the journal. Call after each close.

        Keeps the previous cache on failure — stale verdicts beat no verdicts — and
        re-raises as a DataError so the engine can show the problem instead of the
        learner pretending it knows anything.
        """
        mode = self.stats_mode
        try:
            rows = list(self.journal.bucket_stats(mode=mode) or [])
        except IndbotError as exc:
            self._stats_error = str(exc)
            logger.error("learner: bucket_stats failed: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
            self._stats_error = str(exc)
            logger.exception("learner: bucket_stats failed")
            raise DataError("Could not read bucket statistics", mode=mode) from exc

        # Only pay for the trade scan when bucket_stats cannot supply stop/target.
        needs_trade_scan = any(_pcts_from_row(row) is None for row in rows if isinstance(row, dict))
        trade_pcts = self._bucket_pcts_from_trades(mode) if needs_trade_scan else {}

        verdicts: dict[str, BucketVerdict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            bucket = str(row.get("bucket") or "").strip()
            if not bucket:
                continue
            verdicts[bucket] = self._verdict_from(bucket, row, trade_pcts)

        self._verdicts = verdicts
        self._cached_mode = mode
        self._cached_mode_set = True
        self._last_refresh_ts = clock.epoch_s(clock.now())
        self._failed_at_ts = None
        self._stats_error = None
        blocked = [v.bucket for v in verdicts.values() if v.state == "blocked"]
        logger.info(
            "learner: %d buckets from %s trades, %d blocked",
            len(verdicts),
            mode or "all",
            len(blocked),
        )
        for bucket in blocked:
            logger.info("learner: %s", verdicts[bucket].note)

    def verdict(self, bucket: str) -> BucketVerdict:
        """The cached verdict for `bucket`, or a neutral one when it is unknown.

        Never raises: this sits on the entry path, and a statistics failure must not
        stop the risk gate from being reached. A failure therefore falls back to
        neutral (multiplier 1.0) — the same behaviour as learning switched off, with
        the error surfaced in summary() rather than swallowed.
        """
        self._ensure()
        cached = (self._verdicts or {}).get(str(bucket or ""))
        if cached is not None:
            return cached
        return self._empty_verdict(str(bucket or ""))

    def adjust(self, sig: Signal) -> tuple[Signal, BucketVerdict]:
        """Return (signal with score * multiplier, verdict), plus a reason line.

        The reason is appended verbatim to `Signal.reasons`, which the UI prints, so
        a user who sees a score fall can read why in the same list.
        """
        if not sig.bucket:
            # No bucket means the signal can never be learned from. That is an upstream
            # bug worth seeing, not something to absorb quietly.
            logger.warning(
                "learner: signal %s/%s has no learning bucket; left unadjusted",
                sig.underlying, sig.setup,
            )
        verdict = self.verdict(sig.bucket)
        if not self.enabled:
            # Verdicts stay truthful so the leaderboard still shows what would have
            # happened; they are simply not applied.
            return sig, _with_applied(verdict, False)
        if verdict.multiplier == 1.0:
            return sig, verdict
        reason = f"Learning {verdict.state}: score x{verdict.multiplier:.2f} — {verdict.note}"
        return sig.with_score(sig.score * verdict.multiplier, extra_reason=reason), verdict

    def is_blocked(self, bucket: str) -> bool:
        """True when this bucket's signals must not be traded.

        Returns False when learning is disabled, so the flag in the UI matches what
        the bot actually does.
        """
        if not self.enabled:
            return False
        return self.verdict(bucket).state == "blocked"

    def blocked_buckets(self) -> list[str]:
        """Every currently blocked bucket key, for the snapshot."""
        if not self.enabled:
            return []
        self._ensure()
        return sorted(v.bucket for v in (self._verdicts or {}).values() if v.state == "blocked")

    def leaderboard(self) -> list[BucketVerdict]:
        """Every bucket, best expectancy first, with blocked buckets last always.

        Blocked buckets sink regardless of expectancy: a bucket the bot refuses to
        trade should never sit at the top of a list the user reads as "what works".
        """
        self._ensure()
        verdicts = list((self._verdicts or {}).values())
        return sorted(
            verdicts,
            key=lambda v: (
                1 if v.state == "blocked" else 0,
                -v.expectancy,
                -v.n,
                v.bucket,
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Counts per state, the overall sample size and staleness, for the UI header."""
        self._ensure()
        verdicts = list((self._verdicts or {}).values())
        states = {state: 0 for state in STATES}
        for v in verdicts:
            states[v.state] = states.get(v.state, 0) + 1
        total_n = sum(v.n for v in verdicts)
        total_wins = sum(v.wins for v in verdicts)
        mode = self._cached_mode if self._cached_mode_set else self.stats_mode
        return {
            "enabled": self.enabled,
            "mode": mode,
            "mode_scope": "all modes combined" if mode is None else f"{mode} trades only",
            "learn_across_modes": self.learn_across_modes,
            "confidence": self.confidence,
            "two_sided": self.two_sided,
            "z": round(self.z, 6),
            "min_samples_to_gate": self.min_samples,
            "buckets": len(verdicts),
            "trades": total_n,
            "wins": total_wins,
            "win_rate": round(total_wins / total_n, 4) if total_n > 0 else 0.0,
            "states": states,
            "blocked_buckets": sorted(v.bucket for v in verdicts if v.state == "blocked"),
            "last_refresh_ts": self._last_refresh_ts,
            "last_refresh_at": self.last_refresh_at,
            "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 1),
            "stale": self.is_stale,
            "error": self._stats_error,
            "headline": self._headline(total_n, len(verdicts), states, mode),
        }

    def as_dict(self) -> dict[str, Any]:
        """The whole learning panel: summary plus the leaderboard, JSON-ready."""
        return {
            "summary": self.summary(),
            "leaderboard": [v.as_dict() for v in self.leaderboard()],
        }

    # -- internals ------------------------------------------------------------------

    def _ensure(self) -> None:
        """Refresh if the cache is missing or belongs to another mode. Never raises.

        The read paths (verdict, leaderboard, summary) serve the entry path and the
        UI, so a statistics failure has to surface as an error in the panel rather
        than as an exception out of a tick. After a failure it backs off for
        _RETRY_AFTER_S instead of re-querying a broken database once per lookup.
        """
        if not self._needs_refresh():
            return
        now_ts = clock.epoch_s(clock.now())
        if (
            self._verdicts is not None
            and self._failed_at_ts is not None
            and (now_ts - self._failed_at_ts) < _RETRY_AFTER_S
        ):
            return
        try:
            self.refresh()
            return
        except IndbotError as exc:
            logger.error("learner: verdict cache unavailable: %s", exc)
        except Exception:  # noqa: BLE001 - read path, must not propagate
            logger.exception("learner: verdict cache unavailable")
        self._failed_at_ts = now_ts
        # A failed refresh must never leave another MODE's verdicts in force: a paper
        # block applied to a live signal (or the reverse) is exactly what
        # learn_across_modes=False exists to prevent. Empty reads as neutral, and
        # last_refresh_at goes to None so the UI shows the numbers as unavailable.
        if self._verdicts is None or self._cached_mode != self.stats_mode:
            self._verdicts = {}
            self._cached_mode = self.stats_mode
            self._cached_mode_set = True
            self._last_refresh_ts = None

    def _needs_refresh(self) -> bool:
        """True before the first refresh, or when the mode under study has changed.

        Age alone does not force a refresh — the engine refreshes after every close
        and the UI shows staleness honestly — but a mode switch must, or live signals
        would keep being judged by paper statistics.
        """
        if self._verdicts is None or not self._cached_mode_set:
            return True
        if self._last_refresh_ts is None:
            return True  # the last attempt failed; keep trying, throttled by _ensure
        return self._cached_mode != self.stats_mode

    def _headline(
        self,
        total_n: int,
        buckets: int,
        states: dict[str, int],
        mode: str | None,
    ) -> str:
        """One sentence for the learning panel header."""
        if not self.enabled:
            return (
                f"Learning is switched off: {buckets} setups are being tracked from "
                f"{_plural(total_n, 'trade')} but no score is being adjusted."
            )
        scope = f"{mode} " if mode else ""
        if total_n <= 0:
            return (
                f"No closed {scope}trades yet — every setup stays neutral until it has "
                f"{_plural(self.min_samples, 'trade')}."
            )
        return (
            f"Learning from {_plural(total_n, f'{scope}trade'.strip())} across "
            f"{_plural(buckets, 'setup')}: {states.get('promoted', 0)} promoted, "
            f"{states.get('demoted', 0)} demoted, {states.get('blocked', 0)} blocked."
        )

    def _empty_verdict(self, bucket: str) -> BucketVerdict:
        """A neutral verdict for a bucket with no closed trades behind it."""
        confidence = self.confidence
        min_samples = self.min_samples
        return BucketVerdict(
            bucket=bucket,
            n=0,
            wins=0,
            win_rate=0.0,
            expectancy=0.0,
            lower_bound=0.0,
            state="neutral",
            multiplier=STATE_MULTIPLIERS["neutral"],
            note=build_note(
                "neutral",
                n=0,
                wins=0,
                win_rate=0.0,
                expectancy=0.0,
                lower_bound=0.0,
                break_even=self.config_break_even,
                confidence=confidence,
                min_samples=min_samples,
                mode=self.stats_mode,
            ),
            losses=0,
            break_even=self.config_break_even,
            be_source="config",
            confidence=confidence,
            z=self.z,
            trades_needed=min_samples,
            mode=self.stats_mode,
            applied=self.enabled,
        )

    def _verdict_from(
        self,
        bucket: str,
        row: dict[str, Any],
        trade_pcts: dict[str, tuple[float, float, int]],
    ) -> BucketVerdict:
        """Turn one bucket_stats row into a verdict."""
        n = max(0, _as_int(row.get("n"), 0))
        wins = max(0, min(_as_int(row.get("wins"), 0), n))
        losses = _as_int(row.get("losses"), max(0, n - wins))
        win_rate = wins / n if n > 0 else 0.0  # guard: empty bucket
        expectancy = _expectancy_from_row(row, n)
        confidence = self.confidence
        min_samples = self.min_samples

        be, avg_stop, avg_target, be_source = self._break_even_for(bucket, row, trade_pcts)
        lower = wilson_lower_bound(wins, n, confidence, two_sided=self.two_sided)
        state, multiplier = classify(
            n, expectancy, lower, be, min_samples=min_samples
        )
        return BucketVerdict(
            bucket=bucket,
            n=n,
            wins=wins,
            win_rate=win_rate,
            expectancy=expectancy,
            lower_bound=lower,
            state=state,
            multiplier=multiplier,
            note=build_note(
                state,
                n=n,
                wins=wins,
                win_rate=win_rate,
                expectancy=expectancy,
                lower_bound=lower,
                break_even=be,
                confidence=confidence,
                min_samples=min_samples,
                mode=self.stats_mode,
            ),
            losses=max(0, losses),
            break_even=be,
            be_source=be_source,
            avg_stop_pct=avg_stop,
            avg_target_pct=avg_target,
            confidence=confidence,
            z=z_for_confidence(confidence, two_sided=self.two_sided),
            gross_win=_as_float(row.get("gross_win")) or 0.0,
            gross_loss=_as_float(row.get("gross_loss")) or 0.0,
            trades_needed=max(0, min_samples - n),
            mode=self.stats_mode,
            applied=self.enabled,
        )

    def _break_even_for(
        self,
        bucket: str,
        row: dict[str, Any],
        trade_pcts: dict[str, tuple[float, float, int]],
    ) -> tuple[float, float, float, str]:
        """(break_even, avg_stop_pct, avg_target_pct, source) for one bucket.

        Preference order: the stop/target the journal reports for these trades, then
        the stop/target derived from the trades' own entry/stop/target prices, then —
        only if neither is available — the current config. Config is last on purpose:
        changing stop_pct today must not retroactively re-judge yesterday's trades.
        """
        stated = _pcts_from_row(row)
        if stated is not None:
            return break_even_win_rate(*stated), stated[0], stated[1], "stats"
        stated_be = _first_float(row, "break_even", "be")
        if stated_be is not None and 0.0 < stated_be < 1.0:
            return stated_be, 0.0, 0.0, "stats"
        derived = trade_pcts.get(bucket)
        if derived is not None:
            stop, target, _count = derived
            return break_even_win_rate(stop, target), stop, target, "trades"
        return self.config_break_even, 0.0, 0.0, "config"

    def _bucket_pcts_from_trades(
        self, mode: str | None
    ) -> dict[str, tuple[float, float, int]]:
        """Average stop/target fraction per bucket, read off the closed trades.

        Never raises: a missing average only costs the config fallback, which is
        recorded as `be_source` so the UI can say which reference was used.
        """
        try:
            rows = list(self.journal.trades(mode=mode, limit=_TRADE_SCAN_LIMIT) or [])
        except Exception as exc:  # noqa: BLE001 - statistics only, not an order path
            logger.warning(
                "learner: could not read trades for break-even averages (%s); "
                "falling back to the configured stop/target",
                exc,
            )
            return {}

        stop_floor = _as_float(_get(self.cfg, "min_stop_pct", 0.12)) or 0.0
        acc: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            bucket = str(row.get("bucket") or "").strip()
            if not bucket:
                continue
            if "exit_price" in row and row.get("exit_price") is None:
                continue  # still open: no outcome yet, and bucket_stats will not count it
            pair = _pcts_from_row(row) or _pcts_from_prices(row, stop_floor=stop_floor)
            if pair is not None:
                acc.setdefault(bucket, []).append(pair)

        out: dict[str, tuple[float, float, int]] = {}
        for bucket, pairs in acc.items():
            count = len(pairs)
            if count <= 0:  # guard: divisor
                continue
            out[bucket] = (
                sum(p[0] for p in pairs) / count,
                sum(p[1] for p in pairs) / count,
                count,
            )
        return out


def _with_applied(verdict: BucketVerdict, applied: bool) -> BucketVerdict:
    """A copy of `verdict` flagged as applied or not (BucketVerdict is frozen)."""
    if verdict.applied == applied:
        return verdict
    return replace(verdict, applied=applied)
