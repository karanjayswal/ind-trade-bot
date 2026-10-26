"""Shared value types.

These are the integration contract between the strategy, the chain picker, the risk
gate, the broker and the UI. Signal and OptionLeg are frozen: once a signal is made
it is a fact about a candle that has already closed, and nothing downstream may
rewrite it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, TypedDict

from . import clock


class Candle(TypedDict):
    """One OHLCV bar, exactly as the API returns it.

    `ts` is the candle's OPEN time in epoch SECONDS, and the candle covers
    [ts, ts + interval) — its own open is included, the next candle's open is not.
    """

    ts: int
    o: float
    h: float
    l: float
    c: float
    v: int


SIDES = ("CE", "PE")
MODES = ("backtest", "paper", "live")

#: A CE is the bullish side, a PE the bearish one. The bot is long premium only —
#: it buys a call or buys a put, and never sells one, so the loss is capped at the
#: premium paid and no short-option margin is ever required.
SIDE_FOR_DIRECTION = {"bullish": "CE", "bearish": "PE"}


@dataclass(frozen=True)
class Signal:
    """A setup that fired on a closed candle of the underlying.

    A Signal is not a trade. It still has to survive the learner, the chain picker
    and the risk gate, and most of them will not.
    """

    setup: str
    side: str
    action: str
    underlying: str
    ts: int
    spot: float
    score: float
    reasons: list[str] = field(default_factory=list)
    bucket: str = ""
    stop_pct: float = 0.30
    target_pct: float = 0.50

    @staticmethod
    def make_bucket(setup: str, side: str, ts: int, regime: str) -> str:
        """The learning key: one cell of the setup x time-of-day x regime grid.

        The learner scores buckets, not setups, because the same rule behaves
        differently at 09:20 than at 14:50 and differently in a trend than in a range.
        """
        return f"{setup}:{side}:{clock.tod_bucket(clock.to_ist(ts))}:{regime}"

    @property
    def at(self) -> datetime:
        """The trigger candle's open time, in IST."""
        return clock.to_ist(self.ts)

    @property
    def direction(self) -> str:
        """"bullish" for a CE, "bearish" for a PE."""
        return "bullish" if self.side == "CE" else "bearish"

    def with_score(self, score: float, *, extra_reason: str = "") -> "Signal":
        """A copy with a new score — how the learner applies its multiplier."""
        reasons = list(self.reasons) + ([extra_reason] if extra_reason else [])
        return Signal(
            setup=self.setup,
            side=self.side,
            action=self.action,
            underlying=self.underlying,
            ts=self.ts,
            spot=self.spot,
            score=max(0.0, min(1.0, score)),
            reasons=reasons,
            bucket=self.bucket,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    @property
    def break_even_win_rate(self) -> float:
        """The win rate this signal's stop/target pair needs just to break even.

        With a 30% stop and a 50% target that is 0.375 — the number the learner
        measures a bucket's Wilson lower bound against.
        """
        total = self.stop_pct + self.target_pct
        return (self.stop_pct / total) if total > 0 else 0.5

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["at"] = self.at.isoformat()
        d["direction"] = self.direction
        d["break_even_win_rate"] = round(self.break_even_win_rate, 4)
        return d


@dataclass(frozen=True)
class OptionLeg:
    """One tradable option contract, priced.

    `security_id` goes straight to POST /order. `trading_symbol` is the stable
    identifier for history, because the exchange reuses numeric tokens after expiry.
    """

    underlying: str
    expiry: str
    strike: float
    side: str
    security_id: str
    trading_symbol: str
    lot_size: int
    exchange: str
    ltp: float
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0
    volume: int = 0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    tick_size: float = 0.05

    @property
    def mid(self) -> float:
        """Mid of the top-of-book, falling back to the LTP when a side is missing."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ltp

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread in rupees per unit."""
        return max(0.0, self.ask - self.bid) if (self.bid > 0 and self.ask > 0) else 0.0

    @property
    def spread_pct(self) -> float:
        """Spread as a percentage of the mid — the honest cost of a round trip.

        Returns 100.0 when the book is one-sided, so a leg with no bid is filtered
        out rather than treated as free to trade.
        """
        mid = self.mid
        if mid <= 0:
            return 100.0
        if self.bid <= 0 or self.ask <= 0:
            return 100.0
        return (self.spread / mid) * 100.0

    @property
    def lot_cost(self) -> float:
        """Rupees to buy one lot at the LTP. The number the risk gate sizes against."""
        return self.ltp * self.lot_size

    @property
    def label(self) -> str:
        """"NIFTY 25000 CE" — how the UI names this contract."""
        strike = int(self.strike) if float(self.strike).is_integer() else self.strike
        return f"{self.underlying} {strike} {self.side}"

    def risk_at_stop(self, stop_pct: float) -> float:
        """Rupees at risk on one lot if the stop fills exactly at its price."""
        return self.lot_cost * stop_pct

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            mid=round(self.mid, 2),
            spread=round(self.spread, 2),
            spread_pct=round(self.spread_pct, 3),
            lot_cost=round(self.lot_cost, 2),
            label=self.label,
        )
        return d


@dataclass
class Position:
    """An open (or closed) long-option position: one lot, one contract.

    Mutable, because `high_water` tracks the best premium seen and the exit fields
    are filled in when it closes.
    """

    id: str
    leg: OptionLeg
    signal_setup: str
    bucket: str
    qty: int
    entry_price: float
    entry_ts: int
    entry_order_id: str
    stop_price: float
    target_price: float
    mode: str = "paper"
    high_water: float = 0.0
    trail_armed: bool = False
    exit_price: float | None = None
    exit_ts: int | None = None
    exit_order_id: str | None = None
    exit_reason: str | None = None
    costs: float = 0.0

    def __post_init__(self) -> None:
        if self.high_water <= 0:
            self.high_water = self.entry_price

    @property
    def lots(self) -> int:
        """Units divided by lot size. The bot trades 1."""
        return max(1, self.qty // self.leg.lot_size) if self.leg.lot_size else 1

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def cost_basis(self) -> float:
        """Rupees paid to open, before costs."""
        return self.entry_price * self.qty

    def pnl(self, ltp: float | None = None) -> float:
        """Rupees of profit or loss, realised if closed, marked to `ltp` if open."""
        price = self.exit_price if self.exit_price is not None else ltp
        if price is None:
            return 0.0
        return (price - self.entry_price) * self.qty - self.costs

    def pnl_pct(self, ltp: float | None = None) -> float:
        """P&L as a percentage of the premium paid."""
        if self.cost_basis <= 0:
            return 0.0
        return (self.pnl(ltp) / self.cost_basis) * 100.0

    @property
    def is_win(self) -> bool:
        """A closed trade that made money after costs. Used by the learner."""
        return (not self.is_open) and self.pnl() > 0

    def as_dict(self, ltp: float | None = None) -> dict[str, Any]:
        mark = ltp if self.is_open else self.exit_price
        return {
            "id": self.id,
            "leg": self.leg.as_dict(),
            "label": self.leg.label,
            "signal_setup": self.signal_setup,
            "bucket": self.bucket,
            "mode": self.mode,
            "qty": self.qty,
            "lots": self.lots,
            "entry_price": self.entry_price,
            "entry_ts": self.entry_ts,
            "entry_at": clock.to_ist(self.entry_ts).isoformat(),
            "entry_order_id": self.entry_order_id,
            "stop_price": round(self.stop_price, 2),
            "target_price": round(self.target_price, 2),
            "high_water": round(self.high_water, 2),
            "trail_armed": self.trail_armed,
            "ltp": mark,
            "exit_price": self.exit_price,
            "exit_ts": self.exit_ts,
            "exit_at": clock.to_ist(self.exit_ts).isoformat() if self.exit_ts else None,
            "exit_reason": self.exit_reason,
            "costs": round(self.costs, 2),
            "cost_basis": round(self.cost_basis, 2),
            "pnl": round(self.pnl(mark), 2),
            "pnl_pct": round(self.pnl_pct(mark), 2),
            "is_open": self.is_open,
        }


@dataclass(frozen=True)
class RiskVerdict:
    """The risk gate's answer. `reasons` is populated whether it passed or failed.

    A pass that cannot explain itself is indistinguishable from a bug, so the gate
    always says what it checked.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)
    blocking_rule: str | None = None
    checked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
