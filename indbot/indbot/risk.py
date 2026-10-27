"""The risk gate — the module that turns the user's rupee limits into refusals.

The limits are ₹2,000 a day, ₹1,000 a trade, 3 trades a day, 1 open position, and the
bot buys exactly one lot. This file is the only thing standing between a signal and a
real order, so every rule here is evaluated on every entry and every failure is
reported, not just the first: the user needs the whole list to decide what to change.

THE ARITHMETIC THAT WILL BLOCK MOST TRADES
------------------------------------------
A long option risks the whole premium in theory, but the bot's practical risk is the
stop: `lot_cost * stop_pct`, where `lot_cost = ltp * lot_size`. With a ₹1,000 per-trade
limit and a 30% stop, the most premium one lot may cost is ``1000 / 0.30 = ₹3,333``:

    NIFTY      lot  75  -> max ₹44/unit    (ATM is ₹150-200)
    BANKNIFTY  lot  35  -> max ₹95/unit    (ATM is ₹250-350)
    FINNIFTY   lot  65  -> max ₹51/unit    (ATM is ₹120-180)
    MIDCPNIFTY lot 120  -> max ₹27/unit    (ATM is ₹80-120)
    SENSEX     lot  20  -> max ₹166/unit   (ATM is ₹200-300)

An ATM option therefore almost never passes rule 6. That is the limits working, not a
bug, and this module must never paper over it. It does not widen the stop, it does not
raise the limit, and rule 7 exists specifically to stop the bot "solving" rule 6 with a
5% stop that the bid-ask spread alone would trigger. It blocks, states the arithmetic in
rupees, and offers the user the two honest choices.

Rule 5 is sticky. Once a day has hit the ₹2,000 loss limit, the halt is written to the
journal's meta table keyed by (mode, date), because an automated bot that forgets its
halt on restart turns a ₹2,000 limit into a ₹6,000 one.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, Iterable

from . import clock
from .errors import ConfigError
from .types import MODES, OptionLeg, Position, RiskVerdict, Signal

if TYPE_CHECKING:  # pragma: no cover - import cycle / other agents' modules
    from .config import Config
    from .journal import Journal
    from .learner import Learner

log = logging.getLogger(__name__)

__all__ = [
    "RiskGate",
    "inr",
    "pct",
    "EXIT_STOP",
    "EXIT_TARGET",
    "EXIT_TRAIL",
    "EXIT_FLAT_BY",
    "EXIT_TIME_STOP",
    "EXIT_REASONS",
]

# --- exit reasons -----------------------------------------------------------------
EXIT_STOP = "stop"
EXIT_TARGET = "target"
EXIT_TRAIL = "trail"
EXIT_FLAT_BY = "flat_by"
EXIT_TIME_STOP = "time_stop"
EXIT_REASONS = (EXIT_STOP, EXIT_TARGET, EXIT_TRAIL, EXIT_FLAT_BY, EXIT_TIME_STOP)

# --- rule ids (stable slugs; RiskVerdict.blocking_rule is one of these) ------------
RULE_ARMED = "armed"
RULE_SESSION = "session_window"
RULE_OPEN_POSITIONS = "max_open_positions"
RULE_TRADE_COUNT = "max_trades_per_day"
RULE_DAY_HALT = "day_halt"
RULE_DAY_BUDGET = "day_budget"
RULE_PER_TRADE_RISK = "per_trade_risk"
RULE_STOP_SANITY = "stop_sanity"
RULE_MIN_SCORE = "min_score"
RULE_BUCKET_BLOCKED = "bucket_blocked"
RULE_FUNDS = "funds"

#: Cash headroom over the premium for a live BUY. Covers the flat ₹10 brokerage plus
#: 18% GST, STT/exchange/SEBI charges (see docs §Margin Calculation) and the gap
#: between the quoted LTP and the marketable limit the broker actually sends.
FUNDS_HEADROOM = 1.05

DEFAULT_NO_ENTRY_BEFORE = "09:30"
DEFAULT_NO_ENTRY_AFTER = "14:45"
DEFAULT_FLAT_BY = "15:10"
DEFAULT_TIME_STOP_MINUTES = 45

_STATUS_OK = "ok"
_STATUS_BLOCK = "BLOCK"
_STATUS_SKIP = "skip"


# ----------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------
def _indian_group(digits: str) -> str:
    """Group an integer digit string the Indian way: 1234567 -> '12,34,567'."""
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    chunks: list[str] = []
    while len(head) > 2:
        chunks.insert(0, head[-2:])
        head = head[:-2]
    if head:
        chunks.insert(0, head)
    return ",".join(chunks) + "," + tail


def inr(amount: float | None, *, decimals: int | None = None) -> str:
    """Rupees with Indian digit grouping: 12600 -> '₹12,600', -3780.5 -> '-₹3,780.50'.

    Every rupee figure the gate puts in front of the user goes through here, because a
    limit the user cannot read at a glance is a limit they will not trust.
    """
    if amount is None or not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
        return "₹?"
    value = float(amount)
    if decimals is None:
        # Whole rupees unless the paise actually matter.
        decimals = 0 if abs(value - round(value)) < 0.005 else 2
    rounded = round(value, decimals)
    sign = "-" if rounded < 0 else ""
    text = f"{abs(rounded):.{decimals}f}"
    whole, _, frac = text.partition(".")
    return f"{sign}₹{_indian_group(whole)}" + (f".{frac}" if frac else "")


def pct(fraction: float | None) -> str:
    """A 0..1 fraction as a percentage: 0.30 -> '30%', 0.0794 -> '7.9%'."""
    if fraction is None or not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)):
        return "?%"
    value = float(fraction) * 100.0
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f}%"
    return f"{value:.1f}%"


def _floor_money(value: float) -> float:
    """Round a budget DOWN — rounding a spending cap up is how limits get exceeded."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return float(math.floor(value)) if value >= 1.0 else math.floor(value * 100.0) / 100.0


def _aware(dt: datetime) -> datetime:
    """Treat a naive datetime as IST, never as the machine's local time."""
    return dt.replace(tzinfo=clock.IST) if dt.tzinfo is None else dt


def _is_price(value: Any) -> bool:
    """True for a number we are willing to trade against."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value)) and float(value) > 0


# ----------------------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------------------
class RiskGate:
    """Evaluates every entry rule and every exit condition against the config limits.

    Construction validates the limits themselves, loudly: a non-positive daily or
    per-trade loss limit is a configuration bug that must surface before the first
    order, not as a silently disabled guard halfway through a session.
    """

    def __init__(self, journal: "Journal", cfg: "Config", *, learner: "Learner | None" = None) -> None:
        self.journal = journal
        self.cfg = cfg
        #: Optional. Supplies rule 8's bucket block; the gate works without it.
        self.learner = learner

        for name, default in (("max_daily_loss", 2000.0), ("max_trade_loss", 1000.0), ("min_stop_pct", 0.12)):
            value = self._num(name, default)
            if value <= 0:
                raise ConfigError(
                    f"{name} must be greater than zero, got {value!r} — a non-positive "
                    "limit silently disables the guard it is supposed to enforce.",
                    setting=name,
                    value=value,
                )
        if self._num("max_trade_loss", 1000.0) > self._num("max_daily_loss", 2000.0):
            log.warning(
                "max_trade_loss %s exceeds max_daily_loss %s — a single trade can end the day.",
                self._num("max_trade_loss", 1000.0),
                self._num("max_daily_loss", 2000.0),
            )

    # -- config access -------------------------------------------------------------
    def _cfg(self, name: str, default: Any) -> Any:
        """Read a config attribute, falling back when that key is not defined.

        `time_stop_minutes` and friends are not in the Config dataclass contract, so
        they are read tolerantly rather than requiring another agent's file to change.
        """
        value = getattr(self.cfg, name, None)
        return default if value is None else value

    def _num(self, name: str, default: float) -> float:
        """A config attribute as a float, falling back on anything unparseable."""
        try:
            return float(self._cfg(name, default))
        except (TypeError, ValueError):
            log.warning("config %s=%r is not a number; using %s", name, getattr(self.cfg, name, None), default)
            return float(default)

    def _time_cfg(self, name: str, default: str) -> time:
        """Parse an "HH:MM" config window, falling back to the default when malformed.

        A malformed `flat_by` must not take the exit path down with it, so this logs
        and degrades to the default instead of raising mid-session.
        """
        raw = str(self._cfg(name, default))
        try:
            return clock.parse_hhmm(raw)
        except (ValueError, AttributeError):
            log.error("config %s=%r is not HH:MM; falling back to %s", name, raw, default)
            return clock.parse_hhmm(default)

    # -- entry ---------------------------------------------------------------------
    def check_entry(
        self,
        sig: Signal,
        leg: OptionLeg,
        *,
        open_positions: list[Position],
        mode: str,
        funds: dict | None = None,
        blocked_buckets: Iterable[str] | None = None,
        now: datetime | None = None,
    ) -> RiskVerdict:
        """Evaluate all nine entry rules and report EVERY failure, not just the first.

        Returns a RiskVerdict whose `checked` lists every rule with its outcome (so a
        pass is auditable), whose `reasons` holds the blocking sentences on a failure
        and the supporting arithmetic on a pass, and whose `blocking_rule` names the
        first rule that failed.
        """
        now_dt = _aware(now) if now is not None else clock.now()
        day = now_dt.date()
        mode_norm = str(mode or "").strip().lower()
        positions = [p for p in (open_positions or []) if getattr(p, "is_open", True)]

        # (rule number, rule id, status, sentence)
        rules: list[tuple[int, str, str, str]] = []

        def record(number: int, rule_id: str, status: str, sentence: str) -> None:
            rules.append((number, rule_id, status, sentence))

        # Today's realised numbers are needed by rules 4 and 5. One read, one failure
        # mode: if the journal cannot be read we do NOT know how much has been lost
        # today, and the safe branch when the day's losses are unknown is to refuse.
        stats, stats_error = self._day_stats(mode_norm, day)

        self._rule_1_armed(record, mode_norm)
        self._rule_2_session(record, now_dt)
        self._rule_3_open_positions(record, positions)
        self._rule_4_trade_count(record, stats, stats_error)
        self._rule_5_day_limit(record, sig, leg, positions, mode_norm, day, stats, stats_error)
        self._rule_6_7_sizing(record, sig, leg)
        self._rule_8_signal(record, sig, blocked_buckets)
        self._rule_9_funds(record, leg, mode_norm, funds)

        failures = [r for r in rules if r[2] == _STATUS_BLOCK]
        ok = not failures
        checked = [f"{number}. {rule_id}: {status}" for number, rule_id, status, _ in rules]
        reasons = [r[3] for r in (failures if failures else rules)]
        blocking_rule = failures[0][1] if failures else None

        if failures:
            log.info(
                "entry blocked for %s %s %s: %s",
                sig.underlying,
                sig.setup,
                getattr(leg, "label", "?"),
                "; ".join(f"{r[1]}" for r in failures),
            )
        return RiskVerdict(ok=ok, reasons=reasons, blocking_rule=blocking_rule, checked=checked)

    # -- rule 1 --------------------------------------------------------------------
    def _rule_1_armed(self, record: Any, mode: str) -> None:
        """Rule 1 — live mode must be explicitly armed, and the config must agree."""
        if mode not in MODES:
            record(1, RULE_ARMED, _STATUS_BLOCK,
                   f"Unknown trading mode {mode!r}. Refusing to trade; mode must be one of "
                   f"{', '.join(MODES)}.")
            return
        if mode != "live":
            record(1, RULE_ARMED, _STATUS_OK,
                   f"{mode.capitalize()} mode — no real order is sent, so no arming is required.")
            return
        cfg_mode = str(self._cfg("mode", "paper")).strip().lower()
        if not bool(self._cfg("armed", False)):
            record(1, RULE_ARMED, _STATUS_BLOCK,
                   "Live mode is not armed. Nothing reaches the broker until you type "
                   "ARM LIVE TRADING in the dashboard.")
            return
        if cfg_mode != "live":
            # Defence in depth: the broker asserts this too, but a gate that lets a
            # live order through on a config mismatch is a gate that failed open.
            record(1, RULE_ARMED, _STATUS_BLOCK,
                   f"Asked to trade live but the saved config mode is {cfg_mode!r}. Refusing — "
                   "the requested mode and the saved mode must agree before real money moves.")
            return
        record(1, RULE_ARMED, _STATUS_OK, "Live mode is armed and the saved config agrees.")

    # -- rule 2 --------------------------------------------------------------------
    def _rule_2_session(self, record: Any, now_dt: datetime) -> None:
        """Rule 2 — the market is open and we are inside the entry window."""
        if not clock.is_market_open(now_dt):
            record(2, RULE_SESSION, _STATUS_BLOCK,
                   f"The market is closed at {now_dt:%H:%M on %a %d %b %Y}. Entries only happen "
                   "between 09:15 and 15:30 on a trading day.")
            return
        before = self._time_cfg("no_entry_before", DEFAULT_NO_ENTRY_BEFORE)
        after = self._time_cfg("no_entry_after", DEFAULT_NO_ENTRY_AFTER)
        flat_by = self._time_cfg("flat_by", DEFAULT_FLAT_BY)
        # Safe branch: never open a position we would be forced to flatten minutes
        # later. If flat_by is earlier than no_entry_after, flat_by wins.
        last_entry = min(after, flat_by)
        current = now_dt.time()
        if current < before:
            record(2, RULE_SESSION, _STATUS_BLOCK,
                   f"It is {current:%H:%M}; the entry window opens at {before:%H:%M}. The first "
                   "minutes of the session are skipped because the opening range is not formed yet.")
            return
        if current > last_entry:
            tail = (f"{last_entry:%H:%M}" if last_entry == after
                    else f"{last_entry:%H:%M} (flat_by, which is earlier than no_entry_after)")
            record(2, RULE_SESSION, _STATUS_BLOCK,
                   f"It is {current:%H:%M}; the entry window closed at {tail}. A trade opened now "
                   f"would be force-closed at {flat_by:%H:%M} before it could work.")
            return
        record(2, RULE_SESSION, _STATUS_OK,
               f"{current:%H:%M} is inside the {before:%H:%M}-{last_entry:%H:%M} entry window.")

    # -- rule 3 --------------------------------------------------------------------
    def _rule_3_open_positions(self, record: Any, positions: list[Position]) -> None:
        """Rule 3 — at most `max_open_positions` open at a time."""
        cap = max(0, int(self._num("max_open_positions", 1.0)))
        count = len(positions)
        if count >= cap:
            held = ", ".join(getattr(p.leg, "label", p.id) for p in positions) or "none"
            record(3, RULE_OPEN_POSITIONS, _STATUS_BLOCK,
                   f"{count} position(s) already open ({held}); the cap is {cap}. The bot holds "
                   "one trade at a time so that a bad entry cannot compound into a second.")
            return
        record(3, RULE_OPEN_POSITIONS, _STATUS_OK, f"{count} of {cap} position slots in use.")

    # -- rule 4 --------------------------------------------------------------------
    def _rule_4_trade_count(self, record: Any, stats: dict | None, stats_error: str) -> None:
        """Rule 4 — at most `max_trades_per_day` entries per IST calendar day."""
        cap = max(0, int(self._num("max_trades_per_day", 3.0)))
        if stats is None:
            record(4, RULE_TRADE_COUNT, _STATUS_BLOCK,
                   f"Could not read today's trade count from the journal ({stats_error}). "
                   "Refusing to trade rather than risk exceeding the daily cap of "
                   f"{cap} trades.")
            return
        count = int(stats.get("trade_count", 0) or 0)
        if count >= cap:
            record(4, RULE_TRADE_COUNT, _STATUS_BLOCK,
                   f"{count} trades already taken today and the daily cap is {cap}. "
                   "The cap exists so a bad morning does not become a bad day.")
            return
        record(4, RULE_TRADE_COUNT, _STATUS_OK, f"{count} of {cap} trades used today.")

    # -- rule 5 --------------------------------------------------------------------
    def _rule_5_day_limit(
        self,
        record: Any,
        sig: Signal,
        leg: OptionLeg,
        positions: list[Position],
        mode: str,
        day: date,
        stats: dict | None,
        stats_error: str,
    ) -> None:
        """Rule 5 — the ₹2,000 day-loss halt, plus the budget left for one more trade."""
        limit = self._num("max_daily_loss", 2000.0)
        if stats is None:
            record(5, RULE_DAY_HALT, _STATUS_BLOCK,
                   f"Could not read today's realised P&L from the journal ({stats_error}). "
                   f"With the day's losses unknown the {inr(limit)} daily limit cannot be "
                   "enforced, so the gate refuses.")
            return

        halted, halt_reason = self.day_halted(mode, day=day, stats=stats)
        if halted:
            record(5, RULE_DAY_HALT, _STATUS_BLOCK, halt_reason)
            return

        realized = self._realized(stats)
        record(5, RULE_DAY_HALT, _STATUS_OK,
               f"Realised P&L today is {inr(realized)}, inside the {inr(limit)} daily loss limit.")

        # --- rule 5b: the day's remaining budget --------------------------------
        # Rule 5 as written only looks backwards. With ₹1,500 already lost and a ₹2,000
        # limit, a fresh trade risking ₹1,000 can still end the day at -₹2,500: 25% past
        # the user's hard limit. The bot cannot size below one lot, so the only honest
        # answer is to stand down. Set cfg.strict_day_budget = False to disable.
        if not bool(self._cfg("strict_day_budget", True)):
            record(5, RULE_DAY_BUDGET, _STATUS_SKIP,
                   "Forward day-budget check disabled by config (strict_day_budget=false).")
            return
        # Cap the new trade's risk at the per-trade limit: a trade that risks more than
        # that is rule 6's problem, and letting it also trip this rule would bury the
        # arithmetic the user actually needs to read behind a second, derived reason.
        trade_risk = min(self._trade_risk(sig, leg),
                         float(self._tier(sig)["max_trade_loss"]))
        open_risk = self._open_risk(positions)
        projected = realized - trade_risk - open_risk
        if projected <= -limit:
            extra = f", on top of {inr(open_risk)} already at risk in open positions" if open_risk > 0 else ""
            record(5, RULE_DAY_BUDGET, _STATUS_BLOCK,
                   f"Today stands at {inr(realized)} and this trade risks another "
                   f"{inr(trade_risk)}{extra}, which could end the day at {inr(projected)} — past "
                   f"the {inr(limit)} daily limit. The bot cannot buy less than one lot, so it "
                   f"stands down. Only {inr(max(0.0, limit + realized - open_risk))} of today's "
                   "loss budget is left.")
            return
        record(5, RULE_DAY_BUDGET, _STATUS_OK,
               f"Risking {inr(trade_risk)} leaves the worst case for today at {inr(projected)}, "
               f"inside the {inr(limit)} limit.")

    def _tier(self, sig: Signal) -> dict[str, Any]:
        """The risk tier this signal trades under, from Config.tier_for().

        Derived here rather than accepted as an argument: a caller that forgot to
        pass it would silently get the standard cap, and a risk gate that can be
        bypassed by omission is not a gate. Falls back to the standard limit when the
        config predates the tier settings.
        """
        try:
            return self.cfg.tier_for(sig.score, stop_pct=sig.stop_pct)
        except (AttributeError, TypeError):
            return {
                "tier": "standard",
                "max_trade_loss": self._num("max_trade_loss", 1000.0),
                "stop_pct": float(getattr(sig, "stop_pct", 0.0) or 0.0),
                "target_pct": float(getattr(sig, "target_pct", 0.0) or 0.0),
                "rr": 0.0,
                "note": "",
            }

    # -- rules 6 and 7 -------------------------------------------------------------
    def _rule_6_7_sizing(self, record: Any, sig: Signal, leg: OptionLeg) -> None:
        """Rules 6 and 7 — per-trade rupee risk, and the stop that limit implies.

        These two are the binding constraints for a ₹1,000-per-trade account. Rule 6
        refuses a lot whose stop would cost more than the limit; rule 7 refuses to
        "fix" rule 6 by using a stop so tight the spread alone would trigger it.
        """
        tier = self._tier(sig)
        max_trade_loss = float(tier["max_trade_loss"])
        min_stop = self._num("min_stop_pct", 0.12)
        lot_size = int(getattr(leg, "lot_size", 0) or 0)
        ltp = float(getattr(leg, "ltp", 0.0) or 0.0)
        stop_pct = float(getattr(sig, "stop_pct", 0.0) or 0.0)
        label = getattr(leg, "label", f"{sig.underlying} {sig.side}")

        # Guard every division, and refuse rather than guess: a missing price or lot
        # size means we cannot compute what one lot costs, and an unpriced trade is
        # the one most likely to lose real money.
        if lot_size <= 0 or not _is_price(ltp):
            message = (f"Cannot size {label}: lot size is {lot_size} and the premium is "
                       f"{ltp}. Refusing to trade a contract whose cost cannot be computed.")
            record(6, RULE_PER_TRADE_RISK, _STATUS_BLOCK, message)
            record(7, RULE_STOP_SANITY, _STATUS_BLOCK, message)
            return
        if stop_pct <= 0:
            record(6, RULE_PER_TRADE_RISK, _STATUS_BLOCK,
                   f"The signal for {label} carries a {pct(stop_pct)} stop, which is no stop at "
                   f"all — the whole {inr(ltp * lot_size)} premium would be at risk. Refusing.")
            record(7, RULE_STOP_SANITY, _STATUS_SKIP,
                   "Stop sanity not evaluated: the signal has no usable stop percentage.")
            return

        lot_cost = ltp * lot_size
        risk = lot_cost * stop_pct

        # --- rule 6 ---------------------------------------------------------------
        if risk > max_trade_loss:
            suggested_limit = math.ceil(risk / 100.0) * 100.0
            max_premium = _floor_money(max_trade_loss / (stop_pct * lot_size))
            note = self._premium_budget_note(lot_size, max_trade_loss)
            record(6, RULE_PER_TRADE_RISK, _STATUS_BLOCK,
                   f"1 lot of {label} ({lot_size} units) at {inr(ltp)} is {inr(lot_cost)}; a "
                   f"{pct(stop_pct)} stop risks {inr(risk)}, over the {inr(max_trade_loss)} "
                   f"per-trade limit. Either raise the per-trade limit to {inr(suggested_limit)} "
                   f"or let the bot buy a cheaper strike (max {inr(max_premium)} premium at this "
                   f"lot size).{note}")
        else:
            record(6, RULE_PER_TRADE_RISK, _STATUS_OK,
                   f"1 lot of {label} ({lot_size} units) at {inr(ltp)} is {inr(lot_cost)}; a "
                   f"{pct(stop_pct)} stop risks {inr(risk)}, inside the {inr(max_trade_loss)} "
                   "per-trade limit.")

        # --- rule 7 ---------------------------------------------------------------
        implied_stop = max_trade_loss / lot_cost  # lot_cost > 0, guarded above
        if implied_stop >= 1.0:
            record(7, RULE_STOP_SANITY, _STATUS_OK,
                   f"The whole {inr(lot_cost)} premium is under the {inr(max_trade_loss)} "
                   "per-trade limit, so the stop is never the binding constraint here.")
            return
        if implied_stop < min_stop:
            limit_for_min_stop = min_stop * lot_cost
            max_premium = _floor_money(max_trade_loss / (min_stop * lot_size))
            record(7, RULE_STOP_SANITY, _STATUS_BLOCK,
                   f"1 lot of {label} ({lot_size} units) at {inr(ltp)} is {inr(lot_cost)}; "
                   f"holding the loss to {inr(max_trade_loss)} would need a {pct(implied_stop)} "
                   f"stop, tighter than the {pct(min_stop)} floor — the bid-ask spread alone "
                   f"would trigger it and pay the loss for nothing. Either raise the per-trade "
                   f"limit to {inr(limit_for_min_stop)} (a {pct(min_stop)} stop on "
                   f"{inr(lot_cost)}) or let the bot buy a strike at or under "
                   f"{inr(max_premium)} a unit ({inr(_floor_money(max_trade_loss / min_stop))} "
                   "a lot).")
            return
        record(7, RULE_STOP_SANITY, _STATUS_OK,
               f"A {inr(max_trade_loss)} limit on {inr(lot_cost)} of premium implies a "
               f"{pct(implied_stop)} stop, at or above the {pct(min_stop)} floor.")

    def _premium_budget_note(self, lot_size: int, max_trade_loss: float) -> str:
        """Warn when cfg.max_premium_per_lot lets the chain picker offer blocked legs."""
        try:
            cap = float(self._cfg("max_premium_per_lot", 0.0) or 0.0)
        except (TypeError, ValueError):
            return ""
        if cap <= 0:
            return ""
        cfg_stop = self._num("stop_pct", 0.30)
        if cap * lot_size * cfg_stop <= max_trade_loss:
            return ""
        return (f" Note: max_premium_per_lot is set to {inr(cap)}, looser than this limit allows, "
                "so the chain picker will keep offering strikes this gate keeps blocking.")

    # -- rule 8 --------------------------------------------------------------------
    def _rule_8_signal(self, record: Any, sig: Signal, blocked_buckets: Iterable[str] | None) -> None:
        """Rule 8 — the signal clears `min_score` and its learning bucket is not blocked."""
        min_score = self._num("min_score", 0.55)
        score = float(getattr(sig, "score", 0.0) or 0.0)
        if score < min_score:
            record(8, RULE_MIN_SCORE, _STATUS_BLOCK,
                   f"The {sig.setup} {sig.side} signal on {sig.underlying} scored {score:.2f}, "
                   f"under the {min_score:.2f} minimum. Confluence was thin: "
                   + ("; ".join(sig.reasons[:3]) if sig.reasons else "no confirmations recorded") + ".")
        else:
            record(8, RULE_MIN_SCORE, _STATUS_OK,
                   f"The {sig.setup} {sig.side} signal on {sig.underlying} scored {score:.2f}, "
                   f"at or above the {min_score:.2f} minimum.")

        bucket = getattr(sig, "bucket", "") or ""
        if not bucket:
            record(8, RULE_BUCKET_BLOCKED, _STATUS_SKIP,
                   "No learning bucket on this signal, so the learner's block list was not consulted.")
            return
        if blocked_buckets is not None:
            if bucket in set(blocked_buckets):
                record(8, RULE_BUCKET_BLOCKED, _STATUS_BLOCK,
                       f"The learner has blocked the {bucket} bucket — this setup has lost money "
                       "in this time-of-day and regime often enough to stop taking it.")
            else:
                record(8, RULE_BUCKET_BLOCKED, _STATUS_OK, f"The {bucket} bucket is not blocked.")
            return
        if self.learner is None:
            record(8, RULE_BUCKET_BLOCKED, _STATUS_SKIP,
                   f"No learner attached, so the {bucket} bucket was not vetted for past losses.")
            return
        try:
            if self.learner.is_blocked(bucket):
                note = ""
                try:
                    note = getattr(self.learner.verdict(bucket), "note", "") or ""
                except Exception:  # noqa: BLE001 - the note is cosmetic, the block is not
                    log.warning("learner.verdict(%s) failed while building a block reason", bucket, exc_info=True)
                record(8, RULE_BUCKET_BLOCKED, _STATUS_BLOCK,
                       f"The learner has blocked the {bucket} bucket." + (f" {note}" if note else ""))
            else:
                record(8, RULE_BUCKET_BLOCKED, _STATUS_OK, f"The {bucket} bucket is not blocked.")
        except Exception as exc:  # noqa: BLE001 - advisory gate, never fatal
            # The learner is advisory. Rules 3-7 and 9 are what cap the money, so a
            # broken learner downgrades to "unvetted" rather than halting the bot, and
            # says so out loud instead of pretending the bucket was checked.
            log.warning("learner.is_blocked(%s) failed: %s", bucket, exc, exc_info=True)
            record(8, RULE_BUCKET_BLOCKED, _STATUS_SKIP,
                   f"The learner could not be consulted about the {bucket} bucket ({exc}); "
                   "this signal is unvetted by past results.")

    # -- rule 9 --------------------------------------------------------------------
    def _rule_9_funds(self, record: Any, leg: OptionLeg, mode: str, funds: dict | None) -> None:
        """Rule 9 — in live mode, `detailed_avl_balance.option_buy` covers the lot plus 5%."""
        if mode != "live":
            record(9, RULE_FUNDS, _STATUS_SKIP, f"Funds are not checked in {mode or 'unknown'} mode.")
            return
        lot_size = int(getattr(leg, "lot_size", 0) or 0)
        lot_cost = float(getattr(leg, "lot_cost", 0.0) or 0.0)
        ask = float(getattr(leg, "ask", 0.0) or 0.0)
        # The order goes out as a marketable LIMIT at the ask, so the ask is what the
        # account is actually debited. Sizing on the LTP alone under-states the cash
        # needed whenever the ask is above it; take the larger of the two.
        gross = max(lot_cost, ask * lot_size if (ask > 0 and lot_size > 0) else 0.0)
        needed = gross * FUNDS_HEADROOM
        available = self._option_buy_balance(funds)
        if available is None:
            record(9, RULE_FUNDS, _STATUS_BLOCK,
                   "Could not read detailed_avl_balance.option_buy from the funds payload. "
                   "Refusing to send a live order without knowing the buying balance.")
            return
        if available < needed:
            record(9, RULE_FUNDS, _STATUS_BLOCK,
                   f"Option buying balance is {inr(available)} but 1 lot of "
                   f"{getattr(leg, 'label', '?')} needs {inr(needed)} ({inr(gross)} of premium "
                   f"plus {pct(FUNDS_HEADROOM - 1)} for brokerage, GST and slippage). "
                   f"Short by {inr(needed - available)}.")
            return
        record(9, RULE_FUNDS, _STATUS_OK,
               f"Option buying balance {inr(available)} covers the {inr(needed)} this lot needs.")

    @staticmethod
    def _option_buy_balance(funds: dict | None) -> float | None:
        """Pull `detailed_avl_balance.option_buy` out of a /funds payload, wrapped or not.

        Accepts both the raw envelope ({"status":…, "data":{…}}) and the unwrapped data
        block, because the gate must not depend on where the API client chose to unwrap.
        """
        if not isinstance(funds, dict):
            return None
        block = funds.get("detailed_avl_balance")
        if not isinstance(block, dict):
            data = funds.get("data")
            block = data.get("detailed_avl_balance") if isinstance(data, dict) else None
        if not isinstance(block, dict):
            return None
        value = block.get("option_buy")
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    # -- day halt ------------------------------------------------------------------
    def day_halted(self, mode: str, *, day: date | None = None,
                   stats: dict | None = None) -> tuple[bool, str]:
        """(halted, reason) for `mode` today — the ₹2,000 limit, made sticky.

        The persisted flag is read FIRST and wins. A bot that restarts after hitting its
        daily loss must not resume trading just because its in-memory state was lost;
        that hole is exactly how a ₹2,000 limit becomes a ₹6,000 loss.
        """
        mode_norm = str(mode or "").strip().lower()
        d = day or clock.today()
        key = self._halt_key(mode_norm, d)

        try:
            raw = self.journal.get_meta(key, None)
        except Exception as exc:  # noqa: BLE001 - journal backend is sqlite3
            # Safe branch: if we cannot tell whether the day is halted, treat it as
            # halted. A missed trading day costs nothing; a missed halt costs rupees.
            log.error("could not read the halt flag %s: %s", key, exc, exc_info=True)
            return True, (f"Could not read today's halt flag from the journal ({exc}). "
                          "Treating the day as halted until the journal is readable again.")
        if raw:
            return True, self._halt_reason_from_meta(raw, mode_norm, d)

        if stats is None:
            stats, stats_error = self._day_stats(mode_norm, d)
            if stats is None:
                log.error("could not read day stats for the halt check: %s", stats_error)
                return True, (f"Could not read today's realised P&L ({stats_error}). Treating the "
                              "day as halted until the journal is readable again.")

        limit = self._num("max_daily_loss", 2000.0)
        realized = self._realized(stats)
        if realized <= -limit:
            reason = (f"Day halted: realised P&L is {inr(realized)}, at or past the "
                      f"{inr(limit)} daily loss limit. No more entries today in {mode_norm} mode.")
            self.halt_day(mode_norm, reason, day=d)
            return True, reason
        return False, ""

    def halt_day(self, mode: str, reason: str, *, day: date | None = None) -> None:
        """Persist the day halt for (mode, day) so a restart cannot resume trading.

        Also called by the engine's panic button, which halts the day after flattening.
        """
        mode_norm = str(mode or "").strip().lower()
        d = day or clock.today()
        key = self._halt_key(mode_norm, d)
        payload = json.dumps({
            "reason": reason,
            "mode": mode_norm,
            "day": d.isoformat(),
            "at": clock.now().isoformat(),
        })
        try:
            self.journal.set_meta(key, payload)
            log.warning("DAY HALTED (%s %s): %s", mode_norm, d.isoformat(), reason)
        except Exception as exc:  # noqa: BLE001 - journal backend is sqlite3
            # Do not swallow: the in-memory halt still applies this process, but the
            # user must know the flag will not survive a restart.
            log.error("FAILED to persist the day halt %s (%s). A restart would resume "
                      "trading on a day that already hit its loss limit.", key, exc, exc_info=True)

    def clear_halt(self, mode: str, *, day: date | None = None) -> bool:
        """Manually lift today's halt. Deliberate user override — logged loudly.

        Exists for the case where the halt came from a journal or data fault rather than
        from real losses. It is never called automatically.
        """
        mode_norm = str(mode or "").strip().lower()
        d = day or clock.today()
        key = self._halt_key(mode_norm, d)
        try:
            self.journal.set_meta(key, "")
        except Exception as exc:  # noqa: BLE001 - journal backend is sqlite3
            log.error("could not clear the halt flag %s: %s", key, exc, exc_info=True)
            return False
        log.warning("DAY HALT MANUALLY CLEARED for %s on %s — the daily loss limit is being "
                    "overridden by the user.", mode_norm, d.isoformat())
        return True

    @staticmethod
    def _halt_key(mode: str, day: date) -> str:
        """Meta key for the sticky day halt, scoped to (mode, IST date)."""
        return f"risk.halt.{mode}.{day.isoformat()}"

    def _halt_reason_from_meta(self, raw: str, mode: str, day: date) -> str:
        """Read back a persisted halt, tolerating a plain string instead of JSON."""
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return str(raw)
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or "").strip()
            at = str(parsed.get("at") or "").strip()
            if reason:
                return f"{reason}" + (f" (halted at {at})" if at else "")
        return (f"Day halted for {mode} on {day.isoformat()}; the halt flag is set but carries "
                "no reason.")

    # -- exits ---------------------------------------------------------------------
    def check_exit_required(self, pos: Position, ltp: float, *,
                            now: datetime | None = None) -> str | None:
        """The first exit reason that applies to `pos`, or None to keep holding.

        Order is deliberate: `flat_by` and the stop are checked BEFORE the target,
        because on a gap both can be true in the same tick and the safe branch is the
        exit, not the win.
        """
        if pos is None or not pos.is_open:
            return None
        now_dt = _aware(now) if now is not None else clock.now()

        # 1. flat_by — needs no price, and nothing is worth carrying past it.
        flat_by = clock.at(now_dt.date(), self._time_cfg("flat_by", DEFAULT_FLAT_BY))
        if now_dt >= flat_by:
            return EXIT_FLAT_BY

        if _is_price(ltp):
            stop = self.effective_stop(pos, ltp)
            trailing = self._trail_armed(pos, ltp)
            # 2/3. stop and trailing stop, before the target.
            if ltp <= stop:
                return EXIT_TRAIL if trailing else EXIT_STOP
            # 4. target.
            if _is_price(pos.target_price) and ltp >= pos.target_price:
                return EXIT_TARGET
        else:
            # A missing or zero quote is not a reason to sell. Flag it and fall through
            # to the time-based checks, which do not need a live price.
            log.warning("no usable LTP (%r) for position %s; price-based exits skipped this tick",
                        ltp, pos.id)

        # 5. time stop — held too long with nothing to show for it.
        if self._time_stop_hit(pos, now_dt):
            return EXIT_TIME_STOP
        return None

    def effective_stop(self, pos: Position, ltp: float | None = None) -> float:
        """The stop actually in force: the entry stop, raised by the trail once armed.

        Pure — it never mutates `pos`, so the UI and the engine can both ask.
        """
        stop = float(pos.stop_price or 0.0)
        entry = float(pos.entry_price or 0.0)
        trigger = self._num("trail_trigger_pct", 0.25)
        if entry <= 0 or trigger <= 0:
            return stop  # guard: no entry price means no ratio to trail against
        high_water = max(float(pos.high_water or 0.0), entry, float(ltp) if _is_price(ltp) else 0.0)
        if high_water >= entry * (1.0 + trigger):
            stop = max(stop, entry)  # stage 1: breakeven, the trade can no longer lose
        if high_water >= entry * (1.0 + 2.0 * trigger):
            stop = max(stop, entry * (1.0 + trigger / 2.0))  # stage 2: lock in half the trigger
        return stop

    def update_trail(self, pos: Position, ltp: float) -> float:
        """Advance the high-water mark and the trailing stop; returns the stop to persist.

        Mutates `pos.high_water`, `pos.trail_armed` and `pos.stop_price` in place. The
        stop only ever moves up — a trailing stop that can loosen is not a stop.
        """
        if pos is None:
            return 0.0
        if _is_price(ltp) and float(ltp) > float(pos.high_water or 0.0):
            pos.high_water = float(ltp)
        entry = float(pos.entry_price or 0.0)
        if entry <= 0:
            return float(pos.stop_price or 0.0)  # guard: cannot trail without an entry
        trigger = self._num("trail_trigger_pct", 0.25)
        if trigger > 0 and float(pos.high_water or 0.0) >= entry * (1.0 + trigger):
            pos.trail_armed = True
        new_stop = self.effective_stop(pos)
        if new_stop > float(pos.stop_price or 0.0):
            log.info("position %s: trailing stop raised from %.2f to %.2f (high water %.2f)",
                     pos.id, float(pos.stop_price or 0.0), new_stop, float(pos.high_water or 0.0))
            pos.stop_price = new_stop
        return float(pos.stop_price)

    def _trail_armed(self, pos: Position, ltp: float | None = None) -> bool:
        """True once the premium has made the trail trigger, flagged or merely reached."""
        if bool(getattr(pos, "trail_armed", False)):
            return True
        entry = float(pos.entry_price or 0.0)
        trigger = self._num("trail_trigger_pct", 0.25)
        if entry <= 0 or trigger <= 0:
            return False
        high_water = max(float(pos.high_water or 0.0), float(ltp) if _is_price(ltp) else 0.0)
        return high_water >= entry * (1.0 + trigger)

    def _time_stop_hit(self, pos: Position, now_dt: datetime) -> bool:
        """True when the position has been open too long with no progress toward target."""
        minutes_cap = self._num("time_stop_minutes", float(DEFAULT_TIME_STOP_MINUTES))
        if minutes_cap <= 0:
            return False
        entry = float(pos.entry_price or 0.0)
        if entry <= 0:
            return False  # guard: no entry price, no progress ratio
        try:
            opened_at = clock.to_ist(pos.entry_ts)
        except (TypeError, ValueError, OverflowError, OSError):
            log.warning("position %s has an unusable entry_ts %r; time stop skipped", pos.id, pos.entry_ts)
            return False
        open_minutes = (now_dt - opened_at).total_seconds() / 60.0
        if open_minutes < minutes_cap:
            return False
        # "Progress" is measured on the best premium SEEN, not the current one: a trade
        # that ran up and gave it back has had its chance and is handled by the trail.
        default_progress = self._num("trail_trigger_pct", 0.25) / 2.0
        min_progress = self._num("time_stop_min_progress_pct", default_progress)
        progress = (float(pos.high_water or entry) - entry) / entry
        return progress < min_progress

    # -- journal helpers -----------------------------------------------------------
    def _day_stats(self, mode: str, day: date) -> tuple[dict | None, str]:
        """Today's stats for `mode`, or (None, error) — never a silent zero.

        A journal read that fails must not read as "nothing lost today"; the caller
        turns the error into a block.
        """
        try:
            stats = self.journal.day_stats(day=day, mode=mode)
        except Exception as exc:  # noqa: BLE001 - journal backend is sqlite3
            log.error("journal.day_stats(%s, %s) failed: %s", day, mode, exc, exc_info=True)
            return None, str(exc) or exc.__class__.__name__
        if not isinstance(stats, dict):
            log.error("journal.day_stats returned %r, expected a dict", type(stats).__name__)
            return None, f"day_stats returned {type(stats).__name__}, expected a dict"
        return stats, ""

    @staticmethod
    def _realized(stats: dict) -> float:
        """Realised rupees today, coerced; an unreadable value reads as zero profit."""
        try:
            return float(stats.get("realized_pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            log.warning("day_stats.realized_pnl=%r is not a number; treating as 0",
                        stats.get("realized_pnl"))
            return 0.0

    @staticmethod
    def _trade_risk(sig: Signal, leg: OptionLeg) -> float:
        """Rupees this entry puts at risk: one lot's cost times the signal's stop."""
        try:
            lot_cost = float(getattr(leg, "lot_cost", 0.0) or 0.0)
            stop_pct = float(getattr(sig, "stop_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, lot_cost * stop_pct)

    @staticmethod
    def _open_risk(positions: list[Position]) -> float:
        """Rupees still at risk in open positions, from entry down to each stop."""
        total = 0.0
        for pos in positions:
            try:
                total += max(0.0, (float(pos.entry_price) - float(pos.stop_price)) * int(pos.qty))
            except (TypeError, ValueError):
                continue
        return total
