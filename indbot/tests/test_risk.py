"""Tests for indbot.risk — the nine entry rules and the exit conditions.

The risk gate is the last thing between a signal and a real order, so these tests
are about refusals, not about features:

* Every one of the nine rules is fired in isolation, and each one's REASON is
  checked for the numbers the user needs to act on. A block with a vague reason is
  a bot the user will disarm.
* ALL failures are reported, not just the first. The user has to see the whole list
  to know what to change.
* The ₹1,000 / 30%-stop arithmetic is asserted against a real ₹168 NIFTY ATM leg on
  a 75 lot: ₹12,600 of premium, ₹3,780 at risk, blocked. CONTRACTS.md is explicit
  that this is the limits working and must never be papered over.
* The day halt PERSISTS across a new RiskGate built on a restarted journal. An
  automated bot that forgets its halt on restart turns a ₹2,000 daily limit into a
  ₹6,000 one, so the halt is asserted after the losing trades have been deleted —
  only the persisted flag can produce it.
* check_exit_required prefers flat_by, then the stop, then the target. On a gap more
  than one can be true in a tick and the safe branch is the exit, not the win.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterator

import pytest

# The package root, so this file runs under a bare `pytest` from any directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indbot import clock  # noqa: E402
from indbot.config import Config  # noqa: E402
from indbot.errors import ConfigError  # noqa: E402
from indbot.journal import Journal  # noqa: E402
from indbot.risk import (  # noqa: E402
    EXIT_FLAT_BY,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME_STOP,
    EXIT_TRAIL,
    FUNDS_HEADROOM,
    RULE_ARMED,
    RULE_BUCKET_BLOCKED,
    RULE_DAY_BUDGET,
    RULE_DAY_HALT,
    RULE_FUNDS,
    RULE_MIN_SCORE,
    RULE_OPEN_POSITIONS,
    RULE_PER_TRADE_RISK,
    RULE_SESSION,
    RULE_STOP_SANITY,
    RULE_TRADE_COUNT,
    RiskGate,
    inr,
    pct,
)
from indbot.types import OptionLeg, Position, RiskVerdict, Signal  # noqa: E402

#: Monday 26 Oct 2026 — a weekday, not in clock.HOLIDAYS. 10:30 is inside the
#: 09:30-14:45 entry window and inside the 09:15-15:30 session.
DAY = date(2026, 10, 26)
FAKE_NOW = "2026-10-26 10:30:00"
NOW = clock.at(DAY, time(10, 30))

BUCKET = "orb:CE:morning:trend_up"

#: A ₹168 NIFTY ATM call on a 75 lot: ₹12,600 a lot, the headline block.
ATM_PREMIUM = 168.0
NIFTY_LOT = 75
#: ₹40 a unit on a 75 lot is ₹3,000 — a 30% stop risks ₹900, which fits the limit.
CHEAP_PREMIUM = 40.0


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin indbot.clock to one IST instant so "today" is never the machine's date."""
    monkeypatch.setenv("INDBOT_FAKE_NOW", FAKE_NOW)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Where the journal lives — a temp dir, never the user's data/indbot.db."""
    return tmp_path / "risk.db"


@pytest.fixture()
def journal(db_path: Path) -> Iterator[Journal]:
    """A real SQLite journal, so the gate is tested against real day statistics."""
    j = Journal(db_path)
    try:
        yield j
    finally:
        j.close()


@pytest.fixture()
def cfg() -> Config:
    """The user's stated limits: ₹2,000 a day, ₹1,000 a trade, 3 trades, 1 lot."""
    return Config()


@pytest.fixture()
def gate(journal: Journal, cfg: Config) -> RiskGate:
    """A gate over that journal and those limits."""
    return RiskGate(journal, cfg)


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _leg(
    premium: float = CHEAP_PREMIUM,
    *,
    lot_size: int = NIFTY_LOT,
    underlying: str = "NIFTY",
    strike: float = 25000.0,
    side: str = "CE",
    exchange: str = "NSE",
    spread: float = 0.5,
) -> OptionLeg:
    """One priced, tradable option leg."""
    return OptionLeg(
        underlying=underlying,
        expiry="2026-10-29",
        strike=strike,
        side=side,
        security_id="54321",
        trading_symbol=f"{underlying}29OCT{int(strike)}{side}",
        lot_size=lot_size,
        exchange=exchange,
        ltp=premium,
        bid=premium - spread,
        ask=premium + spread,
        oi=120000,
        volume=90000,
        iv=12.0,
        delta=0.51,
    )


def _signal(
    *,
    # Below high_conviction_score (0.75) on purpose: these tests exercise the
    # STANDARD risk tier. A default of 0.80 silently put every one of them in the
    # high-conviction tier, which doubles the per-trade cap and made rule 6 stop
    # firing. Tier behaviour is covered explicitly in TestHighConvictionTier.
    score: float = 0.60,
    stop_pct: float = 0.30,
    target_pct: float = 0.50,
    bucket: str = BUCKET,
    ts: int | None = None,
) -> Signal:
    """A BUY signal that clears min_score but stays in the standard tier."""
    return Signal(
        setup="orb",
        side="CE",
        action="BUY",
        underlying="NIFTY",
        ts=ts if ts is not None else clock.epoch_s(NOW),
        spot=25000.0,
        score=score,
        reasons=["close above the opening-range high (+0.35)", "EMA9>EMA21>EMA50 (+0.10)"],
        bucket=bucket,
        stop_pct=stop_pct,
        target_pct=target_pct,
    )


def _position(
    *,
    pos_id: str = "pos-1",
    entry: float = CHEAP_PREMIUM,
    stop: float = 28.0,
    target: float = 60.0,
    qty: int = NIFTY_LOT,
    entry_at: time = time(10, 0),
    high_water: float = 0.0,
    trail_armed: bool = False,
    leg: OptionLeg | None = None,
) -> Position:
    """An OPEN position. Defaults risk (40−28)×75 = ₹900, inside every limit."""
    return Position(
        id=pos_id,
        leg=leg if leg is not None else _leg(entry),
        signal_setup="orb",
        bucket=BUCKET,
        qty=qty,
        entry_price=entry,
        entry_ts=clock.epoch_s(clock.at(DAY, entry_at)),
        entry_order_id="order-1",
        stop_price=stop,
        target_price=target,
        mode="paper",
        high_water=high_water,
        trail_armed=trail_armed,
    )


def _blocked_rules(verdict: RiskVerdict) -> set[str]:
    """The rule ids that BLOCKED, parsed out of `verdict.checked`.

    `checked` is the audit trail ("6. per_trade_risk: BLOCK"); reading the ids back
    out of it is how these tests assert that a rule fired *and* that nothing else did.
    """
    out: set[str] = set()
    for entry in verdict.checked:
        _number, _, rest = entry.partition(". ")
        rule_id, _, status = rest.partition(": ")
        if status == "BLOCK":
            out.add(rule_id)
    return out


def _record_closed_trade(
    journal: Journal,
    *,
    pos_id: str,
    pnl_per_unit: float,
    qty: int = NIFTY_LOT,
    entry: float = 100.0,
    entry_at: time = time(9, 45),
    mode: str = "paper",
) -> None:
    """Write one closed trade for DAY so day_stats has something real to report."""
    pos = _position(pos_id=pos_id, entry=entry, stop=entry * 0.7, target=entry * 1.5,
                    qty=qty, entry_at=entry_at)
    pos.mode = mode
    journal.open_trade(pos, mode)
    pos.exit_price = entry + pnl_per_unit
    pos.exit_ts = clock.epoch_s(clock.at(DAY, time(10, 15)))
    pos.exit_reason = "target" if pnl_per_unit > 0 else "stop"
    journal.close_trade(pos)


def _entry(
    gate: RiskGate,
    *,
    sig: Signal | None = None,
    leg: OptionLeg | None = None,
    open_positions: list[Position] | None = None,
    mode: str = "paper",
    now: datetime | None = None,
    **kw: Any,
) -> RiskVerdict:
    """check_entry with this module's defaults filled in."""
    return gate.check_entry(
        sig if sig is not None else _signal(),
        leg if leg is not None else _leg(),
        open_positions=open_positions if open_positions is not None else [],
        mode=mode,
        now=now if now is not None else NOW,
        **kw,
    )


class _StubLearner:
    """A learner stand-in: the gate consults it for rule 8 and must not require it."""

    def __init__(self, blocked: set[str], *, note: str = "", explode: bool = False) -> None:
        self.blocked = blocked
        self.note = note
        self.explode = explode

    def is_blocked(self, bucket: str) -> bool:
        if self.explode:
            raise RuntimeError("learner database is locked")
        return bucket in self.blocked

    def verdict(self, bucket: str) -> Any:
        return type("V", (), {"note": self.note})()


# ----------------------------------------------------------------------------------
# the premise: a leg that fits does pass
# ----------------------------------------------------------------------------------
def test_a_cheap_leg_passes_every_rule(gate: RiskGate) -> None:
    """A ₹40 leg (₹3,000 a lot, ₹900 at risk) clears all nine rules.

    Asserted first, because a gate that blocks everything is indistinguishable from a
    gate that works, and the rest of this file is about blocks.
    """
    verdict = _entry(gate)

    assert verdict.ok is True
    assert verdict.blocking_rule is None
    assert _blocked_rules(verdict) == set()
    # A pass that cannot explain itself is indistinguishable from a bug.
    assert verdict.reasons, "a passing verdict must still say what it checked"
    assert any("₹900" in r for r in verdict.reasons)
    assert any("inside the ₹1,000 per-trade limit" in r for r in verdict.reasons)
    checked_rules = {entry.partition(". ")[2].partition(": ")[0] for entry in verdict.checked}
    assert {
        RULE_ARMED, RULE_SESSION, RULE_OPEN_POSITIONS, RULE_TRADE_COUNT, RULE_DAY_HALT,
        RULE_PER_TRADE_RISK, RULE_STOP_SANITY, RULE_MIN_SCORE, RULE_FUNDS,
    } <= checked_rules


# ----------------------------------------------------------------------------------
# rule 1 — live mode must be armed
# ----------------------------------------------------------------------------------
class TestRule1Armed:
    def test_live_without_arming_is_blocked(self, journal: Journal, cfg: Config) -> None:
        """mode=live with armed False never reaches the broker."""
        cfg.mode = "live"
        cfg.armed = False
        verdict = _entry(RiskGate(journal, cfg), mode="live", funds=_funds(10_000.0))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_ARMED
        assert RULE_ARMED in _blocked_rules(verdict)
        assert any("ARM LIVE TRADING" in r for r in verdict.reasons)

    def test_a_mode_mismatch_fails_closed(self, journal: Journal, cfg: Config) -> None:
        """Asked to trade live while the saved config says paper — refuse.

        Defence in depth: the broker asserts this too, but a gate that lets a live
        order through on a config mismatch is a gate that failed open.
        """
        cfg.mode = "paper"
        cfg.armed = True
        verdict = _entry(RiskGate(journal, cfg), mode="live", funds=_funds(10_000.0))

        assert verdict.blocking_rule == RULE_ARMED
        assert any("saved config mode is 'paper'" in r for r in verdict.reasons)

    def test_an_unknown_mode_is_refused(self, gate: RiskGate) -> None:
        """A typo'd mode must not be treated as harmless."""
        verdict = _entry(gate, mode="livee")

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_ARMED
        assert any("Unknown trading mode" in r for r in verdict.reasons)

    def test_armed_live_clears_rule_one(self, journal: Journal, cfg: Config) -> None:
        """mode=live, cfg.mode=live, armed=True: rule 1 passes and says so."""
        cfg.mode = "live"
        cfg.armed = True
        verdict = _entry(RiskGate(journal, cfg), mode="live", funds=_funds(10_000.0))

        assert RULE_ARMED not in _blocked_rules(verdict)
        assert verdict.ok is True


def _funds(option_buy: float) -> dict:
    """A /funds payload shaped like the API's detailed_avl_balance block."""
    return {"detailed_avl_balance": {"option_buy": option_buy}}


# ----------------------------------------------------------------------------------
# rule 2 — the session and the entry window
# ----------------------------------------------------------------------------------
class TestRule2Session:
    @pytest.mark.parametrize(
        ("when", "needle"),
        [
            (clock.at(date(2026, 10, 25), time(10, 30)), "The market is closed"),   # Sunday
            (clock.at(date(2026, 10, 26), time(8, 0)), "The market is closed"),     # pre-open
            (clock.at(date(2026, 10, 20), time(10, 30)), "The market is closed"),   # Dussehra
        ],
    )
    def test_outside_the_session_is_blocked(
        self, gate: RiskGate, when: datetime, needle: str
    ) -> None:
        """Weekends, holidays and out-of-hours all refuse, and name the time."""
        verdict = _entry(gate, now=when)

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_SESSION
        assert any(needle in r for r in verdict.reasons)

    def test_before_the_entry_window_is_blocked(self, gate: RiskGate) -> None:
        """09:20 is inside the session but before no_entry_before."""
        verdict = _entry(gate, now=clock.at(DAY, time(9, 20)))

        assert verdict.blocking_rule == RULE_SESSION
        assert _blocked_rules(verdict) == {RULE_SESSION}
        assert any("the entry window opens at 09:30" in r for r in verdict.reasons)

    def test_after_the_entry_window_is_blocked(self, gate: RiskGate) -> None:
        """15:00 is past no_entry_after, and the reason names flat_by."""
        verdict = _entry(gate, now=clock.at(DAY, time(15, 0)))

        assert verdict.blocking_rule == RULE_SESSION
        assert any("the entry window closed at 14:45" in r for r in verdict.reasons)
        assert any("15:10" in r for r in verdict.reasons)

    def test_flat_by_wins_when_it_is_earlier_than_no_entry_after(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Never open a position the bot would be forced to flatten minutes later."""
        cfg.no_entry_after = "14:45"
        cfg.flat_by = "13:00"
        verdict = _entry(RiskGate(journal, cfg), now=clock.at(DAY, time(14, 0)))

        assert verdict.blocking_rule == RULE_SESSION
        assert any("13:00 (flat_by, which is earlier than no_entry_after)" in r
                   for r in verdict.reasons)


# ----------------------------------------------------------------------------------
# rule 3 — the open-position cap
# ----------------------------------------------------------------------------------
def test_rule_3_one_position_at_a_time(gate: RiskGate) -> None:
    """With max_open_positions=1 a second entry is refused, naming what is held."""
    held = _position(pos_id="held-1")
    verdict = _entry(gate, open_positions=[held])

    assert verdict.ok is False
    assert verdict.blocking_rule == RULE_OPEN_POSITIONS
    assert _blocked_rules(verdict) == {RULE_OPEN_POSITIONS}
    assert any("the cap is 1" in r and "NIFTY 25000 CE" in r for r in verdict.reasons)


def test_rule_3_ignores_closed_positions(gate: RiskGate) -> None:
    """A closed position does not consume a slot."""
    closed = _position(pos_id="done-1")
    closed.exit_price = 55.0
    closed.exit_ts = clock.epoch_s(NOW)
    closed.exit_reason = "target"

    verdict = _entry(gate, open_positions=[closed])
    assert verdict.ok is True


# ----------------------------------------------------------------------------------
# rule 4 — trades per day
# ----------------------------------------------------------------------------------
def test_rule_4_daily_trade_cap(journal: Journal, gate: RiskGate) -> None:
    """Three entries today and the fourth is refused, counted by IST entry day."""
    for i in range(3):
        _record_closed_trade(journal, pos_id=f"t{i}", pnl_per_unit=1.0)
    assert journal.day_stats(day=DAY, mode="paper")["trade_count"] == 3

    verdict = _entry(gate)

    assert verdict.ok is False
    assert verdict.blocking_rule == RULE_TRADE_COUNT
    assert _blocked_rules(verdict) == {RULE_TRADE_COUNT}
    assert any("3 trades already taken today and the daily cap is 3" in r
               for r in verdict.reasons)


def test_rule_4_counts_only_this_mode(journal: Journal, gate: RiskGate) -> None:
    """Three live trades must not spend the paper day's budget, or vice versa."""
    for i in range(3):
        _record_closed_trade(journal, pos_id=f"L{i}", pnl_per_unit=1.0, mode="live")

    verdict = _entry(gate, mode="paper")
    assert RULE_TRADE_COUNT not in _blocked_rules(verdict)
    assert verdict.ok is True


# ----------------------------------------------------------------------------------
# rule 5 — the daily loss limit, and the budget left for one more trade
# ----------------------------------------------------------------------------------
class TestRule5DayLimit:
    def test_the_day_halts_at_the_daily_loss_limit(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """−₹2,550 realised is past the ₹2,000 limit: no more entries today."""
        _record_closed_trade(journal, pos_id="big-loss", pnl_per_unit=-34.0)
        assert journal.day_stats(day=DAY, mode="paper")["realized_pnl"] == -2550.0

        verdict = _entry(gate)

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_DAY_HALT
        assert any("Day halted" in r and "−₹2,550" in r.replace("-₹", "−₹")
                   for r in verdict.reasons)
        assert any("₹2,000 daily loss limit" in r for r in verdict.reasons)

    def test_the_forward_budget_blocks_a_trade_that_could_breach_the_limit(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """₹1,500 already lost + ₹1,000 at risk = −₹2,500, past a ₹2,000 limit.

        Rule 5 as the contract words it only looks backwards. The bot cannot buy less
        than one lot, so the only honest answer is to stand down.
        """
        _record_closed_trade(journal, pos_id="loss-1500", pnl_per_unit=-20.0)
        assert journal.day_stats(day=DAY, mode="paper")["realized_pnl"] == -1500.0

        verdict = _entry(gate, sig=_signal(), leg=_leg(CHEAP_PREMIUM))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_DAY_BUDGET
        assert any("could end the day at" in r for r in verdict.reasons)
        assert any("cannot buy less than one lot" in r for r in verdict.reasons)
        assert any("loss budget is left" in r for r in verdict.reasons)

    def test_the_forward_budget_counts_risk_already_on_the_table(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """An open position's entry-to-stop risk counts against the day's budget."""
        _record_closed_trade(journal, pos_id="loss-600", pnl_per_unit=-8.0)  # −₹600
        # Entry 100, stop 70, 75 units = ₹2,250 still at risk.
        held = _position(pos_id="held-2", entry=100.0, stop=70.0, target=150.0)

        verdict = RiskGate(gate.journal, gate.cfg).check_entry(
            _signal(), _leg(CHEAP_PREMIUM), open_positions=[held], mode="paper", now=NOW,
        )

        assert RULE_DAY_BUDGET in _blocked_rules(verdict)
        assert any("already at risk in open positions" in r for r in verdict.reasons)

    def test_the_forward_budget_can_be_switched_off(
        self, journal: Journal, cfg: Config
    ) -> None:
        """strict_day_budget=False restores the contract's literal rule 5."""
        _record_closed_trade(journal, pos_id="loss-1500", pnl_per_unit=-20.0)
        cfg.strict_day_budget = False  # type: ignore[attr-defined]

        verdict = _entry(RiskGate(journal, cfg))

        assert RULE_DAY_BUDGET not in _blocked_rules(verdict)
        assert verdict.ok is True


# ----------------------------------------------------------------------------------
# rules 6 and 7 — THE arithmetic
# ----------------------------------------------------------------------------------
class TestRules6And7Sizing:
    """₹1,000 a trade with a 30% stop caps one lot at ₹3,333 of premium."""

    def test_a_168_rupee_nifty_atm_leg_is_blocked_with_the_numbers(
        self, gate: RiskGate
    ) -> None:
        """The headline case from CONTRACTS.md, asserted arithmetic by arithmetic.

        1 lot of NIFTY at ₹168 on a 75 lot is ₹12,600. A 30% stop risks ₹3,780 —
        3.8x the user's ₹1,000 per-trade limit. The gate must block and must show
        the two honest choices: raise the limit, or buy a cheaper strike.
        """
        leg = _leg(ATM_PREMIUM, lot_size=NIFTY_LOT)
        assert leg.lot_cost == pytest.approx(12_600.0)
        assert leg.risk_at_stop(0.30) == pytest.approx(3_780.0)

        verdict = _entry(gate, leg=leg, sig=_signal(stop_pct=0.30))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_PER_TRADE_RISK
        # Rule 7 fires too: holding the loss to ₹1,000 on ₹12,600 needs a 7.9% stop.
        assert _blocked_rules(verdict) == {RULE_PER_TRADE_RISK, RULE_STOP_SANITY}

        reason = next(r for r in verdict.reasons if "per-trade limit" in r and "over the" in r)
        assert "1 lot of NIFTY 25000 CE (75 units)" in reason
        assert "at ₹168" in reason
        assert "is ₹12,600" in reason
        assert "a 30% stop risks ₹3,780" in reason
        assert "over the ₹1,000 per-trade limit" in reason
        # The two honest choices, both with their number attached.
        assert "raise the per-trade limit to ₹3,800" in reason
        assert "max ₹44 premium at this lot size" in reason

    def test_the_stop_sanity_reason_shows_the_implied_stop(self, gate: RiskGate) -> None:
        """Rule 7 on the same leg: ₹1,000 of ₹12,600 is a 7.9% stop, under the 12% floor."""
        verdict = _entry(gate, leg=_leg(ATM_PREMIUM), sig=_signal(stop_pct=0.30))

        reason = next(r for r in verdict.reasons if "floor" in r)
        assert "would need a 7.9% stop" in reason
        assert "tighter than the 12% floor" in reason
        assert "the bid-ask spread alone would trigger it" in reason
        assert "raise the per-trade limit to ₹1,512" in reason   # 12% of ₹12,600
        assert "at or under ₹111 a unit" in reason               # 1000/(0.12x75)
        assert "₹8,333 a lot" in reason

    def test_rule_6_fires_alone_when_the_implied_stop_is_still_sane(
        self, gate: RiskGate
    ) -> None:
        """₹80 a unit is ₹6,000 a lot: over budget, but a ₹1,000 stop is 16.7% — sane."""
        verdict = _entry(gate, leg=_leg(80.0), sig=_signal(stop_pct=0.30))

        assert _blocked_rules(verdict) == {RULE_PER_TRADE_RISK}
        assert any("a 30% stop risks ₹1,800" in r for r in verdict.reasons)

    def test_rule_7_stops_the_bot_solving_rule_6_with_a_tiny_stop(
        self, gate: RiskGate
    ) -> None:
        """A ₹10,000 lot with an 8% stop satisfies rule 6 and must still be refused.

        0.08 x ₹10,000 = ₹800, inside the per-trade limit — so rule 6 passes. But
        holding ₹10,000 of premium to a ₹1,000 loss needs a 10% stop, under the 12%
        floor, and a stop that tight is triggered by the spread alone. This is the
        exact "solution" min_stop_pct exists to refuse.
        """
        leg = _leg(500.0, lot_size=20, underlying="SENSEX", exchange="BSE", strike=82000.0)
        assert leg.lot_cost == pytest.approx(10_000.0)

        verdict = _entry(gate, leg=leg, sig=_signal(stop_pct=0.08))

        assert verdict.ok is False
        assert _blocked_rules(verdict) == {RULE_STOP_SANITY}
        assert verdict.blocking_rule == RULE_STOP_SANITY
        assert any("would need a 10% stop" in r for r in verdict.reasons)

    def test_a_signal_with_no_stop_is_refused(self, gate: RiskGate) -> None:
        """A 0% stop is no stop at all: the whole premium would be at risk."""
        verdict = _entry(gate, sig=_signal(stop_pct=0.0))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_PER_TRADE_RISK
        assert any("which is no stop at all" in r for r in verdict.reasons)

    @pytest.mark.parametrize(("lot_size", "premium"), [(0, 50.0), (75, 0.0), (75, -5.0)])
    def test_an_unpriceable_leg_is_refused(
        self, gate: RiskGate, lot_size: int, premium: float
    ) -> None:
        """No lot size or no price means one lot's cost cannot be computed. Refuse."""
        verdict = _entry(gate, leg=_leg(premium, lot_size=lot_size))

        assert verdict.ok is False
        assert RULE_PER_TRADE_RISK in _blocked_rules(verdict)
        assert RULE_STOP_SANITY in _blocked_rules(verdict)
        assert any("whose cost cannot be computed" in r for r in verdict.reasons)

    def test_a_whole_premium_under_the_limit_never_binds(self, gate: RiskGate) -> None:
        """When one lot costs less than the per-trade limit, rule 7 cannot bind."""
        leg = _leg(10.0, lot_size=75)  # ₹750 a lot, under the ₹1,000 limit
        verdict = _entry(gate, leg=leg)

        assert verdict.ok is True
        assert any("is never the binding constraint here" in r for r in verdict.reasons)

    def test_the_gate_agrees_with_the_configs_own_premium_cap(self, cfg: Config) -> None:
        """effective_max_premium and rule 6 must derive the same ₹44 a unit."""
        assert cfg.effective_max_premium(NIFTY_LOT) == pytest.approx(44.444, abs=1e-3)
        # Rule 6 floors it to a figure the user can act on without exceeding the limit.
        assert inr(44.0) == "₹44"
        assert 44.0 * NIFTY_LOT * cfg.stop_pct <= cfg.max_trade_loss

    def test_a_loose_premium_cap_is_called_out(self, journal: Journal, cfg: Config) -> None:
        """max_premium_per_lot looser than the per-trade limit is flagged in the reason."""
        cfg.max_premium_per_lot = 20_000.0
        verdict = _entry(RiskGate(journal, cfg), leg=_leg(ATM_PREMIUM))

        assert any("looser than this limit allows" in r for r in verdict.reasons)


# ----------------------------------------------------------------------------------
# rule 8 — score and the learner's block list
# ----------------------------------------------------------------------------------
class TestRule8Signal:
    def test_a_thin_signal_is_refused(self, gate: RiskGate) -> None:
        """0.20 is under the 0.55 minimum, and the reason quotes the confluence."""
        verdict = _entry(gate, sig=_signal(score=0.20))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_MIN_SCORE
        assert _blocked_rules(verdict) == {RULE_MIN_SCORE}
        assert any("scored 0.20, under the 0.55 minimum" in r for r in verdict.reasons)
        assert any("opening-range high" in r for r in verdict.reasons)

    def test_the_minimum_score_is_inclusive(self, gate: RiskGate) -> None:
        """Exactly min_score passes — the contract says `score >= min_score`."""
        verdict = _entry(gate, sig=_signal(score=0.55))
        assert RULE_MIN_SCORE not in _blocked_rules(verdict)

    def test_a_blocked_bucket_is_refused(self, gate: RiskGate) -> None:
        """An explicit blocked_buckets list stops the trade and names the bucket."""
        verdict = _entry(gate, blocked_buckets=[BUCKET])

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_BUCKET_BLOCKED
        assert _blocked_rules(verdict) == {RULE_BUCKET_BLOCKED}
        assert any(f"blocked the {BUCKET} bucket" in r for r in verdict.reasons)

    def test_the_learner_is_consulted_when_no_list_is_passed(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A learner attached to the gate supplies rule 8's block, note and all."""
        learner = _StubLearner({BUCKET}, note="Blocked: 6 wins in 24 trades (25%).")
        verdict = _entry(RiskGate(journal, cfg, learner=learner))

        assert verdict.blocking_rule == RULE_BUCKET_BLOCKED
        assert any("6 wins in 24 trades" in r for r in verdict.reasons)

    def test_a_broken_learner_downgrades_to_unvetted_rather_than_halting(
        self, journal: Journal, cfg: Config
    ) -> None:
        """The learner is advisory; rules 3-7 and 9 are what cap the money.

        It must say "unvetted" out loud rather than pretend the bucket was checked.
        """
        learner = _StubLearner(set(), explode=True)
        verdict = _entry(RiskGate(journal, cfg, learner=learner))

        assert verdict.ok is True
        assert any("could not be consulted" in entry or "skip" in entry
                   for entry in verdict.checked if RULE_BUCKET_BLOCKED in entry)

    def test_no_learner_is_not_a_silent_pass(self, gate: RiskGate) -> None:
        """Without a learner the audit trail records the bucket as un-vetted."""
        verdict = _entry(gate)
        assert any(f"{RULE_BUCKET_BLOCKED}: skip" in entry for entry in verdict.checked)


# ----------------------------------------------------------------------------------
# rule 9 — live funds
# ----------------------------------------------------------------------------------
class TestRule9Funds:
    def _live(self, journal: Journal, cfg: Config) -> RiskGate:
        cfg.mode = "live"
        cfg.armed = True
        return RiskGate(journal, cfg)

    def test_an_insufficient_option_buy_balance_is_refused(
        self, journal: Journal, cfg: Config
    ) -> None:
        """₹1,000 of buying power cannot fund a ₹3,000 lot plus charges."""
        verdict = _entry(self._live(journal, cfg), mode="live", funds=_funds(1_000.0))

        assert verdict.ok is False
        assert verdict.blocking_rule == RULE_FUNDS
        assert _blocked_rules(verdict) == {RULE_FUNDS}
        reason = next(r for r in verdict.reasons if "Short by" in r)
        assert "Option buying balance is ₹1,000" in reason
        assert pct(FUNDS_HEADROOM - 1) in reason  # "5%" for brokerage, GST and slippage

    def test_an_unreadable_funds_payload_is_refused(
        self, journal: Journal, cfg: Config
    ) -> None:
        """No balance figure means no live order. Never assume there is cash."""
        for payload in (None, {}, {"detailed_avl_balance": {}}, {"data": {}}):
            verdict = _entry(self._live(journal, cfg), mode="live", funds=payload)
            assert verdict.blocking_rule == RULE_FUNDS
            assert any("Refusing to send a live order" in r for r in verdict.reasons)

    def test_the_ask_not_the_ltp_decides_what_cash_is_needed(
        self, journal: Journal, cfg: Config
    ) -> None:
        """The order goes out as a marketable LIMIT at the ask, so size on the ask.

        LTP ₹40 x 75 = ₹3,000 but the ask is ₹60, so the real debit is ₹4,500 + 5%.
        A balance of ₹3,500 covers the LTP and must still be refused.
        """
        leg = _leg(CHEAP_PREMIUM, spread=20.0)  # bid 20, ask 60
        verdict = _entry(self._live(journal, cfg), leg=leg, mode="live",
                         funds=_funds(3_500.0))

        assert verdict.blocking_rule == RULE_FUNDS
        assert any("₹4,725" in r for r in verdict.reasons)  # 4,500 x 1.05

    def test_funds_are_not_checked_outside_live(self, gate: RiskGate) -> None:
        """Paper mode never needs cash; the audit trail says it was skipped."""
        verdict = _entry(gate, funds=None)
        assert verdict.ok is True
        assert any(f"{RULE_FUNDS}: skip" in entry for entry in verdict.checked)

    def test_a_wrapped_envelope_is_accepted(self, journal: Journal, cfg: Config) -> None:
        """{"data": {...}} and the bare block both work — the gate must not care."""
        verdict = _entry(self._live(journal, cfg), mode="live",
                         funds={"data": _funds(50_000.0)})
        assert verdict.ok is True


# ----------------------------------------------------------------------------------
# every failure is reported, not just the first
# ----------------------------------------------------------------------------------
def test_every_failing_rule_is_reported(journal: Journal, cfg: Config) -> None:
    """Ten rules fail at once and all ten reasons come back.

    The user has to see the whole list to know what to change; a gate that stops at
    the first failure sends them round the loop once per rule.
    """
    # 3 trades today, −₹2,550 realised: rules 4 and 5 both fail. They are filed under
    # "live" because that is the mode this entry is attempted in, and day_stats never
    # aggregates across modes.
    _record_closed_trade(journal, pos_id="a", pnl_per_unit=-34.0, mode="live")
    _record_closed_trade(journal, pos_id="b", pnl_per_unit=-1.0, mode="live")
    _record_closed_trade(journal, pos_id="c", pnl_per_unit=-1.0, mode="live")
    cfg.mode = "paper"      # rule 1: asked for live while the config says paper
    cfg.armed = False

    verdict = RiskGate(journal, cfg).check_entry(
        _signal(score=0.10),                       # rule 8a
        _leg(ATM_PREMIUM),                         # rules 6 and 7
        open_positions=[_position(pos_id="held")],  # rule 3
        mode="live",                               # rule 1
        now=clock.at(DAY, time(15, 20)),           # rule 2
        funds=_funds(5.0),                         # rule 9
        blocked_buckets=[BUCKET],                  # rule 8b
    )

    blocked = _blocked_rules(verdict)
    assert {
        RULE_ARMED,
        RULE_SESSION,
        RULE_OPEN_POSITIONS,
        RULE_TRADE_COUNT,
        RULE_DAY_HALT,
        RULE_PER_TRADE_RISK,
        RULE_STOP_SANITY,
        RULE_MIN_SCORE,
        RULE_BUCKET_BLOCKED,
        RULE_FUNDS,
    } <= blocked
    # One reason per failure — none dropped, and no passing rule's sentence leaked in.
    assert len(verdict.reasons) == len(blocked)
    assert all(isinstance(r, str) and len(r) > 20 for r in verdict.reasons)
    # blocking_rule names the FIRST failure, in rule order.
    assert verdict.blocking_rule == RULE_ARMED
    assert verdict.ok is False


def test_a_single_failure_does_not_hide_the_passing_rules(gate: RiskGate) -> None:
    """`checked` is a full audit trail even when something blocked."""
    verdict = _entry(gate, sig=_signal(score=0.10))

    assert len(verdict.reasons) == 1
    assert len(verdict.checked) >= 9
    assert any(": ok" in entry for entry in verdict.checked)


def test_an_unreadable_journal_refuses_to_trade(cfg: Config) -> None:
    """With the day's losses unknown, the ₹2,000 limit cannot be enforced. Refuse.

    The expensive failure here is reading a broken journal as "nothing lost today".
    """

    class _BrokenJournal:
        def day_stats(self, *a: Any, **kw: Any) -> dict:
            raise sqlite_error()

        def get_meta(self, key: str, default: str | None = None) -> str | None:
            return None

        def set_meta(self, key: str, value: str) -> None:
            return None

    def sqlite_error() -> Exception:
        return RuntimeError("database disk image is malformed")

    verdict = _entry(RiskGate(_BrokenJournal(), cfg))  # type: ignore[arg-type]

    assert verdict.ok is False
    assert {RULE_TRADE_COUNT, RULE_DAY_HALT} <= _blocked_rules(verdict)
    assert any("Refusing to trade rather than risk exceeding the daily cap" in r
               for r in verdict.reasons)
    assert any("cannot be enforced" in r for r in verdict.reasons)


# ----------------------------------------------------------------------------------
# the day halt survives a restart
# ----------------------------------------------------------------------------------
class TestDayHaltPersistence:
    """The restart hole: an in-memory-only halt turns ₹2,000 into ₹6,000."""

    def test_the_halt_survives_a_new_journal_and_a_new_gate(
        self, db_path: Path, cfg: Config
    ) -> None:
        """Halt, delete the losing trades, restart: still halted from the flag alone.

        Deleting the trades is what makes this a real test of persistence. If the
        halt were recomputed from today's P&L it would come back clear, because after
        the delete the day looks flat — which is exactly what a crash-and-restart
        looks like to a bot that only kept the halt in memory.
        """
        first = Journal(db_path)
        try:
            _record_closed_trade(first, pos_id="ruin", pnl_per_unit=-34.0)
            gate = RiskGate(first, cfg)

            halted, reason = gate.day_halted("paper", day=DAY)
            assert halted is True
            assert "−₹2,550" in reason.replace("-₹", "−₹")

            # The flag is persisted, scoped to (mode, IST date).
            assert first.get_meta(f"risk.halt.paper.{DAY.isoformat()}")
            first.clear_mode("paper")
            assert first.day_stats(day=DAY, mode="paper")["realized_pnl"] == 0.0
        finally:
            first.close()

        second = Journal(db_path)          # the restart
        try:
            revived = RiskGate(second, Config())
            halted, reason = revived.day_halted("paper", day=DAY)

            assert halted is True, "a restart must not resume trading on a halted day"
            assert "Day halted" in reason
            assert "halted at" in reason   # read back from the persisted payload

            verdict = revived.check_entry(
                _signal(), _leg(), open_positions=[], mode="paper", now=NOW,
            )
            assert verdict.ok is False
            assert verdict.blocking_rule == RULE_DAY_HALT
        finally:
            second.close()

    def test_the_halt_is_scoped_per_mode_and_per_day(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """Halting paper must not halt live, and tomorrow starts clean."""
        _record_closed_trade(journal, pos_id="ruin", pnl_per_unit=-34.0)
        assert gate.day_halted("paper", day=DAY)[0] is True

        assert gate.day_halted("live", day=DAY) == (False, "")
        assert gate.day_halted("paper", day=date(2026, 10, 27)) == (False, "")

    def test_a_manual_clear_lifts_it_and_is_the_only_thing_that_does(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """clear_halt is a deliberate override — nothing automatic unsets the flag."""
        _record_closed_trade(journal, pos_id="ruin", pnl_per_unit=-34.0)
        assert gate.day_halted("paper", day=DAY)[0] is True

        # Re-asking does not clear it, even after the trades are gone.
        journal.clear_mode("paper")
        assert gate.day_halted("paper", day=DAY)[0] is True

        assert gate.clear_halt("paper", day=DAY) is True
        assert gate.day_halted("paper", day=DAY) == (False, "")

    def test_halt_day_records_the_reason_the_engine_gave(
        self, journal: Journal, gate: RiskGate
    ) -> None:
        """The panic button halts the day too, and its reason is what the UI shows."""
        gate.halt_day("paper", "PANIC: flattened by the user", day=DAY)

        halted, reason = gate.day_halted("paper", day=DAY)
        assert halted is True
        assert "PANIC: flattened by the user" in reason

    def test_an_unreadable_halt_flag_is_treated_as_halted(self, cfg: Config) -> None:
        """If we cannot tell whether the day is halted, assume it is.

        A missed trading day costs nothing; a missed halt costs rupees.
        """

        class _NoMeta:
            def get_meta(self, key: str, default: str | None = None) -> str | None:
                raise RuntimeError("database is locked")

            def set_meta(self, key: str, value: str) -> None:
                return None

            def day_stats(self, *a: Any, **kw: Any) -> dict:
                return {"realized_pnl": 0.0, "trade_count": 0}

        halted, reason = RiskGate(_NoMeta(), cfg).day_halted("paper", day=DAY)  # type: ignore[arg-type]
        assert halted is True
        assert "Treating the day as halted" in reason


# ----------------------------------------------------------------------------------
# exits
# ----------------------------------------------------------------------------------
class TestCheckExitRequired:
    """Order: flat_by, then stop/trail, then target, then the time stop."""

    def test_flat_by_beats_a_position_sitting_on_its_target(
        self, gate: RiskGate
    ) -> None:
        """At 15:10 nothing is worth carrying, even a winner at its target."""
        pos = _position(entry=40.0, stop=28.0, target=60.0)

        assert gate.check_exit_required(pos, 61.0, now=clock.at(DAY, time(15, 10))) == EXIT_FLAT_BY
        assert gate.check_exit_required(pos, 61.0, now=clock.at(DAY, time(15, 25))) == EXIT_FLAT_BY
        # One minute earlier the target still wins — flat_by is not early.
        assert gate.check_exit_required(pos, 61.0, now=clock.at(DAY, time(15, 9))) == EXIT_TARGET

    def test_flat_by_beats_a_stop_too(self, gate: RiskGate) -> None:
        """It needs no price at all, so it is checked before anything else."""
        pos = _position(entry=40.0, stop=28.0, target=60.0)
        assert gate.check_exit_required(pos, 20.0, now=clock.at(DAY, time(15, 10))) == EXIT_FLAT_BY

    def test_the_stop_beats_the_target_when_both_are_true(self, gate: RiskGate) -> None:
        """A stop above the target is the only state where both predicates hold.

        With a normal stop < entry < target no single LTP satisfies both, so this
        inverts the pair — the shape a stop raised past a (tight) target leaves
        behind. The safe branch on that tick is the exit, not the win.
        """
        pos = _position(entry=100.0, stop=120.0, target=110.0, high_water=100.0)
        ltp = 115.0
        assert ltp <= pos.stop_price and ltp >= pos.target_price  # both fire

        assert gate.check_exit_required(pos, ltp, now=clock.at(DAY, time(10, 20))) == EXIT_STOP

    def test_a_trailed_stop_reports_itself_as_a_trail(self, gate: RiskGate) -> None:
        """Once the trail is armed the same exit is reported as `trail`, not `stop`.

        high_water ₹200 on a ₹100 entry is 2x the 25% trigger, so the stop is locked
        at entry + half the trigger = ₹112.50.
        """
        pos = _position(entry=100.0, stop=70.0, target=110.0, high_water=200.0)
        assert gate.effective_stop(pos) == pytest.approx(112.5)

        assert gate.check_exit_required(pos, 112.0, now=clock.at(DAY, time(10, 20))) == EXIT_TRAIL

    def test_the_target_fires_when_nothing_else_does(self, gate: RiskGate) -> None:
        """The happy path, so the ordering tests above mean something."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)
        assert gate.check_exit_required(pos, 151.0, now=clock.at(DAY, time(10, 20))) == EXIT_TARGET
        assert gate.check_exit_required(pos, 150.0, now=clock.at(DAY, time(10, 20))) == EXIT_TARGET

    def test_holding_is_the_default(self, gate: RiskGate) -> None:
        """Between the stop and the target, keep holding."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)
        assert gate.check_exit_required(pos, 101.0, now=clock.at(DAY, time(10, 20))) is None

    @pytest.mark.parametrize("bad_ltp", [0.0, -1.0, float("nan"), None])
    def test_a_missing_quote_is_not_a_reason_to_sell(
        self, gate: RiskGate, bad_ltp: Any
    ) -> None:
        """A zero or absent LTP skips the price exits instead of dumping the lot."""
        pos = _position(entry=100.0, stop=70.0, target=150.0, entry_at=time(10, 15))
        assert gate.check_exit_required(pos, bad_ltp, now=clock.at(DAY, time(10, 20))) is None

    def test_the_time_stop_fires_when_the_trade_has_gone_nowhere(
        self, gate: RiskGate
    ) -> None:
        """45 minutes open with no progress toward the target is a time stop."""
        pos = _position(entry=100.0, stop=70.0, target=150.0, entry_at=time(10, 0))
        assert gate.check_exit_required(pos, 101.0, now=clock.at(DAY, time(10, 44))) is None
        assert gate.check_exit_required(pos, 101.0, now=clock.at(DAY, time(10, 45))) == EXIT_TIME_STOP

    def test_the_time_stop_spares_a_trade_that_has_made_progress(
        self, gate: RiskGate
    ) -> None:
        """Progress is measured on the best premium SEEN; the trail handles give-backs."""
        pos = _position(entry=100.0, stop=70.0, target=150.0, entry_at=time(10, 0),
                        high_water=120.0)
        assert gate.check_exit_required(pos, 110.0, now=clock.at(DAY, time(11, 30))) is None

    def test_a_closed_position_needs_no_exit(self, gate: RiskGate) -> None:
        """Never sell twice."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)
        pos.exit_price = 150.0
        pos.exit_ts = clock.epoch_s(NOW)
        pos.exit_reason = EXIT_TARGET

        assert gate.check_exit_required(pos, 60.0, now=clock.at(DAY, time(10, 20))) is None

    def test_a_malformed_flat_by_degrades_instead_of_raising(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A bad HH:MM must not take the exit path down mid-session."""
        cfg.flat_by = "not a time"
        pos = _position(entry=100.0, stop=70.0, target=150.0)

        # Falls back to the 15:10 default rather than raising.
        assert RiskGate(journal, cfg).check_exit_required(
            pos, 101.0, now=clock.at(DAY, time(15, 11))
        ) == EXIT_FLAT_BY


class TestTrailing:
    """update_trail only ever ratchets the stop upward."""

    def test_the_stop_moves_to_breakeven_then_locks_in_half_the_trigger(
        self, gate: RiskGate
    ) -> None:
        """25% up arms the trail at breakeven; 50% up locks in 12.5%."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)

        assert gate.update_trail(pos, 110.0) == pytest.approx(70.0)   # not yet armed
        assert pos.trail_armed is False

        assert gate.update_trail(pos, 125.0) == pytest.approx(100.0)  # breakeven
        assert pos.trail_armed is True

        assert gate.update_trail(pos, 150.0) == pytest.approx(112.5)  # half the trigger

    def test_the_stop_never_loosens(self, gate: RiskGate) -> None:
        """A trailing stop that can fall back is not a stop."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)
        gate.update_trail(pos, 150.0)
        assert pos.stop_price == pytest.approx(112.5)

        gate.update_trail(pos, 101.0)
        assert pos.stop_price == pytest.approx(112.5)
        assert pos.high_water == pytest.approx(150.0)

    def test_effective_stop_does_not_mutate_the_position(self, gate: RiskGate) -> None:
        """The UI and the engine both ask, so it has to be pure."""
        pos = _position(entry=100.0, stop=70.0, target=150.0)
        before = (pos.stop_price, pos.high_water, pos.trail_armed)

        gate.effective_stop(pos, 200.0)

        assert (pos.stop_price, pos.high_water, pos.trail_armed) == before


# ----------------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------------
class TestConstruction:
    @pytest.mark.parametrize(
        ("setting", "value"),
        [("max_daily_loss", 0.0), ("max_daily_loss", -1.0),
         ("max_trade_loss", 0.0), ("min_stop_pct", 0.0)],
    )
    def test_a_non_positive_limit_is_refused_at_construction(
        self, journal: Journal, cfg: Config, setting: str, value: float
    ) -> None:
        """A zero limit silently disables the guard it is meant to enforce."""
        setattr(cfg, setting, value)
        with pytest.raises(ConfigError) as excinfo:
            RiskGate(journal, cfg)
        assert setting in str(excinfo.value)

    def test_formatting_reads_as_rupees(self) -> None:
        """Indian digit grouping, because a limit the user cannot read is not trusted."""
        assert inr(12_600) == "₹12,600"
        assert inr(3_780.0) == "₹3,780"
        assert inr(1_234_567) == "₹12,34,567"
        assert inr(-2_550.0) == "-₹2,550"
        assert inr(None) == "₹?"
        assert pct(0.30) == "30%"
        assert pct(1_000 / 12_600) == "7.9%"


class TestHighConvictionTier:
    """A high score may risk more, but only at the required reward:risk.

    Asked for directly: "If the confidence is high you can take the high risk
    trades also. But in that case the risk reward ratio should be 1:2 then."
    """

    def test_below_the_threshold_nothing_changes(self) -> None:
        cfg = Config()
        t = cfg.tier_for(cfg.high_conviction_score - 0.01)
        assert t["tier"] == "standard"
        assert t["max_trade_loss"] == cfg.max_trade_loss
        assert t["target_pct"] == cfg.target_pct

    def test_at_the_threshold_the_tier_applies(self) -> None:
        cfg = Config()
        t = cfg.tier_for(cfg.high_conviction_score)
        assert t["tier"] == "high_conviction"
        assert t["max_trade_loss"] >= cfg.max_trade_loss

    def test_the_reward_to_risk_is_exactly_the_configured_ratio(self) -> None:
        """The target is REWRITTEN, not left at the ordinary 1:1.67."""
        cfg = Config()
        for stop in (0.20, 0.30, 0.375):
            t = cfg.tier_for(0.90, stop_pct=stop)
            assert t["stop_pct"] == stop
            assert t["target_pct"] == pytest.approx(stop * cfg.high_conviction_rr)
            assert t["rr"] == pytest.approx(cfg.high_conviction_rr)

    def test_the_tier_is_never_tighter_than_standard(self) -> None:
        """A day cap under the per-trade cap must not punish a better signal.

        The first version clamped to max_daily_loss unconditionally, so a config
        with a 2,000 day cap and a 4,000 per-trade cap gave high-conviction LESS
        room than an ordinary trade.
        """
        cfg = Config().copy_with({"max_trade_loss": 4000.0, "max_daily_loss": 2000.0})
        assert cfg.tier_for(0.90)["max_trade_loss"] >= cfg.max_trade_loss

    def test_it_never_exceeds_the_day_limit_when_that_is_the_wider_bound(self) -> None:
        cfg = Config().copy_with({"max_trade_loss": 1000.0, "max_daily_loss": 1500.0})
        assert cfg.tier_for(0.90)["max_trade_loss"] <= max(cfg.max_daily_loss, cfg.max_trade_loss)

    def test_disabling_it_restores_the_standard_tier(self) -> None:
        cfg = Config().copy_with({"high_conviction_enabled": False})
        assert cfg.tier_for(0.99)["tier"] == "standard"

    def test_the_note_states_the_numbers(self) -> None:
        """The UI prints this verbatim, so it has to read as a sentence."""
        note = Config().tier_for(0.85, stop_pct=0.30)["note"]
        assert "0.85" in note and "2:1" in note
        assert "30%" in note and "60%" in note

    def test_the_gate_uses_the_tier_not_the_standard_cap(self, journal: Journal) -> None:
        """Rule 6 must size against the tier, or the raise does nothing."""
        cfg = Config().copy_with({"max_daily_loss": 50_000.0})
        gate = RiskGate(journal, cfg)
        hc = cfg.tier_for(0.85)["max_trade_loss"]
        # A lot whose stop risks more than standard but less than high-conviction.
        risk_target = (cfg.max_trade_loss + hc) / 2
        ltp = risk_target / (0.30 * 75)
        leg = _leg(round(ltp, 2))
        blocked_std = _blocked_rules(_entry(gate, leg=leg, sig=_signal(score=0.60)))
        blocked_hc = _blocked_rules(_entry(gate, leg=leg, sig=_signal(score=0.85)))
        assert RULE_PER_TRADE_RISK in blocked_std, "standard tier should refuse this lot"
        assert RULE_PER_TRADE_RISK not in blocked_hc, "high-conviction tier should allow it"
