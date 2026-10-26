"""End-to-end integration tests for indbot. No network, no real clock, no real money.

Every test here drives the real `Engine`, `RiskGate`, `ChainPicker`, `Journal`,
`Learner` and brokers against `conftest.FakeClient`. Nothing is mocked out with
`unittest.mock`: the point is to prove the modules agree with each other, and a mock
of a collaborator is a mock of the thing under test.

Run it with any of:

    python3 -m unittest discover -s tests
    python3 -m pytest tests/test_integration.py
    python3 tests/test_integration.py

(`-t .` needs a `tests/__init__.py`, which this package does not have; the suite puts
the package root on `sys.path` itself so it runs from any working directory.)

What each class pins down:

* `TestOrbDayBlockedByRiskArithmetic` — the headline case. A real ORB signal fires,
  the chain picker refuses the at-the-money strike because one lot costs more than the
  per-trade loss limit allows, and the journal keeps that arithmetic verbatim. This is
  the behaviour CONTRACTS.md calls "not a bug", and the one most likely to be
  "fixed" by a future change that quietly widens a stop.
* `TestPaperTradeLifecycle` — the same day, with the limit raised far enough, opens a
  paper position, trails it to break-even and exits on the trail.
* `TestPaperBrokerPlacesNoOrders` — asserted on the recorded wire calls.
* `TestLiveBrokerArmGate` — live mode without `armed` raises OrderError, before
  anything reaches the API.
* `TestSetModeFlattensLivePosition` — leaving live mode squares off first.
* `TestDayHaltPersistence` — the daily-loss halt survives a new RiskGate and a new
  Journal connection over the same file.
* `TestModuleHealth` — every module imports, and `Config.load()` works with no
  config.json present.
* `TestFixtureShapes` — guards the fixtures against drifting away from the documented
  API response shapes, so the tests above keep testing something real.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest as fx  # noqa: E402

from indbot import clock  # noqa: E402
from indbot.broker import LiveBroker, PaperBroker, QuoteOnlyClient  # noqa: E402
from indbot.chain import ChainPicker  # noqa: E402
from indbot.config import ARM_PHRASE, DEFAULTS, Config  # noqa: E402
from indbot.errors import ConfigError, OrderError  # noqa: E402
from indbot.journal import Journal  # noqa: E402
from indbot.risk import RiskGate  # noqa: E402
from indbot.strategy import Strategy  # noqa: E402

fx.quiet_logging()

#: Every module in the package, in dependency order. `TestModuleHealth` imports all
#: of them; the ordering is cosmetic but makes a failure easier to read.
MODULES = (
    "types", "clock", "errors", "config", "api", "instruments", "indicators",
    "strategy", "chain", "journal", "learner", "risk", "broker", "engine",
    "backtest", "server",
)

#: The per-trade limit at which one lot of the chosen NIFTY strike becomes legal.
#: One lot of the 24500 CE costs about Rs.11,194 and a 30% stop risks about Rs.3,358,
#: so Rs.8,000 clears rule 6 with room to spare while staying a plausible setting.
LOOSE_TRADE_LOSS = 8000.0
LOOSE_DAILY_LOSS = 20000.0


class IndbotTestCase(unittest.TestCase):
    """Shared plumbing: a temp directory, a pinned clock and a quiet logger."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="indbot-test-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.pin(fx.ENTRY_AT)

    def pin(self, stamp: str) -> None:
        """Pin `clock.now()` for the rest of the test, restoring it afterwards."""
        ctx = fx.fake_now(stamp)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def assertNoOrders(self, client: fx.FakeClient, why: str = "") -> None:
        """Fail if anything reached an order endpoint on `client`."""
        self.assertEqual(
            [], client.order_calls,
            f"{why or 'no order endpoint should have been called'}; got "
            f"{[name for name, _ in client.order_calls]}",
        )


# ----------------------------------------------------------------------------------
# 1. the ORB day that is correctly refused
# ----------------------------------------------------------------------------------
class TestOrbDayBlockedByRiskArithmetic(IndbotTestCase):
    """A real signal, refused on the premium arithmetic, and the refusal is recorded.

    The default limits (Rs.1,000 a trade, a 30% stop) cap NIFTY premium at Rs.44 a
    unit while an at-the-money call costs about Rs.168. CONTRACTS.md is explicit that
    this must block loudly rather than be papered over, so these assertions are the
    guard on that: a change that lets this trade through has either widened the stop,
    raised the loss limit, or drifted to lottery strikes behind the user's back.
    """

    def setUp(self) -> None:
        super().setUp()
        self.cfg = fx.make_config(self.tmp)                 # default Rs.1,000 per trade
        self.journal = fx.make_journal(self.tmp)
        self.client = fx.FakeClient()
        self.engine = fx.make_engine(self.cfg, self.journal, self.client, self.tmp)
        self.result = self.engine.tick()

    def test_orb_signal_fires_on_the_breakout_candle(self) -> None:
        """The 09:30 close above the opening-range high produces one BUY CE signal."""
        strategy = Strategy(self.cfg)
        signals = strategy.evaluate(
            "NIFTY", fx.nifty_candles(), now_ts=fx.at(fx.TRADING_DAY, "09:35")
        )
        self.assertEqual(1, len(signals), "exactly one setup is enabled, so one signal")
        sig = signals[0]
        self.assertEqual("orb", sig.setup)
        self.assertEqual("CE", sig.side)
        self.assertEqual("BUY", sig.action)
        self.assertEqual(fx.at(fx.TRADING_DAY, "09:30"), sig.ts,
                         "the signal must be stamped with the last CLOSED candle")
        self.assertTrue(sig.bucket.startswith("orb:CE:open:"),
                        f"bucket should be setup:side:tod:regime, got {sig.bucket!r}")
        self.assertGreater(sig.stop_pct, 0.0)
        self.assertTrue(any("range high" in r for r in sig.reasons),
                        f"the ORB reason should name the range it broke: {sig.reasons}")
        self.assertEqual(1, self.result["signals"])

    def test_evaluate_never_reads_a_forming_candle(self) -> None:
        """At 09:35 the 09:35 bar is still forming, so the trigger is the 09:30 bar.

        A signal built from a forming candle is look-ahead bias and the backtest and
        the live loop would disagree, so this is checked rather than assumed.
        """
        strategy = Strategy(self.cfg)
        signals = strategy.evaluate(
            "NIFTY", fx.nifty_candles(), now_ts=fx.at(fx.TRADING_DAY, "09:35")
        )
        self.assertLess(signals[0].ts, fx.at(fx.TRADING_DAY, "09:35"))

    def test_chain_picker_rejects_the_atm_strike_on_the_premium_budget(self) -> None:
        """`pick` returns None and names the at-the-money strike with the arithmetic."""
        picker = ChainPicker(self.client, self.engine.instruments, self.cfg)
        leg, reasons = picker.pick("NIFTY", "CE", spot=24472.0, stop_pct=self.cfg.stop_pct)

        self.assertIsNone(leg, "no strike is affordable under a Rs.1,000 per-trade limit")
        joined = " ".join(reasons)
        self.assertIn("NIFTY 24450 CE", joined, "the ATM strike must be named")
        self.assertIn("allowed to cost", joined, "the rejection must state the budget")
        self.assertIn("stop risks", joined, "the rejection must state the rupees at risk")
        self.assertIn("per-trade budget had already removed", joined,
                      "the root cause — the budget removed the strikes near the money — "
                      "must be stated, not just the per-leg rejections")

        atm_line = next((r for r in reasons if "NIFTY 24450 CE" in r), "")
        self.assertIn("75 units", atm_line, "one lot is 75 units and the reason says so")
        # 167.90 x 75 = Rs.12,592.50. chain.py's formatter rounds that to 12,593 while
        # risk.py's truncates it to 12,592 (see the other test below); both strings are
        # shown in the same UI panel. Asserted as each module actually renders it, so
        # the inconsistency is recorded rather than hidden behind a loose match.
        self.assertIn("12,593", atm_line,
                      "the lot cost must be in the sentence, not just the verdict")
        self.assertIn("3,778", atm_line, "a 30% stop on Rs.12,592.50 risks Rs.3,778")

    def test_risk_gate_blocks_the_atm_leg_with_the_same_arithmetic(self) -> None:
        """Rule 6 reports the lot cost, the rupees at risk and both honest remedies.

        Checked independently of the picker: the gate is the second lock on this door
        and must hold even if a future chain picker offers the leg anyway.
        """
        leg = fx.leg_from_chain(fx.ATM_STRIKE, "CE")
        strategy = Strategy(self.cfg)
        sig = strategy.evaluate(
            "NIFTY", fx.nifty_candles(), now_ts=fx.at(fx.TRADING_DAY, "09:35")
        )[0]
        gate = RiskGate(self.journal, self.cfg)
        verdict = gate.check_entry(sig, leg, open_positions=[], mode="paper")

        self.assertFalse(verdict.ok)
        self.assertEqual("per_trade_risk", verdict.blocking_rule)
        blocked = " ".join(verdict.reasons)
        self.assertIn("12,592", blocked)
        self.assertIn("per-trade limit", blocked)
        self.assertIn("cheaper strike", blocked,
                      "the user must be offered both honest choices, not just the block")

    def test_journal_records_the_signal_as_not_taken_with_that_reason(self) -> None:
        """The signal is written with `taken=False` and the arithmetic in full."""
        rows = self.journal.signals(day=fx.TRADING_DAY)
        self.assertEqual(1, len(rows), "one signal, recorded once per candle")
        row = rows[0]
        self.assertFalse(row["taken"], "the signal must be recorded as NOT taken")
        self.assertEqual("orb", row["setup"])
        self.assertEqual("CE", row["side"])

        reason = row["block_reason"]
        self.assertIn("No tradable strike", reason)
        self.assertIn("NIFTY 24450 CE", reason)
        self.assertIn("allowed to cost", reason)
        self.assertIn("per-trade budget had already removed", reason)
        # The "why no trade" panel renders this verbatim, so it must be prose.
        self.assertNotIn("{", reason)

    def test_nothing_was_traded_and_no_order_was_sent(self) -> None:
        """No position, no trade row, no day P&L and nothing on the wire."""
        self.assertEqual([], self.result["entries"])
        self.assertEqual([], self.journal.trades())
        day = self.journal.day_stats(fx.TRADING_DAY, "paper")
        self.assertEqual(0, day["trade_count"])
        self.assertEqual(0.0, day["realized_pnl"])
        self.assertEqual(0, day["open_count"])
        self.assertNoOrders(self.client, "a blocked signal must not reach the broker")

    def test_the_signal_is_recorded_once_however_often_the_bot_ticks(self) -> None:
        """Fifteen ticks inside one 5-minute bar must not write fifteen rows.

        The poll is 20 seconds and the bar is 5 minutes: without the engine's
        per-candle deduplication the "why no trade" feed is unreadable.
        """
        for _ in range(5):
            self.engine.tick()
        self.assertEqual(1, len(self.journal.signals(day=fx.TRADING_DAY)))
        self.assertNoOrders(self.client)


# ----------------------------------------------------------------------------------
# 2. the same day, traded
# ----------------------------------------------------------------------------------
class TestPaperTradeLifecycle(IndbotTestCase):
    """With the per-trade limit raised, the same day opens, trails and exits.

    The numbers are deterministic and worth stating, because they are the check that
    the fill model is the pessimistic one CONTRACTS.md asks for: the BUY fills through
    the ask (Rs.149.90 ask + one 5-paisa tick, rounded away from the touch, Rs.150.00)
    and the SELL fills through the bid (Rs.148.60 bid - one tick, rounded down,
    Rs.148.50). Paper is never allowed to fill better than live would.
    """

    def setUp(self) -> None:
        super().setUp()
        self.cfg = fx.make_config(
            self.tmp, max_trade_loss=LOOSE_TRADE_LOSS, max_daily_loss=LOOSE_DAILY_LOSS
        )
        self.journal = fx.make_journal(self.tmp)
        self.client = fx.FakeClient()
        self.engine = fx.make_engine(self.cfg, self.journal, self.client, self.tmp)
        self.entry_result = self.engine.tick()
        open_now = self.engine._open_positions()  # noqa: SLF001 - the engine's own book
        self.assertEqual(1, len(open_now), f"expected one position, got {open_now}")
        self.position = open_now[0]

    def test_entry_opens_one_lot_of_a_strike_inside_the_delta_band(self) -> None:
        """One lot, 75 units, ranked nearest a 0.5 |delta| and inside the budget."""
        leg = self.position.leg
        self.assertEqual("NIFTY 24500 CE", leg.label)
        self.assertEqual("CE", leg.side)
        self.assertEqual(75, leg.lot_size)
        self.assertEqual(75, self.position.qty, "qty is UNITS: one lot of 75")
        self.assertEqual(1, self.position.lots)
        self.assertEqual(150.0, self.position.entry_price)
        self.assertEqual("paper", self.position.mode)

        self.assertGreaterEqual(abs(leg.delta), self.cfg.min_delta)
        self.assertLessEqual(abs(leg.delta), self.cfg.max_delta)
        self.assertLessEqual(leg.spread_pct, self.cfg.max_spread_pct)
        self.assertGreaterEqual(leg.oi, self.cfg.min_oi)
        self.assertGreaterEqual(leg.volume, self.cfg.min_volume)
        self.assertLessEqual(
            leg.lot_cost * self.position_stop_pct(), self.cfg.max_trade_loss,
            "a lot that passed the gate must risk no more than the per-trade limit",
        )

    def position_stop_pct(self) -> float:
        """The stop the position was actually opened with, as a fraction of premium."""
        return (self.position.entry_price - self.position.stop_price) / self.position.entry_price

    def test_stop_and_target_are_set_from_the_signal(self) -> None:
        """Stop below entry, target above, both derived from the fill price."""
        self.assertEqual(105.0, self.position.stop_price)   # 150.00 x (1 - 0.30)
        self.assertEqual(225.0, self.position.target_price)  # 150.00 x (1 + 0.50)
        self.assertEqual(150.0, self.position.high_water)
        self.assertFalse(self.position.trail_armed)

    def test_trade_row_is_journalled_as_open(self) -> None:
        """The open is persisted before anything can go wrong with it."""
        rows = self.journal.trades(day=fx.TRADING_DAY, mode="paper")
        self.assertEqual(1, len(rows))
        self.assertEqual("open", rows[0]["status"])
        self.assertEqual(75, rows[0]["qty"])
        self.assertEqual(24500.0, rows[0]["strike"])
        self.assertEqual(fx.EXPIRY, rows[0]["expiry"])
        self.assertEqual("paper", rows[0]["mode"])

    def test_the_taken_signal_is_recorded_as_taken(self) -> None:
        """The journal links the signal to the trade it became."""
        rows = self.journal.signals(day=fx.TRADING_DAY)
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0]["taken"])
        self.assertEqual(self.position.id, rows[0]["trade_id"])

    def test_trail_arms_at_the_trigger_and_lifts_the_stop_to_break_even(self) -> None:
        """At +27% the trail arms and the stop moves up to the entry, never down."""
        fx.set_now("2026-10-26 09:40:00")
        self.client.set_ltp(self.position.leg.security_id, 190.0)   # 150 x 1.267
        result = self.engine.tick()

        self.assertEqual("at_position_cap", result["phase"],
                         "one position is the cap, so the bot must stop before scanning")
        self.assertEqual([], result["exits"])
        self.assertTrue(self.position.trail_armed)
        self.assertEqual(190.0, self.position.high_water)
        self.assertEqual(150.0, self.position.stop_price,
                         "stage one of the trail is break-even")
        self.assertTrue(self.position.is_open)

    def test_position_exits_on_the_trailing_stop_and_is_journalled(self) -> None:
        """Walk it up, then back through the raised stop: exit reason is `trail`."""
        fx.set_now("2026-10-26 09:40:00")
        self.client.set_ltp(self.position.leg.security_id, 190.0)
        self.engine.tick()

        fx.set_now("2026-10-26 09:45:00")
        self.client.set_ltp(self.position.leg.security_id, 145.0)   # under the 150 trail
        result = self.engine.tick()

        self.assertEqual(1, len(result["exits"]))
        exit_row = result["exits"][0]
        self.assertEqual("trail", exit_row["reason"],
                         "below a trail-raised stop is a trail exit, not a stop exit")
        self.assertEqual(148.5, exit_row["exit_price"], "a paper SELL fills through the bid")
        self.assertEqual([], self.engine._open_positions())  # noqa: SLF001

        trades = self.journal.trades(day=fx.TRADING_DAY, mode="paper")
        self.assertEqual(1, len(trades))
        closed = trades[0]
        self.assertEqual("closed", closed["status"])
        self.assertEqual("trail", closed["exit_reason"])
        self.assertEqual(150.0, closed["entry_price"])
        self.assertEqual(148.5, closed["exit_price"])
        # (148.50 - 150.00) x 75 = -112.50, less Rs.20 per order on two orders.
        self.assertAlmostEqual(-152.5, closed["pnl"], places=2)

        day = self.journal.day_stats(fx.TRADING_DAY, "paper")
        self.assertEqual(1, day["trade_count"])
        self.assertEqual(0, day["open_count"])
        self.assertEqual(1, day["losses"])
        self.assertAlmostEqual(-152.5, day["realized_pnl"], places=2)

    def test_exits_at_the_target_when_the_premium_gets_there(self) -> None:
        """A move straight to the target closes the trade as `target`.

        The trail trigger (+25%) is passed on the way to the target (+50%), and
        `check_exit_required` must still report the target rather than the trail,
        because the premium is above the trailed stop, not below it.
        """
        fx.set_now("2026-10-26 09:40:00")
        self.client.set_ltp(self.position.leg.security_id, 230.0)   # over the 225 target
        result = self.engine.tick()

        self.assertEqual(1, len(result["exits"]))
        self.assertEqual("target", result["exits"][0]["reason"])
        self.assertEqual("closed", self.journal.trades(mode="paper")[0]["status"])

    def test_the_whole_lifecycle_sent_no_order_to_the_api(self) -> None:
        """Entry, trail and exit, all simulated: the wire stays clean."""
        fx.set_now("2026-10-26 09:40:00")
        self.client.set_ltp(self.position.leg.security_id, 190.0)
        self.engine.tick()
        fx.set_now("2026-10-26 09:45:00")
        self.client.set_ltp(self.position.leg.security_id, 145.0)
        self.engine.tick()

        self.assertNoOrders(self.client, "paper mode must never reach POST /order")
        self.assertTrue(self.client.method_calls("ltp"), "but it must read quotes")

    def test_snapshot_carries_every_key_the_ui_renders(self) -> None:
        """`snapshot()` answers the contract's keys even mid-position."""
        snap = self.engine.snapshot()
        for key in (
            "mode", "armed", "market_open", "now", "session", "halted", "halt_reason",
            "config", "funds", "day", "underlyings", "positions", "recent_trades",
            "recent_signals", "leaderboard", "blocked_buckets", "log", "errors",
        ):
            self.assertIn(key, snap)
        self.assertEqual("paper", snap["mode"])
        self.assertFalse(snap["armed"])
        self.assertTrue(snap["market_open"])
        self.assertFalse(snap["halted"])
        self.assertEqual([], snap["errors"], f"a clean run must record no errors: {snap['errors']}")
        self.assertEqual(1, len(snap["positions"]))
        self.assertEqual(1, len(snap["underlyings"]))
        view = snap["underlyings"][0]
        self.assertEqual("NIFTY", view["name"])
        self.assertIsNone(view.get("data_error"), "a healthy feed reports no data error")
        self.assertIsNotNone(view["or_high"], "the opening range must be on the card")
        # Everything the UI renders has to survive a JSON round trip.
        self.assertIsInstance(json.dumps(snap, default=str), str)


# ----------------------------------------------------------------------------------
# 3. the paper broker, structurally
# ----------------------------------------------------------------------------------
class TestPaperBrokerPlacesNoOrders(IndbotTestCase):
    """PaperBroker cannot reach an order endpoint, by construction and in practice."""

    def setUp(self) -> None:
        super().setUp()
        self.cfg = fx.make_config(self.tmp)
        self.client = fx.FakeClient()
        self.broker = PaperBroker(self.cfg, self.client)
        self.leg = fx.leg_from_chain(fx.ATM_STRIKE, "CE")

    def test_buy_and_sell_simulate_without_touching_the_api(self) -> None:
        """A full round trip fills on both sides and records nothing on the wire."""
        bought = self.broker.buy(self.leg, 75, remark="indbot/test/CE/NIFTY")
        self.assertEqual("SUCCESS", bought["status"])
        self.assertEqual(75, bought["filled_qty"])
        self.assertTrue(bought["order_id"].startswith("PAPER-"))
        # BUY fills through the ask (167.50) by one tick: 167.55.
        self.assertEqual(167.55, bought["price"])

        sold = self.broker.sell(self.leg, 75, remark="indbot/x-test/CE/NIFTY")
        # SELL fills through the bid (166.05) by one tick: 166.00.
        self.assertEqual(166.0, sold["price"])
        self.assertEqual("paper", sold["mode"])

        self.assertNoOrders(self.client)

    def test_the_client_behind_the_paper_broker_is_quote_only(self) -> None:
        """The guard is in the object graph, not in reviewer discipline."""
        wrapped = self.broker._client  # noqa: SLF001 - that it is wrapped IS the test
        self.assertIsInstance(wrapped, QuoteOnlyClient)
        with self.assertRaises(OrderError):
            wrapped.place_order(txn_type="BUY")
        self.assertNoOrders(self.client)

    def test_paper_refuses_a_qty_that_is_not_a_whole_lot(self) -> None:
        """Paper validates exactly as live does, so paper is a real rehearsal."""
        with self.assertRaises(OrderError) as caught:
            self.broker.buy(self.leg, 40, remark="indbot/test/CE/NIFTY")
        self.assertIn("MultipleOfLotSize", str(caught.exception))
        self.assertNoOrders(self.client)

    def test_paper_refuses_more_lots_than_max_lots(self) -> None:
        """`max_lots` is 1, so two lots is refused on the entry side."""
        with self.assertRaises(OrderError):
            self.broker.buy(self.leg, 150, remark="indbot/test/CE/NIFTY")
        self.assertNoOrders(self.client)


# ----------------------------------------------------------------------------------
# 4. the live arm gate
# ----------------------------------------------------------------------------------
class TestLiveBrokerArmGate(IndbotTestCase):
    """LiveBroker refuses every order unless mode is live AND armed is True."""

    def setUp(self) -> None:
        super().setUp()
        self.client = fx.FakeClient()
        self.leg = fx.leg_from_chain(fx.ATM_STRIKE, "CE")

    def _broker(self, **overrides: Any) -> LiveBroker:
        cfg = fx.make_config(self.tmp, **overrides)
        return LiveBroker(cfg, self.client)

    def test_buy_raises_order_error_when_live_but_not_armed(self) -> None:
        """mode == "live", armed == False: the required case from CONTRACTS.md J."""
        broker = self._broker(mode="live")
        self.assertEqual("live", broker.cfg.mode)
        self.assertIs(False, broker.cfg.armed)

        with self.assertRaises(OrderError) as caught:
            broker.buy(self.leg, 75, remark="indbot/orb/CE/NIFTY")

        message = str(caught.exception)
        self.assertIn("armed", message)
        self.assertIn("mode", message)
        self.assertNoOrders(self.client, "the gate must fire before anything is sent")

    def test_sell_also_raises_when_not_armed(self) -> None:
        """The gate is checked per call and on both sides, not just on entry."""
        broker = self._broker(mode="live")
        with self.assertRaises(OrderError):
            broker.sell(self.leg, 75, remark="indbot/x-stop/CE/NIFTY")
        self.assertNoOrders(self.client)

    def test_disarming_mid_flight_stops_the_next_order(self) -> None:
        """Armed, then disarmed: the second order is refused.

        The arm check runs on every call precisely so the UI can disarm a running
        bot and have it mean something.
        """
        broker = self._broker(mode="live")
        broker.cfg.set_armed(True, confirm=ARM_PHRASE)
        first = broker.buy(self.leg, 75, remark="indbot/orb/CE/NIFTY", signal_ts=1)
        self.assertEqual(75, first["filled_qty"])

        broker.cfg.set_armed(False)
        with self.assertRaises(OrderError):
            broker.buy(self.leg, 75, remark="indbot/orb/CE/NIFTY", signal_ts=2)
        self.assertEqual(1, len(self.client.method_calls("place_order")),
                         "only the armed order should have been sent")

    def test_armed_but_paper_mode_still_refuses(self) -> None:
        """`armed` alone is not enough — the mode has to say live too."""
        cfg = fx.make_config(self.tmp, mode="paper")
        cfg.set_armed(True, confirm=ARM_PHRASE)
        broker = LiveBroker(cfg, self.client)
        with self.assertRaises(OrderError):
            broker.buy(self.leg, 75, remark="indbot/orb/CE/NIFTY")
        self.assertNoOrders(self.client)

    def test_engine_arm_requires_the_exact_phrase(self) -> None:
        """Nothing arms on a click: the confirmation string is the gate."""
        cfg = fx.make_config(self.tmp, mode="live")
        engine = fx.make_engine(cfg, fx.make_journal(self.tmp), self.client, self.tmp)
        with self.assertRaises(ConfigError):
            engine.arm("arm live trading")
        self.assertIs(False, cfg.armed)
        engine.arm(ARM_PHRASE)
        self.assertIs(True, cfg.armed)
        engine.disarm()
        self.assertIs(False, cfg.armed)

    def test_config_update_cannot_arm(self) -> None:
        """A settings POST must never be able to go live by itself."""
        cfg = fx.make_config(self.tmp, mode="live")
        with self.assertRaises(ConfigError):
            cfg.update({"armed": True}, persist=False)
        self.assertIs(False, cfg.armed)


# ----------------------------------------------------------------------------------
# 5. leaving live mode
# ----------------------------------------------------------------------------------
class TestSetModeFlattensLivePosition(IndbotTestCase):
    """`set_mode` out of live squares the position off before swapping the broker.

    Swapping a LiveBroker out from under an open live position would "close" it on
    paper while the real contract stayed open on the exchange — the single most
    expensive bug available in this file.
    """

    def setUp(self) -> None:
        super().setUp()
        self.cfg = fx.make_config(
            self.tmp, mode="live", max_trade_loss=LOOSE_TRADE_LOSS,
            max_daily_loss=LOOSE_DAILY_LOSS,
        )
        self.cfg.set_armed(True, confirm=ARM_PHRASE)
        self.journal = fx.make_journal(self.tmp)
        self.client = fx.FakeClient()
        self.engine = fx.make_engine(self.cfg, self.journal, self.client, self.tmp)
        self.assertIsInstance(self.engine.broker, LiveBroker)
        self.engine.tick()
        open_now = self.engine._open_positions()  # noqa: SLF001
        self.assertEqual(1, len(open_now), f"expected a live position, got {open_now}")
        self.position = open_now[0]
        self.assertEqual("live", self.position.mode)

    def test_live_entry_sends_the_order_body_contracts_specifies(self) -> None:
        """The POST /order body, field by field, including qty in UNITS."""
        sent = self.client.method_calls("place_order")
        self.assertEqual(1, len(sent))
        body = sent[0]
        self.assertEqual("BUY", body["txn_type"])
        self.assertEqual("NSE", body["exchange"])
        self.assertEqual("DERIVATIVE", body["segment"])
        self.assertEqual("INTRADAY", body["product"])
        self.assertEqual("LIMIT", body["order_type"])
        self.assertEqual("DAY", body["validity"])
        self.assertEqual("45018", body["security_id"])
        self.assertEqual(75, body["qty"])
        self.assertIs(False, body["is_amo"])
        self.assertEqual(0, round(body["limit_price"] / 0.05) * 0.05 - body["limit_price"],
                         "the limit price must sit on the tick")
        self.assertLessEqual(len(body["remarks"]), 100)
        self.assertNotEqual("TV-TERMINAL", body["remarks"])
        self.assertEqual({"txn_type", "exchange", "segment", "product", "order_type",
                          "validity", "security_id", "qty", "limit_price", "is_amo",
                          "remarks"}, set(body))

    def test_leaving_live_flattens_first_then_swaps_the_broker(self) -> None:
        """A SELL goes out before the mode changes, and the broker changes after."""
        fx.set_now("2026-10-26 09:40:00")
        self.engine.set_mode("paper")

        order_types = [kw["txn_type"] for kw in self.client.method_calls("place_order")]
        self.assertEqual(["BUY", "SELL"], order_types,
                         "the exit must be sent while the live broker is still in place")
        self.assertEqual([], self.engine._open_positions())  # noqa: SLF001
        self.assertEqual("paper", self.cfg.mode)
        self.assertIsInstance(self.engine.broker, PaperBroker)

    def test_leaving_live_disarms(self) -> None:
        """Any mode change clears `armed`; re-arming is a deliberate act."""
        fx.set_now("2026-10-26 09:40:00")
        self.engine.set_mode("paper")
        self.assertIs(False, self.cfg.armed)

    def test_the_flattened_trade_is_journalled_as_closed(self) -> None:
        """The close is recorded under the mode it was traded in, with its reason."""
        fx.set_now("2026-10-26 09:40:00")
        self.engine.set_mode("paper")

        trades = self.journal.trades(day=fx.TRADING_DAY, mode="live")
        self.assertEqual(1, len(trades))
        closed = trades[0]
        self.assertEqual("closed", closed["status"])
        self.assertEqual("mode_change", closed["exit_reason"])
        self.assertEqual("live", closed["mode"])
        self.assertIsNotNone(closed["exit_order_id"])
        self.assertEqual(75, closed["qty"])
        self.assertEqual([], self.journal.open_trades())

    def test_switching_into_live_does_not_flatten_a_paper_position(self) -> None:
        """Only LEAVING live flattens. Entering it must not sell anything.

        Checked because the obvious implementation of "flatten on mode change" would
        also close a perfectly good paper position on the way in.
        """
        fx.set_now("2026-10-26 09:40:00")
        self.engine.set_mode("paper")
        before = len(self.client.method_calls("place_order"))
        self.engine.set_mode("live")
        self.assertEqual(before, len(self.client.method_calls("place_order")))
        self.assertEqual("live", self.cfg.mode)


# ----------------------------------------------------------------------------------
# 6. the day halt
# ----------------------------------------------------------------------------------
class TestDayHaltPersistence(IndbotTestCase):
    """The daily-loss halt is sticky: a new gate, and a new DB connection, still see it.

    A bot that restarts after hitting its daily loss and resumes trading is how a
    Rs.2,000 limit becomes a Rs.6,000 loss, so this is checked across both a new
    `RiskGate` and a reopened `Journal` rather than only in memory.
    """

    def setUp(self) -> None:
        super().setUp()
        self.db = self.tmp / "halt.db"
        self.cfg = fx.make_config(self.tmp)
        self.journal = Journal(self.db)

    def test_halt_survives_recreating_the_gate(self) -> None:
        """A fresh RiskGate over the same journal reads the persisted flag."""
        gate = RiskGate(self.journal, self.cfg)
        self.assertEqual((False, ""), gate.day_halted("paper"))

        gate.halt_day("paper", "Day halted: realised P&L is -Rs.2,000.")
        self.assertTrue(gate.day_halted("paper")[0])

        reborn = RiskGate(self.journal, self.cfg)
        halted, reason = reborn.day_halted("paper")
        self.assertTrue(halted, "a new gate must not forget today's halt")
        self.assertIn("realised P&L", reason, "the reason must survive too")

    def test_halt_survives_reopening_the_database(self) -> None:
        """The flag is on disk, not in the process — the restart case."""
        RiskGate(self.journal, self.cfg).halt_day("paper", "hit the daily loss limit")
        self.journal.close()

        reopened = Journal(self.db)
        self.addCleanup(reopened.close)
        halted, reason = RiskGate(reopened, self.cfg).day_halted("paper")
        self.assertTrue(halted)
        self.assertIn("daily loss limit", reason)

    def test_the_halt_is_scoped_to_one_mode_and_one_day(self) -> None:
        """Halting paper must not halt live, and tomorrow starts clean."""
        gate = RiskGate(self.journal, self.cfg)
        gate.halt_day("paper", "paper is done for the day")
        self.assertTrue(gate.day_halted("paper")[0])
        self.assertFalse(gate.day_halted("live")[0])
        self.assertFalse(gate.day_halted("paper", day=date(2026, 10, 27))[0])

    def test_a_halted_day_stops_the_engine_before_it_scans(self) -> None:
        """The engine honours the persisted halt on its very first tick."""
        client = fx.FakeClient()
        engine = fx.make_engine(self.cfg, self.journal, client, self.tmp)
        engine.risk.halt_day("paper", "Day halted: realised P&L is -Rs.2,000.")

        result = engine.tick()
        self.assertTrue(result["halted"])
        self.assertEqual("halted", result["phase"])
        self.assertEqual(0, result["signals"], "a halted day must not even scan")
        self.assertEqual([], self.journal.trades())
        self.assertNoOrders(client)

    def test_clearing_the_halt_is_explicit(self) -> None:
        """A halt only lifts when a human lifts it."""
        gate = RiskGate(self.journal, self.cfg)
        gate.halt_day("paper", "hit the limit")
        self.assertTrue(gate.clear_halt("paper"))
        self.assertFalse(gate.day_halted("paper")[0])


# ----------------------------------------------------------------------------------
# 7. module health
# ----------------------------------------------------------------------------------
class TestModuleHealth(unittest.TestCase):
    """Every module imports, and the config loads with nothing on disk."""

    def test_every_module_imports_cleanly(self) -> None:
        """No import-time side effect, no circular import, no missing dependency."""
        failures: list[str] = []
        for name in MODULES:
            try:
                importlib.import_module(f"indbot.{name}")
            except Exception as exc:  # noqa: BLE001 - every failure is reported together
                failures.append(f"indbot.{name}: {type(exc).__name__}: {exc}")
        self.assertEqual([], failures, "modules failed to import:\n" + "\n".join(failures))

    def test_config_load_works_with_no_config_json_present(self) -> None:
        """`Config.load()` with no file is pure defaults, and boots DISARMED."""
        with tempfile.TemporaryDirectory(prefix="indbot-cfg-") as raw:
            missing = Path(raw) / "config.json"
            self.assertFalse(missing.exists())

            cfg = Config.load(missing)
            self.assertEqual(DEFAULTS["mode"], cfg.mode)
            self.assertIs(False, cfg.armed)
            self.assertEqual(DEFAULTS["max_trade_loss"], cfg.max_trade_loss)
            self.assertEqual(DEFAULTS["stop_pct"], cfg.stop_pct)
            self.assertEqual(DEFAULTS["underlyings"], cfg.underlyings)
            self.assertFalse(missing.exists(), "load() must not create the file")

            # Every DEFAULTS key round-trips through as_dict()/update().
            self.assertEqual(set(DEFAULTS), set(cfg.as_dict()))
            self.assertEqual([], cfg.update(cfg.as_dict(), persist=False))

    def test_config_load_with_no_arguments_uses_the_default_path(self) -> None:
        """The no-argument form, with INDBOT_CONFIG pointing at a file that is absent.

        Pinned through the env var rather than relying on the repo not containing a
        config.json, so the test says the same thing on every machine.
        """
        import os

        with tempfile.TemporaryDirectory(prefix="indbot-cfg-") as raw:
            missing = Path(raw) / "nothing-here.json"
            previous = os.environ.get("INDBOT_CONFIG")
            os.environ["INDBOT_CONFIG"] = str(missing)
            try:
                cfg = Config.load()
            finally:
                if previous is None:
                    os.environ.pop("INDBOT_CONFIG", None)
                else:
                    os.environ["INDBOT_CONFIG"] = previous
        self.assertEqual(DEFAULTS["mode"], cfg.mode)
        self.assertIs(False, cfg.armed)

    def test_config_load_boots_disarmed_even_if_the_file_says_armed(self) -> None:
        """A crashed-and-restarted bot must be re-armed by a human who is watching."""
        with tempfile.TemporaryDirectory(prefix="indbot-cfg-") as raw:
            path = Path(raw) / "config.json"
            path.write_text(json.dumps({"mode": "live", "armed": True}), encoding="utf-8")
            cfg = Config.load(path)
        self.assertEqual("live", cfg.mode)
        self.assertIs(False, cfg.armed)

    def test_effective_max_premium_is_per_unit_and_guards_its_division(self) -> None:
        """Rs.1,000 / (30% x 75) = Rs.44.44 a UNIT — the number the whole bot turns on."""
        with tempfile.TemporaryDirectory(prefix="indbot-cfg-") as raw:
            cfg = Config.load(Path(raw) / "config.json")
        self.assertAlmostEqual(44.44, cfg.effective_max_premium(75), places=2)
        self.assertAlmostEqual(3333.33, cfg.effective_max_lot_cost(75), places=2)
        with self.assertRaises(ConfigError):
            cfg.effective_max_premium(0)     # never guess a lot size
        with self.assertRaises(ConfigError):
            cfg.effective_max_premium(-75)


# ----------------------------------------------------------------------------------
# 8. the fixtures themselves
# ----------------------------------------------------------------------------------
class TestFixtureShapes(unittest.TestCase):
    """Guards the fixtures against drifting from the documented response shapes.

    If these fail, the tests above are still green but are no longer testing the API
    this bot talks to — which is worse than a red suite.
    """

    def test_option_chain_strikes_is_an_object_keyed_by_strike(self) -> None:
        """Not an array, and not in numeric key order."""
        data = fx.chain_data()
        self.assertIsInstance(data["strikes"], dict)
        self.assertIn("24450", data["strikes"])
        keys = list(data["strikes"])
        self.assertNotEqual(sorted(keys, key=float), keys,
                            "the fixture must NOT be pre-sorted, or sorting is untested")

    def test_iv_is_a_percentage_and_a_sibling_of_greeks(self) -> None:
        """`iv` sits next to `greeks`, and `greeks` has no rho."""
        node = fx.chain_data()["strikes"]["24450"]["ce"]
        self.assertIn("iv", node)
        self.assertNotIn("iv", node["greeks"])
        self.assertEqual(10.5, node["iv"], "a percentage, e.g. 10.5 means 10.5%")
        self.assertEqual({"delta", "gamma", "theta", "vega"}, set(node["greeks"]))

    def test_put_delta_is_negative(self) -> None:
        """The docs' own example is -0.44, which is why ranking uses |  |delta| - 0.5 |."""
        self.assertLess(fx.chain_data()["strikes"]["24450"]["pe"]["greeks"]["delta"], 0)

    def test_historical_wraps_in_success_not_status(self) -> None:
        """A `success` BOOLEAN, `data` keyed by code, each with its own `candles`."""
        raw = fx.historical_response({fx.NIFTY_CODE: fx.orb_breakout_candles()})
        self.assertIn("success", raw)
        self.assertNotIn("status", raw)
        self.assertIs(True, raw["success"])
        self.assertIn("candles", raw["data"][fx.NIFTY_CODE])
        candle = raw["data"][fx.NIFTY_CODE]["candles"][0]
        self.assertEqual({"ts", "o", "h", "l", "c", "v"}, set(candle))

    def test_candle_ts_is_epoch_seconds_and_request_bounds_are_milliseconds(self) -> None:
        """The unit boundary: `ts` in seconds, `start_time`/`end_time` in milliseconds."""
        client = fx.FakeClient()
        start = clock.epoch_ms(clock.session_open(fx.TRADING_DAY))
        end = clock.epoch_ms(clock.at(fx.TRADING_DAY, clock.parse_hhmm("09:35")))
        candles = client.historical("5minute", [fx.NIFTY_CODE], start, end)[fx.NIFTY_CODE]

        self.assertTrue(candles)
        self.assertEqual(fx.at(fx.TRADING_DAY, "09:15"), candles[0]["ts"])
        self.assertLess(candles[-1]["ts"], end, "a seconds ts is ~1000x below a ms bound")
        self.assertEqual(
            [c["ts"] for c in candles], sorted(c["ts"] for c in candles),
            "candles must be strictly ascending or every indicator is corrupt",
        )
        recorded = client.method_calls("historical")[0]
        self.assertEqual(start, recorded["start_ms"])

    def test_the_index_master_is_three_columns_read_positionally(self) -> None:
        """Column 2 is labelled SEGMENT but holds the index NAME."""
        header, first = fx.INDEX_CSV.splitlines()[0], fx.INDEX_CSV.splitlines()[1]
        self.assertEqual("EXCH,SEGMENT,SECURITY_ID", header)
        self.assertEqual(["NSE", "NIFTY 50", "40000001"], first.split(","))

    def test_the_resolved_index_scrip_beats_the_hardcoded_constant(self) -> None:
        """BANKNIFTY's constant disagrees with the fixture file; the file must win.

        Every scrip id in config except NIFTY's is a guess, so this is the path that
        turns a wrong constant into a logged correction instead of candles for the
        wrong instrument.
        """
        from indbot.config import UNDERLYINGS

        with tempfile.TemporaryDirectory(prefix="indbot-instr-") as raw, \
                fx.fake_now(fx.ENTRY_AT):
            tmp = Path(raw)
            instruments = fx.make_instruments(fx.FakeClient(), tmp)
            self.assertEqual("40000001", instruments.index_scrip("NIFTY 50"))
            self.assertEqual("40000005", instruments.index_scrip("NIFTY BANK"))
            self.assertNotEqual(UNDERLYINGS["BANKNIFTY"]["scrip"],
                                instruments.index_scrip("NIFTY BANK"))
            self.assertEqual(75, instruments.lot_size("NIFTY", fx.EXPIRY))
            self.assertEqual(0.05, instruments.tick_size("45018"))


class TestSuiteIsFast(unittest.TestCase):
    """The suite must stay runnable in a pre-commit hook: offline and quick."""

    def test_a_full_engine_tick_costs_no_network_and_little_time(self) -> None:
        """One tick, end to end, well inside the budget for the whole suite."""
        with tempfile.TemporaryDirectory(prefix="indbot-perf-") as raw, \
                fx.fake_now(fx.ENTRY_AT):
            tmp = Path(raw)
            client = fx.FakeClient()
            engine = fx.make_engine(
                fx.make_config(tmp), fx.make_journal(tmp), client, tmp
            )
            started = time.monotonic()
            engine.tick()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0, f"one tick took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
