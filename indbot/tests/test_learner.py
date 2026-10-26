"""Tests for indbot.learner — the outcome gate that stops a losing setup.

What these tests protect:

* The Wilson lower bound is checked against PUBLISHED values, not against a second
  copy of the same algebra. `_score_test_lower_bound` re-derives the bound by
  bisecting the score test it is the closed-form solution of, so a typo in the
  closed form cannot agree with it by construction.
* The four states are probed at their EXACT float boundaries with `math.nextafter`,
  because `0.375 + 0.10` is not the same double as `0.475` on every value and a gate
  written with `>=` where the contract says `>` would silently promote a bucket that
  has only matched break-even.
* `blocked` is a strict subset of `demoted`, so the evaluation ORDER is load-bearing.
  CONTRACTS.md lists `demoted` first; a literal reading of that order never reaches
  `blocked` and the bot keeps trading a setup it has proven loses money.
* Break-even comes from the trades' own stop/target. Re-judging closed trades with
  today's config would let a `stop_pct` tweak unblock a setup that lost real rupees
  under different rules.
* Paper and live statistics stay apart. A paper losing streak gating a live setup
  (or worse, a lucky paper run promoting one) is the failure `learn_across_modes`
  exists to prevent.
"""

from __future__ import annotations

import math
import sys
from datetime import date, time
from pathlib import Path
from typing import Iterator

import pytest

# The package root, so this file runs under a bare `pytest` from any directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indbot import clock  # noqa: E402
from indbot import learner as learner_mod  # noqa: E402
from indbot.config import Config  # noqa: E402
from indbot.journal import Journal  # noqa: E402
from indbot.learner import (  # noqa: E402
    BLOCK_MARGIN,
    PROMOTE_MARGIN,
    STATE_MULTIPLIERS,
    BucketVerdict,
    Learner,
    break_even_win_rate,
    classify,
    wilson_lower_bound,
    z_for_confidence,
)
from indbot.types import OptionLeg, Position, Signal  # noqa: E402

#: Frozen clock. The learner stamps every refresh with clock.now(), and a journal
#: row's entry day decides which calendar day it belongs to, so the whole module
#: runs at one fixed IST instant rather than at the machine's wall clock.
FAKE_NOW = "2026-10-26 10:30:00"
TRADE_DAY = date(2026, 10, 20)  # a Tuesday, not an exchange holiday

#: The bucket from the CONTRACTS.md §H worked example.
BUCKET = "orb:CE:open:trend_up"


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin indbot.clock to one IST instant for every test in this module."""
    monkeypatch.setenv("INDBOT_FAKE_NOW", FAKE_NOW)


@pytest.fixture()
def journal(tmp_path: Path) -> Iterator[Journal]:
    """A real SQLite journal in a temp dir — never the user's data/indbot.db."""
    j = Journal(tmp_path / "learner.db")
    try:
        yield j
    finally:
        j.close()


@pytest.fixture()
def cfg() -> Config:
    """Stock config: 10 samples to gate, 90% confidence, 30% stop / 50% target."""
    return Config()


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _leg(premium: float, lot_size: int = 25) -> OptionLeg:
    """A priced option leg. The lot size here only scales the P&L arithmetic."""
    return OptionLeg(
        underlying="NIFTY",
        expiry="2026-10-29",
        strike=25000.0,
        side="CE",
        security_id="54321",
        trading_symbol="NIFTY29OCT25000CE",
        lot_size=lot_size,
        exchange="NSE",
        ltp=premium,
        bid=premium - 0.5,
        ask=premium + 0.5,
        oi=120000,
        volume=90000,
        iv=12.0,
        delta=0.50,
    )


def _close_trade(
    journal: Journal,
    *,
    seq: int,
    bucket: str,
    win: bool,
    mode: str = "paper",
    stop_pct: float = 0.30,
    target_pct: float = 0.50,
    entry: float = 100.0,
    qty: int = 25,
    win_move: float = 16.0,
    loss_move: float = 16.64,
) -> Position:
    """Write one CLOSED trade into the journal and return the position.

    The stop and target PRICES are written from `stop_pct`/`target_pct`, which is how
    the learner recovers the risk plan a trade was actually taken with. The default
    moves make a win +₹400 and a loss −₹416 on a 25-unit lot, so 6 wins in 24 trades
    comes to exactly −₹212 a trade — the CONTRACTS.md example.
    """
    entry_ts = clock.epoch_s(clock.at(TRADE_DAY, time(10, 0))) + seq * 60
    pos = Position(
        id=f"{mode}:{bucket}:{seq}",
        leg=_leg(entry, qty),
        signal_setup=bucket.split(":", 1)[0],
        bucket=bucket,
        qty=qty,
        entry_price=entry,
        entry_ts=entry_ts,
        entry_order_id=f"order-{seq}",
        stop_price=entry * (1.0 - stop_pct),
        target_price=entry * (1.0 + target_pct),
        mode=mode,
    )
    journal.open_trade(pos, mode)
    pos.exit_price = entry + (win_move if win else -loss_move)
    pos.exit_ts = entry_ts + 600
    pos.exit_reason = "target" if win else "stop"
    journal.close_trade(pos)
    return pos


def _fill_bucket(
    journal: Journal,
    bucket: str,
    *,
    wins: int,
    losses: int,
    mode: str = "paper",
    stop_pct: float = 0.30,
    target_pct: float = 0.50,
    seq0: int = 0,
    **kw: float,
) -> None:
    """Write `wins` winners and `losses` losers into one bucket."""
    for i in range(wins + losses):
        _close_trade(
            journal,
            seq=seq0 + i,
            bucket=bucket,
            win=i < wins,
            mode=mode,
            stop_pct=stop_pct,
            target_pct=target_pct,
            **kw,  # type: ignore[arg-type]
        )


def _signal(bucket: str = BUCKET, *, score: float = 0.80) -> Signal:
    """A plain BUY signal carrying `bucket`, for adjust()."""
    return Signal(
        setup=bucket.split(":", 1)[0],
        side="CE",
        action="BUY",
        underlying="NIFTY",
        ts=clock.epoch_s(clock.now()),
        spot=25000.0,
        score=score,
        reasons=["opening range high broken (+0.35)"],
        bucket=bucket,
        stop_pct=0.30,
        target_pct=0.50,
    )


def _score_test_lower_bound(wins: int, n: int, z: float) -> float:
    """Re-derive the Wilson lower bound by bisecting the score test, independently.

    The Wilson bound is the smaller root of (p̂ − p)² = z²·p(1−p)/n. Solving it by
    bisection on the inequality instead of by the quadratic formula gives a value
    that shares no algebra with the implementation, so the two agreeing to 1e-12 is
    real evidence rather than two copies of the same mistake.
    """
    p_hat = wins / n
    lo, hi = 0.0, p_hat
    for _ in range(200):
        mid = (lo + hi) / 2.0
        se = math.sqrt(mid * (1.0 - mid) / n)
        if se == 0.0:
            lo = mid
            continue
        if (p_hat - mid) / se > z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ----------------------------------------------------------------------------------
# 1. the Wilson lower bound, against published values
# ----------------------------------------------------------------------------------
class TestWilsonLowerBound:
    """The statistic the whole gate rests on."""

    def test_six_in_twentyfour_at_90_and_95(self) -> None:
        """6/24 at 90% and 95% one-sided match the published Wilson values."""
        # p̂ = 25%, n = 24. One-sided, as the gate's question ("the win rate is at
        # least X") demands: z(0.90) = 1.2816, z(0.95) = 1.6449.
        assert wilson_lower_bound(6, 24, 0.90) == pytest.approx(0.155262025131, abs=1e-12)
        assert wilson_lower_bound(6, 24, 0.95) == pytest.approx(0.135194952502, abs=1e-12)

    def test_six_in_twentyfour_two_sided_95(self) -> None:
        """6/24 at 95% TWO-sided is the textbook 0.1200 (z = 1.96)."""
        assert wilson_lower_bound(6, 24, 0.95, two_sided=True) == pytest.approx(
            0.119993662288, abs=1e-12
        )

    def test_sixty_in_a_hundred_matches_the_textbook_interval(self) -> None:
        """60/100 at 95% two-sided is the standard worked example, (0.5020, 0.6906)."""
        assert wilson_lower_bound(60, 100, 0.95, two_sided=True) == pytest.approx(
            0.502002586793, abs=1e-12
        )

    @pytest.mark.parametrize(
        ("wins", "n", "confidence"),
        [(6, 24, 0.90), (6, 24, 0.95), (1, 11, 0.90), (17, 20, 0.95), (60, 100, 0.99)],
    )
    def test_agrees_with_an_independent_derivation(
        self, wins: int, n: int, confidence: float
    ) -> None:
        """The closed form equals a bisection of the score test it solves."""
        z = z_for_confidence(confidence)
        assert wilson_lower_bound(wins, n, confidence) == pytest.approx(
            _score_test_lower_bound(wins, n, z), abs=1e-10
        )

    def test_zero_wins_bounds_at_zero_and_never_goes_negative(self) -> None:
        """0 wins bounds at 0.0 — the normal approximation would return a negative rate.

        The analytic value is exactly zero; the implementation lands within one
        rounding step of it, which is why this is a tolerance and not an identity.
        """
        for n, confidence in ((10, 0.95), (1, 0.99), (24, 0.90), (3, 0.80)):
            value = wilson_lower_bound(0, n, confidence)
            assert value == pytest.approx(0.0, abs=1e-12)
            assert value >= 0.0  # a negative "win rate" would read as a promotion

    def test_a_perfect_record_does_not_claim_certainty(self) -> None:
        """10/10 at 90% is 85.9%, not 100% — the reason Wilson is used at all."""
        assert wilson_lower_bound(10, 10, 0.90) == pytest.approx(0.858931317918, abs=1e-12)
        assert wilson_lower_bound(10, 10, 0.90) < 1.0

    def test_no_evidence_is_not_evidence(self) -> None:
        """An empty bucket is 0.0, and more wins than trades is clamped, not trusted."""
        assert wilson_lower_bound(0, 0, 0.90) == 0.0
        assert wilson_lower_bound(-5, 0, 0.90) == 0.0
        assert wilson_lower_bound(30, 24, 0.90) == wilson_lower_bound(24, 24, 0.90)

    def test_more_confidence_means_a_lower_bound(self) -> None:
        """Raising the confidence can only make the gate stricter."""
        bounds = [wilson_lower_bound(12, 20, c) for c in (0.80, 0.90, 0.95, 0.99)]
        assert bounds == sorted(bounds, reverse=True)

    def test_stays_inside_the_unit_interval(self) -> None:
        """Every bound is a probability, for every n and every win count."""
        for n in range(1, 40):
            for wins in range(0, n + 1):
                value = wilson_lower_bound(wins, n, 0.90)
                assert 0.0 <= value <= 1.0


class TestBreakEven:
    """stop / (stop + target), and the safe answer when that is undefined."""

    def test_thirty_fifty_needs_thirty_seven_point_five(self) -> None:
        """A 30% stop against a 50% target needs 37.5% of trades to win."""
        assert break_even_win_rate(0.30, 0.50) == pytest.approx(0.375, abs=1e-12)

    @pytest.mark.parametrize(
        ("stop", "target", "expected"),
        [
            (0.20, 0.80, 0.20),
            (0.30, 0.30, 0.50),
            (0.50, 0.25, 2.0 / 3.0),
            (0.12, 0.60, 0.166666666667),
        ],
    )
    def test_ratios(self, stop: float, target: float, expected: float) -> None:
        """The break-even is the stop's share of the stop-plus-target distance."""
        assert break_even_win_rate(stop, target) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize(("stop", "target"), [(0.0, 0.0), (-0.3, 0.5), (0.3, -0.5)])
    def test_a_degenerate_plan_falls_back_to_neutral_not_a_zero_divide(
        self, stop: float, target: float
    ) -> None:
        """An unusable risk plan returns 0.5, which neither promotes nor blocks."""
        assert break_even_win_rate(stop, target) == 0.5


# ----------------------------------------------------------------------------------
# 2. the four states, at their exact boundaries
# ----------------------------------------------------------------------------------
class TestClassifyBoundaries:
    """`classify` is probed one ULP either side of every threshold.

    The thresholds are recomputed here from the module's own constants rather than
    written as literals: `0.375 + 0.10` and `0.475` are not guaranteed to be the same
    double, and a test that hard-codes the literal can pass while the gate is wrong.
    """

    BE = 0.375
    MIN_SAMPLES = 10

    def _state(self, n: int, expectancy: float, lower: float) -> str:
        state, multiplier = classify(
            n, expectancy, lower, self.BE, min_samples=self.MIN_SAMPLES
        )
        assert multiplier == STATE_MULTIPLIERS[state]  # the pair must stay consistent
        return state

    # -- promote boundary: lower_bound > be + 0.10 (STRICTLY greater) --------------
    def test_promote_needs_to_clear_the_margin_strictly(self) -> None:
        """Exactly at be + 0.10 is NOT promoted; one ULP above it is."""
        threshold = self.BE + PROMOTE_MARGIN
        assert self._state(20, 150.0, threshold) == "neutral"
        assert self._state(20, 150.0, math.nextafter(threshold, math.inf)) == "promoted"
        assert self._state(20, 150.0, math.nextafter(threshold, -math.inf)) == "neutral"

    def test_promotion_does_not_need_a_positive_expectancy(self) -> None:
        """Per the contract, promotion is decided by the bound alone."""
        assert self._state(20, -1.0, self.BE + PROMOTE_MARGIN + 0.05) == "promoted"

    # -- demote boundary: lower_bound < be AND expectancy < 0 ----------------------
    def test_demote_needs_the_bound_under_break_even_strictly(self) -> None:
        """Exactly at break-even is neutral; one ULP below it demotes."""
        assert self._state(12, -50.0, self.BE) == "neutral"
        assert self._state(12, -50.0, math.nextafter(self.BE, -math.inf)) == "demoted"

    def test_demote_needs_a_losing_expectancy_strictly(self) -> None:
        """A scratch bucket (expectancy exactly 0) is not demoted."""
        assert self._state(12, 0.0, self.BE - 0.20) == "neutral"
        assert self._state(12, math.nextafter(0.0, -math.inf), self.BE - 0.20) == "demoted"
        assert self._state(12, 0.01, self.BE - 0.20) == "neutral"

    # -- block boundary: n >= 2*min_samples AND expectancy < 0 AND lb < be - 0.10 ---
    def test_block_needs_twice_the_sample_size(self) -> None:
        """At 2 x min_samples it blocks; one trade short it only demotes."""
        lower = self.BE - BLOCK_MARGIN - 0.05
        assert self._state(2 * self.MIN_SAMPLES - 1, -200.0, lower) == "demoted"
        assert self._state(2 * self.MIN_SAMPLES, -200.0, lower) == "blocked"

    def test_block_needs_to_clear_the_margin_strictly(self) -> None:
        """Exactly at be − 0.10 is demoted, not blocked; one ULP below blocks."""
        threshold = self.BE - BLOCK_MARGIN
        assert self._state(24, -200.0, threshold) == "demoted"
        assert self._state(24, -200.0, math.nextafter(threshold, -math.inf)) == "blocked"

    def test_block_needs_a_losing_expectancy(self) -> None:
        """A bucket with a bad win rate but positive expectancy is not blocked.

        That is the big-winners/small-losers profile, and blocking it would cost the
        user a profitable setup on a statistic that does not measure profit.
        """
        assert self._state(24, 1.0, 0.0) == "neutral"
        assert self._state(24, 0.0, 0.0) == "neutral"

    # -- the ordering that CONTRACTS.md gets backwards -----------------------------
    def test_blocked_wins_over_demoted_when_both_apply(self) -> None:
        """`blocked` is a strict subset of `demoted`, so it must be tested first.

        Every blocked bucket also satisfies the demote rule. Evaluating the rules in
        the order CONTRACTS.md lists them (demoted, then blocked) would return
        `demoted` here and leave a proven money-loser trading at 0.75x.
        """
        state, multiplier = classify(
            40, -500.0, 0.05, self.BE, min_samples=self.MIN_SAMPLES
        )
        assert (state, multiplier) == ("blocked", 0.0)

    # -- sample-size gate ----------------------------------------------------------
    def test_below_min_samples_is_always_neutral(self) -> None:
        """n < min_samples is neutral no matter how bad the numbers look."""
        for n in range(0, self.MIN_SAMPLES):
            assert self._state(n, -5000.0, 0.0) == "neutral"

    def test_at_min_samples_the_gate_starts_judging(self) -> None:
        """At exactly min_samples the same numbers demote."""
        assert self._state(self.MIN_SAMPLES, -5000.0, 0.0) == "demoted"

    def test_min_samples_is_never_below_one(self) -> None:
        """A zero or negative min_samples must not make an empty bucket judgeable."""
        assert classify(0, -100.0, 0.0, self.BE, min_samples=0)[0] == "neutral"
        assert classify(0, -100.0, 0.0, self.BE, min_samples=-5)[0] == "neutral"

    def test_multipliers_are_the_contract_values(self) -> None:
        """1.15 / 1.0 / 0.75 / 0.0, fixed by CONTRACTS.md §H."""
        assert STATE_MULTIPLIERS == {
            "promoted": 1.15,
            "neutral": 1.0,
            "demoted": 0.75,
            "blocked": 0.0,
        }


# ----------------------------------------------------------------------------------
# 3. the Learner over a real journal
# ----------------------------------------------------------------------------------
class TestLearnerOverAJournal:
    """End-to-end: closed trades in SQLite, verdicts out."""

    def test_a_setup_below_min_samples_stays_neutral_and_says_so(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Nine straight losses must not gate anything — and the note must explain.

        This is the case where an impatient gate would do real damage: nine losses is
        a plausible run for a setup with a 37.5% break-even, and blocking on it would
        switch off a setup that was never given a chance to prove itself.
        """
        _fill_bucket(journal, BUCKET, wins=0, losses=9)
        learner = Learner(journal, cfg)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert (v.n, v.wins, v.state, v.multiplier) == (9, 0, "neutral", 1.0)
        assert v.expectancy < 0  # the numbers are bad; the gate just will not act yet
        assert v.trades_needed == 1
        assert "Still learning" in v.note
        assert "9 paper trades" in v.note
        assert "1 more before this setup is judged" in v.note
        assert learner.is_blocked(BUCKET) is False

        # The signal's score is untouched while the bucket is still learning.
        sig = _signal()
        adjusted, verdict = learner.adjust(sig)
        assert adjusted.score == pytest.approx(0.80)
        assert adjusted.reasons == sig.reasons
        assert verdict.state == "neutral"

    def test_the_tenth_trade_makes_the_bucket_judgeable(
        self, journal: Journal, cfg: Config
    ) -> None:
        """At exactly min_samples_to_gate the same losing record demotes."""
        _fill_bucket(journal, BUCKET, wins=0, losses=9)
        learner = Learner(journal, cfg)
        learner.refresh()
        assert learner.verdict(BUCKET).state == "neutral"

        _close_trade(journal, seq=99, bucket=BUCKET, win=False)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert (v.n, v.state, v.multiplier) == (10, "demoted", 0.75)
        assert "Demoted" in v.note and "0 wins in 10 trades (0%)" in v.note

    def test_an_empty_bucket_is_neutral_without_a_refresh(
        self, journal: Journal, cfg: Config
    ) -> None:
        """verdict() on an unknown bucket is neutral, and never raises."""
        learner = Learner(journal, cfg)
        v = learner.verdict("vwap_reclaim:PE:midday:range")
        assert (v.n, v.state, v.multiplier) == (0, "neutral", 1.0)
        assert "No trades yet" in v.note
        assert "10 paper trades" in v.note

    def test_the_contract_worked_example_blocks(
        self, journal: Journal, cfg: Config
    ) -> None:
        """6 wins in 24 trades at a 37.5% break-even blocks, with the contract's note.

        CONTRACTS.md §H asks for exactly this sentence, so it is asserted verbatim:
        the note is the only explanation the user gets for a setup going quiet.
        """
        _fill_bucket(journal, BUCKET, wins=6, losses=18)
        learner = Learner(journal, cfg)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert (v.n, v.wins, v.losses) == (24, 6, 18)
        assert v.win_rate == pytest.approx(0.25)
        assert v.expectancy == pytest.approx(-212.0)
        assert v.lower_bound == pytest.approx(0.155262025131, abs=1e-9)
        assert v.break_even == pytest.approx(0.375, abs=1e-9)
        assert (v.state, v.multiplier) == ("blocked", 0.0)
        assert v.is_blocked is True
        assert v.note == (
            "Blocked: 6 wins in 24 trades (25%), break-even needs 37.5%, "
            "average −₹212 a trade."
        )
        assert learner.is_blocked(BUCKET) is True
        assert learner.blocked_buckets() == [BUCKET]

    def test_adjust_zeroes_a_blocked_signal_and_says_why(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A blocked bucket multiplies the score to 0 and appends the reason verbatim."""
        _fill_bucket(journal, BUCKET, wins=6, losses=18)
        learner = Learner(journal, cfg)
        learner.refresh()

        adjusted, verdict = learner.adjust(_signal(score=0.95))
        assert adjusted.score == 0.0
        assert verdict.state == "blocked"
        # The UI prints `reasons`; a score that fell with no explanation reads as a bug.
        assert len(adjusted.reasons) == 2
        assert "Learning blocked" in adjusted.reasons[-1]
        assert verdict.note in adjusted.reasons[-1]

    def test_a_promoted_bucket_raises_the_score_and_clamps_at_one(
        self, journal: Journal, cfg: Config
    ) -> None:
        """9 wins in 12 clears break-even + 10%, so the score is multiplied by 1.15."""
        _fill_bucket(journal, BUCKET, wins=9, losses=3)
        learner = Learner(journal, cfg)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert (v.state, v.multiplier) == ("promoted", 1.15)
        assert v.lower_bound > v.break_even + PROMOTE_MARGIN

        adjusted, _ = learner.adjust(_signal(score=0.60))
        assert adjusted.score == pytest.approx(0.69)
        # Signal.with_score clamps, so a promoted 0.95 cannot become 1.09.
        high, _ = learner.adjust(_signal(score=0.95))
        assert high.score == 1.0

    def test_learning_switched_off_keeps_the_verdict_truthful(
        self, journal: Journal, cfg: Config
    ) -> None:
        """With learning_enabled False the stats still show, they just do not apply."""
        _fill_bucket(journal, BUCKET, wins=6, losses=18)
        learner = Learner(journal, cfg)
        learner.refresh()
        assert learner.verdict(BUCKET).state == "blocked"

        cfg.learning_enabled = False
        adjusted, verdict = learner.adjust(_signal(score=0.95))
        assert adjusted.score == pytest.approx(0.95)  # untouched
        assert verdict.state == "blocked"            # but still reported honestly
        assert verdict.applied is False
        assert learner.is_blocked(BUCKET) is False   # matches what the bot will do
        assert learner.blocked_buckets() == []

    def test_leaderboard_sinks_blocked_buckets(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Blocked buckets sort last whatever their expectancy; the rest by expectancy."""
        _fill_bucket(journal, "orb:CE:open:trend_up", wins=6, losses=18, seq0=0)
        _fill_bucket(journal, "ema_cross:CE:morning:trend_up", wins=9, losses=3, seq0=100)
        _fill_bucket(journal, "pullback:PE:midday:range", wins=5, losses=7, seq0=200)
        learner = Learner(journal, cfg)
        learner.refresh()

        board = learner.leaderboard()
        assert [v.bucket for v in board][-1] == "orb:CE:open:trend_up"
        assert board[-1].state == "blocked"
        unblocked = [v.expectancy for v in board if v.state != "blocked"]
        assert unblocked == sorted(unblocked, reverse=True)
        assert all(isinstance(v, BucketVerdict) for v in board)


# ----------------------------------------------------------------------------------
# 4. break-even comes from the trades, not from today's config
# ----------------------------------------------------------------------------------
class TestBreakEvenSource:
    """The reference win rate is a property of the trades, not of the settings."""

    def test_break_even_is_derived_from_the_trades_own_stop_and_target(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Trades taken on 20%/80% are judged at a 20% break-even, not the config's 37.5%."""
        assert cfg.stop_pct == 0.30 and cfg.target_pct == 0.50  # config says 0.375
        _fill_bucket(journal, BUCKET, wins=3, losses=9, stop_pct=0.20, target_pct=0.80)

        learner = Learner(journal, cfg)
        learner.refresh()
        v = learner.verdict(BUCKET)

        assert v.be_source == "trades"
        assert v.break_even == pytest.approx(0.20, abs=1e-9)
        assert v.avg_stop_pct == pytest.approx(0.20, abs=1e-9)
        assert v.avg_target_pct == pytest.approx(0.80, abs=1e-9)
        assert learner.config_break_even == pytest.approx(0.375, abs=1e-9)
        assert v.break_even != pytest.approx(learner.config_break_even)
        # 25% beats a 20% break-even badly enough not to block at be − 0.10 = 10%.
        assert v.state == "demoted"

    def test_changing_the_config_does_not_re_judge_closed_trades(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A stop_pct tweak must not retroactively move the bar those trades were held to.

        This is the money bug: if break-even followed the live config, widening the
        stop would raise the bar and block a good setup, and tightening it would lower
        the bar and quietly un-block a setup that lost real rupees.
        """
        _fill_bucket(journal, BUCKET, wins=3, losses=9, stop_pct=0.20, target_pct=0.80)
        learner = Learner(journal, cfg)
        learner.refresh()
        before = learner.verdict(BUCKET)

        cfg.stop_pct = 0.50
        cfg.target_pct = 0.50  # config break-even is now 50%
        learner.refresh()
        after = learner.verdict(BUCKET)

        assert learner.config_break_even == pytest.approx(0.50)
        assert after.break_even == pytest.approx(before.break_even, abs=1e-12)
        assert after.be_source == "trades"
        assert after.state == before.state

    def test_the_config_is_only_the_last_resort(
        self, journal: Journal, cfg: Config
    ) -> None:
        """With no usable stop/target on the rows, break-even falls back to the config.

        A zero stop price makes the derived fraction implausible (100%), so the row is
        dropped from the average and `be_source` says "config" out loud rather than
        implying the number came from the trades.
        """
        _fill_bucket(journal, BUCKET, wins=4, losses=8, stop_pct=1.0, target_pct=0.0)
        learner = Learner(journal, cfg)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert v.be_source == "config"
        assert v.break_even == pytest.approx(learner.config_break_even, abs=1e-12)

    def test_a_trailed_stop_does_not_understate_the_risk_taken(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A stop row under half of min_stop_pct is dropped, not believed.

        A trailed stop written over the original would imply a tiny stop fraction and
        therefore a tiny break-even, which makes a losing bucket look like it cleared
        the bar. Dropping the row is the safe direction.
        """
        _fill_bucket(journal, BUCKET, wins=4, losses=8, stop_pct=0.02, target_pct=0.50)
        learner = Learner(journal, cfg)
        learner.refresh()

        v = learner.verdict(BUCKET)
        assert v.be_source == "config"
        assert v.break_even == pytest.approx(0.375, abs=1e-9)


# ----------------------------------------------------------------------------------
# 5. paper and live stay apart
# ----------------------------------------------------------------------------------
class TestModeIsolation:
    """learn_across_modes defaults to False: paper results never gate live ones."""

    BUCKET = "supertrend:CE:close:trend_up"

    def _mixed_journal(self, journal: Journal) -> None:
        """A bucket that is a disaster in paper and excellent in live."""
        _fill_bucket(journal, self.BUCKET, wins=6, losses=18, mode="paper", seq0=0)
        _fill_bucket(journal, self.BUCKET, wins=9, losses=3, mode="live", seq0=200)

    def test_default_is_separate(self, journal: Journal, cfg: Config) -> None:
        """The flag defaults to False even though Config does not define it."""
        assert Learner(journal, cfg).learn_across_modes is False

    def test_paper_losses_do_not_gate_a_live_setup(
        self, journal: Journal, cfg: Config
    ) -> None:
        """The same bucket is blocked in paper and promoted in live, simultaneously."""
        self._mixed_journal(journal)

        paper_cfg = Config()
        paper_cfg.mode = "paper"
        paper = Learner(journal, paper_cfg)
        paper.refresh()

        live_cfg = Config()
        live_cfg.mode = "live"
        live = Learner(journal, live_cfg)
        live.refresh()

        assert paper.stats_mode == "paper"
        assert live.stats_mode == "live"

        pv = paper.verdict(self.BUCKET)
        lv = live.verdict(self.BUCKET)
        assert (pv.n, pv.wins, pv.state) == (24, 6, "blocked")
        assert (lv.n, lv.wins, lv.state) == (12, 9, "promoted")
        assert paper.is_blocked(self.BUCKET) is True
        assert live.is_blocked(self.BUCKET) is False
        assert live.blocked_buckets() == []

    def test_switching_mode_re_reads_the_statistics(
        self, journal: Journal, cfg: Config
    ) -> None:
        """One Learner whose cfg.mode flips must not keep the other mode's verdicts.

        A cached paper block still applied after the user switches to live is exactly
        the cross-contamination this flag exists to stop, so the cache keys on mode.
        """
        self._mixed_journal(journal)
        learner = Learner(journal, cfg)  # cfg.mode == "paper"
        learner.refresh()
        assert learner.verdict(self.BUCKET).state == "blocked"

        cfg.mode = "live"
        v = learner.verdict(self.BUCKET)  # no explicit refresh: the mode change forces it
        assert (v.n, v.state) == (12, "promoted")
        assert v.mode == "live"
        assert learner.is_blocked(self.BUCKET) is False

        cfg.mode = "paper"
        assert learner.verdict(self.BUCKET).state == "blocked"

    def test_a_live_only_bucket_is_invisible_to_paper(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Trades filed under live do not show up in a paper learner at all."""
        _fill_bucket(journal, "orb:PE:open:trend_down", wins=0, losses=24, mode="live")
        learner = Learner(journal, cfg)  # paper
        learner.refresh()

        v = learner.verdict("orb:PE:open:trend_down")
        assert (v.n, v.state) == (0, "neutral")
        assert learner.summary()["trades"] == 0
        assert learner.summary()["mode_scope"] == "paper trades only"

    def test_learn_across_modes_true_combines_them(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Opt in explicitly and both modes are pooled — 24 paper + 12 live = 36."""
        self._mixed_journal(journal)
        learner = Learner(journal, cfg, learn_across_modes=True)
        learner.refresh()

        v = learner.verdict(self.BUCKET)
        assert v.n == 36
        assert v.wins == 15
        assert learner.stats_mode is None
        assert learner.summary()["mode_scope"] == "all modes combined"

    def test_summary_reports_the_scope_it_used(
        self, journal: Journal, cfg: Config
    ) -> None:
        """The UI header must say which mode's trades produced these numbers."""
        self._mixed_journal(journal)
        learner = Learner(journal, cfg)
        learner.refresh()

        summary = learner.summary()
        assert summary["mode"] == "paper"
        assert summary["learn_across_modes"] is False
        assert summary["trades"] == 24
        assert summary["states"]["blocked"] == 1
        assert summary["blocked_buckets"] == [self.BUCKET]
        assert summary["min_samples_to_gate"] == 10
        assert summary["confidence"] == pytest.approx(0.90)
        assert "paper" in summary["headline"]


# ----------------------------------------------------------------------------------
# 6. configuration edges that would cost money
# ----------------------------------------------------------------------------------
class TestConfidenceHandling:
    """A nonsense confidence must make the gate stricter, never looser."""

    def test_a_confidence_below_a_half_falls_back_to_the_default(
        self, journal: Journal, cfg: Config
    ) -> None:
        """z would go negative and turn the lower bound into an upper bound."""
        cfg.gate_confidence = 0.10
        learner = Learner(journal, cfg)
        assert learner.confidence == pytest.approx(learner_mod.DEFAULT_CONFIDENCE)
        assert learner.z > 0

    def test_a_confidence_of_one_is_clamped_not_divided_by(
        self, journal: Journal, cfg: Config
    ) -> None:
        """Only makes the gate stricter, so clamping is safe."""
        cfg.gate_confidence = 1.5
        learner = Learner(journal, cfg)
        assert learner.confidence == pytest.approx(learner_mod.MAX_CONFIDENCE)

    def test_min_samples_never_drops_below_one(
        self, journal: Journal, cfg: Config
    ) -> None:
        """A zero would make a bucket with no trades judgeable."""
        cfg.min_samples_to_gate = 0
        assert Learner(journal, cfg).min_samples == 1
