"""Offline fixtures for the indbot integration suite: a fake API and a synthetic day.

Nothing in here touches the network. `FakeClient` implements the slice of
`INDstocksClient` that the engine, the chain picker and both brokers actually call,
and it records every call so a test can assert on what was *not* called — which is how
"PaperBroker never places an order" is proved.

The response fixtures are copied from `indstocks-api-docs.md`, including the three
shapes that are easy to get wrong and expensive to get wrong:

* the option chain's ``strikes`` is a JSON **object keyed by strike**, not an array;
* ``iv`` is a **percentage** and a **sibling** of ``greeks``, not a member of it;
* the historical endpoints wrap in ``{"success": bool, "data": {...}}`` — a ``success``
  boolean, **not** the ``status`` string the order and funds endpoints use.

The fixtures are stored in that raw wire shape and normalised on the way out by the
same `indbot.api._normalise_candles` the real client uses, so a fixture that drifts
away from the documented shape fails here rather than in production.

This module is importable both as a pytest ``conftest`` and as a plain module from a
``unittest`` run; it defines no pytest fixtures, only builders.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

# --- import bootstrap -------------------------------------------------------------
# The suite runs from anywhere: `python -m unittest discover tests`, `pytest`, or an
# IDE. Put the package root on sys.path rather than depending on the launcher's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indbot import clock  # noqa: E402
from indbot.api import _normalise_candles  # noqa: E402
from indbot.config import Config  # noqa: E402
from indbot.journal import Journal  # noqa: E402
from indbot.types import Candle, OptionLeg  # noqa: E402

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# the pinned session
# ----------------------------------------------------------------------------------

#: Monday 26 October 2026 — a weekday and not in `clock.HOLIDAYS_2026`, so
#: `clock.is_market_open` is True inside session hours and the brokers' session guard
#: does not have to be switched off to let an order through.
TRADING_DAY = date(2026, 10, 26)

#: The previous session, used only to warm EMA50/ADX(14) up so `indicators.regime`
#: returns a real verdict instead of its "range" safe default.
WARMUP_DAY = date(2026, 10, 23)

#: 09:35:00 — the 09:30 candle has just closed, so it is the last CLOSED bar and the
#: signal built from it is still fresh by `Engine._signal_fresh`.
ENTRY_AT = "2026-10-26 09:35:00"

EXPIRY = "2026-10-29"
LATER_EXPIRIES = ("2026-11-05", "2026-11-26")

#: NIFTY's spot through the breakout, chosen so the documented 24450 ATM node sits at
#: the money. `underlying_ltp` in the chain fixture agrees with the 09:30 close to
#: within the chain picker's 1% spot-disagreement warning.
ATM_STRIKE = 24450.0
CHAIN_SPOT = 24471.7
STRIKE_STEP = 50.0
NIFTY_SCRIP = "40000001"
NIFTY_CODE = "NSE_40000001"
NIFTY_LOT = 75

#: The 24450 CE / PE nodes verbatim from the docs' Option Chain example. Every other
#: strike in the ladder is generated around them; these two are copied so the fixture
#: is checkably the documented payload and not an invention.
DOC_ATM_CE: dict[str, Any] = {
    "security_id": "45108",
    "trading_symbol": "NIFTY-Aug2026-24450-CE",
    "last_price": 167.9,
    "previous_close_price": 274.95,
    "oi": 1608490,
    "previous_oi": 1606988,
    "volume": 9079330,
    "top_bid_price": 166.05,
    "top_bid_quantity": 195,
    "top_ask_price": 167.5,
    "top_ask_quantity": 130,
    "iv": 10.5,
    "greeks": {"delta": 0.56, "gamma": 0.0011, "theta": -10.04, "vega": 13.39},
}
DOC_ATM_PE: dict[str, Any] = {
    "security_id": "45109",
    "trading_symbol": "NIFTY-Aug2026-24450-PE",
    "last_price": 117.65,
    "previous_close_price": 88.9,
    "oi": 1579370,
    "previous_oi": 1579129,
    "volume": 11974495,
    "top_bid_price": 117.65,
    "top_bid_quantity": 195,
    "top_ask_price": 119.4,
    "top_ask_quantity": 195,
    "iv": 10.4,
    "greeks": {"delta": -0.44, "gamma": 0.0011, "theta": -9.95, "vega": 13.39},
}

#: `detailed_avl_balance.option_buy` is the only field risk rule 9 reads; the rest of
#: the block is kept so the fixture is recognisably the documented /funds payload.
FUNDS_DATA: dict[str, Any] = {
    "sod_balance": 500000.0,
    "pledge_received": 0,
    "pledge_remained": 0,
    "detailed_avl_balance": {
        "option_sell": 500000.0,
        "future": 500000.0,
        "option_buy": 500000.0,
        "comm_option_buy": 0.0,
        "eq_mis": 500000.0,
        "eq_cnc": 500000.0,
        "eq_mtf": 500000.0,
    },
    "withdrawal_balance": 500000.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
}

PROFILE_DATA: dict[str, Any] = {"client_id": "TEST0001", "name": "Integration Test", "email": "t@example.com"}


# ----------------------------------------------------------------------------------
# the clock
# ----------------------------------------------------------------------------------


@contextmanager
def fake_now(stamp: str) -> Iterator[None]:
    """Pin `clock.now()` to `stamp` ("YYYY-MM-DD HH:MM:SS") for the block.

    Restores the previous INDBOT_FAKE_NOW rather than deleting it, so nested uses and
    a developer running with the env var already set both behave.
    """
    previous = os.environ.get("INDBOT_FAKE_NOW")
    os.environ["INDBOT_FAKE_NOW"] = stamp
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("INDBOT_FAKE_NOW", None)
        else:
            os.environ["INDBOT_FAKE_NOW"] = previous


def set_now(stamp: str) -> None:
    """Move the pinned clock. For a test that steps through a session tick by tick."""
    os.environ["INDBOT_FAKE_NOW"] = stamp


def at(day: date, hhmm: str) -> int:
    """Epoch SECONDS for an "HH:MM" IST time on `day` — the unit of a candle's `ts`."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    return clock.epoch_s(datetime(day.year, day.month, day.day, hour, minute, tzinfo=clock.IST))


# ----------------------------------------------------------------------------------
# synthetic candles
# ----------------------------------------------------------------------------------


def _bar(ts: int, o: float, h: float, l: float, c: float, v: int) -> Candle:
    """One candle, with the high/low widened to contain the body.

    A fixture whose high is below its close is rejected by `Strategy._check_series`,
    which is correct of it and tedious here, so the envelope is enforced at the source.
    """
    return {
        "ts": int(ts),
        "o": round(float(o), 2),
        "h": round(max(h, o, c), 2),
        "l": round(min(l, o, c), 2),
        "c": round(float(c), 2),
        "v": int(v),
    }


def warmup_candles(day: date = WARMUP_DAY, *, start: float = 24150.0) -> list[Candle]:
    """A full prior session of 5-minute bars that drifts gently up.

    Its only job is to warm EMA50 and ADX(14): the ORB trigger reads the current day's
    candles only, and `indicators.vwap` re-anchors at each session open, so this day
    cannot leak into the signal it is there to make computable.
    """
    out: list[Candle] = []
    price = start
    open_s = at(day, "09:15")
    for i in range(75):  # 09:15 .. 15:25
        step = 3.5 if i % 4 else -2.0
        close = price + step
        out.append(_bar(open_s + i * 300, price, max(price, close) + 4.0,
                        min(price, close) - 4.0, close, 90_000 + (i % 7) * 4_000))
        price = close
    return out


def orb_breakout_candles(day: date = TRADING_DAY) -> list[Candle]:
    """The trading day: a tight opening range, then a decisive break above its high.

    09:15/09:20/09:25 are the `or_minutes=15` window and hold 24400-24450. The 09:30
    bar is the first close above 24450, which is exactly the `orb` CE trigger, and the
    bars after it stay above — so ORB fires once, on 09:30, and never again that day
    (the trigger is "the FIRST close outside the range", derived from the series).
    """
    open_s = at(day, "09:15")
    bars: list[Candle] = [
        _bar(open_s + 0,   24420.0, 24450.0, 24405.0, 24435.0, 180_000),
        _bar(open_s + 300, 24435.0, 24448.0, 24400.0, 24412.0, 165_000),
        _bar(open_s + 600, 24412.0, 24446.0, 24402.0, 24441.0, 158_000),
        # 09:30 — the breakout. A wide body on heavy volume, so the confluence score
        # is not pinned to a doji and the ORB reason reads like a real breakout.
        _bar(open_s + 900, 24443.0, 24478.0, 24440.0, 24472.0, 310_000),
    ]
    price = 24472.0
    for i in range(4, 75):  # 09:35 .. 15:25, holding above the range high
        step = 2.5 if i % 3 else -1.5
        close = price + step
        bars.append(_bar(open_s + i * 300, price, max(price, close) + 3.0,
                         min(price, close) - 3.0, close, 120_000 + (i % 5) * 6_000))
        price = close
    return bars


def nifty_candles() -> list[Candle]:
    """Warm-up session plus the ORB day, strictly ascending, as one series."""
    return warmup_candles() + orb_breakout_candles()


def historical_response(candles_by_code: dict[str, list[Candle]]) -> dict[str, Any]:
    """The raw `GET /market/historical` envelope — a `success` BOOLEAN, not `status`.

    This is the shape the fixture is stored in; `FakeClient.historical` normalises it
    through the real `api._normalise_candles`, so a fixture that stops matching the
    documented shape breaks the suite instead of being quietly accepted.
    """
    return {
        "success": True,
        "data": {code: {"candles": [dict(c) for c in candles]}
                 for code, candles in candles_by_code.items()},
    }


# ----------------------------------------------------------------------------------
# the option chain
# ----------------------------------------------------------------------------------


def _ladder_leg(strike: float, side: str, spot: float, seq: int) -> dict[str, Any]:
    """One generated `ce`/`pe` node: intrinsic plus a bell-shaped time value.

    Not a pricing model — just a monotone, realistic ladder. What matters for the
    tests is that premiums fall away from the money while |delta| falls with them, so
    the cheap strikes the per-trade budget *can* afford are exactly the ones the delta
    band then rejects. That is the real shape of the user's problem.
    """
    moneyness = strike - spot
    time_value = 150.0 * math.exp(-((moneyness / 400.0) ** 2))
    intrinsic = max(spot - strike, 0.0) if side == "ce" else max(strike - spot, 0.0)
    last = max(0.5, round(intrinsic + time_value, 2))

    # Top of book: a ~0.9% spread, at least one 5-paisa tick wide, so the mid-priced
    # strikes clear `max_spread_pct` and the near-worthless tails do not.
    half = max(0.025, round(last * 0.0045, 2))
    bid = max(0.05, round((last - half) * 20) / 20)
    ask = round((last + half) * 20) / 20

    ce_delta = 1.0 / (1.0 + math.exp(moneyness / 180.0))
    delta = round(ce_delta, 4) if side == "ce" else round(ce_delta - 1.0, 4)
    weight = math.exp(-((moneyness / 500.0) ** 2))

    return {
        "security_id": str(45000 + seq),
        "trading_symbol": f"NIFTY-Oct2026-{int(strike)}-{side.upper()}",
        "last_price": last,
        "previous_close_price": round(last * 1.1, 2),
        "oi": int(60_000 + 1_500_000 * weight),
        "previous_oi": int(59_000 + 1_480_000 * weight),
        "volume": int(20_000 + 9_000_000 * weight),
        "top_bid_price": bid,
        "top_bid_quantity": 195,
        "top_ask_price": ask,
        "top_ask_quantity": 130,
        # iv is a PERCENTAGE and a SIBLING of greeks, exactly as documented.
        "iv": round(10.0 + abs(moneyness) / 400.0, 2),
        "greeks": {
            "delta": delta,
            "gamma": round(0.0011 * weight, 6),
            "theta": round(-10.0 * weight, 2),
            "vega": round(13.4 * weight, 2),
        },
    }


def chain_data(
    *,
    spot: float = CHAIN_SPOT,
    expiry: str = EXPIRY,
    atm: float = ATM_STRIKE,
    strike_count: int = 10,
) -> dict[str, Any]:
    """The chain's `data` block: `{underlying_ltp, expiry, strikes}`.

    `strikes` is an OBJECT keyed by the strike price as a string, and the keys are
    emitted in an order that is NOT numeric (highest first, '9'-prefixed keys mixed
    in by string sort) on purpose: `ChainPicker._sorted_strikes` promises to sort them
    numerically, and a fixture in helpfully-sorted order would never test that.
    """
    strikes: dict[str, Any] = {}
    seq = 0
    offsets = list(range(-strike_count, strike_count + 1))
    for offset in sorted(offsets, key=lambda o: -o):  # deliberately descending
        strike = atm + offset * STRIKE_STEP
        node = {"ce": _ladder_leg(strike, "ce", spot, seq),
                "pe": _ladder_leg(strike, "pe", spot, seq + 1)}
        if strike == atm:
            # The documented ATM payload wins over the generated one.
            node = {"ce": dict(DOC_ATM_CE), "pe": dict(DOC_ATM_PE)}
        strikes[str(int(strike))] = node
        seq += 2
    return {"underlying_ltp": spot, "expiry": expiry, "strikes": strikes}


def chain_response(**kw: Any) -> dict[str, Any]:
    """The full HTTP envelope the Option Chain returns: `{"status": "success", "data": …}`.

    `INDstocksClient.option_chain` hands its caller the inner `data` block, which is
    what `FakeClient` returns; this exists so the fixture records the whole documented
    payload and a reader can see which half is unwrapped where.
    """
    return {"status": "success", "data": chain_data(**kw)}


def leg_from_chain(strike: float, side: str, *, lot_size: int = NIFTY_LOT) -> OptionLeg:
    """Build one OptionLeg straight from the chain fixture, for a broker-level test.

    Mirrors `ChainPicker._build_leg`, including reading `iv` as a sibling of `greeks`.
    """
    node = chain_data()["strikes"][str(int(strike))][side.lower()]
    greeks = node["greeks"]
    return OptionLeg(
        underlying="NIFTY",
        expiry=EXPIRY,
        strike=float(strike),
        side=side.upper(),
        security_id=str(node["security_id"]),
        trading_symbol=str(node["trading_symbol"]),
        lot_size=lot_size,
        exchange="NSE",
        ltp=float(node["last_price"]),
        bid=float(node["top_bid_price"]),
        ask=float(node["top_ask_price"]),
        oi=int(node["oi"]),
        volume=int(node["volume"]),
        iv=float(node["iv"]),
        delta=float(greeks["delta"]),
        gamma=float(greeks["gamma"]),
        theta=float(greeks["theta"]),
        vega=float(greeks["vega"]),
        tick_size=0.05,
    )


# ----------------------------------------------------------------------------------
# the instrument masters
# ----------------------------------------------------------------------------------

#: `source=index` is three columns and its second column holds the index NAME despite
#: its SEGMENT header — the file the bot must read positionally. NIFTY's id matches
#: the constant in config; BANKNIFTY's deliberately does NOT, so the self-healing
#: "the file wins over the constant" path is exercised by the suite rather than
#: assumed.
INDEX_CSV = (
    "EXCH,SEGMENT,SECURITY_ID\n"
    "NSE,NIFTY 50,40000001\n"
    "NSE,NIFTY BANK,40000005\n"
    "NSE,NIFTY FIN SERVICE,40000003\n"
    "NSE,NIFTY MIDCAP SELECT,40000016\n"
    "BSE,SENSEX,40000100\n"
    "NSE,NIFTY IT,40000004\n"
)

_FNO_HEADER = (
    "EXCH,SEGMENT,SECURITY_ID,INSTRUMENT_NAME,EXPIRY_CODE,TRADING_SYMBOL,LOT_UNITS,"
    "CUSTOM_SYMBOL,EXPIRY_DATE,STRIKE_PRICE,OPTION_TYPE,TICK_SIZE,EXPIRY_FLAG,"
    "SEM_EXCH_INSTRUMENT_TYPE,SERIES,SYMBOL_NAME,UNDERLYING_SCRIP_NAME,FREEZE_QTY,"
    "UPPER_LIMIT,LOWER_LIMIT,ISIN,GENERAL_FACTOR,DELIVERY_UNIT"
)

_FNO_LOTS = (("NIFTY", "NSE", "NFO", 75), ("BANKNIFTY", "NSE", "NFO", 35),
             ("FINNIFTY", "NSE", "NFO", 65), ("MIDCPNIFTY", "NSE", "NFO", 120),
             ("SENSEX", "BSE", "BFO", 20))


def _fno_row(*, exch: str, segment: str, security_id: str, trading_symbol: str,
             lot: int, strike: str, option_type: str, symbol: str) -> str:
    """One 23-column F&O master row, in the documented column order."""
    return ",".join([
        exch, segment, security_id, "OPTIDX", "1", trading_symbol, str(lot),
        trading_symbol, EXPIRY, strike, option_type, "0.05", "W", "OPTIDX", "",
        symbol, symbol, str(lot * 24), "9999", "0.05", "", "0", "",
    ])


def fno_csv() -> str:
    """The F&O master: a lot-size row per index, plus a row per chain contract.

    The per-contract rows are what make `Instruments.tick_size` answer from the file
    instead of falling back to its 5-paise default, so the limit prices the live
    broker builds in these tests are rounded to a tick the master actually published.
    """
    rows = [_FNO_HEADER]
    for symbol, exch, segment, lot in _FNO_LOTS:
        rows.append(_fno_row(exch=exch, segment=segment, security_id=f"T{lot}{symbol[:3]}",
                             trading_symbol=f"{symbol}-{EXPIRY}-FUT", lot=lot,
                             strike="0", option_type="XX", symbol=symbol))
    for key, node in chain_data()["strikes"].items():
        for side in ("ce", "pe"):
            payload = node[side]
            rows.append(_fno_row(
                exch="NSE", segment="NFO", security_id=str(payload["security_id"]),
                trading_symbol=str(payload["trading_symbol"]), lot=NIFTY_LOT,
                strike=key, option_type=side.upper(), symbol="NIFTY",
            ))
    return "\n".join(rows) + "\n"


# ----------------------------------------------------------------------------------
# the fake client
# ----------------------------------------------------------------------------------

#: Every method that reaches an order endpoint. `PaperBroker` must call none of them;
#: `assertNoOrders` is written against this set rather than a hand-listed one so a new
#: order method cannot slip past the assertion.
ORDER_METHODS: frozenset[str] = frozenset({
    "place_order", "modify_order", "cancel_order", "order_book", "order_details",
    "order_trades", "trade_book", "positions", "margin",
})


class FakeClient:
    """An offline stand-in for `INDstocksClient`, recording every call it receives.

    Implements the surface the engine, the chain picker and both brokers use. Market
    data comes from the fixtures above; orders are filled synthetically at their own
    limit price, which is the honest best case and keeps the arithmetic in a test
    checkable by hand.

    `calls` is the point of the class as much as the data is: a test asserts on what
    was never called, and "the paper path cannot place an order" is only a real claim
    if something is watching the wire.
    """

    def __init__(
        self,
        *,
        candles: dict[str, list[Candle]] | None = None,
        chain: dict[str, Any] | None = None,
        expiries_by_underlying: dict[str, list[str]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._candles = candles if candles is not None else {NIFTY_CODE: nifty_candles()}
        self._chain = chain if chain is not None else chain_data()
        self._expiries = expiries_by_underlying or {}
        self._ltp: dict[str, float] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self._order_seq = 0
        self.closed = False
        self.seed_ltp_from_chain()

    # -- recording ------------------------------------------------------------
    def _record(self, method: str, **kw: Any) -> None:
        self.calls.append((method, kw))

    def method_calls(self, method: str) -> list[dict[str, Any]]:
        """Every recorded call to `method`, in order, with its keyword arguments."""
        return [kw for name, kw in self.calls if name == method]

    @property
    def order_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Every recorded call that would have reached an order endpoint."""
        return [(name, kw) for name, kw in self.calls if name in ORDER_METHODS]

    def reset_calls(self) -> None:
        """Forget the call log, keeping the market data and the order book."""
        self.calls.clear()

    # -- market data ----------------------------------------------------------
    def seed_ltp_from_chain(self) -> None:
        """Price every contract in the chain fixture at its own `last_price`."""
        for node in self._chain["strikes"].values():
            for side in ("ce", "pe"):
                payload = node.get(side)
                if isinstance(payload, dict):
                    self._ltp[f"NSE_{payload['security_id']}"] = float(payload["last_price"])

    def set_ltp(self, security_id: str, price: float, *, exchange: str = "NSE") -> None:
        """Move one contract's mark. How a test walks a position to its stop or target."""
        self._ltp[f"{exchange}_{security_id}"] = float(price)

    def profile(self) -> dict[str, Any]:
        self._record("profile")
        return dict(PROFILE_DATA)

    def funds(self) -> dict[str, Any]:
        self._record("funds")
        return dict(FUNDS_DATA)

    def instruments_csv(self, source: str) -> str:
        self._record("instruments_csv", source=source)
        if source == "index":
            return INDEX_CSV
        if source == "fno":
            return fno_csv()
        return _FNO_HEADER + "\n"

    def expiries(self, underlying: str, *, segment: str = "DERIVATIVE") -> list[str]:
        self._record("expiries", underlying=underlying, segment=segment)
        listed = self._expiries.get(underlying)
        if listed is not None:
            return list(listed)
        # Not pre-sorted: `nearest_expiry` promises to sort and to drop past dates.
        return [LATER_EXPIRIES[1], EXPIRY, LATER_EXPIRIES[0]]

    def expired_expiries(self, underlying: str, expiry_from: str, expiry_to: str) -> list[str]:
        self._record("expired_expiries", underlying=underlying,
                     expiry_from=expiry_from, expiry_to=expiry_to)
        return [EXPIRY]

    def expired_contracts(self, underlying: str, expiry: str,
                          instrument_type: str = "OPTIDX") -> list[dict[str, Any]]:
        self._record("expired_contracts", underlying=underlying, expiry=expiry,
                     instrument_type=instrument_type)
        return []

    def historical(self, interval: str, scrip_codes: list[str], start_ms: int,
                   end_ms: int) -> dict[str, list[Candle]]:
        """Fixture candles inside [start_ms, end_ms], normalised exactly as the real
        client normalises them.

        The window is honoured rather than ignored: a test moves the clock forward and
        must see the bars that had closed by then and no others, or "never reads a
        forming candle" is untestable here.
        """
        self._record("historical", interval=interval, scrip_codes=list(scrip_codes),
                     start_ms=start_ms, end_ms=end_ms)
        raw = historical_response(
            {code: self._window(self._candles.get(code, []), start_ms, end_ms)
             for code in scrip_codes}
        )
        if not raw.get("success"):  # pragma: no cover - the fixture always succeeds
            return {code: [] for code in scrip_codes}
        out: dict[str, list[Candle]] = {}
        for code in scrip_codes:
            block = raw["data"].get(code) or {}
            out[code] = _normalise_candles(block.get("candles"), code=code)
        return out

    def historical_expired(self, interval: str, trading_symbols: list[str], start_ms: int,
                           end_ms: int, segment: str = "NFO") -> dict[str, list[Candle]]:
        self._record("historical_expired", interval=interval,
                     trading_symbols=list(trading_symbols), start_ms=start_ms,
                     end_ms=end_ms, segment=segment)
        return {symbol: [] for symbol in trading_symbols}

    @staticmethod
    def _window(candles: Sequence[Candle], start_ms: int, end_ms: int) -> list[Candle]:
        """Candles whose open time falls in the requested MILLISECOND window.

        `ts` is epoch seconds and the request bounds are epoch milliseconds — the unit
        boundary CONTRACTS.md puts in bold, converted here and nowhere else.
        """
        lo = int(start_ms) // 1000
        hi = int(end_ms) // 1000
        return [dict(c) for c in candles if lo <= int(c["ts"]) <= hi]  # type: ignore[misc]

    def ltp(self, scrip_codes: list[str]) -> dict[str, float]:
        self._record("ltp", scrip_codes=list(scrip_codes))
        # A code with no price is ABSENT from the result, never 0.0 — a zero mark
        # would read as a worthless option and trip a stop instantly.
        return {code: self._ltp[code] for code in scrip_codes if code in self._ltp}

    def quotes(self, scrip_codes: list[str]) -> dict[str, dict[str, Any]]:
        self._record("quotes", scrip_codes=list(scrip_codes))
        return {code: {"last_price": self._ltp[code]} for code in scrip_codes if code in self._ltp}

    def depth(self, scrip_codes: list[str]) -> dict[str, dict[str, Any]]:
        self._record("depth", scrip_codes=list(scrip_codes))
        return {}

    def option_chain(self, exchange: str, segment: str, underlying_scrip: str,
                     expiry: str, strike_count: int = 10) -> dict[str, Any]:
        """The chain's `data` block, as `INDstocksClient.option_chain` returns it."""
        self._record("option_chain", exchange=exchange, segment=segment,
                     underlying_scrip=underlying_scrip, expiry=expiry,
                     strike_count=strike_count)
        return self._chain

    # -- orders ---------------------------------------------------------------
    def place_order(self, **kw: Any) -> dict[str, Any]:
        """Accept an order and fill it in full at its own limit price."""
        self._record("place_order", **kw)
        self._order_seq += 1
        order_id = f"FAKE-{self._order_seq:04d}"
        self.orders[order_id] = {
            "order_id": order_id,
            "exch_order_id": f"EX-{self._order_seq:04d}",
            "order_status": "SUCCESS",
            "txn_type": kw.get("txn_type"),
            "security_id": kw.get("security_id"),
            "qty": int(kw.get("qty") or 0),
            "traded_qty": int(kw.get("qty") or 0),
            "traded_price": float(kw.get("limit_price") or 0.0),
            "requested_price": float(kw.get("limit_price") or 0.0),
            "remarks": kw.get("remarks"),
            "id": order_id,
        }
        return {"status": "success", "data": {"order_id": order_id}}

    def modify_order(self, **kw: Any) -> dict[str, Any]:
        self._record("modify_order", **kw)
        return {"status": "success", "data": {"order_id": kw.get("order_id")}}

    def cancel_order(self, order_id: str, *, segment: str = "DERIVATIVE") -> dict[str, Any]:
        self._record("cancel_order", order_id=order_id, segment=segment)
        row = self.orders.get(order_id)
        if row is not None:
            row["order_status"] = "CANCELLED"
        return {"status": "success", "data": {"order_id": order_id, "order_status": "CANCELLED"}}

    def order_book(self) -> list[dict[str, Any]]:
        self._record("order_book")
        return [dict(row) for row in self.orders.values()]

    def order_details(self, order_id: str, *, segment: str = "DERIVATIVE") -> dict[str, Any]:
        self._record("order_details", order_id=order_id, segment=segment)
        return {"status": "success", "data": dict(self.orders.get(order_id, {}))}

    def order_trades(self, order_id: str, *, segment: str = "DERIVATIVE") -> list[dict[str, Any]]:
        self._record("order_trades", order_id=order_id, segment=segment)
        row = self.orders.get(order_id)
        if not row:
            return []
        return [{"quantity": row["traded_qty"], "price": row["traded_price"]}]

    def trade_book(self, *, segment: str = "DERIVATIVE") -> list[dict[str, Any]]:
        self._record("trade_book", segment=segment)
        return []

    def positions(self, segment: str | None = None, product: str | None = None) -> list[dict[str, Any]]:
        self._record("positions", segment=segment, product=product)
        return []

    def margin(self, **kw: Any) -> dict[str, Any]:
        self._record("margin", **kw)
        return {"status": "success", "data": {}}

    def close(self) -> None:
        self.closed = True


# ----------------------------------------------------------------------------------
# object builders
# ----------------------------------------------------------------------------------

#: Settings every test starts from. Deliberately NOT the production defaults:
#: `underlyings`/`enabled_setups` are narrowed to one index and one setup so exactly
#: one signal is in play and an assertion names a specific trade, and `min_score` is
#: dropped to 0 so these tests measure the plumbing rather than re-measuring the
#: confluence table (which is strategy.py's own unit tests' job).
BASE_OVERRIDES: dict[str, Any] = {
    "mode": "paper",
    "underlyings": ["NIFTY"],
    "enabled_setups": ["orb"],
    "min_score": 0.0,
    "interval": "5minute",
    "or_minutes": 15,
    "poll_seconds": 20,
    "slippage_ticks": 1,
    "cost_per_order": 20.0,
}


def make_config(tmp: Path, **overrides: Any) -> Config:
    """A validated Config rooted in `tmp`, never touching the user's config.json.

    `_path` is left None so `update()` cannot write a test's settings over the real
    file; a test that wants persistence calls `save(path)` explicitly.
    """
    values = dict(BASE_OVERRIDES)
    values.update(overrides)
    cfg = Config.load(tmp / "config.json")          # the file does not exist: pure defaults
    cfg._path = None                                 # noqa: SLF001 - test isolation
    cfg.update(values, persist=False)
    return cfg


def make_journal(tmp: Path, name: str = "indbot.db") -> Journal:
    """A Journal on a fresh SQLite file under `tmp`."""
    return Journal(tmp / name)


def make_instruments(client: FakeClient, tmp: Path):
    """An `Instruments` cache under `tmp`, so no test writes into `data/`."""
    from indbot.instruments import Instruments

    return Instruments(client, tmp / "instruments")


def make_engine(cfg: Config, journal: Journal, client: FakeClient, tmp: Path, **kw: Any):
    """An Engine wired to the fake client, with its caches inside `tmp`.

    `capture_logs` defaults to False: the engine installs a logging handler on the
    package logger, and a suite that builds a dozen engines would otherwise stack a
    dozen handlers. One test turns it back on to check the snapshot's `log` key.
    """
    from indbot.engine import Engine

    kw.setdefault("capture_logs", False)
    kw.setdefault("instruments", make_instruments(client, tmp))
    return Engine(cfg, journal, client=client, **kw)


def quiet_logging() -> None:
    """Silence indbot's logging for the suite.

    The modules log at WARNING and CRITICAL by design — an exit, an arm, a blocked
    entry. That is correct in production and noise in a test run, so the package
    logger is muted here rather than by weakening any module.
    """
    logging.getLogger("indbot").setLevel(logging.CRITICAL + 1)
    logging.getLogger("indbot").propagate = False
