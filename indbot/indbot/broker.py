"""Order placement — the only module in indbot that may send a real order.

Everything else in the system produces *intent*: a signal, a chosen leg, a risk
verdict. This module is where intent becomes rupees, so it is deliberately
paranoid and deliberately boring.

Three things are worth knowing before changing anything here:

1. ``qty`` is in UNITS, not lots. One lot of NIFTY is ``leg.lot_size`` units (75 at
   the time of writing). The exchange rejects ``QtyMultipleOfLotSize``, but a qty
   that is wrong *and* still a multiple is a silently wrong position size, which is
   worse than a rejection. Both brokers assert the multiple before sending.

2. We send a marketable ``LIMIT``, never ``MARKET``. INDstocks converts a
   ``MARKET`` derivative order into a ``LIMIT`` at "the live price plus (BUY) or
   minus (SELL) a price-protection band" (docs, Place Order → *Market Orders Are
   Converted to Limit Orders*). That band is not under our control and is not
   documented as a number, so a "market" order is really a limit order at a price
   we cannot predict. We would rather choose the price ourselves, a couple of ticks
   through the touch, and know exactly what the worst fill can be.

3. ``remarks`` is a bookkeeping tag, not an idempotency key. The docs say it
   explicitly: "two orders sent with the same tag are two orders". Deduplication
   therefore happens on our side, in process, before the request is sent.

``PaperBroker`` cannot reach the order endpoints *structurally*: it is handed a
``QuoteOnlyClient`` facade that raises on every order method, so a future edit that
"just adds" a real order call to the paper path fails loudly instead of trading.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import TYPE_CHECKING, Any

from . import clock
from .errors import ApiError, ConfigError, DataError, OrderError
from .types import OptionLeg

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .api import INDstocksClient
    from .config import Config

log = logging.getLogger(__name__)

# --- order body constants -------------------------------------------------
# The bot trades index options intraday, long premium only. These never vary, so
# they are constants rather than parameters: a stray "MARGIN" or "CNC" would change
# what the position costs and how it is squared off.
SEGMENT: str = "DERIVATIVE"
PRODUCT: str = "INTRADAY"
ORDER_TYPE: str = "LIMIT"
VALIDITY: str = "DAY"
IS_AMO: bool = False

#: `remarks` is truncated to this many characters by INDstocks; we truncate first so
#: the value we log is the value they stored.
MAX_REMARK_LEN: int = 100

#: Values reserved for INDstocks' own channel tags. Sending one is rejected with a
#: RequestValidationException. Matching is on the STORED value (i.e. after
#: truncation) and ignores case and surrounding whitespace.
RESERVED_REMARKS: frozenset[str] = frozenset({"TV-TERMINAL"})

# --- order status vocabulary (docs: Orders → Order Status Types) ------------
# Do not invent statuses. These are the documented ones, verbatim, normalised to
# upper case with whitespace around the hyphen collapsed.
STATUS_FILLED: frozenset[str] = frozenset({"SUCCESS"})
STATUS_PARTIAL_TERMINAL: frozenset[str] = frozenset(
    {"PARTIALLY FILLED - CANCELLED", "PARTIALLY FILLED - EXPIRED"}
)
STATUS_DEAD: frozenset[str] = frozenset({"ABORTED", "CANCELLED", "EXPIRED", "FAILED"})
STATUS_PENDING: frozenset[str] = frozenset(
    {
        "QUEUED",
        "O-PENDING",
        "SL-PENDING",
        "PROCESSING",
        "INITIATED",
        "MODIFIED",
        "PENDING",
        "PARTIALLY FILLED",
    }
)
TERMINAL_STATUSES: frozenset[str] = STATUS_FILLED | STATUS_PARTIAL_TERMINAL | STATUS_DEAD

#: Response-envelope words that are NOT order statuses. `POST /order` answers
#: `{"status": "success", "data": {"order_status": ...}}`, so a naive read of
#: "status" picks up the envelope and would look like a fill. ("FAILED" is a real
#: order status and is deliberately absent from this set.)
_ENVELOPE_WORDS: frozenset[str] = frozenset({"ERROR", "FAIL", "FAILURE", "OK"})

# --- defaults -------------------------------------------------------------
DEFAULT_SLIPPAGE_TICKS: int = 2
DEFAULT_PER_ORDER_COST: float = 20.0  # flat brokerage+tax approximation, per order
DEFAULT_FILL_TIMEOUT_S: float = 20.0
DEFAULT_POLL_INTERVAL_S: float = 0.6
DEFAULT_TICK_SIZE: float = 0.05
#: How many consecutive `GET /order` lookup failures to tolerate while polling. The
#: endpoint answers 512 (not 404) for an order it cannot fetch, so a persistent 512
#: is a real lookup failure, not a transient one.
MAX_LOOKUP_FAILURES: int = 4

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------
def _num(value: Any, default: float = 0.0) -> float:
    """Coerce an API numeric-or-empty-string field to a float.

    The order book returns prices and quantities as strings, and as `""` when the
    field does not apply ("traded_price": "" on an unfilled order). A naive
    float("") raises, so every read of those fields goes through here.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def normalise_status(raw: Any) -> str:
    """Upper-case an order status and collapse its spacing to the documented form.

    "Partially Filled-Cancelled" and "PARTIALLY FILLED - CANCELLED" are the same
    state; comparing raw strings against the documented table would miss one of them.
    """
    text = _WS_RE.sub(" ", str(raw or "").strip().upper().replace("_", " "))
    text = re.sub(r"\s*-\s*", " - ", text)
    # "O-PENDING"/"SL-PENDING" are single tokens in the docs, not "X - Y" pairs.
    if text in {"O - PENDING", "SL - PENDING"}:
        text = text.replace(" - ", "-")
    return text


def is_terminal_status(status: str) -> bool:
    """True when the exchange will not change this order's state any further."""
    return normalise_status(status) in TERMINAL_STATUSES


def round_to_tick(price: float, tick: float, *, side: str) -> float:
    """Round `price` to a whole multiple of `tick`, away from the touch.

    A BUY limit rounds UP and a SELL limit rounds DOWN, so rounding can only ever
    make the order *more* marketable. Rounding a BUY down by one tick is how an
    entry sits unfilled while the move happens without us.

    Decimal, not float arithmetic: 43.55 / 0.05 is 870.9999... in binary floating
    point, and a floor() of that is a tick below the price we meant to send.
    """
    if tick <= 0:
        tick = DEFAULT_TICK_SIZE  # a zero/absent tick must never divide
    if price <= 0:
        # Non-positive input (a SELL priced below zero once the slippage buffer comes
        # off) clamps to one tick: zero is rejected as LimitPriceMustBeAboveZero.
        return round(float(Decimal(str(tick))), 4)
    rounding = ROUND_CEILING if side.upper() == "BUY" else ROUND_FLOOR
    dec_tick = Decimal(str(tick))
    steps = (Decimal(str(price)) / dec_tick).to_integral_value(rounding=rounding)
    out = float(steps * dec_tick)
    if out <= 0:
        # A SELL of a 4-paisa option floors to zero. One tick is the lowest price the
        # exchange accepts, and an exit at one tick beats an exit never sent.
        out = float(dec_tick)
    return round(out, 4)


def sanitize_remark(remark: str, *, fallback: str = "indbot") -> str:
    """Make `remark` safe to send: trimmed, <=100 chars, never a reserved tag.

    This deliberately *repairs* rather than refuses. The remark is a cosmetic
    bookkeeping tag; refusing to send an exit because its label was malformed would
    strand a live position, which is the one outcome this system must never have.
    The repair is logged at WARNING so it is still visible.
    """
    text = _CTRL_RE.sub(" ", str(remark or ""))
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        text = fallback
    stored = text[:MAX_REMARK_LEN].strip()  # INDstocks matches the STORED value
    if _WS_RE.sub("", stored).upper() in {r.replace(" ", "") for r in RESERVED_REMARKS}:
        repaired = f"{fallback}-{stored}"[:MAX_REMARK_LEN]
        log.warning("remark %r is a reserved INDstocks tag; sending %r instead", stored, repaired)
        return repaired
    if stored != text:
        log.warning("remark truncated to %d chars: %r", MAX_REMARK_LEN, stored)
    return stored


def make_remark(
    setup: str,
    side: str,
    underlying: str,
    signal_ts: int | None = None,
    *,
    prefix: str = "indbot",
) -> str:
    """Build the per-order tag, e.g. `indbot/orb/CE/NIFTY/1792990800`.

    It is a correlation key for reconciling fills against the journal (it comes back
    on the order book and on every trade-book fill), NOT an idempotency key.
    """
    parts = [prefix, str(setup or "manual"), str(side or ""), str(underlying or "")]
    if signal_ts:
        parts.append(str(int(signal_ts)))
    return sanitize_remark("/".join(p for p in parts if p), fallback=prefix)


def marketable_limit(
    leg: OptionLeg,
    txn_type: str,
    *,
    slippage_ticks: int = DEFAULT_SLIPPAGE_TICKS,
    slip_pct: float = 0.0,
) -> float:
    """The limit price to send: through the touch by a few ticks, rounded to the tick.

    BUY prices off the ask and SELL off the bid, because the bot is taking liquidity
    — an entry resting at the mid is an entry that does not happen. The buffer is
    measured in TICKS rather than as a percentage: on a 2-rupee option a 1% buffer is
    2 paisa, less than one tick, i.e. no buffer at all. `slip_pct` is still honoured
    when configured, and the wider of the two wins.
    """
    tick = leg.tick_size if leg.tick_size and leg.tick_size > 0 else DEFAULT_TICK_SIZE
    side = txn_type.upper()
    if side == "BUY":
        base = leg.ask if leg.ask > 0 else (leg.ltp if leg.ltp > 0 else leg.mid)
    else:
        base = leg.bid if leg.bid > 0 else (leg.ltp if leg.ltp > 0 else leg.mid)
    if base <= 0:
        raise OrderError(
            f"no usable price for {leg.label}: bid={leg.bid} ask={leg.ask} ltp={leg.ltp}; "
            "refusing to send a limit order without a price",
            payload=leg.as_dict(),
            security_id=leg.security_id,
            txn_type=side,
        )
    buffer = max(float(slippage_ticks) * tick, abs(base * float(slip_pct)))
    raw = base + buffer if side == "BUY" else base - buffer
    return round_to_tick(raw, tick, side=side)


def _cfg_value(cfg: Any, names: tuple[str, ...], default: Any) -> Any:
    """First present attribute from `names` on cfg, else `default`.

    CONTRACTS.md does not pin the names of the paper-fill tunables (config.py ships
    `slippage_ticks` and `cost_per_order`), so the broker asks for a few spellings
    and falls back to its own default rather than crashing on a missing attribute.
    """
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return value
    return default


def _unwrap(resp: Any) -> dict[str, Any]:
    """Return the `data` block of an API response, accepting an already-unwrapped dict."""
    if not isinstance(resp, dict):
        return {}
    data = resp.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return resp


def _pick_order_id(payload: dict[str, Any]) -> str:
    """Pull the order id out of a place/modify/details payload."""
    for key in ("order_id", "id", "orderId"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _pick_status(payload: dict[str, Any]) -> str:
    """Pull the ORDER status, never the response envelope's `status: success`."""
    value = payload.get("order_status")
    if value:
        return normalise_status(value)
    value = normalise_status(payload.get("status"))
    if value == "SUCCESS" and "order_status" not in payload and "traded_qty" not in payload:
        # An envelope leaked through: `{"status": "success", "data": {...}}` with no
        # order fields alongside it. Treating that as a fill would report a position
        # we do not have.
        return ""
    if value in _ENVELOPE_WORDS:
        return ""
    return value


# ---------------------------------------------------------------------------
# structural guard for the paper path
# ---------------------------------------------------------------------------
class QuoteOnlyClient:
    """A read-only view of INDstocksClient. Every order method raises.

    PaperBroker holds one of these instead of the real client, so "the paper broker
    never places an order" is enforced by the object graph rather than by reviewer
    discipline. Reaching for `place_order` here is an immediate, loud OrderError.
    """

    def __init__(self, client: "INDstocksClient | None") -> None:
        self._client = client

    @property
    def available(self) -> bool:
        """True when a real client is behind this facade."""
        return self._client is not None

    def ltp(self, scrip_codes: list[str]) -> dict[str, float]:
        """Forward the LTP read, the only market data the paper broker needs."""
        if self._client is None:
            raise DataError("no API client attached to the paper broker")
        return self._client.ltp(scrip_codes)

    def quotes(self, scrip_codes: list[str]) -> dict[str, dict]:
        """Forward the full-quote read (bid/ask refresh)."""
        if self._client is None:
            raise DataError("no API client attached to the paper broker")
        return self._client.quotes(scrip_codes)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            # copy/pickle/repr protocol probes must stay AttributeError, or every
            # generic tool that touches this object blows up with an OrderError.
            raise AttributeError(name)
        raise OrderError(
            f"PaperBroker tried to call client.{name}(); the paper path may only read "
            "quotes. If you need this, you are writing LiveBroker.",
            attempted=name,
        )


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class BaseBroker(ABC):
    """What the engine is allowed to ask a broker to do.

    Four methods, two of which move money. Validation that must hold in both paper
    and live (lot multiples, lot caps, remark hygiene, session hours, in-session
    deduplication) lives here, so paper is a faithful rehearsal of live rather than
    a laxer path.
    """

    mode: str = "base"

    def __init__(
        self,
        cfg: "Config",
        client: "INDstocksClient | None" = None,
        *,
        allow_closed_market: bool = False,
    ) -> None:
        self.cfg = cfg
        self._allow_closed_market = allow_closed_market
        self._lock = threading.RLock()
        self._sent_keys: set[tuple[str, str, str]] = set()
        self._ids = itertools.count(1)

    # -- shared tunables ---------------------------------------------------
    @property
    def slippage_ticks(self) -> int:
        """Ticks through the touch for a marketable limit / a simulated fill."""
        return int(_num(_cfg_value(self.cfg, ("slippage_ticks",), DEFAULT_SLIPPAGE_TICKS),
                        DEFAULT_SLIPPAGE_TICKS))

    @property
    def slip_pct(self) -> float:
        """Optional fractional price buffer, 0.0 unless configured (ticks dominate)."""
        return float(_num(_cfg_value(self.cfg, ("limit_slip_pct", "slip_pct"), 0.0), 0.0))

    @property
    def per_order_cost(self) -> float:
        """Flat rupee cost charged per order (brokerage + taxes, approximated)."""
        return float(
            _num(
                _cfg_value(
                    self.cfg,
                    ("cost_per_order", "per_order_cost", "order_cost", "brokerage_per_order"),
                    DEFAULT_PER_ORDER_COST,
                ),
                DEFAULT_PER_ORDER_COST,
            )
        )

    @property
    def fill_timeout_s(self) -> float:
        """How long to wait for a terminal order status before cancelling."""
        return float(
            _num(
                _cfg_value(self.cfg, ("order_fill_timeout_s", "fill_timeout_s"),
                           DEFAULT_FILL_TIMEOUT_S),
                DEFAULT_FILL_TIMEOUT_S,
            )
        )

    # -- validation shared by both brokers ---------------------------------
    def _validate(self, leg: OptionLeg, qty: int, txn_type: str) -> None:
        """Reject anything the exchange would reject, plus a wrong position size.

        QtyMultipleOfLotSize is an exchange validation, but the dangerous case is the
        one that passes it: a qty that is a clean multiple and still not the size we
        intended. The lot cap is therefore enforced on entries too.
        """
        side = txn_type.upper()
        if not isinstance(qty, int) or isinstance(qty, bool):
            raise OrderError(f"qty must be an int number of UNITS, got {qty!r}",
                             security_id=leg.security_id, txn_type=side)
        if qty <= 0:
            raise OrderError(f"QtyMustBeAboveZero: refusing a {side} of {qty} units of {leg.label}",
                             security_id=leg.security_id, txn_type=side)
        lot = int(leg.lot_size or 0)
        if lot <= 0:
            # Without a lot size we cannot verify the multiple, and an unverified qty
            # on a derivative order is a position of unknown size. Refuse.
            raise OrderError(
                f"{leg.label} has no lot_size; cannot verify that {qty} units is a whole "
                "number of lots",
                security_id=leg.security_id,
                txn_type=side,
            )
        if qty % lot != 0:
            raise OrderError(
                f"QtyMultipleOfLotSize: {qty} units is not a whole multiple of the {lot}-unit "
                f"lot for {leg.label} ({qty / lot:.3f} lots)",
                security_id=leg.security_id,
                txn_type=side,
                qty=qty,
                lot_size=lot,
            )
        if side == "BUY":
            max_lots = int(_num(_cfg_value(self.cfg, ("max_lots",), 1), 1))
            lots = qty // lot
            if max_lots > 0 and lots > max_lots:
                # Entries only. An exit must always be allowed to sell whatever is
                # actually held, even if that is more than the current cap.
                raise OrderError(
                    f"refusing to BUY {lots} lots of {leg.label}: max_lots is {max_lots}",
                    security_id=leg.security_id,
                    txn_type=side,
                    qty=qty,
                )

    def _check_session(self, txn_type: str, leg: OptionLeg) -> None:
        """Refuse an ENTRY outside market hours; warn loudly on an exit.

        An INTRADAY buy sent while the market is shut either bounces or queues as
        O-PENDING and fills unattended at the next open — a position nobody is
        watching. An exit is the opposite case: if the engine believes it must get
        out, let the broker be the one to say no.
        """
        if self._allow_closed_market:
            return
        if clock.is_market_open():
            return
        now_txt = clock.now().isoformat()
        if txn_type.upper() == "BUY":
            raise OrderError(
                f"market is closed at {now_txt}; refusing to open a position in {leg.label}",
                security_id=leg.security_id,
                txn_type="BUY",
            )
        log.critical(
            "EXIT requested for %s while the market is closed (%s) — sending anyway, but it "
            "will not fill until the next session",
            leg.label,
            now_txt,
        )

    # -- in-session deduplication ------------------------------------------
    def _dedupe_key(self, leg: OptionLeg, txn_type: str, signal_ts: int | None,
                    remark: str) -> tuple[str, str, str]:
        """(security_id, BUY/SELL, signal_ts or remark) — the key we refuse twice."""
        discriminator = str(int(signal_ts)) if signal_ts else remark
        return (str(leg.security_id), txn_type.upper(), discriminator)

    def _claim(self, key: tuple[str, str, str], *, enforce: bool) -> bool:
        """Reserve a dedupe key before sending. False when it was already claimed.

        Claimed BEFORE the request, not after, because the duplicate we are guarding
        against is a second call made while the first is still in flight.
        """
        with self._lock:
            if key in self._sent_keys:
                return not enforce
            self._sent_keys.add(key)
            return True

    def _release(self, key: tuple[str, str, str]) -> None:
        """Give a key back after an order that definitively did not fill at all.

        A nothing-filled order left nothing behind, so a later retry of the same
        signal is legitimate. A partial fill keeps its key: something is on the book.
        """
        with self._lock:
            self._sent_keys.discard(key)

    def sent_keys(self) -> list[tuple[str, str, str]]:
        """Everything this session has already sent — for the UI and for tests."""
        with self._lock:
            return sorted(self._sent_keys)

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def buy(self, leg: OptionLeg, qty: int, *, remark: str,
            signal_ts: int | None = None) -> dict[str, Any]:
        """Buy `qty` UNITS of `leg`. Returns {"order_id", "status", "price", ...}."""

    @abstractmethod
    def sell(self, leg: OptionLeg, qty: int, *, remark: str,
             signal_ts: int | None = None) -> dict[str, Any]:
        """Sell `qty` UNITS of `leg` to close. Returns {"order_id", "status", "price", ...}."""

    @abstractmethod
    def ltp(self, leg: OptionLeg) -> float:
        """Last traded premium per unit for `leg`."""

    @abstractmethod
    def open_positions(self) -> list[dict[str, Any]]:
        """Open positions this broker knows about."""

    # -- shared plumbing ----------------------------------------------------
    def _scrip_code(self, leg: OptionLeg) -> str:
        """`NSE_56998` — the scrip-code format the quote endpoints take."""
        return f"{leg.exchange}_{leg.security_id}"

    def _read_ltp(self, leg: OptionLeg, client: Any) -> float:
        """Read the LTP through `client`, falling back to the leg's own snapshot."""
        code = self._scrip_code(leg)
        try:
            data = client.ltp([code]) or {}
        except (ApiError, DataError) as exc:
            if leg.ltp > 0:
                log.warning("LTP lookup failed for %s (%s); using the leg snapshot %.2f",
                            leg.label, exc, leg.ltp)
                return float(leg.ltp)
            raise
        value = data.get(code)
        if value is None:
            suffix = f"_{leg.security_id}"
            for key, candidate in data.items():
                if str(key).endswith(suffix):
                    value = candidate
                    break
        if value is None and len(data) == 1:
            value = next(iter(data.values()))
        price = _num(value, -1.0)
        if price <= 0:
            # A zero mark is treated as missing, not as a worthless option: marking a
            # live position at 0 shows a 100% loss and would trip the stop instantly.
            if leg.ltp > 0:
                log.warning("no LTP for %s in %r; using the leg snapshot %.2f",
                            code, sorted(data), leg.ltp)
                return float(leg.ltp)
            raise DataError(f"no LTP available for {leg.label} ({code})",
                            security_id=leg.security_id)
        return price

    def _result(
        self,
        *,
        order_id: str,
        status: str,
        price: float,
        leg: OptionLeg,
        qty: int,
        filled_qty: float,
        txn_type: str,
        remark: str,
        limit_price: float,
        costs: float = 0.0,
        partial: bool = False,
        raw: Any = None,
    ) -> dict[str, Any]:
        """The dict both brokers return: the contract's three keys plus the audit trail."""
        return {
            "order_id": order_id,
            "status": status,
            "price": round(float(price), 4),
            # Everything below is additive. `filled_qty` matters: when `partial` is
            # true the position is SMALLER than `qty` and must be sized from it.
            "avg_price": round(float(price), 4),
            "qty": int(qty),
            "filled_qty": int(filled_qty),
            "partial": bool(partial),
            "txn_type": txn_type.upper(),
            "mode": self.mode,
            "security_id": leg.security_id,
            "trading_symbol": leg.trading_symbol,
            "label": leg.label,
            "limit_price": round(float(limit_price), 4) if limit_price else None,
            "remarks": remark,
            "costs": round(float(costs), 2),
            "ts": clock.epoch_s(clock.now()),
            "raw": raw,
        }


# ---------------------------------------------------------------------------
# paper
# ---------------------------------------------------------------------------
class PaperBroker(BaseBroker):
    """Simulated fills against the live book. Cannot reach POST /order.

    A BUY fills at the ask plus `slippage_ticks`, a SELL at the bid minus the same,
    each rounded to the tick in the direction that costs us money — so paper P&L is
    pessimistic by exactly the amount live execution usually is. A flat per-order
    cost is reported in `costs` for the engine to charge to the position; it is not
    folded into the fill price, because a corrupted price breaks stop/target maths.
    """

    mode = "paper"

    def __init__(
        self,
        cfg: "Config",
        client: "INDstocksClient | None" = None,
        *,
        allow_closed_market: bool = False,
    ) -> None:
        super().__init__(cfg, client, allow_closed_market=allow_closed_market)
        # The real client is wrapped away and never stored unwrapped: there is no
        # reference on this object through which an order endpoint can be reached.
        self._client = QuoteOnlyClient(client)
        self._book: dict[str, dict[str, Any]] = {}

    # -- fills -------------------------------------------------------------
    def _fill_price(self, leg: OptionLeg, txn_type: str) -> float:
        """Where a taker order would actually fill, in this book, right now."""
        tick = leg.tick_size if leg.tick_size and leg.tick_size > 0 else DEFAULT_TICK_SIZE
        side = txn_type.upper()
        if side == "BUY":
            base = leg.ask if leg.ask > 0 else (leg.ltp if leg.ltp > 0 else leg.mid)
        else:
            base = leg.bid if leg.bid > 0 else (leg.ltp if leg.ltp > 0 else leg.mid)
        if base <= 0:
            # Paper must fail where live would fail, or the backtest lies.
            raise OrderError(
                f"no price to simulate a {side} fill for {leg.label}: "
                f"bid={leg.bid} ask={leg.ask} ltp={leg.ltp}",
                payload=leg.as_dict(),
                security_id=leg.security_id,
                txn_type=side,
            )
        buffer = self.slippage_ticks * tick
        raw = base + buffer if side == "BUY" else base - buffer
        return round_to_tick(raw, tick, side=side)

    def _simulate(self, leg: OptionLeg, qty: int, txn_type: str, *, remark: str,
                  signal_ts: int | None) -> dict[str, Any]:
        """Validate exactly as live does, then book a synthetic fill."""
        side = txn_type.upper()
        remark = sanitize_remark(remark, fallback=f"indbot-{self.mode}")
        self._validate(leg, qty, side)
        self._check_session(side, leg)
        key = self._dedupe_key(leg, side, signal_ts, remark)
        if not self._claim(key, enforce=(side == "BUY" or signal_ts is not None)):
            raise OrderError(
                f"duplicate {side} for {leg.label} already sent this session (key={key})",
                security_id=leg.security_id,
                txn_type=side,
                duplicate=True,
            )
        price = self._fill_price(leg, side)
        order_id = f"PAPER-{clock.epoch_ms(clock.now())}-{next(self._ids)}"
        self._apply_to_book(leg, qty, side, price)
        log.info("[paper] %s %d units of %s at %.2f (order %s, remark %s)",
                 side, qty, leg.label, price, order_id, remark)
        return self._result(
            order_id=order_id,
            status="SUCCESS",
            price=price,
            leg=leg,
            qty=qty,
            filled_qty=qty,
            txn_type=side,
            remark=remark,
            limit_price=price,
            costs=self.per_order_cost,
            raw={"simulated": True},
        )

    def _apply_to_book(self, leg: OptionLeg, qty: int, txn_type: str, price: float) -> None:
        """Keep the simulated position ledger in step with the simulated fills."""
        with self._lock:
            row = self._book.get(leg.security_id)
            if txn_type == "BUY":
                if row is None:
                    self._book[leg.security_id] = {
                        "security_id": leg.security_id,
                        "trading_symbol": leg.trading_symbol,
                        "label": leg.label,
                        "exchange": leg.exchange,
                        "lot_size": leg.lot_size,
                        "net_qty": qty,
                        "avg_price": price,
                        "opened_ts": clock.epoch_s(clock.now()),
                    }
                else:
                    total = row["net_qty"] + qty
                    if total > 0:
                        row["avg_price"] = (row["avg_price"] * row["net_qty"] + price * qty) / total
                    row["net_qty"] = total
                return
            if row is None:
                log.warning("[paper] SELL of %s with no simulated position on the book", leg.label)
                return
            row["net_qty"] -= qty
            if row["net_qty"] <= 0:
                if row["net_qty"] < 0:
                    log.warning(
                        "[paper] SELL took %s net short (%d units) — the engine sold "
                        "more than it holds", leg.label, row["net_qty"])
                self._book.pop(leg.security_id, None)

    # -- interface ----------------------------------------------------------
    def buy(self, leg: OptionLeg, qty: int, *, remark: str,
            signal_ts: int | None = None) -> dict[str, Any]:
        """Simulate buying `qty` UNITS at the ask plus slippage."""
        return self._simulate(leg, qty, "BUY", remark=remark, signal_ts=signal_ts)

    def sell(self, leg: OptionLeg, qty: int, *, remark: str,
             signal_ts: int | None = None) -> dict[str, Any]:
        """Simulate selling `qty` UNITS at the bid minus slippage."""
        return self._simulate(leg, qty, "SELL", remark=remark, signal_ts=signal_ts)

    def ltp(self, leg: OptionLeg) -> float:
        """Live LTP when a client is attached, else the leg's own snapshot."""
        if not self._client.available:
            return float(leg.ltp)
        return self._read_ltp(leg, self._client)

    def open_positions(self) -> list[dict[str, Any]]:
        """The simulated ledger — never the real account's positions."""
        with self._lock:
            return [dict(row) for row in self._book.values() if row["net_qty"] != 0]


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
class LiveBroker(BaseBroker):
    """Real orders on the real account. Refuses unless mode == "live" and armed.

    The arm check runs on every order call, not just at construction: the engine can
    be re-moded or disarmed from the UI while an order is being prepared, and this
    assert is the last thing between a disarmed bot and a real fill.
    """

    mode = "live"

    def __init__(
        self,
        cfg: "Config",
        client: "INDstocksClient | None" = None,
        *,
        allow_closed_market: bool = False,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__(cfg, client, allow_closed_market=allow_closed_market)
        if client is None:
            raise ConfigError("LiveBroker needs an INDstocksClient; refusing to construct one "
                              "that cannot reach the broker")
        self._client = client
        self._poll_interval_s = max(0.1, float(poll_interval_s))

    # -- the arm gate --------------------------------------------------------
    def _require_armed(self, txn_type: str, leg: OptionLeg) -> None:
        """Raise unless cfg.mode == "live" AND cfg.armed is True. Checked per call.

        `is True`, not truthiness: a config that round-tripped through JSON as the
        string "true" is a config we do not understand, and refusing to trade is the
        safe side of that mistake.
        """
        mode = str(getattr(self.cfg, "mode", "")).strip()
        armed = getattr(self.cfg, "armed", False)
        if mode != "live" or armed is not True:
            raise OrderError(
                f"LiveBroker refused a {txn_type.upper()} of {leg.label}: mode={mode!r}, "
                f"armed={armed!r} — live orders require mode 'live' and armed True",
                security_id=leg.security_id,
                txn_type=txn_type.upper(),
                mode=mode,
                armed=bool(armed is True),
            )

    # -- order plumbing ------------------------------------------------------
    def _body(self, leg: OptionLeg, qty: int, txn_type: str, limit_price: float,
              remark: str) -> dict[str, Any]:
        """The POST /order body, exactly as CONTRACTS.md specifies it."""
        return {
            "txn_type": txn_type.upper(),
            "exchange": leg.exchange,
            "segment": SEGMENT,
            "product": PRODUCT,
            "order_type": ORDER_TYPE,
            "validity": VALIDITY,
            "security_id": leg.security_id,
            "qty": int(qty),  # UNITS, a whole multiple of lot_size — asserted above
            "limit_price": limit_price,
            "is_amo": IS_AMO,
            "remarks": remark,
        }

    def _details(self, order_id: str, failures: list[int]) -> dict[str, Any]:
        """One `GET /order` read. Returns {} on a lookup failure rather than raising.

        GET /order answers 512 InternalServerException — not 404 — for an order it
        cannot fetch, including one that does not exist. A few of those right after
        placement are normal propagation delay; a persistent run of them is a real
        lookup failure, so they are counted and surfaced by the caller rather than
        ignored.
        """
        try:
            return _unwrap(self._client.order_details(order_id))
        except Exception as exc:  # noqa: BLE001 - a transport error must not escape mid-order
            failures.append(1)
            log.warning("order_details(%s) failed (%s: %s)", order_id, type(exc).__name__, exc)
            return {}

    def _order_book_row(self, order_id: str = "", remark: str = "") -> dict[str, Any] | None:
        """Find an order in the day's order book by id or by remark.

        Returns the row, `{}` when the book definitively has no such order, and
        `None` when the lookup itself failed — the caller treats "not there" and
        "could not tell" differently, because only the first one makes a resend safe.

        Used only on the failure paths: the docs are explicit that after a timeout or
        an error on placement you confirm the order's real state from the order book
        before concluding anything. Never retry a placement on a guess.
        """
        try:
            rows = self._client.order_book() or []
        except Exception as exc:  # noqa: BLE001 - reconciliation must report, not raise
            log.error("order_book() failed while reconciling order %s / remark %r (%s: %s)",
                      order_id or "-", remark, type(exc).__name__, exc)
            return None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if order_id and str(row.get("id") or "") == str(order_id):
                return row
            if remark and str(row.get("remarks") or "") == remark and not order_id:
                return row
        return {}

    def _fill_of(self, details: dict[str, Any]) -> tuple[float, float]:
        """(filled units, average fill price) from an order-details payload.

        `traded_price` is the broker's own average and is trusted first. When it is
        empty — which happens on a partial fill — the average is rebuilt from the
        actual fills, because a position priced from the LIMIT we asked for rather
        than the price we got reports a P&L that never happened.
        """
        filled = _num(details.get("traded_qty"))
        avg = _num(details.get("traded_price"))
        if filled > 0 and avg <= 0:
            avg = self._avg_from_fills(_pick_order_id(details))
        if filled > 0 and avg <= 0:
            avg = self._avg_from_trade_book(details)
        if filled > 0 and avg <= 0:
            avg = _num(details.get("requested_price"))
            if avg > 0:
                log.warning("no traded price for order %s; falling back to the requested price "
                            "%.2f — P&L for this trade is approximate",
                            _pick_order_id(details) or "?", avg)
        return filled, avg

    def _avg_from_fills(self, order_id: str) -> float:
        """Weighted average from GET /order/trades, when the client exposes it.

        Not promised by CONTRACTS.md, so it is probed rather than assumed; it is the
        most direct answer to "what did this order actually fill at".
        """
        fetch = getattr(self._client, "order_trades", None)
        if not order_id or not callable(fetch):
            return 0.0
        try:
            rows = fetch(order_id) or []
        except Exception as exc:  # noqa: BLE001 - a pricing fallback, never fatal
            log.warning("order_trades(%s) unavailable: %s", order_id, exc)
            return 0.0
        qty = 0.0
        value = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            q = _num(row.get("quantity"))
            value += q * _num(row.get("price"))
            qty += q
        return (value / qty) if qty > 0 else 0.0  # guarded division

    def _avg_from_trade_book(self, details: dict[str, Any]) -> float:
        """Weighted average fill price from the trade book, matched on exch_order_id."""
        exch_id = str(details.get("exch_order_id") or "")
        if not exch_id:
            return 0.0
        try:
            rows = self._client.trade_book() or []
        except Exception as exc:  # noqa: BLE001 - a pricing fallback, never fatal
            # Logged, not swallowed: the caller still has traded_price/requested_price
            # to fall back on, and the order outcome does not depend on this read.
            log.warning("trade_book() unavailable while pricing order %s: %s", exch_id, exc)
            return 0.0
        qty = 0.0
        value = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("exch_order_id") or "") != exch_id:
                continue
            q = _num(row.get("quantity"))
            value += q * _num(row.get("price"))
            qty += q
        return (value / qty) if qty > 0 else 0.0  # guarded division

    def _cancel(self, order_id: str, *, why: str) -> dict[str, Any]:
        """Cancel an order and report the outcome. Never raises; always logs."""
        try:
            resp = _unwrap(self._client.cancel_order(order_id))
            log.warning("cancelled order %s (%s): status=%s", order_id, why,
                        _pick_status(resp) or "?")
            return resp
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
            # A cancel can legitimately fail because the order just filled. The caller
            # re-reads the order before concluding anything, and this failure is
            # carried into the OrderError payload rather than dropped.
            log.error("cancel_order(%s) failed (%s, %s: %s)", order_id, why,
                      type(exc).__name__, exc)
            return {"cancel_error": str(exc), "status_code": getattr(exc, "status", None)}

    def _await_terminal(self, order_id: str, timeout_s: float) -> tuple[dict[str, Any], str, int]:
        """Poll `order_details` until a terminal status or the timeout. (details, status, failures).

        Wall-clock via time.monotonic, deliberately: clock.now() honours
        INDBOT_FAKE_NOW, and a frozen fake clock would make this loop never time out
        while a real order sits on the exchange.
        """
        deadline = time.monotonic() + max(1.0, timeout_s)
        failures: list[int] = []
        details: dict[str, Any] = {}
        status = ""
        while True:
            fresh = self._details(order_id, failures)
            if fresh:
                details = fresh
                status = _pick_status(fresh)
                if status and status in TERMINAL_STATUSES:
                    break
                if status and status not in STATUS_PENDING:
                    # Undocumented status: keep polling (the safe branch — we do not
                    # know that it is terminal) but say so loudly.
                    log.warning("order %s reported an undocumented status %r", order_id, status)
            if len(failures) >= MAX_LOOKUP_FAILURES:
                log.error("order_details(%s) failed %d times; falling back to the order book",
                          order_id, len(failures))
                row = self._order_book_row(order_id=order_id) or {}
                if row:
                    details = row
                    status = _pick_status(row)
                    if status and status in TERMINAL_STATUSES:
                        break
                failures.clear()
            if time.monotonic() >= deadline:
                log.warning("order %s did not reach a terminal status within %.1fs (last=%r)",
                            order_id, timeout_s, status or "unknown")
                break
            time.sleep(self._poll_interval_s)
        return details, status, len(failures)

    def _place(self, leg: OptionLeg, qty: int, txn_type: str, *, remark: str,
               signal_ts: int | None) -> dict[str, Any]:
        """Validate, send one order, wait for the fill, return the average price."""
        side = txn_type.upper()
        is_exit = side == "SELL"
        remark = sanitize_remark(remark, fallback="indbot")

        self._require_armed(side, leg)          # checked again just before the POST
        self._validate(leg, qty, side)
        self._check_session(side, leg)

        key = self._dedupe_key(leg, side, signal_ts, remark)
        # An exit is deduped only on an explicit signal_ts. Refusing a repeated BUY
        # costs an opportunity; refusing a repeated SELL can strand a live position,
        # so the exit path only blocks a duplicate we are certain about.
        if not self._claim(key, enforce=(side == "BUY" or signal_ts is not None)):
            raise OrderError(
                f"duplicate {side} for {leg.label} already sent this session (key={key}); "
                "remarks are not an idempotency key, so this is deduped in-process",
                security_id=leg.security_id,
                txn_type=side,
                duplicate=True,
            )

        limit_price = marketable_limit(
            leg, side, slippage_ticks=self.slippage_ticks, slip_pct=self.slip_pct
        )
        body = self._body(leg, qty, side, limit_price, remark)

        self._require_armed(side, leg)  # last line of defence before money moves
        log.info("[live] placing %s %d units of %s limit %.2f (remark %s)",
                 side, qty, leg.label, limit_price, remark)
        try:
            resp = self._client.place_order(**body)
        except Exception as exc:  # noqa: BLE001 - including a transport timeout
            # The call failed, but the order may still exist — a read timeout after
            # the exchange accepted it looks exactly like a rejection from here, which
            # is why nothing below resends and everything below reconciles. The docs
            # are explicit: confirm the real state from the order book rather than
            # resending on a guess. `remarks` is echoed there, which is what makes
            # the orphan findable at all.
            row = self._order_book_row(remark=remark)
            if row == {} or (is_exit and row is None):
                # Entry: release only when the book positively shows nothing, so an
                # ambiguous lookup can never become a double entry.
                # Exit: release even when the lookup failed — a retryable exit beats
                # one the dedupe set has locked out, and the payload below tells the
                # operator to verify before anything automated tries again.
                self._release(key)
            level = log.critical if is_exit else log.error
            level("[live] %s of %s REJECTED by INDstocks: %s (order book shows %s)",
                  side, leg.label, exc,
                  (row or {}).get("id") if row is not None else "LOOKUP FAILED")
            raise OrderError(
                f"{side} of {qty} units of {leg.label} was rejected: {exc}",
                payload={"body": body, "error_type": type(exc).__name__,
                         "api_status": getattr(exc, "status", None),
                         "api_payload": getattr(exc, "payload", None),
                         "order_book_row": row, "order_book_lookup_ok": row is not None},
                security_id=leg.security_id,
                trading_symbol=leg.trading_symbol,
                txn_type=side,
                qty=qty,
                stranded=is_exit,
            ) from exc

        data = _unwrap(resp)
        order_id = _pick_order_id(data)
        if not order_id:
            # The order may exist without us knowing its id. Do NOT resend — look it
            # up by remark, which the docs guarantee is echoed on the order book.
            row = self._order_book_row(remark=remark)
            log.critical("[live] %s of %s returned no order_id; order book shows %s",
                         side, leg.label,
                         (row or {}).get("id") if row is not None else "LOOKUP FAILED")
            raise OrderError(
                f"{side} of {leg.label} returned no order_id — verify by hand before resending",
                payload={"body": body, "response": resp, "order_book_row": row,
                         "order_book_lookup_ok": row is not None},
                security_id=leg.security_id,
                txn_type=side,
                qty=qty,
                stranded=is_exit,
            )

        details, status, _ = self._await_terminal(order_id, self.fill_timeout_s)
        filled, avg = self._fill_of(details)

        cancel_resp: dict[str, Any] | None = None

        if status in STATUS_FILLED and filled >= qty and avg > 0:
            log.info("[live] %s %s filled %d units at %.2f (order %s)",
                     side, leg.label, int(filled), avg, order_id)
            return self._result(order_id=order_id, status=status, price=avg, leg=leg, qty=qty,
                                filled_qty=filled, txn_type=side, remark=remark,
                                limit_price=limit_price, raw=details)

        if filled <= 0 and status not in STATUS_DEAD:
            # Nothing filled and the order is still live (or its state is unknown).
            # Cancel, then RE-READ: the cancel races the exchange and an order can
            # fill in that window. Reporting "no fill" on a filled order would leave
            # an untracked live position. An already-dead order is left alone —
            # cancelling it just earns an OrderCannotBeCancelled error.
            cancel_resp = self._cancel(order_id, why="no fill within the timeout")
            recheck = self._details(order_id, [])
            if recheck:
                details = recheck
                status = _pick_status(recheck) or status
                filled, avg = self._fill_of(recheck)

        if filled >= qty and avg > 0:
            log.warning("[live] %s %s filled during cancellation at %.2f (order %s)",
                        side, leg.label, avg, order_id)
            return self._result(order_id=order_id, status=status or "SUCCESS", price=avg, leg=leg,
                                qty=qty, filled_qty=filled, txn_type=side, remark=remark,
                                limit_price=limit_price, raw=details)

        payload = {
            "body": body,
            "cancel": cancel_resp,
            "order_id": order_id,
            "status": status or "unknown",
            "details": details or None,
            "filled_qty": int(filled),
            "requested_qty": int(qty),
            "avg_price": round(avg, 4) if avg else None,
            "limit_price": limit_price,
            "remarks": remark,
            "at": clock.now().isoformat(),
        }

        if filled > 0:
            # Partial fill. Real units changed hands, so this is NOT a clean failure.
            if not is_exit:
                self._cancel(order_id, why="partial fill on an entry")
                log.critical(
                    "[live] PARTIAL ENTRY: %d of %d units of %s filled at %.2f (order %s). The "
                    "position is smaller than intended — size the exit from filled_qty.",
                    int(filled), qty, leg.label, avg, order_id,
                )
                return self._result(order_id=order_id, status=status or "PARTIALLY FILLED",
                                    price=avg, leg=leg, qty=qty, filled_qty=filled, txn_type=side,
                                    remark=remark, limit_price=limit_price, partial=True,
                                    raw=details)
            payload["cancel"] = self._cancel(order_id, why="partial fill on an exit")
            remaining = int(qty - filled)
            log.critical(
                "[live] PARTIAL EXIT — %d of %d units of %s sold at %.2f, %d UNITS STILL OPEN "
                "(security_id=%s, order=%s). Square this off by hand.",
                int(filled), qty, leg.label, avg, remaining, leg.security_id, order_id,
            )
            raise OrderError(
                f"EXIT PARTIALLY FILLED: sold {int(filled)} of {qty} units of {leg.label} at "
                f"{avg:.2f}; {remaining} units (security_id={leg.security_id}) are STILL OPEN — "
                f"square off by hand. order_id={order_id}",
                payload=payload,
                security_id=leg.security_id,
                trading_symbol=leg.trading_symbol,
                txn_type=side,
                qty=qty,
                filled_qty=int(filled),
                remaining_qty=remaining,
                order_id=order_id,
                stranded=True,
            )

        # Nothing filled at all.
        if not is_exit:
            self._release(key)
            log.error("[live] entry %s of %s did not fill (status=%s); order %s cancelled",
                      side, leg.label, status or "unknown", order_id)
            raise OrderError(
                f"BUY of {qty} units of {leg.label} did not fill within "
                f"{self.fill_timeout_s:.0f}s (status={status or 'unknown'}); order {order_id} "
                "was cancelled",
                payload=payload,
                security_id=leg.security_id,
                txn_type=side,
                qty=qty,
                order_id=order_id,
                stranded=False,
            )

        log.critical(
            "[live] EXIT FAILED — %s %d units of %s (security_id=%s) did not fill (status=%s); "
            "order %s cancelled. THE POSITION IS STILL OPEN. Square it off by hand.",
            side, qty, leg.label, leg.security_id, status or "unknown", order_id,
        )
        raise OrderError(
            f"EXIT FAILED: {leg.label} ({leg.trading_symbol}, security_id={leg.security_id}) "
            f"{qty} units did not sell (status={status or 'unknown'}, order_id={order_id}, "
            f"limit {limit_price:.2f}). THE POSITION IS STILL OPEN — square off by hand.",
            payload=payload,
            security_id=leg.security_id,
            trading_symbol=leg.trading_symbol,
            txn_type=side,
            qty=qty,
            filled_qty=0,
            remaining_qty=int(qty),
            order_id=order_id,
            stranded=True,
        )

    # -- interface ----------------------------------------------------------
    def buy(self, leg: OptionLeg, qty: int, *, remark: str,
            signal_ts: int | None = None) -> dict[str, Any]:
        """Buy `qty` UNITS of `leg` with a marketable limit, and wait for the fill."""
        return self._place(leg, qty, "BUY", remark=remark, signal_ts=signal_ts)

    def sell(self, leg: OptionLeg, qty: int, *, remark: str,
             signal_ts: int | None = None) -> dict[str, Any]:
        """Sell `qty` UNITS of `leg` to close. Raises loudly rather than stranding it."""
        return self._place(leg, qty, "SELL", remark=remark, signal_ts=signal_ts)

    def ltp(self, leg: OptionLeg) -> float:
        """Last traded premium per unit, from /market/quotes/ltp."""
        return self._read_ltp(leg, self._client)

    def open_positions(self) -> list[dict[str, Any]]:
        """Every open DERIVATIVE position on the account, whatever its product.

        Unfiltered by product on purpose: this read is how a stranded position gets
        noticed, and a filter that hides one is worse than a row the UI ignores. It
        also does NOT apply the arm check — disarming must never blind the operator
        to a position that is still open.
        """
        mode = str(getattr(self.cfg, "mode", "")).strip()
        if mode != "live":
            log.warning("reading live positions while mode is %r", mode)
        rows = self._client.positions(segment="derivative") or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            net = _num(row.get("net_qty"))
            if net == 0:
                continue
            enriched = dict(row)
            enriched.update(
                security_id=str(row.get("security_id") or ""),
                net_qty=int(net),
                avg_price=_num(row.get("avg_price")),
                side=("LONG" if net > 0 else "SHORT"),
                label=" ".join(
                    str(part)
                    for part in (
                        row.get("symbol"),
                        int(_num(row.get("drv_strike_price"))) or "",
                        row.get("drv_option_type"),
                    )
                    if part
                ).strip(),
                source="broker",
            )
            out.append(enriched)
        return out


# ---------------------------------------------------------------------------
def make_broker(
    cfg: "Config",
    client: "INDstocksClient | None" = None,
    **kwargs: Any,
) -> BaseBroker:
    """The broker for `cfg.mode`: LiveBroker only for "live", PaperBroker otherwise.

    Anything that is not exactly "live" gets the simulated broker. An unrecognised
    mode falling through to paper is a wasted signal; falling through to live would
    be a real trade nobody asked for.
    """
    mode = str(getattr(cfg, "mode", "")).strip()
    if mode == "live":
        return LiveBroker(cfg, client, **kwargs)
    return PaperBroker(cfg, client, **kwargs)
