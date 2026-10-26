"""Strike selection — turns a Signal's direction into the one contract to buy.

The option chain gives the full strike ladder with top-of-book, OI, volume, IV and
greeks in a single call. This module normalises that into `OptionLeg`s and then
applies the six filters in CONTRACTS.md section F, in order, recording why every
leg it drops was dropped.

THE ARITHMETIC THAT DOMINATES THIS FILE
---------------------------------------
The user's limit is a maximum loss of ``max_trade_loss`` rupees per trade and a
minimum size of one lot. A long option's loss at a ``stop_pct`` stop is
``premium * lot_size * stop_pct``, so the premium we are allowed to pay is::

    premium_per_unit <= max_trade_loss / (stop_pct * lot_size)

At the defaults (₹1,000, 30%, NIFTY's 75) that is ₹44.44 a unit, while an ATM
NIFTY option trades at ₹150–200. **Filter 2 therefore rejects every at-the-money
strike, and that is the correct answer, not a bug.** This module does not widen the
stop, does not quietly raise the budget and does not drift to 0.05-delta lottery
strikes (filter 3's delta floor is what stops that). It returns ``None`` and spells
the arithmetic out in the first rejection reason, with the two honest choices the
user actually has: raise the per-trade limit, or accept a tighter stop and its
whipsaw cost.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time as _time
from datetime import date

from . import clock
from .config import UNDERLYINGS
from .errors import DataError, IndbotError
from .types import OptionLeg

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, these are type-only
    from .api import INDstocksClient
    from .config import Config
    from .instruments import Instruments

log = logging.getLogger(__name__)

#: Strikes per side of the money, per the option-chain endpoint's `strike_count`.
#: The response holds (2 * strike_count) + 1 strikes.
DEFAULT_STRIKE_COUNT = 10

#: How long a fetched chain may be reused. Seconds, deliberately short: this is the
#: price we size a real order against, and a stale premium is a wrong stop.
CHAIN_TTL_SECONDS = 5.0

#: Used only when the instruments file has no tick size for a contract. Every NSE/BSE
#: index option ticks at 5 paise, so this is a safe floor rather than a guess.
DEFAULT_TICK_SIZE = 0.05

#: The delta we rank towards — a roughly at-the-money option.
ATM_DELTA = 0.5

#: Warn when the caller's spot and the chain's underlying_ltp disagree by more than
#: this percentage; one of the two is stale.
SPOT_DISAGREEMENT_PCT = 1.0

#: One paise of slack on every money comparison, so float noise never rejects a leg
#: that is exactly at the cap. A paise cannot move the risk arithmetic.
MONEY_EPS = 0.01

#: Slack on dimensionless comparisons (delta, percentages).
RATIO_EPS = 1e-9

_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# coercion and formatting helpers
# --------------------------------------------------------------------------- #

def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce an API field to a finite float, falling back to `default`.

    NaN and infinity are treated as missing: either would sail through a `<=`
    comparison in a filter and let an unpriced leg reach an order.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: object, default: int = 0) -> int:
    """Coerce an API field to an int, falling back to `default`."""
    out = _as_float(value, float(default))
    try:
        return int(out)
    except (ValueError, OverflowError):
        return default


def _group(whole: int) -> str:
    """Indian digit grouping: 1234567 -> '12,34,567' (last three, then pairs)."""
    digits = str(abs(int(whole)))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _inr(value: float, dp: int = 0) -> str:
    """Format rupees the way the UI does — '₹12,593' or '₹167.90'.

    Rounds half UP rather than to even, so ₹12,592.50 does not print as ₹12,592 and
    make the arithmetic in a rejection reason fail to add up on the reader's calculator.
    """
    scale = 10 ** dp
    units = int(math.floor(abs(float(value)) * scale + 0.5))
    whole, frac = divmod(units, scale)
    text = _group(whole)
    if dp > 0:
        text = f"{text}.{frac:0{dp}d}"
    return ("-₹" if value < 0 else "₹") + text


def _qty(value: int) -> str:
    """Format a contract count with Indian grouping — '15,78,940 lots of OI'."""
    return _group(int(value))


def _pct(fraction: float, dp: int = 1) -> str:
    """Format a fraction as a percentage, dropping a pointless '.0' — 0.3 -> '30%'."""
    text = f"{fraction * 100:.{dp}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text or '0'}%"


def _a_pct(fraction: float, dp: int = 1) -> str:
    """A percentage with its English article — 'a 30%', 'an 11.3%', 'an 8%'.

    These strings end up in sentences the user reads in the UI, and "a 11% stop"
    reads like a bug in the sentence.
    """
    text = _pct(fraction, dp)
    whole = text.split(".")[0].rstrip("%")
    article = "an" if (whole.startswith("8") or whole in ("11", "18")) else "a"
    return f"{article} {text}"


def _parse_expiry(value: object) -> date | None:
    """Parse a 'YYYY-MM-DD' expiry, or None when it is not one."""
    text = str(value or "").strip()
    if not _EXPIRY_RE.match(text):
        return None
    try:
        y, m, d = (int(part) for part in text.split("-"))
        return date(y, m, d)
    except ValueError:
        return None


def is_tradable(leg: OptionLeg) -> bool:
    """True when a leg has a real price and a real two-sided book (filter 6).

    A leg with no bid cannot be exited, and a leg with no ask cannot be entered, so
    a one-sided book is treated as untradable rather than as a bargain.
    """
    return leg.ltp > 0 and leg.bid > 0 and leg.ask > 0


# --------------------------------------------------------------------------- #
# the picker
# --------------------------------------------------------------------------- #

class ChainPicker:
    """Builds the priced strike ladder for an underlying and picks the leg to buy."""

    def __init__(
        self,
        client: "INDstocksClient",
        instruments: "Instruments",
        cfg: "Config",
        *,
        cache_ttl: float = CHAIN_TTL_SECONDS,
    ) -> None:
        self.client = client
        self.instruments = instruments
        self.cfg = cfg
        self._cache_ttl = max(0.0, _as_float(cache_ttl, CHAIN_TTL_SECONDS))
        # The server thread can call pick() while the engine thread is mid-fetch.
        self._lock = threading.RLock()
        self._expiry_cache: dict[tuple[str, str], str] = {}
        self._scrip_cache: dict[str, str] = {}
        self._chain_cache: dict[tuple[str, str, int], tuple[float, float, list[OptionLeg]]] = {}

    # ---------------------------------------------------------------- metadata

    def _meta(self, underlying: str) -> dict:
        """Static reference data for an index, or DataError for an unknown one."""
        try:
            return UNDERLYINGS[underlying]
        except KeyError as exc:
            raise DataError(
                f"{underlying!r} is not one of the configured underlyings "
                f"({', '.join(sorted(UNDERLYINGS))})",
                underlying=underlying,
            ) from exc

    def underlying_scrip(self, underlying: str) -> str:
        """The underlying's own SECURITY_ID, for the chain's `underlying-scrip`.

        The constants in config are guesses for everything but NIFTY, so the live
        index file wins and a disagreement is logged rather than trusted. Passing an
        option contract's id here is rejected by the API, so this must be the index's.
        """
        with self._lock:
            hit = self._scrip_cache.get(underlying)
        if hit:
            return hit

        meta = self._meta(underlying)
        constant = str(meta.get("scrip") or "").strip()
        label = str(meta.get("label") or underlying)

        resolved = ""
        try:
            resolved = str(self.instruments.index_scrip(label) or "").strip()
        except IndbotError as exc:
            log.warning(
                "index_scrip(%r) failed (%s); falling back to the configured scrip %r",
                label, exc, constant,
            )
        except Exception:  # noqa: BLE001 - logged with a traceback, never swallowed
            log.exception("index_scrip(%r) raised; falling back to the configured scrip %r",
                          label, constant)

        if resolved and constant and resolved != constant:
            log.warning(
                "scrip for %s: config says %s, the instruments file says %s — using the file",
                underlying, constant, resolved,
            )

        scrip = resolved or constant
        if not scrip:
            raise DataError(
                f"could not resolve an underlying scrip for {underlying} "
                f"(label {label!r} is not in the index instruments file)",
                underlying=underlying,
            )
        with self._lock:
            self._scrip_cache[underlying] = scrip
        return scrip

    def nearest_expiry(self, underlying: str) -> str:
        """The first listed expiry on or after today, cached per underlying per day.

        The endpoint documents ascending, upcoming-only dates, but we sort anyway:
        silently trading a far expiry instead of the near one is a different premium,
        a different delta and a different trade.
        """
        day = clock.today()
        key = (underlying, day.isoformat())
        with self._lock:
            hit = self._expiry_cache.get(key)
        if hit:
            return hit

        listed = self.client.expiries(underlying)
        if not listed:
            raise DataError(
                f"/market/instruments/expiries returned no expiries for {underlying}",
                underlying=underlying,
            )

        dated: list[tuple[date, str]] = []
        for raw in listed:
            parsed = _parse_expiry(raw)
            if parsed is None:
                log.warning("ignoring malformed expiry %r for %s", raw, underlying)
                continue
            dated.append((parsed, str(raw).strip()))
        if not dated:
            raise DataError(
                f"no parseable expiry dates for {underlying} in {listed!r}",
                underlying=underlying,
            )

        dated.sort(key=lambda pair: pair[0])
        upcoming = [pair for pair in dated if pair[0] >= day]
        if not upcoming:
            raise DataError(
                f"every expiry listed for {underlying} is in the past "
                f"(latest {dated[-1][1]}, today {day.isoformat()})",
                underlying=underlying,
            )

        chosen_date, chosen = upcoming[0]
        if chosen_date == day:
            # Not blocked here — the contract says "first expiry >= today" — but the
            # user should see it: expiry-day gamma and theta are not the same trade.
            log.warning("%s nearest expiry %s is TODAY — expiry-day premiums decay fast",
                        underlying, chosen)
        with self._lock:
            # Drop other days' entries so the cache cannot grow across a long run.
            self._expiry_cache = {k: v for k, v in self._expiry_cache.items() if k[1] == key[1]}
            self._expiry_cache[key] = chosen
        log.debug("%s nearest expiry: %s (from %d listed)", underlying, chosen, len(dated))
        return chosen

    # ------------------------------------------------------------------- fetch

    def fetch(
        self,
        underlying: str,
        expiry: str | None = None,
        strike_count: int = DEFAULT_STRIKE_COUNT,
    ) -> tuple[float, list[OptionLeg]]:
        """Fetch one expiry's ladder as `(underlying_ltp, legs)`, legs in strike order.

        `strikes` is a JSON object keyed by strike, not an array, and object key order
        is not guaranteed — the keys are sorted numerically here. `iv` is a percentage
        and a sibling of `greeks`; `greeks` holds delta/gamma/theta/vega and no rho.
        The chain carries neither lot size nor tick size, so both come from the
        instruments file.
        """
        resolved_expiry = (expiry or self.nearest_expiry(underlying)).strip()
        if not _EXPIRY_RE.match(resolved_expiry):
            # The API answers a malformed expiry with the same 400 as an unknown
            # underlying, so the format is validated here where the message is useful.
            raise DataError(
                f"expiry must be YYYY-MM-DD, got {resolved_expiry!r}",
                underlying=underlying, expiry=resolved_expiry,
            )

        meta = self._meta(underlying)
        count = max(1, _as_int(strike_count, DEFAULT_STRIKE_COUNT))
        key = (underlying, resolved_expiry, count)

        cached = self._cache_get(key)
        if cached is not None:
            return cached

        raw = self.client.option_chain(
            exchange=str(meta.get("exchange") or "NSE"),
            segment=str(meta.get("chain_segment") or "INDEX"),
            underlying_scrip=self.underlying_scrip(underlying),
            expiry=resolved_expiry,
            strike_count=count,
        )
        if not isinstance(raw, dict):
            raise DataError(
                f"option chain for {underlying} {resolved_expiry} was {type(raw).__name__}, "
                "expected an object",
                underlying=underlying, expiry=resolved_expiry,
            )

        strikes = raw.get("strikes")
        if not isinstance(strikes, dict) or not strikes:
            raise DataError(
                f"option chain for {underlying} {resolved_expiry} carried no strikes",
                underlying=underlying, expiry=resolved_expiry,
            )

        spot = _as_float(raw.get("underlying_ltp"), 0.0)
        if spot <= 0:
            log.warning("option chain for %s %s returned no underlying_ltp",
                        underlying, resolved_expiry)

        chain_expiry = str(raw.get("expiry") or resolved_expiry).strip() or resolved_expiry
        if chain_expiry != resolved_expiry:
            # Trust the response over the request: the legs belong to the expiry the
            # API actually served, and that string is what the journal must record.
            log.warning("asked %s for %s, the chain says %s — using the chain's expiry",
                        underlying, resolved_expiry, chain_expiry)

        lot_size = self._lot_size(underlying, chain_expiry)
        exchange = str(meta.get("exchange") or "NSE")

        legs: list[OptionLeg] = []
        for strike, strike_key in self._sorted_strikes(strikes, underlying):
            node = strikes.get(strike_key)
            if not isinstance(node, dict):
                log.warning("%s %s strike %s is not an object — skipped",
                            underlying, chain_expiry, strike_key)
                continue
            for side in ("CE", "PE"):
                payload = node.get(side.lower())
                if not isinstance(payload, dict):
                    # A strike can legitimately carry only one side; not an error.
                    continue
                leg = self._build_leg(
                    underlying=underlying, expiry=chain_expiry, strike=strike, side=side,
                    payload=payload, lot_size=lot_size, exchange=exchange,
                )
                if leg is not None:
                    legs.append(leg)

        if not legs:
            raise DataError(
                f"option chain for {underlying} {chain_expiry} held {len(strikes)} strikes "
                "but no usable legs",
                underlying=underlying, expiry=chain_expiry,
            )

        log.debug("%s %s: %d legs across %d strikes, spot %.2f, lot %d",
                  underlying, chain_expiry, len(legs), len(strikes), spot, lot_size)
        self._cache_put(key, spot, legs)
        return spot, legs

    def _sorted_strikes(self, strikes: dict, underlying: str) -> list[tuple[float, str]]:
        """`strikes`' keys as (strike, original_key), sorted NUMERICALLY.

        The keys are strings and JSON object order is not guaranteed, so '10000' must
        not sort before '9500'.
        """
        out: list[tuple[float, str]] = []
        for raw_key in strikes:
            strike = _as_float(raw_key, -1.0)
            if strike <= 0:
                log.warning("ignoring non-numeric strike key %r for %s", raw_key, underlying)
                continue
            out.append((strike, raw_key))
        out.sort(key=lambda pair: pair[0])
        return out

    def _build_leg(
        self,
        *,
        underlying: str,
        expiry: str,
        strike: float,
        side: str,
        payload: dict,
        lot_size: int,
        exchange: str,
    ) -> OptionLeg | None:
        """One `ce`/`pe` node as an OptionLeg, or None when it cannot be ordered."""
        security_id = str(payload.get("security_id") or "").strip()
        if not security_id:
            # Without a security_id there is nothing to send to POST /order.
            log.warning("%s %s %s %s has no security_id — skipped",
                        underlying, expiry, strike, side)
            return None

        greeks = payload.get("greeks")
        greeks = greeks if isinstance(greeks, dict) else {}

        # Prices are clamped at zero: a negative premium would make lot_cost negative
        # and sail through the per-trade budget filter. Clamped, it fails filter 6.
        return OptionLeg(
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            side=side,
            security_id=security_id,
            trading_symbol=str(payload.get("trading_symbol") or "").strip(),
            lot_size=lot_size,
            exchange=exchange,
            ltp=max(0.0, _as_float(payload.get("last_price"))),
            bid=max(0.0, _as_float(payload.get("top_bid_price"))),
            ask=max(0.0, _as_float(payload.get("top_ask_price"))),
            oi=max(0, _as_int(payload.get("oi"))),
            volume=max(0, _as_int(payload.get("volume"))),
            # iv is a PERCENTAGE and a sibling of greeks, not a member of it.
            iv=max(0.0, _as_float(payload.get("iv"))),
            delta=_as_float(greeks.get("delta")),
            gamma=_as_float(greeks.get("gamma")),
            theta=_as_float(greeks.get("theta")),
            vega=_as_float(greeks.get("vega")),
            tick_size=self._tick_size(security_id),
        )

    def _lot_size(self, underlying: str, expiry: str) -> int:
        """Lot size from the instruments file. Refuses to continue without one.

        Every rupee figure in this module is `premium * lot_size`, so a guessed lot
        size is a mis-sized order. Instruments already falls back to FALLBACK_LOTS;
        if even that fails we stop rather than invent a number.
        """
        lot = 0
        try:
            lot = _as_int(self.instruments.lot_size(underlying, expiry), 0)
        except IndbotError as exc:
            log.warning("lot_size(%s, %s) failed: %s", underlying, expiry, exc)
        except Exception:  # noqa: BLE001 - logged with a traceback, never swallowed
            log.exception("lot_size(%s, %s) raised", underlying, expiry)
        if lot <= 0:
            raise DataError(
                f"no lot size for {underlying} {expiry}; refusing to price a leg without one",
                underlying=underlying, expiry=expiry,
            )
        return lot

    def _tick_size(self, security_id: str) -> float:
        """Tick size from the instruments file, or the 5-paise default."""
        tick = 0.0
        try:
            tick = _as_float(self.instruments.tick_size(security_id), 0.0)
        except IndbotError as exc:
            log.debug("tick_size(%s) failed: %s", security_id, exc)
        except Exception:  # noqa: BLE001 - logged with a traceback, never swallowed
            log.exception("tick_size(%s) raised", security_id)
        return tick if tick > 0 else DEFAULT_TICK_SIZE

    # ------------------------------------------------------------------- cache

    def _cache_get(self, key: tuple[str, str, int]) -> tuple[float, list[OptionLeg]] | None:
        """A cached ladder if it is younger than the TTL, else None."""
        if self._cache_ttl <= 0:
            return None
        with self._lock:
            entry = self._chain_cache.get(key)
            if entry is None:
                return None
            stamped, spot, legs = entry
            # monotonic, not clock.now(): a cache age is a duration, and INDBOT_FAKE_NOW
            # can be frozen — a frozen wall clock would make this cache immortal.
            if (_time.monotonic() - stamped) > self._cache_ttl:
                self._chain_cache.pop(key, None)
                return None
            return spot, list(legs)

    def _cache_put(self, key: tuple[str, str, int], spot: float, legs: list[OptionLeg]) -> None:
        """Remember a ladder for `cache_ttl` seconds."""
        if self._cache_ttl <= 0:
            return
        with self._lock:
            self._chain_cache[key] = (_time.monotonic(), spot, list(legs))

    def invalidate(self, underlying: str | None = None) -> None:
        """Drop cached ladders, for `underlying` or all of them."""
        with self._lock:
            if underlying is None:
                self._chain_cache.clear()
            else:
                for key in [k for k in self._chain_cache if k[0] == underlying]:
                    self._chain_cache.pop(key, None)

    # -------------------------------------------------------------------- pick

    @staticmethod
    def rank_key(leg: OptionLeg) -> tuple[float, float, float]:
        """Ranking key: nearest a 0.5 delta, then tightest spread, then deepest OI.

        CONTRACTS.md writes this as ``|delta - 0.5|``. A put's delta is NEGATIVE
        (-0.44 in the API's own example), so the literal form would score a 0.44-delta
        put at 0.94 and a 0.05-delta put at 0.55, ranking the deepest out-of-the-money
        put FIRST — the opposite of the intent, and in contradiction with filter 3,
        which bands on ``|delta|``. ``|  |delta| - 0.5 |`` is identical for calls and
        correct for puts.
        """
        return (abs(abs(leg.delta) - ATM_DELTA), leg.spread_pct, -float(leg.oi))

    def pick(
        self,
        underlying: str,
        side: str,
        *,
        spot: float | None = None,
        stop_pct: float | None = None,
        expiry: str | None = None,
        strike_count: int | None = None,
    ) -> tuple[OptionLeg | None, list[str]]:
        """Choose the single best tradable leg, or None plus why nothing qualified.

        Filters run in the order CONTRACTS.md section F gives them, and every dropped
        leg contributes one reason naming its strike and the number that failed. The
        reasons go straight to the UI's "why no trade" panel, so they are sentences.

        Raises DataError if the chain cannot be read: a data outage must not be
        reported as "nothing qualified".
        """
        want = str(side or "").strip().upper()
        if want not in ("CE", "PE"):
            raise DataError(f"side must be CE or PE, got {side!r}", underlying=underlying)

        if strike_count is None:
            # Only consult config when the caller did not say; an explicit argument wins.
            count = _as_int(getattr(self.cfg, "strike_count", DEFAULT_STRIKE_COUNT),
                            DEFAULT_STRIKE_COUNT)
        else:
            count = _as_int(strike_count, DEFAULT_STRIKE_COUNT)
        count = max(1, count)

        chain_spot, legs = self.fetch(underlying, expiry, count)
        reference_spot = spot if (spot is not None and spot > 0) else chain_spot
        if spot is not None and spot > 0 and chain_spot > 0:
            drift = abs(spot - chain_spot) / chain_spot * 100.0
            if drift > SPOT_DISAGREEMENT_PCT:
                log.warning(
                    "%s spot disagreement: signal %.2f vs chain %.2f (%.2f%%) — one is stale",
                    underlying, spot, chain_spot, drift,
                )

        reasons: list[str] = []

        # --- filter 1: correct side -----------------------------------------
        # A wrong-side leg is a partition, not a rejection; emitting 21 "wrong side"
        # lines would bury the reasons that matter. Only an empty side is reported.
        candidates = [leg for leg in legs if leg.side == want]
        if not candidates:
            reasons.append(
                f"The {underlying} {legs[0].expiry} chain came back with {len(legs)} "
                f"{'leg' if len(legs) == 1 else 'legs'} but not one {want} — "
                f"nothing to choose from."
            )
            return None, reasons

        lot_size = candidates[0].lot_size
        stop = _as_float(stop_pct, 0.0) if stop_pct is not None else _as_float(
            getattr(self.cfg, "stop_pct", 0.30), 0.30)
        if stop <= 0:
            # A zero stop would divide by zero below and imply an unlimited budget.
            log.warning("stop_pct was %r for %s; using 0.30 for the premium cap", stop_pct, underlying)
            stop = 0.30

        cap_unit, cap_lot, cap_source = self._premium_cap(lot_size, stop)

        # --- filter 2: the per-trade premium budget (the binding constraint) ---
        affordable: list[OptionLeg] = []
        premium_reasons: list[str] = []
        for leg in candidates:
            if leg.lot_cost <= cap_lot + MONEY_EPS:
                affordable.append(leg)
            else:
                premium_reasons.append(
                    f"{leg.label} at {_inr(leg.ltp, 2)}: one lot ({_qty(lot_size)} units) costs "
                    f"{_inr(leg.lot_cost)} and {_a_pct(stop)} stop risks "
                    f"{_inr(leg.lot_cost * stop)} — over the {_inr(cap_lot)} a lot this trade "
                    f"is allowed to cost."
                )
        if not affordable:
            # The arithmetic comes FIRST: it is the one thing the user needs to read.
            reasons.append(self._premium_arithmetic(underlying, want, candidates,
                                                    lot_size, cap_unit, cap_lot, stop, cap_source))
            reasons.extend(premium_reasons)
            log.info("%s %s: every strike is over the per-trade premium budget of %s a lot",
                     underlying, want, _inr(cap_lot))
            return None, reasons
        reasons.extend(premium_reasons)
        premium_dropped = len(candidates) - len(affordable)

        # --- filter 3: delta band -------------------------------------------
        min_delta = _as_float(getattr(self.cfg, "min_delta", 0.35), 0.35)
        max_delta = _as_float(getattr(self.cfg, "max_delta", 0.60), 0.60)
        if min_delta > max_delta:
            log.warning("min_delta %.2f > max_delta %.2f — swapping", min_delta, max_delta)
            min_delta, max_delta = max_delta, min_delta
        in_band: list[OptionLeg] = []
        for leg in affordable:
            abs_delta = abs(leg.delta)
            if min_delta - RATIO_EPS <= abs_delta <= max_delta + RATIO_EPS:
                in_band.append(leg)
            elif leg.delta == 0.0:
                reasons.append(
                    f"{leg.label}: the chain returned no delta for it, and a leg we cannot "
                    f"measure is not one we buy (band {min_delta:.2f}-{max_delta:.2f})."
                )
            else:
                where = "too far out of the money" if abs_delta < min_delta else "too deep in the money"
                reasons.append(
                    f"{leg.label}: delta {leg.delta:+.2f} (|{abs_delta:.2f}|) is outside the "
                    f"{min_delta:.2f}-{max_delta:.2f} band — {where}."
                )

        # --- filter 4: bid-ask spread ---------------------------------------
        max_spread = _as_float(getattr(self.cfg, "max_spread_pct", 2.0), 2.0)
        tight: list[OptionLeg] = []
        for leg in in_band:
            if leg.spread_pct <= max_spread + RATIO_EPS:
                tight.append(leg)
            elif leg.bid <= 0 or leg.ask <= 0:
                # spread_pct reports 100% for a one-sided book, so say what is actually
                # wrong rather than quoting a spread that does not exist.
                reasons.append(
                    f"{leg.label}: one-sided book (bid {_inr(leg.bid, 2)}, ask "
                    f"{_inr(leg.ask, 2)}) — nothing to buy from or sell back into."
                )
            else:
                reasons.append(
                    f"{leg.label}: spread {leg.spread_pct:.1f}% ({_inr(leg.spread, 2)} on a "
                    f"{_inr(leg.mid, 2)} mid) is over the {max_spread:.1f}% limit — the round "
                    f"trip costs {_inr(leg.spread * lot_size)} a lot before the trade moves."
                )

        # --- filter 5: liquidity --------------------------------------------
        min_oi = _as_int(getattr(self.cfg, "min_oi", 50000), 50000)
        min_volume = _as_int(getattr(self.cfg, "min_volume", 10000), 10000)
        liquid: list[OptionLeg] = []
        for leg in tight:
            thin: list[str] = []
            if leg.oi < min_oi:
                thin.append(f"open interest {_qty(leg.oi)} against a floor of {_qty(min_oi)}")
            if leg.volume < min_volume:
                thin.append(f"volume {_qty(leg.volume)} against a floor of {_qty(min_volume)}")
            if thin:
                reasons.append(f"{leg.label}: {' and '.join(thin)} — too thin to get out of.")
            else:
                liquid.append(leg)

        # --- filter 6: a real price and a real two-sided book -----------------
        survivors: list[OptionLeg] = []
        for leg in liquid:
            missing: list[str] = []
            if leg.ltp <= 0:
                missing.append("no last traded price")
            if leg.bid <= 0:
                missing.append("no bid")
            if leg.ask <= 0:
                missing.append("no ask")
            if missing:
                reasons.append(
                    f"{leg.label}: {', '.join(missing)} — a stale or one-sided book is not tradable."
                )
            else:
                survivors.append(leg)

        if not survivors:
            if premium_dropped:
                # The honest root cause: the budget had already taken the strikes near
                # the money off the table before the later filters ever saw them.
                reasons.append(
                    f"Note: the per-trade budget had already removed {premium_dropped} of "
                    f"{len(candidates)} {underlying} {want} strikes — everything above "
                    f"{_inr(cap_unit, 2)} a unit — so only far-out-of-the-money strikes "
                    f"reached the delta, spread and liquidity checks. "
                    + self._two_options(self._cheapest_in_band(candidates), stop)
                )
            log.info("%s %s: %d candidates, nothing survived the filters", underlying, want, len(candidates))
            return None, reasons

        survivors.sort(key=self.rank_key)
        best = survivors[0]
        chosen = (
            f"Chose {best.label} at {_inr(best.ltp, 2)} (best of {len(survivors)} tradable "
            f"{'strike' if len(survivors) == 1 else 'strikes'}): delta {best.delta:+.2f}, "
            f"spread {best.spread_pct:.1f}%, IV {best.iv:.1f}%, open interest {_qty(best.oi)}. "
            f"One lot is {_qty(lot_size)} units, so it costs {_inr(best.lot_cost)} and "
            f"{_a_pct(stop)} stop risks {_inr(best.lot_cost * stop)}."
        )
        if reference_spot > 0:
            chosen += f" Spot {_inr(reference_spot, 2)}."
        log.info("%s %s pick: %s @ %.2f (delta %.2f, lot cost %.0f)",
                 underlying, want, best.label, best.ltp, best.delta, best.lot_cost)
        # The chosen line leads: a decision that cannot explain itself reads like a bug.
        return best, [chosen] + reasons

    # -------------------------------------------------------- risk arithmetic

    def _premium_cap(self, lot_size: int, stop: float) -> tuple[float, float, str]:
        """The per-unit and per-lot premium cap, and which rule set it.

        Two numbers are reconciled here and the TIGHTER wins, deliberately:

        * ``cfg.effective_max_premium(lot_size)``, documented as
          ``max_trade_loss / (stop_pct * lot_size)`` — rupees PER UNIT.
        * the same constraint recomputed from ``max_trade_loss`` and this signal's own
          ``stop_pct``, which may have been widened by the strategy.

        The config field behind the first is named ``max_premium_per_lot`` while its
        formula yields a per-unit figure, so the two readings differ by a factor of
        ``lot_size``. Taking the minimum means a misread unit can only ever make the
        budget tighter, never let a trade risk more than ``max_trade_loss``.
        """
        max_trade_loss = _as_float(getattr(self.cfg, "max_trade_loss", 1000.0), 1000.0)
        denominator = stop * lot_size
        risk_cap = (max_trade_loss / denominator) if denominator > 0 else 0.0

        cfg_cap = 0.0
        try:
            cfg_cap = _as_float(self.cfg.effective_max_premium(lot_size), 0.0)
        except AttributeError:
            log.warning("cfg has no effective_max_premium(); using the risk-derived cap")
        except IndbotError as exc:
            log.warning("effective_max_premium(%d) failed (%s); using the risk-derived cap",
                        lot_size, exc)
        except Exception:  # noqa: BLE001 - logged with a traceback, never swallowed
            log.exception("effective_max_premium(%d) raised; using the risk-derived cap", lot_size)

        candidates = [value for value in (cfg_cap, risk_cap) if value > 0]
        if not candidates:
            # Both unusable. Zero blocks every strike and the reason explains it,
            # which is the safe direction when we cannot compute a budget.
            log.error("no usable premium cap for lot size %d (stop %.3f, max_trade_loss %.2f)",
                      lot_size, stop, max_trade_loss)
            return 0.0, 0.0, "unknown"
        cap_unit = min(candidates)
        # Only credit the config setting when it is strictly tighter: with
        # max_premium_per_lot = 0 it is derived from max_trade_loss and is the same
        # number, and naming the wrong rule would send the user to the wrong setting.
        source = "config" if (0 < cfg_cap < risk_cap) else "risk"
        return cap_unit, cap_unit * lot_size, source

    @staticmethod
    def _cheapest(candidates: list[OptionLeg]) -> tuple[OptionLeg | None, bool]:
        """The cheapest leg we could actually buy, and whether its book is two-sided.

        Quoting an untradable 5-paise strike as "the cheapest available" would be true
        and useless, so a leg with a real two-sided book is preferred.
        """
        priced = [leg for leg in candidates if leg.ltp > 0]
        tradable = [leg for leg in priced if is_tradable(leg)]
        if tradable:
            return min(tradable, key=lambda leg: leg.ltp), True
        if priced:
            return min(priced, key=lambda leg: leg.ltp), False
        return None, False

    def _in_band(self, leg: OptionLeg) -> bool:
        """True when a leg's |delta| is inside the configured band (filter 3's test)."""
        lo = _as_float(getattr(self.cfg, "min_delta", 0.35), 0.35)
        hi = _as_float(getattr(self.cfg, "max_delta", 0.60), 0.60)
        if lo > hi:
            lo, hi = hi, lo
        return lo - RATIO_EPS <= abs(leg.delta) <= hi + RATIO_EPS

    def _cheapest_in_band(self, candidates: list[OptionLeg]) -> OptionLeg | None:
        """The cheapest tradable leg the delta filter would accept, if any.

        This is the strike the user is actually being kept out of, so it is the one
        whose arithmetic the "raise the limit to X" suggestion must be built from.
        """
        usable = [leg for leg in candidates if is_tradable(leg) and self._in_band(leg)]
        return min(usable, key=lambda leg: leg.ltp) if usable else None

    def _two_options(self, reference: OptionLeg | None, stop: float) -> str:
        """The two honest ways out of the premium cap, priced on one specific leg.

        `reference` must be a leg the budget actually rejected — suggesting a limit
        derived from a strike that already fits would tell the user to lower it.
        """
        min_stop = _as_float(getattr(self.cfg, "min_stop_pct", 0.12), 0.12)
        max_trade_loss = _as_float(getattr(self.cfg, "max_trade_loss", 1000.0), 1000.0)
        if reference is None or reference.lot_cost <= 0:
            return (
                f"Either raise the per-trade limit above {_inr(max_trade_loss)} or accept a "
                f"stop tighter than {_pct(stop)} — nothing priced and tradable exists at this size."
            )
        risk = reference.lot_cost * stop
        suggested = math.ceil(risk / 100.0) * 100.0
        needed_stop = max_trade_loss / reference.lot_cost
        tail = (
            f"or accept {_a_pct(needed_stop)} stop on the {reference.label}, tighter than your "
            f"{_pct(min_stop)} floor"
            if needed_stop < min_stop else
            f"or accept {_a_pct(needed_stop)} stop on the {reference.label} instead of {_pct(stop)}"
        )
        if reference.spread_pct > 0 and reference.spread_pct / 100.0 >= needed_stop:
            tail += (
                f" — though its {reference.spread_pct:.1f}% bid-ask spread is wider than that "
                f"stop, so the spread alone would trigger it"
            )
        return (
            f"Two honest options: raise the per-trade limit from {_inr(max_trade_loss)} to about "
            f"{_inr(suggested)}, {tail}."
        )

    def _premium_arithmetic(self, underlying: str, side: str, candidates: list[OptionLeg],
                            lot_size: int, cap_unit: float, cap_lot: float,
                            stop: float, cap_source: str) -> str:
        """The first reason when the per-trade budget rejects every strike.

        Spells out the cheapest premium, the lot size, what one lot costs, what the
        stop risks and the two choices the user has — in a sentence a person reads,
        because this is the message that fires most often and it must not look like
        a malfunction.
        """
        max_trade_loss = _as_float(getattr(self.cfg, "max_trade_loss", 1000.0), 1000.0)
        cheapest, two_sided = self._cheapest(candidates)

        if cap_unit <= 0:
            return (
                f"No {underlying} {side} could be priced against a per-trade budget: the limit "
                f"({_inr(max_trade_loss)}), the stop ({_pct(stop)}) or the lot size "
                f"({_qty(lot_size)}) is missing or zero, so the bot cannot size a trade. "
                f"Fix those settings before it will trade."
            )
        if cheapest is None:
            return (
                f"None of the {len(candidates)} {underlying} {side} strikes has a traded price, "
                f"so none can be checked against the {_inr(cap_lot)} a lot this trade is allowed "
                f"to cost. The chain looks stale."
            )

        lot_cost = cheapest.lot_cost
        risk = lot_cost * stop
        book = "" if two_sided else " (and even that one has a one-sided book)"
        source = ("your premium-per-lot setting" if cap_source == "config"
                  else f"your {_inr(max_trade_loss)} per-trade loss limit at {_a_pct(stop)} stop")
        return (
            f"No {underlying} {side} is cheap enough to trade. The cheapest of the "
            f"{len(candidates)} {'strike' if len(candidates) == 1 else 'strikes'} is the "
            f"{cheapest.label} at {_inr(cheapest.ltp, 2)}"
            f"{book}; one lot is {_qty(lot_size)} units, so one lot costs {_inr(lot_cost)} and "
            f"{_a_pct(stop)} stop on it would risk {_inr(risk)}. {source.capitalize()} caps the "
            f"premium at {_inr(cap_unit, 2)} a unit, {_inr(cap_lot)} a lot, and every strike in "
            f"the chain is above that. "
            + self._two_options(self._cheapest_in_band(candidates) or cheapest, stop)
        )
