"""Static reference data and the one mutable settings object.

Every tunable in indbot lives in `DEFAULTS` and nowhere else: a number the user is
allowed to change must be a key in this file, not a literal buried in a strategy.
`Config` is loaded once at boot, mutated only through `update()` (which validates
the whole patch before applying any of it), and written back atomically.

Three rules here exist purely so that a bug cannot cost the user rupees:

* `update()` can never raise `armed` from False to True, and changing `mode`
  clears `armed`. Arming is a deliberate act with a typed confirmation
  (`set_armed`), never a side effect of saving a settings form. The engine
  enforces this too; this is the second lock on the same door.
* `load()` always boots DISARMED, whatever config.json says. A bot that crashed
  and restarted must be re-armed by a human who is watching.
* `effective_max_premium()` always answers in rupees PER UNIT, never per lot, so
  a consumer that misreads the unit buys nothing instead of buying far too much.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import clock
from .errors import ConfigError
from .types import MODES

log = logging.getLogger(__name__)

#: The exact phrase the user must type to send real money to the exchange.
#: Shared by the server, the engine and `Config.set_armed` so there is one spelling.
ARM_PHRASE = "ARM LIVE TRADING"

#: The six setups. `strategy.SETUPS` must match this list exactly; it is duplicated
#: here rather than imported because strategy.py imports Config and a cycle would
#: make the package unimportable.
SETUP_NAMES: tuple[str, ...] = (
    "orb",
    "vwap_reclaim",
    "ema_cross",
    "supertrend",
    "rsi_reversal",
    "pullback",
)

# These scrip ids and ws tokens are GUESSES except NIFTY's 40000001. Every one must be
# resolved at runtime from /market/instruments?source=index by matching `label` against
# the file's second column (which holds the index NAME despite its header), and the
# resolved value wins over the constant here. `Instruments.resolve_underlyings()` does
# that and logs a warning per mismatch, so a wrong constant is self-healing rather than
# a silently wrong order.
#: The five index underlyings.
#:
#: Every `scrip` and `label` below was VERIFIED against the live
#: `/market/instruments?source=index` master on 27 Oct 2026 and cross-checked by
#: price level, because four of the five original guesses were wrong in the worst
#: possible way: they returned real candles for the WRONG index. `40000002` is
#: NIFTY 100 (23,733), not BANKNIFTY; `40000003` is BANK NIFTY (54,392), not
#: FINNIFTY. A bot reading those would have computed signals on one index and
#: bought options on another, and nothing would have looked broken.
#:
#: Hence `price_band`: the plausible range for each index. `Instruments` checks the
#: live spot against it and refuses the underlying if it falls outside, so a future
#: id change surfaces as a blocked scan rather than a wrong trade. Widen a band only
#: after checking the index really has moved that far.
#:
#: `hist_segment` is the prefix for the REST `scrip-codes` surface and is NOT the
#: exchange: an NSE *index* is `NIDX_`, a BSE index `BIDX_`. Using `NSE_`/`BSE_`
#: there returns `400 Invalid scrip codes` — the original bug.
UNDERLYINGS: dict[str, dict] = {
    "NIFTY":      {"scrip": "40000001", "exchange": "NSE", "chain_segment": "INDEX",
                   "hist_segment": "NIDX", "fno_segment": "NFO", "strike_step": 50,
                   "label": "NIFTY 50", "price_band": (15000, 40000)},
    "BANKNIFTY":  {"scrip": "40000003", "exchange": "NSE", "chain_segment": "INDEX",
                   "hist_segment": "NIDX", "fno_segment": "NFO", "strike_step": 100,
                   "label": "BANK NIFTY", "price_band": (35000, 90000)},
    "FINNIFTY":   {"scrip": "40000100", "exchange": "NSE", "chain_segment": "INDEX",
                   "hist_segment": "NIDX", "fno_segment": "NFO", "strike_step": 50,
                   "label": "Nifty Financial", "price_band": (15000, 45000)},
    "MIDCPNIFTY": {"scrip": "40000093", "exchange": "NSE", "chain_segment": "INDEX",
                   "hist_segment": "NIDX", "fno_segment": "NFO", "strike_step": 25,
                   "label": "Nifty Midcap Sel", "price_band": (7000, 30000)},
    "SENSEX":     {"scrip": "40000006", "exchange": "BSE", "chain_segment": "INDEX",
                   "hist_segment": "BIDX", "fno_segment": "BFO", "strike_step": 100,
                   "label": "SENSEX", "price_band": (50000, 130000)},
}

#: Lot sizes used ONLY to render the risk table before the instrument master has been
#: downloaded. `instruments.FALLBACK_LOTS` is the authoritative copy and the live CSV
#: beats both — lot sizes change by circular, so never trade off this dict.
_FALLBACK_LOTS: dict[str, int] = {
    # Verified against the live F&O master on 27 Oct 2026. The previous values
    # (75/35/65) were wrong and, because the master lookup was also broken, they
    # silently sized every position. Prefer the CSV; this is the last resort.
    "NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120, "SENSEX": 20,
}

#: Indicative at-the-money premium band, rupees per unit. Reference only — it is here
#: so `describe_risk()` can tell the user "your cap is 44, the thing you want to buy
#: costs 150-200" before they arm, instead of after a week of no trades.
TYPICAL_ATM_PREMIUM: dict[str, tuple[float, float]] = {
    "NIFTY": (150.0, 200.0),
    "BANKNIFTY": (250.0, 350.0),
    "FINNIFTY": (120.0, 180.0),
    "MIDCPNIFTY": (80.0, 120.0),
    "SENSEX": (200.0, 300.0),
}

#: Every tunable, with its default. Keys here, the Config fields and _SPECS are
#: checked against each other at import time, so the three cannot drift apart.
DEFAULTS: dict[str, Any] = {
    # --- what the bot is allowed to do -----------------------------------------
    "mode": "paper",
    "armed": False,
    # All five major index books, as asked for.
    #
    # Worth knowing before you leave them all on: NIFTY, BANKNIFTY and SENSEX carry the
    # deepest books. FINNIFTY and MIDCPNIFTY are thinner — away from their expiry day the
    # oi/volume filters reject nearly every strike, so scanning them mostly spends rate
    # limit (the data bucket is 5 req/s across every underlying). If the log fills with
    # liquidity rejections on those two, drop them from this list and nothing else changes.
    "underlyings": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"],
    # --- risk -------------------------------------------------------------------
    # Deliberately left at the user's chosen 2,000. The high-conviction tier wants
    # 2,500 a trade, which is more than this; rather than quietly raise someone's daily
    # loss limit, `tier_for()` CLAMPS the per-trade cap to this number and says so.
    # Raising it is the account owner's decision, not a side effect of a feature.
    "max_daily_loss": 2000.0,
    "max_trade_loss": 1000.0,
    "max_trades_per_day": 3,
    "max_open_positions": 1,
    "max_lots": 1,
    # --- high-conviction tier ---------------------------------------------------
    # A score at or above `high_conviction_score` may risk up to
    # `high_conviction_max_trade_loss` instead of `max_trade_loss`, and in exchange
    # is REQUIRED to carry a reward:risk of `high_conviction_rr` — the target is
    # recomputed as stop x rr, never left at the ordinary 1:1.67.
    #
    # Read this before raising the cap: `score` is the strategy's own confluence
    # number, not a calibrated probability. Until the learner has closed trades in a
    # bucket it has never been tested against an outcome, so "high conviction" means
    # "this setup ticked more boxes", not "this is more likely to win".
    "high_conviction_enabled": True,
    "high_conviction_score": 0.75,
    "high_conviction_max_trade_loss": 2500.0,
    "high_conviction_rr": 2.0,
    # --- strategy ---------------------------------------------------------------
    "min_score": 0.55,
    "stop_pct": 0.30,
    "target_pct": 0.50,
    "trail_trigger_pct": 0.25,
    "min_stop_pct": 0.12,
    "or_minutes": 15,
    "interval": "5minute",
    "enabled_setups": list(SETUP_NAMES),
    # --- option selection -------------------------------------------------------
    "min_delta": 0.35,
    "max_delta": 0.60,
    "max_spread_pct": 2.0,
    "min_oi": 50000,
    "min_volume": 10000,
    "max_premium_per_lot": 0.0,
    # --- session ----------------------------------------------------------------
    "no_entry_before": "09:30",
    "no_entry_after": "14:45",
    "flat_by": "15:10",
    # --- learning ---------------------------------------------------------------
    "learning_enabled": True,
    "min_samples_to_gate": 10,
    "gate_confidence": 0.90,
    # --- execution / plumbing ---------------------------------------------------
    # Not listed in CONTRACTS.md section A, but sections J, K, L and M all need a
    # knob that has to live somewhere. Defaults are chosen so a module that never
    # reads them behaves the way its section describes.
    "poll_seconds": 20,
    "time_stop_minutes": 45,
    "slippage_ticks": 1,
    "cost_per_order": 20.0,
    "backtest_spread_pct": 1.5,
    "host": "127.0.0.1",
    "port": 8765,
}


# --------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    """One setting's type and permitted range, used for validation and for the
    error message the UI shows when the user types something impossible."""

    kind: str                               # bool|int|float|choice|list|time|host
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] | None = None  # for kind="choice"
    members: tuple[str, ...] | None = None  # permitted items, for kind="list"
    unit: str = ""
    note: str = ""

    def allowed(self) -> str:
        """A plain-English description of what this setting will accept."""
        if self.kind == "bool":
            return "true or false"
        if self.kind in ("int", "float"):
            word = "a whole number" if self.kind == "int" else "a number"
            unit = f" {self.unit}" if self.unit else ""
            if self.lo is not None and self.hi is not None:
                return f"{word} between {_num(self.lo)} and {_num(self.hi)}{unit}"
            if self.lo is not None:
                return f"{word} of at least {_num(self.lo)}{unit}"
            return f"{word}{unit}"
        if self.kind == "choice":
            return "one of " + ", ".join(self.choices or ())
        if self.kind == "list":
            return "a list drawn from " + ", ".join(self.members or ())
        if self.kind == "time":
            return 'an "HH:MM" time between 09:15 and 15:30'
        if self.kind == "host":
            return "a loopback address: 127.0.0.1, localhost or ::1"
        return "a valid value"


#: Reused unit blurb: these settings are fractions, not percentages.
_PCT = "(a fraction of the premium: 0.30 means 30%)"

_SPECS: dict[str, _Spec] = {
    "mode": _Spec("choice", choices=MODES, note="which broker the engine routes through"),
    "armed": _Spec("bool", note="live orders need mode=live AND armed"),
    "underlyings": _Spec("list", members=tuple(UNDERLYINGS), note="indices to scan"),

    "max_daily_loss": _Spec("float", 100.0, 1_000_000.0, unit="rupees"),
    "max_trade_loss": _Spec("float", 100.0, 1_000_000.0, unit="rupees"),
    "max_trades_per_day": _Spec("int", 1, 50),
    "max_open_positions": _Spec("int", 1, 10),
    "max_lots": _Spec("int", 1, 10, note="the bot is designed around 1"),
    "high_conviction_enabled": _Spec("bool", note="let a high score risk more, at 1:2 R:R"),
    "high_conviction_score": _Spec("float", 0.0, 1.0, note="score at or above which the raised cap applies"),
    "high_conviction_max_trade_loss": _Spec("float", 100.0, 1_000_000.0, unit="rupees"),
    "high_conviction_rr": _Spec("float", 1.0, 10.0, note="reward:risk required of a high-conviction trade"),

    "min_score": _Spec("float", 0.0, 1.0),
    "stop_pct": _Spec("float", 0.02, 0.90, unit=_PCT),
    "target_pct": _Spec("float", 0.05, 5.0, unit=_PCT),
    "trail_trigger_pct": _Spec("float", 0.0, 5.0, unit=_PCT),
    "min_stop_pct": _Spec("float", 0.02, 0.90, unit=_PCT),
    "or_minutes": _Spec("int", 5, 120, unit="minutes"),
    "interval": _Spec("choice", choices=tuple(clock.INTERVAL_MINUTES)),
    "enabled_setups": _Spec("list", members=SETUP_NAMES),

    "min_delta": _Spec("float", 0.0, 1.0),
    "max_delta": _Spec("float", 0.0, 1.0),
    "max_spread_pct": _Spec("float", 0.1, 50.0, unit="percent of the mid price"),
    "min_oi": _Spec("int", 0, 100_000_000, unit="contracts"),
    "min_volume": _Spec("int", 0, 100_000_000, unit="contracts"),
    "max_premium_per_lot": _Spec("float", 0.0, 1_000_000.0,
                                 unit="for one lot (0 = derive it from max_trade_loss)"),

    "no_entry_before": _Spec("time"),
    "no_entry_after": _Spec("time"),
    "flat_by": _Spec("time"),

    "learning_enabled": _Spec("bool"),
    "min_samples_to_gate": _Spec("int", 1, 10_000, unit="trades"),
    "gate_confidence": _Spec("float", 0.50, 0.999),

    "poll_seconds": _Spec("int", 2, 600, unit="seconds"),
    "time_stop_minutes": _Spec("int", 1, 375, unit="minutes"),
    "slippage_ticks": _Spec("int", 0, 20, unit="ticks"),
    "cost_per_order": _Spec("float", 0.0, 10_000.0, unit="rupees per order"),
    "backtest_spread_pct": _Spec("float", 0.0, 50.0, unit="percent of the mid price"),
    "host": _Spec("host"),
    "port": _Spec("int", 1024, 65535),
}

#: The server may only bind here. A bot that can move money must not listen on a
#: routable interface; section M says refuse to start, so refuse to configure too.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _num(value: float) -> str:
    """Format a bound without a trailing .0 on whole numbers."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _inr(value: float) -> str:
    """Rupees with Indian digit grouping, e.g. 3333.0 -> "Rs 3,333" rendered with the
    rupee sign. Used in the sentences the UI prints verbatim."""
    n = int(round(value))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        body = s
    else:
        head, tail = s[:-3], s[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        body = ",".join(groups + [tail])
    return f"{sign}₹{body}"


def _bad(key: str, value: Any, spec: _Spec) -> ConfigError:
    """The one error shape every validation failure uses: key, what came in, what is allowed."""
    return ConfigError(
        f"{key} must be {spec.allowed()} (got {value!r})",
        key=key,
        allowed=spec.allowed(),
        value=value if isinstance(value, (str, int, float, bool, type(None))) else repr(value),
    )


def _coerce(key: str, value: Any) -> Any:
    """Validate one setting and return it in its canonical type, or raise ConfigError.

    Bools are rejected for numeric keys on purpose: JSON's `true` is an int in Python
    and `{"max_trade_loss": true}` would otherwise silently become a 1 rupee limit.
    """
    spec = _SPECS[key]

    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise _bad(key, value, spec)
        return value

    if spec.kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad(key, value, spec)
        num = float(value)
        if num != num or num in (float("inf"), float("-inf")):  # NaN / inf
            raise _bad(key, value, spec)
        if spec.kind == "int":
            if abs(num - round(num)) > 1e-9:
                raise _bad(key, value, spec)
            num = float(round(num))
        if spec.lo is not None and num < spec.lo:
            raise _bad(key, value, spec)
        if spec.hi is not None and num > spec.hi:
            raise _bad(key, value, spec)
        return int(num) if spec.kind == "int" else num

    if spec.kind == "choice":
        if not isinstance(value, str) or value not in (spec.choices or ()):
            raise _bad(key, value, spec)
        return value

    if spec.kind == "list":
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise _bad(key, value, spec)
        members = spec.members or ()
        out: list[str] = []
        for item in value:
            name = item.strip()
            if name not in members:
                raise _bad(key, item, spec)
            if name not in out:          # dedupe, keep the user's order
                out.append(name)
        return out

    if spec.kind == "time":
        if not isinstance(value, str):
            raise _bad(key, value, spec)
        try:
            parsed = clock.parse_hhmm(value)
        except (ValueError, TypeError) as exc:
            raise _bad(key, value, spec) from exc
        if not (clock.SESSION_OPEN <= parsed <= clock.SESSION_CLOSE):
            raise _bad(key, value, spec)
        return f"{parsed.hour:02d}:{parsed.minute:02d}"

    if spec.kind == "host":
        if not isinstance(value, str) or value.strip() not in LOOPBACK_HOSTS:
            raise _bad(key, value, spec)
        return value.strip()

    raise ConfigError(f"no validator for {key}", key=key)  # unreachable; guards future edits


def _cross_check(values: dict[str, Any]) -> None:
    """Validate the settings that only make sense against each other.

    Raises on an incoherent pair, warns on a merely unwise one — the user is allowed
    to be aggressive, but not to configure something the engine cannot honour.
    """
    if values["stop_pct"] < values["min_stop_pct"]:
        raise ConfigError(
            f"stop_pct {values['stop_pct']} is tighter than min_stop_pct "
            f"{values['min_stop_pct']}; a stop that tight is triggered by the bid-ask "
            f"spread alone. Lower min_stop_pct first if you really mean it.",
            key="stop_pct",
            allowed=f"at least min_stop_pct ({values['min_stop_pct']})",
        )
    if values["min_delta"] > values["max_delta"]:
        raise ConfigError(
            f"min_delta {values['min_delta']} is above max_delta {values['max_delta']}; "
            f"no strike can satisfy both.",
            key="min_delta",
            allowed=f"at most max_delta ({values['max_delta']})",
        )

    before = clock.parse_hhmm(values["no_entry_before"])
    after = clock.parse_hhmm(values["no_entry_after"])
    flat = clock.parse_hhmm(values["flat_by"])
    if before >= after:
        raise ConfigError(
            f"no_entry_before {values['no_entry_before']} is not before no_entry_after "
            f"{values['no_entry_after']}; that window never opens.",
            key="no_entry_before",
            allowed=f"earlier than no_entry_after ({values['no_entry_after']})",
        )
    if flat <= after:
        raise ConfigError(
            f"flat_by {values['flat_by']} is not after no_entry_after "
            f"{values['no_entry_after']}; a trade opened at the last allowed entry would "
            f"be force-closed in the same breath.",
            key="flat_by",
            allowed=f"later than no_entry_after ({values['no_entry_after']})",
        )

    if values["max_trade_loss"] > values["max_daily_loss"]:
        log.warning(
            "max_trade_loss %s exceeds max_daily_loss %s: one losing trade can blow "
            "through the day limit before the halt can fire.",
            _inr(values["max_trade_loss"]), _inr(values["max_daily_loss"]),
        )
    if values["trail_trigger_pct"] >= values["target_pct"]:
        log.warning(
            "trail_trigger_pct %s is not below target_pct %s, so the trail will never "
            "arm before the target is hit.",
            values["trail_trigger_pct"], values["target_pct"],
        )
    if not values["underlyings"]:
        log.warning("underlyings is empty: the bot will scan nothing.")
    if not values["enabled_setups"]:
        log.warning("enabled_setups is empty: no setup can fire.")
    try:
        bar = clock.interval_minutes(values["interval"])
    except ValueError:
        bar = 0
    if bar and values["or_minutes"] < bar:
        log.warning(
            "or_minutes %s is shorter than one %s candle; the opening range will be "
            "built from a single bar.",
            values["or_minutes"], values["interval"],
        )


def default_config_path() -> Path:
    """Where config.json lives: the package parent, or $INDBOT_CONFIG when set."""
    env = os.environ.get("INDBOT_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "config.json"


# --------------------------------------------------------------------------------
# the settings object
# --------------------------------------------------------------------------------


@dataclass
class Config:
    """Every tunable the bot reads, loaded from config.json over DEFAULTS.

    Mutate it through `update()`, which validates the whole patch before applying
    any of it, so a rejected patch leaves the running bot exactly as it was.
    """

    mode: str = "paper"
    armed: bool = False
    underlyings: list[str] = field(
        default_factory=lambda: ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
    )
    # risk
    max_daily_loss: float = 2000.0
    max_trade_loss: float = 1000.0
    max_trades_per_day: int = 3
    max_open_positions: int = 1
    max_lots: int = 1
    high_conviction_enabled: bool = True
    high_conviction_score: float = 0.75
    high_conviction_max_trade_loss: float = 2500.0
    high_conviction_rr: float = 2.0
    # strategy
    min_score: float = 0.55
    stop_pct: float = 0.30
    target_pct: float = 0.50
    trail_trigger_pct: float = 0.25
    min_stop_pct: float = 0.12
    or_minutes: int = 15
    interval: str = "5minute"
    enabled_setups: list[str] = field(default_factory=lambda: list(SETUP_NAMES))
    # option selection
    min_delta: float = 0.35
    max_delta: float = 0.60
    max_spread_pct: float = 2.0
    min_oi: int = 50000
    min_volume: int = 10000
    max_premium_per_lot: float = 0.0
    # session
    no_entry_before: str = "09:30"
    no_entry_after: str = "14:45"
    flat_by: str = "15:10"
    # learning
    learning_enabled: bool = True
    min_samples_to_gate: int = 10
    gate_confidence: float = 0.90
    # execution / plumbing (see DEFAULTS)
    poll_seconds: int = 20
    time_stop_minutes: int = 45
    slippage_ticks: int = 1
    cost_per_order: float = 20.0
    backtest_spread_pct: float = 1.5
    host: str = "127.0.0.1"
    port: int = 8765

    #: The file this config came from. Only a config that was loaded from (or saved
    #: to) a path writes itself back on `update()`; a bare Config() or a copy_with()
    #: clone never touches the user's file.
    _path: Path | None = field(default=None, repr=False, compare=False)

    # -- loading / saving --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Read config.json, merged over DEFAULTS, and return a validated Config.

        A missing file is fine (pure defaults) and an unknown key is logged and
        ignored, but a malformed file or an out-of-range value raises: booting with
        silently different limits than the ones written down is how money is lost.
        """
        target = Path(path).expanduser() if path else default_config_path()
        values: dict[str, Any] = _fresh_defaults()
        raw: dict[str, Any] = {}

        if target.exists():
            try:
                text = target.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigError(f"could not read {target}: {exc}", path=str(target)) from exc
            if text.strip():
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        f"{target} is not valid JSON ({exc}). Fix or delete it; the bot "
                        f"will not start on a config it cannot read.",
                        path=str(target),
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ConfigError(
                        f"{target} must contain a JSON object, found {type(parsed).__name__}",
                        path=str(target),
                    )
                raw = parsed
        else:
            log.info("no config at %s - using defaults", target)

        for key, value in raw.items():
            if key not in _SPECS:
                # Forward/backward compatible: an old or newer file must not stop the bot.
                log.warning("ignoring unknown setting %r in %s", key, target)
                continue
            values[key] = _coerce(key, value)

        if values.get("armed"):
            log.warning(
                "config.json has armed=true - booting DISARMED. Arming is a live action "
                "confirmed by a human, never a file setting."
            )
        values["armed"] = False  # safe branch: a restart is never automatically live

        _cross_check(values)
        cfg = cls(**values, _path=target)
        log.info(
            "config loaded from %s: mode=%s underlyings=%s max_trade_loss=%s stop=%.0f%%",
            target, cfg.mode, ",".join(cfg.underlyings), _inr(cfg.max_trade_loss),
            cfg.stop_pct * 100,
        )
        return cfg

    def save(self, path: str | Path | None = None) -> None:
        """Write the settings as JSON, atomically.

        Temp file in the same directory, fsync, then os.replace — a crash or a full
        disk mid-write leaves the old config intact rather than a truncated file the
        bot would refuse (or worse, half-read) at the next boot.
        """
        target = Path(path).expanduser() if path else (self._path or default_config_path())
        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        tmp_name: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=".config-", suffix=".json.tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
            tmp_name = None
        except OSError as exc:
            raise ConfigError(f"could not write {target}: {exc}", path=str(target)) from exc
        finally:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:  # best effort; the stale temp file is harmless
                    log.debug("could not remove temp file %s", tmp_name)
        self._path = target
        log.debug("config written to %s", target)

    # -- mutation ----------------------------------------------------------------

    def update(self, patch: dict, *, persist: bool = True,
               strict_unknown: bool = False) -> list[str]:
        """Validate `patch`, apply it all or not at all, and return the changed keys.

        `armed` cannot be raised to True here — see `set_armed`. Changing `mode`
        forces `armed` back to False, so flipping to live through a settings form
        can never leave the bot live-and-armed in one call.

        Persists to the file this config was loaded from unless `persist=False`;
        a Config that was never loaded from a file (a test, or a `copy_with` clone
        used for a backtest sweep) never writes itself back.
        """
        if not isinstance(patch, dict):
            raise ConfigError(f"a config patch must be an object, got {type(patch).__name__}")

        current = self.as_dict()
        cleaned: dict[str, Any] = {}
        for key, raw in patch.items():
            if key not in _SPECS:
                if strict_unknown:
                    raise ConfigError(
                        f"unknown setting {key!r}",
                        key=key,
                        allowed="one of " + ", ".join(sorted(_SPECS)),
                    )
                log.warning("ignoring unknown setting %r in config patch", key)
                continue
            cleaned[key] = _coerce(key, raw)

        # Arming is a separate, deliberate act. Defence in depth: the engine refuses
        # this too, but a settings POST must never be able to go live by itself.
        if "armed" in cleaned:
            wanted = bool(cleaned.pop("armed"))
            if wanted and not self.armed:
                raise ConfigError(
                    "arming is not a settings change: POST /api/arm with "
                    f'confirm="{ARM_PHRASE}" instead.',
                    key="armed",
                    allowed="false (disarming is always permitted here)",
                )
            if not wanted and self.armed:
                cleaned["armed"] = False  # disarming through any path is always allowed

        candidate = dict(current)
        candidate.update(cleaned)

        if "mode" in cleaned and cleaned["mode"] != current["mode"]:
            # Safe branch: any mode change disarms. Re-arming is one deliberate click.
            candidate["armed"] = False

        _cross_check(candidate)

        changed = sorted(k for k in candidate if candidate[k] != current[k])
        for key in changed:
            value = candidate[key]
            setattr(self, key, list(value) if isinstance(value, list) else value)

        if changed:
            log.info("config updated: %s", ", ".join(f"{k}={candidate[k]!r}" for k in changed))
            if persist and self._path is not None:
                self.save()
        return changed

    def set_armed(self, value: bool, *, confirm: str | None = None) -> bool:
        """Arm or disarm. Arming requires `confirm == ARM_PHRASE`; disarming never does.

        This is the only supported way to set `armed=True`, because the one thing
        worse than a bot that will not trade is a bot that started trading by accident.
        """
        if not value:
            was = self.armed
            self.armed = False
            if was:
                log.warning("DISARMED - live orders are blocked")
            return False
        if confirm != ARM_PHRASE:
            raise ConfigError(
                f'to arm, confirm must be exactly "{ARM_PHRASE}"',
                key="armed",
                allowed=f'confirm="{ARM_PHRASE}"',
            )
        if self.mode != "live":
            # Not an error: the user may arm first and switch after. But say so loudly,
            # and note that update()/set_mode will clear it again on the mode change.
            log.warning("armed while mode=%s - no live order is sent until mode=live", self.mode)
        self.armed = True
        log.warning("ARMED - real orders will be placed in live mode")
        return True

    def copy_with(self, patch: dict | None = None) -> "Config":
        """A validated, file-less clone — for a backtest sweep that must not rewrite
        the user's config.json."""
        clone = Config(**self.as_dict())
        clone.armed = False  # a throwaway copy is never the armed one
        if patch:
            clone.update(patch, persist=False)
        return clone

    # -- views -------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The settings as plain JSON types, exactly the keys `update()` accepts."""
        out: dict[str, Any] = {}
        for key in DEFAULTS:
            value = getattr(self, key)
            out[key] = list(value) if isinstance(value, list) else value
        return out

    # -- risk arithmetic ---------------------------------------------------------

    def tier_for(self, score: float, *, stop_pct: float | None = None) -> dict[str, Any]:
        """Which risk tier a signal of this `score` trades under.

        Returns `{tier, max_trade_loss, stop_pct, target_pct, rr, note}`.

        A score at or above `high_conviction_score` may risk up to
        `high_conviction_max_trade_loss`, and in exchange its target is REWRITTEN to
        `stop_pct * high_conviction_rr` — taking more risk buys a wider target, not
        the ordinary 1:1.67. That is the whole bargain; it is not optional, because a
        bigger stake on the same payoff is strictly worse.

        The per-trade figure is never allowed above `max_daily_loss`: a trade that
        could breach the day cap on its own makes the day cap meaningless, and the
        halt would fire only after the damage. Clamped here rather than at the gate so
        every caller sees the same number.
        """
        base_stop = float(stop_pct if stop_pct is not None else self.stop_pct)
        ordinary = {
            "tier": "standard",
            "max_trade_loss": float(self.max_trade_loss),
            "stop_pct": base_stop,
            "target_pct": float(self.target_pct),
            "rr": (float(self.target_pct) / base_stop) if base_stop > 0 else 0.0,
            "note": "",
        }
        if not self.high_conviction_enabled:
            return ordinary
        try:
            if float(score) < float(self.high_conviction_score):
                return ordinary
        except (TypeError, ValueError):
            return ordinary

        cap = float(self.high_conviction_max_trade_loss)
        standard = float(self.max_trade_loss)
        # Clamp to the day limit, but NEVER below the standard cap. Without the outer
        # max() a config with max_daily_loss under max_trade_loss made the
        # high-conviction tier *more* restrictive than the ordinary one — a tier that
        # punishes a better signal, which is plainly not what it is for.
        clamped = max(standard, min(cap, float(self.max_daily_loss)))
        rr = float(self.high_conviction_rr)
        note = (
            f"High conviction ({float(score):.2f} at or above "
            f"{self.high_conviction_score:.2f}): risking up to {_inr(clamped)} instead of "
            f"{_inr(self.max_trade_loss)}, at a required {rr:.0f}:1 reward-to-risk "
            f"({base_stop:.0%} stop, {base_stop * rr:.0%} target)."
        )
        if clamped < cap:
            note += (
                f" Capped at the {_inr(self.max_daily_loss)} day limit — raise"
                f" max_daily_loss to use the full {_inr(cap)}."
            )
        return {
            "tier": "high_conviction",
            "max_trade_loss": clamped,
            "stop_pct": base_stop,
            "target_pct": base_stop * rr,
            "rr": rr,
            "note": note,
        }

    def effective_max_premium(self, lot_size: int) -> float:
        """The most one UNIT of premium may cost, in rupees. Never a per-lot figure.

        With max_premium_per_lot at 0 this is `max_trade_loss / (stop_pct * lot_size)`:
        the premium at which a full stop-out costs exactly the per-trade limit. For
        NIFTY at a 75 lot, a 1,000 rupee limit and a 30% stop that is 44 rupees a unit.

        The unit matters: a caller that mistakes this for a per-lot budget will reject
        every strike (buys nothing) rather than accept a position 75x too large. That
        is the deliberate direction to fail in.
        """
        if lot_size <= 0:
            # Never guess a lot size. A zero here means the instrument master is wrong,
            # and sizing against a guess is exactly how a position ends up 100x too big.
            raise ConfigError(
                f"lot_size must be a positive number of units, got {lot_size}",
                key="lot_size",
                allowed="a positive integer",
            )
        if self.max_premium_per_lot > 0:
            # Read as its name says: rupees for ONE LOT. Divided here so both branches
            # of this function return the same unit.
            return self.max_premium_per_lot / float(lot_size)
        if self.stop_pct <= 0:
            # update() cannot produce this; reaching it means someone set the attribute
            # directly. Refuse rather than divide by zero or fall back to "unlimited".
            raise ConfigError(
                f"stop_pct must be above 0 to derive a premium cap, got {self.stop_pct}",
                key="stop_pct",
                allowed=_SPECS["stop_pct"].allowed(),
            )
        return self.max_trade_loss / (self.stop_pct * float(lot_size))

    def effective_max_lot_cost(self, lot_size: int) -> float:
        """The same cap expressed as rupees for one whole lot — what the user pays."""
        return self.effective_max_premium(lot_size) * float(lot_size)

    def describe_risk(self, lot_sizes: dict[str, int] | None = None) -> list[dict]:
        """Per index: the premium cap the per-trade limit implies, and what it costs.

        The UI renders this before the user arms, so the constraint is visible up
        front: with the default 1,000 rupee limit and a 30% stop, an at-the-money
        option is simply out of budget on every index, and the honest fix is to raise
        the limit or accept a tighter stop — not for the bot to quietly buy junk.
        """
        lot_sizes = lot_sizes or {}
        rows: list[dict] = []
        for name, meta in UNDERLYINGS.items():
            raw = lot_sizes.get(name)
            lot = int(raw) if isinstance(raw, (int, float)) and int(raw) > 0 else 0
            source = "instruments"
            if lot <= 0:
                lot = _FALLBACK_LOTS.get(name, 0)
                source = "fallback"

            row: dict[str, Any] = {
                "underlying": name,
                "label": meta["label"],
                "enabled": name in self.underlyings,
                "lot_size": lot,
                "lot_size_source": source,
                "stop_pct": self.stop_pct,
                "cap_source": ("max_premium_per_lot" if self.max_premium_per_lot > 0
                               else "max_trade_loss / stop_pct"),
            }

            if lot <= 0:
                # No lot size anywhere: say so instead of printing a confident wrong number.
                row.update(
                    max_premium_per_unit=None, max_lot_cost=None, risk_at_cap=None,
                    typical_atm_low=None, typical_atm_high=None, typical_atm_lot_cost=None,
                    fits_typical_atm=None, max_trade_loss_for_atm=None,
                    note=f"Lot size for {name} is unknown until the instrument master loads.",
                )
                rows.append(row)
                continue

            per_unit = self.effective_max_premium(lot)
            lot_cost = per_unit * lot
            risk_at_cap = lot_cost * self.stop_pct
            override = self.max_premium_per_lot > 0
            low, high = TYPICAL_ATM_PREMIUM.get(name, (0.0, 0.0))
            # What the per-trade limit would have to be for a typical ATM option to pass.
            needed = low * lot * self.stop_pct if low > 0 else 0.0
            fits = bool(low > 0 and per_unit >= low)

            if override:
                note = (
                    f"One {name} lot is {lot} units. Your manual cap of "
                    f"{_inr(self.max_premium_per_lot)} a lot allows {_inr(per_unit)} a unit, "
                    f"and a {self.stop_pct * 100:.0f}% stop on a full lot risks "
                    f"{_inr(risk_at_cap)}."
                )
            else:
                note = (
                    f"One {name} lot is {lot} units, so a {_inr(self.max_trade_loss)} "
                    f"per-trade limit with a {self.stop_pct * 100:.0f}% stop caps the premium "
                    f"at {_inr(per_unit)} a unit ({_inr(lot_cost)} a lot)."
                )
            if low > 0:
                if fits:
                    note += (
                        f" A typical at-the-money {name} option trades near {_inr(low)}-"
                        f"{_inr(high)}, which fits."
                    )
                elif override:
                    note += (
                        f" A typical at-the-money {name} option trades near {_inr(low)}-"
                        f"{_inr(high)}, so nothing at the money qualifies: that would need a "
                        f"manual cap of about {_inr(low * lot)} a lot."
                    )
                else:
                    note += (
                        f" A typical at-the-money {name} option trades near {_inr(low)}-"
                        f"{_inr(high)}, so nothing at the money qualifies: it would need a "
                        f"per-trade limit of about {_inr(needed)}, or a tighter stop."
                    )
            if risk_at_cap > self.max_trade_loss + 0.5:
                # The manual cap is looser than the loss limit, so the risk gate will
                # refuse these legs anyway. Say it here rather than let the user wonder
                # why a strike they allowed keeps getting blocked.
                note += (
                    f" Note that the risk gate still caps a single trade at "
                    f"{_inr(self.max_trade_loss)}, so it will block anything above "
                    f"{_inr(self.max_trade_loss / self.stop_pct)} a lot."
                )

            row.update(
                max_premium_per_unit=round(per_unit, 2),
                max_lot_cost=round(lot_cost, 2),
                risk_at_cap=round(risk_at_cap, 2),
                typical_atm_low=low or None,
                typical_atm_high=high or None,
                typical_atm_lot_cost=round(low * lot, 2) if low > 0 else None,
                fits_typical_atm=fits,
                max_trade_loss_for_atm=round(needed, 2) if needed else None,
                note=note,
            )
            rows.append(row)
        return rows


def _fresh_defaults() -> dict[str, Any]:
    """A deep-enough copy of DEFAULTS: the list values must not be shared."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}


def _check_tables_agree() -> None:
    """Fail at import if DEFAULTS, the Config fields and _SPECS have drifted apart.

    Three parallel tables are a maintenance hazard, so the mismatch is caught at
    import time rather than as a missing knob halfway through a trading session.
    """
    declared = {f.name: f for f in fields(Config) if not f.name.startswith("_")}
    missing = sorted(set(declared) - set(DEFAULTS))
    extra = sorted(set(DEFAULTS) - set(declared))
    unspecced = sorted(set(DEFAULTS) - set(_SPECS))
    if missing or extra or unspecced:
        raise ConfigError(
            "config tables disagree - "
            f"fields without a default: {missing}; defaults without a field: {extra}; "
            f"settings without a validator: {unspecced}"
        )
    for name, f in declared.items():
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if default != DEFAULTS[name]:
            raise ConfigError(
                f"Config.{name} defaults to {default!r} but DEFAULTS says {DEFAULTS[name]!r}"
            )


_check_tables_agree()
