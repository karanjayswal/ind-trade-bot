"""The state machine: one loop thread, six tick steps, one lot at a time.

The engine owns every decision that costs money, and the order of its tick is the
safety argument:

1. clock — outside the session nothing trades, but the snapshot keeps serving.
2. manage open positions — exits BEFORE the halt check, so a halted day still gets
   out. A halt must never strand a trade.
3. gates — day halt and the position cap stop everything below this line.
4. scan — candles (cached), strategy, learner.
5. entry — the single best signal, the chain pick, the risk gate, one lot.
6. publish — the snapshot every subscriber and the UI reads.

Two threads touch this object: the loop thread, and the server thread calling
`snapshot()`, `set_mode()`, `arm()` or `panic()`. `_lock` guards the shared state and
is never held across an HTTP call; `_trade_lock` serialises the order paths. `panic()`
deliberately takes neither of the long ones, because it has to work when the loop
thread is wedged.

Every signal is journalled, taken or not, with the exact sentence that blocked it.
A quiet bot that explains itself beats a bot that trades.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterator, Sequence

from . import clock, indicators, token_store
from .api import INDstocksClient
from .broker import BaseBroker, make_broker, make_remark
from .chain import ChainPicker
from .config import ARM_PHRASE, UNDERLYINGS, Config
from .errors import AuthError, ConfigError, DataError, IndbotError, OrderError
from .instruments import Instruments
from .journal import Journal
from .learner import BucketVerdict, Learner
from .risk import RiskGate, inr
from .strategy import Strategy
from .types import MODES, Candle, OptionLeg, Position, RiskVerdict, Signal

log = logging.getLogger(__name__)

#: Log lines kept for the dashboard. The log is the product, but it is not storage.
LOG_CAPACITY = 500
#: Errors kept for the snapshot's `errors` list.
ERROR_CAPACITY = 50

#: Seconds to wait after a candle should have closed before asking for it. The feed
#: publishes a bar a beat after its close; asking at the exact second just wastes a
#: data call and gets the previous bar back.
CANDLE_LAG_S = 8
#: When the feed is behind, retry this often rather than every tick — the data limit
#: is 5 calls a second across all five underlyings and a tight loop would 429.
CANDLE_RETRY_S = 20
#: Bars the indicators need to warm up: EMA50 and ADX(14) both want ~50.
MIN_BARS = 60
#: Never ask for fewer than this many sessions of history.
MIN_SESSIONS = 3
#: Minutes in one NSE/BSE equity session (09:15-15:30).
SESSION_MINUTES = 375

#: How long a /funds snapshot is considered current.
FUNDS_TTL_S = 60.0
#: How long the account profile is considered current.
PROFILE_TTL_S = 3600.0
#: After a failed exit, wait this long before trying again. It never gives up — a
#: stranded position is the worst outcome this system can have — but it must not
#: hammer the order endpoint either.
EXIT_RETRY_S = 45.0

#: A signal older than this many candle intervals is not entered. By the time a
#: stale feed catches up, the price that justified the trigger is gone.
MAX_SIGNAL_AGE_INTERVALS = 1

#: Default seconds between ticks when the config does not say.
DEFAULT_POLL_SECONDS = 20


# ---------------------------------------------------------------------------
# log capture
# ---------------------------------------------------------------------------
class _LogSink(logging.Handler):
    """Keeps the last `capacity` indbot log records so the UI can show them.

    Installed on the `indbot` package logger, not on root, so it captures the chain
    picker's and the risk gate's sentences — the ones the user actually needs — and
    leaves the host application's logging configuration alone.
    """

    def __init__(self, capacity: int = LOG_CAPACITY) -> None:
        super().__init__(level=logging.INFO)
        self.lines: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Append one record. Never raises: a handler that throws breaks every log call."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not kill logging
            message = str(record.msg)
        try:
            with self._lock:
                self.lines.append(
                    {
                        "ts": int(record.created),
                        "at": clock.to_ist(record.created).strftime("%H:%M:%S"),
                        "level": record.levelname,
                        "logger": record.name.split(".")[-1],
                        "msg": message,
                    }
                )
        except Exception:  # noqa: BLE001 - see above
            pass

    def tail(self, limit: int = 250, min_level: str | None = None) -> list[dict[str, Any]]:
        """The newest lines first, optionally filtered to `min_level` and above."""
        with self._lock:
            rows = list(self.lines)
        rows.reverse()
        if min_level:
            floor = logging.getLevelName(str(min_level).upper())
            if isinstance(floor, int):
                rows = [
                    r for r in rows
                    if _level_number(str(r.get("level", ""))) >= floor
                ]
        return rows[: max(0, int(limit))]


def _level_number(name: str) -> int:
    """Numeric level for a level name, 0 when it is not a level we know."""
    value = logging.getLevelName(name.upper())
    return value if isinstance(value, int) else 0


# ---------------------------------------------------------------------------
# candle cache
# ---------------------------------------------------------------------------
@dataclass
class _Series:
    """One underlying's cached candles and when they are worth refetching.

    `next_fetch_at` is epoch SECONDS on the indbot clock (not monotonic) so a faked
    clock replays the same fetch schedule a live session would have had.
    """

    candles: list[Candle] = field(default_factory=list)
    last_ts: int = 0
    fetched_ts: int = 0
    next_fetch_at: int = 0
    fetches: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when there is at least one usable candle in the cache."""
        return bool(self.candles)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class Engine:
    """Runs the bot: one loop thread, a stop Event, and a snapshot the server reads.

    Construct it with a Config and a Journal; every other collaborator is built here
    and can be injected for tests. Nothing is fetched and no thread is started until
    `start()` (or a direct `tick()`), so constructing an Engine is cheap and safe.
    """

    def __init__(
        self,
        cfg: Config,
        journal: Journal,
        *,
        client: INDstocksClient | None = None,
        instruments: Instruments | None = None,
        strategy: Strategy | None = None,
        chain: ChainPicker | None = None,
        learner: Learner | None = None,
        risk: RiskGate | None = None,
        broker: BaseBroker | None = None,
        allow_closed_market: bool = False,
        capture_logs: bool = True,
    ) -> None:
        self.cfg = cfg
        self.journal = journal
        # Signals carry no mode of their own; stamp them with the mode we are in so a
        # paper signal can never be read back as live history.
        self.journal.default_mode = cfg.mode
        self._allow_closed_market = bool(allow_closed_market)

        self._lock = threading.RLock()          # guards shared state and the snapshot
        self._trade_lock = threading.RLock()    # serialises order paths (never held by panic)
        self._tick_lock = threading.RLock()     # one tick at a time, loop or test
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._sink = _LogSink()
        self._capture_logs = bool(capture_logs)
        if self._capture_logs:
            pkg_logger = logging.getLogger(__name__.split(".")[0])
            if self._sink not in pkg_logger.handlers:
                pkg_logger.addHandler(self._sink)

        self._errors: deque[dict[str, Any]] = deque(maxlen=ERROR_CAPACITY)
        self._subscribers: dict[int, Callable[[dict], None]] = {}
        self._sub_seq = 0
        self._published: dict[str, Any] | None = None

        self._positions: dict[str, Position] = {}
        self._exiting: set[str] = set()          # position ids with a SELL in flight
        self._exit_fail: dict[str, dict[str, Any]] = {}
        self._series: dict[str, _Series] = {}
        self._views: dict[str, dict[str, Any]] = {}
        self._seen_signals: set[tuple[str, str, str, int]] = set()
        self._lot_sizes: dict[str, int] = {}
        self._day: date | None = None

        self._funds: dict[str, Any] | None = None
        self._funds_ts: int = 0
        self._account: dict[str, Any] | None = None
        self._account_ts: int = 0

        self._tick_count = 0
        self._last_tick_at: str | None = None
        self._last_tick_ms: float = 0.0
        self._started_at: str | None = None
        #: Raised while an exit is being sent. No entry may be taken in that window:
        #: `_exit_authority` can momentarily re-arm the broker and a BUY must not
        #: slip through on that arm.
        self._exit_window = 0

        self._client = client if client is not None else self._make_client()
        self.instruments = instruments or (
            Instruments(self._client) if self._client is not None else None
        )
        self.strategy = strategy or Strategy(cfg)
        self.learner = learner or Learner(journal, cfg)
        self.risk = risk or RiskGate(journal, cfg, learner=self.learner)
        self.chain = chain or (
            ChainPicker(self._client, self.instruments, cfg)
            if (self._client is not None and self.instruments is not None)
            else None
        )
        self.broker = broker or make_broker(
            cfg, self._client, allow_closed_market=self._allow_closed_market
        )

        # Remembered so a config change made behind the engine's back is noticed.
        self._broker_mode = self.cfg.mode
        self._strategy_interval = str(getattr(cfg, "interval", ""))
        self._strategy_setups = list(getattr(cfg, "enabled_setups", None) or [])

        if self._client is None:
            self._record_error(
                DataError(
                    "No INDstocks access token, so there is no market data and no broker "
                    "connection. Set INDSTOCKS_ACCESS_TOKEN and restart; the dashboard "
                    "will keep serving until then but the bot cannot trade."
                ),
                where="startup",
            )
        log.info(
            "engine ready: mode=%s armed=%s underlyings=%s interval=%s poll=%ss",
            cfg.mode, cfg.armed, ",".join(cfg.underlyings), self.strategy.interval,
            self.poll_seconds,
        )

    # ---------------------------------------------------------------- setup

    def _make_client(self) -> INDstocksClient | None:
        """Build the API client, or None when there is no token.

        Returning None rather than raising keeps the dashboard bootable so it can
        tell the user what is missing. Every path that needs the client checks for it
        and refuses, so a missing token can only ever mean "does not trade".
        """
        try:
            return INDstocksClient()
        except IndbotError as exc:
            log.error("no INDstocks client: %s", getattr(exc, "message", exc))
            return None

    def set_token(self, raw: str, *, persist: bool = True) -> dict[str, Any]:
        """Install a new access token, after proving it works. Returns its description.

        The order matters and is the whole point: the token is validated against
        GET /user/profile with a throwaway client BEFORE anything is replaced, so a
        bad paste cannot take down a working session. Only once the broker has
        answered is the token stored and the live wiring rebuilt.

        Refuses outright while a live position is open. Swapping the credential
        underneath an open trade would, if the new token were wrong, leave a
        position on the exchange that this process can no longer exit.
        """
        token = token_store.validate_shape(raw)

        with self._trade_lock:
            open_now = self._open_positions()
            if open_now and str(self.cfg.mode) == "live":
                raise OrderError(
                    f"refusing to change the access token with {len(open_now)} live "
                    "position(s) open: if the new token were wrong, this process could "
                    "no longer exit them. Close them first, or use panic().",
                    payload={"open": [p.leg.label for p in open_now]},
                )

            # Prove it before trusting it.
            probe = INDstocksClient(token=token)
            try:
                profile = probe.profile()
            except IndbotError as exc:
                probe.close()
                raise AuthError(
                    "that token did not work: "
                    + str(getattr(exc, "message", exc))
                    + ". Tokens expire at 07:00 IST and generating a new one invalidates "
                    "the previous token, so make sure nothing else has minted one since.",
                    masked=token_store.mask(token),
                ) from exc

            if persist:
                token_store.save(token)

            old = self._client
            self._client = probe
            self.instruments = Instruments(probe)
            self.chain = ChainPicker(probe, self.instruments, self.cfg)
            try:
                self.broker = make_broker(
                    self.cfg, probe, allow_closed_market=self._allow_closed_market
                )
            except IndbotError:
                # A broker this mode cannot build is not a reason to lose a working
                # client; the mode gate will report it separately.
                log.exception("new token installed but the broker could not be rebuilt")
            self._broker_mode = self.cfg.mode
            self._account = profile
            self._series.clear()           # refetch on the next tick with the new token
            if old is not None and old is not probe:
                with contextlib.suppress(Exception):
                    old.close()

        with self._lock:
            # The startup "no access token" error is now stale — drop it, or the
            # dashboard keeps showing a red banner about a problem just fixed.
            self._errors = [e for e in self._errors if e.get("where") != "startup"]
        log.warning(
            "access token updated (%s) for client %s",
            token_store.mask(token),
            profile.get("user_id") or profile.get("ucc") or "?",
        )
        self._publish()
        return self.token_info()

    def token_info(self) -> dict[str, Any]:
        """Masked token metadata for the UI. Never contains the token itself."""
        raw, source = token_store.load()
        info = token_store.describe(raw)
        info["source"] = (
            "environment ($INDSTOCKS_ACCESS_TOKEN)" if source == "env"
            else "dashboard" if source == "file" else None
        )
        info["env_override"] = source == "env"
        info["client_attached"] = self._client is not None
        return info

    @property
    def poll_seconds(self) -> int:
        """Seconds between ticks, at least 1."""
        try:
            return max(1, int(getattr(self.cfg, "poll_seconds", DEFAULT_POLL_SECONDS)))
        except (TypeError, ValueError):
            return DEFAULT_POLL_SECONDS

    @property
    def interval_minutes(self) -> int:
        """Minutes per candle, from the strategy (which validated the interval)."""
        return self.strategy.interval_minutes

    @property
    def running(self) -> bool:
        """True while the loop thread is alive."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Spawn the loop thread. Idempotent; recovers any position left open."""
        with self._lock:
            if self.running:
                log.info("engine already running")
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="indbot-engine", daemon=True)
            self._started_at = clock.now().isoformat()
        self._recover_open_positions()
        self._thread.start()
        log.warning("engine STARTED in %s mode (armed=%s)", self.cfg.mode, self.cfg.armed)

    def stop(self) -> None:
        """Signal the loop to finish and wait briefly for it.

        Stopping does NOT close an open position: the user may be restarting the
        process, and selling on a stop would turn a restart into a realised loss.
        Open positions are recovered from the journal on the next `start()`.
        """
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(2.0, min(10.0, float(self.poll_seconds))))
        with self._lock:
            self._thread = None
        if self._positions:
            log.critical(
                "engine STOPPED with %d position(s) still open: %s. They are journalled and "
                "will be picked up again on start(); until then nothing is watching the stop.",
                len(self._positions),
                ", ".join(p.leg.label for p in self._positions.values()),
            )
        else:
            log.warning("engine STOPPED")
        self._publish()

    def close(self) -> None:
        """Stop the loop and detach the log handler. Safe to call twice."""
        self.stop()
        if self._capture_logs:
            with contextlib.suppress(Exception):
                logging.getLogger(__name__.split(".")[0]).removeHandler(self._sink)

    def _loop(self) -> None:
        """Tick until stopped, sleeping the remainder of `poll_seconds` between passes."""
        log.info("tick loop started (every %ss)", self.poll_seconds)
        while not self._stop.is_set():
            started = clock.now()
            try:
                self.tick()
            except OrderError as exc:
                # An order failure is never absorbed: it means money did or did not move
                # when we believed otherwise. Shout, and keep looping so a failed EXIT is
                # retried rather than abandoned. `tick` has already recorded and published.
                log.critical("ORDER FAILURE in tick: %s", exc, exc_info=True)
            except IndbotError as exc:
                log.error("tick failed: %s", getattr(exc, "message", exc), exc_info=True)
            except Exception:  # noqa: BLE001 - the loop must outlive any bug
                log.exception("unexpected error in tick")
            elapsed = (clock.now() - started).total_seconds()
            self._stop.wait(max(1.0, float(self.poll_seconds) - max(0.0, elapsed)))
        log.info("tick loop finished")

    # ----------------------------------------------------------------- tick

    def tick(self) -> dict[str, Any]:
        """One pass of the state machine. Returns a small summary of what it did.

        Safe to call directly from a test or from the server; concurrent calls are
        serialised so two ticks can never place two orders for the same signal.
        """
        with self._tick_lock:
            try:
                return self._tick_once()
            except Exception as exc:  # noqa: BLE001 - recorded, published, then re-raised
                self._record_error(exc, where="tick")
                self._publish()
                raise

    def _tick_once(self) -> dict[str, Any]:
        """The six steps, in the order the safety argument depends on."""
        started = clock.now()
        result: dict[str, Any] = {
            "at": started.isoformat(),
            "phase": "idle",
            "market_open": False,
            "halted": False,
            "halt_reason": "",
            "exits": [],
            "entries": [],
            "signals": 0,
            "scanned": [],
            "errors": [],
        }

        self._sync_config()
        self._roll_day(started)

        # --- step 1: the clock ------------------------------------------------
        market_open = clock.is_market_open(started)
        result["market_open"] = market_open
        if not market_open:
            self._idle(started)
            self._finish_tick(started)
            self._publish()
            return result

        self._fetch_funds()

        # --- step 2: manage open positions (BEFORE the halt check) ------------
        result["phase"] = "manage"
        result["exits"] = self._manage_positions(started)

        # --- step 3: the gates ------------------------------------------------
        halted, halt_reason = self._halt_state()
        result["halted"] = halted
        result["halt_reason"] = halt_reason
        if halted:
            result["phase"] = "halted"
            self._finish_tick(started)
            self._publish()
            return result

        cap = max(0, int(getattr(self.cfg, "max_open_positions", 1) or 0))
        if len(self._open_positions()) >= cap:
            result["phase"] = "at_position_cap"
            self._finish_tick(started)
            self._publish()
            return result

        if self.chain is None:
            result["phase"] = "no_client"
            self._finish_tick(started)
            self._publish()
            return result

        # --- step 4: scan -----------------------------------------------------
        result["phase"] = "scan"
        signals, scanned = self._scan(started)
        result["signals"] = len(signals)
        result["scanned"] = scanned

        # --- step 5: the single best signal -----------------------------------
        if signals:
            result["phase"] = "entry"
            entered = self._consider_entries(signals, started)
            if entered:
                result["entries"].append(entered)

        # --- step 6: publish ---------------------------------------------------
        self._finish_tick(started)
        self._publish()
        result["errors"] = self.errors()
        return result

    def _finish_tick(self, started: datetime) -> None:
        """Stamp the tick counters the dashboard uses to show the loop is alive."""
        with self._lock:
            self._tick_count += 1
            self._last_tick_at = started.isoformat()
            self._last_tick_ms = round((clock.now() - started).total_seconds() * 1000.0, 1)

    def _idle(self, now_dt: datetime) -> None:
        """Outside the session: mark any open position and shout if one is stranded.

        No order is sent. An INTRADAY order placed outside hours either bounces or
        queues to fill unattended at the next open, which is a position nobody is
        watching — strictly worse than carrying the one we already have.
        """
        positions = self._open_positions()
        if not positions:
            return
        for pos in positions:
            with contextlib.suppress(IndbotError):
                self._mark(pos)
        log.critical(
            "market is closed and %d position(s) are still open: %s. The bot will not send "
            "an order outside session hours — square off by hand or wait for the open.",
            len(positions),
            ", ".join(f"{p.leg.label} x{p.qty}" for p in positions),
        )
        self._record_error(
            OrderError(
                f"{len(positions)} position(s) open outside market hours: "
                + ", ".join(p.leg.label for p in positions)
                + ". flat_by should have closed these; verify the position on INDmoney.",
            ),
            where="idle",
        )

    # ------------------------------------------------------- config drift

    def _sync_config(self) -> None:
        """Pick up config changes made outside `set_mode` and rebuild what they affect.

        POST /api/config can change `mode`, `interval` and `enabled_setups` without
        going through this class. The interval and the setup list are cheap to honour.
        A mode change is not: leaving live mode is supposed to flatten first, and a
        LiveBroker swapped out from under an open live position would "exit" it on
        paper while the real contract stays open. That one is refused loudly.
        """
        mode = str(getattr(self.cfg, "mode", "")).strip()
        if mode != self._broker_mode:
            live_positions = [p for p in self._open_positions() if p.mode == "live"]
            if self._broker_mode == "live" and live_positions:
                log.critical(
                    "mode was changed from live to %r outside set_mode() while %d live "
                    "position(s) are open (%s). Keeping the LIVE broker so the position can "
                    "still be squared off, halting the day, and refusing new entries. Use "
                    "panic() or close them by hand.",
                    mode, len(live_positions),
                    ", ".join(p.leg.label for p in live_positions),
                )
                self._record_error(
                    OrderError(
                        f"mode changed to {mode!r} outside set_mode() with a live position "
                        f"open ({', '.join(p.leg.label for p in live_positions)}). The live "
                        "broker is retained so the exit path still works; no new entry will "
                        "be taken. Square off with PANIC or on INDmoney.",
                    ),
                    where="config",
                )
                self.risk.halt_day(
                    "live", "Mode was changed out from under an open live position."
                )
                return  # keep the live broker: the exit path matters more than tidiness
            self.broker = make_broker(
                self.cfg, self._client, allow_closed_market=self._allow_closed_market
            )
            self._broker_mode = mode
            self.journal.default_mode = mode
            log.warning("broker rebuilt for mode=%s (%s)", mode, type(self.broker).__name__)

        interval = str(getattr(self.cfg, "interval", self._strategy_interval))
        setups = list(getattr(self.cfg, "enabled_setups", self._strategy_setups) or [])
        if interval != self._strategy_interval or setups != self._strategy_setups:
            try:
                self.strategy = Strategy(self.cfg)
            except ConfigError as exc:
                log.error("rejecting the new strategy settings: %s", exc)
                self._record_error(exc, where="config")
                return
            if interval != self._strategy_interval:
                with self._lock:
                    self._series.clear()  # bars of a different length are a different series
            self._strategy_interval = interval
            self._strategy_setups = list(setups)
            log.warning(
                "strategy rebuilt: interval=%s setups=%s",
                self.strategy.interval, ",".join(self.strategy.enabled),
            )

    def _roll_day(self, now_dt: datetime) -> None:
        """First tick of a new IST day: clear per-day caches and re-resolve instruments."""
        today = now_dt.date()
        if self._day == today:
            return
        with self._lock:
            self._day = today
            self._series.clear()
            self._views.clear()
            self._seen_signals.clear()
            self._lot_sizes.clear()
        log.info("new trading day %s (%s)", today.isoformat(),
                 "trading day" if clock.is_trading_day(today) else "not a trading day")
        self._refresh_instruments()
        self._fetch_account()
        with contextlib.suppress(Exception):
            self.learner.refresh()

    def _refresh_instruments(self) -> None:
        """Resolve the index scrips and the lot sizes for today, tolerating failure.

        A wrong scrip id is the difference between NIFTY's candles and somebody
        else's, so the live index file always wins over the constant; a mismatch is
        logged by `resolve_underlyings` and shown in the snapshot as unverified.
        """
        if self.instruments is None:
            return
        try:
            self.instruments.resolve_underlyings(self.cfg)
        except IndbotError as exc:
            log.error("could not resolve the index scrips: %s", getattr(exc, "message", exc))
            self._record_error(exc, where="instruments")
        for name in UNDERLYINGS:
            try:
                self._lot_sizes[name] = int(self.instruments.lot_size(name))
            except IndbotError as exc:
                log.warning("no lot size for %s: %s", name, getattr(exc, "message", exc))
            except Exception as exc:  # noqa: BLE001 - CSV parsing is the other agent's code
                log.warning("lot size lookup for %s raised: %s", name, exc)

    # --------------------------------------------------- step 2: positions

    def _open_positions(self) -> list[Position]:
        """A stable list of the positions the engine believes are open."""
        with self._lock:
            return [p for p in self._positions.values() if p.is_open]

    def _manage_positions(self, now_dt: datetime) -> list[dict[str, Any]]:
        """Mark every open position, trail it, and exit the ones that must go.

        Runs before the day-halt check on purpose: a halted day still has to get out
        of the trade it is already in.
        """
        done: list[dict[str, Any]] = []
        for pos in self._open_positions():
            try:
                done.extend(self._manage_one(pos, now_dt))
            except OrderError:
                raise  # an order failure escalates to the loop, loudly
            except IndbotError as exc:
                # A data failure on one position must not stop the others being managed.
                log.error("could not manage %s: %s", pos.leg.label, getattr(exc, "message", exc))
                self._record_error(exc, where=f"position:{pos.id}")
        return done

    def _manage_one(self, pos: Position, now_dt: datetime) -> list[dict[str, Any]]:
        """Refresh one position's mark, advance the trail, and exit if a rule says so."""
        ltp = self._mark(pos)
        self.risk.update_trail(pos, ltp)

        reason = self.risk.check_exit_required(pos, ltp, now=now_dt)
        exit_sig: Signal | None = None
        if reason is None and ltp > 0:
            # Only consult the strategy when no hard rule fired: stop, target, trail
            # and flat_by are not opinions and do not need a candle to be right.
            exit_sig = self._exit_signal(pos, now_dt)
            if exit_sig is not None:
                reason = "signal"

        if reason is None:
            return []

        if not self._exit_due(pos):
            return []

        closed = self._exit(pos, reason, ltp=ltp, signal=exit_sig)
        return [closed] if closed else []

    def _mark(self, pos: Position) -> float:
        """Refresh the position's LTP and write it back onto the (frozen) leg.

        A recovered position's leg has no price at all, and `marketable_limit` refuses
        to send an order without one, so the mark is also what keeps the exit path
        alive across a restart.
        """
        try:
            ltp = float(self.broker.ltp(pos.leg))
        except IndbotError as exc:
            log.warning("no mark for %s: %s", pos.leg.label, getattr(exc, "message", exc))
            return float(pos.leg.ltp or 0.0)
        if ltp > 0 and abs(ltp - float(pos.leg.ltp or 0.0)) > 1e-9:
            pos.leg = replace(pos.leg, ltp=ltp)
        if ltp > float(pos.high_water or 0.0):
            pos.high_water = ltp
        return ltp

    def _exit_signal(self, pos: Position, now_dt: datetime) -> Signal | None:
        """The strategy's invalidation signal for this position, if its candles are cached.

        Candles are fetched in step 4, so on the very first tick after a restart there
        are none yet. That is fine: it only delays the discretionary exit by one tick,
        and never the stop.
        """
        series = self._series.get(pos.leg.underlying)
        if series is None or not series.ok:
            return None
        try:
            return self.strategy.exit_signal(
                pos, series.candles, now_ts=clock.epoch_s(now_dt)
            )
        except IndbotError as exc:
            log.warning("exit_signal for %s failed: %s", pos.leg.label,
                        getattr(exc, "message", exc))
            return None

    def _exit_due(self, pos: Position) -> bool:
        """False while a previous exit attempt is still backing off."""
        with self._lock:
            state = self._exit_fail.get(pos.id)
        if not state:
            return True
        next_try = int(state.get("next_try_at", 0))
        if clock.epoch_s(clock.now()) < next_try:
            return False
        return True

    def _exit(
        self,
        pos: Position,
        reason: str,
        *,
        ltp: float = 0.0,
        signal: Signal | None = None,
    ) -> dict[str, Any] | None:
        """Sell the whole position, journal the close, and refresh the learner.

        `signal_ts` is deliberately NOT passed to the broker: an exit must stay
        retryable, because the broker's in-process dedupe refusing a repeated SELL
        would strand the position it is trying to protect.
        """
        with self._lock:
            if pos.id in self._exiting:
                log.warning("an exit for %s is already in flight; not sending a second", pos.id)
                return None
            self._exiting.add(pos.id)
        try:
            # Deliberately NOT under `_trade_lock`: panic() reaches this and must not be
            # able to block on a loop thread wedged inside an order poll. The `_exiting`
            # set claimed above is the real guard — it is what stops a second SELL
            # turning one long lot into a short one.
            with self._exit_authority(reason):
                qty = int(pos.qty)
                if not (pos.leg.ltp > 0 or pos.leg.bid > 0 or pos.leg.ask > 0):
                    # No price means no limit order can be built. Refusing is the safe
                    # branch — a blind MARKET order into an illiquid option book is how a
                    # 40 rupee exit fills at 4 — but the position stays open, so this is
                    # recorded as a fault and retried, never swallowed.
                    exc = OrderError(
                        f"cannot price an exit for {pos.leg.label}: no LTP, bid or ask is "
                        f"available, so no limit order can be built. THE POSITION IS STILL "
                        f"OPEN — check the market data feed or square off on INDmoney.",
                        security_id=pos.leg.security_id, txn_type="SELL", stranded=True,
                    )
                    self._handle_exit_failure(pos, reason, exc)
                    raise exc
                remark = make_remark(f"x-{reason}", pos.leg.side, pos.leg.underlying)
                log.warning(
                    "EXIT %s: selling %d units of %s (mark %.2f, stop %.2f, target %.2f)",
                    reason, qty, pos.leg.label, ltp, pos.stop_price, pos.target_price,
                )
                try:
                    res = self.broker.sell(pos.leg, qty, remark=remark, signal_ts=None)
                except OrderError as exc:
                    self._handle_exit_failure(pos, reason, exc)
                    raise
                self._apply_exit(pos, reason, res)
        finally:
            with self._lock:
                self._exiting.discard(pos.id)

        if signal is not None:
            self._record(signal, taken=True, reason=f"Exit taken: {reason}.", trade_id=pos.id)
        return {
            "id": pos.id,
            "label": pos.leg.label,
            "reason": reason,
            "exit_price": pos.exit_price,
            "pnl": round(pos.pnl(), 2),
        }

    def _apply_exit(self, pos: Position, reason: str, res: dict[str, Any]) -> None:
        """Write the fill onto the position, journal it, and drop it from the book."""
        price = float(res.get("price") or 0.0)
        pos.exit_price = price
        pos.exit_ts = clock.epoch_s(clock.now())
        pos.exit_order_id = str(res.get("order_id") or "")
        pos.exit_reason = reason
        pos.costs = float(pos.costs or 0.0) + float(res.get("costs") or 0.0)

        with self._lock:
            self._positions.pop(pos.id, None)
            self._exit_fail.pop(pos.id, None)

        try:
            self.journal.close_trade(pos)
        except IndbotError as exc:
            # The money has already moved. Losing the row would understate the day and
            # leave the day-halt switched off, so this is an error, never a shrug.
            log.critical("could not journal the close of %s: %s", pos.id, exc, exc_info=True)
            self._record_error(exc, where=f"journal:close:{pos.id}")

        log.warning(
            "CLOSED %s at %s (%s): P&L %s on %s",
            pos.leg.label, inr(price, decimals=2), reason,
            inr(pos.pnl()), inr(pos.cost_basis),
        )
        with contextlib.suppress(Exception):
            self.learner.refresh()  # the bucket just got one more sample

    def _handle_exit_failure(self, pos: Position, reason: str, exc: OrderError) -> None:
        """Record a failed exit, bank any partial fill, and schedule the retry.

        A partial exit means real units changed hands. The realised part is credited
        into `pos.costs` as a negative cost and `pos.qty` is reduced to what is still
        open, so the eventual full close still totals exactly:
        `(exit - entry) * remaining + (partial - entry) * filled - real costs`.
        """
        context = getattr(exc, "context", {}) or {}
        payload = getattr(exc, "payload", None) or {}
        filled = _as_int(context.get("filled_qty"))
        remaining = _as_int(context.get("remaining_qty"))
        if filled > 0 and 0 < remaining < int(pos.qty):
            avg = _as_float(payload.get("avg_price") if isinstance(payload, dict) else 0.0)
            if avg > 0:
                pos.costs = float(pos.costs or 0.0) - (avg - float(pos.entry_price)) * filled
                log.critical(
                    "PARTIAL EXIT banked for %s: %d units at %.2f, %d units still open",
                    pos.leg.label, filled, avg, remaining,
                )
            pos.qty = remaining

        with self._lock:
            state = self._exit_fail.setdefault(pos.id, {"attempts": 0})
            state["attempts"] = int(state.get("attempts", 0)) + 1
            state["reason"] = reason
            state["error"] = getattr(exc, "message", str(exc))
            state["at"] = clock.now().isoformat()
            state["next_try_at"] = clock.epoch_s(clock.now()) + int(EXIT_RETRY_S)
        log.critical(
            "EXIT FAILED for %s (attempt %d, reason=%s): %s — the position is STILL OPEN; "
            "retrying in %.0fs. Verify it on INDmoney.",
            pos.leg.label, self._exit_fail[pos.id]["attempts"], reason,
            getattr(exc, "message", exc), EXIT_RETRY_S,
        )
        self._record_error(exc, where=f"exit:{pos.id}")

    @contextlib.contextmanager
    def _exit_authority(self, why: str) -> Iterator[bool]:
        """Temporarily satisfy LiveBroker's arm gate so an EXIT can be sent.

        LiveBroker refuses a SELL as well as a BUY unless `cfg.armed is True`, so a
        disarmed bot cannot square off its own live position. The gate exists to stop
        orders nobody asked for, and a flatten is the most deliberate order there is —
        so it is granted here, loudly, for the length of the exit only. Entries are
        blocked for that whole window so a BUY cannot ride in on the temporary arm.
        """
        with self._lock:
            self._exit_window += 1
        rearmed = False
        try:
            if str(getattr(self.cfg, "mode", "")).strip() == "live" and self.cfg.armed is not True:
                log.critical(
                    "temporarily ARMING to send a %s exit on a live position — no entry can "
                    "be taken while this lasts", why,
                )
                self.cfg.set_armed(True, confirm=ARM_PHRASE)
                rearmed = True
            yield rearmed
        finally:
            if rearmed:
                self.cfg.set_armed(False)
            with self._lock:
                self._exit_window = max(0, self._exit_window - 1)

    # -------------------------------------------------------- step 3: gates

    def _halt_state(self) -> tuple[bool, str]:
        """(halted, reason) for the current mode, from the persisted day halt."""
        try:
            return self.risk.day_halted(self.cfg.mode)
        except IndbotError as exc:
            # Safe branch: if we cannot tell, treat the day as halted. A missed day
            # costs nothing; a missed halt costs rupees.
            log.error("halt check failed: %s", getattr(exc, "message", exc))
            self._record_error(exc, where="halt")
            return True, f"Could not read today's halt flag ({getattr(exc, 'message', exc)})."

    # --------------------------------------------------------- step 4: scan

    def _scan(self, now_dt: datetime) -> tuple[list[tuple[Signal, BucketVerdict]], list[str]]:
        """Refresh candles where due, evaluate every underlying, return scored signals.

        Per-underlying failures are recorded and skipped: one bad instrument must not
        stop the scan, let alone the loop.
        """
        names = [str(n).strip().upper() for n in (self.cfg.underlyings or [])]
        self._refresh_candles(names, now_dt)

        out: list[tuple[Signal, BucketVerdict]] = []
        scanned: list[str] = []
        for name in names:
            series = self._series.get(name)
            if series is None or not series.ok:
                self._error_view(name, series)
                continue
            try:
                signals = self.strategy.evaluate(
                    name, series.candles, now_ts=clock.epoch_s(now_dt)
                )
            except IndbotError as exc:
                log.error("strategy failed on %s: %s", name, getattr(exc, "message", exc))
                self._record_error(exc, where=f"strategy:{name}")
                continue
            adjusted: list[tuple[Signal, BucketVerdict]] = []
            for sig in signals:
                try:
                    tuned, verdict = self.learner.adjust(sig)
                except Exception as exc:  # noqa: BLE001 - statistics must not block the gate
                    log.warning("learner.adjust failed for %s: %s", sig.bucket, exc)
                    tuned, verdict = sig, self.learner.verdict(sig.bucket)
                adjusted.append((tuned, verdict))
            adjusted.sort(key=lambda pair: (-pair[0].score, pair[0].setup))
            out.extend(adjusted)
            scanned.append(name)
            self._update_view(name, series, [s for s, _ in adjusted], now_dt)

        out.sort(key=lambda pair: (-pair[0].score, pair[0].underlying, pair[0].setup))
        return out, scanned

    def _error_view(self, name: str, series: _Series | None) -> None:
        """Replace an underlying's card with its data error rather than stale numbers.

        A dashboard that keeps showing last tick's spot through an outage is lying, and
        the point of this panel is that the user can trust what it says.
        """
        meta = UNDERLYINGS.get(name, {})
        with self._lock:
            self._views[name] = {
                "name": name,
                "label": meta.get("label", name),
                "spot": None,
                "change_pct": None,
                "regime": None,
                "or_high": None,
                "or_low": None,
                "last_candle_ts": None,
                "last_candle_at": None,
                "bars": 0,
                "lot_size": self._lot_sizes.get(name),
                "enabled": name in (self.cfg.underlyings or []),
                "data_error": (series.error if series else "") or "no candles yet",
                "signals": [],
                "updated_at": clock.now().isoformat(),
            }

    def _refresh_candles(self, names: Sequence[str], now_dt: datetime) -> None:
        """Fetch the due underlyings in ONE batched historical call.

        `historical` takes five scrip codes per request, which is exactly the number of
        underlyings, so a whole scan costs one data call instead of five. Anything that
        is not due is left alone: the data limit is 5 calls a second and refetching
        every tick would 429 long before it would find a new candle.
        """
        if self.chain is None or self._client is None:
            return
        due: dict[str, str] = {}
        for name in names:
            series = self._series.setdefault(name, _Series())
            if series.ok and clock.epoch_s(now_dt) < series.next_fetch_at:
                continue
            try:
                code = self._scrip_code(name)
            except IndbotError as exc:
                series.error = getattr(exc, "message", str(exc))
                series.next_fetch_at = clock.epoch_s(now_dt) + CANDLE_RETRY_S
                log.error("cannot build a scrip code for %s: %s", name, series.error)
                self._record_error(exc, where=f"candles:{name}")
                continue
            due[name] = code
        if not due:
            return

        start_dt, end_dt = self._history_window(now_dt)
        try:
            data = self._client.historical(
                self.strategy.interval,
                list(due.values()),
                clock.epoch_ms(start_dt),
                clock.epoch_ms(end_dt),
            )
        except IndbotError as exc:
            # One failed batch is one failed fetch, not a dead loop: back every due
            # underlying off and try again on the next tick.
            log.error("historical fetch failed for %s: %s", ",".join(due),
                      getattr(exc, "message", exc))
            self._record_error(exc, where="candles")
            for name in due:
                series = self._series[name]
                series.error = getattr(exc, "message", str(exc))
                series.next_fetch_at = clock.epoch_s(now_dt) + CANDLE_RETRY_S
            return

        for name, code in due.items():
            series = self._series[name]
            candles = list(data.get(code) or [])
            if not candles:
                series.error = f"no candles for {code} in {self.strategy.interval}"
                log.warning("%s: %s", name, series.error)
            elif (band_error := self._price_band_error(name, candles[-1])):
                # A wrong instrument id usually still returns perfectly good candles
                # — for the wrong index. That is the failure that cost four of five
                # underlyings here, and it is invisible unless the level is checked.
                # Refuse the data rather than trade a signal computed on NIFTY 100.
                series.error = band_error
                series.candles = []
                log.error("%s: %s", name, band_error)
                self._record_error(DataError(band_error, underlying=name, code=code),
                                   where="candles")
            else:
                series.candles = candles
                series.last_ts = int(candles[-1].get("ts", 0) or 0)
                series.error = ""
            series.fetched_ts = clock.epoch_s(now_dt)
            series.fetches += 1
            self._schedule_fetch(series, now_dt)

    @staticmethod
    def _price_band_error(name: str, candle: Candle) -> str:
        """Non-empty when `candle` cannot plausibly belong to `name`.

        Guards the one failure the API cannot report: a valid security id for the
        wrong instrument. Every underlying carries a `price_band` in UNDERLYINGS;
        a close outside it means the id is pointing somewhere else.
        """
        meta = UNDERLYINGS.get(name) or {}
        band = meta.get("price_band")
        if not band:
            return ""
        try:
            close = float(candle.get("c") or 0.0)
        except (TypeError, ValueError):
            return ""
        if close <= 0:
            return ""
        low, high = float(band[0]), float(band[1])
        if low <= close <= high:
            return ""
        return (
            f"{name} came back at {close:,.2f}, outside its expected {low:,.0f}-{high:,.0f} "
            f"band. That almost always means the security id points at a different index "
            f"— refusing the data rather than trading a signal computed on the wrong one. "
            f"Run `python3 probe.py codes` and check the id against the index master."
        )

    def _schedule_fetch(self, series: _Series, now_dt: datetime) -> None:
        """Set the next fetch to just after the next candle should close.

        When the feed is behind — the bar we expected is not in the response yet — it
        retries sooner, but never sooner than CANDLE_RETRY_S, so a late feed cannot
        turn into a tight loop against a 5-per-second limit.
        """
        minutes = self.interval_minutes
        step = minutes * 60
        current_open = clock.epoch_s(clock.candle_open_time(now_dt, minutes))
        want_closed = current_open - step          # newest bar that should exist
        after_next_close = current_open + step + CANDLE_LAG_S
        if series.last_ts >= want_closed:
            series.next_fetch_at = after_next_close
        else:
            series.next_fetch_at = min(
                clock.epoch_s(now_dt) + CANDLE_RETRY_S, after_next_close
            )

    def _scrip_code(self, name: str) -> str:
        """`NIDX_40000001` — the underlying's history code for the REST surface.

        The prefix is the INDEX segment (`NIDX` on NSE, `BIDX` on BSE), NOT the
        exchange. `NSE_40000001` is what this used to build and it returns
        `400 Invalid scrip codes`: the scrip-code surface wants the segment the
        instrument lives in, and an index does not live in the equity segment.
        """
        if self.chain is None:
            raise DataError("no API client, so no instrument can be resolved", underlying=name)
        meta = UNDERLYINGS.get(name)
        if not meta:
            raise DataError(
                f"{name!r} is not one of the configured underlyings "
                f"({', '.join(sorted(UNDERLYINGS))})",
                underlying=name,
            )
        segment = str(meta.get("hist_segment") or ("BIDX" if meta["exchange"] == "BSE" else "NIDX"))
        return f"{segment}_{self.chain.underlying_scrip(name)}"

    def _history_window(self, now_dt: datetime) -> tuple[datetime, datetime]:
        """The [start, end] window to request, wide enough to warm up every indicator.

        EMA50 and ADX(14) need about 50 bars, so the window spans enough SESSIONS to
        supply them at the configured interval, clamped to the API's per-interval
        maximum — a wider request comes back silently truncated rather than as an error.
        """
        minutes = max(1, self.interval_minutes)
        per_session = max(1, SESSION_MINUTES // minutes)
        sessions = max(MIN_SESSIONS, (MIN_BARS // per_session) + 2)
        day = now_dt.date()
        for _ in range(sessions - 1):
            day = clock.prev_trading_day(day)
        start = clock.session_open(day)
        max_days = clock.INTERVAL_MAX_DAYS.get(self.strategy.interval)
        if max_days:
            floor = now_dt - timedelta(days=max(1, int(max_days) - 1))
            start = max(start, floor)
        return start, now_dt

    def _update_view(
        self,
        name: str,
        series: _Series,
        signals: list[Signal],
        now_dt: datetime,
    ) -> None:
        """Rebuild one underlying's card for the snapshot: spot, regime, OR, signals."""
        candles = series.candles
        last = candles[-1] if candles else None
        spot = float(last.get("c", 0.0) or 0.0) if last else 0.0
        try:
            regime = indicators.regime(candles)
        except Exception as exc:  # noqa: BLE001 - a view must never break the tick
            log.debug("regime failed for %s: %s", name, exc)
            regime = "range"
        try:
            or_minutes = int(getattr(self.cfg, "or_minutes", 15) or 15)
            opening = indicators.opening_range(candles, or_minutes)
        except Exception as exc:  # noqa: BLE001
            log.debug("opening_range failed for %s: %s", name, exc)
            opening = None
        meta = UNDERLYINGS.get(name, {})
        view = {
            "name": name,
            "label": meta.get("label", name),
            "spot": round(spot, 2),
            "change_pct": self._change_pct(candles, spot),
            "regime": regime,
            "or_high": round(opening[0], 2) if opening else None,
            "or_low": round(opening[1], 2) if opening else None,
            "last_candle_ts": series.last_ts or None,
            "last_candle_at": clock.to_ist(series.last_ts).strftime("%H:%M")
            if series.last_ts else None,
            "bars": len(candles),
            "lot_size": self._lot_sizes.get(name),
            "enabled": name in (self.cfg.underlyings or []),
            "data_error": series.error or None,
            "signals": [s.as_dict() for s in signals],
            "updated_at": now_dt.isoformat(),
        }
        with self._lock:
            self._views[name] = view

    @staticmethod
    def _change_pct(candles: list[Candle], spot: float) -> float | None:
        """Percent change against the previous session's last close, or None.

        Needs a multi-day series; returns None rather than a made-up zero when the
        window only covers today, because a zero would read as a flat market.
        """
        if not candles or spot <= 0:
            return None
        last_day = clock.to_ist(int(candles[-1].get("ts", 0) or 0)).date()
        prev_close: float | None = None
        for candle in reversed(candles):
            ts = int(candle.get("ts", 0) or 0)
            if clock.to_ist(ts).date() < last_day:
                prev_close = float(candle.get("c", 0.0) or 0.0)
                break
        if prev_close is None or prev_close <= 0:
            return None
        return round((spot - prev_close) / prev_close * 100.0, 2)

    # -------------------------------------------------------- step 5: entry

    def _consider_entries(
        self,
        signals: list[tuple[Signal, BucketVerdict]],
        now_dt: datetime,
    ) -> dict[str, Any] | None:
        """Journal every signal, then try to take the single best one.

        The runners-up are recorded with `taken=False` and the reason they lost, so
        the "why no trade" panel accounts for all of them, not just the winner.
        """
        best_sig, _ = signals[0]
        for sig, verdict in signals[1:]:
            self._record(
                sig,
                taken=False,
                reason=(
                    f"A higher-scoring signal was evaluated first this candle: "
                    f"{best_sig.underlying} {best_sig.setup} {best_sig.side} at "
                    f"{best_sig.score:.2f} against this one's {sig.score:.2f}. The bot takes "
                    f"one trade at a time."
                ),
            )
        return self._try_entry(best_sig, now_dt)

    def _try_entry(self, sig: Signal, now_dt: datetime) -> dict[str, Any] | None:
        """Pick a strike, run the risk gate, and buy one lot when everything says yes."""
        if not self._signal_fresh(sig, now_dt):
            age = clock.epoch_s(now_dt) - (sig.ts + self.interval_minutes * 60)
            self._record(
                sig,
                taken=False,
                reason=(
                    f"The {clock.to_ist(sig.ts).strftime('%H:%M')} candle that triggered this "
                    f"closed {max(0, age) // 60} minute(s) ago, more than one "
                    f"{self.strategy.interval} bar. The price that justified the trigger is "
                    f"gone, so the entry is skipped rather than chased."
                ),
            )
            return None

        if self._exit_window > 0:
            self._record(
                sig, taken=False,
                reason="An exit is being sent right now; no entry is taken in that window.",
            )
            return None

        if self.chain is None:
            raise DataError(
                "no API client, so no option chain can be read", underlying=sig.underlying
            )
        try:
            leg, pick_reasons = self.chain.pick(
                sig.underlying, sig.side, spot=sig.spot, stop_pct=sig.stop_pct
            )
        except IndbotError as exc:
            message = getattr(exc, "message", str(exc))
            log.error("chain pick failed for %s %s: %s", sig.underlying, sig.side, message)
            self._record_error(exc, where=f"chain:{sig.underlying}")
            self._record(
                sig, taken=False,
                reason=f"Could not read the {sig.underlying} option chain: {message}",
            )
            return None

        if leg is None:
            # The chain picker's rejection lines name the strike and the number that
            # failed. They are the most useful thing on the dashboard — keep them whole.
            self._record(
                sig, taken=False,
                reason="No tradable strike. " + " ".join(pick_reasons),
            )
            return None

        verdict = self.risk.check_entry(
            sig,
            leg,
            open_positions=self._open_positions(),
            mode=self.cfg.mode,
            funds=self._fetch_funds(),
            blocked_buckets=self._blocked_buckets(),
            now=now_dt,
        )
        if not verdict.ok:
            self._record(
                sig, taken=False,
                reason=" ".join(verdict.reasons) or "Blocked by the risk gate.",
            )
            return None

        return self._enter(sig, leg, pick_reasons, verdict, now_dt)

    def _signal_fresh(self, sig: Signal, now_dt: datetime) -> bool:
        """True when the triggering candle closed within the last candle interval."""
        step = self.interval_minutes * 60
        if step <= 0:
            return False
        closed_at = int(sig.ts) + step
        return (clock.epoch_s(now_dt) - closed_at) <= step * MAX_SIGNAL_AGE_INTERVALS

    def _enter(
        self,
        sig: Signal,
        leg: OptionLeg,
        pick_reasons: list[str],
        verdict: RiskVerdict,
        now_dt: datetime,
    ) -> dict[str, Any] | None:
        """Buy one lot, build the Position, and journal the open."""
        qty = int(leg.lot_size)
        remark = make_remark(sig.setup, sig.side, sig.underlying, sig.ts)
        with self._trade_lock:
            # Re-checked inside the lock and as late as possible: panic() and
            # _exit_authority can raise this flag from another thread at any moment.
            if self._exit_window > 0:
                log.warning("entry abandoned: an exit was started while the order was prepared")
                return None
            log.warning(
                "ENTRY: buying 1 lot (%d units) of %s at about %s on %s/%s score %.2f",
                qty, leg.label, inr(leg.ask or leg.ltp, decimals=2), sig.underlying,
                sig.setup, sig.score,
            )
            res = self.broker.buy(leg, qty, remark=remark, signal_ts=sig.ts)

        price = float(res.get("price") or 0.0)
        filled = int(res.get("filled_qty") or 0) or qty
        if price <= 0:
            # Units changed hands even though the fill price came back unusable. Dropping
            # the position here would leave a real lot with nothing watching its stop,
            # which is strictly worse than an estimated entry price — so it is tracked on
            # the best price we have and the discrepancy is shouted about.
            price = float(leg.ask or leg.ltp or leg.mid or 0.0)
            log.critical(
                "BUY of %s reported no fill price (%s). Tracking the position at the "
                "estimated entry %s so its stop is still managed — VERIFY the real average "
                "price on INDmoney.", leg.label, res, inr(price, decimals=2),
            )
            self._record_error(
                OrderError(
                    f"BUY of {leg.label} filled without a usable price; the position is "
                    f"tracked at an estimated entry of {inr(price, decimals=2)}. Verify it "
                    f"on INDmoney.",
                    payload=res, security_id=leg.security_id, txn_type="BUY",
                ),
                where="entry",
            )
        if price <= 0:
            # No price anywhere. Refuse before a Position with a zero entry makes every
            # downstream ratio — stop, target, P&L percentage — meaningless.
            raise OrderError(
                f"BUY of {leg.label} reported no fill price and the leg has none either: {res}",
                payload=res, security_id=leg.security_id, txn_type="BUY",
            )
        if filled < qty:
            log.critical(
                "PARTIAL ENTRY on %s: %d of %d units filled. The position is sized from the "
                "fill, so the exit sells exactly what is held.",
                leg.label, filled, qty,
            )

        stop_price = max(0.01, price * (1.0 - float(sig.stop_pct)))
        target_price = price * (1.0 + float(sig.target_pct))
        pos = Position(
            id=self._trade_id(sig),
            leg=replace(leg, ltp=price),
            signal_setup=sig.setup,
            bucket=sig.bucket,
            qty=filled,
            entry_price=price,
            entry_ts=clock.epoch_s(now_dt),
            entry_order_id=str(res.get("order_id") or ""),
            stop_price=stop_price,
            target_price=target_price,
            mode=self.cfg.mode,
            high_water=price,
            costs=float(res.get("costs") or 0.0),
        )
        with self._lock:
            self._positions[pos.id] = pos

        try:
            self.journal.open_trade(pos, self.cfg.mode)
        except IndbotError as exc:
            # The lot is real whether or not the row was written. Keep managing it.
            log.critical("could not journal the open of %s: %s", pos.id, exc, exc_info=True)
            self._record_error(exc, where=f"journal:open:{pos.id}")

        self._record(
            sig, taken=True, trade_id=pos.id,
            reason=" ".join(pick_reasons[:1] + verdict.reasons[:1]),
        )
        log.warning(
            "OPEN %s x%d units at %s — stop %s, target %s, risk %s",
            leg.label, filled, inr(price, decimals=2), inr(stop_price, decimals=2),
            inr(target_price, decimals=2), inr((price - stop_price) * filled),
        )
        return {
            "id": pos.id,
            "label": leg.label,
            "setup": sig.setup,
            "qty": filled,
            "entry_price": price,
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
        }

    @staticmethod
    def _trade_id(sig: Signal) -> str:
        """A unique, sortable, human-readable trade id."""
        stamp = clock.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{sig.underlying}-{sig.setup}-{uuid.uuid4().hex[:6]}"

    # ----------------------------------------------------------- journalling

    def _record(
        self,
        sig: Signal,
        *,
        taken: bool,
        reason: str = "",
        trade_id: str | None = None,
    ) -> None:
        """Journal a signal once per candle, with the exact reason it was not taken.

        Deduplicated on `(underlying, setup, side, candle ts)`: with a 20 second poll
        and a 5 minute bar the same trigger is re-evaluated fifteen times, and the
        "why no trade" feed is unreadable if every pass writes a row.
        """
        key = (sig.underlying, sig.setup, sig.side, int(sig.ts))
        with self._lock:
            if key in self._seen_signals:
                return
            self._seen_signals.add(key)
        try:
            self.journal.record_signal(
                sig, taken=taken, reason=reason, mode=self.cfg.mode, trade_id=trade_id
            )
        except IndbotError as exc:
            log.error("could not journal a signal: %s", getattr(exc, "message", exc))
            self._record_error(exc, where="journal:signal")
        if not taken and reason:
            log.info("no trade on %s %s %s: %s", sig.underlying, sig.setup, sig.side, reason)

    def _blocked_buckets(self) -> list[str]:
        """Buckets the learner refuses, tolerating a statistics failure."""
        try:
            return self.learner.blocked_buckets()
        except Exception as exc:  # noqa: BLE001 - learning must never block the gate
            log.warning("could not read the blocked buckets: %s", exc)
            return []

    # ------------------------------------------------------- account / funds

    def _fetch_funds(self, *, force: bool = False) -> dict[str, Any] | None:
        """The cached /funds block, refreshed at most once a minute.

        A stale value is kept on failure rather than blanked: the risk gate blocks a
        live entry when the buying balance is unreadable, and a blanked cache would
        turn one hiccup into a blocked session.
        """
        if self._client is None:
            return None
        now_s = clock.epoch_s(clock.now())
        if not force and self._funds is not None and (now_s - self._funds_ts) < FUNDS_TTL_S:
            return self._funds
        try:
            funds = self._client.funds()
        except IndbotError as exc:
            log.warning("could not read funds: %s", getattr(exc, "message", exc))
            self._record_error(exc, where="funds")
            return self._funds
        with self._lock:
            self._funds = funds
            self._funds_ts = now_s
        return funds

    def _fetch_account(self, *, force: bool = False) -> dict[str, Any] | None:
        """The cached account profile — what the live-mode modal names."""
        if self._client is None:
            return None
        now_s = clock.epoch_s(clock.now())
        if not force and self._account is not None and (now_s - self._account_ts) < PROFILE_TTL_S:
            return self._account
        try:
            profile = self._client.profile()
        except IndbotError as exc:
            log.warning("could not read the account profile: %s", getattr(exc, "message", exc))
            return self._account
        with self._lock:
            self._account = profile
            self._account_ts = now_s
        return profile

    def _account_label(self) -> str:
        """A short, loggable account identifier, or "unknown account"."""
        profile = self._account or self._fetch_account()
        if not isinstance(profile, dict):
            return "unknown account"
        for key in ("client_id", "clientId", "ucc", "user_id", "email", "name"):
            value = profile.get(key)
            if value:
                return str(value)
        return "unknown account"

    # ------------------------------------------------------------- recovery

    def _recover_open_positions(self) -> None:
        """Re-adopt any position the journal still has open in this mode.

        A restart with a live lot on the book is a real scenario, and a position whose
        stop nobody is watching is the expensive kind of bug. The leg is rebuilt from
        the trade row; its price is unknown until the first `_mark`, which is why the
        mark writes the LTP back onto the leg before any exit is attempted.
        """
        try:
            rows = self.journal.open_trades(mode=self.cfg.mode)
        except IndbotError as exc:
            log.error("could not read open trades: %s", getattr(exc, "message", exc))
            self._record_error(exc, where="recovery")
            return
        for row in rows:
            trade_id = str(row.get("id") or "")
            if not trade_id or trade_id in self._positions:
                continue
            try:
                pos = self._position_from_row(row)
            except (KeyError, TypeError, ValueError) as exc:
                log.error("could not rebuild the open trade %s: %s", trade_id, exc)
                continue
            with self._lock:
                self._positions[pos.id] = pos
            log.critical(
                "RECOVERED an open %s position from the journal: %s x%d units entered at %s "
                "(stop %s, target %s). The bot is managing it again from this tick.",
                pos.mode, pos.leg.label, pos.qty, inr(pos.entry_price, decimals=2),
                inr(pos.stop_price, decimals=2), inr(pos.target_price, decimals=2),
            )

    @staticmethod
    def _position_from_row(row: dict[str, Any]) -> Position:
        """Rebuild a Position from a journal `trades` row (no live price yet)."""
        underlying = str(row.get("underlying") or "")
        meta = UNDERLYINGS.get(underlying, {})
        lot_size = _as_int(row.get("lot_size"))
        leg = OptionLeg(
            underlying=underlying,
            expiry=str(row.get("expiry") or ""),
            strike=_as_float(row.get("strike")),
            side=str(row.get("side") or "CE"),
            security_id=str(row.get("security_id") or ""),
            trading_symbol=str(row.get("trading_symbol") or ""),
            lot_size=lot_size,
            exchange=str(row.get("exchange") or meta.get("exchange") or "NSE"),
            ltp=0.0,
        )
        entry_price = _as_float(row.get("entry_price"))
        return Position(
            id=str(row.get("id") or ""),
            leg=leg,
            signal_setup=str(row.get("setup") or ""),
            bucket=str(row.get("bucket") or ""),
            qty=_as_int(row.get("qty")) or lot_size,
            entry_price=entry_price,
            entry_ts=_as_int(row.get("entry_ts")),
            entry_order_id=str(row.get("entry_order_id") or ""),
            stop_price=_as_float(row.get("stop_price")),
            target_price=_as_float(row.get("target_price")),
            mode=str(row.get("mode") or "paper"),
            high_water=_as_float(row.get("high_water")) or entry_price,
            costs=_as_float(row.get("costs")),
        )

    # -------------------------------------------------------------- controls

    def set_mode(self, mode: str) -> None:
        """Switch trading mode, flattening any open position before leaving live.

        Refuses rather than abandoning: if the flatten fails the mode is left alone,
        so the live broker — and with it the only working exit path — stays in place.
        """
        wanted = str(mode or "").strip().lower()
        if wanted not in MODES:
            raise ConfigError(
                f"mode must be one of {', '.join(MODES)}, got {mode!r}",
                key="mode", allowed=", ".join(MODES),
            )
        current = str(self.cfg.mode or "").strip().lower()
        if wanted == current:
            log.info("mode is already %s", wanted)
            return

        with self._trade_lock:
            if current == "live":
                open_now = self._open_positions()
                if open_now:
                    log.warning(
                        "leaving live mode with %d position(s) open — flattening first",
                        len(open_now),
                    )
                    flattened, failed, _skipped = self._flatten_all("mode_change")
                    if failed:
                        raise OrderError(
                            f"refusing to leave live mode: {len(failed)} position(s) could not "
                            f"be squared off ("
                            + "; ".join(f"{f['label']}: {f['error']}" for f in failed)
                            + "). The live broker is kept so the exit can be retried — use "
                            "panic() or close them on INDmoney, then switch mode.",
                            payload={"flattened": flattened, "failed": failed},
                        )

            # Commit the mode only once a broker for it exists. Writing cfg.mode
            # first and building the broker after left mode="live" with no broker
            # when construction failed (no token), so the dashboard showed a red
            # LIVE rail for a bot that could not reach the broker at all. The mode
            # a user sees must always be one the engine can actually act in.
            changed = self.cfg.update({"mode": wanted})
            self.cfg.set_armed(False)
            try:
                broker = make_broker(
                    self.cfg, self._client, allow_closed_market=self._allow_closed_market
                )
            except Exception:
                self.cfg.update({"mode": current})
                self.cfg.set_armed(False)
                log.error("mode %s -> %s failed to build a broker; rolled back to %s",
                          current, wanted, current)
                self._publish()
                raise
            self.journal.default_mode = wanted
            self.broker = broker
            self._broker_mode = wanted
        with contextlib.suppress(Exception):
            self.learner.refresh()   # verdicts are per mode
        log.warning(
            "MODE %s -> %s (disarmed; changed %s)", current, wanted, ",".join(changed) or "nothing"
        )
        self._publish()

    def arm(self, confirm: str) -> None:
        """Arm live trading. `confirm` must be exactly "ARM LIVE TRADING".

        Nothing arms on a click: the phrase is the whole point of the gate, and the
        limits it is arming under are logged so the record says what was agreed to.
        """
        if confirm != ARM_PHRASE:
            raise ConfigError(
                f'to arm, confirm must be exactly "{ARM_PHRASE}"',
                key="confirm", allowed=f'confirm="{ARM_PHRASE}"',
            )
        if str(self.cfg.mode).strip().lower() == "live" and self._client is None:
            raise ConfigError(
                "cannot arm live trading without an INDstocks access token: there is no "
                "connection to place or to cancel an order with.",
                key="armed",
            )
        self.cfg.set_armed(True, confirm=confirm)
        if str(self.cfg.mode).strip().lower() != "live":
            log.warning(
                "armed while mode=%s — switching mode will clear this, so arm AFTER "
                "selecting live", self.cfg.mode,
            )
        log.warning(
            "ARMED LIVE TRADING on %s | per-trade loss %s, daily loss %s, max %d trade(s)/day, "
            "max %d open, max %d lot(s), stop %.0f%%, target %.0f%%, no entry before %s or "
            "after %s, flat by %s, underlyings %s",
            self._account_label(),
            inr(self.cfg.max_trade_loss), inr(self.cfg.max_daily_loss),
            self.cfg.max_trades_per_day, self.cfg.max_open_positions, self.cfg.max_lots,
            self.cfg.stop_pct * 100.0, self.cfg.target_pct * 100.0,
            self.cfg.no_entry_before, self.cfg.no_entry_after, self.cfg.flat_by,
            ",".join(self.cfg.underlyings),
        )
        self._publish()

    def disarm(self, *, force: bool = False) -> None:
        """Disarm live trading, squaring off any open live position first.

        LiveBroker refuses a SELL as well as a BUY when disarmed, so disarming on top
        of an open live position would leave it with no automated exit. The flatten
        therefore happens while still armed, and a failed flatten refuses the disarm so
        the exit can be retried — pass `force=True` to disarm anyway and square off by
        hand.
        """
        live_open = [p for p in self._open_positions() if p.mode == "live"]
        if live_open and str(self.cfg.mode).strip().lower() == "live" and not force:
            log.warning("disarming with %d live position(s) open — flattening first",
                        len(live_open))
            flattened, failed, _skipped = self._flatten_all("disarm")
            if failed:
                raise OrderError(
                    f"refusing to disarm: {len(failed)} live position(s) are still open ("
                    + "; ".join(f"{f['label']}: {f['error']}" for f in failed)
                    + "). A disarmed bot cannot send their exit. Retry, use panic(), or "
                    "disarm(force=True) and square off on INDmoney.",
                    payload={"flattened": flattened, "failed": failed},
                )
        self.cfg.set_armed(False)
        if force and live_open:
            log.critical(
                "DISARMED by force with %d live position(s) open (%s) — the bot can no longer "
                "exit them. Square off on INDmoney.",
                len(live_open), ", ".join(p.leg.label for p in live_open),
            )
        else:
            log.warning("DISARMED — no live order will be sent")
        self._publish()

    def panic(self) -> dict[str, Any]:
        """Flatten everything now, halt the day, disarm, and report what happened.

        Does not go through the tick, does not take `_trade_lock` and never waits on
        the loop thread, so it still works when the loop is wedged. The flatten runs
        BEFORE the disarm because LiveBroker refuses a SELL from a disarmed config; an
        exit already in flight is reported rather than duplicated, because a second
        SELL on one lot opens a short.
        """
        at = clock.now()
        log.critical("PANIC pressed at %s — flattening everything and halting the day",
                     at.isoformat())
        result: dict[str, Any] = {
            "at": at.isoformat(),
            "mode": self.cfg.mode,
            "flattened": [],
            "failed": [],
            "skipped": [],
            "halted": False,
            "armed": self.cfg.armed,
            "broker_positions": [],
            "notes": [],
        }
        with self._exit_authority("panic") as rearmed:
            if rearmed:
                result["notes"].append(
                    "The bot was disarmed, so it re-armed itself for the length of the exit "
                    "only — a disarmed LiveBroker refuses a SELL as well as a BUY."
                )
            flattened, failed, skipped = self._flatten_all("panic")
            result["flattened"] = flattened
            result["failed"] = failed
            result["skipped"] = skipped

            try:
                self.risk.halt_day(
                    self.cfg.mode,
                    f"PANIC at {at.strftime('%H:%M:%S')}: the operator flattened everything "
                    f"and stopped trading for the day.",
                )
                result["halted"] = True
            except Exception as exc:  # noqa: BLE001 - journal backend
                log.critical("could not persist the panic halt: %s", exc, exc_info=True)
                result["notes"].append(f"The day halt could not be persisted: {exc}")

            self.cfg.set_armed(False)
            result["armed"] = False

        try:
            result["broker_positions"] = self.broker.open_positions()
        except IndbotError as exc:
            result["notes"].append(f"Could not read the broker's positions: {exc}")
        if result["broker_positions"]:
            # Not sold automatically: squaring off a contract the engine never opened
            # means guessing its lot size and its price on an order path. Report it.
            log.critical(
                "PANIC: the broker still reports %d open position(s) the bot did not open or "
                "could not close: %s. Square these off on INDmoney.",
                len(result["broker_positions"]),
                ", ".join(
                    str(r.get("label") or r.get("security_id"))
                    for r in result["broker_positions"]
                ),
            )
            result["notes"].append(
                f"{len(result['broker_positions'])} position(s) are still open on the broker. "
                "The bot does not place orders for contracts it did not open — square these "
                "off on INDmoney."
            )
        log.critical(
            "PANIC done: %d flattened, %d failed, %d skipped, halted=%s",
            len(result["flattened"]), len(result["failed"]), len(result["skipped"]),
            result["halted"],
        )
        self._publish()
        return result

    def close_position(self, position_id: str, reason: str = "manual") -> dict[str, Any]:
        """Square off one position by id — the dashboard's manual CLOSE button.

        Not in CONTRACTS.md section K, but section N's positions panel has the button
        and it has to reach the broker through something.
        """
        with self._lock:
            pos = self._positions.get(str(position_id))
        if pos is None or not pos.is_open:
            raise DataError(
                f"no open position with id {position_id!r}", position_id=str(position_id)
            )
        ltp = 0.0
        with contextlib.suppress(IndbotError):
            ltp = self._mark(pos)
        closed = self._exit(pos, reason, ltp=ltp)
        self._publish()
        if closed is None:
            raise OrderError(
                f"an exit for {pos.leg.label} is already in flight; not sending a second",
                security_id=pos.leg.security_id,
            )
        return closed

    def _flatten_all(
        self, reason: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Sell every position the engine holds. Returns (flattened, failed, skipped).

        Never raises: the caller decides what a failure means. Each failure keeps its
        position on the books so the tick loop goes on retrying the exit.
        """
        flattened: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for pos in self._open_positions():
            with self._lock:
                in_flight = pos.id in self._exiting
            if in_flight:
                skipped.append({
                    "id": pos.id,
                    "label": pos.leg.label,
                    "error": "an exit is already in flight; a second SELL would open a short",
                })
                log.critical(
                    "%s: an exit for %s is already in flight — NOT sending a second SELL. "
                    "Verify the order on INDmoney.", reason, pos.leg.label,
                )
                continue
            ltp = 0.0
            with contextlib.suppress(IndbotError):
                ltp = self._mark(pos)
            try:
                closed = self._exit(pos, reason, ltp=ltp)
            except OrderError as exc:
                failed.append({
                    "id": pos.id,
                    "label": pos.leg.label,
                    "qty": pos.qty,
                    "security_id": pos.leg.security_id,
                    "error": getattr(exc, "message", str(exc)),
                })
                continue
            except IndbotError as exc:
                failed.append({
                    "id": pos.id, "label": pos.leg.label, "qty": pos.qty,
                    "security_id": pos.leg.security_id,
                    "error": getattr(exc, "message", str(exc)),
                })
                continue
            if closed is not None:
                flattened.append(closed)
        return flattened, failed, skipped

    # -------------------------------------------------------------- pub/sub

    def subscribe(self, fn: Callable[[dict], None]) -> Callable[[], None]:
        """Register a snapshot listener; the returned callable unsubscribes it."""
        with self._lock:
            self._sub_seq += 1
            key = self._sub_seq
            self._subscribers[key] = fn

        def unsubscribe() -> None:
            """Remove this listener. Safe to call more than once."""
            with self._lock:
                self._subscribers.pop(key, None)

        return unsubscribe

    def _publish(self) -> None:
        """Build the snapshot, store it, and hand it to every subscriber."""
        try:
            snap = self._build_snapshot()
        except Exception as exc:  # noqa: BLE001 - the UI feed must not kill the loop
            log.exception("could not build the snapshot")
            self._record_error(exc, where="snapshot")
            return
        with self._lock:
            self._published = snap
            listeners = list(self._subscribers.values())
        for fn in listeners:
            try:
                fn(snap)
            except Exception as exc:  # noqa: BLE001 - a bad listener is not our problem
                log.warning("a snapshot subscriber raised: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        """Everything the UI renders. Read-only — do not mutate the returned dict.

        Returns the last published snapshot when there is one, so the server thread
        never waits on the loop; before the first tick it builds one on the spot.
        """
        with self._lock:
            published = self._published
        if published is not None:
            return published
        return self._build_snapshot()

    def _build_snapshot(self) -> dict[str, Any]:
        """Assemble the snapshot, degrading each section independently on failure."""
        now_dt = clock.now()
        mode = str(self.cfg.mode)
        halted, halt_reason = _safe(
            "halt",
            self._halt_state,
            (True, "Could not determine today's halt state; treating the day as halted."),
        )
        day = _safe("day_stats", lambda: self.journal.day_stats(now_dt.date(), mode), {})
        positions = self._position_views()
        with self._lock:
            views = [dict(v) for v in self._views.values()]
            errors = list(self._errors)
            exit_window = self._exit_window
            tick_count = self._tick_count
            last_tick_at = self._last_tick_at
            last_tick_ms = self._last_tick_ms
            started_at = self._started_at
            funds = self._funds
            account = self._account
        order = {name: i for i, name in enumerate(self.cfg.underlyings or [])}
        views.sort(key=lambda v: (order.get(str(v.get("name")), 99), str(v.get("name"))))

        return {
            # --- contract keys ------------------------------------------------
            "mode": mode,
            "armed": bool(self.cfg.armed),
            "market_open": clock.is_market_open(now_dt),
            "now": now_dt.isoformat(),
            "session": self._session_view(now_dt),
            "halted": bool(halted),
            "halt_reason": halt_reason,
            "config": _safe("config", self.cfg.as_dict, {}),
            "funds": funds,
            "day": day,
            "underlyings": views,
            "positions": positions,
            "recent_trades": _safe(
                "trades", lambda: self.journal.trades(mode=mode, limit=25), []
            ),
            "recent_signals": _safe(
                "signals", lambda: self.journal.signals(mode=mode, limit=60), []
            ),
            "leaderboard": _safe(
                "leaderboard", lambda: [v.as_dict() for v in self.learner.leaderboard()], []
            ),
            "blocked_buckets": _safe("blocked", self.learner.blocked_buckets, []),
            "log": self._sink.tail(250),
            "errors": errors,
            # --- additions the UI needs, all additive ---------------------------
            "running": self.running,
            "started_at": started_at,
            "tick": {
                "count": tick_count,
                "last_at": last_tick_at,
                "last_ms": last_tick_ms,
                "poll_seconds": self.poll_seconds,
                "interval": self.strategy.interval,
                "exiting": exit_window > 0,
            },
            "account": account,
            # Masked only — token_store.describe() never returns the token, so this
            # is safe to broadcast over the websocket and render in the browser.
            "token": _safe("token", self.token_info, {"present": False}),
            "arm_phrase": ARM_PHRASE,
            "limits": _safe(
                "limits", lambda: self.cfg.describe_risk(self._lot_sizes), []
            ),
            "learning": _safe("learning", self.learner.summary, {}),
            "instruments": _safe(
                "instruments",
                lambda: self.instruments.status() if self.instruments else {},
                {},
            ),
            "connected": self._client is not None,
            "setups": list(self.strategy.enabled),
        }

    def _position_views(self) -> list[dict[str, Any]]:
        """Open positions as JSON, with the stop actually in force and any exit fault."""
        out: list[dict[str, Any]] = []
        for pos in self._open_positions():
            ltp = float(pos.leg.ltp or 0.0)
            # None, not 0.0: Position.pnl(None) is 0.0, while pnl(0.0) is a full loss of
            # the premium. A quote outage must read as "no quote", never as a wipeout.
            row = pos.as_dict(ltp if ltp > 0 else None)
            row["mark_ok"] = ltp > 0
            row["risk_stop"] = round(_safe(
                "effective_stop", lambda: self.risk.effective_stop(pos, ltp), pos.stop_price
            ), 2)
            with self._lock:
                fault = self._exit_fail.get(pos.id)
                row["exiting"] = pos.id in self._exiting
            row["exit_fault"] = dict(fault) if fault else None
            row["risk_at_stop"] = round(
                max(0.0, (pos.entry_price - row["risk_stop"]) * pos.qty), 2
            )
            out.append(row)
        return out

    def _session_view(self, now_dt: datetime) -> dict[str, Any]:
        """Where we are in the trading day, and whether entries are allowed right now."""
        today = now_dt.date()
        trading_day = clock.is_trading_day(today)
        open_at = clock.session_open(today)
        close_at = clock.session_close(today)
        if not trading_day:
            state = "holiday"
        elif now_dt < open_at:
            state = "pre_open"
        elif now_dt > close_at:
            state = "closed"
        else:
            state = "open"
        entry_window = False
        with contextlib.suppress(ValueError, AttributeError):
            entry_window = (
                clock.parse_hhmm(str(self.cfg.no_entry_before))
                <= now_dt.time()
                <= clock.parse_hhmm(str(self.cfg.no_entry_after))
            )
        return {
            "state": state,
            "is_trading_day": trading_day,
            "open": open_at.isoformat(),
            "close": close_at.isoformat(),
            "minutes_into_session": clock.minutes_into_session(now_dt),
            "tod_bucket": clock.tod_bucket(now_dt),
            "no_entry_before": str(self.cfg.no_entry_before),
            "no_entry_after": str(self.cfg.no_entry_after),
            "flat_by": str(self.cfg.flat_by),
            "entries_allowed": bool(state == "open" and entry_window),
        }

    # ---------------------------------------------------------------- errors

    def _record_error(self, exc: BaseException, *, where: str = "") -> None:
        """Add an error to the snapshot's `errors` list, collapsing repeats."""
        if isinstance(exc, IndbotError):
            payload = exc.as_dict()
        else:
            payload = {"type": type(exc).__name__, "message": str(exc), "context": {}}
        payload["where"] = where
        payload["at"] = clock.now().isoformat()
        with self._lock:
            if self._errors:
                last = self._errors[-1]
                if last.get("message") == payload["message"] and last.get("where") == where:
                    last["count"] = int(last.get("count", 1)) + 1
                    last["at"] = payload["at"]
                    return
            payload["count"] = 1
            self._errors.append(payload)

    def errors(self) -> list[dict[str, Any]]:
        """The recorded errors, oldest first."""
        with self._lock:
            return [dict(e) for e in self._errors]

    def clear_errors(self) -> None:
        """Drop the recorded errors — the dashboard's dismiss button."""
        with self._lock:
            self._errors.clear()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _safe(label: str, fn: Callable[[], Any], default: Any) -> Any:
    """Call `fn`, returning `default` and logging when it fails.

    Used only when assembling the snapshot: one unreadable section must not blank the
    whole dashboard, because a frozen dashboard is how a stranded position goes unseen.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a view is never worth a crash
        log.warning("snapshot: %s unavailable (%s)", label, exc)
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """A float from anything the journal or an error payload might hold."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """An int from anything the journal or an error payload might hold."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
