"""Trade journal — the single source of truth for what the bot did and why.

Everything the risk gate, the learner and the UI know about history comes out of
here: a SQLite database in WAL mode, shared by three threads (the engine loop, the
FastAPI server, a backtest worker).

Two decisions in this file are load-bearing, and both exist to protect real money.

1. **Backtest, paper and live rows share the same tables, separated by `mode`.**
   One schema, one set of queries, one place to look. The hazard is equally
   obvious: if a backtest's P&L leaked into the live day stats, `RiskGate`'s
   day-halt would be computed from fictional money and would either stop a healthy
   day or — far worse — fail to stop a losing one. So every write validates its
   mode against `types.MODES` (a typo'd mode would create a pile of rows that
   `day_stats` never sees, which silently disarms the loss limit), and the two
   reads the risk gate depends on, `day_stats` and `equity_curve`, take a concrete
   mode and never aggregate across modes.

2. **"Today" is an IST calendar day, computed in Python from `indbot.clock`.**
   `entry_ts` is a raw epoch. SQLite's `date(entry_ts, 'unixepoch')` would read it
   as UTC and bucket every trade entered after 18:30 IST into the previous day. So
   the day's epoch bounds are computed with `clock` and compared as integers, here
   and in every other date-filtered query.

Locking: one `sqlite3.Connection` with `check_same_thread=False`, and a
module-level `RLock` held across *every* statement, not just writes. A shared
connection is not safe for concurrent cursor use even when reads do not conflict
at the storage layer, and at this volume (a handful of rows a day) the contention
cost is nil.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from datetime import date, time, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

from . import clock
from .errors import ConfigError, DataError
from .types import MODES, Position, Signal

logger = logging.getLogger(__name__)

#: Bumped whenever the schema changes. Stored in `meta` so an upgrade can migrate
#: an existing database instead of the user losing their trade history.
SCHEMA_VERSION = 1

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"

#: Guards every statement on the shared connection. Module level (not per
#: instance) because two Journal objects pointed at the same file would otherwise
#: not serialise against each other.
_DB_LOCK = threading.RLock()

_MAX_LIMIT = 5000


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_ts INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id              TEXT PRIMARY KEY,
        mode            TEXT NOT NULL,
        underlying      TEXT NOT NULL,
        setup           TEXT NOT NULL,
        bucket          TEXT NOT NULL,
        side            TEXT NOT NULL,
        strike          REAL NOT NULL,
        expiry          TEXT NOT NULL,
        trading_symbol  TEXT NOT NULL,
        security_id     TEXT NOT NULL,
        lot_size        INTEGER NOT NULL,
        qty             INTEGER NOT NULL,
        entry_price     REAL NOT NULL,
        entry_ts        INTEGER NOT NULL,
        entry_order_id  TEXT NOT NULL DEFAULT '',
        exit_price      REAL,
        exit_ts         INTEGER,
        exit_order_id   TEXT,
        exit_reason     TEXT,
        stop_price      REAL NOT NULL DEFAULT 0,
        target_price    REAL NOT NULL DEFAULT 0,
        high_water      REAL NOT NULL DEFAULT 0,
        costs           REAL NOT NULL DEFAULT 0,
        pnl             REAL,
        pnl_pct         REAL,
        status          TEXT NOT NULL,
        created_ts      INTEGER NOT NULL,
        updated_ts      INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        mode         TEXT NOT NULL,
        ts           INTEGER NOT NULL,
        recorded_ts  INTEGER NOT NULL,
        underlying   TEXT NOT NULL,
        setup        TEXT NOT NULL,
        side         TEXT NOT NULL,
        action       TEXT NOT NULL,
        bucket       TEXT NOT NULL,
        score        REAL NOT NULL,
        spot         REAL NOT NULL,
        stop_pct     REAL NOT NULL,
        target_pct   REAL NOT NULL,
        reasons      TEXT NOT NULL DEFAULT '[]',
        taken        INTEGER NOT NULL,
        block_reason TEXT NOT NULL DEFAULT '',
        trade_id     TEXT,
        signal_json  TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_trades_mode_entry  ON trades(mode, entry_ts)",
    "CREATE INDEX IF NOT EXISTS ix_trades_bucket      ON trades(bucket)",
    "CREATE INDEX IF NOT EXISTS ix_trades_mode_status ON trades(mode, status)",
    "CREATE INDEX IF NOT EXISTS ix_signals_mode_ts    ON signals(mode, ts)",
    "CREATE INDEX IF NOT EXISTS ix_signals_bucket     ON signals(bucket)",
    "CREATE INDEX IF NOT EXISTS ix_signals_taken      ON signals(mode, taken, ts)",
)


def _migrate_to_1(conn: sqlite3.Connection) -> None:
    """Placeholder for the first migration: v0 databases predate this module."""
    return None


#: version -> migration that upgrades a database *to* that version. Adding a
#: column later is then two lines: bump SCHEMA_VERSION, and register a migration
#: that calls `_add_column`. Nobody has to delete indbot.db to get an upgrade.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: _migrate_to_1,
}


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Add a column if it is absent — the idempotent building block for migrations."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        logger.info("journal: added column %s.%s", table, column)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def day_bounds(day: date) -> tuple[int, int]:
    """Epoch-second bounds [start, end) of one IST calendar day.

    Public because the engine and the backtester need the same definition of a
    day that `day_stats` uses. Never replace this with SQLite's `date()`: that
    function reads a bare epoch as UTC and would push every trade after 18:30 IST
    into the day before.
    """
    start = clock.epoch_s(clock.at(day, time(0, 0)))
    end = clock.epoch_s(clock.at(day + timedelta(days=1), time(0, 0)))
    return start, end


def _check_mode(mode: str) -> str:
    """Validate a mode on a write path, raising rather than creating orphan rows."""
    if mode not in MODES:
        raise ConfigError(
            f"unknown mode {mode!r}; expected one of {', '.join(MODES)}",
            mode=mode,
        )
    return mode


def _clamp_limit(limit: int) -> int:
    """Keep a caller-supplied LIMIT sane; the UI should never be able to OOM us."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 200
    return max(1, min(_MAX_LIMIT, value))


def _loads(raw: Any, fallback: Any) -> Any:
    """json.loads that degrades to `fallback` — one bad row must not break the UI."""
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("journal: unparseable JSON column, falling back")
        return fallback


def _iso(epoch: int | None) -> str | None:
    """IST ISO-8601 for an epoch-seconds column, or None."""
    return clock.to_ist(epoch).isoformat() if epoch else None


def _label(underlying: str, strike: float, side: str) -> str:
    """"NIFTY 25000 CE" — matches OptionLeg.label so the UI reads consistently."""
    s = int(strike) if float(strike).is_integer() else strike
    return f"{underlying} {s} {side}"


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #


class Journal:
    """Durable record of every signal and every trade, in every mode.

    Open one per process and share it; it is thread-safe. Use it as a context
    manager, or call `close()`.
    """

    def __init__(self, path: Path | str, *, default_mode: str = "paper") -> None:
        """Open (and create, if needed) the database at `path`.

        `default_mode` tags writes that carry no explicit mode — notably
        `record_signal`, since a Signal has no mode of its own. It defaults to
        "paper" so a caller that forgets to set it can never mint fake *live*
        history; the engine should assign `journal.default_mode = cfg.mode`.
        """
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self._default_mode = _check_mode(default_mode)
        self._closed = False

        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with _DB_LOCK:
            self._configure()
            self._create_schema()
        logger.info("journal: opened %s (schema v%d)", self.path, SCHEMA_VERSION)

    # -- lifecycle --------------------------------------------------------- #

    @property
    def default_mode(self) -> str:
        """Mode stamped on writes that do not name one."""
        return self._default_mode

    @default_mode.setter
    def default_mode(self, mode: str) -> None:
        """Retag unmoded writes; validated, so a typo cannot create orphan rows."""
        self._default_mode = _check_mode(mode)

    def close(self) -> None:
        """Flush and close the connection. Safe to call twice."""
        with _DB_LOCK:
            if self._closed:
                return
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._closed = True
        logger.info("journal: closed %s", self.path)

    def __enter__(self) -> "Journal":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- schema ------------------------------------------------------------ #

    def _configure(self) -> None:
        """Pragmas. WAL so a reader (the UI) never blocks the engine's write."""
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        # FULL, not NORMAL: this file is the only record that a real rupee loss
        # happened, and the day-halt is computed from it. A few fsyncs a day is a
        # price worth paying to never lose the last closed trade.
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA busy_timeout=10000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    def _create_schema(self) -> None:
        """Create tables and indexes idempotently, then run pending migrations."""
        fresh = (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades'"
            ).fetchone()
            is None
        )
        with self._conn:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            if fresh:
                # A new file is born at the current version; migrations below it
                # would be re-running work the CREATE statements already did.
                self._write_meta(self._conn, "schema_version", str(SCHEMA_VERSION))
            else:
                self._run_migrations(self._conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply every migration above the stored version, in order."""
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        try:
            current = int(row["value"]) if row else 0
        except (TypeError, ValueError):
            current = 0
        if current > SCHEMA_VERSION:
            # A newer build wrote this file. Refuse rather than corrupt it.
            raise DataError(
                f"journal at {self.path} is schema v{current}, this build understands "
                f"v{SCHEMA_VERSION}; upgrade indbot or point at a different file",
                found=current,
                expected=SCHEMA_VERSION,
            )
        for version in range(current + 1, SCHEMA_VERSION + 1):
            migrate = _MIGRATIONS.get(version)
            if migrate is None:
                continue
            logger.info("journal: migrating schema v%d -> v%d", version - 1, version)
            migrate(conn)
        if current != SCHEMA_VERSION:
            self._write_meta(conn, "schema_version", str(SCHEMA_VERSION))

    # -- meta -------------------------------------------------------------- #

    @staticmethod
    def _write_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value, updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (key, value, clock.epoch_s(clock.now())),
        )

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a key/value row — small persistent state like the last run date."""
        with _DB_LOCK, self._conn:
            self._write_meta(self._conn, key, str(value))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """Read a meta value, or `default` when the key is absent."""
        with _DB_LOCK:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    # -- signals ----------------------------------------------------------- #

    def record_signal(
        self,
        sig: Signal,
        *,
        taken: bool,
        reason: str = "",
        mode: str | None = None,
        trade_id: str | None = None,
    ) -> int:
        """Record a signal and whether it was traded; returns the new row id.

        Every signal is written, taken or not. `reason` is why it was *not* taken,
        in full — the "why no trade" panel renders it verbatim, and a bot that
        explains its silence is the whole point of the product.
        """
        mode = _check_mode(mode or self._default_mode)
        reasons = list(sig.reasons or [])
        try:
            payload = json.dumps(sig.as_dict(), default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.warning("journal: signal %s not serialisable, storing a stub", sig.setup)
            payload = "{}"

        with _DB_LOCK, self._conn:
            cur = self._conn.execute(
                "INSERT INTO signals(mode, ts, recorded_ts, underlying, setup, side, action, "
                "bucket, score, spot, stop_pct, target_pct, reasons, taken, block_reason, "
                "trade_id, signal_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mode,
                    int(sig.ts),
                    clock.epoch_s(clock.now()),
                    sig.underlying,
                    sig.setup,
                    sig.side,
                    sig.action,
                    sig.bucket,
                    float(sig.score),
                    float(sig.spot),
                    float(sig.stop_pct),
                    float(sig.target_pct),
                    json.dumps(reasons),
                    1 if taken else 0,
                    reason or "",
                    trade_id,
                    payload,
                ),
            )
        return int(cur.lastrowid or 0)

    def signals(
        self,
        *,
        day: date | None = None,
        limit: int = 200,
        mode: str | None = None,
        taken: bool | None = None,
        underlying: str | None = None,
    ) -> list[dict]:
        """Recent signals, newest first; `taken=False` is the "why no trade" feed.

        `day` filters on the signal's candle time, bucketed by IST calendar day.
        `mode=None` returns every mode — pass `cfg.mode` for a single-mode view.
        """
        where: list[str] = []
        args: list[Any] = []
        if mode is not None:
            if not self._known_mode(mode):
                return []
            where.append("mode = ?")
            args.append(mode)
        if day is not None:
            start, end = day_bounds(day)
            where.append("ts >= ? AND ts < ?")
            args.extend((start, end))
        if taken is not None:
            where.append("taken = ?")
            args.append(1 if taken else 0)
        if underlying is not None:
            where.append("underlying = ?")
            args.append(underlying)

        sql = "SELECT * FROM signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(_clamp_limit(limit))

        with _DB_LOCK:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._signal_row(r) for r in rows]

    @staticmethod
    def _signal_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["reasons"] = _loads(d.get("reasons"), [])
        d["signal"] = _loads(d.pop("signal_json", None), {})
        d["taken"] = bool(d.get("taken"))
        d["at"] = _iso(d.get("ts"))
        d["recorded_at"] = _iso(d.get("recorded_ts"))
        return d

    # -- trades ------------------------------------------------------------ #

    def open_trade(self, pos: Position, mode: str) -> None:
        """Write a newly opened position as an `open` trade row.

        `mode` is the caller's explicit statement and wins over `pos.mode`; a
        mismatch is logged, because a backtest row filed under "live" would feed
        the live day-halt.
        """
        mode = _check_mode(mode)
        if pos.mode and pos.mode != mode:
            logger.warning(
                "journal: trade %s has pos.mode=%r but was opened as %r; using %r",
                pos.id,
                pos.mode,
                mode,
                mode,
            )
        now_s = clock.epoch_s(clock.now())
        leg = pos.leg

        with _DB_LOCK, self._conn:
            existing = self._conn.execute(
                "SELECT status FROM trades WHERE id=?", (pos.id,)
            ).fetchone()
            if existing is not None and existing["status"] == STATUS_CLOSED:
                # Re-opening a closed id would erase a realised P&L and with it
                # the loss the day-halt is counting. Refuse loudly instead.
                raise DataError(
                    f"trade id {pos.id!r} is already closed in the journal; refusing to "
                    "overwrite a realised result",
                    trade_id=pos.id,
                )
            self._conn.execute(
                """
                INSERT INTO trades(
                    id, mode, underlying, setup, bucket, side, strike, expiry,
                    trading_symbol, security_id, lot_size, qty, entry_price, entry_ts,
                    entry_order_id, stop_price, target_price, high_water, costs,
                    status, created_ts, updated_ts)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    mode=excluded.mode, qty=excluded.qty,
                    entry_price=excluded.entry_price, entry_ts=excluded.entry_ts,
                    entry_order_id=excluded.entry_order_id,
                    stop_price=excluded.stop_price, target_price=excluded.target_price,
                    high_water=excluded.high_water, costs=excluded.costs,
                    updated_ts=excluded.updated_ts
                """,
                (
                    pos.id,
                    mode,
                    leg.underlying,
                    pos.signal_setup,
                    pos.bucket,
                    leg.side,
                    float(leg.strike),
                    leg.expiry,
                    leg.trading_symbol,
                    leg.security_id,
                    int(leg.lot_size),
                    int(pos.qty),
                    float(pos.entry_price),
                    int(pos.entry_ts),
                    pos.entry_order_id or "",
                    float(pos.stop_price),
                    float(pos.target_price),
                    float(pos.high_water),
                    float(pos.costs),
                    STATUS_OPEN,
                    now_s,
                    now_s,
                ),
            )
        logger.info(
            "journal: opened %s %s qty=%d @ %.2f (%s)",
            mode,
            _label(leg.underlying, leg.strike, leg.side),
            pos.qty,
            pos.entry_price,
            pos.id,
        )

    def close_trade(self, pos: Position) -> None:
        """Write the exit, the realised P&L and the `closed` status for a position.

        If the opening row is missing (a crash between the fill and the write), the
        trade is INSERTed here instead. Dropping a realised loss on the floor would
        understate the day and leave the day-halt switched off, so the safe branch
        is to record it even with a partly reconstructed row.
        """
        if pos.exit_price is None:
            raise DataError(
                f"close_trade({pos.id!r}) called with no exit_price — the position is "
                "still open and must not be journalled as closed",
                trade_id=pos.id,
            )

        mode = _check_mode(pos.mode or self._default_mode)
        now_s = clock.epoch_s(clock.now())
        pnl = pos.pnl()
        pnl_pct = pos.pnl_pct()  # guards cost_basis <= 0 itself
        exit_ts = int(pos.exit_ts) if pos.exit_ts else now_s
        leg = pos.leg

        with _DB_LOCK, self._conn:
            existing = self._conn.execute(
                "SELECT status, mode FROM trades WHERE id=?", (pos.id,)
            ).fetchone()
            if existing is None:
                logger.error(
                    "journal: no open row for trade %s; inserting the closed trade so its "
                    "P&L is not lost",
                    pos.id,
                )
                self._conn.execute(
                    """
                    INSERT INTO trades(
                        id, mode, underlying, setup, bucket, side, strike, expiry,
                        trading_symbol, security_id, lot_size, qty, entry_price, entry_ts,
                        entry_order_id, stop_price, target_price, high_water, costs,
                        status, created_ts, updated_ts)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        pos.id,
                        mode,
                        leg.underlying,
                        pos.signal_setup,
                        pos.bucket,
                        leg.side,
                        float(leg.strike),
                        leg.expiry,
                        leg.trading_symbol,
                        leg.security_id,
                        int(leg.lot_size),
                        int(pos.qty),
                        float(pos.entry_price),
                        int(pos.entry_ts),
                        pos.entry_order_id or "",
                        float(pos.stop_price),
                        float(pos.target_price),
                        float(pos.high_water),
                        float(pos.costs),
                        STATUS_OPEN,
                        now_s,
                        now_s,
                    ),
                )
            elif existing["status"] == STATUS_CLOSED:
                logger.warning("journal: trade %s closed twice; overwriting the exit", pos.id)

            self._conn.execute(
                """
                UPDATE trades SET
                    exit_price=?, exit_ts=?, exit_order_id=?, exit_reason=?,
                    high_water=?, costs=?, pnl=?, pnl_pct=?, status=?, updated_ts=?
                WHERE id=?
                """,
                (
                    float(pos.exit_price),
                    exit_ts,
                    pos.exit_order_id,
                    pos.exit_reason,
                    float(pos.high_water),
                    float(pos.costs),
                    round(float(pnl), 4),
                    round(float(pnl_pct), 4),
                    STATUS_CLOSED,
                    now_s,
                    pos.id,
                ),
            )
        logger.info(
            "journal: closed %s %s @ %.2f (%s) pnl=%.2f",
            mode,
            _label(leg.underlying, leg.strike, leg.side),
            pos.exit_price,
            pos.exit_reason or "?",
            pnl,
        )

    def trades(
        self,
        *,
        day: date | None = None,
        mode: str | None = None,
        limit: int = 200,
        status: str | None = None,
        bucket: str | None = None,
    ) -> list[dict]:
        """Trades filtered by IST entry day / mode / status, newest entry first.

        `mode=None` returns every mode; each row carries its own `mode`, so a UI
        that mixes them can still tell a backtest from a live fill.
        """
        where: list[str] = []
        args: list[Any] = []
        if mode is not None:
            if not self._known_mode(mode):
                return []
            where.append("mode = ?")
            args.append(mode)
        if day is not None:
            start, end = day_bounds(day)
            where.append("entry_ts >= ? AND entry_ts < ?")
            args.extend((start, end))
        if status is not None:
            where.append("status = ?")
            args.append(status)
        if bucket is not None:
            where.append("bucket = ?")
            args.append(bucket)

        sql = "SELECT * FROM trades"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY entry_ts DESC, rowid DESC LIMIT ?"
        args.append(_clamp_limit(limit))

        with _DB_LOCK:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._trade_row(r) for r in rows]

    def trade(self, trade_id: str) -> dict | None:
        """One trade by id, or None."""
        with _DB_LOCK:
            row = self._conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        return self._trade_row(row) if row else None

    def open_trades(self, *, mode: str | None = None, limit: int = 200) -> list[dict]:
        """Every trade still marked open — how the engine finds a position it was
        holding when it restarted, in any mode."""
        return self.trades(mode=mode, status=STATUS_OPEN, limit=limit)

    @staticmethod
    def _trade_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["is_open"] = d.get("status") == STATUS_OPEN
        d["label"] = _label(d.get("underlying", ""), d.get("strike") or 0.0, d.get("side", ""))
        d["entry_at"] = _iso(d.get("entry_ts"))
        d["exit_at"] = _iso(d.get("exit_ts"))
        return d

    # -- aggregates -------------------------------------------------------- #

    def day_stats(self, day: date | None = None, mode: str = "paper") -> dict:
        """The risk gate's view of one IST calendar day, for one mode only.

        Counts trades by `entry_ts`, so a trade is spent against
        `max_trades_per_day` on the day it was entered. `mode` is required (it
        defaults, but never to "all") because a backtest's P&L mixed in here would
        disable the day-halt — the single most expensive bug this file can have.
        """
        day = day or clock.today()
        if not self._known_mode(mode):
            return self._empty_day(day, mode)
        start, end = day_bounds(day)

        with _DB_LOCK:
            rows = self._conn.execute(
                "SELECT status, pnl, costs FROM trades "
                "WHERE mode=? AND entry_ts >= ? AND entry_ts < ?",
                (mode, start, end),
            ).fetchall()
            stranded = self._conn.execute(
                "SELECT COUNT(*) AS n FROM trades WHERE mode=? AND status=?",
                (mode, STATUS_OPEN),
            ).fetchone()

        realized = 0.0
        wins = losses = scratches = open_count = 0
        gross_win = gross_loss = 0.0
        costs = 0.0
        best: float | None = None
        worst: float | None = None

        for r in rows:
            costs += float(r["costs"] or 0.0)
            if r["status"] == STATUS_OPEN:
                open_count += 1
                continue
            pnl = float(r["pnl"] or 0.0)
            realized += pnl
            best = pnl if best is None else max(best, pnl)
            worst = pnl if worst is None else min(worst, pnl)
            if pnl > 0:
                wins += 1
                gross_win += pnl
            elif pnl < 0:
                losses += 1
                gross_loss += -pnl
            else:
                scratches += 1

        closed = wins + losses + scratches
        return {
            "day": day.isoformat(),
            "mode": mode,
            "realized_pnl": round(realized, 2),
            "trade_count": len(rows),
            "closed_count": closed,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "open_count": open_count,
            # Open rows from ANY day in this mode: a position still open from a
            # previous session is a stranded position, and the UI must shout.
            "open_count_any_day": int(stranded["n"] if stranded else 0),
            "gross_win": round(gross_win, 2),
            "gross_loss": round(gross_loss, 2),
            "costs": round(costs, 2),
            "win_rate": round(wins / closed, 4) if closed else 0.0,
            "best": round(best, 2) if best is not None else 0.0,
            "worst": round(worst, 2) if worst is not None else 0.0,
        }

    @staticmethod
    def _empty_day(day: date, mode: str) -> dict:
        return {
            "day": day.isoformat(),
            "mode": mode,
            "realized_pnl": 0.0,
            "trade_count": 0,
            "closed_count": 0,
            "wins": 0,
            "losses": 0,
            "scratches": 0,
            "open_count": 0,
            "open_count_any_day": 0,
            "gross_win": 0.0,
            "gross_loss": 0.0,
            "costs": 0.0,
            "win_rate": 0.0,
            "best": 0.0,
            "worst": 0.0,
        }

    def equity_curve(self, mode: str = "paper", days: int = 30) -> list[dict]:
        """Realised P&L per day and its running total, for the last `days` days.

        One row per *trading* day in the window, including flat days, so the UI can
        plot a continuous line without filling gaps itself. Days are IST calendar
        days bucketed in Python, and a trade counts on its ENTRY day so the last
        point always equals `day_stats(...)["realized_pnl"]`.
        """
        if not self._known_mode(mode):
            return []
        span = max(1, int(days))
        end_day = clock.today()
        start_day = end_day - timedelta(days=span - 1)
        start_ts, _ = day_bounds(start_day)
        _, end_ts = day_bounds(end_day)

        with _DB_LOCK:
            rows = self._conn.execute(
                "SELECT entry_ts, pnl FROM trades "
                "WHERE mode=? AND status=? AND entry_ts >= ? AND entry_ts < ? "
                "ORDER BY entry_ts ASC",
                (mode, STATUS_CLOSED, start_ts, end_ts),
            ).fetchall()

        per_day: dict[str, dict[str, float]] = {}
        for r in rows:
            key = clock.to_ist(int(r["entry_ts"])).date().isoformat()
            slot = per_day.setdefault(key, {"pnl": 0.0, "trades": 0.0, "wins": 0.0, "losses": 0.0})
            pnl = float(r["pnl"] or 0.0)
            slot["pnl"] += pnl
            slot["trades"] += 1
            if pnl > 0:
                slot["wins"] += 1
            elif pnl < 0:
                slot["losses"] += 1

        out: list[dict] = []
        cum = 0.0
        for d in clock.trading_days(start_day, end_day):
            key = d.isoformat()
            slot = per_day.get(key)
            pnl = slot["pnl"] if slot else 0.0
            cum += pnl
            out.append(
                {
                    "day": key,
                    "pnl": round(pnl, 2),
                    "cum_pnl": round(cum, 2),
                    "trades": int(slot["trades"]) if slot else 0,
                    "wins": int(slot["wins"]) if slot else 0,
                    "losses": int(slot["losses"]) if slot else 0,
                }
            )
        return out

    def bucket_stats(self, mode: str | None = None, *, min_n: int = 0) -> list[dict]:
        """Per learning-bucket outcome stats, best expectancy first.

        Each row: `bucket, n, wins, losses, scratches, gross_win, gross_loss,
        net_pnl, win_rate, expectancy, avg_win, avg_loss, last_ts`. `n` counts only
        CLOSED trades — an open trade has no outcome to learn from — and
        `gross_loss` is a POSITIVE magnitude, with `net_pnl = gross_win -
        gross_loss`. `expectancy` is mean rupees per closed trade and is what the
        learner consumes directly.

        `mode=None` aggregates every mode, which is almost never what the learner
        wants: a thousand backtest trades would drown fifty live ones in the same
        bucket. Pass `cfg.mode`. When a mode-less call actually spans more than one
        mode this logs a warning and reports the contributing modes in `modes`.
        """
        where = ["status = ?"]
        args: list[Any] = [STATUS_CLOSED]
        if mode is not None:
            if not self._known_mode(mode):
                return []
            where.append("mode = ?")
            args.append(mode)

        sql = (
            "SELECT bucket, mode, pnl, entry_ts FROM trades WHERE "
            + " AND ".join(where)
            + " AND bucket IS NOT NULL AND bucket != ''"
        )
        with _DB_LOCK:
            rows = self._conn.execute(sql, args).fetchall()

        acc: dict[str, dict[str, Any]] = {}
        for r in rows:
            b = r["bucket"]
            slot = acc.setdefault(
                b,
                {
                    "bucket": b,
                    "n": 0,
                    "wins": 0,
                    "losses": 0,
                    "scratches": 0,
                    "gross_win": 0.0,
                    "gross_loss": 0.0,
                    "last_ts": 0,
                    "_modes": set(),
                },
            )
            pnl = float(r["pnl"] or 0.0)
            slot["n"] += 1
            slot["_modes"].add(r["mode"])
            slot["last_ts"] = max(int(slot["last_ts"]), int(r["entry_ts"] or 0))
            if pnl > 0:
                slot["wins"] += 1
                slot["gross_win"] += pnl
            elif pnl < 0:
                slot["losses"] += 1
                slot["gross_loss"] += -pnl
            else:
                slot["scratches"] += 1

        out: list[dict] = []
        mixed = False
        for slot in acc.values():
            n = int(slot["n"])
            if n < max(0, int(min_n)):
                continue
            modes = sorted(slot.pop("_modes"))
            if len(modes) > 1:
                mixed = True
            gw = float(slot["gross_win"])
            gl = float(slot["gross_loss"])
            wins = int(slot["wins"])
            losses = int(slot["losses"])
            net = gw - gl
            out.append(
                {
                    "bucket": slot["bucket"],
                    "mode": mode,
                    "modes": modes,
                    "n": n,
                    "wins": wins,
                    "losses": losses,
                    "scratches": int(slot["scratches"]),
                    "gross_win": round(gw, 2),
                    "gross_loss": round(gl, 2),
                    "net_pnl": round(net, 2),
                    # n is non-zero by construction here, but divide defensively:
                    # a NaN expectancy would propagate straight into the gate.
                    "win_rate": round(wins / n, 4) if n else 0.0,
                    "expectancy": round(net / n, 2) if n else 0.0,
                    "avg_win": round(gw / wins, 2) if wins else 0.0,
                    "avg_loss": round(gl / losses, 2) if losses else 0.0,
                    "last_ts": int(slot["last_ts"]),
                    "last_at": _iso(int(slot["last_ts"])),
                }
            )
        if mixed:
            logger.warning(
                "journal.bucket_stats(mode=None) mixed modes in one bucket; the learner "
                "should pass cfg.mode so backtest results cannot gate live trading"
            )
        out.sort(key=lambda r: (-r["expectancy"], -r["n"], r["bucket"]))
        return out

    # -- maintenance ------------------------------------------------------- #

    def clear_mode(self, mode: str) -> int:
        """Delete every trade and signal for a non-live mode; returns rows removed.

        Refuses "live" outright. Clearing live history would wipe the realised
        losses the day-halt is counting, and there is no UI button worth that.
        """
        mode = _check_mode(mode)
        if mode == "live":
            raise ConfigError("refusing to delete live trade history", mode=mode)
        with _DB_LOCK, self._conn:
            n = self._conn.execute("DELETE FROM trades WHERE mode=?", (mode,)).rowcount or 0
            n += self._conn.execute("DELETE FROM signals WHERE mode=?", (mode,)).rowcount or 0
        logger.info("journal: cleared %d %s rows", n, mode)
        return n

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _known_mode(mode: str) -> bool:
        """Read-path mode check: warn and return nothing rather than raise.

        A bad mode in a URL query should give the UI an empty list, not a 500.
        """
        if mode in MODES:
            return True
        logger.warning("journal: ignoring query for unknown mode %r", mode)
        return False


__all__ = [
    "Journal",
    "SCHEMA_VERSION",
    "STATUS_CLOSED",
    "STATUS_OPEN",
    "day_bounds",
]
