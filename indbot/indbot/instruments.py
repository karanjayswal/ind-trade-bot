"""Instrument master cache: every SECURITY_ID, lot size, tick size and freeze quantity.

The four CSVs behind ``GET /market/instruments`` are the only authority for an index's
SECURITY_ID, a contract's lot size and its tick size, and all three can change overnight.
This module caches each file at ``data/instruments/<source>.csv``, refetches it when the
copy is older than 12 hours or when the IST trading day has rolled, and answers every
lookup from dicts built once per file rather than by re-scanning the CSV per call.

Three things in here move real money, and each one is written to fail loudly:

* ``index_scrip`` / ``resolve_underlyings`` decide **which** instrument the bot trades.
  Every hardcoded scrip id in ``config.UNDERLYINGS`` except NIFTY's is a guess, so the
  value resolved from the live index file always wins over the constant and a
  disagreement is logged with both numbers. A label that cannot be resolved keeps its
  constant but is marked ``resolved: False`` so the UI can show it as unverified.
* ``lot_size`` decides **how big** every position is. The CSV wins; ``FALLBACK_LOTS`` is
  used only when the row is genuinely absent, and every resolution is recorded in
  ``lot_size_source`` with the file and expiry that answered it.
* ``tick_size`` decides the limit price the broker may send. See ``_PAISE_THRESHOLD``
  for why an ambiguous value is resolved to the *smaller* tick.

A truncated download, an HTML error page or a JSON error body is never allowed to
replace a cache that is already good.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from . import clock
from .errors import ConfigError, DataError

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .api import INDstocksClient

logger = logging.getLogger(__name__)

#: The four values ``source`` accepts on GET /market/instruments.
SOURCES: tuple[str, ...] = ("equity", "fno", "index", "mcx")

#: Lot sizes to fall back on when the F&O master has no row for an underlying. These are
#: only a floor under a total lookup failure — the CSV is always preferred, and every use
#: of one of these is logged and recorded in ``Instruments.lot_size_source``.
FALLBACK_LOTS: dict[str, int] = {
    "NIFTY": 75,
    "BANKNIFTY": 35,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
    "SENSEX": 20,
}

#: Refetch a cached master older than this. Lot sizes, tick sizes and circuit bands are
#: republished daily, so a copy from yesterday is not safe to size a trade from.
MAX_AGE_SECONDS: float = 12 * 3600.0

#: Column counts the docs promise per source. A mismatch is a warning, not an error —
#: the exchange adds columns from time to time and the named lookups still work.
DOCUMENTED_COLUMNS: dict[str, int] = {"equity": 26, "fno": 23, "mcx": 23, "index": 3}

#: Below this many columns the file is not the instrument master at all.
MIN_COLUMNS: dict[str, int] = {"equity": 20, "fno": 18, "mcx": 18, "index": 3}

#: Documented positions of the 23 shared columns, used when a header name is missing.
#: ``equity`` appends INTRADAY_LEVERAGE / HAIRCUT_PERCENTAGE / PLEDGE_ELIGIBLE after these.
SHARED_COLUMN_POSITIONS: dict[str, int] = {
    "EXCH": 0,
    "SEGMENT": 1,
    "SECURITY_ID": 2,
    "INSTRUMENT_NAME": 3,
    "EXPIRY_CODE": 4,
    "TRADING_SYMBOL": 5,
    "LOT_UNITS": 6,
    "CUSTOM_SYMBOL": 7,
    "EXPIRY_DATE": 8,
    "STRIKE_PRICE": 9,
    "OPTION_TYPE": 10,
    "TICK_SIZE": 11,
    "EXPIRY_FLAG": 12,
    "SEM_EXCH_INSTRUMENT_TYPE": 13,
    "SERIES": 14,
    "SYMBOL_NAME": 15,
    "UNDERLYING_SCRIP_NAME": 16,
    "FREEZE_QTY": 17,
    "UPPER_LIMIT": 18,
    "LOWER_LIMIT": 19,
    "ISIN": 20,
    "GENERAL_FACTOR": 21,
    "DELIVERY_UNIT": 22,
}

#: ``source=index`` is three columns — EXCH, SEGMENT, SECURITY_ID — and the second is
#: **labelled** SEGMENT but **contains the index name**. Parsed positionally, never by
#: header name, exactly as the docs instruct.
INDEX_COLUMN_POSITIONS: dict[str, int] = {"EXCH": 0, "NAME": 1, "SECURITY_ID": 2}

#: NSE/BSE index options tick in 5 paise.
DEFAULT_TICK_SIZE: float = 0.05

#: Some scrip masters publish TICK_SIZE in paise (``5``) and some in rupees (``0.05``).
#: A value at or above this is read as paise and divided by 100.
#:
#: Money: the two failure modes are not symmetric. Treating a real 0.05 tick as 5.00
#: rounds an exit limit by up to five rupees a unit (hundreds of rupees a lot) or misses
#: the fill and strands the position. Treating a real 5.00 tick as 0.05 gets the order
#: rejected by the exchange for a bad price step — loud, immediate and free. So an
#: ambiguous tick always resolves to the smaller value.
_PAISE_THRESHOLD: float = 1.0

#: Where ``contract``/``tick_size``/``freeze_qty`` look for a SECURITY_ID. ``fno`` is the
#: only one fetched on demand; the other two are consulted only if already loaded,
#: because a miss must not trigger two more multi-megabyte downloads, and because the
#: same numeric token can mean different instruments in different segments.
ID_LOOKUP_ORDER: tuple[str, ...] = ("fno", "mcx", "equity")

#: Matches the date half of an ISO timestamp, so a time suffix can be dropped without
#: touching month-name formats like "27-OCT-2026".
_ISO_DATETIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]")

_EXPIRY_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d %Y",
)


def _norm_label(value: Any) -> str:
    """Upper-case and collapse internal whitespace: ``"  nifty  50 "`` -> ``"NIFTY 50"``."""
    return " ".join(str(value or "").split()).upper()


def _squash(value: Any) -> str:
    """Keep only letters and digits, upper-cased: ``"Nifty-50"`` -> ``"NIFTY50"``."""
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _as_float(value: Any, default: float = 0.0) -> float:
    """Parse a CSV cell as a float, returning `default` on anything unparseable."""
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Parse a CSV cell as an int, tolerating ``"75.0"`` and returning `default` on junk."""
    parsed = _as_float(value, float(default))
    try:
        return int(parsed)
    except (OverflowError, ValueError):
        return default


def _parse_expiry(value: Any) -> date | None:
    """Parse an EXPIRY_DATE cell into a date, or None when it is blank or unreadable.

    The docs name the column but not its format, so every plausible encoding the NSE
    masters use is accepted — ISO, day-first, month-name and epoch stamps.
    """
    text = str(value or "").strip()
    if not text:
        return None
    # "2026-08-25T00:00:00" and "2026-08-25 14:30:00" carry a time. Splitting on "T"
    # unconditionally would mangle "27-OCT-2026", so only an ISO date is split that way.
    iso = _ISO_DATETIME_RE.match(text)
    text = iso.group(1) if iso else text.split()[0].strip()
    if not text:
        return None
    if text.isdigit():
        if len(text) == 8:  # YYYYMMDD
            try:
                return datetime.strptime(text, "%Y%m%d").date()
            except ValueError:
                return None
        if len(text) in (10, 13):  # epoch seconds or milliseconds
            raw = int(text)
            seconds = raw / 1000.0 if len(text) == 13 else float(raw)
            try:
                return clock.to_ist(seconds).date()
            except (OverflowError, OSError, ValueError):
                return None
        return None
    for fmt in _EXPIRY_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _column_map(source: str, header: list[str]) -> dict[str, int]:
    """Map column name -> position, by header name with a documented positional fallback.

    The index file's header is known to lie about its second column, so that source is
    mapped positionally and its header is ignored.
    """
    if source == "index":
        return dict(INDEX_COLUMN_POSITIONS)
    by_name = {_squash(name): pos for pos, name in enumerate(header)}
    mapping: dict[str, int] = {}
    for name, pos in SHARED_COLUMN_POSITIONS.items():
        found = by_name.get(_squash(name))
        if found is not None:
            mapping[name] = found
        elif pos < len(header):
            mapping[name] = pos
            logger.warning(
                "source=%s header has no %s column; falling back to position %d (%r)",
                source,
                name,
                pos,
                header[pos],
            )
    return mapping


def _validate_master(source: str, text: str) -> list[str]:
    """Confirm `text` really is the instrument master for `source`; return its header.

    Raises DataError naming the problem. An HTML error page, a JSON error body and a
    header-only truncated file all arrive with HTTP 200 from time to time, and any of
    them would otherwise overwrite a cache that is still good.
    """
    if not text or not text.strip():
        raise DataError(f"instrument master source={source} came back empty", source=source)

    head = text.lstrip()[:512]
    if head[:1] in ("<", "{", "["):
        kind = "an HTML page" if head[:1] == "<" else "a JSON body"
        raise DataError(
            f"instrument master source={source} is {kind}, not CSV: {head[:160]!r}",
            source=source,
        )

    reader = csv.reader(io.StringIO(text))
    try:
        raw_header = next(reader)
    except StopIteration:  # pragma: no cover - guarded by the empty check above
        raise DataError(f"instrument master source={source} has no header row", source=source) from None

    header = [h.strip() for h in raw_header]
    names = {_squash(h) for h in header}
    if "SECURITYID" not in names:
        raise DataError(
            f"instrument master source={source} header has no SECURITY_ID column: {header[:6]}",
            source=source,
        )
    if source != "index" and "TRADINGSYMBOL" not in names:
        raise DataError(
            f"instrument master source={source} header has no TRADING_SYMBOL column: {header[:6]}",
            source=source,
        )

    minimum = MIN_COLUMNS.get(source, 3)
    if len(header) < minimum:
        raise DataError(
            f"instrument master source={source} has {len(header)} columns, expected at least {minimum}",
            source=source,
        )
    documented = DOCUMENTED_COLUMNS.get(source)
    if documented is not None and len(header) != documented:
        logger.warning(
            "source=%s header has %d columns, the docs say %d - reading by name where possible",
            source,
            len(header),
            documented,
        )

    # Sample the first rows: a garbled or half-downloaded file shows up as rows that do
    # not have the header's shape.
    sampled = 0
    malformed = 0
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        sampled += 1
        if len(row) < minimum:
            malformed += 1
        if sampled >= 200:
            break
    if sampled == 0:
        raise DataError(
            f"instrument master source={source} is header-only - truncated download",
            source=source,
        )
    if malformed * 10 > sampled * 3:  # >30% of the sample is the wrong shape
        raise DataError(
            f"instrument master source={source}: {malformed} of the first {sampled} rows have "
            f"fewer than {minimum} columns - the download looks corrupt",
            source=source,
        )
    return header


@dataclass
class _Table:
    """One parsed instrument master, with its lookups built once at load time."""

    source: str
    header: list[str]
    col: dict[str, int]
    rows: list[tuple[str, ...]]
    by_id: dict[str, int]
    by_symbol: dict[str, list[int]]
    index_by_name: dict[str, list[tuple[str, str]]]
    index_by_squash: dict[str, list[tuple[str, str]]]
    mtime: float
    loaded_at: int
    #: Per-symbol lot-size candidates, memoised on first use so EXPIRY_DATE is parsed
    #: for the handful of symbols this bot trades and not for all ~100k F&O rows.
    lot_memo: dict[str, list[tuple[date | None, int, str, str]]] = field(default_factory=dict)

    def cell(self, row: tuple[str, ...], name: str, default: str = "") -> str:
        """Read one named column out of a row, tolerating a short row."""
        pos = self.col.get(name)
        if pos is None or pos >= len(row):
            return default
        return row[pos]


class Instruments:
    """Cached instrument masters and every id, lot, tick and freeze-quantity lookup."""

    def __init__(self, client: "INDstocksClient", cache_dir: Path | str | None = None) -> None:
        """Hold the API client and the directory the CSVs are cached in.

        Nothing is fetched here: each source is downloaded and parsed the first time a
        lookup needs it, so constructing this object is cheap and safe at import time.
        """
        self.client = client
        default_dir = Path(__file__).resolve().parent.parent / "data" / "instruments"
        self.cache_dir = Path(cache_dir) if cache_dir else default_dir
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"cannot create the instrument cache directory {self.cache_dir}: {exc}",
                cache_dir=str(self.cache_dir),
            ) from exc

        self._lock = RLock()
        self._tables: dict[str, _Table] = {}
        #: Per-source: path, row count, age, staleness and the last fetch error. The UI
        #: reads this so "lot size unverified" is visible rather than implied.
        self.cache_status: dict[str, dict[str, Any]] = {}
        #: Per-underlying: which file answered ``lot_size`` and with what.
        self.lot_size_source: dict[str, dict[str, Any]] = {}
        #: Last result of ``resolve_underlyings``, for the UI.
        self.resolved_underlyings: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ cache

    def path_for(self, source: str) -> Path:
        """On-disk path of a cached master, ``data/instruments/<source>.csv``."""
        return self.cache_dir / f"{self._check_source(source)}.csv"

    def refresh(self, source: str, force: bool = False) -> None:
        """Ensure the cached master for `source` is current and parsed into memory.

        A no-op when the cache is fresh and already loaded. Refetches when the copy is
        older than 12h or was written on an earlier IST trading day, because lot sizes
        and circuit bands are republished overnight.
        """
        source = self._check_source(source)
        with self._lock:
            path = self.path_for(source)
            stale, why = self._is_stale(path)
            degraded = False
            if force or stale:
                # False means the fetch failed and the previous copy is still in place.
                degraded = not self._download(
                    source, path, "forced refresh" if force and not stale else why
                )

            if not path.exists():
                raise DataError(
                    f"no instrument master on disk for source={source} at {path}",
                    source=source,
                    path=str(path),
                )
            try:
                mtime = path.stat().st_mtime
            except OSError as exc:
                raise DataError(
                    f"cannot stat the cached master for source={source}: {exc}", source=source
                ) from exc

            table = self._tables.get(source)
            if table is None or table.mtime != mtime:
                self._tables[source] = self._load(source, path, mtime)
            self._note_status(source, path, stale=degraded)

    def _check_source(self, source: str) -> str:
        """Validate a source name against the endpoint's enum."""
        name = str(source or "").strip().lower()
        if name not in SOURCES:
            raise ConfigError(
                f"unknown instrument source {source!r}; expected one of {', '.join(SOURCES)}",
                source=str(source),
            )
        return name

    def _is_stale(self, path: Path) -> tuple[bool, str]:
        """Whether the cached file must be refetched, and the reason in plain English."""
        if not path.exists():
            return True, "no cached copy yet"
        try:
            st = path.stat()
        except OSError as exc:
            return True, f"cannot stat the cached copy ({exc})"
        if st.st_size == 0:
            return True, "the cached copy is empty"

        # clock.now() so INDBOT_FAKE_NOW replays deterministically.
        age = clock.epoch_s(clock.now()) - st.st_mtime
        if age > MAX_AGE_SECONDS:
            return True, f"the cached copy is {age / 3600.0:.1f}h old (limit 12h)"

        written_on = clock.to_ist(st.st_mtime).date()
        today = clock.today()
        reference = today if clock.is_trading_day(today) else clock.prev_trading_day(today)
        if written_on < reference:
            # A weekend copy is fine; one from before the last trading session is not,
            # because lot sizes and bands changed at that session's start.
            return True, f"the cached copy is from {written_on}, the last trading day is {reference}"
        return False, "fresh"

    def _download(self, source: str, path: Path, why: str) -> bool:
        """Fetch, validate and atomically replace one cached master; True if it was replaced.

        A download that does not validate leaves an existing good cache in place, returns
        False and is reported through the log and ``cache_status`` — stale-by-a-day data is
        survivable, an HTML error page parsed as lot sizes is not.
        """
        logger.info("fetching instrument master source=%s (%s)", source, why)
        try:
            text = self.client.instruments_csv(source)
            _validate_master(source, text)
        except Exception as exc:  # noqa: BLE001 - re-raised or reported below, never swallowed
            detail = getattr(exc, "message", None) or str(exc)
            try:
                previous = path.stat() if path.exists() else None
            except OSError:
                previous = None
            if previous is not None and previous.st_size > 0:
                logger.error(
                    "source=%s refresh failed (%s); keeping the cached copy from %s - "
                    "lot sizes and tick sizes may be a day out of date",
                    source,
                    detail,
                    clock.to_ist(previous.st_mtime).isoformat(timespec="minutes"),
                )
                self._note_status(source, path, error=detail, stale=True)
                return False
            self._note_status(source, path, error=detail, stale=True)
            raise DataError(
                f"cannot load the instrument master for source={source} and no cached copy exists: {detail}",
                source=source,
            ) from exc

        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(self.cache_dir), prefix=f".{source}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                    fh.write(text)
                os.replace(tmp_name, path)  # atomic: a crash mid-write cannot truncate the cache
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise DataError(
                f"cannot write the instrument cache for source={source}: {exc}",
                source=source,
                path=str(path),
            ) from exc
        logger.info("instrument master source=%s cached at %s (%d bytes)", source, path, len(text))
        return True

    def _load(self, source: str, path: Path, mtime: float) -> _Table:
        """Parse a cached CSV into row tuples plus the id and symbol lookups."""
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise DataError(
                f"cannot read the cached master for source={source}: {exc}", source=source
            ) from exc

        header = _validate_master(source, text)
        col = _column_map(source, header)

        reader = csv.reader(io.StringIO(text))
        next(reader, None)  # header
        rows: list[tuple[str, ...]] = []
        for raw in reader:
            if not raw or not any(cell.strip() for cell in raw):
                continue
            rows.append(tuple(cell.strip() for cell in raw))

        id_pos = col.get("SECURITY_ID", INDEX_COLUMN_POSITIONS["SECURITY_ID"])
        sym_pos = col.get("SYMBOL_NAME")
        by_id: dict[str, int] = {}
        by_symbol: dict[str, list[int]] = {}
        index_by_name: dict[str, list[tuple[str, str]]] = {}
        index_by_squash: dict[str, list[tuple[str, str]]] = {}
        duplicate_ids = 0
        short_rows = 0

        for i, row in enumerate(rows):
            if id_pos >= len(row):
                short_rows += 1
                continue
            sid = row[id_pos]
            if sid:
                if sid in by_id:
                    duplicate_ids += 1
                else:
                    by_id[sid] = i
            if source == "index":
                # Positional parse: column 1 is the index NAME despite its SEGMENT header.
                name_pos = INDEX_COLUMN_POSITIONS["NAME"]
                exch_pos = INDEX_COLUMN_POSITIONS["EXCH"]
                if name_pos < len(row) and sid:
                    exch = row[exch_pos] if exch_pos < len(row) else ""
                    entry = (exch.upper(), sid)
                    index_by_name.setdefault(_norm_label(row[name_pos]), []).append(entry)
                    index_by_squash.setdefault(_squash(row[name_pos]), []).append(entry)
            elif sym_pos is not None and sym_pos < len(row):
                symbol = _norm_label(row[sym_pos])
                if symbol:
                    by_symbol.setdefault(symbol, []).append(i)

        if duplicate_ids:
            logger.debug("source=%s has %d duplicate SECURITY_ID rows; first row wins", source, duplicate_ids)
        if short_rows:
            logger.warning("source=%s: skipped %d rows with no SECURITY_ID column", source, short_rows)
        logger.info("source=%s parsed: %d rows, %d ids, %d symbols", source, len(rows), len(by_id), len(by_symbol))

        return _Table(
            source=source,
            header=header,
            col=col,
            rows=rows,
            by_id=by_id,
            by_symbol=by_symbol,
            index_by_name=index_by_name,
            index_by_squash=index_by_squash,
            mtime=mtime,
            loaded_at=clock.epoch_s(clock.now()),
        )

    def _note_status(
        self,
        source: str,
        path: Path,
        *,
        error: str | None = None,
        stale: bool | None = None,
    ) -> None:
        """Record cache state for the UI: age, row count and the last fetch error."""
        entry = self.cache_status.setdefault(source, {})
        table = self._tables.get(source)
        age_hours: float | None = None
        written_at: str | None = None
        if path.exists():
            try:
                mtime = path.stat().st_mtime
                age_hours = round((clock.epoch_s(clock.now()) - mtime) / 3600.0, 2)
                written_at = clock.to_ist(mtime).isoformat(timespec="minutes")
            except OSError:
                pass
        entry.update(
            source=source,
            path=str(path),
            exists=path.exists(),
            rows=len(table.rows) if table else 0,
            columns=len(table.header) if table else 0,
            written_at=written_at,
            age_hours=age_hours,
            checked_at=clock.now().isoformat(timespec="seconds"),
        )
        if error is not None:
            entry["last_error"] = error
        entry["stale"] = bool(stale) if stale is not None else bool(error)

    def _ensure(self, source: str) -> _Table:
        """Return the parsed table for `source`, refreshing and loading it if needed."""
        with self._lock:
            self.refresh(source)
            table = self._tables.get(source)
            if table is None:  # pragma: no cover - refresh raises before this
                raise DataError(f"instrument master source={source} is not loaded", source=source)
            return table

    # ------------------------------------------------------------- index file

    def index_scrip(self, label: str, *, exchange: str | None = None) -> str | None:
        """Resolve an index label such as ``"NIFTY 50"`` to its SECURITY_ID, or None.

        The index file is three columns (EXCH, SEGMENT, SECURITY_ID) where column 2 holds
        the index NAME despite its header, so it is read positionally. Matching is
        case-insensitive and whitespace-tolerant, with a punctuation-insensitive second
        pass so ``"Nifty-50"`` still finds ``"NIFTY 50"``.
        """
        wanted = _norm_label(label)
        if not wanted:
            return None
        table = self._ensure("index")

        matches = table.index_by_name.get(wanted)
        matched_by = "name"
        if not matches:
            matches = table.index_by_squash.get(_squash(label))
            matched_by = "squashed name"
        if not matches:
            return None

        if exchange:
            want_exch = str(exchange).strip().upper()
            same_exchange = [m for m in matches if m[0] == want_exch]
            if same_exchange:
                matches = same_exchange
            else:
                # The same index exists once; a differing EXCH string is a labelling
                # difference, not a different instrument. Take it, but say so.
                logger.warning(
                    "index %r resolved on %s, not the expected %s - using it anyway (id=%s)",
                    wanted,
                    ", ".join(sorted({m[0] for m in matches})),
                    want_exch,
                    matches[0][1],
                )

        if len(matches) > 1:
            logger.warning(
                "index %r matches %d rows in the index file (%s); taking the first",
                wanted,
                len(matches),
                ", ".join(f"{e}:{s}" for e, s in matches[:4]),
            )
        logger.debug("index %r -> %s (matched by %s)", wanted, matches[0][1], matched_by)
        return matches[0][1]

    def resolve_underlyings(self, cfg: Any) -> dict[str, dict[str, Any]]:
        """Return the UNDERLYINGS table with every ``scrip`` corrected from the live index file.

        Only NIFTY's hardcoded id is known good; the rest are guesses, so this is what
        turns a wrong constant into a logged correction instead of a silent order on the
        wrong instrument. A label that cannot be resolved keeps its constant and is
        marked ``resolved: False`` so the UI can show it as unverified.

        ``ws`` is deliberately NOT resolved from this file: the docs state the WebSocket
        index tokens live in a different token space (``NIDX:26000`` vs ``40000001``), so
        copying a REST id onto the feed would subscribe to the wrong instrument.
        """
        base = self._base_underlyings(cfg)
        enabled = {str(n).strip().upper() for n in (getattr(cfg, "underlyings", None) or [])}
        resolved: dict[str, dict[str, Any]] = {}

        # Load the index master once. Without it every constant stays in place and is
        # marked unverified, rather than each underlying retrying a failing download.
        index_available = True
        try:
            self._ensure("index")
        except DataError as exc:
            index_available = False
            logger.error(
                "cannot resolve index scrips, the index master is unavailable (%s); keeping the "
                "hardcoded ids and marking all of them UNVERIFIED",
                getattr(exc, "message", exc),
            )

        for name, raw_entry in base.items():
            entry: dict[str, Any] = dict(raw_entry)
            key = str(name).strip().upper()
            constant = str(entry.get("scrip", "") or "")
            entry["scrip_constant"] = constant
            entry["enabled"] = (key in enabled) if enabled else True

            label = str(entry.get("label") or name)
            found: str | None = None
            matched_label: str | None = None
            if index_available:
                for candidate in self._label_candidates(name, entry):
                    found = self.index_scrip(candidate, exchange=entry.get("exchange"))
                    if found:
                        matched_label = candidate
                        break

            if found:
                entry["scrip"] = found
                entry["resolved"] = True
                entry["resolved_from"] = matched_label
                if constant and found != constant:
                    logger.warning(
                        "%s scrip mismatch: config says %s, the index file says %s for %r - using %s",
                        key,
                        constant,
                        found,
                        matched_label,
                        found,
                    )
                    entry["scrip_mismatch"] = True
                else:
                    entry["scrip_mismatch"] = False
            else:
                entry["resolved"] = False
                entry["scrip_mismatch"] = False
                if index_available:
                    logger.warning(
                        "%s: index label %r is not in the index master; keeping the hardcoded "
                        "scrip %s as UNVERIFIED",
                        key,
                        label,
                        constant or "(none)",
                    )
            resolved[name] = entry

        self.resolved_underlyings = resolved
        return resolved

    @staticmethod
    def _label_candidates(name: str, entry: dict[str, Any]) -> list[str]:
        """Labels to try against the index file, best first: label, aliases, then the key."""
        out: list[str] = []
        for candidate in (
            entry.get("label"),
            *(entry.get("index_aliases") or []),
            name,
        ):
            text = str(candidate or "").strip()
            if text and all(_squash(text) != _squash(seen) for seen in out):
                out.append(text)
        return out

    @staticmethod
    def _base_underlyings(cfg: Any) -> dict[str, dict[str, Any]]:
        """The static UNDERLYINGS reference table, from `cfg` or from indbot.config."""
        table = getattr(cfg, "UNDERLYINGS", None)
        if isinstance(table, dict) and table:
            return table
        if isinstance(cfg, dict) and cfg:
            return cfg
        try:
            from .config import UNDERLYINGS  # local import: config.py must not be a hard dep here
        except ImportError as exc:  # pragma: no cover - config.py is always present in a build
            raise ConfigError(f"cannot import indbot.config.UNDERLYINGS: {exc}") from exc
        return UNDERLYINGS

    # -------------------------------------------------------------- F&O lookups

    def lot_size(self, underlying: str, expiry: str | None = None) -> int:
        """Units in one lot of `underlying`, from the F&O master's LOT_UNITS column.

        Matches on SYMBOL_NAME and, when `expiry` is given, on EXPIRY_DATE; otherwise the
        nearest expiry still to come is used, because that is the one the bot trades.
        FALLBACK_LOTS is used only when the lookup genuinely fails, and every resolution
        is recorded in ``lot_size_source`` — a wrong lot size mis-sizes every trade, so
        nothing here guesses silently.
        """
        key = _norm_label(underlying)
        fallback = FALLBACK_LOTS.get(key, 0)

        try:
            table = self._ensure("fno")
        except DataError as exc:
            detail = getattr(exc, "message", None) or str(exc)
            if fallback <= 0:
                raise DataError(
                    f"no lot size for {key}: the F&O master is unavailable ({detail}) and there "
                    f"is no fallback for this underlying",
                    underlying=key,
                ) from exc
            logger.error(
                "lot size for %s falling back to %d units: the F&O master is unavailable (%s)",
                key,
                fallback,
                detail,
            )
            self._record_lot_source(key, expiry, fallback, "fallback", f"F&O master unavailable: {detail}")
            return fallback

        candidates = self._lot_candidates(table, key)
        if not candidates:
            if fallback <= 0:
                raise DataError(
                    f"no lot size for {key}: no SYMBOL_NAME row in the F&O master and no fallback",
                    underlying=key,
                )
            logger.warning(
                "lot size for %s falling back to %d units: no SYMBOL_NAME=%s row in the F&O master",
                key,
                fallback,
                key,
            )
            self._record_lot_source(key, expiry, fallback, "fallback", "no SYMBOL_NAME row in the F&O master")
            return fallback

        wanted = _parse_expiry(expiry) if expiry else None
        chosen: list[tuple[date | None, int, str, str]] = []
        detail = ""

        if expiry:
            wanted_text = str(expiry).strip()
            chosen = [
                c
                for c in candidates
                if (wanted is not None and c[0] == wanted) or _squash(c[2]) == _squash(wanted_text)
            ]
            detail = f"EXPIRY_DATE={wanted_text}"
            if not chosen:
                logger.warning(
                    "lot size for %s: expiry %s is not in the F&O master; using the nearest "
                    "listed expiry instead",
                    key,
                    wanted_text,
                )
                detail = f"expiry {wanted_text} absent, used the nearest listed expiry"

        if not chosen:
            dated = [c for c in candidates if c[0] is not None]
            today = clock.today()
            # Today's own expiry still counts: a weekly expiring today is tradable.
            upcoming = [c for c in dated if c[0] is not None and c[0] >= today]
            upcoming.sort(key=lambda c: c[0] or today)
            if upcoming:
                nearest = upcoming[0][0]
                chosen = [c for c in dated if c[0] == nearest]
                detail = detail or f"nearest expiry {nearest}"
            elif dated:
                latest = max(c[0] for c in dated if c[0] is not None)
                chosen = [c for c in dated if c[0] == latest]
                detail = detail or f"latest listed expiry {latest} (no upcoming expiry in the master)"
            else:
                chosen = candidates
                detail = detail or "no readable EXPIRY_DATE; used every row for this symbol"

        counts = Counter(c[1] for c in chosen)
        if len(counts) > 1:
            # Different lots on one expiry should not happen; say so loudly and take the
            # most common rather than whichever row sorted first.
            logger.warning(
                "lot size for %s is not consistent across %d rows (%s); using the most common",
                key,
                len(chosen),
                ", ".join(f"{lot}x{n}" for lot, n in counts.most_common()),
            )
        lot = counts.most_common(1)[0][0]

        if fallback and lot != fallback:
            logger.info(
                "lot size for %s is %d units per the F&O master (%s); the hardcoded fallback says %d",
                key,
                lot,
                detail,
                fallback,
            )
        self._record_lot_source(key, expiry, lot, "fno:LOT_UNITS", detail)
        return lot

    def _lot_candidates(self, table: _Table, symbol: str) -> list[tuple[date | None, int, str, str]]:
        """(expiry_date, lot, raw_expiry, instrument_name) for one SYMBOL_NAME, memoised.

        Option rows are preferred over futures rows when both exist, because options are
        what this bot trades.
        """
        with self._lock:
            cached = table.lot_memo.get(symbol)
            if cached is not None:
                return cached

            out: list[tuple[date | None, int, str, str]] = []
            for i in table.by_symbol.get(symbol, []):
                row = table.rows[i]
                lot = _as_int(table.cell(row, "LOT_UNITS"), 0)
                if lot <= 0:
                    continue
                raw_expiry = table.cell(row, "EXPIRY_DATE")
                out.append(
                    (
                        _parse_expiry(raw_expiry),
                        lot,
                        raw_expiry,
                        table.cell(row, "INSTRUMENT_NAME"),
                    )
                )
            options = [c for c in out if c[3].upper().startswith("OPT")]
            result = options or out
            table.lot_memo[symbol] = result
            return result

    def _record_lot_source(
        self, underlying: str, expiry: str | None, lot: int, source: str, detail: str
    ) -> None:
        """Record which file answered a lot-size lookup, for the UI to display."""
        self.lot_size_source[underlying] = {
            "underlying": underlying,
            "lot_size": lot,
            "source": source,
            "verified": source.startswith("fno"),
            "expiry": expiry,
            "detail": detail,
            "at": clock.now().isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------- per-contract lookups

    def _row_for_id(self, security_id: str, source: str | None = None) -> tuple[str, _Table, tuple[str, ...]] | None:
        """Find a row by SECURITY_ID, in the F&O master first, then any loaded master.

        Only ``fno`` is fetched on demand: a miss must not trigger two more multi-megabyte
        downloads, and the same numeric token can identify different instruments in
        different segments, so an already-loaded equity file is consulted last.
        """
        sid = str(security_id or "").strip()
        if not sid:
            return None
        order = (self._check_source(source),) if source else ID_LOOKUP_ORDER
        for name in order:
            table = self._tables.get(name)
            if table is None:
                if name != "fno" and not source:
                    continue  # not loaded and not worth downloading for a single id
                try:
                    table = self._ensure(name)
                except DataError as exc:
                    logger.error("cannot search source=%s for security_id=%s: %s", name, sid, exc)
                    continue
            pos = table.by_id.get(sid)
            if pos is not None:
                return name, table, table.rows[pos]
        return None

    def tick_size(self, security_id: str, *, source: str | None = None) -> float:
        """Minimum price step for a contract, in rupees. Defaults to 0.05 when unknown.

        An ambiguous published value is resolved to the smaller tick — see
        ``_PAISE_THRESHOLD``: a too-coarse tick costs real money on an exit, a too-fine
        one only gets the order rejected.
        """
        found = self._row_for_id(security_id, source)
        if found is None:
            logger.warning(
                "no instrument row for security_id=%s; using the default tick of %.2f",
                security_id,
                DEFAULT_TICK_SIZE,
            )
            return DEFAULT_TICK_SIZE

        name, table, row = found
        raw = table.cell(row, "TICK_SIZE")
        value = _as_float(raw, 0.0)
        if value <= 0:
            logger.warning(
                "security_id=%s (source=%s) has TICK_SIZE=%r; using the default tick of %.2f",
                security_id,
                name,
                raw,
                DEFAULT_TICK_SIZE,
            )
            return DEFAULT_TICK_SIZE
        if value >= _PAISE_THRESHOLD:
            converted = value / 100.0  # constant divisor; no division-by-zero path
            logger.warning(
                "security_id=%s (source=%s) publishes TICK_SIZE=%s; reading it as paise -> %.4f "
                "rupees, the safer of the two readings",
                security_id,
                name,
                raw,
                converted,
            )
            value = converted
        return value

    def freeze_qty(self, security_id: str, *, source: str | None = None) -> int:
        """Largest quantity the exchange accepts on one order, or 0 when not published.

        0 means "no published cap", not "no quantity allowed": callers must never divide
        by this without guarding it. The bot trades one lot, so the freeze quantity never
        binds here; it exists for order slicing.
        """
        found = self._row_for_id(security_id, source)
        if found is None:
            logger.warning("no instrument row for security_id=%s; freeze quantity unknown", security_id)
            return 0
        name, table, row = found
        qty = _as_int(table.cell(row, "FREEZE_QTY"), 0)
        if qty <= 0:
            logger.debug("security_id=%s (source=%s) publishes no FREEZE_QTY", security_id, name)
            return 0
        return qty

    def contract(self, security_id: str, *, source: str | None = None) -> dict[str, Any] | None:
        """Every published field for one SECURITY_ID, raw plus typed, or None if absent.

        Raw columns keep their upper-case CSV names; the typed, normalised values are
        added under lower-case keys so callers never re-parse a cell.
        """
        found = self._row_for_id(security_id, source)
        if found is None:
            return None
        name, table, row = found

        out: dict[str, Any] = {}
        for pos, column in enumerate(table.header):
            out[column.strip()] = row[pos] if pos < len(row) else ""

        expiry = _parse_expiry(table.cell(row, "EXPIRY_DATE"))
        strike = _as_float(table.cell(row, "STRIKE_PRICE"), 0.0)
        out.update(
            source=name,
            security_id=table.cell(row, "SECURITY_ID"),
            trading_symbol=table.cell(row, "TRADING_SYMBOL"),
            custom_symbol=table.cell(row, "CUSTOM_SYMBOL"),
            symbol_name=table.cell(row, "SYMBOL_NAME"),
            exchange=table.cell(row, "EXCH").upper(),
            segment=table.cell(row, "SEGMENT"),
            instrument_name=table.cell(row, "INSTRUMENT_NAME"),
            option_type=table.cell(row, "OPTION_TYPE").upper() or None,
            strike=strike if strike > 0 else None,
            expiry=expiry.isoformat() if expiry else None,
            expiry_raw=table.cell(row, "EXPIRY_DATE"),
            lot_size=_as_int(table.cell(row, "LOT_UNITS"), 0),
            tick_size=self.tick_size(security_id, source=name),
            freeze_qty=_as_int(table.cell(row, "FREEZE_QTY"), 0),
            upper_limit=_as_float(table.cell(row, "UPPER_LIMIT"), 0.0),
            lower_limit=_as_float(table.cell(row, "LOWER_LIMIT"), 0.0),
            # On mcx the circuit columns are PERCENTAGES of the previous close, not
            # prices. Flag it so no caller sends 4.0 as a limit price.
            limits_are_percent=(name == "mcx"),
        )
        return out

    # ----------------------------------------------------------------- UI state

    def status(self) -> dict[str, Any]:
        """Cache freshness, lot-size provenance and resolved scrips, for the dashboard."""
        return {
            "cache_dir": str(self.cache_dir),
            "cache": {name: dict(entry) for name, entry in self.cache_status.items()},
            "lot_size_source": {name: dict(entry) for name, entry in self.lot_size_source.items()},
            "underlyings": {
                name: {
                    "scrip": entry.get("scrip"),
                    "scrip_constant": entry.get("scrip_constant"),
                    "resolved": entry.get("resolved", False),
                    "scrip_mismatch": entry.get("scrip_mismatch", False),
                    "label": entry.get("label"),
                    "enabled": entry.get("enabled", True),
                }
                for name, entry in self.resolved_underlyings.items()
            },
            "fallback_lots": dict(FALLBACK_LOTS),
        }
