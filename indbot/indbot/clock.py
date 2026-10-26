"""IST clock and NSE/BSE session helpers.

Every time in indbot is IST. Nothing calls `datetime.now()` directly — it calls
`clock.now()`, which honours INDBOT_FAKE_NOW so a session can be replayed.

The API speaks two units: request windows are epoch MILLISECONDS, candle `ts`
values are epoch SECONDS. `epoch_ms` and `epoch_s` keep that straight.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)

#: Intraday candle buckets are anchored to the 09:15 session open, not to wall-clock
#: :00/:05 boundaries. A 15minute candle falls at 09:15, 09:30, 09:45 …
SESSION_ANCHOR = SESSION_OPEN

# NSE/BSE trading holidays. Verify against the exchange circular each year — a wrong
# holiday only costs a wasted poll, a missing one costs a day of bad data.
HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 26),   # Republic Day
        date(2026, 3, 4),    # Holi
        date(2026, 3, 21),   # Id-ul-Fitr
        date(2026, 3, 27),   # Ram Navami
        date(2026, 3, 31),   # Mahavir Jayanti
        date(2026, 4, 3),    # Good Friday
        date(2026, 4, 14),   # Dr. Ambedkar Jayanti
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 5, 27),   # Bakri Id
        date(2026, 6, 26),   # Muharram
        date(2026, 8, 15),   # Independence Day
        date(2026, 8, 26),   # Ganesh Chaturthi
        date(2026, 10, 2),   # Gandhi Jayanti
        date(2026, 10, 20),  # Dussehra
        date(2026, 11, 9),   # Diwali Balipratipada
        date(2026, 11, 24),  # Guru Nanak Jayanti
        date(2026, 12, 25),  # Christmas
    }
)

HOLIDAYS_2027: frozenset[date] = frozenset(
    {
        date(2027, 1, 26),
        date(2027, 3, 26),
        date(2027, 4, 14),
        date(2027, 5, 1),
        date(2027, 8, 15),
        date(2027, 10, 2),
        date(2027, 12, 25),
    }
)

HOLIDAYS: frozenset[date] = HOLIDAYS_2026 | HOLIDAYS_2027

_FAKE_NOW_ENV = "INDBOT_FAKE_NOW"
_FAKE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def now() -> datetime:
    """Current IST time, or INDBOT_FAKE_NOW when that env var is set.

    The fake clock is how the backtester and the tests replay a session without
    waiting for it. It is read on every call, so it can be changed mid-process.
    """
    raw = os.environ.get(_FAKE_NOW_ENV)
    if raw:
        for fmt in _FAKE_FORMATS:
            try:
                return datetime.strptime(raw.strip(), fmt).replace(tzinfo=IST)
            except ValueError:
                continue
    return datetime.now(IST)


def today() -> date:
    """Today's IST calendar date."""
    return now().date()


def to_ist(epoch_seconds: int | float) -> datetime:
    """Convert a candle's epoch-SECONDS `ts` into an IST datetime."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=IST)


def epoch_s(dt: datetime) -> int:
    """Epoch seconds, the unit of a candle's `ts`."""
    return int(_aware(dt).timestamp())


def epoch_ms(dt: datetime) -> int:
    """Epoch milliseconds, the unit of `start_time` / `end_time` query params."""
    return int(_aware(dt).timestamp() * 1000)


def _aware(dt: datetime) -> datetime:
    """Treat a naive datetime as IST rather than as local time."""
    return dt.replace(tzinfo=IST) if dt.tzinfo is None else dt


def at(d: date, t: time) -> datetime:
    """Combine a date and a time into an IST datetime."""
    return datetime.combine(d, t, tzinfo=IST)


def session_open(d: date | None = None) -> datetime:
    """09:15 IST on `d` (today by default)."""
    return at(d or today(), SESSION_OPEN)


def session_close(d: date | None = None) -> datetime:
    """15:30 IST on `d` (today by default)."""
    return at(d or today(), SESSION_CLOSE)


def is_trading_day(d: date) -> bool:
    """True when `d` is a weekday and not a listed exchange holiday.

    Years beyond the HOLIDAYS table fall back to the weekday test alone, so an
    unlisted holiday reads as a trading day with no candles rather than as an error.
    """
    return d.weekday() < 5 and d not in HOLIDAYS


def is_market_open(dt: datetime | None = None) -> bool:
    """True when `dt` falls inside a live NSE/BSE equity session."""
    dt = _aware(dt or now())
    if not is_trading_day(dt.date()):
        return False
    return SESSION_OPEN <= dt.timetz().replace(tzinfo=None) <= SESSION_CLOSE


def parse_hhmm(value: str) -> time:
    """Parse "15:10" into a time. Raises ValueError on anything else."""
    hh, _, mm = value.strip().partition(":")
    return time(int(hh), int(mm))


def today_at(value: str) -> datetime:
    """Today's IST datetime at an "HH:MM" string, for config windows like flat_by."""
    return at(today(), parse_hhmm(value))


def tod_bucket(dt: datetime | None = None) -> str:
    """Which part of the session `dt` falls in — a dimension of the learning bucket.

    Returns one of "open", "morning", "midday", "close". A 09:20 breakout and a
    14:50 breakout are different trades, and the learner is allowed to find that out.
    """
    dt = _aware(dt or now())
    t = dt.timetz().replace(tzinfo=None)
    if t < time(10, 0):
        return "open"
    if t < time(12, 0):
        return "morning"
    if t < time(14, 0):
        return "midday"
    return "close"


def minutes_into_session(dt: datetime | None = None) -> int:
    """Minutes since 09:15. Negative before the open, capped at the close."""
    dt = _aware(dt or now())
    delta = (dt - session_open(dt.date())).total_seconds() / 60.0
    return int(delta)


def prev_trading_day(d: date | None = None) -> date:
    """The most recent trading day strictly before `d`."""
    cur = (d or today()) - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def trading_days(start: date, end: date) -> list[date]:
    """Every trading day in [start, end], inclusive, ascending."""
    out: list[date] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def candle_open_time(dt: datetime, interval_minutes: int) -> datetime:
    """Snap `dt` back to the open of the candle containing it.

    Buckets are anchored to 09:15, so on 5minute they land at 09:15, 09:20, 09:25 …
    rather than on :00/:05 wall-clock boundaries.
    """
    dt = _aware(dt)
    anchor = at(dt.date(), SESSION_ANCHOR)
    if dt < anchor:
        return anchor
    elapsed = int((dt - anchor).total_seconds() // 60)
    return anchor + timedelta(minutes=(elapsed // interval_minutes) * interval_minutes)


INTERVAL_MINUTES: dict[str, int] = {
    "1minute": 1,
    "2minute": 2,
    "3minute": 3,
    "4minute": 4,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
    "120minute": 120,
    "180minute": 180,
    "240minute": 240,
}

#: Largest window a single historical call may span, per the API's interval table.
INTERVAL_MAX_DAYS: dict[str, int] = {
    **{k: 7 for k in ("1minute", "2minute", "3minute", "4minute", "5minute", "10minute", "15minute", "30minute")},
    **{k: 15 for k in ("60minute", "120minute", "180minute", "240minute")},
    "1day": 365,
    "1week": 365,
    "1month": 365,
}


def interval_minutes(interval: str) -> int:
    """Minutes per candle for an API interval label like "5minute"."""
    try:
        return INTERVAL_MINUTES[interval]
    except KeyError as exc:
        raise ValueError(f"unsupported interval {interval!r}") from exc
