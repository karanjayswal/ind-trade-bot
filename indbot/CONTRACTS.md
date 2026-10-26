# indbot — module contracts

**Read this file in full before writing any code.** Every module below is built by a
separate agent in parallel. These signatures are the integration contract: do not
rename, do not change argument order, do not add required arguments. If you need
something not listed here, add it as a keyword argument with a default.

Target: **Python 3.12**, but keep it 3.10-compatible. Standard library plus only the
packages in `requirements.txt` (`fastapi`, `uvicorn`, `httpx`, `websockets`, `pyotp`).
No pandas, no numpy — indicators are hand-rolled over lists of floats.

Package root: `/Users/AL58376/Downloads/Personal_Project/indbot/`
Python package: `indbot/` (so imports read `from indbot.indicators import ema`).

---

## 0. Ground rules

- **All times are IST.** Use `indbot.clock`, never `datetime.now()` bare.
- **Candle `ts` is epoch SECONDS** (as the API returns). Request params
  `start_time`/`end_time` are epoch **milliseconds**. Convert at the API boundary only.
- **Money is rupees, floats.** Premiums are per-unit; a position's rupee value is
  `premium * lot_size`.
- **Never place an order outside `broker.py`.** No module other than `broker.py` and
  `engine.py` may import the order endpoints.
- Every public function gets a docstring with a one-line summary. Type-hint everything.
- Raise `indbot.errors.IndbotError` subclasses, never bare `Exception`.
- No `print()` in library modules — use `logging.getLogger(__name__)`.

## 1. Shared types — `indbot/types.py` (ALREADY WRITTEN, do not edit)

```python
Candle = dict  # {"ts": int (epoch sec), "o": float, "h": float, "l": float, "c": float, "v": int}

@dataclass(frozen=True)
class Signal:
    setup: str          # "orb" | "vwap_reclaim" | "ema_cross" | "supertrend" | "rsi_reversal" | "pullback"
    side: str           # "CE" (bullish) | "PE" (bearish)
    action: str         # "BUY" (open) | "SELL" (close an open position)
    underlying: str     # "NIFTY" | "BANKNIFTY" | "FINNIFTY" | "MIDCPNIFTY" | "SENSEX"
    ts: int             # epoch seconds of the candle that triggered
    spot: float         # underlying price at trigger
    score: float        # 0.0..1.0 confluence score
    reasons: list[str]  # human-readable, shown verbatim in the UI
    bucket: str         # learning key, from Signal.make_bucket()
    stop_pct: float     # fraction of premium, e.g. 0.30
    target_pct: float   # fraction of premium, e.g. 0.50

@dataclass(frozen=True)
class OptionLeg:
    underlying: str
    expiry: str         # "YYYY-MM-DD"
    strike: float
    side: str           # "CE" | "PE"
    security_id: str    # pass straight to POST /order as security_id
    trading_symbol: str
    lot_size: int
    exchange: str       # "NSE" for NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY, "BSE" for SENSEX
    ltp: float
    bid: float
    ask: float
    oi: int
    volume: int
    iv: float           # percent, e.g. 10.5
    delta: float
    # properties: mid, spread, spread_pct, lot_cost (= ltp * lot_size)

@dataclass
class Position:
    id: str; leg: OptionLeg; signal_setup: str; bucket: str
    qty: int                 # units, = lot_size * lots
    entry_price: float; entry_ts: int; entry_order_id: str
    stop_price: float; target_price: float
    high_water: float        # best premium seen, for trailing
    exit_price: float | None; exit_ts: int | None; exit_reason: str | None
    # properties: pnl(ltp), pnl_pct(ltp), is_open

@dataclass(frozen=True)
class RiskVerdict:
    ok: bool
    reasons: list[str]       # why blocked, or why allowed — always populated
    blocking_rule: str | None
```

`Signal.make_bucket(setup, side, ts, regime) -> str` returns
`f"{setup}:{side}:{tod}:{regime}"` where `tod` is one of
`open` (09:15–10:00), `morning` (10:00–12:00), `midday` (12:00–14:00),
`close` (14:00–15:30), and `regime` is one of `trend_up`, `trend_down`, `range`.

## 2. `indbot/errors.py` (ALREADY WRITTEN, do not edit)

`IndbotError`, and subclasses `ConfigError`, `ApiError` (has `.status`, `.payload`),
`AuthError`, `RateLimitError`, `OrderError`, `DataError`, `RiskBlocked`.

## 3. `indbot/clock.py` (ALREADY WRITTEN, do not edit)

```python
IST: timezone                      # UTC+05:30
def now() -> datetime              # tz-aware IST; honours INDBOT_FAKE_NOW="YYYY-MM-DD HH:MM:SS"
def today() -> date
def to_ist(epoch_seconds: int|float) -> datetime
def epoch_s(dt: datetime) -> int
def epoch_ms(dt: datetime) -> int
def session_open(d: date|None = None) -> datetime     # 09:15
def session_close(d: date|None = None) -> datetime    # 15:30
def is_market_open(dt: datetime|None = None) -> bool  # weekday + 09:15..15:30, NOT holidays
def is_trading_day(d: date) -> bool                   # weekday and not in HOLIDAYS_2026
def tod_bucket(dt: datetime) -> str                   # "open"|"morning"|"midday"|"close"
def minutes_into_session(dt: datetime|None = None) -> int
```

---

# MODULES TO BUILD

## A. `indbot/config.py`

```python
UNDERLYINGS: dict[str, dict]   # see below — the 5 indices, static reference data
DEFAULTS: dict                 # every tunable, with its default

@dataclass
class Config:
    # loaded from config.json, merged over DEFAULTS
    mode: str                  # "backtest" | "paper" | "live"
    armed: bool                # live orders only when mode=="live" AND armed
    underlyings: list[str]
    # risk
    max_daily_loss: float = 2000.0
    max_trade_loss: float = 1000.0
    max_trades_per_day: int = 3
    max_open_positions: int = 1
    max_lots: int = 1
    # strategy
    min_score: float = 0.55
    stop_pct: float = 0.30
    target_pct: float = 0.50
    trail_trigger_pct: float = 0.25
    min_stop_pct: float = 0.12      # refuse a stop tighter than this (see RISK ARITHMETIC)
    or_minutes: int = 15
    interval: str = "5minute"
    enabled_setups: list[str]
    # option selection
    min_delta: float = 0.35
    max_delta: float = 0.60
    max_spread_pct: float = 2.0
    min_oi: int = 50000
    min_volume: int = 10000
    max_premium_per_lot: float = 0.0   # 0 = derive from max_trade_loss/stop_pct
    # session
    no_entry_before: str = "09:30"
    no_entry_after: str = "14:45"
    flat_by: str = "15:10"
    # learning
    learning_enabled: bool = True
    min_samples_to_gate: int = 10
    gate_confidence: float = 0.90

    @classmethod
    def load(cls, path: str|Path|None = None) -> "Config"
    def save(self, path: str|Path|None = None) -> None
    def update(self, patch: dict) -> list[str]   # validates, returns list of changed keys
    def as_dict(self) -> dict
    def effective_max_premium(self, lot_size: int) -> float
        """max_premium_per_lot, or max_trade_loss/(stop_pct*lot_size) when it is 0."""
```

`UNDERLYINGS` — exact content required:

```python
UNDERLYINGS = {
  "NIFTY":      {"scrip": "40000001", "exchange": "NSE", "chain_segment": "INDEX",
                 "ws": "NIDX:26000", "fno_segment": "NFO", "strike_step": 50,  "label": "NIFTY 50"},
  "BANKNIFTY":  {"scrip": "40000002", "exchange": "NSE", "chain_segment": "INDEX",
                 "ws": "NIDX:26009", "fno_segment": "NFO", "strike_step": 100, "label": "NIFTY BANK"},
  "FINNIFTY":   {"scrip": "40000003", "exchange": "NSE", "chain_segment": "INDEX",
                 "ws": "NIDX:26037", "fno_segment": "NFO", "strike_step": 50,  "label": "NIFTY FIN SERVICE"},
  "MIDCPNIFTY": {"scrip": "40000016", "exchange": "NSE", "chain_segment": "INDEX",
                 "ws": "NIDX:26074", "fno_segment": "NFO", "strike_step": 25,  "label": "NIFTY MIDCAP SELECT"},
  "SENSEX":     {"scrip": "40000100", "exchange": "BSE", "chain_segment": "INDEX",
                 "ws": "BIDX:1",      "fno_segment": "BFO", "strike_step": 100, "label": "SENSEX"},
}
```

**These scrip ids and ws tokens are GUESSES except NIFTY's `40000001`.** Every one must be
resolvable at runtime from `/market/instruments?source=index` by matching `label` against
the file's second column, and the resolved value must win over the constant. Write it so a
wrong constant is self-healing, and log a warning when a constant disagrees with the file.

## B. `indbot/api.py`

`class INDstocksClient` — thin, typed, rate-limited wrapper over `https://api.indstocks.com`.

```python
def __init__(self, token: str|None = None, *, base_url: str = "https://api.indstocks.com",
             timeout: float = 15.0) -> None
    # token defaults to os.environ["INDSTOCKS_ACCESS_TOKEN"]; raise AuthError if absent.
```

Header is `Authorization: <token>` — **no `Bearer ` prefix**.

Rate-limit buckets, token-bucket per category, enforced in-process (see docs §Rate Limiting):
`order` 10/s, `data` 5/s, `quote` 5/s, `nontrading` 15/s. Retry `429` and `5xx` with
exponential backoff + jitter, max 4 attempts; never retry a `POST /order`.

Methods (all sync, `httpx.Client`):

```python
# user / funds           category="nontrading"
def profile(self) -> dict
def funds(self) -> dict
# instruments            category="data"
def instruments_csv(self, source: str) -> str          # source: equity|fno|index|mcx
def expiries(self, underlying: str) -> list[str]        # GET /market/instruments/expiries
def expired_expiries(self, underlying: str, expiry_from: str, expiry_to: str) -> list[str]
def expired_contracts(self, underlying: str, expiry: str,
                      instrument_type: str = "OPTIDX") -> list[dict]
# market data            category="data" / "quote"
def historical(self, interval: str, scrip_codes: list[str],
               start_ms: int, end_ms: int) -> dict[str, list[Candle]]
    """<=5 codes per call; batches internally. Returns {code: [candle,...]}, missing keys dropped."""
def historical_expired(self, interval: str, trading_symbols: list[str],
                       start_ms: int, end_ms: int, segment: str = "NFO") -> dict[str, list[Candle]]
    """<=50 symbols per call; batches internally."""
def ltp(self, scrip_codes: list[str]) -> dict[str, float]
def quotes(self, scrip_codes: list[str]) -> dict[str, dict]
def option_chain(self, exchange: str, segment: str, underlying_scrip: str,
                 expiry: str, strike_count: int = 10) -> dict
    """Returns the raw data block: {"underlying_ltp":…, "expiry":…, "strikes": {...}}."""
# orders                 category="order"
def place_order(self, **kw) -> dict         # POST /order
def modify_order(self, **kw) -> dict
def cancel_order(self, order_id: str) -> dict
def order_book(self) -> list[dict]
def order_details(self, order_id: str) -> dict
def trade_book(self) -> list[dict]
def positions(self, segment: str|None = None, product: str|None = None) -> list[dict]
def margin(self, **kw) -> dict
```

`historical` and `historical_expired` normalise the response: this endpoint wraps in
`{"success": bool, "data": {...}}` (NOT `status`), each scrip has `{"candles": [...]}`,
and `candles` can be `null`. Return `{}` entries as `[]`, never `None`.
`option_chain` errors do NOT use the standard envelope — check the HTTP code, then read
whichever of `debug_info` / `message` / `error` is present.

Also in this file:
```python
def generate_token(client_id: str, totp_secret: str, pin: str) -> str
    """TOTP token generation per docs §TOTP Authentication. 1 req/min — rate-limit hard."""
```

## C. `indbot/instruments.py`

Caches the instrument master CSVs on disk (`data/instruments/<source>.csv`, refetch when
older than 12h or when the trading date has rolled) and answers lookups.

```python
class Instruments:
    def __init__(self, client: INDstocksClient, cache_dir: Path) -> None
    def refresh(self, source: str, force: bool = False) -> None
    def index_scrip(self, label: str) -> str | None
        """Resolve 'NIFTY 50' -> '40000001' from source=index. The index file is THREE columns
        (EXCH, SEGMENT, SECURITY_ID) where column 2 holds the index NAME despite its header —
        read it positionally."""
    def lot_size(self, underlying: str, expiry: str | None = None) -> int
        """From source=fno LOT_UNITS. Falls back to FALLBACK_LOTS when the row is absent."""
    def tick_size(self, security_id: str) -> float
    def freeze_qty(self, security_id: str) -> int
    def contract(self, security_id: str) -> dict | None
    def resolve_underlyings(self, cfg) -> dict[str, dict]
        """Returns UNDERLYINGS with 'scrip' corrected from the live index file; logs a warning
        per mismatch."""

FALLBACK_LOTS = {"NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65,
                 "MIDCPNIFTY": 120, "SENSEX": 20}
```

Lot sizes change; always prefer the CSV, use the fallback only when the lookup fails, and
mark which one was used so the UI can show it.

## D. `indbot/indicators.py`

Pure functions, no state, no dependencies. Each takes `list[float]` or `list[Candle]` and
returns a list of the same length, left-padded with `None` where undefined.

```python
def sma(values: list[float], n: int) -> list[float|None]
def ema(values: list[float], n: int) -> list[float|None]
def rsi(values: list[float], n: int = 14) -> list[float|None]
def atr(candles: list[Candle], n: int = 14) -> list[float|None]
def vwap(candles: list[Candle]) -> list[float|None]       # session-anchored, resets each day
def supertrend(candles: list[Candle], n: int = 10, mult: float = 2.0
               ) -> list[tuple[float, int]|None]           # (level, direction) dir=+1 up, -1 down
def adx(candles: list[Candle], n: int = 14) -> list[float|None]
def bollinger(values: list[float], n: int = 20, k: float = 2.0
              ) -> list[tuple[float,float,float]|None]      # (lower, mid, upper)
def opening_range(candles: list[Candle], minutes: int = 15) -> tuple[float, float] | None
    """(high, low) of the candles inside the first `minutes` of the session, else None."""
def body_ratio(c: Candle) -> float        # |c-o| / (h-l), 0.0 when h==l
def regime(candles: list[Candle]) -> str  # "trend_up"|"trend_down"|"range" from ADX + EMA stack
```

Correctness matters more than speed here. Use Wilder's smoothing for RSI/ATR/ADX (not a
plain SMA). `vwap` uses the typical price `(h+l+c)/3` weighted by volume. Guard every
division. These functions are unit-tested against hand-computed fixtures — get them right.

## E. `indbot/strategy.py`

```python
SETUPS: list[str] = ["orb", "vwap_reclaim", "ema_cross", "supertrend", "rsi_reversal", "pullback"]

class Strategy:
    def __init__(self, cfg: Config) -> None
    def evaluate(self, underlying: str, candles: list[Candle], *,
                 now_ts: int | None = None) -> list[Signal]
        """Evaluate every enabled setup against the LAST CLOSED candle. Returns 0..n signals,
        highest score first. Never looks at a forming candle."""
    def exit_signal(self, position: Position, candles: list[Candle]) -> Signal | None
        """An opposing/invalidation signal on the underlying that should close `position`."""
```

Triggers — each fires on the **last closed candle** only:

| setup | CE (bullish) trigger | PE (bearish) trigger |
|---|---|---|
| `orb` | first close above the opening-range high, after `or_minutes` | first close below the OR low |
| `vwap_reclaim` | prior close < VWAP, this close > VWAP, `body_ratio >= 0.5` | mirror |
| `ema_cross` | EMA9 crosses above EMA21 **and** close > VWAP | mirror |
| `supertrend` | supertrend direction flips −1 → +1 | +1 → −1 |
| `rsi_reversal` | RSI(14) crosses up through 30 | crosses down through 70 |
| `pullback` | EMA9>EMA21>EMA50, prior low touched EMA21, this close > prior high | mirror |

Confluence score — start at the trigger's base weight, then add/subtract confirmations.
Document the exact table in the module docstring. Required components:
EMA stack alignment, VWAP side, RSI not extreme against the trade, volume vs 20-bar average,
ADX > 20, candle body ratio. Clamp to `[0, 1]`. Put every contributing term into
`Signal.reasons` as a short phrase like `"EMA9>EMA21>EMA50 (+0.10)"` — the UI prints these,
so they must read as plain English.

Each signal carries `stop_pct`/`target_pct` from config, **widened for a low-ADX regime** and
tightened near `flat_by`. One signal per `(underlying, setup)` per candle; the engine
deduplicates across candles.

## F. `indbot/chain.py`

```python
class ChainPicker:
    def __init__(self, client: INDstocksClient, instruments: Instruments, cfg: Config) -> None
    def nearest_expiry(self, underlying: str) -> str
    def fetch(self, underlying: str, expiry: str | None = None,
              strike_count: int = 10) -> tuple[float, list[OptionLeg]]
        """(underlying_ltp, legs). `strikes` is an object keyed by strike, NOT an array —
        sort the keys numerically. Builds OptionLeg for both ce and pe at every strike."""
    def pick(self, underlying: str, side: str, *, spot: float | None = None
             ) -> tuple[OptionLeg | None, list[str]]
        """Choose the single best tradable leg, or None plus the reasons nothing qualified.
        The reasons list is shown in the UI, so each entry names the strike it rejected and why."""
```

`pick` filters, in this order, recording a rejection reason for each leg dropped:
1. correct `side`
2. `max_premium = cfg.effective_max_premium(lot_size)` → drop `ltp * lot_size` over the
   per-trade budget (**this is the binding constraint — see RISK ARITHMETIC**)
3. `min_delta <= |delta| <= max_delta`
4. `spread_pct <= max_spread_pct`
5. `oi >= min_oi` and `volume >= min_volume`
6. `ltp > 0` and `bid > 0` and `ask > 0`

Rank survivors by `|delta - 0.5|` ascending, tie-break on tighter `spread_pct`, then
higher `oi`. If nothing survives filter 2, say so explicitly — that reason is the one the
user most needs to see.

## G. `indbot/journal.py`

SQLite at `data/indbot.db`, `sqlite3` with `check_same_thread=False` and a module-level
`threading.RLock` around writes. WAL mode on.

```python
class Journal:
    def __init__(self, path: Path) -> None            # creates schema if absent
    def record_signal(self, sig: Signal, *, taken: bool, reason: str = "") -> int
    def open_trade(self, pos: Position, mode: str) -> None
    def close_trade(self, pos: Position) -> None
    def trades(self, *, day: date|None = None, mode: str|None = None,
               limit: int = 200) -> list[dict]
    def signals(self, *, day: date|None = None, limit: int = 200) -> list[dict]
    def day_stats(self, day: date|None = None, mode: str = "paper") -> dict
        """{"realized_pnl", "trade_count", "wins", "losses", "open_count"} — drives the risk gate."""
    def equity_curve(self, mode: str = "paper", days: int = 30) -> list[dict]
    def bucket_stats(self, mode: str|None = None) -> list[dict]
        """Per learning bucket: n, wins, losses, gross_win, gross_loss, expectancy."""
    def set_meta(self, key: str, value: str) -> None
    def get_meta(self, key: str, default: str|None = None) -> str|None
```

Tables: `signals`, `trades`, `meta`. `trades` columns at minimum: `id, mode, underlying,
setup, bucket, side, strike, expiry, trading_symbol, security_id, lot_size, qty,
entry_price, entry_ts, entry_order_id, exit_price, exit_ts, exit_order_id, exit_reason,
stop_price, target_price, pnl, pnl_pct, status`. Index `(mode, entry_ts)` and `(bucket)`.
`day_stats` must count only trades whose `entry_ts` falls on that IST calendar day.

## H. `indbot/learner.py`

Outcome-gated setups. No ML — transparent statistics the UI can explain.

```python
@dataclass(frozen=True)
class BucketVerdict:
    bucket: str; n: int; wins: int; win_rate: float
    expectancy: float          # mean rupees per trade
    lower_bound: float         # Wilson lower bound on win rate at cfg.gate_confidence
    state: str                 # "promoted" | "neutral" | "demoted" | "blocked"
    multiplier: float          # score multiplier applied to a signal in this bucket
    note: str                  # one plain sentence the UI shows verbatim

class Learner:
    def __init__(self, journal: Journal, cfg: Config) -> None
    def refresh(self) -> None                       # recompute from the journal; call after each close
    def verdict(self, bucket: str) -> BucketVerdict
    def adjust(self, sig: Signal) -> tuple[Signal, BucketVerdict]
        """Return the signal with score *= verdict.multiplier, and the verdict."""
    def leaderboard(self) -> list[BucketVerdict]    # best expectancy first
    def is_blocked(self, bucket: str) -> bool
```

Rules — implement exactly:
- `n < cfg.min_samples_to_gate` → `neutral`, multiplier `1.0`, note says it is still learning
  and how many more trades it needs.
- Wilson lower bound at `cfg.gate_confidence` on the win rate, with the break-even win rate
  `be = stop_pct / (stop_pct + target_pct)` as the reference.
- `lower_bound > be + 0.10` → `promoted`, multiplier `1.15`.
- `lower_bound < be` **and** `expectancy < 0` → `demoted`, multiplier `0.75`.
- `n >= 2 * min_samples_to_gate` and `expectancy < 0` and `lower_bound < be - 0.10`
  → `blocked`, multiplier `0.0`.
- Otherwise `neutral`.

`note` must always explain the state in one sentence with the numbers in it, e.g.
`"Blocked: 6 wins in 24 trades (25%), break-even needs 37.5%, average −₹212 a trade."`

## I. `indbot/risk.py`

```python
class RiskGate:
    def __init__(self, journal: Journal, cfg: Config) -> None
    def check_entry(self, sig: Signal, leg: OptionLeg, *, open_positions: list[Position],
                    mode: str, funds: dict | None = None) -> RiskVerdict
    def check_exit_required(self, pos: Position, ltp: float, *,
                            now: datetime | None = None) -> str | None
        """Returns an exit reason ("stop"|"target"|"trail"|"flat_by"|"time_stop") or None."""
    def day_halted(self, mode: str) -> tuple[bool, str]
```

`check_entry` evaluates every rule and reports **all** failures, not just the first:
1. mode is `live` and not `armed` → block
2. market open, and `no_entry_before <= now <= no_entry_after`
3. `len(open_positions) < max_open_positions`
4. `day_stats.trade_count < max_trades_per_day`
5. `realized_pnl > -max_daily_loss` (the day-halt)
6. **per-trade risk**: `leg.lot_cost * sig.stop_pct <= max_trade_loss`
7. **stop sanity**: the stop implied by `max_trade_loss` is not tighter than `min_stop_pct`
8. `sig.score >= min_score` and the bucket is not blocked
9. in `live` mode, `funds.detailed_avl_balance.option_buy >= leg.lot_cost * 1.05`

Rule 6/7 is the one that will fire most — see RISK ARITHMETIC. Its reason must be
arithmetic and specific:
`"1 lot of NIFTY 25000 CE at ₹168 is ₹12,600; a 30% stop risks ₹3,780, over the ₹1,000
per-trade limit. Either raise the per-trade limit to ₹3,800 or let the bot buy a cheaper
strike (max ₹44 premium at this lot size)."`

## J. `indbot/broker.py`

```python
class BaseBroker(ABC):
    mode: str
    def buy(self, leg: OptionLeg, qty: int, *, remark: str) -> dict      # {"order_id", "status", "price"}
    def sell(self, leg: OptionLeg, qty: int, *, remark: str) -> dict
    def ltp(self, leg: OptionLeg) -> float
    def open_positions(self) -> list[dict]

class PaperBroker(BaseBroker):   # mode = "paper"
    """Simulated fills against the live book: a BUY fills at `ask` (plus slippage_ticks),
    a SELL at `bid` (minus slippage_ticks). Charges a configurable per-order cost.
    Never touches POST /order."""

class LiveBroker(BaseBroker):    # mode = "live"
    """Real orders. Refuses outright unless cfg.mode == 'live' and cfg.armed is True —
    assert it on every single call, not just at construction."""
```

Live order body, exactly:
```python
{"txn_type": "BUY"|"SELL", "exchange": leg.exchange, "segment": "DERIVATIVE",
 "product": "INTRADAY", "order_type": "LIMIT", "validity": "DAY",
 "security_id": leg.security_id, "qty": qty,          # UNITS, a whole multiple of lot_size
 "limit_price": <price rounded to tick_size>, "is_amo": False, "remarks": remark}
```
- `qty` is **units, not lots**. One lot = `leg.lot_size` units.
- Prefer `LIMIT` with a marketable price (BUY at `ask * (1 + slip)`, SELL at `bid * (1 - slip)`)
  rounded to the tick size, because a `MARKET` order is converted to a protected `LIMIT`
  anyway and the band is not under your control.
- `remarks` ≤ 100 chars, must not be `TV-TERMINAL`, and is **not** an idempotency key —
  dedupe on your side before sending.
- After placing, poll `order_details` until terminal, with a timeout; on a non-fill,
  cancel and report. Return the average fill price.

`LiveBroker.sell` must never leave a position stranded: if the exit order is rejected,
raise `OrderError` with the full payload so the engine can alert loudly.

## K. `indbot/engine.py`

The state machine that ties it together. One thread, a tick loop, thread-safe snapshot
for the server.

```python
class Engine:
    def __init__(self, cfg: Config, journal: Journal, *, client: INDstocksClient|None = None) -> None
    def start(self) -> None                 # spawn the loop thread
    def stop(self) -> None
    def tick(self) -> dict                  # one pass; also callable directly from tests
    def snapshot(self) -> dict              # everything the UI renders — see below
    def set_mode(self, mode: str) -> None   # closes open positions first when leaving live
    def arm(self, confirm: str) -> None     # requires confirm == "ARM LIVE TRADING"
    def disarm(self) -> None
    def panic(self) -> dict                 # flatten every open position NOW, halt the day
    def subscribe(self, fn: Callable[[dict], None]) -> Callable[[], None]   # returns unsubscribe
```

Tick order, every `poll_seconds` (default 20):
1. refresh clock; if outside the session, idle (but keep serving the snapshot)
2. for each open position: refresh LTP → update `high_water` → `risk.check_exit_required`
   → if an exit reason, SELL 1 lot and close the trade
3. if the day is halted or at the position cap, stop here
4. for each underlying: fetch candles (cached, only refetch when a new candle should have
   closed) → `strategy.evaluate` → `learner.adjust`
5. take the single best signal across all underlyings → `chain.pick` →
   `risk.check_entry` → if `ok`, BUY 1 lot and open the trade; otherwise record the
   signal with its block reason so the UI can show *why nothing was traded*
6. publish the snapshot to subscribers

`snapshot()` returns:
```python
{"mode", "armed", "market_open", "now", "session", "halted", "halt_reason",
 "config": cfg.as_dict(), "funds": {...}|None, "day": {...day_stats},
 "underlyings": [{"name","label","spot","change_pct","regime","or_high","or_low",
                  "last_candle_ts","signals":[...]}],
 "positions": [...], "recent_trades": [...], "recent_signals": [...],
 "leaderboard": [...], "blocked_buckets": [...],
 "log": [{"ts","level","msg"}], "errors": [...]}
```

Every rejected entry is logged with its reason. **A quiet bot that explains itself beats a
bot that trades.** The log is the product.

## L. `indbot/backtest.py`

Replays the strategy over **real historical option premiums**.

```python
@dataclass
class BacktestResult:
    # summary + trades + equity curve + per-bucket stats, all JSON-serialisable
    ...

class Backtester:
    def __init__(self, client: INDstocksClient, instruments: Instruments, cfg: Config) -> None
    def run(self, underlying: str, start: date, end: date, *,
            progress: Callable[[dict], None] | None = None) -> BacktestResult
```

Method, per trading day in the window:
1. `expired_expiries(underlying, ...)` → choose the expiry that was nearest to that day
   (the one the live bot would have traded).
2. `expired_contracts(underlying, expiry, "OPTIDX")` → the full chain, with `lot_size`
   (one response, 460–480 rows for NIFTY).
3. Underlying candles for the day: `historical(interval, [f"NSE_{scrip}"], ...)`.
   SENSEX uses its BSE scrip.
4. Walk the day candle by candle. At each closed candle run `Strategy.evaluate` on the
   candles **up to and including** that one — never beyond, or the backtest is a lie.
5. On a signal, pick the strike the live `ChainPicker` would have picked, using the
   premium from the expired candles at that timestamp as `ltp` (synthesise `bid`/`ask`
   from a configurable spread assumption, and say in the result that you did).
6. Fetch that contract's own candles via `historical_expired` and simulate the position
   bar by bar: stop, target, trail, `flat_by`. Fill a stop at the bar's close, not its
   low — assuming the best price inside the bar is the classic backtest lie.
7. Apply the same `RiskGate` rules and per-day limits as live.

Batch `historical_expired` 50 symbols at a time and cache every response under
`data/backtest_cache/` keyed by symbol+window, because the data rate limit is 5/s and a
year of NIFTY is thousands of calls. Report **cache hits, API calls, and days skipped for
missing data** in the result — silent gaps read as a clean backtest when they are not.

The result must state its own assumptions in a `caveats: list[str]`: the spread model, the
stop-fill rule, no slippage beyond the spread, no brokerage unless configured, survivorship
of the chosen expiry.

## M. `indbot/server.py`

FastAPI app, serves the UI and a websocket.

```python
def create_app(engine: Engine, journal: Journal, cfg: Config) -> FastAPI
```

Routes:
```
GET  /                       -> web/index.html
GET  /static/*               -> web/
GET  /api/state              -> engine.snapshot()
GET  /api/config             -> cfg.as_dict()
POST /api/config             -> cfg.update(patch); returns changed keys
POST /api/mode               -> {"mode": "backtest"|"paper"|"live"}
POST /api/arm                -> {"confirm": "ARM LIVE TRADING"}
POST /api/disarm
POST /api/panic              -> flatten everything, halt
POST /api/start  POST /api/stop
GET  /api/trades?day=&mode=  GET /api/signals
GET  /api/leaderboard
POST /api/backtest           -> {"underlying","start","end"} -> {"job_id"}
GET  /api/backtest/{job_id}  -> progress or result
GET  /api/health
WS   /ws                     -> pushes the snapshot on every engine publish
```

Bind to `127.0.0.1` only. No auth, but **refuse to start if the bind host is not
loopback** — this process can move real money. Backtests run in a
`ThreadPoolExecutor(max_workers=1)` with their progress readable by job id.

## N. `web/index.html`, `web/app.js`, `web/style.css`

A single-page dashboard, vanilla JS, no build step, no CDN. Dark by default, light via
`prefers-color-scheme`. It must work on a phone screen.

Layout, top to bottom:
1. **Header** — mode selector (BACKTEST / PAPER / LIVE), ARM/DISARM, big red PANIC, a
   connection dot, the clock and the session state.
2. **The live-mode gate** — switching to LIVE opens a modal that states the account, the
   limits and today's P&L, and requires typing `ARM LIVE TRADING`. Nothing arms on a click.
3. **Day strip** — realised P&L, trades used of the cap, open positions, day-halt banner.
4. **Underlyings** — one card per index: spot, change, regime, opening range, the live
   signals with their scores and `reasons`.
5. **Positions** — the open position: strike, entry, LTP, P&L, stop, target, trailing
   state, and a manual CLOSE.
6. **Why no trade** — the rejected-signal feed with each block reason in full. This panel
   is as prominent as the positions panel; it is what makes the bot trustworthy.
7. **Learning** — the bucket leaderboard: setup, n, win rate, expectancy, state, note.
   Blocked buckets styled clearly.
8. **Backtest** — pick an underlying and a date range, run, watch progress, read the
   equity curve, the trade list and the caveats.
9. **Log** — the engine log, newest first, filterable by level.

Rules: never show a number without its unit; P&L always carries a sign and a colour;
reasons render as sentences, not JSON. Reconnect the websocket with backoff and show the
disconnected state honestly rather than freezing stale numbers. Rupee amounts use
`toLocaleString('en-IN')`.

---

## RISK ARITHMETIC — read this before writing `chain.py` or `risk.py`

The user's limits are **₹1,000 max loss per trade** and **1 lot minimum**. With a 30% stop
on the premium, that caps the position at `1000 / 0.30 = ₹3,333` of premium:

| underlying | lot | max premium per unit | a typical ATM premium |
|---|---|---|---|
| NIFTY | 75 | ₹44 | ₹150–200 |
| BANKNIFTY | 35 | ₹95 | ₹250–350 |
| FINNIFTY | 65 | ₹51 | ₹120–180 |
| MIDCPNIFTY | 120 | ₹28 | ₹80–120 |
| SENSEX | 20 | ₹167 | ₹200–300 |

**An ATM option will therefore almost never pass the gate.** That is not a bug and you must
not paper over it: do not silently widen the stop, do not silently allow a bigger loss, and
do not quietly drift to far-OTM lottery strikes. Make the arithmetic visible — the gate
blocks, the reason states the numbers, and the UI offers the user the two honest choices
(raise the per-trade limit, or accept a tighter stop with its whipsaw cost). Default
`min_stop_pct = 0.12` exists to stop the bot from "solving" this with a 5% stop that the
spread alone would trigger.
