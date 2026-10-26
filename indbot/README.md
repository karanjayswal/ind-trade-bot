# indbot

An automated Indian index-options trading bot. It scans the five major index
option chains, generates call/put signals from the underlying's price action,
buys one lot when a signal survives every gate, sells that lot on a stop,
target, trailing stop or opposing signal — and keeps score of its own setups so
the ones that keep losing stop firing.

It runs in three modes, in this order: **backtest → paper → live**. Live is
behind a switch you have to arm by typing a phrase, every session.

> **This software places real orders with real money.** Most people who day
> trade lose money, and most people who automate day trading lose it faster.
> Nothing here is financial advice. Read [Before you trade real
> money](#before-you-trade-real-money) before you touch the LIVE toggle.

---

## The constraint you should understand first

This is the single most important thing about running this bot, and it is
arithmetic, not opinion.

The default limits are **₹1,000 maximum loss per trade** and a **30% stop** on
the option premium. Together they cap a position at `1000 / 0.30 = ₹3,333` of
premium. Index options trade in lots, and one lot is a lot of units:

| Index | Lot size | Max premium per unit that fits ₹1,000 | Typical ATM premium |
|---|---:|---:|---:|
| NIFTY | 75 | ₹44 | ₹150–200 |
| BANKNIFTY | 35 | ₹95 | ₹250–350 |
| FINNIFTY | 65 | ₹51 | ₹120–180 |
| MIDCPNIFTY | 120 | ₹28 | ₹80–120 |
| SENSEX | 20 | ₹167 | ₹200–300 |

**An at-the-money option will therefore almost never pass the risk gate.** That
is not a bug. The bot blocks the trade and tells you the arithmetic:

```
1 lot of NIFTY 25000 CE (75 units) at ₹168 is ₹12,600; a 30% stop risks ₹3,780,
over the ₹1,000 per-trade limit. Either raise the per-trade limit to ₹3,800 or
let the bot buy a cheaper strike (max ₹44 premium at this lot size).
```

You have three honest choices, and the UI shows you all three:

1. **Raise the per-trade limit** to roughly ₹3,400–3,800 so one ATM lot fits.
   Then set `max_daily_loss` above it, or a single trade can breach the day cap
   before the halt fires.
2. **Accept a tighter stop.** The floor is `min_stop_pct` (12% by default)
   because below that the bid-ask spread alone triggers the stop and you pay the
   loss for nothing.
3. **Let it buy cheap far-OTM strikes.** These are low-delta lottery tickets
   that mostly expire worthless. The bot will do it if you tell it to; it will
   not drift there on its own.

What the bot will *not* do is quietly widen your stop or quietly buy something
cheaper than you asked for. Four of the five indices now have **monthly-only**
expiries (NSE kept weeklies on NIFTY alone), and monthly premiums are pricier
than weeklies, which tightens this further.

---

## Install

```bash
git clone https://github.com/karanjayswal/ind-trade-bot.git
cd ind-trade-bot/indbot
python3 -m pip install -r requirements.txt
```

Python 3.10 or newer. Five dependencies: `fastapi`, `uvicorn`, `httpx`,
`websockets`, `pyotp`.

### Get an access token

From the [INDstocks dashboard](https://indstocks.com):

```bash
export INDSTOCKS_ACCESS_TOKEN="your_token_here"
```

Or generate one with TOTP, if you have that set up:

```bash
python3 run.py token          # prompts for client id, TOTP secret and PIN
```

Token generation is limited to **one request a minute** and repeated failures
lock the account out, so the client throttles it hard and never retries a
failure automatically.

The token lives in the environment, never in a file and never in the UI.
`config.json` is in `.gitignore` for the same reason. Tokens expire — when calls
start failing with `TokenException`, generate a new one.

### Check it works before you trust it

```bash
python3 run.py doctor
```

Verifies the token, the profile, your funds, the instrument master and one live
option chain, and prints a remedy for whatever fails. Run this first. An empty
dashboard usually means a dead token.

---

## Use it in the order it was built for

### 1. Backtest

```bash
python3 run.py backtest --underlying NIFTY --start 2026-09-01 --end 2026-10-20
```

Replays the strategy over **real historical option premiums** pulled from the
expired-contract endpoints — not a Black-Scholes approximation. It applies the
same risk gate and the same daily limits as live.

Read the `caveats` in the result before you believe it. Among them: stops fill
at the bar's **close**, not its low (assuming the best price inside a bar is the
classic backtest lie); bid/ask is synthesised from a spread assumption because
the expired-candle endpoint returns OHLCV only; and with no greeks in that data,
strikes are selected by moneyness rather than delta. The result also reports
days skipped for missing data, because a silent gap reads as a clean backtest
when it is not.

### 2. Paper

```bash
python3 run.py serve --mode paper
```

Opens the dashboard at <http://127.0.0.1:8765>. Signals, sizing and fills are
simulated against the live book — a buy fills at the ask, a sell at the bid,
plus slippage and a per-order cost. `PaperBroker` structurally cannot reach the
order endpoints; there is a test that proves it.

Watch it for days. The interesting panel is **Why no trade** — most signals are
rejected, and the reasons are the product.

### 3. Live

Switch the mode to LIVE in the UI. You will have to type `ARM LIVE TRADING`
exactly; a click alone never arms it. Arming is per session, and the bot boots
disarmed even if the config file says otherwise.

```bash
python3 run.py serve --mode live   # still starts disarmed
```

---

## How it decides

**Signals** are generated on the *underlying index*, on closed 5-minute candles
only. Six setups:

| Setup | CE (bullish) trigger | PE (bearish) trigger |
|---|---|---|
| `orb` | first close above the opening-range high | first close below the OR low |
| `vwap_reclaim` | crosses back above VWAP with a decisive body | mirror |
| `ema_cross` | EMA9 crosses above EMA21, price above VWAP | mirror |
| `supertrend` | supertrend flips up | flips down |
| `rsi_reversal` | RSI(14) crosses up through 30 | crosses down through 70 |
| `pullback` | pulls back to EMA21 in an uptrend, then takes the prior high | mirror |

Each trigger is then scored for confluence — EMA stack, VWAP side, RSI, volume
versus the 20-bar average, ADX, candle body — and every term that moved the
score appears in the signal's `reasons`, in English, in the UI. A signal below
`min_score` (0.55) never reaches the chain picker.

**Strike selection** takes the chain for the nearest expiry and filters in order:
side, the per-trade premium budget, delta band (0.35–0.60), spread (≤2%), open
interest and volume, and a two-sided book. Survivors rank by closeness to 0.50
delta. Every rejected strike records *why*.

**Risk** checks nine rules and reports **every** failure, not just the first:
arm state, the session window, the position cap, the trade count, the day loss
limit, the per-trade risk arithmetic, the stop-sanity floor, the score and
bucket gate, and available funds. The day halt persists in the database, so
restarting the bot cannot resume trading on a day that already hit its limit.

**Exits**, in priority order: `flat_by` time, stop, target, trailing stop, time
stop. Flat-by and stop are checked *before* target, because on a gap both can be
true and the safe branch is the exit.

---

## How it learns from its mistakes

No machine learning. Transparent statistics you can audit and argue with.

Every trade is journalled against a **bucket** — `setup : side : time-of-day :
regime` — so a 09:20 breakout in a trend is scored separately from a 14:50
breakout in a range. For each bucket the learner computes the win rate, the mean
rupees per trade, and the **Wilson score lower bound** on the win rate, then
compares that against the break-even win rate the trades' own stop/target implies
(`stop / (stop + target)` — 37.5% for a 30/50 pair).

| State | Condition | Effect on a signal's score |
|---|---|---|
| `neutral` | fewer than 10 trades | ×1.00 — still learning |
| `promoted` | lower bound clears break-even by 10 points | ×1.15 |
| `demoted` | lower bound under break-even and expectancy negative | ×0.75 |
| `blocked` | 20+ trades, negative expectancy, well under break-even | ×0.00 |

A blocked bucket stops firing and says exactly why:

> Blocked: 6 wins in 24 trades (25%), break-even needs 37.5%, average −₹212 a trade.

Paper and live statistics are kept separate by default (`learn_across_modes:
false`), so a paper losing streak does not silently block a live setup.

---

## The dashboard

Four tabs. **LIVE** holds only what matters while a position is open: a ticker
tape of the five indices, today's P&L as the hero figure, the open position, the
signals, and **why no trade**. **LEARNING** has the bucket leaderboard and trade
history, **BACKTEST** the runner and its equity curve, **SETTINGS** the risk
limits with the constraint table beside them.

Two deliberate design decisions worth knowing about:

- **Mode is unmistakable.** A rail across the top and a border on the page:
  grey for backtest, blue for paper, red for live. Paper and live must not look
  alike at a glance.
- **Colour never carries P&L direction alone.** Profit-green and loss-red
  measure an OKLab ΔE of **4.0** under deuteranopia — for roughly one man in
  twelve they are the same colour, while normal vision reads them at 33.9, which
  is why most trading UIs ship this broken. So every money figure carries an
  explicit `+`/`−`, every price move carries a ▲/▼ with a text equivalent, and
  charts carry sign through position against a zero rule.

On websocket loss the numbers grey out and say so. A frozen P&L that still looks
live is the most dangerous thing a trading UI can do.

The server binds to **loopback only** and refuses to start otherwise. There is
no authentication, so a non-loopback bind would be an unauthenticated trading
API on your network.

---

## Tests

```bash
python3 -m pytest tests/ -q
```

277 tests and 336 subtests, fully offline, in about a second. They run against
a fake client built from the documented wire shapes, and they assert
hand-computed values rather than whatever the code happens to produce. The ones
worth knowing about:

- `test_evaluate_never_reads_a_forming_candle` — look-ahead is the bug that
  makes a backtest a lie, and the same function runs live and in the backtest
- `test_the_whole_lifecycle_sent_no_order_to_the_api` — paper mode is proven,
  not promised
- `test_halt_survives_reopening_the_database` — the restart hole that turns a
  ₹2,000 limit into a ₹6,000 loss
- `test_config_load_boots_disarmed_even_if_the_file_says_armed`
- `test_disarming_mid_flight_stops_the_next_order`
- `test_effective_max_premium_is_per_unit_and_guards_its_division` — a real
  unit-confusion bug, caught during the build, that would have sized a position
  75× too large

---

## Configuration

Copy `config.example.json` to `config.json` and edit, or change everything in
the SETTINGS tab. `config.json` is gitignored.

The settings most worth your attention:

| Key | Default | What it does |
|---|---|---|
| `max_daily_loss` | 2000.0 | Realised loss that halts the day. Persists across restarts. |
| `max_trade_loss` | 1000.0 | Per-trade cap. **This is what blocks ATM strikes** — see above. |
| `max_trades_per_day` | 3 | Entries per day. |
| `max_open_positions` | 1 | Concurrent positions. |
| `stop_pct` / `target_pct` | 0.30 / 0.50 | Stop and target, as a fraction of premium. |
| `min_stop_pct` | 0.12 | Refuses a stop so tight the spread would trigger it. |
| `min_score` | 0.55 | Confluence score a signal needs to be considered. |
| `flat_by` | "15:10" | Everything is closed by this time, win or lose. |
| `underlyings` | all five | Drop FINNIFTY/MIDCPNIFTY if the log fills with liquidity rejections. |

`armed` cannot be set through a config patch. Arming is a separate, deliberate
act, enforced in the config layer, the engine and the server independently.

---

## Layout

```
indbot/
  indbot/
    clock.py         IST time, session windows, holidays, candle bucketing
    types.py         Signal, OptionLeg, Position, RiskVerdict
    errors.py        the exception hierarchy
    config.py        settings, validation, the risk constraint table
    api.py           INDstocks REST client: rate limits, retries, wire-shape quirks
    instruments.py   instrument master cache, lot sizes, self-healing scrip ids
    indicators.py    EMA, Wilder RSI/ATR/ADX, session VWAP, supertrend, regime
    strategy.py      the six setups and the confluence score
    chain.py         option chain to one tradable strike, with reasons
    journal.py       SQLite: signals, trades, day stats, buckets
    learner.py       Wilson-bound outcome gating
    risk.py          the nine rules and the persistent day halt
    broker.py        PaperBroker and LiveBroker behind one interface
    engine.py        the tick loop and the snapshot the UI renders
    backtest.py      replay over real expired-contract premiums
    server.py        loopback FastAPI + websocket
  web/               the dashboard: index.html, app.js, style.css
  tests/             offline suite
  CONTRACTS.md       the module contract this was built against
  run.py             serve | backtest | doctor
```

`CONTRACTS.md` is the specification the modules were written against. It is kept
in the repo because it documents the integration contract — and because it
contains the bug described above, which is a useful thing to have on the record.

---

## Before you trade real money

**Most people who day trade lose money.** Automation does not change the odds;
it changes the speed. Nothing in this repository is financial advice, and the
author is not a licensed adviser. Decide for yourself, and talk to someone
qualified if you need advice.

**A stop is a plan, not a guarantee.** In a fast market or across a gap, an exit
fills well past its price. Every "max loss" figure here is the loss *at the stop
price*, which is not the loss you will necessarily take.

**Options decay.** A long option loses value every day you hold it, and on
expiry day it can go to zero in minutes. The bot is long premium only — it buys
calls and puts and never sells them — so a loss is capped at the premium paid
and no short-option margin is ever required. That is the safe direction, and it
is also the one that bleeds theta.

**The backtest is a simulation, not a prediction.** It is built from what those
contracts actually did. The future can do things that window never did.

**Market data can lag, and the API can rate-limit you.** The data bucket is five
requests a second across every underlying. Check anything that matters against
INDmoney before you act on it.

**Taxes.** Intraday options gains in India are speculative/business income, not
capital gains. Nothing here is tax advice.

**Start in backtest. Then paper. Then live, small.** In that order. The bot was
built to make that sequence easy for a reason.

---

## Credits

- Option and market data, and order execution, via the
  [INDstocks API](https://api-docs.indstocks.com/) (INDmoney).
- `day-trader-skill-main/` in this repository is
  [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) by
  Jake Schincariol, MIT licensed, included as the reference the strategy
  structure was adapted from. Its own LICENSE travels with it. It targets US
  equities on free Yahoo data; indbot is a different system for Indian index
  options on a broker API, and shares none of its code.
- `indstocks-api-docs.md` is a local copy of INDmoney's published API
  documentation, kept for offline reference. It is their work, not mine.
