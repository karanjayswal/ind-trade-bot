#!/usr/bin/env python3
"""indbot command line.

    python3 run.py serve     [--port 8777] [--mode paper]
    python3 run.py backtest  --underlying NIFTY --start 2026-09-01 --end 2026-10-20
    python3 run.py doctor    [--underlying NIFTY]
    python3 run.py token     # mint a 24h access token from your TOTP secret

`doctor` is the command to run first and after every token refresh: it proves the
token works, the account is F&O enabled, the instrument masters parse and one option
chain comes back — and it prints the premium budget your per-trade limit implies,
which is the number that decides whether this bot can trade at all.

This file is the only entry point allowed to print; everything inside `indbot/`
logs instead. Logs go to stderr and to data/indbot.log (rotated, 5 x 2 MB).
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_PATH = DATA_DIR / "indbot.log"
DB_PATH = DATA_DIR / "indbot.db"

if str(ROOT) not in sys.path:  # allow `python3 run.py` from any working directory
    sys.path.insert(0, str(ROOT))

from indbot import clock  # noqa: E402
from indbot.errors import ApiError, AuthError, ConfigError, DataError, IndbotError  # noqa: E402

log = logging.getLogger("indbot.cli")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
os.environ["REQUESTS_CA_BUNDLE"] = '/Users/AL58376/Downloads/Personal_Project/cacert.pem'
os.environ["SSL_CERT_FILE"] = '/Users/AL58376/Downloads/Personal_Project/cacert.pem'
os.environ["NODE_EXTRA_CA_CERTS"] = '/Users/AL58376/Downloads/Personal_Project/cacert.pem'
os.environ["AWS_CA_BUNDLE"] = '/Users/AL58376/Downloads/Personal_Project/cacert.pem'


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    """Wrap `text` in an ANSI colour, or return it unchanged when not a terminal."""
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _green(t: str) -> str:
    return _c(t, "32")


def _red(t: str) -> str:
    return _c(t, "31")


def _yellow(t: str) -> str:
    return _c(t, "33")


def _bold(t: str) -> str:
    return _c(t, "1")


def _inr(amount: float | None, dp: int = 0) -> str:
    """Rupees with Indian digit grouping, e.g. 1,23,456. "-" when unknown."""
    if amount is None:
        return "-"
    neg = amount < 0
    whole = f"{abs(float(amount)):.{dp}f}"
    int_part, _, frac = whole.partition(".")
    if len(int_part) > 3:
        head, tail = int_part[:-3], int_part[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        int_part = ",".join(groups + [tail])
    out = f"₹{int_part}" + (f".{frac}" if frac else "")
    return ("-" + out) if neg else out


def _say(text: str = "") -> None:
    print(text, flush=True)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class _ISTFormatter(logging.Formatter):
    """Log timestamps in IST, because every other time in this system is IST.

    Deliberately formats the record's own clock time rather than `clock.now()`: a
    replay under INDBOT_FAKE_NOW must still produce a log you can read in order.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        stamp = datetime.fromtimestamp(record.created, tz=clock.IST)
        return stamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S") + " IST"


def configure_logging(level: int = logging.INFO, *, log_path: Path = LOG_PATH) -> None:
    """Send logs to stderr and to a rotating data/indbot.log.

    Rotation is 5 x 2 MB: a full trading day at poll_seconds=20 writes a few MB, and
    the previous days are exactly what you need when reconstructing a bad fill.
    """
    root = logging.getLogger()
    root.setLevel(min(level, logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(_ISTFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(stream)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        rotating.setLevel(logging.DEBUG if level <= logging.DEBUG else logging.INFO)
        rotating.setFormatter(
            _ISTFormatter("%(asctime)s %(levelname)-7s %(name)s [%(threadName)s]: %(message)s")
        )
        root.addHandler(rotating)
    except OSError as exc:
        # Not fatal: losing the file log is annoying, refusing to trade over it is worse.
        print(_yellow(f"warning: cannot write {log_path}: {exc}"), file=sys.stderr)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# shared wiring
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> Any:
    """Load config.json and apply the CLI overrides (without rewriting the file)."""
    from indbot.config import Config

    cfg = Config.load(getattr(args, "config", None))
    patch: dict[str, Any] = {}
    if getattr(args, "mode", None):
        patch["mode"] = args.mode
    if getattr(args, "port", None):
        patch["port"] = int(args.port)
    if getattr(args, "host", None):
        patch["host"] = args.host
    if patch:
        # persist=False: a one-off `--mode paper` must not silently rewrite the
        # settings the user keeps in config.json.
        cfg.update(patch, persist=False)
        log.info("CLI overrides applied for this run only: %s", patch)
    return cfg


def _make_client(*, required: bool) -> Any:
    """Build an INDstocksClient, or return None when there is no token and one is optional."""
    from indbot.api import TOKEN_ENV, INDstocksClient

    try:
        return INDstocksClient()
    except AuthError as exc:
        if required:
            raise
        log.warning("no API client: %s", exc)
        print(
            _yellow(
                f"warning: ${TOKEN_ENV} is not set, so there is no market data. "
                "The dashboard will start but the engine cannot scan. "
                "Run `python3 run.py doctor` for the fix."
            ),
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the engine and serve the dashboard on loopback."""
    from indbot.journal import Journal
    from indbot.server import assert_loopback, run_server

    cfg = _load_config(args)
    # Fail before touching the broker or the DB: a non-loopback host is a refusal,
    # not a warning. create_app checks again, and the app refuses remote peers too.
    host = assert_loopback(cfg.host, what="bind host")

    journal = Journal(DB_PATH, default_mode=cfg.mode)
    client = _make_client(required=False)

    from indbot.engine import Engine

    engine = Engine(cfg, journal, client=client)

    _say(_bold("indbot"))
    _say(f"  mode        {cfg.mode}" + ("" if cfg.mode != "live" else _red("  (LIVE)")))
    _say(f"  armed       {cfg.armed}")
    _say(f"  underlyings {', '.join(cfg.underlyings)}")
    _say(f"  limits      {_inr(cfg.max_trade_loss)}/trade, {_inr(cfg.max_daily_loss)}/day, "
         f"{cfg.max_trades_per_day} trades, {cfg.max_lots} lot")
    _say(f"  dashboard   http://{host}:{cfg.port}")
    _say(f"  log         {LOG_PATH}")
    if cfg.mode == "live" and not cfg.armed:
        _say(_yellow("  live mode is NOT armed - no order will be placed until you arm it in the UI"))
    _say()

    started = False
    try:
        if not args.no_start:
            engine.start()
            started = True
        run_server(engine, journal, cfg, host=host, port=cfg.port,
                   log_level="debug" if args.verbose else "info")
        return EXIT_OK
    except KeyboardInterrupt:  # pragma: no cover - uvicorn usually swallows this
        _say("\nstopping")
        return EXIT_OK
    finally:
        if started:
            try:
                # close() stops the loop and detaches the engine's log sink. It does
                # NOT flatten anything - see the warning below.
                engine.close()
            except Exception:  # noqa: BLE001 - shutdown must not mask the real error
                log.exception("engine shutdown failed")
        _warn_about_open_positions(engine)
        try:
            journal.close()
        except Exception:  # noqa: BLE001
            log.exception("journal.close() failed during shutdown")


def _warn_about_open_positions(engine: Any) -> None:
    """Shout if the process is exiting with a position still live at the broker.

    Stopping the loop does not flatten anything. A user who Ctrl-Cs at 15:05 with an
    open lot needs to know the exchange still has it; silence here costs real money.
    """
    try:
        snapshot = engine.snapshot()
        positions = [p for p in (snapshot.get("positions") or []) if p.get("is_open", True)]
    except Exception:  # noqa: BLE001
        log.debug("could not read positions at shutdown", exc_info=True)
        return
    if not positions:
        return
    log.error("exiting with %d open position(s)", len(positions))
    _say(_red(_bold(f"\n  !! {len(positions)} POSITION(S) STILL OPEN AT THE BROKER !!")))
    for pos in positions:
        _say(_red(f"     {pos.get('label', pos.get('id'))}  qty {pos.get('qty')}  "
                  f"entry {_inr(pos.get('entry_price'), 2)}  P&L {_inr(pos.get('pnl'), 2)}"))
    _say(_red("     Stopping indbot does not close them. Square off in the INDmoney app"
              " or restart and use PANIC."))


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    """Replay one underlying over a date range against real historical premiums."""
    from indbot.backtest import Backtester
    from indbot.instruments import Instruments

    start, end = args.start, args.end
    if end < start:
        # Checked before anything is built: a usage error should not demand a token.
        _say(_red(f"end {end.isoformat()} is before start {start.isoformat()}"))
        return EXIT_USAGE

    cfg = _load_config(args)
    client = _make_client(required=True)
    instruments = Instruments(client)
    # A file-less clone: a backtest must never rewrite config.json or inherit `armed`.
    backtester = Backtester(client, instruments, cfg.copy_with())

    _say(_bold(f"backtest {args.underlying} {start.isoformat()} .. {end.isoformat()}"))
    _say(f"  interval {cfg.interval}, stop {cfg.stop_pct:.0%}, target {cfg.target_pct:.0%}, "
         f"max loss {_inr(cfg.max_trade_loss)}/trade")
    _say()

    last_line = {"n": 0}

    def progress(update: dict) -> None:
        """One line per trading day, overwritten in place on a terminal."""
        day = update.get("day", "?")
        done = update.get("done", 0)
        total = update.get("total", 0) or 1
        trades = update.get("trades", 0)
        pnl = update.get("pnl", 0.0)
        line = (f"  [{done:>3}/{total:<3}] {day}  trades {trades:<3} "
                f"P&L {_inr(pnl, 2):>12}")
        last_line["n"] += 1
        end_ch = "\r" if (_COLOR and done < total) else "\n"
        print(line.ljust(72), end=end_ch, flush=True)

    try:
        result = backtester.run(args.underlying, start, end, progress=progress)
    except IndbotError as exc:
        _say(_red(f"\nbacktest failed: {exc}"))
        return EXIT_FAIL

    _print_backtest_result(result)

    if args.json:
        target = Path(args.json).expanduser()
        import json

        target.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
        _say(f"\n  full result written to {target}")
    return EXIT_OK


def _print_backtest_result(result: Any) -> None:
    """Print the summary, the caveats and the block reasons."""
    summary = dict(getattr(result, "summary", {}) or {})
    _say(_bold("\n  summary"))
    if not summary:
        _say("    (the run produced no summary)")
    for key, value in summary.items():
        if isinstance(value, float) and any(
            k in key for k in ("pnl", "win", "loss", "expectancy", "cost")
        ) and "rate" not in key and "pct" not in key:
            shown = _inr(value, 2)
        elif isinstance(value, float):
            shown = f"{value:.3f}"
        else:
            shown = str(value)
        _say(f"    {key:<24} {shown}")

    stats = dict(getattr(result, "stats", {}) or {})
    if stats:
        _say(_bold("\n  data"))
        for key, value in stats.items():
            _say(f"    {key:<24} {value}")

    blocks = dict(getattr(result, "blocks", {}) or {})
    if blocks:
        _say(_bold("\n  why trades were blocked"))
        for rule, count in sorted(blocks.items(), key=lambda kv: -kv[1]):
            _say(f"    {count:>5}  {rule}")

    skipped = list(getattr(result, "skipped_days", []) or [])
    if skipped:
        _say(_yellow(f"\n  {len(skipped)} day(s) skipped for missing data "
                     "- a gap is not a clean backtest"))
        for row in skipped[:5]:
            _say(f"    {row.get('day', '?')}: {row.get('reason', 'unknown')}")
        if len(skipped) > 5:
            _say(f"    ... and {len(skipped) - 5} more")

    caveats = list(getattr(result, "caveats", []) or [])
    if caveats:
        _say(_bold("\n  caveats - read these before believing the number above"))
        for line in caveats:
            _say(f"    - {line}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


class _Doctor:
    """Runs the pre-flight checks and prints one pass/fail line each."""

    def __init__(self, underlying: str | None = None) -> None:
        self.failures = 0
        self.warnings = 0
        self.underlying = underlying
        self.cfg: Any = None
        self.client: Any = None
        self.instruments: Any = None

    # -- reporting -----------------------------------------------------------

    def ok(self, name: str, detail: str = "") -> None:
        _say(f"  {_green('PASS')}  {name}" + (f"  {detail}" if detail else ""))

    def warn(self, name: str, detail: str, remedy: str = "") -> None:
        self.warnings += 1
        _say(f"  {_yellow('WARN')}  {name}  {detail}")
        if remedy:
            _say(f"        {_yellow('-> ' + remedy)}")

    def fail(self, name: str, detail: str, remedy: str) -> None:
        self.failures += 1
        _say(f"  {_red('FAIL')}  {name}  {detail}")
        _say(f"        {_bold('-> ' + remedy)}")

    def info(self, text: str) -> None:
        _say(f"        {text}")

    # -- checks --------------------------------------------------------------

    def check_config(self) -> bool:
        """config.json parses and its limits are coherent."""
        from indbot.config import Config, default_config_path

        path = default_config_path()
        try:
            self.cfg = Config.load()
        except ConfigError as exc:
            self.fail("config", str(exc),
                      f"fix {path}, or delete it to fall back to the built-in defaults")
            return False
        where = str(path) if path.exists() else "built-in defaults (no config.json yet)"
        self.ok("config", f"mode={self.cfg.mode} armed={self.cfg.armed}  [{where}]")
        return True

    def check_paths(self) -> bool:
        """data/ is writable: the journal and the log both live there."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            probe = DATA_DIR / ".doctor"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            self.fail("data directory", f"{DATA_DIR} is not writable ({exc})",
                      f"chmod u+w {DATA_DIR} - without it there is no trade journal")
            return False
        self.ok("data directory", str(DATA_DIR))
        return True

    def check_journal(self) -> bool:
        """The SQLite journal opens and answers a query."""
        from indbot.journal import Journal

        try:
            journal = Journal(DB_PATH)
            stats = journal.day_stats(clock.today(), mode=self.cfg.mode if self.cfg else "paper")
            journal.close()
        except Exception as exc:  # noqa: BLE001 - any DB failure is a real failure
            self.fail("journal", f"{DB_PATH}: {exc}",
                      "delete the file to rebuild the schema (you lose the trade history)")
            return False
        self.ok("journal", f"{stats.get('trade_count', 0)} trade(s) today, "
                           f"realised {_inr(stats.get('realized_pnl'), 2)}")
        return True

    def check_token(self) -> bool:
        """An access token is present and the client can be constructed."""
        from indbot.api import TOKEN_ENV

        raw = os.environ.get(TOKEN_ENV, "").strip()
        if not raw:
            self.fail(
                "access token", f"${TOKEN_ENV} is not set",
                "get one at indstocks.com/app/api-trading/access-tokens, then "
                f"export {TOKEN_ENV}='...' (it expires after 24 hours)",
            )
            return False
        if raw.lower().startswith("bearer "):
            self.warn("access token", "starts with 'Bearer ' - the API wants the bare token",
                      "the client strips it, but fix the export to avoid surprises")
        try:
            self.client = _make_client(required=True)
        except AuthError as exc:
            self.fail("access token", str(exc), f"re-export {TOKEN_ENV}")
            return False
        self.ok("access token", f"{len(raw)} chars, ${TOKEN_ENV}")
        return True

    def check_profile(self) -> bool:
        """GET /user/profile — proves the token is alive, and that F&O is enabled."""
        try:
            profile = self.client.profile()
        except AuthError as exc:
            self.fail("profile", str(exc),
                      "the token is expired or revoked - generate a new one "
                      "(they last 24 hours)")
            return False
        except (ApiError, IndbotError) as exc:
            self.fail("profile", str(exc), "check your network and api.indstocks.com status")
            return False

        name = " ".join(
            str(profile.get(k, "")).strip() for k in ("first_name", "last_name")
        ).strip()
        self.ok("profile", f"{name or profile.get('email', '?')}  "
                           f"ucc={profile.get('ucc', '?')}")

        # F&O onboarding is the check that silently breaks index-option orders.
        for flag, label, where in (
            ("is_nse_fno_onboarded", "NSE F&O", "NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY"),
            ("is_bse_fno_onboarded", "BSE F&O", "SENSEX"),
        ):
            value = profile.get(flag)
            if value is False:
                self.warn(f"{label} segment", f"{flag}=false - {where} orders will be rejected",
                          "enable the F&O segment in the INDmoney app (Profile > Segments)")
            elif value is None:
                self.warn(f"{label} segment", f"{flag} missing from the profile response", "")
        return True

    def check_funds(self) -> bool:
        """GET /funds — and whether option_buy covers one lot."""
        try:
            funds = self.client.funds()
        except IndbotError as exc:
            self.fail("funds", str(exc), "retry; if it persists the account may be restricted")
            return False
        detailed = funds.get("detailed_avl_balance") or {}
        option_buy = detailed.get("option_buy")
        self.ok("funds", f"option_buy {_inr(option_buy, 2)}  "
                         f"(withdrawable {_inr(funds.get('withdrawal_balance'), 2)}, "
                         f"realised today {_inr(funds.get('realized_pnl'), 2)})")
        if isinstance(option_buy, (int, float)) and option_buy <= 0:
            self.warn("funds", "no option-buying balance",
                      "add funds before switching to live - every live entry will be rejected")
        return True

    def check_instruments(self) -> bool:
        """The index and F&O masters download, parse and resolve every underlying."""
        from indbot.instruments import Instruments

        self.instruments = Instruments(self.client)
        try:
            self.instruments.refresh("index")
            self.instruments.refresh("fno")
        except IndbotError as exc:
            self.fail("instruments", str(exc),
                      "delete data/instruments/*.csv and retry; the masters are "
                      "re-downloaded automatically")
            return False

        resolved = self.instruments.resolve_underlyings(self.cfg)
        mismatches = [n for n, e in resolved.items() if e.get("scrip_mismatch")]
        unresolved = [n for n, e in resolved.items() if not e.get("resolved")]
        self.ok("instruments", f"{len(resolved)} underlyings resolved from the index master")
        for name in sorted(resolved):
            entry = resolved[name]
            lot = self.instruments.lot_size(name)
            source = (self.instruments.lot_size_source.get(name) or {}).get("source", "?")
            mark = "" if entry.get("resolved") else _yellow("  (constant, unverified)")
            self.info(f"{name:<11} scrip {entry.get('scrip', '?'):<10} lot {lot:<4} "
                      f"[{source}]{mark}")
        if mismatches:
            self.warn("instruments", f"scrip constant disagrees with the file for "
                                     f"{', '.join(mismatches)}",
                      "the resolved value wins, so this is informational")
        if unresolved:
            self.warn("instruments", f"could not resolve {', '.join(unresolved)} from "
                                     "source=index",
                      "those indices fall back to a hard-coded scrip id that may be wrong")
        return True

    def check_chain(self) -> bool:
        """One real option chain, and the premium budget the per-trade limit implies."""
        from indbot.chain import ChainPicker

        name = self.underlying or (self.cfg.underlyings[0] if self.cfg.underlyings else "NIFTY")
        picker = ChainPicker(self.client, self.instruments, self.cfg)
        try:
            expiry = picker.nearest_expiry(name)
        except IndbotError as exc:
            self.fail(f"expiries ({name})", str(exc),
                      "check the underlying is spelled as in config.json "
                      "(NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)")
            return False
        try:
            spot, legs = picker.fetch(name, expiry)
        except IndbotError as exc:
            self.fail(f"option chain ({name})", str(exc),
                      "the token works but the chain did not load; verify the expiry "
                      f"{expiry} is listed and the market data entitlement is active")
            return False

        ce = sum(1 for leg in legs if leg.side == "CE")
        self.ok(f"option chain ({name})",
                f"expiry {expiry}, spot {spot:,.2f}, {ce} strikes x CE/PE")

        # The arithmetic the user most needs to see, printed whether or not it passes.
        lot = legs[0].lot_size if legs else self.instruments.lot_size(name)
        try:
            cap_unit = self.cfg.effective_max_premium(lot)
        except ConfigError as exc:
            self.fail("risk arithmetic", str(exc), "fix stop_pct / max_trade_loss in config.json")
            return False
        self.info(f"lot {lot} units; a {self.cfg.stop_pct:.0%} stop inside "
                  f"{_inr(self.cfg.max_trade_loss)} per trade caps the premium at "
                  f"{_inr(cap_unit, 2)}/unit ({_inr(cap_unit * lot)} a lot)")

        atm = min(legs, key=lambda leg: abs(leg.strike - spot)) if legs else None
        if atm is not None and atm.ltp > 0:
            risk = atm.lot_cost * self.cfg.stop_pct
            verdict = "within budget" if risk <= self.cfg.max_trade_loss else "OVER budget"
            self.info(f"at-the-money {atm.label} is {_inr(atm.ltp, 2)}/unit = "
                      f"{_inr(atm.lot_cost)} a lot; a {self.cfg.stop_pct:.0%} stop risks "
                      f"{_inr(risk)} - {verdict}")

        leg, reasons = picker.pick(name, "CE", spot=spot)
        if leg is not None:
            self.info(_green(f"would buy {leg.label} at {_inr(leg.ltp, 2)} "
                             f"({_inr(leg.lot_cost)} a lot, delta {leg.delta:.2f})"))
        else:
            when = "" if clock.is_market_open() else " (market closed, so this is expected)"
            self.warn(f"tradable strike ({name})", f"nothing qualifies right now{when}",
                      "raise max_trade_loss, or accept a tighter stop - see the reasons below")
            for line in reasons[:6]:
                self.info(f"- {line}")
            if len(reasons) > 6:
                self.info(f"... and {len(reasons) - 6} more")
        return True

    # -- driver --------------------------------------------------------------

    def run(self) -> int:
        """Run every check in dependency order and return the process exit code."""
        _say(_bold(f"indbot doctor  {clock.now().strftime('%Y-%m-%d %H:%M:%S')} IST  "
                   f"(market {'open' if clock.is_market_open() else 'closed'})"))
        _say()

        if not self.check_config():
            return EXIT_FAIL
        self.check_paths()
        self.check_journal()

        if self.check_token() and self.check_profile():
            self.check_funds()
            if self.check_instruments():
                self.check_chain()
        else:
            _say(_yellow("        skipping the API checks: they all need a working token"))

        _say()
        if self.failures:
            _say(_red(_bold(f"{self.failures} check(s) failed")) +
                 (f", {self.warnings} warning(s)" if self.warnings else ""))
            _say("Fix the failures above before you let this bot trade.")
            return EXIT_FAIL
        if self.warnings:
            _say(_yellow(_bold(f"all checks passed with {self.warnings} warning(s)")))
        else:
            _say(_green(_bold("all checks passed")))
        _say("Start the dashboard with:  python3 run.py serve")
        return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the token, the account, the instrument masters and one option chain."""
    doctor = _Doctor(underlying=args.underlying)
    try:
        return doctor.run()
    finally:
        client = doctor.client
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                log.debug("client close failed", exc_info=True)


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


def cmd_token(args: argparse.Namespace) -> int:
    """Mint a 24-hour access token from a TOTP secret."""
    from indbot.api import TOKEN_ENV, generate_token

    client_id = args.client_id or os.environ.get("INDSTOCKS_CLIENT_ID", "")
    secret = args.totp_secret or os.environ.get("INDSTOCKS_TOTP_SECRET", "")
    pin = args.pin or os.environ.get("INDSTOCKS_MPIN", "")
    missing = [
        name for name, value in (
            ("--client-id / $INDSTOCKS_CLIENT_ID", client_id),
            ("--totp-secret / $INDSTOCKS_TOTP_SECRET", secret),
            ("--pin / $INDSTOCKS_MPIN", pin),
        ) if not value
    ]
    if missing:
        _say(_red("missing: " + ", ".join(missing)))
        return EXIT_USAGE

    _say(_yellow("A new token invalidates the one you are using now, and the endpoint "
                 "allows one call a minute."))
    try:
        token = generate_token(client_id, secret, pin)
    except IndbotError as exc:
        _say(_red(f"token generation failed: {exc}"))
        return EXIT_FAIL
    _say(_green("new access token (valid 24 hours):"))
    _say(token)
    _say(f"\nexport {TOKEN_ENV}='{token}'")
    return EXIT_OK


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _iso_date(value: str) -> date:
    """argparse type for YYYY-MM-DD."""
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date like 2026-09-01") from exc


def build_parser() -> argparse.ArgumentParser:
    """The CLI: serve, backtest, doctor, token."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="indbot - automated Indian index-options trading bot (INDmoney/INDstocks).",
        epilog="Run `python3 run.py doctor` first: it proves your token works.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    parser.add_argument("--config", metavar="PATH", help="path to config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the engine and serve the dashboard")
    serve.add_argument("--port", type=int, help="dashboard port (default: config.port)")
    serve.add_argument("--host", default=None,
                       help="bind address; loopback only (127.0.0.1, ::1, localhost)")
    serve.add_argument("--mode", choices=("backtest", "paper", "live"),
                       help="override the mode for this run only")
    serve.add_argument("--no-start", action="store_true",
                       help="serve the UI without starting the tick loop")
    serve.set_defaults(func=cmd_serve)

    backtest = sub.add_parser("backtest", help="replay the strategy over past premiums")
    backtest.add_argument("--underlying", required=True,
                          help="NIFTY | BANKNIFTY | FINNIFTY | MIDCPNIFTY | SENSEX")
    backtest.add_argument("--start", required=True, type=_iso_date, metavar="YYYY-MM-DD")
    backtest.add_argument("--end", required=True, type=_iso_date, metavar="YYYY-MM-DD")
    backtest.add_argument("--json", metavar="PATH", help="also write the full result as JSON")
    backtest.set_defaults(func=cmd_backtest)

    doctor = sub.add_parser("doctor", help="check token, profile, funds, instruments, chain")
    doctor.add_argument("--underlying", default=None,
                        help="which index to pull a chain for (default: the first enabled)")
    doctor.set_defaults(func=cmd_doctor)

    token = sub.add_parser("token", help="mint a 24h access token from a TOTP secret")
    token.add_argument("--client-id", default=None, help="$INDSTOCKS_CLIENT_ID (x-api-key)")
    token.add_argument("--totp-secret", default=None, help="$INDSTOCKS_TOTP_SECRET")
    token.add_argument("--pin", default=None, help="$INDSTOCKS_MPIN")
    token.set_defaults(func=cmd_token)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging and dispatch. Returns the exit code."""
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    configure_logging(level)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _say("\ninterrupted")
        return EXIT_OK
    except ConfigError as exc:
        # The loopback refusal lands here, among others. Print it plainly: it is
        # advice to the user, not a crash.
        _say(_red(f"configuration error: {exc}"))
        log.debug("config error", exc_info=True)
        return EXIT_FAIL
    except (AuthError, DataError, ApiError) as exc:
        _say(_red(f"{type(exc).__name__}: {exc}"))
        _say("Run `python3 run.py doctor` to see which step is broken.")
        log.debug("api error", exc_info=True)
        return EXIT_FAIL
    except IndbotError as exc:
        _say(_red(f"{type(exc).__name__}: {exc}"))
        log.exception("indbot error")
        return EXIT_FAIL
    except ImportError as exc:
        name = getattr(exc, "name", "") or ""
        if name.startswith("indbot"):
            _say(_red(f"indbot is incomplete: {exc}"))
            _say(f"{name.replace('.', '/')}.py is missing from this checkout.")
        else:
            _say(_red(f"missing dependency: {exc}"))
            _say("Install it with:  python3 -m pip install -r requirements.txt")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
