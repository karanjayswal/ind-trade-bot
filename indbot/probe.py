#!/usr/bin/env python3
"""Probe the two live failures: index scrip codes, and why a chain yields no legs.

Run from the indbot folder with your token exported:

    python3 probe.py                 # both probes
    python3 probe.py codes           # just the scrip-code hunt
    python3 probe.py chain SENSEX    # just the chain filter report

Nothing here places an order or writes to the journal. It is read-only.

Why it exists
-------------
Two errors came out of a live run:

  GET /market/historical/5minute failed (400): Invalid scrip codes
  option chain for SENSEX 2026-10-29 held 21 strikes but no usable legs

The first is an identifier-format bug. `/market/instruments?source=index` lists
NIFTY 50 as SECURITY_ID 40000001, and that is the right value for the option
chain's `underlying-scrip`. But the docs warn that the WebSocket surface uses
`NIDX:26000` for the same index, and that the two surfaces may not share a token.
The historical endpoint takes `SEGMENT_TOKEN` — and which SEGMENT and which TOKEN
an index wants there is not documented. So rather than guess, this tries every
plausible combination and reports which one the API actually accepts.

The second is a filter question: the chain came back with strikes, so the data is
there; something in the picker rejected all of them. This prints the chain and the
filter verdict strike by strike, so the binding filter is visible rather than
inferred.
"""

from __future__ import annotations

import sys
from datetime import timedelta

from indbot import clock
from indbot.api import INDstocksClient, resolve_ca_bundle
from indbot.config import UNDERLYINGS, Config
from indbot.errors import IndbotError

#: Segment prefixes worth trying for an index on the REST scrip-code surface.
SEGMENTS = ["NIDX", "BIDX", "NSE", "BSE", "IDX"]

#: Token spaces seen in the docs for the same index.
EXTRA_TOKENS = {
    "NIFTY": ["26000"],
    "BANKNIFTY": ["26009"],
    "FINNIFTY": ["26037"],
    "MIDCPNIFTY": ["26074"],
    "SENSEX": ["1", "51"],
}


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def probe_codes(client: INDstocksClient) -> dict[str, str]:
    """Try every (segment, token) pair per index; report what the API accepts."""
    hr("1. INDEX SCRIP CODES — which SEGMENT_TOKEN does /market/historical accept?")

    # The index master is the documented source for one of the two token spaces.
    print("\n-- /market/instruments?source=index (read positionally: col 2 is the NAME) --")
    master: dict[str, str] = {}
    try:
        for line in client.instruments_csv("index").splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                master[parts[1].upper()] = parts[2]
        for name, meta in UNDERLYINGS.items():
            label = str(meta["label"]).upper()
            print("   %-11s label=%-22s master=%s" % (name, label, master.get(label, "NOT FOUND")))
    except IndbotError as exc:
        print("   could not read the index master:", exc)

    end = clock.now()
    start = end - timedelta(days=3)
    s_ms, e_ms = clock.epoch_ms(start), clock.epoch_ms(end)

    working: dict[str, str] = {}
    for name, meta in UNDERLYINGS.items():
        label = str(meta["label"]).upper()
        tokens: list[str] = []
        if label in master:
            tokens.append(master[label])
        if str(meta.get("scrip")) not in tokens:
            tokens.append(str(meta["scrip"]))
        tokens += [t for t in EXTRA_TOKENS.get(name, []) if t not in tokens]

        print("\n-- %s (trying %d codes) --" % (name, len(SEGMENTS) * len(tokens)))
        found = None
        for token in tokens:
            for seg in SEGMENTS:
                code = f"{seg}_{token}"
                try:
                    data = client.historical("5minute", [code], s_ms, e_ms)
                except IndbotError as exc:
                    msg = str(exc)
                    detail = "Invalid scrip codes" if "scrip" in msg.lower() else msg[:58]
                    print("   %-18s ->  %s" % (code, detail))
                    continue
                candles = data.get(code) or []
                if candles:
                    last = candles[-1]
                    print("   %-18s ->  OK  %d candles, last %s close %s"
                          % (code, len(candles), clock.to_ist(last["ts"]).strftime("%d %b %H:%M"),
                             last["c"]))
                    found = code
                    break
                print("   %-18s ->  accepted but EMPTY" % code)
            if found:
                break
        if found:
            working[name] = found
        else:
            print("   !! nothing worked for %s" % name)
    return working


def probe_chain(client: INDstocksClient, cfg: Config, name: str) -> None:
    """Print a chain strike by strike with the filter verdict for each leg."""
    hr(f"2. OPTION CHAIN — {name}: why did no leg qualify?")
    meta = UNDERLYINGS.get(name)
    if not meta:
        print("unknown underlying", name)
        return

    try:
        expiries = client.expiries(name)
    except IndbotError as exc:
        print("could not list expiries:", exc)
        return
    if not expiries:
        print("no upcoming expiries returned")
        return
    expiry = expiries[0]
    print(f"nearest expiry: {expiry}   (all upcoming: {', '.join(expiries[:5])})")

    try:
        chain = client.option_chain(
            str(meta["exchange"]), str(meta["chain_segment"]), str(meta["scrip"]), expiry, 10
        )
    except IndbotError as exc:
        print("chain fetch failed:", exc)
        return

    strikes = chain.get("strikes") or {}
    spot = chain.get("underlying_ltp")
    print(f"underlying_ltp: {spot}   strikes returned: {len(strikes)}")

    # Lot size decides the premium budget, and the budget is usually the binding filter.
    lot = 0
    try:
        lot = client_lot(client, name, expiry)
    except Exception:
        pass
    if not lot:
        from indbot.instruments import FALLBACK_LOTS
        lot = FALLBACK_LOTS.get(name, 1)
        print(f"lot size: {lot} (fallback — could not read LOT_UNITS)")
    else:
        print(f"lot size: {lot} (from the fno master)")

    budget = cfg.effective_max_premium(lot)
    print(f"premium budget: Rs {budget:,.2f}/unit  "
          f"(= max_trade_loss {cfg.max_trade_loss:,.0f} / (stop {cfg.stop_pct:.0%} x lot {lot}))")
    print(f"filters: delta {cfg.min_delta}-{cfg.max_delta}, spread <= {cfg.max_spread_pct}%, "
          f"oi >= {cfg.min_oi:,}, volume >= {cfg.min_volume:,}\n")

    hdr = ("strike", "side", "ltp", "lot cost", "delta", "spr%", "oi", "vol", "verdict")
    print("%-9s %-4s %9s %10s %7s %6s %10s %9s  %s" % hdr)
    print("-" * 100)

    reasons: dict[str, int] = {}
    for key in sorted(strikes, key=lambda k: float(k)):
        for side in ("ce", "pe"):
            leg = (strikes[key] or {}).get(side) or {}
            if not leg:
                continue
            ltp = float(leg.get("last_price") or 0)
            bid = float(leg.get("top_bid_price") or 0)
            ask = float(leg.get("top_ask_price") or 0)
            oi = int(leg.get("oi") or 0)
            vol = int(leg.get("volume") or 0)
            delta = abs(float((leg.get("greeks") or {}).get("delta") or 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else ltp
            spread_pct = ((ask - bid) / mid * 100) if (mid > 0 and bid > 0 and ask > 0) else 999.0
            cost = ltp * lot

            why = []
            if ltp <= 0 or bid <= 0 or ask <= 0:
                why.append("no two-sided book")
            if ltp > budget:
                why.append(f"premium Rs{ltp:.2f} > budget Rs{budget:.2f}")
            if not (cfg.min_delta <= delta <= cfg.max_delta):
                why.append(f"delta {delta:.2f} outside band")
            if spread_pct > cfg.max_spread_pct:
                why.append(f"spread {spread_pct:.2f}% too wide")
            if oi < cfg.min_oi:
                why.append(f"oi {oi:,} < {cfg.min_oi:,}")
            if vol < cfg.min_volume:
                why.append(f"vol {vol:,} < {cfg.min_volume:,}")

            verdict = "PASS" if not why else why[0]
            for w in why:
                tag = ("premium" if "premium" in w else "delta" if "delta" in w
                       else "spread" if "spread" in w else "oi" if w.startswith("oi")
                       else "volume" if w.startswith("vol") else "book")
                reasons[tag] = reasons.get(tag, 0) + 1
            print("%-9s %-4s %9.2f %10.0f %7.2f %6.2f %10d %9d  %s"
                  % (key, side.upper(), ltp, cost, delta, min(spread_pct, 999), oi, vol, verdict))

    total = len(strikes) * 2
    print("\n-- how many legs each filter rejected (a leg can fail several) --")
    for tag, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("   %-9s %3d of %d legs" % (tag, n, total))
    print("\nThe filter at the top of that list is the one to change first.")


def client_lot(client: INDstocksClient, name: str, expiry: str) -> int:
    """LOT_UNITS for this underlying, via the same lookup the engine uses.

    Originally this parsed the CSV here and matched on column 15, SYMBOL_NAME —
    which is EMPTY on every F&O row, so it always reported "fallback". Going
    through Instruments means the probe exercises the real code path instead of a
    second, differently-broken copy of it.
    """
    from pathlib import Path

    from indbot.instruments import Instruments

    inst = Instruments(client, Path(__file__).resolve().parent / "data" / "instruments")
    return int(inst.lot_size(name, expiry))


def main() -> int:
    args = sys.argv[1:]
    what = args[0] if args else "all"
    which = args[1].upper() if len(args) > 1 else "SENSEX"

    ca = resolve_ca_bundle()
    print(f"CA bundle: {ca or 'system default'}")
    try:
        client = INDstocksClient()
    except IndbotError as exc:
        print("cannot build a client:", exc)
        return 1

    cfg = Config.load()
    try:
        if what in ("all", "codes"):
            working = probe_codes(client)
            if working:
                hr("RESULT — working index codes")
                for k, v in working.items():
                    print(f"   {k:<11} {v}")
                print("\nPaste this back and the engine's _scrip_code() will be fixed to match.")
        if what in ("all", "chain"):
            probe_chain(client, cfg, which)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
