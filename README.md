# ind-trade-bot

An automated Indian index-options trading bot, plus the reference material it was
built from.

**→ The bot and its full documentation are in [`indbot/`](indbot/README.md).**

```bash
cd indbot
python3 -m pip install -r requirements.txt
export INDSTOCKS_ACCESS_TOKEN="your_token"
python3 run.py doctor          # prove the token and the data path work
python3 run.py serve           # dashboard on http://127.0.0.1:8765
```

---

## What is in here

| path | what it is |
|---|---|
| [`indbot/`](indbot/) | **The bot.** Scans the five major index option chains, generates call/put signals, buys one lot when a signal clears every gate, sells it on a stop, target, trailing stop or opposing signal, and keeps score of its own setups. Backtest → paper → live, with live behind a typed-phrase arming gate. |
| [`indstocks-api-docs.md`](indstocks-api-docs.md) | A local copy of INDmoney's published INDstocks API documentation, kept for offline reference while building. Their work, not mine. |
| [`day-trader-skill-main/`](day-trader-skill-main/) | [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) by Jake Schincariol, MIT licensed, included as the reference the strategy structure was adapted from. Its own LICENSE travels with it. It targets US equities on free Yahoo data; `indbot` is a different system for Indian index options on a broker API, and shares none of its code. |

## Read this before you run it

This software places real orders with real money. Most people who day trade lose
money, and automating it changes the speed, not the odds. Nothing here is
financial advice.

The one thing worth knowing up front: with the default **₹1,000 per-trade loss
limit** and a **30% stop**, a position is capped at ₹3,333 of premium. Against
real lot sizes that means an at-the-money option will almost never pass the risk
gate — NIFTY's lot of 65 allows ₹51 a unit against a typical ATM premium of
₹150–200. That is deliberate, and the bot blocks the trade and shows you the
arithmetic rather than quietly widening your stop or drifting to far-OTM strikes.

[The full explanation, and the three honest ways to change it, is in the bot's
README.](indbot/README.md#the-constraint-you-should-understand-first)

Start in backtest. Then paper. Then live, small. In that order.
