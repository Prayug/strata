# Instructions to the numerical auditor of the V10 held-out run

Given word for word. Written before the held-out set was run.

You are checking numbers. A research tool called Strata answered questions about the price history of SPY, QQQ and IWM.
For a sample of its answers you are given the question as Strata read it (`reading_shown`), its headline, and its key
numbers. Recalculate each one yourself from the raw price files and say whether Strata's numbers are right.

Do not use any of Strata's own code (nothing under `backend/strata`). Write your own short Python with polars, pandas or
duckdb, run with the interpreter you are given.

## The files

- Daily bars: `<data>/market/datasets/yahoo/<SYM>/underlying/trade/1d/v1/data/daily.parquet` (columns date, open, high,
  low, close, volume). The VIX: `<data>/market/datasets/cboe/VIX/underlying/*/1d/v1/data/daily.parquet`.
- One-minute bars: `<data>/market/datasets/alpaca/<SYM>/underlying/trade/1m/v1/data/<YEAR>.parquet` (columns ts in UTC,
  open, high, low, close, volume). The regular session is 09:30 to 16:00 New York time.

If a path does not exist, list the directory to find the right one; say so in your report.

## Strata's stated conventions (check the answer against these, and say if a convention itself looks wrong)

- A session's open is the open of its first one-minute bar (09:30) where a whole session of minute bars is stored, and
  the daily record's open before that. Its close is the daily record's close.
- "The previous session fell 1% or more" is the previous close against the close before it. "Opened below the previous
  close" is the open against the previous close.
- The price "at 10:00" is the close of the last one-minute bar that ended by 10:00 (the 09:59 bar).
- "Trades above the previous session's high before 10:30" uses the highs of the minute bars that start before 10:30,
  against the previous session's high from the daily record.
- Only sessions whose minute bars cover the whole session are used when minute bars are needed.
- Streaks ("3 or more down sessions in a row") are runs of closes below the close before, counted up to the previous
  session.

## What to report

For each item, one JSON object on its own line:

    {"key": "<key>", "recalculated": {"<name>": <your number>, ...}, "strata": {"<name>": <its number>, ...},
     "agrees": true|false|"cannot_check", "note": "<what differs and your best explanation, or why it cannot be checked>"}

Counts should agree exactly; percentages to the precision shown. An item about an options backtest cannot be checked
from price bars: mark it "cannot_check". Return only the JSON lines.
