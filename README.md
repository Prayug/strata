# Strata — options research terminal

Strata is a local research platform for options strategies. You can describe a strategy or a
market question in plain English, or build it visually; a deterministic Python engine runs it
on **real historical market data** and reports what happened, with the assumptions and data
limits stated next to every number.

> **Real data first.** A fresh install downloads free, real market history (no account
> needed). Simulated prices are never used as a default and never substituted when real data
> is missing. If a request cannot be answered with the data on disk, Strata says exactly what
> is missing instead.

---

## Contents
- [Quick start](#quick-start)
- [Three research modes](#three-research-modes)
- [AI Strategy Lab](#ai-strategy-lab)
- [Market data](#market-data)
- [Rules and the expression language](#rules-and-the-expression-language)
- [Engine assumptions](#engine-assumptions-read-this)
- [Optimisation & overfitting controls](#optimisation--overfitting-controls)
- [REST API, CLI & Python strategies](#rest-api-cli--python-strategies)
- [Testing & validation](#testing--validation)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

## Quick start

Requirements: Python 3.11+ (3.12 tested), Node 20+ (22 tested), `npm` or `pnpm`. `uv` is used if present.

```bash
scripts/setup.sh     # venv + backend, frontend deps, database, free real market data (a few MB)
scripts/dev.sh       # API on :8765, UI on :5199
```
Open http://localhost:5199. API docs are at http://localhost:8765/docs.

`setup.sh` downloads daily history since 1993 and recent intraday bars for SPY, QQQ and IWM,
plus VIX, from Yahoo Finance and Cboe. To add real option chains (free, about 350 MB for five
years of SPY):

```bash
backend/.venv/bin/strata fetch SPY --options              # end-of-day chains, last five years
backend/.venv/bin/strata fetch SPY --options --years 2015-2025
```

Docker: `docker compose up --build` serves the API and built UI on http://localhost:8000 and
runs the same free download on first start (`STRATA_SKIP_FETCH=1` to skip it).

### First five minutes
1. The app opens on the **AI Lab**. Ask *"How often does SPY rise 1% from the open after falling
   3% or more the previous trading day?"* The interpretation, the answer with its confidence
   interval, the baseline, and every qualifying session appear within a few seconds.
2. Follow up: *"Change the previous-day decline to 2% and compare the results."* A new version is
   created with a field-level diff; the original run is kept.
3. After `strata fetch SPY --options`, try *"Buy a 30DTE ATM SPY call at the close after a 2% down
   day and hold for 5 days"* for a backtest on observed option quotes.
4. **Market Explorer** builds the same studies with a form; **Strategy Builder** builds option
   strategies leg by leg; **Data** shows what is stored and what each dataset cannot answer.

Keyboard: `⌘K` command palette · `g` + key to switch page (a, m, b, r, o, t, p, d, s) · `⌘↵` run a
backtest in the builder · `⌘\` collapse the sidebar.

## Three research modes

Every result is labelled with the mode that produced it.

| Mode | When it is used | What the numbers are |
|---|---|---|
| **Historical options backtest** | Observed option quotes cover the request (symbol, dates, time of day, expirations). | Fills at stored bid/ask quotes. Quote-based and trade-print-based data are distinguished; a bid-ask spread is never invented from OHLC bars. |
| **Historical market study** | The question is about the underlying, or option quotes are missing. | How the underlying behaved: frequency with a Wilson 95% interval, baseline on all sessions, MFE/MAE, time to target, by year and regime. Never presented as option P&L. |
| **Modeled options backtest** | Only when you opt in (a button, or by asking for modeled premiums). | Real underlying prices with theoretical premiums (Barone-Adesi-Whaley American or Black-Scholes; volatility from VIX, realised or fixed; stated skew and spread). Labelled *estimated* on every screen. |

Strata recommends a mode from the data on disk. It never recommends modeled premiums, never
mixes them into a historical backtest, and never uses end-of-day option data to simulate an
intraday fill. For example, with end-of-day SPY chains stored:

> *"Buy one ATM SPY 0DTE call at 6:30 AM Pacific after a 3% down day, exit when SPY moves up 1%
> from entry or at 12:45 PM Pacific."* → the strategy is interpreted in full (09:30 ET entry,
> exact 0DTE, underlying target, 15:45 ET exit), the historical backtest is refused with the
> reason (*"SPY option history is end-of-day only… an entry at the open and 0DTE contracts need
> intraday option quotes"*), and the lab offers the market study on the underlying, or a modeled
> estimate if you choose it.

## AI Strategy Lab

```
your words ──► interpreter ──► Draft (structured, validated) ──► compiler ──► engine config
                (rules or LLM)                                     │  every default recorded
                                                                   ▼  as an assumption
                 grounded explanation ◄── stored result ◄── job system ◄── coverage check
```

- **The interpreter only fills in a `Draft`.** The deterministic engine calculates every
  financial result. There is no separate "AI simulation".
- **Works with no API key.** A built-in rule-based interpreter handles entries, exits, clock
  times, previous-day moves, gaps, VWAP, opening ranges, streaks, VIX filters, sweeps
  ("test 0DTE, 1DTE and 2DTE"), comparisons, "puts instead", out-of-sample splits and
  "show me every losing trade".
- **Optional language models** (Settings): Anthropic Claude, OpenAI, Google Gemini, OpenRouter,
  or a local Ollama model, behind one provider-independent interface. The model may only choose
  from a fixed list of tools (list datasets, check coverage, look up functions, validate an
  expression, read computed results, propose a draft, run a sweep, show trades, out-of-sample).
  It cannot run code or SQL. Its drafts are validated server-side and sent back for repair when
  invalid; if it fails, the built-in interpreter takes over.
- **Keys stay on the server.** They are stored in the local database or read from the
  environment; the browser only ever receives the last four characters.
- **Nothing is assumed silently.** The interpretation panel lists every field as *you said*,
  *assumed* or *missing*. No stop, target or expiration is invented without being shown, and
  every field is editable.
- **Units are kept apart.** A 1% move of SPY, a 1% return on the option, a $1 move and a 100%
  premium gain are four different fields. "The previous day fell 3%" means close-to-close unless
  you say open-to-close or gap.
- **Times.** Pacific, Mountain, Central and Eastern are converted to New York time by the
  server (not by the model). A time without a timezone uses the default in Settings and is
  flagged as an assumption. `open` and `close` follow half days.
- **Versions.** Each change is a new version with a diff; earlier runs are never overwritten.
  Restore any version and continue from it as a branch.
- **Grounded explanations.** The default explanation is assembled from numbers read out of the
  stored result. If a language model rewords it, every number in its text is checked against
  those facts; one unmatched number and the model's text is discarded.

## Market data

### Free sources (implemented and tested against the live services)

| Source | Account | What it provides | Licence |
|---|---|---|---|
| Yahoo Finance | none | Daily bars for the full listing history with dividends and splits; intraday bars: 60-minute (~2 years), 5-minute (~60 days), 1-minute (~30 days, rolling) | Yahoo terms (personal research) |
| Cboe | none | Official daily VIX history since 1990 | Cboe published data |
| EOD options archive | none | Full end-of-day chains for SPY and IWM 2008–2025 and QQQ 2011–2025: bid/ask, sizes, volume, open interest, IV, Greeks | MIT (P. Dubach dataset mirror) |
| DoltHub `post-no-preference/options` | none | End-of-day chains since 2019 for about 2,000 symbols (four expirations, ~27 strikes; no volume/OI) | CC BY-SA 4.0 |
| Alpaca | free key | 1-minute stock bars since 2016; option *trade* bars since Feb 2024 (no historical option quotes on the free plan) | Alpaca data agreement |

Ingestion estimates size before large downloads, resumes, retries with backoff, skips what is
already stored, rejects crossed or negative quotes, nulls implausible vendor IV/Greeks so they
are recomputed from the observed quotes, and records provider, resolution, licence and a
dataset version for every file. **Data → Get free data** does all of this from the UI.

**What free data cannot do:** no free source we could verify offers *intraday option quotes*.
0DTE and time-of-day option backtests therefore need a paid feed (ThetaData, Databento,
Polygon/Massive adapters are included and never called automatically) or your own files
(CSV/Parquet import; OptionsDX-style files import directly). Until then those requests get a
market study, or a modeled estimate if you opt in.

### Storage
```
market/daily/<SYM>.parquet               daily bars
market/underlying/<SYM>/<YYYY>.parquet    1-minute bars
market/bars/<res>/<SYM>/<YYYY>.parquet    5m / 15m / 30m / 60m bars
market/options/<SYM>/<YYYY>/<date>.parquet  quote snapshots (one file per session)
market/manifests/<SYM>.json               per-file sha256 → dataset version
```
The engine reads the finest bars stored for each session. Session closes come from the daily
record where one exists, so day-over-day returns match published closes.

### Developer sandbox
`strata sandbox` (or **Data → Developer sandbox**) generates clearly labelled **SIMULATED**
symbols (`DEMO`, `DEMOX`, `DEMOVIX`) for testing the software without market data. They are
never a default, never substituted for missing data, and a backtest only reads them when the
strategy names one and `allow_synthetic` is set.

## Rules and the expression language

The visual builder, Market Explorer and the AI Lab share one rule registry. Besides the
form-based conditions (times, weekdays, indicators, VIX, IV rank, events…), any rule can be an
expression:

```
previous_session.return_pct <= -3
underlying.price >= entry.underlying_price * 1.01
consecutive_bearish_candles("5m") >= 3
crosses_above(price, vwap)
return_pct("SPY", sessions=5) < -4
first_touch(previous_high, previous_low) == 1
gap_pct <= -0.5 and price > opening_range_high(15)
```

Available: session and previous-session prices and returns, gaps and gap fill, VWAP and distance
from it, opening range, previous week/month highs and lows, rolling highs/lows, retracements,
first-touch ordering, time above/below a level, consecutive days and candles on any timeframe,
crossovers, RSI/SMA/EMA/ATR/MACD/Bollinger, realised volatility and its percentile, volume
ratios, cross-symbol returns and relative strength, VIX, IV rank, events, and position fields
for exits. The reference is searchable in the UI (the book icon next to any rule).

Expressions are parsed into an AST and interpreted against a whitelist of syntax, names,
fields and functions. They are never `eval`'d; imports, attribute access, subscripts, lambdas
and unknown names are rejected when the rule is saved.

**Missing inputs are tri-state.** If a rule cannot be evaluated (no VIX that day, no intraday
bars for an intraday rule), the session is *excluded and reported with the reason*. It is never
counted as "condition false".

## Engine assumptions (read this)

- **Separate clocks.** Rules are evaluated on the underlying's bars: at the open, every
  `signal_interval_minutes`, at the close, and at any explicit clock time the session's bars
  can observe. Orders fill at the first *fresh* option quote at or after signal time plus
  `latency_seconds`, within `max_fill_delay_minutes`; otherwise the order expires and is
  reported. Signal time and fill time are recorded separately on every trade.
- **No look-ahead.** A bar is visible only once it has ended. Daily indicators use completed
  sessions. Quotes time-stamped after the decision are rejected. The test suite perturbs all
  data after a cut point and checks earlier decisions are bit-identical, and injects three
  deliberate look-ahead bugs that must be caught.
- **Fills.** Conservative by default (buy at the ask, sell at the bid), midpoint, or custom,
  plus configurable slippage, commissions and fees. Option data made of trade prints is flagged
  and filled at the last trade with an explicit, separately configured slippage; no spread is
  invented.
- **Exact expirations.** `0DTE` means the contract expiring that day. If none is listed the
  trade is skipped and reported; it never rolls to a later expiry. Other targets can be nearest,
  exact, at-least or at-most.
- **Exits.** Time and underlying-based rules are evaluated on the underlying clock; premium-based
  rules need a fresh quote. An exit requested without a fresh quote is deferred to the next one
  and recorded with both timestamps. If a loss-side and a profit-side rule fire together, the
  loss is recorded and the trade is flagged.
- **Missing data is never filled with model prices** in historical mode. Stale quotes are
  rejected for entries; missing quotes defer exits; contracts at expiration settle at intrinsic.
- **Settlement.** Cash-settled index options settle at the close (or the open for AM-settled
  monthlies). Physically settled options such as SPY deliver stock, sold at the next open with
  stock slippage. Short American options can be assigned early.
- **Splits.** Vendor split-adjusted price history is re-expressed in as-traded terms so it lines
  up with as-traded option strikes; open contracts are adjusted on the split date.
- **Margin.** Strategy-based Reg-T or cash. Margin deficits liquidate positions.
- **Calendar.** NYSE holidays, early closes and DST via `zoneinfo`. Storage is UTC; rules use
  New York time.
- **Reproducibility.** A run is keyed by the configuration, every dataset version it reads, the
  event calendar and the engine version. Identical requests return the stored result.

## Optimisation & overfitting controls
- **Search.** Grid (seeded subsample if it exceeds the budget), random, or Bayesian (GP with a
  Matérn-5/2 kernel and EI acquisition, in parallel batches). Each configuration is a full
  backtest in its own worker process.
- **Ranking.** Configurations are ranked on in-sample data only. A minimum trade count governs
  eligibility.
- **Diagnostics reported with every study:**
  - Deflated Sharpe ratio, using the trial count including previous studies of the same saved
    strategy.
  - Probability of backtest overfitting (CSCV).
  - Neighbourhood stability.
  - Optimum-at-range-edge warnings.
  - Leverage warnings.
  - Hold-out degradation and IS/OOS rank correlation.
  - Walk-forward efficiency, with a stitched out-of-sample equity curve.
- **Monte Carlo.** Trade bootstrap (i.i.d.) and moving-block bootstrap of daily returns, which
  preserves serial dependence.

## REST API, CLI & Python strategies

FastAPI serves OpenAPI docs at `/docs`. Long-running work returns a run id; progress streams
over server-sent events at `GET /api/jobs/{id}/stream`.

| Endpoint | Purpose |
|---|---|
| `/api/ai/sessions` · `…/{id}/messages` · `…/run` · `…/edit` · `…/apply` · `…/restore` · `…/finalize` · `POST /api/ai/interpret` | Research conversations: interpret, run, edit, version, explain |
| `POST /api/study/run` · `GET /api/study/session` | Event studies and session replay |
| `POST /api/backtests` · `GET /api/runs/{id}` · `GET /api/backtests/{id}/trades[.csv]` | Option backtests, results, trades |
| `GET /api/data/coverage` · `/api/data/sources` · `POST /api/data/sources/plan` · `…/ingest` · `POST /api/data/quickstart` · `GET /api/data/verify/{sym}` | Coverage matrix, free sources, size estimates, downloads, cross-checks |
| `GET/PUT /api/settings` · `POST /api/settings/llm/test` | Settings (keys are write-only) |
| `POST /api/expr/validate` · `GET /api/meta` · `GET /api/templates` | Expression validation, registries, presets |
| `POST /api/optimizations` · `POST /api/portfolios` · `POST /api/research/run` · `/api/strategies` | Optimisation, portfolios, option-quote research, saved strategies |

```bash
strata init [--fetch]            # database (+ free real data)
strata fetch [SYMBOLS…] [--options] [--years 2020-2025] [--no-intraday]
strata serve --port 8765
strata backtest config.json --save
strata import file.csv --kind option_quotes --symbol SPY
strata ingest polygon SPY 2024-01-02 2024-03-28      # paid providers: only when you run it
strata sandbox                   # SIMULATED developer symbols
```

Code strategies subclass `strata.sdk.Strategy` and use the same point-in-time context, execution,
margin and settlement models as the visual builder:
```python
from strata.sdk import Strategy, LegSpec, StrikeSel

class MyStrategy(Strategy):
    name, symbol = "RSI put spread", "SPY"
    def on_snapshot(self, ctx):
        for pos in ctx.positions():
            if ctx.position_pnl_pct(pos) >= 50:
                ctx.close(pos, "profit_target")
        if ctx.minute == 630 and not ctx.positions() and ctx.indicator("SPY", "rsi", {"period": 14}, "5m") > 50:
            sel, why, _ = ctx.select([LegSpec(id="sp", side="sell", right="P", dte=0, strike=StrikeSel(method="delta", value=0.15)),
                                      LegSpec(id="lp", side="buy", right="P", dte=0, strike=StrikeSel(method="leg_offset", value=-5, ref_leg="sp"))])
            if sel:
                ctx.open(sel, units=1)
```
Run it with `strata backtest examples/python_strategy.py --save`. The result appears in the UI
alongside builder runs. JSON `BacktestConfig` files work the same way.

## Testing & validation
```bash
cd backend && .venv/bin/python -m pytest      # 117 tests, about 20 seconds, no network
```
Coverage includes the calendar, Black-Scholes and the American approximation (against the
published Barone-Adesi-Whaley table), indicators and their causality, margin, fill models,
hand-calculated engine scenarios (settlement, assignment, splits, signal-versus-fill timing,
expired orders, exact DTE, tri-state rules), look-ahead and reproducibility, the expression
whitelist, hand-calculated event studies (including daily-bar ambiguity), the interpreter on
the example prompts, timezone conversion, the compiler and coverage assessor, grounded
explanations, every LLM adapter with a mocked transport, the model tool loop and its repair
path, free-source normalisers, and the settings API.

Acceptance checks run on real data for this build:

| | Check | Result |
|---|---|---|
| A | Clean copy of the tracked files → `setup.sh`, tests, frontend build, API round trip | passes |
| B | Free SPY download, cross-checked against an independent price table | 6,570 sessions compared; 99.97% of closes within 0.1%, largest difference 0.113% |
| C | Real SPY end-of-day chains 2020–2025 (13.7M rows) → backtest; every fill compared with the original archive rows | 130 of 130 fills identical |
| D | "How often does SPY rise 1% from the open after falling 3% or more the previous trading day?" | market study, 105 qualifying sessions since 1993 |
| E | 0DTE call at 6:30 AM Pacific with end-of-day option data | interpreted in full, historical backtest refused with the reason, study offered, no option P&L fabricated |
| F | "Change the previous-day decline to 2% and compare the results." | new version, one-field diff, before/after table, original run kept |
| G | Symbol with no data | reported; nothing simulated in its place |
| H | Look-ahead | perturbation and injected-bug tests |
| I | Reproducibility | identical fingerprint on re-run; cached by config + dataset + engine version |

## Project layout
```
backend/strata/
  ai/          draft (schema, compiler, coverage), nl_rules (offline interpreter), llm (providers),
               explain (grounded text), agent (sessions, tools, versions)
  core/        calendar, Black-Scholes / American pricing, causal indicators
  data/        Parquet store, catalog, importers, modeled chains, providers/ (free sources + paid adapters)
  engine/      models, market (point-in-time data), expr (rule language), conditions, rules,
               selection, execution, margin, portfolio, engine (event loop), templates
  research/    study (event-study engine), option-quote research queries, SQL console
  analytics/ · optimize/ · portfolio/
  api/         FastAPI app, launch (job submission + caching), settings, SQLite state, jobs, results
frontend/src/  pages/ailab, pages/Explorer, pages/builder, pages/results, pages/DataPage, pages/Settings …
scripts/       setup.sh · dev.sh · init_db.sh · docker-entrypoint.sh
```

## Known limitations
- **Free option data is end-of-day.** Intraday and 0DTE option backtests need an intraday quote
  feed. The paid-provider adapters are written against each vendor's documented API and
  unit-tested with sample payloads, but have not been exercised against live endpoints here.
  The Alpaca source has not been run live either (it needs a key).
- **Free intraday underlying history is short** (about a month of 1-minute bars from Yahoo), so
  studies that need intraday timing have small samples until more history is loaded. Sessions
  with only daily bars still count for touch questions, with timing reported as unknown.
- **Modeled premiums** ignore real skew dynamics, liquidity and event pricing. They are estimates.
- **The offline interpreter** understands common phrasings, not arbitrary English. When it cannot
  read a request it says so rather than guessing. A language model widens what can be phrased.
- **Local models.** Small local models can be slow or unreliable at structured output; the lab
  falls back to the built-in interpreter when a model fails.
- **Margin** is strategy-based Reg-T, not portfolio margin. Special dividends and mergers are
  not modelled.
- **Scale.** Application state uses SQLite; single user.
