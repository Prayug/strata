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
- [What V3 adds](#what-v3-adds)
- [Four research modes](#four-research-modes)
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
1. The app opens on the **AI Lab**. Ask *"How often does SPY rise 1% after a 3% down day?"* The
   lab shows how it read the question, checks coverage, runs it, and opens a report: the
   frequency with its confidence interval, the baseline on all sessions, a calendar of every
   qualifying session, and the robustness checks.
2. Follow up in the same notebook: *"Compare that with 2% and 4% down days"*, *"What time of
   day does it usually hit the target?"*, *"Show every qualifying historical session"*.
3. *"Test ATM 0DTE calls on those days using historical option trade prices."* The lab finds
   the qualifying sessions on underlying data first, downloads option trade bars only for those
   sessions and the contracts they could select, and reports how many signals could actually be
   executed. Then *"Compare the results with 7DTE calls"*, *"Include commissions and conservative
   slippage"*, *"Show me every skipped signal and why"*, *"Export the complete research report"*.
4. **Pin** two results and open **Compare**; click any day in the calendar to replay it with the
   option contract's own prints underneath; press `?` for the keyboard shortcuts.
5. **Market Explorer**, **Strategy Builder**, **Backtests** and **Data** remain for working
   by hand. Everything the lab runs appears there too.

## What V3 adds

| Area | Change |
|---|---|
| Data ingestion | Alpaca requests follow `next_page_token` to the last page, de-duplicate, validate every record and resume. A session is marked complete only when its request was read to the end; a partial session is stored as partial and never used as if complete. |
| Point-in-time contracts | Contracts are discovered as of the session being traded: active **and** since-expired definitions, expirations on or after that session, strikes around the price at the signal. Original OCC identifiers, expirations, multipliers and adjusted (non-standard) contracts are preserved. |
| Selective acquisition | Two stages: signals are found on cheap underlying data, then option data is fetched only for the qualifying sessions and the contracts those signals could select. Unusually large requests show size, time and cost ($0.00 for free sources) and wait for approval. |
| Trade-priced backtests | A fourth mode executes on observed option **trade prints** (free: Yahoo for the last weeks, Alpaca since Feb 2024). A print is used only while it is recent; otherwise the order is deferred, then rejected. No spread is invented, and trade bars are never read as quotes. |
| Coverage | Session-level and contract-level coverage are reported separately from results: requested vs. usable sessions, qualifying vs. executable signals, and every skipped signal with its reason. A stated period that cannot be covered stops and asks instead of silently shrinking. |
| Storage | `provider / symbol / data type / observation / resolution / version`, per-file provenance, append-only manifests. Older data directories are migrated automatically (`strata migrate --dry-run` shows the plan). |
| Research agent | Multi-step investigations, variant comparisons, exploratory discovery with an untouched final test period, out-of-sample and robustness checks, all computed by the deterministic engine. |
| Stateful rules | Ordered event sequences ("drops 0.5%, then recovers half"), scale-ins tied to the first entry, cross-symbol conditions, per-position state and entry-dependent exits are first-class engine constructs, not approximations. |
| Reports | Overview, Calendar, Trades/Sessions, Statistics, Robustness, Methodology and Raw data; every headline number opens its observations; exports to HTML, Markdown, JSON and CSV. |
| Interface | A new research workspace: composer with quick/review modes, live pipeline stages, inline editable interpretation with a real-day preview, calendar and synchronized replay, pinned comparisons, searchable notebooks and runs, keyboard navigation. |
| Reliability and security | Structured outputs and strict tool schemas, retries with backoff, timeouts, cancellation, token usage, model discovery, an explicit fallback policy. Credentials live in the OS keychain (or a 0600 file), never in the database, logs, exports or prompts. Non-local requests need a token. |
| Validation | `strata validate` proves the chain on real observations from an empty directory; opt-in live tests per provider; browser end-to-end tests; CI. |

## Four research modes

Every result is labelled with the mode that produced it, on every screen and in every export.

| Mode | When it is used | What the numbers are |
|---|---|---|
| **Options backtest on bid/ask quotes** | Observed option quotes cover the request. | Fills at stored bid/ask. |
| **Options backtest on trade prints** | Intraday option trades are stored or can be fetched free. | Fills at the price of the most recent observed trade, only while it is at most `max_trade_age_seconds` old (default 300). There is no bid/ask in this data and none is created; an optional percentage slippage is applied and disclosed. Reports show the share of signals executed, print ages, stale marks and cost sensitivity. |
| **Market study** | The question is about the underlying, or you choose it. | How the underlying behaved: frequency with a Wilson 95% interval, baseline on all sessions, MFE/MAE, time to target, by year and regime. Never presented as option P&L. |
| **Modeled options backtest** | Only when you opt in. | Real underlying prices with theoretical premiums. Labelled *theoretical* everywhere; never chosen automatically and never mixed with observed prices. |

`auto` (the default for option strategies) picks whichever kind of *observed* price covers more of
the qualifying sessions and says why. If neither covers them, the run stops with choices:
connect a free Alpaca account, accept the covered subset, switch to the market study, or opt in
to modeled premiums. Missing data is never read as "the condition was not met".

## AI Strategy Lab

```
your words ─► interpreter ─► Draft ─► questions / confirmation ─► research pipeline
              (rules or LLM)  (structured, validated)              │
                                                                   ▼
   1 Interpreting strategy      2 Checking historical coverage     3 Identifying qualifying sessions
   4 Downloading missing data   5 Validating contracts             6 Running simulation
   7 Generating research report ─► report · calendar · replay · comparison · exports
```

- **The interpreter only fills in a `Draft`.** The deterministic engine calculates every
  statistic. The model can choose from a fixed list of tools with strict argument schemas; it
  cannot run code or SQL, and any number it writes that is not in the stored result is discarded.
- **Works with no API key.** The built-in interpreter reads entries, exits, clock times,
  previous-day moves, gaps, VWAP, opening ranges, streaks, expirations in plain words
  ("expiring in 7 days", "a month out"), dated periods, ordered sequences, scale-ins,
  cross-symbol conditions, costs, comparisons and follow-ups. A complex request is never handed
  silently to a weaker interpreter: the notebook shows who read each request, and the fallback
  policy is a setting.
- **Quick research or review first.** Quick mode runs immediately and lists every assumption.
  Review mode shows the interpretation and waits. Either way an ambiguous detail is asked once,
  in context: *"You said target 1%. Do you mean a 1% move in SPY or a 1% gain on the option?"*
- **Unsupported parts are named.** If part of a request cannot be represented, the exact words
  are quoted, nothing is substituted, and nothing runs until you confirm or rephrase.
- **Editable interpretation.** Plain English beside the structured configuration, inline
  controls for every field, and *Check it on a real day*: each rule evaluated tick by tick on an
  actual stored session before anything runs.
- **Stateful rules.** *"Wait until SPY drops 0.5% from the open, then buy calls when it recovers
  half of that decline"* becomes two ordered stages (`from_open_pct <= -0.5`, then
  `price >= low_since("drop") + 0.5 * (open - low_since("drop"))`). Sequences are evaluated only
  on 5-minute or finer bars; coarser sessions are excluded and reported.
- **Investigations and discovery.** *"Find out whether SPY tends to reverse after large down
  days"* runs a planned set of variants with p-values adjusted for the number of comparisons.
  *"Investigate mean reversion after unusually large SPY declines"* explores a parameter grid on
  the first 60% of history, validates on the next 20%, and looks at the final 20% once, only
  for candidates. Every configuration tested is kept, including the ones that showed nothing.
- **Reports.** Overview, Calendar, Trades, Statistics, Robustness, Methodology and Raw data.
  Headline numbers open the observations behind them; the calendar colours sessions by outcome
  (qualified, traded, skipped, missing data) and replays a day with the traded contract's own
  prints under the underlying chart.
- **Comparison workspace.** Pin results to compare them on one time axis. Differences in
  pricing basis, period, costs and symbol are listed above the numbers, a shared window can be
  re-run in one click, and theoretical results are never shown beside observed ones without a
  prominent warning.
- **Credentials.** Stored in the macOS Keychain, the Linux Secret Service, or an owner-only
  file; never in the database, logs, exports or model prompts. The browser sees only the last
  four characters.

## Market data

### Free sources (implemented and tested against the live services)

| Source | Account | What it provides | Licence |
|---|---|---|---|
| Yahoo Finance | none | Daily bars for the full listing history with dividends and splits; intraday bars: 60-minute (~2 years), 5-minute (~60 days), 1-minute (~30 days, rolling) | Yahoo terms (personal research) |
| Cboe | none | Official daily VIX history since 1990 | Cboe published data |
| EOD options archive | none | Full end-of-day chains for SPY and IWM 2008–2025 and QQQ 2011–2025: bid/ask, sizes, volume, open interest, IV, Greeks | MIT (P. Dubach dataset mirror) |
| DoltHub `post-no-preference/options` | none | End-of-day chains since 2019 for about 2,000 symbols (four expirations, ~27 strikes; no volume/OI) | CC BY-SA 4.0 |
| Yahoo option trades | none | 1-minute option **trade** bars for contracts still listed or expired within about a day (rolling window: capture 0DTE history as it happens) | Yahoo terms (personal research) |
| Alpaca | free key | 1-minute stock bars since 2016; 1-minute option **trade** bars since Feb 2024, including expired contracts (no historical option quotes on the free plan) | Alpaca data agreement |

Ingestion estimates size before large downloads, resumes, retries with backoff, skips what is
already stored, rejects crossed or negative quotes, nulls implausible vendor IV/Greeks so they
are recomputed from the observed quotes, and records provider, resolution, licence and a
dataset version for every file. **Data → Get free data** does all of this from the UI.

**What free data cannot do:** no free source offers *intraday option bid/ask quotes*. Intraday
and 0DTE option backtests therefore run on **trade prints** (see the modes above), with the
limits that implies, or on a paid quote feed (ThetaData, Databento, Polygon/Massive adapters are
included and are never called automatically) or your own files. **Settings → Connect a free
Alpaca account** walks through the three steps and tests both APIs.

### Storage
```
market/datasets/<provider>/<SYMBOL>/<underlying|options>/<trade|quote|index>/<resolution>/v<N>/
    dataset.json            provider, licence, adjustment, limitations
    data/…                  daily.parquet · <YYYY>.parquet · <YYYY>/<date>.parquet
    sessions.json           per-session status: complete | partial (never used as complete)
    index.parquet           option trade bars: one row per fully fetched (session, contract)
    contracts.parquet       contract definitions as the provider gave them
    provenance.json         per-file source request and fetch time
    manifests/<seq>-<hash>.json · HEAD     append-only content versions
market/reference/events · corporate_actions
```
Datasets from different providers are never merged, so symbol-level metadata cannot leak from
one provider's observations to another's, and a trade bar cannot be read as a quote. The
content version of every dataset a run read is part of its cache key.

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

**Stateful rules (V3).** An entry can be an ordered *sequence* of named stages; each stage is
looked for only after the one before it has happened, and later stages (and exits) can read
what earlier ones recorded:

```
sequence:  drop     from_open_pct <= -0.5
           recover  price >= low_since("drop") + 0.5 * (open - low_since("drop"))
add:       price <= entry.underlying_price * (1 - 0.005)        # scale in, tied to the first lot
exit:      price >= vwap                                         # closes every lot of the position
also:      QQQ.high <= QQQ.prev_high        occurred("breakout")        minutes_since("drop") <= 30
           stage_price("drop")   high_since("rally")   ref("level")   position.adds   group.pnl_pct
```
A stage cannot complete on the tick that armed it, sequences reset every session, and a
sequence is only evaluated on 5-minute or finer bars.

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
| `/api/ai/sessions` · `…/{id}/messages` · `…/answer` · `…/confirm` · `…/decide` · `…/edit` · `…/restore` · `…/pin` · `…/finalize` · `POST /api/ai/interpret` · `POST /api/ai/preview` · `GET /api/ai/search` | Research notebooks: interpret, ask, confirm, decide on coverage, edit, version, preview on a real day, search |
| `GET /api/runs/{id}/pipeline` · `…/calendar` · `…/observations` · `…/robustness` · `…/report?format=json|md|html` · `…/export?what=…` · `…/day?day=` | Stage progress, calendar, the observations behind each statistic, reports and exports, one-day replay |
| `GET /api/compare?ids=` · `POST /api/compare/rerun` · `GET /api/runs/search` | Comparison workspace and run search |
| `GET /api/data/datasets` · `POST /api/data/alpaca/test` · `GET /api/settings/llm/models` · `…/llm/usage` | Datasets with provenance, guided Alpaca check, model discovery, token usage |
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
strata validate [--keep]         # prove the whole chain on real observations (see below)
strata migrate [--dry-run]       # move an older data directory to the current layout
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
cd backend && .venv/bin/python -m pytest          # 213 tests, about 25 seconds, no network
cd backend && .venv/bin/strata validate           # ten steps on real observations, ~10 seconds
cd backend && STRATA_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -rs   # opt-in, per provider
cd frontend && pnpm build && pnpm e2e             # browser tests (PW_CHANNEL=chrome uses your Chrome)
```
The unit suite runs with downloads off and never touches the OS keychain. On top of the V2
coverage it adds regression tests for truncated and paginated responses, partial sessions,
missing contracts, point-in-time contract discovery (future prices cannot change which
contracts are selected), trade-print execution (stale prints, deferred and rejected orders, no
invented spread), ordered sequences and scale-ins, sequences on coarse bars, plain-language
expirations and periods, the clarifying-question and confirmation flows, coverage decisions,
investigation and discovery statistics, provider retries, cancellation and usage tracking,
credential storage, and reproducibility of every kind of run.

**`strata validate`** starts from an empty data directory and uses free, keyless sources only:

| # | Step | Result on this build |
|---|---|---|
| 1 | The data directory starts empty | no datasets, runs or credentials |
| 2 | Download a small actual SPY dataset | 8,514 daily bars since 1993 and 7,800 one-minute bars (0.8 MB) |
| 3 | Verify the stored candles | 6,570 closes compared with an independent source, 99.97% within 0.1%; 20 minute sessions agree with their daily bar |
| 4 | Download legitimate option observations | 2,609 one-minute trade bars for 18 contracts, only for the sessions and strikes the strategy needs |
| 5 | Run a hand-written strategy | 3 trades, every fill an observed trade print |
| 6 | Run the same strategy written as a sentence | read by the built-in interpreter, 3 trades |
| 7 | Both produce identical results | same contracts, times, prices, commissions and P&L; configurations equal in every setting that affects behaviour |
| 8 | Premiums match the source records | 6 fill prices recomputed from stored bars without the engine and matched against freshly downloaded contract histories |
| 9 | Interrupted downloads are never marked complete | connection cut after 2 requests: 2 of 6 contracts recorded, 4 reported failed; the resume fetched only the missing 4, no duplicates |
| 10 | No simulated data anywhere | every dataset has a licence and manifest; every run priced on observed trade prints |

The figures in steps 2 to 9 depend on the day it is run. A step that cannot reach a provider
fails; it is never skipped.

**Live tests** (`STRATA_LIVE=1`): Yahoo bars, Cboe VIX, the independent price archive, Yahoo
option trades and the validation command need only network access. Alpaca and each language
model run when their key is in the environment and otherwise report themselves as skipped.

**Browser tests** drive the built app in a real browser against a small seeded fixture store
with known answers: asking a question and opening the observations behind a number, the
calendar and day replay, trade-priced follow-ups, skipped signals, exports, review mode with a
clarifying question and the real-day preview, an unsupported clause, pinned comparison, and
write-only credentials.

**CI** (`.github/workflows/ci.yml`): backend tests, reproducibility and look-ahead tests,
frontend type check and build, browser tests; the live job runs weekly and on request.

Acceptance prompts, run through the browser on real SPY history for this build:

| Prompt | Outcome |
|---|---|
| How often does SPY rise 1% after a 3% down day? | market study: 68 of 105 sessions since 1993 (64.8%, interval 55.3% to 73.2%) against 17.3% on all sessions |
| Compare that with 2% and 4% down days. | three variants on one chart with intervals and baseline; warns about picking the best of several |
| What time of day does SPY usually hit the target? | answered from the sessions that have intraday bars, stating how few those are |
| Show every qualifying historical session. | the 105 sessions, sortable, each replayable |
| Test ATM 0DTE calls on those days using historical option trade prices. | 1 of 105 qualifying sessions has free option prints; that trade is shown and the other 104 are listed as not testable, not as losses |
| Compare the results with 7DTE calls. | new version, before/after, same exit kept and disclosed |
| Include commissions and conservative slippage. | new version with the cost assumptions listed |
| Show me every skipped signal and why it was skipped. | 104 signals, each with its reason |
| Test whether the results hold up out of sample. | split run; reports that only one side has option data, so there is nothing to compare |
| Export the complete research report. | HTML, Markdown, JSON and CSV |

## Project layout
```
backend/strata/
  ai/          draft (schema, compiler), nl_rules + nl_stateful (built-in interpreter), research
               (investigations, follow-ups), llm (providers), explain (grounded text), agent
  core/        calendar, Black-Scholes / American pricing, causal indicators
  data/        store (datasets, manifests), migrate, catalog, planner (requirements, coverage,
               selective acquisition), importers, providers/ (free sources + paid adapters)
  engine/      models, market (point-in-time data, trade-print chains), expr (rule language),
               sequence (ordered stages), scan (signals only), rules, selection, execution,
               margin, portfolio, engine (event loop)
  research/    pipeline (the seven stages), study, stats, discovery, explain_day, views
               (calendar, observations, comparison), report (HTML / Markdown / CSV)
  analytics/ · optimize/ · portfolio/
  api/         FastAPI app, research_api, launch (jobs + caching), settings (credential stores), db
  validate.py  the `strata validate` command
backend/tests/ unit suite, test_live.py (opt-in), test_reproducibility.py, e2e_seed.py
frontend/src/  pages/lab (research workspace), pages/Explorer, pages/builder, pages/results,
               pages/DataPage, pages/Settings …
frontend/e2e/  browser tests            .github/workflows/ci.yml
scripts/       setup.sh · dev.sh · init_db.sh · docker-entrypoint.sh
```

## Known limitations
- **Free intraday option history is shallow.** Yahoo serves option trade bars only for contracts
  still listed or expired within about a day, so 0DTE history has to be captured as it happens;
  Alpaca's free plan reaches back to February 2024. Before that, intraday option backtests need
  a paid provider or imported files. A request that reaches further back is told so.
- **Trade prints are not quotes.** A trade-priced backtest cannot know the spread that was
  available. It refuses stale prints and reports executability, but it is still an
  approximation of what could have been filled.
- **Not exercised against live endpoints here:** Alpaca (no key was available: its pagination,
  contract discovery and resume logic are covered with a mocked transport and by an opt-in live
  test), the paid-provider adapters, the language-model providers (mocked transport plus opt-in
  live tests), and writing to the real macOS Keychain (the command construction is unit-tested).
- **Language models.** Cancellation is checked between requests, not in the middle of one.
  Gemini uses JSON mode without a schema. Small local models can be unreliable at structured
  output; what happens then is the fallback setting.
- **Discovery explores the underlying.** It searches conditions and targets on price history;
  it does not search over option structures.
- **Free intraday underlying history is short** (about a month of 1-minute bars from Yahoo), so
  intraday rules have small samples until more history is loaded. Sessions without fine enough
  bars are excluded and counted, not treated as "no signal".
- **Modeled premiums** ignore real skew dynamics, liquidity and event pricing. They are estimates.
- **The built-in interpreter** understands common phrasings, not arbitrary English. When it
  cannot read part of a request it quotes that part rather than guessing.
- **Margin** is strategy-based Reg-T, not portfolio margin. Special dividends and mergers are
  not modelled.
- **Scale.** Application state uses SQLite; single user.
