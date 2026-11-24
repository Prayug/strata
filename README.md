# Strata — options backtesting & research terminal

Strata is a local research platform for building, backtesting, optimising and stress-testing
options strategies on historical chain data. A Python event-driven engine runs the backtests,
and a React terminal-style UI sits on top: a visual strategy builder, a results dashboard,
trade replay, an optimisation lab, market research, portfolio testing and data management.

> **Synthetic demo data.** The repository ships a generator for a clearly labelled
> **SYNTHETIC** dataset (`DEMO`, `DEMOX`, `DEMOVIX`) so you can explore everything without a
> data subscription. It is simulated, not market history. Every backtest, study and chart
> computed on it carries a `SYNTHETIC` badge. Connect a provider or import your own files to
> research real markets.

---

## Contents
- [Quick start](#quick-start)
- [What's inside](#whats-inside)
- [Engine assumptions](#engine-assumptions-read-this)
- [Data](#data)
- [Optimisation & overfitting controls](#optimisation--overfitting-controls)
- [REST API & Python strategy interface](#rest-api--python-strategy-interface)
- [Testing & validation](#testing--validation)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

## Quick start

### Local (recommended for development)
Requirements: Python 3.11+ (3.12 tested), Node 20+ (22 tested), and `pnpm` or `npm`. `uv` is used if present.

```bash
scripts/setup.sh     # venv + backend, frontend deps, SQLite schema, synthetic demo data (~1 min)
scripts/dev.sh       # API on :8765, UI on :5199
```
Open http://localhost:5199. The interactive API docs are at http://localhost:8765/docs.

Manual equivalent:
```bash
cp .env.example .env
cd backend && uv venv -p 3.12 .venv && uv pip install -e '.[dev]'     # or python -m venv + pip
.venv/bin/strata init --demo            # create app.db + generate the SYNTHETIC dataset
.venv/bin/strata serve --port 8765      # API (background jobs run in worker processes)
cd ../frontend && pnpm install && pnpm dev --port 5199
```

### Docker
```bash
docker compose up --build        # http://localhost:8000 (API + built UI in one container)
```
On first start the container creates the database and generates the demo dataset into
`./data` (mounted volume). Set `STRATA_SKIP_DEMO=1` to skip the dataset.

### First steps in the UI
1. **Saved Strategies → Add examples** loads seven ready-to-run strategies: a 0DTE iron condor,
   a 30-DTE put spread, a short strangle, a covered call, a cash-settled iron fly, a calendar,
   and an ORB breakout spread.
2. Open one in the **Strategy Builder** (`g b`). The right panel resolves the legs against a real
   stored chain snapshot and shows payoff, Greeks, margin and the selected contracts.
3. Press **Run backtest** (`⌘↵`). Progress streams live, and the result opens in **Backtest
   Results**. Click any trade to open **Trade Replay**.
4. Use **More → Optimise these parameters** to launch a study with hold-out or walk-forward validation.

Keyboard: `⌘K` command palette · `g` + key to switch page (o, b, r, t, l, m, p, s, d) · `⌘S` save · `⌘↵` run ·
`⌘\` collapse the sidebar · replay: `space` play/pause, `←/→` step, `Home/End`.

## What's inside

| Area | Highlights |
|---|---|
| **Strategy Builder** | Legs (long/short calls & puts, stock legs) with independent side, type, ratio, DTE (incl. 0DTE; nearest / at-least / at-most), and strike by delta, premium, % OTM, ATM ± points/strikes/%, fixed, or offset from another leg. Per-leg IV, spread, bid, volume and OI filters. 23 templates (verticals, condors, flies, straddles/strangles, calendars, diagonals, covered calls, CSPs, butterflies, BWB, ratios). Drag-to-reorder, duplicate and remove legs and conditions. Live payoff (expiration + T+0), Greeks, BPR (Reg-T & cash), POP estimate. |
| **Rule engine** | Nested AND/OR/NOT trees. Entry: times, windows, sessions, weekday, underlying move, opening-range breakouts, previous-day levels, gaps, intraday range, RSI/MACD/VWAP/EMA/SMA/ATR/Bollinger/realised vol on 1m–1d, comparisons and crossovers, VIX level/change, ATM IV, IV rank/percentile, trend and volatility regime, scheduled events. Exits: % or $ targets/stops, trailing stops, time exits, DTE exits, max hold, underlying thresholds and strike breaches, Greek limits, per-leg or per-group rules, custom rule trees, and rolling. Also max entries per day, cooldowns, overlap limits and re-entry policy. |
| **Engine** | Chronological and event-driven. Bid/ask-aware fills (conservative by default, midpoint, or custom), slippage, commissions and fees, multipliers, Reg-T/cash buying power, sizing (fixed / % risk / % BP), liquidity caps and partial fills, stale-quote rejection, missing-quote handling, cash and physical settlement, early assignment (extrinsic or dividend), split adjustments, margin calls, portfolio limits, mark-to-market accounting. Deterministic and fingerprinted. |
| **Results** | 30+ metrics (Sharpe, Sortino, Calmar, CVaR, expectancy ± s.e., profit factor, streaks, BP and capital utilisation, exposure-adjusted return, costs…), equity vs buy-and-hold vs cash, underwater, rolling Sharpe, monthly heatmap, return distribution, MFE/MAE, breakdowns by time, weekday, DTE, delta, VIX, regime and exit reason, portfolio Greeks, trade table, Monte Carlo (trade and block bootstrap), and diagnostics explaining every skipped signal. `$ / % / R` toggle. |
| **Trade Replay** | Candles with entry/exit markers and strike lines, position P&L and per-leg premiums, play/step/scrub, bid/ask/fill per leg at entry, at the cursor and at exit, Greeks, why the engine entered (each condition with its value, plus contract-selection reasoning) and exited, the event log, and the original chain at entry. |
| **Optimisation Lab** | Grid, random, and Bayesian (GP expected improvement), run in parallel worker processes. Hold-out or walk-forward (rolling/anchored) validation. Heatmaps, Pareto frontier, sensitivity, stability scores, deflated Sharpe ratio, PBO via CSCV, OOS degradation, and cumulative trial counts per strategy. |
| **Market Research** | Move-after-open frequencies (Wilson CIs), 0DTE premium decay by VIX regime, IV vs realised volatility, VIX-regime behaviour, event-day premium and IV crush, custom conditional distributions (bootstrap CIs, Welch/KS tests), an entry-time sensitivity sweep, and a read-only DuckDB SQL console over the data lake. |
| **Portfolio** | Strategies run in one engine loop sharing cash, buying power and position limits, compared with the naive sum of standalone backtests, buy-and-hold and cash. Correlation matrix, contribution, simultaneous positions, and entries skipped for capital. |
| **Data** | ThetaData, Polygon/Massive, Databento and ORATS adapters. CSV/Parquet import with validation. Per-session Parquet store, coverage calendar, gap reports, chain viewer, storage use, dataset versions. |

## Engine assumptions (read this)

- **Clock.** Decisions happen only at option-chain snapshot times (15 minutes in the demo data;
  whatever your data provides). Underlying 1-minute bars are used for indicators and price
  conditions.
- **No look-ahead.**
  - The price at time `t` is the close of the last bar that *ended* at or before `t`.
  - Multi-minute bars count as complete only after their last minute.
  - Daily indicators use completed sessions only.
  - IV rank uses prior sessions only.
  - The chain is the latest snapshot at or before `t`.

  The test suite perturbs all data after a cut point and checks that earlier decisions are
  bit-identical. It also injects three deliberate look-ahead bugs and verifies each one is caught.
- **Fills.**
  - *Conservative* (default): buy at the ask, sell at the bid.
  - *Midpoint*: fill at the mid.
  - *Custom*: mid ± a fraction of the half-spread.

  Every model then adds configurable slippage. Orders fill at the quotes of the snapshot the
  decision was made on (no latency model, so use slippage or the conservative model for
  that). A position is never evaluated for exit on its entry snapshot.
- **Exits** are evaluated at each snapshot on observable quotes. By default the P&L used is the
  cost to close under the fill model. No intrabar fill is ever assumed:
  - If a loss-side and a profit-side rule fire together, the loss-side reason is recorded and
    the trade is flagged `ambiguous_exit`.
  - If the 1-minute underlying path between snapshots could have touched both the stop and the
    target (delta–gamma estimate), the trade is flagged `intrabar_ambiguous`.
- **Missing data is never filled with model prices.**
  - Stale quotes (older than `max_quote_age_seconds`) are rejected for entries.
  - Missing or stale quotes defer exits to the next fresh snapshot; marks carry the last
    observed quote.
  - Contracts at or after their expiration close are valued at intrinsic and settled, never traded.
  - Black-Scholes is used only for the builder's payoff curves (labelled estimates) and to
    compute IV/Greeks from *observed* mids when a provider omits them.
- **Settlement.**
  - Cash-settled contracts (index) settle at the PM close, or at the open for AM-settled
    monthlies.
  - Physically settled contracts deliver stock, which is sold at the next session open with
    stock slippage (configurable).
  - Early assignment of short American options happens when extrinsic value falls below a
    threshold, or before an ex-dividend date.
- **Margin.**
  - Defined-risk structures hold their worst expiration value, with longer-dated long legs
    valued at intrinsic (conservative).
  - Uncovered shorts use the CBOE/Reg-T naked formula; strangles use the greater side plus the
    other side's premium.
  - The cash model secures puts and rejects naked calls.
  - BPR = requirement − premium received. Margin deficits liquidate positions.
- **Calendar.** NYSE holidays (rule-based, plus special closures), early closes, and DST via
  `zoneinfo`. All storage is UTC; all rules use New York time.
- **Reproducibility.** A run is keyed by `sha256(config) + dataset manifest version + events
  version + engine version`. Identical requests return the cached result, and the engine has
  no randomness.

## Data

Layout under `STRATA_DATA_DIR/market` (columnar, one Parquet file per option session):
```
underlying/<SYM>/<YYYY>.parquet        1-minute bars (bar start, UTC)
options/<SYM>/<YYYY>/<YYYY-MM-DD>.parquet   quote snapshots: ts, expiration, strike, right, bid, ask,
                                       sizes, volume, OI, iv, greeks, underlying_price, quote_ts
daily/<SYM>.parquet · derived/<SYM>_iv.parquet · events/events.parquet · corporate_actions/<SYM>.parquet
manifests/<SYM>.json                   per-file sha256 → dataset version
```

**Providers** (keys go in `.env`; downloads are cached under `data/cache/<provider>`, and sessions
already stored are skipped):

| Provider | Setup | What is ingested |
|---|---|---|
| ThetaData | Run the Theta Terminal; `THETADATA_BASE_URL` | `/v2/hist/stock/ohlc` 1-min bars, `/v2/bulk_hist/option/quote` + `greeks` at your interval, open interest |
| Polygon / Massive | `POLYGON_API_KEY` (Options Advanced for quotes) | 1-min aggregates; contracts as-of each day (incl. expired); last NBBO quote at each snapshot (quote age kept) |
| Databento | `DATABENTO_API_KEY`, `pip install 'strata[databento]'` | `OPRA.PILLAR` `cbbo-1m` + `definition`; `DBEQ.BASIC` `ohlcv-1m` |
| ORATS | `ORATS_API_KEY` | `hist/strikes` end-of-day snapshot with ORATS IV/Greeks; `hist/dailies` bars |

Ingest from **Data Management → Providers & ingestion**, or from the CLI:
`strata ingest polygon SPY 2024-01-02 2024-03-28 --interval 15`.

**Imports.** CSV or Parquet, from the UI or with `strata import file.csv --kind option_quotes --symbol SPY`.

| Kind | Required columns |
|---|---|
| `option_quotes` | `ts, expiration, strike, right, bid, ask` (optional: sizes, volume, OI, IV, Greeks, `underlying_price`, `quote_ts`) |
| `underlying_minutes` | `ts, open, high, low, close` (optional: `volume`) |
| `underlying_daily` | `date, open, high, low, close` |
| `events` | `date, event_type` |
| `corporate_actions` | `ex_date, action, ratio, amount` |

Naive timestamps are read as New York time. Invalid rows (crossed quotes, negative prices) are
rejected and counted, never repaired.

**Coverage.** The Data page shows, per symbol:
- the date ranges and resolution
- missing sessions and partial sessions
- snapshot gaps, on a calendar heatmap
- quote rows, storage use and the dataset version

It also has a chain viewer for any stored snapshot.

**Synthetic generator** (`strata demo-data --start … --end … --seed …`):
- Square-root stochastic variance with a leverage effect.
- Student-t intraday shocks with a U-shaped intraday volatility profile.
- Overnight gaps and jumps.
- Synthetic CPI/NFP/FOMC events with event variance and IV crush.
- A skewed IV surface with a variance risk premium.
- Quotes with realistic spreads, sizes, volume and OI.

Deliberate data defects are injected (stale and missing quotes, outages, one missing session)
so the engine's handling of bad data is exercised.

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

## REST API & Python strategy interface
FastAPI serves OpenAPI docs at `/docs`. Long-running work returns a run id. Progress streams over
server-sent events at `GET /api/jobs/{id}/stream`, and runs can be cancelled with
`POST /api/jobs/{id}/cancel`.

| Endpoint | Purpose |
|---|---|
| `POST /api/backtests` · `GET /api/runs/{id}` · `GET /api/backtests/{id}/trades[.csv]` · `GET /api/backtests/{id}/trades/{tid}` | Backtests, results, trades, trade replay |
| `POST /api/optimizations` · `POST /api/optimizations/{id}/apply` | Optimisation studies |
| `POST /api/portfolios` | Combined portfolio backtests |
| `POST /api/research/run` · `POST /api/research/sql` · `POST /api/research/entry-timing` | Market research |
| `GET /api/data/catalog` · `/api/data/chain/{sym}` · `/api/data/candles/{sym}` · `POST /api/data/import` · `/api/data/ingest` · `/api/data/demo` | Data management |
| `/api/strategies` (CRUD, `/preview`, `/validate`, `/optimizable`, `/examples`) | Strategies |

Code strategies subclass `strata.sdk.Strategy` and use the same point-in-time context, execution,
margin and settlement models as the visual builder:
```python
from strata.sdk import Strategy, LegSpec, StrikeSel

class MyStrategy(Strategy):
    name, symbol = "RSI put spread", "DEMO"
    def on_snapshot(self, ctx):
        for pos in ctx.positions():
            if ctx.position_pnl_pct(pos) >= 50:
                ctx.close(pos, "profit_target")
        if ctx.minute == 630 and not ctx.positions() and ctx.indicator("DEMO", "rsi", {"period": 14}, "5m") > 50:
            sel, why, _ = ctx.select([LegSpec(id="sp", side="sell", right="P", dte=0, strike=StrikeSel(method="delta", value=0.15)),
                                      LegSpec(id="lp", side="buy", right="P", dte=0, strike=StrikeSel(method="leg_offset", value=-5, ref_leg="sp"))])
            if sel:
                ctx.open(sel, units=1)
```
Run it with `strata backtest examples/python_strategy.py --save`. The result appears in the UI
alongside builder runs. JSON `BacktestConfig` files work the same way.

## Testing & validation
```bash
cd backend && .venv/bin/python -m pytest
```
56 tests (about 10 seconds) cover:
- **Calendar.** NYSE holidays, observed rules, special closures, early closes, DST.
- **Pricing.** Black-Scholes against textbook values, put-call parity, IV round-trips.
- **Indicators.** Hand-checked values, and causality checks: altering future bars must not
  change past values.
- **Margin.** Hand-calculated Reg-T and cash requirements for spreads, condors, flies, naked
  shorts, strangles, ratios, covered calls, calendars and diagonals.
- **Fill models, commissions and liquidity caps.**
- **Engine scenarios with hand-calculated results.**
  - Hold to expiry, profit target, midpoint fills, and ITM cash settlement.
  - Stop/target ambiguity.
  - Missing quotes deferring an exit.
  - Stale-quote rejection, insufficient buying power, partial fills.
  - Physical assignment, early assignment, splits.
  - Re-entry rules, explanations, and the accounting identity.
  - Never trading an expiring contract at the close.
- **Look-ahead and reproducibility.**
  - Data perturbed after cut points, with three injected bugs that must be caught.
  - Bit-identical reruns.
  - Entry times across DST and holidays.
- **Providers, research and optimisation.** Provider normalisers against sample vendor payloads,
  CSV import validation, research statistics, Monte Carlo seeding, DSR/PBO behaviour, and an
  end-to-end optimisation.

## Project layout
```
backend/strata/
  core/        calendar, Black-Scholes/IV (Numba), causal indicators (Numba)
  data/        Parquet store, catalog, importers, synthetic generator, derived IV, providers/
  engine/      models (configs), market (point-in-time data), selection, execution, margin,
               portfolio (accounting), conditions (rule engine), rules (builder strategies),
               engine (event loop), templates
  analytics/   metrics, breakdowns, Monte Carlo
  optimize/    search (grid/random/GP-EI), walk-forward, overfitting diagnostics
  research/    research queries + SQL console
  portfolio/   combined portfolio analysis
  api/         FastAPI app, SQLite state, background jobs (SSE), result storage, builder preview
  sdk.py · cli.py
frontend/src/  React + TypeScript + Tailwind; pages/ (one per workspace), components/ (ui, charts)
scripts/       setup.sh · dev.sh · init_db.sh · schema.sql · docker-entrypoint.sh
```

## Known limitations
- **Provider adapters** are written against each vendor's documented API, and their normalisers
  are unit-tested with sample payloads. They have **not** been exercised against live endpoints
  in this build. Expect small adjustments for your subscription tier or API version
  (ThetaData v3, for example, changed paths).
- **Synthetic pricing.**
  - The demo market prices American options with Black-Scholes and no dividends, so the early
    exercise premium of puts is ignored.
  - Theta is per trading day.
  - Results on it say nothing about real markets.
- **Exit timing.** Exits fill at snapshots. With coarse snapshots (15 minutes, or end-of-day for
  ORATS) stops can gap through their levels; that is intended, but use finer data for 0DTE
  research.
- **Margin.**
  - Margin is strategy-based Reg-T, not portfolio margin.
  - Cross-position offsets between different positions are not recognised.
  - Hard-to-borrow costs and interest on short stock are not modelled.
- **Corporate actions.** Adjustments cover splits (integer and fractional ratios) and dividend
  early-exercise checks. Special dividends and mergers are not modelled.
- **Scale.** Application state uses SQLite. For multi-user deployments, swap in PostgreSQL
  (the data-access layer is isolated in `api/db.py`).
