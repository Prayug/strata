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
- [What V5 adds](#what-v5-adds)
- [What V4 adds](#what-v4-adds)
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
The sidebar is three workflows: **Explore** (ask, patterns, knowledge base), **Build** (strategy IDE,
visual builder) and **Validate** (backtests, forward validation, experiment registry). *Simple* shows the
main path through each; *Advanced* adds the specialist tools. The calculations are the same in both.

1. The app opens on **Ask**. A first run is offered the free download and four questions the stored data can already answer. Ask *"How often does SPY rise 1% after a 3% down day?"* The
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

## What V5 adds
V5 removes practical limits rather than adding surface: research is no longer confined to seven
templates, a strategy can be read and edited as text, what has been tested is tracked across
every notebook, and the sources of historical options data are laid out with what was actually
verified. Nothing from V2-V4 was rebuilt.

| Area | What it does |
| --- | --- |
| **Research graphs** | A question becomes a typed, validated graph of seventeen primitives (data, feature, filter, combine, event, sequence, forward return, target/stop, conditional probability, matched comparison, distribution, correlation, regression, sweep, robustness, validation, options backtest). The language model, when one is configured, may only propose a graph; the same validator checks it and the deterministic engines run it. No generated code is executed. A step whose input is only known after the outcome starts is refused before anything runs. |
| **A reader that composes** | Conditions are read as subject x time span x predicate, so "the morning session sold off, QQQ outperformed SPY and VIX rose overnight" needs no template. Vague sizes ("large") are resolved from the symbol's own history and stated as assumptions; words that were not understood are listed, never dropped. The same reader now serves strategy descriptions, so a strategy can use any condition a question can. |
| **Matched comparisons** | The plain comparison next to three adjusted ones (stratification, nearest neighbours with a caliper, regression adjustment with week-clustered errors) and a balance table. The verdict is "persists", "mixed", "not distinguishable", "reverses" or "not estimable": when the methods disagree it says so. It is association on matched sessions, not causation, and it uses only values known at the decision time. |
| **Pattern explorer** | Pick an event (fourteen built in, or a custom rule): every occurrence, the fan of price paths after it, forward returns at six horizons with Holm-adjusted tests, regimes, and the years it clusters in. Each look is recorded in the registry. It describes the underlying only; a strategy goes through a tested finding. |
| **Analyst memory and knowledge base** | Every finished analysis leaves findings with an evidence grade (confirmed, holds but not independent, supported, suggestive, holds on seen data, not confirmed, not supported, insufficient) and the reasons for it. Follow-ups work on what is stored: "apply the same idea to QQQ", "which of our findings still hold after 2020?", "combine the opening-range condition with the volatility filter", "which of these results are actually independent?". With nothing stored it says so. |
| **Experiment registry** | One record across all notebooks and tools: fingerprints, exact repeats, closely related configurations (same symbol, overlapping inputs and outcome), phases (exploration, development, validation, confirmation, forward), and every held-back final period with who else evaluated those dates. A final period examined by related work elsewhere is never called an independent confirmation, and option positions priced on it before the pattern was scored there count as a look. |
| **Strategy IDE** | The strategy as text (defaults left out, exact round trip), with syntax colours, completion and inline documentation for the rule language, error markers on the line they belong to, an outline, a form for the selected part with how often a rule was true historically, and a debugger that replays the entry rule on a real session tick by tick with watches and a breakpoint. "Replace this fixed stop with an ATR-based stop" is applied to the selected section only; the changed lines are marked, and an instruction that cannot be expressed changes nothing and says so. |
| **More of a strategy can be expressed** | Independent entry systems (each may carry its own legs, so calls after a decline and puts after a rally), staged exits with partial closes and a tightened stop, ATR targets and stops, trailing thresholds on the underlying, sizing by expression or by ATR risk, portfolio-level conditions (`portfolio.open_positions`, `portfolio.day_pnl_pct`, ...), a delta hedge in shares, and saved signals (`signal("name")`). Each has a regression test with hand-computed numbers. |
| **Execution fidelity** | More stress cases (best achievable, impatient, auction windows, expiration-day spreads, early or never assigned) and an execution-quality score built from price evidence, coverage, freshness, liquidity, spread and exits. It never reads the P&L, and a component that cannot be measured is left out and named. |
| **ML research (optional)** | Logistic, ridge, lasso, tree, forest and gradient-boosted models scored out of sample with purged, embargoed chronological folds and a final test set at the end of history. Reported against the base rate (Brier and log-loss skill), with calibration, a time-shifted placebo, permutation importance and a leakage probe that scrambles the future. Random splits are not offered. The only bridge to a strategy is a saved walk-forward signal, priced by the ordinary backtester. `pip install -e "backend[ml]"`. |
| **Forward validation** | Freeze a finished backtest or study: the configuration is stored with a hash and the last stored session as its activation date. Only later sessions count; fewer than ten forward observations are reported as a record, not a verdict. A change is a new version with its own date. Paper only: there is no order entry anywhere in Strata. |
| **Provider marketplace** | Data -> Sources lists twenty-one sources of historical options data with advertised coverage, resolution, price and limits next to what this installation verified with a real request, coverage for SPY, QQQ, IWM and SPX, and a cost estimate for only the observations one strategy needs. Alpaca has a guided connection (credentials, entitlements, an expired contract, its trade bars, a small backtest). |
| **Collector** | Survives restarts (a cut-off cycle is detected by its heartbeat and resumed), retries failed sessions with growing delays, probes its sources first, checks what it stored, recovers a gap while the source still serves it, and leaves a note in the app. It states what is recoverable after a miss and what is not. |
| **Three workflows** | Explore, Build and Validate, with a simple and an advanced level, onboarding that says what the stored data can already answer, and an in-app notification list. |

### A real session, start to finish
These nine requests were typed into the running application on 2026-11-25, in one notebook, on
downloaded market history (SPY daily bars 1993-2026, QQQ, IWM, VIX, and SPY end-of-day option
quotes 2020-2025). The numbers are what it answered.

| Request | What came back |
| --- | --- |
| *Does SPY tend to recover after large down days, even after controlling for volatility?* | An 11-step graph. "Large" was taken as a fall of 2.25% or more (the largest 5% of declines) and said so. The next close was higher on 116 of 197 such sessions (58.9%) against 53.2% of 6,614 others, p 0.11: within chance. Against sessions matched on realised volatility and the VIX the difference is +12.2 points (95% interval +3.9 to +19.9). Graded **suggestive**. Not a candidate after correction, so the final test period (2020-02-19 onward) was left untouched. |
| *Compare that behavior with QQQ and IWM.* | Each symbol's own down days, on the one period all three have and that stops before the held-back sessions (2000-05-30 to 2020-02-18): SPY 56.4% of 163 against 54.0%, QQQ 54.5% of 347 against 54.0%, IWM 53.2% of 252 against 53.1%. None distinguishable after correction. |
| *Show the distribution of price paths following those events.* | 197 paths with percentile bands: median next-session return +0.32%, middle half -1.00% to +1.46% (other sessions: median +0.06%). |
| *Find which historical conditions are associated with a recovery.* | Eleven values scanned among the 197 sessions. No term is distinguishable from zero after correcting for the number scanned. |
| *Build a strategy from the strongest supported finding.* | It answered that no finding is graded "supported"; the strongest is "suggestive". It drafted ten option structures so the idea can be tested, priced none, and pointed out that option prices are stored only for dates inside the held-back period. |
| *Backtest that strategy using observed historical options prices.* | Ten structures on observed end-of-day bid/ask quotes; 52 of the 252 qualifying sessions have option prices and the rest are listed as not testable. At-the-money call, 7 days: 52 trades, net +$56, 44% winners. The put control: -$2,639. Iron condor: -$2,454. The price pattern does not carry over into option profit. |
| *Challenge the results with worse execution assumptions.* | With no costs the signal shows +$798; as simulated +$56: spread and fees take 93%. The result is zero when every fill is 0.1% worse, and -$762 at 1% worse. Execution-quality score 89 of 100, which grades the evidence behind the fills, not the result. |
| *Check whether I've already tested similar configurations on this holdout.* | The exact configuration: 5 times in this notebook and in 2 others. Ten closely related ones. The final period is "already seen" (ten option positions were priced on it), and 60 configurations in 5 other notebooks touched those dates, so a result on it would not be an independent confirmation. |
| *Start a frozen forward-validation experiment.* | The strategy the execution challenge was run on was frozen with its hash; activation 2026-11-25; no forward session exists yet, and it says so. |

### Historical options data: what is advertised and what was verified
Provider pages were read on 2026-11-25 and are linked from each card in the app. "Verified"
means a real request from this installation returned the data.

| Source | Cost | Advertised | Verified here |
| --- | --- | --- | --- |
| End-of-day option archive (public mirror, MIT) | free | SPY, QQQ, IWM end-of-day chains with bid/ask, greeks, 2008-2025 | **verified** |
| Yahoo Finance option trade bars | free | 1-minute trade bars for contracts still listed or just expired | **verified** (an expired contract returned nothing) |
| DoltHub options | free | recent end-of-day chains, about three expirations per date | **verified** |
| Cboe | free | VIX and index history; delayed chain snapshots | **verified** (index history) |
| Alpha Vantage | $49.99 a month | end-of-day chains from 2008 | parser verified on the demo symbol only; SPY, QQQ, IWM need a paid key |
| Alpaca | free key | 1-minute option trade bars from February 2024, expired contracts included; quotes with the $99 plan | **unverified**: no credentials. The guided connection reports "not connected" |
| Massive (formerly Polygon) | free key; paid tiers | per-contract minute aggregates, 2 to 20+ years by tier | **unverified**: no credentials |
| Databento OPRA | pay per use, $125 credit | trades, 1-minute bars and 1-minute bid/ask from 2013-04 | **unverified**: no credentials |
| ThetaData | from $40 a month | 1-minute quotes from 2020 (Value), earlier on higher tiers | **unverified**: needs their terminal |
| ORATS | subscription or one-time | 1-minute data from 2020-08 | **unverified** |
| Cboe DataShop, historicaloptiondata.com, OptionsDX, FirstRate, WRDS OptionMetrics, algoseek sample | paid or institutional | files | import only; layouts tested on files written with the documented columns |
| EODHD, Tradier, IBKR, marketdata.app, QuantConnect | varies | - | not integrated (each card says why) |

SPX index options have no free source; they enter through imported files. Strata never calls a
paid service by itself: a paid source shows a quoted cost and stops there.

### Unattended collection
Free sources drop intraday data quickly, so the archive has to be built as time passes. Nothing
is collected while the machine is asleep or off, or while neither the app nor the collector is
running. To collect without the app, on any small always-on machine:

```bash
docker compose --profile collector up -d collector     # collects after each close, restarts with the machine
docker compose ps                                      # "healthy" while the daemon reports every minute
backend/.venv/bin/strata collect --daemon              # the same without Docker (systemd, tmux, launchd)
backend/.venv/bin/strata collect --once                # from cron
```
A machine with one shared core and 1 GB of memory is enough; a cycle for three symbols is a few
hundred requests and tens of megabytes a week. What a missed session costs:

| Data | Can it be fetched later? |
| --- | --- |
| Daily bars | always |
| 60-minute bars | for about two years |
| 5-minute bars | for about two months |
| 1-minute bars | for about a month (a free Alpaca key reaches back to 2016) |
| Option trade bars (Yahoo) | only while the contract is listed, and about a day after it expires. After that it is gone from this source and is recorded as missing, never filled in |

## What V4 adds
V4 turns Strata from a tool that answers a question into one that helps decide whether an edge
is real. Nothing from V3 was rebuilt; these sit on top of it.

| Area | What it does |
| --- | --- |
| **Hypothesis Lab** | A broad question ("Investigate whether SPY tends to reverse following large down days") becomes a bounded plan stated before anything runs: hypothesis, null, baseline, data needed, every configuration (at most 60), primary measure, minimum sample, and training / validation / final-test periods. All configurations are reported, including the ones that show nothing. |
| **Research ledger** | Every configuration ever evaluated in a notebook is recorded, shown or not. Significance is corrected for all of them. The final test period has a status (untouched, used once, already seen, contaminated); tuning after its results were shown contaminates it and a new holdout is required. |
| **Dependence-aware statistics** | Calendar-cluster and block bootstraps, a circular-shift permutation test, Newey-West errors, paired comparisons on identical sessions, Holm / Benjamini-Hochberg / Benjamini-Yekutieli corrections, purged and embargoed cross-validation, uniqueness weights for overlapping holds, regime tables. Thin samples can never be graded above "insufficient". |
| **Adversarial reviewer** | "Challenge this strategy" examines sixteen ways a result can mislead and runs the experiments that settle them: prices after the decision are scrambled to detect look-ahead, each parameter is nudged, the same position is run without the entry condition, and execution is stressed. |
| **Execution uncertainty** | The same configuration re-simulated under a ladder of worse assumptions (slippage, order delay, price age, minimum volume, participation, partial and missed fills, multi-leg legging, bid/ask instead of prints). Where only prints exist the spread is reported as unknown, never invented. |
| **Adaptive contract discovery** | Selections that land on the edge of the downloaded strike band are detected, the band is widened for those sessions and the run repeated. Regression tests require a selective download to choose the same contracts as the complete chain. |
| **Data** | A background collector that archives intraday bars and option trade prints while free sources still serve them; streaming imports of CSV / ZIP / Parquet in ten vendor layouts; live verification of what each source actually delivers. |
| **Research surfaces** | A node canvas of each notebook (branch from any result), a sensitivity explorer (two-assumption heatmaps with thin and stable cells marked), and an encyclopedia of twelve market-behaviour studies computed on your stored data. |
| **Quantitative reasoning** | Questions about a result are answered by deterministic tools: sub-periods, whether two thresholds really differ, cost survival, concentration, how much volatility explains, which assumptions matter most, why options differ from the underlying. |
| **Benchmarks** | `strata bench` times realistic workloads on a reproducible fixture, records every run with its commit, and fails CI on a regression. |

### The workflow, on real data
Run in the browser on the stored SPY history (8,514 sessions, 1993-2026); the numbers are what
the engine computed on 2026-11-25.

1. *"Investigate whether SPY tends to reverse following large down days."*
2. A plan of **36 configurations** (bound 60): six decline thresholds x five outcomes, plus a
   split by volatility. Training 1993-2013, validation 2013-2020, final test 2020-02-18 onward.
3. Daily SPY and VIX history is already stored; nothing needed downloading.
4. All 36 are evaluated and listed. Five are distinguishable from the baseline after correction;
   four of those also agree with the validation period and become candidates.
5. The four that survive are all the same outcome: *1% up before 1% down during the next
   session* after declines of 1% to 2.5% (for example 38.2% of 186 training sessions against
   17.5% on all sessions). The plain reversal rows did **not** survive: after a 3% decline the
   next session closed above its open on 61.8% of 68 sessions against 51.3%, which is within
   what chance produces once 36 tries and clustering are allowed for.
6. Taking the 2% version further gives the full study: 104 of 342 sessions (37.8%) against
   14.9%. The reviewer's verdict on it: about **70% of that difference is explained by when
   these sessions occur** (93% of them have the volatility index at 20 or more, against 34% of
   other sessions); within the same volatility band the difference is +7 points.
7. Option structures on the close-entry form, priced on stored end-of-day bid/ask quotes: 66 of
   the 342 qualifying sessions have quotes (2020-2025). At-the-money 7DTE call +$1,126
   (+$17 per trade, 43.9% winners); the put on the same sessions -$4,877; straddle -$3,751;
   iron condor -$2,799; debit spread -$1,236.
8. Challenging the best of them: **three serious problems.** Only 19% of signals could be
   priced; without its three best trades the result is -$2,860; it reaches zero when every fill
   is 1.1% worse, less than a typical option spread. With whole weeks resampled the average
   trade's p-value against zero is 0.74.
9. Chronological validation is built into steps 2-5 (the final period was scored once and is
   marked "used once"), and the selection itself is cross-validated with purging and an embargo.
10. Every step is a node on the canvas and a section of the exportable report.

The honest summary of that session: a real pattern in how far SPY moves after a decline, mostly
a volatility effect, and no evidence of a profitable option position.

### Statistics that allow for dependence
Large moves cluster, so sessions that qualify together are not independent draws. Each claim
therefore carries two numbers: the usual one and one from resampling whole calendar weeks
(extended to blocks of weeks when neighbouring weeks are correlated). The wider interval and the
larger p-value are the ones used. On simulated data where hits arrive in runs, the independence
interval covered the true value 62% of the time and the cluster interval 93%; a true null was
called significant 14.5% of the time by the ordinary test and 7.5% by the cluster test. Those
calibration checks are part of the test suite.

The ledger counts distinct configurations, so re-running the same one is free and running a
plan of 36 costs 36. A plan's own drill-down ("Investigate further") is the same configuration
and does not contaminate the final period; a new threshold tested afterwards does. Option
positions priced on an already-selected condition are counted as trials but do not re-tune the
condition.

### Look-ahead
V4 found and closed one hole in V3's rule language: `close(0)`, `session_high(0)` and negative
counts such as `return_pct(-1)` addressed a session that had not finished. Counts below one are
now rejected when a rule is validated and again when it is evaluated. The reviewer's
future-scramble test alters every price after the decision instant and requires the rule to
give the same answer; a regression test re-opens the hole on purpose and checks the test
catches it.

### Data sources: what was actually verified
`strata providers` (and Data -> Sources) makes a few small read-only requests per source.
Measured from this installation on 2026-11-25:

| Source | Live check | Measured |
| --- | --- | --- |
| Yahoo Finance (prices) | verified | daily from 1993-01-29; 60-minute bars from 2023-12-28; 5-minute from 2026-09-01; 1-minute for the last 7 sessions per request; 0% of expected bars missing |
| Cboe (VIX) | verified | 9,323 daily rows from 1990-01-02 |
| Yahoo Finance option trades | verified | 8,754 one-minute trade bars for 12 at-the-money contracts; a contract that expired weeks ago returned nothing, so history must be collected while it is served |
| End-of-day option archive | verified | SPY chains listed for 2008-2025 |
| DoltHub options | verified | latest chain date 2026-11-24, 214 SPY rows |
| **Alpaca** | **unverified** | no credentials were available. Its pagination, contract discovery, resume and entitlement checks are covered by mocked tests and an opt-in live test; whether a free account can read what the documentation says has not been confirmed here. |

The paid adapters and the cloud language-model providers are likewise unverified against live
endpoints. Other legitimate sources (Cboe DataShop, OptionsDX, historicaloptiondata.com, Massive
flat files, Databento, ThetaData, Stooq, FirstRate / Kibot) are listed in the app with their
cost and how to use them; they enter through the bulk importer, never through an automatic call.

### The collector
Off by default. When switched on (Data -> Collector) it runs after each close, or on demand, or
from cron with `strata collect --once`. It refreshes the watch list's bars and downloads option
trade bars for 0, 1, 3, 7, 14 and 30 DTE contracts around the previous close, nearest the money
first, within a request budget and rate limit. It only issues GET requests for market data: the
single HTTP helper refuses any trading path, and a test scans the code for anything else.
A session it did not reach while the data was still served is recorded as missed, not filled in.
One real cycle from this machine: 246 requests in 69 seconds stored 14,797 trade bars for SPY
and QQQ; 24 contract-sessions failed and were left for the next cycle.

### Benchmarks
`strata bench` on the fixture (simulated on purpose, so it is reproducible offline):

| Workload | Time |
| --- | --- |
| Generate and store 61 sessions of intraday bars and option chains, two symbols | 1.71 s |
| Load cached bars, build the price series, read one chain | 0.009 s |
| One backtest, 60 sessions of 30-minute chains | 0.27 s |
| Sweep of 12 backtests | 2.91 s |
| Feature table for 16 years of sessions | 0.025 s |
| Hypothesis plan, 36 configurations with cluster tests | 0.74 s |
| One configuration, session-by-session engine | 0.121 s |
| The same configuration, vectorised | 0.0001 s (about 1,200x) |
| Three backtests in three processes | 0.87 s |

Peak memory 544 MB. The vectorised feature engine is the optimisation the profile called for:
discovery and maps no longer re-walk every session per configuration, and a test requires it to
match the session engine exactly. Runs are appended to `benchmarks/history.jsonl`;
`strata bench --check` compares against `benchmarks/baseline.json` after normalising by a
calibration loop. The browser suite records how long the sensitivity map takes to render.

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
strata bulk-import PATH [--inspect]   # a file, ZIP or folder in a vendor layout, streamed
strata collect [--once|--daemon|--status]   # the market-data collector (read-only)
strata providers [yahoo cboe ...]     # live check of what each source actually delivers
strata bench [--check] [--real]       # time the research workloads and record the result
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
cd backend && .venv/bin/python -m pytest          # 367 tests (12 opt-in live ones skip), about a minute, no network
cd backend && .venv/bin/strata validate           # fourteen steps on real observations, ~15 seconds
cd backend && .venv/bin/strata bench --check      # performance against the recorded baseline
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

V4 adds: calibration of the dependence-aware intervals and tests on simulated clustered data;
purged cross-validation and overlapping holding periods; every session feature against the
rule-language expression it is tied to, and vectorised outcomes against the session engine; an
independent plain-Python recomputation of a study; rejection of zero and negative look-backs and
the future-scramble test (including a deliberately leaky rule it must catch); plan bounds,
reporting of every configuration, and noise producing no confirmation; ledger counting, holdout
contamination and the requirement for a new holdout; execution assumptions (slippage, delay,
volume, participation, partial and missed fills, legging); selective versus complete-chain
contract selection for six strike methods and adaptive widening; contract adjustments and
constructed identifiers; streaming imports from a ZIP in bounded memory and out-of-order
sessions; collector persistence, resumption, request budget, missed sessions and the read-only
guarantee; and provider verification with and without credentials. `strata validate` gained
three steps: a plan checked against the session engine and against plain arithmetic on real SPY
history, the future-scramble test on real data, and live source verification. The browser suite
gained six tests covering the plan flow, the reviewer, the canvas, the sensitivity map, the
encyclopedia and the data panels.

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
               (investigations, follow-ups), llm (providers), explain (grounded text), agent,
               nl_graph (the composing reader), agent_v5 (graph, memory and registry follow-ups)
  core/        calendar, Black-Scholes / American pricing, causal indicators
  data/        store (datasets, manifests), migrate, catalog, planner (requirements, coverage,
               selective and adaptive acquisition), importers, bulk (streaming imports),
               providers/ (free sources, paid adapters, bench = live verification, market =
               the provider marketplace and cost estimates)
  engine/      models, market (point-in-time data, trade-print chains), expr (rule language),
               sequence (ordered stages), scan (signals only), rules, selection, execution,
               margin, portfolio, engine (event loop)
  research/    pipeline (the seven stages), study, stats, discovery, explain_day, views
               (calendar, observations, comparison), report (HTML / Markdown / CSV)
               V4: robust (dependence-aware statistics), ledger (trials and holdouts), features
               (vectorised session features), hypothesis (plans, option structures), sensitivity,
               encyclopedia, reasoning, challenge (adversarial review), execution (stress ladder)
               V5: graph (typed research graphs), evidence (the one comparison every analysis
               uses), matching (matched comparisons), patterns (event explorer), registry
               (cross-notebook trials and holdouts), memory (findings), ml (optional), forward
               (frozen experiments), signals (saved signal lists)
  collector/   the background market-data collector
  bench.py     the `strata bench` command
  analytics/ · optimize/ · portfolio/
  api/         FastAPI app, research_api, v4_api, v5_api, ide (strategy IDE: compact form, section
               edits, debugger), launch (jobs + caching), settings (credential stores), notify, db
  validate.py  the `strata validate` command
backend/tests/ unit suite, test_live.py (opt-in), test_reproducibility.py, e2e_seed.py
frontend/src/  pages/lab (research workspace, Canvas, Research and Graph views), pages/explore
               (Patterns, Findings, MlLab), pages/build (Ide, CodeEditor), pages/validate (Forward,
               Registry), pages/research (Sensitivity, Encyclopedia), pages/data (Collector,
               Marketplace, bulk import), pages/builder, pages/results …
benchmarks/    baseline.json, history.jsonl (backend), frontend.jsonl (browser render times)
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
- **The reader composes, but it is still a reader.** Questions are built from subjects, time
  spans and predicates it knows; a phrase outside them is reported as not understood, and the
  graph can be written directly or proposed by a configured language model. The seven Hypothesis
  Lab families remain as shortcuts.
- **The one worked example found no edge.** The session above is a faithful record: the pattern
  is "suggestive" at best and does not survive option costs. V5 was built to say that plainly.
- **Forward validation needs time and a running collector.** An experiment frozen today has no
  forward evidence until sessions are collected after it; option prices for a past session often
  cannot be fetched later.
- **Staged exits, ATR exits and hedges are simulated at the engine's tick**, on the bars and
  option prices stored. On end-of-day option data only close-to-close behaviour can be priced.
- **The ML page predicts the underlying's outcome.** It is not an option-profit model, and on the
  default study it reports no out-of-sample skill.
- **The option stage is only as wide as the option data.** End-of-day quotes (2020-2025 for SPY)
  can price entries at the close; entries at the open or intraday need trade prints, which
  exist only where they were collected. Coverage is stated per signal and reviewed as a bias.
- **SPX options** have no free source here and no strike grid configured; SPX is supported
  through imported files.
- **The volatility control is coarse.** Four fixed bands of the volatility index's previous close.
- **Bootstrap intervals are approximations.** They are calibrated in tests at realistic sample
  sizes and still under-cover when dependence is extreme or weeks are few.
- **The registry knows what Strata ran, not what you read elsewhere.** A new notebook's final
  period on dates other notebooks already evaluated is reported as not independent, but the
  registry cannot know what was learned outside the application.
- **The collector needs a machine that is awake** at the close; a laptop that sleeps will miss
  sessions and will say so.
- **Free intraday underlying history is short** (about a month of 1-minute bars from Yahoo), so
  intraday rules have small samples until more history is loaded. Sessions without fine enough
  bars are excluded and counted, not treated as "no signal".
- **Modeled premiums** ignore real skew dynamics, liquidity and event pricing. They are estimates.
- **The built-in interpreter** understands common phrasings, not arbitrary English. When it
  cannot read part of a request it quotes that part rather than guessing.
- **Margin** is strategy-based Reg-T, not portfolio margin. Special dividends and mergers are
  not modelled.
- **Scale.** Application state uses SQLite; single user.
