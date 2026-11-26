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
- [What V6 adds](#what-v6-adds)
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
runs the same free download on first start (`STRATA_SKIP_FETCH=1` to skip it). The port is
published on this computer only (`127.0.0.1`); `STRATA_PUBLISH=127.0.0.1:8010` in `.env` moves it
when 8000 is taken. **Run one copy at a time on a data folder.** A copy on the computer and a copy
in a container cannot share one database, so the second one to start goes on standby: it shows
which copy holds the folder and where to open it, saves nothing, and continues by itself about 20
seconds after the other stops. Rebuild the image after pulling a new version
(`docker compose up --build -d`): an old container keeps serving the old application.

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

## What V6 adds
V6 adds no data framework. It answers one question, on real observations: **can the strategy this
project started from be backtested, and if not, what exactly is in the way?** Then it fixes what an
audit of the forward record, the engine, the statistics and the reader found.

> *Buy ATM SPY 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in SPY, only on days when SPY
> fell at least 3% the previous trading day.*

### Where the data stands
The software could always run this strategy. What was missing was **observed intraday option
prices for the sessions it trades on**. On 2026-11-25 a free Alpaca paper key was connected, the
access test was run with it, and the history it serves was downloaded and checked. Counted from the
files on the development machine afterwards (`Data > Readiness`, `strata usecase`,
`strata bars-audit SPY`, `strata options-audit SPY`):

| Stored for SPY | Sessions | Period | Good for |
| --- | --- | --- | --- |
| Daily bars (Yahoo) | 8,515 | 1993-01-29 to 2026-11-25 | the signal (which days qualify) |
| 1-minute bars, consolidated tape (Alpaca) | 2,741 (2,740 whole) | 2016-01-04 to 2026-11-25 | intraday entries and exits of the underlying |
| End-of-day option chains, bid/ask | 1,495 | 2020-01-02 to 2025-12-12 | positions entered and exited at the close; whether a 0DTE contract was listed. **Never** an intraday fill |
| 1-minute option **trade bars** (Alpaca) | 718 (711 whole, 7 the provider holds only in part) | 2024-01-18 to 2026-11-25 | intraday option backtests on observed trades. No bid/ask |
| Intraday option bid/ask | 0 whole, 1 partial | 2020-03-06 09:30 to 12:00 (a vendor's public sample) | one session, priced by the reference simulator only |

The option set is 43.4 million bars over 331,147 contract-sessions (629 MB): every contract within
3% of the previous close expiring within 7 days, calls and puts, on every session, plus the
contracts the original strategy could select around the opening price on its own sessions. QQQ and
IWM were not asked for and are unchanged; the same two commands download them.

**What the key returns (measured).** `strata access alpaca`, about 140 read-only requests:
- 100 of 100 expired contract-sessions on ten dates (2024-02-14 to 2026-11-18) returned one-minute
  trade bars, every expiration found through the listing endpoint; individual trades are served too.
- **Option history starts on 2024-01-18**: every session from 2024-01-02 was asked and none before
  it returned a bar. 2024-02-02 returns nothing at all.
- Stock minute bars come from the consolidated tape (`sip`), 390 of 390 minutes; **they start on
  2016-01-04**, and the session before returns nothing.
- No OPRA agreement on the account (real-time OPRA quotes answer 403); history does not need one.
  There is no historical option bid/ask on any Alpaca plan.

**The original strategy on those observations.** The rule fires on **105 sessions** since 1993
(plain arithmetic on daily closes and the engine's own scan give the same 105; the sentence and a
hand-written configuration compile to the same strategy). Of those 105:

| | Sessions |
| --- | --- |
| Priced by the engine on stored observations | **5** (Alpaca trade bars) |
| Priced by the independent reference only (a partly observed session) | 1 (2020-03-06, bid/ask sample, +$150.70) |
| No contract expiring that day was listed (known from stored end-of-day chains) | 10 |
| No intraday option prices stored | 89 |

| Session | Fall the day before | Entry | Exit | Net, one contract | Worst to best print |
| --- | --- | --- | --- | --- | --- |
| 2025-04-04 | -4.93% | 09:31, 525 call at 4.54 | 15:55 at 0.01 (time) | -$454.30 | -$498.30 to -$418.30 |
| 2025-04-07 | -5.85% | 09:31, 488 call at 7.64 | 09:57 at 9.82 (SPY +1%) | +$216.70 | +$105.70 to +$234.70 |
| 2025-04-11 | -4.38% | 09:31, 524 call at 6.22 | 12:33 at 7.25 (SPY +1%) | +$101.70 | +$25.70 to +$112.70 |
| 2026-11-06 | -4.41% | 09:31, 736 call at 3.65 | 15:55 at 0.27 (time) | -$339.30 | -$347.30 to -$272.30 |
| 2026-11-23 | -3.64% | 09:31, 722 call at 4.97 | 15:55 at 0.81 (time) | -$417.30 | -$574.30 to -$373.30 |

Total **-$892.50**, 2 winners of 5; between -$1,288.50 and -$716.50 if every fill had been the worst
or the best price the contract traded at in the minute the fill came from (trade bars carry no bid
or ask, so that range is the only observed measure of how far a real fill could have been from the
print used). The engine and the reference simulator agree on all five. These are **all** the
sessions the rule produced in the 34 months for which any free intraday option history exists.
Five trades are five observations: nothing about the strategy's profitability is established, and
the app says so wherever a number from them is shown.

| Variant | Rule fired (since 2024-01-18) | Priced | Net | Worst to best print | Winners |
| --- | --- | --- | --- | --- | --- |
| after a fall of 1% or more | 1,079 (75) | 75 | +$2,384.50 | -$962.50 to +$4,268.50 | 38 of 75 |
| after a fall of 2% or more | 342 (16) | 16 | -$81.80 | -$919.80 to +$429.20 | 8 of 16 |
| after a fall of 4% or more | 49 (4) | 4 | -$475.20 | -$714.20 to -$343.20 | 2 of 4 |
| baseline: an ATM 0DTE call at 10:00, out at 15:30, every session | 718 | 707 | -$376.14 | | 269 of 707 |

At 1% the sign of the result depends on which print of the entry minute is assumed. A rule that
fires often can now be tested on 711 sessions (the baseline runs in 100 seconds); the original
rule cannot, because it fired five times.

### What is still out of reach
Of the 105 sessions, 10 had no such contract and **57 fall before 2010**, which no source compared
here reaches at any price (SPY had no weekly expirations then, so a same-day contract existed only
on monthly expiration days). **38 could be reached by data at all**; 5 are stored
(`POST /api/data/budget`, or ask *"Estimate what it would cost to acquire the missing observations"*).

| Budget | What it buys | Sessions reached | Evidence |
| --- | --- | --- | --- |
| stored now | Alpaca trade bars | **5** | measured |
| $0 | Databento's $125 new-account credit: 1-minute bid/ask from 2013-04, about $0.02 of it for 351 contract-sessions | up to 27 | the provider's description; never run from here |
| $25 | OptionsDX 30-minute bid/ask chains for the eight years that hold signals (2010-2011, 2015-2016, 2018-2020, 2022), $24 | up to 38 | storefront prices; the importer was verified on the vendor's sample |
| $50, $100, $200 | nothing more for this strategy | up to 38 | |

A session is counted when it lies inside a source's history; whether the contract was listed or
traded that day is established only for 26 of the 105. A 30-minute snapshot sees an exit at the next
snapshot, not when it happens. Even all 38 would be a small sample.

**University of Washington** (pages read 2026-11-25): WRDS is managed by the Foster School. Faculty
and PhD students ask the Foster data research consultant for an account; undergraduate and master's
students are told to ask their instructor. **OptionMetrics is not in the list of WRDS datasets
Foster publishes**, and OptionMetrics is end-of-day in any case, so it could not price this strategy.
TAQ is on the list and would supply minute bars of the underlying, which are now stored from 2016
anyway. Bloomberg is terminal-only in the Foster Business Library with export limits. UW's
guidelines allow saving data for a single assignment and forbid systematic downloading and use with
AI tools, so the only route is a file you export yourself and import (`Data > Bulk import`). Strata
never logs in to any of it.

### Checked against the source
| What | How | Result |
| --- | --- | --- |
| Option trade bars | 32 stored contract-sessions (2024-02-22 to 2026-11-13, quiet and busy contracts) fetched again as 141,605 **individual trades** and rebuilt into minute bars by separate code (`strata options-audit SPY --verify 32`) | 5,558 of 5,559 bars identical in open, high, low and close (99.98%); 95.16% also in volume; no bar without a trade, no traded minute without a bar |
| Underlying minute bars | the exchange calendar, the daily record, and Yahoo's minute bars where both exist (`strata bars-audit SPY`) | 2,712 sessions of 390 bars, the 21 early closes of 210, the four halted sessions of March 2020 short by the halt; last bar within 0.05% of the official close on 2,697 of 2,740; all 8,190 bars of the 21 shared sessions match Yahoo's, every close within 0.02% |
| Every priced trade | the engine against a reference simulator that shares no code with it | 100 of 100 trades across the four thresholds agree on contract, times, prices, fees and P&L |
| The underlying study | the 35 qualifying sessions since 2016 recomputed from minute bars | the daily and the minute record agree on all 35 (19 reached +1% from the open) |
| Alpaca against Yahoo, the same contracts and minutes | 481 contract-sessions on 2026-11-23 and 11-24, 73,317 common minutes | the minute's close is identical on 54.9%; 65.6% of Yahoo's bars hold one price (25.7% of Alpaca's); Yahoo has the 09:30 bar for 42 contracts, Alpaca for 388; daily volume within 3% |

Yahoo's option bars are therefore closer to samples of the last price than to trade bars, and
Alpaca's are read first where both exist. The trade reported earlier for 2026-11-23 from Yahoo's
bars (721 call, 09:32 at 5.50, -$360.30) is superseded by the one above; both are in the report.

### What running it against the live service found
None of this was visible while the key-gated code had only run against recorded doubles. Each has
a regression test.

| Found | Effect | Now |
| --- | --- | --- |
| A key saved in an old container (the V2 build, still on port 8000) never reached the database: the container and the app shared one data folder, and SQLite's write-ahead log is not coordinated across a container boundary | a committed write was dropped a minute later | **One machine at a time** opens a data folder (`strata/core/instance.py`): a lease with a heartbeat. The other copy goes on standby, says on the whole page who holds the folder and where to open it, and continues by itself about 20 seconds after that copy stops. Taking the lease is a claim that a live holder refuses |
| The route that serves the interface returned **any readable file** for a path with encoded `..` | whoever could reach the port could read the database or a key file | only files inside the build directory are served; an unknown `/api` path is a 404 |
| Inside Docker the computer's own browser arrives from the container network and was refused as remote | the container build could not be used at all since V3 | the compose file publishes the port on `127.0.0.1` only and declares it (`STRATA_LOOPBACK_PUBLISHED`); the Host name must still be local |
| A running copy remembered "no key" for its whole life; the app on the computer and a container read different credential stores | "I saved it and it says not set" | an absence is remembered for 3 seconds; the native app also reads the protected file a container writes; *where* a key was stored (never the key) is recorded, so a container says "saved in the macOS Keychain, enter it here as well" instead of "not set" |
| Alpaca lists `multiplier: "0"` for every contract that expired on or before 2024-04-26 (5,584 of 76,080 definitions) | those trades were multiplied by zero, then the margin code divided by it | the contract size and deliverable are read instead; the engine refuses a non-positive multiplier and says so |
| Alpaca holds seven early sessions only in part (2024-01-22, -23, -24, -29, 02-02, 02-26, 03-15) | an entry would slide to the first print the provider happens to have | recorded as partial with the times; coverage, readiness and the engine leave them out |
| The rule that removes Yahoo's running-total volume flagged 32 real block prints on the consolidated tape (SPY, 2018-09-21 10:43: 18.7 million shares, 17.7% of the day) | real volume dropped from VWAP on expiration Fridays | applied only to feeds that produce the artefact |
| On 70 of 2,741 sessions the daily high or low lies outside every regular-session minute bar, 65 of them at about the previous close | the audit said prints were missing | a daily bar counts late and reference-priced trade reports; the finding says so, and a session with intraday bars takes its range from them |
| A band around the previous close misses the at-the-money strike after a gap open (2026-11-23: close 689.81, open 724.33) | the bulk download alone could not price that session | `strata usecase --acquire` and the lab fetch the contracts around the price at the signal |
| The engine expanded every session's option data even when nothing traded | 98 seconds to price 5 sessions in the lab | a session's chain is built on first use; all 100 use-case trades are identical before and after |

### What the audit found and fixed
An independent reference simulator (`strata/audit/reference.py`, sharing no code with the engine)
and property-based tests (cash + marked positions = equity, fees, exits never exceed open quantity,
no look-ahead) found these. Each has a regression test.

| Found | Effect before the fix |
| --- | --- |
| The official close was rejected when it was more than 1% from the last intraday bar | "fell 4%" did not fire on a day it should have (SPY 2025-04-09: hourly bars end 542.84, official close 548.62), and expiration settled on the wrong price |
| A position with a small net delta was treated as having no direction | an underlying target never fired; a stop fired either way |
| Short shares had an unbounded margin requirement | a hedged long call or short put was margin-called at entry |
| Hedge shares stayed open after the options expired | phantom stock exposure and P&L |
| The hedge record left out its closing trade and deliveries | cash did not equal the recorded flows |
| A staged exit followed by expiration overwrote the leg's exit price | wrong per-leg P&L |
| The last daily row still counted closed positions | wrong open-position count |
| Yahoo 1-minute bars carry the day's running volume in 5 to 9 bars a session | raw minute volume summed to 289% of the daily volume; those bars now get no weight in VWAP or volume rules |

**Forward validation** used the last *stored* session as the activation date, so a stale dataset
back-dated an experiment and a later backfill could slip in-sample days into the forward record.
Activation is now an immutable UTC instant from the clock; the first session that counts comes from
the exchange calendar (activation during market hours starts with the next session); every decision,
fill, exit, open position and missing-data event is appended to a hash-chained ledger that the
database refuses to update or delete; data that changes after a session was sealed is recorded as a
revision, never applied; open positions are read back from the ledger after a restart.

**ML research** scored every candidate on the reserved test period, so the best-looking one could be
picked from it. Candidates are now compared on purged cross-validation only; the selected
configuration is frozen; the test period is scored for that one model, on request, pinned to what
was frozen (the step is refused if the model, features or rows changed); and every scoring is
recorded in a table that cannot be edited. A second configuration scored on the same period makes
both "tested repeatedly", never "confirmed". A failed leakage check is its own verdict.

**Reading requests.** One intent schema and one sequence for every request: understand, validate
against the sentence, check the data, plan, execute, explain, follow up. Every size and clock time
in a sentence must have a place in the interpretation or the part is named as unread. Measured with
`strata eval-nl` on 150 written requests:

| Set | Cases | Correct | Silently wrong |
| --- | --- | --- | --- |
| Development (the reader was fixed against these) | 110 | 60.9% before, 99.1% after | 25.5% before, 0% after |
| Held-out, **first run** (written afterwards; the honest estimate) | 40 | **75.0%** | **20.0%** |
| Held-out, after fixing what it exposed | 40 | 92.5% | 2.5% |

The built-in reader was evaluated. No language model is configured on the development machine, so
the model-backed path is **not run** (`strata eval-nl --llm` says so rather than scoring zero).

**Security and reliability.** The research SQL console checked a list of forbidden words that a
comment or a quoted path walked around: it could read any file on the computer (the stored API keys
among them), write files and request web addresses. It now runs one read-only statement, as the
database's own parser sees it, on a connection whose file access is confined to the market store by
the engine, with network access and extensions off and the configuration locked (its views had also
pointed at a folder layout abandoned in V3, so it returned nothing; they work again). Also: ZIP and
gzip bombs and member counts are bounded before extraction; provider and symbol names cannot be
paths; remote callers can import only from `<data>/imports`; the local-model address must be a
loopback or private host; store writes are atomic and locked across processes, so the application
and a collector container sharing one data directory no longer lose each other's sessions; the
collector's one-cycle-at-a-time claim is atomic; `strata backup` / `strata restore` copy the
database with SQLite's online backup, hash every file, exclude credentials and refuse a damaged
backup.

**Interface.** Coverage comes before profitability everywhere: the plan, the result card, the
report, comparisons and the forward page lead with "N of M qualifying sessions could be priced",
large, with what the rest are missing. The notebook shows the seven stages of the current piece of
research and each opens its surface; a result hands the same configuration to the IDE, the visual
builder, forward validation and the registry. `Data > Readiness` shows the five kinds of research
per symbol and year, the provider access state and steps, the original question's coverage and the
bar audit.

### Acceptance: software, data access, research
Run on 2026-11-25 against the running application over HTTP (as the browser does), with the Alpaca
key connected and its history stored. The runs carry the engine label 6.0.0: the number was raised
to 6.0.1 afterwards for the changes listed above, with no change to the engine in between.

| | Status |
| --- | --- |
| **Software verified** | 482 backend tests (12 opt-in live ones skip), 24 browser tests, `strata validate` 14 of 14 on freshly downloaded real observations, `strata bench --check` without regression |
| **Data access verified** | **Alpaca** with a paper key: stock minute bars since 2016-01-04, option trade bars and individual trades since 2024-01-18, expired contracts included. Yahoo daily, intraday and option bars; Cboe VIX; the end-of-day option archive; the OptionsDX importer on the vendor's public 5-minute sample. **BLOCKED:** Massive and Databento (need a key each), ThetaData and ORATS (a subscription), the language-model path (a model key or a local Ollama) |
| **Research verified** | The underlying study on 105 sessions, its 35 sessions since 2016 confirmed on minute bars. The option strategy on **5 of 105** sessions by the engine (6 with the reference), every trade recomputed independently. Nothing about its profitability is established: no free history holds more of its sessions |

| Prompt | What the application answered |
| --- | --- |
| How often does SPY rise 1% from the open after a 3% down day? | 68 of 105 sessions (64.8%, 95% interval 55.3% to 73.2%) against 17.2% of all 8,513 sessions |
| Test buying ATM SPY 0DTE calls on those days. | Before running: 5 of 105 can be priced; 89 have no stored option prices, 10 had no such contract, 1 is partly observed. Result: 5 trades, -$892.50, 2 winners. "With so few trades the averages say very little about what to expect." |
| Run the same strategy after 1% and 2% declines. | Coverage first: 75 of 1,079 and 16 of 342 qualifying sessions priced; +$2,384.50 over 75 trades and -$81.80 over 16, "a description of what happened, not evidence" |
| Compare 0DTE, 1DTE, and 7DTE calls. | 5 of 105 priced for each; from -$892.50 (0DTE) to -$469.50 (7DTE); every variant flagged as fewer than 30 observations |
| Enter at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move. | New version; entry at the first print after 09:30 ET, exit 15:45 ET; 5 trades, -$604.50 |
| Show every actual option fill and its source. | Five fills, each an observed trade print from `alpaca/SPY/options/trade/1m/v1`; no bid or ask existed and none is shown |
| Show exactly which historical sessions could not be tested. | All 100, each with its reason: 89 no intraday option prices stored, 10 no contract listed, 1 partly observed |
| Estimate what it would cost to acquire the missing observations. | The budget table above, with the 57 pre-2010 sessions named as unreachable and Alpaca named as the one route that was measured |

New commands: `strata access alpaca`, `strata download SOURCE DATASET SYMBOLS` (resumable),
`strata usecase [--acquire]`, `strata bars-audit [SYMBOLS]`, `strata options-audit [SYMBOL] [--verify N]`,
`strata eval-nl [--llm]`, `strata backup DEST`, `strata restore SRC [--verify]`.

To repeat the downloads: `strata download alpaca bars_1m SPY` (51 seconds for eleven years) and
`strata download alpaca option_trades SPY --band 3 --max-dte 7` (40 minutes, 6,190 requests inside
the free limit of 200 a minute; what is stored is skipped), then `strata usecase --acquire`.

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
| **ML research (optional)** | Logistic, ridge, lasso, tree, forest and gradient-boosted models scored out of sample with purged, embargoed chronological folds and a final test set at the end of history (since V6 that set is scored once, on request, for the frozen model only). Reported against the base rate (Brier and log-loss skill), with calibration, a time-shifted placebo, permutation importance and a leakage probe that scrambles the future. Random splits are not offered. The only bridge to a strategy is a saved walk-forward signal, priced by the ordinary backtester. `pip install -e "backend[ml]"`. |
| **Forward validation** | Freeze a finished backtest or study: the configuration is stored with a hash and its activation instant (V5 used the last stored session as the date, which V6 replaced with the clock and a sealed ledger: see above). Only later sessions count; fewer than ten forward observations are reported as a record, not a verdict. A change is a new version with its own date. Paper only: there is no order entry anywhere in Strata. |
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
| Yahoo Finance option bars | free | 1-minute bars for contracts still listed or just expired | **verified** (an expired contract returned nothing). Compared with Alpaca's trade bars they behave like samples of the last price, not trade bars: see [Checked against the source](#checked-against-the-source) |
| DoltHub options | free | recent end-of-day chains, about three expirations per date | **verified** |
| Cboe | free | VIX and index history; delayed chain snapshots | **verified** (index history) |
| Alpha Vantage | $49.99 a month | end-of-day chains from 2008 | parser verified on the demo symbol only; SPY, QQQ, IWM need a paid key |
| Alpaca | free key | 1-minute option trade bars from February 2024, expired contracts included; no historical bid/ask on any plan (the $99 plan adds the live OPRA feed) | **verified** with a paper key on 2026-11-25: 100 of 100 expired contract-sessions returned trade bars; the history starts on 2024-01-18, not February; seven early sessions are incomplete at the provider; stock minute bars start on 2016-01-04 |
| Massive (formerly Polygon) | free key; paid tiers | per-contract minute aggregates, 2 to 20+ years by tier | **unverified**: no credentials |
| Databento OPRA | pay per use, $125 credit | trades, 1-minute bars and 1-minute bid/ask from 2013-04 | **unverified**: no credentials |
| ThetaData | from $40 a month | 1-minute quotes from 2020 (Value), earlier on higher tiers | **unverified**: needs their terminal |
| ORATS | subscription or one-time | 1-minute data from 2020-08 | **unverified** |
| OptionsDX | $0 to $50 per symbol-year | bid/ask snapshots of whole chains, SPY 2010-2023 and QQQ 2012-2023 | importer **verified on the vendor's public 5-minute sample** (one partial session, 2020-03-06); no purchased file has been imported |
| Cboe DataShop, historicaloptiondata.com, FirstRate, WRDS OptionMetrics, algoseek sample | paid or institutional | files | import only; layouts tested on files written with the documented columns |
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
| 1-minute bars | for about a month (a free Alpaca key reaches back to 2016-01-04: `strata download alpaca bars_1m SPY`) |
| Option trade bars (Yahoo) | only while the contract is listed, and about a day after it expires. After that it is gone from this source and is recorded as missing, never filled in |

The application and the collector may share one data directory (V6): every store write is atomic and
takes a lock across processes, and only one collection cycle can claim the database at a time. On
macOS a native process and a Docker container do not see each other's file locks, so run both the
same way. This is how intraday 0DTE history accumulates for free: each close that the collector is
running for adds that session's option prints, and no close it was not running for can be recovered
from this source. `strata backup DEST` copies the database and the data with their hashes.

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
| **Alpaca** | unverified when V4 was written; **verified in V6** | a paper key was connected on 2026-11-25: see [Where the data stands](#where-the-data-stands) |

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
| Trade-priced backtests | A fourth mode executes on observed option **trade prints** (free: Alpaca since 2024-01-18, verified against its individual trades; Yahoo's bars for the last weeks). A print is used only while it is recent; otherwise the order is deferred, then rejected. No spread is invented, and trade bars are never read as quotes. |
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
| Alpaca | free key | 1-minute stock bars from the consolidated tape since 2016-01-04; 1-minute option **trade** bars since 2024-01-18, including expired contracts (no historical option quotes on any plan). Both measured with a paper key | Alpaca data agreement |

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
strata access alpaca [--json]    # what this account's key actually returns (exit 2 = blocked on credentials)
strata download alpaca bars_1m SPY                      # 1-minute bars since 2016; resumable (exit 2 = no key)
strata download alpaca option_trades SPY --band 3       # option trade bars: strikes within 3% of the previous close, <= 7 days out
strata usecase [--symbol SPY] [--acquire]               # the original 0DTE question on stored observations; --acquire first
                                                        # fetches the contracts its sessions need around the price at the signal
strata bars-audit [SYMBOLS]      # missing minutes, early closes, DST, volume artefacts, mixed feeds, two feeds against each other
strata options-audit [SYMBOL] [--verify N]              # stored option bars: sessions absent, empty or partial at the provider;
                                                        # --verify refetches N contract-sessions as individual trades and rebuilds the bars
strata eval-nl [--llm]           # interpretation accuracy on 150 written requests
strata backup DEST [--no-market] # database (online backup) + data, hashed; credentials excluded
strata restore SRC [--verify]    # refuses a backup that does not verify
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
cd backend && .venv/bin/python -m pytest          # 482 tests (12 opt-in live ones skip), a few minutes, no network
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

V6 adds 115 tests in thirteen files: the forward ledger (activation from the clock, the stale-dataset
scenario, backfills, tampering, restarts), the reference simulator against the engine with
property-based cases, the bar audit, the budget arithmetic, the access classifier against a
recorded double (credentials redacted, trading paths refused), the OptionsDX layout, the ML test-set
record (a real relationship, pure noise, a feature that knows the answer, and the pinned final
test), readiness and the data check, the interpretation evaluation, and the security cases
(traversal, archive bombs, malformed files, code in a rule, the SQL console, SSRF, a misbehaving
model, credential exposure, two processes writing one dataset, backup and restore, the interface
route, a container published on loopback). Two more files came from running Alpaca for real:
`test_v6_alpaca.py` (a session the provider holds only in part, the zero multiplier, volume on the
consolidated tape, bars rebuilt from trades, two providers compared, paging and the rate limit, the
download command) and `test_v6_instance.py` (the lease: a live copy on another kernel keeps the
database closed, liveness is the counter moving and not the clock, a wrong claim is refused, a
frozen holder yields, two real processes, standby in the web application, credentials on both
sides of a container boundary). The browser suite gained six tests: coverage before profitability,
data readiness and the provider's state, the forward ledger, the two-step ML study, the standby
page, and a key stored where this copy cannot read it.

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
               nl_graph (the composing reader), agent_v5 (graph, memory and registry follow-ups),
               V6: intent (one schema and seven stages for every request), evalset (150 written
               requests with their expected readings)
  core/        calendar, Black-Scholes / American pricing, causal indicators
               V6: instance (one machine at a time opens a data folder: the lease, standby, the claim protocol)
  data/        store (datasets, manifests), migrate, catalog, planner (requirements, coverage,
               selective and adaptive acquisition), importers, bulk (streaming imports),
               providers/ (free sources, paid adapters, bench = live verification, market =
               the provider marketplace, cost estimates and V6 budget table, access = what an
               account's key actually returns), V6: readiness (the five capabilities and the
               data check before a run), quality (volume artefacts)
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
               V6: usecase (the original question on stored observations)
  audit/       V6: reference (an independent simulator that shares no code with the engine),
               bars (missing minutes, early closes, DST, volume artefacts, mixed feeds, two feeds against
               each other), options (stored option bars against the calendar, against the provider's
               individual trades, and one provider against another)
  collector/   the background market-data collector
  backup.py    V6: `strata backup` / `strata restore`
  bench.py     the `strata bench` command
  analytics/ · optimize/ · portfolio/
  api/         FastAPI app, research_api, v4_api, v5_api, v6_api, ide (strategy IDE: compact form, section
               edits, debugger), launch (jobs + caching), settings (credential stores), notify, db
  validate.py  the `strata validate` command
backend/tests/ unit suite, test_live.py (opt-in), test_reproducibility.py, e2e_seed.py
frontend/src/  pages/lab (research workspace, Canvas, Research and Graph views), pages/explore
               (Patterns, Findings, MlLab), pages/build (Ide, CodeEditor), pages/validate (Forward,
               Registry), pages/research (Sensitivity, Encyclopedia), pages/data (Collector,
               Marketplace, bulk import, Readiness), pages/builder, pages/results …
               pages/lab/Coverage.tsx: the coverage block, stage rail, data check, fills, budget
benchmarks/    baseline.json, history.jsonl (backend), frontend.jsonl (browser render times)
frontend/e2e/  browser tests            .github/workflows/ci.yml
scripts/       setup.sh · dev.sh · init_db.sh · docker-entrypoint.sh
```

## Known limitations
- **The strategy this project started from has not been backtested.** One of its 105 qualifying
  sessions can be priced by the engine on stored observations and one more by the reference
  simulator. Everything else waits on intraday option prices that are not stored: see
  [What would close the gap](#what-would-close-the-gap). No key-gated provider has been reached
  from this installation, so their coverage is what they publish, not what was measured.
- **The reader is right about three times in four on wording it has not seen**, and before the
  V6 guards one answer in five was wrong without saying so (the held-out first run). Read the
  "How I read this" card; the Interpretation panel is editable for a reason.
- **The reference simulator is a second opinion, not a second engine.** It covers option
  positions on one underlying with the exit rules in its docstring. Agreement with it is evidence
  for those cases only.
- **One copy at a time on a data folder.** A process on the computer and one in a container do not
  see each other's file locks or SQLite's shared memory, so they cannot share a database. The lease
  in `<data>/instance.json` enforces it for every copy that knows about it (version 6 on): the
  second copy stands by. A container built from an older version does not know the lease and must
  be rebuilt or stopped. The application and the collector may run together when both are native
  or both in Docker.
- **Free intraday option history is shallow.** Yahoo serves option trade bars only for contracts
  still listed or expired within about a day, and its bars behave like samples of the last price;
  Alpaca's free plan reaches back to 2024-01-18 on trade bars, with seven early sessions incomplete
  at the provider. Before that, intraday option backtests need a paid provider or imported files.
  A request that reaches further back is told so. A rule that fires a few times a year (the
  original 3% rule: five sessions since 2024) cannot be tested on this history however complete it is.
- **The stored option set is a band, not the whole chain.** The bulk download holds contracts
  within 3% of the previous close expiring within 7 days. After a gap open the at-the-money strike
  can lie outside it; the lab and `strata usecase --acquire` fetch the contracts around the price
  at the signal for the sessions a strategy needs, and a contract that is not stored is reported
  as not stored, never priced another way.
- **Trade prints are not quotes.** A trade-priced backtest cannot know the spread that was
  available. It refuses stale prints and reports executability, but it is still an
  approximation of what could have been filled.
- **Not exercised against live endpoints here:** the paid-provider adapters (Massive, Databento,
  ThetaData, ORATS) and the language-model providers (mocked transport plus opt-in live tests).
  Alpaca and the macOS Keychain were exercised for real on 2026-11-25; Alpaca's coverage for QQQ
  and IWM was not measured.
- **Two free sources disagree about option minutes.** Where Alpaca and Yahoo hold the same contract
  and minute, the close differs on 45% of them. Alpaca's bars reproduce from its own trade records
  and are used first; no third source was available to settle which is closer to the tape.
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
