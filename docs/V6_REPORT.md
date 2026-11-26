# Strata V6: delivery report

Written 2026-11-25, after a free Alpaca paper key was connected, the access test was run with it
and the history it serves was downloaded and checked. Every figure is counted from files on the
development machine or was returned by the running application that day. Coverage that was not
measured here is labelled as the provider's own description.

Short version: **the access problem is solved for January 2024 onward and unsolvable for free
before it.** Alpaca's free account returns one-minute trade bars for every expired SPY contract
that was asked for; 43.4 million of them are now stored and verified against the provider's own
trade records. The original strategy is priced on every session it produced in that period, and
that is **5 sessions**: the rule is too rare for 34 months of history to test it.

## The thirteen questions

**1. What was the exact historical data-access blocker?**
Observed intraday option prices for the sessions the strategy trades on. The signal needs only
daily bars, which go back to 1993. An intraday fill needs an option price observed in that minute,
and the only free source that had been reached (Yahoo) serves a contract's bars while it is listed
and for about a day after it expires. No Alpaca key was stored, so every Alpaca endpoint answered
401. Once the key was in, the blocker for 2024-01-18 onward was gone. Before that date no free
source serves intraday option history at all; that part is a matter of money or of the prices not
existing (no same-day SPY expirations outside monthly expiration days before 2010).

**2. Which providers were authenticated?**
Alpaca, with a paper key stored in the macOS Keychain. The key is accepted by the paper host;
the account has no OPRA agreement (real-time OPRA quotes answer 403), which history does not need.
Nothing else needs or has a credential: Yahoo, Cboe, the end-of-day archive, DoltHub and the public
OptionsDX sample were reached without one. Massive, Databento, ThetaData, ORATS and every language
model remain without credentials.

**3. Which providers returned actual expired-contract observations?**
Alpaca. The access test asked for 100 expired contract-sessions on ten dates from 2024-02-14 to
2026-11-18 (0DTE at and around the money, one week out, one month out; calls and puts) and all 100
returned one-minute trade bars; all 100 expirations were found by the listing endpoint, none had
to be constructed. Individual trades are served too. Yahoo returned nothing for a contract expired
more than a day.

**4. What is the earliest verified intraday option date?**
**2024-01-18**: every session from 2024-01-02 was asked for 18 contracts around the money and none
before 2024-01-18 returned a bar. The first weeks are damaged at the provider: 2024-02-02 returns
nothing at all, 2024-01-23 stops at 10:48 New York time, and 2024-01-22, 01-24, 01-29, 02-26 and
03-15 start between 09:48 and 12:55. Those seven sessions are stored, recorded as partial and never
priced. (One vendor sample session of bid/ask from 2020-03-06, 09:30 to 12:00, is also stored.)

**5. How many SPY historical intraday sessions are usable?**
- Underlying, 1-minute bars from the consolidated tape: **2,740 whole sessions**, 2016-01-04 to
  2026-11-25 (2,741 stored; 2019-08-12 stops at 15:29 and is treated as partial). 1,065,106 bars.
- Options, 1-minute trade bars: **711 whole sessions**, 2024-01-18 to 2026-11-25 (718 stored, the
  7 above partial). 43,364,573 bars over 331,147 contract-sessions, 310,967 of which traded.
  Stored: every contract within 3% of the previous close expiring within 7 days, calls and puts,
  on every session; plus the contracts the original strategy and its 1-day and 7-day variants
  could select around the opening price on each of their qualifying sessions.
- Options, intraday bid/ask: 0 whole sessions, 1 partial.

**6. How many QQQ and IWM sessions are usable?**
Both were downloaded from Alpaca after SPY, with the same commands, and checked the same way.

| | QQQ | IWM |
| --- | --- | --- |
| Underlying, 1-minute bars | 2,741 whole sessions, 2016-01-04 to 2026-11-25 (1,065,095 bars) | 2,740 whole sessions (2019-08-12 stops at 15:29, as for SPY); 1,065,103 bars |
| Options, 1-minute trade bars | 711 whole sessions of 718, 2024-01-18 to 2026-11-25; 34,403,727 bars over 300,586 contract-sessions | 711 whole of 718; 10,488,629 bars over 134,700 contract-sessions |
| Same-day expirations | on all 718 sessions | on 693 of the 718 (none on 25 early sessions) |
| Stored bars against individual trades | 24 contract-sessions, 36,488 trades: 4,113 of 4,113 bars identical | 24 contract-sessions, 33,186 trades: 3,939 of 3,940 identical |
| Baseline: ATM 0DTE call at 10:00, out at 15:30, every session | 702 sessions traded, -$5,194.16, 251 winners | 683 sessions traded, -$2,171.84, 236 winners |

The seven sessions that are incomplete for SPY are incomplete for QQQ and IWM too: the holes are in
the provider's feed. Neither symbol has intraday bid/ask or end-of-day option chains stored.

**7. What is the resolution of each dataset?**

| Dataset | Symbol | Observation | Resolution | Sessions | Period |
| --- | --- | --- | --- | --- | --- |
| yahoo underlying | SPY, QQQ, IWM | trade bars | daily | 8,515 / 6,973 / 6,664 | 1993 / 1999 / 2000 to 2026-11-25 |
| **alpaca underlying** | SPY, QQQ, IWM | trade bars, consolidated tape | 1 minute | 2,741 each | 2016-01-04 to 2026-11-25 |
| yahoo underlying | SPY, QQQ, IWM | trade bars, feed not documented | 1 / 5 / 60 minutes | 21 / 61 / 729 | from 2026-10-27 / 2026-08-31 / 2023-12-27 |
| cboe | VIX | index level | daily | 9,324 | 1990-01-02 to 2026-11-25 |
| github_eod | SPY | bid/ask quotes | end of day | 1,495 | 2020-01-02 to 2025-12-12 |
| **alpaca options** | SPY, QQQ, IWM | trade bars | 1 minute | 718 each (711 whole) | 2024-01-18 to 2026-11-25 |
| yahoo_options | SPY, QQQ | sampled last prices (see 8) | 1 minute | 7 / 2 | 2026-11-16 to 2026-11-24 |
| optionsdx (public sample) | SPY | bid/ask snapshots | 5 minutes | 1, partial | 2020-03-06 |

**8. Are observations trades, quotes, indicative prices, or modeled prices?**
- Alpaca option bars are **trades**. 32 stored contract-sessions from 2024-02-22 to 2026-11-13,
  quiet and busy contracts alike, were fetched again as 141,605 individual trades and rebuilt into
  one-minute bars by separate code: 5,558 of the 5,559 stored bars have the same open, high, low
  and close (99.98%), 5,290 also the same volume (95.16%); no stored bar lacks a trade and no
  traded minute lacks a bar. There is no bid and no ask, and none is shown or assumed.
- Yahoo's option bars turned out to be **closer to samples of the last price than to trade bars**.
  On the 481 SPY contract-sessions both providers hold (2026-11-23 and 11-24, 73,317 common
  minutes): the minute's close is identical on 54.9%, all four prices on 27.0%; 65.6% of Yahoo's
  bars hold a single price against 25.7% of Alpaca's; the 09:30 bar exists for 42 contracts in
  Yahoo and 388 in Alpaca; daily volume agrees within 3%. Alpaca's bars are used first.
- github_eod and optionsdx are quotes. Alpaca's *latest* quotes on this plan are indicative and are
  not stored. Nothing modeled is used unless a run is explicitly switched to theoretical premiums.

**9. Which intraday strategies were successfully backtested?**
On real observations, through the engine:

| Strategy | Sessions the rule fired on | Priced | Net result, one contract | Worst to best print | Winners |
| --- | --- | --- | --- | --- | --- |
| The original: ATM 0DTE call at the open after a fall of 3% or more, out at +1% in SPY or 15:55 | 105 since 1993; **5** since 2024-01-18 | 5 | -$892.50 | -$1,288.50 to -$716.50 | 2 of 5 |
| The same after a fall of 4% or more | 49; 4 | 4 | -$475.20 | -$714.20 to -$343.20 | 2 of 4 |
| The same after a fall of 2% or more | 342; 16 | 16 | -$81.80 | -$919.80 to +$429.20 | 8 of 16 |
| The same after a fall of 1% or more | 1,079; 75 | 75 | +$2,384.50 | -$962.50 to +$4,268.50 | 38 of 75 |
| Baseline: ATM 0DTE call at 10:00, out at 15:30, every session | 718 | 707 | -$376.14 | not computed | 269 of 707 |
| The same baseline on QQQ | 718 | 702 | -$5,194.16 | not computed | 251 of 702 |
| The same baseline on IWM | 718 | 683 | -$2,171.84 | not computed | 236 of 683 |

"Worst to best print" restates a result with every fill taken at the worst, then the best, price
the contract traded at in the minute the fill came from. Trade bars carry no bid or ask, so this is
the only observed measure of how far a real fill could have been from the print used; in the
opening minute of a 0DTE contract it is wide (2026-11-23, 722 call: 4.89 to 6.48).

The five sessions of the original strategy:

| Session | Fall the day before | Entry | Exit | Net |
| --- | --- | --- | --- | --- |
| 2025-04-04 | -4.93% | 09:31, 525 call at 4.54 | 15:55 at 0.01 (time) | -$454.30 |
| 2025-04-07 | -5.85% | 09:31, 488 call at 7.64 | 09:57 at 9.82 (SPY +1%) | +$216.70 |
| 2025-04-11 | -4.38% | 09:31, 524 call at 6.22 | 12:33 at 7.25 (SPY +1%) | +$101.70 |
| 2026-11-06 | -4.41% | 09:31, 736 call at 3.65 | 15:55 at 0.27 (time) | -$339.30 |
| 2026-11-23 | -3.64% | 09:31, 722 call at 4.97 | 15:55 at 0.81 (time) | -$417.30 |

Five trades establish nothing about the strategy. The 1% variant has 75; its sign depends on which
print of the entry minute is assumed, so it is not evidence of an edge either. The P1 goal of a
*meaningful* intraday backtest is met for rules that fire often (the baseline: 707 sessions in 100
seconds) and **not met for the original rule**, which fired five times in the only period any free
intraday option history covers.

**10. Which results were independently verified against source records?**
- Qualifying sessions at 1%, 2%, 3% and 4%: plain arithmetic on daily closes equals the engine's
  scan (1,079 / 342 / 105 / 49), and the sentence compiles to the hand-written configuration.
- Every priced trade: the engine and a reference simulator that shares no code with it agree on
  contract, times, prices, fees and P&L on **100 of 100** trades across the four thresholds.
- Option bars against the provider's individual trades: answer 8.
- Underlying minute bars: 2,712 sessions hold 390 bars, the 21 early closes 210, the four sessions
  with a market-wide halt in March 2020 hold 376 to 386; the last bar is within 0.05% of the
  official close on 2,697 of 2,740 sessions; on the 21 sessions Yahoo also holds, all 8,190 bars
  match by start time and every close is within 0.02%.
- The underlying study (68 of 105 sessions rose 1% from the open): the 35 qualifying sessions
  since 2016 were recomputed from minute bars and agree with the daily record on every one (19).
- The 2026-11-23 trade reported earlier from Yahoo's bars (721 call, 09:32 at 5.50, -$360.30) is
  **superseded**: Alpaca's trade bars show prints in the opening minute that Yahoo's bars lack, so
  the fill is at 09:31 on the 722 call at 4.97, -$417.30. Both are in the report, side by side.

**11. Which providers or workflows remain blocked?**

| Blocked | On what |
| --- | --- |
| Intraday option prices before 2024-01-18 | money or nothing: see 13 |
| Massive (Polygon), Databento | a key each (Databento: an account with its $125 credit) |
| ThetaData, ORATS | a subscription and, for ThetaData, its terminal |
| Model-backed reading of requests, and its evaluation | a model key or a local Ollama |
| UW data (WRDS TAQ, Bloomberg) | a UW login, an instructor's or Foster's approval for WRDS, and a manual export |

**12. What credentials or subscriptions do I need to supply?**
None for what is done. For more: a Databento account (free, with $125 of credit) to reach
2013-2023 on bid/ask, and $24 of OptionsDX files to reach 2010-2012. A model key only if you want
requests read by a language model instead of the built-in reader.

**13. What is the cheapest realistic path to substantially more coverage?**
For the original strategy, 38 of its 105 sessions can be reached by any source at all (10 had no
same-day contract; 57 fall before 2010).

| Step | Cost | Sessions reached | Evidence |
| --- | --- | --- | --- |
| stored now (Alpaca trade bars) | $0 | **5**, plus 1 partial sample session | measured |
| Databento new-account credit, 1-minute bid/ask | $0 (about $0.02 of $125) | up to 27 | provider's description and unit prices; never run |
| OptionsDX 30-minute chains for 2010-2011, 2015-2016, 2018-2020, 2022 | $24 | up to 38 | storefront prices; the importer was verified on the vendor's sample |
| $50 to $200 | - | nothing more for this strategy | - |

Even all 38 would be a small sample. The more useful path is the one now open: strategies that
fire often can be tested on 711 sessions today, and the collector adds each new session for free.

## What running it for real found

None of these was visible while the key-gated code had only run against recorded doubles.

| Found | Effect | Fix |
| --- | --- | --- |
| The key you saved first went into a two-day-old container on port 8000 (the V2 build) sharing the data folder with the current app. SQLite's write-ahead log is not coordinated across the container boundary, and the write was dropped a minute later | a saved credential silently lost; the same could have happened to any run | One machine at a time opens a data folder: a lease file with a heartbeat; the other copy goes on standby, says who holds the folder and where to open it, and takes over by itself. A claim must go unanswered for longer than a heartbeat, so a live holder always wins |
| The route that serves the interface returned **any file on disk** for a path with encoded `..` (`/..%2f..%2fbackend%2fpyproject.toml` came back) | anyone who could reach the port could read the database or a stored key file | Only files inside the build directory are served; unknown `/api` paths are a 404 |
| Inside Docker the computer's own browser arrives from the container network, so **every request was refused as remote**. The container build has been unusable since V3; port 8000 was still serving V2 | the documented Docker start did not work | The compose file publishes on 127.0.0.1 only and declares it; the Host name must still be local |
| A running copy remembered "no key" for its whole life, and the native app and a container read different credential stores | "I saved it and it says not set" | An absence is remembered for 3 seconds; the native app also reads the protected file a container writes; where a key is stored (never the key) is recorded so the other side can say so |
| Alpaca lists `multiplier: "0"` for every contract that expired on or before 2024-04-26 (5,584 of 76,080 definitions) | the first full run multiplied those trades by zero, then divided by it | Size and deliverable are read; the engine refuses a non-positive multiplier; stored definitions repaired from the cached listings |
| Alpaca holds seven early sessions only in part | an entry would slide to the first print the provider happens to have | Recorded as partial; coverage, readiness and the engine leave them out and say why |
| The rule that removes Yahoo's running-total volume flagged 32 real block prints on the consolidated tape (SPY, 2018-09-21 10:43: 18.7 million shares) | real volume dropped from VWAP on expiration Fridays | Applied only to feeds that produce the artefact |
| On 70 of 2,741 sessions the daily high or low lies outside every regular-session minute bar, 65 of them at about the previous close | the audit said prints were missing from the bars | A daily bar counts late and reference-priced trade reports; the wording says so, and the engine takes a session's range from the bars |
| A band around the previous close misses the at-the-money strike after a gap open (2026-11-23: close 689.81, open 724.33) | the bulk download alone could not price that session | `strata usecase --acquire`, and the lab, fetch the contracts around the price at the signal |
| The engine expanded every session's option data even when nothing traded | the lab took 98 seconds to price 5 sessions | A session's chain is built on first use; the 100 trades of the use-case report are identical before and after |
| Port 8000 on this Mac is also used by another local project on 127.0.0.1 | `localhost:8000` showed a mixture of two applications | This machine's container is published on **http://localhost:8010** (`.env`, not committed) |

## Acceptance

| | Status |
| --- | --- |
| Software verified | 482 backend tests (12 opt-in live ones skip), 115 of them in thirteen V6 files; 24 browser tests; `strata validate` 14 of 14 on freshly downloaded observations; `strata bench --check` without regression |
| Data access verified | Alpaca (stock minute bars since 2016-01-04, option trade bars and trades since 2024-01-18, expired contracts included), Yahoo, Cboe, the end-of-day archive, DoltHub, the OptionsDX sample importer |
| Data access BLOCKED | Massive, Databento (credentials); ThetaData, ORATS (subscription); UW (login and permission); a language model (key) |
| Research verified | the underlying study on 105 sessions (68 reached +1% from the open, 64.8%, 95% interval 55.3% to 73.2%, against 17.3% of all sessions; the 35 sessions since 2016 confirmed on minute bars); the option strategy's five trades, engine against reference |
| Research NOT established | whether the original strategy is profitable: 5 of 105 sessions priced, and no more exist in any free history |

The eight acceptance prompts were sent to the running application over HTTP on 2026-11-25 with the
key connected (notebook "How often does SPY rise 1% from the open after a 3% down day?"; the runs
carry the engine label 6.0.0, which was raised to 6.0.1 afterwards with no change to the engine in between):

| Prompt | What the application answered |
| --- | --- |
| How often does SPY rise 1% from the open after a 3% down day? | 68 of 105 sessions (64.8%, 95% interval 55.3% to 73.2%) against 17.2% of all 8,513 sessions |
| Test buying ATM SPY 0DTE calls on those days. | Before running: 5 of 105 can be priced, 89 have no stored option prices, 10 had no such contract, 1 is partly observed. Result: 5 trades, -$892.50 |
| Run the same strategy after 1% and 2% declines. | 75 of 1,079 priced, +$2,384.50; 16 of 342 priced, -$81.80; "a description of what happened, not evidence" |
| Compare 0DTE, 1DTE, and 7DTE calls. | 5 of 105 priced for each; -$892.50 (0DTE) to -$469.50 (7DTE); all flagged as fewer than 30 observations |
| Enter at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move. | 5 trades, -$604.50 |
| Show every actual option fill and its source. | five fills, each an observed trade print from `alpaca/SPY/options/trade/1m/v1`; no bid or ask shown |
| Show exactly which historical sessions could not be tested. | 100 of 105: 89 no intraday option prices stored, 10 no contract listed, 1 partly observed |
| Estimate what it would cost to acquire the missing observations. | 57 sessions before 2010 unreachable; up to 27 with Databento's free credit; up to 38 with $24 of files |

These runs are recorded in the experiment registry like any others.

## Sanitized request diagnostics: Alpaca

Access test of 2026-11-26T06:18:50Z, about 140 read-only requests, 200 allowed a minute, none
refused for rate. No key, token or account identifier is in the record.

| Asked for | Endpoint | Status | Classified as |
| --- | --- | --- | --- |
| contract listing | paper-api.alpaca.markets/v2/options/contracts | 200 | ok (the key belongs to the paper environment) |
| stock minute bars 2019-06-03, feed=sip | data.alpaca.markets/v2/stocks/bars | 200 | ok, 390 of 390 minutes |
| stock minute bars 2015-12-31 / 2016-01-04 | data.alpaca.markets/v2/stocks/bars | 200 / 200 | no data / ok: the history starts 2016-01-04 |
| latest option quote, feed=opra | data.alpaca.markets/v1beta1/options/quotes/latest | 403 "OPRA agreement is not signed" | entitlement |
| latest option quote, feed=indicative | same | 200 | ok (derived values; not stored) |
| option bars, ten dates, 10 contracts each | data.alpaca.markets/v1beta1/options/bars | 200 | ok, 100 of 100 contract-sessions |
| option trades, same | data.alpaca.markets/v1beta1/options/trades | 200 | ok |
| option bars, every session 2024-01-02 to 2024-01-17 | same | 200 | no data |
| option bars, 2024-02-02 | same | 200 | no data: a hole in the provider's history |

Before the key existed the same endpoints answered 401 and none 404, so the addresses and
parameters were right and only the key was missing.

## Not done

- A meaningful test of the original 3% rule: the observations do not exist in any free history.
- Intraday option history before 2024-01-18 (a Databento account, or $24 of files).
- Any measurement of Massive, Databento, ThetaData, ORATS or a language model.
- A third source to settle which of Alpaca's and Yahoo's option bars is closer to the tape for the
  minutes where they differ (Alpaca's reproduce from its own trades; Yahoo publishes no trades).
