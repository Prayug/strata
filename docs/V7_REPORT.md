# Strata V7: delivery report

Written 2026-11-26. Every figure below was counted from files on the development machine or returned by the
running application that day. Where something was not measured it says so. Evidence files are in
`docs/v7_evidence/`; the acceptance runs are notebooks in the application and can be opened there.

**Short version.** The stored history (89.1 million one-minute option trade bars for SPY, QQQ and IWM) can now be
worked on from one screen: type a strategy, a comparison or a question, see how it was read, see how much of it the
stored data can price, run it, open any trade over the stored prints, check every fill against the bar it came
from. Five things the work found matter more than the features:

1. **You asked the application a question three times while this was being built, and it answered a different one
   each time.** "How often does SPY's first 9:30 candle predict its overall day?" was run as a study with no
   condition and no outcome, and came back as the share of all sessions that closed higher, with nothing flagged.
   The correction you typed ("no I meant 6:30a, PST") got a list of example prompts. Both are fixed and the question
   now has a real answer from the stored bars (section 5). It is the clearest evidence in this report of what
   finding 2 means in use.
2. **Sentences nobody tuned the readers on are read correctly about half the time.** On 34 sealed sentences the
   built-in readers got 16 right and answered a *different question without saying so* on 9. Two small local models,
   reading through a form of typed fields, got the same 16 right with 4 silent misreadings. With both reading and
   every disagreement asked before anything runs, silent misreadings fell to 2 or 3 of 34. That mode is now the
   default when a local model is selected. No hosted model was measured: there is no key.
3. **Nothing here measures the bid/ask spread.** The quote comparison is BLOCKED on a Databento key. What was
   measured instead is the first print after each decision, which moved results a few percent in both directions.
4. **No configuration that was tested shows evidence of an edge.** The best cell of every grid stops being
   distinguishable from zero once the number of cells is counted, and what ranked best in the earlier part of the
   history did not rank best in the later part.
5. **Running the application for real found defects that tests on fixtures had not**: an exit condition read as an
   entry condition as well, a strategy that kept opening positions after its own exit time, a planner that asked
   the provider about data already on disk (minutes of waiting before a nine-second run), a run whose summary said
   74 sessions after it had traded 75, a contract dated on a day the market was closed that never expired and put
   $12,823 into a result none of its trades had made, and a local model server that loaded a 6 GB model as 13 GB.
   Each is fixed and has a test.

## What was tested live, with fixtures, and what is blocked

| Area | Live, on the stored dataset or the real service | Fixtures only | Blocked |
| --- | --- | --- | --- |
| Collector (new sessions for SPY, QQQ, IWM) | From an empty directory: 6 of 6 symbol-sessions identical to the archive. Real configuration switched on; one cycle ran | Missed days, budgets, deferrals, recovery | A session that has not happened yet: the next cycle is due 2026-11-27 16:20 New York time |
| On-demand contracts (0-90DTE, exact dates, deep ITM/OTM, delta, premium, gaps) | 13 cases against Alpaca; a second request fetched nothing | Selection rules on hand-written chains | |
| Other symbols (DIA, AAPL, NVDA, TSLA, MSFT, META) | One-session probe of each. Nothing stored | | |
| Strategy readiness | Every acceptance run reports its own coverage | Partial sessions, missing expirations and strikes, gaps, stale prints, multi-day positions | |
| Historical bid/ask (Databento) | | Adapter, parser, cost arithmetic, comparison | **BLOCKED: no key.** The adapter has never run against the live service |
| Individual trades (Alpaca) | Four runs, two exit policies | Latency, conditions, size, leg timing, order of target and stop | |
| Sentences to strategies, grids and questions | The acceptance sentences and the question you typed, through the running application | 19 development sentences, compared by where the rule fires on stored sessions | |
| Language models | Two local models (gemma3:4b, qwen2.5vl:7b) on 34 sealed sentences | Routing, grounding, cross-check | **BLOCKED: no key** for Anthropic, OpenAI, Gemini, OpenRouter |
| Replay, tear sheet, fills, experiment grid | In the browser on real runs; 28 browser tests on a fixture store | | |
| Forward validation | Incremental evaluation equal to a continuous run on real data; a paper experiment frozen today | Restarts, revisions, engine changes | Forward sessions: none exists yet for an experiment frozen today |
| Benchmarks | The stored dataset itself, each workload in a fresh process | The reproducible fixture suite (`strata bench --check`) | |

## 1. The dataset: kept, continued, extended on demand

Nothing was downloaded again. `GET /api/dataset/options` reads the index of each stored dataset:

| Symbol | Sessions | Period | Contract-sessions (with a trade) | One-minute trade bars |
| --- | --- | --- | --- | --- |
| SPY | 718 | 2024-01-18 to 2026-11-25 | 335,210 (314,171) | 43,757,048 |
| QQQ | 718 | 2024-01-18 to 2026-11-25 | 303,311 (285,734) | 34,800,936 |
| IWM | 718 | 2024-01-18 to 2026-11-25 | 135,863 (130,618) | 10,590,036 |

89,148,020 bars at the end of this work, 15.5 bytes a bar on disk: the 88.3 million of the V6 download, the 728,410
of the gap top-up below, and 122,477 fetched on demand since for the contracts strategies asked for. They are trade
bars: no bid, no ask.

**Does the collector add new sessions by itself?** It did not: it was off, set to SPY and QQQ on Yahoo. It now reads
what the archive holds (symbols, strike range, expirations) and continues exactly that. Checked from an empty
directory holding only the underlying history: one cycle collected the last two sessions of all three symbols, and
every bar of every contract it stored is identical to the archive's (`collector_fresh_directory.json`):

| | 2026-11-24 | 2026-11-25 |
| --- | --- | --- |
| SPY | identical | 484 contracts, 78,603 bars: identical |
| QQQ | 415 contracts, 57,611 bars: identical | 486 contracts, 68,210 bars: identical |
| IWM | 168 contracts, 19,067 bars: identical | 176 contracts, 17,229 bars: identical |

The archive holds a few more contracts on those days than the collector asks for: the ones fetched on demand since.
**The real collector is now switched on** (SPY, QQQ, IWM; Alpaca; every listed contract within 3% of the previous
close expiring within 7 days). One cycle ran: 21 requests, nothing new, because the archive was already current
(2026-11-26 is a market holiday). With this source a missed day costs nothing: expired contracts stay available and
the next cycle collects every session since the last stored one, oldest first, whole sessions only. It still cannot
collect while the computer is asleep or the application is not running.

**The strike range follows a gap at the open, and nothing later.** The stored range was 3% around the previous
close. After a large overnight move the opening price can sit at its edge. The range is now widened so that it
reaches half the band beyond the session's *opening* price, never by where the session went afterwards (that would
choose contracts with hindsight). Topping up the three archives for it took 117 requests and added 728,410 bars
(`gap_topup_totals.txt`). Intraday the price still leaves the stored strikes on some sessions; those are fetched
when a strategy asks.

**A leg is filled with the contract it asked for, or not at all.** The first live run of the expansion matrix found
three silent substitutions: a 30DTE leg trading a stored 7DTE contract, an 8% in-the-money call filled 49 strikes
away, and a 20-delta put filled at a 0.52 delta. On selectively stored data the engine now refuses those
(`contract_not_stored`, `no_recent_trade`) and the planner fetches what is missing. The matrix after the fix
(`on_demand_expansion_after_stricter_selection.json`), each case on 7 signal sessions unless stated:

| Case | Stored before | Fetched | Traded after |
| --- | --- | --- | --- |
| ATM 0DTE, ATM 30DTE, 30-delta 7DTE, 10-delta 30DTE put, $0.50 premium 0DTE | everything | nothing | 7 of 7 |
| ATM 90DTE | 20 of 42 contract-sessions | 22 (6 requests) | 6, 1 with no print near the exit |
| 5% OTM 7DTE | 44 of 49 | 5 | 7 |
| 8% ITM 7DTE | none in range | 16, after two widening rounds | still short: deep in-the-money weeklies rarely print, and that is what the run reports |
| An exact expiration date (2026-12-18) | everything | nothing | 6, 1 uncovered exit |
| QQQ ATM 0DTE call on the +7.8% gap day; SPY ATM put after a 3.9% intraday fall | 7 of 7; 4 of 6 | nothing; 2 | 1; 1 |

A second request for the same contracts fetched nothing in every case.

**Other symbols** were probed for one session each and not stored (`further_symbols_probe.json`). All six return
expired contracts from the same endpoint. A year at the same band would be about 8 MB for DIA, 34 for AAPL, 24 for
NVDA, 46 for TSLA, 26 for MSFT and 33 for META, against 278 for SPY. Stocks list Monday, Wednesday and Friday
expirations with 2.50 strikes; DIA lists weeklies only. A pre-split NVDA session needed the as-traded strike range
(fixed: 50 contracts on 2024-01-19 where the split-adjusted range found none).

## 2. Readiness is decided per signal, with the engine's own rules

`backend/strata/data/readiness.py` used to count a session as usable when it had some option data. It now takes
every signal through the checks the engine itself makes: is the session stored and whole, is the expiration the leg
would choose stored, is the contract inside the stored strikes, did it print inside the entry window, is the exit
window covered. A plan says "262 pass every check ... 3 have the contract stored but no trade inside the entry
window, 1 needs a strike outside the stored range, 1 is held only in part by the provider", and a run then confirms
each one. Sessions that can only *potentially* be priced are never called executable. Regression tests cover partial
sessions, missing expirations, missing strikes, large opening gaps, stale prints and multi-day positions
(`tests/test_v7_readiness.py`, `tests/test_v7_expansion.py`).

The same determination now decides what the provider is asked for. Before, the contracts of *every* signal were
looked up at the provider even when all of them were on disk: a strategy whose data was entirely stored waited two
to four minutes for a run that takes nine seconds. Only the signals the stored observations do not fully price are
sent to the acquisition step now. The first acceptance sentence went from 48 seconds to 6.

**A defect the acceptance run found here.** On 2024-12-19 the contracts had been fetched around SPY's price at the
signal; by the first print the price had moved four strikes, and the strike the entry then needed was one below the
fetched range. The application asked for a go-ahead on "74 of the 75 qualifying sessions", then fetched the missing
strike while simulating, traded all 75, and reported them under a summary that still said 74 could be priced and
"the remaining 1 could not be tested". Nothing was mispriced, but two statements about one run disagreed. Now a
strike outside the stored range is fetched for those sessions *before* coverage is reported, so nobody is asked to
give up a session that could simply be fetched; and a run that does widen the range while simulating counts its
coverage again from the store as it then stands. With downloads off the session is reported as not priced and
nothing is substituted (`tests/test_v7_expansion.py`, two tests through the whole pipeline with a provider class
that records what it is asked for).

## 3. Historical bid and ask: BLOCKED, and exactly what is needed

No Databento key is stored, so nothing below has run against the live service. Until it has, what the adapter
would retrieve is Databento's documentation, not a measurement.

What exists: an adapter for OPRA consolidated quotes sampled each minute (`cbbo-1m`), with a staged check
(`strata access databento`, or Data > Readiness > Historical bid and ask). The free stages verify the key, the
dataset, its date range and unit prices, then ask Databento itself for the price of a small sample
(`metadata.get_cost`). The sample is downloaded only when a budget is typed that covers that quoted price, and never
more than it; the remaining credit has no API, so Strata tracks what its own requests were quoted. After a download
it checks contract identifiers and clocks against the overlapping Alpaca trade bars, then re-prices the same
strategy on quotes (conservative and midpoint) beside the trade-print result. No spread is ever inferred.

To unblock it:
1. Open https://databento.com and choose Sign up.
2. Databento states that new users receive $125 of free credit for historical data, applied before any charge and
   expiring six months after it is issued.
3. Sign in and open the API keys page of the portal (https://databento.com/portal/keys). The key is a 32-character
   string that starts with `db-`.
4. Open the Billing page (https://databento.com/portal/billing). Databento's documentation says the remaining
   credit is shown there and that a monthly usage limit can be set, beyond which further historical requests are
   blocked. Set it to an amount you accept. Strata cannot see either figure.
5. In Strata: Settings > Data provider keys > Databento > Save.
6. Run the free checks. They end with Databento's own price for the sample.
7. If that price is acceptable, type a budget that covers it and download the sample.

Databento's documentation also says OPRA data before 2023-03-28 was built from subsampled data rather than packet
captures. The stored Alpaca history starts 2024-01-18, so the overlap is not affected.

## 4. Execution on individual trades

A minute-bar backtest fills at the last print it has seen, which happened *before* its decision. The execution
check fills the same decisions on the first eligible individual print strictly *after* each one, from Alpaca's
trade tape, asking only for the minutes around each fill and keeping what it receives. It models: latency; a
maximum wait; the exchange's trade-condition codes (single-leg prints by default; multi-leg and late prints
excluded); a minimum size; partial fills against print size; a delay after the opening bell; the gap between the
legs of a spread; and two exit policies. With `decision_clock` the exit is the first print after the engine's exit
decision. With `resting_orders` a target or stop rests in the market and fills when a print trades through it,
which also settles which of the two came first when both levels fall inside one minute.

A print is still not a quote. It says somebody traded there; it does not say this order would have.

Measured (the acceptance section has the runs made through the application today):

| Run | Trades | Minute bars | First print after the decision | Note |
| --- | --- | --- | --- | --- |
| SPY 0DTE call after a 3% down day | 5 | −$892.50 | −$978.50 | entries a median 0.7 s after the decision, 1.05% worse |
| SPY 0DTE call after a 1% down day | 75 | +$2,384.50 | +$2,496.50 | 75 of 75 repriced; entries a median 0.44 s after the decision, 0.33% worse on average; the same with resting orders |
| SPY ATM 0DTE call, +30% / −15% premium, one month | 23 | −$399.90 | −$408.90 | as resting orders: −$273.40, one exit reason changed |
| QQQ, two 0DTE calls on a VWAP reclaim, half off at +25%, 15% premium stop (acceptance F) | 254 | +$775.60 | +$557.60 | as resting orders: +$249.60, 14 positions left for a different reason |

The last row is the one to read. A strategy with a tight premium stop and a partial target is decided by which print
fills it: the same 254 decisions are worth $775.60 on minute bars, $557.60 on the first print after each decision,
and $249.60 when the stop and the target rest in the market from the fill. Its tear sheet says the same thing
another way: the result reaches zero at fills 0.4% worse. None of the three figures includes a spread.

Once the prints are in the provider cache the check itself costs 0.4 seconds for 23 positions and makes no request.

### The account and its trades

While reading the acceptance figures one of them did not add up: a 1DTE variant reported a net result of $19,796.03
over 74 trades whose results sum to $6,972.68. The cause is in the calendar. 2025-01-09 was a national day of
mourning; the market was closed, and the contracts dated that day expired at the close of the 8th, keeping their
symbols. The engine settled a leg when the session's date equalled the contract's date, a day that never came. A
call dated the 9th and bought on the 8th therefore stayed in the account for two years, marked at intrinsic value
as SPY rose, and its mark was in the account's net result while the trade list had nothing for it.

Fixed in three places: a contract dated on a day the market is closed is settled at the close of the last session
before that date; "days to expiration" count to that session, so on the 8th such a contract is a same-day contract
and is not what a "1DTE" leg selects; and a leg whose expiration has passed with no session to settle it on is
settled at the next open and reported, never carried on. The tear sheet now compares the account's own net result
with the sum of its trades and says so in red when they differ by more than fifty cents.

Checked on the stored history: of the ten single runs of this acceptance only that variant was affected, and four
benchmark runs are trade-for-trade identical with the fix switched off and on, each with an account that equals its
trades (`engine_same_runs_effective_expiration.json`). The engine version went from 7.0.0 to 7.0.1 so that nothing
computed before the fix is served from the cache, and the whole acceptance below was run again on 7.0.1.

## 5. Sentences become inspectable configurations

Every sentence of the brief compiles to a strategy whose rule can be read as an expression and run. They are checked
by behaviour, not by spelling: the interpretation's entry rule and a hand-written reference rule are both run over
227 stored sessions and must fire at the same minutes of the same days (`strata/ai/eval_v7.py`).

| Sentence of the brief | Read as |
| --- | --- |
| Buy ATM SPY calls at the open after a red previous day | entry 09:30; `previous_session.return_pct < 0` |
| Buy QQQ puts when the first 15-minute candle breaks the previous day's low | `opening_range_low(15) < prev_low`, checked from 09:45, when that candle is complete |
| Buy SPY calls when price reclaims VWAP after declining 0.5% from the open | a sequence: `from_open_pct <= -0.5`, then `crosses_above(price, vwap)` |
| ... opening-range breakouts with cross-symbol confirmation | `price > opening_range_high(30)` and `QQQ.price > QQQ.opening_range_high(30)`, on QQQ's own bars |
| Buy 0DTE calls after three consecutive bearish five-minute candles followed by a bullish reversal | a sequence of two candle counts |
| Test a 30% premium profit target versus a 15% premium stop | a change to the strategy: premium target 30, premium stop 15 (not a move of the underlying) |
| Compare ATM and OTM contracts across multiple entry times | a grid; "OTM" and "multiple" name no values, so 1% OTM and four times are assumed, and said to be |
| Compare exits after 15, 30, 60, and 120 minutes; the same signal across SPY, QQQ, and IWM; 0DTE ... 30DTE | grids of 4, 3 and 5 |
| Test multiple entries and partial exits | asked: it says neither how many nor at what level |

New in the language: minutes held after the fill, several entries a session, partial exits at a premium or an
underlying level, another symbol's own levels (`QQQ.opening_range_high(30)`), exact expiration dates. An engine
configuration with a key the engine does not know is now refused instead of ignored: `"pct": 30` under a stop had
silently become the default 50%.

Two reading defects were found by typing into the running application, and fixed:
- "... exit at 3:55 PM or **when SPY has risen 1%**" was read as the underlying target *and* as an entry condition.
  The strategy then asked for a day that fell 1% and rose 1%, and never traded. A clause inside an exit statement
  is now an exit and is never offered to any reader as a condition of the entry.
- A strategy that waits for an event, with "exit at 3:30 PM", kept opening positions after 3:30 and carried two into
  expiration: 9 of 263 trades, which lost $874 while the whole run lost $99. The reading now states, as an
  assumption that can be edited, that no position is opened from the exit time on.

### What you typed, and what it answered

At about 01:00 Pacific on 2026-11-26, while this work was in progress, one question was typed into the running
application in three wordings, with one correction (they are in its notebook list; I found them there, nobody
reported them):

| Typed | What the application did then |
| --- | --- |
| How often is SPY's overall day, predicted by the first 9:30 open candle? | Read "9:30" as Pacific time (12:30 in New York). Ran a study with no condition and no outcome: "2,740 SPY sessions met the condition ... higher on 53.6% of sessions" |
| no I meant 6:30a, PST | "I couldn't turn that into a strategy, a market question or a change to the current one", followed by example prompts |
| How often is SPY's overall day, predicted by the first 9:30ET open candle? | The same empty study from the open: "8,514 SPY sessions met the condition ... higher on 52.3%" |
| How often does SPY's first 9:30ET candle, predict its overall day? (a new notebook) | The same again |

Nothing was flagged as not understood. "Predicted by the first candle" was dropped, and the number that came back
is the share of all sessions that close above their open: an answer to a question nobody asked. This is what a
silent misreading looks like from the chair, and it is the failure the sealed measurement in section 7 counts.

What happens now, for the same sentences (`tests/test_v7_questions.py`, and a browser test):

- **The question is two directions and how often they agree.** The first candle's direction (its close against
  the session's open) is compared with the direction of the whole session, open to close, and with the direction of
  the *rest* of the session, from the candle's close on. Both are reported because they are different claims: the
  whole session contains the candle, so some agreement is built in; only the rest of the session was still unknown
  when the candle closed. No candle length was named, so 5-, 15- and 30-minute candles are all measured and none is
  assumed. A sentence that does ask about "the rest of the day" gets that measure alone.
- **A frequency question with nothing to count is not run.** If the readers find neither an event nor a
  condition, and nothing they can flag, the reply says so and nothing is run.
- **A correction is a correction.** After "no", "I meant", "actually", "sorry" and the like, a value on its own
  replaces the one thing in the current reading it can replace: a clock time, a percentage, calls or puts, an
  expiration, a symbol. If it could replace two (a rise of 1% after a decline of 3%, then "no I meant 2%"; an entry
  time and an exit time, then a bare clock) nothing is changed and the reply names both. Words that say what they
  change ("no stop loss") go to the readers as they always did.
- **Clock times as typed.** "6:30a", "6:45p", and a zone after a comma ("6:30a, PST") are times. "9:30" with no zone
  is the opening bell in New York whatever zone is set for other times. "Change the entry time to 6:45 AM Pacific"
  had been setting the entry to "6.0".

The question as you typed it, run through the application on the stored SPY minute bars (acceptance M: twelve
studies, 14 seconds):

| First candle | Sessions | Matched the whole session, open to close | Matched the rest of the session |
| --- | --- | --- | --- |
| 5 minutes | 2,714 | 58.0% (95% interval 56.1% to 59.8%) | 50.7% (48.9% to 52.6%) |
| 15 minutes | 2,731 | 61.1% (59.3% to 63.0%) | 50.4% (48.5% to 52.3%) |
| 30 minutes | 2,728 | 64.8% (63.0% to 66.5%) | 51.2% (49.4% to 53.1%) |

Unrelated directions would match on about 50.0%. So: SPY's first candle "predicts" the day 58% to 65% of the time,
more the longer the candle, and nearly all of that is the candle being part of the day. From the candle's close to
the session's close its direction matched about as often as a coin, for every length measured, on 2,731 sessions
from 2016-01-04 to 2026-11-25. It is a statement about direction only, on the underlying, with no costs.

Your two original notebooks still show the old answers. The same question asked again is in the notebook
"V7 acceptance · the first candle and the day".

These sentences were seen before the readers were changed for them, so they say nothing about sentences that have
not been typed yet. The sealed set of section 7 was not opened, scored or edited for any of this.

## 6. Bounded experiments from one sentence

"Compare entry times from 9:30 to 11:00 in 15-minute increments, prior-day declines from 0.5% to 3%, 0DTE versus
1DTE, and several profit targets" is 7 x 6 x 2 x 3 = 252 configurations. Ranges and steps are expanded by
arithmetic, values that were not given are assumed and said to be, and a grid above 300 cells is refused whole,
never trimmed. Configurations that share an entry rule share one signal scan, and each is then simulated only on the
sessions where its rule fired. Nothing is downloaded while a grid runs: each cell reports what the stored data could
not price.

The result is built as a map, not as a search for a winner. Every cell carries trades, net result, average per
trade with a resampling interval, win rate, drawdown, the result with fills 2% and 5% worse, the share of its
signals that could be priced, and the earlier 70% of sessions against the later 30%. The screen marks the most
stable region (the cell whose worst neighbour is best) beside the highest single cell, plots every cell's earlier
average against its later one, and states the best cell's p-value after counting the cells. Every cell is recorded
in the experiment registry as a trial.

The sentence above, run through the application on the stored SPY history (acceptance C; base strategy: an ATM
call at 10:00 after a 0.5% down day, 143 signal sessions):

- 252 of 252 configurations traded, in 107 seconds (42 entry rules scanned once each, 9 seconds; 824 CPU-seconds
  across the worker processes). Assumed and said: three profit targets (25%, 50%, 100% of premium), a 0.5% step for
  the decline, Eastern time for the entry times.
- 213 of them (84.5%) end above zero and the median result is +$1,020.80. That is what buying calls after down days
  did in a market that rose over the period, not a finding about any one setting.
- The best cell (09:45, 1DTE, 50% target, 0.5% decline) made +$8,927.05 over 143 trades: t = 2.81, p = 0.005 on
  its own and **p = 1.00 as the best of 252**.
- Chosen on the earlier 70% of sessions alone, the best would have been 10:45, 1DTE, 100% target, 1.5% decline
  (+$287.30 a trade over 27 trades). On the later sessions it made −$130.12 a trade over 14.
- Across the 126 configurations with enough trades in both parts, the ranking by the earlier part and by the later
  part correlate at **−0.56**: what looked better earlier mostly looked worse later.
- The most stable region is around 10:45, 1DTE, 50% target, 2.5% decline: the cell and its six neighbours all make
  at least +$56.96 a trade. The highest single expectancy (+$190.95 a trade) sits beside neighbours as low as +$36.37.

Every cell is in `experiment_252_configurations.json`.

## 7. A real model as a planner, measured on sentences nobody had seen

**The set.** 34 sentences (16 strategies, 12 grids, 6 that must be flagged as not representable) were written and
sealed before any V7 reader existed; the hash of the set is in `unseen_seal.txt` and a test fails if a sentence is
edited. Each reader was scored on it **once**. Nothing was changed in response to these sentences.

**The models.** No hosted provider has a key, so "a real model" here is the two models installed in the local
Ollama server: gemma3:4b and qwen2.5vl:7b. Through the full planner (every tool, free expressions, 20,000
characters of instructions) both failed on development sentences: one picked an unrelated tool, the other wrote
expressions that do not exist. One call took 520 seconds, because the server had loaded the 6 GB model with its
128,000-token maximum context: 13.5 GB on a 16 GB machine. The application now asks for a bounded context on every
call (12,288 tokens by default), and a call takes 4 to 8 seconds.

**The form.** For a local model the planner is a form of 25 typed fields under a JSON schema the server enforces
while the model writes (`strata/ai/planner_form.py`). The model copies what the sentence says into fields; it
writes no expression, and the form has no field that could hold a price or a result. Everything after is
deterministic: a signal is one of twenty named ones and the application writes its expression; a number must occur
in the sentence or it is removed and reported; a clock time must be one of the sentence's, and its zone is read
from the sentence; a signal needs words for it in the sentence; a number attached to a signal that does not take it
is flagged; lists and ranges are expanded by arithmetic.

**Results, first and only scoring of each reader:**

| Reader | Correct | Wrong, and said so | Wrong, silently |
| --- | --- | --- | --- |
| Built-in readers | 16 of 34 | 9 | **9** |
| gemma3:4b alone, through the form | 16 | 14 | 4 |
| qwen2.5vl:7b alone, through the form | 16 | 14 | 4 |
| Built-in first, the model only for what they flag ("assist"), gemma3:4b | 16 | 8 | 10 |
| The same, qwen2.5vl:7b | 16 | 9 | 9 |
| Both read; a difference is asked ("check"), gemma3:4b | 18 | 14 | **2** |
| The same, qwen2.5vl:7b | 17 | 14 | **3** |

All readers flagged all six sentences that cannot be represented. On the 19 development sentences the same readers
score 19 (built-in), 17 (gemma) and 13 (qwen): the gap between those figures and 16 of 34 is what "tuned on" means.

How to read this:
- Accuracy on an unseen sentence is about one in two, for every reader. The unseen strategies are harder than the
  brief's: several conditions at once, a second symbol, a time limit, a streak of sessions.
- The readers differ in what happens when they are wrong. The built-in readers are confident: 9 silent misreadings.
  A small model through the form is cautious: 4.
- "Assist", the mode first built, does not help: the model is consulted only where the built-in readers already
  know they are unsure, and once it replaced a flagged reading with a wrong one that passed the checks.
- "Check" was designed after those aggregate scores were known and before any of the sentences or per-sentence
  results had been looked at. It is the default with a local model now. Its scores are a first scoring of that
  reader, but the decision to make it the default was taken from this same set, so they are not an independent test
  of that decision. It also asks about readings that were right: 5 and 6 of the 34.
- One disclosure: a worked example in the form's prompt mentions "retail sentiment", and so does one sealed
  must-flag sentence. Every reader, including the built-in ones that have no prompt, flagged all six of those.

After the scoring, the readers were changed from other evidence: a test sentence of mine ("ATM versus 1% OTM" as a
strike comparison; the digits of a clock time not counting as a stated size), sentences I typed into the running
application and the acceptance runs (exit clauses; the test period inside an exit clause; an opening sale mistaken
for an exit; the last-entry assumption), and the sentences you typed (the first-candle question, corrections,
clock formats, a frequency question with nothing to count). None of it came from a sealed sentence, but some of it
was written after I had seen the per-sentence table of the sealed results, so I cannot claim the changes are
blind to it. The scored code is identified by hash in each result file. The shipped readers have not been scored
on the sealed set and will not be: a second scoring would no longer be a first attempt.

To use a local model: Settings > Language model > Ollama, and pick a model. Each request then takes several
seconds longer, and the server holds the model in memory. Nothing is sent anywhere.

## 8. Market replay on stored observations

The Replay tab of a run, and the Replay page for any stored session, draw what is on disk and nothing else: the
underlying's stored one-minute candles with VWAP and moving averages computed by the engine's own functions, the
previous session's close, high and low, and the opening range; below it one contract's stored trade bars with
volume. A minute in which the contract did not trade has no bar, the chart has a gap there, and the position's
mark across the gap says how old it is. A trade is laid over both charts with the minute the rule held, the minute
it was filled, the age of the print it was filled on, its target and stop as price levels, and its running result
marked on the leg's stored prints. One cursor moves through all charts; "only what was known" draws nothing after
it. Any other contract or expiration stored for the session can be switched to, and its bars are read only then.
Long spans are aggregated and the page says to what.

Loading on the stored dataset: a session with its levels 0.15 s, a contract's prints 0.006 s, a trade laid over
them 0.09 s, a year of candles 0.20 s.

Three defects were found by using it on runs made through the lab, after its tests had passed on directly made
runs: no stop, target or clock-exit level was drawn (section 13, G); at a laptop-sized window the contract's chart
sat below the fold, because the charts had fixed heights and a wheel over a chart zooms it; and with the pointer
resting on a chart, the cursor snapped back to the pointer whenever the slider or the keyboard moved it, because
the chart reports a "pointer move" each time it redraws. All three are fixed; the first has a backend test and the
third is what the browser test of the scrubber now exercises.

## 9. What a strategy did, in the order a reader should take it in

The tear sheet puts first what decides whether anything after it means anything:

1. trades actually taken; 2. what was asked for against what the stored data could price; 3. the net result after
costs; 4. the average trade; 5. expected value per trade with its interval and the p-value after counting the
configurations tried; 6. maximum drawdown; 7. consecutive losses; 8. the distribution of results; 9. years and
months; 10. time of entry; 11. sensitivity to worse fills, with the individual-print comparison when it has been
run; 12. dependence on the largest trades; 13. the same position without the signal (a baseline that enters at the
same time on every session, run on request and never counted as a trial).

One line at the top is the only place that speaks about an edge, and it is separate from the figures: "no evidence
of an edge", "too few trades to judge", "mixed evidence" or "evidence of an edge on this data".

**Every fill can be checked.** The Fills tab lists each fill with the time it was decided, the time it was filled,
the print it used and the stored bar that print came from, read back from the store now, with a CSV of the same.
On the first acceptance run 150 of 150 fills match their stored bar. A position held to expiration is listed as a
settlement at intrinsic value, not as a fill: there is no print to compare it with.

## 10. Forward validation is incremental

Until V7 an evaluation replayed the whole window since activation and reported whatever positions and cash that
replay produced. The account is now a stored thing: after each sealed session the engine's full state (cash, open
positions with every attribute, pending exits, what each strategy last did) is appended to a table that refuses
updates and deletes, with its hash, the engine version and a fingerprint of the observations the session read. An
evaluation loads the last state and runs only the sessions after it. On real data, one multi-day spread strategy
run continuously and run in 2, 4, 28, 188 and 125 single-session pieces ends in the same state on every field.

If stored data is revised later, `Reconcile` re-runs only the affected session from its own stored starting state,
compares, and appends a revision record. Sealed entries, stored states and every later decision stay as they were.
An engine upgrade appends an `engine_change` entry and continues from the stored state.

Acceptance K froze the SPY strategy of acceptance A as a paper experiment on 2026-11-26, a market holiday: engine
7.0.1, configuration hash `157545ab8a177cdc`, a ledger of one entry. The first session that can count is
2026-11-27. It has **no forward evidence**: no session has closed since it was frozen, and the evaluation says
"waiting", not a result. Nothing about it is a validation of anything yet.

## 11. Performance on the stored dataset

`strata bench --dataset` runs each workload in a fresh process on the stored history, read-only, and records wall
time, CPU time and peak memory (10 cores; the machine was also running other programs, so CPU time is the steadier
figure):

| Workload | Before this pass | After | CPU s | Peak memory |
| --- | --- | --- | --- | --- |
| SPY, a same-day call every session, 718 sessions (710 trades) | 14.95 s | 8.89 s (8.48 s again in the same process) | 13.1 | 648 MB |
| QQQ, the same (708 trades) | 14.15 s | 8.95 s | 12.8 | 514 MB |
| IWM, the same (687 trades) | 11.04 s | 6.88 s | 9.6 | 474 MB |
| Three symbols in one run (2,105 trades) | 37.61 s | 24.06 s | 35.2 | 753 MB |
| A two-leg spread a week out, held for days | 11.44 s | 9.11 s | 12.0 | 652 MB |
| An ordered rule checked every minute, premium target and stop | 7.06 s | 6.23 s | 7.6 | 498 MB |
| 100 configurations, 227 sessions, worker processes | 56.59 s | 35.96 s | 256 | 1,115 MB in the largest worker |
| Replay loading (session, contract list, contract, trade, a year) | 0.91 s | 0.70 s | 1.4 | 327 MB |
| Individual-trade check from the cache, 23 positions | 0.38 s | 0.38 s | 0.4 | 172 MB |

"Before" is already after the earlier V7 work: expanding a session's chain with array operations instead of a
slice per contract (7.6 times faster), evaluating the clock before the other rules of an entry and stopping at the
first plain "no", loading only the expirations a strategy can select, computing implied volatility and greeks only
for the rows that are read, and running a grid's configurations only on the sessions of their signals. Those took
one real run from 56.1 s to 8.8 s. The pass measured above read only the needed expirations and columns from each
day's file (4.3 ms to 1.0 ms a session) and computed a single row's greeks with the scalar kernels.

Every change was accepted only with trade-for-trade identical results on real runs, greeks included
(`engine_same_runs_*.json`): six runs for the first set, four for the second, and four again for the expiration
fix of section 4.

Market data on disk: 2,572 MB in all, of which the three option archives are 1,381 MB. The individual-trade cache
is 3.8 MB.

## 12. The interface

The primary screen was rebuilt around the path a piece of research takes, and checked in a real browser at each
step:

- **A command line across the top.** Strategies, comparisons, changes and questions are typed in the same place.
- **The reading as a ticket under it.** Every field the sentence was read into is a cell; what was assumed is marked;
  what could not be represented is a red row; an open question (a target that could mean the underlying or the
  premium, two readers that disagree) is asked there with one-click answers. A cell opens the editor.
- **The stages on one line**: understood, checked against the sentence, coverage (72 of 75), plan, run, explained.
- **The workbench**: overview, tear sheet, trades, replay, fills, execution, calendar, statistics, robustness,
  methodology. A comparison opens as a grid with its heatmap; a question about two directions opens as a strip per
  candle length on one percent axis, each figure with its interval and the level unrelated directions would show.
  Nothing navigates away.
- **The log** of the conversation at the side, collapsible.

**Focus** (title bar) folds the reading and the suggestions away and gives the workbench their height; on the
replay the run's heading is one line and the charts share the height there is, so that the underlying and the
contract are both on screen on a laptop.

The landing page shows what is stored on the machine (bars per symbol, whether the collector is continuing it,
whether any quote source is connected) and sentences that run on it. The Data page has the collector's source
picker and the quotes section; the Forward page has the stored account and Reconcile; Settings has the reading
mode and the local-model context.

## Tests

598 backend tests pass with downloads off (610 with the 12 opt-in live ones), 128 of them new in thirteen
`tests/test_v7_*.py` files; 28 browser tests pass against the fixture store; `strata validate` and
`strata bench --check` were run on engine 7.0.1 (validation: 14 of 14 steps on real observations from an empty
directory; benchmark: 17 fixture workloads, none flagged against the recorded baseline).

One finding about the suite itself: six test files switch `time.sleep` off for the whole process to skip a
provider's retry delays. Every job scheduler left alive by an earlier test then stopped sleeping and spun, and under
load one collector test that takes 1.5 seconds alone ran for more than four minutes before I stopped it. The
scheduler now waits on an event, which a patched `sleep` cannot turn into a spin.

## 13. Acceptance on the stored dataset

Every item below was typed into the running application as a sentence (the same HTTP calls the interface makes),
on engine 7.0.1, after the last engine fix, by `docs/v7_evidence/acceptance_driver.py`. The record is
`acceptance.json`; each item is a notebook titled
"V7 acceptance · ..." that can be opened. Where the application stopped to ask about coverage, the first choice it
offered ("run on the sessions that are covered") was taken, and that is recorded with the question.

**A. One strategy over the whole stored period, on each symbol.** "Buy ATM {symbol} 0DTE calls at the open after a
1% down day, exit at 3:55 PM Eastern or when {symbol} has risen 1%, from 2024-01-18 to 2026-11-25."

| | Signals | Priced | Trades | Net after costs | Average, 95% interval | p | With fills 2% worse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 75 | 75 | 75 | +$2,384.50 | +$31.79 (−27.10 to +93.82) | 0.31 | +$1,541.06 |
| QQQ | 121 | 121 | 121 | −$1,037.30 | −$8.57 (−55.35 to +39.67) | 0.72 | −$2,413.46 |
| IWM | 129 | 123 | 123 | +$931.10 | +$7.57 (−12.15 to +26.60) | 0.45 | +$322.12 |

IWM asked first: 5 of its sessions have no same-day expiration stored and one is held only in part by the
provider. Each run took 6 to 8 seconds. Every interval includes zero; the tear sheet of each says "no evidence of
an edge". SPY's fills were checked against their stored bars: 150 of 150.

**B. The same signal across the three symbols**, as one grid in SPY's notebook: 3 of 3 traded; the best (SPY) is
p = 0.94 as the best of three. Chosen on the earlier sessions it made +$78.03 a trade over 49; on the later
sessions −$55.34 over 26.

**C. A bounded sweep from one sentence**: 252 configurations, section 6.

**D. ATM against OTM across entry times.** "Compare ATM and OTM contracts across multiple entry times" names neither
a distance nor a time: 1% OTM and 09:30, 10:00, 11:00, 13:00 were assumed and said. 8 of 8 traded, 5 above zero; the
best (13:00, ATM: +$3,744.70 over 141 trades) is p = 1.00 as the best of eight, and the earlier ranking correlates
with the later one at +0.12.

**E. Several expirations.** "Buy ATM SPY calls at the open after a 1% down day, hold for 1 trading day", then "Test
0DTE, 1DTE, 3DTE, 7DTE, and 30DTE contracts." No expiration was stated in the first sentence, so 10DTE was assumed
and said; its run priced 75 of 75.

| | Priced of 75 | Trades | Net | A trade |
| --- | --- | --- | --- | --- |
| 0DTE | 75 | 75 | +$5,696.71 | +$75.96 |
| 1DTE | 75 | 75 | +$6,689.38 | +$89.19 |
| 3DTE | 75 | 75 | +$6,871.50 | +$91.62 |
| 7DTE | 75 | 75 | +$5,777.50 | +$77.03 |
| 30DTE | 30 | 30 | +$4,788.41 | +$159.61 |

The 30DTE row describes 30 sessions: on the other 45 the contract had no print near the entry or the exit, and the
comparison says so before it says anything else. In every row the account's net result equals the sum of its
trades; on engine 7.0.0 the 1DTE row did not (section 4). All five are positive in a period when SPY rose; none
was compared with the same position without the signal.

**F. A stateful strategy.** "Buy two ATM QQQ 0DTE calls when price reclaims VWAP after declining 0.5% from the open,
sell half when the premium is up 25%, with a 15% premium stop, exit at 3:30 PM Eastern." Read as an ordered rule
(down 0.5% from the open, then a cross back above VWAP), two contracts, a partial exit, a premium stop, a clock
exit, and one assumption: no entry from 15:30 on. The rule fired 257 times on 256 sessions; 254 were priced. 254
trades, **+$775.60** after $660.40 of commissions, +$3.05 a trade (−11.07 to +17.21), p = 0.67; positive only
because of its three largest trades; zero at fills 0.4% worse.

**G. A replay.** Trade 1 of F (QQQ, 2024-01-31, the 420 call): the underlying's 390 stored one-minute bars with the
previous session's levels and the opening ranges, the contract's 390 stored trade bars (a print in every minute),
the entry and exit on them, the stop at 1.22 (15% of the 1.44 paid), the partial exit at 1.80 and the 15:30 clock
exit, and the position marked minute by minute; loaded in 0.03 seconds. All 508 fills of the run match their
stored bars. The levels were missing from the first acceptance run: the replay read the request a lab run was made
from, which does not validate as an engine configuration, and drew no stop, target or clock exit on any run made
through the lab. It reads the configuration the engine ran now, and a test makes a run that way.

**H. Minute bars against individual prints**: section 4. SPY (A): +$2,384.50 on bars, +$2,496.50 on first prints,
the same with resting orders. QQQ (F): +$775.60, +$557.60, +$249.60.

**I. A matched comparison.** "Does SPY tend to recover after down days of 1.5% or more, even after controlling for
volatility?" became a research graph of 11 steps. On the training period the next session closed higher on 270 of
472 such sessions (57.2%) against 53.1% of 6,339 others (p = 0.077 with weeks resampled); against sessions matched
on realised volatility and the volatility index the difference is +8.4 points (+3.6 to +13.1). Not a candidate
after correcting for the 2 configurations tried (adjusted p = 0.13). The final test period (2020-02-19 on) was not
examined.

**J. Out of sample.** SPY (A) split at 2026-01-20: +$3,823.30 over 49 trades before, **−$1,438.80 over 26 after**.
The application adds that every earlier run in the notebook used the full period, so agreement would have been
necessary, not sufficient. "Is this statistically significant?": no; the interval on the average includes zero
(p = 0.31, 1.00 after the 6 configurations tried in that notebook).

**K. A prospective paper experiment**: frozen, waiting for its first session (section 10).

**L. A quote-based comparison**: BLOCKED, no Databento key (section 3). Nothing was mocked in its place.

**M. The question you typed**: section 5.

Model-backed interpretation on unseen prompts: section 7 (two local models; hosted models BLOCKED on a key).

## What changed on your machine that you did not type

- **The collector is on**, continuing SPY, QQQ and IWM from Alpaca. Turn it off under Data > Collector.
- **A paper experiment was frozen** for the acceptance ("V7 acceptance: SPY 0DTE calls after a 1% down day"). It is
  evaluated on each new session while it is active; stop it under Forward validation if you do not want it.
- **Notebooks.** The acceptance left 9 notebooks titled "V7 acceptance · ...". Earlier attempts, made while defects
  were still being found, left 26 more; those were deleted on 2026-11-26 once you said to, after this report was
  first written. Their runs are still listed under Backtests and every configuration they ran is still in the
  experiment registry, which cannot be deleted by design; one duplicate finding went with its notebook. One more
  notebook, with your first-candle question as its title, is the run I made in the browser while fixing it.
- 1,996 contract-sessions (122,477 bars) were fetched on demand from Alpaca, at no cost, by the expansion matrix
  and the acceptance runs, after the 728,410 bars of the gap top-up. One consequence: two benchmark runs trade one more
  QQQ session (2026-11-05) than they did this morning.
- The engine is 7.0.1. Runs stored earlier today under 7.0.0 were made by builds that still had the defects
  described above; they are not reused.
- Nothing was bought, no account was created, no key was typed, no order endpoint exists.

## What is not done

- **Quotes.** Blocked on a Databento key. Until then every result is on trade prints and no figure includes the spread.
- **Hosted models** were not measured. The full planner is unchanged for them and unverified on V7's new tools.
- **Accuracy on unseen sentences is about one in two**, and the question you typed is what the other half looks
  like. Read the ticket before relying on a run. Select a local model to have disagreements asked.
- **Corrections cover a value on its own** (a time, a percentage, calls or puts, an expiration, a symbol) and the
  edits the readers already knew. Anything else after "no, I meant" is read as a new request or answered with what
  could not be placed.
- **The first-candle answer is about direction only**, on the underlying. It says nothing about how far the
  session went or about any option position.
- **A grid does not fetch what it is missing.** It reports the signals it could not price; run the single strategy
  to fetch them, then run the grid again.
- **Far expirations are thinly traded.** A 30DTE or 90DTE leg often has no print near the entry or the exit; runs
  report those as not priced.
- **The earlier/later split is not an untouched test set.** Both parts are history that has been examined.
- **Seven early sessions** are held only in part by the provider, for all three symbols.
- **On a laptop screen the replay is tight.** Both charts now share the height there is and Focus folds the
  reading away, but at 1440 x 860 the contract's price pane is about 40 pixels tall until Focus is pressed.
- **A market closure the calendar does not know** would bring the expiration defect back for contracts dated that
  day. The calendar holds the closures through 2026; the guard on the tear sheet would show it.
- **The Docker image was not rebuilt.** Docker Desktop was not running when the work ended, so the container (the
  copy published on port 8010 on this machine) still holds the V6 build and is stopped. `docker compose up --build -d`
  rebuilds and starts it. Until then V7 runs from `scripts/dev.sh` (http://localhost:5199).

## Sources

Databento facts in section 3 are from Databento's own pages, read on 2026-11-26:
- https://databento.com/docs/quickstart
- https://databento.com/docs/faqs/usage-pricing-and-data-credits
- https://databento.com/docs/api-historical/metadata/metadata-get-cost
- https://databento.com/datasets/OPRA.PILLAR
- https://databento.com/docs/portal/api-keys
- https://databento.com/docs/knowledge-base/portal/billing/manage-monthly-limit
- https://databento.com/docs/api-reference-historical/metadata/metadata-list-unit-prices
