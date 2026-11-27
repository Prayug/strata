# Strata V9: delivery report

Written 2026-11-27. Every figure below was produced by the application, its evaluation harness or the test suites on the
development machine that day (Apple M5, 16 GB), from the stored history: SPY, QQQ and IWM daily since 1993/1999/2000,
one-minute bars since 2016, and Alpaca's one-minute option trade bars from 2024-01-18. Evidence is in
`docs/v9_evidence/`. Where something was not measured, or is blocked, it says so.

**Short version.**

1. **On sentences written before V9 and never used to fix it, the reader missed both targets by a wide margin.** In
   the primary configuration (the local model consulted on what the built-in readers cannot place), 21 of 55 answerable
   messages were read correctly (38%; the target was 90%) and 14 (25%; the target was under 2%) were answered as a
   different question without saying so. The built-in readers alone, as installed: 35% and 20%. The model reading
   first: 47% and 27%. The same code had read 74 to 83% of the development items correctly, but those items were used to
   fix it. Of 12 questions with numbers to reproduce, 9 were answered and every number in them matched the independent
   recalculation; the other 3 were misread.
2. **A language model can now be the first reader of every message, and nothing it says is trusted without a check.**
   One reader (`strata/ai/model_reader.py`) serves every provider. Its reading is in Strata's own question language,
   each part quotes the message, numbers and times are re-read from those words by the deterministic code, and a clause
   no part holds stops the run. Hosted models are supported but **were not measured**: you chose no paid calls. The local
   model measured, qwen3:8b (downloaded with your approval), helped most when reading first, and did not reduce silent
   errors.
3. **Every answer is now recalculated independently, and the recalculation found four real errors.** A second
   calculation in DuckDB SQL, written apart from the engines, checks each count and average and re-prices option
   backtests trade by trade from the raw prints. It reproduced V8's failed sealed answers to the session and found: a
   study that ended at the last minute bar instead of the official close; a VIX condition that ran on a different
   number than the reading said (50 QQQ sessions); two definitions of a session's open used side by side; and 0 DTE
   trades on half days that were exercised instead of sold. All four are fixed, and every answer now carries the
   check's verdict.
4. **Position sizing in a trader's terms** runs on the existing engine, and the answer keeps the premium paid, the most a
   trade could lose, the loss the stop allowed and the loss actually taken apart.
5. **No independent tester was available.** Three unscripted conversations were held in the real application, on real
   data, at desktop, laptop and phone sizes; eleven problems they found were fixed and are listed below. Two more,
   found in the sealed run's output (a model's rephrasing that stated a false observation, and a data preview that
   looked only at 1993 to 2008), were fixed after scoring; neither changes how a message is read.
6. **Docker: rebuilt.** `strata:latest` holds V9 (version 0.9.0, with both fixes made after scoring); the container on
   http://localhost:8010 stands by while the copy running on the Mac holds the data folder, as designed.

## What was measured, and what is blocked

| | Status on 2026-11-27 |
| --- | --- |
| Model reader, local (Ollama qwen3:8b) | built, measured on development data and once on the sealed set |
| Model reader, hosted (Anthropic and the other providers in `llm.py`) | built, tested with stand-in models; **not measured** (no paid calls, by your decision) |
| Independent recalculation of answers | built; on in every answer; covers frequency, movement and agreement studies, timing and event profiles, and single-leg long option backtests with clock entry and exit (fixed contracts, a premium budget or a risk budget) |
| Position sizing | built and used in the browser on real data; re-priced independently for contracts, $ premium and $ risk |
| Bid/ask (quote) execution | **unverified**: no historical intraday quotes are stored; Databento is still blocked on credentials |
| Independent usability tester | **none available**; the conversations below were held by the developer |

## Understanding the question

### What reads a message now

```
message ──► conversation actions on what is on screen (undo, compare, explore, next steps, a question about a selection)
        ──► llm.mode
              rules        the built-in readers (V8's question reader, then the older strategy, graph and experiment readers)
              rules_first  the built-in readers; the model reads only a message they could not place completely   (auto, local model)
              model_first  a follow-up goes to the deterministic edit layer; a new message to the model, checked   (auto, hosted model)
```

The model's reply is one JSON object under a schema: what kind of message it is (a question, a strategy, a change, an
action on the result, something unanswerable, something unclear) and the parts of the question, each with the exact
words of the message it comes from. Then, deterministically:

- **Vocabulary.** Only Strata's condition, outcome and command names exist (enforced by the schema and checked again; a
  reply with any other field is refused and said, and the built-in readers read the message).
- **Grounding.** A part whose quoted words are not in the message is dropped (it did not come from the user). The model's
  numbers are never used: percentages, counts, minutes, days and clock times are re-read from the quoted words; a
  percentage needs a % in its words ("3 red days in a row" is a count, not a 3% fall); each condition's words must name
  what it is about (a VIX clause filed as a previous-day move is refused); a ticker must be the one its words name.
- **Coverage.** Every word of the message that carries meaning must lie inside a quoted part; otherwise the message
  goes to the built-in readers, never to a partial run.
- **Cross-check.** When the built-in reader also reads the whole sentence and the two readings differ, the user is asked
  which was meant, both shown. When they differ only in what an ambiguous outcome is measured against ("close higher":
  than the open, or than the previous close), both are measured and said, as an ambiguous word always was.
- **Edits.** A change may only touch what the new message quotes; a condition the message neither restates nor removes
  is kept.

V6's tool-loop planner and V7's form planner, the two model readers this replaces, are no longer on the request path;
the tests that exercised them through the agent were ported to the V9 reader with the same guarantees (a broken or
hostile model can do nothing but describe a request; no credential or local path is ever sent to a model; the reading
order is a setting; two readings that disagree are asked about).

### Measured on development data

Development data: the V9 brief's examples (7 single questions, 2 conversations), a fresh probe written during V9 in
phrasings neither reader was built on (25 + 3 conversations), and V8's post-fix probe (28 + 2). They were used to fix
the readers, so they are not an estimate of accuracy on new sentences; the sealed set is.

| reader (final code) | correct | asked | silent | seconds per message (median) |
| --- | --- | --- | --- | --- |
| built-in readers, no model (as installed) | 65 of 88 | 19 | 4 | 0.1 |
| `rules_first` with qwen3:8b (`auto` for a local model) | 68 of 88 | 16 | 4 | 0.4 to 3 |
| `model_first` with qwen3:8b | 73 of 88 | 10 | 5 | 5.4 |

Before V9's fixes to the readers, the built-in readers scored 56 of 88 with 11 silent errors on the same items, and the
model first 55 with 14: most of the gain on this data came from fixing the deterministic checks and readers, which
both arrangements share. The sets were used for those fixes, so these figures flatter both readers, by far: on the
sealed set the same three arrangements read 35%, 38% and 47% of answerable messages correctly (below), and the order
of the two model arrangements reversed.

What the model did badly, and what the checks now catch: it invented a ticker that was not in the sentence, copied
parts from its examples, filed "VIX under 15" as a previous-day fall, read "3 red days" as a 3% fall, and read "puts on
spy" as selling puts. What it did well: phrasings the built-in tagger has no pattern for. On conversations it was
worse than the deterministic edit layer, which kept the context of every measured edit; a follow-up therefore goes to
that layer first even when the model reads first.

### Bugs in the built-in readers found on the way (development data, not the sealed set)

A ticker with no stored history ("buy tsla puts at the open") ran silently on SPY; it is now declined with the closest
question that can be answered offered. "How many times did QQQ drop 3% in a day" counted the day after each drop.
"Down 0.5% at 10am" was read as the previous day's fall. A follow-up that only added a condition changed the kind of
question. "What if it gapped down but yesterday was green" turned a 1% gap condition into any gap. "Does the first
30-minute candle predict the rest of QQQ's session" went to the hypothesis lab while every other phrasing of the same
question went to the agreement reading; there is now one engine for it.

## The sealed V9 set

**Written before any V9 code, scored once after all of it.** 62 single messages (55 the stored data can answer, 7 it
cannot), 7 conversations of 31 turns, and 12 questions whose numbers must be reproduced, with the expected readings
fixed in advance (sha256 `868db97d…`, `docs/v9_evidence/sealed_v9.txt`). On 2026-11-27, after the last change to the
readers, `python -m strata.ai.score_sealed_v9` scored it in the three configurations registered with the hash, then ran
the numerical questions with the primary one. The run is recorded in `sealed_v9_scored.txt` and the script refuses a
second. Nothing in the readers changed afterwards. Results: `sealed_v9_A.json`, `sealed_v9_B.json`, `sealed_v9_C.json`,
`sealed_v9_answers.json`.

"Correct" is the expected reading or a registered alternative; "asked" is a question or a flagged clause, with nothing
run; "silent" is a different question answered without saying so or, for a request that cannot be answered, something
run.

| configuration | answerable messages read correctly | silent errors on them | the 7 unanswerable: declined with the closest question | conversation turns: correct / asked / silent | context kept | seconds per message, median (90th percentile) |
| --- | --- | --- | --- | --- | --- | --- |
| A: built-in readers (as installed) | 19 of 55 (35%) | 11 (20%) | 1 (4 declined without it, 2 run) | 20 / 10 / 1 | 8 of 9 | 0.1 (11.4) |
| **B: qwen3:8b, rules first (`auto`): primary** | **21 of 55 (38%)** | **14 (25%)** | 1 (4 without it, 2 run) | 21 / 9 / 1 | 8 of 9 | 2.7 (13.9) |
| C: qwen3:8b, model first | 26 of 55 (47%) | 15 (27%) | 6 (1 run) | 24 / 6 / 1 | 9 of 10 | 4.9 (17.4) |

**Numerical answers (B): 9 of 12 reproduced.** In each of the nine, every number the item names was within its
tolerance of the independent recalculation. The three misses were readings, not calculations: "Does the first
15-minute candle predict the rest of SPY's session?" went to a research plan, which reported the rest of the session
(50.4%, as the reference) but not the whole day (61.0%); "Which weekday has the best average open-to-close return for
QQQ?" was not read ("not 'open', 'close'"); "On days SPY gaps up 0.5% or more, what is the average move from 10:00 ET to
the close?" went to a research graph, which reported how often SPY closed above its entry (712 of 1,261 sessions)
instead of the average move. In four of the nine the reply's prose was the local model's rephrasing of the result, and
two of those contained a false sentence built from true numbers: "SPY closed above its open in 2,741 sessions out of
2,741 evaluated" (2,741 is the number of sessions; 1,486 closed above), and the same for "traded 0.5% above its open in
2,740 sessions", dated from 1993 although the measurement starts in 2016. The scorer looks for the numbers and passed
them. Results no longer use that rephrasing (below).

**Against the targets: both missed, by a wide margin.** At least 90% of answerable requests read correctly: 38% in the
primary configuration, 35% and 47% in the others. Under 2% silent errors: 25%, 20% and 27%.

**A stricter count, made afterwards.** The scorer (V8's, registered with the set) counts any question or flag as correct
where a question is acceptable, even "I couldn't turn that into a strategy". Read one by one, four such replies in B and C
and five in A were not questions about what was ambiguous (one or two single messages, three conversation turns), and two
replies scored silent were the expected reading written differently ("a green day" as a positive day's return; "at
least 1.5%" as an expression for the previous session's return, which selects the same days). Counted that way: A 18 of
55 with 10 silent, B 21 with 13, C 26 with 14; conversation turns 18, 19 and 22 of 31, none silent
(`sealed_v9_hand_audit.json`, made by `sealed_v9_hand_audit.py`, which lists every reply judged; same author, after
scoring; nothing was changed because of it).

### What went wrong (B: every miss read by hand)

| | kind | messages | examples |
| --- | --- | --- | --- |
| silent | the wrong kind of question | 5 | "when does IWM usually put in its high on gap up days" became a put strategy; "do tuesdays tend to be green" and "when yesterday was up, is today usually up too" became first-candle studies; "takes out yesterday's high in the first hour" became the average move after a cross; "a 1% gap up keep going til close" lost the 1% and became an average move |
| silent | a clock time or an option's gain read as the stock's move | 3 | "out by 1245 pst" became an exit when SPY is up 1,245%; "sell when they're up 40%" (QQQ calls) became QQQ up 40%; "sell when it's up 20%" (IWM puts) IWM up 20% |
| silent | a part dropped or invented | 3 | "between 6:30 and 7:30 pacific" answered for the whole day; "close red aftr a gren day" lost the green day; "max 3 a day ... stop trading for the day after losing $200" lost the limit of 3 and gained a down-day condition |
| silent | ran instead of asking | 2 | "what's the best time to buy calls"; "buy calls when it dips" (read as any fall below the open, on SPY) |
| silent | the same reading, written differently | 1 | see the stricter count |
| asked | taken by the older research-plan readers | 7 | "how far does iwm usually run after gapping up 1%", "typical spy path on days it opens more than 1% lower", "show me how spy trades the day after a 3% drop", two gap-fill questions: a research plan to confirm instead of an answer |
| asked | a clause not read | 10 | "the next day", "friday's close", "the first 5 minute candle is red", "1% otm", "sell 2 days later", "a 30% loss", "when SPY is up half a percent", "are fridays worse", "best day of the week", "is the last hour more volatile than the first" |
| asked | a reasonable question the set did not allow for | 3 | "You said stop 40%. Do you mean ..." where the set expected the option's price |
| unanswerable | declined without the closest question | 4 | the day after FOMC, tomorrow, RSI, earnings weeks |
| unanswerable | drafted anyway | 2 | iron condors since 2015 and calls "every morning in 2022": option prices are stored from 2024; both were drafted as strategies instead of being declined with the closest question |

In conversations the misses compound. When the first message went to a research plan (v9c6, "how often does qqq fill a
gap down"), the follow-ups had nothing to change. In v9c7, after "what if yesterday was red too" went to a research
plan, "show me the worst days" replied that nothing had been answered yet, although the first question had been. In
v9c5 the first message's condition ("when it's down 1% from the open") was never read, so each of the four changes was
shown with it still missing.

### The language model's part

Consulted only on what the built-in readers could not place (B), qwen3:8b turned five of the messages they had asked
about into answers: two right, three silently wrong (two first-candle readings and the 1,245% exit). Reading first (C)
it did better: five more messages right than B, all among those B had asked about or misread ("gaps up half a percent
... filled", "after 3 red days in a row", "the day after a 3% drop", "how far does iwm usually run", "sell when they're
up 40%"); six of the seven unanswerable requests declined with the closest question offered; 24 of 31 conversation turns.
Its silent errors did not fall (15, with two new ones: "flat by 3:30pm ET" read as an exit when SPY is up 3%, and "what
fraction of mondays does spy open above friday's close" read as a first-candle study), and it took about twice as long.

The default for a local model (`auto`, rules first) was chosen on the development data, which favoured it; the sealed
set does not. Changing the default because of this result would make C's figures an optimistic estimate, so it stays
as registered, and `llm.mode` is a setting. Hosted models were not measured.

### Why the development figures were so much higher

On the development items the same code read 74 to 83% correctly. Those items were used to find reader bugs, and each
fix, general in intent, was confirmed on them. On sentences nobody had looked at, what decides is how much phrasing the
readers cover, and that is far from complete; V7's sealed set showed the same (16 of 34). The brief asked for this not
to be solved with more special-case patterns. V9's answer was the model reader, and a small local model under strict
grounding checks did not close the gap.

### What the misses point to (not done here: doing it would use the set up)

1. Take the V5 to V7 research-plan readers off plain questions: they should read only an explicit request to
   investigate (7 single messages and 2 conversation turns here).
2. A clock time after "by", "out by" or "flat by" is an exit time, never a price target; "up 20%" after an option is the
   option's price unless the stock is named, and an unclear case asks.
3. The kind of question: a weekday question is a breakdown, "when does it put in its high" is timing, "yesterday up,
   today up?" predicts from the previous day.
4. A message with no symbol, time or condition asks, instead of running SPY with defaults.
5. Every decline offers the closest question.

A set written before those changes would be needed to measure them.

### Found in the sealed run's output, fixed after scoring

Neither changes how a message is read, so neither changes a figure above; both have regression tests
(`tests/test_v9_output.py`).

1. **A model's rephrasing could state a false observation.** With a model configured, a result's text was the model's
   rewording of the calculated facts, accepted when every number in it was one of the result's. That check cannot see a
   true number attached to the wrong quantity (the two sentences above). Results now always use the calculated text
   (`Agent.finalize`). In the notebook the rewording appeared only for results without an answer card; the workspace
   (`/lab`) showed it, marked "worded by" the model.
2. **The data preview of a strategy looked only at its first 4,000 sessions.** An entry at the open on SPY's daily
   history since 1993 fires on 8,515 sessions; the preview kept 1993 to 2008, said that none of them had option prices,
   and offered no run on the sessions that do. It now counts every session: "The entry rule fires on 8,515 sessions
   (1993-01-29 to 2026-11-25). 710 pass every check on stored option observations ..." (`market.needs_of`).

## Independently verifying answers

### V8's six mismatches, explained

| V8 item | what happened | kind | now |
| --- | --- | --- | --- |
| a1, SPY close above open after a 2% down day | 190 of 342 vs 189: on 2020-04-16 the official close (279.10) was below the open (279.15) but the last minute bar (279.21) above it; the study ended at the last bar although the engine says the official close wins | calculation error | fixed: a horizon that ends at a session close ends at the official close (ENGINE_VERSION 7.0.2); 189 of 342 |
| a9, QQQ after opening 0.5% below the previous close | 297 of 1,257 vs 295 of 1,259: the application's open is the first 9:30 minute bar's, V8's check used the official daily open; 12 sessions fall on different sides of -0.5% | data-source difference | the definition is stated in every check; the reference uses the application's |
| a3, best weekday for IWM | asked back instead of answering | interpretation | read correctly by the V9 readers (development check) |
| a5, "good strategy or just luck" | ran the strategy, dropped the clause about luck | interpretation (dropped clause) | the reliability answer is routed separately |
| a6, 2021 backtest | did not name the date option prices start | presentation | fixed late in V8 (the plan names the stored span) |
| a8, average move after a 1% gap down | answered a frequency instead of an average | interpretation | movement questions read as movement (development check) |

### Found by the recalculation in V9

| | what | effect | fix |
| --- | --- | --- | --- |
| VIX | "VIX was above 30 the day before" compiled to `vix`, the level at the moment of entry (the day's own VIX open), while the reading said the previous close | QQQ: 311 of 640 sessions vs 289 of 623 | compiles to `vix_prev_close`; matches |
| Opens | profiles (timing, breakdown, event counts) used the daily record's open, studies the first minute bar's | "how often does SPY gap up 1%": 352 in a profile, 353 in a study | one definition everywhere |
| Incomplete sessions | the timing profile counted a session whose minute bars stop at 15:31 (SPY 2019-08-12) as an early close | the day's high possibly missing | the bar store's coverage rule everywhere |
| Half days | a clock exit after a 13:00 close never fired; the 0 DTE position expired, was exercised into shares and sold at the next open | QQQ 2024-11-29: +$300.13 reported, +$179.70 when sold at the close | the exit fires at the close |

### What every answer now says

The check runs when an answer is built and never changes it. Over a fixed battery (`python -m
strata.research.reference_report`, `docs/v9_evidence/reference_checks.json`): 15 of 15 studies (frequency and movement), 2 of 2 profiles (the timing of the day's high, an event count), 2 of 2 agreement questions (every count of the outcome matrix) and 553 of 553 trades of two backtests the same. Two fresh backtests
were re-priced trade by trade: QQQ calls, 311 of 311 trades and $6,032.70; IWM puts at 2 contracts, 242 of 242 and
$3,210.80. In the browser, every sized IWM run was re-priced as well (140 of 140 trades each).

Definitions the recalculation states with its results: a session is a regular-hours day of the daily record; its open
is the first one-minute bar's open where whole sessions of minute bars are stored, the daily record's open before;
its close is the official closing print; minute bars are Alpaca's, used for a session only when they cover it (first bar
within 15 minutes of the open, last within 15 minutes of the close); options are priced at the close of the last bar
ending no more than 300 seconds before the instant, with no slippage on prints, $0.65 per contract per side.

## Conversation

- **What next.** "What would you investigate next?" answers from the result: too few trades or days, whether it could be
  luck, whether the entry minute matters, the losing trades, sizing; each suggestion has its reason and runs on a click,
  and nothing already asked in the notebook is suggested again.
- **A group against the rest.** "Is there anything interesting about the losing trades?" compares the group with the rest
  of the same result, leads with what was known when the trade was entered (gap, previous day, VIX, weekday, entry time,
  premium), reports for each difference how often random splits of the same observations give one as large, and says how
  many things were compared. What happened during the trades (the underlying moved against a put) is shown as how they
  lost, never as a filter.
- **Selections.** Rows selected in the explore sheet can be asked about in the notebook ("What sets them apart?").
- **Unanswerable questions.** Forecasts, event calendars, indicators, unstored tickers and option prices before 2024 are
  declined with the reason, and the closest question that can be answered is a button; nothing changed runs by itself.

## Position sizing

Typed ("spend $500 a trade", "risk 2% of a $25k account", "start with 10k and reinvest", "stop trading for the day
after losing $200", "buy 3 contracts") or set in the strategy sentence, compiled onto the existing engine:

| said | engine |
| --- | --- |
| N contracts | fixed |
| $ of premium per trade | `premium_dollars`: floor(budget / premium of one contract); a contract over budget is not bought, and counted |
| % of the account in premium | `premium_pct` |
| $ or % of the account at risk | `risk_dollars` / `risk_pct`: risk of one contract = the loss at the stop when there is one (premium x stop %), else the maximum loss (the premium of a bought option) |
| start with $X | the backtest's starting capital |
| reinvest | percentages of current equity (otherwise of the starting capital) |
| stop for the day after losing $X | no new entry once the strategy's positions closed that session lost $X |

On real data (IWM 0 DTE puts at 10:15 after a 1% up day, out at 15:00, 140 trades): $400 of premium bought 2 to 14
contracts (median 5), -$3,364.60; $250 at risk, 1 to 8, -$1,372.30; $200 at risk, 1 to 7, -$1,534.70. Each answer listed
the premium paid (median $164 to $363), that it is also the most a bought option can lose, and the realised losses, and
each was re-priced independently to the cent.

## Charts

The rest of the session after up and after down first candles (median and middle half, from the candle's close), the
spread of trade results (bars open those trades), the question before and after a change, and the comparison table for
a group. Existing: outcome counts, the outcome matrix, price-path bands, timing and breakdowns, the account curve,
version comparisons and sweeps, the payoff illustration.

## Consolidation

Removed from the request path: V6's tool-loop planner and V7's form planner and their routing (`Agent.planner`,
`_model_reading`, `_cross_checked`); the functions stay as libraries for their unit tests. Unified: one engine for
first-candle prediction questions; one definition of a session's open and of a covered session across studies,
profiles and the reference. Twenty API endpoints are called by neither the frontend nor the tests; they are kept for
backward compatibility (scripts may use them) and listed in `docs/v9_evidence/unused_endpoints.txt`. Saved research is
unchanged; study results computed before V9 are recomputed once (the engine version is part of the cache key).

## The application, used

Three conversations, not scripted in advance, in the real application on the stored history (no model selected, as
installed). Problems found and fixed in them are numbered.

1. **Laptop, 1440×860.** "Buy ATM IWM 0DTE puts at 10:15 ET after a 1% up day, sell at 15:00 ET, spend $400 a trade",
   then "is there anything interesting about the losing trades?", "what would you investigate next?", "risk $300 per
   trade instead", "risk $250 per trade", "risk $200 per trade". Fixed: (1) "Why it matters" said "a positive total" when
   the total was -$3,364.60; (2) the check line printed money without a dollar sign; (3) the comparison of the losing
   trades led with the day's own move (puts lose when the price rises: how they lost, not a lead) and now leads with
   what was known at entry; (4) an empty "Why it matters" heading; (5) "what next" suggested a question already asked;
   (6) the change line printed the sizing as raw data; (7) the before/after chart compared with the wrong earlier
   answer; (8) risk-based sizing was not re-priced independently. The dev server's reload interrupted one run while
   code was being edited; it was asked again.
2. **Laptop.** "does a green first 15 minutes on IWM tell you anything about the rest of the day?" Read as the agreement
   of the first 15-minute candle with the rest of the session: 50.9% the same way against 50.0% for unrelated directions
   and 51.6% for always guessing up; every count recalculated independently; the new overlay shows the median rest of
   the day after up and after down candles barely separating.
3. **Desktop, 1920×1080, then phone, 375×812.** "if spy gaps down more than 1% does it usually fill the gap by noon?"
   (15 of 135 days, 11.1%, recalculated the same), then "what if it gapped down but yesterday was green", "undo", and
   the same again. Fixed: (9) the follow-up turned "gaps down more than 1%" into any gap down (said, but not meant) and
   now keeps the 1%: 8 of 60 days, 13.3%; (10) the answer restored by undo was worded without its question ("the
   outcome happened"); (11) "5,774 sessions could not be measured (missing bars)" now says they have only daily prices
   before 2016, so a time of day cannot be checked. On the phone the page has no horizontal overflow; the before/after
   bars collapsed to a stub and were fixed.

Screens: `docs/v9_evidence/screens/` (desktop, laptop, phone). Time to a useful answer in these sessions: a question
read by the built-in readers is answered in 1 to 3 seconds when its data is stored and its result cached, 10 to 25
seconds when it runs (a backtest that downloads missing option days took longer); a reply that needs the local model
adds the model's 2 to 15 seconds (median about 3).

## Tests

- Backend: 643 tests (622 from V8, 19 in `tests/test_v9.py`, 2 in `tests/test_v9_output.py` for the fixes made after
  scoring); all pass except the one skipped without real data. Four legacy tests of the replaced model planners were ported to the V9 reader with the same guarantees
  (`test_v7_planner.py`, `test_v6_security.py`); one routing test now expects the unified candle-question engine
  (`test_v4_research.py`).
- Browser: 37 tests (33 from V5 to V8, 4 new in `frontend/e2e/v9.spec.ts`), all pass; TypeScript and the production
  build are clean.
- The independent recalculation battery: `python -m strata.research.reference_report` (all the same, above).

## What V9 does not do

- Hosted models were not measured. The reading accuracy above is the built-in readers' and qwen3:8b's.
- The 90% / under 2% targets were missed: on the sealed set, 38% of answerable messages read correctly and 25% answered as
  a different question without saying so, in the primary configuration (35% and 20% without a model, 47% and 27% with
  the model reading first).
- Quote (bid/ask) execution is unverified until Databento is connected.
- The sealed V9 set is now used: its misses are described above, so a reader change made because of them would have to
  be measured on a new set written before the change.
- The sealed items were written by the same person as the code; two conversation turns paraphrase the brief's examples,
  and four sizing items were looked at during development (disclosed in `docs/v9_evidence/sealed_v9.txt`).

## What changes on your machine

- A local model, qwen3:8b (5.2 GB), is installed in Ollama. Strata does not use it until you select Ollama under Settings,
  Language model (then `auto` reads with the built-in readers first and asks the model about what they cannot place).
- Study and backtest results computed before V9 are recomputed once when asked again (engine version 7.0.2).
- Notebooks created while testing: f1f9a59d3569 (IWM puts with sizing), 5c8285746aed (IWM first candle),
  46ef75799a34 (SPY gap fill). They can be deleted from the notebook list.
