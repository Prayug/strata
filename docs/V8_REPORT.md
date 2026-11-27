# Strata V8: delivery report

Written 2026-11-27. Every figure below was produced by the application or the test suites on the development machine
that day, from the stored history (SPY, QQQ and IWM daily since 1993/1999/2000, one-minute since 2016, 89.1 million
one-minute option trade bars from 2024-01-18). Evidence is in `docs/v8_evidence/`. Where something was not measured,
or is blocked, it says so.

**Short version.**

1. **The first screen is now a question box, and the answer comes first, in words.** "Is the first five-minute candle a
   good predictor of the day?" is answered on the stored history in under a second when cached and about ten seconds
   when not: *"The first 5-minute candle matched the direction of the whole day 58.0% of the time. However, it matched
   the direction of the rest of the session, after the candle closed, only 50.7% of the time. The candle looks
   predictive mainly because its own move is part of the whole day."* Then the outcome matrix, the baselines it is
   compared with (50.0% if the directions were unrelated; 53.9% for always guessing "up"), and four numbers, three of
   which open the exact days behind them.
2. **A new question reader** (strata/ai/semantics.py) maps every clause of a sentence to a slot or reports it as not
   understood, and a conversation layer (converse.py) edits one investigation across messages ("use 15 min", "nah do
   qqq", "only when yesterday was red", "what if I bought calls instead?", "compare both", "undo"), saying each change
   in one line.
3. **Measured on sentences nobody tuned on, it is better than V7 but not good enough.** On the sealed V8 set (written
   before development, scored once): 37 of 52 questions read correctly, 4 asked a focused question back, **11 (21%)
   were read wrongly without saying so**. In conversations, 25 of 35 turns were right and the context of the
   conversation was kept 16 of 16 times. After fixing the general causes of those errors (not the sentences), a fresh
   set written afterwards scored 19 of 28 correct, 7 asked, **2 silent (7%)**. These are small samples written by the
   same author as the reader: they are indications, not a guarantee.
4. **Four of ten sealed answers matched an independent calculation within tolerance.** The first run of that part was
   invalid (the evaluation harness gave the runs an empty data folder, so they failed); it was re-run with only the
   harness fixed and no change to the application. The six differences are explained below, one of them a real
   interpretation error.
5. **No independent tester was available.** The thirteen usability workflows were run by script against the real
   application, with clicks, needed knowledge and seconds recorded; that measures the application, not a person.
6. **The Docker image was not rebuilt**: Docker Desktop is waiting for an administrator password on this Mac. The
   commands are at the end.

## What was measured, and what is blocked

| | Status on 2026-11-27 |
| --- | --- |
| Answer-first notebook, three detail levels, explorable numbers, adaptive charts | built and used in the browser on the stored history; 5 browser tests (`frontend/e2e/v8.spec.ts`) |
| Sealed V8 question set (52 questions, 8 conversations) | scored once: 37/52 correct, 4 asked, 11 silent; conversation turns 25/35, context kept 16/16 |
| Sealed answers against independent calculations (10) | first run invalid (harness); re-run with only the harness fixed: 4/10 within tolerance |
| Fresh set written after the post-sealed fixes (28 questions, 2 conversations) | scored once: 19/28 correct, 7 asked, 2 silent; conversation turns 8/8 |
| The thirteen usability workflows | all completed by script on real data, 23.5 s of application time in total; **no independent tester** |
| Historical bid/ask (Databento) | **BLOCKED: no key.** Unchanged from V7; the plain-language data summary says so |
| Docker image | **NOT REBUILT: Docker Desktop is waiting for an administrator password** |
| Hosted language models | not used; nothing in V8 depends on a model |

## 1. Understanding the question

V7 read 16 of 34 unseen sentences correctly and 9 wrongly without saying so. The cause was structural: each older
reader matched a whole sentence shape and, when a shape matched partly, ran what it had. V8 adds a layer in front of
them, `strata/ai/semantics.py`:

- **Normalisation** with an offset map back to the typed text: slang and typos ("yday", "tp", "spy's", "intrday",
  "hw"), number words ("five-minute", "half a percent"), within-one-letter spelling for words of five letters or more.
- **A tagger of composable mentions**: instrument, kind of question, events (gaps, the previous session's move, moves
  from the open, the day's change at a time, streaks, VIX, weekdays, VWAP, the first candle), outcomes (touch, close
  against a reference, a level, recover/continue, extremes), anchors (the rest of the day, the whole day, the first
  hour, "within 5 days", "a week after", "by noon"), clock times with their zone, periods. Each mention owns its
  characters; a symbol inside a VIX condition is that condition's, not the subject.
- **Clause coverage**: every word that carries meaning must be claimed by a mention. Unclaimed words stop the run:
  "How often does SPY rise 1% from the open when retail sentiment is euphoric?" is read up to the condition and then
  answered with *"I understood part of this (...) but not 'retail', 'sentiment', 'euphoric', so nothing was run."*
- **A question frame** per kind (frequency, agreement, movement, timing, breakdown, strategy), compiled to the
  existing engine's configurations. The older readers, research graphs and strategy readers are all still there: when
  the new layer cannot read a sentence completely, they try, and their reading is then held against what the new layer
  did recognise (an event, a clock time, a percentage, an exit, an order of events). Anything they dropped is listed as
  not represented and the run waits for "run without it" instead of answering a different question.
- **Ambiguity measured, not guessed**: "recover", "bounce" and "keep falling" have more than one usual meaning; each is
  measured side by side and said ("after a gap, recovering first means the gap fills"). A candle length that was not
  stated is measured at 5, 15 and 30 minutes. "Up at 10:30" is read as up on the day, as a quote shows it, and said.
- **Unanswerable questions are refused with the reason**: forecasts ("will spy go up tomorrow"), events without a real
  calendar (FOMC, CPI, earnings: the stored event table is simulated), symbols that are not stored.

What it measured, in order of honesty:

| Set | When written | Result | What it is |
| --- | --- | --- | --- |
| Sealed V8, 52 questions | before development; hash `be3d9323…` recorded | 37 correct, 4 asked, **11 silent** | the estimate for sentences nobody tuned on |
| Sealed V8, 35 conversation turns | same | 25 correct, 5 asked, **5 silent**; context kept 16/16 | same |
| Probe 1, 30 questions | mid-development, scored once | 17 correct, 9 asked, 4 silent | an earlier estimate; development data since |
| Probe 2, 28 questions | after the sealed score and its fixes, scored once | 19 correct, 7 asked, **2 silent** | the estimate after those fixes |
| Development, probe 1 and V6 sets | used while building | 20/20, 30/30, V6 110/110 and 38/40 (0 silent; the V7 code scored 109/110 and 37/40 with 1 silent) | regression lines, not estimates |

By level, the sealed questions were read correctly for 17 of 20 beginner, 15 of 21 intermediate and 5 of 11 expert
sentences. The silent errors were concentrated in strategy sentences that the older strategy reader read partly
(an exit "sold at lunch", "30% tp / 20% sl", "out at 9am pt", "first rallies 0.3% ... and then breaks below the open"),
two data questions ("how did ... do in 2022" lost its condition; "do 0dte calls print on fomc days?" ran instead of
declining) and three questions the research-graph reader took over. After the sealed score, the strategy clause check
was widened (every clock time, every percentage, a stated exit and an order of events must have a place in the draft,
or the run waits and lists them), the common exit phrasings are now read, and a mis-sized kept condition ("gap ups
>1%" read as any gap up) is corrected and said. Probe 2 was written after that, avoiding the sealed set's wording.

The scorer was also corrected after the sealed score: a plan that stopped and listed an unrepresented part is a question
back, not a silent run. The sealed score above used the old rule and was not recomputed.

## 2. Conversations

The notebook keeps one question frame per conversation, a stack of earlier frames, and the question it last asked.
Each change is one line under the message, with Undo:

- "use 15 min" → *Changed the candle length from 5 minutes to 15 minutes. Everything else stays the same.*
- "nah do qqq" → *Changed SPY to QQQ. Everything else stays the same.*
- "Change the previous-day decline to 3.8% and compare the results" → *Now: the previous session fell 3.8% or more
  (was 3%). Everything else stays the same.* (V8 found that this sentence had been changing the outcome instead of the
  condition; it now names the condition, and is a test.)
- "what if I bought calls instead?" → *Turned the question into an option strategy, with the same conditions*, plus
  what had to be assumed (entry when the candle closes, at-the-money same-day contracts).
- A bare number that could replace two things asks which, with one-click answers; a correction that fits one is
  applied. Restoring an old version brings its question back with it.

## 3. Answers first, then evidence, then the method

Every result now has an `answer`: a headline in plain words, what happened, why it matters when it does, what it cannot
tell you, a few numbers (most of them openable), one picture chosen by the kind of question, follow-up questions
derived from the frame, and, behind "Technical details", the intervals, tests, adjustments, the full V7 report and the
configuration. The level chosen is remembered.

- **Pictures by question**: counts split into the days it happened and did not (with the any-day rate where it is a fair
  comparison, and none where it is not: a gap "fills" automatically on days without a gap); the outcome matrix for
  prediction questions with the independence baseline and the always-guess-up baseline computed on the same days;
  price paths from a common start with bands, the median and the extreme days; half-hour bars for time of day; group
  bars with intervals for weekdays and months; the running total with every trade as a dot for strategies; two curves
  for comparisons; bars per setting for sweeps.
- **Statistics in words, correctly**: "That average is not reliably different from zero ... a range that includes
  losing money"; the p-value is *how often a gap at least this large would appear if there were no difference*, never
  the chance that a strategy works; "We tried 252 variations ..." when trials were counted.
- **Comparisons in sentences**, naming each version by what differs: *"With exit at 3:55 PM ET, it made more money
  overall: +$4,075.83 after fees, against +$3,566.83 with exit at 2:30 PM ET. The other had a shorter worst losing
  streak (8 against 11 losses in a row)."* Ties are called ties; whether the difference is within chance is said.

## 4. Every number opens what it is made of

`POST /api/v8/explore` returns the rows behind any number: the days of a study (all, happened, did not, highest,
lowest, ended up/down), the days left out and why, the classified days of a prediction question, the trades of a
backtest with the premium in and out per share and the multiplier, the signals that could not be traded, the sessions
behind a time-of-day or weekday answer. In the notebook they open in a side sheet, with a replay of any day or trade.
Selecting days and asking *what do these days have in common?* compares them with every stored day (gap, previous day,
open-to-close, VIX, weekday mix) and says when the sample is too small. Free-text questions about a selection are not
supported; that is listed below.

## 5. Strategies as sentences

A strategy plan is shown as: *Buy [QQQ] [calls] with [0 days] to expiration at [9:45 AM Eastern]. Only enter when [the
first 15-minute candle closed up] and [the previous session closed down] [+ add a condition]. Take profit when [...].
Stop when [...]. Close any remaining position at [3:55 PM Eastern].* Every bracket is editable in place (times in
Eastern, Central, Mountain or Pacific, remembered); conditions are added from plain templates joined by "and" or "or".
Changes are collected and said before anything runs ("Changed the exit time from 3:55 PM Eastern to 2:30 PM Eastern");
running creates a new version and the old one folds to its result. The rule editor and the IDE are one link away;
ordered events ("first ..., then ...") are still edited there.

## 6. Options trades explained from their own prices

Each backtest answer includes one actual trade told step by step with the contract multiplier of the trade itself.
"Explain this like I'm new to options" walks through a typical trade (what a call is, what it cost per share and per
contract, how much the underlying and the premium moved, and the same money in shares) with a payoff picture marked
*Illustration, not historical data*. "Why did the option lose value if SPY went up?" finds the trades where that
happened and splits the price into intrinsic and time value. V8 also found that the biggest-loss explanation blamed
"time value running out" for a trade that lost because QQQ fell 1.6%; a loss on a wrong-way move is now explained by
the move. Questions asked while a newer version is still running say that they describe the previous result.

## 7. Data, in plain words

`GET /api/v8/scope/{symbol}` and the landing page keep four things apart: underlying prices (daily since 1993 for SPY,
minutes since 2016), real option trades (January 2024 to November 2026, trade prices, no bid/ask), end-of-day bid/ask
quotes (January 2020 to December 2025), theoretical prices (estimates, only when chosen); and intraday bid/ask, which
is not stored, so quote-based execution has not been verified. A plan with nothing priceable now ends with *"Real
historical option trades for SPY are stored from January 2024 to November 2026: option profits can be tested only in
that span."* Nothing is substituted: no synthetic data, no model prices unless chosen.

## 8. The interface

The landing page and the notebook have their own chrome: a question box, starting questions that the stored data can
answer (each checked against the store), earlier investigations as questions and findings, saved strategies, and what
the data can and cannot answer. No infrastructure statistics. The notebook reads like an article: serif headlines,
hairline rules instead of cards, amber only on numbers that open something, green and red only in charts. Earlier
questions fold to their finding; a side index lists them. All specialist tools stay under "All tools" and in the
command palette; the V7 workspace is at `/lab`.

**Replay at 1440 × 860.** In the side sheet the two price charts and the position each have their full height (250,
230 and 110 pixels) without Focus mode, either chart can be made taller in place, and V8 fixed a defect where a
resized panel moved one chart's time window but not the other's (a window change is now shared only on a real
gesture, and a resize restores the shared window).

## 9. The thirteen workflows

Run by script (`frontend/walkthrough/`) on the real application and data at 1440 × 860, from the landing page, without
settings, the classic workspace or documentation. Seconds are the application's time to show the answer, after the
click or Enter; reading and typing time are not included, and several studies were already cached.

| # | Workflow | How | Clicks | Knowledge needed | Seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | Does the opening candle predict SPY's day | clicked a starting question | 1 | none | 0.8 |
| 2 | Change the candle length | typed "use 15 min" | 1 | none | 0.2 |
| 3 | Correct the ticker informally | typed "nah do qqq" | 1 | the ticker | 0.2 |
| 4 | Add a previous-day condition | typed "only when yesterday was red" | 1 | none | 0.2 |
| 5 | Inspect the observations | clicked "Days the candle was wrong" | 1 | none | 0.5 |
| 6 | Turn it into an options strategy | typed "what if I bought calls instead?" | 1 | what a call is | 10.7 |
| 7 | Change entry and exit | clicked the exit time in the sentence, set 2:30 PM | 3 | none | 0.2 |
| 8 | Run the backtest | clicked "Run with these changes" | 1 | none | 9.7 |
| 9 | Why the largest loss happened | typed the question | 1 | none | 0.1 |
| 10 | Compare two variants | typed "compare both" | 1 | none | 0.1 |
| 11 | Could the difference be luck | typed the question | 1 | none | 0.1 |
| 12 | The option prices used | typed "show me the option prices used", clicked the count | 2 | what a premium is | 0.6 |
| 13 | Back to the original investigation | clicked it in the notebook index, unfolded it, opened its answer | 3 | none | 0.1 |

The first scripted run found a real problem (fixed): a question asked right after "Run with these changes" was answered
about the previous version without saying so. Screenshots at 1920 × 1080, 1440 × 860 and 375 × 812 are in
`docs/v8_evidence/screens/`; the phone layout is one column with no horizontal scrolling (a browser test checks it).
**No one unfamiliar with the application has used it.** That test remains to be done.

## Tests

- Backend: 622 tests (24 new in `tests/test_v8.py`); all pass, one skipped where it needs the real stored data.
- Browser: 33 Playwright tests, all pass (5 new). The older specs now start at `/lab`, where the V7 workspace moved. One
  older test compared a New York session date with a UTC date and failed every evening; it now compares New York dates.
- `strata validate`: 14 of 14 steps pass. `strata bench --check`: passes. Typecheck and build: clean.
- Expectations changed in older tests, each because of a deliberate change: the V6 reader test accepts the new reader
  ("frame") where the research graph used to read the sentence.

## What changes on your machine

- The package version is 0.8.0. The engine version is unchanged (7.0.1): no trade or study outcome is computed differently.
- A market study run before V8 is computed once more when asked again: its stored result lacks the price paths the V8
  movement answers draw (`STUDY_OUTPUT` in `strata/api/launch.py`). Backtests are reused as before.
- The landing page is now the V8 question page; the V7 workspace is at `/lab`. Notebooks made before V8 open in both.
- The evaluations ran in scratch databases. Testing in the browser and the scripted walkthrough left five notebooks in
  yours, created 2026-11-27 01:34 to 02:30 UTC: four titled "Is the first five-minute candle a good predictor of the
  day?" (one from manual testing, three from walkthrough runs; the last, `ae5a9c4d7015`, is the complete one) and one
  "On SPY, after the previous session fell 2% or more ...". They can be deleted from the notebook list; their runs and
  registry trials are kept by design.

## What is not done

- **Silent misreadings are not eliminated**: 21% on the sealed set, 7% on the fresh set after fixes, on small sets
  written by the same author as the reader. Strategy sentences remain the weakest part.
- **Sealed answers: 4 of 10 matched within tolerance.** Of the six others: two differ by one or two sessions from the
  independent calculation because the application reads opens and closes from one-minute bars where they exist and the
  check reads daily bars (190 vs 189 of 342; 297 of 1,257 vs 295 of 1,259); one asked for an average move and got a
  frequency (a real misreading by the research-graph reader); one was asked back ("highest", "open", "close" not read);
  one strategy answer did not say in its text that option data starts in January 2024 (fixed afterwards, not re-scored);
  one did not use the words the item required in its headline or first paragraph.
- **No language model reads frames.** The brief allows structured model output; V8's reader is deterministic, and the
  V6/V7 model readers are unchanged and optional.
- **Questions about a selection** are limited to "what do these days have in common?".
- **Ordered events in the sentence builder** ("first ..., then ...") are edited in the IDE, not in the sentence.
- **No independent usability test**, and **the Docker image is not rebuilt** (below).
- Historical bid/ask remains BLOCKED on a Databento key.

## Rebuilding the container

Docker Desktop is running but waiting for an administrator password (its privileged helper). After approving it:

```bash
cd ~/Coding/strata && docker compose build strata && docker compose up -d strata
```

The container publishes on http://localhost:8010 (`.env`). While the native copy (API on 8765) holds the data folder,
the container shows its standby page pointing to the running copy; that is by design (one copy at a time).
