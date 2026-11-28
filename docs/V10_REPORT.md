# Strata V10: delivery report

V10 had one job: read what the user typed, and never answer a different question without saying so. Nothing else was
added until that improved. This report says what was measured, on questions the developer did not write and did not see,
and where the targets were missed.

**The targets were missed, by a wide margin.** On the held-out questions (147 conversations, 234 turns, written by six
separate agents and never seen by the developer), judged by separate agents:

| on the 175 supported turns (in scope, an answer expected) | measured | target |
| --- | --- | --- |
| read correctly on the first attempt | 19 = **10.9%** | 90% |
| read correctly counting one clarification | 32 = **18.3%** | |
| answered a different question without saying so | 41 = **23.4%** | under 2% |
| showed a wrong reading and waited (the user could see it) | 87 | |
| not understood, and said so | 43 | |

The same readers score 302 of 338 on the regression set and answer 63 of 97 development questions at once. Those are
sets the developer had seen. V10 repeated V9's mistake in a new form: the rules were fitted to the sentences in front of
the developer, and they do not carry to new sentences, even from the same authors. What V10 did achieve is narrower: when
it misreads, it now usually shows the misreading and waits (87 turns) or says it did not understand (43) instead of
answering; but it still answered wrongly without saying so on 41 turns, most of them after the user's correction.

A numerical audit of the 26 answered price-history turns, recalculated from the raw files without Strata's code, agreed
on 13 and differed on 12. One was a condition shown in the reading and not applied in the calculation; two were a
measurement that did not match its own description; one was a mislabelled figure; six were a meaningless "0 of N days"
answer from an older answer path; one was a boundary difference that Strata's own check had flagged. They are listed
below, with what was fixed after the run. Nothing fixed after the run has been measured on fresh questions.

## Contents

- [How it was evaluated](#how-it-was-evaluated)
- [The held-out result](#the-held-out-result)
- [What reads a message now](#what-reads-a-message-now)
- [The V9 failures, one by one](#the-v9-failures-one-by-one)
- [Confirming before running](#confirming-before-running)
- [What is declined](#what-is-declined)
- [Local models, measured](#local-models-measured)
- [Independent verification](#independent-verification)
- [Results on screen](#results-on-screen)
- [The application, used](#the-application-used)
- [Regression and development figures](#regression-and-development-figures)
- [Tests](#tests)
- [What V10 does not do](#what-v10-does-not-do)
- [What changes on your machine](#what-changes-on-your-machine)

## How it was evaluated

V9's own development questions said 74 to 83% of readings were right. The sealed set said 38%. The development questions
had been written by the same author as the readers, so they measured what the author had thought of. V10 changes who
writes the questions and who marks them.

- **What is in scope was fixed first.** `docs/v10_evidence/eval/product_brief.md` describes Strata as its user sees it:
  what is stored, what it can work out, what it cannot. It was written from the V9 README before any reader changed, and
  given word for word to the question authors and to the judges.
- **Six separate agents wrote the questions.** Each was given the product brief and a trader's voice, and was forbidden
  to read any file on this computer. Five wrote a development file and a held-out file; a sixth wrote held-out questions
  only, so one voice is in no development data. With each turn the author recorded what they meant, what a careful
  assistant should do first (answer, ask or decline), what they would reply if asked, other readings they would accept,
  and whether the request is simple or complex.
- **The held-out files were sealed unread.** 147 items, 234 turns (175 expecting an answer, 28 a question, 31 a refusal),
  hashed on 2026-11-27 23:46 UTC (`heldout_seal.txt`). The developer never opened them. The run script prints counts only.
- **The rubric was written before the questions** (`judging_rubric.md`): twelve verdicts, from "right" to "silently
  wrong", and a second verdict after a simulated user's reply.
- **The developer runs; others judge.** Every held-out turn went through the application as installed, each item in a
  notebook of its own, and every answered turn was executed on the stored history. When Strata asked or waited, a
  simulated user (an agent holding the author's intent) replied, at most twice per turn (`simulated_user.md`). Two fresh
  judge agents gave each turn a verdict from what the user saw; a third decided where they differed
  (`judge_instructions.md`). A separate agent recalculated a sample of the answers from the raw files without Strata's
  code (`audit_instructions.md`).
- **Two sets that are not the measure.** The 133 development turns were read by the developer and built against. The
  338 turns of the V8 and V9 sets (all seen before) are a permanent regression set, scored by machine.

The configurations were registered in `heldout_seal.txt` before the run, and the set was scored once.

## The held-out result

One run, on 2026-11-27 (registered 2026-11-28 04:01 UTC, code hash `fc676275907d3c5a...`), of the application as
installed: the built-in readers, no language model, the go-ahead on requests with several parts switched on. The record
is in `docs/v10_evidence/heldout_run/A/`.

**What happened, before any judging.** Of 234 turns, Strata's first response was an answer on 43, a reading shown for a
go-ahead on 97, a question on 63 and a refusal on 31. The six simulated users gave 287 replies: 19 were the go-ahead
button, 7 a choice, and 261 were typed, almost all of them corrections. After the replies 108 turns had an answer, 63
still showed a reading waiting, 29 still asked and 34 were declined.

**The verdicts.** Two judges per turn; they agreed on 225 of 234 turns and a third judge decided the other 9. (The
agreement was checked for copying: the batch judged before any other verdict existed agrees with its counterparts on 29
of 30 turns, and no two reasons are the same text.) The judges found 175 turns supported (in scope with an answer
expected), 28 where the author expected a question, and 31 out of scope.

| supported turns (175) | turns | share |
| --- | --- | --- |
| R: answered, reading right | 14 | 8.0% |
| C: complex, reading shown and right, waited | 5 | 2.9% |
| **correct on the first attempt (R + C)** | **19** | **10.9%** |
| A: asked, and it was needed | 4 | 2.3% |
| U: asked or waited when it was not needed | 7 | 4.0% |
| W: showed a wrong reading for confirmation | 87 | 49.7% |
| X: said it could not read or answer it | 43 | 24.6% |
| S: answered a different question without saying so | 15 | 8.6% |

After the simulated user's replies (at most two):

| first verdict | ended right (R2) | still not understood | then silently wrong | error |
| --- | --- | --- | --- | --- |
| W (87) | 8 | 58 | 20 | 1 |
| X (43; 13 had nothing to reply to) | 2 | 24 | 4 | |
| U (7) | 4 | 1 | 2 | |
| A (4) | 1 | 3 | | |

- **Correct counting one clarification (R + C + R2 after A, U or W): 32 of 175 = 18.3%.**
- **Silent failures at either stage: 41 of 175 = 23.4%** (15 at the first response, 26 only after a reply).
- Unnecessary clarification: 7 (4.0%). Errors: 1.
- By the authors' own label: of 43 simple supported turns, 7 were right, 19 showed a wrong reading, 10 were not
  understood, 6 were asked about needlessly and 1 was silently wrong. Of 132 complex ones, 12 were right or confirmed
  right.
- First turns of a conversation (106): 11 right or confirmed right. Later turns (69): 8 right, 29 not understood, 12
  silently wrong. Follow-ups are read worse than first messages, because a first reading that was wrong is what the
  follow-up then builds on.

Where the author expected a question (28 turns): 3 were asked as needed, 3 were answered with the reading said, 10 showed
a wrong reading, 8 were not understood, 3 were silently wrong; after the reply 4 ended right and 8 silently wrong.

Out of scope (31 turns): 26 were declined with the reason, 14 of those with the nearest answerable question offered; 4
showed a reading and waited; 1 was answered as something else.

**By author.** The sixth author has no development file. The result is no worse for that voice than for the five whose
development questions the readers were built against, which says the readers were fitted to particular sentences, not to
an author's way of writing.

| author | supported turns | first attempt | with one clarification | silently wrong |
| --- | --- | --- | --- | --- |
| alex (no development file) | 37 | 13.5% | 18.9% | 21.6% |
| casey | 30 | 10.0% | 26.7% | 16.7% |
| jordan | 29 | 0.0% | 6.9% | 31.0% |
| morgan | 28 | 17.9% | 25.0% | 17.9% |
| riley | 27 | 3.7% | 7.4% | 40.7% |
| sam | 24 | 20.8% | 25.0% | 12.5% |
| the five with development files | 138 | 10.1% | 18.1% | 23.9% |

**Time to a first response**: median 0.1 seconds, 90th percentile 0.5 seconds over all turns; on the 43 turns answered at
once, median 0.7 seconds and 90th percentile 22.8 seconds (an options backtest).

**What it got wrong, in the judges' words, grouped.** The misreadings are not a few repeated slips. They are spread over
what a question can say:

- *A state at a time of day as a condition* ("still below it at 11:00", "up more than 0.6% on the day at 3 PM"): read as
  the previous session's move, or dropped.
- *Two groups in one question* (VIX below 15 against VIX above 20; up mornings against down mornings; one period against
  another): merged into one contradictory set of conditions, or one half dropped.
- *Measures other than a frequency*: the typical size of a move, the longest streak, returns by calendar month, the
  path of the day and the time of its low, a comparison with "all Tuesdays".
- *Trade structures and their parts*: a call butterfly was run as one long call; strangles by a distance in percent;
  shares held against calls; five spreads read as one; "$500 of premium" and "2% of the account" as the size.
- *Exits*: a stop when the price is up 150% was read as a profit target; an exit three trading days later; "let it
  expire" overridden by a default exit.
- *Expirations*: the next day, the Friday of the following week.
- *Follow-ups about a result* (list the losers, why did the worst trade lose, could it be luck) on a conversation whose
  earlier turns never ran: "there is no finished run".
- *Corrections*: "I could not read that", "Nothing changed", "that is already how the current version is set up" while
  the reading on screen still showed the old value.

**How the silent failures happened.** At the first response (15): a call butterfly run as a long call, twice, with the
strike labelled "not stated"; a trade "on those mornings" run on every session; a list of "the days that matched your
rules" that was every session; answers from an older path with no reading and an unnamed "outcome". After a reply (26):
a correction changed one part of a reading that was wrong in several, and the changed reading ran; or a restatement was
taken by an older reader and run as an unconditioned study.

**The local model as a proposer (configuration B), measured as a shadow.** On the 148 turns where the application
stopped, granite3.3:8b with the constrained reply offered a rewrite on 7 (median 2.7 seconds a call). Two judges each
found 2 of the 7 right, 4 partly right and 1 wrong. On the other 141 the rewrite was rejected by the checks or not made:
61 the built-in reader could not read either, 43 changed the content, 25 the model said could not be done, 12 it said
were missing something. It would not have changed the result: the proposer can only rewrite toward what the built-in
reader reads, and that is the part that is too narrow.

**Numerical audit.** A separate agent recalculated all 26 answered price-history turns from the raw price files, with its
own code. 13 agree, 12 differ, 1 could not be checked:

| what differs | turns | what it is |
| --- | --- | --- |
| "the outcome happened on 0 of N days (0.0%)" with no reading | 6 | A study of how the price moved (which has no outcome to count) answered by the older frequency wording, after a restatement that an older reader took. The day counts are right; the "0 of N" is meaningless. |
| the condition in the reading was not applied | 1 | "When it is below the open at 12:00: how often it closes above its open" counted all 2,741 sessions (54.3%); on the 1,254 sessions that meet the condition it is 25.6%. The reader held the condition, the handoff to the study engine dropped it without a word. The worst defect found. |
| the measurement is not the one described | 2 | "From that session's close to the close 5 sessions later" was computed from the next session's open (+0.85% against +0.62% or +0.79% by the two readings of the sentence). |
| a figure under the wrong label | 1 | Of two variants, the one labelled "the next session closes above its own open" was the close five sessions later against that open. |
| an undisclosed second level | 1 | "+1% from the open" was counted as "+1% before -0.5%", which was asked for but not said in the answer; its 19.4% leaves 780 undecided days out of the denominator without saying so. |
| a boundary difference | 1 | "Trades below the previous session's low": 3,881 of 8,515 by the engine, 3,870 by Strata's own check (which flagged the disagreement on screen), 3,824 strictly below by the auditor. |
| could not be checked | 1 | A comparison whose headline showed internal labels and 0.0% on both sides. |

Strata's own independent check, on the answers shown at the end of a turn: 12 fully verified, 32 partly (options
backtests), 18 not covered, 2 disagreements shown, 15 answers of kinds that carry no check line, 1 failed run. Of the
audit's 12 differences, 10 were on answers the check did not cover and 1 on an answer it had flagged; the dropped
condition was in an answer it did not cover.

**Explanations.** The judges flagged a sentence that contradicts the numbers or claims too much on 44 turns. The
repeated ones: "allowing for how few days there are" said of thousands of days; for a single short call or put, "the most
such a trade can lose is the distance between the strikes less the credit", which is false (there is one strike and no
such cap); "the premium is the most the trade could lose" next to a larger realised loss of a position of several
contracts; a negative t-statistic called "above the conventional threshold" and a net loss an "edge".

### What was changed after the run

The run and the audit exposed defects that produce a wrong number or a silent wrong answer. These were fixed afterwards,
each with a test in `backend/tests/test_v10_heldout.py`. **None of this has been measured on fresh questions**, and the
figures above are the result of V10, not these.

| found by | defect | change |
| --- | --- | --- |
| audit | A condition in the reading was not applied ("below the open at 12:00": all 2,741 sessions counted). | The study engine can no longer leave a condition out: a question with such a condition is run by the relation engine, which applies it (321 of 1,254 sessions, 25.6%, the auditor's figure, fully verified), or it is said that the condition cannot be combined. |
| audit | "The outcome happened on 0 of N days" (6 answers). | A study with no target is worded as a movement; an older reader's study that holds nothing of the message is not run. |
| audit | "N sessions later" was measured from the next open, not from the close its description named (2 answers). | Measured from the close of the day the condition was met to the close N sessions after it (SPY after a 2% down day, 5 sessions: +0.79%, median +0.69%, higher on 59.5%, the auditor's figures for the author's meaning). |
| audit | A figure under the wrong label (a five-session outcome labelled "the next session"). | With a horizon in sessions one reading of the outcome is measured, labelled with the horizon. |
| judges | A call butterfly run as one long call (4 turns). | A position Strata cannot test is declined by name. |
| judges | A correction changed one part of a wrong reading and the reading ran (20 turns). | A correction to a reading or a trade that is still waiting shows the corrected version and waits again. |
| judges | "Buy a put on those mornings" ran on every session when no earlier question had been answered. | Days the conversation does not hold are asked for. |
| judges | "Changed review, start" as a label; a comparison of a version with itself. | Why a reading waits is not a change to it. |
| judges | "How few days" said of 8,515 days; a naked short call given "the distance between the strikes less the credit" as its maximum loss; a negative t-statistic called an edge; "the premium is the most it can lose" next to a larger loss. | Each sentence is now said only when it is true of the numbers beside it. |

Left open, known from the run: the list of "the days that matched your rules" can be the days of a different question
than the one on screen; the answer for "+1% before -0.5%" does not say its second level and leaves undecided days out of
its percentage without saying so; "trades below the previous session's low" differs by 11 sessions between the engine and
Strata's own check (shown as a disagreement) and by more from a strict count; sizing by risk measures to the stop where
the user meant the spread's maximum loss.

## What reads a message now

The request path was audited before anything changed. V9 routed a message through, in order: a check for unstored
tickers; the V8 question reader and conversation layer; a correction reader; then about fifteen older readers (execution
stress, challenge, the first-candle reader, the experiment reader, the V5 router, research questions, follow-ups, the
research-graph planner, the hypothesis planner, the research-question detector, the strategy reader), and a V8 check of
whatever they produced. Each could claim a message. A simple question with one unfamiliar word fell through to a planner
that ran a different investigation.

V10 does not add a reader. It puts one typed description of what a message says, and one gate, around the readers that
exist (`backend/strata/ai/intent.py`, `Agent.interpret` in `agent.py`):

| step | what it does | where |
| --- | --- | --- |
| understand | The existing readers produce one reading: a question frame, a trade, or a command. | `Agent._read` |
| resolve | A time with no zone takes the zone the message names or gives away ("the 830 open"), the conversation's, or the one in Settings; otherwise New York time when that is plausible, said; otherwise the one zone that fits; otherwise it is asked. | `intent.message_zone`, `zone_revealed`, `nl_rules.find_times` |
| validate | Everything the message says (each time, percentage with what it is a percentage of, dollar amount, count, expiration, delta, strike, weekday, ticker, year) must be held by the reading as the same kind of thing. A clock time held as a percentage, a percentage of the option held as a move of the underlying, a number that is in the reading and not in the message: each is an issue. | `intent.said`, `held_draft`, `held_frame`, `check` |
| repair | Where the words leave one way to place a thing, it is placed and said. | `intent.repair`, `fill_trade` |
| decide | Run; or show the reading with what is wrong or missing and wait; or ask one question whose answers are complete messages; or decline. | `Agent._establish` |
| data | An options test of a period with no stored option prices is declined; one with no period runs on the stored option history, said. | `Agent._establish`, `_launch` |
| execute, explain | Unchanged engines. The answer carries the reading as a sentence and an independent check. | |

Three rules were meant to hold everywhere: a reading never runs with something of the message unaccounted for; an older
planner's reading is not shown in place of the question reader's when the question reader had read the message as a
simple question; and a follow-up that changes nothing is told so, not re-run as if it had been understood. The held-out
run found where they did not hold: the inventory types quantities, times and names, so a word that carries the meaning
without a number (a butterfly, "on those mornings", "typical") can go unaccounted for; an older reader could still run a
study on a restatement; and a correction could run a reading that was still wrong elsewhere. Those three were closed
after the run (see the held-out result). The first limit is structural: a check built on what the rules recognise cannot
see what they do not.

Existing notebooks and strategies open as before: no stored format changed, and the two new behaviours that change what
a message does (waiting on a complex reading; New York time for a time without a zone) are settings.

## The V9 failures, one by one

Each failure named in the V10 brief was traced to its cause before it was fixed.

| message | what V9 did | cause | what V10 does |
| --- | --- | --- | --- |
| "Out by 1245 PST" | a 1,245% profit target | Four digits were not read as a time, so the number was free, and the target reader took it. Nothing checked that a clock is not a percentage. | `find_times` reads "1245 PST", "0935", "by 2" (an hour after a word that introduces a time). The typed check would also stop it: a clock held as a percentage is "mistyped". |
| "When does IWM put in its high?" | a put strategy | Any option word made the message a trade. | A trade is a structure, an expiration, a strike word, or a trading verb with an option noun (`intent.is_trade`). "Put in its high" is a time-of-day question. |
| "Sell when they're up 40%" | 40% on the underlying | A percentage after "sell when" was read as a move of the ticker. | `about_percent`: the noun the number is attached to, then the subject of its clause ("they" are the options), then the plausibility of the size (SPY does not move 10% while a trade is open). When the words do not say, it is asked: "40% of the option's price, or of SPY's price?" |
| a time window ignored | answered for the whole session | Windows were anchors of one reader and dropped by others. | Relation questions carry their own window ("above yesterday's high before 10:30"), and an unused clock time is an issue. |
| weekday questions | opening-candle studies | The cue for "what kind of question" was read first and swallowed the words around it. | Kinds are read last, from cue words only. Weekday questions have their own measures: an average move, or a share of days ("is Monday or Friday more likely to close red?"). |
| previous-day conditions disappearing | the condition was dropped | The same cue-reading problem: in "when VIX is above 25 how often does QQQ ...", "is above 25 how often" became the cue. | Fixed with it. A condition said and not held is "unused", and the reading waits. |
| an unsupported ticker replaced with SPY | ran on SPY | V9 already declined a named, unstored ticker. A trade that named no ticker still ran on the default. | A trade that names no ticker is asked: "Which ticker should this trade?" |

Found on the way, in the same class (a reading that ran and was not what was said), and fixed:

- "trade under yesterday's high" was read as trading *above* it (an older pattern ignored "under").
- "buying a straddle at 9:35 and selling it at 11:30" became a *short* straddle (any "sell" in the message did that).
- "sell a put spread: short the 20 delta, long the strike 5 points under it" became a put *debit* spread that is bought.
- "let it expire", with an expiration the reader missed ("expires that day"), was compiled as a 14-day option while the
  sentence on screen said "0 days".
- "stop trading for the day after losing $200" added the condition "SPY fell in the previous session".
- "remove the vix condition" moved the whole question onto VIX.
- "how does that change in high-volatility periods?" turned a question about the low into one about the high.
- "exited an hour later" moved a default exit past the close and ignored the rest of the message.
- "Are Fridays any different?" asked after a question about red days dropped the red-day condition.
- A spread, condor or straddle was shown in the strategy sentence as "calls".

## Confirming before running

The brief asks that a simple, unambiguous question be answered at once and that a complex or uncertain one show its
reading first. V10 does both, by the reading's own structure and not by the sentence's length:

- **Simple**: a question with at most one condition and no set clock time; a trade with at most one condition and one
  way out. Answered at once, with the reading above the answer.
- **Complex**: two or more of: a condition, a second step, a set clock time, each exit after the first, a sizing rule,
  several legs. The reading is shown and waits for "Yes, run it". (`intent.complexity`; setting `ai.confirm_complex`.)
- **Uncertain**: something in the message is not in the reading, or is in it as the wrong kind of thing. The reading is
  shown with the reason in red, and waits.
- **Mostly unspecified** ("backtest SPY iron condors"): the defaults are filled in, shown, and it waits.
- **A research plan** (the older planner's reading of a question with several steps) is the one reading that cannot be
  checked slot by slot. It never starts on its own: it is shown as its sentence, with anything of the message it does not
  hold, and waits.

What is waiting is answered by its button or in the user's own words: "yes", "looks right, go ahead", "no, not what I
meant", "the first one", or a correction, which changes the reading that is waiting.

The reading is one sentence, not a form:

> On SPY, when the previous session fell 1% or more: how often it closes above its price at 10:00 New York time
> (7:00 AM Pacific), over the whole stored history.

The ticker, each condition and the period are underlined. Clicking one lists its changes under the sentence ("Remove
this condition", "Since 2020", "QQQ"); choosing one sends it as a follow-up through the same reader as anything typed, so
the change is said back and can be undone. A change made to a reading that is still waiting edits that reading. A trade
is shown as its editable sentence (V8), which now names every structure, and lists underneath whatever it has no slot
for: the strikes, a spread's width, exits on the underlying, weekdays, steps, partial exits, the period, and any value
that was not stated and is a default.

## What is declined

`intent.out_of_scope` holds the product's limits in one place, and is applied to new requests and follow-ups alike,
before any reader can make something else of the message:

- scheduled events and news (Fed days, CPI, jobs, earnings, option expiration weeks, holidays);
- technical indicators (RSI, MACD, moving averages, "its 50-day"); VWAP is the exception;
- fills at the bid and the ask ("redo it buying at the ask"): no quotes are stored, fills are priced from trades;
- forecasts and advice ("is tomorrow more likely red or green?", "should I hold my calls?");
- an options test of a period before the first stored option session (2024-01-18).

The reply says why, says that nothing was run, and where it can, offers the nearest request it can answer as a button,
without running it: the same request without the unsupported part, the same conditions as a question about the past, or
a re-run under worse execution assumptions.

## Local models, measured

The free local-model workflow of V9 is unchanged: with a model configured in Settings, the V9 arrangement (the built-in
readers first, the model for what they cannot place, every model reading checked part by part) still applies.

V10 tested a narrower job for a local model, on the theory that a small model fails less when asked for less: when the
built-in reader stops on a message, the model rewrites the message as one plain sentence of the kind the built-in
reader is known to read (`backend/strata/ai/proposer.py`). The rewrite is used only if every number, time, ticker and
weekday of the message is in it as the same kind of thing and nothing was added, and only if the built-in reader reads
it completely. Even then it is not run: it is shown as a choice under the reply.

Compared on the 41 development turns the built-in reader stopped on, with three ways of putting the task (one prompt
with every form and example; a prompt with only the form and examples of the kind of message at hand; the same with the
reply constrained to a JSON schema). "Right", "partly" and "wrong" are the developer's own reading of each accepted offer
against what the author meant; this is development data.

| model | way of asking | turns | offers accepted | right | partly | wrong | median time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3:4b | one prompt, all forms | 41 | 1 | 0 | 0 | 1 | 2.0s |
| qwen3:4b | only the relevant form | 41 | 0 | 0 | 0 | 0 | 2.4s |
| qwen3:4b | relevant form, reply constrained | 41 | 8 | 0 | 2 | 6 | 3.2s |
| gemma3:4b | one prompt, all forms | 41 | 6 | 0 | 0 | 6 | 3.2s |
| gemma3:4b | only the relevant form | 41 | 4 | 2 | 0 | 2 | 2.7s |
| gemma3:4b | relevant form, reply constrained | 41 | 6 | 1 | 0 | 5 | 3.1s |
| qwen3:8b | one prompt, all forms | 41 | 4 | 1 | 0 | 3 | 3.0s |
| qwen3:8b | only the relevant form | 41 | 3 | 0 | 0 | 3 | 4.3s |
| qwen3:8b | relevant form, reply constrained | 41 | 4 | 1 | 1 | 2 | 5.5s |
| granite3.3:8b | one prompt, all forms | 41 | 3 | 1 | 0 | 2 | 3.8s |
| granite3.3:8b | only the relevant form | 41 | 5 | 3 | 0 | 2 | 4.7s |
| granite3.3:8b | relevant form, reply constrained | 41 | 4 | 3 | 0 | 1 | 5.7s |
| gemma3:12b | one prompt, all forms | 6 | 2 | 1 | 0 | 1 | 13.0s |
| gemma3:12b | only the relevant form | 5 | 2 | 1 | 0 | 1 | 7.8s |
| gemma3:12b | relevant form, reply constrained | 5 | 1 | 0 | 0 | 1 | 8.9s |

- Most rewrites were rejected by the checks: 223 changed the content (a number dropped or added), 211 could not be read
  by the built-in reader either.
- Of the 53 that passed, 14 were right, 3 partly right and 36 wrong. The content check cannot see a condition said in
  words and dropped ("the morning after a big red day"), or a unit changed ("10 points below the money" became
  "10-delta").
- Constraining the reply helped the two better models and did not help the others. Showing only the relevant form and
  examples did not help on its own.
- gemma3:12b gave the most careful rewrites and cannot be used here: with the application, a browser and the dev servers
  running it took this 16 GB machine to 19 to 20 GB of swap, and each call took 8 to 17 seconds. It was stopped after
  five turns.
- The best arrangement that fits in memory, granite3.3:8b with the constrained reply, made 4 offers on 41 turns: 3
  right, 1 wrong, at a median 5.7 seconds per call.

So adding a language model does not by itself improve understanding here. The proposer ships **off**. Settings has the
switch; with it on, a suggestion is labelled as a model's suggestion and is never run unless chosen. The arrangement was
also measured on the held-out set as a shadow (see the held-out result). No hosted model was called and nothing was
paid for.

## Independent verification

V9 recalculated frequency, movement and first-candle answers in SQL, apart from the engines, and re-priced one form of
options backtest trade by trade. V10 extends both, and names three levels on every answer:

| level | meaning |
| --- | --- |
| **Fully verified** | A second calculation, written separately in SQL on the raw files, found the same sessions and the same figures. |
| **Partly verified** | Part of the result was recalculated and agrees. The line says which part was not. |
| **Not independently verified** | No second calculation covers this result. Said in so many words. |
| **Disagreement** | The two calculations differ. Shown in a red block above the answer, with the sessions that differ and the instruction to treat the numbers as unconfirmed. |

New in V10 (`backend/strata/research/reference.py`, `verify.py`):

- **Relation questions** with several conditions, a time of day and a second step ("when it opens above yesterday's
  high, how often does it trade back under it before the close"): sessions and counts recalculated.
- **Time-of-day distributions**: the part of the day of the session's high or low, the average move and the average
  high-to-low range of every part.
- **Conditional averages and shares by weekday or month**, with conditions.
- **How often an event happens**, with conditions.
- **Any options backtest** that the full re-pricing does not cover (several legs, targets and stops, sizing rules,
  positions held across sessions): every option entry price is looked up again in the raw option files by a separate
  reader, and every trade's result and the total are re-added from the trades' own cash flows. This is "partly
  verified": which sessions qualified, which strikes the rule chose and exits before expiration are not re-derived, and
  the line says so.

On real data the thirteen question forms tried all agreed with the engine to the session. Comparative research
(a comparison of two versions) is verified through its parts: each version carries its own check. Partial exits and
dynamic sizing are covered only by the fill check, so they are at most "partly verified".

Historical bid/ask execution remains unverified: no quote source is connected, every fill is priced from recorded
trades, and every options answer says so.

## Results on screen

No dashboard, card, grade or page was added. What changed is what an answer says:

- the reading, as a sentence, above every answer, with what was assumed, what was read as the writer's own comment, and
  anything left out;
- the verification level;
- for an options trade, the amounts that get confused, kept apart: the premium paid (or the credit received, and that
  the most a credit trade can lose is larger than the credit), the loss a stop was meant to allow, the losses that were
  actually realised and how often the price jumped past the stop, and the total after fees;
- one sentence that puts the two percentages side by side: "between entry and exit SPY itself moved a median of +0.74%
  on the winning trades and -0.16% on the losing ones, while the position's result was a median of +166% and -93% of the
  premium paid. A target or a stop given as a percentage here is of the position's price, not of SPY's."

A stop is never described as the most that can be lost.

Follow-ups on what is on screen (the losing trades, the selected days, "what do they have in common", "is that just
noise") already existed in V9 and were kept; V10 made them reachable in more wordings and made a follow-up that asks for
another view of the same question ("Are Fridays any different?", "when in the day does that happen?") keep that
question's ticker, conditions and period, and say that it did.

## The application, used

The application was run and used by hand in the browser, at desktop and phone widths, on the real stored history. What
that found, all fixed:

- **Fourteen seconds of empty page** after pressing Enter on a trade: the question now appears at once with "Reading the
  question", and a reading that waits is shown without first checking which sessions have option prices.
- **A wall of data-coverage text** where the go-ahead should be ("None passes every check on stored option
  observations: 0.0% of them ..."): replaced by the reason it waits and one button.
- **A five-minute run that came back with a question instead of a result.** A trade with no period asked for the
  underlying's whole history, found no option prices before 2024, and stopped to ask how to proceed. Option tests now
  run on the stored option history and say so; a dated run asks only when more than a tenth of its sessions cannot be
  priced.
- **The strategy sentence said "calls"** for a put credit spread, and "0 days to expiration" for a trade compiled with 14.
- **A research plan that stopped to ask had no buttons** in the notebook, and showed "Research graph: 9 steps" as its
  reading. It now has a sentence and its choices.
- **The menu of a reading's part ran off a phone's screen**: the choices are now a row under the sentence.
- A reply to something that could not be read was a fixed list of long example prompts. It now says what it was not
  read as, what in it was recognised, and two or three things that can be said at that point.

The scoring procedure was then rehearsed on nine development items, with a simulated user typing its replies. It found a
second kind of problem, in what happens after Strata asks: in five of the nine items Strata was still asking after two
replies. These were fixed before the held-out run, and each has a test:

- **A typed go-ahead was not understood.** Only the button, or the words "run it", ran a waiting reading. "Yes", "looks
  right, go ahead" and "yep that's what I meant" were answered with "I could not read that". A reply that only agrees or
  only refuses now answers whatever is waiting (`intent.go_ahead`); a choice can be named by its label or its place ("the
  first one"); a bare level ("40%") answers the question that asked for it.
- **"Put a stop on it" was read as buying a put**, and replaced the spread on screen. "Put" the verb is no longer the
  option. A part asked for without a level is asked about, and no level is filled in. "No, same trade as before, plus
  ..." sets the refused reading aside and is read against the trade before it.
- **"Buy a put instead"**, said of a spread, changed only the delta and ran. A follow-up that names a position to open is
  now applied, the check flags a reading that still holds the old position, and a change between a sold and a bought
  position waits, because its exits were written for the other one.
- **"at 10:30, 20 delta"** was taken by the older comparison reader as a list of two deltas, 30 and 20, and two
  configurations were run. A number of a clock time is never part of a list, and a comparison whose list cuts through
  another typed quantity is not taken.
- **"Buy it back if it doubles"**, on a sold spread, was a profit target. It is the stop. "Stop at 2x the credit" is asked
  about: the price to buy back at, or the size of the loss.
- **Three entry times a day** ("at 10:30, noon and 1:30") produced "10:30 is not used in this reading" three times over.
  One trade here has one entry time a day; that is now said in a sentence, with what can be run instead.
- **"Does SPY tend to reverse during lunch?"** was answered with "reverse after what?". It is read as the move from 11:30
  to 13:30 New York time against the morning's move, with the hours stated, and recalculated independently. "How far does
  QQQ bounce after a bad open?" shows the three sizes it tries for "bad", and takes the size the user then gives.

A second rehearsal, on 29 more development items that stop to ask or wait, found these, also fixed and tested:

- **An older research planner could still run a different question.** "If SPY is red 2 days in a row and then opens
  lower, does it bounce?" was planned as "SPY fell in the previous session", and ran. A research plan is the one reading
  that cannot be checked slot by slot, so it now never starts on its own: it is shown as its sentence and waits, a streak
  the message states must be a streak in the plan, and when the plan lost a condition the question reader had read, the
  question reader's reading is the one shown.
- **"STC at +40% / -25% or noon pt"** kept the default 3:55 exit: a time said by its name was in nobody's inventory. Noon
  is now a time like any other, in the zone of the message. The correction "the exit is noon Pacific, not 3:55" set the
  exit to 3:55: what a message says a thing is NOT is no longer read as the request.
- **"Between 9:30 and 10"** was read as "from 10:00 to the close". A window said with "between", "the 10 to 11 hour",
  "closed below the day before two days in a row", "north of 25" and "the Friday before" (of a Monday) are now read.
- **A covered call said by its parts** ("own 100 shares, sell the call 30 days out") was a naked short call, and its
  ticker was lost with the sentence that was the writer's own comment ("my brother-in-law swears by covered calls on
  IWM"). Both are read; the comparison with simply holding the shares, which is not calculated, is said before it runs. A
  trade in the shares themselves is declined with the nearest question about the price.
- **"The put 10 points below the money"** had no place, and in a spread "short strike 5 points below the money, 5 wide"
  lost the strike to the width without a word. A distance in dollars from the money is that many listed strikes (SPY, QQQ
  and IWM list a strike every dollar), and is kept apart from the width.
- **"SPY itself has moved more than 0.7%"** was asked about (of the option, or of SPY?). On a straddle or a condor a move
  exit is now worded "in either direction", which is how it has always been run.

## Regression and development figures

Not the measure of V10, and reported for what they are: sets the developer has seen.

**Regression set** (338 turns from the V8 and V9 development, probe and sealed sets; scored by machine against the
answer keys written for those versions; a reading that is complete and waits only for its go-ahead is given it, as a
user would, and counted):

| | read as the key says | asked or waited instead | ran something else |
| --- | --- | --- | --- |
| V9, as delivered | 250 | 67 | 21 |
| V10 | 302 | 34 | 2 |

Of the 302, 28 waited for the go-ahead first. The two that still count as "ran something else" are disagreements with
the old keys, not misreadings: "Buy SPY 0DTE calls at the open, take profit at 25%" runs with the 25% on the option's
price (the V8 key expected a question; V10 asks only when the words leave it open, and "take profit" on an option trade
does not), and a VWAP cross is held in a different notation than the key's.

**Development questions** (133 turns by five independent authors, read by the developer and built against, so not a
measure). What the application does first:

| the author expected | turns | answered at once | complete, waits for the go-ahead | waits, naming what is missing | asks | declines |
| --- | --- | --- | --- | --- | --- | --- |
| an answer | 97 | 63 | 17 | 8 | 7 | 2 |
| a question | 18 | 5 | | 2 | 11 | |
| a refusal | 18 | | | | | 18 |

V9 answered 31 of the 97 at once. Of the 29 simple ones (the authors' label) 28 are answered at once. The two declined
are trades with several fixed entry times a day, which one trade here cannot hold.

These are the figures of the code that was scored on the held-out set. They count what the application did first, not
whether a judge would call the reading right, which is why they looked so much better than the held-out result: on the
held-out set, 87 of the readings shown were wrong. After the changes made after the run the regression set stands at
301 / 35 / 2, and the development questions at 55 answered at once, 17 complete and waiting, 14 waiting with what is
missing named, 9 asked and 2 declined (this harness gives no go-ahead, and a follow-up to a trade that is still waiting
now waits too).

## Tests

- **Backend**: 690 tests pass (`pytest`, the live-service tests deselected as before; one skipped). 7 are regressions
  from the held-out run (`backend/tests/test_v10_heldout.py`) and 35 are new in `backend/tests/test_v10.py`: clock times and zones, the typed inventory, the check and the repairs, relation questions
  and their calculation against a direct count, the confirm flow, what is declined, the reading as a sentence with
  parts, replies in a person's own words, trade follow-ups, research plans that wait, and the proposer's content check.
- **Browser** (Playwright, on a seeded fixture store): 41 tests pass, 4 of them new (`frontend/e2e/zz_v10.spec.ts`): a
  simple question answered at once and marked fully verified; a request with several parts that shows its reading and
  waits, with a part changed by a click; a go-ahead typed in a person's own words; a refusal with its reason. One older
  test caught a regression on the way: with a Pacific zone confirmed, "the first 9:30 candle" was read as 12:30 New
  York. The tests were run with the installed Chrome (`PW_CHANNEL=chrome pnpm e2e`); the Playwright browser download is
  not on this machine and nothing was downloaded.
- The frontend type-checks and builds.

## What V10 does not do

What the held-out run shows V10 cannot do, in order of how much it matters:

- **Read questions it has not seen.** One in nine supported turns was read correctly on the first attempt. The readers are
  rules written against sentences; they were measured at 302 of 338 and 63 of 97 on sentences the developer had read, and
  at 19 of 175 on sentences he had not. More rules of the same kind will raise the first two numbers again and not the
  third. This is the main finding of V10, and the brief's own warning: the generation of complexity added to compensate
  (the typed inventory, the check, the repairs, the gate) made misreadings visible more often than it made readings right.
- **Stay silent-error free after a correction.** 26 of the 41 silent failures came after the user's reply. The change made
  after the run (a correction to a waiting reading waits again) addresses the largest group; it has not been measured.
- **Condition on a state at a time of day** ("still below the open at 10:30", "up 0.6% on the day at 3 PM") except in the
  one form the relation questions cover; compare **two groups or two periods** in one question; measure the **typical
  size** of a move, the **longest streak**, **returns by calendar month**, the **path** of an average day.
- **Trade structures outside its list**: butterflies, calendars, diagonals, ratio spreads (now declined by name); several
  fixed entries a day in one plan; rolling a position; a trade in the shares themselves; a comparison of an option trade
  with holding the shares.
- **Follow up on a conversation whose earlier turns did not run**: listing losers, explaining a trade and "could it be
  luck" need a finished run, and when the first reading was wrong there is none.
- **Explain a term or a method** asked in passing ("what does 15 delta mean", "does a penny through count").
- **Verify everything it answers**: of the answers shown at the end of a held-out turn, 12 were fully verified, 32 partly,
  and 35 not at all or of a kind with no check; the audit's wrong answers were all among the unverified ones.
- **Price a fill at the bid or the ask.** No quote source is connected: every option fill is priced from recorded trades,
  and historical bid/ask execution remains unverified.
- **Use a local model to close the gap.** Measured twice (41 development turns, 148 held-out turns): as a proposer of a
  checked rewrite it makes few offers and most are not right. A model that reads the message itself was V9's approach and
  is not re-measured here.

What the next version should not do is add readers. A fresh untouched question set is needed before any claim of
improvement: the held-out set of V10 has now been read by the developer, and joins the regression set.

## What changes on your machine

- **A time without a zone** is read as New York time when that is plausible, and said. Choose your zone in Settings and
  every such time is read in it. (Until V10 the default was Pacific.)
- **A request with several parts waits** for a go-ahead before it runs. Settings, "A request with several parts", turns
  this off.
- **An options test with no period** runs on the stored option history (from 2024-01-18) instead of asking how to
  proceed.
- Stored notebooks, strategies and results are unchanged, and nothing was downloaded, bought or sent.
