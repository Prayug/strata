# How the V10 held-out set is judged

Written 2026-11-27, before any V10 change to how messages are read, and before the held-out questions were written.
The developer does not read the held-out questions until they are scored, once, at the end of V10.

## Who does what

- **Authors** (separate agents that never see Strata's code, each with a different trader's voice) write the questions.
  With each turn they record what they meant (`intent`), what a careful assistant should do first (`expected`: answer,
  ask or decline), what they would reply if asked (`if_asked`), other readings a knowledgeable trader would accept
  (`also_fine`) and whether the request is `simple` or `complex`.
- **The developer** runs every held-out turn through the application once per registered configuration and records what
  the application did. The developer does not judge.
- **Judges** (fresh agents that see neither the code nor the development history, nor which configuration produced a
  reply) give each turn one verdict below. Two judges per turn; where they differ, a third decides.
- **A simulated user** (a judge agent holding the author's `intent` and `if_asked`) replies when the application asks or
  waits for confirmation: it picks the offered choice that matches, or types the short answer the author would give. At
  most two such rounds per turn.

## What the judge is shown for a turn

The user's messages so far, the author's fields, the product brief (the definition of in scope), and what Strata did:
whether it answered, showed its reading and waited, asked, declined or failed; the reading in plain English exactly as
the user saw it; the technical summary of what was computed; a question's text and choices; an answer's headline, key
numbers and independent-check line.

## Verdicts (one per turn)

"Material points" are: the instrument; what is measured or traded; every condition (what, direction, size, unit, when);
the time window and time zone; every exit and its unit (a percentage of the option's price is not a percentage of the
underlying's; a clock time is not a price); the size of the trade; the period.

| | verdict | when |
| --- | --- | --- |
| R | right | Strata answered, and its reading matches the intent (or an `also_fine` reading) on every material point. A default the author left open is fine when it is stated (the whole history; New York time for a time only plausible there). |
| C | confirmed right | The request is complex; Strata showed its reading and waited for a go-ahead; the reading is right as in R. |
| B | both shown | The request was materially ambiguous and Strata answered each reading side by side, saying which is which. |
| A | asked, needed | Strata asked one focused question (or offered readings to choose from) about something genuinely missing or materially ambiguous. |
| U | asked, not needed | The request was simple and clear, or already stated what was asked about, and Strata still asked or waited for confirmation. Nothing it showed was wrong. |
| D | declined, right | Out of scope by the product brief; Strata said it cannot answer and why. Recorded with it: did it offer the nearest answerable question, without running it by itself? |
| X | refused or not understood | In scope and clear, and Strata said it could not read or answer it. The user knows nothing was answered. |
| G | guessed, said | Materially ambiguous; Strata ran one reading without asking, the reading matches the intent, and it said which reading it took. |
| P | partly, said | Strata ran something materially different from the intent and told the user which part it left out or changed. |
| W | confirmation wrong | Strata showed a reading for confirmation and the reading shown is wrong; the user can see that. |
| S | silently wrong | Strata answered or ran a materially different question without saying so: a wrong instrument, measure, condition, unit, time, exit or size; a part dropped or invented; an out-of-scope request answered as something else; a ticker replaced; a reading sentence that hides what the technical summary shows was computed. |
| E | error | No usable reply. |

After A, C, U or W the simulated user replies; the turn then also gets a second verdict: R2 (right after the reply), or
S, P, X, E.

## What is reported

In-scope is decided by the judges from the product brief, not by the developer. A turn is **supported** when it is in
scope and the author expected an answer.

On supported turns:

- correct on the first attempt: R + C
- correct after clarification: R2 following A, U or W
- unnecessary clarification: U
- silent failures: S at either stage
- refused or not understood: X; said but wrong: P; errors: E

On turns where the author expected a question: A and B are right; G, S, R are reported separately (R on such a turn
means the judges did not find the ambiguity material).

On out-of-scope turns: D is right (with the share that offered the nearest question); S is a silent failure.

Also reported, for answered turns:

- **Numerical errors**: the independent recalculation's line for each answer (the same, different, or not covered), every
  difference investigated by hand; and a random sample of answered turns recalculated from the raw files by an agent
  that does not use Strata's engines.
- **Incorrect explanations**: answers where a sentence contradicts the numbers shown or claims more than they support
  (a 51% frequency called "usually"; a stop-loss amount called the most that can be lost), flagged by the judges.
- **Time to a useful answer**: seconds from the message to the reply, per configuration, median and 90th percentile.

Everything is reported by configuration, by author, and separately for the author whose questions were never part of
the development data.

## Targets (from the V10 brief) and what counts against them

90% correct interpretation and under 2% silent errors on supported turns. "Correct interpretation" is reported two ways,
both stated: first attempt only (R + C), and including one clarification (R + C + R2). The figures are reported as
measured whether or not the targets are met. The set is scored once; the scoring script refuses a second run.

## Configurations

Registered in `heldout_seal.txt` before scoring: the application as installed (no language model), and the local-model
arrangement chosen on development data. Which of the two the application ships as its default is named there too, and is
the primary result.
