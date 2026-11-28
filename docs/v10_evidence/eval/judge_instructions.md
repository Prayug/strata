# Instructions to the judges of the V10 held-out run

Given word for word, together with `product_brief.md` and `judging_rubric.md`. Written before the held-out set was run.

You are judging how a research tool called Strata handled messages typed by traders. You are given a file of turns. For
each turn you see: the conversation so far, the message, what its author meant (`author.intent`), what they expected
(`author.expected`: answer, ask or decline), other readings they accept (`author.also_fine`), and what Strata did:

- `stage_1_what_strata_did`: its first response. `disposition` is one of answered, waiting (it showed its reading and
  waits for a go-ahead), asked, declined, error. `reading_shown` is the reading exactly as the user saw it;
  `assumed_shown`, `left_out_shown`, `not_represented_shown`, `defaults_filled_in`, `why_it_waits` and
  `change_shown` are the other things it said about the reading; `technical_summary_of_what_was_set_up` is what was
  actually configured to run; `answers` holds each answer's headline, sentences, key numbers and independent-check line.
- `simulated_user_replies` and `after_each_reply`: when Strata asked or waited, a simulated user replied (at most twice),
  and Strata's response to each reply is listed in the same form.

Judge each turn by the rubric. Do not consider whether the numbers are correct (that is audited separately); judge
whether what Strata read, asked, declined or said matches what the author meant, and whether it was said openly.

For every turn return one JSON object on its own line, with exactly these fields:

    {"key": "<the turn's key>",
     "in_scope": true|false,                      by the product brief, is this something Strata is expected to answer?
     "verdict": "R|C|B|A|U|D|X|G|P|W|S|E",          the stage-1 verdict of the rubric
     "after_reply": "R2|S|P|X|E|none",             when a simulated user replied: the final outcome; else "none"
     "offered_nearest": true|false|null,           for D only: did it offer the nearest answerable question without running it?
     "explanation_problem": "<empty, or one sentence>",  a sentence of an answer that contradicts its numbers or claims more than they support
     "why": "<one or two sentences: the material point that decided the verdict>"}

Points that decide between verdicts:

- The reading must match the intent (or an `also_fine` reading) on every material point of the rubric. A default the
  author left open is fine when Strata states it. If `technical_summary_of_what_was_set_up` shows something the
  reading shown to the user does not say and that changes the question, that is S.
- "waiting" with a correct reading on a request the author marked complex is C. On a request that is simple and clear it
  is U. "waiting" with a wrong reading is W (the user can see it).
- A reading that leaves a part out and says so, and then runs: P. If it waits instead of running, judge it as waiting
  (W if what it shows is not what was meant; the user's reply then decides `after_reply`).
- Declining something the product brief says is in scope is X. Declining something out of scope is D.
- When the author expected a question (`expected: ask`) and Strata answered: G if it ran one reading that matches the
  intent and said which reading it took; R if you judge the ambiguity not material; S if the reading it ran is not what
  was meant and it did not ask.
- `after_reply`: R2 when, after the simulated user's reply or replies, Strata ended on a reading or answer that matches
  the intent; S if it then ran something materially different without saying so; P if it ran something different and
  said so; X if it still did not understand; E for an error.

Return only the JSON lines, one per turn, in the order of the file, and nothing else.
