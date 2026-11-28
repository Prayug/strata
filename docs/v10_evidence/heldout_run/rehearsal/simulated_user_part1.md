# Simulated user log: configuration `rehearsal`, part 1/1

One line per reply: the key, what was pressed or typed, and why. `pending` returned `[]` at the end.

- `brief/bd-04#1` (reply 1 of 2): typed "A gap down of at least half a percent at the open; the bounce is from the low of the first half hour to the day's high after it." Why: the reading did not match (sizes nobody gave, a fall from the open, measured to the close) and it said the bounce part was not used, so the author's `if_asked` as written.
- `brief/bd-04#1` (reply 2 of 2): typed "No, the bounce is how far it rises from the lowest price of the first 30 minutes to the highest price later that day, not the move to the close." Why: the gap down was now in the reading but the thing measured was still the move to the close; corrected that one most important point. Turn ended with Strata saying it could not read the message.
- `brief/bd-06#1` (reply 1 of 2): typed "Whatever it did from the open until lunch, does it go the other way between about 11:30 and 1:30 New York time." Why: Strata asked what move is being reversed and showed no choices, so the author's `if_asked` as written.
- `brief/bd-06#1` (reply 2 of 2): typed "Does the direction SPY takes in the morning tend to turn around over the lunch hours, roughly 11:30 to 13:30 New York time?" Why: Strata could not read the `if_asked` just typed, so the request said again once in one plain sentence taken from the intent. Turn ended with Strata asking again what the turn is after.
- `jordan/jd-14#1` (reply 1 of 1): pressed the go-ahead. Why: the reading matched the intent on every point (2 same-day SPY put credit spreads, short near 20 delta, long 5 lower, 10:30 New York time, only above session VWAP, held to expiration, whole stored option period). The backtest ran and answered.
- `jordan/jd-14#3` (reply 1 of 2): typed "No, same trade as before with the 50% of credit profit target, plus a stop: out if SPY drops 0.4% from where it was when I sold the spread." Why: the reading was a different trade throughout (one long put, defaults everywhere) and the stop part was marked as not represented, so one sentence from the intent with the `if_asked` stop.
- `jordan/jd-14#3` (reply 2 of 2): typed "No, I'm selling the put spread from before, not buying a put: 2 contracts at 10:30 if SPY is above VWAP, expiring that day, short the 20 delta, long the strike 5 points under it." Why: the two exits were now right but the position was still a long put; corrected the position, the most important thing wrong. Turn ended with a reading waiting for a go-ahead: the spread was back but the profit target and the stop were gone.
- `jordan/jd-14#5` (reply 1 of 2): typed "Not just 10:30. Up to three entries a day, at 10:30, 12:00 and 1:30 PM, each only if SPY is above VWAP at that time." Why: the reading had only the 10:30 entry (and no target or stop); the screen itself said 1:30 was not used, so the entries were corrected first as the most important miss of this message.
- `jordan/jd-14#5` (reply 2 of 2): typed "Up to three entries a day: one at 10:30 in the morning, one at noon and one at 1:30 in the afternoon, each only if SPY is above VWAP at that time." Why: Strata said the three times were not used and asked to say that part another way; a short answer in the author's voice. Turn ended with Strata still asking the same thing; no reply was left for the missing target and stop.
- `jordan/jd-14#6` (reply 1 of 2): typed "No, everything the same, just QQQ, still 5 wide. Then both results together." Why: the reading was a default long SPY put, wrong throughout, and said expiration, entry, strike and exit were left open; the author's `if_asked`.
- `jordan/jd-14#6` (reply 2 of 2): typed "No, not buying a put: sell same-day QQQ put credit spreads, short about 20 delta and long 5 points lower, at 10:30, 12:00 and 1:30 PM whenever QQQ is above VWAP, take profit at 50% of the credit, stop out if QQQ falls 0.4% below its price at entry, otherwise hold to expiration, on a $30,000 account risking 3% of it on each trade with no compounding, no new trades after losing $1,500 in a day, and show it next to the SPY result." Why: only the ticker had changed, the rest was still wrong throughout, so the whole request restated in one sentence taken from the intent. Turn ended with a reading waiting for a go-ahead (expiration about 14 days out, only the 1:30 entry, no SPY comparison).

## Totals

- Replies given: 11 (go-ahead button: 1, choice: 0, typed: 10).
- Turns that needed me: 6 (`brief/bd-04#1`, `brief/bd-06#1`, `jordan/jd-14#1`, `jordan/jd-14#3`, `jordan/jd-14#5`, `jordan/jd-14#6`).
- Turns where Strata had still not answered after my two replies: 5 (`brief/bd-04#1`, `brief/bd-06#1`, `jordan/jd-14#3`, `jordan/jd-14#5`, `jordan/jd-14#6`). 3 of them ended on a question or a could-not-read message (`brief/bd-04#1`, `brief/bd-06#1`, `jordan/jd-14#5`), 2 on a reading waiting for a go-ahead (`jordan/jd-14#3`, `jordan/jd-14#6`).
- `jordan/jd-14#2` and `jordan/jd-14#4` never paused for me.

## Judgment calls

- `brief/bd-06#1` reply 2: the message Strata could not read was the `if_asked` itself, so it was not typed a second time; the request was said again from the intent instead (case 3, second half).
- `jordan/jd-14#1`: "stop loss" was listed under defaults filled in, but the ways out on screen named only holding to expiration, so the reading was taken as matching.
- `jordan/jd-14#5`: two things were wrong at once (entry times, and the target and stop lost in the turn before). One correction per reply was given, entries first; the second reply was used up by Strata asking for the times again.
- `jordan/jd-14#6` reply 2: dollar signs were escaped in the shell so that the amounts reached Strata as typed.
