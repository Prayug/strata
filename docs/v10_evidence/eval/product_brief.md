# Strata, as its user sees it

Given, word for word, to the authors of the V10 evaluation questions and to the judges of the held-out set. Written
2026-11-27 from the V9 README and report, before any V10 change to how messages are read. It is the definition of "in
scope" for the evaluation: what Strata is expected to answer was fixed here, before the questions existed.

Strata is a research tool one trader runs on their own computer. You type a question or a trading idea in plain English
into one box, the way you would text a research assistant, and it answers from stored price history, never from opinion.
You can then follow up in the same conversation.

## What it has stored

- SPY, QQQ and IWM: daily prices since 1993, 1999 and 2000; one-minute prices since 2016 (the regular session, 9:30 to
  16:00 New York time).
- The VIX, daily, usable as a condition ("when VIX is above 25").
- Options on those three: one-minute trade prices from 2024-01-18 onward. Trades only; there is no bid/ask history.

## What it can work out (in scope)

1. Questions about what the price did in the past:
   - how often something happened, how far the price typically moved, what the typical path looked like;
   - when in the day things happen (the time of the day's high or low, which part of the day moves most);
   - differences between weekdays or months;
   - whether one thing tends to go with another (the early direction and the rest of the day, yesterday and today);
   - any of these only on days that meet conditions: what the previous day or days did (direction, size, streaks), how
     the day opened against the previous close, where the price stood by a certain time of day, the direction of the
     first minutes, the weekday, the level of the VIX, the price against VWAP;
   - measured over a chosen part of the day (between two clock times, or until the close), the next day, or several days
     later, and over a chosen range of years;
   - with outcomes such as: closes above or below the open or the previous close; reaches a given percentage move; gets
     back to the previous close; trades through the previous day's high or low; recovers, reverses, keeps going.
2. Backtests of options trades on the stored option prices (2024-01-18 onward):
   - buying or selling calls and puts; straddles, strangles, vertical spreads, iron condors and butterflies, covered calls;
   - which expiration (same day, or some days out) and which strike (at, in or out of the money, or a delta);
   - getting in at a time of day, optionally only when conditions like those above hold; once or several times a day;
   - getting out at a time of day, after a holding time, when the option's price is up or down by a percentage, when the
     underlying has moved by a percentage, or at expiration;
   - how much to trade: a number of contracts, dollars of premium per trade, dollars or a percentage of the account at
     risk, a starting account size, reinvesting gains, a daily loss limit.
3. Follow-ups in the same conversation: changing one part of the question, undoing, comparing versions; listing the days
   or trades behind a number; asking why a particular trade or day went the way it did; asking what a group of days or
   trades has in common; whether a result could be luck; what to look at next; explaining a term.

## What it cannot do (out of scope)

It should say so, say why, and ideally offer the nearest thing it can answer.

- Predictions, advice, anything about the future.
- Any ticker other than SPY, QQQ and IWM (the VIX only as a condition).
- Scheduled events and news (Fed days, CPI, earnings, expiration calendars) and fundamentals.
- Technical indicators (RSI, MACD, moving averages and the like) in the question box. VWAP is the exception.
- Option prices before 2024-01-18, and fills based on bid/ask quotes.
