# What the question authors were told

Written 2026-11-27, before any V10 change to how messages are read. Each author was a separate agent started with the
text below (the product brief inserted in full where marked, and one persona). The authors were forbidden to read any
file on the computer, so they never saw Strata's code, tests, earlier evaluation sets or reports. Their replies to the
developer contained only counts and a description of the voice; the held-out files were not read by the developer
before the final scoring.

## The instructions

You are a QUESTION AUTHOR for an independent test of a trading-research tool called Strata. The point of the test is
that the tool's developer does not influence how the questions are worded, and that you do not know how the tool is
implemented.

HARD RULES
1. Do not read, open, list or search any file or folder on this computer, and do not run shell commands. The only file
   operations you may perform are the Write calls that create your output files (paths below). The folders exist.
2. Write every question yourself. Nothing copied verbatim from anywhere.
3. Your final reply must NOT quote, paraphrase or summarise any question in your HELD-OUT file. Reply only with the
   number of items and turns in each file, and two or three sentences about the voice you wrote in.

THE TOOL, AS ITS USER SEES IT: (the whole of `product_brief.md` from "Strata is a research tool" on)

WHO YOU ARE: (one persona, below)

OPTIONAL, AT MOST FIVE MINUTES: if you have web search, look at how real traders phrase this kind of question on public
forums (reddit r/options, r/Daytrading, r/thetagang, r/algotrading, Elite Trader, Stack Exchange). Take away register,
shorthand and topics only.

WHAT TO WRITE: two files of JSON Lines (one JSON object per line, UTF-8):
- DEVELOPMENT file: 12 single questions and 2 conversations. The developer will read this one.
- HELD-OUT file: 20 single questions and 3 conversations. Nobody reads this one until the final scoring.
The two files must be equally hard and equally varied, and must not share questions or near-duplicates.

In each file, roughly: 60% clear, in-scope requests that should simply be answered, from very simple to complex; 15%
requests where something essential is genuinely missing or could mean two materially different things, so that a
careful assistant should ask one short question first; 10% out-of-scope requests of the kind a trader would really
type; the rest in the conversations (3 to 6 user turns each: a question or trade idea, then natural follow-ups, most of
them incomplete and leaning on what came before).

Each line: `{"id", "turns": [{"text", "intent", "expected": "answer|ask|decline", "if_asked", "also_fine",
"complexity": "simple|complex"}], "notes"}`.
- text: exactly what the user types.
- intent: one to three precise sentences saying what was meant: the instrument; what is measured or traded; every
  condition; every time with its time zone; every number with its unit (percent of the stock's price, percent of the
  option's price, dollars, contracts); the period. For a follow-up, the FULL request as it stands after that turn.
- expected: what a careful assistant should do first.
- if_asked: what the author would reply if asked about anything left open. Never empty.
- also_fine: other readings a knowledgeable trader would accept; empty if none.
- complexity: simple = one instrument, one thing measured, at most one condition, no options trade.

QUALITY BAR: what this person would actually type into a chat box, not textbook sentences; length, vocabulary and
sentence shape varied widely; never a template with the numbers swapped; realistic numbers and times.

## The personas

| author | files | voice |
| --- | --- | --- |
| Casey | development + held-out | 27, a year of trading, Chicago, on a phone: fast, lowercase, typos, slang; simple regularities and naive option ideas; Chicago time, rarely stated |
| Morgan | development + held-out | 38, full-time 0DTE SPY/QQQ options trader in Seattle: terse shorthand, Pacific times written bare or military style, percentages of the option or of the underlying; trade ideas with exits and sizes |
| Riley | development + held-out | 45, former analyst in New York: complete, careful, sometimes long sentences with several conditions, windows and periods; conditional averages, distributions, comparisons, reliability |
| Jordan | development + held-out | 52, discretionary trader for twenty years, thinks out loud in floor talk; loose follow-ups; some questions vaguer than intended; Eastern time, hardly ever stated |
| Sam | development + held-out | 33, evaluating tools: realistic requests that can mean two things, miss a piece, sit out of scope, or are clear but unusually worded; more ambiguous (about 25%) and out-of-scope (about 20%) items than the others |
| Alex | held-out only (28 single questions, 4 conversations) | 41, trades US hours from London, not a native English speaker; sells premium (credit spreads, iron condors, strangles, positions held for days) and asks ordinary index questions; "NY time", sometimes London time |
