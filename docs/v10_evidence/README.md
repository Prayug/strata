# V10 evidence

What V10 was measured on, and the record of the measurement. The report is [../V10_REPORT.md](../V10_REPORT.md).

| path | what it is |
| --- | --- |
| `eval/product_brief.md` | What Strata is and what is in scope, as its user sees it. Given word for word to the question authors and the judges. Written before any V10 change. |
| `eval/authoring_brief.md` | The instructions the six question authors were given. |
| `eval/judging_rubric.md` | The twelve verdicts and what is reported. Written before the questions. |
| `eval/simulated_user.md` | How the agents reply that stand in for the user when Strata asks or waits. |
| `eval/judge_instructions.md` | What a judge is shown and what it returns. |
| `eval/audit_instructions.md` | How the numerical auditor recalculates a sample of answers without Strata's code. |
| `eval/dev/*.jsonl` | 133 development turns by five authors. Read by the developer and built against. |
| `eval/heldout/*.jsonl` | 147 held-out items, 234 turns, by six authors. Sealed unread on 2026-11-27 and run once. |
| `eval/heldout_seal.txt`, `heldout_seal.json` | The hashes of the held-out files, and the configurations registered before the run with a hash of the code. |
| `heldout_run/A/` | The one held-out run: `state.json` (every turn, what was shown at each stage, the simulated user's replies), `judge_packets.jsonl` (what the judges were given), `judging/` (batches, each judge's verdicts), `verdicts_final.json`, `shadow_offers.jsonl` (what the local-model proposer would have offered), `audit_sample.jsonl` and `audit_report.jsonl`. |
| `heldout_run/rehearsal/`, `rehearsal2/` | Two rehearsals of the procedure on development questions (`state.json` only; they changed the application and the run tool before the held-out run). |
| `tools/` | The scripts used around the run: registration, splitting the packets between judges, the tally, the audit sample, the regression and development measurement, and the prompts given to the agents. |
| `models/` | The local-model proposer benchmark on development turns. |
| `regression_baseline_v9.json`, `dev_baseline_v9.json` | What V9 did on the regression set and on the development questions, before V10. |

The sandbox database and result files of each run (`heldout_run/*/sandbox/`) are working files and are not kept in git.
