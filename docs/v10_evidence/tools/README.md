# The scripts and prompts used around the V10 held-out run

Kept as they were run. They were written in the session's scratch folder, so some carry that folder's absolute path;
`ho` is the wrapper for `python -m strata.ai.heldout_v10` with the run's environment.

| file | what it did |
| --- | --- |
| `register.py` | Wrote the configurations and a hash of every `.py` file under `backend/strata` into `eval/heldout_seal.txt`, before the run. |
| `ho` | `python -m strata.ai.heldout_v10 <command>` offline, on the stored market history. |
| `simulated_user_prompt.txt` | What each simulated user was given (`eval/simulated_user.md` word for word, plus the commands). `PART` was replaced by the user's share, for example `3/6`. |
| `judge_split.py` | Split the judges' packets into batches, twice with different shuffles, so the two judges of a turn saw it in different batches. |
| `judge_prompt.txt` | What each judge was given: the product brief, the rubric and the instructions by path, and its batch. |
| `judge_tally.py` | Merged the verdicts (two judges, a third where they differed) and printed the rubric's figures. |
| `offer_judge_prompt.txt` | What the judge of the local model's offers was given. |
| `audit_sample.py`, `auditor_prompt.txt` | The random sample of answered price-history turns (seed 10) and what the numerical auditor was given. |
| `checks_tally.py` | Counts of dispositions, replies and verification levels over the run. |
| `measure.py` | The regression sets and the development questions through the application (not the held-out measure). |
