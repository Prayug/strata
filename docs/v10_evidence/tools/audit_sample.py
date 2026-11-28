"""A random sample of answered price-history turns of a held-out run, for the numerical auditor. Prints counts only.
Usage: audit_sample.py <config> <n>"""
import json, sys, random
from pathlib import Path
cfg, n = sys.argv[1], int(sys.argv[2])
d = Path("/Users/prayugsigdel/Coding/strata/docs/v10_evidence/heldout_run") / cfg
rows = [json.loads(l) for l in (d / "judge_packets.jsonl").read_text().splitlines() if l.strip()]
pool = []
for r in rows:
    stages = [r["stage_1_what_strata_did"]] + list(r.get("after_each_reply") or [])
    last = stages[-1]
    ans = [a for a in (last.get("answers") or []) if not a.get("failed") and a.get("numbers")]
    tech = last.get("technical_summary_of_what_was_set_up") or {}
    if not ans or tech.get("type") in ("strategy", None) or tech.get("structure"):
        continue  # options backtests cannot be recalculated from price bars
    a = ans[-1]
    pool.append({"key": r["key"], "reading_shown": last.get("reading_shown"), "assumed_shown": last.get("assumed_shown"), "headline": a.get("headline"), "what_happened": a.get("what_happened"),
                 "numbers": a.get("numbers"), "strata_independent_check": a.get("independent_check"), "technical_summary": tech})
random.Random(10).shuffle(pool)
pick = pool[:n]
(d / "audit_sample.jsonl").write_text("\n".join(json.dumps(x, default=str) for x in pick) + "\n")
print(f"answered price-history turns: {len(pool)}; sampled {len(pick)} (seed 10) into {d / 'audit_sample.jsonl'}")
