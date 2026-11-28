"""Counts only: how far the answers of a held-out run were independently verified, the dispositions per stage, and replies."""
import json, sys, collections
from pathlib import Path
cfg = sys.argv[1]
d = Path("/Users/prayugsigdel/Coding/strata/docs/v10_evidence/heldout_run") / cfg
rows = [json.loads(l) for l in (d / "judge_packets.jsonl").read_text().splitlines() if l.strip()]
first = collections.Counter(r["stage_1_what_strata_did"].get("disposition") for r in rows)
print("turns:", len(rows), "| first response:", dict(first))
final = collections.Counter()
chk = collections.Counter()
views = collections.Counter()
nrep = collections.Counter()
kinds = collections.Counter()
for r in rows:
    stages = [r["stage_1_what_strata_did"]] + list(r.get("after_each_reply") or [])
    last = stages[-1]
    final[last.get("disposition")] += 1
    nrep[len(r.get("simulated_user_replies") or [])] += 1
    for rep in r.get("simulated_user_replies") or []:
        kinds["go-ahead button" if rep.startswith("(pressed") else "choice" if rep.startswith("(chose") else "typed"] += 1
    for a in (last.get("answers") or []):
        if a.get("failed"):
            chk["failed run"] += 1
            continue
        c = a.get("independent_check") or {}
        chk[c.get("status") or "no check line"] += 1
        views[(a.get("view"), c.get("status") or "none")] += 1
print("after the simulated user's replies:", dict(final))
print("replies per turn:", dict(sorted(nrep.items())), "| kinds of reply:", dict(kinds))
print("independent check of the answers shown at the end of a turn:", dict(chk))
print("by kind of answer:", {f"{k[0]}:{k[1]}": v for k, v in sorted(views.items(), key=str)})
