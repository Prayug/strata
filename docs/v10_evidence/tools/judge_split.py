"""Split the judge packets into batches (files), without printing their content. Usage: judge_split.py <config> <per_batch>"""
import json, sys, random
from pathlib import Path
cfg, per = sys.argv[1], int(sys.argv[2])
d = Path("/Users/prayugsigdel/Coding/strata/docs/v10_evidence/heldout_run") / cfg
rows = [json.loads(l) for l in (d / "judge_packets.jsonl").read_text().splitlines() if l.strip()]
# whole conversations stay together; two different splits (A and B) so that the two judges of a turn see different batches
by_item = {}
for r in rows:
    by_item.setdefault(r["key"].split("#")[0], []).append(r)
for name, seed in (("A", 1), ("B", 2)):
    items = list(by_item.items())
    random.Random(seed).shuffle(items)
    batches, cur = [], []
    for k, rs in items:
        if cur and len(cur) + len(rs) > per:
            batches.append(cur); cur = []
        cur += rs
    if cur: batches.append(cur)
    out = d / "judging"
    out.mkdir(exist_ok=True)
    for i, b in enumerate(batches, 1):
        (out / f"batch_{name}{i:02d}.jsonl").write_text("\n".join(json.dumps(x) for x in b) + "\n")
    print(name, len(batches), "batches,", [len(b) for b in batches])
