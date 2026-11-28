"""Regression sets and the development questions through the application: counts, and a page to read."""
import json, sys, time
from collections import Counter
from pathlib import Path
from strata.ai import eval_v10 as V

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
t0 = time.time()
if "--no-regression" not in sys.argv:
    res = V.run_regression()
    for name, s in res["sets"].items():
        print(f"{name:16s} {s['summary']}")
    print("REGRESSION TOTAL", res["total"], f"{time.time() - t0:.0f}s")
    (out / "regression_now.json").write_text(json.dumps({"total": res["total"], "failures": V.regression_failures(res), "confirmed": res["confirmed"]}, indent=1, default=str))
items = []
for f in sorted((V.EVAL_DIR / "dev").glob("*.jsonl")):
    for it in V.load_authored(f):
        it["id"] = f"{f.stem}/{it['id']}"
        items.append(it)
obs = V.observe_items(items, fast=True)
tab = Counter()
cx = Counter()
for it in obs:
    for t in it["turns"]:
        s = t["saw"]
        d = s["disposition"]
        why = ""
        if d == "waiting":
            rv = (s.get("draft") or {}).get("review") or []
            why = "complex" if rv == [V.COMPLEX_LINE] or "several parts" in ((s.get("question") or {}).get("text") or "") else "reasons"
        tab[(t["expected"], d + (":" + why if why else ""))] += 1
        if t["expected"] == "answer":
            cx[(t.get("complexity"), d + (":" + why if why else ""))] += 1
for k in sorted(tab):
    print(k, tab[k])
print("--- by the author's label (answer-expected turns)")
for k in sorted(cx, key=str):
    print(k, cx[k])
(out / "dev_now.json").write_text(json.dumps(obs, indent=1, default=str))
(out / "dev_now.txt").write_text(V.show(obs))
print(f"{time.time() - t0:.0f}s")
