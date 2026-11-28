"""Merge the judges' verdicts for a held-out run and print the figures of the rubric. Prints counts only.
Usage: judge_tally.py <config> [--write-disagreements]"""
import json, sys, collections, statistics
from pathlib import Path
cfg = sys.argv[1]
d = Path("/Users/prayugsigdel/Coding/strata/docs/v10_evidence/heldout_run") / cfg
J = d / "judging"
packets = {json.loads(l)["key"]: json.loads(l) for l in (d / "judge_packets.jsonl").read_text().splitlines() if l.strip()}

def load(prefix):
    out = {}
    for f in sorted(J.glob(f"verdicts_{prefix}*.jsonl")):
        for l in f.read_text().splitlines():
            l = l.strip()
            if not l.startswith("{"): continue
            try: v = json.loads(l)
            except Exception: continue
            if v.get("key") in packets: out[v["key"]] = v
    return out
A, B, C = load("A"), load("B"), load("C")
print(f"turns {len(packets)}; judged by first judge {len(A)}, second {len(B)}, third {len(C)}")
same = lambda x, y: x and y and (x.get("verdict"), x.get("after_reply") or "none", bool(x.get("in_scope"))) == (y.get("verdict"), y.get("after_reply") or "none", bool(y.get("in_scope")))
final, disagree, missing = {}, [], []
for k in packets:
    a, b, c = A.get(k), B.get(k), C.get(k)
    if a is None or b is None:
        missing.append(k); continue
    if same(a, b): final[k] = a
    elif c is not None:
        final[k] = c if not (same(a, c) or same(b, c)) else (a if same(a, c) else b)
        final[k] = {**final[k], "third": True}
    else: disagree.append(k)
print(f"agreed {sum(1 for k in final if not final[k].get('third'))}, decided by a third judge {sum(1 for k in final if final[k].get('third'))}, still open {len(disagree)}, not judged twice {len(missing)}")
if "--write-disagreements" in sys.argv and disagree:
    (J / "batch_C01.jsonl").write_text("\n".join(json.dumps(packets[k]) for k in disagree) + "\n")
    print("wrote", len(disagree), "turns for the third judge to", J / "batch_C01.jsonl")
if disagree or missing:
    sys.exit(0)
rows = []
for k, v in final.items():
    p = packets[k]
    rows.append({"key": k, "file": p["author_file"], "expected": p["author"].get("expected"), "complexity": p["author"].get("complexity"), "in_scope": bool(v.get("in_scope")), "v": v.get("verdict"),
                 "after": v.get("after_reply") or "none", "nearest": v.get("offered_nearest"), "expl": (v.get("explanation_problem") or "").strip(), "third": bool(v.get("third")),
                 "disp": p["stage_1_what_strata_did"].get("disposition"), "sec": p["stage_1_what_strata_did"].get("seconds"), "why": v.get("why")})
(d / "verdicts_final.json").write_text(json.dumps(rows, indent=1))

def report(rs, title):
    sup = [r for r in rs if r["in_scope"] and r["expected"] == "answer"]
    n = len(sup)
    c = collections.Counter(r["v"] for r in sup)
    first = c["R"] + c["C"]
    after = sum(1 for r in sup if r["v"] in ("A", "U", "W") and r["after"] == "R2")
    silent = sum(1 for r in sup if r["v"] == "S" or r["after"] == "S")
    pc = lambda x: f"{100 * x / n:.1f}%" if n else "n/a"
    print(f"\n== {title}: {len(rs)} turns; supported (in scope, an answer expected): {n}")
    print(f"   correct on the first attempt (R+C): {first} = {pc(first)}   [R {c['R']}, C {c['C']}]")
    print(f"   correct after one clarification (R2 after A, U or W): {after} = {pc(after)}   -> including it: {first + after} = {pc(first + after)}")
    print(f"   unnecessary clarification (U): {c['U']} = {pc(c['U'])};  asked, needed (A): {c['A']};  confirmation wrong (W): {c['W']};  both shown (B): {c['B']};  guessed, said (G): {c['G']}")
    print(f"   silent failures (S at either stage): {silent} = {pc(silent)};  partly, said (P): {c['P']};  refused or not understood (X): {c['X']};  declined (D): {c['D']};  errors (E): {c['E']}")
    ask = [r for r in rs if r["in_scope"] and r["expected"] == "ask"]
    ca = collections.Counter(r["v"] for r in ask)
    print(f"   author expected a question: {len(ask)} turns: A {ca['A']}, B {ca['B']}, G {ca['G']}, R {ca['R']}, C {ca['C']}, U {ca['U']}, W {ca['W']}, S {ca['S']}, P {ca['P']}, X {ca['X']}, D {ca['D']}, E {ca['E']}; after the reply R2: {sum(1 for r in ask if r['after'] == 'R2')}, S: {sum(1 for r in ask if r['after'] == 'S')}")
    oos = [r for r in rs if not r["in_scope"]]
    co = collections.Counter(r["v"] for r in oos)
    print(f"   out of scope (by the judges): {len(oos)} turns: D {co['D']} (nearest question offered on {sum(1 for r in oos if r['v'] == 'D' and r['nearest'])}), S {co['S']}, other {len(oos) - co['D'] - co['S']} {dict((k, v) for k, v in co.items() if k not in ('D', 'S'))}")
    odd = [r for r in rs if r["in_scope"] and r["expected"] == "decline"]
    print(f"   in scope by the judges although the author expected a refusal: {len(odd)} {dict(collections.Counter(r['v'] for r in odd))}")
report(rows, "ALL AUTHORS")
for f in sorted({r["file"] for r in rows}):
    report([r for r in rows if r["file"] == f], f"author {f}" + (" (no development file by this author)" if f == "alex" else ""))
report([r for r in rows if r["file"] != "alex"], "authors who also wrote development questions")
secs = sorted(r["sec"] for r in rows if isinstance(r["sec"], (int, float)))
ans = sorted(r["sec"] for r in rows if isinstance(r["sec"], (int, float)) and r["disp"] == "answered")
if secs:
    print(f"\ntime to the first response: median {statistics.median(secs):.1f}s, 90th percentile {secs[int(len(secs) * 0.9)]:.1f}s over {len(secs)} turns; answered turns only: median {statistics.median(ans):.1f}s, 90th percentile {ans[int(len(ans) * 0.9)]:.1f}s over {len(ans)}")
print("explanation problems flagged by the judges:", sum(1 for r in rows if r["expl"]))
print("by simple/complex (supported turns):", dict(collections.Counter((r["complexity"], r["v"]) for r in rows if r["in_scope"] and r["expected"] == "answer")))
