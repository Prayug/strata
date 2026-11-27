"""Post-hoc hand audit of the sealed V9 results (same author, after scoring; nothing in the readers changed because of it).

Two rules, applied to every configuration:
  R1 demote: on an item where a question is acceptable (clarify_ok), the pre-registered scorer counts ANY flag or question
     as correct. Here it counts only when it asks about what is actually ambiguous, with a correct partial reading.
  R2 promote: a reading the scorer marked silent that is mathematically the same as an accepted one.
Prints strict counts next to the pre-registered ones and writes docs/v9_evidence/sealed_v9_hand_audit.json."""
import json
import sys
from pathlib import Path

EVID = Path(__file__).resolve().parent

# R1: (response text starts with / contains) -> verdict. Judged one by one from the responses in the result files.
FOCUSED = {
    "does spy bounce?": True,                                   # '"Bounce" after what?' asks exactly what is missing
    "how often does TSLA gap up 2%?": True,                     # unsupported: declined, stored symbols offered (a button)
    "average move from 10am to the close on fridays for IWM": False,  # A: did not read 'close'
    "what happens when the market tanks": False,                # names the vague clause as unreadable; no question about the size of the fall
    ("v9c5", 0): False,                                         # "buy spy puts at 10 when it's down 1% from the open": the condition was not read; '10' not asked about
    ("v9c5", 2): False,                                         # "exit at 12": "I couldn't turn that into a strategy ..."
    ("v9c7", 0): False,                                         # "does a strong first hour ...": partial reading wrong ("by weekday"), then 'strong' not understood
    # C (model first): the requests that cannot be answered were declined with the reason and the closest question offered
    # as a button (expected: a decline with an alternative; not a clarify_ok item, listed so every flagged reply is judged)
    "how does SPY do the day after FOMC?": True,
    "will qqq go up tomorrow": True,
    "how often does RSI under 30 lead to a bounce on QQQ": True,
    "backtest SPY iron condors since 2015": True,
    "what's SPY's average return during earnings season weeks": True,
    ("v9c6", 1): True,                                          # "what about on CPI days?": no calendar stored, closest question offered
}


def equivalent(got: dict) -> str | None:
    """R2: readings that are the same as an accepted one, written differently."""
    if got.get("outcome") == "event:day_return>0":
        return "a positive day's return is a close above the previous close (accepted)"
    conds = [c.replace(" ", "") for c in got.get("conditions") or []]
    if sorted(conds) == sorted(["prev_return<0", "expr:return_pct(1)<=-1.5"]):
        return "return_pct(1) is the last completed session's close-to-close return: the same days as prev_return<=-1.5"
    return None


def audit(name: str) -> dict:
    r = json.loads((EVID / f"sealed_v9_{name}.json").read_text())
    changes, unjudged = [], []

    def judge(key, x):
        g = x["got"]
        if x["status"] == "correct" and g.get("response") in ("flag", "clarify"):
            v = FOCUSED.get(key)
            if v is None:
                unjudged.append((key, g.get("text", "")[:300]))
                return x["status"]
            if not v:
                changes.append({"item": key if isinstance(key, str) else f"{key[0]} turn {key[1] + 1}", "from": "correct", "to": "asked", "rule": "R1", "response": g.get("text", "")[:200]})
                return "asked"
        if x["status"] == "silent":
            why = equivalent(g)
            if why:
                changes.append({"item": key if isinstance(key, str) else f"{key[0]} turn {key[1] + 1}", "from": "silent", "to": "correct", "rule": "R2", "why": why})
                return "correct"
        return x["status"]

    singles = [{**x, "strict": judge(x["text"], x)} for x in r["singles"]]
    turns = [{**t, "strict": judge((c["id"], i), t)} for c in r["conversations"] for i, t in enumerate(c["turns"])]
    cnt = lambda xs, k, f="strict": sum(1 for x in xs if x[f] == k)  # noqa: E731
    sup = [x for x in singles if x.get("supported")]
    out = {"config": name,
           "pre_registered": r["summary"],
           "strict": {"single": {"n": len(singles), "correct": cnt(singles, "correct"), "asked": cnt(singles, "asked"), "silent": cnt(singles, "silent")},
                      "supported": {"n": len(sup), "correct": cnt(sup, "correct"), "silent": cnt(sup, "silent")},
                      "turns": {"n": len(turns), "correct": cnt(turns, "correct"), "asked": cnt(turns, "asked"), "silent": cnt(turns, "silent")}},
           "changes": changes}
    if unjudged:
        out["UNJUDGED"] = unjudged
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["A", "B", "C"]
    res = [audit(n) for n in names if (EVID / f"sealed_v9_{n}.json").exists()]
    for a in res:
        print(a["config"], "pre", json.dumps(a["pre_registered"]["supported"]), json.dumps(a["pre_registered"]["turns"]))
        print("  strict", json.dumps(a["strict"]))
        for c in a["changes"]:
            print("   ", c["rule"], c["from"], "->", c["to"], "|", c["item"][:70])
        for u in a.get("UNJUDGED", []):
            print("   UNJUDGED", u)
    if "--write" in sys.argv or True:
        (EVID / "sealed_v9_hand_audit.json").write_text(json.dumps({"rules": __doc__, "configs": res}, indent=1, default=str))
