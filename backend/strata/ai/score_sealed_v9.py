"""Score the sealed V9 set ONCE, in the configurations pre-registered in docs/v9_evidence/sealed_v9.txt:
  A built-in readers only;  B llm.mode auto with qwen3:8b (rules first; PRIMARY);  C model_first with qwen3:8b.
Then the sealed numerical answers once, with B. Results are written to docs/v9_evidence/. Refuses to run twice: the marker
file docs/v9_evidence/sealed_v9_scored.txt records the run."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVID = Path(__file__).resolve().parents[3] / "docs" / "v9_evidence"
MARK = EVID / "sealed_v9_scored.txt"


def main() -> None:
    from strata.ai import eval_v9 as V
    from strata.ai import evalset_v9 as E
    if MARK.exists():
        sys.exit(f"the sealed V9 set was already scored ({MARK.read_text().strip()}); it is scored once")
    assert E.sealed_hash() == "868db97d358ffaadcca113cc37d48e37adc318237a87c28a5d0bd2bec389d1e9", "the sealed set changed"
    MARK.write_text(f"scoring started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    out = {}
    for name, model, policy in (("A", None, None), ("B", "qwen3:8b", "auto"), ("C", "qwen3:8b", "model_first")):
        t0 = time.time()
        r = V.evaluate("sealed", model, policy=policy)
        r["config"], r["wall_seconds"] = name, round(time.time() - t0, 1)
        (EVID / f"sealed_v9_{name}.json").write_text(json.dumps(r, indent=1, default=str))
        out[name] = r["summary"]
        print(name, json.dumps(r["summary"]), flush=True)
    t0 = time.time()
    a = V.evaluate_answers("sealed", "qwen3:8b", policy="auto")
    a["wall_seconds"] = round(time.time() - t0, 1)
    (EVID / "sealed_v9_answers.json").write_text(json.dumps(a, indent=1, default=str))
    print("answers", a["ok"], "of", a["n"], flush=True)
    with MARK.open("a") as f:
        f.write(f"scoring finished {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n" + json.dumps({"summaries": out, "answers": [a["ok"], a["n"]]}) + "\n")


if __name__ == "__main__":
    main()
