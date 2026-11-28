"""Register the configurations of the V10 held-out run in heldout_seal.txt, with a hash of the code as it stands."""
import hashlib, subprocess, time
from pathlib import Path
root = Path("/Users/prayugsigdel/Coding/strata")
files = sorted((root / "backend/strata").rglob("*.py"))
h = hashlib.sha256()
for f in files:
    h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")
seal = root / "docs/v10_evidence/eval/heldout_seal.txt"
s = seal.read_text()
old = "Configurations to be scored: registered below BEFORE scoring (not yet registered)."
assert old in s
now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
new = f"""Configurations to be scored, registered {now}, before the held-out set was run (the developer has still not opened
the files; the readers are frozen from here until every verdict is in):

A  (primary: what the application does as installed) The built-in readers only; no language model reads or proposes.
   Default settings: ai.confirm_complex on (a reading with several parts waits for a go-ahead), no time zone confirmed (a
   time with no zone is New York time when that is plausible), llm.proposer off. Run offline (STRATA_OFFLINE=1): nothing is
   downloaded during the run. Every answered turn is executed on the stored history.

B  (the local-model arrangement chosen on development data) A plus granite3.3:8b as a proposer whose reply is constrained to
   a schema (llm.proposer = schema). Chosen because, on the 41 development turns the built-in reader stopped on at the
   time of the benchmark, it made 4 offers of which the developer read 3 as right and 1 as wrong; every other model and
   way of asking tested was worse (docs/v10_evidence/models/proposer_dev_summary.txt). The built-in reader has changed
   since that benchmark; the comparison was not repeated. It is not the default. It is measured as a SHADOW of run A
   (strata.ai.heldout_v10): on each turn where A stops, the rewrite B would offer is stored out of sight, is not shown to
   the simulated user, and is judged afterwards against the author's intent. A full second set of conversations is not run.

Scored once: `python -m strata.ai.heldout_v10 start A --shadow-model granite3.3:8b --shadow-variant schema` refuses a second
run. Simulated users follow eval/simulated_user.md; judges follow eval/judge_instructions.md (two per turn, a third where
they differ); the numerical audit follows eval/audit_instructions.md on a random sample of answered price-history turns
(seed 10).

Code at registration: sha256 {h.hexdigest()} over every .py file under backend/strata, each as its path from the
repository root, a zero byte, its bytes and a zero byte, in path order ({len(files)} files).

Before registering, the procedure itself was rehearsed twice on development questions (9 items, then 29 items), with
simulated users. The rehearsals changed the application (docs/V10_REPORT.md, "The application, used") and the run tool
(a fixed share of conversations per simulated user; one waiting turn shown at a time, because a longer listing was cut off
on the simulated user's screen). Nothing of the held-out set was run, shown or read during them."""
seal.write_text(s.replace(old, new))
print("registered", now, h.hexdigest()[:16], len(files), "files")
