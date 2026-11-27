"""The independent recalculation, run over a fixed battery of questions on the stored history, written to
docs/v9_evidence/reference_checks.json. `python -m strata.research.reference_report` re-creates it."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

QUESTIONS = [
    "How often does SPY close above its open on Mondays?",
    "When QQQ gaps down 0.5% or more, how often does it fill the gap the same day?",
    "How often does IWM close above the previous close the day after a 1.5% down day?",
    "How often does SPY rise 0.7% from the open before noon ET?",
    "How often does QQQ close below its open when VIX was above 30 the previous day?",
    "After 4 down days in a row, how often does SPY close higher?",
    "How often does IWM trade below the previous day's low after a gap down of 0.3%?",
    "how often does qqq close above its open after the first 15 minute candle is green",
    "How often does SPY close below the previous close the day after a 2.5% up day?",
    "How often does IWM close above its open when the first 30 minute candle is red?",
    "How often does QQQ touch -0.8% from the open before 11:30 ET after a gap up of 0.5%?",
    "After 3 up days in a row, how often does IWM close below its open?",
    "What does SPY do in the first two hours after a 1% gap up?",
    "What happens to IWM the day after a 2% down day?",
    "average move from 11am ET to the close on Thursdays for QQQ",
]
AGREEMENT = ["Does the first 15-minute candle predict the rest of the day for IWM?", "how often does the first 30 minute candle of QQQ agree with the whole day?"]
BACKTESTS = [("Buy 1 ATM QQQ 0DTE call at 9:45 ET after a down day, sell at 14:30 ET, using option trade prices", {"symbol": "QQQ", "right": "call", "entry": "09:45", "exit": "14:30", "prev": -1e-9, "contracts": 1}),
             ("Buy 2 ATM IWM 0DTE puts at 10:15 ET after a 0.5% up day, sell at 15:00 ET, using option trade prices", {"symbol": "IWM", "right": "put", "entry": "10:15", "exit": "15:00", "prev": 0.5, "contracts": 2})]


def run(out: str | None = None) -> dict:
    from strata.ai import semantics as SM
    from strata.ai.agent import Agent
    from strata.ai.draft import compile_draft
    from strata.ai.eval_v8 import InlineJobs, _fresh
    from strata.api import db
    from strata.api.results import load_trades
    from strata.config import get_settings, override_settings
    from strata.data.store import MarketStore
    from strata.research import profile as PF
    from strata.research.reference import Reference
    from strata.research.reference_options import compare_trades, option_backtest
    from strata.research.study import run_study
    store, R = MarketStore(), Reference()
    known = ["SPY", "QQQ", "IWM", "VIX"]
    rows = []
    for q in QUESTIONS:
        r = SM.read(q, known, "SPY")
        d = SM.to_route(r).get("draft")
        res = run_study(compile_draft(d, store)["study"], store, max_sessions_out=9000)
        st = res["stats"]
        ref = R.check_frame(r.frame) or {}
        if ref.get("kind") == "movement":
            same = ref["n"] == st["qualifying_sessions"] and abs(float(st["avg_end_return_pct"]) - float(ref["mean"])) < 5e-4
            rows.append({"question": q, "kind": "movement", "app": [st["qualifying_sessions"], round(float(st["avg_end_return_pct"]), 4)], "reference": [ref["n"], round(float(ref["mean"]), 4)], "same": same})
        else:
            same = (ref.get("n"), ref.get("k")) == (st["qualifying_sessions"], st["target_first"])
            rows.append({"question": q, "kind": "frequency", "app": [st["target_first"], st["qualifying_sessions"]], "reference": [ref.get("k"), ref.get("n")], "same": same})
    prof = []
    t = PF.timing(store, "SPY", measure="high_time", bucket=30)
    rr = R.extreme_bucket("SPY", "high", 30)
    prof.append({"what": "SPY: half hour of the day's high", "app": [t["top"]["start"], t["sessions"], round(t["top"]["share"], 3)], "reference": [rr["bucket"], rr["sessions"], round(rr["share"], 3)],
                 "same": (t["top"]["start"], t["sessions"]) == (rr["bucket"], rr["sessions"])})
    e = PF.event_frequency(store, "SPY", {"key": "gap", "op": ">=", "value": 1.0})
    rr = R.frequency("SPY", [], {"key": "event", "event": {"key": "gap", "op": ">=", "value": 1.0}})
    prof.append({"what": "SPY: gaps of 1% or more", "app": [e["k"], e["n"]], "reference": [rr["k"], rr["n"]], "same": (e["k"], e["n"]) == (rr["k"], rr["n"])})
    before = get_settings().data_dir
    tmp = Path(tempfile.mkdtemp(prefix="strata-refreport-"))
    _fresh(tmp)
    bts, agr = [], []
    try:
        for text in AGREEMENT:
            ag = Agent(InlineJobs(), store, llm_factory=lambda: None)
            sid = db.create_session("ref")["id"]
            ag.post(sid, text)
            for _ in range(3):
                ag.finalize(sid)
            chk = next(((m["content"].get("answer") or {}).get("check") for m in db.list_messages(sid) if m["kind"] == "result"), None)
            agr.append({"question": text, "check": chk})
        for text, args in BACKTESTS:
            ag = Agent(InlineJobs(), store, llm_factory=lambda: None)
            sid = db.create_session("ref")["id"]
            t0 = time.time()
            ag.post(sid, text)
            for _ in range(3):
                ag.finalize(sid)
            rid = next((m["content"].get("run_id") for m in reversed(db.list_messages(sid)) if m["kind"] == "result" and (m["content"] or {}).get("run_id")), None)
            app = load_trades(rid) if rid else []
            ref = option_backtest(args, R)
            cmp = compare_trades(app, ref)
            bts.append({"strategy": text, "app_trades": len(app), "app_pnl": round(sum(x["pnl"] for x in app), 2), "reference_trades": ref["trades"], "reference_pnl": ref["pnl"],
                        "same": cmp["same"], "different": cmp["different"], "only_app": cmp["only_app"], "only_reference": cmp["only_reference"], "seconds": round(time.time() - t0, 1)})
    finally:
        override_settings(data_dir=before)
    rep = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "studies": rows, "profiles": prof, "agreement": agr, "backtests": bts,
           "summary": {"studies_same": sum(1 for x in rows if x["same"]), "studies": len(rows), "profiles_same": sum(1 for x in prof if x["same"]), "profiles": len(prof),
                       "agreement_same": sum(1 for x in agr if (x["check"] or {}).get("status") == "same"), "agreement": len(agr),
                       "trades_same": sum(x["same"] for x in bts), "trades": sum(x["app_trades"] for x in bts)}}
    if out:
        Path(out).write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    import sys
    r = run(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[3] / "docs" / "v9_evidence" / "reference_checks.json"))
    print(json.dumps(r["summary"]))
