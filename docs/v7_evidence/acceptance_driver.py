"""V7 acceptance on the stored dataset, through the running application (the same HTTP calls the interface makes).

Each item is a sentence typed into a research notebook, read by the application, run by the engine on stored observations
and reported by the application. This script only sends the sentences, waits, and writes down what came back.

    python docs/v7_evidence/acceptance_driver.py            # every item, against the application on 127.0.0.1:8765
    python docs/v7_evidence/acceptance_driver.py E G        # some of them (the others are kept from acceptance.json)

It creates notebooks, runs and one paper experiment (item K) in the application it talks to, and item E and others may
fetch missing contracts from the connected provider. It places no order: the application has no order endpoint."""
import json, sys, time, urllib.request, urllib.error
BASE = "http://127.0.0.1:8765"
OUT = str(__import__("pathlib").Path(__file__).with_name("acceptance.json"))

def call(method, path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "detail": e.read().decode()[:400]}

def notebook(title):
    return call("POST", "/api/ai/sessions", {"title": title})["id"]

def title_notebook(sid, title):
    """The application titles a notebook after its first sentence. These are renamed so they can be told from anything typed by hand."""
    call("PATCH", f"/api/ai/sessions/{sid}", {"title": ("V7 acceptance · " + title)[:80]})

PENDING = ("plan", "sweep", "run", "investigation", "discovery", "hypothesis", "stress", "challenge", "graph", "experiment")
def say(sid, text, wait=1800):
    t0 = time.time()
    r = call("POST", f"/api/ai/sessions/{sid}/messages", {"text": text})
    if "_error" in r:
        return {"text": text, "error": r}
    while time.time() - t0 < wait:
        call("POST", f"/api/ai/sessions/{sid}/finalize", {})
        ms = call("GET", f"/api/ai/sessions/{sid}")["messages"]
        open_ = [m for m in ms if m["kind"] in PENDING and (m["content"].get("runs") or []) and not m["content"].get("finalized")]
        if not open_:
            break
        time.sleep(1.5)
    ms = call("GET", f"/api/ai/sessions/{sid}")["messages"]
    i = max(j for j, m in enumerate(ms) if m["role"] == "user" and m["content"].get("text") == text)
    out = {"text": text, "seconds": round(time.time() - t0, 1), "replies": ms[i + 1:]}
    dec = next((m for m in out["replies"] if m["kind"] == "decision" and not m["content"].get("resolved")), None)
    if dec is not None:
        # the stored data cannot cover every qualifying session of the period that was asked for, and the application asks
        # before it runs anything. The answer given here is the first choice it offers: run on the sessions that are covered
        out["decision"] = {"asked": dec["content"]["text"], "choices": [c["label"] for c in dec["content"]["choices"]], "chosen": dec["content"]["choices"][0]["label"]}
        call("POST", f"/api/ai/sessions/{sid}/decide", {"message_id": dec["id"], "choice": dec["content"]["choices"][0]["id"]})
        while time.time() - t0 < wait:
            call("POST", f"/api/ai/sessions/{sid}/finalize", {})
            ms = call("GET", f"/api/ai/sessions/{sid}")["messages"]
            if not [m for m in ms if m["kind"] in PENDING and (m["content"].get("runs") or []) and not m["content"].get("finalized")]:
                break
            time.sleep(1.5)
        out["replies"] = call("GET", f"/api/ai/sessions/{sid}")["messages"][i + 1:]
        out["seconds"] = round(time.time() - t0, 1)
    return out

def brief(reply):
    """What the application said back, without the bulk."""
    out = []
    for m in reply.get("replies", []):
        c = m["content"]
        row = {"kind": m["kind"], "view": c.get("view"), "state": c.get("state"), "interpreter": c.get("interpreter"), "text": (c.get("text") or "")[:1400]}
        if m["kind"] == "plan":
            row["read_as"] = [f"{i['label']}: {str(i['value']).split('   [')[0]}" + ("" if i["status"] == "stated" else f" [{i['status']}]") for i in c.get("items") or []]
            row["unsupported"], row["questions"] = c.get("unsupported"), [q["text"] for q in c.get("questions") or [] if q.get("answered") is None]
        for k in ("run_id", "coverage", "configurations", "summary", "dims"):
            if c.get(k) is not None:
                row[k] = c[k] if k != "coverage" else {x: c[k].get(x) for x in ("priced", "of", "counts", "first", "last") if isinstance(c[k], dict)}
        if c.get("runs"):
            row["runs"] = [r["run_id"] for r in c["runs"]]
        out.append(row)
    return {"sentence": reply["text"], "seconds": reply.get("seconds"), "error": reply.get("error"), "decision": reply.get("decision"), "said": out}

def run_of(reply, view=None):
    for m in reversed(reply.get("replies", [])):
        c = m["content"]
        if m["kind"] == "result" and c.get("run_id") and (view is None or c.get("view") == view):
            return c["run_id"]
    return None

res = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "engine": call("GET", "/api/health").get("engine_version"), "dataset": call("GET", "/api/dataset/options"), "items": {}}
which = set(sys.argv[1:]) or {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"}
def keep():
    json.dump(res, open(OUT, "w"), indent=1, default=str)
try:
    prev = json.load(open(OUT))
    res["items"].update(prev.get("items") or {})
    res["runs"] = prev.get("runs") or {}
except Exception:
    res["runs"] = {}
R = res["runs"]

if "A" in which:  # a full-period strategy on each of the three symbols
    out = {}
    for sym in ("SPY", "QQQ", "IWM"):
        sid = notebook(f"V7 acceptance: {sym} 0DTE calls after a 1% down day, whole stored period")
        rep = say(sid, f"Buy ATM {sym} 0DTE calls at the open after a 1% down day, exit at 3:55 PM Eastern or when {sym} has risen 1%, from 2024-01-18 to 2026-11-25")
        rid = run_of(rep)
        R[f"full_{sym}"] = {"notebook": sid, "run": rid}
        card = call("GET", f"/api/backtests/{rid}/performance?sid={sid}") if rid else None
        out[sym] = {**brief(rep), "notebook": sid, "run": rid, "card": None if not card or "_error" in card else {s["id"]: s["text"] for s in card["sections"]}, "evidence": None if not card or "_error" in card else card["evidence"]["text"]}
        title_notebook(sid, f"{sym} 0DTE calls after a 1% down day, whole period")
        print("A", sym, rid, (out[sym]["card"] or {}).get("net"), flush=True); res["items"]["A_full_period_each_symbol"] = out; keep()

if "B" in which:  # the same signal across the three symbols, as one grid
    sid = R["full_SPY"]["notebook"]
    rep = say(sid, "Compare the same signal across SPY, QQQ and IWM.")
    R["cross_symbol"] = {"notebook": sid, "run": (rep["replies"][-1]["content"].get("run_id") if rep.get("replies") else None)}
    title_notebook(sid, "SPY 0DTE calls after a 1% down day; across symbols; out of sample")
    res["items"]["B_cross_symbol_comparison"] = brief(rep); print("B", R["cross_symbol"], flush=True); keep()

if "D" in which:  # ATM against OTM across entry times (the brief's own sentence: both lists are assumed, and said to be)
    sid = notebook("V7 acceptance: ATM versus OTM across entry times")
    a = say(sid, "Buy SPY 0DTE calls after a 0.5% down day, sell after 60 minutes, from 2024-01-18 to 2026-11-25")
    rep = say(sid, "Compare ATM and OTM contracts across multiple entry times.")
    R["atm_otm"] = {"notebook": sid, "run": (rep["replies"][-1]["content"].get("run_id") if rep.get("replies") else None)}
    title_notebook(sid, "ATM versus OTM across entry times")
    res["items"]["D_atm_versus_otm"] = {"base": brief(a), "comparison": brief(rep)}; print("D", R["atm_otm"], flush=True); keep()

if "E" in which:  # several expirations: the stored universe reaches 7 days, and the grid says what it could not price
    sid = notebook("V7 acceptance: expirations from 0 to 30 days")
    a = say(sid, "Buy ATM SPY calls at the open after a 1% down day, hold for 1 trading day, from 2024-01-18 to 2026-11-25")
    rep = say(sid, "Test 0DTE, 1DTE, 3DTE, 7DTE, and 30DTE contracts.")
    R["dte"] = {"notebook": sid, "run": (rep["replies"][-1]["content"].get("run_id") if rep.get("replies") else None)}
    title_notebook(sid, "expirations from 0 to 30 days")
    res["items"]["E_multiple_expirations"] = {"base": brief(a), "comparison": brief(rep)}; print("E", R["dte"], flush=True); keep()

if "F" in which:  # a stateful strategy: an ordered rule, a partial exit, a premium stop, a clock exit
    sid = notebook("V7 acceptance: VWAP reclaim after a decline, with a partial exit")
    rep = say(sid, "Buy two ATM QQQ 0DTE calls when price reclaims VWAP after declining 0.5% from the open, sell half when the premium is up 25%, with a 15% premium stop, exit at 3:30 PM Eastern, from 2024-01-18 to 2026-11-25")
    rid = run_of(rep)
    R["stateful"] = {"notebook": sid, "run": rid}
    card = call("GET", f"/api/backtests/{rid}/performance?sid={sid}") if rid else None
    title_notebook(sid, "VWAP reclaim after a decline, partial exit, premium stop")
    res["items"]["F_complex_stateful"] = {**brief(rep), "run": rid, "card": None if not card or "_error" in card else {s["id"]: s["text"] for s in card["sections"]}}; print("F", rid, flush=True); keep()

if "G" in which:  # a replay: the stored bars of the underlying and of the contract, the trade over them
    rid = R["stateful"]["run"] or R["full_SPY"]["run"]
    t0 = time.time(); tr = call("GET", f"/api/backtests/{rid}/trades/1/observed"); dt = time.time() - t0
    marks = [x for x in tr["marks"]["pnl"] if x is not None]
    res["items"]["G_replay"] = {"run": rid, "trade": tr["trade"], "seconds_to_load": round(dt, 3), "underlying": {"stored": tr["underlying"]["stored"], "candles": len(tr["underlying"]["candles"]), "note": tr["underlying"]["note"], "levels": {k: tr["underlying"]["sessions"][0].get(k) for k in ("previous", "opening_range")}},
                               "legs": [{"contract": l["occ"], "stored_bars": len(l["bars"]), "sessions": l["sessions"], "entry_price": l["entry_price"], "exit_price": l["exit_price"]} for l in tr["legs"]],
                               "markers": tr["markers"], "exit_levels": tr["levels"], "marks": {"minutes_marked": len(marks), "first": marks[:3], "last": marks[-1] if marks else None, "oldest_print_seconds_max": max([x for x in tr["marks"]["oldest_print_seconds"] if x is not None] or [None])},
                               "fills_check": {k: v for k, v in call("GET", f"/api/backtests/{rid}/fills").items() if k in ("checked_against_stored_bars", "matching")}}
    print("G", res["items"]["G_replay"]["marks"], res["items"]["G_replay"]["fills_check"], flush=True); keep()

if "H" in which:  # the same decisions on individual prints, both exit policies
    out = {}
    for name in ("full_SPY", "stateful"):
        rid = R[name]["run"]
        for pol in ("decision_clock", "resting_orders"):
            j = call("POST", f"/api/backtests/{rid}/hires", {"exit_policy": pol})
            t0 = time.time()
            while time.time() - t0 < 1500:
                st = call("GET", f"/api/runs/{j['id']}")
                if st.get("status") in ("done", "error", "cancelled"):
                    break
                time.sleep(2)
            rep = (call("GET", f"/api/backtests/{rid}/hires").get("reports") or {}).get(pol) or {}
            sm = rep.get("summary") or {}
            out[f"{name}:{pol}"] = {"run": rid, "job": st.get("status"), "seconds": round(time.time() - t0, 1), "requests": rep.get("requests"), "text": sm.get("text"),
                                    **{k: sm.get(k) for k in ("positions", "repriced", "entries_not_filled", "bar_pnl_repriced", "tick_pnl_repriced", "difference", "exit_reason_changed", "entry_seconds_after_decision", "entry_price_worse_pct", "exit_price_worse_pct")}}
            print("H", name, pol, out[f"{name}:{pol}"]["bar_pnl_repriced"], out[f"{name}:{pol}"]["tick_pnl_repriced"], flush=True)
            res["items"]["H_minute_bars_against_individual_prints"] = out; keep()

if "I" in which:  # a matched historical comparison (the research graph)
    sid = notebook("V7 acceptance: recovery after down days, controlling for volatility")
    rep = say(sid, "Does SPY tend to recover after down days of 1.5% or more, even after controlling for volatility?")
    title_notebook(sid, "recovery after down days, controlling for volatility")
    res["items"]["I_matched_comparison"] = brief(rep); print("I", [x["kind"] for x in res["items"]["I_matched_comparison"]["said"]], flush=True); keep()

if "J" in which:  # out of sample: the later part of the history against the earlier, and the search counted
    sid = R["full_SPY"]["notebook"]
    a = say(sid, "Test whether the results hold up out of sample")
    b = say(sid, "Is this statistically significant?")
    res["items"]["J_out_of_sample"] = {"later_sessions": brief(a), "significance": brief(b)}; print("J done", flush=True); keep()

if "K" in which:  # a prospective paper experiment: frozen now, judged only on sessions that do not exist yet
    rid = R["full_SPY"]["run"]
    ex = call("POST", "/api/forward", {"run_id": rid, "name": "V7 acceptance: SPY 0DTE calls after a 1% down day"})
    ev = call("POST", f"/api/forward/{ex.get('id')}/evaluate", {}) if ex.get("id") else None
    acct = call("GET", f"/api/forward/{ex.get('id')}/account") if ex.get("id") else None
    full = call("GET", f"/api/forward/{ex.get('id')}") if ex.get("id") else {}
    res["items"]["K_prospective_paper_experiment"] = {"experiment": ex.get("id"), "frozen": {k: full.get(k) for k in ("name", "status", "activation_date", "first_session", "config_hash", "engine_version", "version")}, "activation": full.get("activation"),
                                                      "evaluation": None if not ev else {k: ev.get(k) for k in ("status", "forward_sessions", "first", "last", "notes", "method", "account")}, "stored_states": len((acct or {}).get("states") or []), "ledger": full.get("ledger"), "error": ex if "_error" in ex else None}
    print("K", res["items"]["K_prospective_paper_experiment"]["frozen"], flush=True); keep()

if "L" in which:  # historical quotes: blocked, and it says what is needed
    d = call("GET", "/api/data/access/databento")
    res["items"]["L_quote_based_comparison"] = {"status": "BLOCKED" if not d.get("key_stored") else d.get("status"), "reason": d.get("reason"), "needs": d.get("needs"), "how": d.get("how"), "unverified": d.get("unverified")}
    print("L", res["items"]["L_quote_based_comparison"]["status"], flush=True); keep()

if "M" in which:  # typed into the application by its user on 2026-11-26, and answered then with a different, unconditional statistic
    sid = notebook("V7 acceptance: the first candle and the day")
    rep = say(sid, "How often does SPY's first 9:30ET candle, predict its overall day?")
    msgs = rep.get("replies") or []
    plan = next((m["content"].get("plan") for m in msgs if m["kind"] == "investigation"), None) or {}
    out = next((m["content"] for m in reversed(msgs) if m["kind"] == "result"), {})
    r_ = out.get("report") or {}
    res["items"]["M_question_typed_by_the_user"] = {"sentence": rep["text"], "seconds": rep.get("seconds"), "notebook": sid, "read_as": {k: plan.get(k) for k in ("hypothesis", "method", "interpretation_notes")}, "variants": [v["label"] for v in plan.get("variants") or []],
                                                    "level": r_.get("level"), "verdict": r_.get("verdict"), "findings": r_.get("findings"), "limitations": r_.get("limitations"),
                                                    "agreement": [{k: t.get(k) for k in ("minutes", "scope", "from", "sessions", "agree", "agreement", "ci95", "if_unrelated", "distinguishable", "first", "last")} for t in r_.get("agreement") or []]}
    title_notebook(sid, "the first candle and the day (a question typed by hand on 2026-11-26)")
    print("M", r_.get("level"), (r_.get("verdict") or "")[:200], flush=True); keep()

if "C" in which:  # the bounded sweep of the brief: 252 configurations from one sentence
    sid = notebook("V7 acceptance: 252 configurations from one sentence")
    a = say(sid, "Buy ATM SPY calls at 10:00 AM Eastern after a 0.5% down day, from 2024-01-18 to 2026-11-25")
    rep = say(sid, "Compare entry times from 9:30 to 11:00 in 15-minute increments, prior-day declines from 0.5% to 3%, 0DTE versus 1DTE, and several profit targets.", wait=3600)
    rid = rep["replies"][-1]["content"].get("run_id") if rep.get("replies") else None
    R["sweep"] = {"notebook": sid, "run": rid}
    exp = call("GET", f"/api/experiments/{rid}") if rid else {}
    res["items"]["C_bounded_sweep"] = {"base": brief(a), "experiment": brief(rep), "run": rid, "summary": exp.get("summary"), "stability": (exp.get("stability") or {}).get("text"), "notes": exp.get("notes")}
    title_notebook(sid, "252 configurations from one sentence")
    print("C", rid, (exp.get("summary") or {}).get("text", "")[:300], flush=True); keep()
keep()
print("written", OUT)
