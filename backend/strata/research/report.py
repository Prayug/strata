"""The research report: one persistent document per result.

`build_report` assembles everything a reader needs to judge a result - the question, how
it was interpreted and what was assumed, which data answered it and what that data could
not cover, the sample, every statistic with the observations behind it, robustness and
out-of-sample checks, the limitations, and the configuration that reproduces it. The same
structure feeds the interactive view and the exports (HTML, Markdown, JSON).

All numbers are read from stored results or computed by `research.stats`; this module
formats, it does not estimate.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from strata.ai import explain as EX
from strata.api.results import load_trades
from strata.research import stats as ST
from strata.research.views import observations


def build_report(run: dict, res: dict, context: dict | None = None, trials: int = 1, store=None) -> dict:
    """context: {question, title, items, assumptions, unsupported, notes, interpreter, exec_mode, version_id, session_id}"""
    ctx = context or {}
    study = run["kind"] == "study"
    facts = EX.study_facts(res) if study else EX.backtest_facts(res)
    caveats = EX.study_caveats(res) if study else EX.backtest_caveats(res)
    summary = EX.study_text(res, facts) if study else EX.backtest_text(res, facts)
    cov = res.get("coverage") or {}
    meta = res.get("meta") or {}
    trades = [] if study else load_trades(run["id"])
    rob = ST.study_robustness(res, trials) if study else ST.backtest_robustness(res, trades, trials)
    sources = []
    if study:
        for sym, info in ((cov.get("underlying") or {}).get("by_symbol") or {}).items():
            sources.append({"what": f"{sym} prices ({info['role']})", "detail": ", ".join(f"{n} sessions {k}" for k, n in (info.get("resolution_mix") or {}).items()) or "none stored", "limitations": []})
    else:
        for sym, used in (meta.get("datasets_used") or {}).items():
            sources.append({"what": f"{sym} option {used['observation']}", "detail": f"{used['dataset']} ({used['resolution']}, {used.get('license') or 'licence not recorded'})", "limitations": used.get("limitations") or []})
        for sym, r in (meta.get("data_resolution") or {}).items():
            sources.append({"what": f"{sym} prices", "detail": ", ".join(f"{n} sessions {k.replace(':', ' ')}" for k, n in (r.get("underlying_sessions_by_provider") or {}).items()) or "n/a", "limitations": []})
        if meta.get("pricing") == "modeled":
            sources.append({"what": "Option premiums", "detail": "THEORETICAL (pricing model) - not observed prices", "limitations": meta.get("pricing_assumptions") or []})
    obs = observations(run, res, "all", limit=20000)
    out = {
        "title": ctx.get("title") or run.get("name") or "Research report", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": "study" if study else "backtest",
        "question": ctx.get("question") or run.get("name") or "", "mode": cov.get("mode") or ("study" if study else meta.get("pricing")), "mode_label": cov.get("mode_label") or ("Historical market study" if study else "Options backtest"),
        "mode_reason": cov.get("mode_reason"), "symbol": res.get("symbol") if study else ", ".join(sorted({s["symbol"] for s in res["config"]["strategies"]})),
        "period": res["period"] if study else [res["config"]["start"], res["config"]["end"]],
        "interpretation": {"items": ctx.get("items") or [], "assumptions": ctx.get("assumptions") or [], "unsupported": ctx.get("unsupported") or [], "notes": ctx.get("notes") or [], "interpreter": ctx.get("interpreter")},
        "data": {"sources": sources, "requested": cov.get("requested"), "usable": (cov.get("underlying") or {}).get("usable"), "excluded": (cov.get("underlying") or {}).get("excluded") or [],
                 "signals": {k: v for k, v in (cov.get("signals") or {}).items() if k != "dates"}, "executable": cov.get("executable"), "options": cov.get("options"),
                 "downloads": cov.get("downloads") or [], "unavailable": cov.get("unavailable") or [], "awaiting_approval": cov.get("awaiting_approval") or [], "by_symbol": (cov.get("underlying") or {}).get("by_symbol") or {}},
        "summary": summary, "facts": facts, "limitations": caveats, "robustness": rob,
        "observations": {"title": obs["title"], "columns": obs["columns"], "rows": obs["rows"], "count": obs["count"]},
        "reproduce": {"run_id": run["id"], "created": run.get("created_at"), "engine_version": run.get("engine_version"), "config_hash": run.get("config_hash"), "dataset_versions": run.get("dataset_versions") or meta.get("dataset_versions"),
                      "config": res.get("config"), "pipeline": [{k: s.get(k) for k in ("label", "status", "detail")} for s in (res.get("pipeline") or [])]},
        "trials": trials,
    }
    if study:
        st = res["stats"]
        out["sample"] = {"qualifying": st.get("qualifying_sessions"), "evaluated": st.get("evaluated_sessions"), "excluded": st.get("excluded_sessions"), "resolution_mix": st.get("resolution_mix")}
        out["outcomes"] = {k: st.get(k) for k in ("target_first", "stop_first", "neither", "ambiguous")}
        out["baseline"] = {"probability": st.get("baseline_probability"), "ci95": st.get("baseline_ci95"), "sessions": st.get("baseline_sessions"), "lift": st.get("lift"), "p_value": st.get("p_value_vs_rest")}
        out["by_year"] = res.get("by_year")
        out["regimes"] = res.get("regimes")
        out["charts"] = {"cumulative": res.get("cumulative"), "time_of_day": res.get("time_of_day"), "end_hist": res.get("end_hist"), "mfe_hist": res.get("mfe_hist"), "mae_hist": res.get("mae_hist"), "sensitivity": res.get("sensitivity")}
        out["excluded"] = res.get("excluded") or []
    else:
        m = res.get("metrics") or {}
        ser = res.get("series") or {}
        cap = (res["config"].get("capital") or {}).get("initial_capital", 100000)
        out["sample"] = {"trades": m.get("trades"), "sessions": m.get("sessions"), "qualifying": (cov.get("signals") or {}).get("qualifying"), "executable": (cov.get("executable") or {}).get("signals")}
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.get("exit_reason") or "?"] = reasons.get(t.get("exit_reason") or "?", 0) + 1
        out["outcomes"] = reasons
        out["metrics"] = m
        bd = res.get("breakdowns") or {}
        out["by_year"] = bd.get("year")
        out["regimes"] = {"vix": bd.get("vix"), "trend": bd.get("regime"), "weekday": bd.get("weekday"), "time_of_day": bd.get("time_of_day")}
        out["charts"] = {"equity": [[d, (v - cap) if v is not None else None] for d, v in zip(ser.get("dates") or [], ser.get("equity") or [])], "drawdown": [[d, v] for d, v in zip(ser.get("dates") or [], ser.get("drawdown") or [])],
                         "distribution": (res.get("distributions") or {}).get("dollars")}
        out["cost_sensitivity"] = res.get("cost_sensitivity")
        out["executability"] = meta.get("executability")
        out["skipped"] = cov.get("skipped_signals")
        out["execution"] = res["config"].get("execution")
        out["excluded"] = [{"reason": k, "count": len(v), "dates": v[-12:]} for k, v in ((res.get("diagnostics") or {}).get("excluded_sessions") or {}).items()]
    return out


# ----------------------------------------------------------------------------- exports
def _f(v, unit=None) -> str:
    if v is None:
        return "n/a"
    if unit == "prob":
        return f"{v * 100:.1f}%"
    if unit == "usd":
        return f"{v:+,.2f}"
    if isinstance(v, float):
        return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.2f}"
    return str(v)


def render_markdown(r: dict) -> str:
    L: list[str] = [f"# {r['title']}", "", f"*{r['mode_label']} - {r['symbol']} - {r['period'][0]} to {r['period'][1]} - generated {r['generated']}*", ""]
    L += ["## Research question", "", r["question"], ""]
    L += ["## Summary", "", r["summary"], ""]
    if r.get("mode_reason"):
        L += [f"**How it was run:** {r['mode_reason']}", ""]
    L += ["## Interpretation", ""]
    for it in r["interpretation"]["items"]:
        L.append(f"- **{it['label']}** ({it['status']}): {it['value']}")
    if r["interpretation"]["assumptions"]:
        L += ["", "### Assumptions", ""] + [f"- {a['field']}: {a['value']} - {a['reason']}" for a in r["interpretation"]["assumptions"]]
    if r["interpretation"]["unsupported"]:
        L += ["", "### Parts of the request that were not represented", ""] + [f"- \"{u}\"" for u in r["interpretation"]["unsupported"]]
    for n in r["interpretation"]["notes"]:
        L.append(f"- Note: {n}")
    d = r["data"]
    L += ["", "## Data sources and coverage", ""]
    for s in d["sources"]:
        L.append(f"- **{s['what']}**: {s['detail']}")
        L += [f"  - Limitation: {x}" for x in s["limitations"][:4]]
    if d.get("requested"):
        L.append(f"- Requested: {d['requested']['start']} to {d['requested']['end']} ({d['requested']['sessions']:,} sessions{', as stated' if d['requested'].get('stated') else ', all available history (no period was stated)'})")
    if d.get("usable"):
        L.append(f"- Sessions with the prices the rules need: {d['usable']['sessions']:,}")
    if d.get("signals"):
        L.append(f"- Sessions on which the entry rule fired: {d['signals'].get('qualifying')} of {d['signals'].get('evaluated_sessions')} evaluated ({d['signals'].get('not_evaluable_sessions')} could not be evaluated)")
    if d.get("executable") and r["kind"] == "backtest":
        L.append(f"- Of those, sessions with usable observed option prices: {d['executable']['signals']} of {d['executable']['of']}")
    for x in (d.get("options") or {}).get("issues", [])[:5] if d.get("options") else []:
        L.append(f"  - Not executable: {x['reason']} ({x['signals']})")
    for x in d["downloads"]:
        L.append(f"- Downloaded: {x.get('label')} {x.get('dataset')} for {', '.join(x.get('symbols') or [])} - {x.get('why')}" + (f" (failed: {x['error']})" if x.get("error") else ""))
    for x in d["unavailable"]:
        L.append(f"- Not available: {x.get('detail')}")
    for x in d["awaiting_approval"]:
        L.append(f"- Could be expanded with a larger download (not run without approval): {x.get('label')} - {x.get('why')}")
    L += ["", "## Results", "", "| Statistic | Value |", "|---|---|"] + [f"| {f['label']} | {f['shown']} |" for f in r["facts"]]
    rb = r["robustness"]
    L += ["", "## Robustness", "", f"**{rb['verdict']}**", ""] + [f"- {'PASS' if c['ok'] else 'NOT PASSED'} - {c['name']}: {c['detail']}" for c in rb["checks"]]
    if r["trials"] > 1:
        L.append(f"- {r['trials']} configurations were tested in this research session; significance is adjusted for that.")
    if r.get("by_year"):
        L += ["", "## By year", ""]
        if r["kind"] == "study":
            L += ["| Year | Sessions | Outcome | Frequency | Avg end return |", "|---|---|---|---|---|"] + [f"| {y['year']} | {y['n']} | {y['target']} | {_f(y['probability'], 'prob')} | {y['avg_end_return_pct']:+.2f}% |" for y in r["by_year"]]
        else:
            L += ["| Year | Trades | Net P&L | Win rate |", "|---|---|---|---|"] + [f"| {y['key']} | {y['trades']} | {y['pnl']:+,.2f} | {_f(y['win_rate'], 'prob')} |" for y in r["by_year"]]
    if r.get("cost_sensitivity") and r["cost_sensitivity"].get("rows"):
        cs = r["cost_sensitivity"]
        L += ["", "## Transaction-cost sensitivity", "", cs["method"], "", "| Every fill worse by | Net P&L | Win rate |", "|---|---|---|"]
        L += [f"| {x['extra_slippage_pct']}% | {x['net_pnl']:+,.2f} | {_f(x['win_rate'], 'prob')} |" for x in cs["rows"]]
        if cs.get("breakeven_slippage_pct") is not None:
            L.append(f"\nThe net result reaches zero when every fill is about {cs['breakeven_slippage_pct']:.1f}% worse.")
    if r.get("skipped") and r["skipped"].get("count"):
        L += ["", "## Signals that did not become trades", "", f"{r['skipped']['count']} in total: " + ", ".join(f"{k} ({v})" for k, v in (r["skipped"].get("by_kind") or {}).items()) + ".", "", "| Date | Reason |", "|---|---|"]
        L += [f"| {x['date']} | {x['reason']} |" for x in r["skipped"]["rows"][-60:]]
    L += ["", "## Limitations", ""] + [f"- {c}" for c in r["limitations"]]
    o = r["observations"]
    L += ["", f"## {o['title']} ({o['count']})", "", "| " + " | ".join(o["columns"]) + " |", "|" + "---|" * len(o["columns"])]
    L += ["| " + " | ".join(_cell(row.get(c)) for c in o["columns"]) + " |" for row in o["rows"][-400:]]
    rp = r["reproduce"]
    L += ["", "## Reproduce", "", f"- Run `{rp['run_id']}`, engine {rp['engine_version']}, configuration hash `{rp['config_hash']}`", f"- Dataset versions: `{json.dumps(rp['dataset_versions'])}`", "", "```json", json.dumps(rp["config"], indent=1, default=str), "```", ""]
    return "\n".join(L)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("|", "/")


def _svg_line(points: list, w=820, h=200, color="#0ea5a4", zero=True) -> str:
    pts = [(i, p[1]) for i, p in enumerate(points) if p[1] is not None]
    if len(pts) < 2:
        return ""
    ys = [y for _, y in pts]
    lo, hi = min(ys + ([0] if zero else [])), max(ys + ([0] if zero else []))
    if hi == lo:
        hi = lo + 1
    sx = lambda i: 46 + i / (len(points) - 1) * (w - 56)
    sy = lambda y: 10 + (hi - y) / (hi - lo) * (h - 34)
    path = " ".join(f"{'M' if k == 0 else 'L'}{sx(i):.1f},{sy(y):.1f}" for k, (i, y) in enumerate(pts))
    axis = f'<line x1="46" x2="{w - 10}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#999" stroke-dasharray="3 3"/>' if zero else ""
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img"><text x="4" y="{sy(hi) + 4:.0f}" font-size="10" fill="#666">{hi:,.0f}</text><text x="4" y="{sy(lo) + 4:.0f}" font-size="10" fill="#666">{lo:,.0f}</text>'
            f'{axis}<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"/><text x="46" y="{h - 4}" font-size="10" fill="#666">{points[0][0]}</text>'
            f'<text x="{w - 10}" y="{h - 4}" font-size="10" fill="#666" text-anchor="end">{points[-1][0]}</text></svg>')


def _svg_bars(rows: list[tuple[str, float | None]], w=820, h=180, fmt=lambda v: f"{v:,.0f}") -> str:
    rows = [(k, v) for k, v in rows if v is not None]
    if not rows:
        return ""
    lo, hi = min(0.0, min(v for _, v in rows)), max(0.0, max(v for _, v in rows))
    if hi == lo:
        hi = lo + 1
    bw = (w - 60) / len(rows)
    sy = lambda y: 12 + (hi - y) / (hi - lo) * (h - 40)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img"><line x1="40" x2="{w - 10}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#999"/>']
    for i, (k, v) in enumerate(rows):
        x = 44 + i * bw
        y0, y1 = sy(max(v, 0)), sy(min(v, 0))
        out.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{max(bw - 4, 2):.1f}" height="{max(y1 - y0, 1):.1f}" fill="{"#0ea5a4" if v >= 0 else "#e11d48"}"><title>{html.escape(str(k))}: {fmt(v)}</title></rect>')
        if len(rows) <= 34:
            out.append(f'<text x="{x + bw / 2 - 2:.1f}" y="{h - 6}" font-size="9" fill="#666" text-anchor="middle">{html.escape(str(k))[-4:]}</text>')
    return "".join(out) + "</svg>"


def render_html(r: dict) -> str:
    e = html.escape
    H: list[str] = []
    add = H.append
    add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{e(r['title'])}</title><style>"
        "body{font:14px/1.5 -apple-system,Segoe UI,Inter,sans-serif;color:#16181d;max-width:980px;margin:32px auto;padding:0 20px}h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px;padding-top:14px;border-top:1px solid #e3e5ea}"
        "h3{font-size:13px;margin:16px 0 6px}.meta{color:#5b6270;font-size:12.5px}table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #eceef2;vertical-align:top}"
        "th{color:#5b6270;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;background:#eef1f5;color:#444;margin-right:6px}"
        ".tag.assumed{background:#fff3d6;color:#7a5200}.tag.unsupported{background:#fde2e4;color:#8a1c2b}.ok{color:#0a7d6b;font-weight:600}.no{color:#b3261e;font-weight:600}.box{background:#f6f7f9;border-radius:8px;padding:12px 14px;margin:8px 0}"
        ".warn{background:#fff7e6;border-left:3px solid #e0a100}pre{background:#f6f7f9;padding:12px;border-radius:8px;overflow:auto;font-size:11.5px}ul{padding-left:20px;margin:6px 0}.scroll{max-height:520px;overflow:auto}"
        "@media print{.scroll{max-height:none}}</style></head><body>")
    add(f"<h1>{e(r['title'])}</h1><div class='meta'>{e(r['mode_label'])} &middot; {e(r['symbol'])} &middot; {r['period'][0]} to {r['period'][1]} &middot; generated {r['generated']} &middot; Strata engine {e(str(r['reproduce']['engine_version']))}</div>")
    add(f"<h2>Research question</h2><p>{e(r['question'])}</p>")
    add(f"<h2>Summary</h2><div class='box'>{e(r['summary'])}</div>")
    if r.get("mode_reason"):
        add(f"<p class='meta'>{e(r['mode_reason'])}</p>")
    add("<h2>Interpretation</h2><table>" + "".join(f"<tr><th style='width:150px'>{e(it['label'])}</th><td><span class='tag {e(it['status'])}'>{e(it['status'])}</span>{e(str(it['value']))}</td></tr>" for it in r["interpretation"]["items"]) + "</table>")
    if r["interpretation"]["assumptions"]:
        add("<h3>Assumptions made</h3><ul>" + "".join(f"<li><b>{e(str(a['field']))}</b>: {e(str(a['value']))} &mdash; {e(a['reason'])}</li>" for a in r["interpretation"]["assumptions"]) + "</ul>")
    if r["interpretation"]["unsupported"]:
        add("<div class='box warn'><b>Not represented:</b> " + "; ".join(f"&ldquo;{e(u)}&rdquo;" for u in r["interpretation"]["unsupported"]) + ". Nothing was substituted for these parts.</div>")
    if r["interpretation"]["notes"]:
        add("<ul>" + "".join(f"<li>{e(n)}</li>" for n in r["interpretation"]["notes"]) + "</ul>")
    d = r["data"]
    add("<h2>Data sources and coverage</h2><table><tr><th>Used for</th><th>Dataset</th></tr>" + "".join(
        f"<tr><td>{e(s['what'])}</td><td>{e(s['detail'])}" + "".join(f"<div class='meta'>Limitation: {e(x)}</div>" for x in s["limitations"][:4]) + "</td></tr>" for s in d["sources"]) + "</table><ul>")
    if d.get("requested"):
        add(f"<li>Requested period: {d['requested']['start']} to {d['requested']['end']} ({d['requested']['sessions']:,} sessions; {'stated in the request' if d['requested'].get('stated') else 'all available history - no period was stated'})</li>")
    if d.get("usable"):
        add(f"<li>Sessions with the prices the rules need: <b>{d['usable']['sessions']:,}</b></li>")
    for x in d["excluded"][:6]:
        add(f"<li>Excluded: {e(x['reason'])} ({x['sessions']:,} sessions)</li>")
    if d.get("signals"):
        add(f"<li>Sessions on which the rule fired: <b>{d['signals'].get('qualifying')}</b> of {d['signals'].get('evaluated_sessions'):,} evaluated; {d['signals'].get('not_evaluable_sessions')} could not be evaluated (kept apart from sessions without a signal)</li>")
    if d.get("executable") and r["kind"] == "backtest":
        add(f"<li>Sessions with usable observed option prices: <b>{d['executable']['signals']}</b> of {d['executable']['of']}</li>")
    if d.get("options"):
        for x in d["options"].get("issues", [])[:6]:
            add(f"<li>Not executable: {e(x['reason'])} ({x['signals']})</li>")
    for x in d["downloads"]:
        add(f"<li>Downloaded automatically (free): {e(str(x.get('label')))} {e(str(x.get('dataset')))} &mdash; {e(str(x.get('why')))}" + (f" <span class='no'>failed: {e(x['error'])}</span>" if x.get("error") else "") + "</li>")
    for x in d["unavailable"]:
        add(f"<li>Not available: {e(str(x.get('detail')))}</li>")
    for x in d["awaiting_approval"]:
        add(f"<li>Coverage could be expanded with a larger download (not run without approval): {e(str(x.get('label')))} &mdash; {e(str(x.get('why')))}</li>")
    add("</ul>")
    add("<h2>Results</h2><table><tr><th>Statistic</th><th>Value</th></tr>" + "".join(f"<tr><td>{e(f['label'])}</td><td class='n'>{e(f['shown'])}</td></tr>" for f in r["facts"]) + "</table>")
    ch = r.get("charts") or {}
    if ch.get("equity"):
        add("<h3>Net profit over time ($)</h3>" + _svg_line(ch["equity"]))
        add("<h3>Drawdown</h3>" + _svg_line([[a, (b or 0) * 100] for a, b in ch["drawdown"]], h=140, color="#e11d48"))
    if r.get("by_year"):
        if r["kind"] == "study":
            add("<h3>Frequency of the outcome by year (%)</h3>" + _svg_bars([(y["year"], (y["probability"] or 0) * 100 if y["probability"] is not None else None) for y in r["by_year"]], fmt=lambda v: f"{v:.0f}%"))
            add("<table><tr><th>Year</th><th>Sessions</th><th>Outcome</th><th>Frequency</th><th>Avg end return</th></tr>" + "".join(
                f"<tr><td>{y['year']}</td><td class='n'>{y['n']}</td><td class='n'>{y['target']}</td><td class='n'>{_f(y['probability'], 'prob')}</td><td class='n'>{y['avg_end_return_pct']:+.2f}%</td></tr>" for y in r["by_year"]) + "</table>")
        else:
            add("<h3>Net P&amp;L by year ($)</h3>" + _svg_bars([(y["key"], y["pnl"]) for y in r["by_year"]]))
            add("<table><tr><th>Year</th><th>Trades</th><th>Net P&amp;L</th><th>Win rate</th></tr>" + "".join(
                f"<tr><td>{y['key']}</td><td class='n'>{y['trades']}</td><td class='n'>{y['pnl']:+,.2f}</td><td class='n'>{_f(y['win_rate'], 'prob')}</td></tr>" for y in r["by_year"]) + "</table>")
    for name, rows in ((r.get("regimes") or {}).items() if r.get("regimes") else []):
        if not rows:
            continue
        add(f"<h3>By {e(name.replace('_', ' '))}</h3><table>")
        if r["kind"] == "study":
            add("<tr><th>Group</th><th>Sessions</th><th>Frequency</th><th>Avg end return</th></tr>" + "".join(f"<tr><td>{e(str(x['key']))}</td><td class='n'>{x['n']}</td><td class='n'>{_f(x['probability'], 'prob')}</td><td class='n'>{x['avg_end_return_pct']:+.2f}%</td></tr>" for x in rows))
        else:
            add("<tr><th>Group</th><th>Trades</th><th>Net P&amp;L</th><th>Win rate</th></tr>" + "".join(f"<tr><td>{e(str(x['key']))}</td><td class='n'>{x['trades']}</td><td class='n'>{x['pnl']:+,.2f}</td><td class='n'>{_f(x['win_rate'], 'prob')}</td></tr>" for x in rows))
        add("</table>")
    rb = r["robustness"]
    add(f"<h2>Robustness and out-of-sample</h2><div class='box'><b>{e(rb['verdict'])}</b></div><table>" + "".join(
        f"<tr><td style='width:90px' class='{'ok' if c['ok'] else 'no'}'>{'Passed' if c['ok'] else 'Not passed'}</td><td><b>{e(c['name'])}</b><div class='meta'>{e(c['detail'])}</div></td></tr>" for c in rb["checks"]) + "</table>")
    if r["trials"] > 1:
        add(f"<p class='meta'>{r['trials']} configurations were tested in this research session; the significance threshold is adjusted for that.</p>")
    cs = r.get("cost_sensitivity")
    if cs and cs.get("rows"):
        add(f"<h2>Transaction-cost sensitivity</h2><p class='meta'>{e(cs['method'])}</p><table><tr><th>Every fill worse by</th><th>Net P&amp;L</th><th>Win rate</th><th>Per trade</th></tr>" + "".join(
            f"<tr><td>{x['extra_slippage_pct']}%</td><td class='n'>{x['net_pnl']:+,.2f}</td><td class='n'>{_f(x['win_rate'], 'prob')}</td><td class='n'>{_f(x['avg_trade'], 'usd')}</td></tr>" for x in cs["rows"]) + "</table>")
        if cs.get("breakeven_slippage_pct") is not None:
            add(f"<p>The net result reaches zero when every fill is about {cs['breakeven_slippage_pct']:.1f}% worse.</p>")
    sk = r.get("skipped")
    if sk and sk.get("count"):
        add(f"<h2>Signals that did not become trades ({sk['count']})</h2><p class='meta'>These are data and execution limits. None is counted as a win or a loss.</p><div class='scroll'><table><tr><th>Date</th><th>Kind</th><th>Reason</th></tr>" + "".join(
            f"<tr><td>{x['date']}</td><td>{e(str(x.get('kind')))}</td><td>{e(str(x.get('reason')))}</td></tr>" for x in sk["rows"]) + "</table></div>")
    if r.get("excluded"):
        add("<h3>Sessions that could not be evaluated</h3><ul>" + "".join(f"<li>{e(x['reason'])} ({x['count']})</li>" for x in r["excluded"][:10]) + "</ul>")
    add("<h2>Limitations</h2><ul>" + "".join(f"<li>{e(c)}</li>" for c in r["limitations"]) + "</ul>")
    o = r["observations"]
    add(f"<h2>{e(o['title'])} ({o['count']})</h2><div class='scroll'><table><tr>" + "".join(f"<th>{e(c.replace('_', ' '))}</th>" for c in o["columns"]) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td{' class=n' if isinstance(row.get(c), (int, float)) else ''}>{e(_cell(row.get(c)))}</td>" for c in o["columns"]) + "</tr>" for row in o["rows"]) + "</table></div>")
    rp = r["reproduce"]
    add(f"<h2>Methodology and reproduction</h2><ul><li>Run <code>{e(rp['run_id'])}</code> created {e(str(rp['created']))}; engine {e(str(rp['engine_version']))}; configuration hash <code>{e(str(rp['config_hash']))}</code></li>"
        f"<li>Dataset versions: <code>{e(json.dumps(rp['dataset_versions']))}</code></li><li>Every number above was calculated by the deterministic engine from stored market data. No language model produced a result.</li></ul>")
    if rp["pipeline"]:
        add("<table><tr><th>Stage</th><th>Outcome</th></tr>" + "".join(f"<tr><td>{e(str(s['label']))}</td><td>{e(str(s['status']))}: {e(str(s['detail']))}</td></tr>" for s in rp["pipeline"]) + "</table>")
    add(f"<h3>Configuration</h3><pre>{e(json.dumps(rp['config'], indent=1, default=str))}</pre></body></html>")
    return "".join(H)


def csv_text(columns: list[str], rows: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow(["" if r.get(c) is None else r.get(c) for c in columns])
    return buf.getvalue()
