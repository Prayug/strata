/* One answer, at three depths (V8 brief section 6):
     Answer     the conclusion, a few numbers, one picture, the limits, where to go next. A beginner reads it in 15 seconds.
     Explore    the observations themselves: days, trades, classifications, filters, replays.
     Technical  intervals, tests, adjustments, provenance, configuration: the full report the engine produced.
   The level is remembered. Nothing is removed between levels; it is moved. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { Loader2 } from "lucide-react";
import type { AiMessage } from "@/lib/lab";
import { Report } from "@/pages/lab/Report";
import { InvestigationView, SweepView } from "@/pages/lab/Views";
import { type Level, fmtDate, useLevel, useObservations } from "./data";
import { PayoffViz, Viz } from "./Viz";

type Open = (spec: any) => void;

function Figure({ n, onOpen }: { n: any; onOpen: Open }) {
  const body = (<><span className="v">{n.value}</span><span className="l">{n.label}{n.detail ? <span className="faint"> · {n.detail}</span> : null}</span></>);
  return n.explore ? <button className="figure" onClick={() => onOpen(n.explore)} data-testid="figure">{body}</button> : <div className="figure">{body}</div>;
}

function ExploreLevel({ a, onOpen }: { a: any; onOpen: Open }) {
  const specs = (a.numbers || []).filter((n: any) => n.explore && n.explore.kind !== "replay");
  const [i, setI] = useState(0);
  const spec = specs[i]?.explore || null;
  const q = useObservations(spec && !["timing", "group", "dates"].includes(spec.kind) ? spec : null);
  if (!specs.length) return <div className="small mute">This answer has no list of observations behind it.</div>;
  return (
    <div data-testid="content-explore">
      <div className="seg mb-4 flex-wrap">{specs.map((n: any, k: number) => <button key={k} aria-pressed={k === i} onClick={() => setI(k)}>{n.label} <span className="num faint">{n.value}</span></button>)}</div>
      {q.isLoading && <div className="small mute flex items-center gap-2"><Loader2 className="size-4 animate-spin" />Reading the observations…</div>}
      {q.data && (
        <>
          <div className="small soft mb-2">{q.data.title}: <span className="num amber">{q.data.count.toLocaleString()}</span> {q.data.caption}</div>
          <div className="max-h-[420px] overflow-auto">
            <table className="obs"><thead><tr>{q.data.columns.slice(0, 7).map((c) => <th key={c.key} className={c.unit ? "r" : ""}>{c.label}</th>)}</tr></thead>
              <tbody>{q.data.rows.slice(0, 60).map((r: any, k: number) => <tr key={k}>{q.data!.columns.slice(0, 7).map((c) => <td key={c.key} className={c.unit ? "r num" : ""}>{c.key === "date" ? fmtDate(r.date) : r[c.key] == null ? "—" : typeof r[c.key] === "number" ? (c.unit === "%" ? `${r[c.key] > 0 ? "+" : ""}${r[c.key].toFixed(2)}%` : c.unit === "$" ? `${r[c.key] < 0 ? "−" : ""}$${Math.abs(r[c.key]).toFixed(2)}` : r[c.key].toLocaleString()) : String(r[c.key])}</td>)}</tr>)}</tbody>
            </table>
          </div>
        </>
      )}
      <button className="act mt-4" onClick={() => onOpen(spec)}>Open all {specs[i]?.value} with filters, selection and replay</button>
    </div>
  );
}

function TechnicalLevel({ m, a, sid }: { m: AiMessage; a: any; sid: string }) {
  const c = m.content || {};
  const [tab, setTab] = useState("overview");
  const [which, setWhich] = useState("all");
  const [day, setDay] = useState<string | null>(null);
  return (
    <div data-testid="content-technical">
      {(a?.details || []).length > 0 && (
        <dl className="mb-6 grid gap-x-8 gap-y-2 small" style={{ gridTemplateColumns: "minmax(160px, 260px) 1fr" }}>
          {a.details.map((d: any, i: number) => <div key={i} className="contents"><dt className="mute">{d.label}</dt><dd className="num soft">{d.value}</dd></div>)}
        </dl>
      )}
      {(a?.terms || []).length > 0 && (
        <div className="mb-6">
          <div className="kicker mb-2">The words used, and their technical names</div>
          <dl className="grid gap-x-8 gap-y-2 small" style={{ gridTemplateColumns: "minmax(160px, 260px) 1fr" }}>
            {a.terms.map((t: any, i: number) => <div key={i} className="contents"><dt><span className="soft">{t.plain}</span><br /><span className="faint tiny">{t.technical}</span></dt><dd className="soft">{t.meaning}</dd></div>)}
          </dl>
        </div>
      )}
      {c.view === "single" && c.run_id && (
        <div className="lab h-[720px] border" style={{ borderColor: "var(--rule)" }}>
          <Report runId={c.run_id} sid={sid} trials={1} tab={tab} setTab={setTab} which={which} setWhich={setWhich} onDay={setDay} day={day} onAsk={() => {}} onPin={() => {}} pinned={false} />
        </div>
      )}
      {c.view === "investigation" && <div className="lab"><InvestigationView c={c} onOpen={() => {}} onPin={() => {}} pinned={[]} /></div>}
      {(c.view === "sweep" || c.view === "compare") && c.table && <div className="lab"><SweepView c={c} onOpen={() => {}} onPin={() => {}} pinned={[]} /></div>}
      {c.view === "fills" && c.fills && (
        <div className="max-h-[520px] overflow-auto" data-testid="fills-table">
          <table className="obs"><thead><tr>{["Day", "Contract", "Side", "In at", "Price in", "Out at", "Price out", "Priced from"].map((h) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>{c.fills.rows.map((r: any, i: number) => <tr key={i}><td className="num">{fmtDate(r.date)}</td><td>{r.contract}</td><td>{r.side}</td><td className="num">{r.entry_time}</td><td className="num r">${Number(r.entry_price).toFixed(2)}</td><td className="num">{r.exit_time}</td><td className="num r">${Number(r.exit_price).toFixed(2)}</td><td className="mute">{r.source_text}</td></tr>)}</tbody>
          </table>
        </div>
      )}
      {!["single", "investigation", "sweep", "compare", "fills"].includes(c.view) && c.text && <p className="small soft whitespace-pre-line">{c.text}</p>}
      <div className="tiny mute mt-4">Everything here is also in the <Link className="link" to={`/lab/${sid}`}>full research workspace</Link>, with the method, the data used and the raw configuration.</div>
    </div>
  );
}

export function AnswerView({ m, sid, onOpen, onAsk, onTrade }: { m: AiMessage; sid: string; onOpen: Open; onAsk: (t: string) => void; onTrade: (runId: string, id: number) => void }) {
  const [level, setLevel] = useLevel();
  const c = m.content || {};
  const a = c.answer;
  if (!a) {
    return (
      <div data-testid="answer-plain">
        <p className="lede whitespace-pre-line">{c.text}</p>
        {c.view && c.view !== "failed" && <div className="mt-6"><TechnicalLevel m={m} a={null} sid={sid} /></div>}
      </div>
    );
  }
  const runId = a.viz?.run_id || c.run_id;
  const levels: [Level, string][] = [["answer", "Answer"], ["explore", "Explore"], ["technical", "Technical details"]];
  return (
    <article data-testid="answer" data-kind={a.kind}>
      <div className="seg mb-6" role="tablist" aria-label="Detail">{levels.map(([k, l]) => <button key={k} aria-pressed={level === k} onClick={() => setLevel(k)} data-testid={`level-${k}`}>{l}</button>)}</div>
      {level === "answer" && (
        <div data-testid="content-answer">
          <h2 className="headline" data-testid="headline">{a.headline}</h2>
          {(a.happened || []).length > 0 && <div className="mt-4 flex flex-col gap-2.5">{a.happened.map((h: string, i: number) => <p key={i} className="lede">{h}</p>)}</div>}
          {a.matters && <p className="lede mt-4"><span className="kicker mr-2" style={{ color: "var(--paper-2)" }}>Why it matters</span>{a.matters}</p>}
          {(a.numbers || []).length > 0 && (
            <div className="mt-8 grid gap-x-8 gap-y-5 rule pt-5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }} data-testid="figures">
              {a.numbers.slice(0, 5).map((n: any, i: number) => <Figure key={i} n={n} onOpen={onOpen} />)}
            </div>
          )}
          {a.viz && <div className="mt-8"><Viz a={a} onOpen={onOpen} onTrade={(id) => onTrade(runId, id)} /></div>}
          {a.illustration && <div className="mt-8"><PayoffViz ill={a.illustration} /></div>}
          {(a.story?.lines || []).length > 0 && <div className="mt-6 flex flex-col gap-2">{a.story.lines.map((l: string, i: number) => <p key={i} className="lede">{l}</p>)}</div>}
          {(a.trade || c.trade_id != null) && <button className="link small mt-3" onClick={() => onTrade(a.trade?.run_id || c.run_id, a.trade?.trade_id ?? c.trade_id)}>Replay this trade</button>}
          {a.example && (
            <div className="mt-8 rule pt-5">
              <div className="kicker mb-2">One actual trade</div>
              {(a.example.lines || []).map((l: string, i: number) => <p key={i} className="small soft mb-1.5">{l}</p>)}
              {a.example.trade != null && <button className="link small" onClick={() => onTrade(a.example.run_id || runId, a.example.trade)}>Replay this trade</button>}
            </div>
          )}
          {(a.limits || []).filter(Boolean).length > 0 && (
            <div className="mt-8 aside-rule" data-testid="limits">
              <div className="kicker mb-1.5">What this cannot tell you</div>
              {a.limits.filter(Boolean).map((l: string, i: number) => <p key={i} className="small soft mb-1">{l}</p>)}
            </div>
          )}
          {(a.followups || []).length > 0 && (
            <div className="mt-9" data-testid="followups">
              <div className="kicker mb-1">Where to go next</div>
              <div className="qlist">
                {a.followups.slice(0, 6).map((f: any, i: number) => (
                  <button key={i} className="qrow" onClick={() => onAsk(f.text)} data-testid="followup">
                    <span className="q" style={{ fontSize: 16.5 }}>{f.text}</span>
                    <span className="tiny faint text-right">{f.why}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {level === "explore" && <ExploreLevel a={a} onOpen={onOpen} />}
      {level === "technical" && <TechnicalLevel m={m} a={a} sid={sid} />}
    </article>
  );
}
