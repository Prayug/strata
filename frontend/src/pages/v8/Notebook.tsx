/* One research notebook: questions and answers in order, the newest open, earlier ones folded to their finding.
   Every turn keeps the question as typed and how it was read; a change says what changed and can be undone. */
import { ArrowUp, ChevronDown, ChevronRight, Loader2, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AiMessage } from "@/lib/lab";
import { AnswerView } from "./Answer";
import { ExploreSheet } from "./Explore";
import { ReadingLine } from "./Reading";
import { StrategySentence, describeChange } from "./StrategySentence";
import { said, turns, useNotebook } from "./data";

function Progress({ c }: { c: any }) {
  const run = (c.runs || [])[c.runs.length - 1];
  const q = useQuery({ queryKey: ["run-progress", run?.run_id], queryFn: () => api.get<any>(`/api/runs/${run.run_id}`), enabled: !!run, refetchInterval: 1200 });
  const p = q.data?.progress;
  return (
    <div className="small soft flex items-center gap-3" data-testid="working">
      <Loader2 className="size-4 animate-spin" style={{ color: "var(--amber)" }} />
      <span>{(c.runs || []).length > 1 ? `Measuring ${c.runs.length} variations on the stored history` : "Measuring it on the stored history"}{p != null ? ` · ${Math.round(p * 100)}%` : ""}</span>
      {q.data?.message && <span className="tiny mute truncate">{q.data.message}</span>}
    </div>
  );
}

function Needs({ m, sid, nb }: { m: AiMessage; sid: string; nb: ReturnType<typeof useNotebook> }) {
  const c = m.content || {};
  const qs = (c.questions || []).filter((q: any) => q.answered == null);
  const unsupported: string[] = c.draft?.unsupported || [];
  return (
    <div className="flex flex-col gap-4" data-testid="needs-input">
      {qs.map((q: any) => (
        <div key={q.id}>
          <p className="lede">{q.text}</p>
          <div className="mt-2 flex flex-wrap gap-2">{q.options.map((o: any, i: number) => <button key={i} className="act" onClick={() => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/answer`, { version_id: c.version_id, question_id: q.id, option: i }))}>{o.label}</button>)}</div>
        </div>
      ))}
      {unsupported.length > 0 && (
        <div className="aside-rule">
          <p className="lede">Part of the request could not be represented, so nothing ran yet: {unsupported.map((u) => `"${u}"`).join(", ")}.</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <button className="act" onClick={() => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/confirm`, { version_id: c.version_id }))}>Run without it</button>
            <span className="tiny mute self-center">or rephrase that part below.</span>
          </div>
        </div>
      )}
      {!qs.length && !unsupported.length && c.state === "awaiting_confirmation" && (() => {
        const review: string[] = c.draft?.review || [];
        const several = review.filter((x) => /several parts/.test(x));
        const wrong = review.filter((x) => !/several parts/.test(x));
        const why = review.length ? [] : [String(c.text || "").replace(/^Here is how I read that (strategy|question)\.\s*/, "").replace(/^Updated the (strategy|study)\.\s*/, "")].filter(Boolean);
        return (
          <div className="aside-rule" data-testid="confirm-strategy">
            <div className="kicker mb-2">{wrong.length ? "Check this reading: something in it may not be what you meant" : "Check this reading before it runs"}</div>
            {wrong.map((x, i) => <p key={i} className="small mb-1.5" style={{ color: "var(--down)" }} data-testid="confirm-reason">{x}</p>)}
            {several.length > 0 && !wrong.length && <p className="small soft mb-1.5">This trade has several parts. The sentence above is how each one was read.</p>}
            {why.map((x, i) => <p key={i} className="small soft mb-1.5">{x}</p>)}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button className="act primary" disabled={nb.act.isPending} onClick={() => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/confirm`, { version_id: c.version_id }))} data-testid="run-strategy">Yes, run it</button>
              <span className="tiny mute">Nothing has run. Click an underlined value above to change it first, or say what to change below.</span>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

type Ctx = { latestResult?: number; latestPlan?: number; lastId?: number; prevDraft: Record<number, any>; prevAnswer: Record<number, any>; expanded: Set<number>; toggle: (id: number) => void };

function Reply({ m, sid, nb, open, onOpen, onTrade, symbols, ctx }: { m: AiMessage; sid: string; nb: ReturnType<typeof useNotebook>; open: boolean; onOpen: (s: any) => void; onTrade: (r: string, id: number) => void; symbols: string[]; ctx: Ctx }) {
  const c = m.content || {};
  if (m.kind === "result") {
    const full = open && (m.id === ctx.latestResult || ctx.expanded.has(m.id));
    if (full) return <AnswerView m={m} sid={sid} onOpen={onOpen} onAsk={nb.send} onTrade={onTrade} prev={ctx.prevAnswer[m.id]} />;
    return (
      <button className="flex items-start gap-2 border-0 bg-transparent p-0 text-left" style={{ cursor: "pointer", color: "inherit" }} onClick={() => ctx.toggle(m.id)} data-testid="folded-result">
        <ChevronRight size={15} className="mt-1 shrink-0 faint" /><span className="serif text-[18px] leading-snug soft">{c.answer?.headline || String(c.text || "").split(". ")[0]}</span>
      </button>
    );
  }
  if (m.kind === "trades") return <div className="flex flex-wrap items-center gap-3"><span className="lede">{c.text}</span>{c.explore && <button className="act" onClick={() => onOpen(c.explore)} data-testid="open-observations">Open the list</button>}</div>;
  if (["plan", "investigation", "run", "sweep", "hypothesis", "graph", "experiment", "discovery"].includes(m.kind)) {
    const strat = m.kind === "plan" && c.draft?.kind === "options";
    const needs = m.kind === "plan" && ["awaiting_answer", "awaiting_confirmation", "blocked"].includes(c.state);
    const editable = strat && m.id === ctx.latestPlan;
    const prev = ctx.prevDraft[m.id];
    const edited = strat && prev && !c.change && (c.diff || []).length > 0 ? describeChange(prev, c.draft, 0) : [];
    // a research plan that stopped to ask (a part it could not read, data to download): its choices, which the notebook did not show
    const waitingPlan = m.kind === "graph" && ["awaiting", "needs_data"].includes(c.state) && (c.choices || []).length > 0;
    return (
      <div className="flex flex-col gap-4" id={`msg-${m.id}`}>
        {edited.length > 0 && <p className="small soft" data-testid="version-change"><span className="kicker mr-2">New version</span>{edited.join(" ")} Everything else stays the same.</p>}
        {editable && open && <StrategySentence draft={c.draft} symbols={symbols} busy={nb.act.isPending} more={c.sentence?.more} effective={c.effective} onRun={(d) => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/edit`, { draft: d, run: true }))} />}
        {!c.reading && !strat && c.text && !waitingPlan && <p className="small soft">{String(c.text).slice(0, 400)}</p>}
        {waitingPlan && (
          <div data-testid="plan-decision">
            <p className="lede">{c.text}</p>
            <div className="mt-2 flex flex-wrap gap-2">{c.choices.map((x: any) => <button key={x.id} className={x.id === "cancel" ? "act" : "act primary"} onClick={() => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/graph/decide`, { message_id: m.id, choice: x.id }))}>{x.label}</button>)}</div>
          </div>
        )}
        {needs ? <Needs m={m} sid={sid} nb={nb} /> : (c.runs || []).length > 0 && !c.finalized ? <Progress c={c} /> : null}
      </div>
    );
  }
  if (m.kind === "decision") {
    return (
      <div data-testid="decision">
        <p className="lede">{c.text}</p>
        <div className="mt-2 flex flex-wrap gap-2">{(c.choices || []).map((x: any) => <button key={x.id} className="act" onClick={() => nb.act.mutate(() => api.post(`/api/ai/sessions/${sid}/decide`, { message_id: m.id, choice: x.id }))}>{x.label}</button>)}</div>
      </div>
    );
  }
  // suggestions of what to look at next, each with its reason; a click asks it
  if ((c.next_steps || []).length > 0) {
    return (
      <div data-testid="next-steps">
        <p className="lede">Where I would look next, given this result:</p>
        <div className="qlist mt-2">
          {c.next_steps.map((x: any, i: number) => (
            <button key={i} className="qrow" onClick={() => nb.send(x.ask)} data-testid="next-step">
              <span className="q" style={{ fontSize: 16.5 }}>{x.ask}</span><span className="tiny faint text-right">{x.why}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }
  // a reading shown before it runs: the sentence with its parts to change, why it waits, and the go-ahead
  if (c.confirm && c.reading) {
    const live = m.id === ctx.lastId;
    const reasons: string[] = c.confirm.reasons || [];
    return (
      <div className="aside-rule" data-testid="confirm-reading">
        <div className="kicker mb-2">{reasons.length ? "Check this reading: part of your message is not in it" : "Check this reading before it runs"}</div>
        <p className="serif" style={{ fontSize: 20, lineHeight: 1.6 }}><ReadingLine parts={c.reading_parts} text={c.reading} live={live} onSay={nb.send} /></p>
        {reasons.map((r, i) => <p key={i} className="small mt-2" style={{ color: "var(--down)" }} data-testid="confirm-reason">{r}</p>)}
        {live ? (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button className="act primary" disabled={nb.act.isPending} onClick={() => nb.send("Run it as read")} data-testid="run-as-read">{reasons.length ? "Run it as read" : "Yes, run it"}</button>
            <span className="tiny mute">Nothing has run. Click an underlined part to change it, or say it another way below.</span>
          </div>
        ) : <p className="tiny mute mt-2">This reading was not run.</p>}
      </div>
    );
  }
  // a question back (clarify), a refusal (with the closest question that can be answered, when there is one), a note
  return (
    <div data-testid={c.declined ? "declined" : c.clarify ? "clarify" : "note"}>
      <p className={c.declined ? "lede soft" : "lede"}>{c.text}</p>
      {(c.options || []).length > 0 && <div className="mt-3 flex flex-wrap gap-2">{c.options.map((o: any, i: number) => <button key={i} className={o.ask ? "act primary" : "act"} onClick={() => nb.send(o.ask || o.label)} data-testid={o.ask ? "closest-question" : "clarify-option"}>{o.label}</button>)}</div>}
    </div>
  );
}

export function Notebook({ sid }: { sid: string }) {
  const nb0 = useNotebook(sid);
  const [sending, setSending] = useState<string | null>(null);
  const nb = useMemo(() => ({ ...nb0, send: (t: string) => { setSending(t); nb0.send(t); } }), [nb0]); // eslint-disable-line
  useEffect(() => { if (!nb0.act.isPending) setSending(null); }, [nb0.act.isPending]);
  const [text, setText] = useState("");
  const [spec, setSpec] = useState<any>(null);
  const [frame, setFrame] = useState<any>(null);
  const [opened, setOpened] = useState<Set<number>>(new Set());
  const [selection, setSelection] = useState<{ count: number; what: string } | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const meta = useQuery({ queryKey: ["v8-symbols"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 300_000 });
  const symbols = (meta.data || []).filter((r: any) => !r.synthetic && r.option_trades).map((r: any) => r.symbol);
  useEffect(() => {
    const k = `v8-first-${sid}`;
    const first = sessionStorage.getItem(k);
    if (first) { sessionStorage.removeItem(k); nb.send(first); }
  }, [sid]); // eslint-disable-line
  const ts = useMemo(() => turns(nb.messages), [nb.messages]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const ctx: Ctx = useMemo(() => {
    const plans = nb.messages.filter((m) => m.kind === "plan");
    const prevDraft: Record<number, any> = {};
    let last: any = null;
    for (const p of plans) { if (last) prevDraft[p.id] = last; if (p.content?.draft) last = p.content.draft; }
    // an answer that follows a change (a turn with a change line) is shown against the answer before it
    const prevAnswer: Record<number, any> = {};
    const lastOfKind: Record<string, any> = {};
    let changed = false;
    for (const m of nb.messages) {
      if (m.role === "user") changed = false;
      if (m.content?.change) changed = true;
      const a = m.kind === "result" ? m.content?.answer : null;
      if (a?.viz?.kind) { if (changed && lastOfKind[a.viz.kind]) prevAnswer[m.id] = lastOfKind[a.viz.kind]; lastOfKind[a.viz.kind] = a; }
    }
    return { latestResult: [...nb.messages].reverse().find((m) => m.kind === "result")?.id, latestPlan: plans[plans.length - 1]?.id, lastId: nb.messages[nb.messages.length - 1]?.id, prevDraft, prevAnswer, expanded,
             toggle: (id: number) => setExpanded((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; }) };
  }, [nb.messages, expanded]);
  // a new answer opens at its question, not at the bottom of the page; while a question is being read, the bottom is shown
  const lastQ = [...nb.messages].reverse().find((m) => m.role === "user")?.id;
  const lastResult = [...nb.messages].reverse().find((m) => m.role === "assistant" && m.kind === "result")?.id;
  useEffect(() => { if (nb.act.isPending) end.current?.scrollIntoView({ block: "end", behavior: "smooth" }); }, [nb.act.isPending]);
  useEffect(() => {
    // a re-run inside the same turn (an edited strategy) opens at its own version, not at the top of the turn
    const planBefore = [...nb.messages].reverse().find((m) => m.kind === "plan" && (!lastResult || m.id < lastResult) && (!lastQ || m.id > lastQ));
    const firstReply = nb.messages.find((m) => lastQ && m.id > lastQ && m.role === "assistant" && m.kind !== "text");
    const el = planBefore && firstReply && planBefore.id !== firstReply.id ? document.getElementById(`msg-${planBefore.id}`) : lastQ ? document.getElementById(`turn-${lastQ}`) : null;
    const t = setTimeout(() => el?.scrollIntoView({ block: "start", behavior: "smooth" }), 60);
    return () => clearTimeout(t);
  }, [lastQ, lastResult]); // eslint-disable-line
  const submit = () => { const t = text.trim(); if (!t) return; nb.send(t); setText(""); };
  const lastIdx = ts.length - 1;
  const openSpec = (s: any, fr?: any) => { setFrame(fr || null); setSpec(s); };
  const undo = () => nb.send("undo");
  return (
    <div className="flex min-h-full flex-col">
      <div className="wide flex flex-1 gap-10 px-5 pb-40 pt-8 md:px-8">
        <nav className="hidden w-[230px] shrink-0 lg:block" aria-label="Questions in this notebook">
          <div className="sticky top-6">
            <div className="kicker mb-2">In this notebook</div>
            <ol className="flex flex-col gap-2.5">
              {ts.filter((t) => t.q).map((t, i) => {
                const res = t.a.find((m) => m.kind === "result");
                return (
                  <li key={t.q!.id}><a href={`#turn-${t.q!.id}`} className="block text-[13px] leading-snug" style={{ color: i === ts.length - 1 ? "var(--paper)" : "var(--mute)" }}>{said(t.q!)}</a>
                    {res?.content?.answer?.headline && <div className="tiny faint mt-0.5 line-clamp-2">{res.content.answer.headline}</div>}</li>
                );
              })}
            </ol>
          </div>
        </nav>
        <main className="col min-w-0 flex-1" style={{ maxWidth: 820 }}>
          {nb.q.isLoading && <div className="small mute">Opening the notebook…</div>}
          {ts.map((t, i) => {
            const latest = i === lastIdx;
            const user = t.q;
            const change = t.a.find((m) => m.content?.change)?.content?.change;
            const assumed: string[] = (t.a.find((m) => m.content?.assumed)?.content?.assumed || []).map((x: any) => (typeof x === "string" ? x : x.text)).filter(Boolean);
            // how this question was read, said once under it; a reading that is still waiting says itself in its own block
            const read = t.a.find((m) => m.content?.reading && !m.content?.confirm)?.content;
            const leftOut: string[] = t.a.find((m) => m.content?.left_out)?.content?.left_out || [];
            const isOpen = latest || (user && opened.has(user.id));
            return (
              <section key={user?.id ?? i} id={user ? `turn-${user.id}` : undefined} className={i ? "rule mt-10 pt-8" : ""} data-testid="turn">
                {user && (
                  <header className="mb-4">
                    <div className="flex items-start gap-2">
                      {!latest && <button aria-label={isOpen ? "Fold" : "Unfold"} className="mt-1" style={{ background: "none", border: 0, color: "var(--faint)", cursor: "pointer" }} onClick={() => setOpened((s) => { const n = new Set(s); if (n.has(user.id)) n.delete(user.id); else n.add(user.id); return n; })}>{isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</button>}
                      <h1 className={latest ? "display" : "serif text-[22px] leading-tight"} style={latest ? { fontSize: "clamp(26px, 3vw, 36px)" } : undefined} data-testid="question">{said(user)}</h1>
                    </div>
                    {change && (
                      <div className="mt-3 flex flex-wrap items-center gap-3 small" data-testid="change">
                        <span className="soft">{change.text}</span>
                        {latest && change.undo && <button className="link inline-flex items-center gap-1" onClick={undo} data-testid="undo"><RotateCcw size={13} />Undo</button>}
                      </div>
                    )}
                    {read && (
                      <p className="small mute mt-3" data-testid="reading"><span className="kicker mr-2">Read as</span>
                        <span className="soft"><ReadingLine parts={read.reading_parts} text={read.reading} live={latest && !nb.act.isPending} onSay={nb.send} /></span></p>
                    )}
                    {leftOut.map((x, k) => <p key={k} className="tiny mt-1.5" style={{ color: "var(--down)" }} data-testid="left-out">{x} The answer below does not include it.</p>)}
                    {assumed.length > 0 && <ul className="tiny mute mt-2 flex flex-col gap-0.5" data-testid="assumed">{assumed.map((x, k) => <li key={k}>Assumed: {x}</li>)}</ul>}
                  </header>
                )}
                <div className="flex flex-col gap-6">
                  {t.a.filter((m) => m.kind !== "text" || m.role === "assistant").map((m) => (
                    <Reply key={m.id} m={m} sid={sid} nb={nb} open={!!isOpen} ctx={ctx} symbols={symbols.length ? symbols : ["SPY", "QQQ", "IWM"]}
                      onOpen={(s) => openSpec(s, m.content?.frame)} onTrade={(r, id) => openSpec({ kind: "replay", run_id: r, trade: id })} />
                  ))}
                  {latest && nb.act.isPending && !sending && <div className="small mute flex items-center gap-2"><Loader2 className="size-4 animate-spin" />Working…</div>}
                </div>
              </section>
            );
          })}
          {sending && nb.act.isPending && (
            <section className={ts.length ? "rule mt-10 pt-8" : ""} data-testid="turn-sending">
              <h1 className="display" style={{ fontSize: "clamp(26px, 3vw, 36px)" }}>{sending}</h1>
              <div className="small mute mt-4 flex items-center gap-2"><Loader2 className="size-4 animate-spin" />Reading the question…</div>
            </section>
          )}
          <div ref={end} />
        </main>
      </div>
      <div className="fixed inset-x-0 bottom-0 z-20" style={{ background: "linear-gradient(to top, var(--ink-0) 72%, transparent)" }}>
        <div className="mx-auto w-full px-5 pb-5 pt-6 md:px-8" style={{ maxWidth: 820 + 230 + 40, paddingLeft: undefined }}>
          <div className="lg:ml-[270px]">
            {selection && (
              <div className="mb-2 flex items-center gap-2 tiny" data-testid="selection-chip">
                <span className="soft" style={{ border: "1px solid var(--rule-2)", padding: "2px 8px" }}>Asking about {selection.count} selected {selection.what}</span>
                <button className="link" onClick={() => { setSelection(null); api.post("/api/v8/selection", { session_id: sid, ids: [] }).catch(() => {}); }} aria-label="Clear the selection">clear</button>
              </div>
            )}
            <div className="ask compact">
              <textarea ref={box} rows={1} value={text} placeholder={typeof window !== "undefined" && window.innerWidth < 640 ? "Ask a follow-up or change something…" : "Ask a follow-up, or change something: “use 5 min”, “do QQQ”, “only when yesterday was red”…"} onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} data-testid="composer" aria-label="Ask" />
              <button className="send" onClick={submit} disabled={!text.trim() || nb.act.isPending} aria-label="Send"><ArrowUp size={16} /></button>
            </div>
            {nb.pending && <div className="tiny mute mt-1.5">Working on the stored history. You can keep typing.</div>}
          </div>
        </div>
      </div>
      {spec && <ExploreSheet spec={spec} frame={frame} onClose={() => setSpec(null)}
        onAsk={(ids, what, q) => { api.post("/api/v8/selection", { session_id: sid, ids, what }).then(() => { setSelection({ count: ids.length, what }); setSpec(null); if (q) nb.send(q); }).catch(() => {}); }} />}
    </div>
  );
}

export default Notebook;
