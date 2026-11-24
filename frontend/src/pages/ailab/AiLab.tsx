/* AI Strategy Lab: describe a strategy or a market question in plain English; the
   deterministic engine runs it. Left: research log. Centre: report. Right: how the
   request was interpreted, what was assumed and what the stored data can support. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUp, Check, ChevronRight, CircleDashed, CornerDownRight, Database, GitBranch, History, Loader2, PanelBottomClose, PanelBottomOpen, PanelRightClose, PanelRightOpen, Pencil, Play, Plus, RotateCcw, Settings2, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Group, Panel as RPanel, Separator } from "react-resizable-panels";
import { Link, useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { DataTable, Empty, ErrorBox, Loading, PageHeader, Panel, Pnl, SyntheticBadge, type Col } from "@/components/common";
import { BacktestReport } from "@/components/research/BacktestReport";
import { SessionReplay, SessionsTable, StudyReport } from "@/components/research/StudyReport";
import { Button, Input, NumInput, Segmented, Select, Tip } from "@/components/ui";
import { MODE_LABEL, MODE_LONG, MODE_TONE, type AiMessage, type AiPayload, type AiSession, type AiVersion, type Coverage, type Mode, type PlanItem, type RunRef } from "@/lib/ai";
import { api, queryClient, useRun } from "@/lib/api";
import { ago, cn, etTime, minutesLabel, money, num, REASON_LABEL } from "@/lib/format";
import type { Trade } from "@/lib/types";

export default function AiLab() {
  const { sid } = useParams();
  return sid ? <Workspace key={sid} sid={sid} /> : <Landing />;
}

function ModeBadge({ mode, className }: { mode: Mode; className?: string }) {
  return <Tip content={MODE_LONG[mode]}><span className={cn("inline-flex h-4 items-center rounded-[2px] border px-1.5 text-[10px] font-semibold uppercase tracking-wider", MODE_TONE[mode], className)}>{MODE_LABEL[mode]}</span></Tip>;
}

function useAiMeta() {
  return useQuery({ queryKey: ["ai-meta"], queryFn: () => api.get<any>("/api/ai/meta"), staleTime: 30_000 });
}

function InterpreterChip() {
  const meta = useAiMeta();
  const llm = meta.data?.llm;
  if (!llm) return null;
  const label = llm.error ? "model unavailable" : llm.active ? llm.active.split(":").slice(1).join(":") || llm.active : "built-in interpreter";
  return (
    <Tip content={llm.error ? `${llm.error}. Requests fall back to the built-in interpreter.` : llm.active ? `Requests are interpreted by ${llm.active}. It only fills in a structured draft; every number is calculated by the engine.` : "No language model configured: a built-in rule-based interpreter reads your requests. Add a key under Settings for freer phrasing."}>
      <Link to="/settings" className={cn("flex h-5 items-center gap-1 rounded-sm border px-1.5 text-2xs", llm.error ? "border-warn/50 text-warn" : "border-line-2 text-muted hover:text-fg")}>
        <Settings2 className="size-3" /> {label}
      </Link>
    </Tip>
  );
}

// ============================================================================ landing
function Landing() {
  const nav = useNavigate();
  const meta = useAiMeta();
  const sessions = useQuery({ queryKey: ["ai-sessions"], queryFn: () => api.get<AiSession[]>("/api/ai/sessions") });
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage") });
  const [text, setText] = useState("");
  const start = useMutation({
    mutationFn: async (t: string) => {
      const s = await api.post<AiSession>("/api/ai/sessions", { title: t.slice(0, 70) });
      await api.post(`/api/ai/sessions/${s.id}/messages`, { text: t });
      return s;
    },
    onSuccess: (s) => { queryClient.invalidateQueries({ queryKey: ["ai-sessions"] }); nav(`/lab/${s.id}`); },
    onError: (e: any) => toast.error(e.message),
  });
  const del = useMutation({ mutationFn: (id: string) => api.del(`/api/ai/sessions/${id}`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["ai-sessions"] }) });
  const real = (cov.data || []).filter((r) => !r.synthetic && r.kind !== "vol_index" && r.symbol !== "VIX" && (r.daily || Object.keys(r.intraday || {}).length));
  const noData = cov.isSuccess && real.length === 0;
  const go = () => text.trim() && !start.isPending && start.mutate(text.trim());
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="AI Strategy Lab" sub={<span className="text-2xs text-muted">plain English in, deterministic backtests out</span>}><InterpreterChip /></PageHeader>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto flex max-w-[1080px] flex-col gap-4 px-6 py-7">
          <div>
            <div className="label mb-1.5">Describe a strategy or ask how the market behaves</div>
            <div className="rounded-md border border-line-2 bg-bg-1 focus-within:border-accent/70">
              <textarea autoFocus value={text} onChange={(e) => setText(e.target.value)} rows={3}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); go(); } }}
                placeholder="How often does SPY rise 1% from the open after falling 3% or more the previous trading day?"
                className="block w-full resize-none bg-transparent px-3 pt-2.5 text-[14px] leading-relaxed outline-none placeholder:text-faint" />
              <div className="flex items-center gap-2 px-2 pb-2">
                <span className="text-2xs text-faint">Enter to run · Shift+Enter for a new line · times may be Pacific or Eastern</span>
                <span className="flex-1" />
                <Button variant="primary" onClick={go} disabled={!text.trim() || start.isPending || noData}>{start.isPending ? <Loader2 className="animate-spin" /> : <ArrowUp />} Run</Button>
              </div>
            </div>
            {start.error ? <ErrorBox className="mt-2" error={start.error} /> : null}
          </div>
          {noData && <FirstData />}
          <div className="grid grid-cols-[1.25fr_1fr] gap-4">
            <div>
              <div className="label mb-1.5">Try one of these</div>
              <div className="flex flex-col border border-line bg-bg-1">
                {(meta.data?.examples || []).map((ex: string) => (
                  <button key={ex} disabled={noData || start.isPending} onClick={() => start.mutate(ex)}
                    className="group flex items-start gap-2 border-b border-line/70 px-2.5 py-1.5 text-left text-xs text-fg-2 last:border-0 hover:bg-bg-2 hover:text-fg disabled:opacity-40">
                    <ChevronRight className="mt-0.5 size-3 shrink-0 text-faint group-hover:text-accent" />{ex}
                  </button>
                ))}
              </div>
              <div className="label mb-1.5 mt-4">How a request is answered</div>
              <div className="grid grid-cols-3 gap-px border border-line bg-line text-2xs">
                {(["historical", "study", "modeled"] as Mode[]).map((m) => (
                  <div key={m} className="bg-bg-1 p-2.5">
                    <ModeBadge mode={m} />
                    <p className="mt-1.5 leading-relaxed text-muted">{{
                      historical: "Used when observed option quotes cover the request. Entries and exits fill at real bid/ask quotes; nothing is interpolated.",
                      study: "Used for questions about the underlying, or when option quotes are missing. Reports price behaviour only, never option P&L.",
                      modeled: "Only when you opt in. Real underlying prices with theoretical premiums, labelled as estimates on every screen.",
                    }[m]}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex flex-col gap-4">
              <div>
                <div className="label mb-1.5 flex items-center justify-between">Market data on this machine <Link to="/data" className="normal-case tracking-normal text-accent hover:underline">manage</Link></div>
                <div className="border border-line bg-bg-1">
                  {cov.isLoading ? <Loading /> : real.length === 0 ? <div className="p-2.5 text-xs text-muted">No real market data stored yet.</div> : (
                    <table className="tbl"><thead><tr><th>Symbol</th><th>Daily</th><th>Intraday</th><th>Options</th></tr></thead><tbody>
                      {real.map((r) => (
                        <tr key={r.symbol}>
                          <td className="font-medium text-fg">{r.symbol}</td>
                          <td className="num text-fg-2">{r.daily ? `${r.daily.first.slice(0, 4)}–${r.daily.last.slice(0, 4)}` : "—"}</td>
                          <td className="num text-fg-2">{Object.keys(r.intraday || {}).length ? Object.entries(r.intraday).map(([k, v]: any) => `${k} ${v.first.slice(2, 7)}→`).join(" · ") : <span className="text-faint">none</span>}</td>
                          <td className="num text-fg-2">{r.options ? `${r.options.resolution} ${r.options.first.slice(0, 4)}–${r.options.last.slice(0, 4)}` : <span className="text-faint">none</span>}</td>
                        </tr>
                      ))}
                    </tbody></table>
                  )}
                </div>
                <p className="mt-1 text-2xs leading-relaxed text-faint">If a request needs data that is not stored, the lab says exactly what is missing. It never substitutes simulated prices.</p>
              </div>
              <div>
                <div className="label mb-1.5">Recent research</div>
                <div className="border border-line bg-bg-1">
                  {(sessions.data || []).length === 0 ? <div className="p-2.5 text-xs text-muted">Nothing yet.</div> : (sessions.data || []).slice(0, 12).map((s) => (
                    <div key={s.id} className="group flex items-center gap-2 border-b border-line/70 px-2.5 py-1.5 last:border-0 hover:bg-bg-2">
                      <Link to={`/lab/${s.id}`} className="min-w-0 flex-1 truncate text-xs text-fg-2 group-hover:text-fg">{s.title}</Link>
                      <span className="num text-2xs text-faint">{ago(s.updated_at)}</span>
                      <button className="text-faint opacity-0 hover:text-down group-hover:opacity-100" onClick={() => confirm("Delete this research session? Its runs stay under Backtests.") && del.mutate(s.id)}><Trash2 className="size-3" /></button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function FirstData() {
  const [job, setJob] = useState<string | null>(null);
  const run = useRun(job || undefined, true);
  const go = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/quickstart", { symbols: ["SPY", "QQQ", "IWM"], intraday: true }), onSuccess: (r) => setJob(r.id), onError: (e: any) => toast.error(e.message) });
  useEffect(() => {
    if (run.data?.status === "done") {
      toast.success("Market data downloaded");
      queryClient.invalidateQueries({ queryKey: ["coverage"] });
      queryClient.invalidateQueries({ queryKey: ["meta"] });
    }
  }, [run.data?.status]);
  const busy = go.isPending || (job && run.data?.status !== "done" && run.data?.status !== "error");
  return (
    <div className="flex items-center gap-3 border border-accent/40 bg-accent/[0.06] px-3 py-2.5">
      <Database className="size-4 shrink-0 text-accent" />
      <div className="min-w-0 flex-1 text-xs">
        <div className="text-fg">No market data yet. Download free real history to begin.</div>
        <div className="text-2xs text-muted">Daily bars since 1993 and recent intraday bars for SPY, QQQ, IWM plus VIX - a few MB from Yahoo Finance and CBOE, no account needed. {run.data?.status === "error" ? <span className="text-down">Failed: {run.data.error?.split("\n")[0]}</span> : busy ? <span className="text-fg-2">{run.data?.message || "starting"}…</span> : null}</div>
      </div>
      <Button variant="primary" disabled={!!busy} onClick={() => go.mutate()}>{busy ? <Loader2 className="animate-spin" /> : <Database />} Get free data</Button>
    </div>
  );
}

// ============================================================================ workspace
const PENDING = ["plan", "sweep", "run"];
const isPending = (m: AiMessage) => PENDING.includes(m.kind) && m.content?.runs?.length > 0 && !m.content.finalized;

function Workspace({ sid }: { sid: string }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["ai", sid], queryFn: () => api.get<AiPayload>(`/api/ai/sessions/${sid}`) });
  const [sel, setSel] = useState<number | null>(null); // selected result message
  const [focusRun, setFocusRun] = useState<string | null>(null);
  const [tradeFilter, setTradeFilter] = useState<"all" | "win" | "loss">("all");
  const [showRight, setShowRight] = useState(true);
  const [showBottom, setShowBottom] = useState(true);
  const messages = q.data?.messages || [];
  const pending = messages.some(isPending);
  const jobs = useQuery({ queryKey: ["jobs", "fast"], queryFn: () => api.get<{ active: any[] }>("/api/jobs"), refetchInterval: pending ? 900 : false, enabled: pending });

  // finished runs become result messages (explanation + comparison) on the server
  useEffect(() => {
    if (!pending) return;
    let stop = false;
    const tick = async () => {
      try {
        const r = await api.post<{ messages: AiMessage[] }>(`/api/ai/sessions/${sid}/finalize`);
        if (!stop && r.messages.length) {
          queryClient.invalidateQueries({ queryKey: ["ai", sid] });
          queryClient.invalidateQueries({ queryKey: ["runs"] });
          setSel(r.messages[r.messages.length - 1].id);
          setFocusRun(null);
        }
      } catch { /* retried on the next tick */ }
    };
    const h = setInterval(tick, 1200);
    tick();
    return () => { stop = true; clearInterval(h); };
  }, [pending, sid]);

  const results = messages.filter((m) => m.kind === "result");
  const cur = results.find((m) => m.id === sel) || results[results.length - 1] || null;
  const lastPlan = [...messages].reverse().find((m) => m.kind === "plan") || null;
  const act = useMutation({
    mutationFn: (fn: () => Promise<any>) => fn(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["ai", sid] }),
    onError: (e: any) => toast.error(e.message),
  });
  const send = (text: string) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/messages`, { text }));
  const runMode = (version_id: number, mode: Mode) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/run`, { version_id, mode }));
  const edit = (draft: any) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/edit`, { draft, run: true }));
  const restore = (version_id: number) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/restore`, { version_id }));

  // a "show me the losing trades" reply drives the bottom pane
  const lastTrades = [...messages].reverse().find((m) => m.kind === "trades");
  useEffect(() => {
    if (lastTrades) {
      setTradeFilter(lastTrades.content.filter || "all");
      setFocusRun(lastTrades.content.run_id);
      setShowBottom(true);
    }
  }, [lastTrades?.id]);

  if (q.isLoading) return <Loading />;
  if (q.error) return <div className="p-4"><ErrorBox error={q.error} /><Button className="mt-2" onClick={() => nav("/")}>Back to the lab</Button></div>;
  const data = q.data!;
  const mainRun: string | null = focusRun || cur?.content.run_id || cur?.content.table?.rows?.[cur.content.table.rows.length - 1]?.run_id || null;
  return (
    <div className="flex h-full flex-col">
      <PageHeader title={<span className="flex items-center gap-2"><Link to="/" className="text-muted hover:text-fg">AI Lab</Link><ChevronRight className="size-3 text-faint" /><span className="truncate">{data.session.title}</span></span>}>
        <InterpreterChip />
        <Tip content="Trade / session log"><Button size="icon" variant="ghost" onClick={() => setShowBottom((v) => !v)}>{showBottom ? <PanelBottomClose /> : <PanelBottomOpen />}</Button></Tip>
        <Tip content="Interpretation panel"><Button size="icon" variant="ghost" onClick={() => setShowRight((v) => !v)}>{showRight ? <PanelRightClose /> : <PanelRightOpen />}</Button></Tip>
        <Button onClick={() => nav("/")}><Plus /> New research</Button>
      </PageHeader>
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <RPanel defaultSize="29%" minSize="20%" maxSize="45%" className="bg-bg-1">
          <Log messages={messages} busy={act.isPending} onSend={send} onSelect={(id) => { setSel(id); setFocusRun(null); }} selected={cur?.id ?? null} jobs={jobs.data?.active || []} onRunMode={runMode} draftKind={lastPlan?.content.kind} />
        </RPanel>
        <Separator />
        <RPanel minSize="30%">
          <Group orientation="vertical">
            <RPanel minSize="25%">
              <div className="h-full overflow-auto">
                {cur ? <Report msg={cur} results={results} onSelect={(id) => { setSel(id); setFocusRun(null); }} focusRun={mainRun} onFocus={setFocusRun} onAdopt={edit} />
                  : pending ? <Empty title="Running…" icon={Loader2}>The engine is working through the sessions. Results appear here as soon as the run finishes.</Empty>
                    : <Empty title="No result yet">{lastPlan?.content.coverage?.messages?.[0] || "Describe a strategy or a market question in the log on the left."}</Empty>}
              </div>
            </RPanel>
            {showBottom && mainRun && <><Separator /><RPanel defaultSize="34%" minSize="14%"><BottomLog runId={mainRun} filter={tradeFilter} setFilter={setTradeFilter} /></RPanel></>}
          </Group>
        </RPanel>
        {showRight && <><Separator /><RPanel defaultSize="25%" minSize="18%" maxSize="38%" className="bg-bg-1">
          <Inspector data={data} plan={lastPlan} onRunMode={runMode} onEdit={edit} onRestore={restore} busy={act.isPending || pending} />
        </RPanel></>}
      </Group>
    </div>
  );
}

// ---------------------------------------------------------------------------- research log
function Log({ messages, busy, onSend, onSelect, selected, jobs, onRunMode, draftKind }: {
  messages: AiMessage[]; busy: boolean; onSend: (t: string) => void; onSelect: (id: number) => void; selected: number | null; jobs: any[]; onRunMode: (v: number, m: Mode) => void; draftKind?: string;
}) {
  const end = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [messages.length, busy]);
  const submit = () => { if (text.trim() && !busy) { onSend(text.trim()); setText(""); } };
  const chips = draftKind === "study"
    ? ["Change the previous-day decline to 2% and compare", "How sensitive is this to the target?", "Add a 0.5% stop", "Only since 2020", "Test out of sample", "Show the sessions that missed"]
    : draftKind === "options" ? ["What if I bought puts instead?", "Test 7DTE, 14DTE and 30DTE", "How sensitive is this to slippage?", "Show me every losing trade", "Test out of sample"] : [];
  return (
    <div className="flex h-full flex-col">
      <div className="label flex h-7 shrink-0 items-center border-b border-line px-2.5">Research log</div>
      <div className="min-h-0 flex-1 overflow-auto px-2.5 py-2">
        <div className="flex flex-col gap-2.5">
          {messages.map((m) => <LogEntry key={m.id} m={m} onSelect={onSelect} selected={selected} jobs={jobs} onRunMode={onRunMode} />)}
          {busy && <div className="flex items-center gap-2 text-2xs text-muted"><Loader2 className="size-3 animate-spin" /> reading the request…</div>}
          <div ref={end} />
        </div>
      </div>
      {chips.length > 0 && (
        <div className="flex shrink-0 gap-1 overflow-x-auto border-t border-line px-2 py-1.5">
          {chips.map((c) => <button key={c} disabled={busy} onClick={() => onSend(c)} className="shrink-0 rounded-sm border border-line-2 px-1.5 py-0.5 text-2xs text-muted hover:border-faint hover:text-fg disabled:opacity-40">{c}</button>)}
        </div>
      )}
      <div className="shrink-0 border-t border-line p-2">
        <div className="flex items-end gap-1.5 rounded-sm border border-line-2 bg-bg-2 p-1.5 focus-within:border-accent/70">
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder="Refine it, compare a change, or ask something new…"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
            className="block min-h-0 flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-faint" />
          <Button variant="primary" size="icon" disabled={!text.trim() || busy} onClick={submit}><ArrowUp /></Button>
        </div>
      </div>
    </div>
  );
}

function StatusPill({ s }: { s: PlanItem["status"] }) {
  const m = { stated: ["you said", "text-faint"], assumed: ["assumed", "text-warn"], missing: ["missing", "text-down"] }[s];
  return <span className={cn("shrink-0 text-[9.5px] uppercase tracking-wider", m[1])}>{m[0]}</span>;
}

function RunLine({ r, jobs }: { r: RunRef; jobs: any[] }) {
  const run = useRun(r.run_id, true);
  const j = jobs.find((x) => x.id === r.run_id);
  const st = run.data?.status || j?.status || "queued";
  const p = Math.round(((j?.progress ?? run.data?.progress) || 0) * 100);
  return (
    <div className="flex items-center gap-2 text-2xs">
      {st === "done" ? <Check className="size-3 text-up" /> : st === "error" || st === "cancelled" ? <X className="size-3 text-down" /> : <Loader2 className="size-3 animate-spin text-accent" />}
      <span className="min-w-0 flex-1 truncate text-fg-2">{r.label}</span>
      <span className="num text-faint">{st === "done" ? (r.cached ? "cached" : "done") : st === "running" ? `${p}%` : st}</span>
    </div>
  );
}

function LogEntry({ m, onSelect, selected, jobs, onRunMode }: { m: AiMessage; onSelect: (id: number) => void; selected: number | null; jobs: any[]; onRunMode: (v: number, mode: Mode) => void }) {
  const c = m.content || {};
  if (m.role === "user") {
    return <div className="border-l-2 border-accent pl-2 text-sm leading-relaxed text-fg">{c.text}</div>;
  }
  if (m.kind === "plan") {
    const cov: Coverage = c.coverage;
    const blocked = !c.runs?.length && !c.errors?.length;
    return (
      <div className="border border-line bg-bg-0">
        <div className="flex items-center gap-1.5 border-b border-line px-2 py-1">
          <span className="label text-fg-2">{c.parent_id ? `Version ${c.version_id} · change` : `Version ${c.version_id} · interpretation`}</span>
          <span className="flex-1" />
          {c.mode && <ModeBadge mode={c.mode} />}
          <Tip content={`Read by: ${c.interpreter}`}><span className="text-[9.5px] uppercase tracking-wider text-faint">{c.interpreter === "built-in rules" ? "rules" : c.interpreter === "manual edit" ? "edit" : "model"}</span></Tip>
        </div>
        {c.diff?.length > 0 && (
          <div className="border-b border-line px-2 py-1">
            {c.diff.map((d: any) => <div key={d.field} className="flex gap-1.5 text-2xs"><span className="w-20 shrink-0 text-muted">{d.field.replace("study.", "").replace(/_/g, " ")}</span><span className="min-w-0 text-fg-2"><span className="text-down/80 line-through">{String(d.from)}</span> <span className="text-up">{String(d.to)}</span></span></div>)}
          </div>
        )}
        <div className="px-2 py-1.5 text-xs leading-relaxed text-fg-2">{c.text}</div>
        {c.errors?.length > 0 && <ErrorBox className="m-2 mt-0" error={c.errors.join("\n")} />}
        {blocked && cov && (
          <div className="flex flex-wrap gap-1.5 border-t border-line px-2 py-1.5">
            {cov.modes.study?.ok && <Button size="sm" onClick={() => onRunMode(c.version_id, "study")}><Play /> Run the market study</Button>}
            {cov.modes.modeled?.ok && <Tip content="Opt in to theoretical premiums. Results are labelled as estimates everywhere."><Button size="sm" variant="outline" onClick={() => onRunMode(c.version_id, "modeled")}>Estimate with modeled premiums</Button></Tip>}
            <Link to="/data"><Button size="sm" variant="ghost"><Database /> Get option data</Button></Link>
          </div>
        )}
        {c.runs?.length > 0 && <div className="flex flex-col gap-1 border-t border-line px-2 py-1.5">{[...(c.compare_with || []), ...c.runs].map((r: RunRef) => <RunLine key={r.run_id + r.label} r={r} jobs={jobs} />)}</div>}
      </div>
    );
  }
  if (m.kind === "sweep" || m.kind === "run") {
    return (
      <div className="border border-line bg-bg-0">
        <div className="flex items-center gap-1.5 border-b border-line px-2 py-1"><span className="label text-fg-2">{m.kind === "sweep" ? `Variants · ${c.what}` : "Run"}</span><span className="flex-1" />{c.runs?.[0] && <ModeBadge mode={c.runs[0].mode} />}</div>
        <div className="px-2 py-1.5 text-xs leading-relaxed text-fg-2">{c.text}</div>
        <div className="flex flex-col gap-1 border-t border-line px-2 py-1.5">{(c.runs || []).map((r: RunRef) => <RunLine key={r.run_id + r.label} r={r} jobs={jobs} />)}</div>
      </div>
    );
  }
  if (m.kind === "result") {
    const on = selected === m.id;
    return (
      <button onClick={() => onSelect(m.id)} className={cn("block w-full border bg-bg-0 text-left transition-colors", on ? "border-accent/60" : "border-line hover:border-line-2")}>
        <div className="flex items-center gap-1.5 border-b border-line px-2 py-1">
          <CornerDownRight className="size-3 text-accent" />
          <span className="label text-fg-2">{c.view === "sweep" ? "Variant results" : c.view === "compare" ? "Result + comparison" : c.view === "failed" ? "Run failed" : "Result"}</span>
          <span className="flex-1" />
          {c.mode && <ModeBadge mode={c.mode} />}
        </div>
        <div className="px-2 py-1.5 text-xs leading-relaxed text-fg">{c.text}</div>
        {c.compare_text && <div className="border-t border-line px-2 py-1.5 text-xs leading-relaxed text-fg-2">{c.compare_text}</div>}
        {c.explanation_rejected && <div className="border-t border-line px-2 py-1 text-2xs text-warn">{c.explanation_rejected}</div>}
        <div className="flex items-center gap-2 border-t border-line px-2 py-1 text-[9.5px] uppercase tracking-wider text-faint">
          <span>{c.explanation_source && c.explanation_source !== "deterministic" ? `worded by ${c.explanation_source} · every number checked against the engine` : "numbers and wording straight from the engine"}</span>
          <span className="flex-1" /><span className={on ? "text-accent" : ""}>{on ? "shown" : "view"}</span>
        </div>
      </button>
    );
  }
  return <div className={cn("whitespace-pre-wrap text-xs leading-relaxed", m.kind === "error" ? "text-down" : m.kind === "note" ? "text-muted" : "text-fg-2")}>{c.text}</div>;
}

// ---------------------------------------------------------------------------- report
function Report({ msg, results, onSelect, focusRun, onFocus, onAdopt }: { msg: AiMessage; results: AiMessage[]; onSelect: (id: number) => void; focusRun: string | null; onFocus: (id: string) => void; onAdopt: (draft: any) => void }) {
  const c = msg.content;
  const src = useQuery({ queryKey: ["ai-msg-src", c.source_message], queryFn: async () => null, enabled: false });
  void src;
  return (
    <div className="flex min-h-full flex-col">
      {results.length > 1 && (
        <div className="flex h-7 shrink-0 items-center gap-0 overflow-x-auto border-b border-line bg-bg-1 px-1.5">
          {results.map((r, i) => (
            <button key={r.id} onClick={() => onSelect(r.id)} className={cn("h-7 shrink-0 border-b-2 px-2.5 text-2xs -mb-px", r.id === msg.id ? "border-accent text-fg" : "border-transparent text-muted hover:text-fg")}>
              {i + 1} · {r.content.view === "sweep" ? `variants: ${(r.content.what || "").split(" (")[0]}` : r.content.view === "compare" ? `v${r.content.version_id} vs previous` : r.content.view === "failed" ? "failed" : `v${r.content.version_id}`}
            </button>
          ))}
        </div>
      )}
      {c.view === "failed" ? <div className="p-3"><ErrorBox error={c.text} /></div> : null}
      {c.table && <CompareTable c={c} focusRun={focusRun} onFocus={onFocus} onAdopt={onAdopt} />}
      {c.caveats?.length > 0 && c.view === "sweep" && <div className="border-b border-line px-3 py-1.5 text-2xs leading-relaxed text-muted">{c.caveats.map((t: string, i: number) => <div key={i}>• {t}</div>)}</div>}
      {focusRun && c.view !== "failed" && <RunReport runId={focusRun} />}
    </div>
  );
}

function CompareTable({ c, focusRun, onFocus, onAdopt }: { c: any; focusRun: string | null; onFocus: (id: string) => void; onAdopt: (draft: any) => void }) {
  const t = c.table;
  const key = t.kind === "study" ? "probability" : "net_pnl";
  const chart = useMemo(() => ({
    grid: { left: 54, right: 14, top: 14, bottom: 46 },
    tooltip: { trigger: "axis", formatter: (ps: any[]) => { const r = t.rows[ps[0].dataIndex]; return `${r.label}<br/>${t.kind === "study" ? `frequency ${r.shown.probability} (CI ${r.shown.probability_ci_low} – ${r.shown.probability_ci_high})<br/>${r.shown.qualifying_sessions} sessions` : `net ${r.shown.net_pnl}<br/>${r.shown.trades} trades, win rate ${r.shown.win_rate}`}`; } },
    xAxis: { type: "category", data: t.rows.map((r: any) => r.label.replace(/^(previous-day move threshold|underlying target|underlying stop|days to expiration|entry time|premium profit target|premium stop|holding period) /, "")), axisLabel: { fontSize: 9.5, interval: 0, rotate: t.rows.length > 6 ? 30 : 0, width: 90, overflow: "truncate" } },
    yAxis: { type: "value", max: t.kind === "study" ? 1 : undefined, axisLabel: { formatter: (v: number) => (t.kind === "study" ? `${Math.round(v * 100)}%` : `${v >= 1000 || v <= -1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)}`) } },
    series: [
      { type: "bar", barMaxWidth: 34, data: t.rows.map((r: any) => ({ value: r.values[key], itemStyle: { color: t.kind === "study" ? ((r.values.qualifying_sessions || 0) < 30 ? "#3a4353" : C.accent) : (r.values[key] >= 0 ? C.up : C.down) } })) },
      ...(t.kind === "study" ? [{ type: "custom", z: 5, data: t.rows.map((r: any, i: number) => [i, r.values.probability_ci_low, r.values.probability_ci_high]), renderItem: (_: any, a: any) => {
        const lo = a.coord([a.value(0), a.value(1)]), hi = a.coord([a.value(0), a.value(2)]);
        return { type: "group", children: [{ type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: { stroke: C.fg, lineWidth: 1 } }, { type: "line", shape: { x1: lo[0] - 4, y1: lo[1], x2: lo[0] + 4, y2: lo[1] }, style: { stroke: C.fg } }, { type: "line", shape: { x1: hi[0] - 4, y1: hi[1], x2: hi[0] + 4, y2: hi[1] }, style: { stroke: C.fg } }] };
      } }] : []),
    ],
  }), [t, key]);
  return (
    <div className="border-b border-line">
      <div className="flex items-center gap-2 border-b border-line bg-bg-1 px-3 py-1.5 text-2xs text-muted">
        <span className="label text-fg-2">{c.view === "sweep" ? `Variants of ${c.what}` : "Before / after"}</span>
        <span>{t.kind === "study" ? "bars = historical frequency with 95% confidence interval; grey = fewer than 30 sessions" : "bars = net P&L after costs"}</span>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-px bg-line">
        <div className="min-w-0 bg-bg-0"><EChart option={chart} height={Math.max(170, 60 + t.rows.length * 22)} /></div>
        <div className="min-w-0 overflow-auto bg-bg-0">
          <table className="tbl">
            <thead><tr><th>Variant</th>{t.columns.slice(0, t.kind === "study" ? 5 : 6).map((col: any) => <th key={col.key} className="r">{col.label}</th>)}<th /></tr></thead>
            <tbody>
              {t.rows.map((r: any) => {
                const ref = [...(c.runs || [])].find((x: any) => x.run_id === r.run_id);
                return (
                  <tr key={r.run_id + r.label} onClick={() => onFocus(r.run_id)} className={cn("cursor-pointer", focusRun === r.run_id && "[&>td]:bg-accent/[0.08]")}>
                    <td className="max-w-56 truncate text-fg">{r.label}</td>
                    {t.columns.slice(0, t.kind === "study" ? 5 : 6).map((col: any) => <td key={col.key} className={cn("r num", col.key === key && "text-fg", col.unit === "usd" && (r.values[col.key] > 0 ? "text-up" : r.values[col.key] < 0 ? "text-down" : ""))}>{r.shown[col.key]}</td>)}
                    <td className="r">{ref?.draft && <Tip content="Make this variant the current version"><button className="text-2xs text-accent hover:underline" onClick={(e) => { e.stopPropagation(); onAdopt(ref.draft); }}>use</button></Tip>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RunReport({ runId }: { runId: string }) {
  const run = useRun(runId, true);
  if (run.isLoading) return <Loading />;
  if (run.error) return <ErrorBox className="m-2" error={run.error} />;
  const r = run.data!;
  if (r.status === "error") return <ErrorBox className="m-2" error={r.error?.split("\n")[0] || "run failed"} />;
  if (r.status !== "done" || !r.result) return <Loading label={r.message || "running"} />;
  return r.kind === "study" ? <StudyReport res={r.result} showTable={false} /> : <BacktestReport run={r} />;
}

// ---------------------------------------------------------------------------- bottom log
function BottomLog({ runId, filter, setFilter }: { runId: string; filter: "all" | "win" | "loss"; setFilter: (f: "all" | "win" | "loss") => void }) {
  const run = useRun(runId, true);
  const [pick, setPick] = useState<any>(null);
  useEffect(() => setPick(null), [runId]);
  if (!run.data || run.data.status !== "done" || !run.data.result) return <div className="h-full bg-bg-1" />;
  const r = run.data;
  if (r.kind === "study") {
    const res = r.result;
    const row = pick || [...(res.sessions || [])].reverse().find((x: any) => filter === "all" || (filter === "win" ? x.outcome === "target" : x.outcome !== "target"));
    return (
      <div className="flex h-full flex-col bg-bg-0">
        <div className="flex h-7 shrink-0 items-center gap-2 border-b border-line bg-bg-1 px-2.5">
          <span className="label text-fg-2">Sessions</span>
          <Segmented value={filter} onChange={setFilter} options={[{ value: "all", label: `All ${res.sessions?.length ?? 0}` }, { value: "win", label: "Hit" }, { value: "loss", label: "Missed" }]} />
          <span className="text-2xs text-faint">click a session to replay it</span>
        </div>
        <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-px bg-line">
          <div className="min-h-0 min-w-0 bg-bg-0"><SessionsTable res={res} filter={filter} onPick={setPick} selected={row?.date} /></div>
          <div className="min-h-0 min-w-0 bg-bg-0">{row ? <SessionReplay symbol={res.symbol} row={row} height={170} /> : <Empty title="No sessions" />}</div>
        </div>
      </div>
    );
  }
  return <TradeLog runId={runId} filter={filter} setFilter={setFilter} modeled={r.result.meta?.pricing === "modeled"} />;
}

function TradeLog({ runId, filter, setFilter, modeled }: { runId: string; filter: "all" | "win" | "loss"; setFilter: (f: "all" | "win" | "loss") => void; modeled: boolean }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["trades", runId], queryFn: () => api.get<(Trade & { signal_ts?: number; fill_delay_seconds?: number; exit_delay_seconds?: number })[]>(`/api/backtests/${runId}/trades`) });
  const rows = (q.data || []).filter((t) => filter === "all" || (filter === "win" ? t.pnl > 0 : t.pnl <= 0));
  const delay = (s?: number) => (!s ? <span className="text-faint">same tick</span> : <span className="text-warn">+{s >= 3600 ? `${(s / 3600).toFixed(1)}h` : `${Math.round(s / 60)}m`}</span>);
  const cols: Col<any>[] = [
    { key: "id", label: "#", render: (t) => <span className="num text-faint">{t.id}</span>, sort: (t) => t.id },
    { key: "sig", label: "Signal (ET)", render: (t) => <span className="num text-fg-2">{etTime(t.signal_ts ?? t.entry_ts)}</span>, sort: (t) => t.signal_ts ?? t.entry_ts },
    { key: "fill", label: "Fill", render: (t) => <span className="num">{etTime(t.entry_ts, false)} {delay(t.fill_delay_seconds)}</span>, title: "When the order actually filled, and how long after the signal" },
    { key: "exit", label: "Exit", render: (t) => <span className="num text-fg-2">{etTime(t.exit_ts)} {t.exit_delay_seconds ? delay(t.exit_delay_seconds) : null}</span>, sort: (t) => t.exit_ts },
    { key: "legs", label: "Contracts", render: (t) => <span className="num text-fg-2">{t.legs.map((l: any) => (l.kind === "stock" ? `${l.ratio}sh` : `${l.ratio > 0 ? "+" : ""}${l.ratio} ${l.strike}${l.right} ${l.expiration?.slice(5)}`)).join("  ")}</span> },
    { key: "px", label: "Entry → exit px", align: "r", render: (t) => <span className="num text-fg-2">{t.legs.length === 1 ? `${num(t.legs[0].entry_price)} → ${num(t.legs[0].exit_price)}` : money(t.entry_cash, 0, true)}</span> },
    { key: "spot", label: "Underlying", align: "r", render: (t) => <span className="num text-muted">{num(t.entry_spot)} → {num(t.exit_spot)}</span> },
    { key: "pnl", label: "P&L", align: "r", render: (t) => <Pnl v={t.pnl} fmt={(x) => money(x, 2, true)} />, sort: (t) => t.pnl },
    { key: "hold", label: "Hold", align: "r", render: (t) => <span className="num text-fg-2">{minutesLabel(t.holding_minutes)}</span>, sort: (t) => t.holding_minutes },
    { key: "why", label: "Exit reason", render: (t) => <span className="text-fg-2">{REASON_LABEL[t.exit_reason] || t.exit_reason}</span>, sort: (t) => t.exit_reason },
    { key: "flags", label: "Flags", render: (t) => <span className="text-2xs text-warn">{(t.flags || []).join(" ")}</span> },
  ];
  return (
    <div className="flex h-full flex-col bg-bg-0">
      <div className="flex h-7 shrink-0 items-center gap-2 border-b border-line bg-bg-1 px-2.5">
        <span className="label text-fg-2">Trade log</span>
        <Segmented value={filter} onChange={setFilter} options={[{ value: "all", label: `All ${q.data?.length ?? 0}` }, { value: "win", label: "Winners" }, { value: "loss", label: "Losers" }]} />
        {modeled && <span className="text-2xs text-warn">prices are model estimates</span>}
        <span className="text-2xs text-faint">signal and fill times are recorded separately · click a trade for the full replay</span>
        <span className="flex-1" />
        <a href={`/api/backtests/${runId}/trades.csv`} className="text-2xs text-muted hover:text-fg">CSV</a>
      </div>
      {q.isLoading ? <Loading /> : <DataTable rows={rows} cols={cols} rowKey={(t) => t.id} onRowClick={(t) => nav(`/replay/${runId}/${t.id}`)} className="flex-1" />}
    </div>
  );
}

// ---------------------------------------------------------------------------- inspector
function Inspector({ data, plan, onRunMode, onEdit, onRestore, busy }: { data: AiPayload; plan: AiMessage | null; onRunMode: (v: number, m: Mode) => void; onEdit: (d: any) => void; onRestore: (v: number) => void; busy: boolean }) {
  const [editing, setEditing] = useState(false);
  const cur = data.versions.find((v) => v.id === data.current_version) || null;
  const shown = plan && plan.content.version_id === data.current_version ? plan : [...data.messages].reverse().find((m) => m.kind === "plan" && m.content.version_id === data.current_version) || plan;
  const c = shown?.content;
  useEffect(() => setEditing(false), [data.current_version]);
  if (!c || !cur) return <Empty title="Nothing interpreted yet">The interpretation of your request, every assumed default and the data it needs appear here.</Empty>;
  const cov: Coverage = c.coverage;
  return (
    <div className="flex h-full flex-col overflow-auto">
      <Panel className="border-0 border-b" dense title={`Interpretation · version ${cur.id}`} right={<Button size="sm" variant={editing ? "default" : "ghost"} onClick={() => setEditing((v) => !v)}><Pencil /> {editing ? "Close" : "Edit"}</Button>}>
        {editing ? <DraftEditor draft={cur.draft} busy={busy} onApply={(d) => { onEdit(d); setEditing(false); }} /> : (
          <div className="flex flex-col">
            {(c.items as PlanItem[]).map((it, i) => (
              <div key={i} className="flex items-start gap-2 border-b border-line/60 px-2.5 py-1 last:border-0">
                <span className="w-[74px] shrink-0 pt-px text-2xs text-muted">{it.label}</span>
                <span className={cn("min-w-0 flex-1 text-xs leading-snug", it.status === "assumed" ? "text-warn/90" : "text-fg")}>{it.value}</span>
                <StatusPill s={it.status} />
              </div>
            ))}
          </div>
        )}
      </Panel>
      {c.assumptions?.length > 0 && !editing && (
        <Panel className="border-0 border-b" dense title={`Defaults I filled in (${c.assumptions.length})`}>
          <div className="flex flex-col gap-1 px-2.5 py-1.5">
            {c.assumptions.map((a: any, i: number) => <div key={i} className="text-2xs leading-relaxed text-fg-2"><span className="text-warn">{String(a.field).replace(/_/g, " ")}{a.value !== null && a.value !== undefined ? ` = ${a.value}` : ""}</span> - {a.reason}</div>)}
            <div className="text-2xs text-faint">Nothing here was stated in your request. Use Edit, or just tell me what to change.</div>
          </div>
        </Panel>
      )}
      <Panel className="border-0 border-b" dense title={`What the stored ${cov.symbol || ""} data supports`}>
        <div className="flex flex-col">
          {(["historical", "study", "modeled"] as Mode[]).map((m) => {
            const info = cov.modes[m];
            if (!info) return null;
            const ran = cur.runs.some((r) => r.mode === m);
            return (
              <div key={m} className="border-b border-line/60 px-2.5 py-1.5 last:border-0">
                <div className="flex items-center gap-1.5">
                  {info.ok ? <Check className="size-3 text-up" /> : <CircleDashed className="size-3 text-faint" />}
                  <span className={cn("text-xs", info.ok ? "text-fg" : "text-muted")}>{MODE_LABEL[m]}</span>
                  {cov.recommended === m && <span className="text-[9.5px] uppercase tracking-wider text-accent">recommended</span>}
                  <span className="flex-1" />
                  {info.ok && <Button size="sm" variant="ghost" disabled={busy} onClick={() => onRunMode(cur.id, m)}><Play /> {ran ? "Re-run" : m === "modeled" ? "Opt in" : "Run"}</Button>}
                </div>
                <div className="mt-0.5 pl-[18px] text-2xs leading-relaxed text-muted">{info.ok ? <>{info.window ? <span className="num text-fg-2">{info.window[0]} → {info.window[1]} · {info.resolution} · </span> : null}{info.note}</> : info.reason}</div>
              </div>
            );
          })}
        </div>
        {!cov.modes.historical?.ok && c.kind === "options" && <div className="border-t border-line px-2.5 py-1.5"><Link to="/data" className="text-2xs text-accent hover:underline">Find option data for {cov.symbol} →</Link></div>}
      </Panel>
      <Panel className="border-0" dense title={<span className="flex items-center gap-1.5"><History className="size-3" /> Versions</span>}>
        <div className="flex flex-col">
          {[...data.versions].reverse().map((v) => <VersionRow key={v.id} v={v} current={v.id === data.current_version} onRestore={onRestore} depth={depthOf(v, data.versions)} />)}
        </div>
      </Panel>
    </div>
  );
}

function depthOf(v: AiVersion, all: AiVersion[]): number {
  let d = 0, p = v.parent_id;
  while (p && d < 6) { d++; p = all.find((x) => x.id === p)?.parent_id ?? null; }
  return d;
}

function VersionRow({ v, current, onRestore, depth }: { v: AiVersion; current: boolean; onRestore: (id: number) => void; depth: number }) {
  return (
    <div className={cn("group flex items-center gap-1.5 border-b border-line/60 px-2.5 py-1 last:border-0", current && "bg-accent/[0.06]")}>
      <span style={{ width: depth * 8 }} className="shrink-0" />
      {v.parent_id ? <GitBranch className="size-3 shrink-0 text-faint" /> : <span className="size-3 shrink-0" />}
      <span className="num w-6 shrink-0 text-2xs text-faint">v{v.id}</span>
      <span className={cn("min-w-0 flex-1 truncate text-xs", current ? "text-fg" : "text-fg-2")}>{v.label}</span>
      <span className="num text-[9.5px] text-faint">{v.runs.length ? `${v.runs.length} run${v.runs.length > 1 ? "s" : ""}` : "not run"}</span>
      {current ? <span className="text-[9.5px] uppercase tracking-wider text-accent">current</span>
        : <Tip content="Go back to this version. Later versions are kept; new changes branch from here."><button onClick={() => onRestore(v.id)} className="text-faint opacity-0 hover:text-fg group-hover:opacity-100"><RotateCcw className="size-3" /></button></Tip>}
    </div>
  );
}

const STRUCTURES = ["long_call", "long_put", "short_put", "short_call", "bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread", "iron_condor", "iron_butterfly", "long_straddle", "short_straddle", "long_strangle", "short_strangle", "covered_call"];

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="flex items-center gap-2"><span className="w-[92px] shrink-0 text-2xs text-muted">{label}</span><div className="min-w-0 flex-1">{children}</div></div>;
}

/** Direct edits to the structured request. Times here are New York time. */
function DraftEditor({ draft, onApply, busy }: { draft: any; onApply: (d: any) => void; busy: boolean }) {
  const [d, setD] = useState<any>(() => JSON.parse(JSON.stringify(draft)));
  const set = (p: any) => setD((x: any) => ({ ...x, ...p }));
  const st = d.study || { target: null, stop: null, reference: "entry", measure: "touch", horizon_kind: "session", horizon_time: "12:00", horizon_sessions: 1 };
  const setSt = (p: any) => set({ study: { ...st, ...p } });
  const [bad, setBad] = useState<Record<number, string>>({});
  const check = async (i: number, expr: string) => {
    const r = await api.post<any>("/api/expr/validate", { expr });
    setBad((b) => ({ ...b, [i]: r.valid ? "" : r.errors.join("; ") }));
  };
  const options = d.kind === "options";
  return (
    <div className="flex flex-col gap-1.5 p-2.5">
      <Row label="Symbol"><Input sz="sm" className="num uppercase" value={d.symbol} onChange={(e) => set({ symbol: e.target.value.toUpperCase(), symbol_stated: true })} /></Row>
      <Row label={options ? "Entry time (ET)" : "Start (ET)"}><Input sz="sm" className="num" placeholder="open, close or HH:MM - blank = on signal" value={d.entry_time || ""} onChange={(e) => set({ entry_time: e.target.value.trim() || null, entry_time_stated: null })} /></Row>
      <div className="label mt-1">Conditions</div>
      {(d.conditions || []).map((cnd: any, i: number) => (
        <div key={i} className="flex flex-col gap-0.5">
          <div className="flex items-center gap-1">
            <Input sz="sm" className="font-mono text-2xs" value={cnd.expr} onChange={(e) => set({ conditions: d.conditions.map((x: any, j: number) => (j === i ? { ...x, expr: e.target.value, text: "", source: "stated" } : x)) })} onBlur={(e) => check(i, e.target.value)} />
            <Button size="iconSm" variant="ghost" onClick={() => set({ conditions: d.conditions.filter((_: any, j: number) => j !== i) })}><X /></Button>
          </div>
          {bad[i] ? <span className="text-2xs text-down">{bad[i]}</span> : cnd.text ? <span className="text-2xs text-faint">{cnd.text}</span> : null}
        </div>
      ))}
      <Button size="sm" variant="ghost" className="self-start" onClick={() => set({ conditions: [...(d.conditions || []), { expr: "previous_session.return_pct <= -1", text: "", source: "stated" }] })}><Plus /> Condition</Button>
      {options ? <>
        <div className="label mt-1">Position</div>
        <Row label="Structure"><Select sz="sm" value={d.structure || "long_call"} onChange={(v) => set({ structure: v })} options={STRUCTURES.map((s) => ({ value: s, label: s.replace(/_/g, " ") }))} /></Row>
        <Row label="Days to expiry"><NumInput sz="sm" allowNull min={0} value={d.dte} onChange={(v) => set({ dte: v })} placeholder="default" /></Row>
        <Row label="Strike"><div className="flex gap-1"><Select sz="sm" className="w-28" value={d.strike?.method || "atm"} onChange={(v) => set({ strike: { method: v, value: v === "delta" ? 0.3 : v === "pct_otm" ? 1 : 0 } })} options={[{ value: "atm", label: "at the money" }, { value: "delta", label: "delta" }, { value: "pct_otm", label: "% OTM" }]} />
          {d.strike && d.strike.method !== "atm" && <NumInput sz="sm" step={d.strike.method === "delta" ? 0.05 : 0.5} value={d.strike.value} onChange={(v) => set({ strike: { ...d.strike, value: v ?? 0 } })} />}</div></Row>
        <Row label="Quantity"><NumInput sz="sm" min={1} value={d.quantity} onChange={(v) => set({ quantity: v ?? 1 })} /></Row>
        <div className="label mt-1">Exits</div>
        <Row label="Underlying +%"><NumInput sz="sm" allowNull step={0.25} suffix="%" value={d.underlying_target_pct} onChange={(v) => set({ underlying_target_pct: v })} placeholder="none" /></Row>
        <Row label="Underlying −%"><NumInput sz="sm" allowNull step={0.25} suffix="%" value={d.underlying_stop_pct} onChange={(v) => set({ underlying_stop_pct: v })} placeholder="none" /></Row>
        <Row label="Premium gain"><NumInput sz="sm" allowNull step={10} suffix="%" value={d.premium_target_pct} onChange={(v) => set({ premium_target_pct: v })} placeholder="none" /></Row>
        <Row label="Premium loss"><NumInput sz="sm" allowNull step={10} suffix="%" value={d.premium_stop_pct} onChange={(v) => set({ premium_stop_pct: v })} placeholder="none" /></Row>
        <Row label="Exit time (ET)"><Input sz="sm" className="num" placeholder="HH:MM or close" value={d.exit_time || ""} onChange={(e) => set({ exit_time: e.target.value.trim() || null, exit_time_stated: null })} /></Row>
        <Row label="Hold sessions"><NumInput sz="sm" allowNull min={1} value={d.hold_sessions} onChange={(v) => set({ hold_sessions: v })} placeholder="none" /></Row>
      </> : <>
        <div className="label mt-1">Outcome</div>
        <Row label="Target"><NumInput sz="sm" allowNull step={0.25} suffix="%" value={st.target?.kind === "pct" ? st.target.value : null} onChange={(v) => setSt({ target: v === null ? null : { kind: "pct", value: v, expr: null } })} placeholder={st.target?.kind === "expr" ? st.target.expr : "none"} /></Row>
        <Row label="Stop"><NumInput sz="sm" allowNull step={0.25} suffix="%" value={st.stop?.kind === "pct" ? st.stop.value : null} onChange={(v) => setSt({ stop: v === null ? null : { kind: "pct", value: v, expr: null } })} placeholder={st.stop?.kind === "expr" ? st.stop.expr : "none"} /></Row>
        <Row label="Counts when"><Select sz="sm" value={st.measure || "touch"} onChange={(v) => setSt({ measure: v })} options={[{ value: "touch", label: "price touches the level" }, { value: "close", label: "the horizon ends beyond it" }]} /></Row>
        <Row label="Measured from"><Select sz="sm" value={st.reference || "entry"} onChange={(v) => setSt({ reference: v })} options={[{ value: "entry", label: "the starting price" }, { value: "open", label: "the session open" }, { value: "prev_close", label: "the previous close" }]} /></Row>
        <Row label="Horizon"><div className="flex gap-1"><Select sz="sm" className="w-32" value={st.horizon_kind || "session"} onChange={(v) => setSt({ horizon_kind: v })} options={[{ value: "session", label: "that session" }, { value: "time", label: "until a time" }, { value: "sessions", label: "N sessions" }]} />
          {st.horizon_kind === "time" && <Input sz="sm" className="num" value={st.horizon_time} onChange={(e) => setSt({ horizon_time: e.target.value })} />}
          {st.horizon_kind === "sessions" && <NumInput sz="sm" min={1} value={st.horizon_sessions} onChange={(v) => setSt({ horizon_sessions: v ?? 1 })} />}</div></Row>
      </>}
      <div className="label mt-1">Period</div>
      <div className="flex gap-1"><Input sz="sm" type="date" className="num" value={d.start || ""} onChange={(e) => set({ start: e.target.value || null })} /><Input sz="sm" type="date" className="num" value={d.end || ""} onChange={(e) => set({ end: e.target.value || null })} /></div>
      <div className="text-2xs text-faint">Blank dates use all available history. Times are New York time.</div>
      <Button variant="primary" className="mt-1" disabled={busy || Object.values(bad).some(Boolean)} onClick={() => onApply(d)}><Play /> Apply as a new version and run</Button>
    </div>
  );
}
