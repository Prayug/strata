/* One research notebook: the conversation on the left, and on the right whatever it produced:
   the report of a result, the editable interpretation, a set of variants, or the pinned comparison. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, FileText, GitCompareArrows, History, Keyboard, Network, PanelRightClose, PanelRightOpen, Pencil, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Kbd, Tip } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { cn } from "@/lib/format";
import { MODE_NAME, useAiMeta, type AiMessage, type AiPayload } from "@/lib/lab";
import { Empty, Skeleton, useHotkeys } from "./bits";
import { Canvas } from "./Canvas";
import { Compare } from "./Compare";
import { CostView, DataCheckView, FillsView, StageRail, UntestedView, type Alt } from "./Coverage";
import { Composer, InterpreterChip, type ComposerHandle } from "./Landing";
import { Message, type Actions, type OpenTarget } from "./Messages";
import { Timeline } from "./Pipeline";
import { Report, REPORT_TABS } from "./Report";
import { GraphView, MemoryView, RegistryCheckView, StrategyPlanView } from "./Graph";
import { AnalysisView, ChallengeView, ExecutionView, HoldoutChip, HypothesisView } from "./Research";
import { Setup } from "./Setup";
import { DiscoveryView, InvestigationView, SweepView } from "./Views";

type View = OpenTarget | { type: "compare" } | { type: "canvas" } | null;
const PENDING = ["plan", "sweep", "run", "investigation", "discovery", "hypothesis", "stress", "challenge", "graph"];
const MSG_VIEWS = ["sweep", "investigation", "discovery", "hypothesis", "execution", "challenge", "analysis", "graph", "graph_batch", "fills", "untested", "cost"];
const MSG_KINDS = ["memory", "registry", "strategy_plan"];  // answers that open on the right without being a run result
const isPending = (m: AiMessage) => PENDING.includes(m.kind) && m.content?.runs?.length > 0 && !m.content.finalized;
const TAB_OF: Record<string, string> = { robustness: "robustness", skipped: "observations", breakdown: "statistics", timing: "statistics" };

/** Where a freshly arrived assistant message should take the workspace. */
function targetOf(m: AiMessage): { view: View; tab?: string; which?: string } | null {
  const c = m.content || {};
  if (m.kind === "result") {
    if (MSG_VIEWS.includes(c.view)) return { view: { type: "message", messageId: m.id } };
    if (c.view === "failed" || c.view === "export" || !c.run_id) return null;
    return { view: { type: "run", runId: c.run_id, messageId: m.id }, tab: TAB_OF[c.view] || "overview", which: c.view === "skipped" ? "skipped" : undefined };
  }
  if (MSG_KINDS.includes(m.kind)) return { view: { type: "message", messageId: m.id } };
  if (m.kind === "trades" && c.run_id) return { view: { type: "run", runId: c.run_id }, tab: "observations", which: c.filter === "all" ? "all" : c.mode === "study" ? (c.filter === "win" ? "target" : "neither") : c.filter };
  if (m.kind === "plan" && ["awaiting_confirmation", "awaiting_answer", "blocked"].includes(c.state)) return { view: { type: "setup", versionId: c.version_id } };
  return null;
}

const strip = (l: string) => String(l || "").replace(/^v\d+\s*-\s*/, "");
/** A result is named after the question that produced it. */
function resultName(m: AiMessage, all: AiMessage[]) {
  const c = m.content;
  const kind = { sweep: `Comparison${c.what ? `: ${c.what}` : ""}`, investigation: `${c.plan?.stage === "options" ? "Option structures" : "Investigation"}: ${c.plan?.question || c.plan?.hypothesis || ""}`,
    discovery: `Exploration: ${c.discovery?.hypothesis || c.discovery?.question || ""}`, hypothesis: `Plan: ${c.question || ""}`, execution: `Execution stress: ${strip(c.label || "").replace(/^Execution stress: /, "")}`,
    challenge: `Review: ${strip(c.label || "").replace(/^Challenge: /, "")}`, fills: "Every fill and its source", untested: "Sessions that could not be tested", cost: "What the missing data would cost", analysis: `Answer: ${c.analysis?.title || ""}`, graph: `Graph: ${c.question || c.label || ""}`, graph_batch: `Re-test: ${c.label || ""}` }[c.view as string];
  if (kind) return kind.slice(0, 80);
  const asked = [...all].reverse().find((x) => x.role === "user" && x.id < m.id)?.content?.text;
  return `${String(asked || strip(c.label || c.title) || "Result").slice(0, 64)}${c.mode && c.mode !== "study" ? ` (${(MODE_NAME[c.mode] || c.mode).replace("Options · ", "")})` : ""}`;
}

const STUDY_NEXT = ["Challenge this result", "How much of the result is explained by volatility?", "Test option structures on this", "Compare that with 2% and 4% down days", "What time of day does it usually hit the target?", "Show every qualifying historical session", "Test ATM 0DTE calls on those days using historical option trade prices", "Is this statistically significant?", "Test whether the results hold up out of sample", "Export the complete research report"];
const GRAPH_NEXT = ["Compare that behavior with QQQ and IWM.", "Show the distribution of price paths following those events.", "Find which historical conditions are associated with that outcome.", "Build a strategy from the strongest supported finding.", "Check whether I've already tested similar configurations on this holdout.", "What have we found so far?"];
const OPTION_NEXT = ["Show exactly which historical sessions could not be tested", "Show every actual option fill and its source", "Estimate what it would cost to acquire the missing observations", "Challenge this strategy", "Stress-test the execution", "Are most profits concentrated in a handful of sessions?", "Compare the results with 7DTE calls", "Include commissions and conservative slippage", "Show me every skipped signal and why", "Show me the losing trades", "Test whether the results hold up out of sample", "Export the complete research report"];

export function Workspace({ sid }: { sid: string }) {
  const nav = useNavigate();
  const meta = useAiMeta();
  const q = useQuery({ queryKey: ["ai", sid], queryFn: () => api.get<AiPayload>(`/api/ai/sessions/${sid}`) });
  const [view, setView] = useState<View>(null);
  const [tab, setTab] = useState("overview");
  const [which, setWhich] = useState("all");
  const [day, setDay] = useState<string | null>(null);
  const [showRight, setShowRight] = useState(true);
  const [help, setHelp] = useState(false);
  const [wide, setWide] = useState(false);
  const comp = useRef<ComposerHandle>(null);
  const end = useRef<HTMLDivElement>(null);
  const data = q.data;
  const messages = data?.messages || [];
  const pending = messages.some(isPending);

  // while someone is working on the canvas, new results appear there as nodes instead of taking the workspace away
  const onCanvas = useRef(false);
  const go = (t: { view: View; tab?: string; which?: string } | null) => {
    if (!t) return;
    onCanvas.current = t.view?.type === "canvas";
    setView(t.view);
    if (t.tab) setTab(t.tab);
    if (t.which) setWhich(t.which);
    setDay(null);
    setShowRight(true);
  };
  const arrived = (ms: AiMessage[] | undefined) => {
    if (onCanvas.current) return;
    const hit = [...(ms || [])].reverse().map(targetOf).find(Boolean);
    if (hit) go(hit);
  };
  const act = useMutation({
    mutationFn: (fn: () => Promise<any>) => fn(),
    onSuccess: (r: any) => { queryClient.invalidateQueries({ queryKey: ["ai", sid] }); queryClient.invalidateQueries({ queryKey: ["notebooks"] }); arrived(r?.messages); },
    onError: (e: any) => toast.error(e.message),
  });
  const send = (text: string) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/messages`, { text }));

  // the first question typed on the landing page is sent here, once
  useEffect(() => {
    const k = `lab-first-${sid}`;
    const first = sessionStorage.getItem(k);
    if (first) {
      sessionStorage.removeItem(k);
      send(first);
    }
    setView(null);
    setTab("overview");
    setDay(null);
  }, [sid]); // eslint-disable-line

  // finished runs become result messages on the server; poll while anything is in flight
  useEffect(() => {
    if (!pending) return;
    let stop = false;
    const tick = async () => {
      try {
        const r = await api.post<{ messages: AiMessage[] }>(`/api/ai/sessions/${sid}/finalize`);
        if (!stop && r.messages.length) {
          queryClient.invalidateQueries({ queryKey: ["ai", sid] });
          queryClient.invalidateQueries({ queryKey: ["runs"] });
          queryClient.invalidateQueries({ queryKey: ["coverage"] });
          arrived(r.messages);
        }
      } catch { /* retried on the next tick */ }
    };
    const h = setInterval(tick, 1100);
    tick();
    return () => { stop = true; clearInterval(h); };
  }, [pending, sid]); // eslint-disable-line

  useEffect(() => { end.current?.scrollIntoView({ block: "end", behavior: "smooth" }); }, [messages.length, act.isPending]);

  // every result this notebook holds: single reports, sets of variants, investigations, explorations
  const results = useMemo(() => messages.filter((m) => m.kind === "result" && ["single", "compare", ...MSG_VIEWS].includes(m.content?.view) && targetOf(m)), [messages]);
  const plans = useMemo(() => messages.filter((m) => m.kind === "plan"), [messages]);
  const lastPlan = plans[plans.length - 1];
  // what the right-hand side shows when nothing was picked explicitly: the newest result, else the interpretation in progress
  const eff: View = useMemo(() => {
    if (view) return view;
    const lastRes = [...messages].reverse().find((m) => m.kind === "result" && targetOf(m));
    if (lastRes && (!lastPlan || lastRes.id > lastPlan.id || lastPlan.content.finalized)) return targetOf(lastRes)!.view;
    if (lastPlan) return { type: "setup", versionId: lastPlan.content.version_id };
    return null;
  }, [view, messages, lastPlan]);
  const pinned = data?.pinned || [];
  const runNames = useMemo(() => Object.fromEntries(results.filter((m) => m.content.run_id && !["fills", "untested", "cost"].includes(m.content.view)).map((m) => [m.content.run_id, resultName(m, messages)])), [results, messages]);
  const pin = (runId: string, on?: boolean) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/pin`, { run_id: runId, on: on ?? !pinned.includes(runId) }));
  const actions: Actions = {
    answer: (version_id, question_id, option) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/answer`, { version_id, question_id, option })),
    confirm: (version_id) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/confirm`, { version_id })),
    decide: (message_id, choice) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/decide`, { message_id, choice })),
    open: (t) => go({ view: t, tab: t.type === "run" ? t.tab || "overview" : undefined, which: t.type === "run" ? t.which : undefined }),
    pin: (id) => pin(id),
    refresh: () => queryClient.invalidateQueries({ queryKey: ["ai", sid] }),
    rerun: (version_id, mode) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/run`, { version_id, mode })),
    select: (run_id, key, what) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/hypothesis/select`, { run_id, key, what })),
    analyse: (kind, run_id) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/analysis`, { kind, run_id })),
    graphDecide: (message_id, choice) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/graph/decide`, { message_id, choice })),
    ask: (text) => send(text),
    strategyFrom: (finding_id) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/strategy-from-finding`, { finding_id: finding_id ?? null })),
    backtestPlan: () => act.mutate(() => api.post(`/api/ai/sessions/${sid}/backtest-plan`, {})),
    pickAlt: (version_id, alt: Alt) => {
      // each alternative is an explicit choice by the person; none of them alters the strategy that was asked for
      const mode = ({ study: "study", subset: "auto", modeled: "modeled" } as Record<string, string>)[alt.id];
      if (mode) act.mutate(() => api.post(`/api/ai/sessions/${sid}/run`, { version_id, mode }));
      else nav(alt.id === "connect" ? "/data?tab=readiness" : "/data?tab=bulk");
    },
    busy: act.isPending,
    pinned,
  };
  const setMode = (m: "quick" | "review") => act.mutate(() => api.post(`/api/ai/sessions/${sid}/mode`, { exec_mode: m }));

  const runView = eff?.type === "run" ? eff : null;
  const curId = eff?.type === "message" ? eff.messageId : runView ? ((results.some((m) => m.id === runView.messageId) ? runView.messageId : undefined) ?? [...results].reverse().find((m) => m.content.run_id === runView.runId)?.id ?? null) : null;
  const stepResult = (dir: 1 | -1) => {
    if (!results.length) return;
    const i = results.findIndex((m) => m.id === curId);
    const m = results[(i + dir + results.length) % results.length];
    go(targetOf(m));
  };
  useHotkeys({
    "/": (e) => { e.preventDefault(); comp.current?.focus(); },
    ...Object.fromEntries(REPORT_TABS.map(([k], i) => [String(i + 1), () => { if (runView) setTab(k); }])),
    c: () => go({ view: { type: "compare" } }),
    v: () => go({ view: { type: "canvas" } }),
    e: () => lastPlan && go({ view: { type: "setup", versionId: lastPlan.content.version_id } }),
    "[": () => stepResult(-1),
    "]": () => stepResult(1),
    "?": () => setHelp((h) => !h),
    Escape: () => { setHelp(false); (document.activeElement as HTMLElement)?.blur?.(); },
  }, [runView?.runId, results.length, lastPlan?.id]);

  if (q.isLoading) return <div className="lab flex h-full flex-col gap-3 p-6"><Skeleton className="h-8 w-1/3" /><Skeleton className="h-40" /></div>;
  if (q.isError || !data) return <div className="lab h-full"><Empty title="This notebook could not be opened">{String((q.error as any)?.message || "")}<div className="mt-3"><button className="btn sm" onClick={() => nav("/")}>Back to the lab</button></div></Empty></div>;

  const lastRes = results[results.length - 1];
  const lastRun = [...results].reverse().find((m) => ["single", "compare"].includes(m.content.view));
  const optionRes = lastRun && lastRun.content.mode && lastRun.content.mode !== "study";
  // follow-ups are about the newest study or backtest; a plan's own next steps are the buttons on its candidates
  const graphLast = lastRes && ["graph", "graph_batch"].includes(lastRes.content.view) && (!lastRun || lastRes.id > lastRun.id);
  const chips = act.isPending || pending ? [] : graphLast ? GRAPH_NEXT : !lastRun || lastRes?.content.view === "hypothesis" ? [] : optionRes ? OPTION_NEXT : STUDY_NEXT;
  const planMsg = eff?.type === "setup" ? plans.find((m) => m.content.version_id === eff.versionId) || lastPlan : null;
  const msg = eff?.type === "message" || eff?.type === "data" ? messages.find((m) => m.id === eff.messageId) : null;
  // the stages of the newest request, and the surface each one opens
  // The rail follows the newest piece of research (a request that was planned or run), not the newest sentence: a question
  // about its result is its "follow up" stage. What happened after the plan was written is read from the conversation.
  const stagedOf = (pred: (st: any[]) => boolean) => [...messages].reverse().find((m) => m.role === "assistant" && pred(m.content?.intent?.stages || []));
  const launched = [...messages].reverse().find((m) => m.role === "assistant" && m.kind !== "result" && (m.content?.intent?.stages || []).some((x: any) => ["plan", "execute"].includes(x.stage)));
  const staged = launched || stagedOf((st) => st.length > 0);
  const stages = (() => {  // computed in place: this sits below the early returns, where a hook may not be
    if (!staged) return [];
    const st: any[] = (staged.content.intent.stages as any[]).map((x) => ({ ...x }));
    const set = (stage: string, status: string, detail: string) => { const x = st.find((y) => y.stage === stage); if (x) Object.assign(x, { status, detail }); else st.push({ stage, status, detail }); };
    const after = messages.filter((m) => m.id > staged.id);
    const res = after.filter((m) => m.kind === "result");
    const failed = res.some((m) => m.content?.view === "failed");
    const done = res.some((m) => m.content?.view !== "failed") || (staged.kind === "result" && staged.content?.view !== "failed");
    const ex = st.find((x) => x.stage === "execute");
    if (ex && ex.status === "running" && (done || failed || staged.content?.finalized)) set("execute", failed && !done ? "failed" : "done", failed && !done ? "the run did not finish" : "finished");
    if (done) set("explain", "done", "the result is reported with its coverage, its observations and its limits");
    if (staged.kind === "sweep" && !st.some((x) => x.stage === "plan")) set("plan", "done", String(staged.content?.text || "a set of variants of the current version"));
    // a set of variants is checked against the data run by run: its coverage is read from the result's own table
    const rows: any[] = res.map((m) => m.content?.table?.rows || []).find((r) => r.some((x: any) => Number.isFinite(x.values?.qualifying))) || [];
    if (!st.some((x) => x.stage === "data") && rows.length) {
      const of = Math.max(...rows.map((x) => x.values.qualifying || 0)), priced = Math.min(...rows.map((x) => x.values.executable || 0));
      const short = rows.some((x) => (x.values.executable || 0) < (x.values.qualifying || 0));
      st.push({ stage: "data", status: short ? "attention" : "done", priced, of, detail: rows.map((x) => `${String(x.label).replace(/^v\d+\s*-?\s*/, "")}: ${(x.values.executable || 0).toLocaleString()} of ${(x.values.qualifying || 0).toLocaleString()} qualifying sessions priced`).join("; ") });
    }
    const asks = after.filter((m) => m.role === "user").length;
    if (asks) set("followup", "done", `${asks} follow-up question${asks === 1 ? "" : "s"} on this result`);
    return st;
  })();
  const checked = [...messages].reverse().find((m) => m.kind === "plan" && m.content?.data_check);
  const openStage = (id: string) => {
    if (id === "data") { if (checked) go({ view: { type: "data", messageId: checked.id } }); return; }
    if (["understand", "validate", "plan"].includes(id)) { if (lastPlan) go({ view: { type: "setup", versionId: lastPlan.content.version_id } }); return; }
    if (id === "execute" || id === "explain") { if (lastRes) go(targetOf(lastRes)); return; }
    comp.current?.focus();
  };
  const liveRun = [...messages].reverse().find(isPending);

  const right = (
    <div className="flex h-full min-h-0 flex-col bg-bg-0">
      <div className="flex h-11 shrink-0 items-center gap-1 border-b border-line px-3">
        {results.length > 0 && (
          <select className="field !h-8 max-w-[46%] cursor-pointer truncate" aria-label="Result shown" data-testid="result-picker"
            value={curId ?? ""} onChange={(e) => { const m = results.find((x) => String(x.id) === e.target.value); if (m) go(targetOf(m)); }}>
            {curId === null && <option value="">Results in this notebook ({results.length})</option>}
            {results.map((m, i) => <option key={m.id} value={m.id} className="bg-bg-2">{i + 1}. {resultName(m, messages)}</option>)}
          </select>
        )}
        <span className="flex-1" />
        {lastPlan && <button className={cn("btn quiet sm", eff?.type === "setup" && "!bg-bg-3 !text-fg")} onClick={() => go({ view: { type: "setup", versionId: lastPlan.content.version_id } })} title="How the request was read, editable (E)"><Pencil />Interpretation</button>}
        <button className={cn("btn quiet sm", eff?.type === "canvas" && "!bg-bg-3 !text-fg")} onClick={() => go({ view: { type: "canvas" } })} title="The whole notebook as a graph you can branch from (V)" data-testid="open-canvas"><Network />Canvas</button>
        <button className={cn("btn quiet sm", eff?.type === "compare" && "!bg-bg-3 !text-fg")} onClick={() => go({ view: { type: "compare" } })} title="Pinned results side by side (C)" data-testid="open-compare"><GitCompareArrows />Compare{pinned.length > 0 && <span className="num rounded-full bg-accent/20 px-1.5 text-[11px] text-accent">{pinned.length}</span>}</button>
        <Tip content="Keyboard shortcuts (?)"><button className="btn quiet sm !px-1.5" onClick={() => setHelp((h) => !h)} aria-label="Keyboard shortcuts"><Keyboard /></button></Tip>
        <button className="btn quiet sm !px-1.5" onClick={() => setShowRight(false)} aria-label="Hide the workspace" title="Hide the workspace"><PanelRightClose /></button>
      </div>
      {help && (
        <div className="shrink-0 border-b border-line bg-bg-1 px-4 py-3 text-xs dim">
          <div className="flex flex-wrap gap-x-5 gap-y-1.5">
            {[["/", "write a question"], ["V", "research canvas"], ["1 – 7", "report sections"], ["← →", "previous / next marked session"], ["[ ]", "previous / next result"], ["C", "comparison"], ["E", "edit the interpretation"], ["⌘K", "search everything"], ["Esc", "close"]].map(([k, l]) => <span key={k} className="flex items-center gap-1.5"><Kbd>{k}</Kbd>{l}</span>)}
          </div>
        </div>
      )}
      <div className={cn("min-h-0 flex-1", eff?.type === "canvas" ? "overflow-hidden" : "overflow-auto")}>
        {eff?.type === "canvas" && <Canvas sid={sid} stamp={messages.length} busy={act.isPending} wide={wide} onWide={setWide} onOpen={(t) => { setWide(false); actions.open(t); }}
          onBranch={(text, version_id) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/branch`, { text, version_id }))} onSelect={actions.select} onAnalyse={actions.analyse} />}
        {eff?.type === "message" && msg?.content.view === "hypothesis" && <HypothesisView runId={msg.content.run_id} busy={act.isPending} onSelect={(key, what) => actions.select(msg.content.run_id, key, what)} />}
        {eff?.type === "message" && msg?.content.view === "execution" && <ExecutionView runId={msg.content.run_id} />}
        {eff?.type === "message" && msg?.content.view === "challenge" && <ChallengeView runId={msg.content.run_id} />}
        {eff?.type === "message" && msg?.content.view === "analysis" && <AnalysisView c={msg.content} />}
        {eff?.type === "message" && msg && ["graph", "graph_batch"].includes(msg.content.view) && <GraphView key={msg.content.run_id} runId={msg.content.run_id} findings={msg.content.findings || []} busy={act.isPending} onStrategy={(id) => actions.strategyFrom(id)} onAsk={send} />}
        {eff?.type === "data" && msg?.content.data_check && <DataCheckView check={msg.content.data_check} busy={act.isPending} onPick={(x) => actions.pickAlt(msg.content.version_id, x)} />}
        {eff?.type === "message" && msg?.content.view === "fills" && <FillsView c={msg.content} />}
        {eff?.type === "message" && msg?.content.view === "untested" && <UntestedView c={msg.content} />}
        {eff?.type === "message" && msg?.content.view === "cost" && <CostView c={msg.content} busy={act.isPending} onPick={(x) => actions.pickAlt(msg.content.version_id, x)} />}
        {eff?.type === "message" && msg?.kind === "memory" && <MemoryView key={messages.length} sid={sid} onAsk={send} />}
        {eff?.type === "message" && msg?.kind === "registry" && <RegistryCheckView c={msg.content} />}
        {eff?.type === "message" && msg?.kind === "strategy_plan" && <StrategyPlanView c={msg.content} busy={act.isPending} onBacktest={actions.backtestPlan} />}
        {eff?.type === "run" && <Report key={eff.runId} runId={eff.runId} sid={sid} trials={data.trials || 1} tab={tab} setTab={setTab} which={which} setWhich={setWhich} day={day} onDay={(d) => { setDay(d); if (d && tab !== "calendar") setTab("calendar"); }} onAsk={send} onPin={(id) => pin(id)} pinned={pinned.includes(eff.runId)} />}
        {eff?.type === "setup" && planMsg && <Setup plan={planMsg.content} busy={act.isPending} refresh={actions.refresh} onConfirm={actions.confirm} onApply={(draft, run) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/edit`, { draft, run }))} />}
        {eff?.type === "message" && msg?.content.view === "sweep" && <SweepView c={msg.content} onOpen={(id) => go({ view: { type: "run", runId: id }, tab: "overview" })} onPin={(id) => pin(id)} pinned={pinned} />}
        {eff?.type === "message" && msg?.content.view === "investigation" && <InvestigationView c={msg.content} onOpen={(id) => go({ view: { type: "run", runId: id }, tab: "overview" })} onPin={(id) => pin(id)} pinned={pinned} />}
        {eff?.type === "message" && msg?.content.view === "discovery" && <DiscoveryView c={msg.content} />}
        {eff?.type === "compare" && <Compare ids={pinned} names={runNames} onOpen={(id) => go({ view: { type: "run", runId: id }, tab: "overview" })} onUnpin={(id) => pin(id, false)}
          onSwap={async (from, to) => { for (const id of from) await api.post(`/api/ai/sessions/${sid}/pin`, { run_id: id, on: false }); for (const id of to) await api.post(`/api/ai/sessions/${sid}/pin`, { run_id: id, on: true }); queryClient.invalidateQueries({ queryKey: ["ai", sid] }); }} />}
        {!eff && (liveRun ? <div className="mx-auto max-w-xl p-8"><div className="card px-5 py-4"><Timeline runId={liveRun.content.runs[liveRun.content.runs.length - 1].run_id} onDone={actions.refresh} title="Working on it" /></div></div>
          : <Empty icon={FileText} title="Results appear here">Ask a question on the left. The report, the calendar of qualifying sessions and the comparison of pinned results open on this side.</Empty>)}
      </div>
    </div>
  );

  const left = (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-line px-3">
        <button className="btn quiet sm !px-1.5" onClick={() => nav("/")} aria-label="All research" title="All research"><ArrowLeft /></button>
        <Sparkles className="size-4 shrink-0 text-accent" />
        <Title sid={sid} title={data.session.title} />
        <span className="flex-1" />
        {data.trials > 1 && <Tip content="How many distinct configurations this notebook has tested so far, including every one inside plans, maps and stress runs - not only the ones shown. Significance figures are corrected for it."><span className="chip !h-6" data-testid="trials-chip"><History className="size-3.5" />{data.trials} tested</span></Tip>}
        {((data as any).ledger?.holdouts || []).slice(-1).map((h: any) => <HoldoutChip key={h.id} h={h} compact />)}
        <InterpreterChip />
        {!showRight && <button className="btn quiet sm !px-1.5" onClick={() => setShowRight(true)} aria-label="Show the workspace" title="Show the workspace"><PanelRightOpen /></button>}
      </div>
      {stages.length > 0 && <div className="shrink-0 border-b border-line bg-bg-1/60 px-3 py-1"><StageRail stages={stages} onStage={openStage} /></div>}
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={cn("mx-auto flex w-full flex-col gap-3 px-4 py-4", showRight ? "max-w-[720px]" : "max-w-[820px]")} data-testid="conversation">
          {messages.length === 0 && !act.isPending && <div className="pt-10 text-center text-[13.5px] dim">Ask a question about the market, or describe a strategy in plain words.</div>}
          {messages.map((m) => <Message key={m.id} m={m} a={actions} />)}
          {act.isPending && <div className="flex items-center gap-2 px-1 text-[13px] dim"><span className="dot bg-accent pulse" />Reading the request…</div>}
          <div ref={end} />
        </div>
      </div>
      <div className="shrink-0 border-t border-line bg-bg-0 px-4 pb-3 pt-2.5">
        <div className={cn("mx-auto w-full", showRight ? "max-w-[720px]" : "max-w-[820px]")}>
          <Composer ref={comp} onSend={send} busy={act.isPending} mode={data.exec_mode} onMode={setMode} chips={chips} placeholder={messages.length ? "Ask a follow-up, change something, or start a new question" : (meta.data?.examples?.[0] ? `e.g. ${meta.data.examples[0]}` : "Ask a question")} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="lab h-full min-h-0">
      {showRight && wide && eff?.type === "canvas" ? right : showRight ? (
        <Group orientation="horizontal" id="lab-split">
          <Panel defaultSize="40%" minSize="26%" id="lab-chat">{left}</Panel>
          <Separator className="w-px bg-line transition-colors hover:bg-accent data-[separator=active]:bg-accent" />
          <Panel defaultSize="60%" minSize="34%" id="lab-work">{right}</Panel>
        </Group>
      ) : left}
    </div>
  );
}

function Title({ sid, title }: { sid: string; title: string }) {
  const [edit, setEdit] = useState(false);
  const [v, setV] = useState(title);
  useEffect(() => setV(title), [title]);
  const save = async () => {
    setEdit(false);
    if (v.trim() && v !== title) {
      await api.patch(`/api/ai/sessions/${sid}`, { title: v.trim() });
      queryClient.invalidateQueries({ queryKey: ["ai", sid] });
      queryClient.invalidateQueries({ queryKey: ["notebooks"] });
    } else setV(title);
  };
  return edit
    ? <input autoFocus className="field !h-7 min-w-0 flex-1" value={v} onChange={(e) => setV(e.target.value)} onBlur={save} onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") { setV(title); setEdit(false); } }} aria-label="Notebook title" />
    : <button className="min-w-0 truncate text-left text-[14px] font-medium hover:text-accent" onClick={() => setEdit(true)} title="Rename this notebook">{title}</button>;
}
