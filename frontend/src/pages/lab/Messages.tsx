/* The conversation: what was asked, how it was read, what is running, and what came back. */
import { Check, ChevronRight, Download, FileText, FlaskConical, GitCompareArrows, HelpCircle, Layers, Pencil, Pin, Play, ShieldAlert, ShieldQuestion, X, Zap } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { MODE_NAME, type AiMessage, type PlanItem, type Question, type RunRef } from "@/lib/lab";
import { Callout, ModeBadge, StatusChip, T } from "./bits";
import { Timeline, usePipeline } from "./Pipeline";

export type OpenTarget =
  | { type: "run"; runId: string; tab?: string; which?: string; messageId?: number }
  | { type: "setup"; versionId: number }
  | { type: "message"; messageId: number };

export interface Actions {
  answer: (versionId: number, questionId: string, option: number) => void;
  confirm: (versionId: number) => void;
  decide: (messageId: number, choice: string) => void;
  open: (t: OpenTarget) => void;
  pin: (runId: string) => void;
  refresh: () => void;
  rerun: (versionId: number, mode: string) => void;
  select: (runId: string, key: string, what: "study" | "options") => void;
  analyse: (kind: "stress" | "challenge", runId: string) => void;
  busy: boolean;
  pinned: string[];
}

function Items({ items, limit = 7 }: { items: PlanItem[]; limit?: number }) {
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, limit);
  return (
    <div className="flex flex-col gap-1.5">
      {shown.map((it, i) => (
        <div key={i} className="flex items-baseline gap-2 text-[13px]">
          <span className="w-[104px] shrink-0 text-xs dim">{it.label}</span>
          <span className={cn("min-w-0 flex-1", it.status === "unsupported" && "text-down")}><T>{String(it.value).split("   [")[0]}</T></span>
          {it.status !== "stated" && <StatusChip status={it.status} />}
        </div>
      ))}
      {items.length > limit && <button className="self-start text-xs text-info hover:underline" onClick={() => setAll((a) => !a)}>{all ? "Show less" : `Show ${items.length - limit} more`}</button>}
    </div>
  );
}

function Ask({ q, onPick, busy }: { q: Question; onPick: (i: number) => void; busy: boolean }) {
  const done = q.answered !== undefined && q.answered !== null;
  return (
    <div className={cn("rounded-[10px] border px-3.5 py-3", done ? "border-line bg-bg-2/40" : "border-accent/50 bg-accent/[0.06]")} data-testid="question">
      <div className="flex gap-2.5">
        <HelpCircle className={cn("size-4 shrink-0 mt-0.5", done ? "text-faint" : "text-accent")} />
        <div className="min-w-0 flex-1">
          <div className="text-[13.5px] font-medium">{q.text}</div>
          {q.hint && !done && <div className="mt-0.5 text-xs dim">{q.hint}</div>}
          <div className="mt-2.5 flex flex-wrap gap-2">
            {q.options.map((o, i) => (
              <button key={i} disabled={busy || done} onClick={() => onPick(i)} title={o.detail}
                className={cn("btn sm", done && q.answered === i && "!border-up/50 !bg-up/10 !text-up", done && q.answered !== i && "opacity-40")}>
                {done && q.answered === i && <Check />}{o.label}
              </button>
            ))}
          </div>
          {!done && q.options[0]?.detail && <div className="mt-2 text-xs dim">{q.options.map((o) => `${o.label}: ${o.detail}`).join(" · ")}</div>}
        </div>
      </div>
    </div>
  );
}

function RunLine({ r, onDone }: { r: RunRef; onDone: () => void }) {
  const { state, live } = usePipeline(r.run_id, onDone);
  const cur = state?.stages?.find((s) => s.status === "running");
  const n = state?.stages?.filter((s) => ["done", "skipped"].includes(s.status)).length || 0;
  const st = state?.status;
  return (
    <div className="flex items-center gap-2.5 text-xs">
      <span className={cn("dot", st === "error" ? "bg-down" : live ? "bg-accent pulse" : state?.outcome === "needs_decision" ? "bg-warn" : "bg-up")} />
      <span className="w-[42%] truncate soft">{r.label}</span>
      <span className="min-w-0 flex-1 truncate dim">{st === "error" ? state?.error : live ? (cur ? `${cur.label}${cur.detail ? ": " + cur.detail : ""}` : "queued") : state?.outcome === "needs_decision" ? "not covered by the data" : r.cached ? "reused an identical earlier run" : "done"}</span>
      {live && <span className="num text-faint">{n}/7</span>}
    </div>
  );
}

/** Progress of a job that is not a research pipeline (a plan, a stress ladder, a review). */
function JobLine({ r, onDone }: { r: RunRef; onDone: () => void }) {
  const { state, live } = usePipeline(r.run_id, onDone);
  const pct = Math.round(((state?.progress as number) || 0) * 100);
  return (
    <div className="flex flex-col gap-1.5 text-xs" data-testid="job-line">
      <div className="flex items-center gap-2.5">
        <span className={cn("dot", state?.status === "error" ? "bg-down" : live ? "bg-accent pulse" : "bg-up")} />
        <span className="min-w-0 flex-1 truncate soft">{state?.status === "error" ? state?.error : live ? state?.message || "queued" : "finished"}</span>
        {live && <span className="num text-faint">{pct}%</span>}
      </div>
      {live && <div className="h-1 overflow-hidden rounded-full bg-bg-3"><div className="h-full bg-accent transition-[width] duration-300" style={{ width: `${Math.max(pct, 3)}%` }} /></div>}
    </div>
  );
}

function PlanCard({ c }: { c: any }) {
  const [all, setAll] = useState(false);
  const p = c.plan || {};
  const cfgs: any[] = p.configs || [];
  const groups: Record<string, any[]> = {};
  for (const x of cfgs) (groups[x.outcome_text + (x.tag ? ` · ${x.tag}` : "")] ||= []).push(x);
  const need: any[] = c.data?.required || [];
  return (
    <div data-testid="plan-card">
      <div className="text-[14px] font-medium">{p.hypothesis}</div>
      <div className="mt-1 text-[12.5px] dim">Tested against: {p.null}</div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[12.5px]">
        <div><span className="dim">Configurations </span><span className="num">{cfgs.length}</span><span className="dim"> of at most {p.max_configs}</span></div>
        <div><span className="dim">Minimum sample </span><span className="num">{p.min_sample}</span><span className="dim"> observations</span></div>
        <div className="col-span-2"><span className="dim">Primary measure </span>{p.metric_label}</div>
        <div className="col-span-2"><span className="dim">Baseline </span>{p.baseline}</div>
        <div className="col-span-2"><span className="dim">Periods </span>training (selection), validation (can only veto), final test (scored once, for candidates only)</div>
      </div>
      {need.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {need.map((n, i) => <span key={i} className={cn("chip !h-6", !n.ok && "!border-warn/50 !text-warn")} title={n.note || ""}>{n.ok ? "✓" : "!"} {n.what} {n.symbol}: {Number(n.sessions).toLocaleString()} sessions{n.first ? `, ${n.first} → ${n.last}` : ""}</span>)}
        </div>
      )}
      <button className="mt-2.5 text-xs text-info hover:underline" onClick={() => setAll((a) => !a)}>{all ? "Hide the list" : `Show all ${cfgs.length} configurations that will be tested`}</button>
      {all && (
        <div className="mt-2 max-h-64 overflow-auto rounded-lg border border-line px-3 py-2 text-xs">
          {Object.entries(groups).map(([k, xs]) => (
            <div key={k} className="mb-2 last:mb-0"><div className="dim">Outcome: {k}</div>{xs.map((x) => <div key={x.key} className="pl-3 soft">· {x.label.replace(/ \[.*\]$/, "")}</div>)}</div>
          ))}
        </div>
      )}
      {(p.controls || []).map((n: string, i: number) => <div key={i} className="mt-1.5 text-xs dim">Control: {n}</div>)}
      {(p.notes || []).slice(0, 3).map((n: string, i: number) => <div key={i} className="mt-1 text-xs dim">· {n}</div>)}
    </div>
  );
}

function Card({ children, className, tone, ...rest }: { children: React.ReactNode; className?: string; tone?: "accent" } & React.HTMLAttributes<HTMLDivElement>) {
  return <div {...rest} className={cn("card fade-in px-4 py-3.5", tone === "accent" && "!border-accent/40", className)}>{children}</div>;
}

function MiniTable({ table, onOpen }: { table: any; onOpen: (runId: string) => void }) {
  if (!table?.rows?.length) return null;
  const cols = table.columns.slice(0, 5);
  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-line">
      <table className="t">
        <thead><tr><th>Variant</th>{cols.map((c: any) => <th key={c.key} className="r">{c.label}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((r: any) => (
            <tr key={r.run_id} className="click" onClick={() => onOpen(r.run_id)} title="Open this result">
              <td className="max-w-[220px] truncate">{r.label.replace(/^v\d+\s*-?\s*/, "")}</td>
              {cols.map((c: any) => <td key={c.key} className="r num">{r.shown[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Connecting the free data account without leaving the conversation: the two values are typed by the person,
    sent once to the local server (which stores them in the OS credential store) and never shown again. */
function AlpacaConnect({ c, a }: { c: any; a: Actions }) {
  const [id, setId] = useState("");
  const [secret, setSecret] = useState("");
  const [state, setState] = useState<{ busy?: boolean; res?: any }>({});
  const go = async () => {
    setState({ busy: true });
    try {
      if (id && secret) await api.put("/api/settings", { settings: {}, secrets: { ALPACA_API_KEY: id.trim(), ALPACA_API_SECRET: secret.trim() } });
      const res = await api.post<any>("/api/data/alpaca/test");
      setState({ res });
      if (res.ok) { setId(""); setSecret(""); }
    } catch (e: any) {
      setState({ res: { ok: false, message: e.message } });
    }
  };
  const r = state.res;
  return (
    <Card data-testid="alpaca-connect">
      <div className="eyebrow mb-1.5">Connect a free Alpaca account</div>
      <div className="text-[13.5px] soft">{c.text}</div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <input className="field" type="password" autoComplete="off" placeholder="Key ID" aria-label="Alpaca key ID" value={id} onChange={(e) => setId(e.target.value)} />
        <input className="field" type="password" autoComplete="off" placeholder="Secret" aria-label="Alpaca secret" value={secret} onChange={(e) => setSecret(e.target.value)} />
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <button className="btn primary sm" disabled={state.busy || (!!id !== !!secret)} onClick={go}>{state.busy ? "Testing…" : id && secret ? "Save and test" : "Test the saved key"}</button>
        <a className="text-xs text-info hover:underline" href="/settings#alpaca">How to get a key (three steps)</a>
      </div>
      {r && (
        <div className={cn("mt-2.5 rounded-lg border px-3 py-2 text-[13px]", r.ok ? "border-up/40 bg-up/[0.07]" : "border-down/40 bg-down/[0.07]")}>
          <div className={r.ok ? "text-up" : "text-down"}>{r.ok ? "Connected" : "Not connected"}</div>
          <div className="soft">{r.message}</div>
          {r.ok && <button className="btn sm mt-2" disabled={a.busy} onClick={() => a.rerun(c.version_id, c.mode || "auto")}><Play />Run it again with the new data</button>}
        </div>
      )}
      <div className="mt-2 text-xs dim">Free plan: market data only, delayed 15 minutes, never an order. Large downloads still ask first.</div>
    </Card>
  );
}

export function Message({ m, a }: { m: AiMessage; a: Actions }) {
  const c = m.content || {};
  if (m.role === "user") {
    return <div className="fade-in ml-10 self-end rounded-[14px] rounded-br-[4px] bg-bg-3 px-4 py-2.5 text-[14px]" data-testid="msg-user">{c.text}</div>;
  }
  if (m.kind === "plan") {
    const open = (c.questions || []).filter((q: Question) => q.answered === undefined || q.answered === null);
    const waiting = c.state === "awaiting_confirmation";
    return (
      <Card tone={open.length || waiting ? "accent" : undefined}>
        <div className="mb-2.5 flex items-center gap-2">
          <span className="eyebrow">How I read this</span>
          <ModeBadge mode={c.mode || (c.kind === "study" ? "study" : "auto")} className="!h-5 !text-[11px]" />
          <span className="flex-1" />
          <span className="text-[11px] text-faint">v{c.version_id} · {c.interpreter}</span>
        </div>
        <Items items={c.items || []} />
        {(c.notes || []).length > 0 && <div className="mt-2.5 flex flex-col gap-1 text-xs dim">{c.notes.map((n: string, i: number) => <div key={i}>· {n}</div>)}</div>}
        {(c.unsupported || []).length > 0 && (
          <Callout tone="bad" className="mt-3" title="Part of the request could not be represented">
            {c.unsupported.map((u: string) => `"${u}"`).join("; ")}. Nothing was substituted for it.
          </Callout>
        )}
        {(c.errors || []).length > 0 && c.state === "blocked" && <Callout tone="bad" className="mt-3" title="This cannot run yet">{c.errors.join("; ")}</Callout>}
        {(c.questions || []).length > 0 && <div className="mt-3 flex flex-col gap-2">{c.questions.map((q: Question) => <Ask key={q.id} q={q} busy={a.busy} onPick={(i) => a.answer(c.version_id, q.id, i)} />)}</div>}
        {waiting && open.length === 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2" data-testid="confirm">
            <button className="btn primary sm" disabled={a.busy} onClick={() => a.confirm(c.version_id)}><Play />{(c.unsupported || []).length ? "Run without that part" : "Looks right, run it"}</button>
            <button className="btn sm" onClick={() => a.open({ type: "setup", versionId: c.version_id })}><Pencil />Edit first</button>
            <span className="text-xs dim">{(c.assumptions || []).length ? `${c.assumptions.length} assumption${c.assumptions.length !== 1 ? "s" : ""} to check` : "Nothing was assumed"}</span>
          </div>
        )}
        {c.note && <div className="mt-2.5 rounded-lg bg-bg-2 px-3 py-2 text-xs soft" data-testid="plan-note">{c.note}</div>}
        {(c.runs || []).length > 0 && !c.finalized && <div className="mt-3.5 border-t border-line pt-3.5"><Timeline runId={c.runs[c.runs.length - 1].run_id} onDone={a.refresh} /></div>}
        {(c.runs || []).length > 0 && c.finalized && <div className="mt-3 border-t border-line pt-2.5"><Timeline runId={c.runs[c.runs.length - 1].run_id} compact /></div>}
        <div className="mt-2.5 flex items-center gap-1">
          <button className="btn quiet sm !px-2" onClick={() => a.open({ type: "setup", versionId: c.version_id })}><Pencil />Edit the interpretation</button>
          {(c.diff || []).length > 0 && <span className="text-xs dim truncate">changed: {c.diff.map((d: any) => d.field.replace("study.", "").replace(/_/g, " ")).join(", ")}</span>}
        </div>
      </Card>
    );
  }
  if (m.kind === "decision") {
    const done = c.resolved;
    return (
      <Card tone={done ? undefined : "accent"} className="!px-0 !py-0 overflow-hidden">
        <div className="flex gap-3 bg-warn/[0.07] px-4 py-3.5">
          <ShieldQuestion className="size-[18px] shrink-0 text-warn mt-0.5" />
          <div className="min-w-0">
            <div className="eyebrow !text-warn">The data cannot cover all of this</div>
            <div className="mt-1 text-[13.5px]">{c.text}</div>
          </div>
        </div>
        <div className="flex flex-col gap-1.5 px-3 py-3" data-testid="decision">
          {(c.choices || []).map((ch: any) => (
            <button key={ch.id} disabled={a.busy || !!done} onClick={() => a.decide(m.id, ch.id)}
              className={cn("group flex items-start gap-3 rounded-lg border border-transparent px-3 py-2 text-left transition-colors", done ? (done === ch.id ? "border-up/40 bg-up/[0.07]" : "opacity-40") : "hover:border-line-2 hover:bg-bg-2")}>
              <span className={cn("mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border", done === ch.id ? "border-up text-up" : "border-line-2 text-faint group-hover:border-accent group-hover:text-accent")}>
                {done === ch.id ? <Check className="size-3" strokeWidth={3} /> : <ChevronRight className="size-3" />}
              </span>
              <span className="min-w-0">
                <span className="block text-[13.5px] font-medium">{ch.label}{ch.kind === "mode" && ch.id === "mode:modeled" && <span className="ml-2 text-[11px] font-normal text-warn">estimates, not observed prices</span>}</span>
                <span className="block text-xs dim">{ch.detail}</span>
              </span>
            </button>
          ))}
        </div>
      </Card>
    );
  }
  if (["hypothesis", "stress", "challenge"].includes(m.kind)) {
    const runs: RunRef[] = c.runs || [];
    return (
      <Card data-testid={`pending-${m.kind}`}>
        <div className="mb-2 flex items-center gap-2">
          <span className="eyebrow">{m.kind === "hypothesis" ? "Research plan" : m.kind === "stress" ? "Execution stress test" : "Adversarial review"}</span>
          {m.kind === "hypothesis" && <span className="chip !h-5 !text-[11px]">bounded · {c.plan?.configs?.length} configurations</span>}
        </div>
        {m.kind === "hypothesis" ? <PlanCard c={c} /> : <div className="text-[13.5px] soft">{c.text}</div>}
        {m.kind === "hypothesis" && <div className="mt-2.5 text-[12.5px] dim">{c.text}</div>}
        {!c.finalized && runs[0] && <div className="mt-3 border-t border-line pt-3"><JobLine r={runs[0]} onDone={a.refresh} /></div>}
      </Card>
    );
  }
  if (["sweep", "investigation", "discovery", "run"].includes(m.kind)) {
    const runs: RunRef[] = c.runs || [];
    return (
      <Card>
        <div className="mb-2 flex items-center gap-2">
          <span className="eyebrow">{m.kind === "investigation" ? "Research plan" : m.kind === "discovery" ? "Exploration" : m.kind === "sweep" ? "Variants" : "Run"}</span>
          {m.kind === "run" && <ModeBadge mode={c.mode} className="!h-5 !text-[11px]" />}
        </div>
        {m.kind === "investigation" && c.plan && (
          <div className="mb-2.5">
            <div className="text-[13.5px] font-medium">{c.plan.hypothesis}</div>
            <div className="mt-1 text-xs dim">{c.plan.method}</div>
            {(c.plan.interpretation_notes || []).map((n: string, i: number) => <div key={i} className="mt-1 text-xs dim">· {n}</div>)}
          </div>
        )}
        {m.kind !== "investigation" && <div className="text-[13.5px] soft"><T>{c.text}</T></div>}
        {(c.not_explored || []).map((n: string, i: number) => <div key={i} className="mt-1.5 text-xs dim">· {n}</div>)}
        {runs.length === 1 && !c.finalized ? <div className="mt-3 border-t border-line pt-3"><Timeline runId={runs[0].run_id} onDone={a.refresh} /></div>
          : <div className="mt-3 flex flex-col gap-1.5 border-t border-line pt-3">{runs.map((r) => <RunLine key={r.run_id} r={r} onDone={a.refresh} />)}</div>}
        {(c.skipped || []).length > 0 && <div className="mt-2 text-xs text-warn">Not run: {c.skipped.map((s: any) => `${s.label} (${s.reason})`).join("; ")}</div>}
      </Card>
    );
  }
  if (m.kind === "result") {
    const v = c.view;
    if (v === "failed") return <Callout tone="bad" title="The run did not finish">{c.text}</Callout>;
    const runId: string | undefined = c.run_id;
    const pinned = runId ? a.pinned.includes(runId) : false;
    const cov = c.coverage;
    if (v === "hypothesis" || v === "execution" || v === "challenge" || v === "analysis") {
      const SEVG: Record<string, string> = { high: "✕", medium: "!", low: "·", ok: "✓", na: "–" };
      return (
        <Card data-testid={`result-${v}`}>
          <div className="mb-2 flex items-center gap-2">
            <span className="eyebrow">{{ hypothesis: "What the plan found", execution: "Execution uncertainty", challenge: "Review", analysis: c.analysis?.title || "Answer" }[v as string]}</span>
            <span className="flex-1" /><span className="text-[11px] text-faint" title="Assembled from the engine's numbers; no language model wrote this.">calculated</span>
          </div>
          <div className="text-[13.5px] soft"><T>{c.text}</T></div>
          {v === "hypothesis" && (c.candidates || []).length > 0 && (
            <div className="mt-3 flex flex-col gap-1.5">
              {c.candidates.map((x: any) => (
                <div key={x.key} className="rounded-lg border border-line px-3 py-2">
                  <div className="text-[13px]">{x.label}</div>
                  <div className="mt-0.5 text-xs dim">{x.outcome_text} · {x.status_label}{x.grade ? ` · ${x.grade.label.toLowerCase()}` : ""}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    <button className="btn quiet sm" disabled={a.busy} onClick={() => a.select(c.run_id, x.key, "study")}><Layers />Investigate further</button>
                    <button className="btn quiet sm" disabled={a.busy} onClick={() => a.select(c.run_id, x.key, "options")}><FlaskConical />Test option structures</button>
                  </div>
                </div>
              ))}
            </div>
          )}
          {v === "challenge" && (
            <div className="mt-2.5 flex flex-col gap-1">
              {(c.findings || []).filter((f: any) => ["high", "medium"].includes(f.severity)).slice(0, 6).map((f: any) => (
                <div key={f.topic} className="flex gap-2 text-[13px]"><span className={cn("w-3 shrink-0 text-center font-semibold", f.severity === "high" ? "text-down" : "text-warn")}>{SEVG[f.severity]}</span><span><span className="dim">{f.name}: </span>{f.title}</span></div>
              ))}
            </div>
          )}
          {v === "execution" && c.summary && (
            <div className="mt-2.5 grid grid-cols-3 gap-2 text-xs">
              {[["Signal only", c.summary.signal_net_pnl], ["As simulated", c.summary.simulated_net_pnl], ["Adverse", c.summary.adverse_net_pnl]].map(([l, x]: any) => (
                <div key={l} className="rounded-lg bg-bg-2 px-2.5 py-1.5"><div className="dim">{l}</div><div className={cn("num text-[13px]", (x ?? 0) >= 0 ? "text-up" : "text-down")}>{x == null ? "—" : `${x < 0 ? "-" : ""}$${Math.abs(x).toLocaleString("en-US", { maximumFractionDigits: 0 })}`}</div></div>
              ))}
            </div>
          )}
          {(c.caveats || []).length > 0 && v === "analysis" && <div className="mt-2 text-xs dim">{c.caveats[0]}</div>}
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn sm" data-testid="open-view" onClick={() => a.open({ type: "message", messageId: m.id })}>
              {v === "challenge" ? <ShieldAlert /> : v === "execution" ? <Zap /> : <FileText />}{{ hypothesis: "Open every configuration", execution: "Open the stress ladder", challenge: "Open the review", analysis: "Open the numbers" }[v as string]}
            </button>
          </div>
        </Card>
      );
    }
    const title = { single: "Result", compare: "Before and after", sweep: `Comparison${c.what ? `: ${c.what}` : ""}`, investigation: "Findings", discovery: "What the exploration found", robustness: "Is it real?",
      breakdown: c.by === "year" ? "By year" : "By day of the week", timing: "Time of day", skipped: "Signals that were not traded", export: "Report ready" }[v as string] || "Result";
    return (
      <Card data-testid={`result-${v}`}>
        <div className="mb-2 flex items-center gap-2">
          <span className="eyebrow">{title}</span>
          {c.mode && ["single", "compare", "sweep"].includes(v) && <ModeBadge mode={c.mode} className="!h-5 !text-[11px]" />}
          {v === "robustness" && c.robustness && <span className={cn("chip !h-5 !text-[11px]", { supported: "!text-up !border-up/40", mixed: "!text-warn !border-warn/40", none: "", insufficient: "!text-warn !border-warn/40" }[c.robustness.level as string])}>{({ supported: "holds up", mixed: "mixed", none: "no edge found", insufficient: "too few observations" } as any)[c.robustness.level]}</span>}
          <span className="flex-1" />
          {c.explanation_source && c.explanation_source !== "deterministic" && <span className="text-[11px] text-faint" title="Worded by the language model from the engine's numbers; any number it added would have been rejected.">worded by {String(c.explanation_source).split(":").slice(-1)[0]}</span>}
        </div>
        {v === "investigation" && c.report ? (
          <>
            <div className="text-[14px] font-medium">{c.report.verdict}</div>
            <ul className="mt-2 flex flex-col gap-1 text-[13px] soft">{c.report.findings.map((f: string, i: number) => <li key={i}>· {f}</li>)}</ul>
          </>
        ) : v === "robustness" && c.robustness ? (
          <>
            <div className="text-[14px] font-medium">{c.robustness.verdict}</div>
            <div className="mt-2.5 flex flex-col gap-1.5">
              {c.robustness.checks.map((k: any, i: number) => (
                <div key={i} className="flex gap-2.5 text-[13px]">
                  <span className={cn("mt-0.5 grid size-4 shrink-0 place-items-center rounded-full", k.ok ? "bg-up/15 text-up" : "bg-down/15 text-down")}>{k.ok ? <Check className="size-2.5" strokeWidth={3.5} /> : <X className="size-2.5" strokeWidth={3.5} />}</span>
                  <span><span className="font-medium">{k.name}.</span> <span className="soft"><T>{k.detail}</T></span></span>
                </div>
              ))}
            </div>
            {c.robustness.trials > 1 && <div className="mt-2 text-xs dim">{c.robustness.trials} configurations have been tested in this notebook; significance is adjusted for that.</div>}
          </>
        ) : v === "export" ? (
          <>
            <div className="text-[13.5px] soft">{c.text}</div>
            <div className="mt-3 flex flex-col gap-1.5">
              {(c.links || []).map((l: any) => (
                <a key={l.href} href={l.href} target={l.href.includes("format=html") ? "_blank" : undefined} download={l.href.includes("format=html") ? undefined : l.file} rel="noreferrer"
                  className="flex items-center gap-2.5 rounded-lg border border-line px-3 py-2 text-[13px] transition-colors hover:border-accent/60 hover:bg-bg-2">
                  {l.href.includes("format=html") ? <FileText className="size-4 text-accent" /> : <Download className="size-4 dim" />}<span className="flex-1">{l.label}</span><span className="text-[11px] text-faint">{l.file}</span>
                </a>
              ))}
            </div>
          </>
        ) : (
          <div className="prose text-[13.5px] soft"><T>{v === "compare" && c.compare_text ? c.compare_text : c.text}</T></div>
        )}
        {c.compatibility?.warning && <Callout tone="warn" className="mt-3">{c.compatibility.warning}</Callout>}
        {["compare", "sweep", "investigation"].includes(v) && <MiniTable table={c.table} onOpen={(id) => a.open({ type: "run", runId: id })} />}
        {v === "single" && cov && cov.signals?.qualifying !== undefined && cov.executable && cov.executable.signals < cov.signals.qualifying && (
          <Callout tone="warn" className="mt-3" title={`${cov.executable.signals} of ${cov.signals.qualifying} qualifying sessions could be tested`}>
            {(cov.options?.issues || []).slice(0, 2).map((x: any) => `${x.reason} (${x.signals})`).join("; ")}
          </Callout>
        )}
        {(c.failed || []).length > 0 && v !== "failed" && <div className="mt-2 text-xs text-warn">Not completed: {c.failed.map((f: any) => `${f.label}`).join("; ")}</div>}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {runId && !["export", "sweep", "investigation", "discovery"].includes(v) && (
            <button className="btn sm" data-testid="open-report" onClick={() => a.open({ type: "run", runId, messageId: m.id, tab: ({ robustness: "robustness", skipped: "observations", breakdown: "statistics", timing: "statistics" } as any)[v] || "overview", which: v === "skipped" ? "skipped" : undefined })}>
              <FileText />{v === "single" || v === "compare" ? "Open the report" : "Open in the workspace"}
            </button>
          )}
          {["sweep", "investigation", "discovery"].includes(v) && <button className="btn sm" data-testid="open-view" onClick={() => a.open({ type: "message", messageId: m.id })}><GitCompareArrows />{v === "discovery" ? "Open the exploration" : v === "investigation" ? "Open the findings" : "Open the comparison"}</button>}
          {runId && ["single", "compare"].includes(v) && <button className={cn("btn quiet sm", pinned && "!text-accent")} onClick={() => a.pin(runId)} title="Pin this result to the comparison workspace"><Pin />{pinned ? "Pinned" : "Pin to compare"}</button>}
          {c.mode_reason && v === "single" && <span className="text-xs dim truncate" title={c.mode_reason}>{MODE_NAME[c.mode] || ""}</span>}
        </div>
      </Card>
    );
  }
  if (m.kind === "trades") {
    return (
      <Card>
        <div className="text-[13.5px] soft">{c.text}</div>
        <button className="btn sm mt-2.5" onClick={() => a.open({ type: "run", runId: c.run_id, tab: "observations", which: c.filter === "all" ? "all" : c.mode === "study" ? (c.filter === "win" ? "target" : "neither") : c.filter })}><FileText />Open the list</button>
      </Card>
    );
  }
  if (m.kind === "error") return <Callout tone="bad">{c.text}</Callout>;
  if (m.kind === "note" && c.action === "connect_alpaca") return <AlpacaConnect c={c} a={a} />;
  return <div className="fade-in whitespace-pre-wrap rounded-[12px] px-1 text-[13.5px] soft">{c.text}</div>;
}
