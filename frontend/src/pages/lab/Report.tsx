/* The research report of one result: Overview, Calendar, Observations, Statistics, Robustness, Methodology, Raw data.
   Every number shown here comes from the run's stored result; clicking a headline number opens the observations behind it. */
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Blocks, Check, Code2, Download, ExternalLink, FileText, Pin, Scale, Snowflake, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { pp, sgn, usd, type Fact } from "@/lib/lab";
import { useBuilder } from "@/lib/store";
import { Bar, Callout, Empty, ModeBadge, Section, Skeleton, StatTile, StatusChip, T, Term, useHotkeys } from "./bits";
import { ExecutionCheck, FillsLedger, PerformanceView } from "./Performance";
import { ReplayView } from "./ReplayView";
import { CalendarView } from "./CalendarView";
import { ResultCoverage } from "./Coverage";
import { Timeline } from "./Pipeline";

export const REPORT_TABS = [["overview", "Overview"], ["calendar", "Calendar"], ["observations", "Trades"], ["statistics", "Statistics"], ["robustness", "Robustness"], ["methodology", "Methodology"], ["raw", "Raw data"]] as const;
// a backtest on option observations: what it did (in the order a reader should take it), each trade, the trade over the
// stored observations, every fill against its stored bar, and the same decisions on individual prints
export const BACKTEST_TABS = [["overview", "Overview"], ["performance", "Tear sheet"], ["observations", "Trades"], ["replay", "Replay"], ["fills", "Fills"], ["execution", "Execution"], ["calendar", "Calendar"], ["statistics", "Statistics"],
  ["robustness", "Robustness"], ["methodology", "Methodology"], ["raw", "Raw data"]] as const;

export function useReport(runId: string, sid?: string, trials = 1) {
  return useQuery({ queryKey: ["report", runId, sid, trials], queryFn: () => api.get<any>(`/api/runs/${runId}/report?trials=${trials}${sid ? `&sid=${sid}` : ""}`), staleTime: 60_000 });
}
const f = (facts: Fact[], k: string) => facts?.find((x) => x.key === k);

export function Report({ runId, sid, trials, tab, setTab, which, setWhich, onDay, day, onAsk, onPin, pinned }: {
  runId: string; sid?: string; trials: number; tab: string; setTab: (t: string) => void; which: string; setWhich: (w: string) => void; onDay: (d: string | null) => void; day: string | null;
  onAsk: (text: string) => void; onPin: (id: string) => void; pinned: boolean;
}) {
  const rep = useReport(runId, sid, trials);
  const [year, setYear] = useState<number | null>(null);
  const [trade, setTrade] = useState(1);
  const r = rep.data;
  const tabs: readonly (readonly [string, string])[] = r && r.kind !== "study" ? BACKTEST_TABS : REPORT_TABS;
  useHotkeys(Object.fromEntries(tabs.slice(0, 9).map(([k], i) => [String(i + 1), () => setTab(k)])), [runId, tabs.length]);
  if (rep.isLoading) return <div className="flex flex-col gap-3 p-5"><Skeleton className="h-8 w-2/3" /><Skeleton className="h-24" /><Skeleton className="h-64" /></div>;
  if (rep.isError || !r) return <Empty title="This result cannot be opened">{String((rep.error as any)?.message || "")}</Empty>;
  const study = r.kind === "study";
  const openObs = (w: string) => { setWhich(w); setTab("observations"); };
  const openYear = (y: string) => { setYear(Number(y)); onDay(null); setTab("calendar"); };
  const nTrades = Number(String(f(r.facts, "trades")?.shown ?? "").replace(/[^0-9]/g, "")) || undefined;
  const openTrade = (id: number) => { setTrade(id); setTab("replay"); };
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="report">
      <div className={cn("shrink-0 px-5", tab === "replay" ? "py-1.5" : "pt-4 pb-3")}>
        <div className={cn("flex gap-3", tab === "replay" ? "items-center" : "items-start")}>
          {/* on the replay the charts need the height: the heading is one line there */}
          <div className={cn("min-w-0 flex-1", tab === "replay" && "flex items-center gap-3")}>
            <h2 className={cn("truncate", tab === "replay" && "!text-[13.5px] min-w-0")} title={r.question}>{r.question || r.title}</h2>
            <div className={cn("flex items-center gap-2 text-xs dim", tab === "replay" ? "shrink-0" : "mt-1.5 flex-wrap")}>
              <ModeBadge mode={r.mode} />
              <span>{r.symbol}</span><span>·</span><span>{r.period[0]} to {r.period[1]}</span>
              {r.mode_reason && tab !== "replay" && <Tip content={r.mode_reason}><span className="term">why this mode</span></Tip>}
            </div>
          </div>
          <button className={cn("btn sm", pinned && "!border-accent/60 !text-accent")} onClick={() => onPin(runId)} title="Pin to the comparison workspace"><Pin />{pinned ? "Pinned" : "Pin"}</button>
          <a className="btn sm" href={`/api/runs/${runId}/report?format=html${sid ? `&sid=${sid}` : ""}&trials=${trials}`} target="_blank" rel="noreferrer" title="Open the full report as a standalone page"><ExternalLink />Export</a>
        </div>
      </div>
      <div className="tabbar shrink-0" role="tablist">
        {tabs.map(([k, l], i) => (
          <button key={k} role="tab" aria-selected={tab === k} data-testid={`tab-${k}`} className={cn("tab", tab === k && "on")} onClick={() => setTab(k)}>
            {k === "observations" && study ? "Sessions" : l}{i < 9 && <span className="num text-[10px] text-faint">{i + 1}</span>}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {tab === "overview" && <Overview r={r} openObs={openObs} setTab={setTab} openYear={openYear} />}
        {tab === "calendar" && <CalendarView runId={runId} day={day} onDay={onDay} year={year} />}
        {tab === "observations" && <Observations runId={runId} study={study} which={which} setWhich={setWhich} onDay={onDay} day={day} onTrade={study ? undefined : openTrade} />}
        {tab === "performance" && !study && <PerformanceView runId={runId} sid={sid} trials={trials} onAsk={onAsk} />}
        {tab === "replay" && !study && <ReplayView runId={runId} tradeId={trade} onTrade={setTrade} tradeCount={nTrades} />}
        {tab === "fills" && !study && <FillsLedger runId={runId} onTrade={openTrade} />}
        {tab === "execution" && !study && <ExecutionCheck runId={runId} />}
        {tab === "statistics" && <Statistics r={r} openObs={openObs} />}
        {tab === "robustness" && <Robustness r={r} onAsk={onAsk} />}
        {tab === "methodology" && <Methodology r={r} runId={runId} />}
        {tab === "raw" && <Raw r={r} runId={runId} sid={sid} trials={trials} />}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------- overview
function Funnel({ r }: { r: any }) {
  const d = r.data;
  const study = r.kind === "study";
  const rows: { label: string; n: number | undefined; note?: string }[] = [
    { label: "Sessions in the requested period", n: d.requested?.sessions, note: d.requested?.stated ? "the period you asked for" : "all available history (no period was stated)" },
    { label: "Have the prices the rules need", n: d.usable?.sessions, note: (d.excluded || []).slice(0, 2).map((x: any) => `${x.sessions.toLocaleString()} excluded: ${x.reason}`).join(" · ") },
    { label: "Conditions met", n: d.signals?.qualifying, note: d.signals?.not_evaluable_sessions ? `${d.signals.not_evaluable_sessions} more could not be evaluated (kept apart, not counted as "no")` : undefined },
  ];
  if (!study) rows.push({ label: "Could be traded on observed option prices", n: d.executable?.signals, note: (d.options?.issues || []).slice(0, 2).map((x: any) => `${x.signals} not: ${x.reason}`).join(" · ") });
  const max = Math.max(...rows.map((x) => x.n || 0), 1);
  return (
    <div className="flex flex-col gap-2.5">
      {rows.filter((x) => x.n !== undefined && x.n !== null).map((x, i) => (
        <div key={i}>
          <div className="flex items-baseline justify-between gap-3 text-[13px]"><span className="soft">{x.label}</span><span className="num font-medium">{x.n!.toLocaleString()}</span></div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-bg-3"><div className={cn("h-full rounded-full", i === rows.length - 1 || (study && i === 2) ? "bg-accent" : "bg-info/70")} style={{ width: `${Math.max(1.2, Math.sqrt((x.n || 0) / max) * 100)}%` }} /></div>
          {x.note && <div className="mt-1 text-xs dim">{x.note}</div>}
        </div>
      ))}
    </div>
  );
}

/** One strategy, every tool: what was asked in words can be edited as code or with forms, and then followed forward.
    Nothing is copied by hand; each tool opens on the configuration this result was computed from. */
function NextSteps({ r }: { r: any }) {
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const cfg = r.reproduce.config;
  const runId = r.reproduce.run_id;
  const load = (to: string) => {
    useBuilder.getState().load(cfg.strategies[0], null);
    useBuilder.getState().patch({ start: cfg.start, end: cfg.end, capital: cfg.capital, execution: cfg.execution });
    nav(to);
  };
  const freeze = async () => {
    setBusy(true);
    try {
      const ex = await api.post<any>("/api/forward", { run_id: runId });
      toast.success(`Frozen. The first session that counts is ${ex.first_session}.`);
      nav(`/forward/${ex.id}`);
    } catch (e: any) { toast.error(e.message); } finally { setBusy(false); }
  };
  const steps: [any, string, string, () => void, string][] = [
    [Code2, "Edit it as code", "The same strategy as text in the IDE, with a debugger that replays its rules on a real session.", () => load("/ide"), "ide"],
    [Blocks, "Edit it with forms", "Legs, entry rules and exits in the visual builder.", () => load("/builder"), "builder"],
    [Snowflake, "Freeze it and follow it forward", "Stored with a hash and the time on the clock. Only sessions that have not happened yet count.", freeze, "forward"],
    [Scale, "See what has been tested before", "Repeats, related configurations and held-back periods across every notebook.", () => nav("/registry"), "registry"],
  ];
  return (
    <Section title="Take this strategy further" sub="the other tools open on exactly this configuration">
      <div className="grid gap-2 sm:grid-cols-2" data-testid="next-steps">
        {steps.map(([Icon, title, sub, go, id]) => (
          <button key={id} disabled={busy} onClick={go} data-next={id} className="flex items-start gap-2.5 rounded-lg border border-line px-3 py-2.5 text-left transition-colors hover:border-accent/60 hover:bg-bg-2">
            <Icon className="mt-0.5 size-4 shrink-0 text-accent" />
            <span className="min-w-0 flex-1"><span className="block text-[13px] font-medium">{title}</span><span className="mt-0.5 block text-xs dim">{sub}</span></span>
            <ArrowRight className="mt-1 size-3.5 shrink-0 text-faint" />
          </button>
        ))}
      </div>
    </Section>
  );
}

function Overview({ r, openObs, setTab, openYear }: { r: any; openObs: (w: string) => void; setTab: (t: string) => void; openYear: (y: string) => void }) {
  const study = r.kind === "study";
  const F: Fact[] = r.facts;
  const cs = r.data.executable;
  const sig = r.data.signals?.qualifying;
  const small = study ? (r.sample.qualifying || 0) < 30 : (r.sample.trades || 0) < 30;
  return (
    <div className="flex flex-col gap-4 p-5">
      {r.mode === "modeled" && <Callout tone="warn" title="Modeled premiums">Option prices here are THEORETICAL estimates from a pricing model. They are not observed market prices and the dollar figures are approximate.</Callout>}
      {r.mode === "trades" && <Callout tone="info" title="Priced on option trade prints">Fills use prices at which the contracts actually traded. This data has no <Term k="bid/ask" />, so no spread is assumed; a print is only used while it is recent.</Callout>}
      {(r.interpretation.unsupported || []).length > 0 && <Callout tone="bad" title="Part of the request was not represented">{r.interpretation.unsupported.map((u: string) => `"${u}"`).join("; ")}</Callout>}
      {!study && <ResultCoverage cov={r.data} mode={r.mode} onOpen={() => openObs("skipped")} />}
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))" }}>
        {study ? (
          <>
            <StatTile label="How often" value={f(F, "probability")?.shown ?? "—"} sub={`${f(F, "target_first")?.shown ?? 0} of ${f(F, "qualifying_sessions")?.shown ?? 0} sessions`} tone="text-accent-2" onClick={() => openObs("target")} />
            <StatTile label={<Term k="baseline">On all sessions</Term>} value={f(F, "baseline_probability")?.shown ?? "—"} sub={`${f(F, "evaluated_sessions")?.shown ?? ""} sessions`} onClick={() => openObs("baseline")} />
            <StatTile label="Difference" value={r.baseline?.lift !== null && r.baseline?.lift !== undefined ? `${r.baseline.lift > 0 ? "+" : ""}${(r.baseline.lift * 100).toFixed(1)} pts` : "—"} sub={f(F, "p_value") ? <>p {f(F, "p_value")!.shown}</> : undefined} tone={r.baseline?.lift > 0 ? "text-up" : r.baseline?.lift < 0 ? "text-down" : undefined} hint="Percentage points above or below the baseline, and how likely a gap this size is by chance." />
            <StatTile label={<Term k="confidence interval">95% range</Term>} value={`${f(F, "probability_ci_low")?.shown ?? "—"} – ${f(F, "probability_ci_high")?.shown ?? "—"}`} sub="where the true frequency plausibly lies" />
            <StatTile label="Average end of day" value={f(F, "avg_end_return_pct")?.shown ?? "—"} sub="from the starting price" onClick={() => openObs("all")} />
          </>
        ) : (
          <>
            <StatTile label="Net result" value={usd(r.metrics?.net_pnl, 2)} sub="after costs" tone={r.metrics?.net_pnl > 0 ? "text-up" : r.metrics?.net_pnl < 0 ? "text-down" : undefined} onClick={() => openObs("all")} />
            <StatTile label="Trades" value={r.sample.trades ?? 0} sub={sig !== undefined && sig !== null ? `from ${sig} qualifying sessions` : undefined} onClick={() => openObs("all")} />
            <StatTile label="Winners" value={pp(r.metrics?.win_rate)} sub={r.sample.trades ? `${Math.round((r.metrics?.win_rate || 0) * r.sample.trades)} of ${r.sample.trades}` : undefined} onClick={() => openObs("win")} />
            <StatTile label="Average per trade" value={usd(r.metrics?.expectancy, 2)} tone={r.metrics?.expectancy > 0 ? "text-up" : r.metrics?.expectancy < 0 ? "text-down" : undefined} />
            <StatTile label="Deepest drawdown" value={pp(r.metrics?.max_drawdown)} sub="of account equity" hint="The largest peak-to-trough fall in the account's value." />
            {cs && sig > cs.signals && <StatTile label="Not traded" value={sig - cs.signals} sub="qualified, no option prices" tone="text-warn" onClick={() => openObs("skipped")} />}
          </>
        )}
      </div>
      {small && <Callout tone="warn" title={study ? `Only ${r.sample.qualifying} qualifying sessions` : `Only ${r.sample.trades} trade${r.sample.trades === 1 ? "" : "s"}`}>
        Too few observations to draw a conclusion. Treat the numbers as a description of what happened, not as evidence of an edge.
      </Callout>}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Section title={study ? "By year" : "Net profit over time"} sub={study ? "click a bar to see those sessions" : undefined}>
          {study ? <YearChart r={r} onYear={openYear} /> : <EquityChart r={r} />}
        </Section>
        <Section title="From request to evidence" sub="what was covered"><Funnel r={r} /></Section>
      </div>
      <Section title="In words">
        <div className="prose text-[14px] leading-[1.6] soft"><T>{r.summary}</T></div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button className="btn sm" onClick={() => setTab("robustness")}>Is this real? See the robustness checks</button>
          <button className="btn quiet sm" onClick={() => setTab("methodology")}>How it was calculated</button>
        </div>
      </Section>
      {!study && r.reproduce?.config?.strategies?.length > 0 && <NextSteps r={r} />}
      {(r.limitations || []).length > 0 && (
        <Section title="Limitations" sub={`${r.limitations.length}`} open={r.limitations.length <= 4}>
          <ul className="flex flex-col gap-1.5 text-[13px] soft">{r.limitations.map((x: string, i: number) => <li key={i} className="flex gap-2"><span className="text-faint">·</span><span><T>{x}</T></span></li>)}</ul>
        </Section>
      )}
    </div>
  );
}

function EquityChart({ r }: { r: any }) {
  const eq: [string, number | null][] = r.charts?.equity || [];
  const dd: [string, number | null][] = r.charts?.drawdown || [];
  if (eq.length < 2) return <div className="py-8 text-center text-[13px] dim">A single session: there is no curve to draw. See the trade under Trades.</div>;
  const option = {
    grid: [{ left: 56, right: 16, top: 12, height: "58%" }, { left: 56, right: 16, top: "76%", bottom: 26 }],
    tooltip: { trigger: "axis" },
    xAxis: [{ type: "category", data: eq.map((x) => x[0]), gridIndex: 0, axisLabel: { show: false } }, { type: "category", data: dd.map((x) => x[0]), gridIndex: 1 }],
    yAxis: [{ type: "value", gridIndex: 0, axisLabel: { formatter: fmtMoneyAxis } }, { type: "value", gridIndex: 1, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } }],
    series: [
      { type: "line", data: eq.map((x) => x[1]), showSymbol: false, lineStyle: { width: 1.6, color: C.accent }, areaStyle: { color: "rgba(232,163,61,0.08)" }, name: "Net profit ($)", markLine: { silent: true, symbol: "none", lineStyle: { color: C.faint, type: "dashed" }, data: [{ yAxis: 0 }], label: { show: false } } },
      { type: "line", data: dd.map((x) => x[1]), showSymbol: false, xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 1, color: C.down }, areaStyle: { color: "rgba(239,83,80,0.12)" }, name: "Drawdown" },
    ],
  };
  return <EChart option={option} height={300} />;
}

function YearChart({ r, onYear }: { r: any; onYear?: (y: string) => void }) {
  const years: any[] = r.by_year || [];
  const study = r.kind === "study";
  if (!years.length) return <div className="py-8 text-center text-[13px] dim">No yearly breakdown.</div>;
  const base = r.baseline?.probability;
  const option = {
    grid: { left: 48, right: 14, top: 14, bottom: 26 },
    tooltip: { trigger: "axis", formatter: (p: any) => { const y = years[p[0].dataIndex]; return study ? `${y.year}<br/>${y.target} of ${y.n} sessions${y.probability !== null ? ` (${(y.probability * 100).toFixed(0)}%)` : ""}` : `${y.key}<br/>${y.trades} trades, ${y.pnl >= 0 ? "+" : ""}$${Math.round(y.pnl).toLocaleString()}`; } },
    xAxis: { type: "category", data: years.map((y) => (study ? y.year : y.key)) },
    yAxis: { type: "value", ...(study ? { max: 1, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } } : { axisLabel: { formatter: fmtMoneyAxis } }) },
    series: [{
      type: "bar", barMaxWidth: 22,
      data: years.map((y) => (study ? { value: y.probability, itemStyle: { color: y.n < 5 ? "rgba(76,141,255,0.35)" : C.info } } : { value: y.pnl, itemStyle: { color: y.pnl >= 0 ? C.up : C.down } })),
      ...(study && base !== null && base !== undefined ? { markLine: { silent: true, symbol: "none", lineStyle: { color: C.accent, type: "dashed" }, data: [{ yAxis: base }], label: { formatter: "baseline", color: C.accent, fontSize: 10 } } } : {}),
    }],
  };
  return (
    <>
      <EChart option={option} height={250} onClick={(p) => onYear?.(String(p.name))} />
      {study && <div className="mt-1 text-xs dim">Faded bars are years with fewer than 5 qualifying sessions. The dashed line is the frequency on all sessions.</div>}
    </>
  );
}

// ----------------------------------------------------------------------------- observations
const STUDY_FILTERS = [["all", "Qualifying"], ["target", "Target reached"], ["stop", "Stop reached"], ["neither", "Neither"], ["ambiguous", "Ambiguous"], ["baseline", "Every evaluated session"], ["excluded", "Could not be evaluated"]];
const BT_FILTERS = [["all", "All trades"], ["win", "Winners"], ["loss", "Losers"], ["target", "Hit a target"], ["stop", "Hit a stop"], ["time", "Closed by time"], ["skipped", "Qualified, not traded"], ["excluded", "Could not be evaluated"]];
const HEAD: Record<string, string> = { end_return_pct: "end %", mfe_pct: "best %", mae_pct: "worst %", prev_return_pct: "prev day %", gap_pct: "gap %", entry_minute: "start", minutes_to_target: "min to target",
  target_level: "target", stop_level: "stop", entry_price: "entry", exit_price: "exit", vix_prev: "VIX", fill_delay_s: "fill delay s", price_source: "priced on", exit_reason: "closed by", entry_spot: "underlying in", exit_spot: "underlying out" };

function cell(k: string, v: any) {
  if (v === null || v === undefined || v === "") return <span className="text-faint">—</span>;
  if (k === "outcome") return <span className={({ target: "text-up", stop: "text-down", ambiguous: "text-warn" } as any)[v] || "dim"}>{({ target: "target", stop: "stop", neither: "neither", ambiguous: "ambiguous" } as any)[v] || v}</span>;
  if (k === "pnl") return <span className={cn("num", v > 0 ? "text-up" : v < 0 ? "text-down" : "")}>{usd(v, 2)}</span>;
  if (k.endsWith("_pct")) return <span className={cn("num", v > 0 ? "text-up" : v < 0 ? "text-down" : "")}>{sgn(v, 2, "%")}</span>;
  if (k === "entry_minute" || k === "minute") return <span className="num">{`${String(Math.floor(v / 60)).padStart(2, "0")}:${String(v % 60).padStart(2, "0")}`}</span>;
  if (k === "price_source") return ({ trade_print: "trade print", quote: "bid/ask quote", model: "model", stock: "stock" } as any)[v] || v;
  if (k === "exit_reason" || k === "kind" || k === "class") return String(v).replace(/_/g, " ");
  if (typeof v === "number") return <span className="num">{Number.isInteger(v) ? v : v.toFixed(Math.abs(v) < 10 ? 3 : 2)}</span>;
  return String(v);
}

function Observations({ runId, study, which, setWhich, onDay, day, onTrade }: { runId: string; study: boolean; which: string; setWhich: (w: string) => void; onDay: (d: string | null) => void; day: string | null; onTrade?: (id: number) => void }) {
  const q = useQuery({ queryKey: ["obs", runId, which], queryFn: () => api.get<any>(`/api/runs/${runId}/observations?which=${which}`), staleTime: 60_000 });
  const [sort, setSort] = useState<{ k: string; d: 1 | -1 }>({ k: "date", d: -1 });
  const rows = useMemo(() => {
    const rs = [...(q.data?.rows || [])];
    rs.sort((a, b) => (a[sort.k] === b[sort.k] ? 0 : (a[sort.k] ?? -Infinity) < (b[sort.k] ?? -Infinity) ? -sort.d : sort.d));
    return rs.slice(0, 1500);
  }, [q.data, sort]);
  const cols: string[] = q.data?.columns || [];
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1.5 px-5 py-3">
        {(study ? STUDY_FILTERS : BT_FILTERS).map(([k, l]) => <button key={k} className={cn("chip", which === k && "on")} onClick={() => setWhich(k)}>{l}</button>)}
        <span className="flex-1" />
        <span className="text-xs dim">{q.data ? `${q.data.count.toLocaleString()} ${q.data.title.toLowerCase()}` : ""}</span>
        <a className="btn quiet sm" href={`/api/runs/${runId}/export?what=${which === "skipped" ? "skipped" : which === "baseline" ? "baseline" : which === "excluded" ? "excluded" : "observations"}`} download><Download />CSV</a>
      </div>
      <div className="min-h-0 flex-1 overflow-auto border-t border-line">
        {q.isLoading ? <div className="p-5"><Skeleton className="h-40" /></div> : rows.length === 0 ? <Empty title="Nothing in this group" /> : (
          <table className="t" data-testid="observations">
            <thead><tr>{cols.map((c) => <th key={c} className={cn("cursor-pointer select-none", typeof rows[0]?.[c] === "number" && c !== "id" && "r")} onClick={() => setSort((s) => ({ k: c, d: s.k === c ? ((-s.d) as 1 | -1) : -1 }))}>{HEAD[c] || c.replace(/_/g, " ")}{sort.k === c ? (sort.d > 0 ? " ↑" : " ↓") : ""}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.date}-${r.id ?? i}`} className={cn("click", day === r.date && "sel")} onClick={() => (onTrade && typeof r.id === "number" && which !== "skipped" ? onTrade(r.id) : onDay(r.date))} title={onTrade ? "Replay this trade on the stored observations" : "Replay this session"}>
                  {cols.map((c) => <td key={c} className={cn(typeof r[c] === "number" && c !== "id" && "r", c === "reason" && "!whitespace-normal max-w-[520px]")}>{cell(c, r[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {q.data?.count > rows.length && <div className="shrink-0 border-t border-line px-5 py-1.5 text-xs dim">Showing {rows.length.toLocaleString()} of {q.data.count.toLocaleString()}. The CSV has every row.</div>}
    </div>
  );
}

// ----------------------------------------------------------------------------- statistics
function Hist({ h, color = C.info, fmt = (v: number) => v.toFixed(1) }: { h: any; color?: string; fmt?: (v: number) => string }) {
  const xs: number[] = h?.x || (h?.edges ? h.edges.slice(0, -1).map((e: number, i: number) => (e + h.edges[i + 1]) / 2) : []);
  if (!xs.length) return <div className="py-6 text-center text-xs dim">Not enough observations.</div>;
  return <EChart height={190} option={{ grid: { left: 40, right: 10, top: 10, bottom: 24 }, tooltip: { trigger: "axis" }, xAxis: { type: "category", data: xs.map(fmt) }, yAxis: { type: "value" },
    series: [{ type: "bar", data: h.counts.map((c: number, i: number) => ({ value: c, itemStyle: { color: xs[i] >= 0 ? color : C.down } })), barCategoryGap: "8%" }] }} />;
}

function GroupTable({ rows, study }: { rows: any[]; study: boolean }) {
  if (!rows?.length) return <div className="text-xs dim">No data.</div>;
  return (
    <table className="t">
      <thead><tr><th>Group</th><th className="r">{study ? "Sessions" : "Trades"}</th><th className="r">{study ? "How often" : "Net P&L"}</th><th className="r">{study ? "Avg end" : "Winners"}</th></tr></thead>
      <tbody>{rows.map((x) => (
        <tr key={x.key ?? x.year}><td>{x.key ?? x.year}</td><td className="r num">{x.n ?? x.trades}</td>
          <td className="r num">{study ? pp(x.probability) : <span className={x.pnl > 0 ? "text-up" : x.pnl < 0 ? "text-down" : ""}>{usd(x.pnl, 0)}</span>}</td>
          <td className="r num">{study ? sgn(x.avg_end_return_pct, 2, "%") : pp(x.win_rate)}</td></tr>
      ))}</tbody>
    </table>
  );
}

function Statistics({ r, openObs }: { r: any; openObs: (w: string) => void }) {
  const study = r.kind === "study";
  const o = r.outcomes || {};
  const cs = r.cost_sensitivity;
  const tod = r.charts?.time_of_day;
  const cum = r.charts?.cumulative;
  return (
    <div className="flex flex-col gap-4 p-5">
      <Section title={study ? "What happened on qualifying sessions" : "How the trades were closed"} sub="click a group to list it">
        {study ? (
          <>
            <Bar height={14} parts={[{ value: o.target_first || 0, className: "bg-up", label: "target reached first" }, { value: o.stop_first || 0, className: "bg-down", label: "stop reached first" }, { value: o.neither || 0, className: "bg-info", label: "neither" }, { value: o.ambiguous || 0, className: "bg-warn", label: "ambiguous" }]} />
            <div className="mt-3 flex flex-wrap gap-2">
              {[["target", "Target reached", o.target_first, "bg-up"], ["stop", "Stop reached", o.stop_first, "bg-down"], ["neither", "Neither", o.neither, "bg-info"], ["ambiguous", "Ambiguous order", o.ambiguous, "bg-warn"]].filter((x) => x[2]).map(([k, l, n, cl]) => (
                <button key={k as string} className="chip" onClick={() => openObs(k as string)}><span className={cn("dot", cl as string)} />{l} <span className="num">{n as number}</span></button>
              ))}
            </div>
          </>
        ) : (
          <div className="flex flex-wrap gap-2">{Object.entries(o).map(([k, n]) => <button key={k} className="chip" onClick={() => openObs(/target/.test(k) ? "target" : /stop/.test(k) ? "stop" : /time|hold|dte|expir/.test(k) ? "time" : "all")}>{k.replace(/_/g, " ")} <span className="num">{n as number}</span></button>)}</div>
        )}
      </Section>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title={study ? "Where sessions ended, from the starting price (%)" : "Profit and loss per trade ($)"}>
          <Hist h={study ? r.charts?.end_hist : r.charts?.distribution} color={C.up} fmt={(v) => (study ? v.toFixed(1) : Math.round(v).toString())} />
        </Section>
        <Section title="By year"><div className="max-h-[230px] overflow-auto"><GroupTable rows={r.by_year || []} study={study} /></div></Section>
      </div>
      {study && (cum || tod) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {cum && <Section title="How quickly the target is reached" sub={`${cum.sessions} sessions with intraday prices`}>
            <EChart height={190} option={{ grid: { left: 44, right: 10, top: 10, bottom: 24 }, tooltip: { trigger: "axis" }, xAxis: { type: "category", data: cum.minutes.map((m: number) => `${m}m`) }, yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
              series: [{ type: "line", data: cum.probability, showSymbol: false, lineStyle: { color: C.accent, width: 1.6 }, areaStyle: { color: "rgba(232,163,61,0.08)" } }] }} />
            <div className="text-xs dim">Share of those sessions that had reached the target, by minutes after the start.</div>
          </Section>}
          {tod && <Section title="When the target was reached"><Hist h={{ x: tod.minutes, counts: tod.hits }} color={C.accent} fmt={(v) => `${Math.round(v)}m`} /></Section>}
        </div>
      )}
      {r.regimes && (
        <div className="grid gap-4 lg:grid-cols-2">
          {Object.entries(r.regimes).filter(([, rows]) => (rows as any[])?.length).map(([k, rows]) => (
            <Section key={k} title={({ vix: "By volatility level (VIX)", trend: "By market trend", weekday: "By day of the week", time_of_day: "By entry time" } as any)[k] || k}><GroupTable rows={rows as any[]} study={study} /></Section>
          ))}
        </div>
      )}
      {cs?.rows?.length > 0 && (
        <Section title="If fills had been worse" sub="transaction-cost sensitivity">
          <table className="t"><thead><tr><th>Every fill worse by</th><th className="r">Net result</th><th className="r">Winners</th><th className="r">Per trade</th></tr></thead>
            <tbody>{cs.rows.map((x: any) => <tr key={x.extra_slippage_pct}><td>{x.extra_slippage_pct}%</td><td className={cn("r num", x.net_pnl > 0 ? "text-up" : "text-down")}>{usd(x.net_pnl, 2)}</td><td className="r num">{pp(x.win_rate)}</td><td className="r num">{usd(x.avg_trade, 2)}</td></tr>)}</tbody></table>
          <div className="mt-2 text-xs dim">{cs.method}{cs.breakeven_slippage_pct !== null && cs.breakeven_slippage_pct !== undefined ? ` The result reaches zero at about ${cs.breakeven_slippage_pct.toFixed(1)}% worse fills.` : ""}</div>
        </Section>
      )}
      {r.executability && r.mode === "trades" && (
        <Section title="Could the signals actually be executed?">
          <dl className="kv text-[13px]">
            <dt>Signals in the simulated window</dt><dd className="num">{r.executability.signals}</dd>
            <dt>Executed</dt><dd className="num">{r.executability.executed}{r.executability.executed_pct !== null ? ` (${r.executability.executed_pct.toFixed(0)}%)` : ""}</dd>
            <dt>Age of the print used</dt><dd>{r.executability.median_entry_print_age_seconds !== null ? `${r.executability.median_entry_print_age_seconds}s median, ${r.executability.max_entry_print_age_seconds}s at most` : "—"} <span className="dim">(refused beyond {r.executability.max_trade_age_seconds}s)</span></dd>
            <dt>How often the contracts traded</dt><dd>{r.executability.median_contract_minutes_traded_pct !== null ? `in ${r.executability.median_contract_minutes_traded_pct.toFixed(0)}% of the session's minutes (median)` : "—"}</dd>
            <dt>Marks using a stale print</dt><dd className="num">{r.executability.stale_marks}</dd>
            <dt>Exits delayed for lack of a print</dt><dd className="num">{r.executability.deferred_exits}</dd>
          </dl>
        </Section>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------- robustness
function Robustness({ r, onAsk }: { r: any; onAsk: (t: string) => void }) {
  const rb = r.robustness;
  const study = r.kind === "study";
  const tone = ({ supported: "good", mixed: "warn", none: "info", insufficient: "warn" } as const)[rb.level as "supported"] || "info";
  const part = (x: any) => (!x ? "—" : study ? `${pp(x.probability)} of ${x.n} (baseline ${pp(x.baseline_probability)})` : x.n ? `${usd(x.mean, 2)} per trade over ${x.n}` : "no trades");
  return (
    <div className="flex flex-col gap-4 p-5">
      <Callout tone={tone} title={rb.verdict} />
      <Section title="Checks">
        <div className="flex flex-col gap-3">
          {rb.checks.map((k: any, i: number) => (
            <div key={i} className="flex gap-3">
              <span className={cn("mt-0.5 grid size-5 shrink-0 place-items-center rounded-full", k.ok ? "bg-up/15 text-up" : "bg-down/15 text-down")}>{k.ok ? <Check className="size-3" strokeWidth={3} /> : <X className="size-3" strokeWidth={3} />}</span>
              <div><div className="text-[13.5px] font-medium">{k.name}</div><div className="text-[13px] soft"><T>{k.detail}</T></div></div>
            </div>
          ))}
        </div>
      </Section>
      <div className="grid gap-3 md:grid-cols-2">
        <StatTile label={`First 70% of history${rb.split_date ? ` (before ${rb.split_date})` : ""}`} value={study ? pp(rb.in_sample?.probability) : usd(rb.in_sample?.mean, 2)} sub={part(rb.in_sample)} />
        <StatTile label={<Term k="out of sample">{`Last 30% (from ${rb.split_date || "—"})`}</Term>} value={study ? pp(rb.out_of_sample?.probability) : usd(rb.out_of_sample?.mean, 2)} sub={part(rb.out_of_sample)} />
      </div>
      <Section title="How many things were tried" sub={`${rb.trials} in this notebook`}>
        <div className="text-[13px] soft">
          Every configuration tested in this notebook is counted, whatever it showed. Trying many and keeping the best one makes luck look like skill, so the significance threshold is tightened accordingly
          {rb.adjusted_p_value !== null && rb.adjusted_p_value !== undefined ? <> (adjusted <Term k="p-value" />: {rb.adjusted_p_value < 0.001 ? "< 0.001" : rb.adjusted_p_value.toFixed(3)})</> : null}.
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button className="btn sm" onClick={() => onAsk("Test whether the results hold up out of sample.")}>Run it on two separate periods</button>
          {!study && <button className="btn sm" onClick={() => onAsk("Include commissions and conservative slippage.")}>Apply conservative costs</button>}
          <button className="btn sm" onClick={() => onAsk("Show me whether this performs differently on Mondays.")}>Check by day of the week</button>
        </div>
      </Section>
    </div>
  );
}

// ----------------------------------------------------------------------------- methodology
function Methodology({ r, runId }: { r: any; runId: string }) {
  const d = r.data;
  const it = r.interpretation;
  const [json, setJson] = useState(false);
  return (
    <div className="flex flex-col gap-4 p-5">
      <Section title="Interpretation" sub={it.interpreter ? `read by ${it.interpreter}` : undefined} right={<button className="btn quiet sm" onClick={() => setJson((j) => !j)}>{json ? "Hide" : "Show"} structured configuration</button>}>
        <div className={cn("grid gap-5", json && "lg:grid-cols-2")}>
          <div className="flex flex-col gap-2">
            {(it.items || []).map((x: any, i: number) => {
              const [plain, expr] = String(x.value).split("   [");
              return (
                <div key={i} className="flex items-baseline gap-2.5 text-[13px]">
                  <span className="w-28 shrink-0 text-xs dim">{x.label}</span>
                  <span className="min-w-0 flex-1"><T>{plain}</T>{expr && <code className="ml-2 rounded bg-bg-3 px-1.5 py-0.5 font-mono text-[11.5px] text-fg-2">{expr.replace(/\]$/, "")}</code>}</span>
                  <StatusChip status={x.status} />
                </div>
              );
            })}
            {(it.items || []).length === 0 && <div className="text-[13px] dim">This run was started from a saved configuration, not from a question.</div>}
          </div>
          {json && <pre className="max-h-[420px] overflow-auto rounded-lg bg-bg-2 p-3 font-mono text-[11.5px] leading-[1.5] text-fg-2">{JSON.stringify(r.reproduce.config, null, 1)}</pre>}
        </div>
        {(it.assumptions || []).length > 0 && (
          <div className="mt-4 border-t border-line pt-3">
            <div className="eyebrow mb-2">Assumptions made</div>
            <ul className="flex flex-col gap-1.5 text-[13px]">{it.assumptions.map((a: any, i: number) => <li key={i}><span className="font-medium">{String(a.field).replace(/_/g, " ")}: {String(a.value)}</span> <span className="dim">— {a.reason}</span></li>)}</ul>
          </div>
        )}
      </Section>
      <Section title="Data sources and coverage">
        <div className="flex flex-col gap-2.5">
          {d.sources.map((s: any, i: number) => (
            <div key={i} className="text-[13px]">
              <div><span className="font-medium">{s.what}</span> <span className="soft">— {s.detail}</span></div>
              {s.limitations.slice(0, 4).map((l: string, j: number) => <div key={j} className="text-xs dim">· {l}</div>)}
            </div>
          ))}
        </div>
        {Object.keys(d.by_symbol || {}).length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-line">
            <table className="t"><thead><tr><th>Symbol</th><th>Used for</th><th>Needs</th><th>Stored</th><th className="r">Usable sessions</th><th>Gaps</th></tr></thead>
              <tbody>{Object.entries(d.by_symbol).map(([sym, v]: any) => (
                <tr key={sym}><td className="font-medium">{sym}</td><td>{v.role}</td><td>{v.needs}</td><td className="num">{v.stored.first ? `${v.stored.first} → ${v.stored.last}` : "nothing"}</td><td className="r num">{v.usable_sessions.toLocaleString()}</td>
                  <td className="!whitespace-normal max-w-[380px] text-xs dim">{v.issues.length ? v.issues.map((x: any) => `${x.sessions.toLocaleString()}: ${x.reason}`).join("; ") : "none"}</td></tr>
              ))}</tbody></table>
          </div>
        )}
        {(d.downloads || []).length > 0 && <div className="mt-3"><div className="eyebrow mb-1.5">Downloaded automatically (free)</div>{d.downloads.map((x: any, i: number) => <div key={i} className={cn("text-xs", x.error ? "text-down" : "soft")}>· {x.label} {x.dataset} for {(x.symbols || []).join(", ")} — {x.why}{x.error ? ` (failed: ${x.error})` : ""}</div>)}</div>}
        {(d.unavailable || []).length > 0 && <div className="mt-3"><div className="eyebrow mb-1.5">Not available from a free source</div>{d.unavailable.map((x: any, i: number) => <div key={i} className="text-xs soft">· {x.detail}</div>)}</div>}
        {(d.awaiting_approval || []).length > 0 && <div className="mt-3"><div className="eyebrow mb-1.5">Could be expanded (large download, not run without approval)</div>{d.awaiting_approval.map((x: any, i: number) => <div key={i} className="text-xs soft">· {x.label}: {x.why}</div>)}</div>}
      </Section>
      <Section title="How the run went" open={false}><Timeline runId={runId} /></Section>
      <Section title="Reproduce this result" open={false}>
        <dl className="kv text-[13px]">
          <dt>Run</dt><dd className="num">{r.reproduce.run_id}</dd><dt>Engine</dt><dd className="num">{r.reproduce.engine_version}</dd><dt>Configuration hash</dt><dd className="num">{r.reproduce.config_hash}</dd>
          <dt>Dataset versions</dt><dd className="font-mono text-[11.5px] break-all dim">{JSON.stringify(r.reproduce.dataset_versions)}</dd>
        </dl>
        <div className="mt-2 text-xs dim">The same configuration on the same dataset versions and engine gives the same result. No language model produced any number in this report.</div>
      </Section>
    </div>
  );
}

function Raw({ r, runId, sid, trials }: { r: any; runId: string; sid?: string; trials: number }) {
  const q = `${sid ? `&sid=${sid}` : ""}&trials=${trials}`;
  const links = [["Report (HTML)", `/api/runs/${runId}/report?format=html${q}`, FileText], ["Report (Markdown)", `/api/runs/${runId}/report?format=md${q}`, Download], ["Everything (JSON)", `/api/runs/${runId}/report?format=json${q}`, Download],
    [r.kind === "study" ? "Qualifying sessions (CSV)" : "Trade log (CSV)", `/api/runs/${runId}/export?what=observations`, Download], ...(r.kind === "study" ? [["Every evaluated session (CSV)", `/api/runs/${runId}/export?what=baseline`, Download]] : [["Skipped signals (CSV)", `/api/runs/${runId}/export?what=skipped`, Download]]),
    ["Configuration (JSON)", `/api/runs/${runId}/export?what=config`, Download]] as [string, string, any][];
  return (
    <div className="flex flex-col gap-4 p-5">
      <Section title="Export">
        <div className="grid gap-2 sm:grid-cols-2">{links.map(([l, h, I]) => <a key={h} className="btn !justify-start" href={h} target={h.includes("format=html") ? "_blank" : undefined} download={h.includes("format=html") ? undefined : true} rel="noreferrer"><I />{l}</a>)}</div>
      </Section>
      <Section title="Every statistic" sub="as stored">
        <table className="t"><thead><tr><th>Statistic</th><th className="r">Value</th><th className="r">Exact</th></tr></thead>
          <tbody>{r.facts.map((x: Fact) => <tr key={x.key}><td>{x.label}</td><td className="r num">{x.shown}</td><td className="r num dim">{x.value}</td></tr>)}</tbody></table>
      </Section>
      <Section title="Configuration" open={false}><pre className="max-h-[520px] overflow-auto rounded-lg bg-bg-2 p-3 font-mono text-[11.5px] leading-[1.5] text-fg-2">{JSON.stringify(r.reproduce.config, null, 1)}</pre></Section>
    </div>
  );
}
