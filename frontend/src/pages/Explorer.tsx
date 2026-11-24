/* Market Explorer: ask how the underlying behaves (event studies), look at stored
   prices and option chains, and run the option-quote research templates. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { CandlestickChart, Database, FlaskRound, Loader2, Microscope, Play, Plus, Table2, Timer, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { LWChart, dayTime, etChartTime } from "@/components/charts/LWChart";
import { Empty, ErrorBox, Loading, PageHeader, Panel, SyntheticBadge } from "@/components/common";
import { ExprInput } from "@/components/research/ExprInput";
import { SessionReplay, StudyReport } from "@/components/research/StudyReport";
import { Button, Field, Input, NumInput, Segmented, Select, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui";
import { api, queryClient, useMeta, useRun } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { ChainViewer } from "@/pages/DataPage";
import { EntryTiming, Queries, Sql } from "@/pages/Research";

export function useCoverage() {
  return useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage") });
}

export default function Explorer() {
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Market Explorer" sub={<span className="text-2xs text-muted">how the market has behaved, measured directly from stored prices</span>} />
      <Tabs defaultValue="study" className="flex min-h-0 flex-1 flex-col">
        <TabsList>
          <TabsTrigger value="study"><FlaskRound className="size-3" /> Event study</TabsTrigger>
          <TabsTrigger value="chart"><CandlestickChart className="size-3" /> Prices</TabsTrigger>
          <TabsTrigger value="chain"><Table2 className="size-3" /> Option chain</TabsTrigger>
          <TabsTrigger value="quotes"><Microscope className="size-3" /> Option-quote studies</TabsTrigger>
          <TabsTrigger value="timing"><Timer className="size-3" /> Entry-time sensitivity</TabsTrigger>
          <TabsTrigger value="sql"><Database className="size-3" /> SQL</TabsTrigger>
        </TabsList>
        <TabsContent value="study" className="min-h-0 flex-1"><EventStudy /></TabsContent>
        <TabsContent value="chart" className="min-h-0 flex-1"><Prices /></TabsContent>
        <TabsContent value="chain" className="min-h-0 flex-1 overflow-auto"><ChainTab /></TabsContent>
        <TabsContent value="quotes" className="min-h-0 flex-1"><Queries /></TabsContent>
        <TabsContent value="timing" className="min-h-0 flex-1 overflow-auto"><EntryTiming /></TabsContent>
        <TabsContent value="sql" className="min-h-0 flex-1"><Sql /></TabsContent>
      </Tabs>
    </div>
  );
}

const PRESETS: { label: string; cond: string[]; entry: string; target: number | null; stop: number | null; measure?: "touch" | "close"; horizon?: any; targetExpr?: string; stopExpr?: string }[] = [
  { label: "Bounce after a 3% down day", cond: ["previous_session.return_pct <= -3"], entry: "open", target: 1, stop: null },
  { label: "Gap down, then recover half the gap by noon", cond: ["gap_pct <= -0.5"], entry: "open", target: null, stop: null, targetExpr: "open + 0.5 * (prev_close - open)", horizon: { kind: "time", time: "12:00" } },
  { label: "Yesterday's high before yesterday's low", cond: [], entry: "open", target: null, stop: null, targetExpr: "prev_high", stopExpr: "prev_low" },
  { label: "Three down days in a row, higher five sessions later", cond: ["consecutive_down_days() >= 3"], entry: "close", target: 0, stop: null, measure: "close", horizon: { kind: "sessions", sessions: 5 } },
  { label: "+1% before −0.5% when VIX is above 25", cond: ["vix_prev_close > 25"], entry: "open", target: 1, stop: -0.5 },
];

function EventStudy() {
  const meta = useMeta().data as any;
  const cov = useCoverage();
  const symbols = (cov.data || []).filter((r) => r.daily || Object.keys(r.intraday || {}).length).map((r) => r.symbol);
  const [f, setF] = useState<any>({ symbol: "", start: "", end: "", entry_time: "open", conds: ["previous_session.return_pct <= -3"], target: 1 as number | null, stop: null as number | null, targetExpr: "", stopExpr: "",
    measure: "touch", reference: "entry", horizon: { kind: "session", time: "12:00", sessions: 1 } });
  const [valid, setValid] = useState<Record<number, boolean>>({});
  const [runId, setRunId] = useState<string | null>(null);
  const [pick, setPick] = useState<any>(null);
  const sym = f.symbol || meta?.default_symbol || symbols[0] || "SPY";
  const run = useRun(runId || undefined, true);
  const recent = useQuery({ queryKey: ["runs", "study"], queryFn: () => api.get<any[]>("/api/study/runs?limit=25") });
  useEffect(() => { if (run.data?.status === "done") queryClient.invalidateQueries({ queryKey: ["runs", "study"] }); setPick(null); }, [run.data?.status, runId]);
  const config = () => ({
    symbol: sym, start: f.start || null, end: f.end || null, entry_time: f.entry_time, reference: f.reference, measure: f.measure,
    condition: { id: "root", kind: "group", op: "and", children: f.conds.filter((c: string) => c.trim()).map((c: string, i: number) => ({ id: `c${i}`, kind: "cond", type: "expr", params: { expr: c } })) },
    target: f.targetExpr.trim() ? { kind: "expr", expr: f.targetExpr.trim() } : f.target === null ? null : { kind: "pct", value: f.target },
    stop: f.stopExpr.trim() ? { kind: "expr", expr: f.stopExpr.trim() } : f.stop === null ? null : { kind: "pct", value: f.stop },
    horizon: f.horizon, label: `${sym}: ${f.conds.filter((c: string) => c.trim()).join(" and ") || "all sessions"}`,
  });
  const go = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/study/run", { config: config() }), onSuccess: (r) => setRunId(r.id), onError: (e: any) => toast.error(e.message) });
  const preset = (p: (typeof PRESETS)[number]) => setF({ ...f, conds: p.cond.length ? p.cond : [""], entry_time: p.entry, target: p.target, stop: p.stop, targetExpr: p.targetExpr || "", stopExpr: p.stopExpr || "", measure: p.measure || "touch", horizon: { kind: "session", time: "12:00", sessions: 1, ...(p.horizon || {}) } });
  const row = cov.data?.find((r) => r.symbol === sym);
  const res = run.data?.status === "done" ? run.data.result : null;
  const anyBad = f.conds.some((c: string, i: number) => c.trim() && valid[i] === false);
  return (
    <div className="grid h-full grid-cols-[330px_1fr]">
      <div className="min-h-0 overflow-auto border-r border-line bg-bg-1">
        <div className="space-y-2.5 p-3">
          <div className="grid grid-cols-2 gap-2">
            <Field label="Symbol"><Select value={sym} onChange={(v) => setF({ ...f, symbol: v })} options={symbols.length ? symbols : [sym]} /></Field>
            <Field label="Start measuring at" hint="New York time. 'open' and 'close' follow half days."><Input className="num" value={f.entry_time} onChange={(e) => setF({ ...f, entry_time: e.target.value.trim() })} placeholder="open, close or HH:MM" /></Field>
            <Field label="From"><Input sz="sm" type="date" className="num" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
            <Field label="To"><Input sz="sm" type="date" className="num" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          </div>
          {row && <div className="text-2xs leading-relaxed text-muted">Stored: {row.daily ? `daily ${row.daily.first} → ${row.daily.last}` : "no daily bars"}{Object.entries(row.intraday || {}).map(([k, v]: any) => ` · ${k} from ${v.first}`).join("")}. Blank dates use everything.</div>}
          <div>
            <div className="mb-1 flex items-center justify-between"><span className="label">On sessions where (all must hold)</span><Button size="sm" variant="ghost" onClick={() => setF({ ...f, conds: [...f.conds, ""] })}><Plus /> Rule</Button></div>
            <div className="flex flex-col gap-1.5">
              {f.conds.map((c: string, i: number) => (
                <div key={i} className="flex items-start gap-1">
                  <ExprInput className="flex-1" value={c} onChange={(v) => setF({ ...f, conds: f.conds.map((x: string, j: number) => (j === i ? v : x)) })} onValid={(ok) => setValid((m) => (m[i] === ok ? m : { ...m, [i]: ok }))} />
                  <Button size="iconSm" variant="ghost" className="mt-0.5" onClick={() => setF({ ...f, conds: f.conds.filter((_: string, j: number) => j !== i) })}><X /></Button>
                </div>
              ))}
              {f.conds.length === 0 && <div className="text-2xs text-faint">No rule: every session is measured (this is the baseline).</div>}
            </div>
          </div>
          <div>
            <div className="label mb-1">How often does the price…</div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Target" hint="Signed percent from the reference: +1 = rises 1%, −1 = falls 1%. Or give a price level below."><NumInput allowNull step={0.25} suffix="%" value={f.target} onChange={(v) => setF({ ...f, target: v })} placeholder="none" /></Field>
              <Field label="Stop (optional)" hint="If set, the result is 'target before stop'."><NumInput allowNull step={0.25} suffix="%" value={f.stop} onChange={(v) => setF({ ...f, stop: v })} placeholder="none" /></Field>
              <Field label="…or target level" className="col-span-2"><Input sz="sm" className="font-mono text-2xs" placeholder="e.g. prev_high   or   open + 0.5 * (prev_close - open)" value={f.targetExpr} onChange={(e) => setF({ ...f, targetExpr: e.target.value })} /></Field>
              <Field label="…or stop level" className="col-span-2"><Input sz="sm" className="font-mono text-2xs" placeholder="e.g. prev_low" value={f.stopExpr} onChange={(e) => setF({ ...f, stopExpr: e.target.value })} /></Field>
              <Field label="Counts when"><Select value={f.measure} onChange={(v) => setF({ ...f, measure: v })} options={[{ value: "touch", label: "price touches it" }, { value: "close", label: "horizon ends beyond it" }]} /></Field>
              <Field label="Measured from"><Select value={f.reference} onChange={(v) => setF({ ...f, reference: v })} options={[{ value: "entry", label: "the starting price" }, { value: "open", label: "the session open" }, { value: "prev_close", label: "the previous close" }]} /></Field>
              <Field label="Within"><Select value={f.horizon.kind} onChange={(v) => setF({ ...f, horizon: { ...f.horizon, kind: v } })} options={[{ value: "session", label: "that session" }, { value: "time", label: "until a time (ET)" }, { value: "sessions", label: "N sessions" }]} /></Field>
              {f.horizon.kind === "time" ? <Field label="Until"><Input className="num" value={f.horizon.time} onChange={(e) => setF({ ...f, horizon: { ...f.horizon, time: e.target.value } })} /></Field>
                : f.horizon.kind === "sessions" ? <Field label="Sessions"><NumInput min={1} max={60} value={f.horizon.sessions} onChange={(v) => setF({ ...f, horizon: { ...f.horizon, sessions: v ?? 1 } })} /></Field> : <div />}
            </div>
          </div>
          <Button variant="primary" className="w-full" disabled={go.isPending || anyBad} onClick={() => go.mutate()}>{go.isPending ? <Loader2 className="animate-spin" /> : <Play />} Run study</Button>
          <div className="text-2xs leading-relaxed text-faint">A study measures the underlying only. Sessions whose rules cannot be evaluated are excluded and listed, never counted as "no".</div>
        </div>
        <div className="border-t border-line">
          <div className="label px-3 pb-1 pt-2">Starting points</div>
          {PRESETS.map((p) => <button key={p.label} onClick={() => preset(p)} className="block w-full px-3 py-1 text-left text-xs text-fg-2 hover:bg-bg-2 hover:text-fg">{p.label}</button>)}
        </div>
        {(recent.data || []).length > 0 && (
          <div className="border-t border-line">
            <div className="label px-3 pb-1 pt-2">Recent studies</div>
            {(recent.data || []).slice(0, 12).map((r) => (
              <button key={r.id} onClick={() => setRunId(r.id)} className={cn("flex w-full items-center gap-2 px-3 py-1 text-left hover:bg-bg-2", runId === r.id && "bg-bg-3")}>
                <span className="min-w-0 flex-1 truncate text-xs text-fg-2">{r.name}</span>
                <span className="num text-2xs text-accent">{r.summary?.probability !== null && r.summary?.probability !== undefined ? `${(r.summary.probability * 100).toFixed(0)}%` : ""}</span>
                <span className="num text-2xs text-faint">{ago(r.created_at)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="min-h-0 overflow-auto">
        {!runId ? <Empty title="Build a question on the left" icon={FlaskRound}>Pick a condition, an outcome and a horizon. The report shows the historical frequency with its confidence interval, the baseline on all sessions, results by year and regime, and every qualifying session.</Empty>
          : run.isLoading || (run.data && ["queued", "running"].includes(run.data.status)) ? <Loading label={run.data?.message || "Running study"} />
            : run.data?.status === "error" ? <ErrorBox className="m-3" error={run.data.error?.split("\n")[0]} />
              : res ? (
                <div>
                  <StudyReport res={res} onPick={setPick} selected={pick?.date} />
                  {pick && <Panel className="border-0 border-t" title="Session replay" dense><SessionReplay symbol={res.symbol} row={pick} height={280} /></Panel>}
                </div>
              ) : null}
      </div>
    </div>
  );
}

function Prices() {
  const meta = useMeta().data as any;
  const cov = useCoverage();
  const symbols = (cov.data || []).filter((r) => r.daily || Object.keys(r.intraday || {}).length);
  const [sym, setSym] = useState("");
  const [tf, setTf] = useState("1d");
  const s = sym || meta?.default_symbol || symbols[0]?.symbol || "SPY";
  const q = useQuery({ queryKey: ["candles", s, tf], queryFn: () => api.get<any>(`/api/data/candles/${s}?tf=${tf}&limit=6000`) });
  const row = symbols.find((r) => r.symbol === s);
  const series = useMemo(() => {
    const bars: number[][] = q.data?.bars || [];
    const t = (b: number[]) => (tf === "1d" ? dayTime(new Date(b[0] * 1000).toISOString().slice(0, 10)) : etChartTime(b[0]));
    return [
      { type: "candles" as const, data: bars.map((b) => ({ time: t(b), open: b[1], high: b[2], low: b[3], close: b[4] })) },
      { type: "histogram" as const, pane: 1, data: bars.map((b) => ({ time: t(b), value: b[5] || 0, color: b[4] >= b[1] ? "rgba(38,166,154,0.5)" : "rgba(239,83,80,0.5)" })), options: { priceFormat: { type: "volume" } } },
    ];
  }, [q.data, tf]);
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-line px-2.5 py-1.5">
        <Select sz="sm" className="w-28" value={s} onChange={setSym} options={symbols.map((r) => r.symbol)} />
        <Segmented value={tf} onChange={setTf} options={["1d", "60m", "30m", "15m", "5m", "1m"].map((v) => ({ value: v, label: v }))} />
        <SyntheticBadge on={row?.synthetic} />
        <span className="num text-2xs text-muted">{q.data?.bars?.length ?? 0} bars{q.data?.source_resolution && q.data.source_resolution !== tf ? ` · built from stored ${q.data.source_resolution} bars` : ""} · New York time</span>
        {q.data?.note && <span className="text-2xs text-warn">{q.data.note}</span>}
        <span className="flex-1" />
        {row && <span className="num text-2xs text-faint">stored: {row.daily ? `1d ${row.daily.first}→${row.daily.last}` : ""}{Object.entries(row.intraday || {}).map(([k, v]: any) => ` · ${k} ${v.first}→`).join("")}</span>}
      </div>
      <div className="min-h-0 flex-1">{q.isLoading ? <Loading /> : q.error ? <ErrorBox className="m-2" error={q.error} /> : q.data?.bars?.length ? <LWChart height="100%" timeVisible={tf !== "1d"} series={series} /> : <Empty title={`No ${tf} bars stored for ${s}`}>{q.data?.note || "Download them under Data."}</Empty>}</div>
    </div>
  );
}

function ChainTab() {
  const cov = useCoverage();
  const withOpts = (cov.data || []).filter((r) => r.options);
  const [sym, setSym] = useState("");
  const s = sym || withOpts.find((r) => !r.synthetic)?.symbol || withOpts[0]?.symbol;
  if (cov.isLoading) return <Loading />;
  if (!s) return <Empty title="No option chains stored" icon={Table2}>Download free end-of-day chains for SPY, QQQ or IWM under Data → Get free data. Nothing is synthesised in their place.</Empty>;
  const row = withOpts.find((r) => r.symbol === s);
  return (
    <div>
      <div className="flex items-center gap-2 border-b border-line px-2.5 py-1.5">
        <Select sz="sm" className="w-28" value={s} onChange={setSym} options={withOpts.map((r) => r.symbol)} />
        <span className="num text-2xs text-muted">{row.options.resolution === "EOD" ? "end-of-day snapshot (16:00 ET) per session" : `${row.options.resolution} snapshots`} · {row.options.quote_type} · {row.options.first} → {row.options.last} · via {row.options.provider}{row.options.license ? ` · ${row.options.license}` : ""}</span>
      </div>
      <ChainViewer key={s} sym={s} defaultTime={row.options.resolution === "EOD" ? "16:00" : "10:00"} />
    </div>
  );
}
