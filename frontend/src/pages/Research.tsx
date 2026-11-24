import { useMutation, useQuery } from "@tanstack/react-query";
import { Database, Microscope, Play, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { DataTable, Empty, ErrorBox, JobBar, Loading, PageHeader, Panel, type Col } from "@/components/common";
import { Button, Field, Input, NumInput, Select, Switch, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui";
import { api, useJob, useMeta, useRun } from "@/lib/api";
import { cn, num, pct } from "@/lib/format";
import { useBuilder } from "@/lib/store";
import type { ChartSpec } from "@/lib/types";

type Filter = { field: string; op: string; value: any };

export function Queries() {
  const meta = useMeta().data;
  const catalog = useQuery({ queryKey: ["catalog"], queryFn: () => api.get<any[]>("/api/data/catalog") });
  // these templates read intraday option quotes; end-of-day chains cannot answer them
  const usable = (catalog.data || []).filter((c) => c.options && c.options.resolution !== "EOD");
  const symbols = usable.map((c) => c.symbol);
  const [tpl, setTpl] = useState("intraday_move");
  const [p, setP] = useState<Record<string, any>>({ symbol: "", thresholds: [0.25, 0.5, 1.0], structure: "straddle", ref_time: "09:45", group_by_vix: true, recenter: true, window: 21, event_types: ["FOMC", "CPI", "NFP"], target: "oc_ret", filters: [] as Filter[] });
  const sym = p.symbol && symbols.includes(p.symbol) ? p.symbol : symbols[0];
  const run = useMutation({ mutationFn: () => api.post<any>("/api/research/run", { template: tpl, params: Object.fromEntries(Object.entries({ ...p, symbol: sym }).filter(([k]) => meta?.research[tpl]?.params.includes(k))) }) });
  const t = meta?.research[tpl];
  const has = (k: string) => t?.params.includes(k);
  const filters: Filter[] = p.filters;
  const setF = (f: Filter[]) => setP({ ...p, filters: f });
  if (catalog.isSuccess && symbols.length === 0) {
    return <Empty icon={Microscope} title="These studies need intraday option quotes">Straddle decay, intraday implied-volatility behaviour and the other templates here read option quotes through the session. Only end-of-day chains (or none) are stored, so they cannot be answered - and nothing is estimated in their place. Use the Event study tab for questions about the underlying, or load intraday option data under Data.</Empty>;
  }
  return (
    <div className="grid h-full grid-cols-[300px_1fr]">
      <div className="min-h-0 overflow-auto border-r border-line bg-bg-1">
        <div className="label px-3 pt-2.5 pb-1">Question templates</div>
        {Object.entries(meta?.research || {}).filter(([k]) => k !== "entry_timing").map(([k, v]) => (
          <button key={k} onClick={() => { setTpl(k); run.reset(); }} className={cn("block w-full border-l-2 px-3 py-1.5 text-left", tpl === k ? "border-accent bg-bg-3" : "border-transparent hover:bg-bg-2")}>
            <div className="text-xs text-fg">{v.label}</div><div className="text-2xs text-muted leading-snug">{v.question}</div>
          </button>
        ))}
        <div className="border-t border-line mt-2 p-3 space-y-2">
          <div className="label">Parameters</div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Symbol"><Select value={sym || ""} onChange={(v) => setP({ ...p, symbol: v })} options={usable.map((c) => ({ value: c.symbol, label: c.meta?.synthetic ? `${c.symbol} (simulated)` : c.symbol }))} /></Field>
            <div />
            <Field label="From"><Input sz="sm" type="date" className="num" value={p.start || ""} onChange={(e) => setP({ ...p, start: e.target.value || undefined })} /></Field>
            <Field label="To"><Input sz="sm" type="date" className="num" value={p.end || ""} onChange={(e) => setP({ ...p, end: e.target.value || undefined })} /></Field>
            {has("thresholds") && <Field label="Thresholds %" className="col-span-2"><Input sz="sm" className="num" value={p.thresholds.join(", ")} onChange={(e) => setP({ ...p, thresholds: e.target.value.split(",").map((x) => Number(x.trim())).filter((x) => Number.isFinite(x) && x > 0) })} /></Field>}
            {has("structure") && <Field label="Structure"><Select value={p.structure} onChange={(v) => setP({ ...p, structure: v })} options={["straddle", "call", "put"]} /></Field>}
            {has("ref_time") && <Field label="Reference time"><Input sz="sm" className="num" value={p.ref_time} onChange={(e) => setP({ ...p, ref_time: e.target.value })} /></Field>}
            {has("group_by_vix") && <label className="col-span-2 flex items-center gap-2 text-xs text-fg-2"><Switch checked={p.group_by_vix} onChange={(v) => setP({ ...p, group_by_vix: v })} /> Split by VIX regime</label>}
            {has("recenter") && <label className="col-span-2 flex items-center gap-2 text-xs text-fg-2"><Switch checked={p.recenter} onChange={(v) => setP({ ...p, recenter: v })} /> Rolling ATM strike (isolate time decay)</label>}
            {has("window") && <Field label="RV window (sessions)"><NumInput sz="sm" value={p.window} onChange={(v) => setP({ ...p, window: v ?? 21 })} /></Field>}
            {has("event_types") && <Field label="Events" className="col-span-2"><Input sz="sm" value={p.event_types.join(",")} onChange={(e) => setP({ ...p, event_types: e.target.value.split(",").map((x) => x.trim().toUpperCase()).filter(Boolean) })} /></Field>}
            {has("target") && <Field label="Outcome" className="col-span-2"><Select value={p.target} onChange={(v) => setP({ ...p, target: v })} options={Object.entries(meta?.research_targets || {}).map(([value, label]) => ({ value, label }))} /></Field>}
          </div>
          {has("filters") && (
            <div className="space-y-1">
              <div className="flex items-center justify-between"><span className="label">Conditions (known at the open)</span><Button size="sm" variant="ghost" onClick={() => setF([...filters, { field: "gap_pct", op: ">", value: 0.5 }])}><Plus /> Add</Button></div>
              {filters.map((f, i) => (
                <div key={i} className="flex items-center gap-1">
                  <Select sz="sm" className="w-28" value={f.field} onChange={(v) => setF(filters.map((x, j) => j === i ? { field: v, op: v === "weekday" || v === "event" ? (v === "event" ? "is" : "in") : v === "trend" ? "=" : ">", value: v === "weekday" ? ["Mon"] : v === "event" ? ["FOMC", "CPI", "NFP"] : v === "trend" ? "bull" : 0 } : x))}
                    options={Object.entries(meta?.research_filters || {}).map(([value, label]) => ({ value, label }))} />
                  {f.field === "weekday" ? <Input sz="sm" className="flex-1" value={(f.value || []).join(",")} onChange={(e) => setF(filters.map((x, j) => j === i ? { ...x, value: e.target.value.split(",").map((s) => s.trim()) } : x))} />
                    : f.field === "event" ? <><Select sz="sm" className="w-16" value={f.op} onChange={(v) => setF(filters.map((x, j) => j === i ? { ...x, op: v } : x))} options={["is", "not"]} /><Input sz="sm" className="flex-1" value={(f.value || []).join(",")} onChange={(e) => setF(filters.map((x, j) => j === i ? { ...x, value: e.target.value.split(",").map((s) => s.trim().toUpperCase()) } : x))} /></>
                    : f.field === "trend" ? <Select sz="sm" className="flex-1" value={f.value} onChange={(v) => setF(filters.map((x, j) => j === i ? { ...x, value: v } : x))} options={["bull", "bear", "neutral"]} />
                    : <><Select sz="sm" className="w-14" value={f.op} onChange={(v) => setF(filters.map((x, j) => j === i ? { ...x, op: v } : x))} options={[">", "<", ">=", "<="]} /><NumInput sz="sm" className="flex-1" step={0.1} value={f.value} onChange={(v) => setF(filters.map((x, j) => j === i ? { ...x, value: v ?? 0 } : x))} /></>}
                  <Button size="iconSm" variant="ghost" onClick={() => setF(filters.filter((_, j) => j !== i))}><Trash2 /></Button>
                </div>
              ))}
            </div>
          )}
          <Button variant="primary" className="w-full" onClick={() => run.mutate()} disabled={run.isPending}><Play /> {run.isPending ? "Querying…" : "Run query"}</Button>
        </div>
      </div>
      <div className="min-h-0 overflow-auto">
        {run.isPending && <Loading label="Scanning stored data" />}
        {run.error && <ErrorBox className="m-3" error={run.error} />}
        {!run.data && !run.isPending && !run.error && <Empty icon={Microscope} title={t?.question || "Pick a question"}>Results show sample sizes and 95% confidence intervals. Conditions only use information available at the open; outcomes are realised later.</Empty>}
        {run.data && <ResearchResult r={run.data} />}
      </div>
    </div>
  );
}

function ResearchResult({ r }: { r: any }) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-3 border-b border-line bg-bg-1 px-3 h-9">
        <span className="text-sm font-medium">{r.title}</span>
        <span className="num text-2xs text-muted">n = {r.sample_size}</span>
        {r.filters?.length > 0 && <span className="text-2xs text-info">{r.filters.join(" · ")}</span>}
      </div>
      {r.stats && <StatsBlock stats={r.stats} />}
      <div className="grid grid-cols-2 gap-px bg-line">
        {r.charts.map((c: ChartSpec, i: number) => <Panel key={i} className={cn("border-0", r.charts.length === 1 && "col-span-2")} title={c.title}><SpecChart spec={c} /></Panel>)}
      </div>
      {r.tables?.map((t: any, i: number) => <ResearchTable key={i} t={t} />)}
      {r.notes?.length > 0 && <ul className="px-3 py-2 text-2xs text-faint list-disc list-inside space-y-0.5">{r.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>}
    </div>
  );
}

function StatsBlock({ stats }: { stats: any }) {
  const desc = Object.entries(stats).filter(([, v]) => v && typeof v === "object" && "n" in (v as any));
  const scalars = Object.entries(stats).filter(([, v]) => typeof v === "number" || v === null);
  return (
    <div className="border-b border-line">
      {desc.length > 0 && (
        <table className="tbl">
          <thead><tr><th>Series</th><th className="r">n</th><th className="r">mean</th><th className="r">95% CI (bootstrap)</th><th className="r">median</th><th className="r">std</th><th className="r">p5</th><th className="r">p95</th><th className="r">min</th><th className="r">max</th></tr></thead>
          <tbody>{desc.map(([k, v]: any) => (
            <tr key={k}><td className="text-fg">{k.replace(/_/g, " ")}</td><td className="r num">{v.n}</td><td className="r num">{num(v.mean, 3)}</td><td className="r num text-fg-2">{v.mean_ci95 ? `[${num(v.mean_ci95[0], 3)}, ${num(v.mean_ci95[1], 3)}]` : "—"}</td>
              <td className="r num">{num(v.median, 3)}</td><td className="r num">{num(v.std, 3)}</td><td className="r num">{num(v.p05, 3)}</td><td className="r num">{num(v.p95, 3)}</td><td className="r num text-fg-2">{num(v.min, 2)}</td><td className="r num text-fg-2">{num(v.max, 2)}</td></tr>
          ))}</tbody>
        </table>
      )}
      {scalars.length > 0 && <div className="flex flex-wrap gap-5 px-3 py-2 text-xs">{scalars.map(([k, v]) => <span key={k}><span className="text-muted">{k.replace(/_/g, " ")}</span> <span className="num">{num(v as number, 4)}</span></span>)}</div>}
    </div>
  );
}

function ResearchTable({ t }: { t: any }) {
  const rows = t.rows || [];
  if (!rows.length) return null;
  const keys = Object.keys(rows[0]);
  const fmt = (v: any) => {
    if (v && typeof v === "object" && "p" in v) return <span className="num">{pct(v.p, 1)} <span className="text-faint text-2xs">[{pct(v.ci[0], 0)}–{pct(v.ci[1], 0)}] k={v.k}</span></span>;
    if (Array.isArray(v)) return <span className="num text-fg-2">[{v.map((x) => num(x, 3)).join(", ")}]</span>;
    if (typeof v === "number") return <span className="num">{num(v, Math.abs(v) < 10 ? 3 : 1)}</span>;
    return <span>{String(v)}</span>;
  };
  return (
    <Panel className="border-0 border-t" title={t.title}>
      <table className="tbl"><thead><tr>{keys.map((k) => <th key={k} className={k === keys[0] ? "" : "r"}>{k.replace(/_/g, " ")}</th>)}</tr></thead>
        <tbody>{rows.map((r: any, i: number) => <tr key={i}>{keys.map((k) => <td key={k} className={k === keys[0] ? "" : "r"}>{fmt(r[k])}</td>)}</tr>)}</tbody></table>
    </Panel>
  );
}

export function SpecChart({ spec, height = 260 }: { spec: ChartSpec; height?: number }) {
  const opt = useMemo(() => {
    const base: any = { grid: { left: 52, right: 16, top: 26, bottom: 34 }, legend: { data: spec.series.map((s) => s.name) }, tooltip: { trigger: spec.type === "scatter" ? "item" : "axis" } };
    const yFmt = spec.yFormat === "pct" ? { axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } } : {};
    if (spec.type === "scatter") {
      const all = spec.series.flatMap((s) => s.data);
      const lo = Math.min(...all.map((d: any) => Math.min(d[0], d[1]))), hi = Math.max(...all.map((d: any) => Math.max(d[0], d[1])));
      return { ...base, xAxis: { type: "value", name: spec.xLabel, nameLocation: "middle", nameGap: 22, scale: true }, yAxis: { type: "value", name: spec.yLabel, scale: true },
        series: [...spec.series.map((s) => ({ name: s.name, type: "scatter", symbolSize: 3.5, data: s.data, itemStyle: { opacity: 0.6 } })), ...(spec.diagonal ? [{ type: "line", data: [[lo, lo], [hi, hi]], symbol: "none", lineStyle: { color: C.faint, type: "dashed" } }] : [])] };
    }
    if (spec.type === "histogram") {
      return { ...base, xAxis: { type: "category", data: spec.x!.map((v: number) => v.toFixed(2)), name: spec.xLabel, nameLocation: "middle", nameGap: 22 }, yAxis: { type: "value", ...{ axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } } },
        series: spec.series.map((s, i) => ({ name: s.name, type: "bar", data: s.data, barGap: "-100%", barCategoryGap: "4%", itemStyle: { color: i === 0 ? "rgba(232,163,61,0.75)" : "rgba(122,132,150,0.35)" } })) };
    }
    if (spec.type === "band") {
      const series: any[] = [];
      spec.series.forEach((s, i) => {
        const col = C.palette[i % C.palette.length];
        if (s.p25 && s.p75) {
          series.push({ type: "line", data: s.p25, stack: `b${i}`, symbol: "none", lineStyle: { opacity: 0 }, tooltip: { show: false } });
          series.push({ type: "line", data: s.p75.map((v, j) => v - s.p25![j]), stack: `b${i}`, symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: col, opacity: 0.12 }, tooltip: { show: false } });
        }
        series.push({ name: s.name, type: "line", data: s.data, symbol: "none", lineStyle: { color: col, width: 1.5 }, itemStyle: { color: col } });
      });
      return { ...base, xAxis: { type: "category", data: spec.x }, yAxis: { type: "value", name: spec.yLabel, scale: true }, series };
    }
    if (spec.type === "bar_ci") {
      const series: any[] = spec.series.map((s) => ({ name: s.name, type: "bar", data: s.data, barMaxWidth: 22 }));
      spec.series.forEach((s, si) => series.push({
        type: "custom", name: `${s.name} CI`, tooltip: { show: false },
        renderItem: (params: any, api: any) => {
          const ci = s.ci?.[params.dataIndex];
          if (!ci || ci[0] === null) return null;
          const n = spec.series.length;
          const bandW = api.size([1, 0])[0];
          const off = (si - (n - 1) / 2) * Math.min(22, bandW / (n + 1)) * 1.1;
          const x = api.coord([params.dataIndex, 0])[0] + off;
          const y0 = api.coord([0, ci[0]])[1], y1 = api.coord([0, ci[1]])[1];
          return { type: "group", children: [{ type: "line", shape: { x1: x, y1: y0, x2: x, y2: y1 }, style: { stroke: "#d6dce5", lineWidth: 1 } }, { type: "line", shape: { x1: x - 3, y1: y0, x2: x + 3, y2: y0 }, style: { stroke: "#d6dce5" } }, { type: "line", shape: { x1: x - 3, y1: y1, x2: x + 3, y2: y1 }, style: { stroke: "#d6dce5" } }] };
        },
        data: s.data.map((_, i) => i), z: 10,
      }));
      return { ...base, legend: { data: spec.series.map((s) => s.name) }, xAxis: { type: "category", data: spec.x }, yAxis: { type: "value", ...yFmt }, series };
    }
    return { ...base, xAxis: { type: "category", data: spec.x, axisLabel: { hideOverlap: true } }, yAxis: { type: "value", name: spec.yLabel, scale: spec.type === "line" },
      series: spec.series.map((s) => ({ name: s.name, type: spec.type === "line" ? "line" : "bar", data: s.data, symbol: "none", lineStyle: { width: 1.2 }, barMaxWidth: 26, connectNulls: false })) };
  }, [spec]);
  return <EChart option={opt} height={height} />;
}

export function EntryTiming() {
  const b = useBuilder();
  const [times, setTimes] = useState("09:45,10:00,10:15,10:30,11:00,11:30,12:00,13:00,14:00,15:00");
  const [rid, setRid] = useState<string | null>(null);
  const job = useJob(rid);
  const run = useRun(rid || undefined, true);
  const findPath = (): string | null => {
    const walk = (n: any, path: string): string | null => {
      if (n.kind === "cond" && n.type === "entry_times") return `${path}.params.times`;
      for (const [i, c] of (n.children || []).entries()) { const r = walk(c, `${path}.children.${i}`); if (r) return r; }
      return null;
    };
    return walk(b.strategy.entry.conditions, "strategies.0.entry.conditions");
  };
  const path = findPath();
  const start = useMutation({
    mutationFn: () => api.post<{ id: string }>("/api/research/entry-timing", { config: b.backtestConfig(), path, times: times.split(",").map((t) => t.trim()).filter(Boolean) }),
    onSuccess: (r) => setRid(r.id), onError: (e: any) => toast.error(e.message),
  });
  const rows = run.data?.result?.rows || [];
  return (
    <div className="p-3 space-y-3">
      <div className="flex items-end gap-2">
        <div className="text-xs text-fg-2 pb-1.5">Strategy: <span className="text-fg">{b.strategy.name}</span> (builder draft) · {b.start} → {b.end}</div>
        <Field label="Entry times to test" className="flex-1 max-w-xl"><Input value={times} onChange={(e) => setTimes(e.target.value)} className="num" /></Field>
        <Button variant="primary" disabled={!path || start.isPending} onClick={() => start.mutate()}><Play /> Run sweep</Button>
      </div>
      {!path && <ErrorBox error="The builder strategy needs an 'Entry at times' condition to sweep entry time." />}
      <JobBar job={job} />
      {rows.length > 0 && (
        <div className="grid grid-cols-2 gap-px bg-line">
          <Panel className="border-0" title="Net P&L and Sharpe by entry time">
            <EChart height={260} option={{ grid: { left: 56, right: 44, top: 24, bottom: 28 }, legend: { data: ["net P&L", "Sharpe"] }, tooltip: { trigger: "axis" },
              xAxis: { type: "category", data: rows.map((r: any) => r.time) }, yAxis: [{ type: "value" }, { type: "value", splitLine: { show: false } }],
              series: [{ name: "net P&L", type: "bar", data: rows.map((r: any) => ({ value: r.net_pnl, itemStyle: { color: (r.net_pnl ?? 0) >= 0 ? C.up : C.down } })) }, { name: "Sharpe", type: "line", yAxisIndex: 1, data: rows.map((r: any) => r.sharpe), itemStyle: { color: C.accent } }] } as any} />
          </Panel>
          <Panel className="border-0" title="Results">
            <table className="tbl"><thead><tr><th>Time</th><th className="r">Trades</th><th className="r">Net P&L</th><th className="r">Sharpe</th><th className="r">Win %</th><th className="r">Expectancy</th><th className="r">Max DD</th></tr></thead>
              <tbody>{rows.map((r: any) => <tr key={r.time}><td className="num">{r.time}</td><td className="r num">{r.trades}</td><td className={cn("r num", (r.net_pnl ?? 0) >= 0 ? "text-up" : "text-down")}>{num(r.net_pnl, 0)}</td><td className="r num">{num(r.sharpe)}</td><td className="r num">{pct(r.win_rate, 0)}</td><td className="r num">{num(r.expectancy)}</td><td className="r num text-down">{pct(r.max_drawdown, 1)}</td></tr>)}</tbody></table>
            <div className="p-2 text-2xs text-faint">A robust edge changes smoothly with entry time; a single spike surrounded by losers is usually noise.</div>
          </Panel>
        </div>
      )}
    </div>
  );
}

export function Sql() {
  const [sql, setSql] = useState("-- views: bars(symbol, ts, open, high, low, close, volume), daily, options, iv_history, events\nSELECT symbol, date_trunc('month', date) AS month, avg((high - low) / open) * 100 AS avg_range_pct, count(*) AS sessions\nFROM daily WHERE symbol = 'DEMO'\nGROUP BY 1, 2 ORDER BY 2");
  const run = useMutation({ mutationFn: () => api.post<any>("/api/research/sql", { sql: sql.split("\n").filter((l) => !l.trim().startsWith("--")).join("\n") }) });
  const cols: Col<{ k: number; r: any[] }>[] = (run.data?.columns || []).map((c: string, i: number) => ({ key: String(i), label: c, render: ({ r }: { r: any[] }) => <span className={typeof r[i] === "number" ? "num" : ""}>{typeof r[i] === "number" ? num(r[i], Math.abs(r[i]) < 100 ? 4 : 2) : String(r[i] ?? "—")}</span>, sort: ({ r }: { r: any[] }) => r[i] }));
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-col gap-1.5 border-b border-line p-2.5">
        <textarea value={sql} onChange={(e) => setSql(e.target.value)} onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run.mutate(); } }}
          spellCheck={false} className="field h-36 resize-y py-1.5 font-mono text-xs leading-5" />
        <div className="flex items-center gap-2">
          <Button variant="primary" onClick={() => run.mutate()} disabled={run.isPending}><Play /> Run (⌘↵)</Button>
          <span className="text-2xs text-faint">Read-only DuckDB over the Parquet store · timestamps in America/New_York · 5,000-row limit · 20s timeout</span>
          {run.data && <span className="ml-auto num text-2xs text-muted">{run.data.row_count} rows{run.data.truncated ? " (truncated)" : ""}</span>}
        </div>
      </div>
      {run.error && <ErrorBox className="m-2" error={run.error} />}
      {run.data && <DataTable rows={run.data.rows.map((r: any[], k: number) => ({ k, r }))} cols={cols} rowKey={(x) => x.k} className="flex-1" />}
    </div>
  );
}
