/* Event-study report: the answer, its uncertainty, and where it comes from. */
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { C, EChart } from "@/components/charts/EChart";
import { LWChart, dayTime, etChartTime } from "@/components/charts/LWChart";
import { DataTable, ErrorBox, Loading, Panel, Stat, SyntheticBadge, Warnings, type Col } from "@/components/common";
import { api } from "@/lib/api";
import { cn, num, pct } from "@/lib/format";

const p1 = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const sp = (v: number | null | undefined, dp = 2) => (v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(dp)}%`);
const OUTCOME: Record<string, [string, string]> = { target: ["hit", "text-up"], stop: ["stopped", "text-down"], neither: ["no", "text-muted"], ambiguous: ["ambiguous", "text-warn"] };

export function describeLevel(cfg: any): string {
  const t = cfg?.target;
  if (!t) return "no target";
  if (t.kind === "expr") return `reaches ${t.expr}`;
  const unit = t.kind === "pct" ? "%" : " pts";
  const ref = ({ entry: "start", open: "open", prev_close: "prev close" } as any)[cfg.reference || "entry"];
  if (cfg.measure === "close") return t.value === 0 ? `ends ${cfg.direction === "down" ? "below" : "above"} ${ref}` : `ends ${t.value > 0 ? "≥ +" : "≤ "}${t.value}${unit} vs ${ref}`;
  return `touches ${t.value >= 0 ? "+" : ""}${t.value}${unit} from ${ref}`;
}

export function StudyKpis({ res }: { res: any }) {
  const s = res.stats;
  const hasT = !!res.config.target;
  const lift = s.lift;
  return (
    <div className="grid gap-px border-b border-line bg-line [&>*]:bg-bg-1" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(128px, 1fr))" }}>
      <Stat label={hasT ? "Historical frequency" : "Sessions"} value={hasT ? p1(s.probability) : s.qualifying_sessions} tone={hasT ? "text-accent" : ""}
        sub={hasT ? `95% CI ${p1(s.probability_ci95?.[0])} – ${p1(s.probability_ci95?.[1])}` : "met the condition"} hint="Share of qualifying sessions where the outcome occurred, with a Wilson 95% confidence interval. A past frequency, not a forecast." />
      <Stat label="Sample" value={`${s.target_first ?? 0} / ${s.qualifying_sessions}`} sub={`of ${s.evaluated_sessions} evaluated`} hint="Outcome count over sessions that met the condition" />
      <Stat label="Baseline" value={p1(s.baseline_probability)} sub="all evaluated sessions" hint="How often the same outcome occurs on every evaluated session, condition or not" />
      <Stat label="Lift vs baseline" value={lift === null || lift === undefined ? "—" : `${lift >= 0 ? "+" : ""}${(lift * 100).toFixed(1)} pp`} tone={lift > 0 ? "text-up" : lift < 0 ? "text-down" : ""}
        sub={s.p_value_vs_rest === null || s.p_value_vs_rest === undefined ? "" : `p ${s.p_value_vs_rest < 0.001 ? "< 0.001" : s.p_value_vs_rest.toFixed(3)}`} hint="Two-proportion z-test of qualifying sessions against all the others" />
      <Stat label="Avg best move" value={sp(s.avg_mfe_pct)} tone="text-up" sub={`median ${sp(s.median_mfe_pct)}`} hint="Maximum favourable excursion after the starting point" />
      <Stat label="Avg worst move" value={sp(s.avg_mae_pct)} tone="text-down" sub={`median ${sp(s.median_mae_pct)}`} hint="Maximum adverse excursion after the starting point" />
      <Stat label="Avg end return" value={sp(s.avg_end_return_pct)} tone={s.avg_end_return_pct > 0 ? "text-up" : "text-down"} sub={`${p1(s.pct_positive_end)} ended higher`} hint="Underlying return from the starting point to the end of the horizon" />
      <Stat label="Time to target" value={s.median_minutes_to_target === null || s.median_minutes_to_target === undefined ? "—" : `${Math.round(s.median_minutes_to_target)} min`}
        sub={s.timed_target_hits ? `median of ${s.timed_target_hits} timed hits` : "needs intraday bars"} hint="Only measurable on sessions with intraday bars" />
    </div>
  );
}

export function StudyCharts({ res }: { res: any }) {
  const hasT = !!res.config.target;
  const years = useMemo(() => {
    const y = res.by_year || [];
    return {
      grid: { left: 34, right: 40, top: 28, bottom: 24 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => { const r = y[ps[0].dataIndex]; return `${r.year}: ${r.n} sessions${hasT ? `, ${r.target} hits (${p1(r.probability)})` : ""}<br/>avg end return ${sp(r.avg_end_return_pct)}`; } },
      legend: { left: 4, right: undefined },
      xAxis: { type: "category", data: y.map((r: any) => r.year) },
      yAxis: [{ type: "value", minInterval: 1 }, { type: "value", max: hasT ? 1 : undefined, axisLabel: { formatter: (v: number) => (hasT ? `${Math.round(v * 100)}%` : `${v.toFixed(1)}`) }, splitLine: { show: false } }],
      series: [
        { name: "qualifying sessions", type: "bar", data: y.map((r: any) => r.n), itemStyle: { color: "#2b3442" }, barMaxWidth: 18 },
        { name: hasT ? "frequency" : "avg end return", type: "line", yAxisIndex: 1, data: y.map((r: any) => (hasT ? r.probability : r.avg_end_return_pct)), symbolSize: 5, lineStyle: { width: 1.5, color: C.accent }, itemStyle: { color: C.accent }, connectNulls: false },
      ],
    };
  }, [res, hasT]);
  const cum = useMemo(() => res.cumulative && {
    grid: { left: 44, right: 14, top: 26, bottom: 30 },
    tooltip: { trigger: "axis", formatter: (ps: any[]) => `${ps[0].axisValue} min after start: ${p1(ps[0].data)} of ${res.cumulative.sessions} intraday sessions had reached the target` },
    xAxis: { type: "category", data: res.cumulative.minutes, name: "minutes after start", nameLocation: "middle", nameGap: 20, nameTextStyle: { color: C.faint, fontSize: 10 } },
    yAxis: { type: "value", max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    series: [{ type: "line", step: "end", data: res.cumulative.probability, showSymbol: false, lineStyle: { color: C.accent, width: 1.5 }, areaStyle: { color: "rgba(232,163,61,0.12)" } }],
  }, [res]);
  const reg = (key: "vix" | "trend") => {
    const rows = (res.regimes?.[key] || []).filter((r: any) => r.n > 0);
    return {
      grid: { left: 44, right: 14, top: 20, bottom: 24 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => { const r = rows[ps[0].dataIndex]; return `${r.key}: ${r.n} sessions${hasT ? `, ${p1(r.probability)} (CI ${p1(r.ci95?.[0])}–${p1(r.ci95?.[1])})` : ""}<br/>avg end ${sp(r.avg_end_return_pct)}`; } },
      xAxis: { type: "category", data: rows.map((r: any) => `${r.key}\nn=${r.n}`), axisLabel: { fontSize: 9.5, lineHeight: 12 } },
      yAxis: { type: "value", max: hasT ? 1 : undefined, axisLabel: { formatter: (v: number) => (hasT ? `${Math.round(v * 100)}%` : v.toFixed(1)) } },
      series: [{ type: "bar", barMaxWidth: 26, data: rows.map((r: any) => ({ value: hasT ? r.probability : r.avg_end_return_pct, itemStyle: { color: r.n < 20 ? "#3a4353" : C.info } })) },
        ...(hasT ? [{ type: "custom", data: rows.map((r: any, i: number) => [i, r.ci95?.[0], r.ci95?.[1]]), z: 5, renderItem: (_: any, api: any) => {
          const lo = api.coord([api.value(0), api.value(1)]), hi = api.coord([api.value(0), api.value(2)]);
          return { type: "group", children: [{ type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: { stroke: C.fg, lineWidth: 1 } }, { type: "line", shape: { x1: lo[0] - 4, y1: lo[1], x2: lo[0] + 4, y2: lo[1] }, style: { stroke: C.fg } }, { type: "line", shape: { x1: hi[0] - 4, y1: hi[1], x2: hi[0] + 4, y2: hi[1] }, style: { stroke: C.fg } }] };
        } }] : [])],
    };
  };
  const hist = (h: any, color: string) => h && {
    grid: { left: 36, right: 10, top: 12, bottom: 22 },
    tooltip: { trigger: "axis", formatter: (ps: any[]) => `${Number(ps[0].axisValue).toFixed(2)}%: ${ps[0].data} sessions` },
    xAxis: { type: "category", data: h.x.map((v: number) => v.toFixed(2)), axisLabel: { interval: Math.floor(h.x.length / 6), formatter: (v: string) => `${Number(v).toFixed(1)}%` } },
    yAxis: { type: "value", minInterval: 1 },
    series: [{ type: "bar", data: h.counts, barCategoryGap: "8%", itemStyle: { color } }],
  };
  const sens = useMemo(() => {
    const s = res.sensitivity;
    if (!s) return null;
    const data: any[] = [];
    s.probability.forEach((row: number[], i: number) => row.forEach((v, j) => data.push([j, i, v === null ? "-" : +(v * 100).toFixed(1)])));
    return {
      grid: { left: 58, right: 16, top: 12, bottom: 40 },
      tooltip: { formatter: (p: any) => `target ${s.direction === "up" ? "+" : "−"}${s.targets_pct[p.value[0]]}% before stop ${s.direction === "up" ? "−" : "+"}${s.stops_pct[p.value[1]]}%: ${p.value[2]}%<br/>ambiguous share ${(s.ambiguous_share[p.value[1]][p.value[0]] * 100).toFixed(0)}%` },
      xAxis: { type: "category", data: s.targets_pct.map((v: number) => `${v}%`), name: `target (${s.direction})`, nameLocation: "middle", nameGap: 24, nameTextStyle: { color: C.faint, fontSize: 10 }, splitArea: { show: false } },
      yAxis: { type: "category", data: s.stops_pct.map((v: number) => `${v}%`), name: "stop", nameTextStyle: { color: C.faint, fontSize: 10 } },
      visualMap: { show: false, min: 0, max: 100, inRange: { color: ["#1a202b", "#5a4a22", "#e8a33d"] } },
      series: [{ type: "heatmap", data, label: { show: true, fontSize: 9.5, color: "#d6dce5", fontFamily: "JetBrains Mono", formatter: (p: any) => (p.value[2] === "-" ? "" : `${p.value[2]}`) }, itemStyle: { borderColor: "#090b0f", borderWidth: 1 } }],
    };
  }, [res]);
  return (
    <div className="grid gap-px bg-line [&>*]:min-w-0" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
      <Panel className="border-0" title="By year" dense><EChart option={years} height={190} /></Panel>
      {cum ? <Panel className="border-0" title={`Share that reached the target, by time since the start (${res.cumulative.sessions} intraday sessions)`} dense><EChart option={cum} height={190} /></Panel>
        : <Panel className="border-0" title="By trend regime (grey = fewer than 20 sessions)" dense><EChart option={reg("trend")} height={190} /></Panel>}
      <Panel className="border-0" title="By volatility regime (previous VIX close; grey = fewer than 20 sessions)" dense><EChart option={reg("vix")} height={170} /></Panel>
      {sens ? <Panel className="border-0" title={`Target-before-stop frequency, % (${res.sensitivity.sessions} sessions)`} dense><EChart option={sens} height={170} /></Panel>
        : <Panel className="border-0" title="Return at the end of the horizon" dense>{res.end_hist ? <EChart option={hist(res.end_hist, C.info)} height={170} /> : null}</Panel>}
      <Panel className="border-0" title="Best move after the start (MFE)" dense>{res.mfe_hist ? <EChart option={hist(res.mfe_hist, C.up)} height={130} /> : null}</Panel>
      <Panel className="border-0" title="Worst move after the start (MAE)" dense>{res.mae_hist ? <EChart option={hist(res.mae_hist, C.down)} height={130} /> : null}</Panel>
    </div>
  );
}

export function StudyNotes({ res }: { res: any }) {
  const items = [
    ...(res.synthetic ? [{ level: "info", text: "SIMULATED data: this symbol is synthetic sandbox data, not market history." }] : []),
    ...(res.notes || []).map((t: string) => ({ level: /ambiguous|Only \d+/.test(t) ? "warn" : "info", text: t })),
    ...(res.excluded || []).map((e: any) => ({ level: "warn", text: `${e.count} session(s) excluded - ${e.reason}${e.dates?.length ? ` (e.g. ${e.dates.slice(0, 3).join(", ")})` : ""}. Excluded sessions are not counted as failures.` })),
  ];
  return items.length ? <div className="p-2"><Warnings items={items} /></div> : null;
}

export function SessionsTable({ res, filter = "all", onPick, selected, height = "100%" }: { res: any; filter?: "all" | "win" | "loss"; onPick?: (r: any) => void; selected?: string | null; height?: number | string }) {
  const rows = useMemo(() => (res.sessions || []).filter((r: any) => filter === "all" || (filter === "win" ? r.outcome === "target" : r.outcome !== "target")).slice().reverse(), [res, filter]);
  const cols: Col<any>[] = [
    { key: "date", label: "Session", render: (r) => <span className="num">{r.date}</span>, sort: (r) => r.date },
    { key: "out", label: "Outcome", render: (r) => <span className={cn("text-2xs font-medium uppercase tracking-wide", OUTCOME[r.outcome]?.[1])}>{OUTCOME[r.outcome]?.[0] || r.outcome}</span>, sort: (r) => r.outcome },
    { key: "prev", label: "Prev day", align: "r", render: (r) => <span className={cn("num", r.prev_return_pct > 0 ? "text-up" : "text-down")}>{sp(r.prev_return_pct)}</span>, sort: (r) => r.prev_return_pct },
    { key: "gap", label: "Gap", align: "r", render: (r) => <span className="num text-fg-2">{sp(r.gap_pct)}</span>, sort: (r) => r.gap_pct },
    { key: "entry", label: "Start px", align: "r", render: (r) => <span className="num">{num(r.entry_price)}</span> },
    { key: "mfe", label: "Best", align: "r", render: (r) => <span className="num text-up">{sp(r.mfe_pct)}</span>, sort: (r) => r.mfe_pct },
    { key: "mae", label: "Worst", align: "r", render: (r) => <span className="num text-down">{sp(r.mae_pct)}</span>, sort: (r) => r.mae_pct },
    { key: "end", label: "End", align: "r", render: (r) => <span className={cn("num", r.end_return_pct > 0 ? "text-up" : "text-down")}>{sp(r.end_return_pct)}</span>, sort: (r) => r.end_return_pct },
    { key: "ttt", label: "To target", align: "r", render: (r) => <span className="num text-fg-2">{r.minutes_to_target === null || r.minutes_to_target === undefined ? "—" : `${Math.round(r.minutes_to_target)}m`}</span>, sort: (r) => r.minutes_to_target },
    { key: "vix", label: "VIX", align: "r", render: (r) => <span className="num text-muted">{num(r.vix_prev, 1)}</span>, sort: (r) => r.vix_prev },
    { key: "res", label: "Bars", align: "r", render: (r) => <span className={cn("num text-2xs", r.resolution === "1d" ? "text-faint" : "text-fg-2")}>{r.resolution}</span> },
  ];
  return <DataTable rows={rows} cols={cols} rowKey={(r) => r.date} onRowClick={onPick} selected={selected} height={height} />;
}

/** One session's price path with the levels the study measured. */
export function SessionReplay({ symbol, row, height = 260 }: { symbol: string; row: any; height?: number }) {
  const q = useQuery({ queryKey: ["study-session", symbol, row.date], queryFn: () => api.get<any>(`/api/study/session?symbol=${symbol}&day=${row.date}`) });
  const series = useMemo(() => {
    const d = q.data;
    if (!d) return [];
    const daily = d.resolution === "1d";
    const lines = [
      ...(row.target_level ? [{ price: row.target_level, color: C.up, title: "target", style: "solid" as const }] : []),
      ...(row.stop_level ? [{ price: row.stop_level, color: C.down, title: "stop", style: "solid" as const }] : []),
      { price: row.entry_price, color: C.accent, title: "start" },
      ...(d.prev_close ? [{ price: d.prev_close, color: "#7a8496", title: "prev close" }] : []),
    ];
    return [{ type: "candles" as const, data: d.bars.map((b: number[]) => ({ time: daily ? dayTime(new Date(b[0] * 1000).toISOString().slice(0, 10)) : etChartTime(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })), priceLines: lines,
      markers: daily ? [{ time: dayTime(row.date), position: "aboveBar" as const, color: C.accent, shape: "arrowDown" as const, text: row.date }] : [] }];
  }, [q.data, row]);
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorBox className="m-2" error={q.error} />;
  const daily = q.data.resolution === "1d";
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 overflow-hidden whitespace-nowrap border-b border-line px-2.5 py-1 text-2xs text-muted">
        <span className="num text-fg">{symbol} {row.date}</span>
        <span className={cn("font-medium uppercase", OUTCOME[row.outcome]?.[1])}>{OUTCOME[row.outcome]?.[0]}</span>
        <span className="truncate" title={(row.why || []).join(" · ")}>{daily ? "daily bars only for this session - the touch is known from the day's range, its timing is not" : `${q.data.resolution} bars, New York time`}</span>
      </div>
      <LWChart height={height} timeVisible={!daily} series={series} />
    </div>
  );
}

export function StudyReport({ res, onPick, selected, showTable = true }: { res: any; onPick?: (r: any) => void; selected?: string | null; showTable?: boolean }) {
  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line bg-bg-1 px-3 py-1.5 text-2xs text-muted">
        <span className="rounded-[2px] border border-info/50 bg-info/10 px-1.5 py-px font-semibold tracking-wider text-info">MARKET STUDY</span>
        <SyntheticBadge on={res.synthetic} />
        <span className="num text-fg">{res.symbol}</span><span className="num">{res.period[0]} → {res.period[1]}</span>
        <span>outcome: <b className="font-medium text-fg-2">{describeLevel(res.config)}</b>{res.config.stop ? <> before stop <b className="font-medium text-fg-2">{res.config.stop.kind === "expr" ? res.config.stop.expr : `${res.config.stop.value}${res.config.stop.kind === "pct" ? "%" : " pts"}`}</b></> : null}</span>
        <span>resolution: {Object.entries(res.stats.resolution_mix || {}).map(([k, v]) => `${v}×${k}`).join(" · ") || "—"}</span>
        <span className="text-faint">underlying prices only - not option returns</span>
      </div>
      <StudyKpis res={res} />
      <StudyNotes res={res} />
      <StudyCharts res={res} />
      {showTable && <Panel className="border-0 border-t" title={`Qualifying sessions (${res.sessions?.length ?? 0}${res.sessions_truncated ? ", most recent" : ""})`} dense><SessionsTable res={res} onPick={onPick} selected={selected} height={300} /></Panel>}
    </div>
  );
}
export { p1 as fmtProb, sp as fmtSignedPct, pct as _pct };
