/* A bounded experiment: one strategy over a grid of settings.

   The screen is built around the question the report asks: not "which cell is best" but "where does the answer stop
   depending on the cell". The map shows every configuration; the stable region and the best single cell are both marked,
   and a cell can be opened down to its trades. Every number is the server's. */
import { useQuery } from "@tanstack/react-query";
import { Download, Grid3x3, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import { C, EChart } from "@/components/charts/EChart";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";

type Cell = { i: number; key: Record<string, any>; label: string; symbol: string; status: string; error?: string; metrics: any; signal_sessions: number };
const FIELD: Record<string, string> = { symbol: "Underlying", dte: "Expiration", entry_time: "Entry time", exit_minutes: "Minutes held", strike: "Strike", premium_target: "Premium target", premium_stop: "Premium stop",
  underlying_target: "Underlying target", prev_decline: "Previous-day decline" };
const METRICS: [string, string, (m: any) => number | null, "usd" | "pct" | "n"][] = [
  ["expectancy", "Average per trade", (m) => m.expectancy, "usd"], ["net", "Net result", (m) => (m.trades ? m.net : null), "usd"], ["later", "Average, later sessions", (m) => m.later?.expectancy ?? null, "usd"],
  ["worse", "Net with 5% worse fills", (m) => m.sensitivity?.["5%"] ?? null, "usd"], ["win", "Win rate", (m) => m.win_rate, "pct"], ["trades", "Trades", (m) => m.trades, "n"], ["cov", "Signals priced", (m) => m.coverage?.pct ?? null, "pct"],
];
const usd = (v: number | null | undefined, dp = 0) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : `${v < 0 ? "−" : v > 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`);
const show = (kind: string, v: number | null | undefined) => (v === null || v === undefined ? "–" : kind === "usd" ? usd(v) : kind === "pct" ? `${v.toFixed(0)}%` : String(v));

export function valueLabel(field: string, v: any): string {
  if (field === "strike") { const [m, x] = v as [string, number]; return m === "atm" ? "ATM" : m === "pct_otm" ? `${x}% OTM` : `${Math.round(x * 100)}Δ`; }
  if (field === "dte") return `${v}DTE`;
  if (field === "exit_minutes") return v === "close" ? "close" : `${v}m`;
  if (["premium_target", "premium_stop", "underlying_target", "prev_decline"].includes(field)) return `${v}%`;
  return v === "open" ? "09:30" : String(v);
}
const same = (a: any, b: any) => JSON.stringify(a) === JSON.stringify(b);

export function ExperimentView({ runId, onAsk }: { runId: string; onAsk?: (text: string) => void }) {
  const q = useQuery({ queryKey: ["experiment", runId], queryFn: () => api.get<any>(`/api/experiments/${runId}`), retry: false });
  const [metric, setMetric] = useState("expectancy");
  const [axes, setAxes] = useState<[string, string] | null>(null);
  const [slice, setSlice] = useState<Record<string, number>>({});
  const [sel, setSel] = useState<number | null>(null);
  const [sort, setSort] = useState<[string, 1 | -1]>(["expectancy", -1]);
  const d = q.data;
  const dims: { field: string; values: any[]; assumed?: boolean }[] = d?.dims || [];
  const [xf, yf] = axes || [dims[0]?.field, dims[1]?.field];
  const rest = dims.filter((x) => x.field !== xf && x.field !== yf);
  const met = METRICS.find((m) => m[0] === metric)!;
  const cells: Cell[] = d?.cells || [];
  const stab = d?.stability || {};
  const stableIds = useMemo(() => {
    // the stable cell and the cells one step from it: the region the statement is about
    const s = cells.find((c) => c.i === stab.stable);
    if (!s) return new Set<number>();
    const near = cells.filter((c) => dims.filter((x) => !same(c.key[x.field], s.key[x.field])).length === 1 && dims.every((x) => same(c.key[x.field], s.key[x.field]) || (x.field !== "symbol" && Math.abs(x.values.findIndex((v) => same(v, c.key[x.field])) - x.values.findIndex((v) => same(v, s.key[x.field]))) === 1)));
    return new Set([s.i, ...near.map((c) => c.i)]);
  }, [d]); // eslint-disable-line
  const inSlice = (c: Cell) => rest.every((x) => same(c.key[x.field], x.values[slice[x.field] ?? 0]));
  const xd = dims.find((x) => x.field === xf), yd = dims.find((x) => x.field === yf);
  const grid = useMemo(() => {
    if (!xd) return [];
    return cells.filter(inSlice).map((c) => ({ c, x: xd.values.findIndex((v) => same(v, c.key[xd.field])), y: yd ? yd.values.findIndex((v) => same(v, c.key[yd.field])) : 0, v: c.status === "ok" ? met[2](c.metrics) : null }));
  }, [d, xf, yf, slice, metric]); // eslint-disable-line
  const picked = cells.find((c) => c.i === sel) || null;

  if (q.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Reading the experiment…</div>;
  if (q.isError || !d) return <div className="p-6 text-[13px] dim">The experiment has not finished, or its result is not stored. {String((q.error as any)?.message || "")}</div>;
  const sm = d.summary || {};
  const vals = grid.map((g) => g.v).filter((v): v is number => v !== null && Number.isFinite(v));
  const lim = Math.max(1e-9, ...vals.map((v) => Math.abs(v)));
  const diverging = met[3] === "usd";
  const heat = {
    grid: { left: 74, right: 18, top: 10, bottom: 44 },
    tooltip: { formatter: (p: any) => { const g = grid[p.dataIndex]; const m = g.c.metrics; return `<b>${g.c.label}</b><br/>${met[1]}: ${show(met[3], g.v)}<br/>${m.trades} trades · win ${m.win_rate ?? "–"}% · priced ${m.coverage?.pct ?? "–"}%${m.ci95 ? `<br/>average between ${usd(m.ci95[0])} and ${usd(m.ci95[1])}` : ""}`; } },
    xAxis: { type: "category", data: xd!.values.map((v) => valueLabel(xd!.field, v)), name: FIELD[xd!.field], nameLocation: "middle", nameGap: 28, splitArea: { show: false } },
    yAxis: { type: "category", data: yd ? yd.values.map((v) => valueLabel(yd.field, v)) : [""], name: yd ? FIELD[yd.field] : "", nameLocation: "middle", nameGap: 58, splitArea: { show: false } },
    visualMap: diverging ? { show: false, min: -lim, max: lim, inRange: { color: ["#8f2f2d", "#3a2325", "#161b24", "#17332f", "#1c8076"] } } : { show: false, min: Math.min(...vals, 0), max: Math.max(...vals, 1), inRange: { color: ["#161b24", "#1f3b5c", "#4c8dff"] } },
    series: [{ type: "heatmap", data: grid.map((g) => ({ value: [g.x, g.y, g.v ?? "-"], itemStyle: { borderColor: g.c.i === sel ? "#fff" : stableIds.has(g.c.i) ? C.accent : g.c.i === stab.best ? "#d6dce5" : "#090b0f",
        borderWidth: g.c.i === sel ? 2 : stableIds.has(g.c.i) ? 1.5 : g.c.i === stab.best ? 1 : 1, borderType: g.c.i === stab.best && !stableIds.has(g.c.i) ? "dashed" : "solid" } })),
      label: { show: grid.length <= 80, color: "#d6dce5", fontSize: 10.5, fontFamily: "JetBrains Mono", formatter: (p: any) => show(met[3], grid[p.dataIndex].v) }, emphasis: { itemStyle: { borderColor: "#fff", borderWidth: 1.5 } } }],
  };
  const both = cells.filter((c) => c.metrics.earlier?.trades >= 5 && c.metrics.later?.trades >= 5);
  const elim = Math.max(1, ...both.flatMap((c) => [Math.abs(c.metrics.earlier.expectancy), Math.abs(c.metrics.later.expectancy)]));
  const scatter = {
    grid: { left: 54, right: 14, top: 12, bottom: 40 },
    tooltip: { formatter: (p: any) => `<b>${both[p.dataIndex].label}</b><br/>earlier ${usd(p.value[0])} a trade<br/>later ${usd(p.value[1])} a trade` },
    xAxis: { type: "value", min: -elim, max: elim, name: `average before ${d.split}`, nameLocation: "middle", nameGap: 26, axisLabel: { formatter: (v: number) => usd(v) } },
    yAxis: { type: "value", min: -elim, max: elim, name: "average after", nameLocation: "middle", nameGap: 40, axisLabel: { formatter: (v: number) => usd(v) } },
    series: [{ type: "scatter", symbolSize: 7, data: both.map((c) => ({ value: [c.metrics.earlier.expectancy, c.metrics.later.expectancy], itemStyle: { color: c.i === sel ? "#fff" : stableIds.has(c.i) ? C.accent : c.metrics.later.expectancy > 0 ? C.up : C.down, opacity: 0.85 } })),
      markLine: { silent: true, symbol: "none", lineStyle: { color: C.faint, type: "dashed" }, data: [[{ coord: [-elim, -elim] }, { coord: [elim, elim] }], { xAxis: 0 }, { yAxis: 0 }] } }],
  };
  const rows = [...cells].sort((a, b) => {
    const f = (c: Cell) => (sort[0] === "label" ? c.label : (METRICS.find((m) => m[0] === sort[0])?.[2](c.metrics) ?? -Infinity));
    const x = f(a), y = f(b);
    return (x < y ? -1 : x > y ? 1 : 0) * sort[1];
  });
  const Th = ({ k, children, r = true }: { k: string; children: React.ReactNode; r?: boolean }) => <th className={cn(r && "r", "cursor-pointer select-none")} onClick={() => setSort(([c, s]) => [k, c === k ? (-s as 1 | -1) : -1])}>{children}{sort[0] === k ? (sort[1] < 0 ? " ↓" : " ↑") : ""}</th>;

  return (
    <div className="flex flex-col" data-testid="view-experiment">
      <div className="border-b border-line px-5 py-4">
        <div className="eyebrow mb-1.5 flex items-center gap-2"><Grid3x3 className="size-3.5 text-accent" />Experiment · {sm.configurations} configurations · {d.window[0]} → {d.window[1]} · {d.observation?.split(":")[0]}</div>
        <h2 className="leading-snug">{d.title?.replace(/^Experiment:\s*/, "")}</h2>
        <p className="mt-2 max-w-[980px] text-[13.5px] leading-relaxed soft" data-testid="experiment-summary">{sm.text}</p>
        {(d.notes || []).length > 0 && <div className="mt-2 text-xs text-warn">{d.notes.join(" ")}</div>}
      </div>
      <div className="grid grid-cols-1 gap-px bg-line xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="bg-bg-0 px-5 py-4">
          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
            <span className="dim">Colour</span>
            <div className="flex flex-wrap gap-1">{METRICS.map(([k, l]) => <button key={k} className={cn("chip", metric === k && "on")} onClick={() => setMetric(k)} data-testid={`exp-metric-${k}`}>{l}</button>)}</div>
          </div>
          {dims.length > 1 && (
            <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
              <label className="flex items-center gap-1.5 dim">across<select className="field !h-7 !w-auto" value={xf} onChange={(e) => setAxes([e.target.value, e.target.value === yf ? xf : yf])}>{dims.map((x) => <option key={x.field} value={x.field}>{FIELD[x.field]}</option>)}</select></label>
              <label className="flex items-center gap-1.5 dim">down<select className="field !h-7 !w-auto" value={yf} onChange={(e) => setAxes([e.target.value === xf ? yf : xf, e.target.value])}>{dims.map((x) => <option key={x.field} value={x.field}>{FIELD[x.field]}</option>)}</select></label>
              {rest.map((x) => (
                <span key={x.field} className="flex items-center gap-1.5"><span className="dim">{FIELD[x.field]}</span>
                  {x.values.map((v, j) => <button key={j} className={cn("chip", (slice[x.field] ?? 0) === j && "on")} onClick={() => setSlice((s) => ({ ...s, [x.field]: j }))}>{valueLabel(x.field, v)}</button>)}</span>
              ))}
            </div>
          )}
          <EChart option={heat} height={Math.max(150, 56 + (yd ? yd.values.length : 1) * 40)} onClick={(p) => setSel(grid[p.dataIndex]?.c.i ?? null)} />
          <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs dim">
            <span><span className="mr-1.5 inline-block size-2.5 border-[1.5px] border-accent align-[-1px]" />stable region</span>
            <span><span className="mr-1.5 inline-block size-2.5 border border-dashed border-fg align-[-1px]" />highest single cell</span>
            <span>click a cell for its trades</span>
          </div>
          <p className="mt-3 max-w-[900px] text-[13px] soft" data-testid="experiment-stability">{stab.text}</p>
        </div>
        <div className="bg-bg-0 px-4 py-4">
          <div className="eyebrow mb-1">Earlier against later</div>
          {both.length >= 3 ? <EChart option={scatter} height={230} onClick={(p) => setSel(both[p.dataIndex]?.i ?? null)} /> : <div className="py-6 text-xs dim">Too few configurations have five trades on both sides of {d.split}.</div>}
          <div className="mt-1 text-xs dim">{sm.earlier_later_rank_correlation != null ? `Rank correlation ${sm.earlier_later_rank_correlation > 0 ? "+" : ""}${sm.earlier_later_rank_correlation} across ${sm.compared_on_both} configurations. ` : ""}Points on the dashed line did the same in both parts. Both parts are history that has been examined.</div>
          {sm.best?.p_adjusted_for_configurations != null && (
            <div className="mt-3 border-t border-line pt-3 text-xs">
              <div className="eyebrow mb-1">The search, counted</div>
              <div className="soft">Best cell: <span className="num">{sm.best.label}</span>, t = {sm.best.t}. As one test p = {sm.best.p_unadjusted?.toFixed(3)}; as the best of {sm.with_trades}, <span className={cn("num", sm.best.p_adjusted_for_configurations < 0.05 ? "text-up" : "text-warn")}>p = {sm.best.p_adjusted_for_configurations.toFixed(2)}</span>.</div>
            </div>
          )}
        </div>
      </div>
      {picked && <CellPanel runId={runId} c={picked} split={d.split} onClose={() => setSel(null)} onAsk={onAsk} />}
      <div className="border-t border-line">
        <div className="flex items-center gap-2 px-5 py-2.5"><span className="eyebrow">Every configuration</span><span className="flex-1" />
          <a className="btn quiet sm" href={`/api/experiments/${runId}/table.csv`} download data-testid="experiment-csv"><Download />CSV</a></div>
        <div className="max-h-[420px] overflow-auto">
          <table className="t" data-testid="experiment-table">
            <thead><tr><Th k="label" r={false}>Configuration</Th><Th k="trades">Trades</Th><Th k="net">Net</Th><Th k="expectancy">Per trade</Th><th className="r">95% interval</th><Th k="win">Win</Th><th className="r">Earlier</th><Th k="later">Later</Th><Th k="worse">5% worse</Th><Th k="cov">Priced</Th></tr></thead>
            <tbody>{rows.map((c) => { const m = c.metrics; return (
              <tr key={c.i} className={cn("click", c.i === sel && "sel")} onClick={() => setSel(c.i)}>
                <td><span className="num">{c.label}</span>{c.i === stab.stable && <Tip content="The centre of the most stable region"><span className="ml-2 text-[10.5px] uppercase tracking-wide text-accent">stable</span></Tip>}{c.i === stab.best && <span className="ml-2 text-[10.5px] uppercase tracking-wide dim">highest</span>}{c.status !== "ok" && <span className="ml-2 text-xs text-down" title={c.error}>failed</span>}</td>
                <td className="r num">{m.trades}</td><td className={cn("r num", m.net > 0 ? "text-up" : m.net < 0 ? "text-down" : "")}>{m.trades ? usd(m.net) : "–"}</td><td className="r num">{usd(m.expectancy, 2)}</td>
                <td className="r num dim">{m.ci95 ? `${usd(m.ci95[0])} … ${usd(m.ci95[1])}` : "–"}</td><td className="r num">{m.win_rate != null ? `${m.win_rate}%` : "–"}</td>
                <td className="r num dim">{usd(m.earlier?.expectancy)}<span className="text-faint"> /{m.earlier?.trades}</span></td><td className={cn("r num", (m.later?.expectancy ?? 0) > 0 ? "text-up" : (m.later?.expectancy ?? 0) < 0 ? "text-down" : "")}>{usd(m.later?.expectancy)}<span className="text-faint"> /{m.later?.trades}</span></td>
                <td className="r num">{usd(m.sensitivity?.["5%"])}</td><td className={cn("r num", (m.coverage?.pct ?? 100) < 100 && "text-warn")}>{m.coverage?.pct != null ? `${m.coverage.pct}%` : "–"}</td>
              </tr>); })}</tbody>
          </table>
        </div>
        {(d.errors || []).length > 0 && <div className="px-5 py-2 text-xs text-warn">{d.errors.length} combination(s) could not be built: {d.errors[0].label}: {d.errors[0].error}</div>}
      </div>
    </div>
  );
}

function CellPanel({ runId, c, split, onClose, onAsk }: { runId: string; c: Cell; split: string; onClose: () => void; onAsk?: (t: string) => void }) {
  const t = useQuery({ queryKey: ["experiment-trades", runId, c.i], queryFn: () => api.get<any[]>(`/api/experiments/${runId}/trades/${c.i}`) });
  const m = c.metrics;
  const not = Object.entries(m.coverage?.not_executed || {}) as [string, number][];
  return (
    <div className="border-t border-accent/40 bg-bg-1 px-5 py-4" data-testid="experiment-cell">
      <div className="flex items-baseline gap-3"><h3 className="!text-[14px] !text-fg num">{c.label}</h3><span className="text-xs dim">{c.symbol} · {m.trades} trades from {m.coverage?.signals ?? "?"} signals</span><span className="flex-1" /><button className="btn quiet sm" onClick={onClose}>Close</button></div>
      <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-[12.5px] md:grid-cols-4">
        <div><span className="dim">net </span><span className={cn("num", m.net > 0 ? "text-up" : "text-down")}>{usd(m.net, 2)}</span></div><div><span className="dim">per trade </span><span className="num">{usd(m.expectancy, 2)}</span>{m.ci95 && <span className="dim"> ({usd(m.ci95[0])} … {usd(m.ci95[1])})</span>}</div>
        <div><span className="dim">win rate </span><span className="num">{m.win_rate ?? "–"}%</span></div><div><span className="dim">largest fall </span><span className="num">{usd(-m.max_drawdown)}</span></div>
        <div><span className="dim">before {split} </span><span className="num">{usd(m.earlier?.expectancy, 2)}</span><span className="dim"> ×{m.earlier?.trades}</span></div><div><span className="dim">after </span><span className="num">{usd(m.later?.expectancy, 2)}</span><span className="dim"> ×{m.later?.trades}</span></div>
        <div><span className="dim">2% worse fills </span><span className="num">{usd(m.sensitivity?.["2%"])}</span></div><div><span className="dim">5% worse fills </span><span className="num">{usd(m.sensitivity?.["5%"])}</span></div>
      </div>
      {not.length > 0 && <div className="mt-2 text-xs text-warn">Not priced (not losses): {not.map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ")}.{m.coverage?.fetchable > 0 && onAsk ? <button className="ml-2 underline" onClick={() => onAsk("Estimate what it would cost to acquire the missing observations")}>What would the missing observations take?</button> : null}</div>}
      <div className="mt-3 max-h-64 overflow-auto border border-line">
        <table className="t"><thead><tr><th>Session</th><th>In</th><th>Out</th><th>Contract</th><th className="r">Paid</th><th className="r">Sold</th><th className="r">Result</th><th>Why it closed</th></tr></thead>
          <tbody>{t.isLoading ? <tr><td colSpan={8} className="dim">Loading…</td></tr> : (t.data || []).map((x: any, i: number) => (
            <tr key={i}><td className="num">{x.date}</td><td className="num">{x.entry_time}</td><td className="num">{x.exit_date && x.exit_date !== x.date ? `${x.exit_date.slice(5)} ` : ""}{x.exit_time}</td><td className="num">{x.strike}{x.right} {x.expiration?.slice(5)}{x.legs > 1 ? ` +${x.legs - 1}` : ""}</td>
              <td className="r num">{x.entry?.toFixed(2)}</td><td className="r num">{x.exit?.toFixed(2)}</td><td className={cn("r num", x.pnl > 0 ? "text-up" : "text-down")}>{usd(x.pnl, 2)}</td><td className="dim">{String(x.reason || "").replace(/_/g, " ")}</td></tr>))}</tbody></table>
      </div>
    </div>
  );
}
