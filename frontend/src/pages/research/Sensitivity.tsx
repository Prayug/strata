/* The sensitivity explorer: one result seen across a grid of two assumptions, so a single best cell is never
   read without its neighbours. Every cell shows its sample; thin, incomplete, significant and stable cells are
   marked with glyphs and borders as well as colour, and every cell opens the sessions (or the backtest) behind it. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Grid3x3, Loader2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { pp, usd } from "@/lib/lab";
import { Callout, Empty, Skeleton, StatTile } from "../lab/bits";
import { GTable, pv } from "../lab/Research";

type Cell = any;
const FMT: Record<string, (v: number) => string> = {
  probability: (v) => pp(v), lift: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`, avg_end_return_pct: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}%`, q05_end_return_pct: (v) => `${v.toFixed(2)}%`, avg_mae_pct: (v) => `${v.toFixed(2)}%`,
  n: (v) => v.toLocaleString(), expectancy: (v) => usd(v), net_pnl: (v) => usd(v), max_drawdown: (v) => pp(v), win_rate: (v) => pp(v), trades: (v) => String(v),
};
const DIVERGING = new Set(["lift", "avg_end_return_pct", "expectancy", "net_pnl", "q05_end_return_pct", "avg_mae_pct"]);

function tint(layer: string, v: number | null | undefined, lo: number, hi: number, base?: number | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "transparent";
  if (DIVERGING.has(layer)) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const a = Math.min(1, Math.abs(v) / m) * 0.5;
    return v >= 0 ? `rgba(38,166,154,${a})` : `rgba(239,83,80,${a})`;
  }
  if (layer === "probability" && base !== null && base !== undefined) {
    const m = Math.max(Math.abs(hi - base), Math.abs(lo - base)) || 1;
    const a = Math.min(1, Math.abs(v - base) / m) * 0.5;
    return v >= base ? `rgba(38,166,154,${a})` : `rgba(239,83,80,${a})`;
  }
  return `rgba(76,141,255,${((v - lo) / ((hi - lo) || 1)) * 0.45})`;
}

export function Heatmap({ map, layer, onPick, picked }: { map: any; layer: string; onPick: (c: Cell) => void; picked: Cell | null }) {
  const cells: Cell[] = map.cells;
  const nx = map.x.labels.length, ny = map.y.labels.length;
  const vals = cells.filter((c) => c.status === "ok" && c[layer] !== null && c[layer] !== undefined).map((c) => c[layer] as number);
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
  const at = (xi: number, yi: number) => cells.find((c) => c.xi === xi && c.yi === yi);
  const f = FMT[layer] || ((v: number) => v.toFixed(2));
  return (
    <div className="overflow-auto" data-testid="heatmap">
      <table className="border-separate" style={{ borderSpacing: 3 }}>
        <thead>
          <tr><th /><th colSpan={nx} className="pb-1 text-center text-xs font-medium dim">{map.x.label}</th></tr>
          <tr><th className="pr-2 text-right align-bottom text-xs font-medium dim" style={{ maxWidth: 170 }}>{map.y.label}</th>{map.x.labels.map((l: string, i: number) => <th key={i} className="num px-1 pb-1 text-center text-[11.5px] font-normal soft">{l}</th>)}</tr>
        </thead>
        <tbody>
          {Array.from({ length: ny }, (_, yi) => (
            <tr key={yi}>
              <th className="num whitespace-nowrap pr-2 text-right text-[11.5px] font-normal soft">{map.y.labels[yi]}</th>
              {Array.from({ length: nx }, (_, xi) => {
                const c = at(xi, yi);
                if (!c || c.status !== "ok") return <td key={xi}><div className="grid h-[58px] w-[92px] place-items-center rounded-md border border-dashed border-line text-[11px] text-faint" title={c?.error}>no data</div></td>;
                const v = c[layer];
                const marks = [c.significant && "◆", c.thin && "○", c.incomplete && "≈"].filter(Boolean).join(" ");
                return (
                  <td key={xi}>
                    <Tip content={`${c.condition || `${map.x.label} ${c.x}, ${map.y.label} ${c.y}`}${c.outcome_text ? ` → ${c.outcome_text}` : ""}\n${c.n} ${map.map === "options" ? "trades" : "observations"}${c.thin ? " (too few to rely on)" : ""}${c.incomplete ? "\npart of the period could not be evaluated" : ""}${c.significant ? "\ndistinguishable from the baseline after correcting for every cell" : ""}${c.stable ? "\nneighbouring cells agree" : c.neighbours ? "\nneighbouring cells do not all agree" : ""}`}>
                      <button onClick={() => onPick(c)} data-testid="cell" data-thin={c.thin ? "1" : "0"} data-stable={c.stable ? "1" : "0"}
                        className={cn("relative flex h-[58px] w-[92px] flex-col items-center justify-center rounded-md text-center transition-shadow hover:shadow-[0_0_0_2px_rgba(232,163,61,0.6)]",
                          c.thin ? "border border-dashed border-faint" : c.stable ? "border-2 border-fg-2/70" : "border border-line-2", picked && picked.xi === xi && picked.yi === yi && "shadow-[0_0_0_2px_#e8a33d]")}
                        style={{ background: tint(layer, v, lo, hi, c.baseline_probability) }}>
                        <span className={cn("num text-[13.5px] font-medium", c.thin && "opacity-55")}>{v === null || v === undefined ? "—" : f(v)}</span>
                        <span className="num text-[10.5px] dim">n {c.n}</span>
                        {marks && <span className="absolute right-1 top-0.5 text-[10px] leading-none text-fg">{marks}</span>}
                      </button>
                    </Tip>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs dim">
        <span><span className="text-fg">◆</span> distinguishable after correction</span><span><span className="text-fg">○</span> dashed: fewer than 30 {map.map === "options" ? "trades" : "observations"}</span>
        <span><span className="text-fg">≈</span> part of the period not evaluable</span><span><span className="inline-block size-3 translate-y-0.5 rounded-sm border-2 border-fg-2/70" /> thick border: neighbours agree (stable region)</span>
        <span>colour: {DIVERGING.has(layer) ? "green above zero, red below" : layer === "probability" ? "green above the baseline, red below" : "darker is larger"}</span>
      </div>
    </div>
  );
}

export function CellPanel({ symbol, cell, start, end, priorTrials, onClose }: { symbol: string; cell: Cell; start?: string | null; end?: string | null; priorTrials: number; onClose: () => void }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["cell", symbol, JSON.stringify(cell.preds), JSON.stringify(cell.outcome), start, end], queryFn: () => api.post<any>("/api/sensitivity/cell", { symbol, preds: cell.preds, outcome: cell.outcome, start, end }) });
  const open = useMutation({
    mutationFn: () => api.post<{ id: string }>("/api/ai/sessions/from-cell", { symbol, preds: cell.preds, outcome: cell.outcome, prior_trials: priorTrials, start, end }),
    onSuccess: (r) => nav(`/lab/${r.id}`), onError: (e: any) => toast.error(e.message),
  });
  const d = q.data;
  const hist = useMemo(() => d?.histogram && ({
    grid: { left: 40, right: 10, top: 26, bottom: 30 }, legend: { data: ["qualifying sessions", "all other sessions"] }, tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: d.histogram.x.map((x: number) => x.toFixed(2)), name: "return over the horizon, %", nameLocation: "middle", nameGap: 20, axisLabel: { interval: 5 } },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [{ name: "qualifying sessions", type: "bar", data: d.histogram.condition, itemStyle: { color: C.accent, opacity: 0.8 }, barGap: "-100%" },
      { name: "all other sessions", type: "line", step: "middle", symbol: "none", data: d.histogram.baseline, lineStyle: { color: C.fg, width: 1.2, type: "dashed" }, color: C.fg }],
  }), [d]);
  return (
    <div className="flex h-full w-[440px] shrink-0 flex-col border-l border-line bg-bg-1" data-testid="cell-panel">
      <div className="flex items-start gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1"><div className="text-[13.5px] font-medium">{cell.condition || d?.condition}</div><div className="mt-0.5 text-xs dim">{cell.outcome_text || d?.outcome_text}</div></div>
        <button className="btn quiet sm !px-1" onClick={onClose} aria-label="Close"><X /></button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
        <div className="grid grid-cols-3 gap-2">
          <StatTile label="Observations" value={cell.n} tone={cell.thin ? "text-warn" : undefined} sub={cell.thin ? "too few to rely on" : undefined} />
          <StatTile label="Frequency" value={pp(cell.probability ?? cell.frequency)} sub={cell.ci95?.[0] != null ? `${pp(cell.ci95[0])} to ${pp(cell.ci95[1])}` : undefined} hint="The interval resamples whole calendar weeks." />
          <StatTile label="All sessions" value={pp(cell.baseline_probability ?? cell.baseline)} sub={cell.p_adjusted != null ? `p ${pv(cell.p_adjusted)} corrected` : cell.p_value != null ? `p ${pv(cell.p_value)}` : undefined} />
        </div>
        {q.isLoading && <Skeleton className="mt-3 h-40" />}
        {hist && <div className="mt-3"><div className="text-xs dim">Where the sessions ended, against all other sessions</div><EChart option={hist} height={190} /></div>}
        {d?.by_year?.length > 1 && (
          <div className="mt-2"><div className="mb-1 text-xs dim">By year (hits / sessions)</div>
            <div className="flex flex-wrap gap-1">{d.by_year.map((y: any) => <span key={y.year} className="num rounded bg-bg-2 px-1.5 py-0.5 text-[11px]" title={`${y.hits} of ${y.n}`}>{y.year.slice(2)} <span className="dim">{y.hits}/{y.n}</span></span>)}</div></div>
        )}
        {d && Object.keys(d.excluded || {}).length > 0 && <Callout tone="info" className="mt-3">{Object.entries(d.excluded).map(([k, v]) => `${v} sessions: ${k}`).join("; ")}.</Callout>}
        {d && <div className="mt-3"><GTable table={{ title: `Sessions (${d.rows.length.toLocaleString()}, newest first)`, columns: d.columns, rows: [...d.rows].reverse() }} max={300} /></div>}
      </div>
      <div className="border-t border-line px-4 py-3">
        <button className="btn primary w-full" disabled={open.isPending || !d?.study} onClick={() => open.mutate()} data-testid="cell-to-lab">{open.isPending ? <Loader2 className="animate-spin" /> : <ArrowUpRight />}Open as a study in the AI Lab</button>
        <div className="mt-1.5 text-[11px] text-faint">{d && !d.study ? "A move in either direction has no single study." : priorTrials > 1 ? `You are choosing one of ${priorTrials} cells; the notebook will count all ${priorTrials} as tried.` : "The notebook runs it session by session with a full report."}</div>
      </div>
    </div>
  );
}

export default function Sensitivity() {
  const nav = useNavigate();
  const presets = useQuery({ queryKey: ["sens-presets"], queryFn: () => api.get<any>("/api/sensitivity/presets"), staleTime: 600_000 });
  const ency = useQuery({ queryKey: ["ency-list"], queryFn: () => api.get<any>("/api/encyclopedia"), staleTime: 120_000 });
  const runs = useQuery({ queryKey: ["bt-for-maps"], queryFn: () => api.get<any[]>("/api/runs/search?kind=backtest&limit=40"), staleTime: 20_000 });
  const [kind, setKind] = useState<"underlying" | "options">("underlying");
  const [pid, setPid] = useState("decline_vs_horizon");
  const [symbol, setSymbol] = useState("SPY");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [layer, setLayer] = useState<string | null>(null);
  const [cell, setCell] = useState<Cell | null>(null);
  const [runId, setRunId] = useState("");
  const [job, setJob] = useState<string | null>(null);
  useEffect(() => { const s = ency.data?.symbols; if (s?.length && !s.includes(symbol)) setSymbol(s[0]); }, [ency.data]); // eslint-disable-line
  useEffect(() => { setCell(null); setLayer(null); }, [pid, kind, symbol]);
  const und = useQuery({ queryKey: ["sens-map", pid, symbol, start, end], enabled: kind === "underlying", staleTime: 300_000,
    queryFn: () => api.post<any>("/api/sensitivity/underlying", { symbol, preset: pid, start: start || null, end: end || null }) });
  const launch = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/sensitivity/options", { run_id: runId, preset: pid }), onSuccess: (r) => setJob(r.id), onError: (e: any) => toast.error(e.message) });
  const jobRun = useQuery({ queryKey: ["sens-job", job], enabled: !!job, queryFn: () => api.get<any>(`/api/runs/${job}`), refetchInterval: (s) => (["done", "error", "cancelled"].includes(s.state.data?.status) ? false : 1000) });
  const opt = useQuery({ queryKey: ["sens-opt", job], enabled: !!job && jobRun.data?.status === "done", queryFn: () => api.get<any>(`/api/runs/${job}/sensitivity`) });
  const map = kind === "underlying" ? und.data : opt.data;
  const list: any[] = (kind === "underlying" ? presets.data?.underlying : presets.data?.options) || [];
  const cur = list.find((p) => p.id === pid);
  const lay = layer || map?.layer || "lift";
  const openBt = useMutation({ mutationFn: (cfg: any) => api.post<{ id: string }>("/api/backtests", { config: cfg, name: `${map?.title}: ${cell?.x} / ${cell?.y}` }), onSuccess: (r) => nav(`/results/${r.id}`), onError: (e: any) => toast.error(e.message) });
  const okBt = (runs.data || []).filter((r) => r.status === "done" && r.trades !== undefined && r.trades !== null);
  return (
    <div className="lab flex h-full min-h-0">
      <aside className="flex w-[268px] shrink-0 flex-col gap-4 overflow-auto border-r border-line bg-bg-1 px-3.5 py-4">
        <div><div className="eyebrow flex items-center gap-2"><Grid3x3 className="size-3.5 text-accent" />Sensitivity explorer</div><div className="mt-1.5 text-[12.5px] dim">How a result changes when two assumptions change. A cell is only as trustworthy as its neighbours.</div></div>
        <div className="inline-flex rounded-lg border border-line-2 bg-bg-2 p-0.5 text-xs">
          {([["underlying", "Underlying behaviour"], ["options", "Option backtests"]] as const).map(([k, l]) => <button key={k} onClick={() => { setKind(k); setPid(k === "underlying" ? "decline_vs_horizon" : "dte_vs_delta"); }} className={cn("h-7 flex-1 rounded-md px-2", kind === k ? "bg-bg-4 text-fg" : "text-muted hover:text-fg")} data-testid={`kind-${k}`}>{l}</button>)}
        </div>
        <div className="flex flex-col gap-1">
          {list.map((p) => <button key={p.id} onClick={() => setPid(p.id)} className={cn("rounded-lg border px-3 py-2 text-left text-[13px] transition-colors", pid === p.id ? "border-accent/60 bg-accent/[0.07]" : "border-transparent hover:bg-bg-2")} data-testid={`map-${p.id}`}>{p.title}<div className="mt-0.5 text-[11.5px] dim">{p.question}</div></button>)}
        </div>
        {kind === "underlying" ? (
          <div className="flex flex-col gap-2 border-t border-line pt-3">
            <label className="text-xs dim">Symbol<select className="field mt-1" value={symbol} onChange={(e) => setSymbol(e.target.value)} aria-label="Symbol">{(ency.data?.symbols || [symbol]).filter((s: string) => !/VIX/.test(s)).map((s: string) => <option key={s}>{s}</option>)}</select></label>
            <label className="text-xs dim">From<input type="date" className="field mt-1" value={start} onChange={(e) => setStart(e.target.value)} aria-label="From" /></label>
            <label className="text-xs dim">To<input type="date" className="field mt-1" value={end} onChange={(e) => setEnd(e.target.value)} aria-label="To" /></label>
          </div>
        ) : (
          <div className="flex flex-col gap-2 border-t border-line pt-3">
            <label className="text-xs dim">Start from a finished option backtest
              <select className="field mt-1" value={runId} onChange={(e) => { setRunId(e.target.value); setJob(null); }} aria-label="Backtest"><option value="">Choose…</option>{okBt.map((r) => <option key={r.id} value={r.id}>{String(r.name).replace(/^v\d+ - /, "").slice(0, 46)} ({r.trades} trades)</option>)}</select></label>
            <button className="btn primary sm" disabled={!runId || launch.isPending || (!!job && !["done", "error"].includes(jobRun.data?.status))} onClick={() => launch.mutate()} data-testid="run-option-map">Run this map</button>
            <div className="text-[11.5px] text-faint">Re-runs the backtest for every cell on the option data already stored. Nothing is downloaded or modelled.</div>
          </div>
        )}
      </aside>
      <main className="min-w-0 flex-1 overflow-auto px-6 py-5">
        {cur && <><h2>{cur.title}</h2><p className="mt-1 text-[13.5px] soft">{cur.question}</p></>}
        {kind === "options" && !job && <div className="mt-6"><Empty icon={Grid3x3} title="Choose a backtest and run the map">{okBt.length ? "Each cell is a full backtest by the same engine, counted in the research ledger." : "No finished option backtest exists yet. Run one in the AI Lab first."}</Empty></div>}
        {kind === "options" && job && jobRun.data?.status !== "done" && <div className="card mt-5 px-4 py-3 text-[13px]">{jobRun.data?.status === "error" ? <span className="text-down">{String(jobRun.data?.error || "failed").split("\n")[0]}</span> : <><Loader2 className="mr-2 inline size-4 animate-spin text-accent" />{jobRun.data?.message || "queued"} <span className="num dim">{Math.round((jobRun.data?.progress || 0) * 100)}%</span></>}</div>}
        {kind === "underlying" && und.isLoading && <Skeleton className="mt-5 h-72" />}
        {kind === "underlying" && und.isError && <Callout tone="bad" className="mt-5">{String((und.error as any)?.message)}</Callout>}
        {map && (
          <div className="mt-4 flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-2">
              {Object.entries(map.layers as Record<string, string>).map(([k, l]) => <button key={k} className={cn("chip", lay === k && "on")} onClick={() => setLayer(k)} data-testid={`layer-${k}`}>{l}</button>)}
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="chip !h-6">{map.summary.cells} cells</span>
              <span className={cn("chip !h-6", map.summary.thin > 0 && "!border-warn/50 !text-warn")}>{map.summary.thin} with too few {map.map === "options" ? "trades" : "observations"}</span>
              {map.map !== "options" && <span className="chip !h-6">{map.summary.significant} distinguishable after correction</span>}
              <span className="chip !h-6">{map.summary.stable} in a stable region</span>
              {map.synthetic && <span className="chip !h-6 !border-synth/50 !text-synth">simulated sandbox data</span>}
              {map.window && <span className="chip !h-6 num">{map.window[0]} → {map.window[1]}</span>}
            </div>
            <Heatmap map={map} layer={lay} onPick={setCell} picked={cell} />
            <ul className="flex flex-col gap-1.5 text-[13px] soft">{map.notes.map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
            {map.ledger && <div className="text-xs dim">{map.ledger.configurations} configurations have been viewed in this explorer for {symbol}. Opening a cell in the AI Lab carries the size of the map with it.</div>}
          </div>
        )}
      </main>
      {cell && map?.map !== "options" && <CellPanel symbol={symbol} cell={cell} start={start || null} end={end || null} priorTrials={map?.summary.cells || 1} onClose={() => setCell(null)} />}
      {cell && map?.map === "options" && (
        <div className="flex w-[360px] shrink-0 flex-col border-l border-line bg-bg-1 px-4 py-3" data-testid="cell-panel">
          <div className="flex items-center gap-2"><div className="flex-1 text-[13.5px] font-medium">{map.x.label}: {cell.x} · {map.y.label}: {cell.y}</div><button className="btn quiet sm !px-1" onClick={() => setCell(null)} aria-label="Close"><X /></button></div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <StatTile label="Trades" value={cell.trades} tone={cell.thin ? "text-warn" : undefined} sub={cell.thin ? "too few to rely on" : undefined} /><StatTile label="Net" value={usd(cell.net_pnl)} tone={(cell.net_pnl ?? 0) >= 0 ? "text-up" : "text-down"} />
            <StatTile label="Per trade" value={usd(cell.expectancy, 2)} /><StatTile label="Win rate" value={pp(cell.win_rate)} />
          </div>
          <div className="mt-2 text-xs dim">{cell.stable ? "The neighbouring cells agree in sign." : cell.neighbours ? `Only ${cell.neighbours_agree} of ${cell.neighbours} neighbouring cells agree in sign: do not single this cell out.` : "No comparable neighbouring cells."}</div>
          <button className="btn primary mt-4" disabled={openBt.isPending} onClick={() => openBt.mutate(cell.config)}><ArrowUpRight />Open this backtest</button>
          <div className="mt-1.5 text-[11px] text-faint">Runs (or reuses) exactly this configuration and opens its full result.</div>
        </div>
      )}
    </div>
  );
}
