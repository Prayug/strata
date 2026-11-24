import { useMutation, useQueries, useQuery } from "@tanstack/react-query";
import { Blocks, Copy, FolderOpen, GitCompare, Play, Sparkles, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { DataTable, Empty, Loading, PageHeader, Panel, Pnl, type Col } from "@/components/common";
import { Button, Input } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { ago, cn, money, num, pct } from "@/lib/format";
import { useBuilder, usesSandbox } from "@/lib/store";

export default function Strategies() {
  const nav = useNavigate();
  const list = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies") });
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string[]>([]);
  const [compare, setCompare] = useState(false);
  const inv = () => queryClient.invalidateQueries({ queryKey: ["strategies"] });
  const dup = useMutation({ mutationFn: (id: string) => api.post(`/api/strategies/${id}/duplicate`), onSuccess: inv });
  const del = useMutation({ mutationFn: (id: string) => api.del(`/api/strategies/${id}`), onSuccess: inv });
  const seed = useMutation({ mutationFn: () => api.post<any[]>("/api/strategies/examples"), onSuccess: (r) => { inv(); toast.success(`Added ${r.length} example strategies`); } });
  const run = useMutation({
    mutationFn: (s: any) => api.post<{ id: string }>("/api/backtests", { config: { ...useBuilder.getState().backtestConfig(), strategies: [s.config], label: s.name, allow_synthetic: usesSandbox([s.config]) }, strategy_id: s.id }),
    onSuccess: (r) => nav(`/results/${r.id}`), onError: (e: any) => toast.error(e.message),
  });
  const rows = (list.data || []).filter((s) => !q || s.name.toLowerCase().includes(q.toLowerCase()) || s.symbol.toLowerCase().includes(q.toLowerCase()));
  const open = (s: any) => { useBuilder.getState().load(s.config, s.id); nav("/builder"); };
  const cols: Col<any>[] = [
    { key: "sel", label: "", width: 18, render: (s) => <input type="checkbox" className="accent-[var(--color-accent)]" checked={sel.includes(s.id)} onClick={(e) => e.stopPropagation()} onChange={(e) => setSel(e.target.checked ? [...sel, s.id] : sel.filter((x) => x !== s.id))} /> },
    { key: "name", label: "Strategy", render: (s) => <span className="text-fg">{s.name}</span>, sort: (s) => s.name },
    { key: "sym", label: "Symbol", render: (s) => <span className="num">{s.symbol}</span>, sort: (s) => s.symbol },
    { key: "legs", label: "Structure", render: (s) => <span className="num text-2xs text-fg-2">{s.config.legs.map((l: any) => l.kind === "stock" ? `${l.side === "buy" ? "+" : "-"}${l.quantity}sh` : `${l.side === "buy" ? "+" : "-"}${l.quantity}${l.right} ${l.dte}d`).join(" · ")}</span> },
    { key: "exits", label: "Exits", render: (s) => <span className="text-2xs text-muted">{[s.config.exits.profit_target.enabled && `PT ${s.config.exits.profit_target.value}%`, s.config.exits.stop_loss.enabled && `SL ${s.config.exits.stop_loss.value}%`, s.config.exits.time_exit?.enabled && `@${s.config.exits.time_exit.time}`, s.config.exits.dte_exit?.enabled && `DTE≤${s.config.exits.dte_exit.dte}`].filter(Boolean).join(" · ")}</span> },
    { key: "pnl", label: "Last run P&L", align: "r", render: (s) => s.last_run ? <Pnl v={s.last_run.summary?.net_pnl} fmt={(x) => money(x, 0, true)} /> : <span className="text-faint">—</span>, sort: (s) => s.last_run?.summary?.net_pnl },
    { key: "sharpe", label: "Sharpe", align: "r", render: (s) => <span className="num">{num(s.last_run?.summary?.sharpe)}</span>, sort: (s) => s.last_run?.summary?.sharpe },
    { key: "dd", label: "Max DD", align: "r", render: (s) => <span className="num text-down">{pct(s.last_run?.summary?.max_drawdown, 1)}</span> },
    { key: "trials", label: "Trials", align: "r", render: (s) => <span className="num text-muted" title="Configurations evaluated in optimisation studies (used for multiple-testing adjustment)">{s.trials || 0}</span> },
    { key: "upd", label: "Updated", align: "r", render: (s) => <span className="text-muted">{ago(s.updated_at)}</span>, sort: (s) => s.updated_at },
    { key: "act", label: "", render: (s) => (
      <span className="flex gap-0.5" onClick={(e) => e.stopPropagation()}>
        <Button size="iconSm" variant="ghost" title="Run backtest (builder date range & capital)" onClick={() => run.mutate(s)}><Play /></Button>
        <Button size="iconSm" variant="ghost" title="Duplicate" onClick={() => dup.mutate(s.id)}><Copy /></Button>
        <Button size="iconSm" variant="ghost" title="Delete" onClick={() => confirm(`Delete strategy "${s.name}"? Its past runs are kept.`) && del.mutate(s.id)}><Trash2 /></Button>
      </span>
    ) },
  ];
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Saved Strategies" sub={<span className="text-2xs text-muted">{rows.length} strategies</span>}>
        <Input sz="sm" className="w-52" placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Button disabled={sel.length < 2} onClick={() => setCompare(true)}><GitCompare /> Compare {sel.length > 1 ? sel.length : ""}</Button>
        <Button variant="ghost" onClick={() => seed.mutate()}><Sparkles /> Add examples</Button>
        <Button variant="primary" onClick={() => nav("/builder")}><Blocks /> New</Button>
      </PageHeader>
      <div className={cn("min-h-0", compare ? "h-[38%] border-b border-line" : "flex-1")}>
        {list.isLoading ? <Loading /> : rows.length === 0 ? (
          <Empty icon={FolderOpen} title="No saved strategies" action={<Button onClick={() => seed.mutate()}><Sparkles /> Add example strategies</Button>}>Save a strategy from the builder (⌘S), or add a set of examples covering spreads, condors, strangles, calendars and covered calls.</Empty>
        ) : <DataTable rows={rows} cols={cols} rowKey={(s) => s.id} onRowClick={open} />}
      </div>
      {compare && <Compare ids={sel} strategies={list.data || []} onClose={() => setCompare(false)} />}
    </div>
  );
}

function Compare({ ids, strategies, onClose }: { ids: string[]; strategies: any[]; onClose: () => void }) {
  const chosen = strategies.filter((s) => ids.includes(s.id));
  const withRuns = chosen.filter((s) => s.last_run);
  const results = useQueries({ queries: withRuns.map((s) => ({ queryKey: ["run", s.last_run.id], queryFn: () => api.get<any>(`/api/runs/${s.last_run.id}`) })) });
  const loaded = results.map((r) => r.data).filter(Boolean);
  const series = useMemo(() => loaded.map((r: any, k: number) => ({
    type: "line" as const, data: r.result.series.dates.map((d: string, i: number) => ({ time: dayTime(d), value: (r.result.series.equity[i] / r.result.config.capital.initial_capital - 1) * 100 })),
    options: { color: C.palette[k % C.palette.length], lineWidth: 1.5, title: withRuns[k]?.name },
  })), [loaded.length]); // eslint-disable-line
  const corr = useMemo(() => {
    if (loaded.length < 2) return null;
    const maps = loaded.map((r: any) => new Map<string, number>(r.result.series.dates.map((d: string, i: number) => [d, r.result.series.daily_return[i]])));
    const common = [...maps[0].keys()].filter((d) => maps.every((m) => m.has(d)));
    const vec = maps.map((m) => common.map((d) => m.get(d) || 0));
    const c = (a: number[], b: number[]) => { const ma = a.reduce((x, y) => x + y, 0) / a.length, mb = b.reduce((x, y) => x + y, 0) / b.length; let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return sa && sb ? s / Math.sqrt(sa * sb) : 0; };
    return vec.map((a) => vec.map((b) => c(a, b)));
  }, [loaded.length]); // eslint-disable-line
  const metrics: [string, (m: any) => string][] = [["Net P&L", (m) => money(m.net_pnl, 0, true)], ["CAGR", (m) => pct(m.cagr, 1)], ["Sharpe", (m) => num(m.sharpe)], ["Sortino", (m) => num(m.sortino)], ["Max DD", (m) => pct(m.max_drawdown, 1)], ["Trades", (m) => String(m.trades)], ["Win rate", (m) => pct(m.win_rate, 0)], ["Profit factor", (m) => num(m.profit_factor)], ["Expectancy", (m) => money(m.expectancy, 2)], ["CVaR 95%", (m) => pct(m.cvar_95_daily, 2)], ["Max BP used", (m) => money(m.max_bp_used, 0)]];
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-8 items-center justify-between border-b border-line px-2.5"><span className="label text-fg-2">Comparison of each strategy's latest backtest</span><Button size="sm" variant="ghost" onClick={onClose}>Close</Button></div>
      {chosen.length > withRuns.length && <div className="px-2.5 py-1 text-2xs text-warn">{chosen.length - withRuns.length} selected strategies have no completed backtest yet — run them first.</div>}
      <div className="grid min-h-0 flex-1 grid-cols-[1.4fr_1fr] gap-px bg-line overflow-auto">
        <Panel className="border-0" title="Equity (% return), each on its own capital"><LWChart series={series} height={280} /></Panel>
        <Panel className="border-0" title="Daily-return correlation">
          {corr ? <EChart height={280} option={{ grid: { left: 110, right: 10, top: 10, bottom: 60 }, xAxis: { type: "category", data: withRuns.map((s) => s.name), axisLabel: { rotate: 25, fontSize: 9 } }, yAxis: { type: "category", data: withRuns.map((s) => s.name), inverse: true, axisLabel: { fontSize: 9 } }, visualMap: { min: -1, max: 1, show: false, inRange: { color: ["#4c8dff", "#141922", "#ef5350"] } }, series: [{ type: "heatmap", data: corr.flatMap((row, i) => row.map((v, j) => [j, i, +v.toFixed(2)])), label: { show: true, color: "#d6dce5", fontSize: 10 } }] } as any} /> : <Loading />}
        </Panel>
        <Panel className="col-span-2 border-0" title="Side by side">
          <table className="tbl">
            <thead><tr><th>Metric</th>{withRuns.map((s) => <th key={s.id} className="r">{s.name}</th>)}</tr></thead>
            <tbody>{metrics.map(([l, f]) => <tr key={l}><td className="text-muted">{l}</td>{loaded.map((r: any, i) => <td key={i} className="r num">{f(r.result.metrics)}</td>)}</tr>)}</tbody>
          </table>
          <div className="p-2 text-2xs text-faint">Side-by-side results are independent backtests. To see how they behave together with shared capital and margin, run them in the Portfolio page.</div>
        </Panel>
      </div>
    </div>
  );
}
