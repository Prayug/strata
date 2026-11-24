import { useMutation, useQuery } from "@tanstack/react-query";
import { BarChart3, Blocks, Briefcase, Database, FlaskConical, Microscope, Sigma, Sparkles } from "lucide-react";
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { DataTable, Empty, Loading, PageHeader, Panel, Pnl, Stat, StatusDot, SyntheticBadge, type Col } from "@/components/common";
import { Button } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { ago, bytes, money, num, pct } from "@/lib/format";
import type { RunRow } from "@/lib/types";

export default function Overview() {
  const nav = useNavigate();
  const ov = useQuery({ queryKey: ["overview"], queryFn: () => api.get<any>("/api/overview"), refetchInterval: 5000 });
  const demo = useQuery({ queryKey: ["candles", "DEMO", "1d"], queryFn: () => api.get<any>("/api/data/candles/DEMO?tf=1d") });
  const vix = useQuery({ queryKey: ["candles", "DEMOVIX", "1d"], queryFn: () => api.get<any>("/api/data/candles/DEMOVIX?tf=1d") });
  const seed = useMutation({ mutationFn: () => api.post<any[]>("/api/strategies/examples"), onSuccess: (r) => { queryClient.invalidateQueries({ queryKey: ["strategies"] }); queryClient.invalidateQueries({ queryKey: ["overview"] }); toast.success(`Added ${r.length} example strategies`); nav("/strategies"); } });
  const market = useMemo(() => {
    const bars = demo.data?.bars || [];
    const v = vix.data?.bars || [];
    const t = (s: number) => dayTime(new Date(s * 1000).toISOString().slice(0, 10));
    return [
      { type: "candles" as const, data: bars.map((b: number[]) => ({ time: t(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })) },
      { type: "line" as const, pane: 1, data: v.map((b: number[]) => ({ time: t(b[0]), value: b[4] })), options: { color: C.synth, lineWidth: 1, title: "DEMOVIX" } },
    ];
  }, [demo.data, vix.data]);
  if (ov.isLoading) return <Loading />;
  const d = ov.data || {};
  const runs: RunRow[] = d.recent_runs || [];
  const done = runs.filter((r) => r.status === "done" && (r.kind === "backtest" || r.kind === "portfolio"));
  const cols: Col<RunRow>[] = [
    { key: "s", label: "", render: (r) => <StatusDot status={r.status} />, width: 14 },
    { key: "n", label: "Recent activity", render: (r) => <span className="flex items-center gap-1.5"><span className="truncate max-w-80 text-fg">{r.name}</span><SyntheticBadge on={r.synthetic} /></span> },
    { key: "k", label: "Kind", render: (r) => <span className="text-muted">{r.kind.replace("_", " ")}</span> },
    { key: "p", label: "Result", align: "r", render: (r) => r.kind === "optimization" ? <span className="num text-fg-2">{r.summary?.n_configs ?? "—"} cfgs · DSR {num(r.summary?.dsr)}</span> : r.summary?.net_pnl !== undefined ? <Pnl v={r.summary.net_pnl} fmt={(x) => money(x, 0, true)} /> : <span className="text-muted">{r.status === "running" ? `${Math.round(r.progress * 100)}%` : r.status}</span> },
    { key: "sh", label: "Sharpe", align: "r", render: (r) => <span className="num">{num(r.summary?.sharpe)}</span> },
    { key: "t", label: "", align: "r", render: (r) => <span className="text-muted">{ago(r.created_at)}</span> },
  ];
  const go = (r: RunRow) => nav(r.kind === "optimization" ? `/optimize/${r.id}` : r.kind === "portfolio" ? `/portfolio/${r.id}` : r.kind === "backtest" ? `/results/${r.id}` : "/data");
  const best = done.filter((r) => r.summary?.sharpe !== null && r.summary?.sharpe !== undefined).sort((a, b) => (b.summary.sharpe ?? 0) - (a.summary.sharpe ?? 0))[0];
  const symbols = d.symbols || [];
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Overview" sub={<span className="text-2xs text-muted">engine {d.engine_version}</span>}>
        <Button onClick={() => nav("/data")}><Database /> Data</Button>
        <Button variant="primary" onClick={() => nav("/builder")}><Blocks /> New strategy</Button>
      </PageHeader>
      <div className="grid grid-cols-6 border-b border-line bg-bg-1 [&>*]:border-r [&>*]:border-line">
        <Stat label="Datasets" value={symbols.length} sub={`${symbols.filter((s: any) => s.has_options).length} with option chains`} />
        <Stat label="Synthetic" value={symbols.filter((s: any) => s.synthetic).length} tone="text-synth" sub="demo symbols" />
        <Stat label="Saved strategies" value={d.strategies} />
        <Stat label="Completed runs" value={done.length} sub="last 12 shown" />
        <Stat label="Best recent Sharpe" value={num(best?.summary?.sharpe)} sub={best?.name} />
        <Stat label="Storage" value={bytes(d.storage_bytes)} sub="Parquet market store" />
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] gap-px overflow-auto bg-line">
        <div className="flex min-w-0 flex-col gap-px bg-line">
          <Panel className="border-0" title="DEMO (synthetic) · daily candles with DEMOVIX" right={<SyntheticBadge on />}>
            {demo.data?.bars?.length ? <LWChart series={market} height={330} /> : <Empty icon={FlaskConical} title="No demo data">Generate the synthetic dataset in Data Management, or run <span className="font-mono">strata init --demo</span>.</Empty>}
          </Panel>
          <Panel className="border-0" title="Datasets">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Provider</th><th>Range</th><th className="r">Storage</th></tr></thead>
              <tbody>{symbols.map((s: any) => (
                <tr key={s.symbol} className="cursor-pointer" onClick={() => nav("/data")}><td className="flex items-center gap-1.5 text-fg">{s.symbol}<SyntheticBadge on={s.synthetic} /></td><td className="text-fg-2">{s.provider}</td><td className="num text-fg-2">{s.first} → {s.last}</td><td className="r num">{bytes(s.storage_bytes)}</td></tr>
              ))}</tbody>
            </table>
          </Panel>
        </div>
        <div className="flex min-w-0 flex-col gap-px bg-line">
          <Panel className="border-0" title="Workflow">
            <div className="grid grid-cols-2 gap-px bg-line">
              {[
                [Blocks, "Build", "Assemble legs, entry rules and exits visually", "/builder"],
                [BarChart3, "Backtest", "Event-driven, bid/ask-aware, reproducible", "/results"],
                [Sigma, "Optimise", "Grid / random / Bayesian with DSR & PBO", "/optimize"],
                [Microscope, "Research", "Conditional distributions & SQL on the data", "/research"],
                [Briefcase, "Portfolio", "Shared capital, margin and concurrency", "/portfolio"],
                [Database, "Data", "Providers, imports, coverage & gaps", "/data"],
              ].map(([I, t, s, to]: any) => (
                <button key={t} onClick={() => nav(to)} className="flex items-start gap-2 bg-bg-1 p-2.5 text-left hover:bg-bg-2">
                  <I className="mt-0.5 size-3.5 text-accent" /><span><span className="block text-xs text-fg">{t}</span><span className="text-2xs text-muted">{s}</span></span>
                </button>
              ))}
            </div>
            {d.strategies === 0 && (
              <div className="flex items-center justify-between gap-2 border-t border-line p-2.5">
                <span className="text-xs text-fg-2">No saved strategies yet. Load seven ready-to-run examples on the demo data?</span>
                <Button size="sm" onClick={() => seed.mutate()}><Sparkles /> Add examples</Button>
              </div>
            )}
          </Panel>
          <Panel className="border-0 flex-1" title="Recent runs">
            {runs.length ? <DataTable rows={runs} cols={cols} rowKey={(r) => r.id} onRowClick={go} /> : <Empty title="Nothing run yet">Press ⌘↵ in the Strategy Builder to run your first backtest.</Empty>}
          </Panel>
          {best && (
            <Panel className="border-0" title="Best recent backtest">
              <div className="grid grid-cols-4">
                <Stat label="Run" value={<span className="text-xs">{best.name}</span>} />
                <Stat label="Net P&L" value={money(best.summary.net_pnl, 0, true)} tone={best.summary.net_pnl >= 0 ? "text-up" : "text-down"} />
                <Stat label="Sharpe" value={num(best.summary.sharpe)} />
                <Stat label="Max DD" value={pct(best.summary.max_drawdown, 1)} tone="text-down" />
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
