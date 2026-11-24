import { useMutation, useQuery } from "@tanstack/react-query";
import { Briefcase, Play, Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { cancelJob, Empty, ErrorBox, JobBar, Loading, PageHeader, Panel, StatusDot, SyntheticBadge, Warnings } from "@/components/common";
import { Button, Field, Input, NumInput, Select } from "@/components/ui";
import { api, queryClient, useJob, useRun } from "@/lib/api";
import { ago, cn, money, num, pct } from "@/lib/format";
import { useBuilder } from "@/lib/store";
import type { RunRow, StrategyConfig } from "@/lib/types";
import { KpiStrip } from "./results/RunDetail";

export default function Portfolio() {
  const { id } = useParams();
  const nav = useNavigate();
  const runs = useQuery({ queryKey: ["runs", "portfolio"], queryFn: () => api.get<RunRow[]>("/api/backtests?kind=portfolio"), refetchInterval: 4000 });
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Portfolio" sub={<span className="text-2xs text-muted">combined testing with shared capital, margin and position limits</span>}>
        <Button variant={id ? "default" : "primary"} onClick={() => nav("/portfolio")}><Plus /> New portfolio test</Button>
      </PageHeader>
      <div className="grid min-h-0 flex-1 grid-cols-[250px_1fr]">
        <div className="min-h-0 overflow-auto border-r border-line bg-bg-1">
          <div className="label px-2.5 pt-2 pb-1">Portfolio runs</div>
          {(runs.data || []).map((r) => (
            <button key={r.id} onClick={() => nav(`/portfolio/${r.id}`)} className={cn("flex w-full flex-col gap-0.5 border-l-2 px-2.5 py-1.5 text-left hover:bg-bg-2", id === r.id ? "border-accent bg-bg-3" : "border-transparent")}>
              <span className="flex items-center gap-1.5 text-xs"><StatusDot status={r.status} /><span className="truncate">{r.name.replace("Portfolio: ", "")}</span></span>
              <span className="num text-2xs text-muted">{r.status === "done" ? `${money(r.summary?.net_pnl, 0, true)} combined · ${money(r.summary?.naive_net_pnl, 0, true)} naive` : r.status}</span>
              <span className="text-2xs text-faint">{ago(r.created_at)}</span>
            </button>
          ))}
          {!runs.data?.length && <div className="px-2.5 text-2xs text-faint">No portfolio tests yet.</div>}
        </div>
        <div className="min-h-0 overflow-auto">{id ? <PortfolioRun id={id} /> : <NewPortfolio />}</div>
      </div>
    </div>
  );
}

type Item = { key: string; label: string; config: StrategyConfig; allocation: number };

function NewPortfolio() {
  const nav = useNavigate();
  const [sp] = useSearchParams();
  const b = useBuilder();
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies") });
  const [items, setItems] = useState<Item[]>([]);
  const [pick, setPick] = useState("builder");
  const [capital, setCapital] = useState(b.capital.initial_capital);
  const [maxBp, setMaxBp] = useState(b.capital.max_bp_usage_pct);
  const [maxPos, setMaxPos] = useState(b.capital.max_concurrent_positions);
  const [start, setStart] = useState(b.start);
  const [end, setEnd] = useState(b.end);
  useEffect(() => {
    if (sp.get("add") === "builder" && !items.length) setItems([{ key: "builder", label: `${b.strategy.name} (draft)`, config: b.strategy, allocation: 0.5 }]);
  }, []); // eslint-disable-line
  const add = () => {
    if (pick === "builder") setItems([...items, { key: `builder${items.length}`, label: `${b.strategy.name} (draft)`, config: b.strategy, allocation: 0.25 }]);
    else {
      const s = strategies.data?.find((x) => x.id === pick);
      if (s) setItems([...items, { key: s.id + items.length, label: s.name, config: s.config, allocation: 0.25 }]);
    }
  };
  const total = items.reduce((a, x) => a + x.allocation, 0);
  const run = useMutation({
    mutationFn: () => {
      const names = new Set<string>();
      const strats = items.map((x) => {
        let name = x.config.name;
        while (names.has(name)) name += "′";
        names.add(name);
        return { ...x.config, name, allocation: x.allocation };
      });
      return api.post<{ id: string }>("/api/portfolios", { config: { strategies: strats, start, end, capital: { ...b.capital, initial_capital: capital, max_bp_usage_pct: maxBp, max_concurrent_positions: maxPos }, execution: b.execution, allow_synthetic: true, label: "" } });
    },
    onSuccess: (r) => { queryClient.invalidateQueries({ queryKey: ["runs", "portfolio"] }); nav(`/portfolio/${r.id}`); },
    onError: (e: any) => toast.error(e.message),
  });
  return (
    <div className="p-3 space-y-3 max-w-5xl">
      <Panel title="Strategies & capital allocation" right={<span className={cn("num text-2xs", total > 1.0001 ? "text-down" : "text-muted")}>allocated {pct(total, 0)} of shared equity</span>}>
        <table className="tbl">
          <thead><tr><th>Strategy</th><th>Symbol</th><th>Legs</th><th className="r">Allocation</th><th /></tr></thead>
          <tbody>
            {items.map((x, i) => (
              <tr key={x.key}>
                <td className="text-fg">{x.label}</td><td className="num">{x.config.symbol}</td>
                <td className="num text-2xs text-fg-2">{x.config.legs.map((l) => `${l.side === "buy" ? "+" : "-"}${l.quantity}${l.kind === "stock" ? "sh" : l.right} ${l.dte}d`).join(" ")}</td>
                <td className="r w-32"><NumInput sz="sm" value={Math.round(x.allocation * 1000) / 10} suffix="%" onChange={(v) => setItems(items.map((y, j) => (j === i ? { ...y, allocation: Math.max(0.001, (v ?? 0) / 100) } : y)))} /></td>
                <td className="w-8"><Button size="iconSm" variant="ghost" onClick={() => setItems(items.filter((_, j) => j !== i))}><Trash2 /></Button></td>
              </tr>
            ))}
            {!items.length && <tr><td colSpan={5} className="text-faint py-3">Add two or more strategies. Allocation is the share of portfolio equity each strategy sizes against; all of them draw on the same cash and buying power.</td></tr>}
          </tbody>
        </table>
        <div className="flex gap-2 p-2 border-t border-line">
          <Select className="w-80" value={pick} onChange={setPick} options={[{ value: "builder", label: `Builder draft: ${b.strategy.name}` }, ...(strategies.data || []).map((s) => ({ value: s.id, label: `${s.name} · ${s.symbol}` }))]} />
          <Button onClick={add}><Plus /> Add</Button>
        </div>
      </Panel>
      <div className="grid grid-cols-5 gap-2">
        <Field label="Shared capital"><NumInput value={capital} step={10000} onChange={(v) => setCapital(v ?? 100000)} /></Field>
        <Field label="Max BP usage %"><NumInput value={maxBp} onChange={(v) => setMaxBp(v ?? 100)} suffix="%" /></Field>
        <Field label="Max concurrent positions"><NumInput value={maxPos} onChange={(v) => setMaxPos(v ?? 50)} /></Field>
        <Field label="From"><Input type="date" className="num" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
        <Field label="To"><Input type="date" className="num" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
      </div>
      <Button variant="primary" disabled={items.length < 1 || total > 1.0001 || run.isPending} onClick={() => run.mutate()}><Play /> Run combined portfolio</Button>
      <p className="text-2xs text-faint max-w-3xl leading-relaxed">
        The combined run executes every strategy inside one event loop with a single account, so concurrent positions compete for cash, buying power and the
        portfolio position limit. Each strategy is also backtested alone on its allocated capital; the difference between the combined result and the naive
        sum of standalone results shows what independent backtests overstate.
      </p>
    </div>
  );
}

function PortfolioRun({ id }: { id: string }) {
  const run = useRun(id, true);
  const job = useJob(run.data && ["queued", "running"].includes(run.data.status) ? id : null);
  if (run.isLoading) return <Loading />;
  const r = run.data!;
  if (["queued", "running"].includes(r.status)) return <div className="flex h-full items-center justify-center"><JobBar job={job || { id, status: r.status, progress: r.progress, message: r.message }} onCancel={() => cancelJob(id)} /></div>;
  if (r.status !== "done") return <ErrorBox className="m-3" error={`${r.status}: ${r.message}\n${r.error || ""}`} />;
  return <PortfolioResult run={r} />;
}

function PortfolioResult({ run }: { run: RunRow }) {
  const nav = useNavigate();
  const res = run.result;
  const pf = res.portfolio;
  const init = res.config.capital.initial_capital;
  const eq = useMemo(() => [
    { type: "line" as const, data: pf.dates.map((d: string, i: number) => ({ time: dayTime(d), value: pf.combined_equity[i] })), options: { color: C.accent, lineWidth: 2, title: "combined (shared capital)" } },
    { type: "line" as const, data: pf.dates.map((d: string, i: number) => ({ time: dayTime(d), value: pf.naive_sum_equity[i] })), options: { color: C.info, lineWidth: 1, lineStyle: 2, title: "naive sum" } },
    { type: "line" as const, data: res.series.dates.map((d: string, i: number) => ({ time: dayTime(d), value: res.series.benchmark[i] })).filter((p: any) => p.value !== null), options: { color: "#56b6c2", lineWidth: 1, title: "buy & hold" } },
    { type: "line" as const, data: res.series.dates.map((d: string, i: number) => ({ time: dayTime(d), value: res.series.cash[i] })), options: { color: C.faint, lineWidth: 1, lineStyle: 2, title: "cash" } },
  ], [pf, res]);
  const contrib = useMemo(() => pf.strategies.map((s: string, k: number) => ({ type: "line" as const, data: pf.dates.map((d: string, i: number) => ({ time: dayTime(d), value: pf.per_strategy_pnl[s]?.[i] })), options: { color: C.palette[k % C.palette.length], lineWidth: 1.5, title: s } })), [pf]);
  const corr = pf.correlation;
  const concurrency = {
    grid: { left: 40, right: 12, top: 24, bottom: 26 }, legend: { data: pf.strategies }, tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: pf.dates, axisLabel: { formatter: (v: string) => v.slice(0, 7) } }, yAxis: { type: "value" },
    series: pf.strategies.map((s: string, k: number) => ({ name: s, type: "line", stack: "p", areaStyle: { opacity: 0.35 }, symbol: "none", lineStyle: { width: 0.8 }, color: C.palette[k % C.palette.length], data: pf.concurrent_positions[s] })),
  };
  const contribBars = {
    grid: { left: 140, right: 20, top: 10, bottom: 24 }, tooltip: { trigger: "axis" },
    xAxis: { type: "value", axisLabel: { formatter: fmtMoneyAxis } }, yAxis: { type: "category", data: pf.rows.map((r: any) => r.strategy) },
    series: [{ name: "combined", type: "bar", data: pf.rows.map((r: any) => ({ value: r.combined_pnl, itemStyle: { color: r.combined_pnl >= 0 ? C.up : C.down } })), barMaxWidth: 14 },
      { name: "standalone", type: "bar", data: pf.rows.map((r: any) => r.standalone_pnl), itemStyle: { color: "rgba(122,132,150,0.45)" }, barMaxWidth: 14 }],
  };
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 border-b border-line bg-bg-1 px-3 h-9">
        <Briefcase className="size-3.5 text-accent" /><span className="text-sm font-medium truncate">{run.name}</span><SyntheticBadge on={res.meta.synthetic} />
        <span className="num text-2xs text-muted">{res.config.start} → {res.config.end} · shared ${(init / 1000).toFixed(0)}k</span>
        <span className="flex-1" />
        <Button variant="ghost" onClick={() => nav(`/results/${run.id}`)}>Full results & trades</Button>
      </div>
      <KpiStrip m={res.metrics} />
      <div className="border-b border-line p-2.5"><Warnings items={[
        { level: "info", text: `Combined P&L ${money(pf.combined_equity.at(-1) - init, 0, true)} vs naive sum of standalone backtests ${money(pf.naive_sum_equity.at(-1) - init, 0, true)}. Max drawdown combined ${pct(pf.combined_max_dd, 1)} vs naive ${pct(pf.naive_max_dd, 1)}.` },
        ...Object.entries(pf.capital_constrained_skips || {}).map(([k, v]) => ({ level: "warn", text: `${v} entries skipped by shared constraints: ${k}` })),
      ]} /></div>
      <div className="grid grid-cols-3 gap-px bg-line">
        <Panel className="col-span-2 border-0" title="Combined equity vs naive sum, buy & hold and cash"><LWChart series={eq} height={300} /></Panel>
        <Panel className="border-0" title="Correlation of daily P&L">
          {corr.labels.length > 1 ? <EChart height={300} option={{
            grid: { left: 100, right: 20, top: 10, bottom: 70 }, tooltip: { formatter: (p: any) => `${corr.labels[p.value[0]]} × ${corr.labels[p.value[1]]}: ${p.value[2]}` },
            xAxis: { type: "category", data: corr.labels, axisLabel: { rotate: 30, fontSize: 9 } }, yAxis: { type: "category", data: corr.labels, inverse: true, axisLabel: { fontSize: 9 } },
            visualMap: { min: -1, max: 1, show: false, inRange: { color: ["#4c8dff", "#141922", "#ef5350"] } },
            series: [{ type: "heatmap", data: corr.matrix.flatMap((row: any[], i: number) => row.map((v, j) => [j, i, v === null ? "-" : +v.toFixed(2)])), label: { show: true, color: "#d6dce5", fontSize: 10, fontFamily: "JetBrains Mono" } }],
          } as any} /> : <Empty title="Need two or more strategies" />}
        </Panel>
        <Panel className="col-span-2 border-0" title="Cumulative P&L contribution by strategy (combined run)"><LWChart series={contrib} height={240} /></Panel>
        <Panel className="border-0" title="P&L: combined vs standalone"><EChart height={240} option={contribBars as any} /></Panel>
        <Panel className="col-span-2 border-0" title="Simultaneous positions (max per session)"><EChart height={220} option={concurrency as any} /></Panel>
        <Panel className="border-0" title="Per-strategy">
          <table className="tbl">
            <thead><tr><th>Strategy</th><th className="r">Alloc</th><th className="r">Trades comb/alone</th><th className="r">Contribution</th><th className="r">Alone Sharpe</th></tr></thead>
            <tbody>{pf.rows.map((r: any) => (
              <tr key={r.strategy}><td className="text-fg truncate max-w-40">{r.strategy}</td><td className="r num">{pct(r.allocation, 0)}</td><td className="r num">{r.combined_trades}/{r.standalone_trades}</td><td className="r num">{pct(r.contribution_pct, 0)}</td><td className="r num">{num(r.standalone_sharpe)}</td></tr>
            ))}</tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}
