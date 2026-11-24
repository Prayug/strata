import { useMutation, useQuery } from "@tanstack/react-query";
import { Blocks, Download, PlayCircle, RotateCw, Sigma, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { cancelJob, DataTable, Empty, ErrorBox, JobBar, Loading, PageHeader, Panel, Pnl, Stat, SyntheticBadge, Warnings, type Col } from "@/components/common";
import { Button, Field, NumInput, Segmented, Select, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui";
import { api, queryClient, useJob, useRun } from "@/lib/api";
import { cn, etTime, minutesLabel, money, num, pct, REASON_LABEL, tone } from "@/lib/format";
import { useBuilder, useUi } from "@/lib/store";
import type { RunRow, Trade } from "@/lib/types";

export function RunDetail({ id }: { id: string }) {
  const run = useRun(id, true);
  const job = useJob(run.data && ["queued", "running"].includes(run.data.status) ? id : null);
  if (run.isLoading) return <Loading />;
  if (run.error) return <ErrorBox className="m-3" error={run.error} />;
  const r = run.data!;
  if (r.status === "queued" || r.status === "running") {
    return (
      <div className="flex h-full flex-col">
        <PageHeader title={r.name} sub={<SyntheticBadge on={r.synthetic} />} />
        <div className="flex flex-1 flex-col items-center justify-center gap-3">
          <div className="text-sm text-fg-2">Backtest {r.status === "queued" ? "queued" : "running"} in a background worker</div>
          <JobBar job={job || { id, status: r.status, progress: r.progress, message: r.message }} onCancel={() => cancelJob(id)} />
          <div className="text-2xs text-faint">Chronological event loop · conservative fills · results are cached by config hash + dataset version</div>
        </div>
      </div>
    );
  }
  if (r.status !== "done") {
    return (
      <div className="flex h-full flex-col">
        <PageHeader title={r.name} />
        <div className="p-3 space-y-2">
          <ErrorBox error={`Run ${r.status}: ${r.message || ""}\n\n${(r.error || "").slice(0, 3000)}`} />
          <RerunButton run={r} />
        </div>
      </div>
    );
  }
  return <ResultView run={r} />;
}

function RerunButton({ run }: { run: RunRow }) {
  const nav = useNavigate();
  const m = useMutation({
    mutationFn: () => api.post<{ id: string }>(run.kind === "portfolio" ? "/api/portfolios" : "/api/backtests", { config: run.config || run.result?.config, force: true, name: run.name }),
    onSuccess: (x) => nav(run.kind === "portfolio" ? `/portfolio/${x.id}` : `/results/${x.id}`),
    onError: (e: any) => toast.error(e.message),
  });
  return <Button onClick={() => m.mutate()} disabled={m.isPending}><RotateCw /> Re-run</Button>;
}

export function useTrades(id: string) {
  return useQuery({ queryKey: ["trades", id], queryFn: () => api.get<Trade[]>(`/api/backtests/${id}/trades`) });
}

export function KpiStrip({ m }: { m: any }) {
  const items: [string, React.ReactNode, string?, string?, string?][] = [
    ["Net P&L", money(m.net_pnl, 0, true), tone(m.net_pnl), pct(m.total_return, 2, true)],
    ["CAGR", pct(m.cagr, 1, true), tone(m.cagr), `${num(m.years, 1)} yrs`],
    ["Sharpe", num(m.sharpe), "", `Sortino ${num(m.sortino)}`, "Annualised from daily equity returns (excess of the configured risk-free rate)"],
    ["Calmar", num(m.calmar), "", `vol ${pct(m.volatility, 1)}`],
    ["Max DD", pct(m.max_drawdown, 1), "text-down", money(m.max_drawdown_dollars, 0), "Peak-to-trough on daily marked equity"],
    ["Trades", m.trades ?? 0, "", `${num(m.exposure ? m.exposure * 100 : null, 0)}% exposure`],
    ["Win rate", pct(m.win_rate, 1), "", `${m.max_consecutive_wins ?? 0}W / ${m.max_consecutive_losses ?? 0}L streak`],
    ["Expectancy", money(m.expectancy, 2, true), tone(m.expectancy), `±${money(m.expectancy_std_err, 2)} s.e.`, "Mean P&L per trade with its standard error"],
    ["Profit factor", num(m.profit_factor), "", `payoff ${num(m.payoff_ratio)}`],
    ["Avg win / loss", <span key="w"><span className="text-up">{money(m.avg_winner, 0)}</span> / <span className="text-down">{money(m.avg_loser, 0)}</span></span>, "", `worst ${money(m.largest_loser, 0)}`],
    ["Return on risk", pct(m.return_on_capital_at_risk, 1), "", `avg ${pct(m.avg_return_on_risk, 1)}/trade`, "Total P&L / total buying power committed"],
    ["Max BP used", money(m.max_bp_used, 0), "", `${pct(m.max_capital_utilization, 0)} of equity`],
    ["Capital util.", pct(m.avg_capital_utilization, 1), "", "average BP / equity"],
    ["CVaR 95% (day)", pct(m.cvar_95_daily, 2), "text-down", `VaR ${pct(m.var_95_daily, 2)}`, "Mean of the worst 5% of daily returns"],
    ["Trade CVaR 95%", money(m.trade_cvar_95, 0), "text-down", `t-stat ${num(m.t_stat)}`],
    ["Costs", money(-(m.total_commissions + m.total_slippage), 0), "text-down", `comm ${money(m.total_commissions, 0)} · slip ${money(m.total_slippage, 0)}`, "Commissions plus execution cost versus the quote midpoint"],
    ["Avg hold", minutesLabel(m.avg_holding_minutes), "", `median ${minutesLabel(m.median_holding_minutes)}`],
    ["Exposure-adj. return", pct(m.exposure_adjusted_return, 1), "", "CAGR / time in market"],
  ];
  return (
    <div className="grid grid-cols-9 border-b border-line bg-bg-1 [&>*]:border-r [&>*]:border-line [&>*:nth-child(9n)]:border-r-0 [&>*:nth-child(-n+9)]:border-b">
      {items.map(([l, v, t, s, h]) => <Stat key={l} label={l} value={v} tone={t} sub={s} hint={h} />)}
    </div>
  );
}

function ResultView({ run }: { run: RunRow }) {
  const nav = useNavigate();
  const res = run.result;
  const m = res.metrics;
  const meta = res.meta;
  const { returnMode, set } = useUi();
  const trades = useTrades(run.id);
  const del = useMutation({ mutationFn: () => api.del(`/api/runs/${run.id}`), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["runs"] }); nav("/results"); } });
  const openInBuilder = () => {
    const cfg = res.config;
    useBuilder.getState().load(cfg.strategies[0], run.strategy_id || null);
    useBuilder.getState().patch({ start: cfg.start, end: cfg.end, capital: cfg.capital, execution: cfg.execution });
    nav("/builder");
  };
  return (
    <div className="flex h-full flex-col">
      <PageHeader title={run.name} sub={<>
        <SyntheticBadge on={meta.synthetic} />
        <span className="num text-2xs text-muted">{res.config.start} → {res.config.end} · {meta.sessions} sessions · {meta.runtime_seconds}s</span>
        <Tip content={<div className="num text-2xs space-y-0.5"><div>config {meta.config_hash}</div>{Object.entries(meta.dataset_versions).map(([k, v]) => <div key={k}>{k} data {v as string}</div>)}<div>events {meta.events_version}</div><div>engine {meta.engine_version}</div></div>}>
          <span className="font-mono text-2xs text-faint cursor-help">#{meta.config_hash} · data {Object.values(meta.dataset_versions)[0] as string}</span>
        </Tip>
      </>}>
        <Segmented value={returnMode} onChange={(v) => set({ returnMode: v })} options={[{ value: "dollars", label: "$" }, { value: "pct", label: "%" }, { value: "risk", label: "R", title: "Risk-normalised: P&L divided by buying power at risk" }]} />
        <Button variant="ghost" onClick={openInBuilder}><Blocks /> Builder</Button>
        <Button variant="ghost" onClick={() => { openInBuilder(); nav("/optimize?from=builder"); }}><Sigma /> Optimise</Button>
        <a href={`/api/backtests/${run.id}/trades.csv`}><Button variant="ghost"><Download /> CSV</Button></a>
        <RerunButton run={run} />
        <Button variant="ghost" size="icon" onClick={() => confirm("Delete this run and its stored results?") && del.mutate()}><Trash2 /></Button>
      </PageHeader>
      <KpiStrip m={m} />
      {(m.warnings?.length > 0 || meta.synthetic) && (
        <div className="border-b border-line px-3 py-1.5">
          <Warnings items={[...(meta.synthetic ? [{ level: "info", text: "Computed on the SYNTHETIC demonstration dataset: performance reflects a simulated market, not history." }] : []), ...m.warnings.map((t: string) => ({ level: "warn", text: t }))]} />
        </div>
      )}
      <Tabs defaultValue="perf" className="flex min-h-0 flex-1 flex-col">
        <TabsList>
          <TabsTrigger value="perf">Performance</TabsTrigger>
          <TabsTrigger value="dist">Distribution</TabsTrigger>
          <TabsTrigger value="break">Breakdown</TabsTrigger>
          <TabsTrigger value="greeks">Greeks & risk</TabsTrigger>
          <TabsTrigger value="trades">Trades <span className="num text-faint">{m.trades}</span></TabsTrigger>
          <TabsTrigger value="mc">Monte Carlo</TabsTrigger>
          <TabsTrigger value="diag">Diagnostics</TabsTrigger>
        </TabsList>
        <div className="min-h-0 flex-1 overflow-auto">
          <TabsContent value="perf"><Performance res={res} trades={trades.data} mode={returnMode} /></TabsContent>
          <TabsContent value="dist"><Distribution res={res} trades={trades.data} mode={returnMode} /></TabsContent>
          <TabsContent value="break"><Breakdown res={res} mode={returnMode} /></TabsContent>
          <TabsContent value="greeks"><GreeksTab res={res} /></TabsContent>
          <TabsContent value="trades" className="h-full"><TradesTab runId={run.id} trades={trades.data} /></TabsContent>
          <TabsContent value="mc"><MonteCarlo runId={run.id} initial={res.config.capital.initial_capital} /></TabsContent>
          <TabsContent value="diag"><Diagnostics res={res} /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

function Performance({ res, trades, mode }: { res: any; trades?: Trade[]; mode: string }) {
  const s = res.series;
  const init = res.config.capital.initial_capital;
  const series = useMemo(() => {
    const d = s.dates as string[];
    const conv = (v: number) => (mode === "pct" ? (v / init - 1) * 100 : v);
    if (mode === "risk") {
      if (!trades) return [];
      const byDay = new Map<string, number>();
      for (const t of trades) {
        const k = (t.exit_time || t.entry_time).slice(0, 10);
        byDay.set(k, (byDay.get(k) || 0) + (t.return_on_risk || 0));
      }
      let cum = 0;
      return [{ type: "baseline" as const, data: d.map((x) => ({ time: dayTime(x), value: (cum += byDay.get(x) || 0) })), options: { baseValue: { type: "price", price: 0 }, title: "cum. R" } }];
    }
    return [
      { type: "area" as const, data: d.map((x, i) => ({ time: dayTime(x), value: conv(s.equity[i]) })), options: { lineColor: C.accent, topColor: "rgba(232,163,61,0.22)", bottomColor: "rgba(232,163,61,0.0)", title: "strategy" } },
      { type: "line" as const, data: d.map((x, i) => ({ time: dayTime(x), value: s.benchmark[i] === null ? undefined : conv(s.benchmark[i]) })).filter((p) => p.value !== undefined), options: { color: "#4c8dff", lineWidth: 1, title: `buy & hold ${res.meta.benchmark}`, priceScaleId: "left" } },
      { type: "line" as const, data: d.map((x, i) => ({ time: dayTime(x), value: conv(s.cash[i]) })), options: { color: "#535d6e", lineWidth: 1, lineStyle: 2, title: "cash" } },
    ];
  }, [s, mode, trades, init, res.meta.benchmark]);
  const dd = useMemo(() => [{ type: "area" as const, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.drawdown[i] * 100 })), options: { lineColor: C.down, topColor: "rgba(239,83,80,0.05)", bottomColor: "rgba(239,83,80,0.35)", invertFilledArea: true, lineWidth: 1 } }], [s]);
  const rs = useMemo(() => [{ type: "line" as const, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.rolling_sharpe[i] })).filter((p: any) => p.value !== null), options: { color: C.info, lineWidth: 1 }, priceLines: [{ price: 0, color: "#535d6e" }] }], [s]);
  const bp = useMemo(() => [{ type: "histogram" as const, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.bp_used[i], color: "rgba(76,141,255,0.55)" })) }], [s]);
  return (
    <div className="grid grid-cols-3 gap-px bg-line">
      <Panel className="col-span-3 border-0" title={mode === "risk" ? "Cumulative R (P&L ÷ buying power at risk, by exit date)" : `Equity vs benchmarks ${mode === "pct" ? "(% return)" : "($)"}`} right={<span className="text-2xs text-muted">orange strategy & grey cash @ {pct(res.config.capital.cash_benchmark_rate, 1)} (right axis) · blue buy & hold (left axis)</span>}>
        <LWChart series={series} height={300} extra={{ leftPriceScale: { visible: mode !== "risk", borderColor: "#1f2631" } }} />
      </Panel>
      <Panel className="border-0" title="Underwater (drawdown %)"><LWChart series={dd} height={170} /></Panel>
      <Panel className="border-0" title="Rolling Sharpe (63 sessions)"><LWChart series={rs} height={170} /></Panel>
      <Panel className="border-0" title="Buying power in use ($)"><LWChart series={bp} height={170} /></Panel>
    </div>
  );
}

function Distribution({ res, trades, mode }: { res: any; trades?: Trade[]; mode: string }) {
  const dist = res.distributions[mode];
  const hist = useMemo(() => ({
    grid: { left: 50, right: 12, top: 24, bottom: 34 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: dist.edges.slice(0, -1).map((e: number, i: number) => ((e + dist.edges[i + 1]) / 2)), axisLabel: { formatter: (v: string) => (mode === "dollars" ? fmtMoneyAxis(Number(v)) : mode === "pct" ? `${Number(v).toFixed(0)}%` : Number(v).toFixed(2)) } },
    yAxis: { type: "value", name: "trades" },
    series: [{ type: "bar", barCategoryGap: "8%", data: dist.counts.map((c: number, i: number) => ({ value: c, itemStyle: { color: (dist.edges[i] + dist.edges[i + 1]) / 2 >= 0 ? C.up : C.down } })) }],
  }), [dist, mode]);
  const heat = useMemo(() => {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const years = Array.from(new Set(res.series.monthly.map((m: any) => m.month.slice(0, 4)))) as string[];
    const data = res.series.monthly.map((m: any) => [Number(m.month.slice(5)) - 1, years.indexOf(m.month.slice(0, 4)), m.return === null ? null : +(m.return * 100).toFixed(2)]);
    const yearly = years.map((y) => res.series.monthly.filter((m: any) => m.month.startsWith(y)).reduce((a: number, m: any) => a * (1 + (m.return || 0)), 1) - 1);
    const mx = Math.max(0.5, ...data.map((d: any) => Math.abs(d[2] || 0)));
    return {
      grid: { left: 44, right: 64, top: 10, bottom: 26 },
      tooltip: { formatter: (p: any) => `${months[p.value[0]]} ${years[p.value[1]]}: ${p.value[2]}%` },
      xAxis: { type: "category", data: months, splitArea: { show: false } },
      yAxis: [{ type: "category", data: years }, { type: "category", data: yearly.map((y) => `${(y * 100).toFixed(1)}%`), position: "right", axisLabel: { color: C.fg } }],
      visualMap: { min: -mx, max: mx, show: false, inRange: { color: ["#ef5350", "#3a1f22", "#141922", "#16332f", "#26a69a"] } },
      series: [{ type: "heatmap", data, label: { show: true, formatter: (p: any) => (p.value[2] === null ? "" : p.value[2].toFixed(1)), fontSize: 10, color: "#d6dce5", fontFamily: "JetBrains Mono" }, itemStyle: { borderColor: "#090b0f", borderWidth: 2 } }],
    };
  }, [res]);
  const mfe = useMemo(() => trades && ({
    grid: { left: 56, right: 14, top: 24, bottom: 34 },
    tooltip: { formatter: (p: any) => `#${p.data[2]} MAE ${money(p.data[0], 0)} / MFE ${money(p.data[1], 0)} → P&L ${money(p.data[3], 0)}` },
    xAxis: { type: "value", name: "MAE ($)", nameLocation: "middle", nameGap: 22, axisLabel: { formatter: fmtMoneyAxis } },
    yAxis: { type: "value", name: "MFE ($)", axisLabel: { formatter: fmtMoneyAxis } },
    series: [{ type: "scatter", symbolSize: 4, data: trades.map((t) => [t.mae, t.mfe, t.id, t.pnl]), itemStyle: { color: (p: any) => (p.data[3] > 0 ? C.up : C.down), opacity: 0.75 } }],
  }), [trades]);
  return (
    <div className="grid grid-cols-2 gap-px bg-line">
      <Panel className="border-0" title={`Trade return distribution (${mode === "dollars" ? "$" : mode === "pct" ? "% of premium" : "R multiple"})`} right={<span className="num text-2xs text-muted">mean {num(dist.mean)} · median {num(dist.median)} · p5 {num(dist.p05)} · p95 {num(dist.p95)}</span>}>
        <EChart option={hist as any} height={260} />
      </Panel>
      <Panel className="border-0" title="Maximum favourable vs adverse excursion">{mfe ? <EChart option={mfe as any} height={260} /> : <Loading />}</Panel>
      <Panel className="col-span-2 border-0" title="Monthly returns (%)"><EChart option={heat as any} height={Math.max(120, 40 + 28 * new Set(res.series.monthly.map((m: any) => m.month.slice(0, 4))).size)} /></Panel>
    </div>
  );
}

const GROUP_TITLES: Record<string, string> = { time_of_day: "By entry time (30m)", weekday: "By day of week", dte: "By DTE at entry", delta: "By short-leg |delta|", vix: "By VIX at entry", exit_reason: "By exit reason", regime: "By trend / vol regime", strategy: "By strategy" };

function Breakdown({ res, mode }: { res: any; mode: string }) {
  const b = res.breakdowns;
  if (!b || !Object.keys(b).length) return <Empty title="No trades to break down" />;
  return (
    <div className="grid grid-cols-2 gap-px bg-line">
      {Object.keys(GROUP_TITLES).filter((g) => b[g] && (g !== "strategy" || b[g].length > 1)).map((g) => {
        const rows = b[g];
        const val = (r: any) => (mode === "risk" ? r.avg_ror : mode === "pct" ? r.win_rate * 100 : r.avg_pnl);
        const opt = {
          grid: { left: 52, right: 44, top: 22, bottom: 28 },
          legend: { data: [mode === "risk" ? "avg R" : mode === "pct" ? "win rate %" : "avg P&L", "trades"] },
          tooltip: { trigger: "axis" },
          xAxis: { type: "category", data: rows.map((r: any) => (g === "exit_reason" ? REASON_LABEL[r.key] || r.key : r.key)) },
          yAxis: [{ type: "value", axisLabel: { formatter: mode === "dollars" ? fmtMoneyAxis : undefined } }, { type: "value", splitLine: { show: false } }],
          series: [
            { name: mode === "risk" ? "avg R" : mode === "pct" ? "win rate %" : "avg P&L", type: "bar", barMaxWidth: 26, data: rows.map((r: any) => ({ value: val(r), itemStyle: { color: mode === "pct" ? C.info : (val(r) || 0) >= 0 ? C.up : C.down } })) },
            { name: "trades", type: "line", yAxisIndex: 1, symbolSize: 4, lineStyle: { width: 1, color: C.faint }, itemStyle: { color: C.muted }, data: rows.map((r: any) => r.trades) },
          ],
        };
        return <Panel key={g} className="border-0" title={GROUP_TITLES[g]} right={<span className="text-2xs text-faint">total {money(rows.reduce((a: number, r: any) => a + r.pnl, 0), 0, true)}</span>}><EChart option={opt as any} height={200} /></Panel>;
      })}
    </div>
  );
}

function GreeksTab({ res }: { res: any }) {
  const s = res.series;
  const mk = (k: string, color: string, pane: number) => ({ type: "line" as const, pane, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.greeks[k][i] })), options: { color, lineWidth: 1, title: k } });
  const g = useMemo(() => [mk("delta", C.accent, 0), mk("gamma", C.info, 1), mk("theta", C.up, 2), mk("vega", "#c678dd", 3)], [s]); // eslint-disable-line
  const vix = useMemo(() => [{ type: "line" as const, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.vix[i] })).filter((p: any) => p.value !== null), options: { color: C.synth, lineWidth: 1 } },
    { type: "histogram" as const, pane: 1, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.open_positions[i], color: "rgba(122,132,150,0.5)" })) }], [s]);
  return (
    <div className="grid grid-cols-3 gap-px bg-line">
      <Panel className="col-span-2 border-0" title="Portfolio Greeks at each close (Δ shares · Γ per $ · Θ $/day · ν $/vol pt)"><LWChart series={g} height={420} /></Panel>
      <Panel className="border-0" title="Vol index & open positions"><LWChart series={vix} height={420} /></Panel>
    </div>
  );
}

function TradesTab({ runId, trades }: { runId: string; trades?: Trade[] }) {
  const nav = useNavigate();
  const [filter, setFilter] = useState("all");
  if (!trades) return <Loading />;
  const rows = trades.filter((t) => filter === "all" || (filter === "win" ? t.pnl > 0 : filter === "loss" ? t.pnl <= 0 : t.flags.length > 0));
  const cols: Col<Trade>[] = [
    { key: "id", label: "#", render: (t) => <span className="num text-faint">{t.id}</span>, sort: (t) => t.id },
    { key: "entry", label: "Entry (ET)", render: (t) => <span className="num">{etTime(t.entry_ts)}</span>, sort: (t) => t.entry_ts },
    { key: "exit", label: "Exit", render: (t) => <span className="num text-fg-2">{etTime(t.exit_ts)}</span>, sort: (t) => t.exit_ts },
    { key: "legs", label: "Structure", render: (t) => <span className="num text-fg-2">{t.legs.map((l) => l.kind === "stock" ? `${l.ratio > 0 ? "+" : ""}${l.ratio}sh` : `${l.ratio > 0 ? "+" : ""}${l.ratio}${l.strike}${l.right}`).join(" ")}</span> },
    { key: "dte", label: "DTE", align: "r", render: (t) => <span className="num">{t.entry_dte}</span>, sort: (t) => t.entry_dte },
    { key: "delta", label: "|Δ|", align: "r", render: (t) => <span className="num">{num(t.short_delta, 2)}</span>, sort: (t) => t.short_delta },
    { key: "units", label: "Qty", align: "r", render: (t) => <span className="num">{t.units}</span>, sort: (t) => t.units },
    { key: "prem", label: "Premium", align: "r", render: (t) => <span className={cn("num", t.entry_cash > 0 ? "text-up/80" : "text-fg-2")}>{money(t.entry_cash, 0, true)}</span>, sort: (t) => t.entry_cash },
    { key: "pnl", label: "P&L", align: "r", render: (t) => <Pnl v={t.pnl} fmt={(x) => money(x, 2, true)} />, sort: (t) => t.pnl },
    { key: "ror", label: "R", align: "r", render: (t) => <Pnl v={t.return_on_risk} fmt={(x) => num(x, 2, true)} />, sort: (t) => t.return_on_risk },
    { key: "mfe", label: "MFE / MAE", align: "r", render: (t) => <span className="num text-2xs"><span className="text-up">{money(t.mfe, 0)}</span> / <span className="text-down">{money(t.mae, 0)}</span></span>, sort: (t) => t.mae },
    { key: "hold", label: "Hold", align: "r", render: (t) => <span className="num text-fg-2">{minutesLabel(t.holding_minutes)}</span>, sort: (t) => t.holding_minutes },
    { key: "vix", label: "VIX", align: "r", render: (t) => <span className="num text-fg-2">{num(t.entry_vix, 1)}</span>, sort: (t) => t.entry_vix },
    { key: "exit_reason", label: "Exit", render: (t) => <span className={cn(["stop_loss", "trailing_stop", "underlying_stop", "margin_call", "strike_breach"].includes(t.exit_reason) ? "text-down" : t.exit_reason === "profit_target" ? "text-up" : "text-fg-2")}>{REASON_LABEL[t.exit_reason] || t.exit_reason}</span>, sort: (t) => t.exit_reason },
    { key: "flags", label: "Flags", render: (t) => <span className="text-2xs text-warn">{t.flags.join(" ")}</span> },
    { key: "go", label: "", render: (t) => <PlayCircle className="size-3.5 text-faint" /> },
  ];
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-8 items-center gap-2 border-b border-line px-2.5">
        <Segmented value={filter} onChange={setFilter} options={[{ value: "all", label: `All ${trades.length}` }, { value: "win", label: "Winners" }, { value: "loss", label: "Losers" }, { value: "flag", label: "Flagged" }]} />
        <span className="text-2xs text-faint">Click a trade to open the replay workspace</span>
      </div>
      <DataTable rows={rows} cols={cols} rowKey={(t) => t.id} onRowClick={(t) => nav(`/replay/${runId}/${t.id}`)} className="flex-1" height="calc(100vh - 290px)" />
    </div>
  );
}

function MonteCarlo({ runId, initial }: { runId: string; initial: number }) {
  const [method, setMethod] = useState<"trade" | "block">("block");
  const [nSims, setN] = useState(2000);
  const [block, setBlock] = useState(10);
  const [seed, setSeed] = useState(7);
  const mc = useMutation({ mutationFn: () => api.post<any>(`/api/backtests/${runId}/montecarlo`, { method, n_sims: nSims, block, seed }) });
  const d = mc.data;
  const fan = d && !d.error && {
    grid: { left: 60, right: 14, top: 24, bottom: 30 },
    tooltip: { trigger: "axis", valueFormatter: (v: number) => money(v, 0) },
    xAxis: { type: "category", data: d.fan.x, name: method === "trade" ? "trade #" : "session", nameLocation: "middle", nameGap: 20 },
    yAxis: { type: "value", scale: true, axisLabel: { formatter: fmtMoneyAxis } },
    series: [
      { type: "line", data: d.fan.p05, lineStyle: { opacity: 0 }, stack: "a", symbol: "none", name: "p05" },
      { type: "line", data: d.fan.p95.map((v: number, i: number) => v - d.fan.p05[i]), lineStyle: { opacity: 0 }, stack: "a", symbol: "none", areaStyle: { color: "rgba(76,141,255,0.12)" }, name: "p05–p95", tooltip: { show: false } },
      { type: "line", data: d.fan.p25, lineStyle: { opacity: 0 }, stack: "b", symbol: "none", name: "p25" },
      { type: "line", data: d.fan.p75.map((v: number, i: number) => v - d.fan.p25[i]), lineStyle: { opacity: 0 }, stack: "b", symbol: "none", areaStyle: { color: "rgba(76,141,255,0.22)" }, name: "p25–p75", tooltip: { show: false } },
      { type: "line", data: d.fan.p50, symbol: "none", lineStyle: { color: C.accent, width: 1.5 }, name: "median" },
      { type: "line", data: d.fan.x.map(() => initial), symbol: "none", lineStyle: { color: C.faint, type: "dashed", width: 1 }, name: "start" },
    ],
  };
  const hist = (h: any, title: string, fmt: (v: number) => string, color: string) => ({
    title: { text: title }, grid: { left: 40, right: 10, top: 28, bottom: 26 },
    xAxis: { type: "category", data: h.edges.slice(0, -1).map((e: number, i: number) => (e + h.edges[i + 1]) / 2), axisLabel: { formatter: (v: string) => fmt(Number(v)) } },
    yAxis: { type: "value" }, series: [{ type: "bar", data: h.counts, itemStyle: { color }, barCategoryGap: "6%" }],
  });
  return (
    <div className="flex flex-col">
      <div className="flex items-end gap-2 border-b border-line p-2.5">
        <Field label="Method" className="w-56"><Select value={method} onChange={setMethod} options={[{ value: "block", label: "Block bootstrap (daily returns)" }, { value: "trade", label: "Trade bootstrap (i.i.d.)" }]} /></Field>
        <Field label="Simulations" className="w-24"><NumInput value={nSims} step={500} onChange={(v) => setN(v ?? 2000)} /></Field>
        {method === "block" && <Field label="Block length (sessions)" className="w-36" hint="Longer blocks preserve more serial dependence (volatility clustering, streaks)"><NumInput value={block} onChange={(v) => setBlock(v ?? 10)} /></Field>}
        <Field label="Seed" className="w-20"><NumInput value={seed} onChange={(v) => setSeed(v ?? 7)} /></Field>
        <Button variant="primary" onClick={() => mc.mutate()} disabled={mc.isPending}>{mc.isPending ? "Simulating…" : "Run simulation"}</Button>
        <span className="text-2xs text-faint ml-2 max-w-md">Resamples realised outcomes to show the range of paths this edge could have produced. It cannot reveal risks absent from the sample.</span>
      </div>
      {mc.error && <ErrorBox className="m-2" error={mc.error} />}
      {d?.error && <ErrorBox className="m-2" error={d.error} />}
      {!d && !mc.isPending && <Empty title="Run a simulation to see the distribution of outcomes" />}
      {d && !d.error && (
        <div className="grid grid-cols-3 gap-px bg-line">
          <Panel className="col-span-2 row-span-2 border-0" title={`Equity fan · ${d.n_sims} paths (p5–p95, p25–p75, median)`}><EChart option={fan as any} height={360} /></Panel>
          <Panel className="border-0" title="Outcome summary">
            <table className="tbl">
              <tbody>
                <tr><td className="text-muted">Final equity p5 / p50 / p95</td><td className="r num">{money(d.final_equity.p05, 0)} / {money(d.final_equity.p50, 0)} / {money(d.final_equity.p95, 0)}</td></tr>
                <tr><td className="text-muted">Max drawdown p50 / p5 (worst)</td><td className="r num text-down">{pct(d.max_drawdown.p50, 1)} / {pct(d.max_drawdown.p05, 1)}</td></tr>
                <tr><td className="text-muted">P(ending below start)</td><td className="r num">{pct(d.prob_loss, 1)}</td></tr>
                {d.prob_ruin !== undefined && <tr><td className="text-muted">P(equity ≤ {pct(1 - d.ruin_threshold, 0)} of start)</td><td className="r num">{pct(d.prob_ruin, 2)}</td></tr>}
                {d.sharpe && <tr><td className="text-muted">Sharpe p5 / p50 / p95</td><td className="r num">{num(d.sharpe.p05)} / {num(d.sharpe.p50)} / {num(d.sharpe.p95)}</td></tr>}
                {d.prob_sharpe_below_0 !== undefined && <tr><td className="text-muted">P(Sharpe &lt; 0)</td><td className="r num">{pct(d.prob_sharpe_below_0, 1)}</td></tr>}
              </tbody>
            </table>
          </Panel>
          <Panel className="border-0" title="Max drawdown distribution"><EChart option={hist(d.dd_hist, "", (v) => `${(v * 100).toFixed(0)}%`, C.down) as any} height={180} /></Panel>
        </div>
      )}
    </div>
  );
}

function Diagnostics({ res }: { res: any }) {
  const d = res.diagnostics;
  const meta = res.meta;
  const skipped = Object.entries(d.skipped_entries || {}).sort((a: any, b: any) => b[1] - a[1]);
  const counts: [string, number, string][] = [
    ["Entry signals", d.signals, "condition trees that evaluated true"], ["Entries filled", d.entries, ""], ["Deferred exits", d.deferred_exits, "exit wanted but no fresh quote / size; retried next snapshot"],
    ["Partial fills", d.partial_fills, "limited by displayed size"], ["Stale marks", d.stale_marks, "quote older than max age"], ["Missing marks", d.missing_marks, "contract absent from a snapshot; last mark kept"],
    ["Ambiguous exits", d.ambiguous_exits, "loss & profit rules fired together; loss recorded"], ["Assignments", d.assignments, ""], ["Exercises", d.exercises, ""], ["Margin calls", d.margin_calls, ""],
  ];
  return (
    <div className="grid grid-cols-3 gap-px bg-line">
      <Panel className="border-0" title="Engine counters">
        <table className="tbl"><tbody>{counts.map(([l, v, h]) => <tr key={l}><td><Tip content={h}><span className="cursor-help">{l}</span></Tip></td><td className="r num">{v}</td></tr>)}</tbody></table>
      </Panel>
      <Panel className="border-0" title="Why signals did not become trades">
        {skipped.length === 0 ? <Empty title="Every signal was filled" /> : (
          <table className="tbl"><tbody>{skipped.map(([k, v]) => <tr key={k}><td className="whitespace-normal text-fg-2">{k}</td><td className="r num">{v as number}</td></tr>)}</tbody></table>
        )}
      </Panel>
      <Panel className="border-0" title="Data quality">
        <table className="tbl"><tbody>
          <tr><td>Synthetic data</td><td className="r">{meta.synthetic ? <SyntheticBadge on /> : "no"}</td></tr>
          <tr><td>Missing option sessions</td><td className="r num">{meta.data_quality.missing_sessions}</td></tr>
          {Object.entries(d.missing_sessions || {}).map(([s, days]: any) => <tr key={s}><td className="text-muted pl-4">{s}</td><td className="r num text-2xs text-fg-2 whitespace-normal">{days.slice(0, 8).join(", ")}{days.length > 8 ? ` +${days.length - 8}` : ""}</td></tr>)}
          <tr><td>Greeks computed from quotes</td><td className="r">{meta.data_quality.computed_greeks ? "yes" : "provider-supplied"}</td></tr>
        </tbody></table>
        <div className="p-2"><Warnings items={(d.warnings || []).map((t: string) => ({ level: "warn", text: t }))} /></div>
      </Panel>
      <Panel className="border-0" title="Execution assumptions">
        <table className="tbl"><tbody>{Object.entries(meta.execution).map(([k, v]) => <tr key={k}><td className="text-muted">{k}</td><td className="r num">{String(v)}</td></tr>)}</tbody></table>
      </Panel>
      <Panel className="border-0" title="Capital & margin">
        <table className="tbl"><tbody>{Object.entries(meta.capital).map(([k, v]) => <tr key={k}><td className="text-muted">{k}</td><td className="r num">{String(v)}</td></tr>)}</tbody></table>
      </Panel>
      <Panel className="border-0" title="Reproducibility">
        <table className="tbl"><tbody>
          <tr><td className="text-muted">Config hash</td><td className="r font-mono">{meta.config_hash}</td></tr>
          {Object.entries(meta.dataset_versions).map(([k, v]) => <tr key={k}><td className="text-muted">{k} dataset version</td><td className="r font-mono">{v as string}</td></tr>)}
          <tr><td className="text-muted">Events version</td><td className="r font-mono">{meta.events_version}</td></tr>
          <tr><td className="text-muted">Engine version</td><td className="r font-mono">{meta.engine_version}</td></tr>
          <tr><td className="text-muted">Runtime</td><td className="r num">{meta.runtime_seconds}s</td></tr>
        </tbody></table>
        <div className="p-2 text-2xs text-faint">Re-running the same configuration against the same dataset and engine version reproduces these results exactly (the engine has no randomness).</div>
      </Panel>
    </div>
  );
}
