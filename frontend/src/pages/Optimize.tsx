import { useMutation, useQuery } from "@tanstack/react-query";
import { Blocks, Plus, Sigma, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { cancelJob, DataTable, Empty, ErrorBox, JobBar, Loading, PageHeader, Panel, Stat, StatusDot, SyntheticBadge, Warnings, type Col } from "@/components/common";
import { Button, Field, NumInput, Segmented, Select, Switch, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui";
import { api, queryClient, useJob, useMeta, useRun } from "@/lib/api";
import { ago, cn, money, num, pct } from "@/lib/format";
import { useBuilder } from "@/lib/store";
import type { RunRow } from "@/lib/types";

export default function Optimize() {
  const { id } = useParams();
  const nav = useNavigate();
  const list = useQuery({ queryKey: ["runs", "opt"], queryFn: () => api.get<RunRow[]>("/api/optimizations"), refetchInterval: 4000 });
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Optimization Lab" sub={<span className="text-2xs text-muted">parameter research with overfitting controls</span>}>
        <Button variant={id ? "default" : "primary"} onClick={() => nav("/optimize")}><Plus /> New study</Button>
      </PageHeader>
      <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr]">
        <div className="min-h-0 overflow-auto border-r border-line bg-bg-1">
          <div className="label px-2.5 pt-2 pb-1">Studies</div>
          {(list.data || []).map((r) => (
            <button key={r.id} onClick={() => nav(`/optimize/${r.id}`)} className={cn("flex w-full flex-col gap-0.5 border-l-2 px-2.5 py-1.5 text-left hover:bg-bg-2", id === r.id ? "border-accent bg-bg-3" : "border-transparent")}>
              <span className="flex items-center gap-1.5 text-xs"><StatusDot status={r.status} /><span className="truncate">{r.name}</span></span>
              <span className="num text-2xs text-muted">
                {r.status === "running" ? `${Math.round(r.progress * 100)}% · ${r.message}` : r.status === "done" ? `${r.summary?.n_configs} cfgs · best ${num(r.summary?.best_objective)} · DSR ${num(r.summary?.dsr)}` : r.status}
              </span>
              <span className="text-2xs text-faint">{ago(r.created_at)}</span>
            </button>
          ))}
          {!list.data?.length && <div className="px-2.5 text-2xs text-faint">No studies yet.</div>}
        </div>
        <div className="min-h-0 min-w-0 overflow-auto">{id ? <Study id={id} /> : <NewStudy />}</div>
      </div>
    </div>
  );
}

type PSel = { path: string; label: string; kind: string; current: any; on: boolean; min: number; max: number; step: number };

function toMin(t: string) { const [h, m] = String(t).split(":").map(Number); return h * 60 + (m || 0); }
const NICE = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500];
const nice = (x: number) => NICE.find((n) => n >= x - 1e-9) ?? x;

function NewStudy() {
  const nav = useNavigate();
  const [sp] = useSearchParams();
  const meta = useMeta().data;
  const b = useBuilder();
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies") });
  const [source, setSource] = useState<string>("builder");
  const strat = source === "builder" ? b.strategy : strategies.data?.find((s) => s.id === source)?.config;
  const opt = useQuery({ queryKey: ["optimizable", JSON.stringify(strat)], queryFn: () => api.post<any[]>("/api/strategies/optimizable", strat), enabled: !!strat });
  const [sel, setSel] = useState<PSel[]>([]);
  useEffect(() => {
    if (!opt.data) return;
    setSel(opt.data.map((p: any) => {
      const c = p.kind === "time" ? toMin((p.current || "10:00").split(",")[0]) : Number(p.current ?? 0);
      const isDelta = p.label.includes("delta");
      const step = p.kind === "time" ? 15 : p.kind === "int" ? 1 : isDelta ? 0.05 : nice(Math.max(Math.abs(c) / 4, 0.5));
      const span = p.kind === "time" ? 60 : p.kind === "int" ? 2 : isDelta ? 0.1 : step * 2;
      let lo = +(c - span).toFixed(4), hi = +(c + span).toFixed(4);
      if (p.kind === "time") { lo = Math.max(570, lo); hi = Math.min(960, hi); }
      if (isDelta) { lo = Math.max(0.05, lo); hi = Math.min(0.95, hi); }
      if (p.kind === "int" && p.label.includes("DTE")) lo = Math.max(0, lo);
      if (/target|stop/i.test(p.label)) lo = Math.max(step, lo);
      return { ...p, on: false, min: lo, max: hi, step };
    }));
  }, [opt.data]);
  const [method, setMethod] = useState<"grid" | "random" | "bayes">("grid");
  const [objective, setObjective] = useState("sharpe");
  const [maxEvals, setMaxEvals] = useState(120);
  const [minTrades, setMinTrades] = useState(30);
  const [validation, setValidation] = useState<"none" | "holdout" | "walk_forward">("holdout");
  const [holdout, setHoldout] = useState(30);
  const [folds, setFolds] = useState(4);
  const [anchored, setAnchored] = useState(false);
  const [seed, setSeed] = useState(42);
  const active = sel.filter((p) => p.on);
  const gridOf = (p: PSel) => Math.max(1, Math.floor((p.max - p.min) / (p.step || 1) + 1e-9) + 1);
  const total = active.reduce((a, p) => a * gridOf(p), 1);
  const evals = Math.min(total, maxEvals);
  const perEval = validation === "walk_forward" ? folds + 1 : 1;
  const submit = useMutation({
    mutationFn: () => {
      const base = { ...b.backtestConfig(), strategies: [strat] };
      const params = active.map((p) => p.kind === "time"
        ? { path: p.path, label: p.label, kind: "time", min: p.min, max: p.max, step: p.step }
        : { path: p.path, label: p.label, kind: p.kind === "int" ? "int" : "float", min: p.min, max: p.max, step: p.step });
      return api.post<{ id: string }>("/api/optimizations", {
        config: { base, params, method, objective, max_evals: maxEvals, seed, min_trades: minTrades, validation, holdout_pct: holdout, wf_folds: folds, wf_anchored: anchored },
        strategy_id: source === "builder" ? b.strategyId : source,
      });
    },
    onSuccess: (r) => { queryClient.invalidateQueries({ queryKey: ["runs", "opt"] }); nav(`/optimize/${r.id}`); },
    onError: (e: any) => toast.error(e.message),
  });
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_320px] gap-px bg-line min-h-full">
      <div className="bg-bg-0 p-3 space-y-3 min-w-0">
        <div className="flex items-end gap-2">
          <Field label="Strategy" className="w-80">
            <Select value={source} onChange={setSource} options={[{ value: "builder", label: `Builder draft: ${b.strategy.name}` }, ...(strategies.data || []).map((s) => ({ value: s.id, label: `${s.name}${s.trials ? ` · ${s.trials} prior trials` : ""}` }))]} />
          </Field>
          <span className="text-2xs text-muted pb-1.5">{b.start} → {b.end} · ${(b.capital.initial_capital / 1000).toFixed(0)}k · {b.execution.fill_model} fills (from builder settings)</span>
        </div>
        <Panel title="Parameters to sweep" right={<span className="text-2xs text-muted">{active.length} selected · grid {total.toLocaleString()} combinations</span>}>
          {opt.isLoading ? <Loading /> : (
            <div className="overflow-x-auto"><table className="tbl">
              <thead><tr><th /><th>Parameter</th><th>Path</th><th className="r">Current</th><th className="r">From</th><th className="r">To</th><th className="r">Step</th><th className="r">Values</th></tr></thead>
              <tbody>
                {sel.map((p, i) => (
                  <tr key={p.path} className={cn(!p.on && "opacity-60")}>
                    <td><Switch checked={p.on} onChange={(v) => setSel((s) => s.map((x, j) => (j === i ? { ...x, on: v } : x)))} /></td>
                    <td className="text-fg">{p.label}</td>
                    <td className="font-mono text-2xs text-faint max-w-56 truncate" title={p.path}>{p.path.replace("strategies.0.", "")}</td>
                    <td className="r num">{String(p.current)}</td>
                    {(["min", "max", "step"] as const).map((k) => (
                      <td key={k} className="r w-24">
                        {p.kind === "time" && k !== "step" ? (
                          <input className="field field-sm num w-20" value={`${String(Math.floor(p[k] / 60)).padStart(2, "0")}:${String(p[k] % 60).padStart(2, "0")}`} onChange={(e) => setSel((s) => s.map((x, j) => (j === i ? { ...x, [k]: toMin(e.target.value) } : x)))} />
                        ) : (
                          <NumInput sz="sm" className="w-20" value={p[k]} step={p.step} onChange={(v) => setSel((s) => s.map((x, j) => (j === i ? { ...x, [k]: v ?? 0 } : x)))} />
                        )}
                      </td>
                    ))}
                    <td className="r num text-muted">{gridOf(p)}</td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}
        </Panel>
        <div className="text-2xs text-faint leading-relaxed max-w-3xl">
          Every configuration is a full, independent backtest in a separate worker process. Rankings use in-sample data only; the study reports the
          deflated Sharpe ratio (adjusted for the number of configurations tried, including earlier studies of the same saved strategy), the probability
          of backtest overfitting via combinatorially-symmetric cross-validation, neighbourhood stability, and out-of-sample degradation.
        </div>
      </div>
      <div className="bg-bg-1 p-3 space-y-3">
        <Field label="Search method">
          <Segmented value={method} onChange={setMethod} options={[{ value: "grid", label: "Grid" }, { value: "random", label: "Random" }, { value: "bayes", label: "Bayesian (GP-EI)" }]} />
        </Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Objective"><Select value={objective} onChange={setObjective} options={(meta?.objectives || ["sharpe"]).map((o) => ({ value: o, label: o.replace(/_/g, " ") }))} /></Field>
          <Field label="Max evaluations"><NumInput value={maxEvals} step={10} min={1} onChange={(v) => setMaxEvals(v ?? 100)} /></Field>
          <Field label="Min trades" hint="Configurations with fewer trades are not eligible as 'best'"><NumInput value={minTrades} onChange={(v) => setMinTrades(v ?? 30)} /></Field>
          <Field label="Seed"><NumInput value={seed} onChange={(v) => setSeed(v ?? 42)} /></Field>
        </div>
        <Field label="Validation">
          <Segmented value={validation} onChange={setValidation} options={[{ value: "holdout", label: "IS / OOS holdout" }, { value: "walk_forward", label: "Walk-forward" }, { value: "none", label: "None" }]} />
        </Field>
        {validation === "holdout" && <Field label="Out-of-sample % (most recent)"><NumInput value={holdout} min={5} max={80} onChange={(v) => setHoldout(v ?? 30)} suffix="%" /></Field>}
        {validation === "walk_forward" && (
          <div className="grid grid-cols-2 gap-2">
            <Field label="Folds"><NumInput value={folds} min={2} max={12} onChange={(v) => setFolds(v ?? 4)} /></Field>
            <label className="flex items-center gap-2 self-end pb-1.5 text-xs text-fg-2"><Switch checked={anchored} onChange={setAnchored} /> Anchored window</label>
          </div>
        )}
        <div className="rounded-sm border border-line-2 bg-bg-2 p-2 text-xs space-y-1">
          <div className="flex justify-between"><span className="text-muted">Backtests to run</span><span className="num">{(evals * perEval + (validation === "holdout" ? Math.min(10, evals) : 0)).toLocaleString()}</span></div>
          {total > maxEvals && method === "grid" && <div className="text-warn text-2xs">Grid exceeds the budget; a seeded random subset of {maxEvals} grid points will be evaluated.</div>}
          {active.length === 0 && <div className="text-2xs text-faint">Select at least one parameter.</div>}
        </div>
        <Button variant="primary" className="w-full" disabled={!active.length || submit.isPending || !strat} onClick={() => submit.mutate()}><Sigma /> Launch study</Button>
      </div>
    </div>
  );
}

function Study({ id }: { id: string }) {
  const run = useRun(id, true);
  const job = useJob(run.data && ["queued", "running"].includes(run.data.status) ? id : null);
  if (run.isLoading) return <Loading />;
  const r = run.data!;
  if (["queued", "running"].includes(r.status)) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3">
        <div className="text-sm text-fg-2">{r.name}</div>
        <JobBar job={job || { id, status: r.status, progress: r.progress, message: r.message }} onCancel={() => cancelJob(id)} />
        <div className="text-2xs text-faint">Configurations run in parallel worker processes</div>
      </div>
    );
  }
  if (r.status !== "done") return <ErrorBox className="m-3" error={`${r.status}: ${r.message}\n${r.error || ""}`} />;
  return <StudyResult run={r} />;
}

function StudyResult({ run }: { run: RunRow }) {
  const nav = useNavigate();
  const o = run.result;
  const params: any[] = o.params;
  const configs: any[] = o.configs;
  const best = o.best_index !== null ? configs[o.best_index] : null;
  const pareto = new Set<number>(o.pareto || []);
  const [xp, setXp] = useState(params[0]?.path);
  const [yp, setYp] = useState(params[1]?.path || params[0]?.path);
  const [metric, setMetric] = useState<string>(o.objective);
  const apply = useMutation({
    mutationFn: (index: number) => api.post<any>(`/api/optimizations/${run.id}/apply`, { index }),
    onSuccess: (cfg) => {
      useBuilder.getState().load(cfg.strategies[0], null);
      useBuilder.getState().patch({ start: cfg.start, end: cfg.end, capital: cfg.capital, execution: cfg.execution, dirty: true });
      toast.success("Configuration loaded into the builder");
      nav("/builder");
    },
  });
  const short = (p: string) => params.find((x) => x.path === p)?.label || p.split(".").slice(-2).join(".");
  const mval = (c: any) => (metric === o.objective ? c.objective : c.metrics?.[metric]);
  const cols: Col<any>[] = [
    { key: "i", label: "#", render: (c) => <span className="num text-faint">{c.i}{c.i === o.best_index && <span className="ml-1 text-accent">★</span>}{pareto.has(c.i) && <span className="ml-1 text-info" title="Pareto-efficient (CAGR vs drawdown)">◆</span>}</span>, sort: (c) => c.i },
    ...params.map((p) => ({ key: p.path, label: short(p.path), render: (c: any) => <span className="num">{String(c.params[p.path])}</span>, sort: (c: any) => c.params[p.path] })),
    { key: "obj", label: o.objective, align: "r" as const, render: (c: any) => <span className="num text-fg">{num(c.objective)}</span>, sort: (c: any) => c.objective },
    { key: "stab", label: <Tip content="Mean objective of grid neighbours ÷ own objective (1 = plateau)"><span>Stability</span></Tip>, align: "r" as const, render: (c: any) => <span className={cn("num", c.stability !== null && c.stability < 0.6 ? "text-warn" : "")}>{num(c.stability)}</span>, sort: (c: any) => c.stability },
    { key: "trades", label: "Trades", align: "r" as const, render: (c: any) => <span className={cn("num", (c.metrics?.trades ?? 0) < 30 && "text-warn")}>{c.metrics?.trades ?? "—"}</span>, sort: (c: any) => c.metrics?.trades },
    { key: "pnl", label: "Net P&L", align: "r" as const, render: (c: any) => <span className={cn("num", (c.metrics?.net_pnl ?? 0) >= 0 ? "text-up" : "text-down")}>{money(c.metrics?.net_pnl, 0, true)}</span>, sort: (c: any) => c.metrics?.net_pnl },
    { key: "sharpe", label: "Sharpe", align: "r" as const, render: (c: any) => <span className="num">{num(c.metrics?.sharpe)}</span>, sort: (c: any) => c.metrics?.sharpe },
    { key: "dd", label: "Max DD", align: "r" as const, render: (c: any) => <span className="num text-down">{pct(c.metrics?.max_drawdown, 1)}</span>, sort: (c: any) => c.metrics?.max_drawdown },
    { key: "wr", label: "Win %", align: "r" as const, render: (c: any) => <span className="num">{pct(c.metrics?.win_rate, 0)}</span>, sort: (c: any) => c.metrics?.win_rate },
    { key: "util", label: "Max BP %", align: "r" as const, render: (c: any) => <span className={cn("num", (c.metrics?.max_capital_utilization ?? 0) > 0.8 && "text-warn")}>{pct(c.metrics?.max_capital_utilization, 0)}</span>, sort: (c: any) => c.metrics?.max_capital_utilization },
    { key: "go", label: "", render: (c: any) => <Button size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); apply.mutate(c.i); }}><Blocks /> Builder</Button> },
  ];
  const heat = useMemo(() => {
    const px = params.find((p) => p.path === xp), py = params.find((p) => p.path === yp);
    if (!px || !py) return null;
    const xs = px.grid, ys = py.grid;
    const cell = new Map<string, number>();
    for (const c of configs) {
      const v = mval(c);
      if (v === null || v === undefined) continue;
      const k = `${c.params[xp]}|${c.params[yp]}`;
      cell.set(k, Math.max(cell.get(k) ?? -Infinity, v));
    }
    const data = [];
    for (let a = 0; a < xs.length; a++) for (let bb = 0; bb < ys.length; bb++) {
      const v = cell.get(`${xs[a]}|${ys[bb]}`);
      data.push([a, bb, v === undefined ? "-" : +v.toFixed(3)]);
    }
    const vals = data.map((d) => d[2]).filter((v) => v !== "-") as number[];
    return {
      grid: { left: 64, right: 70, top: 14, bottom: 40 },
      tooltip: { formatter: (p: any) => `${short(xp)}=${xs[p.value[0]]}, ${short(yp)}=${ys[p.value[1]]}: ${p.value[2]}` },
      xAxis: { type: "category", data: xs.map(String), name: short(xp), nameLocation: "middle", nameGap: 26 },
      yAxis: { type: "category", data: ys.map(String), name: short(yp) },
      visualMap: { min: Math.min(...vals), max: Math.max(...vals), calculable: true, orient: "vertical", right: 4, top: "middle", itemHeight: 120, textStyle: { color: C.muted, fontSize: 10 }, inRange: { color: ["#3a1f22", "#ef5350", "#e8a33d", "#26a69a", "#9fe3d9"] } },
      series: [{ type: "heatmap", data, label: { show: xs.length * ys.length <= 120, fontSize: 9, color: "#090b0f", fontFamily: "JetBrains Mono" }, itemStyle: { borderColor: "#090b0f", borderWidth: 1 } }],
    };
  }, [xp, yp, metric, configs, params]); // eslint-disable-line
  const scatter = useMemo(() => {
    const pts = configs.filter((c) => c.metrics).map((c) => [Math.abs(c.metrics.max_drawdown ?? 0) * 100, (c.metrics.cagr ?? 0) * 100, c.i, c.metrics.sharpe ?? 0]);
    const front = (o.pareto || []).map((i: number) => configs[i]).filter((c: any) => c?.metrics).map((c: any) => [Math.abs(c.metrics.max_drawdown) * 100, c.metrics.cagr * 100]).sort((a: any, b: any) => a[0] - b[0]);
    return {
      grid: { left: 52, right: 16, top: 22, bottom: 36 },
      tooltip: { formatter: (p: any) => (p.seriesIndex === 0 ? `#${p.data[2]} · DD ${p.data[0].toFixed(1)}% · CAGR ${p.data[1].toFixed(2)}% · Sharpe ${p.data[3].toFixed(2)}` : "Pareto frontier") },
      xAxis: { type: "value", name: "max drawdown %", nameLocation: "middle", nameGap: 22 },
      yAxis: { type: "value", name: "CAGR %" },
      visualMap: { show: false, dimension: 3, min: Math.min(...pts.map((p) => p[3])), max: Math.max(...pts.map((p) => p[3])), inRange: { color: ["#ef5350", "#e8a33d", "#26a69a"] } },
      series: [{ type: "scatter", symbolSize: 6, data: pts }, { type: "line", data: front, lineStyle: { color: C.info, width: 1.2 }, symbol: "diamond", symbolSize: 7, itemStyle: { color: C.info } }],
    };
  }, [configs, o.pareto]);
  const ds = o.deflated_sharpe || {};
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 border-b border-line bg-bg-1 px-3 h-9">
        <span className="text-sm font-medium">{run.name}</span>
        <SyntheticBadge on={o.synthetic} />
        <span className="num text-2xs text-muted">{o.n_configs} configs · {o.method} · {o.validation} · {o.runtime_seconds}s</span>
      </div>
      <div className="grid grid-cols-8 border-b border-line bg-bg-1 [&>*]:border-r [&>*]:border-line">
        <Stat label="Best objective" value={num(best?.objective)} sub={o.objective} />
        <Stat label="Deflated Sharpe" value={num(ds.dsr)} tone={ds.dsr !== null && ds.dsr < 0.95 ? "text-warn" : "text-up"} sub={`E[max SR|${ds.n_trials} trials] ${num(ds.expected_max_sharpe_annual)}`} hint="Probability the best Sharpe beats the expected maximum of N unskilled trials (Bailey & López de Prado)" />
        <Stat label="PBO (CSCV)" value={pct(o.pbo?.pbo, 0)} tone={(o.pbo?.pbo ?? 0) > 0.4 ? "text-warn" : ""} sub={`${o.pbo?.combinations ?? 0} splits`} hint="Probability of backtest overfitting: how often the in-sample winner ranks below median out of sample" />
        <Stat label="OOS degradation" value={pct(o.oos_degradation, 0)} tone={(o.oos_degradation ?? 0) > 0.5 ? "text-warn" : ""} sub="best IS config" />
        <Stat label="IS→OOS rank corr." value={num(o.is_oos_rank_corr)} sub="top configs, Spearman" />
        <Stat label="WF efficiency" value={num(o.wf_efficiency)} sub="OOS / IS CAGR" />
        <Stat label="Total trials" value={o.total_trials} sub="incl. earlier studies" />
        <Stat label="Best trades" value={best?.metrics?.trades ?? "—"} tone={(best?.metrics?.trades ?? 0) < 30 ? "text-warn" : ""} sub={`stability ${num(best?.stability)}`} />
      </div>
      <div className="border-b border-line p-2.5"><Warnings items={o.warnings} /></div>
      <Tabs defaultValue="table">
        <TabsList>
          <TabsTrigger value="table">Configurations</TabsTrigger>
          <TabsTrigger value="heat">Heatmap & frontier</TabsTrigger>
          <TabsTrigger value="sens">Sensitivity</TabsTrigger>
          <TabsTrigger value="valid">Validation</TabsTrigger>
        </TabsList>
        <TabsContent value="table"><DataTable rows={configs} cols={cols} rowKey={(c) => c.i} defaultSort={{ key: "obj", dir: -1 }} height="calc(100vh - 300px)" selected={o.best_index} /></TabsContent>
        <TabsContent value="heat">
          <div className="grid grid-cols-2 gap-px bg-line">
            <Panel className="border-0" title="Parameter heatmap (max over other parameters)" right={<>
              <Select sz="sm" className="w-32" value={xp} onChange={setXp} options={params.map((p) => ({ value: p.path, label: short(p.path) }))} />
              <Select sz="sm" className="w-32" value={yp} onChange={setYp} options={params.map((p) => ({ value: p.path, label: short(p.path) }))} />
              <Select sz="sm" className="w-28" value={metric} onChange={setMetric} options={[o.objective, "sharpe", "net_pnl", "cagr", "max_drawdown", "win_rate", "trades"].filter((v, i, a) => a.indexOf(v) === i)} />
            </>}>{heat ? <EChart option={heat as any} height={360} /> : <Empty title="Select two parameters" />}</Panel>
            <Panel className="border-0" title="Risk vs return · Pareto frontier (◆)"><EChart option={scatter as any} height={360} /></Panel>
          </div>
        </TabsContent>
        <TabsContent value="sens">
          <div className="grid grid-cols-2 gap-px bg-line">
            {o.sensitivity.map((s: any) => (
              <Panel key={s.path} className="border-0" title={`${s.label}: objective by value`}>
                <EChart height={210} option={{
                  grid: { left: 48, right: 12, top: 24, bottom: 26 }, legend: { data: ["mean", "max", "others at best"] }, tooltip: { trigger: "axis" },
                  xAxis: { type: "category", data: s.values.map((v: any) => String(v.value)) }, yAxis: { type: "value", scale: true },
                  series: [
                    { name: "mean", type: "line", data: s.values.map((v: any) => v.mean), lineStyle: { color: C.info }, itemStyle: { color: C.info } },
                    { name: "max", type: "line", data: s.values.map((v: any) => v.max), lineStyle: { color: C.faint, type: "dashed" }, itemStyle: { color: C.faint } },
                    { name: "others at best", type: "line", data: s.values.map((v: any) => v.at_best), lineStyle: { color: C.accent }, itemStyle: { color: C.accent }, connectNulls: true },
                  ],
                } as any} />
              </Panel>
            ))}
          </div>
        </TabsContent>
        <TabsContent value="valid"><Validation o={o} /></TabsContent>
      </Tabs>
    </div>
  );
}

function Validation({ o }: { o: any }) {
  const oos = o.oos || [];
  const wf = o.walk_forward || [];
  return (
    <div className="grid grid-cols-2 gap-px bg-line">
      {oos.length > 0 && (
        <Panel className="border-0 col-span-2" title={`Hold-out: top in-sample configurations re-run on the unseen period · ${o.objective}`}>
          <EChart height={240} option={{
            grid: { left: 48, right: 16, top: 24, bottom: 28 }, legend: { data: ["in-sample", "out-of-sample"] }, tooltip: { trigger: "axis" },
            xAxis: { type: "category", data: oos.map((x: any) => `#${x.index}`) }, yAxis: { type: "value" },
            series: [{ name: "in-sample", type: "bar", data: oos.map((x: any) => x.is?.[o.objective]), itemStyle: { color: C.info } }, { name: "out-of-sample", type: "bar", data: oos.map((x: any) => x.oos?.[o.objective]), itemStyle: { color: C.accent } }],
          } as any} />
        </Panel>
      )}
      {wf.length > 0 && (
        <>
          <Panel className="border-0 min-w-0" title={`Walk-forward folds · efficiency ${num(o.wf_efficiency)}`}>
            <div className="overflow-x-auto"><table className="tbl">
              <thead><tr><th>Fold</th><th>Train</th><th>Test</th><th>Chosen params</th><th className="r">IS {o.objective}</th><th className="r">OOS {o.objective}</th><th className="r">OOS P&L</th></tr></thead>
              <tbody>{wf.map((f: any) => (
                <tr key={f.fold}><td className="num">{f.fold + 1}</td><td className="num text-2xs">{f.train[0]}→{f.train[1]}</td><td className="num text-2xs">{f.test[0]}→{f.test[1]}</td>
                  <td className="num text-2xs text-fg-2 max-w-60 truncate" title={Object.entries(f.best_params).map(([k, v]) => `${k}=${v}`).join("\n")}>{Object.entries(f.best_params).map(([k, v]) => `${o.params.find((p: any) => p.path === k)?.label || k.split(".").slice(-1)[0]}=${v}`).join(" · ")}</td>
                  <td className="r num">{num(f.train_metrics?.[o.objective])}</td><td className="r num">{num(f.test_metrics?.[o.objective])}</td><td className={cn("r num", (f.test_metrics?.net_pnl ?? 0) >= 0 ? "text-up" : "text-down")}>{money(f.test_metrics?.net_pnl, 0, true)}</td></tr>
              ))}</tbody>
            </table></div>
          </Panel>
          <Panel className="border-0 min-w-0" title="Stitched out-of-sample equity (re-optimised each fold)">
            <LWChart height={240} series={[{ type: "area", data: (o.wf_equity?.dates || []).map((d: string, i: number) => ({ time: dayTime(d), value: o.wf_equity.equity[i] })), options: { lineColor: C.accent, topColor: "rgba(232,163,61,0.2)", bottomColor: "rgba(232,163,61,0)" } }]} />
          </Panel>
        </>
      )}
      {o.pbo?.logits && (
        <Panel className="border-0" title="CSCV: rank logit of the in-sample winner out of sample (≤0 = overfit)">
          <EChart height={220} option={(() => {
            const L: number[] = o.pbo.logits;
            const lo = Math.min(...L), hi = Math.max(...L), nb = 20, w = (hi - lo) / nb || 1;
            const counts = Array(nb).fill(0);
            for (const x of L) counts[Math.min(nb - 1, Math.floor((x - lo) / w))]++;
            return { grid: { left: 36, right: 10, top: 14, bottom: 24 }, xAxis: { type: "category", data: counts.map((_, i) => (lo + (i + 0.5) * w).toFixed(1)) }, yAxis: { type: "value" },
              series: [{ type: "bar", data: counts.map((c, i) => ({ value: c, itemStyle: { color: lo + (i + 0.5) * w <= 0 ? C.down : C.up } })) }] } as any;
          })()} />
        </Panel>
      )}
      {o.pbo?.is_vs_oos && (
        <Panel className="border-0" title="CSCV: in-sample vs out-of-sample Sharpe of the selected configuration">
          <EChart height={220} option={{ grid: { left: 44, right: 12, top: 14, bottom: 30 }, xAxis: { type: "value", name: "IS", nameLocation: "middle", nameGap: 18 }, yAxis: { type: "value", name: "OOS" },
            series: [{ type: "scatter", symbolSize: 4, data: o.pbo.is_vs_oos, itemStyle: { color: C.info, opacity: 0.7 } }] } as any} />
        </Panel>
      )}
      {!oos.length && !wf.length && <Panel className="border-0 col-span-2"><Empty title="No out-of-sample validation in this study">Choose hold-out or walk-forward validation when launching a study.</Empty></Panel>}
    </div>
  );
}
