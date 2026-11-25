/* Comparison workspace: pinned results side by side on one time axis. Anything that makes two results
   not like-for-like (pricing basis, period, costs, symbol) is disclosed above the numbers, never below them. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { GitCompareArrows, PinOff, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { MODE_NAME, sgn, usd } from "@/lib/lab";
import { Callout, Empty, Section, Skeleton } from "./bits";
import { usePipeline } from "./Pipeline";

const PRICING: Record<string, string> = { historical: "observed bid/ask quotes", trades: "observed trade prints", modeled: "THEORETICAL premiums", "underlying prices (study)": "underlying prices" };

function Waiting({ id, label, onDone }: { id: string; label: string; onDone: (id: string, ok: boolean) => void }) {
  const { state, live } = usePipeline(id);
  useEffect(() => { if (state && !live) onDone(id, state.status === "done" && state.outcome !== "needs_decision"); }, [live, state?.status]); // eslint-disable-line
  const cur = state?.stages?.find((s) => s.status === "running");
  return (
    <div className="flex items-center gap-2.5 text-xs">
      <span className={cn("dot", live ? "bg-accent pulse" : state?.status === "error" || state?.outcome === "needs_decision" ? "bg-warn" : "bg-up")} />
      <span className="w-1/2 truncate soft">{label}</span>
      <span className="flex-1 truncate dim">{live ? cur?.label || "queued" : state?.status === "error" ? state?.error : state?.outcome === "needs_decision" ? "the data does not cover this window" : "done"}</span>
    </div>
  );
}

export function Compare({ ids, names, onOpen, onUnpin, onSwap }: { ids: string[]; names?: Record<string, string>; onOpen: (runId: string) => void; onUnpin: (runId: string) => void; onSwap: (from: string[], to: string[]) => void }) {
  const key = ids.join(",");
  const q = useQuery({ queryKey: ["compare", key], queryFn: () => api.get<any>(`/api/compare?ids=${key}`), enabled: ids.length >= 2, staleTime: 30_000 });
  const [rerun, setRerun] = useState<{ from: string; run_id: string; label: string }[] | null>(null);
  const [finished, setFinished] = useState<Record<string, boolean>>({});
  const m = useMutation({
    mutationFn: (w: [string, string]) => api.post<any>("/api/compare/rerun", { ids, start: w[0], end: w[1] }),
    onSuccess: (r) => { setFinished({}); setRerun(r.runs); },
  });
  useEffect(() => { setRerun(null); setFinished({}); }, [key]);
  useEffect(() => {
    if (!rerun || !rerun.length || rerun.some((r) => !(r.run_id in finished))) return;
    const ok = rerun.filter((r) => finished[r.run_id]);
    if (ok.length >= 2) onSwap(ok.map((r) => r.from), ok.map((r) => r.run_id));
  }, [finished, rerun]); // eslint-disable-line
  // results are named after the question that produced them where the notebook knows it
  const c = useMemo(() => {
    const d = q.data;
    if (!d) return d;
    const nm = (id: string, l: string) => names?.[id] || String(l).replace(/^v\d+\s*-?\s*/, "");
    const byLabel: Record<string, string> = Object.fromEntries(d.runs.map((r: any) => [r.label, nm(r.id, r.label)]));
    const rel = (l: string) => byLabel[l] || l;
    return { ...d, runs: d.runs.map((r: any) => ({ ...r, label: nm(r.id, r.label) })),
      compatibility: d.compatibility && { ...d.compatibility, differences: (d.compatibility.differences || []).map((x: any) => ({ ...x, values: x.values.map((v: any) => ({ ...v, label: rel(v.label) })) })) },
      paired: d.paired && { ...d.paired, only_in: d.paired.only_in.map((x: any) => ({ ...x, label: rel(x.label) })), difference: d.paired.difference && { ...d.paired.difference, label: Object.entries(byLabel).reduce((t, [a, b]) => t.split(a).join(b), String(d.paired.difference.label || "")) } } };
  }, [q.data, names]);
  const runs: any[] = c?.runs || [];
  const study = runs[0]?.kind === "study";
  const curves = useMemo(() => ({
    tooltip: { trigger: "axis", valueFormatter: (v: any) => (v === null || v === undefined ? "—" : study ? `${Number(v).toFixed(2)}%` : usd(Number(v))) },
    legend: { type: "scroll", left: 4, right: 4, top: 0 },
    grid: { left: 56, right: 18, top: 34, bottom: 28 },
    xAxis: { type: "time" },
    yAxis: { type: "value", scale: true, axisLabel: { formatter: (v: number) => (study ? `${v.toFixed(0)}%` : fmtMoneyAxis(v)) } },
    series: runs.map((r, i) => ({ name: r.label, type: "line", step: study ? "end" : undefined, showSymbol: (r.curve || []).length < 3, symbolSize: 9, lineStyle: { width: 1.8, type: r.pricing === "modeled" ? "dashed" : "solid" }, color: C.palette[i % C.palette.length], data: r.curve || [], connectNulls: true })),
  }), [c]); // eslint-disable-line
  const dd = useMemo(() => ({
    tooltip: { trigger: "axis", valueFormatter: (v: any) => (v === null || v === undefined ? "—" : `${(Number(v) * 100).toFixed(1)}%`) },
    grid: { left: 56, right: 18, top: 10, bottom: 28 },
    xAxis: { type: "time" },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: runs.filter((r) => r.drawdown).map((r) => ({ name: r.label, type: "line", showSymbol: (r.drawdown || []).length < 3, symbolSize: 7, lineStyle: { width: 1.2 }, areaStyle: { opacity: 0.08 }, color: C.palette[runs.indexOf(r) % C.palette.length], data: r.drawdown })),
  }), [c]); // eslint-disable-line

  if (ids.length < 2) {
    return <Empty icon={GitCompareArrows} title="Pin two or more results to compare them">Use <b>Pin</b> on any result or report. Pinned results appear here on one time axis, with everything that differs between them spelled out.</Empty>;
  }
  if (q.isLoading) return <div className="flex flex-col gap-3 p-5"><Skeleton className="h-16" /><Skeleton className="h-64" /></div>;
  if (q.isError || !c) return <Empty title="These results cannot be compared">{String((q.error as any)?.message || "")}</Empty>;
  const comp = c.compatibility || {};
  const modeledMix = runs.some((r) => r.pricing === "modeled") && runs.some((r) => r.pricing !== "modeled");
  const table = c.table;
  const d = c.paired?.difference;
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="compare">
      <div className="flex items-center gap-2"><h2>Comparison</h2><span className="text-xs dim">{runs.length} pinned results</span></div>
      {modeledMix && <Callout tone="bad" title="Theoretical and observed prices are mixed here">At least one of these results uses model-estimated option premiums and another uses prices that were actually observed. Their profit figures cannot be compared; the dashed line is the theoretical one.</Callout>}
      {comp.warning && !modeledMix && <Callout tone="warn" title="Not like for like">{comp.warning}</Callout>}
      {(comp.differences || []).length > 0 ? (
        <div className="card overflow-hidden" data-testid="compat">
          <div className="px-4 pt-3 text-[13px] font-medium">What differs between these results</div>
          <table className="t mt-2">
            <thead><tr><th>Setting</th>{runs.map((r) => <th key={r.id}>{r.label}</th>)}</tr></thead>
            <tbody>
              {comp.differences.map((x: any) => (
                <tr key={x.field}><td className={cn(x.severity === "high" ? "text-warn" : "soft")}>{x.label}</td>{x.values.map((v: any, i: number) => <td key={i} className="soft">{v.value}</td>)}</tr>
              ))}
            </tbody>
          </table>
          {comp.shared_window && comp.differences.some((x: any) => x.field === "period") && (
            <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-3">
              <span className="text-[13px] soft">Every result has data for <span className="num text-fg">{comp.shared_window[0]}</span> to <span className="num text-fg">{comp.shared_window[1]}</span>.</span>
              <button className="btn sm" disabled={m.isPending || !!rerun} onClick={() => m.mutate(comp.shared_window)} data-testid="rerun-shared"><RefreshCw />Re-run all on that window</button>
              {m.isError && <span className="text-xs text-down">{String((m.error as any)?.message)}</span>}
            </div>
          )}
          {rerun && <div className="flex flex-col gap-1.5 border-t border-line px-4 py-3">{rerun.map((r) => <Waiting key={r.run_id} id={r.run_id} label={r.label} onDone={(id, ok) => setFinished((f) => (id in f ? f : { ...f, [id]: ok }))} />)}
            <div className="text-xs dim">The originals are kept; the re-run results replace them in this comparison when they finish.</div></div>}
        </div>
      ) : <Callout tone="good" title="Like for like">Same symbol, period, pricing basis and execution costs.</Callout>}

      <div className="card overflow-x-auto">
        <table className="t">
          <thead><tr><th>Result</th><th>Priced on</th><th>Period</th>{(table?.columns || []).map((k: any) => <th key={k.key} className="r">{k.label}</th>)}<th /></tr></thead>
          <tbody>
            {runs.map((r, i) => {
              const row = table?.rows?.find((x: any) => x.run_id === r.id);
              return (
                <tr key={r.id} className="click" onClick={() => onOpen(r.id)} title="Open this result's report">
                  <td className="max-w-[300px] truncate" title={r.label}><span className="mr-2 inline-block size-2.5 rounded-sm align-middle" style={{ background: C.palette[i % C.palette.length] }} />{r.label}</td>
                  <td className={cn(r.pricing === "modeled" ? "text-warn" : "soft")}>{PRICING[r.pricing] || MODE_NAME[r.pricing] || r.pricing || "—"}</td>
                  <td className="num dim whitespace-nowrap">{r.period?.[0]} → {r.period?.[1]}</td>
                  {(table?.columns || []).map((k: any) => <td key={k.key} className="r num">{row?.shown?.[k.key] ?? "—"}</td>)}
                  <td className="r"><button className="btn quiet sm !px-1.5" title="Remove from the comparison" aria-label="Unpin" onClick={(e) => { e.stopPropagation(); onUnpin(r.id); }}><PinOff /></button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {c.mixed_kinds && <div className="border-t border-line px-4 py-2.5 text-xs dim">A market study counts how the underlying moved; a backtest counts what an option position earned. Their numbers are shown in separate units.</div>}
      </div>

      {!c.mixed_kinds && (
        <Section title={study ? "Cumulative end-of-period return of the qualifying sessions" : "Net profit over time"} sub="one shared time axis; move the pointer to read every result on the same date">
          <EChart option={curves} height={300} group={`cmp-${key}`} />
          {!study && dd.series.length > 0 && <><div className="mt-3 mb-1 text-xs dim">Drawdown from the previous peak</div><EChart option={dd} height={150} group={`cmp-${key}`} /></>}
        </Section>
      )}

      {c.paired && (
        <Section title="On the same dates" sub={`${c.paired.dates_in_common} date${c.paired.dates_in_common === 1 ? "" : "s"} appear in every result`}>
          {c.paired.only_in.some((x: any) => x.count > 0) && <div className="mb-2 text-[13px] dim">Not shared: {c.paired.only_in.filter((x: any) => x.count > 0).map((x: any) => `${x.count} only in ${x.label}`).join("; ")}.</div>}
          {d && d.n > 0 && (
            <div className="mb-3 text-[13.5px]">
              <span className="soft">{d.label}: </span>
              <span className={cn("num font-medium", d.mean > 0 ? "text-up" : d.mean < 0 ? "text-down" : "")}>{study ? sgn(d.mean, 2, "%") : usd(d.mean, 2)}</span>
              <span className="soft"> on average per shared date</span>
              {d.ci95?.[0] !== null && d.ci95?.[0] !== undefined && <span className="dim"> (95% interval {study ? sgn(d.ci95[0], 2, "%") : usd(d.ci95[0], 2)} to {study ? sgn(d.ci95[1], 2, "%") : usd(d.ci95[1], 2)}{d.p_value !== null && d.p_value !== undefined ? `, p = ${d.p_value.toFixed(3)}` : ""}, {d.n} dates)</span>}
              {d.n < 30 && <span className="text-warn"> · too few shared dates to conclude anything</span>}
            </div>
          )}
          {c.paired.rows.length > 0 ? (
            <div className="max-h-[340px] overflow-auto rounded-lg border border-line">
              <table className="t">
                <thead><tr><th>Date</th>{runs.map((r) => <th key={r.id} className="r">{r.label}</th>)}</tr></thead>
                <tbody>
                  {[...c.paired.rows].reverse().map((p: any) => (
                    <tr key={p.date}><td className="num">{p.date}</td>{p.values.map((v: number, i: number) => <td key={i} className={cn("r num", v > 0 ? "text-up" : v < 0 ? "text-down" : "")}>{study ? sgn(v, 2, "%") : usd(v, 2)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="text-[13px] dim">These results share no dates, so there is nothing to pair.</div>}
        </Section>
      )}
    </div>
  );
}
