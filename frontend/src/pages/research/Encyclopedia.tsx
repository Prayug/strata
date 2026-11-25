/* The market behaviour encyclopedia: reproducible historical studies, computed on the stored data when opened
   and reconfigurable in place. Nothing here is a stored number. Any row opens its sessions, and any study can be
   handed to the AI Lab as a hypothesis to research properly. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { BookOpenText, FlaskConical, Loader2, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { pp } from "@/lib/lab";
import { Callout, Empty, Skeleton, StatTile } from "../lab/bits";
import { pv } from "../lab/Research";
import { CellPanel } from "./Sensitivity";

function Chart({ ch }: { ch: any }) {
  const option = useMemo(() => {
    const thin: boolean[] = ch.thin || [];
    const series: any[] = ch.series.map((s: any, i: number) => i === 0
      ? { name: s.name, type: "bar", barMaxWidth: 34, data: s.values.map((v: number | null, j: number) => ({ value: v, itemStyle: { color: C.accent, opacity: thin[j] ? 0.35 : 0.9, borderColor: thin[j] ? C.accent : undefined, borderType: thin[j] ? "dashed" : undefined, borderWidth: thin[j] ? 1 : 0, borderRadius: [3, 3, 0, 0] } })) }
      : { name: s.name, type: "line", symbol: "diamond", symbolSize: 8, lineStyle: { width: 1, type: "dashed" }, color: C.fg, data: s.values });
    if (ch.ci) series.push({ type: "custom", name: "95% interval", silent: true, data: ch.ci.map((c: any, i: number) => [i, c[0], c[1]]),
      renderItem: (_: any, a: any) => { if (a.value(1) === null || Number.isNaN(a.value(1))) return null; const lo = a.coord([a.value(0), a.value(1)]), hi = a.coord([a.value(0), a.value(2)]);
        return { type: "group", children: [{ type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: { stroke: C.fg, lineWidth: 1.2 } }, { type: "line", shape: { x1: lo[0] - 4, y1: lo[1], x2: lo[0] + 4, y2: lo[1] }, style: { stroke: C.fg } }, { type: "line", shape: { x1: hi[0] - 4, y1: hi[1], x2: hi[0] + 4, y2: hi[1] }, style: { stroke: C.fg } }] }; } });
    return {
      title: { text: ch.title }, grid: { left: 44, right: 12, top: 44, bottom: 78 }, legend: { top: 18, data: ch.series.map((s: any) => s.name) }, tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      xAxis: { type: "category", data: ch.x.map((x: string) => x.split(" : ")[0]), axisLabel: { interval: 0, rotate: ch.x.length > 6 ? 28 : 0, width: 120, overflow: "truncate", fontFamily: "Inter", fontSize: 10.5 } },
      yAxis: { type: "value", min: 0, axisLabel: { formatter: (v: number) => `${v.toFixed(0)}${ch.unit || ""}` } }, series,
    };
  }, [ch]);
  return <EChart option={option} height={300} />;
}

export default function Encyclopedia() {
  const nav = useNavigate();
  const { id } = useParams();
  const list = useQuery({ queryKey: ["ency-list"], queryFn: () => api.get<any>("/api/encyclopedia"), staleTime: 120_000 });
  const [q, setQ] = useState("");
  const [params, setParams] = useState<Record<string, any>>({});
  const [applied, setApplied] = useState<Record<string, any>>({});
  const [cell, setCell] = useState<any>(null);
  const entries: any[] = list.data?.entries || [];
  const cur = entries.find((e) => e.id === id) || entries[0];
  useEffect(() => { setParams({}); setApplied({}); setCell(null); }, [cur?.id]);
  const sym = String(applied.symbol || params.symbol || (list.data?.symbols?.includes("SPY") ? "SPY" : list.data?.symbols?.[0]) || "SPY");
  const run = useQuery({ queryKey: ["ency", cur?.id, JSON.stringify(applied), sym], enabled: !!cur, staleTime: 300_000, queryFn: () => api.post<any>(`/api/encyclopedia/${cur.id}`, { ...applied, symbol: sym }) });
  const toLab = useMutation({
    mutationFn: async (text: string) => { const s = await api.post<any>("/api/ai/sessions", { title: text.slice(0, 70) }); sessionStorage.setItem(`lab-first-${s.id}`, text); return s.id as string; },
    onSuccess: (sid) => nav(`/lab/${sid}`), onError: (e: any) => toast.error(e.message),
  });
  const shown = entries.filter((e) => !q || `${e.title} ${e.blurb} ${e.category}`.toLowerCase().includes(q.toLowerCase()));
  const cats = [...new Set(shown.map((e) => e.category))];
  const r = run.data;
  const table = r?.tables?.[0];
  return (
    <div className="lab flex h-full min-h-0">
      <aside className="flex w-[280px] shrink-0 flex-col border-r border-line bg-bg-1">
        <div className="px-3.5 pb-2 pt-4">
          <div className="eyebrow flex items-center gap-2"><BookOpenText className="size-3.5 text-accent" />Market behaviour</div>
          <div className="mt-1.5 text-[12.5px] dim">Historical studies computed on your stored data. Change any setting and they recompute.</div>
          <div className="relative mt-3"><Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-faint" /><input className="field !pl-8" placeholder="Search studies" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search studies" /></div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-2 pb-4">
          {cats.map((c) => (
            <div key={c} className="mt-2"><div className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-faint">{c}</div>
              {shown.filter((e) => e.category === c).map((e) => (
                <button key={e.id} onClick={() => nav(`/encyclopedia/${e.id}`)} data-testid={`entry-${e.id}`} className={cn("block w-full rounded-lg border px-2.5 py-2 text-left transition-colors", cur?.id === e.id ? "border-accent/60 bg-accent/[0.07]" : "border-transparent hover:bg-bg-2")}>
                  <div className="text-[13px]">{e.title}</div><div className="mt-0.5 text-[11.5px] dim">{e.blurb}</div>
                </button>
              ))}</div>
          ))}
          {list.isSuccess && shown.length === 0 && <div className="px-2 py-6 text-center text-xs dim">No study matches that search.</div>}
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-auto px-6 py-5">
        {!cur ? (list.isLoading ? <Skeleton className="h-64" /> : <Empty icon={BookOpenText} title="No studies" />) : (
          <div className="mx-auto flex max-w-[1080px] flex-col gap-4" data-testid="ency-entry">
            <div className="flex items-start gap-4">
              <div className="min-w-0 flex-1"><div className="eyebrow">{cur.category}</div><h2 className="mt-1">{cur.title}</h2><p className="mt-1 text-[13.5px] soft">{cur.blurb}</p></div>
              {r && <button className="btn primary" disabled={toLab.isPending} onClick={() => toLab.mutate(r.hypothesis)} data-testid="to-hypothesis" title={r.hypothesis}>{toLab.isPending ? <Loader2 className="animate-spin" /> : <FlaskConical />}Research this as a hypothesis</button>}
            </div>
            <form className="card flex flex-wrap items-end gap-3 px-4 py-3" onSubmit={(e) => { e.preventDefault(); setApplied({ ...params }); }}>
              {cur.params.map((p: any) => {
                const v = params[p.name] ?? applied[p.name] ?? p.default ?? "";
                const set = (x: any) => setParams((s) => ({ ...s, [p.name]: x }));
                return (
                  <label key={p.name} className="text-xs dim">{p.label}
                    {p.kind === "symbol" ? <select className="field mt-1 !w-28" value={p.name === "symbol" ? (params.symbol || sym) : v} onChange={(e) => set(e.target.value)} aria-label={p.label}>{(list.data?.symbols || []).filter((s: string) => p.name !== "symbol" || !/VIX/.test(s)).map((s: string) => <option key={s}>{s}</option>)}</select>
                      : p.kind === "choice" ? <select className="field mt-1 !w-28" value={v} onChange={(e) => set(Number(e.target.value))} aria-label={p.label}>{p.choices.map((c: any) => <option key={c}>{c}</option>)}</select>
                      : p.kind === "date" ? <input type="date" className="field mt-1 !w-40" value={v || ""} onChange={(e) => set(e.target.value || null)} aria-label={p.label} />
                      : <input type="number" className="field mt-1 !w-28" value={v} step={p.step} min={p.min} max={p.max} onChange={(e) => set(e.target.value === "" ? "" : Number(e.target.value))} aria-label={p.label} />}
                  </label>
                );
              })}
              <button className="btn" type="submit" data-testid="ency-apply">Recompute</button>
              {r && <span className="ml-auto text-xs dim num">{r.sessions?.toLocaleString()} sessions · {r.window?.[0]} → {r.window?.[1]}</span>}
            </form>
            {run.isLoading && <Skeleton className="h-72" />}
            {run.isError && <Callout tone="bad">{String((run.error as any)?.message)}</Callout>}
            {r && (
              <>
                {r.synthetic && <Callout tone="warn">This symbol is simulated sandbox data, not market history.</Callout>}
                <div className="card px-4 py-3.5 text-[14px] leading-relaxed" data-testid="ency-summary">{r.summary}</div>
                {r.stats?.length > 0 && <div className="grid grid-cols-2 gap-3 md:grid-cols-4">{r.stats.map((s: any) => <StatTile key={s.label} label={s.label} value={s.value} />)}</div>}
                {(r.charts || []).map((ch: any, i: number) => <div key={i} className="card px-3 py-3"><Chart ch={ch} /><div className="px-1 text-[11.5px] dim">Faded, dashed bars have fewer than 30 observations. Whiskers are 95% intervals that resample whole weeks; diamonds are all evaluated sessions.</div></div>)}
                {table && (
                  <div className="overflow-auto rounded-[10px] border border-line">
                    <table className="t" data-testid="ency-table">
                      <thead><tr><th>Condition → outcome</th><th className="r">Obs.</th><th className="r">Frequency</th><th className="r">All sessions</th><th className="r">Difference</th><th className="r">Avg move</th><th className="r"><Tip content="Corrected for every row of this table (Holm), from a test that resamples whole weeks."><span className="term">p, corrected</span></Tip></th><th>Reading</th></tr></thead>
                      <tbody>
                        {table.rows.map((x: any, i: number) => (
                          <tr key={i} className="click" onClick={() => setCell({ ...x.cell, condition: x.condition, outcome_text: x.label.split(" : ")[1], n: x.n, probability: x.frequency, baseline_probability: x.baseline, ci95: x.ci95, p_adjusted: x.p_adjusted, thin: x.thin })} title="Open the sessions behind this row">
                            <td className="max-w-[360px]">{x.label}</td>
                            <td className={cn("r num", x.thin && "text-warn")}>{x.n}{x.thin ? " ⚠" : ""}</td><td className="r num">{pp(x.frequency)}</td><td className="r num dim">{pp(x.baseline)}</td>
                            <td className={cn("r num", (x.difference ?? 0) > 0 ? "text-up" : "text-down")}>{x.difference == null ? "—" : `${x.difference > 0 ? "+" : ""}${(x.difference * 100).toFixed(1)} pts`}</td>
                            <td className="r num">{x.avg_return_pct == null ? "—" : `${x.avg_return_pct > 0 ? "+" : ""}${x.avg_return_pct.toFixed(2)}%`}</td><td className="r num">{pv(x.p_adjusted)}</td>
                            <td className="whitespace-nowrap text-xs">{x.thin ? <span className="text-warn">○ too few observations</span> : x.distinguishable ? <span className="text-up">◆ differs from the baseline</span> : <span className="dim">– within chance</span>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <ul className="flex flex-col gap-1.5 text-[13px] soft">{r.notes.map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
                <div className="card flex items-center gap-3 px-4 py-3"><FlaskConical className="size-4 shrink-0 text-accent" /><div className="min-w-0 flex-1 text-[13px]"><span className="dim">As a research question: </span>{r.hypothesis}</div>
                  <button className="btn sm" onClick={() => toLab.mutate(r.hypothesis)}>Open in the AI Lab</button></div>
              </>
            )}
          </div>
        )}
      </main>
      {cell && <CellPanel symbol={sym} cell={cell} start={applied.start || null} end={applied.end || null} priorTrials={table?.rows.length || 1} onClose={() => setCell(null)} />}
    </div>
  );
}
