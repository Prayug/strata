/* Pattern explorer: choose a market event, see every historical occurrence and what price did afterwards.
   It describes the underlying only. Each exploration is recorded in the experiment registry like any other look at
   the data, and a strategy is never built from here directly: it goes through a tested finding in a notebook. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, GitCompareArrows, Layers, Loader2, Plus, Sparkles, Waves, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { ExprInput } from "@/components/research/ExprInput";
import { SyntheticBadge } from "@/components/common";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { openNotebook, pp } from "@/lib/lab";
import { Callout, Empty, Section, Skeleton, StatTile } from "../lab/bits";
import { Forest, PathChart } from "../lab/Graph";
import { pv } from "../lab/Research";

type Param = { name: string; label: string; default: any; min?: number; max?: number; step?: number; unit?: string; options?: string[]; kind?: string };
type Ev = { id: string; title: string; group: string; anchor: string; params: Param[]; needs: string };

const sr = (x: number | null | undefined, dp = 2) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(dp)}%`);
const defaults = (e?: Ev) => Object.fromEntries((e?.params || []).map((p) => [p.name, p.default]));

function ParamField({ p, value, onChange }: { p: Param; value: any; onChange: (v: any) => void }) {
  if (p.kind === "expr") return <label className="flex flex-col gap-1"><span className="text-xs dim">{p.label}</span><ExprInput value={String(value ?? "")} onChange={onChange} /></label>;
  if (p.kind === "tick") return (
    <label className="flex flex-col gap-1"><span className="text-xs dim">{p.label}</span>
      <select className="field" value={value} onChange={(e) => onChange(e.target.value)}>{["open", "close", "10:00", "10:30", "11:00", "12:00", "13:00", "14:00", "15:00"].map((o) => <option key={o} value={o}>{o === "open" ? "the open" : o === "close" ? "the close" : `${o} ET`}</option>)}</select>
    </label>
  );
  if (p.options) return (
    <label className="flex flex-col gap-1"><span className="text-xs dim">{p.label}</span>
      <select className="field" value={value} onChange={(e) => onChange(e.target.value)}>{p.options.map((o) => <option key={o} value={o}>{o}</option>)}</select>
    </label>
  );
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between text-xs dim"><span>{p.label}</span><span className="num text-fg">{value}{p.unit}</span></span>
      <input type="range" min={p.min} max={p.max} step={p.step} value={Number(value)} onChange={(e) => onChange(Number(e.target.value))} className="accent-[var(--color-accent)]" aria-label={p.label} />
    </label>
  );
}

function HorizonTable({ rows, main }: { rows: any[]; main: number }) {
  return (
    <div className="overflow-auto">
      <table className="t">
        <thead><tr><th>Measured</th><th className="r">Occurrences</th><th className="r">Average</th><th className="r">Higher</th><th className="r" title="Average and share finishing higher on every other session">Other sessions</th><th className="r">Difference</th><th className="r" title="95% interval of the average, resampling whole calendar weeks">95% interval</th><th className="r" title="Adjusted for the number of horizons shown (Holm)">p (adjusted)</th><th className="r" title="Average best and worst point reached inside the window">Best / worst</th></tr></thead>
        <tbody>
          {rows.map((h) => {
            const clear = h.p_adjusted !== null && h.p_adjusted !== undefined && h.p_adjusted < 0.05 && !h.thin;
            return (
              <tr key={h.horizon} className={cn(h.horizon === main && "sel")}>
                <td>{h.label}{h.thin && <span className="ml-1.5 text-[11px] text-warn" title="Fewer than 30 occurrences">thin</span>}</td>
                <td className="r num">{h.n}</td>
                <td className="r num" title={`median ${sr(h.median)}`}>{sr(h.mean)}</td>
                <td className="r num">{pp(h.positive)}</td>
                <td className="r num dim">{sr(h.rest_mean)} · {pp(h.rest_positive)}</td>
                <td className="r num">{sr(h.difference)}</td>
                <td className="r num dim">{h.mean_ci95 ? `${sr(h.mean_ci95[0])} to ${sr(h.mean_ci95[1])}` : "—"}</td>
                <td className="r num">{pv(h.p_adjusted)}<span className="ml-1.5 text-[11px]" title={clear ? "Distinguishable from other sessions after adjustment" : "Not distinguishable from other sessions"}>{clear ? "● differs" : "○ within chance"}</span></td>
                <td className="r num dim">{sr(h.avg_mfe, 1)} / {sr(h.avg_mae, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Patterns() {
  const nav = useNavigate();
  const cat = useQuery({ queryKey: ["pattern-events"], queryFn: () => api.get<{ events: Ev[]; symbols: string[]; default_symbol: string; horizons: number[] }>("/api/patterns/events") });
  const [symbol, setSymbol] = useState("");
  const [eid, setEid] = useState("down_day");
  const [params, setParams] = useState<Record<string, any>>({});
  const [period, setPeriod] = useState<{ start: string; end: string }>({ start: "", end: "" });
  const [horizon, setHorizon] = useState(10);
  const [mainH, setMainH] = useState(5);
  const [tgt, setTgt] = useState<{ on: boolean; target: number; stop: number | null; direction: "up" | "down" }>({ on: false, target: 1, stop: null, direction: "up" });
  const [overlays, setOverlays] = useState<{ id: string; [k: string]: any }[]>([]);
  const [focus, setFocus] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);

  const events = cat.data?.events || [];
  const ev = events.find((e) => e.id === eid);
  useEffect(() => { if (cat.data && !symbol) setSymbol(cat.data.default_symbol); }, [cat.data, symbol]);
  useEffect(() => { setParams(defaults(ev)); }, [eid, events.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useMutation({
    mutationFn: () => api.post<any>("/api/patterns/explore", {
      symbol, event: { id: eid, ...params }, start: period.start || null, end: period.end || null, horizon, main_horizon: Math.min(mainH, horizon),
      target: tgt.on ? tgt.target : null, stop: tgt.on ? tgt.stop : null, direction: tgt.direction, overlays,
    }),
    onError: (e: any) => toast.error(e.message),
  });
  const r = run.data;
  // first look: run once when the catalogue arrives so the page is never empty
  useEffect(() => { if (symbol && ev && !run.data && !run.isPending && Object.keys(params).length === (ev.params || []).length) run.mutate(); }, [symbol, ev, params]); // eslint-disable-line react-hooks/exhaustive-deps

  const groups = useMemo(() => {
    const by: Record<string, Ev[]> = {};
    for (const e of events) (by[e.group] ||= []).push(e);
    return Object.entries(by);
  }, [events]);
  const main = r?.horizons?.find((h: any) => h.horizon === r.main_horizon) || r?.horizons?.[0];
  const goAsk = async (text: string) => {
    setOpening(true);
    try { nav(await openNotebook(text)); } catch (e: any) { toast.error(e.message); setOpening(false); }
  };
  const question = r ? `Is ${r.symbol} more likely to rise over the next ${r.main_horizon} sessions after ${String(r.event.text).replace(new RegExp(`^${r.symbol}\\s+`), "it ")}? Control for volatility and check robustness.` : "";

  return (
    <div className="lab flex h-full min-h-0">
      <aside className="flex w-[304px] shrink-0 flex-col border-r border-line bg-bg-1" aria-label="Event to explore">
        <div className="border-b border-line px-4 py-3">
          <div className="eyebrow flex items-center gap-2"><Waves className="size-3.5 text-accent" />Pattern explorer</div>
          <div className="mt-1 text-xs dim">Pick an event. See what price did after every time it happened.</div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
          <label className="flex flex-col gap-1"><span className="text-xs dim">Symbol</span>
            <select className="field" value={symbol} onChange={(e) => setSymbol(e.target.value)} data-testid="pattern-symbol">{(cat.data?.symbols || []).map((s) => <option key={s}>{s}</option>)}</select>
          </label>
          <div className="mt-4 flex flex-col gap-3">
            {groups.map(([g, list]) => (
              <div key={g}>
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-faint">{g}</div>
                <div className="flex flex-col gap-0.5">
                  {list.map((e) => (
                    <button key={e.id} onClick={() => setEid(e.id)} data-testid={`event-${e.id}`}
                      className={cn("flex items-center justify-between rounded-md px-2 py-1.5 text-left text-[13px] transition-colors hover:bg-bg-3", eid === e.id ? "bg-bg-3 text-fg shadow-[inset_2px_0_0_var(--color-accent)]" : "soft")}>
                      <span>{e.title}</span>{e.needs === "intraday" && <span className="text-[10.5px] text-faint">intraday</span>}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-3 border-t border-line px-4 py-3">
          {(ev?.params || []).map((p) => <ParamField key={p.name} p={p} value={params[p.name] ?? p.default} onChange={(v) => setParams((s) => ({ ...s, [p.name]: v }))} />)}
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1"><span className="text-xs dim">From</span><input type="date" className="field" value={period.start} onChange={(e) => setPeriod((s) => ({ ...s, start: e.target.value }))} /></label>
            <label className="flex flex-col gap-1"><span className="text-xs dim">To</span><input type="date" className="field" value={period.end} onChange={(e) => setPeriod((s) => ({ ...s, end: e.target.value }))} /></label>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1"><span className="text-xs dim">Follow for</span>
              <select className="field" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>{[5, 10, 20].map((h) => <option key={h} value={h}>{h} sessions</option>)}</select></label>
            <label className="flex flex-col gap-1"><span className="text-xs dim">Headline at</span>
              <select className="field" value={Math.min(mainH, horizon)} onChange={(e) => setMainH(Number(e.target.value))}>{(cat.data?.horizons || []).filter((h) => h <= horizon).map((h) => <option key={h} value={h}>{h} session{h > 1 ? "s" : ""}</option>)}</select></label>
          </div>
          <label className="flex items-center gap-2 text-xs soft"><input type="checkbox" checked={tgt.on} onChange={(e) => setTgt((s) => ({ ...s, on: e.target.checked }))} className="accent-[var(--color-accent)]" />Also ask whether a price target was reached</label>
          {tgt.on && (
            <div className="grid grid-cols-3 gap-2">
              <label className="flex flex-col gap-1"><span className="text-xs dim">Move</span><select className="field" value={tgt.direction} onChange={(e) => setTgt((s) => ({ ...s, direction: e.target.value as any }))}><option value="up">up</option><option value="down">down</option></select></label>
              <label className="flex flex-col gap-1"><span className="text-xs dim">Target %</span><input type="number" step={0.25} min={0.1} className="field num" value={tgt.target} onChange={(e) => setTgt((s) => ({ ...s, target: Number(e.target.value) }))} /></label>
              <label className="flex flex-col gap-1"><span className="text-xs dim">Stop %</span><input type="number" step={0.25} min={0} className="field num" placeholder="none" value={tgt.stop ?? ""} onChange={(e) => setTgt((s) => ({ ...s, stop: e.target.value ? Number(e.target.value) : null }))} /></label>
            </div>
          )}
          <button className="btn primary" onClick={() => run.mutate()} disabled={run.isPending || !symbol} data-testid="pattern-run">{run.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}Explore</button>
          <div className="text-[11px] dim">Each exploration is recorded in the experiment registry: looking at dates counts as having seen them.</div>
        </div>
      </aside>

      <div className="min-w-0 flex-1 overflow-auto">
        <div className="mx-auto flex w-full max-w-[1120px] flex-col gap-4 px-6 py-6">
          {cat.isLoading && <Skeleton className="h-40" />}
          {cat.data && !cat.data.symbols.length && <Empty title="No market history is stored yet" icon={Waves}>Download free daily history on the Data page, then every event here has occurrences to show.</Empty>}
          {run.isPending && !r && <><Skeleton className="h-24" /><Skeleton className="h-80" /></>}
          {r && (
            <div className={cn("flex flex-col gap-4", run.isPending && "opacity-60")} data-testid="pattern-result">
              <div>
                <div className="flex flex-wrap items-center gap-2 text-xs dim"><span>{r.symbol}</span><span>·</span><span>{r.window?.[0]} to {r.window?.[1]}</span><SyntheticBadge on={r.synthetic} /></div>
                <h1 className="mt-1 !text-[22px]">{r.event.text}</h1>
                <p className="mt-1.5 text-[14px] soft">
                  {r.n ? <>It happened <b className="text-fg">{r.n.toLocaleString()}</b> times in {r.evaluated.toLocaleString()} sessions ({pp(r.frequency)}).{r.not_evaluable > 0 && ` ${r.not_evaluable} session${r.not_evaluable > 1 ? "s" : ""} could not be evaluated and ${r.not_evaluable > 1 ? "are" : "is"} left out, not counted as a miss.`}</> : "No stored session satisfies this event."}
                </p>
              </div>
              {r.n > 0 && main && (
                <>
                  <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                    <StatTile label={`Average, ${main.label}`} value={sr(main.mean)} sub={`other sessions ${sr(main.rest_mean)}`} hint="Close-to-close change of the underlying from the anchor. No option prices are involved." />
                    <StatTile label="Finished higher" value={pp(main.positive)} sub={`other sessions ${pp(main.rest_positive)}`} />
                    <StatTile label="Difference vs other sessions" value={sr(main.difference)} sub={main.mean_ci95 ? `average 95% interval ${sr(main.mean_ci95[0])} to ${sr(main.mean_ci95[1])}` : undefined} />
                    <StatTile label="Could this be chance?" value={main.thin ? "too few" : pv(main.p_adjusted)} sub={main.thin ? "fewer than 30 occurrences" : main.p_adjusted < 0.05 ? "● differs after adjustment" : "○ within what chance produces"} hint="Permutation test of the average against other sessions, adjusted for the number of horizons shown." />
                  </div>

                  <Section title="Price paths after the event" sub={`${r.paths.shown} of ${r.paths.n} paths drawn · anchored at the ${r.paths.entry === "close" ? "event close" : r.paths.entry}`}
                    right={focus && <button className="btn quiet sm" onClick={() => nav(`/explore?symbol=${r.symbol}&date=${focus}`)}>Open {focus}<ArrowRight /></button>}>
                    <PathChart paths={r.paths} height={360} onPick={setFocus} overlays={(r.overlays || []).filter((o: any) => o.bands).map((o: any) => ({ name: o.event.text, steps: o.steps, bands: o.bands }))} />
                    <div className="mt-2 text-xs dim">Shaded bands hold the middle half and the 5th to 95th percentile of paths. The dashed line is the average path from every other session. Click a faint line to pick that occurrence.</div>
                  </Section>

                  <Section title="Forward returns by horizon" sub="every horizon is tested, so the p-values are adjusted for that">
                    <HorizonTable rows={r.horizons} main={r.main_horizon} />
                    <div className="mt-4"><Forest unit="%" scale={1} rows={r.horizons.map((h: any) => ({ label: h.label, estimate: h.difference, lo: h.mean_ci95 ? h.mean_ci95[0] - h.rest_mean : null, hi: h.mean_ci95 ? h.mean_ci95[1] - h.rest_mean : null, n: h.n, note: `${h.n} occurrences` }))} /></div>
                  </Section>

                  {r.target && (
                    <Section title={`Target: ${r.target.outcome_text}`}>
                      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                        <StatTile label="Reached the target" value={pp(r.target.evidence.probability)} sub={`${r.target.evidence.hits} of ${r.target.evidence.n}`} />
                        <StatTile label="Other sessions" value={pp(r.target.evidence.rest_probability ?? r.target.evidence.baseline_probability)} />
                        <StatTile label="Stopped first" value={String(r.target.stopped)} sub={r.target.ambiguous ? `${r.target.ambiguous} ambiguous bars left out` : undefined} />
                        <StatTile label="Could this be chance?" value={pv(r.target.evidence.p_value)} sub={r.target.time_to_target ? `typically reached ${r.target.time_to_target.median_step}` : undefined} />
                      </div>
                      {r.target.time_to_target && <div className="mt-2 text-xs dim">{r.target.time_to_target.note}</div>}
                    </Section>
                  )}

                  <div className="grid gap-4 lg:grid-cols-2">
                    {(r.regimes || []).map((g: any) => (
                      <Section key={g.title} title={g.title} sub={g.missing ? `${g.missing} occurrence${g.missing > 1 ? "s" : ""} without the reading` : undefined}>
                        <table className="t"><thead><tr><th>Regime</th><th className="r">Occurrences</th><th className="r">Average</th><th className="r">Higher</th><th className="r">95% interval</th></tr></thead>
                          <tbody>{g.rows.map((x: any) => <tr key={x.regime}><td>{x.regime}{x.thin && <span className="ml-1.5 text-[11px] text-warn">thin</span>}</td><td className="r num">{x.n}</td><td className="r num">{sr(x.mean)}</td><td className="r num">{pp(x.positive)}</td><td className="r num dim">{x.mean_ci95 ? `${sr(x.mean_ci95[0])} to ${sr(x.mean_ci95[1])}` : "—"}</td></tr>)}</tbody></table>
                      </Section>
                    ))}
                    <Section title="When it happened" sub="occurrences per year">
                      <EChart height={190} option={{ grid: { left: 34, right: 8, top: 10, bottom: 22 }, tooltip: { trigger: "axis" }, xAxis: { type: "category", data: r.by_year.map((y: any) => y.year) }, yAxis: { type: "value", minInterval: 1 },
                        series: [{ type: "bar", data: r.by_year.map((y: any) => y.n), itemStyle: { color: C.accent }, barMaxWidth: 14 }] }} />
                    </Section>
                  </div>

                  <Section title="Compare with another event" sub="overlaid on the path chart, on the same symbol and period" open={overlays.length > 0}
                    right={<select className="field !h-7 !w-56 !text-xs" value="" onChange={(e) => { const o = events.find((x) => x.id === e.target.value); if (o) setOverlays((s) => [...s, { id: o.id, ...defaults(o) }].slice(0, 4)); }} aria-label="Add an event to compare">
                      <option value="">Add an event…</option>{events.filter((e) => e.id !== "custom").map((e) => <option key={e.id} value={e.id}>{e.title}</option>)}</select>}>
                    {overlays.length === 0 && <div className="text-xs dim">Add up to four events to see their median paths against this one, and how many dates they share.</div>}
                    <div className="flex flex-col gap-2">
                      {overlays.map((o, i) => {
                        const res = (r.overlays || []).find((x: any) => x.event?.id === o.id);
                        return (
                          <div key={`${o.id}${i}`} className="card-2 flex items-center gap-3 px-3 py-2 text-[13px]">
                            <GitCompareArrows className="size-4 dim" /><span className="flex-1">{res?.event?.text || events.find((e) => e.id === o.id)?.title}</span>
                            {res?.error ? <span className="text-xs text-down">{res.error}</span> : res ? <span className="text-xs dim">{res.n} occurrences · {res.overlap_with_main} shared with the main event</span> : <span className="text-xs dim">press Explore to add it</span>}
                            <button className="btn quiet sm !px-1.5" onClick={() => setOverlays((s) => s.filter((_, k) => k !== i))} aria-label="Remove comparison"><X /></button>
                          </div>
                        );
                      })}
                    </div>
                    {overlays.length > 0 && <button className="btn sm mt-3" onClick={() => run.mutate()} disabled={run.isPending}><Plus />Update comparison</button>}
                  </Section>

                  {(r.notes || []).length > 0 && <Callout tone="info" title="How to read this">{r.notes.map((n: string, i: number) => <div key={i} className={cn(i > 0 && "mt-1.5")}>{n}</div>)}</Callout>}

                  <div className="card flex flex-wrap items-center gap-3 px-4 py-3.5">
                    <Layers className="size-5 text-accent" />
                    <div className="min-w-[260px] flex-1">
                      <div className="text-[13.5px] font-medium">Take it further in a notebook</div>
                      <div className="text-xs dim">A price pattern is not a strategy. The notebook tests it against a matched baseline and a held-back period first; an options strategy can then be built from the finding and priced on observed option trades.</div>
                    </div>
                    <button className="btn primary" disabled={opening} onClick={() => goAsk(question)} data-testid="pattern-investigate">{opening ? <Loader2 className="animate-spin" /> : <Sparkles />}Investigate this pattern</button>
                  </div>
                </>
              )}
            </div>
          )}
          {run.isError && !r && <Callout tone="bad" title="This event could not be evaluated">{(run.error as any)?.message}</Callout>}
        </div>
      </div>
    </div>
  );
}
