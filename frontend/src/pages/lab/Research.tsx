/* Workspace views of V4 research results: a finished hypothesis plan (every configuration, not only the winners),
   the adversarial review, the execution stress ladder and a reasoning answer. Status is always carried by a
   glyph and a word as well as a colour, so nothing depends on colour alone. */
import { useQuery } from "@tanstack/react-query";
import { CircleSlash, FlaskConical, Layers, ListTree, Lock, LockOpen, ShieldAlert, ShieldCheck, Table2, TriangleAlert } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { Dialog, Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { pp, usd } from "@/lib/lab";
import { Callout, Empty, Section, Skeleton, StatTile, T } from "./bits";

export const pv = (p: number | null | undefined) => (p === null || p === undefined ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));
const pts = (x: number | null | undefined, dp = 1) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(dp)} pts`);

/* ----------------------------------------------------------------------------- generic table */
const PCT = /frequency|probability|baseline|win_rate|^difference$|^shift$|difference_from_baseline|other_frequency/;
const USD = /pnl|per_trade|expectancy|net_|^net$|change_in/;
export function fmtCell(col: string, v: any): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  if (/p_value|p_adjusted/.test(col)) return pv(v);
  if (/shift|^difference$|difference_from_baseline/.test(col)) return pts(v);
  if (PCT.test(col)) return pp(v);
  if (USD.test(col)) return usd(v);
  if (/_pct$/.test(col)) return `${v.toFixed(2)}%`;
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
}
const head = (c: string) => c.replace(/_pct$/, " %").replace(/_/g, " ").replace(/^\w/, (m) => m.toUpperCase());

export function GTable({ table, onRow, max = 400 }: { table: { title?: string; columns: string[]; rows: any[] }; onRow?: (r: any) => void; max?: number }) {
  if (!table?.rows?.length) return null;
  return (
    <div>
      {table.title && <div className="mb-1.5 text-xs dim">{table.title}</div>}
      <div className="max-h-[420px] overflow-auto rounded-lg border border-line">
        <table className="t">
          <thead><tr>{table.columns.map((c, i) => <th key={c} className={i ? "r" : ""}>{head(c)}</th>)}</tr></thead>
          <tbody>
            {table.rows.slice(0, max).map((r, i) => (
              <tr key={i} className={onRow ? "click" : ""} onClick={() => onRow?.(r)}>
                {table.columns.map((c, j) => <td key={c} className={cn(j ? "r num" : "max-w-[340px]", typeof r[c] === "number" && USD.test(c) && (r[c] > 0 ? "text-up" : r[c] < 0 ? "text-down" : ""))}>{fmtCell(c, r[c])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.rows.length > max && <div className="mt-1 text-xs dim">Showing the first {max} of {table.rows.length.toLocaleString()} rows.</div>}
    </div>
  );
}

/* ----------------------------------------------------------------------------- hypothesis plan result */
const STATUS: Record<string, { glyph: string; word: string; tone: string }> = {
  "confirmed on untouched data": { glyph: "◆", word: "Confirmed on untouched data", tone: "text-up" },
  "confirmed on the final period (not tracked)": { glyph: "◆", word: "Confirmed on the final period", tone: "text-up" },
  "not confirmed": { glyph: "◇", word: "Candidate, not confirmed", tone: "text-warn" },
  "candidate - untested": { glyph: "◇", word: "Candidate, too few final observations", tone: "text-warn" },
  "candidate - no untouched data": { glyph: "◇", word: "Candidate, no untouched data left", tone: "text-warn" },
  exploratory: { glyph: "○", word: "Did not qualify", tone: "text-muted" },
};
const statusOf = (r: any) => (r.status !== "ok" ? { glyph: "–", word: "Could not be evaluated", tone: "text-faint" } : STATUS[r.status_label] || STATUS.exploratory);
const HOLD: Record<string, { word: string; tone: string; Icon: any }> = {
  sealed: { word: "untouched", tone: "text-up border-up/40", Icon: Lock }, used: { word: "used once", tone: "text-info border-info/40", Icon: LockOpen },
  exposed: { word: "already seen by earlier runs", tone: "text-warn border-warn/40", Icon: TriangleAlert }, contaminated: { word: "contaminated", tone: "text-down border-down/40", Icon: CircleSlash },
};

export function HoldoutChip({ h, compact }: { h: any; compact?: boolean }) {
  if (!h) return null;
  const s = HOLD[h.status] || HOLD.used;
  return (
    <Tip content={`Final test period ${h.start} to ${h.end}: ${s.word}.${h.reason ? ` ${h.reason}.` : ""}${h.status === "contaminated" ? " A new holdout is required before anything can be called confirmed." : ""}`}>
      <span className={cn("chip !h-6", s.tone)} data-testid="holdout-chip"><s.Icon className="size-3.5" />{compact ? `Final period: ${s.word}` : `Final test period ${h.start} → ${h.end}: ${s.word}`}</span>
    </Tip>
  );
}

function Periods({ res }: { res: any }) {
  const bases = Object.entries(res.periods || {}).filter(([, p]) => p) as [string, any][];
  if (!bases.length) return null;
  return (
    <div className="flex flex-col gap-3">
      {bases.map(([b, p]) => {
        const tot = p.train_sessions + p.validate_sessions + p.test_sessions || 1;
        const h = res.holdouts?.[b];
        const segs = [["Training", p.train, p.train_sessions, "bg-info/25", "selection happens here"], ["Validation", p.validate, p.validate_sessions, "bg-accent/25", "can only veto"], ["Final test", p.test, p.test_sessions, "bg-up/20", "scored once, for candidates"]] as const;
        return (
          <div key={b}>
            <div className="mb-1.5 flex items-center gap-2 text-xs dim">
              <span className="font-medium text-fg-2">{b === "daily" ? "Rows answered from daily bars" : "Rows that need intraday bars"}</span>
              <span>{p.sessions.toLocaleString()} sessions</span>
              <span className="flex-1" />
              {h ? <HoldoutChip h={h} /> : <span className="chip !h-6">Final period not tracked across runs</span>}
            </div>
            <div className="flex h-12 w-full overflow-hidden rounded-lg border border-line">
              {segs.map(([name, span, n, tone, why]) => (
                <div key={name} className={cn("flex min-w-0 flex-col justify-center border-r border-line px-2.5 last:border-r-0", tone)} style={{ width: `${Math.max(12, (n / tot) * 100)}%` }}>
                  <div className="truncate text-xs font-medium">{name} <span className="num font-normal dim">{n.toLocaleString()}</span></div>
                  <div className="num truncate text-[11px] dim" title={why}>{span[0]} → {span[1]}</div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Forest({ rows, onPick, picked }: { rows: any[]; onPick: (k: string) => void; picked: string | null }) {
  const ok = rows.filter((r) => r.status === "ok" && r.train?.lift !== null && r.train?.lift !== undefined);
  const option = useMemo(() => {
    const y = ok.map((r) => r.key);
    const sym = (r: any) => (r.status_label?.startsWith("confirmed") ? "diamond" : r.candidate ? "emptyDiamond" : r.train.n < 30 ? "emptyCircle" : "circle");
    return {
      grid: { left: 8, right: 18, top: 26, bottom: 30, containLabel: false },
      tooltip: { trigger: "item", formatter: (p: any) => { const r = ok[p.dataIndex]; if (!r) return ""; const t = r.train; return `${r.label}<br/><span style="color:${C.muted}">${r.outcome_text}</span><br/>${pp(t.probability)} of ${t.n} vs ${pp(t.baseline_probability)} on all sessions<br/>difference ${pts(t.lift)}${t.lift_ci95?.[0] != null ? `, 95% interval ${pts(t.lift_ci95[0])} to ${pts(t.lift_ci95[1])}` : ""}<br/>${statusOf(r).word}`; } },
      xAxis: { type: "value", name: "difference from the baseline, percentage points (training period)", nameLocation: "middle", nameGap: 20, axisLabel: { formatter: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}` } },
      yAxis: { type: "category", data: y, inverse: true, axisLabel: { show: false }, axisLine: { show: false } },
      series: [
        { type: "custom", silent: true, data: ok.map((r, i) => [i, (r.train.lift_ci95?.[0] ?? r.train.lift) * 100, (r.train.lift_ci95?.[1] ?? r.train.lift) * 100]),
          renderItem: (_: any, a: any) => { const lo = a.coord([a.value(1), a.value(0)]), hi = a.coord([a.value(2), a.value(0)]); return { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: { stroke: C.faint, lineWidth: 1.5 } }; } },
        { type: "scatter", data: ok.map((r, i) => ({ value: [r.train.lift * 100, i], symbol: sym(r), symbolSize: r.candidate ? 12 : 8, itemStyle: { color: r.status_label?.startsWith("confirmed") ? C.up : r.candidate ? C.accent : C.muted, borderColor: r.key === picked ? C.fg : undefined, borderWidth: r.key === picked ? 2 : 1 } })),
          markLine: { silent: true, symbol: "none", lineStyle: { color: C.muted, type: "dashed" }, data: [{ xAxis: 0 }], label: { formatter: "baseline", color: C.muted, fontSize: 10 } } },
      ],
    };
  }, [ok, picked]);
  if (!ok.length) return null;
  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs dim">
        <span><span className="text-up">◆</span> confirmed</span><span><span className="text-accent">◇</span> candidate</span><span>● did not qualify</span><span>○ fewer than 30 observations</span><span>— 95% interval (weeks resampled)</span>
      </div>
      <div className="overflow-auto" style={{ maxHeight: 460 }}>
        <EChart option={option} height={Math.max(180, ok.length * 16 + 70)} onClick={(p) => { const r = ok[p.dataIndex]; if (r) onPick(r.key); }} />
      </div>
    </div>
  );
}

function BlockLine({ name, b }: { name: string; b: any }) {
  if (!b) return <div className="text-xs dim">{name}: not scored</div>;
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 text-[13px]">
      <span className="w-24 shrink-0 text-xs dim">{name}</span>
      <span className="num">{pp(b.probability)} of {b.n}</span>
      <span className="dim">all sessions <span className="num">{pp(b.baseline_probability)}</span></span>
      <span className={cn("num", (b.lift ?? 0) > 0 ? "text-up" : "text-down")}>{pts(b.lift)}</span>
      {b.p_value !== null && b.p_value !== undefined && <span className="dim">p <span className="num">{pv(b.p_value)}</span>{b.p_iid !== null && b.p_iid !== undefined && <Tip content="The same test if every session were an independent draw. The gap between the two is the cost of clustering."><span className="text-faint"> (independent: {pv(b.p_iid)})</span></Tip>}</span>}
      {b.avg_end_return_pct !== null && b.avg_end_return_pct !== undefined && <span className="dim">avg move <span className="num">{b.avg_end_return_pct.toFixed(2)}%</span></span>}
      {b.ambiguous > 0 && <span className="text-warn">{b.ambiguous} ambiguous</span>}
    </div>
  );
}

function Observations({ runId, row, onClose }: { runId: string; row: any; onClose: () => void }) {
  const q = useQuery({ queryKey: ["hyp-obs", runId, row.key], queryFn: () => api.get<any>(`/api/runs/${runId}/hypothesis/observations?key=${row.key}`) });
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={<span className="text-[14px]">{row.label}</span>} width={860}>
      <div className="text-xs dim mb-2">{row.outcome_text}</div>
      {q.isLoading && <Skeleton className="h-40" />}
      {q.data && (
        <>
          {q.data.hidden_test_period && <Callout tone="info" className="mb-3">Sessions of the final test period are not listed: this configuration was not scored on it, and looking at them would use it up.</Callout>}
          <GTable table={{ columns: q.data.columns, rows: [...q.data.rows].reverse() }} max={600} />
        </>
      )}
    </Dialog>
  );
}

export function HypothesisView({ runId, onSelect, busy }: { runId: string; onSelect: (key: string, what: "study" | "options") => void; busy?: boolean }) {
  const q = useQuery({ queryKey: ["hypothesis", runId], queryFn: () => api.get<any>(`/api/runs/${runId}/hypothesis`), staleTime: 60_000 });
  const [filter, setFilter] = useState<"all" | "candidates" | "enough" | "failed">("all");
  const [open, setOpen] = useState<string | null>(null);
  const [obs, setObs] = useState<any>(null);
  if (q.isLoading) return <div className="flex flex-col gap-3 p-5"><Skeleton className="h-8 w-2/3" /><Skeleton className="h-28" /><Skeleton className="h-64" /></div>;
  if (q.isError || !q.data) return <Empty title="This plan's result could not be opened">{String((q.error as any)?.message || "")}</Empty>;
  const res = q.data;
  const rows: any[] = res.configs;
  const minN = res.plan.min_sample;
  const shown = rows.filter((r) => filter === "all" || (filter === "candidates" && r.candidate) || (filter === "enough" && r.status === "ok" && r.train.n >= minN) || (filter === "failed" && r.status !== "ok"));
  const fam: [string, any[]][] = [];
  for (const r of shown) {
    const k = `${r.outcome_text}${r.tag ? `  ·  ${r.tag}` : ""}`;
    const g = fam.find((x) => x[0] === k);
    if (g) g[1].push(r); else fam.push([k, [r]]);
  }
  const cv = res.selection_cv || {};
  return (
    <div className="mx-auto flex max-w-[1180px] flex-col gap-4 p-5" data-testid="hypothesis-view">
      <div>
        <div className="eyebrow flex items-center gap-2"><FlaskConical className="size-3.5 text-accent" />Hypothesis · {res.symbol} · underlying behaviour</div>
        <h2 className="mt-1.5">{res.question}</h2>
        <div className="mt-2 grid gap-2 text-[13.5px] md:grid-cols-2">
          <div className="card px-3.5 py-2.5"><div className="text-xs dim">Hypothesis</div><div className="mt-0.5">{res.hypothesis}</div></div>
          <div className="card px-3.5 py-2.5"><div className="text-xs dim">What it is tested against</div><div className="mt-0.5 soft">{res.plan.null}</div><div className="mt-1 text-xs dim">Baseline: {res.plan.baseline}</div></div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatTile label="Configurations tested" value={res.tested} sub={`bound ${res.plan.max_configs}`} hint="Every configuration of the plan, listed below whatever it showed." />
        <StatTile label="Could be evaluated" value={res.evaluable} sub={res.tested - res.evaluable ? `${res.tested - res.evaluable} lack the data` : "all of them"} />
        <StatTile label="Candidates" value={res.candidates.length} sub="selected on training data" tone={res.candidates.length ? "text-accent" : undefined} />
        <StatTile label="Confirmed" value={res.confirmed.length} sub="on the final test period" tone={res.confirmed.length ? "text-up" : undefined} />
        <StatTile label="Corrected for" value={res.trials.session} sub="configurations tried in this notebook" hint="Significance is corrected (Holm) for every configuration tried in the research session, not only the ones in this plan." />
      </div>
      {res.contaminated && <Callout tone="bad" title="The final test period is contaminated">{res.contaminated.reason}. Nothing found from here on can be called confirmed until a new holdout exists.</Callout>}
      <Section title="Periods" sub="selection never sees the final test period">
        <Periods res={res} />
      </Section>
      <Section title="From plan to evidence" sub="how many configurations survived each requirement">
        <div className="flex flex-col gap-1.5">
          {res.funnel.map((f: any, i: number) => (
            <div key={i} className="flex items-center gap-3 text-[13px]">
              <span className="w-[46%] shrink-0 truncate soft">{f.step}</span>
              <div className="h-4 flex-1 overflow-hidden rounded bg-bg-3"><div className={cn("h-full", i === res.funnel.length - 1 ? "bg-up/60" : "bg-accent/45")} style={{ width: `${(f.n / Math.max(res.tested, 1)) * 100}%`, minWidth: f.n ? 3 : 0 }} /></div>
              <span className="num w-9 text-right">{f.n}</span>
            </div>
          ))}
        </div>
      </Section>
      <Section title="Every configuration, with its uncertainty" sub="sorted as planned, not by result">
        <Forest rows={rows} picked={open} onPick={(k) => { setOpen(k); document.getElementById(`cfg-${k}`)?.scrollIntoView({ block: "center", behavior: "smooth" }); }} />
      </Section>
      <Section title={`All ${res.tested} configurations`} right={
        <div className="flex gap-1">
          {([["all", `All ${rows.length}`], ["candidates", `Candidates ${rows.filter((r) => r.candidate).length}`], ["enough", `≥ ${minN} observations`], ["failed", `Not evaluable ${rows.filter((r) => r.status !== "ok").length}`]] as const).map(([k, l]) => (
            <button key={k} className={cn("chip", filter === k && "!border-accent/60 !text-fg")} onClick={() => setFilter(k)}>{l}</button>
          ))}
        </div>}>
        <div className="overflow-auto rounded-lg border border-line">
          <table className="t" data-testid="hypothesis-table">
            <thead><tr><th>Status</th><th>Condition</th><th className="r">Obs.</th><th className="r"><Tip content="How often the outcome happened on qualifying sessions, and (smaller) on all evaluated sessions."><span className="term">Frequency</span></Tip></th><th className="r">Difference</th><th className="r"><Tip content="p-value after correcting for every configuration tried in this notebook (Holm), from a test that resamples whole weeks."><span className="term">p, corrected</span></Tip></th><th>Validation</th><th>Final test</th></tr></thead>
            <tbody>
              {fam.map(([name, rs]) => (
                <Fragment key={name}>
                  <tr><td colSpan={8} className="!bg-bg-2/60 !py-1.5 text-xs dim"><ListTree className="mr-1.5 inline size-3.5" />Outcome: {name}</td></tr>
                  {rs.map((r) => {
                    const s = statusOf(r), t = r.train, v = r.validate, te = r.test;
                    return (
                      <Fragment key={r.key}>
                        <tr id={`cfg-${r.key}`} className={cn("click", open === r.key && "!bg-bg-2")} onClick={() => setOpen(open === r.key ? null : r.key)}>
                          <td className={cn("whitespace-nowrap", s.tone)}><span className="mr-1.5">{s.glyph}</span><span className="text-xs">{s.word}</span></td>
                          <td className="min-w-[200px] max-w-[330px] !whitespace-normal">{r.condition}</td>
                          {r.status === "ok" ? (
                            <>
                              <td className={cn("r num", t.n < minN && "text-warn")} title={t.n < minN ? `fewer than the ${minN} the plan requires` : undefined}>{t.n}{t.n < minN ? " ⚠" : ""}</td>
                              <td className="r num whitespace-nowrap">{pp(t.probability)} <span className="text-[11px] dim">vs {pp(t.baseline_probability)}</span></td>
                              <td className={cn("r num", (t.lift ?? 0) > 0 ? "text-up" : "text-down")}>{pts(t.lift)}</td><td className="r num">{pv(r.p_adjusted)}</td>
                              <td className="whitespace-nowrap text-xs">{v.n < 8 ? <span className="dim">{v.n} obs.</span> : ((v.lift ?? 0) > 0) === ((t.lift ?? 0) > 0) ? <span className="text-up">✓ agrees ({pts(v.lift)}, {v.n})</span> : <span className="text-down">✕ opposite ({pts(v.lift)}, {v.n})</span>}</td>
                              <td className="whitespace-nowrap text-xs">{te ? <span className={r.status_label.startsWith("confirmed") ? "text-up" : "text-warn"}>{pp(te.probability)} of {te.n} vs {pp(te.baseline_probability)}</span> : <span className="text-faint">not scored</span>}</td>
                            </>
                          ) : <td colSpan={6} className="text-xs dim">{r.error}</td>}
                        </tr>
                        {open === r.key && r.status === "ok" && (
                          <tr><td colSpan={8} className="!bg-bg-1 !p-0 !whitespace-normal">
                            <div className="sticky left-0 flex max-w-[min(680px,calc(100vw-120px))] flex-col gap-2 px-4 py-3">
                              <BlockLine name="Training" b={t} /><BlockLine name="Validation" b={v} /><BlockLine name="Final test" b={te} />
                              {r.test_note && <div className="text-xs dim">Final test: {r.test_note}.</div>}
                              {r.rejected_because?.length > 0 && <div className="text-[13px]"><span className="dim">Did not qualify because: </span>{r.rejected_because.join("; ")}.</div>}
                              {r.grade && <div className="text-[13px]"><span className="dim">Evidence: </span><span className="font-medium">{r.grade.label}</span>{r.grade.reasons?.length ? <span className="dim"> — {r.grade.reasons.join("; ")}</span> : null}</div>}
                              {r.options_cover && <div className="text-xs dim">Observed option prices for these sessions: {r.options_cover.quotes_usable ? `${r.options_cover.with_quotes} of ${r.options_cover.qualifying} have end-of-day bid/ask quotes` : "end-of-day quotes cannot price an entry that is not at the close"}; {r.options_cover.with_trades} have intraday trade prints.</div>}
                              <div className="mt-1 flex flex-wrap gap-2">
                                <button className="btn sm" disabled={busy || !r.study} onClick={() => onSelect(r.key, "study")} data-testid="select-study" title={r.study ? "Run this exact configuration session by session: full report, calendar and replay" : "A move in either direction has no single study"}><Layers />Investigate further</button>
                                <button className="btn sm" disabled={busy} onClick={() => onSelect(r.key, "options")} data-testid="select-options" title="Stage two: price option positions on these sessions with observed premiums"><FlaskConical />Test option structures</button>
                                <button className="btn quiet sm" onClick={() => setObs(r)} data-testid="see-sessions"><Table2 />See the sessions</button>
                              </div>
                            </div>
                          </td></tr>
                        )}
                      </Fragment>
                    );
                  })}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      {Object.keys(cv).length > 0 && (
        <Section title="Does choosing the best one carry over?" sub="purged, embargoed cross-validation of the selection itself" open={false}>
          {Object.entries(cv).map(([b, v]: [string, any]) => (
            <div key={b} className="mb-3 last:mb-0">
              <div className="text-[13.5px]">In {v.k} chronological folds, the configuration with the largest effect on the training sessions averaged <span className="num">{pts(v.mean_train)}</span> there and <span className="num">{pts(v.mean_test)}</span> on the held-out block
                {v.share_below_median !== null && <>; it ranked in the lower half out of sample in <span className="num">{Math.round(v.share_below_median * v.k)}</span> of {v.k} folds</>}. {v.distinct_choices} different configuration{v.distinct_choices !== 1 ? "s were" : " was"} chosen.</div>
              <div className="mt-1 text-xs dim">{v.note}</div>
            </div>
          ))}
        </Section>
      )}
      <Section title="How to read this" open={false}>
        <ul className="flex flex-col gap-1.5 text-[13px] soft">{res.notes.map((n: string, i: number) => <li key={i}>· <T>{n}</T></li>)}{(res.plan.notes || []).map((n: string, i: number) => <li key={`p${i}`}>· {n}</li>)}</ul>
      </Section>
      {obs && <Observations runId={runId} row={obs} onClose={() => setObs(null)} />}
    </div>
  );
}

/* ----------------------------------------------------------------------------- execution stress */
export function ExecutionTable({ stress }: { stress: any }) {
  const s = stress.summary;
  const rows: any[] = stress.rows;
  const ok = rows.filter((r) => r.status === "ok");
  const option = useMemo(() => ({
    grid: { left: 250, right: 60, top: 8, bottom: 26 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: (p: any[]) => { const r = ok[p[0].dataIndex]; return `${r.label}<br/><span style="color:${C.muted}">${r.assumption || ""}</span><br/>net ${usd(r.net_pnl)} · ${r.trades} trades${r.missed ? ` · ${r.missed} signals not filled` : ""}`; } },
    xAxis: { type: "value", axisLabel: { formatter: fmtMoneyAxis } },
    yAxis: { type: "category", inverse: true, data: ok.map((r) => r.label), axisLabel: { width: 236, overflow: "truncate", fontFamily: "Inter", fontSize: 11 } },
    series: [{ type: "bar", barMaxWidth: 14, data: ok.map((r) => ({ value: r.net_pnl, itemStyle: { color: r.id === "base" ? C.accent : r.id === "signal" ? C.info : (r.net_pnl ?? 0) >= 0 ? C.up : C.down, opacity: r.id === "base" || r.id === "signal" ? 1 : 0.75 } })),
      label: { show: true, position: "right", color: C.muted, fontSize: 10, formatter: (p: any) => `${ok[p.dataIndex].trades} tr.` }, markLine: { silent: true, symbol: "none", lineStyle: { color: C.muted }, data: [{ xAxis: 0 }], label: { show: false } } }],
  }), [ok]);
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Signal only, no costs" value={usd(s.signal_net_pnl)} sub="not achievable: the pattern on paper" tone="text-info" />
        <StatTile label="As simulated" value={usd(s.simulated_net_pnl)} sub={`${s.trades} trades`} tone={(s.simulated_net_pnl ?? 0) >= 0 ? "text-up" : "text-down"} />
        <StatTile label="Adverse combination" value={usd(s.adverse_net_pnl)} sub="several assumptions worse at once" tone={(s.adverse_net_pnl ?? 0) >= 0 ? "text-up" : "text-down"} />
        <StatTile label="Break-even slippage" value={s.breakeven_slippage_pct == null ? "—" : `${s.breakeven_slippage_pct.toFixed(1)}%`} sub={(s.simulated_net_pnl ?? 0) <= 0 ? "already negative as simulated" : s.breakeven_slippage_pct == null ? "positive at every level tested" : "per fill, against you"} />
      </div>
      {s.thin && <Callout tone="warn" title={`Only ${s.trades} trades`}>The differences between rows describe these trades. They are not estimates of what execution will cost.</Callout>}
      <EChart option={option} height={Math.max(160, ok.length * 22 + 40)} />
      <div className="overflow-auto rounded-lg border border-line">
        <table className="t" data-testid="execution-table">
          <thead><tr><th>Assumption</th><th>Scenario</th><th className="r">Trades</th><th className="r">Not filled</th><th className="r">Partial</th><th className="r">Net</th><th className="r">Per trade</th><th className="r">vs simulated</th></tr></thead>
          <tbody>
            {rows.map((r) => r.status === "ok" ? (
              <tr key={r.id} className={r.id === "base" ? "!bg-accent/[0.06]" : ""}>
                <td className="text-xs dim">{r.group}</td>
                <td><Tip content={r.assumption}><span className="term">{r.label}</span></Tip>{r.thin && r.id !== "base" && <span className="ml-2 text-[11px] text-warn">few trades</span>}</td>
                <td className="r num">{r.trades}</td><td className={cn("r num", r.missed > 0 && "text-warn")}>{r.missed ?? "—"}</td><td className="r num">{r.partial_fills || "—"}</td>
                <td className={cn("r num", (r.net_pnl ?? 0) >= 0 ? "text-up" : "text-down")}>{usd(r.net_pnl)}</td><td className="r num">{usd(r.expectancy, 2)}</td>
                <td className={cn("r num", (r.delta_net_pnl ?? 0) < 0 ? "text-down" : "dim")}>{r.id === "base" ? "—" : usd(r.delta_net_pnl)}</td>
              </tr>
            ) : (
              <tr key={r.id}><td className="text-xs dim">{r.group}</td><td className="dim">{r.label}</td><td colSpan={6} className="text-xs text-warn">{r.status === "not_testable" ? "Cannot be tested: " : "Failed: "}{r.note}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="flex flex-col gap-1.5 text-[13px] soft">{stress.notes.map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
    </div>
  );
}

export function ExecutionView({ runId }: { runId: string }) {
  const q = useQuery({ queryKey: ["execution", runId], queryFn: () => api.get<any>(`/api/runs/${runId}/execution`), staleTime: 60_000 });
  if (q.isLoading) return <div className="p-5"><Skeleton className="h-64" /></div>;
  if (!q.data) return <Empty title="The stress test has no result">{String((q.error as any)?.message || "")}</Empty>;
  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-4 p-5" data-testid="execution-view">
      <div><div className="eyebrow">Execution uncertainty</div><h2 className="mt-1.5">How the result changes as fills get worse</h2>
        <p className="mt-1 max-w-[760px] text-[13.5px] soft">The same rules, re-simulated by the same engine under an explicit ladder of execution assumptions. A trade print shows that someone traded at that price, not that this order would have been filled there.</p></div>
      <ExecutionTable stress={q.data} />
    </div>
  );
}

/* ----------------------------------------------------------------------------- adversarial review */
const SEV: Record<string, { glyph: string; word: string; tone: string; ring: string }> = {
  high: { glyph: "✕", word: "Serious", tone: "text-down", ring: "border-down/45 bg-down/[0.06]" }, medium: { glyph: "!", word: "Weakness", tone: "text-warn", ring: "border-warn/40 bg-warn/[0.05]" },
  low: { glyph: "·", word: "Minor", tone: "text-fg-2", ring: "border-line" }, ok: { glyph: "✓", word: "Checked, nothing found", tone: "text-up", ring: "border-line" }, na: { glyph: "–", word: "Not applicable / not checkable", tone: "text-faint", ring: "border-line border-dashed" },
};

export function ChallengeView({ runId }: { runId: string }) {
  const q = useQuery({ queryKey: ["challenge", runId], queryFn: () => api.get<any>(`/api/runs/${runId}/challenge`), staleTime: 60_000 });
  const [open, setOpen] = useState<string | null>(null);
  if (q.isLoading) return <div className="p-5"><Skeleton className="h-64" /></div>;
  if (!q.data) return <Empty title="The review has no result">{String((q.error as any)?.message || "")}</Empty>;
  const r = q.data;
  const bad = r.counts.high > 0;
  return (
    <div className="mx-auto flex max-w-[1000px] flex-col gap-4 p-5" data-testid="challenge-view">
      <div><div className="eyebrow flex items-center gap-2">{bad ? <ShieldAlert className="size-3.5 text-down" /> : <ShieldCheck className="size-3.5 text-up" />}Adversarial review · {r.subject === "study" ? "market study" : "option backtest"}</div>
        <h2 className="mt-1.5">How this result could be misleading</h2></div>
      <Callout tone={bad ? "bad" : r.counts.medium ? "warn" : "good"} title={r.verdict} />
      <div className="flex flex-wrap gap-2 text-xs">
        {(["high", "medium", "low", "ok", "na"] as const).map((k) => <span key={k} className={cn("chip !h-6", SEV[k].tone)}><span className="font-semibold">{SEV[k].glyph}</span>{r.counts[k]} {SEV[k].word.toLowerCase()}</span>)}
        {r.tests_run?.length > 0 && <span className="chip !h-6">Tests run: {r.tests_run.join(", ")}</span>}
      </div>
      <div className="flex flex-col gap-2">
        {r.findings.map((f: any) => {
          const s = SEV[f.severity];
          return (
            <div key={f.topic} className={cn("rounded-[10px] border px-4 py-3", s.ring)} data-testid={`finding-${f.topic}`} data-severity={f.severity}>
              <div className="flex items-baseline gap-2.5">
                <span className={cn("w-4 shrink-0 text-center text-[15px] font-semibold", s.tone)}>{s.glyph}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <span className="text-xs dim">{f.name}</span><span className={cn("text-[11px] font-medium uppercase tracking-wide", s.tone)}>{s.word}</span>
                    {f.test && <span className="chip !h-5 !text-[10.5px]">tested: {f.test}</span>}
                  </div>
                  <div className="mt-0.5 text-[14px] font-medium">{f.title}</div>
                  <div className="mt-1 text-[13px] soft"><T>{f.evidence}</T></div>
                  {f.table && <button className="mt-1.5 text-xs text-info hover:underline" onClick={() => setOpen(open === f.topic ? null : f.topic)}>{open === f.topic ? "Hide the numbers" : "Show the numbers"}</button>}
                  {f.table && open === f.topic && <div className="mt-2"><GTable table={f.table} /></div>}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {r.stress && <Section title="The execution stress ladder behind this review" open={false}><ExecutionTable stress={r.stress} /></Section>}
      <ul className="flex flex-col gap-1 text-xs dim">{r.notes.map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
    </div>
  );
}

/* ----------------------------------------------------------------------------- reasoning answer */
export function AnalysisView({ c }: { c: any }) {
  const a = c.analysis || {};
  return (
    <div className="mx-auto flex max-w-[980px] flex-col gap-4 p-5" data-testid="analysis-view">
      <div><div className="eyebrow">Reasoning · calculated from the stored observations</div><h2 className="mt-1.5">{a.title}</h2></div>
      {!a.answerable && <Callout tone="warn" title="The stored data cannot fully answer this" />}
      <div className="card px-4 py-3.5 text-[14px] leading-relaxed"><T>{a.answer}</T></div>
      {a.facts?.length > 0 && <div className="grid grid-cols-2 gap-3 md:grid-cols-4">{a.facts.map((f: any) => <StatTile key={f.label} label={f.label} value={f.value} />)}</div>}
      {(a.tables || []).map((t: any, i: number) => <GTable key={i} table={t} />)}
      {a.limits?.length > 0 && <Callout tone="info" title="Limits of this answer"><ul className="mt-1 flex flex-col gap-1">{a.limits.map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul></Callout>}
    </div>
  );
}
