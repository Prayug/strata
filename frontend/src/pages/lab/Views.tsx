/* Workspace views of multi-run results: a set of variants, an investigation, an exploration of many configurations. */
import { Check, FlaskConical, Pin, X } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import { C, EChart, fmtMoneyAxis } from "@/components/charts/EChart";
import { Tip } from "@/components/ui";
import { cn } from "@/lib/format";
import { pp, sgn, usd } from "@/lib/lab";
import { Callout, ModeBadge, Section, StatTile, T, Term } from "./bits";

const short = (l: string) => String(l).replace(/^v\d+\s*-?\s*/, "");
const pv = (p: number | null | undefined) => (p === null || p === undefined ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));

function VariantChart({ table, onOpen }: { table: any; onOpen: (id: string) => void }) {
  const study = table.kind === "study";
  const rows: any[] = table.rows;
  const option = useMemo(() => {
    const full = rows.map((r) => short(r.label));
    // the words every label shares become the axis title, so the ticks only show what differs
    const words = full.map((n) => n.split(" "));
    let k = 0;
    while (words.length > 1 && k < words[0].length - 1 && words.every((w) => w.length > k + 1 && w[k] === words[0][k])) k++;
    if (k < 3) k = 0; // two shared words ("after a") are not worth an axis title
    const names = full.map((n) => n.split(" ").slice(k).join(" "));
    const shared = k ? words[0].slice(0, k).join(" ") : "";
    const xAxis = { type: "category", data: names, name: shared, nameLocation: "middle", nameGap: 28, nameTextStyle: { color: C.muted, fontSize: 11 }, axisLabel: { interval: 0, width: 130, overflow: "truncate" } };
    if (study) {
      const base = rows.map((r) => r.values.baseline_probability);
      return {
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: any[]) => { const r = rows[ps[0].dataIndex]; return `${full[ps[0].dataIndex]}<br/>${r.shown.probability} of ${r.shown.qualifying_sessions} sessions<br/>95% interval ${r.shown.probability_ci_low} to ${r.shown.probability_ci_high}<br/>baseline ${r.shown.baseline_probability}`; } },
        legend: { data: ["after the condition", "all sessions (baseline)"] },
        grid: { left: 48, right: 14, top: 30, bottom: shared ? 46 : 30 },
        xAxis,
        yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
        series: [
          { name: "after the condition", type: "bar", barMaxWidth: 46, itemStyle: { color: C.accent, borderRadius: [3, 3, 0, 0] }, data: rows.map((r) => r.values.probability) },
          { name: "all sessions (baseline)", type: "line", symbol: "diamond", symbolSize: 8, lineStyle: { width: 1, type: "dashed" }, color: C.muted, data: base },
          // 95% interval whiskers
          { type: "custom", name: "95% interval", silent: true, data: rows.map((r, i) => [i, r.values.probability_ci_low, r.values.probability_ci_high]),
            renderItem: (_: any, api: any) => {
              const lo = api.coord([api.value(0), api.value(1)]), hi = api.coord([api.value(0), api.value(2)]);
              if ([lo[1], hi[1]].some((v) => !Number.isFinite(v))) return null;
              const st = { stroke: C.fg, lineWidth: 1.2 };
              return { type: "group", children: [{ type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: st }, { type: "line", shape: { x1: lo[0] - 5, y1: lo[1], x2: lo[0] + 5, y2: lo[1] }, style: st }, { type: "line", shape: { x1: hi[0] - 5, y1: hi[1], x2: hi[0] + 5, y2: hi[1] }, style: st }] };
            } },
        ],
      };
    }
    return {
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: any[]) => { const r = rows[ps[0].dataIndex]; return `${full[ps[0].dataIndex]}<br/>net ${r.shown.net_pnl} over ${r.shown.trades} trades<br/>${r.shown.win_rate} winners, ${r.shown.expectancy} per trade`; } },
      grid: { left: 56, right: 14, top: 16, bottom: shared ? 46 : 30 },
      xAxis,
      yAxis: { type: "value", axisLabel: { formatter: fmtMoneyAxis } },
      series: [{ type: "bar", barMaxWidth: 46, data: rows.map((r) => ({ value: r.values.net_pnl, itemStyle: { color: (r.values.net_pnl || 0) >= 0 ? C.up : C.down, borderRadius: 3 } })) }],
    };
  }, [table]); // eslint-disable-line
  return <EChart option={option} height={270} onClick={(p) => rows[p.dataIndex] && onOpen(rows[p.dataIndex].run_id)} />;
}

function VariantTable({ table, onOpen, onPin, pinned }: { table: any; onOpen: (id: string) => void; onPin: (id: string) => void; pinned: string[] }) {
  return (
    <div className="card overflow-x-auto">
      <table className="t">
        <thead><tr><th>Variant</th>{table.columns.map((c: any) => <th key={c.key} className="r">{c.label}</th>)}<th /></tr></thead>
        <tbody>
          {table.rows.map((r: any) => {
            const n = r.values.qualifying_sessions ?? r.values.trades ?? 0;
            return (
              <tr key={r.run_id} className="click" onClick={() => onOpen(r.run_id)} title="Open this variant's full report">
                <td>{short(r.label)}{n < 30 && <Tip content="Fewer than 30 observations: indicative only."><span className="ml-2 text-[11px] text-warn">small sample</span></Tip>}</td>
                {table.columns.map((c: any) => <td key={c.key} className="r num">{r.shown[c.key]}</td>)}
                <td className="r"><button className={cn("btn quiet sm !px-1.5", pinned.includes(r.run_id) && "!text-accent")} title="Pin to the comparison" aria-label="Pin" onClick={(e) => { e.stopPropagation(); onPin(r.run_id); }}><Pin /></button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function SweepView({ c, onOpen, onPin, pinned }: { c: any; onOpen: (id: string) => void; onPin: (id: string) => void; pinned: string[] }) {
  const table = c.table;
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="view-sweep">
      <div className="flex items-center gap-2"><h2>{c.what ? `Comparing ${c.what}` : "Variants compared"}</h2><ModeBadge mode={c.mode} /></div>
      <div className="text-[14px] soft"><T>{c.text}</T></div>
      {c.compatibility?.warning && <Callout tone="warn" title="Not like for like">{c.compatibility.warning}</Callout>}
      {table?.rows?.length > 0 && <><div className="card px-3 py-3"><VariantChart table={table} onOpen={onOpen} /></div><VariantTable table={table} onOpen={onOpen} onPin={onPin} pinned={pinned} /></>}
      {(c.failed || []).length > 0 && <Callout tone="warn" title="Not completed">{c.failed.map((f: any) => `${short(f.label)}${f.reason ? `: ${f.reason}` : ""}`).join("; ")}</Callout>}
      <div className="text-xs dim">Click a bar or a row to open that variant's report. The more variants are tried, the more likely the best one looks good by chance.</div>
    </div>
  );
}

const LEVEL: Record<string, [string, string]> = { supported: ["holds up", "text-up border-up/40 bg-up/10"], mixed: ["mixed evidence", "text-warn border-warn/40 bg-warn/10"], none: ["no edge found", "text-fg-2 border-line-2 bg-bg-2"], insufficient: ["too few observations", "text-warn border-warn/40 bg-warn/10"] };

/** Two directions and how often they agreed. One strip per candle length on a shared percent axis: a filled marker for the
    whole session, a ring for the rest of it, each with its 95% interval, and a dashed line where unrelated directions would
    sit. A marker whose interval reaches the dashed line is not distinguishable from it. */
function Agreement({ rows, onOpen }: { rows: any[]; onOpen: (id: string) => void }) {
  const mins = Array.from(new Set(rows.map((r) => r.minutes as number))).sort((a, b) => a - b);
  const nums = rows.flatMap((r) => [r.ci95?.[0], r.ci95?.[1], r.if_unrelated]).filter((v) => typeof v === "number") as number[];
  const lo = Math.min(0.45, Math.floor(Math.min(...nums) * 20) / 20), hi = Math.max(0.6, Math.ceil(Math.max(...nums) * 20) / 20);
  const x = (v: number) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%`;
  const ticks: number[] = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.05) ticks.push(Math.round(t * 100) / 100);
  const has = (scope: string) => rows.some((r) => r.scope === scope);
  const grid = { gridTemplateColumns: `104px minmax(180px, 1fr)${has("day") ? " 132px" : ""}${has("rest") ? " 132px" : ""}` };
  const cell = (r: any | undefined) => r && (
    <button className="group text-right leading-tight" onClick={() => onOpen(r.up.run_id)}
      title={`Up candle: this happened on ${pp(r.up.probability)} of ${r.up.n.toLocaleString()} sessions (${pp(r.up.all_sessions)} of all sessions; adjusted p ${pv(r.up.adjusted_p_value)}).\nDown candle: ${pp(r.down.probability)} of ${r.down.n.toLocaleString()} (${pp(r.down.all_sessions)} of all sessions; adjusted p ${pv(r.down.adjusted_p_value)}).\nOpens the up-candle study.`}>
      <div className={cn("num text-[15px] font-medium group-hover:underline", !r.distinguishable && "text-fg-2")}>{pp(r.agreement)}</div>
      <div className="num text-[11px] text-faint">{pp(r.ci95?.[0])} – {pp(r.ci95?.[1])}</div>
    </button>
  );
  return (
    <div className="card px-4 py-3.5" data-testid="agreement">
      <div className="eyebrow mb-1">How often the two directions matched</div>
      <div className="mb-2.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11.5px] dim">
        {has("day") && <span className="flex items-center gap-1.5"><span className="inline-block size-2.5 rounded-full bg-accent" />whole session, open to close (contains the candle)</span>}
        {has("rest") && <span className="flex items-center gap-1.5"><span className="inline-block size-2.5 rounded-full border-2 border-fg bg-bg-1" />rest of the session, after the candle</span>}
        <span className="flex items-center gap-1.5"><span className="inline-block h-3 border-l border-dashed border-fg-2" />if the directions were unrelated</span>
      </div>
      <div className="grid items-end gap-x-4 border-b border-line pb-1 text-[10.5px] uppercase tracking-[0.06em] text-muted" style={grid}>
        <span>First candle</span>
        <span className="relative h-4 normal-case tracking-normal">{ticks.map((t) => <span key={t} className="num absolute -translate-x-1/2 text-faint" style={{ left: x(t) }}>{(t * 100).toFixed(0)}%</span>)}</span>
        {has("day") && <span className="text-right">Whole session</span>}
        {has("rest") && <span className="text-right">Rest of session</span>}
      </div>
      {mins.map((m) => {
        const day = rows.find((r) => r.minutes === m && r.scope === "day"), rest = rows.find((r) => r.minutes === m && r.scope === "rest");
        const un = [day?.if_unrelated, rest?.if_unrelated].filter((v) => typeof v === "number") as number[];
        const any = day || rest;
        return (
          <div key={m} className="grid items-center gap-x-4 border-b border-line/60 py-1.5 last:border-b-0" style={grid} data-testid={`agreement-${m}`}>
            <div className="leading-tight"><div className="text-[13px] font-medium">{m} minute{m === 1 ? "" : "s"}</div><div className="num text-[11px] text-faint">{any.sessions.toLocaleString()} sessions</div></div>
            <div className="relative h-9">
              {ticks.map((t) => <span key={t} className="absolute inset-y-0 border-l border-line/50" style={{ left: x(t) }} />)}
              {un.length > 0 && <span className="absolute inset-y-0 border-l border-dashed border-fg-2" style={{ left: x(un.reduce((a, b) => a + b, 0) / un.length) }} />}
              {day && <>
                <span className="absolute top-[9px] h-[3px] -translate-y-1/2 rounded-full bg-accent/45" style={{ left: x(day.ci95[0]), right: `calc(100% - ${x(day.ci95[1])})` }} />
                <span className="absolute top-[9px] size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent" style={{ left: x(day.agreement) }} />
              </>}
              {rest && <>
                <span className="absolute top-[26px] h-[3px] -translate-y-1/2 rounded-full bg-fg/30" style={{ left: x(rest.ci95[0]), right: `calc(100% - ${x(rest.ci95[1])})` }} />
                <span className="absolute top-[26px] size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-fg bg-bg-1" style={{ left: x(rest.agreement) }} />
              </>}
            </div>
            {has("day") && <div className="text-right">{cell(day)}</div>}
            {has("rest") && <div className="text-right">{cell(rest)}</div>}
          </div>
        );
      })}
      <div className="mt-2 text-[11.5px] dim">A lighter figure is one where the two studies behind it (up candles, down candles) do not both differ from all other sessions once every comparison here is counted. Intervals are 95%. Hover a figure for the up and down halves.</div>
    </div>
  );
}

export function InvestigationView({ c, onOpen, onPin, pinned }: { c: any; onOpen: (id: string) => void; onPin: (id: string) => void; pinned: string[] }) {
  const rep = c.report || {};
  const plan = c.plan || {};
  const study = rep.kind === "study";
  const lv = LEVEL[rep.level] || LEVEL.none;
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="view-investigation">
      <div>
        <div className="eyebrow mb-1.5 flex items-center gap-2"><FlaskConical className="size-3.5 text-accent" />Investigation</div>
        <h2>{plan.hypothesis || "Findings"}</h2>
        {plan.method && <div className="mt-1.5 text-[13px] dim">{plan.method}</div>}
      </div>
      <div className="card px-4 py-3.5">
        <div className="flex items-start gap-3">
          <span className={cn("mt-0.5 inline-flex h-6 shrink-0 items-center rounded-full border px-2.5 text-xs font-medium", lv[1])}>{lv[0]}</span>
          <div className="text-[14.5px] font-medium leading-snug">{rep.verdict}</div>
        </div>
        <ul className="mt-3 flex flex-col gap-1.5 text-[13.5px] soft">{(rep.findings || []).map((f: string, i: number) => <li key={i} className="flex gap-2"><span className="text-faint">·</span><span><T>{f}</T></span></li>)}</ul>
        {rep.comparisons > 1 && <div className="mt-3 text-xs dim">{rep.comparisons} variants were compared{rep.trials > rep.comparisons ? `, ${rep.trials} configurations in this notebook so far` : ""}; p-values are adjusted for that.</div>}
      </div>
      {c.compatibility?.warning && <Callout tone="warn" title="Not like for like">{c.compatibility.warning}</Callout>}
      {(rep.agreement || []).length > 0 && <Agreement rows={rep.agreement} onOpen={onOpen} />}
      {(plan.interpretation_notes || []).length > 0 && (rep.agreement || []).length > 0 && (
        <ul className="-mt-1 flex flex-col gap-1 text-[12.5px] dim" data-testid="investigation-notes">{plan.interpretation_notes.map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
      )}
      {c.table?.rows?.length > 0 && !(rep.agreement || []).length && <div className="card px-3 py-3"><VariantChart table={c.table} onOpen={onOpen} /></div>}
      {(rep.rows || []).length > 0 && (
        <div className="card overflow-x-auto">
          <table className="t">
            <thead>{study
              ? <tr><th>Variant</th><th className="r">Sessions</th><th className="r">Frequency</th><th className="r"><Term k="confidence interval">95% interval</Term></th><th className="r"><Term k="baseline">Baseline</Term></th><th className="r">Difference</th><th className="r"><Term k="p-value">Adjusted p</Term></th><th className="r">Avg end return</th><th className="r"><Term k="out of sample">Recent 30%</Term></th><th className="r">Years above</th><th /></tr>
              : <tr><th>Variant</th><th>Priced on</th><th className="r">Qualifying</th><th className="r">Traded</th><th className="r">Net P&L</th><th className="r">Win rate</th><th className="r">Per trade</th><th className="r">Max drawdown</th><th /></tr>}</thead>
            <tbody>
              {rep.rows.map((r: any) => (
                <tr key={r.run_id} className="click" onClick={() => onOpen(r.run_id)}>
                  <td>{short(r.label)}</td>
                  {study ? (
                    <>
                      <td className={cn("r num", r.n < 30 && "text-warn")}>{r.n}</td><td className="r num">{pp(r.probability)}</td><td className="r num dim">{pp(r.ci95?.[0])} – {pp(r.ci95?.[1])}</td><td className="r num">{pp(r.baseline)}</td>
                      <td className={cn("r num", (r.lift || 0) > 0 ? "text-up" : (r.lift || 0) < 0 ? "text-down" : "")}>{r.lift === null || r.lift === undefined ? "—" : `${r.lift > 0 ? "+" : ""}${(r.lift * 100).toFixed(1)} pts`}</td>
                      <td className={cn("r num", r.adjusted_p_value !== null && r.adjusted_p_value < 0.05 ? "text-fg" : "dim")}>{pv(r.adjusted_p_value)}</td><td className="r num">{sgn(r.avg_end_return_pct, 2, "%")}</td>
                      <td className="r num dim">{r.oos_n ? `${pp(r.oos_probability)} of ${r.oos_n}` : "—"}</td><td className="r num dim">{r.years_above ? `${r.years_above[0]} of ${r.years_above[1]}` : "—"}</td>
                    </>
                  ) : (
                    <>
                      <td className={cn(r.pricing === "modeled" ? "text-warn" : "soft")}>{({ historical: "bid/ask quotes", trades: "trade prints", modeled: "THEORETICAL" } as any)[r.pricing] || r.pricing || "—"}</td>
                      <td className="r num">{r.qualifying ?? "—"}</td><td className={cn("r num", (r.trades || 0) < 30 && "text-warn")}>{r.trades ?? 0}</td><td className={cn("r num", (r.net_pnl || 0) > 0 ? "text-up" : (r.net_pnl || 0) < 0 ? "text-down" : "")}>{usd(r.net_pnl, 2)}</td>
                      <td className="r num">{pp(r.win_rate)}</td><td className="r num">{usd(r.expectancy, 2)}</td><td className="r num">{pp(r.max_drawdown)}</td>
                    </>
                  )}
                  <td className="r"><button className={cn("btn quiet sm !px-1.5", pinned.includes(r.run_id) && "!text-accent")} aria-label="Pin" title="Pin to the comparison" onClick={(e) => { e.stopPropagation(); onPin(r.run_id); }}><Pin /></button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {(rep.limitations || []).length > 0 && <Section title="What this does not tell you"><ul className="flex flex-col gap-1.5 text-[13px] soft">{rep.limitations.map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul></Section>}
    </div>
  );
}

const DSTATUS: Record<string, string> = { "confirmed on untouched data": "text-up border-up/40 bg-up/10", "candidate - untested": "text-warn border-warn/40 bg-warn/10", "not confirmed": "text-down border-down/40 bg-down/10", exploratory: "text-fg-2 border-line-2 bg-bg-2" };

export function DiscoveryView({ c }: { c: any }) {
  const d = c.discovery || {};
  const [show, setShow] = useState<"all" | "candidates" | "confirmed">("all");
  const [sort, setSort] = useState<[string, 1 | -1]>(["q", 1]);
  const [open, setOpen] = useState<string | null>(null);
  const configs: any[] = d.configs || [];
  const rows = useMemo(() => {
    const val = (x: any) => ({ q: x.q_value ?? 9, n: x.explore?.n ?? -1, p: x.explore?.probability ?? -1, lift: x.explore?.lift ?? -9, stab: x.stability ?? -1, label: x.label }[sort[0]] as any);
    return configs.filter((x) => (show === "all" ? true : show === "candidates" ? d.candidates.includes(x.key) : d.confirmed.includes(x.key)))
      .sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort[1]);
  }, [configs, show, sort]); // eslint-disable-line
  const scatter = useMemo(() => {
    const ok = configs.filter((x) => x.status === "ok" && x.explore?.lift !== null && x.explore?.lift !== undefined);
    const grp = (f: (x: any) => boolean, name: string, color: string, size: number) => ({ name, type: "scatter", symbolSize: size, itemStyle: { color, opacity: 0.85 }, data: ok.filter(f).map((x) => [x.explore.n, x.explore.lift * 100, x.label, x.q_value]) });
    return {
      tooltip: { formatter: (p: any) => `${p.value[2]}<br/>${p.value[0]} sessions, ${p.value[1] > 0 ? "+" : ""}${p.value[1].toFixed(1)} pts vs baseline<br/>adjusted p ${pv(p.value[3])}` },
      legend: {},
      grid: { left: 50, right: 16, top: 30, bottom: 40 },
      xAxis: { type: "log", name: "qualifying sessions in the explore period", nameLocation: "middle", nameGap: 26, min: 1 },
      yAxis: { type: "value", name: "difference from baseline (points)", axisLabel: { formatter: (v: number) => `${v > 0 ? "+" : ""}${v}` } },
      series: [
        grp((x) => !d.candidates.includes(x.key), "explored", C.faint, 7),
        grp((x) => d.candidates.includes(x.key) && !d.confirmed.includes(x.key), "candidate", C.accent, 11),
        grp((x) => d.confirmed.includes(x.key), "confirmed on untouched data", C.up, 13),
      ],
    };
  }, [configs]); // eslint-disable-line
  const th = (k: string, l: React.ReactNode, cls = "r") => <th className={cn(cls, "cursor-pointer select-none")} onClick={() => setSort(([a, dir]) => (a === k ? [k, dir === 1 ? -1 : 1] : [k, k === "q" || k === "label" ? 1 : -1]))}>{l}{sort[0] === k && <span className="ml-1 text-faint">{sort[1] === 1 ? "↑" : "↓"}</span>}</th>;
  const sp = d.splits || {};
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="view-discovery">
      <div>
        <div className="eyebrow mb-1.5 flex items-center gap-2"><FlaskConical className="size-3.5 text-synth" />Exploration</div>
        <h2>{d.hypothesis || d.question}</h2>
        <div className="mt-2 text-[14px] soft"><T>{c.text}</T></div>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Configurations tested" value={d.tested} sub={`${d.evaluable} had enough data to evaluate`} hint="Every one is listed below, including the ones that showed nothing." />
        <StatTile label="Candidates" value={d.candidates?.length ?? 0} sub="passed every screen on the explore and validation periods" />
        <StatTile label="Confirmed" value={d.confirmed?.length ?? 0} tone={d.confirmed?.length ? "text-up" : undefined} sub="held on the untouched final period" />
        <StatTile label="Final test period" value={d.test_period_used ? "used once" : "still sealed"} sub={sp.test ? `${sp.test[0]} to ${sp.test[1]}` : ""} hint="The most recent 20% of history is looked at only once, only for the candidates. If nothing qualified, it stays untouched." />
      </div>
      {sp.explore && (
        <div className="card px-4 py-3">
          <div className="mb-2 text-xs dim">How the history was split, so that the search cannot grade its own homework</div>
          <div className="flex h-7 overflow-hidden rounded-md text-[11px] font-medium">
            <div className="flex items-center justify-center bg-info/20 text-info" style={{ width: "60%" }}>explore · {sp.explore[0].slice(0, 4)}–{sp.explore[1].slice(0, 4)}</div>
            <div className="flex items-center justify-center bg-synth/20 text-synth" style={{ width: "20%" }}>validate</div>
            <div className={cn("flex items-center justify-center", d.test_period_used ? "bg-up/20 text-up" : "bg-bg-3 text-faint")} style={{ width: "20%" }}>{d.test_period_used ? "final test" : "sealed"}</div>
          </div>
          <div className="mt-1.5 flex justify-between num text-[10.5px] text-faint"><span>{sp.explore[0]}</span><span>{sp.validate?.[0]}</span><span>{sp.test?.[0]}</span><span>{sp.test?.[1]}</span></div>
        </div>
      )}
      <Section title="Every configuration that was tested" sub="size of the effect against how many sessions support it">
        <EChart option={scatter} height={280} />
        <div className="mt-1 text-xs dim">Large differences on very few sessions (top left) are what chance looks like. Candidates need a meaningful sample, an effect that survives adjusting for the number of tests, neighbours that agree, and the same direction in the validation period.</div>
      </Section>
      <div className="flex flex-wrap items-center gap-1.5">
        {([["all", `All ${configs.length}`], ["candidates", `Candidates ${d.candidates?.length ?? 0}`], ["confirmed", `Confirmed ${d.confirmed?.length ?? 0}`]] as const).map(([k, l]) => <button key={k} className={cn("chip", show === k && "on")} onClick={() => setShow(k)}>{l}</button>)}
      </div>
      <div className="card overflow-x-auto">
        <table className="t">
          <thead><tr>{th("label", "Configuration", "")}<th>Status</th>{th("n", "Sessions")}{th("p", "Frequency")}<th className="r">Baseline</th>{th("lift", "Difference")}{th("q", <Term k="p-value">Adjusted p</Term>)}{th("stab", "Neighbours agree")}<th className="r">Validation</th><th className="r">Final test</th></tr></thead>
          <tbody>
            {rows.map((x) => {
              const e = x.explore || {}, v = x.validate || {}, t = x.test;
              return (
                <Fragment key={x.key}>
                  <tr className="click" onClick={() => setOpen((o) => (o === x.key ? null : x.key))}>
                    <td>{x.label}</td>
                    <td>{x.status !== "ok" ? <span className="text-xs text-warn">{x.error || "could not be evaluated"}</span> : <span className={cn("inline-flex h-5 items-center rounded-full border px-2 text-[11px] whitespace-nowrap", DSTATUS[x.status_label] || DSTATUS.exploratory)}>{x.status_label}</span>}</td>
                    <td className={cn("r num", (e.n || 0) < 30 && "text-warn")}>{e.n ?? "—"}</td><td className="r num">{pp(e.probability)}</td><td className="r num dim">{pp(e.baseline_probability)}</td>
                    <td className={cn("r num", (e.lift || 0) > 0 ? "text-up" : (e.lift || 0) < 0 ? "text-down" : "")}>{e.lift === null || e.lift === undefined ? "—" : `${e.lift > 0 ? "+" : ""}${(e.lift * 100).toFixed(1)} pts`}</td>
                    <td className={cn("r num", x.q_value !== null && x.q_value !== undefined && x.q_value < 0.05 ? "text-fg" : "dim")}>{pv(x.q_value)}</td>
                    <td className="r num dim">{x.stability === null || x.stability === undefined ? "—" : pp(x.stability, 0)}</td>
                    <td className="r num dim">{v.n ? `${pp(v.probability)} of ${v.n}` : "—"}</td>
                    <td className="r num">{t ? `${pp(t.probability)} of ${t.n}` : <span className="text-faint">not looked at</span>}</td>
                  </tr>
                  {open === x.key && (
                    <tr><td colSpan={10} className="!bg-bg-2/60 !py-3">
                      <div className="flex flex-col gap-1.5 text-[13px]">
                        {(x.rejected_because || []).length > 0 ? x.rejected_because.map((r: string, i: number) => <div key={i} className="flex gap-2"><X className="mt-0.5 size-3.5 shrink-0 text-down" /><span className="soft">{r}</span></div>)
                          : <div className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-up" /><span className="soft">Passed every screen on the explore and validation periods.</span></div>}
                        {x.test_note && <div className="flex gap-2"><span className="mt-0.5 size-3.5 shrink-0 text-center text-faint">→</span><span className="soft">{x.test_note}</span></div>}
                        <div className="dim text-xs">Qualifying sessions ended {sgn(e.avg_end_return_pct, 2, "%")} on average versus {sgn(e.other_sessions_avg_end_return_pct, 2, "%")} on other sessions{e.end_return_p_value !== null && e.end_return_p_value !== undefined ? ` (p = ${pv(e.end_return_p_value)})` : ""}. A higher hit rate with no difference in where sessions end usually just means bigger swings in both directions.</div>
                      </div>
                    </td></tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      {(c.caveats || []).length > 0 && <Section title="How to read this" open={false}><ul className="flex flex-col gap-1.5 text-[13px] soft">{c.caveats.map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul></Section>}
    </div>
  );
}
