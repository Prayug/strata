/* Research graphs in the notebook: the plan as it was read (steps, assumptions, anything not understood), the
   full result of every analysis, and the memory / registry / strategy answers built on them. Evidence grades and
   statuses always carry a glyph and a word; colour is never the only signal. */
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { Background, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { Boxes, FlaskConical, GitFork, History, Lock, Play, Route, Scale, ShieldCheck, Snowflake, Workflow } from "lucide-react";
import { memo, useMemo, useState } from "react";
import { C, EChart } from "@/components/charts/EChart";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { pp } from "@/lib/lab";
import { Callout, Empty, Section, Skeleton, StatTile, T } from "./bits";
import { GTable, pv } from "./Research";

export const GRADE: Record<string, { glyph: string; tone: string; help: string }> = {
  confirmed: { glyph: "◆", tone: "text-up border-up/40 bg-up/[0.08]", help: "Held on a final test period that nothing had examined, here or elsewhere in Strata." },
  "holds, not independent": { glyph: "◈", tone: "text-warn border-warn/40 bg-warn/[0.08]", help: "Held on this notebook's final test period, but related configurations elsewhere had already examined those dates." },
  supported: { glyph: "●", tone: "text-info border-info/40 bg-info/[0.08]", help: "Distinguishable from the baseline with no open weakness, but not validated on a final test period." },
  suggestive: { glyph: "◐", tone: "text-fg-2 border-line-2 bg-bg-2", help: "Distinguishable on its own test, with an open weakness." },
  "holds on seen data": { glyph: "◇", tone: "text-warn border-warn/40 bg-warn/[0.08]", help: "Held on a final test period that had already been looked at, so it is not a confirmation." },
  "not confirmed": { glyph: "✕", tone: "text-down border-down/40 bg-down/[0.07]", help: "A candidate that failed on the final test period." },
  "not supported": { glyph: "○", tone: "text-muted border-line-2 bg-bg-2", help: "Within what chance produces." },
  insufficient: { glyph: "–", tone: "text-muted border-line-2 bg-bg-2", help: "Too few observations to say anything." },
};

export function GradeBadge({ g, className }: { g: string; className?: string }) {
  const x = GRADE[g] || GRADE.insufficient;
  return <span title={x.help} className={cn("inline-flex h-[22px] shrink-0 items-center gap-1.5 rounded-full border px-2 text-[11.5px] font-medium whitespace-nowrap", x.tone, className)}><span aria-hidden>{x.glyph}</span>{g}</span>;
}

const sp = (x: number | null | undefined, dp = 1) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(dp)} pts`);
const sr = (x: number | null | undefined, dp = 2) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(dp)}%`);
const TICK: Record<string, string> = { open: "known from the open", close: "known at the close", event: "at the event" };
const tick = (t?: string | null) => (t ? TICK[t] || `known at ${t} ET` : "");

/* ----------------------------------------------------------------------------- price paths (shared with the pattern explorer) */
export function PathChart({ paths, height = 320, onPick, overlays = [] }: { paths: any; height?: number; onPick?: (date: string) => void; overlays?: { name: string; steps: string[]; bands: any }[] }) {
  const option = useMemo(() => {
    if (!paths?.bands) return null;
    const b = paths.bands;
    const x = paths.steps;
    const band = (lo: (number | null)[], hi: (number | null)[], name: string, opacity: number) => [
      { name: `${name}-lo`, type: "line", stack: name, stackStrategy: "all", data: lo, symbol: "none", lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
      { name, type: "line", stack: name, stackStrategy: "all", data: hi.map((h, i) => (h === null || lo[i] === null ? null : h - (lo[i] as number))), symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: C.accent, opacity }, silent: true, tooltip: { show: false } },
    ];
    const ext = [...(b.p05 || []), ...(b.p95 || [])].filter((v: any) => typeof v === "number" && Number.isFinite(v)).map((v: number) => Math.abs(v));
    const yr = ext.length ? Math.max(0.5, Math.ceil(Math.max(...ext) * 1.35 * 2) / 2) : 0;
    const shown: number[][] = (paths.paths || []).slice(-160);
    const dates: string[] = (paths.dates || []).slice(-160);
    return {
      grid: { left: 48, right: 18, top: 30, bottom: 28 },
      legend: { data: ["Median", "Average", ...(paths.baseline ? ["Other sessions (average)"] : []), ...overlays.map((o) => o.name)], top: 0, right: 4 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => {
        const i = ps[0]?.dataIndex ?? 0;
        return `<b>${x[i]}</b> · ${b.n[i]} paths<br/>median ${sr(b.median[i])}<br/>average ${sr(b.mean[i])}<br/>middle half ${sr(b.p25[i])} to ${sr(b.p75[i])}<br/>5th to 95th ${sr(b.p05[i])} to ${sr(b.p95[i])}`
          + (paths.baseline ? `<br/><span style="color:${C.muted}">other sessions ${sr(paths.baseline.mean[i])}</span>` : "");
      } },
      xAxis: { type: "category", data: x, boundaryGap: false },
      // scaled to the 5th-95th percentile band: a few extreme paths would otherwise flatten everything else (they are clipped, not removed)
      yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%` }, min: yr ? -yr : undefined, max: yr ? yr : undefined, scale: true },
      series: [
        ...shown.map((p, i) => ({ id: `p${i}`, name: dates[i], type: "line", data: p, symbol: "none", lineStyle: { width: 0.7, color: C.fg, opacity: 0.07 }, emphasis: { lineStyle: { width: 1.6, opacity: 0.9, color: C.info } }, tooltip: { show: false }, z: 1 })),
        ...band(b.p05, b.p95, "5th to 95th percentile", 0.1), ...band(b.p25, b.p75, "Middle half", 0.2),
        ...(paths.baseline ? [{ name: "Other sessions (average)", type: "line", data: paths.baseline.mean, symbol: "none", lineStyle: { width: 1.3, type: "dashed", color: C.muted }, z: 4 }] : []),
        { name: "Average", type: "line", data: b.mean, symbol: "none", lineStyle: { width: 1.4, type: "dotted", color: C.accent }, z: 5 },
        { name: "Median", type: "line", data: b.median, symbol: "circle", symbolSize: 5, lineStyle: { width: 2.2, color: C.accent }, itemStyle: { color: C.accent }, z: 6, markLine: { silent: true, symbol: "none", lineStyle: { color: C.faint, type: "solid", width: 1 }, data: [{ yAxis: 0 }], label: { show: false } } },
        ...overlays.map((o, i) => ({ name: o.name, type: "line", data: x.map((s: string) => { const k = o.steps.indexOf(s); return k < 0 ? null : o.bands.median[k]; }), symbol: "none", lineStyle: { width: 1.8, color: C.palette[(i + 1) % C.palette.length] }, z: 5 })),
      ],
    };
  }, [paths, overlays]);
  if (!option) return <Empty title="No paths to draw" />;
  return <EChart option={option} height={height} onClick={(p) => p.seriesId?.startsWith("p") && onPick?.(p.seriesName)} />;
}

/* ----------------------------------------------------------------------------- estimates with intervals */
export function Forest({ rows, unit = "pts", scale = 100 }: { rows: { label: string; estimate: number | null; lo?: number | null; hi?: number | null; n?: number | null; note?: string }[]; unit?: string; scale?: number }) {
  const vals = rows.flatMap((r) => [r.estimate, r.lo, r.hi]).filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v)).map((v) => v * scale);
  if (!vals.length) return null;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const pad = (hi - lo) * 0.08 || 1;
  const a = lo - pad, b = hi + pad;
  const X = (v: number) => `${((v * scale - a) / (b - a)) * 100}%`;
  return (
    <div className="flex flex-col gap-2.5" data-testid="forest">
      {rows.map((r) => (
        <div key={r.label} className="grid grid-cols-[minmax(150px,220px)_1fr_auto] items-center gap-3 text-[12.5px]">
          <div className="min-w-0"><div className="truncate">{r.label}</div>{r.note && <div className="truncate text-[11px] dim">{r.note}</div>}</div>
          <div className="relative h-6">
            <div className="absolute inset-y-0 w-px bg-line-2" style={{ left: X(0) }} />
            {r.lo !== null && r.lo !== undefined && r.hi !== null && r.hi !== undefined && <div className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-fg-2/50" style={{ left: X(r.lo), width: `calc(${X(r.hi)} - ${X(r.lo)})` }} />}
            {r.estimate !== null && <div className={cn("absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45", r.lo !== null && r.lo !== undefined && r.hi !== null && r.hi !== undefined && (r.lo > 0 || r.hi < 0) ? "bg-accent" : "border border-fg-2 bg-bg-0")} style={{ left: X(r.estimate) }} />}
          </div>
          <div className="num w-[150px] text-right">{r.estimate === null ? "not estimable" : `${r.estimate * scale > 0 ? "+" : ""}${(r.estimate * scale).toFixed(scale === 100 ? 1 : 2)} ${unit}`}
            {r.lo !== null && r.lo !== undefined && r.hi !== null && r.hi !== undefined && <span className="dim"> [{(r.lo * scale).toFixed(1)}, {(r.hi * scale).toFixed(1)}]</span>}</div>
        </div>
      ))}
      <div className="text-[11px] dim">A filled marker means the 95% interval excludes zero. An open marker means it does not.</div>
    </div>
  );
}

/* ----------------------------------------------------------------------------- the graph itself */
const CAT_X: Record<string, number> = { Data: 0, Features: 1, Events: 1, Conditions: 2, Outcomes: 2, Analysis: 3, Validation: 3, Strategy: 3 };
const CAT_TONE: Record<string, string> = { Data: "#7a8496", Features: "#4c8dff", Conditions: "#e8a33d", Events: "#c678dd", Outcomes: "#26a69a", Analysis: "#d6dce5", Validation: "#98c379", Strategy: "#56b6c2" };

function nodeLine(n: any): string {
  const p = n.params || {};
  if (n.op === "data") return `${p.symbol}${p.start ? ` from ${p.start}` : ""}${p.end ? ` to ${p.end}` : ""}`;
  if (n.op === "feature") return p.label || p.name || p.expr;
  if (n.op === "filter") return p.label || (p.expr ? p.expr : `${p.op} ${p.value}${p.hi !== undefined && p.hi !== null ? ` .. ${p.hi}` : ""}`);
  if (n.op === "combine") return `all of ${(n.inputs?.masks || []).length} conditions (${p.how || "and"})`;
  if (n.op === "forward_return") return `${p.direction || "up"} from ${p.entry || "open"}, ${String(p.horizon || "session").replace("sessions:", "+").replace("session", "to the close")}`;
  if (n.op === "target_stop") return `${p.target}% ${p.direction || "up"}${p.stop ? ` before ${p.stop}%` : ""}, from ${p.entry || "open"}`;
  if (n.op === "sweep") return `${p.param}: ${(p.values || []).join(", ")}`;
  if (n.op === "event") return p.label || p.expr;
  if (n.op === "sequence") return (p.steps || []).map((s: any) => s.name).join(" → ");
  return "";
}

const StepNode = memo(({ data }: NodeProps<Node<{ n: any }>>) => {
  const n = data.n;
  const tone = CAT_TONE[n.category] || "#7a8496";
  return (
    <div className={cn("w-[212px] rounded-lg border bg-bg-1 px-2.5 py-1.5 text-left", n.status === "error" ? "border-down/60" : "border-line-2")}>
      <Handle type="target" position={Position.Left} className="!size-1 !border-0 !bg-line-2" />
      <div className="flex items-center gap-1.5"><span className="size-1.5 rounded-full" style={{ background: tone }} /><span className="text-[10px] font-medium uppercase tracking-wide" style={{ color: tone }}>{n.title}</span></div>
      <div className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug text-fg-2">{nodeLine(n) || n.id}</div>
      {n.known_at && ["series", "mask", "outcome", "anchor"].includes(n.output) && <div className="mt-0.5 text-[10px] text-faint">{n.output === "outcome" ? `starts ${n.known_at === "event" ? "at the event" : n.known_at}` : tick(n.known_at)}</div>}
      {n.status === "error" && <div className="mt-0.5 line-clamp-2 text-[10px] text-down">{n.error}</div>}
      <Handle type="source" position={Position.Right} className="!size-1 !border-0 !bg-line-2" />
    </div>
  );
});
StepNode.displayName = "StepNode";
const stepTypes = { step: StepNode };

export function GraphDiagram({ nodes, height = 300 }: { nodes: any[]; height?: number }) {
  const { ns, es } = useMemo(() => {
    const rows: Record<number, number> = {};
    const ns: Node[] = nodes.map((n) => {
      const col = CAT_X[n.category] ?? 3;
      const row = (rows[col] = (rows[col] ?? -1) + 1);
      return { id: n.id, type: "step", position: { x: col * 262, y: row * 78 }, data: { n }, draggable: false, selectable: false };
    });
    const es: Edge[] = nodes.flatMap((n) => Object.values(n.inputs || {}).flat().map((src: any) => ({ id: `${src}>${n.id}`, source: String(src), target: n.id, style: { stroke: "#2b3442", strokeWidth: 1.2 } })));
    return { ns, es };
  }, [nodes]);
  return (
    <div style={{ height }} className="overflow-hidden rounded-lg border border-line bg-bg-0" data-testid="graph-diagram">
      <ReactFlow nodes={ns} edges={es} nodeTypes={stepTypes} fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.3} maxZoom={1.3} forceColorMode="dark" proOptions={{ hideAttribution: true }} nodesConnectable={false} elementsSelectable={false} panOnScroll>
        <Background color="#161c26" gap={22} />
      </ReactFlow>
    </div>
  );
}

/* ----------------------------------------------------------------------------- the card in the conversation */
export function GraphCard({ c, busy, onDecide, children }: { c: any; busy: boolean; onDecide: (choice: string) => void; children?: React.ReactNode }) {
  const g = c.graph || {};
  const clauses: any[] = g.core?.clauses || [];
  const oc = g.core?.outcome;
  const analyses = (g.nodes || []).filter((n: any) => n.output === "result");
  const waiting = ["awaiting", "needs_data"].includes(c.state);
  return (
    <div className={cn("card fade-in px-4 py-3.5", waiting && "!border-accent/40")} data-testid="graph-card">
      <div className="mb-2 flex items-center gap-2">
        <Workflow className="size-3.5 text-accent" /><span className="eyebrow">{g.batch ? "Stored findings, re-run" : "Research graph"}</span>
        {(g.nodes || []).length > 0 && <span className="chip !h-5 !text-[11px]">{g.nodes.length} steps · {analyses.length} analys{analyses.length === 1 ? "is" : "es"}</span>}
        <span className="flex-1" /><span className="text-[11px] text-faint">{g.planner}</span>
      </div>
      {clauses.length > 0 && (
        <div className="flex flex-col gap-1.5 text-[13px]">
          {clauses.map((cl, i) => (
            <div key={i} className="flex items-baseline gap-2"><span className="w-[70px] shrink-0 text-xs dim">{i === 0 ? "When" : "and"}</span><span className="min-w-0 flex-1">{cl.label}</span><span className="shrink-0 text-[11px] text-faint">{tick(cl.at)}</span></div>
          ))}
          {oc && <div className="flex items-baseline gap-2"><span className="w-[70px] shrink-0 text-xs dim">Outcome</span><span className="min-w-0 flex-1">{oc.op === "target_stop" ? `reaches ${oc.params.target}% ${oc.params.direction}${oc.params.stop ? ` before ${oc.params.stop}% the other way` : ""}` : `finishes ${oc.params.direction === "down" ? "lower" : "higher"}${oc.params.threshold ? ` by ${oc.params.threshold}% or more` : ""}`}, measured from {oc.params.entry === "open" ? "the open" : oc.params.entry === "close" ? "that session's close" : `${oc.params.entry} ET`} {String(oc.params.horizon).startsWith("sessions:") ? `over ${oc.params.horizon.split(":")[1]} session${oc.params.horizon.split(":")[1] === "1" ? "" : "s"}` : "to the close"}</span></div>}
          {(g.core?.controls || []).length > 0 && <div className="flex items-baseline gap-2"><span className="w-[70px] shrink-0 text-xs dim">Controls</span><span className="min-w-0 flex-1 soft">{(g.controls_labels || g.core.controls).join("; ")} — compared with sessions that looked the same on these</span></div>}
        </div>
      )}
      {(g.batch || []).length > 0 && <ul className="flex flex-col gap-1 text-[13px] soft">{g.batch.slice(0, 8).map((b: string, i: number) => <li key={i}>· {b}</li>)}</ul>}
      {analyses.length > 0 && <div className="mt-2.5 flex flex-wrap gap-1.5">{analyses.map((n: any) => <span key={n.id} className="chip !h-6">{n.title}</span>)}</div>}
      {(g.assumptions || []).length > 0 && (
        <div className="mt-3 rounded-lg border border-warn/30 bg-warn/[0.05] px-3 py-2" data-testid="graph-assumptions">
          <div className="text-[11px] font-medium uppercase tracking-wide text-warn">Assumed, because the question did not say</div>
          <ul className="mt-1 flex flex-col gap-1 text-[12.5px] soft">{g.assumptions.map((x: string, i: number) => <li key={i}>· {x}</li>)}</ul>
        </div>
      )}
      {(g.unsupported || []).length > 0 && <Callout tone="bad" className="mt-3" title="Not understood, and not part of the analysis">{g.unsupported.map((u: string) => `"${u}"`).join("; ")}. Nothing was substituted for it.</Callout>}
      {(c.data || []).length > 0 && <div className="mt-2.5 flex flex-wrap gap-1.5">{c.data.map((n: any, i: number) => <span key={i} className={cn("chip !h-6", !n.ok && "!border-warn/50 !text-warn")}>{n.ok ? "✓" : "!"} {n.what} {n.symbol}: {Number(n.sessions).toLocaleString()} sessions{n.first ? `, from ${n.first}` : ""}</span>)}</div>}
      <div className="mt-2.5 text-[12.5px] dim"><T>{c.text}</T></div>
      {waiting && (c.choices || []).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2" data-testid="graph-choices">
          {c.choices.map((ch: any, i: number) => <button key={ch.id} className={cn("btn sm", i === 0 && ch.id !== "cancel" && "primary")} disabled={busy} onClick={() => onDecide(ch.id)}>{i === 0 && ch.id !== "cancel" && <Play />}{ch.label}</button>)}
        </div>
      )}
      {children}
    </div>
  );
}

/* ----------------------------------------------------------------------------- sections of the result */
function Evidence({ r }: { r: any }) {
  const e = r.evidence;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <StatTile label="When the condition held" value={pp(e.probability)} sub={`${e.hits} of ${e.n} sessions`} hint="How often the outcome happened on the sessions that satisfied every condition. Sessions where target and stop fell inside one bar are left out and counted separately." />
      <StatTile label="On the other sessions" value={pp(e.rest_probability)} sub={`${Number(e.rest_n).toLocaleString()} sessions`} />
      <StatTile label="Difference" value={sp(e.lift_vs_rest)} sub={e.lift_ci95?.[0] !== null && e.lift_ci95?.[0] !== undefined ? `95% interval ${sp(e.lift_ci95[0])} to ${sp(e.lift_ci95[1])}` : "interval not estimable"} tone={e.p_value !== null && e.p_value < 0.05 ? "text-accent-2" : undefined} />
      <StatTile label="p-value (weeks resampled)" value={pv(e.p_value)} sub={e.p_value === null ? "too few independent weeks" : `${e.clusters} weeks with a qualifying session`} hint="Whole calendar weeks are resampled, because sessions in the same week are not independent observations. This is before any correction for other configurations tried." />
    </div>
  );
}

function Probability({ r, onFinding }: { r: any; onFinding?: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const e = r.evidence;
  return (
    <Section title="How often it happened" sub={`${r.symbol} · ${r.window?.[0]} to ${r.window?.[1]}`} id="probability">
      <div className="mb-3 text-[13.5px]"><span className="dim">When </span>{r.condition}<span className="dim"> — outcome: </span>{r.outcome}</div>
      <Evidence r={r} />
      <div className="mt-3 flex flex-col gap-1.5 text-[13px] soft">{(r.summary || []).map((s: string, i: number) => <p key={i}><T>{s}</T></p>)}</div>
      {r.thin && <Callout tone="warn" className="mt-3" title={`Only ${e.n} qualifying sessions`}>Fewer than {r.min_sample}: the numbers describe those sessions and support no conclusion about a tendency.</Callout>}
      {(e.ambiguous > 0 || r.unknown_condition > 0) && <div className="mt-2 text-xs dim">{e.ambiguous > 0 && `${e.ambiguous} qualifying session${e.ambiguous === 1 ? "" : "s"} left out because target and stop fell inside the same bar (the order cannot be known). `}{r.unknown_condition > 0 && `${Number(r.unknown_condition).toLocaleString()} sessions where the condition could not be evaluated are excluded, not counted as "condition false".`}</div>}
      {onFinding}
      {(r.by || []).length > 0 && (
        <div className="mt-4" data-testid="scan">
          <div className="mb-1.5 text-xs dim">Among the qualifying sessions only: the outcome by thirds of each value, top third against bottom third (p-values corrected for all {r.by.length} values scanned)</div>
          <div className="overflow-auto rounded-lg border border-line"><table className="t">
            <thead><tr><th>Value at the decision</th><th className="r">Low third</th><th className="r">Middle</th><th className="r">High third</th><th className="r">High − low</th><th className="r">Adjusted p</th></tr></thead>
            <tbody>{[...r.by].sort((a: any, b: any) => (a.p_adjusted ?? 9) - (b.p_adjusted ?? 9)).map((b: any) => (
              <tr key={b.feature}>
                <td className="max-w-[280px] truncate" title={b.label}>{b.p_adjusted !== null && b.p_adjusted !== undefined && b.p_adjusted < 0.05 ? <span className="mr-1.5 text-accent" title="Distinguishable after correction">◆</span> : <span className="mr-1.5 text-faint">○</span>}{b.label}</td>
                {b.buckets?.length === 3 ? b.buckets.map((k: any) => <td key={k.name} className="r num">{pp(k.probability)}<span className="dim"> /{k.n}</span></td>) : <td colSpan={3} className="r dim">{b.note || (b.buckets || []).map((k: any) => `${pp(k.probability)} /${k.n}`).join("  ·  ")}</td>}
                <td className="r num">{sp(b.difference)}</td><td className="r num">{pv(b.p_adjusted)}</td>
              </tr>))}</tbody>
          </table></div>
          <div className="mt-1.5 text-xs dim">A split found by looking at the qualifying sessions is within-sample. Treat a marked row as a new hypothesis to validate, not as a result.</div>
        </div>
      )}
      {(r.sessions || []).length > 0 && (
        <div className="mt-3">
          <button className="text-xs text-info hover:underline" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"} the {r.sessions.length.toLocaleString()} qualifying sessions behind these numbers</button>
          {open && <div className="mt-2"><GTable table={{ columns: ["date", "outcome", "end_return_pct"], rows: [...r.sessions].reverse() }} max={600} /></div>}
        </div>
      )}
    </Section>
  );
}

function Matched({ r }: { r: any }) {
  const f = r.frequency, rt = r.returns;
  const rows = (x: any) => [
    { label: "Plain comparison", estimate: x.ordinary?.estimate ?? null, lo: x.ordinary?.ci95?.[0], hi: x.ordinary?.ci95?.[1], note: `${x.ordinary?.n ?? 0} against ${Number(x.ordinary?.comparison_n ?? 0).toLocaleString()} other sessions` },
    { label: "Nearest comparable sessions", estimate: x.nearest?.estimate ?? null, lo: x.nearest?.ci95?.[0], hi: x.nearest?.ci95?.[1], note: x.nearest ? `${x.nearest.matched} matched, ${x.nearest.unmatched} without a comparable session` : "not estimable" },
    { label: "Within strata", estimate: x.stratified?.estimate ?? null, lo: x.stratified?.ci95?.[0], hi: x.stratified?.ci95?.[1], note: x.stratified ? `${x.stratified.strata} strata, ${x.stratified.bins_per_control} bins per control` : "not estimable" },
    { label: "Regression adjustment", estimate: x.regression?.estimate ?? null, lo: x.regression?.ci95?.[0], hi: x.regression?.ci95?.[1], note: "outcome ~ condition + controls, week-clustered" },
  ];
  const tone = { persists: "good", mixed: "warn", not_distinguishable: "warn", reverses: "bad", not_estimable: "warn" }[f.verdict as string] as any;
  const word = { persists: "The difference persists", mixed: "Unsettled: the methods disagree", not_distinguishable: "No longer distinguishable from zero", reverses: "The difference changes sign", not_estimable: "Could not be estimated" }[f.verdict as string];
  return (
    <Section title="Against comparable sessions" sub={`matched on ${f.controls.map((c: any) => c.label).join(", ")}`} id="matched">
      <Callout tone={tone} title={word}><T>{f.sentence}</T></Callout>
      <div className="mt-4 text-xs dim">Difference in how often the outcome happened (percentage points)</div>
      <div className="mt-2"><Forest rows={rows(f)} /></div>
      {rt?.verdict && rt.verdict !== "not_estimable" && (<><div className="mt-4 text-xs dim">Difference in the average return to the end of the horizon (%)</div><div className="mt-2"><Forest rows={rows(rt)} unit="%" scale={1} /></div></>)}
      {(f.balance || []).length > 0 && (
        <div className="mt-4">
          <div className="mb-1.5 text-xs dim">How different the two groups were on each control, in standard deviations (0 = identical)</div>
          <div className="overflow-auto rounded-lg border border-line"><table className="t">
            <thead><tr><th>Control</th><th className="r">Qualifying sessions</th><th className="r">Other sessions</th><th className="r">Matched sessions</th><th className="r">Gap before</th><th className="r">Gap after</th></tr></thead>
            <tbody>{f.balance.map((b: any) => (
              <tr key={b.name}><td>{b.label}</td><td className="r num">{b.treated_mean?.toFixed(2)}</td><td className="r num">{b.comparison_mean?.toFixed(2)}</td><td className="r num">{b.matched_mean?.toFixed(2) ?? "—"}</td>
                <td className="r num">{b.before?.toFixed(2) ?? "—"}</td><td className={cn("r num", Math.abs(b.after_nearest ?? 0) > 0.25 ? "text-warn" : "")}>{b.after_nearest?.toFixed(2) ?? "—"}{Math.abs(b.after_nearest ?? 0) > 0.25 ? " !" : ""}</td></tr>))}</tbody>
          </table></div>
        </div>
      )}
      <ul className="mt-3 flex flex-col gap-1 text-xs dim">{(f.notes || []).map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
    </Section>
  );
}

function Robustness({ r }: { r: any }) {
  const years = r.years || [];
  const opt = { grid: { left: 44, right: 10, top: 14, bottom: 24 }, tooltip: { trigger: "axis", formatter: (ps: any[]) => `${ps[0].name}: ${sp(ps[0].value / 100)} over ${years[ps[0].dataIndex].n} sessions` },
    xAxis: { type: "category", data: years.map((y: any) => y.year) }, yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}` } },
    series: [{ type: "bar", data: years.map((y: any) => ({ value: y.lift === null ? null : y.lift * 100, itemStyle: { color: y.n < 5 ? C.faint : (y.lift ?? 0) >= 0 ? C.up : C.down, opacity: y.n < 5 ? 0.5 : 0.9 } })), barMaxWidth: 16 }] };
  return (
    <Section title="Where it breaks" sub={r.flags?.length ? `${r.flags.length} weak point${r.flags.length === 1 ? "" : "s"}` : "no weak point found"} id="robustness">
      {r.flags?.length > 0 ? <Callout tone="warn" title="Weak points"><ul className="flex flex-col gap-1">{r.flags.map((f: string, i: number) => <li key={i}>· {f}</li>)}</ul></Callout> : <Callout tone="good" title="Nothing here singles out a weakness">Sub-periods, calendar years, each clause removed in turn and neighbouring thresholds all point the same way. That is not a validation on unseen data.</Callout>}
      {years.length > 1 && (<><div className="mt-4 text-xs dim">Difference from the baseline by calendar year, percentage points (faded: fewer than 5 qualifying sessions){r.years_agree ? ` — ${r.years_agree[0]} of ${r.years_agree[1]} usable years agree with the whole period` : ""}</div><EChart option={opt} height={170} /></>)}
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        {(r.periods || []).length > 0 && <GTable table={{ title: "Halves and thirds of the history", columns: ["label", "start", "end", "n", "probability", "baseline_probability"], rows: r.periods }} />}
        {(r.ablation || []).length > 0 && <GTable table={{ title: "Each condition removed in turn: does it add evidence?", columns: ["removed", "n_without", "probability_without", "contributes"], rows: r.ablation }} />}
      </div>
      {(r.neighbours || []).length > 0 && <div className="mt-3"><GTable table={{ title: `Neighbouring thresholds${r.neighbours_agree ? ` (${r.neighbours_agree[0]} of ${r.neighbours_agree[1]} keep the direction)` : ""}`, columns: ["clause", "value", "n", "probability"], rows: r.neighbours }} /></div>}
    </Section>
  );
}

const VSTATUS: Record<string, { word: string; tone: "good" | "warn" | "bad" | "info"; glyph: string }> = {
  confirmed: { word: "Confirmed on untouched data", tone: "good", glyph: "◆" }, confirmed_not_independent: { word: "Holds, but not an independent confirmation", tone: "warn", glyph: "◈" },
  holds_on_seen_data: { word: "Holds on data that had been seen", tone: "warn", glyph: "◇" }, not_confirmed: { word: "Not confirmed", tone: "bad", glyph: "✕" },
  not_a_candidate: { word: "Not a candidate: the final test period was not examined", tone: "info", glyph: "○" }, candidate_untested: { word: "A candidate that could not be tested", tone: "warn", glyph: "◇" }, not_estimable: { word: "Too little history to validate", tone: "warn", glyph: "–" },
};

function Validation({ r }: { r: any }) {
  const st = VSTATUS[r.status] || VSTATUS.not_estimable;
  const rows = [["Training", r.train], ["Validation", r.validate], ["Final test", r.test]].filter(([, x]) => x).map(([k, x]: any) => ({ period: k, from: x.start, to: x.end, n: x.n, probability: x.probability, baseline_probability: x.baseline_probability, p_value: k === "Final test" ? x.p_one_sided : x.p_value }));
  const h = r.holdout;
  return (
    <Section title="On data it was not chosen on" sub={`${st.glyph} ${st.word}`} id="validation">
      <Callout tone={st.tone} title={st.word}>{(r.summary || [])[0]}</Callout>
      {rows.length > 0 && <div className="mt-3"><GTable table={{ columns: ["period", "from", "to", "n", "probability", "baseline_probability", "p_value"], rows }} /></div>}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs dim">
        {h && <span className="chip !h-6"><Lock className="size-3" />final test period {h.start} to {h.end}: {h.label}</span>}
        <span className="chip !h-6">{r.trials} configuration{r.trials === 1 ? "" : "s"} tried in this notebook{r.p_adjusted !== null && r.p_adjusted !== undefined ? ` · adjusted p ${pv(r.p_adjusted)}` : ""}</span>
        {r.global_exposure?.configurations > 0 && <span className="chip !h-6 !border-warn/50 !text-warn">{r.global_exposure.configurations} configuration{r.global_exposure.configurations === 1 ? "" : "s"} in {r.global_exposure.notebooks} other notebook{r.global_exposure.notebooks === 1 ? "" : "s"} covered those dates ({r.global_exposure.related} related)</span>}
      </div>
      {(r.reasons || []).length > 0 && r.status !== "confirmed" && <ul className="mt-2 flex flex-col gap-1 text-xs dim">{r.reasons.map((x: string, i: number) => <li key={i}>· {x}</li>)}</ul>}
      {(r.notes || []).map((x: string, i: number) => <div key={i} className="mt-2 text-xs dim">{x}</div>)}
    </Section>
  );
}

function Sweep({ r }: { r: any }) {
  const sym = r.param === "symbol";
  return (
    <Section title={sym ? "The same analysis on each symbol" : `Across values of ${r.what}`} sub={r.same_direction === false ? "the direction changes" : r.same_direction ? "same direction throughout" : undefined} id="sweep">
      <div className="overflow-auto rounded-lg border border-line"><table className="t" data-testid="sweep-table">
        <thead><tr><th>{sym ? "Symbol" : "Value"}</th><th className="r">Sessions</th><th className="r">Frequency</th><th className="r">Baseline</th><th className="r">Difference</th><th className="r">Adjusted p</th><th>Reading</th></tr></thead>
        <tbody>{r.rows.map((x: any, i: number) => x.status !== "ok" ? <tr key={i}><td>{String(x.value)}</td><td colSpan={6} className="text-warn">{x.error}</td></tr> : (
          <tr key={i}><td className="font-medium">{String(x.value).replace("sessions:", "+").replace(/^\+(\d+)$/, "$1 sessions later")}</td><td className="r num">{x.n}</td><td className="r num">{pp(x.probability)}</td><td className="r num">{pp(x.baseline_probability)}</td>
            <td className={cn("r num", x.lift > 0 ? "text-up" : x.lift < 0 ? "text-down" : "")}>{sp(x.lift)}</td><td className="r num">{pv(x.p_adjusted)}</td>
            <td className="dim">{x.thin ? "– too few sessions" : x.distinguishable ? <span className="text-accent-2">◆ distinguishable</span> : "○ within chance"}{x.verdict ? ` · matched: ${String(x.verdict).replace(/_/g, " ")}` : ""}</td></tr>))}</tbody>
      </table></div>
      <ul className="mt-2 flex flex-col gap-1 text-xs dim">{(r.summary || []).map((s: string, i: number) => <li key={i}>· {s}</li>)}</ul>
    </Section>
  );
}

function Distribution({ r }: { r: any }) {
  const h = r.histogram;
  const opt = h && { grid: { left: 44, right: 12, top: 26, bottom: 26 }, legend: { data: ["Qualifying sessions", "Other sessions"] }, tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: h.edges.slice(0, -1).map((e: number, i: number) => ((e + h.edges[i + 1]) / 2).toFixed(2)), axisLabel: { formatter: (v: string) => `${v}%` } }, yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [{ name: "Other sessions", type: "bar", data: h.rest, itemStyle: { color: C.faint, opacity: 0.55 }, barGap: "-100%" }, { name: "Qualifying sessions", type: "bar", data: h.condition, itemStyle: { color: C.accent, opacity: 0.8 } }] };
  const q = r.quantiles;
  return (
    <Section title="What followed" sub={`${r.n} qualifying sessions`} id="distribution">
      {r.paths?.bands && (<><div className="mb-1 text-xs dim">Price paths after the decision, % from the entry price. Thin lines are individual occurrences{r.paths.shown < r.paths.n ? ` (the latest ${r.paths.shown})` : ""}; bands are percentiles across all {r.paths.n}.</div><PathChart paths={r.paths} /></>)}
      {opt && (<><div className="mt-4 text-xs dim">Return to the end of the horizon</div><EChart option={opt} height={190} /></>)}
      {q?.condition && <div className="mt-3"><GTable table={{ columns: ["group", "n", "mean", "p05", "p25", "p50", "p75", "p95"], rows: [{ group: "Qualifying sessions", ...q.condition }, ...(q.rest ? [{ group: "Other sessions", ...q.rest }] : [])] }} /></div>}
      {r.excursions?.mfe && <div className="mt-2 text-xs dim">Furthest in favour before the horizon ended: median {sr(r.excursions.mfe.p50)}; furthest against: median {sr(r.excursions.mae.p50)}.</div>}
    </Section>
  );
}

function Regression({ r }: { r: any }) {
  return (
    <Section title="Which values go with the outcome, together" sub={r.r2 !== null && r.r2 !== undefined ? `R² ${r.r2.toFixed(3)} over ${r.n} sessions` : undefined} id="regression">
      <div className="text-[13px] soft">{(r.summary || [])[0]}</div>
      {(r.terms || []).length > 0 && (
        <div className="mt-3"><Forest unit={r.target === "hit" ? "pts" : "%"} scale={r.target === "hit" ? 100 : 1} rows={[...r.terms].sort((a: any, b: any) => Math.abs(b.coef_per_sd) - Math.abs(a.coef_per_sd)).map((t: any) => ({ label: t.label, estimate: t.coef_per_sd, lo: t.ci95[0], hi: t.ci95[1], note: `adjusted p ${pv(t.p_adjusted)}` }))} /></div>
      )}
      <div className="mt-2 text-xs dim">Each row is the change in {r.target === "hit" ? "the probability of the outcome" : "the return"} for one standard deviation more of that value, holding the others fixed. Week-clustered errors; p-values corrected for all terms. Association, not cause.</div>
    </Section>
  );
}

function Strategies({ r, onBacktest, busy }: { r: any; onBacktest?: () => void; busy?: boolean }) {
  return (
    <Section title="Option structures that express it" sub={r.convertible ? `${r.variants.length} drafted, none priced` : "not convertible"} id="strategies">
      <Callout tone={r.convertible ? "info" : "warn"}>{(r.summary || [])[0]}</Callout>
      {(r.variants || []).length > 0 && <ul className="mt-3 grid gap-1.5 md:grid-cols-2">{r.variants.map((v: any, i: number) => <li key={i} className="rounded-lg border border-line px-3 py-2 text-[13px]">{v.label}</li>)}</ul>}
      {r.convertible && onBacktest && <button className="btn primary sm mt-3" disabled={busy} onClick={onBacktest}><FlaskConical />Backtest them on observed option prices</button>}
    </Section>
  );
}

/* ----------------------------------------------------------------------------- the whole result */
export function GraphView({ runId, findings = [], busy, onStrategy, onAsk }: { runId: string; findings?: any[]; busy?: boolean; onStrategy?: (findingId: number) => void; onAsk?: (text: string) => void }) {
  const q = useQuery({ queryKey: ["graph", runId], queryFn: () => api.get<any>(`/api/runs/${runId}/graph`), staleTime: 300_000 });
  const [showGraph, setShowGraph] = useState(false);
  if (q.isLoading) return <div className="flex flex-col gap-3 p-5"><Skeleton className="h-8 w-2/3" /><Skeleton className="h-40" /><Skeleton className="h-56" /></div>;
  if (q.isError || !q.data) return <Empty title="This result could not be loaded">{String((q.error as any)?.message || "")}</Empty>;
  const res = q.data;
  if (res.kind === "graph_batch") return <BatchView res={res} />;
  const by = (k: string) => res.nodes.filter((n: any) => n.result?.kind === k).map((n: any) => n.result);
  const data = res.nodes.find((n: any) => n.op === "data")?.info;
  const errors = res.nodes.filter((n: any) => n.status === "error");
  const main = findings.find((f) => f.kind === "pattern");
  const next = ["Compare that behavior with QQQ and IWM.", "Show the distribution of price paths following those events.", "Find which historical conditions are associated with that outcome.", "Build a strategy from the strongest supported finding.", "Check whether I've already tested similar configurations on this holdout."];
  return (
    <div className="mx-auto flex max-w-[1040px] flex-col gap-4 p-5" data-testid="graph-view">
      <div>
        <div className="eyebrow flex items-center gap-2"><Workflow className="size-3.5" />Research graph · {res.symbols.join(", ")} · calculated, not written by a model</div>
        <h2 className="mt-1.5 leading-snug">{res.question || "Research graph"}</h2>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs dim">
          {main && <GradeBadge g={main.evidence} />}
          {data && <span>{Number(data.sessions).toLocaleString()} sessions, {data.first} to {data.last}</span>}
          <span>· {res.nodes.length} steps · {res.trials} new configuration{res.trials === 1 ? "" : "s"} recorded</span>
          <button className="text-info hover:underline" onClick={() => setShowGraph((s) => !s)} data-testid="toggle-graph">{showGraph ? "Hide the graph" : "Show the graph"}</button>
        </div>
        {main?.why?.length > 0 && <div className="mt-1.5 text-xs dim">Why this grade: {main.why.join("; ")}.</div>}
      </div>
      {showGraph && <GraphDiagram nodes={res.nodes} />}
      {errors.length > 0 && <Callout tone="bad" title="Not every step could be computed">{errors.map((n: any) => `${n.title}: ${n.error}`).join("; ")}</Callout>}
      {(res.unsupported || []).length > 0 && <Callout tone="bad" title="Part of the question is not in this analysis">{res.unsupported.map((u: string) => `"${u}"`).join("; ")}</Callout>}
      {/* a comparison across symbols is the answer to "compare that with...", so it leads; the base symbol's own full-history result follows */}
      {by("sweep").filter((r: any) => r.param === "symbol").map((r: any, i: number) => <Sweep key={`sym${i}`} r={r} />)}
      {by("probability").map((r: any, i: number) => <Probability key={i} r={r} onFinding={findings.filter((f) => f.convertible && onStrategy).length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">{findings.filter((f) => f.kind === "pattern" && f.convertible).slice(0, 1).map((f) => <button key={f.id} className="btn sm" disabled={busy} onClick={() => onStrategy?.(f.id)} data-testid="finding-to-strategy"><Route />Draft option strategies from this finding</button>)}</div>) : undefined} />)}
      {by("matched").map((r: any, i: number) => <Matched key={i} r={r} />)}
      {by("sweep").filter((r: any) => r.param !== "symbol").map((r: any, i: number) => <Sweep key={i} r={r} />)}
      {by("distribution").map((r: any, i: number) => <Distribution key={i} r={r} />)}
      {by("regression").map((r: any, i: number) => <Regression key={i} r={r} />)}
      {by("correlation").map((r: any, i: number) => <Section key={i} title="Correlation" id="correlation"><div className="text-[13px] soft">{(r.summary || [])[0]}</div></Section>)}
      {by("robustness").map((r: any, i: number) => <Robustness key={i} r={r} />)}
      {by("validation").map((r: any, i: number) => <Validation key={i} r={r} />)}
      {by("strategies").map((r: any, i: number) => <Strategies key={i} r={r} busy={busy} />)}
      {findings.filter((f) => f.kind === "refinement").length > 0 && (
        <Section title="New hypotheses this turned up" sub="found within the sample; not validated" id="refinements">
          <div className="flex flex-col gap-2">{findings.filter((f) => f.kind === "refinement").map((f) => (
            <div key={f.id} className="flex items-start gap-3 rounded-lg border border-line px-3 py-2.5"><GradeBadge g={f.evidence} /><div className="min-w-0 flex-1 text-[13px]">{f.statement}</div>
              {f.convertible && onStrategy && <button className="btn quiet sm" disabled={busy} onClick={() => onStrategy(f.id)}><Route />Draft strategies</button>}</div>))}</div>
        </Section>
      )}
      {(res.assumptions || []).length > 0 && <Callout tone="warn" title="Assumed, because the question did not say"><ul className="flex flex-col gap-1">{res.assumptions.map((x: string, i: number) => <li key={i}>· {x}</li>)}</ul></Callout>}
      <Callout tone="info" title="Limits of this analysis"><ul className="flex flex-col gap-1">{(res.limits || []).map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul></Callout>
      {onAsk && <div className="flex flex-wrap gap-2">{next.map((t) => <button key={t} className="chip" disabled={busy} onClick={() => onAsk(t)}>{t}</button>)}</div>}
    </div>
  );
}

const VERDICT: Record<string, string> = { holds: "◆ holds", "same direction, not distinguishable": "◐ same direction, not distinguishable", reversed: "✕ reversed", "points the other way, not distinguishable": "◇ points the other way",
  "too few sessions": "– too few sessions", distinguishable: "◆ distinguishable", "within chance": "○ within chance" };

export function BatchView({ res }: { res: any }) {
  const still = res.what === "still_hold";
  return (
    <div className="mx-auto flex max-w-[1040px] flex-col gap-4 p-5" data-testid="batch-view">
      <div><div className="eyebrow flex items-center gap-2"><History className="size-3.5" />{still ? "Stored findings, re-tested with unchanged conditions" : "Related configurations"}</div><h2 className="mt-1.5">{res.title}</h2></div>
      <div className="card px-4 py-3.5 text-[14px] leading-relaxed">{(res.summary || []).join(" ")}</div>
      <div className="overflow-auto rounded-lg border border-line"><table className="t">
        <thead><tr><th>Configuration</th>{still && <th className="r">Originally</th>}<th className="r">{still ? "Now" : "Frequency"}</th><th className="r">Sessions</th><th className="r">Other sessions</th><th className="r">p</th><th>Reading</th></tr></thead>
        <tbody>{res.rows.map((r: any, i: number) => r.status !== "ok" ? <tr key={i}><td className="max-w-[360px] truncate">{r.label}</td><td colSpan={6} className="text-warn">{r.error}</td></tr> : (
          <tr key={i}><td className="max-w-[360px] whitespace-normal">{r.label}<div className="text-[11px] dim">{r.symbol} · {r.window?.[0]} to {r.window?.[1]}</div></td>
            {still && <td className="r num">{pp(r.original?.probability)}<span className="dim"> /{r.original?.n}</span></td>}
            <td className="r num">{pp(r.probability)}</td><td className="r num">{r.n}</td><td className="r num">{pp(r.rest_probability)}</td><td className="r num">{pv(still ? r.p_value : r.p_adjusted)}</td><td className="dim">{VERDICT[r.verdict] || r.verdict}</td></tr>))}</tbody>
      </table></div>
      <Callout tone="info" title="Limits"><ul className="flex flex-col gap-1">{(res.notes || []).map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul></Callout>
    </div>
  );
}

/* ----------------------------------------------------------------------------- memory, registry, drafted strategies, forward */
export function MemoryView({ sid, onAsk }: { sid: string; onAsk?: (t: string) => void }) {
  const q = useQuery({ queryKey: ["memory", sid], queryFn: () => api.get<any>(`/api/ai/sessions/${sid}/memory`) });
  if (q.isLoading) return <div className="p-5"><Skeleton className="h-40" /></div>;
  const m = q.data;
  if (!m) return <Empty title="No memory yet" />;
  const List = ({ items, empty }: { items: any[]; empty: string }) => items.length ? <ul className="flex flex-col gap-1.5 text-[13px]">{items.map((x, i) => <li key={i} className="flex gap-2"><span className="text-faint">·</span><span className="min-w-0 soft">{x.text || x.condition}{x.times > 1 ? <span className="dim"> · used {x.times} times</span> : null}</span></li>)}</ul> : <div className="text-[13px] dim">{empty}</div>;
  return (
    <div className="mx-auto flex max-w-[980px] flex-col gap-4 p-5" data-testid="memory-view">
      <div><div className="eyebrow flex items-center gap-2"><Boxes className="size-3.5" />What this notebook has established · assembled from its stored runs</div><h2 className="mt-1.5">Research memory</h2></div>
      <div className="card px-4 py-3.5 text-[14px] leading-relaxed">{m.text}</div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Questions asked" value={m.hypotheses.length} /><StatTile label="Configurations tested" value={m.ledger.configurations} hint="Every distinct configuration, including the ones inside sweeps and plans." />
        <StatTile label="Findings on record" value={m.findings.length} /><StatTile label="Set aside" value={m.discarded.length} />
      </div>
      <Section title="Findings" sub="strongest evidence first">
        {m.findings.length ? <div className="flex flex-col gap-2">{[...m.findings].sort((a: any, b: any) => b.rank - a.rank).map((f: any) => <div key={f.id} className="flex items-start gap-3 rounded-lg border border-line px-3 py-2.5"><GradeBadge g={f.evidence} /><div className="min-w-0 text-[13px]">{f.statement}<div className="mt-0.5 text-[11px] dim">{f.created_at?.slice(0, 10)} · run {f.run_id}</div></div></div>)}</div> : <div className="text-[13px] dim">Nothing is on record as supported yet. Nothing is listed here unless a run produced it.</div>}
      </Section>
      <Section title="Set aside" sub="within chance, too few observations, or not confirmed" open={false}>
        <div className="flex flex-col gap-2">{m.discarded.map((f: any) => <div key={f.id} className="flex items-start gap-3 rounded-lg border border-line px-3 py-2.5"><GradeBadge g={f.evidence} /><div className="min-w-0 text-[13px] soft">{f.statement}</div></div>)}{!m.discarded.length && <div className="text-[13px] dim">Nothing has been set aside.</div>}</div>
      </Section>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Conditions already tested"><List items={m.tested_conditions} empty="No condition has been tested yet." /></Section>
        <Section title="Open items"><List items={m.open_questions} empty="Nothing is open." /></Section>
        <Section title="Assumptions made along the way" open={false}><List items={m.assumptions} empty="Nothing was assumed." /></Section>
        <Section title="Data limitations met" open={false}><List items={m.limitations} empty="None recorded." /></Section>
      </div>
      {onAsk && <div className="flex flex-wrap gap-2">{["Which of our previous findings still hold after 2020?", "Which of these results are actually independent?", "Find something similar to what we found today."].map((t) => <button key={t} className="chip" onClick={() => onAsk(t)}>{t}</button>)}</div>}
    </div>
  );
}

export function RegistryCheckView({ c }: { c: any }) {
  const k = c.check || {};
  return (
    <div className="mx-auto flex max-w-[1000px] flex-col gap-4 p-5" data-testid="registry-check">
      <div><div className="eyebrow flex items-center gap-2"><Scale className="size-3.5" />Experiment registry · every notebook</div><h2 className="mt-1.5">Has this been tested before?</h2><div className="mt-1 text-[13px] dim">{c.about}</div></div>
      <div className="card px-4 py-3.5"><ul className="flex flex-col gap-2 text-[13.5px] leading-relaxed">{(k.sentences || []).map((s: string, i: number) => <li key={i}>{s}</li>)}</ul></div>
      {(k.holdouts || []).map((h: any) => (
        <Section key={h.id} title={`Final test period · ${h.basis}`} sub={`${h.start} to ${h.end}`}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label="Status in this notebook" value={<span className="!text-[17px]">{h.label}</span>} sub={h.reason || undefined} tone={h.status === "sealed" ? "text-up" : h.status === "used" ? "text-info" : "text-warn"} />
            <StatTile label="Scored on it here" value={h.scored} sub={h.this_configuration_scored ? "including this configuration" : "this configuration was not"} />
            <StatTile label="Covered those dates here" value={h.here} hint="Configurations tested in this notebook on a window that reaches into the final test period." />
            <StatTile label="Elsewhere in Strata" value={h.elsewhere.configurations} sub={`${h.elsewhere.notebooks} notebook${h.elsewhere.notebooks === 1 ? "" : "s"} · ${h.elsewhere.related} related`} tone={h.elsewhere.configurations ? "text-warn" : "text-up"} />
          </div>
          {h.elsewhere.by_notebook?.length > 0 && <div className="mt-3"><GTable table={{ title: "Other notebooks that evaluated configurations on those dates", columns: ["title", "configurations", "related", "last"], rows: h.elsewhere.by_notebook }} /></div>}
        </Section>
      ))}
      {(k.exact || []).length > 0 && <GTable table={{ title: "The identical configuration", columns: ["title", "kind", "evaluations", "phase", "first", "last"], rows: k.exact }} />}
      {(k.related || []).length > 0 && <GTable table={{ title: "Closely related configurations (same symbol, overlapping conditions and outcome)", columns: ["label", "title", "similarity", "n", "p_value", "phase", "last"], rows: k.related }} />}
      <Callout tone="info" title="What this is for">The registry records and warns; it never blocks an experiment. A result is called independently confirmed only when its final test period was untouched in its own notebook and no related configuration elsewhere had examined those dates.</Callout>
    </div>
  );
}

export function StrategyPlanView({ c, busy, onBacktest }: { c: any; busy?: boolean; onBacktest: () => void }) {
  const f = c.finding || {};
  return (
    <div className="mx-auto flex max-w-[980px] flex-col gap-4 p-5" data-testid="strategy-plan">
      <div><div className="eyebrow flex items-center gap-2"><GitFork className="size-3.5" />From a finding to strategies · nothing priced yet</div><h2 className="mt-1.5">Option structures for: {c.label}</h2></div>
      <div className="card flex items-start gap-3 px-4 py-3.5"><GradeBadge g={f.evidence} /><div className="min-w-0 text-[13.5px] leading-relaxed">{f.statement}{f.why?.length > 0 && <div className="mt-1 text-xs dim">{f.why.join("; ")}</div>}</div></div>
      {!["confirmed", "holds, not independent", "supported", "suggestive"].includes(f.evidence) && <Callout tone="bad" title="This finding is not supported by its evidence">The structures below are exploratory. A pattern that is within chance in the underlying has nothing for an option position to capture.</Callout>}
      <Callout tone="warn" title="A pattern is not a strategy result">These are drafts. Whether any of them would have made money is decided only by pricing the trades on observed option prices, after spreads, time decay and fees.{c.cover?.sentence ? ` ${c.cover.sentence}` : ""}</Callout>
      <div className="grid gap-2 md:grid-cols-2">{(c.variants || []).map((v: any, i: number) => (
        <div key={i} className="rounded-lg border border-line px-3.5 py-3"><div className="text-[13.5px] font-medium">{v.label}</div>
          <div className="mt-1.5 flex flex-col gap-0.5 text-xs dim">{(v.items || []).filter((it: any) => ["Position", "Entry", "Exit", "Expiration", "Strike"].some((k) => String(it.label).startsWith(k))).slice(0, 4).map((it: any, j: number) => <div key={j} className="truncate"><span className="text-faint">{it.label}: </span>{String(it.value).split("   [")[0]}</div>)}</div></div>))}</div>
      {(c.notes || []).map((n: string, i: number) => <div key={i} className="text-xs dim">· {n}</div>)}
      <div><button className="btn primary" disabled={busy || c.launched} onClick={onBacktest} data-testid="backtest-plan"><FlaskConical />{c.launched ? "Backtests launched" : "Backtest them on observed option prices"}</button></div>
    </div>
  );
}

export function ForwardCard({ ex }: { ex: any }) {
  return (
    <div className="mt-3 rounded-lg border border-line bg-bg-2/40 px-3.5 py-3 text-[13px]">
      <div className="flex items-center gap-2"><Snowflake className="size-4 text-info" /><span className="font-medium">{ex.name}</span><span className="chip !h-5 !text-[11px]">frozen v{ex.version}</span></div>
      <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-1 text-xs dim"><div>First session that counts <span className="num text-fg-2">{ex.first_session || ex.activation_date}</span></div><div>Configuration hash <span className="num text-fg-2">{ex.config_hash}</span></div></div>
      <a className="btn sm mt-2.5" href={`/forward/${ex.id}`}><ShieldCheck />Open the forward record</a>
    </div>
  );
}
