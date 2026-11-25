/* The session calendar of a run, and the replay of a single day: price, the option contracts that were traded,
   and how every rule evaluated - so "why did it enter here and not there" can be read off the screen. */
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronLeft, ChevronRight, Minus, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { C } from "@/components/charts/EChart";
import { dayTime, etChartTime, LWChart, type LWSeries } from "@/components/charts/LWChart";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { hhmmOf, sgn, STATUS_NAME, usd } from "@/lib/lab";
import { Callout, Empty, Skeleton, T, useHotkeys } from "./bits";

// the wheel scrolls the page; drag to pan and pinch to zoom still work
const NO_WHEEL = { handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }, handleScale: { mouseWheel: false, pinch: true, axisPressedMouseMove: true } };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const iso = (d: Date) => `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;

function YearGrid({ year, days, closed, start, end, sel, onPick, focus }: { year: number; days: Record<string, any>; closed: Set<string>; start: string; end: string; sel: string | null; onPick: (d: string) => void; focus: string | null }) {
  // columns = weeks, rows = Monday..Friday
  const weeks = useMemo(() => {
    const first = new Date(Date.UTC(year, 0, 1));
    const cur = new Date(first);
    cur.setUTCDate(cur.getUTCDate() - ((cur.getUTCDay() + 6) % 7));
    const out: (string | null)[][] = [];
    while (cur.getUTCFullYear() <= year) {
      const w: (string | null)[] = [];
      for (let i = 0; i < 7; i++) {
        if (i < 5) w.push(cur.getUTCFullYear() === year ? iso(cur) : null);
        cur.setUTCDate(cur.getUTCDate() + 1);
      }
      out.push(w);
      if (cur.getUTCFullYear() > year) break;
    }
    return out;
  }, [year]);
  const monthAt = weeks.map((w) => { const d = w.find((x) => x && x.endsWith("-01")) || (w[0]?.slice(8) && Number(w[0]!.slice(8)) <= 7 ? w[0] : null); return d ? Number(d.slice(5, 7)) - 1 : -1; });
  const cols = { gridTemplateColumns: `26px repeat(${weeks.length}, minmax(0, 1fr))` };
  return (
    <div className="flex flex-col gap-[3px]" style={{ maxWidth: 26 + weeks.length * 20 }}>
      <div className="grid gap-[3px] text-[10.5px] text-faint" style={cols}>
        <span />
        {weeks.map((_, i) => <span key={i} className="overflow-visible whitespace-nowrap">{monthAt[i] >= 0 && (i === 0 || monthAt[i] !== monthAt[i - 1]) ? MONTHS[monthAt[i]] : ""}</span>)}
      </div>
      {[0, 1, 2, 3, 4].map((row) => (
        <div key={row} className="grid items-center gap-[3px]" style={cols}>
          <span className="text-[10.5px] text-faint">{["Mon", "", "Wed", "", "Fri"][row]}</span>
          {weeks.map((w, i) => {
            const d = w[row];
            if (!d || d < start || d > end || closed.has(d)) return <span key={i} className="day off" />;
            const x = days[d];
            const st = x?.s || "none";
            const dim = focus && st !== focus;
            return (
              <Tip key={i} content={<span><b>{d}</b><br />{STATUS_NAME[st]}{x?.v !== undefined && x?.v !== null ? ` · ${x.v > 0 ? "+" : ""}${Number(x.v).toFixed(2)}` : ""}{x?.why ? <><br /><span className="opacity-80">{String(x.why).slice(0, 140)}</span></> : null}</span>}>
                <button aria-label={`${d}: ${STATUS_NAME[st]}`} data-status={st} data-date={d} className={cn("day", st !== "none" && `s-${st}`, sel === d && "sel", dim && "opacity-25")} onClick={() => onPick(d)} />
              </Tip>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export function CalendarView({ runId, day, onDay, year: wanted }: { runId: string; day: string | null; onDay: (d: string | null) => void; year?: number | null }) {
  const q = useQuery({ queryKey: ["calendar", runId], queryFn: () => api.get<any>(`/api/runs/${runId}/calendar`), staleTime: 60_000 });
  const c = q.data;
  const marked = useMemo(() => (c ? Object.keys(c.days).sort() : []), [c]);
  const years = useMemo(() => { if (!c) return []; const a = Number(c.start.slice(0, 4)), b = Number(c.end.slice(0, 4)); return Array.from({ length: b - a + 1 }, (_, i) => b - i); }, [c]);
  const [year, setYear] = useState<number | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  useEffect(() => {
    if (!c) return;
    setYear(day ? Number(day.slice(0, 4)) : wanted ? wanted : marked.length ? Number(marked[marked.length - 1].slice(0, 4)) : Number(c.end.slice(0, 4)));
  }, [c, runId]); // eslint-disable-line
  useEffect(() => { if (day) setYear(Number(day.slice(0, 4))); }, [day]);
  useEffect(() => { if (wanted && c) setYear(wanted); }, [wanted, c]);
  const step = (dir: 1 | -1) => {
    const list = focus ? marked.filter((d) => c.days[d].s === focus) : marked;
    if (!list.length) return;
    const i = day ? list.indexOf(day) : -1;
    const next = i < 0 ? (dir > 0 ? list.find((d) => !day || d > day) || list[0] : [...list].reverse().find((d) => !day || d < day) || list[list.length - 1]) : list[(i + dir + list.length) % list.length];
    onDay(next);
  };
  useHotkeys({ ArrowRight: () => step(1), ArrowLeft: () => step(-1), j: () => step(1), k: () => step(-1), Escape: () => onDay(null) }, [marked, day, focus]);
  if (q.isLoading) return <div className="p-5"><Skeleton className="h-40" /></div>;
  if (!c) return <Empty title="No calendar for this result" />;
  const closed = new Set<string>(c.closed || []);
  const perYear: Record<number, number> = {};
  for (const d of marked) perYear[Number(d.slice(0, 4))] = (perYear[Number(d.slice(0, 4))] || 0) + (c.days[d].q ? 1 : 0);
  const maxY = Math.max(1, ...Object.values(perYear));
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="calendar">
      <div className="flex flex-wrap items-center gap-1.5">
        {[...c.legend, ...(c.counts.none && !c.legend.some((l: any) => l.id === "none") ? [{ id: "none", label: "Did not qualify" }] : [])].map((l: any) => (
          <button key={l.id} className={cn("chip", focus === l.id && "on")} onClick={() => setFocus((f) => (f === l.id ? null : l.id))} disabled={l.id === "none"}>
            <span className={cn("dot", l.id === "none" ? "bg-bg-4" : `s-${l.id}`)} style={{ borderRadius: 3 }} />{l.label}<span className="num text-faint">{(c.counts[l.id] || 0).toLocaleString()}</span>
          </button>
        ))}
        <span className="flex-1" />
        <button className="btn quiet sm !px-1.5" onClick={() => step(-1)} title="Previous marked session (←)"><ChevronLeft /></button>
        <button className="btn quiet sm !px-1.5" onClick={() => step(1)} title="Next marked session (→)"><ChevronRight /></button>
      </div>
      {years.length > 1 && (
        <div className="flex items-end gap-[3px] overflow-x-auto pb-1" style={{ scrollbarWidth: "none" }}>
          {[...years].reverse().map((y) => (
            <button key={y} onClick={() => setYear(y)} title={`${y}: ${perYear[y] || 0} qualifying session${perYear[y] === 1 ? "" : "s"}`} className="group flex min-w-0 flex-1 flex-col items-center gap-1">
              <span className={cn("w-full rounded-sm transition-colors", year === y ? "bg-accent" : "bg-bg-4 group-hover:bg-faint")} style={{ height: 4 + ((perYear[y] || 0) / maxY) * 30 }} />
              <span className={cn("num text-[10px]", year === y ? "text-fg" : "text-faint")}>{String(y).slice(2)}</span>
            </button>
          ))}
        </div>
      )}
      <div className="card px-4 py-3.5">
        <div className="mb-2 flex items-center gap-2">
          <button className="btn quiet sm !px-1.5" disabled={year === years[years.length - 1]} onClick={() => setYear((y) => (y || 0) - 1)}><ChevronLeft /></button>
          <span className="num text-[15px] font-medium">{year}</span>
          <button className="btn quiet sm !px-1.5" disabled={year === years[0]} onClick={() => setYear((y) => (y || 0) + 1)}><ChevronRight /></button>
          <span className="text-xs dim">{perYear[year || 0] || 0} qualifying session{perYear[year || 0] === 1 ? "" : "s"} · click a day to replay it</span>
        </div>
        {year && <YearGrid year={year} days={c.days} closed={closed} start={c.start} end={c.end} sel={day} onPick={onDay} focus={focus} />}
      </div>
      {day ? <DayView runId={runId} day={day} onClose={() => onDay(null)} /> : <div className="text-[13px] dim">Pick a session to see its prices, the contracts traded, and why the rules did or did not fire. Use ← and → to walk through the marked sessions.</div>}
    </div>
  );
}

// ----------------------------------------------------------------------------- one day
const TONE: Record<string, string> = { target: "text-up", win: "text-up", stop: "text-down", loss: "text-down", neither: "text-info", ambiguous: "text-warn", skipped: "text-synth", missing: "text-warn", none: "dim" };

export function DayView({ runId, day, onClose }: { runId: string; day: string; onClose?: () => void }) {
  const q = useQuery({ queryKey: ["day", runId, day], queryFn: () => api.get<any>(`/api/runs/${runId}/day?day=${day}`), staleTime: 60_000 });
  const [cursor, setCursor] = useState<number | null>(null);
  const d = q.data;
  const price = useMemo<LWSeries[]>(() => {
    if (!d?.chart) return [];
    const daily = d.chart.resolution === "1d";
    const tm = (t: number) => (daily ? dayTime(new Date(t * 1000).toISOString().slice(0, 10)) : etChartTime(t));
    const lines: any[] = [];
    const o = d.observation;
    if (o?.target_level) lines.push({ price: o.target_level, color: C.up, title: "target", style: "solid" });
    if (o?.stop_level) lines.push({ price: o.stop_level, color: C.down, title: "stop", style: "solid" });
    if (d.chart.prev_close) lines.push({ price: d.chart.prev_close, color: C.muted, title: "prev close" });
    if (d.chart.prev_high) lines.push({ price: d.chart.prev_high, color: C.faint, title: "prev high" });
    if (d.chart.prev_low) lines.push({ price: d.chart.prev_low, color: C.faint, title: "prev low" });
    const marks: any[] = [];
    if (!daily) {
      const sig = d.rules?.signal;
      if (sig && !(d.trades || []).length) marks.push({ time: etChartTime(sig.t), position: "belowBar", color: C.accent, shape: "arrowUp", text: "signal" });
      for (const t of d.trades || []) {
        if (t.signal_ts && t.signal_ts !== t.entry_ts) marks.push({ time: etChartTime(t.signal_ts), position: "belowBar", color: C.accent, shape: "circle", text: "signal" });
        if (t.entry_time && t.entry_ts) marks.push({ time: etChartTime(t.entry_ts), position: "belowBar", color: C.info, shape: "arrowUp", text: t.lot > 1 ? `add ${t.lot}` : "fill" });
        if (t.exit_ts && t.exit_date === day) marks.push({ time: etChartTime(t.exit_ts), position: "aboveBar", color: t.pnl > 0 ? C.up : C.down, shape: "arrowDown", text: String(t.exit_reason || "exit").replace(/_/g, " ") });
      }
      for (const [name, st] of Object.entries<any>(d.rules?.stages || {})) marks.push({ time: etChartTime(st.t), position: "aboveBar", color: C.synth, shape: "circle", text: name });
      marks.sort((a, b) => (a.time as number) - (b.time as number));
    } else marks.push({ time: dayTime(day), position: "aboveBar", color: C.accent, shape: "arrowDown", text: day });
    return [{ type: "candles", data: d.chart.bars.map((b: number[]) => ({ time: tm(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })), priceLines: lines, markers: marks }];
  }, [d, day]);
  const opt = useMemo<LWSeries[]>(() => (d?.option_series || []).map((s: any, i: number) => ({
    type: "line" as const, data: s.points.map((p: number[]) => ({ time: etChartTime(p[0]), value: p[1] })), options: { color: C.palette[(i + 1) % C.palette.length], title: s.label, lineWidth: 1.5, pointMarkersVisible: s.points.length < 80 },
  })), [d]);
  if (q.isLoading) return <div className="card p-4"><Skeleton className="h-64" /></div>;
  if (q.isError || !d) return <Callout tone="bad">This session could not be opened: {String((q.error as any)?.message || "")}</Callout>;
  const daily = d.chart?.resolution === "1d";
  return (
    <div className="card fade-in overflow-hidden" data-testid="day-view">
      <div className="flex items-start gap-3 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2.5">
            <span className="num text-[15px] font-medium">{new Date(`${day}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", year: "numeric", month: "short", day: "numeric", timeZone: "UTC" })}</span>
            <span className={cn("text-[13px] font-medium", TONE[d.status])}>{STATUS_NAME[d.status] || d.status}</span>
            {d.chart && <span className="text-xs dim">{daily ? "only a daily bar is stored for this session" : `${d.chart.resolution} bars · New York time`}</span>}
          </div>
          <div className="mt-1 text-[13.5px] soft"><T>{d.why}</T></div>
        </div>
        {onClose && <button className="btn quiet sm !px-1.5" onClick={onClose} aria-label="Close"><X /></button>}
      </div>
      {d.chart ? <LWChart height={300} timeVisible={!daily} series={price} onCrosshair={setCursor} extra={NO_WHEEL} /> : <div className="px-4 py-6 text-[13px] dim">{d.chart_error || "No prices are stored for this session."}</div>}
      {opt.length > 0 && (
        <div className="border-t border-line">
          <div className="flex items-center gap-2 px-4 pt-2.5 text-xs dim">
            <span className="eyebrow">Option contracts</span>
            {(d.option_series || []).map((s: any, i: number) => <span key={i} className="flex items-center gap-1.5"><span className="dot" style={{ background: C.palette[(i + 1) % C.palette.length] }} />{s.label} · {s.kind} · {s.points.length} observations</span>)}
          </div>
          <LWChart height={150} timeVisible series={opt} cursorTime={cursor} extra={NO_WHEEL} />
        </div>
      )}
      {(d.trades || []).length > 0 && (
        <div className="border-t border-line px-4 py-3">
          <div className="eyebrow mb-2">What was traded</div>
          <div className="flex flex-col gap-3">
            {d.trades.map((t: any) => (
              <div key={t.id} className="text-[13px]">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                  <span className="font-medium">{t.lot > 1 ? `Added lot ${t.lot}` : "Entry"}</span>
                  <span className="soft">signal <span className="num">{t.signal_time}</span> → filled <span className="num">{t.entry_time}</span>{t.fill_delay_seconds ? <span className="dim"> ({Math.round(t.fill_delay_seconds / 60)} min later)</span> : null}</span>
                  <span className="soft">closed <span className="num">{t.exit_date !== day ? `${t.exit_date} ` : ""}{t.exit_time}</span> by {String(t.exit_reason || "").replace(/_/g, " ")}</span>
                  <span className={cn("num font-medium", t.pnl > 0 ? "text-up" : "text-down")}>{usd(t.pnl, 2)}</span>
                </div>
                <div className="mt-1 flex flex-col gap-0.5 text-xs dim">
                  {t.legs.map((l: any, i: number) => (
                    <div key={i}>
                      {l.side} {l.contracts} × {l.strike}{l.right} {l.expiration} — in at <span className="num soft">{l.entry_price?.toFixed(2)}</span>
                      {l.price_source === "trade_print" ? <> (trade print {l.entry_print?.toFixed(2)}; no bid/ask in this data)</> : l.entry_bid !== null && l.entry_bid !== undefined ? <> (bid {l.entry_bid?.toFixed(2)} / ask {l.entry_ask?.toFixed(2)})</> : l.price_source === "model" ? " (model price)" : ""}
                      , out at <span className="num soft">{l.exit_price?.toFixed(2)}</span>{l.price_source === "trade_print" && l.exit_print !== null && l.exit_print !== undefined ? ` (trade print ${l.exit_print.toFixed(2)})` : ""}{l.exit_reason && l.exit_reason !== t.exit_reason ? `, ${String(l.exit_reason).replace(/_/g, " ")}` : ""}
                    </div>
                  ))}
                  {(t.exit_explain || []).slice(0, 2).map((x: string, i: number) => <div key={`x${i}`}>exit trigger: {x}</div>)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {d.observation && (
        <div className="border-t border-line px-4 py-3 text-[13px]">
          <div className="eyebrow mb-1.5">What was measured</div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 soft">
            <span>start {hhmmOf(d.observation.entry_minute ?? 570)} at <span className="num">{d.observation.entry_price}</span></span>
            {d.observation.target_level !== null && <span>target <span className="num">{Number(d.observation.target_level).toFixed(2)}</span></span>}
            {d.observation.stop_level !== null && d.observation.stop_level !== undefined && <span>stop <span className="num">{Number(d.observation.stop_level).toFixed(2)}</span></span>}
            <span>best {sgn(d.observation.mfe_pct, 2, "%")}</span><span>worst {sgn(d.observation.mae_pct, 2, "%")}</span><span>ended {sgn(d.observation.end_return_pct, 2, "%")}</span>
            {d.observation.minutes_to_target !== null && d.observation.minutes_to_target !== undefined && <span>target after {Math.round(d.observation.minutes_to_target)} min</span>}
          </div>
        </div>
      )}
      <Rules r={d.rules} />
    </div>
  );
}

export function Rules({ r }: { r: any }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!r || !(r.rules || []).length) return r?.summary ? <div className="border-t border-line px-4 py-3 text-[13px] dim">{r.summary}</div> : null;
  return (
    <div className="border-t border-line px-4 py-3" data-testid="rules">
      <div className="mb-2 flex items-center gap-2"><span className="eyebrow">How each rule evaluated</span><span className="text-xs dim">{r.evaluations} evaluation{r.evaluations === 1 ? "" : "s"} on {r.resolution} bars</span></div>
      <div className="flex flex-col">
        {r.rules.map((x: any, i: number) => {
          const at = x.at_signal;
          const state = at ? at.passed : x.held_ticks > 0 ? null : false;
          return (
            <div key={i} className="border-b border-line/60 last:border-0">
              <button className="flex w-full items-start gap-2.5 py-2 text-left" onClick={() => setOpen((o) => (o === i ? null : i))}>
                <span className={cn("mt-0.5 grid size-4 shrink-0 place-items-center rounded-full", state === true ? "bg-up/15 text-up" : state === false ? "bg-down/15 text-down" : "bg-warn/15 text-warn")}>
                  {state === true ? <Check className="size-2.5" strokeWidth={3.5} /> : state === false ? <X className="size-2.5" strokeWidth={3.5} /> : <Minus className="size-2.5" strokeWidth={3.5} />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[13px]">{x.kind === "stage" && <span className="mr-1.5 rounded bg-synth/15 px-1.5 py-px text-[10.5px] font-medium text-synth">step</span>}<code className="font-mono text-[12px]">{x.label}</code></span>
                  <span className="block text-xs dim">{at ? at.text : x.last}</span>
                </span>
                <span className="shrink-0 text-right text-xs dim">{x.first_true ? <>first true <span className="num soft">{x.first_true}</span></> : "never true"}<br />{x.held_ticks} of {x.ticks}{x.not_evaluable_ticks ? `, ${x.not_evaluable_ticks} n/a` : ""}</span>
              </button>
              {open === i && x.changes.length > 0 && (
                <div className="mb-2 ml-6 flex flex-col gap-0.5 rounded-lg bg-bg-2 px-3 py-2 text-xs">
                  {x.changes.map((c: any, j: number) => <div key={j} className="flex gap-2"><span className="num w-10 shrink-0 dim">{c.time}</span><span className={cn("w-10 shrink-0", c.passed === true ? "text-up" : c.passed === false ? "text-down" : "text-warn")}>{c.passed === true ? "true" : c.passed === false ? "false" : "n/a"}</span><span className="soft">{c.text}</span></div>)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
