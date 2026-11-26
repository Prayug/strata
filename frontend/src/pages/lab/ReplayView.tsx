/* Market replay on stored observations.

   Two charts over the same minutes: the underlying's stored bars with the levels the engine tests rules against, and one
   option contract's stored trade bars. A minute in which the contract did not trade has no bar: the gap is real, and the
   position's mark across it says how old it is. A trade of a run is laid over both with the minute it was decided, the
   minute it was filled and the age of the print it was filled on. The scrubber moves one cursor through both charts; with
   "only what was known" on, nothing after the cursor is drawn. */
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Eye, EyeOff, Loader2, Pause, Play, SkipBack } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { LWChart, etChartTime, type LWRange, type LWSeries } from "@/components/charts/LWChart";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";

const usd = (v: number | null | undefined, dp = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : `${v < 0 ? "−" : v > 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`);
const clock = (t: number) => new Date(t * 1000).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour12: false, hour: "2-digit", minute: "2-digit" });
const stamp = (t: number) => new Date(t * 1000).toLocaleString("en-US", { timeZone: "America/New_York", hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
const age = (s: number | null | undefined) => (s == null ? "–" : s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const AXIS = { rightPriceScale: { borderColor: "#1f2631", scaleMargins: { top: 0.08, bottom: 0.08 }, minimumWidth: 72 } };  // one axis width: the same minute sits at the same x in every chart
const LEVELS: [string, string][] = [["vwap", "VWAP"], ["ema9", "EMA 9"], ["ema21", "EMA 21"], ["or", "Opening range"], ["prev", "Yesterday"]];

type Props = { runId?: string; tradeId?: number; symbol?: string; date?: string; onTrade?: (tid: number) => void; tradeCount?: number };

export function ReplayView({ runId, tradeId, symbol, date, onTrade, tradeCount }: Props) {
  const tradeMode = !!runId && tradeId != null;
  const tq = useQuery({ queryKey: ["replay-trade", runId, tradeId], queryFn: () => api.get<any>(`/api/backtests/${runId}/trades/${tradeId}/observed`), enabled: tradeMode, retry: false });
  const sym = (tradeMode ? tq.data?.trade?.symbol : symbol) as string | undefined;
  const day = (tradeMode ? tq.data?.trade?.entry_date : date) as string | undefined;
  const sq = useQuery({ queryKey: ["replay-session", sym, day], queryFn: () => api.get<any>(`/api/replay/${sym}/${day}`), enabled: !!sym && !!day && (!tradeMode || !!tq.data), staleTime: 300_000 });
  const [occ, setOcc] = useState<string | null>(null);  // a contract other than the trade's own
  const [leg, setLeg] = useState(0);
  const und = tradeMode ? tq.data?.underlying : sq.data?.underlying;
  const legs: any[] = tq.data?.legs || [];
  const own = legs[leg];
  const cq = useQuery({ queryKey: ["replay-contract", sym, occ, und?.start, und?.end], queryFn: () => api.get<any>(`/api/replay/${sym}/contract/${occ}?start=${und.start}&end=${und.end}${und.aggregated ? `&bucket=${und.candle_seconds}` : ""}`), enabled: !!occ && !!und });
  useEffect(() => { setOcc(null); setLeg(0); }, [runId, tradeId, sym, day]);
  const exps: any[] = sq.data?.contracts?.expirations || [];
  // with no trade to follow, start on the at-the-money call of the nearest expiration
  useEffect(() => {
    if (tradeMode || occ || !exps.length || !und?.sessions?.[0]?.open) return;
    const rows = (exps[0].rows as any[]).filter((r) => r.traded && r.right === "C");
    const best = rows.sort((a, b) => Math.abs(a.strike - und.sessions[0].open) - Math.abs(b.strike - und.sessions[0].open))[0];
    if (best) setOcc(best.occ);
  }, [exps.length, und?.start]); // eslint-disable-line
  const contract = occ ? cq.data : own ? { occ: own.occ, strike: own.strike, right: own.right, expiration: own.expiration, bars: own.bars, candle_seconds: own.candle_seconds, sessions: own.sessions } : null;
  const onOwn = tradeMode && !occ;

  const [on, setOn] = useState<Record<string, boolean>>({ vwap: true, ema9: false, ema21: false, or: true, prev: true });
  const [cur, setCur] = useState<number | null>(null);   // index into the underlying's candles
  const [playing, setPlaying] = useState(false);
  const [known, setKnown] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [range, setRange] = useState<LWRange | null>(null);
  // the charts share the height there is. With fixed heights the contract's chart sat below the fold on a laptop screen, and
  // a wheel over a chart zooms it instead of scrolling to the next one
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const [avail, setAvail] = useState(0);
  useEffect(() => {
    if (!box) return;
    const ro = new ResizeObserver(() => setAvail(box.clientHeight));
    ro.observe(box);
    setAvail(box.clientHeight);
    return () => ro.disconnect();
  }, [box]);
  const candles: number[][] = und?.candles || [];
  const n = candles.length;
  const times = useMemo(() => candles.map((c) => c[0]), [und]);
  const sec: number = und?.candle_seconds || 60;
  useEffect(() => { setCur(null); setPlaying(false); setRange(null); }, [und?.start, und?.end, sym]);
  const timer = useRef<any>(null);
  useEffect(() => {
    clearInterval(timer.current);
    if (!playing || !n) return;
    timer.current = setInterval(() => setCur((i) => { const j = (i ?? -1) + 1; if (j >= n) { setPlaying(false); return n - 1; } return j; }), Math.max(16, 1000 / speed));
    return () => clearInterval(timer.current);
  }, [playing, n, speed]);
  // the whole span, as positions on the time scale: every chart is given the same window, so a minute is at the same x in each
  // and, with "only what was known", the bars advance from left to right instead of the chart refitting to what is drawn
  const full = useMemo(() => ({ from: -0.5, to: Math.max(1, n) - 0.5 }), [n]);
  const idx = cur ?? (n ? n - 1 : 0);
  const tNow = n ? times[idx] + sec : 0;               // what had happened by the END of the candle under the cursor
  const cut = known ? idx : n - 1;

  // the contract's bars placed on the underlying's minutes: a minute without a print stays empty
  const optBy = useMemo(() => { const m = new Map<number, number[]>(); for (const b of contract?.bars || []) m.set(Math.floor(b[0] / sec) * sec, b); return m; }, [contract, sec]);
  const marksAt = useMemo(() => { const m = new Map<number, number>(); (tq.data?.marks?.t || []).forEach((t: number, i: number) => m.set(t, i)); return m; }, [tq.data]);
  const markers = (tq.data?.markers || []) as any[];
  const mk = (kind: string) => markers.find((m) => m.kind === kind);
  const snap = (t: number) => { let lo = 0, hi = n - 1; while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (times[mid] <= t) lo = mid; else hi = mid - 1; } return times[lo]; };

  const undSeries: LWSeries[] = useMemo(() => {
    if (!n) return [];
    const ses = und.sessions?.find((s: any) => s.stored) || {};
    const T = (i: number) => etChartTime(times[i]);
    const line = (key: string, color: string, w = 1) => ({ type: "line" as const, data: candles.map((_, i) => (i <= cut && und[key][i] != null ? { time: T(i), value: und[key][i] } : { time: T(i) })), options: { color, lineWidth: w, lastValueVisible: false, crosshairMarkerVisible: false } });
    const priceLines: any[] = [];
    if (on.prev && ses.previous) priceLines.push({ price: ses.previous.close, color: "#7a8496", title: "prev close" }, { price: ses.previous.high, color: "#535d6e", title: "prev high" }, { price: ses.previous.low, color: "#535d6e", title: "prev low" });
    if (on.or && ses.opening_range?.["15"] && und.sessions.length === 1) priceLines.push({ price: ses.opening_range["15"].high, color: "#4c8dff", title: "first 15m high" }, { price: ses.opening_range["15"].low, color: "#4c8dff", title: "first 15m low" });
    for (const l of (onOwn || tradeMode ? tq.data?.levels?.underlying || [] : [])) if (l.price) priceLines.push({ price: l.price, color: l.rule.includes("stop") ? "#ef5350" : "#26a69a", title: l.rule });
    const ms = tradeMode ? markers.filter((m) => m.t <= times[cut] + sec).map((m) => ({ time: etChartTime(snap(m.kind === "decision" || m.kind === "exit_decision" ? m.t : m.t - 1)), position: m.kind === "exit" ? "aboveBar" : "belowBar", color: m.kind === "entry" ? "#e8a33d" : m.kind === "exit" ? "#d6dce5" : "#7a8496",
      shape: m.kind === "entry" ? "arrowUp" : m.kind === "exit" ? "arrowDown" : "circle", text: m.kind === "decision" ? "rule held" : m.kind === "entry" ? "in" : m.kind === "exit" ? "out" : m.kind.replace("_", " ") })) : [];
    const out: LWSeries[] = [{ type: "candles", data: candles.map((c, i) => (i <= cut ? { time: T(i), open: c[1], high: c[2], low: c[3], close: c[4] } : { time: T(i) })), priceLines, markers: ms as any }];
    if (on.vwap) out.push(line("vwap", "#e8a33d", 1.5));
    if (on.ema9) out.push(line("ema9", "#4c8dff"));
    if (on.ema21) out.push(line("ema21", "#c678dd"));
    return out;
  }, [und, on, cut, tq.data, onOwn]); // eslint-disable-line

  const optSeries: LWSeries[] = useMemo(() => {
    if (!n || !contract) return [];
    const T = (i: number) => etChartTime(times[i]);
    const priceLines: any[] = [];
    const ms: any[] = [];
    if (onOwn && own) {
      priceLines.push({ price: own.entry_price, color: "#e8a33d", title: `${own.side === "buy" ? "paid" : "received"} ${own.entry_price.toFixed(2)}`, style: "solid" });
      for (const l of tq.data?.levels?.premium || []) if (l.price) priceLines.push({ price: l.price, color: l.rule.includes("stop") ? "#ef5350" : "#26a69a", title: l.rule });
      const e = mk("entry"), x = mk("exit");
      if (e && e.t <= times[cut] + sec) ms.push({ time: etChartTime(snap(e.t - 1)), position: "belowBar", color: "#e8a33d", shape: "arrowUp", text: `${own.side} ${own.entry_price.toFixed(2)}` });
      if (x && x.t <= times[cut] + sec && own.exit_price != null) ms.push({ time: etChartTime(snap(x.t - 1)), position: "aboveBar", color: "#d6dce5", shape: "arrowDown", text: `${own.side === "buy" ? "sell" : "buy"} ${own.exit_price.toFixed(2)}` });
    }
    return [
      { type: "candles", data: candles.map((_, i) => { const b = optBy.get(times[i]); return b && i <= cut ? { time: T(i), open: b[1], high: b[2], low: b[3], close: b[4] } : { time: T(i) }; }), priceLines, markers: ms },
      { type: "histogram", pane: 1, data: candles.map((_, i) => { const b = optBy.get(times[i]); return b && i <= cut ? { time: T(i), value: b[5], color: b[4] >= b[1] ? "rgba(38,166,154,0.55)" : "rgba(239,83,80,0.55)" } : { time: T(i) }; }) },
    ];
  }, [contract, optBy, cut, onOwn, tq.data]); // eslint-disable-line

  const pnlSeries: LWSeries[] = useMemo(() => {
    const m = tq.data?.marks;
    if (!tradeMode || !m?.t?.length || !n) return [];
    return [{ type: "baseline", data: candles.map((_, i) => { const j = marksAt.get(times[i] + sec) ?? marksAt.get(times[i]); return j != null && m.pnl[j] != null && i <= cut ? { time: etChartTime(times[i]), value: m.pnl[j] } : { time: etChartTime(times[i]) }; }), options: { baseValue: { type: "price", price: 0 } } }];
  }, [tq.data, cut, n]); // eslint-disable-line

  if (tradeMode && tq.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Reading the stored observations of this trade…</div>;
  if (tradeMode && (tq.isError || !tq.data)) return <div className="p-6 text-[13px] dim">This trade cannot be replayed: {String((tq.error as any)?.message || "no stored trade")}</div>;
  if (!tradeMode && sq.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Reading the session…</div>;
  if (!n) return <div className="p-6 text-[13px] dim">{und?.note || "No intraday bars are stored for this session."}</div>;

  const t = tq.data?.trade;
  const c = candles[idx];
  // state at the cursor: the last print of the contract that had become observable, and the position's mark
  let last: number[] | null = null;
  for (const b of contract?.bars || []) { if (b[6] <= tNow) last = b; else break; }
  const mi = tradeMode ? (marksAt.get(tNow) ?? null) : null;
  const pnlNow = mi != null ? tq.data.marks.pnl[mi] : null;
  const ageNow = mi != null ? tq.data.marks.oldest_print_seconds[mi] : last ? tNow - last[6] : null;
  const expSel = exps.find((e) => e.rows.some((r: any) => r.occ === (occ || own?.occ)));
  const cursorTime = cur == null ? undefined : etChartTime(times[idx]);
  const hover = (tt: number | null) => { if (tt == null || playing) return; let lo = 0, hi = n - 1; while (lo < hi) { const mid = (lo + hi + 1) >> 1; if ((etChartTime(times[mid]) as number) <= tt) lo = mid; else hi = mid - 1; } setCur(lo); };

  // 250 / 230 / 110 where there is room; otherwise the two price charts split what there is (never under 132 and 150: below
  // that a candle is a pixel) and the marked position follows them, one scroll away
  const labels = 22;
  const room = Math.max(0, avail - 2 * labels - 2);
  const fits = avail === 0 || avail >= 250 + 230 + (tradeMode ? 110 + labels : 0) + 2 * labels;
  const hU = fits ? 250 : Math.max(132, Math.round(room * 0.47));
  const hO = fits ? 230 : Math.max(150, room - Math.max(132, Math.round(room * 0.47)));
  const hP = fits ? 110 : 96;
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="view-replay">
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line px-4 py-2">
        {tradeMode && onTrade && <span className="flex items-center gap-0.5"><button className="btn quiet sm !px-1.5" disabled={tradeId! <= 1} onClick={() => onTrade(tradeId! - 1)} aria-label="Previous trade"><ChevronLeft /></button><span className="num text-xs dim">trade {tradeId}{tradeCount ? ` / ${tradeCount}` : ""}</span><button className="btn quiet sm !px-1.5" disabled={!!tradeCount && tradeId! >= tradeCount} onClick={() => onTrade(tradeId! + 1)} aria-label="Next trade"><ChevronRight /></button></span>}
        <span className="num text-[13px] font-medium">{sym} · {und.start}{und.end !== und.start ? ` → ${und.end}` : ""}</span>
        {t && <span className={cn("num text-[13px]", t.pnl > 0 ? "text-up" : "text-down")}>{usd(t.pnl)}</span>}
        {t && <span className="text-xs dim">{String(t.exit_reason || "").replace(/_/g, " ")} · {t.units} × {legs.map((l) => `${l.side === "buy" ? "+" : "−"}${l.strike}${l.right}`).join(" ")}</span>}
        <span className="flex-1" />
        {LEVELS.map(([k, l]) => <button key={k} className={cn("chip", on[k] && "on")} onClick={() => setOn((o) => ({ ...o, [k]: !o[k] }))}>{l}</button>)}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1.5 border-b border-line bg-bg-1 px-4 py-1.5 text-xs" data-testid="replay-contract-bar">
        <span className="dim">Contract</span>
        {legs.map((l, i) => <button key={l.occ} className={cn("chip", onOwn && leg === i && "on")} onClick={() => { setOcc(null); setLeg(i); }}>{l.side === "buy" ? "+" : "−"}{l.strike}{l.right} {l.expiration.slice(5)} <span className="dim">this trade</span></button>)}
        {exps.length > 0 && (
          <>
            <select className="field !h-7 !w-auto" aria-label="Expiration" value={expSel?.expiration || ""} onChange={(e) => { const ex = exps.find((x) => x.expiration === e.target.value); const base = contract?.strike ?? und.sessions[0].open; const right = contract?.right || "C";
              const r = (ex.rows as any[]).filter((x) => x.right === right && x.traded).sort((a, b) => Math.abs(a.strike - base) - Math.abs(b.strike - base))[0]; if (r) setOcc(r.occ); }}>
              {!expSel && <option value="">expiration…</option>}{exps.map((e) => <option key={e.expiration} value={e.expiration}>{e.expiration} · {e.dte}DTE · {e.traded} traded</option>)}</select>
            {expSel && <select className="field !h-7 !w-auto" aria-label="Strike" value={occ || own?.occ || ""} onChange={(e) => setOcc(e.target.value)}>
              {(expSel.rows as any[]).filter((r) => r.right === (contract?.right || "C")).map((r) => <option key={r.occ} value={r.occ} disabled={!r.traded}>{r.strike}{r.right}{r.traded ? ` · ${r.bars} min` : " · no print"}</option>)}</select>}
            {contract && <button className="chip" onClick={() => { const other = (expSel?.rows as any[] | undefined)?.find((r) => r.strike === contract.strike && r.right !== contract.right && r.traded); if (other) setOcc(other.occ); }}>{contract.right === "C" ? "calls → puts" : "puts → calls"}</button>}
          </>
        )}
        <span className="flex-1" />
        {contract && <span className="num dim">{contract.bars?.length ?? 0} of {n} {sec === 60 ? "minutes" : "candles"} have a print{sec > 60 ? ` · ${sec / 60}-minute candles` : ""}</span>}
      </div>
      <div className="min-h-0 flex-1 overflow-auto" ref={setBox}>
        <div className="border-b border-line"><div className="px-4 pt-1.5 text-[10.5px] uppercase tracking-[0.07em] text-muted">{sym} · stored {und.stored} bars</div>
          <LWChart series={undSeries} height={hU} timeVisible fit={false} onCrosshair={hover} cursorTime={cursorTime as any} range={range || full} onRange={setRange} extra={AXIS} /></div>
        <div className="border-b border-line"><div className="px-4 pt-1.5 text-[10.5px] uppercase tracking-[0.07em] text-muted">{contract ? `${contract.strike}${contract.right} expiring ${contract.expiration} · trade prints (no bid/ask) · volume below` : "pick a contract"}</div>
          {occ && cq.isLoading ? <div className="flex items-center justify-center text-xs dim" style={{ height: hO }}><Loader2 className="mr-2 size-4 animate-spin" />Reading this contract's prints…</div>
            : <LWChart series={optSeries} height={hO} timeVisible fit={false} onCrosshair={hover} cursorTime={cursorTime as any} range={range || full} onRange={setRange} extra={AXIS} />}</div>
        {tradeMode && pnlSeries.length > 0 && <div className="border-b border-line"><div className="px-4 pt-1.5 text-[10.5px] uppercase tracking-[0.07em] text-muted">Position marked on its stored prints (profit before costs)</div>
          <LWChart series={pnlSeries} height={hP} timeVisible fit={false} onCrosshair={hover} cursorTime={cursorTime as any} range={range || full} onRange={setRange} extra={AXIS} /></div>}
        <div className={cn("grid grid-cols-1 gap-px bg-line", tradeMode && "lg:grid-cols-2")}>
          {tradeMode && <div className="bg-bg-0 px-4 py-3 text-[12.5px]">
            <div className="eyebrow mb-1.5">What happened, in order</div>
            {markers.map((m, i) => <div key={i} className="flex gap-2 py-0.5"><span className="num w-[92px] shrink-0 dim">{stamp(m.t)}</span><span className={cn(m.kind === "entry" && "text-accent")}>{m.kind.replace("_", " ")}: <span className="soft">{m.text}</span>{m.print_ts ? <span className="dim"> (print at {clock(m.print_ts)})</span> : null}</span></div>)}
            <div className="eyebrow mb-1 mt-3">Why it entered</div>
            {(tq.data.entry_explain || []).map((e: any, i: number) => <div key={i} className="num py-px text-[11.5px] soft">{e.selection ? "· " : e.passed ? "✓ " : "✕ "}{e.text}</div>)}
            <div className="eyebrow mb-1 mt-3">Why it closed</div>
            {(tq.data.exit_explain || []).map((e: any, i: number) => <div key={i} className="num py-px text-[11.5px] soft">{e.text}</div>)}
          </div>}
          <div className="bg-bg-0 px-4 py-3 text-[12px] dim">
            <div className="eyebrow mb-1.5">How to read this</div>
            <p>{tradeMode ? tq.data.reading : "Each option bar is the trades of one minute in which the contract traded. A minute without a bar is a minute without a trade: nothing is drawn across it. These are trade prints, not quotes."}</p>
            <p className="mt-1.5">{und.levels} {und.note}</p>
            {(contract?.sessions || []).map((s: any) => <div key={s.date} className="num mt-1">{s.date}: {s.stored ? `${s.bars} bars, ${s.volume?.toLocaleString() ?? 0} contracts, prints in ${s.minutes_with_prints_pct}% of the session's minutes` : "no option data stored"}</div>)}
          </div>
        </div>
      </div>
      <div className="shrink-0 border-t border-line bg-bg-1 px-4 py-2" data-testid="replay-scrubber">
        <div className="flex items-center gap-2">
          <button className="btn quiet sm !px-1.5" onClick={() => { setCur(0); setPlaying(false); }} aria-label="To the open"><SkipBack /></button>
          <button className="btn sm !px-2" onClick={() => { if (cur == null || idx >= n - 1) setCur(0); setPlaying((p) => !p); }} aria-label={playing ? "Pause" : "Play"} data-testid="replay-play">{playing ? <Pause /> : <Play />}</button>
          <input type="range" min={0} max={n - 1} value={idx} onChange={(e) => { setPlaying(false); setCur(Number(e.target.value)); }} className="h-1.5 min-w-0 flex-1 cursor-pointer accent-[#e8a33d]" aria-label="Time" data-testid="replay-slider" />
          <select className="field !h-7 !w-auto" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Speed">{[2, 8, 20, 60].map((s) => <option key={s} value={s}>{s} {sec === 60 ? "min" : "bars"}/s</option>)}</select>
          <button className={cn("chip", known && "on")} onClick={() => setKnown((k) => !k)} title="Draw nothing after the cursor: the charts show only what had happened by then" data-testid="replay-known">{known ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}only what was known</button>
          {range && (Math.abs(range.from - full.from) > 1 || Math.abs(range.to - full.to) > 1) && <button className="chip" onClick={() => setRange(null)} title="Show the whole span again">whole {und.sessions?.length > 1 ? "span" : "session"}</button>}
        </div>
        <div className="num mt-1.5 flex flex-wrap gap-x-5 gap-y-0.5 text-[11.5px]" data-testid="replay-readout">
          <span className="text-fg">{stamp(tNow)} ET</span>
          <span><span className="dim">{sym} </span>O {c[1]?.toFixed(2)} H {c[2]?.toFixed(2)} L {c[3]?.toFixed(2)} C {c[4]?.toFixed(2)}</span>
          {und.vwap?.[idx] != null && <span><span className="dim">VWAP </span>{und.vwap[idx].toFixed(2)}</span>}
          {contract && <span><span className="dim">{contract.strike}{contract.right} last print </span>{last ? `${last[4].toFixed(2)} at ${clock(last[6] - 1)}` : "none yet"}{last ? <span className={cn((tNow - last[6]) > 300 ? "text-warn" : "dim")}> ({age(tNow - last[6])} old)</span> : null}</span>}
          {tradeMode && <span><span className="dim">position </span>{pnlNow != null ? <span className={pnlNow > 0 ? "text-up" : pnlNow < 0 ? "text-down" : ""}>{usd(pnlNow)}</span> : tNow < (mk("entry")?.t ?? 0) ? "not open yet" : tNow > (mk("exit")?.t ?? Infinity) ? "closed" : "no mark"}{pnlNow != null && ageNow != null && ageNow > 300 ? <span className="text-warn"> on a print {age(ageNow)} old</span> : null}</span>}
        </div>
      </div>
    </div>
  );
}

/* The replay as a page of its own: any stored session and contract, without a run. */
export function ReplayExplorer({ initial }: { initial?: { symbol?: string; date?: string } }) {
  const [sym, setSym] = useState(initial?.symbol || "SPY");
  const sq = useQuery({ queryKey: ["replay-sessions", sym], queryFn: () => api.get<any>(`/api/replay/${sym}/sessions`) });
  const [day, setDay] = useState<string | null>(initial?.date || null);
  const days: any[] = sq.data?.sessions || [];
  const d = day && days.some((x) => x.date === day) ? day : days[0]?.date;
  const i = days.findIndex((x) => x.date === d);
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="replay-explorer">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-line px-4 py-2 text-xs">
        <span className="eyebrow">Stored sessions</span>
        {["SPY", "QQQ", "IWM"].map((s) => <button key={s} className={cn("chip", sym === s && "on")} onClick={() => { setSym(s); setDay(null); }}>{s}</button>)}
        <button className="btn quiet sm !px-1.5" disabled={i < 0 || i >= days.length - 1} onClick={() => setDay(days[i + 1].date)} aria-label="Earlier session"><ChevronLeft /></button>
        <input type="date" className="field !h-7 !w-auto" value={d || ""} min={sq.data?.first} max={sq.data?.last} onChange={(e) => setDay(e.target.value)} aria-label="Session" />
        <button className="btn quiet sm !px-1.5" disabled={i <= 0} onClick={() => setDay(days[i - 1].date)} aria-label="Later session"><ChevronRight /></button>
        <span className="dim">{sq.data?.count ? `${sq.data.count} sessions from ${sq.data.first} to ${sq.data.last}` : sq.isLoading ? "…" : sq.data?.note}{i >= 0 ? ` · ${days[i].contracts} contracts, ${days[i].bars.toLocaleString()} bars this session` : day ? " · no option data stored on that date" : ""}</span>
      </div>
      <div className="min-h-0 flex-1">{d ? <ReplayView key={`${sym}-${d}`} symbol={sym} date={d} /> : <div className="p-6 text-[13px] dim">{sq.isLoading ? "Loading…" : `No option observations are stored for ${sym}.`}</div>}</div>
    </div>
  );
}
