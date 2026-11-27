/* Thin React wrapper around TradingView Lightweight Charts v5. */
import {
  AreaSeries, BaselineSeries, CandlestickSeries, ColorType, createChart, createSeriesMarkers, CrosshairMode, HistogramSeries, LineSeries, LineStyle,
  type IChartApi, type ISeriesApi, type SeriesMarker, type Time, type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

export type LWSeries = {
  type: "line" | "area" | "histogram" | "candles" | "baseline";
  data: any[];
  options?: Record<string, any>;
  markers?: SeriesMarker<Time>[];
  priceLines?: { price: number; color: string; title?: string; style?: "solid" | "dashed" }[];
  pane?: number;
};

const offsets = new Map<number, number>();
/** Shift a UTC epoch so the chart's UTC axis reads as New York wall-clock time. */
export function etChartTime(ts: number): UTCTimestamp {
  const day = Math.floor(ts / 86400);
  let off = offsets.get(day);
  if (off === undefined) {
    const d = new Date(ts * 1000);
    const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).formatToParts(d);
    const g = (t: string) => Number(parts.find((p) => p.type === t)?.value);
    const asUtc = Date.UTC(g("year"), g("month") - 1, g("day"), g("hour"), g("minute"), g("second")) / 1000;
    off = asUtc - ts;
    offsets.set(day, off);
  }
  return (ts + off) as UTCTimestamp;
}
export const dayTime = (iso: string) => iso as unknown as Time;

const BASE = {
  layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#7a8496", fontSize: 10.5, fontFamily: "JetBrains Mono, monospace", attributionLogo: false },
  grid: { vertLines: { color: "rgba(31,38,49,0.55)" }, horzLines: { color: "rgba(31,38,49,0.55)" } },
  rightPriceScale: { borderColor: "#1f2631", scaleMargins: { top: 0.08, bottom: 0.08 } },
  timeScale: { borderColor: "#1f2631", secondsVisible: false },
  crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "#535d6e", labelBackgroundColor: "#222a37" }, horzLine: { color: "#535d6e", labelBackgroundColor: "#222a37" } },
};

export type LWRange = { from: number; to: number };

/** `range` / `onRange` share one visible window between charts drawn over the same time points (bars are positions on the
    time scale, so two charts with the same points and the same logical range show the same minutes one above the other). */
export function LWChart({ series, height = 260, timeVisible = false, className, onCrosshair, onClick, fit = true, extra, cursorTime, range, onRange }: {
  series: LWSeries[]; height?: number | string; timeVisible?: boolean; className?: string;
  onCrosshair?: (t: number | null) => void; onClick?: (t: number | null) => void; fit?: boolean; extra?: Record<string, any>; cursorTime?: number | null;
  range?: LWRange | null; onRange?: (r: LWRange) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const handles = useRef<ISeriesApi<any>[]>([]);
  const cbs = useRef({ onCrosshair, onClick, onRange });
  cbs.current = { onCrosshair, onClick, onRange };
  const quiet = useRef(0);  // until when (ms) range changes are this chart's own doing (data set, fitted, a range it was given) and not a gesture
  // A range change is shared only while a hand is on this chart (wheel, drag, touch). A chart that is resized (a panel
  // opening, a window getting wider) moves its own visible range: that is not a gesture, and other charts must not follow it.
  const gesture = useRef(0);
  const rangeRef = useRef<LWRange | null | undefined>(range);
  rangeRef.current = range;

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, { ...BASE, autoSize: true, ...(extra || {}), timeScale: { ...BASE.timeScale, timeVisible, ...(extra?.timeScale || {}) } } as any);
    chart.current = c;
    // Only a pointer that really moved (or left) is reported. The library also fires this when the chart redraws under a
    // pointer that is standing still: with the mouse parked on a chart, the replay's cursor snapped back to it every time
    // the slider or the keyboard moved it.
    c.subscribeCrosshairMove((p) => { if (p.time && !(p as any).sourceEvent) return; cbs.current.onCrosshair?.(p.time ? (p.time as number) : null); });
    c.subscribeClick((p) => cbs.current.onClick?.(p.time ? (p.time as number) : null));
    c.timeScale().subscribeVisibleLogicalRangeChange((r) => {
      if (!r || !cbs.current.onRange || performance.now() < quiet.current || performance.now() > gesture.current) return;
      cbs.current.onRange({ from: r.from, to: r.to });
    });
    const host = el.current;
    const brief = () => { gesture.current = performance.now() + 900; };
    const hold = () => { gesture.current = Number.POSITIVE_INFINITY; };
    const release = () => { if (gesture.current === Number.POSITIVE_INFINITY) gesture.current = performance.now() + 300; };
    host.addEventListener("wheel", brief, { passive: true });
    host.addEventListener("pointerdown", hold);
    host.addEventListener("touchstart", hold, { passive: true });
    window.addEventListener("pointerup", release);
    window.addEventListener("touchend", release);
    // after a resize, the window it was given is put back (the library keeps the bar spacing and moves the window instead)
    const ro = new ResizeObserver(() => {
      const want = rangeRef.current;
      if (!want) return;
      quiet.current = performance.now() + 250;
      requestAnimationFrame(() => { try { c.timeScale().setVisibleLogicalRange(want as any); } catch { /* no data yet */ } });
    });
    ro.observe(host);
    return () => {
      ro.disconnect();
      host.removeEventListener("wheel", brief);
      host.removeEventListener("pointerdown", hold);
      host.removeEventListener("touchstart", hold);
      window.removeEventListener("pointerup", release);
      window.removeEventListener("touchend", release);
      c.remove();
      chart.current = null;
      handles.current = [];
    };
  }, [timeVisible]);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    for (const h of handles.current) {
      try {
        c.removeSeries(h);
      } catch { /* chart rebuilt */ }
    }
    handles.current = [];
    for (const s of series) {
      const def = { line: LineSeries, area: AreaSeries, histogram: HistogramSeries, candles: CandlestickSeries, baseline: BaselineSeries }[s.type];
      const defaults: Record<string, any> = {
        line: { lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true },
        area: { lineWidth: 1.5, priceLineVisible: false },
        histogram: { priceLineVisible: false, lastValueVisible: false },
        candles: { upColor: "#26a69a", downColor: "#ef5350", borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350", priceLineVisible: false },
        baseline: { lineWidth: 1.5, priceLineVisible: false, topLineColor: "#26a69a", bottomLineColor: "#ef5350", topFillColor1: "rgba(38,166,154,0.18)", topFillColor2: "rgba(38,166,154,0.02)", bottomFillColor1: "rgba(239,83,80,0.02)", bottomFillColor2: "rgba(239,83,80,0.18)" },
      };
      const h = c.addSeries(def as any, { ...defaults[s.type], ...(s.options || {}) }, s.pane ?? 0);
      h.setData(s.data.filter((d) => d && d.time !== undefined && (d.value === undefined || Number.isFinite(d.value))));
      if (s.markers?.length) createSeriesMarkers(h, s.markers);
      for (const pl of s.priceLines || []) h.createPriceLine({ price: pl.price, color: pl.color, lineWidth: 1, lineStyle: pl.style === "solid" ? LineStyle.Solid : LineStyle.Dashed, axisLabelVisible: true, title: pl.title || "" });
      handles.current.push(h);
    }
    quiet.current = performance.now() + 250;
    if (fit) c.timeScale().fitContent();
  }, [series, fit]);

  useEffect(() => {
    const c = chart.current;
    if (!c || cursorTime === undefined) return;
    if (cursorTime === null || !handles.current[0]) {
      c.clearCrosshairPosition();
      return;
    }
    try {
      c.setCrosshairPosition(NaN as any, cursorTime as Time, handles.current[0]);
    } catch { /* outside range */ }
  }, [cursorTime]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !range) return;
    const cur = c.timeScale().getVisibleLogicalRange();
    if (cur && Math.abs(cur.from - range.from) < 0.05 && Math.abs(cur.to - range.to) < 0.05) return;
    quiet.current = performance.now() + 250;
    try {
      c.timeScale().setVisibleLogicalRange(range as any);
    } catch { /* no data yet */ }
  }, [range, series]);

  return <div ref={el} className={className} style={{ height, width: "100%" }} />;
}
