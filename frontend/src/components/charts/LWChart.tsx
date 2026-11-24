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

export function LWChart({ series, height = 260, timeVisible = false, className, onCrosshair, onClick, fit = true, extra, cursorTime }: {
  series: LWSeries[]; height?: number | string; timeVisible?: boolean; className?: string;
  onCrosshair?: (t: number | null) => void; onClick?: (t: number | null) => void; fit?: boolean; extra?: Record<string, any>; cursorTime?: number | null;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const handles = useRef<ISeriesApi<any>[]>([]);
  const cbs = useRef({ onCrosshair, onClick });
  cbs.current = { onCrosshair, onClick };

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, { ...BASE, autoSize: true, ...(extra || {}), timeScale: { ...BASE.timeScale, timeVisible, ...(extra?.timeScale || {}) } } as any);
    chart.current = c;
    c.subscribeCrosshairMove((p) => cbs.current.onCrosshair?.(p.time ? (p.time as number) : null));
    c.subscribeClick((p) => cbs.current.onClick?.(p.time ? (p.time as number) : null));
    return () => {
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

  return <div ref={el} className={className} style={{ height, width: "100%" }} />;
}
