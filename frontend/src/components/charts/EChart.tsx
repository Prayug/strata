/* ECharts wrapper with the terminal theme. Used for statistical charts. */
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import { CalendarComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([BarChart, LineChart, ScatterChart, HeatmapChart, CustomChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent,
  VisualMapComponent, MarkLineComponent, CalendarComponent, CanvasRenderer]);

export const C = {
  up: "#26a69a", down: "#ef5350", accent: "#e8a33d", info: "#4c8dff", muted: "#7a8496", faint: "#535d6e", line: "#1f2631", fg: "#d6dce5", synth: "#b980ff",
  palette: ["#e8a33d", "#4c8dff", "#26a69a", "#c678dd", "#ef5350", "#56b6c2", "#d19a66", "#98c379", "#e06c75", "#61afef"],
};

const THEME = {
  color: C.palette,
  backgroundColor: "transparent",
  textStyle: { fontFamily: "Inter, sans-serif", fontSize: 11, color: C.muted },
  grid: { left: 52, right: 16, top: 26, bottom: 30, containLabel: false },
  title: { textStyle: { color: C.fg, fontSize: 12, fontWeight: 500 }, left: 4, top: 0 },
  legend: { textStyle: { color: C.muted, fontSize: 10.5 }, itemWidth: 10, itemHeight: 6, top: 0, right: 4, icon: "rect" },
  tooltip: { backgroundColor: "#1a202b", borderColor: "#2b3442", textStyle: { color: C.fg, fontSize: 11 }, confine: true },
  categoryAxis: { axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false }, axisLabel: { color: C.muted, fontSize: 10, fontFamily: "JetBrains Mono" }, splitLine: { show: false } },
  valueAxis: { axisLine: { show: false }, axisLabel: { color: C.muted, fontSize: 10, fontFamily: "JetBrains Mono" }, splitLine: { lineStyle: { color: "rgba(31,38,49,0.7)" } }, nameTextStyle: { color: C.faint, fontSize: 10 } },
  timeAxis: { axisLine: { lineStyle: { color: C.line } }, axisLabel: { color: C.muted, fontSize: 10 }, splitLine: { show: false } },
};
echarts.registerTheme("strata", THEME);

export function EChart({ option, height = 240, className, onClick, notMerge = true }: { option: any; height?: number | string; className?: string; onClick?: (p: any) => void; notMerge?: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const cb = useRef(onClick);
  cb.current = onClick;
  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, "strata", { renderer: "canvas" });
    inst.current = c;
    c.on("click", (p) => cb.current?.(p));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
    };
  }, []);
  useEffect(() => {
    inst.current?.setOption(option, { notMerge });
  }, [option, notMerge]);
  return <div ref={el} className={className} style={{ height, width: "100%" }} />;
}

export const axisPct = { axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } };
export const fmtMoneyAxis = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1)}k` : v.toFixed(0));
