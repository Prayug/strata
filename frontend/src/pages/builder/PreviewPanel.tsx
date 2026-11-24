import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { EChart, C, fmtMoneyAxis } from "@/components/charts/EChart";
import { ErrorBox, Loading, SyntheticBadge } from "@/components/common";
import { Input, Select, Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn, money, num, pct } from "@/lib/format";
import { useBuilder } from "@/lib/store";

const TIMES = Array.from({ length: 27 }, (_, i) => {
  const m = 570 + i * 15;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
});

function useDebounced<T>(v: T, ms: number) {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

export function PreviewPanel() {
  const { strategy, previewDay, previewTime, execution, patch } = useBuilder();
  const body = useDebounced(JSON.stringify({ strategy: { name: strategy.name, symbol: strategy.symbol, legs: strategy.legs }, day: previewDay, time: previewTime, execution }), 300);
  const q = useQuery({ queryKey: ["preview", body], queryFn: () => api.post<any>("/api/strategies/preview", JSON.parse(body)), placeholderData: (p) => p });
  const d = q.data;
  const option = useMemo(() => {
    if (!d?.payoff) return null;
    const x: number[] = d.payoff.x;
    const be: number[] = d.breakevens || [];
    return {
      grid: { left: 54, right: 12, top: 22, bottom: 26 },
      legend: { data: ["At front expiration", "Now (T+0, est.)"] },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => money(v, 0, true), axisPointer: { type: "cross", label: { backgroundColor: "#222a37" } } },
      xAxis: { type: "value", min: x[0], max: x[x.length - 1], axisLabel: { formatter: (v: number) => v.toFixed(0) } },
      yAxis: { type: "value", axisLabel: { formatter: fmtMoneyAxis } },
      series: [
        {
          name: "At front expiration", type: "line", showSymbol: false, data: x.map((v, i) => [v, d.payoff.expiration[i]]), lineStyle: { width: 1.6, color: C.accent },
          areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(38,166,154,0.16)" }, { offset: 1, color: "rgba(239,83,80,0.16)" }] }, origin: 0 },
          markLine: {
            symbol: "none", silent: true, label: { color: C.muted, fontSize: 10 },
            data: [
              { yAxis: 0, lineStyle: { color: "#3a4452", type: "solid" }, label: { show: false } },
              { xAxis: d.spot, lineStyle: { color: C.info, type: "dashed" }, label: { formatter: `spot ${d.spot.toFixed(2)}`, color: C.info } },
              ...be.map((b) => ({ xAxis: b, lineStyle: { color: C.faint, type: "dotted" }, label: { formatter: `BE ${b.toFixed(1)}` } })),
            ],
          },
        },
        { name: "Now (T+0, est.)", type: "line", showSymbol: false, data: x.map((v, i) => [v, d.payoff.t0[i]]), lineStyle: { width: 1, color: C.info, type: "dashed" } },
      ],
    } as any;
  }, [d]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-line px-2.5">
        <span className="label text-fg-2">Live preview</span>
        <SyntheticBadge on={d?.synthetic} />
        <span className="flex-1" />
        <Tip content="Reference session for contract selection (latest stored session on or before this date)">
          <Input sz="sm" type="date" className="w-[118px] num" value={previewDay || d?.date || ""} min={d?.available_days?.[0]} max={d?.available_days?.[1]}
            onChange={(e) => patch({ previewDay: e.target.value || null })} />
        </Tip>
        <Select sz="sm" className="w-[64px]" value={previewTime} onChange={(v) => patch({ previewTime: v })} options={TIMES} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {q.isLoading && <Loading label="Resolving contracts" />}
        {q.error && <ErrorBox className="m-2" error={q.error} />}
        {d?.error && (
          <div className="m-2 rounded-sm border border-warn/40 bg-warn/[0.06] p-2 text-xs">
            <div className="font-medium text-warn mb-0.5">No contracts selected on {d.date} {d.time}</div>
            <div className="text-fg-2">{d.error}</div>
          </div>
        )}
        {d && !d.error && (
          <div className={cn("transition-opacity", q.isFetching && "opacity-60")}>
            <div className="px-2.5 pt-2 text-2xs text-muted num">{d.symbol} · {d.date} {d.time} ET · spot {num(d.spot)} · front exp {d.payoff?.front_expiration}</div>
            {d.fallback && <div className="mx-2.5 mt-1 rounded-sm border border-warn/40 bg-warn/[0.06] px-2 py-1 text-2xs text-warn">{d.fallback}</div>}
            {option && <EChart option={option} height={230} />}
            <div className="grid grid-cols-3 border-y border-line">
              {[
                ["Net premium", money(d.credit_conservative, 2, true), `mid ${money(d.credit_mid, 2, true)}`, d.credit_conservative >= 0 ? "text-up" : "text-down"],
                ["Max profit", d.max_profit === null ? "unlimited" : money(d.max_profit, 0), "at front expiration", "text-up"],
                ["Max loss", d.max_loss === null ? "unlimited" : money(-d.max_loss, 0), "at front expiration", "text-down"],
                ["BPR (Reg-T)", money(d.margin?.regt_bpr, 0), `req ${money(d.margin?.regt_requirement, 0)}`, ""],
                ["BPR (cash)", d.margin?.cash_bpr === null ? "n/a" : money(d.margin?.cash_bpr, 0), "cash-secured", ""],
                ["POP (est.)", pct(d.pop_estimate, 0), "lognormal, straddle-implied", ""],
              ].map(([l, v, s, t]) => (
                <div key={l as string} className="border-r border-line px-2.5 py-1.5 last:border-r-0 [&:nth-child(3n)]:border-r-0 [&:nth-child(-n+3)]:border-b">
                  <div className="label">{l}</div>
                  <div className={cn("num text-sm", t as string)}>{v}</div>
                  <div className="text-2xs text-faint truncate">{s}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-4 border-b border-line">
              {[["Δ delta", d.greeks.delta, 1], ["Γ gamma", d.greeks.gamma, 2], ["Θ theta/day", d.greeks.theta, 2], ["ν vega/pt", d.greeks.vega, 2]].map(([l, v, dp]) => (
                <div key={l as string} className="px-2.5 py-1.5 border-r border-line last:border-r-0">
                  <div className="label">{l}</div>
                  <div className={cn("num text-sm", (v as number) > 0 ? "text-up" : (v as number) < 0 ? "text-down" : "")}>{num(v as number, dp as number, true)}</div>
                </div>
              ))}
            </div>
            <table className="tbl">
              <thead><tr><th>Leg</th><th>Side</th><th>Exp</th><th className="r">Strike</th><th className="r">Bid</th><th className="r">Ask</th><th className="r">Fill</th><th className="r">IV</th><th className="r">Δ</th></tr></thead>
              <tbody>
                {d.legs.map((l: any) => l.kind === "stock" ? (
                  <tr key={l.id}><td className="font-mono">{l.id}</td><td className={l.qty > 0 ? "text-up" : "text-down"}>{l.qty > 0 ? "+" : ""}{l.qty} sh</td><td colSpan={7} className="text-muted">stock @ {num(l.price)}</td></tr>
                ) : (
                  <tr key={l.id}>
                    <td className="font-mono">{l.id}</td>
                    <td className={l.qty > 0 ? "text-up" : "text-down"}>{l.qty > 0 ? "+" : ""}{l.qty} {l.right}</td>
                    <td className="num text-fg-2">{l.expiration.slice(5)} <span className="text-faint">{l.dte}d</span></td>
                    <td className="r num">{l.strike}</td><td className="r num text-fg-2">{num(l.bid)}</td><td className="r num text-fg-2">{num(l.ask)}</td>
                    <td className="r num">{num(l.fill)}</td><td className="r num text-fg-2">{l.iv ? pct(l.iv, 1) : "—"}</td><td className="r num">{num(l.delta, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <ul className="px-2.5 py-2 text-2xs text-faint space-y-0.5 list-disc list-inside">{(d.notes || []).map((n: string) => <li key={n}>{n}</li>)}</ul>
          </div>
        )}
      </div>
    </div>
  );
}
