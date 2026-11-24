/* Compact option-backtest report used inside the research workspace. */
import { ExternalLink } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { C } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { Panel, Stat, SyntheticBadge, Warnings } from "@/components/common";
import { Button } from "@/components/ui";
import { cn, minutesLabel, money, num, pct, tone } from "@/lib/format";
import type { RunRow } from "@/lib/types";

export function PricingBadge({ modeled, className }: { modeled: boolean; className?: string }) {
  return modeled ? (
    <span className={cn("rounded-[2px] border border-warn/60 bg-warn/10 px-1.5 py-px text-[10px] font-semibold tracking-wider text-warn", className)} title="Option premiums are theoretical estimates from a pricing model, not observed market prices">MODELED · ESTIMATED PREMIUMS</span>
  ) : (
    <span className={cn("rounded-[2px] border border-up/50 bg-up/10 px-1.5 py-px text-[10px] font-semibold tracking-wider text-up", className)} title="Fills and marks use observed historical option quotes">HISTORICAL OPTIONS · OBSERVED QUOTES</span>
  );
}

export function dataResolutionLines(meta: any): string[] {
  return Object.entries(meta?.data_resolution || {}).map(([sym, r]: any) => {
    const mix = Object.entries(r.underlying_sessions_by_resolution || {}).map(([k, n]) => `${n} sessions at ${k}`).join(", ");
    return `${sym} - options: ${r.options}${r.trade_based ? " (some fills from last-trade prints; no spread invented)" : ""}; underlying: ${mix}`;
  });
}

export function excludedRows(diag: any): { reason: string; count: number; sample: string[] }[] {
  const ex = diag?.excluded_sessions;
  if (!ex) return [];
  return Object.entries(ex).map(([reason, days]: any) => ({ reason, count: days.length, sample: days.slice(0, 4) })).sort((a, b) => b.count - a.count);
}

export function BacktestReport({ run }: { run: RunRow }) {
  const res = run.result;
  const m = res.metrics, meta = res.meta, d = res.diagnostics || {};
  const modeled = meta.pricing === "modeled";
  const s = res.series;
  const init = res.config.capital.initial_capital;
  const equity = useMemo(() => [
    { type: "baseline" as const, data: (s.dates as string[]).map((x, i) => ({ time: dayTime(x), value: s.equity[i] - init })), options: { baseValue: { type: "price", price: 0 }, title: "net P&L $" } },
  ], [s, init]);
  const dd = useMemo(() => [{ type: "area" as const, data: s.dates.map((x: string, i: number) => ({ time: dayTime(x), value: s.drawdown[i] * 100 })), options: { lineColor: C.down, topColor: "rgba(239,83,80,0.05)", bottomColor: "rgba(239,83,80,0.3)", invertFilledArea: true, lineWidth: 1 } }], [s]);
  const expired = Array.isArray(d.expired_orders) ? d.expired_orders.length : d.expired_orders || 0;
  const excluded = excludedRows(d);
  const skipped = Object.entries(d.skipped_entries || {}).sort((a: any, b: any) => b[1] - a[1]).slice(0, 5);
  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line bg-bg-1 px-3 py-1.5 text-2xs text-muted">
        <PricingBadge modeled={modeled} />
        <SyntheticBadge on={meta.synthetic} />
        <span className="num text-fg">{(meta.symbols || []).filter((x: string) => res.config.strategies.some((st: any) => st.symbol === x)).join(", ")}</span>
        <span className="num">{res.config.start} → {res.config.end} · {meta.sessions} sessions</span>
        <span className="flex-1" />
        <Link to={`/results/${run.id}`}><Button size="sm" variant="ghost"><ExternalLink /> Full report</Button></Link>
      </div>
      <div className="grid gap-px border-b border-line bg-line [&>*]:bg-bg-1" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(128px, 1fr))" }}>
        <Stat label="Net P&L" value={money(m.net_pnl, 0, true)} tone={tone(m.net_pnl)} sub={pct(m.total_return, 2, true)} />
        <Stat label="Trades" value={m.trades ?? 0} sub={`${m.max_consecutive_losses ?? 0} max losing streak`} />
        <Stat label="Win rate" value={pct(m.win_rate, 1)} sub={<span><span className="text-up">{money(m.avg_winner, 0)}</span> / <span className="text-down">{money(m.avg_loser, 0)}</span></span>} />
        <Stat label="Per trade" value={money(m.expectancy, 2, true)} tone={tone(m.expectancy)} sub={`± ${money(m.expectancy_std_err, 2)} s.e.`} hint="Mean P&L per trade and its standard error" />
        <Stat label="Profit factor" value={num(m.profit_factor)} sub={`t-stat ${num(m.t_stat)}`} hint="A t-statistic below about 2 means the average trade is not distinguishable from zero" />
        <Stat label="Max drawdown" value={pct(m.max_drawdown, 1)} tone="text-down" sub={money(m.max_drawdown_dollars, 0)} />
        <Stat label="Costs" value={money(-((m.total_commissions || 0) + (m.total_slippage || 0)), 0)} tone="text-down" sub={`comm ${money(m.total_commissions, 0)} · slip ${money(m.total_slippage, 0)}`} />
        <Stat label="Avg hold" value={minutesLabel(m.avg_holding_minutes)} sub={`Sharpe ${num(m.sharpe)}`} />
      </div>
      <div className="p-2">
        <Warnings items={[
          ...(modeled ? [{ level: "warn", text: `Estimated premiums: ${(meta.pricing_assumptions || []).join(" ")} Dollar results are approximations, not what the market paid.` }] : []),
          ...dataResolutionLines(meta).map((t) => ({ level: "info", text: t })),
          ...(expired ? [{ level: "warn", text: `${expired} entry signal(s) expired: no fresh option quote arrived within the allowed delay, so they were not filled late.` }] : []),
          ...excluded.map((e) => ({ level: "warn", text: `${e.count} session(s) excluded - ${e.reason} (e.g. ${e.sample.join(", ")}). A rule that cannot be evaluated is never treated as "condition false".` })),
          ...(m.trades > 0 && m.trades < 30 ? [{ level: "warn", text: `Only ${m.trades} trades: averages from a sample this small are unreliable.` }] : []),
          ...(m.warnings || []).slice(0, 3).map((t: string) => ({ level: "warn", text: t })),
        ]} />
      </div>
      <div className="grid grid-cols-1 gap-px bg-line">
        <Panel className="border-0" title="Cumulative net P&L ($)" dense><LWChart height={200} series={equity} /></Panel>
        <Panel className="border-0" title="Drawdown (%)" dense><LWChart height={90} series={dd} /></Panel>
      </div>
      {m.trades === 0 && skipped.length > 0 && (
        <Panel className="border-0 border-t" title="Why signals did not become trades" dense>
          <table className="tbl"><tbody>{skipped.map(([k, v]) => <tr key={k}><td className="whitespace-normal text-fg-2">{k}</td><td className="r num">{v as number}</td></tr>)}</tbody></table>
        </Panel>
      )}
    </div>
  );
}
