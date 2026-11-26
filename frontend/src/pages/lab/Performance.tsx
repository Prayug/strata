/* What a strategy did, in the order a reader should take it in: the sample and what the stored data could price first,
   the result after costs, how well the average is known, the path, where the result came from, what is left with worse
   fills, and the same position without the signal. A historical result and evidence of an edge are kept apart: the line
   at the top is the only place that speaks about an edge.

   Fills: every option fill with the stored bar its price was taken from, read back from the store.
   Execution: the same decisions filled on the first individual print after them. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, Download, Loader2, Play, X } from "lucide-react";
import { useState } from "react";
import { C, EChart } from "@/components/charts/EChart";
import { api, queryClient, useJob } from "@/lib/api";
import { cn } from "@/lib/format";

const usd = (v: number | null | undefined, dp = 0) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : `${v < 0 ? "−" : v > 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`);
const tone = (v: number | null | undefined) => (v == null || v === 0 ? "" : v > 0 ? "text-up" : "text-down");
const EVID: Record<string, [string, string]> = { supported: ["evidence of an edge on this data", "border-up/50 text-up"], mixed: ["mixed evidence", "border-warn/50 text-warn"], none: ["no evidence of an edge", "border-line-2 text-fg-2"], insufficient: ["too few trades to judge", "border-line-2 text-fg-2"] };

function Fig({ k, children, sub }: { k: string; children: React.ReactNode; sub?: React.ReactNode }) {
  return <div className="min-w-0"><div className="text-[10.5px] uppercase tracking-[0.07em] text-muted">{k}</div><div className="num mt-0.5 text-[15px] leading-5">{children}</div>{sub && <div className="text-[11px] dim">{sub}</div>}</div>;
}

function Row({ n, s, children }: { n: number; s: any; children?: React.ReactNode }) {
  return (
    <section className="grid grid-cols-[34px_minmax(0,1fr)] border-b border-line" data-testid={`perf-${s.id}`}>
      <div className="num border-r border-line bg-bg-1 pt-3 text-center text-[11px] text-faint">{String(n).padStart(2, "0")}</div>
      <div className="px-4 py-3">
        <div className="flex items-baseline gap-3"><h3 className="!text-[13px] !text-fg">{s.title}</h3>{s.warning && <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-down">does not reconcile</span>}</div>
        <p className={cn("mt-1 max-w-[880px] text-[13px] leading-relaxed", s.warning ? "text-down" : "soft")}>{s.text}</p>
        {children && <div className="mt-2.5">{children}</div>}
      </div>
    </section>
  );
}

export function PerformanceView({ runId, sid, trials, onAsk }: { runId: string; sid?: string; trials?: number; onAsk?: (t: string) => void }) {
  const q = useQuery({ queryKey: ["performance", runId, sid, trials], queryFn: () => api.get<any>(`/api/backtests/${runId}/performance?${sid ? `sid=${sid}&` : ""}trials=${trials || 1}`), retry: false });
  const [job, setJob] = useState<string | null>(null);
  const js = useJob(job, () => { setJob(null); queryClient.invalidateQueries({ queryKey: ["performance", runId] }); });
  const base = useMutation({ mutationFn: () => api.post<{ id: string }>(`/api/backtests/${runId}/baseline`), onSuccess: (r) => setJob(r.id) });
  if (q.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Reading the trades…</div>;
  if (q.isError || !q.data) return <div className="p-6 text-[13px] dim">No performance card for this run: {String((q.error as any)?.message || "it has no stored trades")}</div>;
  const d = q.data;
  const S: Record<string, any> = Object.fromEntries(d.sections.map((s: any) => [s.id, s]));
  const ev = EVID[d.evidence.level] || EVID.none;
  const f = (id: string) => S[id]?.figures || {};
  const hist = f("distribution").histogram;
  const months: any[] = f("calendar").months || [];
  const mmax = Math.max(1, ...months.map((m) => Math.abs(m.net)));
  const tod: any[] = f("time_of_day").half_hours || [];
  const tmax = Math.max(1, ...tod.map((m) => Math.abs(m.net)));
  const conc = f("concentration"), net = f("net").net as number;
  const ex = f("expectancy");
  const span = ex.ci95 ? Math.max(Math.abs(ex.ci95[0]), Math.abs(ex.ci95[1]), Math.abs(ex.mean)) * 1.15 || 1 : 1;
  const pos = (v: number) => `${50 + (v / span) * 50}%`;
  return (
    <div data-testid="view-performance">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-bg-1 px-5 py-3">
        <span className={cn("inline-flex h-7 items-center border px-2.5 text-[11.5px] font-medium uppercase tracking-[0.06em]", ev[1])} data-testid="perf-evidence">{ev[0]}</span>
        <span className="min-w-0 flex-1 text-[13px] soft">{d.evidence.text}</span>
        <span className="text-[11px] dim">{d.observation} · {d.window?.[0]} → {d.window?.[1]}{d.evidence.configurations_tried > 1 ? ` · ${d.evidence.configurations_tried} configurations tried` : ""}</span>
      </div>
      {S.trades && <Row n={1} s={S.trades}><div className="grid grid-cols-4 gap-4"><Fig k="Trades">{f("trades").trades}</Fig><Fig k="Winners"><span className="text-up">{f("trades").winners}</span></Fig><Fig k="Losers"><span className="text-down">{f("trades").losers}</span></Fig><Fig k="Sessions traded">{f("trades").sessions_with_a_trade}</Fig></div></Row>}
      {S.coverage && <Row n={2} s={S.coverage}>
        {f("coverage").signals ? (
          <div className="max-w-[640px]">
            <div className="flex h-3 w-full overflow-hidden bg-bg-3" title={`${f("coverage").priced} of ${f("coverage").signals} signals priced`}><div className="bg-up/70" style={{ width: `${Math.max(f("coverage").priced_pct, 0.6)}%` }} /><div className="flex-1 bg-warn/45" /></div>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px] dim"><span><span className="text-up">■</span> {f("coverage").priced?.toLocaleString()} priced on {f("coverage").observation}</span>
              {Object.entries(f("coverage").not_priced_by_reason || {}).map(([k, v]: any) => <span key={k}><span className="text-warn">■</span> {Number(v).toLocaleString()} {k.replace(/_/g, " ")}</span>)}
              {Object.keys(f("coverage").not_priced_by_reason || {}).length > 1 && <span className="text-faint">(a session can be counted under more than one reason)</span>}</div>
          </div>) : null}
      </Row>}
      {S.net && <Row n={3} s={S.net}><div className="grid grid-cols-2 gap-4 md:grid-cols-5"><Fig k="Before costs"><span className={tone(f("net").gross)}>{usd(f("net").gross, 2)}</span></Fig><Fig k="Commissions">{usd(-f("net").commissions, 2)}</Fig><Fig k="Assumed slippage">{usd(-f("net").slippage, 2)}</Fig>
        <Fig k="Net"><span className={cn("text-[17px] font-medium", tone(net))}>{usd(net, 2)}</span></Fig><Fig k="Of premium traded">{f("net").pct_of_premium_traded != null ? `${f("net").pct_of_premium_traded}%` : "–"}</Fig></div></Row>}
      {S.average && <Row n={4} s={S.average}><div className="grid grid-cols-3 gap-4 md:grid-cols-6"><Fig k="Mean"><span className={tone(f("average").mean)}>{usd(f("average").mean, 2)}</span></Fig><Fig k="Median">{usd(f("average").median, 2)}</Fig><Fig k="Win rate">{f("average").win_rate}%</Fig>
        <Fig k="Average win"><span className="text-up">{usd(f("average").average_win, 2)}</span></Fig><Fig k="Average loss"><span className="text-down">{usd(f("average").average_loss, 2)}</span></Fig><Fig k="Profit factor">{f("average").profit_factor ?? "–"}</Fig></div></Row>}
      {S.expectancy && <Row n={5} s={S.expectancy}>
        {ex.ci95 && (
          <div className="max-w-[640px]">
            <div className="relative h-6">
              <div className="absolute top-1/2 h-px w-full bg-line-2" /><div className="absolute top-0 h-full w-px bg-fg-2" style={{ left: "50%" }} />
              <div className={cn("absolute top-[9px] h-1.5", ex.ci95[0] > 0 ? "bg-up" : ex.ci95[1] < 0 ? "bg-down" : "bg-muted")} style={{ left: pos(ex.ci95[0]), width: `calc(${pos(ex.ci95[1])} - ${pos(ex.ci95[0])})` }} />
              <div className="absolute top-[5px] h-3.5 w-0.5 bg-fg" style={{ left: pos(ex.mean) }} />
            </div>
            <div className="num flex justify-between text-[11px] dim"><span>{usd(-span)}</span><span>zero</span><span>{usd(span)}</span></div>
          </div>)}
      </Row>}
      {S.drawdown && <Row n={6} s={S.drawdown} />}
      {S.streaks && <Row n={7} s={S.streaks} />}
      {S.distribution && <Row n={8} s={S.distribution}>
        {hist && <EChart height={150} option={{ grid: { left: 44, right: 10, top: 8, bottom: 22 }, tooltip: { trigger: "axis", formatter: (p: any) => `${usd(hist.edges[p[0].dataIndex])} … ${usd(hist.edges[p[0].dataIndex + 1])}: ${p[0].value} trades` },
          xAxis: { type: "category", data: hist.counts.map((_: any, i: number) => usd((hist.edges[i] + hist.edges[i + 1]) / 2)), axisLabel: { interval: Math.ceil(hist.counts.length / 8) } }, yAxis: { type: "value", minInterval: 1 },
          series: [{ type: "bar", barCategoryGap: "8%", data: hist.counts.map((v: number, i: number) => ({ value: v, itemStyle: { color: (hist.edges[i] + hist.edges[i + 1]) / 2 >= 0 ? C.up : C.down, opacity: 0.8 } })) }] }} />}
      </Row>}
      {S.calendar && <Row n={9} s={S.calendar}>
        <div className="flex flex-wrap gap-[3px]" data-testid="perf-months">{months.map((m) => (
          <div key={m.key} title={`${m.key}: ${usd(m.net, 2)} over ${m.trades} trades, ${m.win_rate}% won`} className="num grid h-9 w-[46px] place-items-center text-[10px] leading-3"
            style={{ background: m.net >= 0 ? `rgba(38,166,154,${0.12 + 0.6 * Math.abs(m.net) / mmax})` : `rgba(239,83,80,${0.12 + 0.6 * Math.abs(m.net) / mmax})` }}><span className="text-fg-2">{m.key.slice(2)}</span><span>{usd(m.net)}</span></div>))}</div>
      </Row>}
      {S.time_of_day && <Row n={10} s={S.time_of_day}>
        {tod.length > 1 && <div className="flex items-end gap-1.5">{tod.map((m) => (
          <div key={m.key} className="flex w-12 flex-col items-center" title={`${m.key}: ${usd(m.net, 2)} over ${m.trades} trades`}><div className="flex h-14 w-full items-end"><div className={cn("w-full", m.net >= 0 ? "bg-up/70" : "bg-down/70")} style={{ height: `${Math.max(4, Math.abs(m.net) / tmax * 100)}%` }} /></div><div className="num mt-0.5 text-[10px] dim">{m.key}</div><div className="num text-[10px]">{m.trades}</div></div>))}</div>}
      </Row>}
      {S.execution && <Row n={11} s={S.execution}>
        <table className="t max-w-[640px]"><thead><tr><th>Every fill worse by</th>{(f("execution").rows || []).map((r: any) => <th key={r.worse_fills_pct} className="r">{r.worse_fills_pct}%</th>)}<th className="r">reaches zero at</th></tr></thead>
          <tbody><tr><td className="dim">net result</td>{(f("execution").rows || []).map((r: any) => <td key={r.worse_fills_pct} className={cn("r num", tone(r.net))}>{usd(r.net)}</td>)}<td className="r num">{f("execution").breakeven_pct != null ? `${f("execution").breakeven_pct}%` : "–"}</td></tr></tbody></table>
        {f("execution").individual_trades && <div className="mt-2 text-xs dim">On individual prints: {usd(f("execution").individual_trades.tick_pnl_repriced, 2)} against {usd(f("execution").individual_trades.bar_pnl_repriced, 2)} on minute bars ({f("execution").individual_trades.repriced} trades). See the Execution tab.</div>}
      </Row>}
      {S.concentration && <Row n={12} s={S.concentration}>
        <div className="grid max-w-[640px] gap-1">{[["all trades", net], ["without the best trade", conc.net_without_best], ["without the best three", conc.net_without_best_3], [`without the best ${conc.best_5pct_trades} (5%)`, conc.net_without_best_5pct]].map(([l, v]: any) => (
          <div key={l} className="grid grid-cols-[190px_1fr_90px] items-center gap-2 text-[12px]"><span className="dim">{l}</span><div className="relative h-2.5 bg-bg-3"><div className={cn("absolute top-0 h-full", v >= 0 ? "bg-up/70" : "bg-down/70")} style={{ left: v >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(v) / Math.max(1, Math.abs(net), Math.abs(conc.net_without_best_5pct)) * 50)}%`, width: `${Math.min(50, Math.abs(v) / Math.max(1, Math.abs(net), Math.abs(conc.net_without_best_5pct)) * 50)}%` }} /><div className="absolute left-1/2 top-0 h-full w-px bg-fg-2" /></div><span className={cn("num text-right", tone(v))}>{usd(v)}</span></div>))}</div>
      </Row>}
      {S.baseline && <Row n={13} s={S.baseline}>
        {S.baseline.figures ? <div className="grid grid-cols-2 gap-4 md:grid-cols-5"><Fig k="Baseline trades">{S.baseline.figures.trades}</Fig><Fig k="Baseline per trade"><span className={tone(S.baseline.figures.average)}>{usd(S.baseline.figures.average, 2)}</span></Fig><Fig k="With the signal"><span className={tone(f("average").mean)}>{usd(f("average").mean, 2)}</span></Fig>
          <Fig k="Difference"><span className={tone(S.baseline.figures.difference_in_average)}>{usd(S.baseline.figures.difference_in_average, 2)}</span></Fig><Fig k="p">{S.baseline.figures.p_value != null ? S.baseline.figures.p_value.toFixed(3) : "–"}</Fig></div>
          : <button className="btn sm" disabled={base.isPending || !!job} onClick={() => base.mutate()} data-testid="perf-baseline-run">{job ? <Loader2 className="animate-spin" /> : <Play />}{job ? `Running the baseline… ${Math.round((js?.progress || 0) * 100)}%` : "Run the baseline"}</button>}
        {base.isError && <div className="mt-1 text-xs text-down">{String((base.error as any)?.message)}</div>}
      </Row>}
      <div className="px-5 py-3 text-[11.5px] dim">{d.evidence.reading}{onAsk && <button className="ml-2 underline" onClick={() => onAsk("Challenge this strategy")}>Challenge this result</button>}</div>
    </div>
  );
}

/* ----------------------------------------------------------------------------- every fill against its stored bar */
export function FillsLedger({ runId, onTrade }: { runId: string; onTrade?: (tid: number) => void }) {
  const q = useQuery({ queryKey: ["fills", runId], queryFn: () => api.get<any>(`/api/backtests/${runId}/fills`) });
  const [only, setOnly] = useState(false);
  if (q.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Reading every fill back from the store…</div>;
  const d = q.data;
  if (!d) return <div className="p-6 text-[13px] dim">No fills are stored for this run.</div>;
  const rows: any[] = only ? d.fills.filter((x: any) => x.matches_stored_bar === false) : d.fills;
  const bad = d.checked_against_stored_bars - d.matching;
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="view-fills">
      <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-line px-5 py-2.5 text-[13px]">
        <span className={cn("num", bad ? "text-warn" : "text-up")}>{d.matching} of {d.checked_against_stored_bars}</span><span className="soft">fills match the stored bar they were priced from</span>
        {bad > 0 && <button className={cn("chip", only && "on")} onClick={() => setOnly((o) => !o)}>{bad} that do not</button>}
        {d.settled_at_expiration > 0 && <span className="text-xs dim">{d.settled_at_expiration} settled at expiration at intrinsic value: not fills, so there is no print to compare</span>}
        <span className="flex-1" /><a className="btn quiet sm" href={`/api/backtests/${runId}/fills.csv`} download data-testid="fills-csv"><Download />CSV of every fill</a>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="t" data-testid="fills-table">
          <thead><tr><th className="r">Trade</th><th>Action</th><th>Contract</th><th>Decided</th><th>Filled</th><th className="r">Fill</th><th className="r">Print used</th><th className="r">Against order</th><th>Stored bar</th><th className="r">Its close</th><th className="r">Low – high</th><th className="r">Volume</th><th className="r">Print age</th><th>Matches</th></tr></thead>
          <tbody>{rows.slice(0, 2000).map((x, i) => (
            <tr key={i} className={cn(onTrade && "click")} onClick={() => onTrade?.(x.trade)}>
              <td className="r num">{x.trade}</td><td>{x.action === "settlement" ? <span className="text-warn">settled</span> : <>{x.side} <span className="dim">({x.action})</span></>}</td><td className="num">{x.contract}</td><td className="num dim">{x.decided_et?.slice(5)}</td><td className="num">{x.filled_et?.slice(5)}</td>
              <td className="r num">{x.fill_price?.toFixed(2)}</td><td className="r num">{x.print_used?.toFixed(2) ?? "–"}</td><td className="r num dim">{x.adjustment_against_the_order ? x.adjustment_against_the_order.toFixed(3) : "0"}</td>
              <td className="num dim">{x.stored_bar_start_et ?? "–"}</td><td className="r num">{x.stored_bar_close?.toFixed(2) ?? "–"}</td><td className="r num dim">{x.stored_bar_low != null ? `${x.stored_bar_low.toFixed(2)} – ${x.stored_bar_high.toFixed(2)}` : "–"}</td><td className="r num dim">{x.stored_bar_volume ?? "–"}</td>
              <td className="r num">{x.print_age_seconds != null ? `${x.print_age_seconds}s` : "–"}</td><td>{x.matches_stored_bar === true ? <Check className="size-3.5 text-up" /> : x.matches_stored_bar === false ? <span className="flex items-center gap-1 text-warn"><X className="size-3.5" />{x.note || "stored bar differs"}</span> : <span className="dim" title={x.note || ""}>{x.action === "settlement" ? "intrinsic value, not a trade" : "–"}</span>}</td>
            </tr>))}</tbody>
        </table>
      </div>
      <div className="shrink-0 border-t border-line px-5 py-2 text-[11.5px] dim">{d.reading}</div>
    </div>
  );
}

/* ----------------------------------------------------------------------------- execution: minute bars, individual prints, quotes */
export function ExecutionCheck({ runId }: { runId: string }) {
  const q = useQuery({ queryKey: ["hires", runId], queryFn: () => api.get<any>(`/api/backtests/${runId}/hires`) });
  const db = useQuery({ queryKey: ["databento-access"], queryFn: () => api.get<any>("/api/data/access/databento"), retry: false });
  const [job, setJob] = useState<string | null>(null);
  const [policy, setPolicy] = useState("decision_clock");
  const js = useJob(job, () => { setJob(null); queryClient.invalidateQueries({ queryKey: ["hires", runId] }); queryClient.invalidateQueries({ queryKey: ["performance", runId] }); });
  const run = useMutation({ mutationFn: (p: string) => api.post<{ id: string }>(`/api/backtests/${runId}/hires`, { exit_policy: p }), onSuccess: (r) => setJob(r.id) });
  const [open, setOpen] = useState<number | null>(null);
  if (q.isLoading) return <div className="flex items-center gap-2 p-6 text-[13px] dim"><Loader2 className="size-4 animate-spin" />Loading…</div>;
  const d = q.data || {};
  const rep = d.reports?.[policy];
  const s = rep?.summary;
  const dbd = db.data;
  return (
    <div data-testid="view-execution-check">
      <div className="border-b border-line px-5 py-4">
        <div className="eyebrow mb-1.5">Execution, three ways</div>
        <p className="max-w-[900px] text-[13.5px] soft">{d.what}</p>
        <p className="mt-1.5 max-w-[900px] text-xs dim">{d.limits}</p>
      </div>
      <div className="grid grid-cols-1 gap-px bg-line md:grid-cols-3">
        <div className="bg-bg-0 px-4 py-3.5"><div className="eyebrow">1 · Minute bars</div><div className="mt-1 text-[12.5px] soft">The backtest as it ran: each fill is the close of the last one-minute trade bar that had ended.</div>
          <div className={cn("num mt-2 text-[20px]", tone(s?.bar_pnl_repriced))}>{s ? usd(s.bar_pnl_repriced, 2) : "–"}</div><div className="text-[11px] dim">{s ? `the ${s.repriced} trades that could be repriced` : "run the check to compare like with like"}</div></div>
        <div className="bg-bg-0 px-4 py-3.5"><div className="eyebrow">2 · Individual prints</div><div className="mt-1 text-[12.5px] soft">The same decisions, filled on the first eligible print strictly after each one.</div>
          <div className={cn("num mt-2 text-[20px]", tone(s?.tick_pnl_repriced))} data-testid="hires-tick">{s ? usd(s.tick_pnl_repriced, 2) : "–"}</div>
          <div className="text-[11px] dim">{s ? `difference ${usd(s.difference, 2)} · entries a median ${s.entry_seconds_after_decision?.median ?? "–"}s after the decision, at prices ${Math.abs(s.entry_price_worse_pct?.mean ?? 0).toFixed(1)}% ${(s.entry_price_worse_pct?.mean ?? 0) >= 0 ? "worse" : "better"} on average · ${s.entries_not_filled} not filled` : d.available ? "not run yet" : d.blocked}</div></div>
        <div className="bg-bg-0 px-4 py-3.5"><div className="eyebrow">3 · Bid and ask</div><div className="mt-1 text-[12.5px] soft">Quotes are the only observation that says what an order could have been filled at.</div>
          <div className="num mt-2 text-[20px] text-faint">–</div>
          <div className="text-[11px] text-warn" data-testid="quotes-blocked">{dbd?.status === "blocked" || !dbd?.key ? "BLOCKED: no historical quote source is connected (Databento key not set). Nothing is substituted for it." : dbd?.summary || "see the Data page"}</div></div>
      </div>
      <div className="flex flex-wrap items-center gap-2 border-b border-t border-line px-5 py-2.5">
        <span className="text-xs dim">Exits filled</span>
        {[["decision_clock", "when the engine decided"], ["resting_orders", "as resting target and stop orders"]].map(([k, l]) => <button key={k} className={cn("chip", policy === k && "on")} onClick={() => setPolicy(k)}>{l}</button>)}
        <span className="flex-1" />
        <button className="btn sm" disabled={!d.available || run.isPending || !!job} onClick={() => run.mutate(policy)} data-testid="hires-run">{job ? <Loader2 className="animate-spin" /> : <Play />}{job ? `${js?.message || "Asking for the prints around each fill"} ${Math.round((js?.progress || 0) * 100)}%` : rep ? "Run again" : "Run the check"}</button>
        {rep && <a className="btn quiet sm" href={`/api/backtests/${runId}/hires.csv?policy=${policy}`} download><Download />CSV</a>}
      </div>
      {run.isError && <div className="px-5 py-2 text-xs text-down">{String((run.error as any)?.message)}</div>}
      {rep ? (
        <div className="max-h-[520px] overflow-auto">
          <table className="t" data-testid="hires-table">
            <thead><tr><th className="r">Trade</th><th>Entered</th><th>Status</th><th className="r">Bar fill</th><th className="r">Print fill</th><th className="r">After decision</th><th className="r">Bar exit</th><th className="r">Print exit</th><th>Exit</th><th className="r">Bar result</th><th className="r">Print result</th><th className="r">Difference</th></tr></thead>
            <tbody>{(rep.positions || []).map((p: any) => { const en = p.legs?.[0]?.entry || {}, xi = p.legs?.[0]?.exit || {}; return (
              <tr key={p.id} className="click" onClick={() => setOpen(open === p.id ? null : p.id)}>
                <td className="r num">{p.id}</td><td className="num">{p.date} <span className="dim">{en.et?.slice(0, 12) || ""}</span></td><td className={cn(p.status === "repriced" ? "" : "text-warn")}>{String(p.status).replace(/_/g, " ")}</td>
                <td className="r num dim">{en.bar_price?.toFixed(2) ?? "–"}</td><td className="r num">{en.print?.toFixed(2) ?? "–"}</td><td className="r num dim">{en.seconds_after_decision != null ? `${en.seconds_after_decision}s` : "–"}</td>
                <td className="r num dim">{xi.bar_price?.toFixed(2) ?? "–"}</td><td className="r num">{xi.print?.toFixed(2) ?? "–"}</td><td className="dim">{String(p.tick?.exit_reason || p.bar?.exit_reason || "").replace(/_/g, " ")}{p.exit_reason_changed ? " (changed)" : ""}</td>
                <td className={cn("r num", tone(p.bar?.pnl))}>{usd(p.bar?.pnl, 2)}</td><td className={cn("r num", tone(p.tick?.pnl))}>{usd(p.tick?.pnl, 2)}</td><td className={cn("r num", tone(p.pnl_difference))}>{usd(p.pnl_difference, 2)}</td>
              </tr>); })}</tbody>
          </table>
        </div>
      ) : <div className="px-5 py-6 text-[13px] dim">{d.available ? "The check has not been run for this run with this exit policy. It asks the provider only for the minutes around each fill, and keeps what it receives." : d.blocked}</div>}
      {s?.conditions_of_prints_used && <div className="border-t border-line px-5 py-2.5 text-[11.5px] dim">Prints used, by the exchange's condition: {Object.entries(s.conditions_of_prints_used).map(([k, v]: any) => `${k} ×${v.count} (${v.meaning})`).join("; ")}.</div>}
    </div>
  );
}
