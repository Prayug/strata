import { useQuery } from "@tanstack/react-query";
import { Check, ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { C } from "@/components/charts/EChart";
import { LWChart, etChartTime, type LWSeries } from "@/components/charts/LWChart";
import { DataTable, Empty, ErrorBox, Loading, PageHeader, Panel, Pnl, StatusDot, SyntheticBadge, type Col } from "@/components/common";
import { Button, Select, Tip } from "@/components/ui";
import { api, useRun } from "@/lib/api";
import { cn, etTime, minutesLabel, money, num, pct, REASON_LABEL } from "@/lib/format";
import type { Explain, RunRow, Trade } from "@/lib/types";
import { ReplayExplorer, ReplayView } from "./lab/ReplayView";

export default function Replay() {
  const { id, tid } = useParams();
  if (!id || !tid) return <Start />;
  return <TradeReplay id={id} tid={Number(tid)} />;
}

/** A run priced on trade prints is replayed on the stored observations themselves; a run priced on a quote chain keeps the chain view. */
function TradeReplay({ id, tid }: { id: string; tid: number }) {
  const nav = useNavigate();
  const run = useRun(id);
  if (run.isLoading) return <Loading />;
  const r: any = run.data;
  const prints = r?.summary?.mode === "trades" || r?.config?.pricing?.mode === "trades";
  if (!prints) return <Workspace id={id} tid={tid} />;
  return (
    <div className="lab flex h-full min-h-0 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-3 text-[13px]"><button className="btn quiet sm" onClick={() => nav("/replay")}><ChevronLeft />Replay</button><span className="truncate font-medium">{r?.name}</span></div>
      <div className="min-h-0 flex-1"><ReplayView runId={id} tradeId={tid} onTrade={(t) => nav(`/replay/${id}/${t}`)} tradeCount={Number(r?.summary?.trades) || undefined} /></div>
    </div>
  );
}

function Start() {
  const [what, setWhat] = useState<"sessions" | "trades">("sessions");
  return (
    <div className="lab flex h-full min-h-0 flex-col" data-testid="replay-start">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-4">
        <span className="text-[14px] font-medium">Market replay</span>
        <span className="text-xs dim">stored observations only: nothing is drawn where nothing was recorded</span>
        <span className="flex-1" />
        {([["sessions", "Any stored session"], ["trades", "A trade of a backtest"]] as const).map(([k, l]) => <button key={k} className={cn("chip", what === k && "on")} onClick={() => setWhat(k)}>{l}</button>)}
      </div>
      <div className="min-h-0 flex-1">{what === "sessions" ? <ReplayExplorer /> : <Picker />}</div>
    </div>
  );
}

function Picker() {
  const nav = useNavigate();
  const runs = useQuery({ queryKey: ["runs", "replay"], queryFn: () => api.get<RunRow[]>("/api/backtests?kind=backtest,portfolio&limit=100") });
  const done = (runs.data || []).filter((r) => r.status === "done");
  const [rid, setRid] = useState<string>("");
  const cur = rid || done[0]?.id;
  const trades = useQuery({ queryKey: ["trades", cur], queryFn: () => api.get<Trade[]>(`/api/backtests/${cur}/trades`), enabled: !!cur });
  const cols: Col<Trade>[] = [
    { key: "id", label: "#", render: (t) => <span className="num text-faint">{t.id}</span>, sort: (t) => t.id },
    { key: "entry", label: "Entry", render: (t) => <span className="num">{etTime(t.entry_ts)}</span>, sort: (t) => t.entry_ts },
    { key: "legs", label: "Structure", render: (t) => <span className="num text-fg-2">{t.legs.map((l) => `${l.ratio > 0 ? "+" : ""}${l.ratio}${l.strike || ""}${l.right || "sh"}`).join(" ")}</span> },
    { key: "pnl", label: "P&L", align: "r", render: (t) => <Pnl v={t.pnl} fmt={(x) => money(x, 2, true)} />, sort: (t) => t.pnl },
    { key: "exit", label: "Exit", render: (t) => <span className="text-fg-2">{REASON_LABEL[t.exit_reason] || t.exit_reason}</span>, sort: (t) => t.exit_reason },
    { key: "flags", label: "Flags", render: (t) => <span className="text-2xs text-warn">{t.flags.join(" ")}</span> },
  ];
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Trade Replay" sub={<span className="text-2xs text-muted">pick a run and a trade to inspect</span>}>
        <Select className="w-96" value={cur || ""} onChange={setRid} options={done.map((r) => ({ value: r.id, label: `${r.name} · ${r.summary?.trades ?? 0} trades` }))} />
      </PageHeader>
      {runs.isLoading ? <Loading /> : !done.length ? <Empty title="No completed backtests">Run a backtest first; every trade can then be replayed step by step.</Empty> : (
        <DataTable rows={trades.data || []} cols={cols} rowKey={(t) => t.id} onRowClick={(t) => nav(`/replay/${cur}/${t.id}`)} className="flex-1" />
      )}
    </div>
  );
}

function ExplainList({ items, empty }: { items?: Explain[]; empty: string }) {
  if (!items?.length) return <div className="px-2.5 py-2 text-xs text-faint">{empty}</div>;
  return (
    <ul className="py-1">
      {items.map((e, i) => (
        <li key={i} className="flex items-start gap-1.5 px-2.5 py-[3px] text-xs" style={{ paddingLeft: 10 + Math.max(0, (e.depth || 1) - 1) * 12 }}>
          {e.selection ? <span className="mt-[3px] size-1.5 shrink-0 rounded-full bg-info" /> : e.passed ? <Check className="size-3 shrink-0 text-up mt-px" /> : <X className="size-3 shrink-0 text-down mt-px" />}
          <span className={cn("num text-2xs leading-4", e.group ? "text-muted" : e.passed || e.selection ? "text-fg" : "text-muted")}>{e.text}</span>
        </li>
      ))}
    </ul>
  );
}

function Workspace({ id, tid }: { id: string; tid: number }) {
  const nav = useNavigate();
  const run = useRun(id);
  const q = useQuery({ queryKey: ["replay", id, tid], queryFn: () => api.get<any>(`/api/backtests/${id}/trades/${tid}`) });
  const all = useQuery({ queryKey: ["trades", id], queryFn: () => api.get<Trade[]>(`/api/backtests/${id}/trades`) });
  const d = q.data;
  const t: Trade | undefined = d?.trade;
  const path = d?.path;
  const n = path?.t?.length || 0;
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("4");
  useEffect(() => { setI(0); setPlaying(false); }, [id, tid]);
  useEffect(() => {
    if (!playing || n < 2) return;
    const h = setInterval(() => setI((x) => { if (x >= n - 1) { setPlaying(false); return x; } return x + 1; }), 1000 / Number(speed));
    return () => clearInterval(h);
  }, [playing, n, speed]);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      if (e.key === "ArrowRight") setI((x) => Math.min(n - 1, x + 1));
      if (e.key === "ArrowLeft") setI((x) => Math.max(0, x - 1));
      if (e.key === "Home") setI(0);
      if (e.key === "End") setI(n - 1);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [n]);
  const idx = all.data?.findIndex((x) => x.id === tid) ?? -1;
  const prevT = idx > 0 ? all.data![idx - 1].id : null;
  const nextT = idx >= 0 && idx < (all.data?.length || 0) - 1 ? all.data![idx + 1].id : null;

  const candleSeries = useMemo<LWSeries[]>(() => {
    if (!d || !t) return [];
    const markers: any[] = [
      { time: etChartTime(Math.floor(t.entry_ts / (d.candle_minutes * 60)) * d.candle_minutes * 60), position: "belowBar", color: C.accent, shape: "arrowUp", text: `ENTRY ${num(t.entry_spot)}` },
    ];
    if (t.exit_ts) markers.push({ time: etChartTime(Math.floor(t.exit_ts / (d.candle_minutes * 60)) * d.candle_minutes * 60), position: "aboveBar", color: t.pnl >= 0 ? C.up : C.down, shape: "arrowDown", text: `EXIT ${REASON_LABEL[t.exit_reason] || t.exit_reason}` });
    const lines = t.legs.filter((l) => l.kind === "option").map((l) => ({ price: l.strike, color: l.ratio < 0 ? "rgba(239,83,80,0.8)" : "rgba(38,166,154,0.8)", title: `${l.ratio > 0 ? "+" : ""}${l.ratio}${l.strike}${l.right}` }));
    const be = t.max_profit !== null ? [] : [];
    return [{ type: "candles", data: (d.candles || []).map((c: number[]) => ({ time: etChartTime(c[0]), open: c[1], high: c[2], low: c[3], close: c[4] })), markers: markers.sort((a, b) => a.time - b.time), priceLines: [...lines, ...be] }];
  }, [d, t]);

  const premSeries = useMemo<LWSeries[]>(() => {
    if (!path || !t) return [];
    const out: LWSeries[] = [{ type: "baseline", data: path.t.map((x: number, k: number) => ({ time: etChartTime(x), value: path.pnl_mark[k] })), options: { baseValue: { type: "price", price: 0 }, title: "unrealised P&L (mid)" } },
      { type: "line", data: path.t.map((x: number, k: number) => ({ time: etChartTime(x), value: path.pnl_close[k] })), options: { color: C.muted, lineWidth: 1, lineStyle: 2, title: "if closed now" } }];
    const colors = C.palette;
    Object.entries(d.legs || {}).forEach(([li, lp]: any, k) => {
      const leg = t.legs[Number(li)];
      out.push({ type: "line", pane: 1, data: lp.t.map((x: number, j: number) => ({ time: etChartTime(x), value: lp.mid[j] })).filter((p: any) => p.value !== null), options: { color: colors[k % colors.length], lineWidth: 1.5, title: `${leg.ratio > 0 ? "+" : ""}${leg.ratio} ${leg.strike}${leg.right}` } });
    });
    return out;
  }, [path, d, t]);

  if (q.isLoading) return <Loading label="Loading trade" />;
  if (q.error) return <ErrorBox className="m-3" error={q.error} />;
  if (!t) return null;
  const at = n ? Math.min(i, n - 1) : 0;
  const curT = n ? path.t[at] : t.entry_ts;
  const cursorChart = n ? etChartTime(curT) : null;
  const pnlNow = n ? path.pnl_mark[at] : t.pnl;
  const mfeSoFar = n ? Math.max(0, ...path.pnl_mark.slice(0, at + 1)) : t.mfe;
  const maeSoFar = n ? Math.min(0, ...path.pnl_mark.slice(0, at + 1)) : t.mae;
  const legAt = (li: number) => {
    const lp = d.legs?.[String(li)];
    if (!lp) return null;
    let j = -1;
    for (let k = 0; k < lp.t.length; k++) if (lp.t[k] <= curT) j = k;
    return j >= 0 ? { bid: lp.bid[j], ask: lp.ask[j], mid: lp.mid[j], iv: lp.iv[j], delta: lp.delta[j] } : null;
  };
  const chainRows = d.chain?.rows || [];
  const legKeys = new Set(t.legs.filter((l) => l.kind === "option").map((l) => `${l.expiration}|${l.strike}|${l.right}`));

  return (
    <div className="flex h-full flex-col">
      <PageHeader title={<span>Trade #{t.id} <span className="text-muted font-normal">· {t.strategy}</span></span>} sub={<>
        <SyntheticBadge on={run.data?.synthetic} />
        <span className="num text-2xs text-muted">{t.symbol} · {etTime(t.entry_ts)} → {etTime(t.exit_ts)} ET · {minutesLabel(t.holding_minutes)}</span>
        <Pnl v={t.pnl} fmt={(x) => money(x, 2, true)} className="text-sm" />
        <span className={cn("text-xs", t.pnl >= 0 ? "text-up" : "text-down")}>{REASON_LABEL[t.exit_reason] || t.exit_reason}</span>
        {t.flags.map((f) => <span key={f} className="rounded-[2px] border border-warn/40 px-1 text-[10px] text-warn">{f}</span>)}
      </>}>
        <Button variant="ghost" onClick={() => nav(`/results/${id}`)}>Run</Button>
        <Button size="icon" variant="ghost" disabled={!prevT} onClick={() => nav(`/replay/${id}/${prevT}`)}><ChevronLeft /></Button>
        <span className="num text-2xs text-muted">{idx + 1}/{all.data?.length ?? "…"}</span>
        <Button size="icon" variant="ghost" disabled={!nextT} onClick={() => nav(`/replay/${id}/${nextT}`)}><ChevronRight /></Button>
      </PageHeader>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.55fr)_minmax(380px,1fr)] gap-px bg-line">
        <div className="flex min-h-0 flex-col gap-px bg-line">
          <Panel className="border-0 flex-[1.15]" title={`Underlying · ${d.candle_minutes === 390 ? "daily" : `${d.candle_minutes}m`} candles · strikes as price lines`}>
            <LWChart series={candleSeries} height="100%" timeVisible className="h-full min-h-[200px]" cursorTime={cursorChart as any} />
          </Panel>
          <Panel className="border-0 flex-1" title="Position P&L (top) · leg premiums (bottom)">
            <LWChart series={premSeries} height="100%" timeVisible className="h-full min-h-[200px]" cursorTime={cursorChart as any} onClick={(tm) => {
              if (tm === null || !n) return;
              let best = 0;
              for (let k = 0; k < n; k++) if (etChartTime(path.t[k]) <= tm) best = k;
              setI(best);
            }} />
          </Panel>
          <div className="flex h-10 shrink-0 items-center gap-2 bg-bg-1 px-2.5">
            <Tip content="Start (Home)"><Button size="icon" variant="ghost" onClick={() => setI(0)}><SkipBack /></Button></Tip>
            <Tip content="Step back (←)"><Button size="icon" variant="ghost" onClick={() => setI((x) => Math.max(0, x - 1))}><ChevronLeft /></Button></Tip>
            <Tip content="Play / pause (space)"><Button size="icon" variant="primary" onClick={() => { if (at >= n - 1) setI(0); setPlaying((p) => !p); }}>{playing ? <Pause /> : <Play />}</Button></Tip>
            <Tip content="Step forward (→)"><Button size="icon" variant="ghost" onClick={() => setI((x) => Math.min(n - 1, x + 1))}><ChevronRight /></Button></Tip>
            <Tip content="End"><Button size="icon" variant="ghost" onClick={() => setI(n - 1)}><SkipForward /></Button></Tip>
            <input type="range" min={0} max={Math.max(0, n - 1)} value={at} onChange={(e) => setI(Number(e.target.value))} className="flex-1 accent-[var(--color-accent)]" />
            <span className="num w-36 text-right text-xs">{etTime(curT)} ET</span>
            <span className="num w-14 text-right text-2xs text-muted">{at + 1}/{n}</span>
            <Select sz="sm" className="w-16" value={speed} onChange={setSpeed} options={[{ value: "1", label: "1×" }, { value: "4", label: "4×" }, { value: "12", label: "12×" }, { value: "30", label: "30×" }]} />
          </div>
        </div>
        <div className="min-h-0 overflow-auto bg-bg-1">
          <div className="grid grid-cols-4 border-b border-line">
            {[["Unrealised P&L", <Pnl key="u" v={pnlNow} fmt={(x) => money(x, 2, true)} />], ["Underlying", num(n ? path.spot[at] : t.entry_spot)], ["MFE so far", <span key="f" className="num text-up">{money(mfeSoFar, 2)}</span>], ["MAE so far", <span key="a" className="num text-down">{money(maeSoFar, 2)}</span>],
              ["Δ position", num(n ? path.delta[at] : null, 1)], ["Γ", num(n ? path.gamma[at] : null, 2)], ["Θ / day", num(n ? path.theta[at] : null, 2)], ["ν / pt", num(n ? path.vega[at] : null, 2)]].map(([l, v], k) => (
              <div key={k} className="border-r border-line px-2.5 py-1.5 last:border-r-0 [&:nth-child(4)]:border-r-0 [&:nth-child(-n+4)]:border-b">
                <div className="label">{l as string}</div><div className="num text-sm">{v as any}</div>
              </div>
            ))}
          </div>
          <Panel className="border-0 border-b" title="Legs" dense>
            <table className="tbl">
              <thead><tr><th>Leg</th><th className="r">Entry b/a</th><th className="r">Fill</th><th className="r">Now b/a</th><th className="r">Exit b/a</th><th className="r">Fill</th><th className="r">Δ in→out</th><th className="r">IV in</th><th className="r">P&L</th></tr></thead>
              <tbody>
                {t.legs.map((l, li) => {
                  const now = legAt(li);
                  return (
                    <tr key={li}>
                      <td className="num"><span className={l.ratio > 0 ? "text-up" : "text-down"}>{l.ratio > 0 ? "+" : ""}{l.ratio * (t.units || 1)}</span> {l.kind === "stock" ? "stock" : `${l.strike}${l.right}`} <span className="text-faint">{l.expiration?.slice(5)}</span></td>
                      <td className="r num text-fg-2">{num(l.entry_bid)}/{num(l.entry_ask)}</td><td className="r num">{num(l.entry_price)}</td>
                      <td className="r num text-fg-2">{now ? `${num(now.bid)}/${num(now.ask)}` : "—"}</td>
                      <td className="r num text-fg-2">{num(l.exit_bid)}/{num(l.exit_ask)}</td><td className="r num">{num(l.exit_price)}</td>
                      <td className="r num text-2xs">{num(l.entry_delta, 2)}→{num(l.exit_delta, 2)}</td><td className="r num text-fg-2">{l.entry_iv ? pct(l.entry_iv, 1) : "—"}</td>
                      <td className="r"><Pnl v={l.pnl} fmt={(x) => money(x, 0, true)} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="flex gap-4 px-2.5 py-1.5 text-2xs text-muted num border-t border-line">
              <span>premium {money(t.entry_cash, 2, true)}</span><span>commissions {money(t.commissions, 2)}</span><span>slippage vs mid {money(t.slippage, 2)}</span><span>BPR {money(t.max_risk, 0)}</span><span>DTE {t.entry_dte}</span><span>VIX {num(t.entry_vix, 1)}</span><span>regime {t.entry_regime}</span>
            </div>
          </Panel>
          <Panel className="border-0 border-b" title="Why the engine entered" dense><ExplainList items={t.entry_explain} empty="No explanation recorded." /></Panel>
          <Panel className="border-0 border-b" title={`Why it exited · ${REASON_LABEL[t.exit_reason] || t.exit_reason}`} dense>
            <ExplainList items={t.exit_explain} empty={["expiration", "assigned", "exercised", "end_of_backtest"].includes(t.exit_reason) ? `Held until ${t.exit_reason.replace(/_/g, " ")}; settled per the contract specification.` : "No rule explanation recorded."} />
          </Panel>
          <Panel className="border-0 border-b" title="Event log" dense>
            <ul className="py-1">
              {(d.events || []).map((e: any, k: number) => (
                <li key={k} className="flex gap-2 px-2.5 py-[3px] text-2xs">
                  <button className="num w-28 shrink-0 text-left text-muted hover:text-accent" onClick={() => { let b = 0; for (let j = 0; j < n; j++) if (path.t[j] <= e.t) b = j; setI(b); }}>{etTime(e.t)}</button>
                  <StatusDot status={e.kind === "exit" ? "done" : e.kind.includes("deferred") || e.kind.includes("margin") ? "error" : "queued"} />
                  <span className="num text-fg-2 leading-4">{e.msg}</span>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel className="border-0" title={`Options chain at entry · ${etTime(d.chain?.ts)} · spot ${num(d.chain?.spot)}`} dense>
            {!chainRows.length ? <div className="p-2 text-xs text-faint">Chain snapshot unavailable.</div> : (
              <table className="tbl">
                <thead><tr><th className="r">Δ</th><th className="r">Bid</th><th className="r">Ask</th><th className="text-center">Strike · Exp</th><th className="r">Bid</th><th className="r">Ask</th><th className="r">Δ</th></tr><tr><th colSpan={3} className="r text-info">CALLS</th><th /><th colSpan={3} className="text-info">PUTS</th></tr></thead>
                <tbody>
                  {chainRows.map((r: any) => {
                    const hc = legKeys.has(`${r.expiration}|${r.strike}|C`), hp = legKeys.has(`${r.expiration}|${r.strike}|P`);
                    const itmC = r.strike < d.chain.spot;
                    return (
                      <tr key={`${r.expiration}${r.strike}`}>
                        <td className={cn("r num", itmC && "bg-bg-2", hc && "!bg-accent/20")}>{num(r.call?.delta, 2)}</td>
                        <td className={cn("r num", itmC && "bg-bg-2", hc && "!bg-accent/20")}>{num(r.call?.bid)}</td>
                        <td className={cn("r num", itmC && "bg-bg-2", hc && "!bg-accent/20")}>{num(r.call?.ask)}</td>
                        <td className="text-center num font-medium">{r.strike} <span className="text-faint text-2xs">{r.expiration.slice(5)}</span></td>
                        <td className={cn("r num", !itmC && "bg-bg-2", hp && "!bg-accent/20")}>{num(r.put?.bid)}</td>
                        <td className={cn("r num", !itmC && "bg-bg-2", hp && "!bg-accent/20")}>{num(r.put?.ask)}</td>
                        <td className={cn("r num", !itmC && "bg-bg-2", hp && "!bg-accent/20")}>{num(r.put?.delta, 2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
