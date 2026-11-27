/* The observations behind a number, in a sheet beside the answer: the exact days or trades, the reason each untested day
   was left out, a replay of any one of them, and "what do these days have in common?" for a selection. */
import { useMutation } from "@tanstack/react-query";
import { ArrowLeft, Loader2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { ReplayView } from "@/pages/lab/ReplayView";
import { fmtDate, useObservations } from "./data";

const FILTERS: Record<string, [string, string][]> = {
  sessions: [["all", "All days"], ["hit", "It happened"], ["miss", "It didn't"], ["best", "Highest"], ["worst", "Lowest"]],
  trades: [["all", "All trades"], ["winners", "Made money"], ["losers", "Lost money"], ["best", "Biggest gains"], ["worst", "Biggest losses"]],
  classification: [["all", "All days"], ["right", "Same way"], ["wrong", "Different ways"]],
};

function cellText(v: any, unit?: string) {
  if (v === null || v === undefined || v === "") return <span className="faint">—</span>;
  if (typeof v === "number") {
    if (unit === "%") return <span className={v > 0 ? "up" : v < 0 ? "down" : ""}>{v > 0 ? "+" : v < 0 ? "−" : ""}{Math.abs(v).toFixed(2)}%</span>;
    if (unit === "$") return <span className={v > 0 ? "up" : v < 0 ? "down" : ""}>{v < 0 ? "−" : ""}${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>;
    return v.toLocaleString();
  }
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

export function ExploreSheet({ spec, frame, onClose, onAsk }: { spec: any; frame?: any; onClose: () => void; onAsk?: (ids: string[], what: string, question?: string) => void }) {
  // the answer's own explore kinds for profile answers ("timing", "group", "dates") are the profile observations of its frame
  const base = useMemo(() => {
    if (!spec) return null;
    if (["timing", "group", "dates"].includes(spec.kind)) return { kind: "profile", frame, filter: spec.kind === "dates" ? "all" : spec.filter };
    return spec;
  }, [spec, frame]);
  const [flt, setFlt] = useState<string>(base?.filter || "all");
  useEffect(() => setFlt(base?.filter || "all"), [base]);
  const cur = base && base.kind !== "replay" ? { ...base, filter: flt } : base;
  const q = useObservations(cur);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [replay, setReplay] = useState<{ runId?: string; tradeId?: number; symbol?: string; date?: string } | null>(spec?.kind === "replay" ? { runId: spec.run_id, tradeId: spec.trade } : null);
  useEffect(() => { setSel(new Set()); }, [JSON.stringify(cur)]); // eslint-disable-line
  useEffect(() => { if (spec?.kind === "replay") setReplay({ runId: spec.run_id, tradeId: spec.trade }); }, [spec]);
  const days = useMutation({ mutationFn: (dates: string[]) => api.post<any>("/api/v8/days", { symbol: q.data?.symbol || frame?.symbol || "SPY", dates }) });
  useEffect(() => { days.reset(); }, [sel.size]); // eslint-disable-line
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") (replay && spec?.kind !== "replay" ? setReplay(null) : onClose()); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [replay, onClose, spec]);
  const t = q.data;
  const rows: any[] = t?.rows || [];
  const keyOf = (r: any) => String(r.id ?? r.date);
  const open = (r: any) => {
    if (t?.replay?.type === "trade") setReplay({ runId: t.replay.run_id, tradeId: r.id });
    else if (r.date && r.date !== "...") setReplay({ symbol: t?.symbol || t?.replay?.symbol || frame?.symbol, date: r.date });
  };
  const filters = FILTERS[t?.kind || cur?.kind || ""] || [];
  const selDates = rows.filter((r) => sel.has(keyOf(r))).map((r) => r.date).filter(Boolean);
  return (
    <>
      <div className="v8-scrim" onClick={onClose} />
      <aside className="v8-sheet v8" style={replay ? { width: "min(1080px, 100vw)" } : undefined} role="dialog" aria-label="Observations" data-testid="explore-sheet">
        <div className="flex items-center gap-3 border-b px-5 py-3" style={{ borderColor: "var(--rule)" }}>
          {replay && spec?.kind !== "replay" && <button className="act" onClick={() => setReplay(null)} aria-label="Back to the list"><ArrowLeft />Back</button>}
          <div className="min-w-0 flex-1">
            <div className="kicker">{replay ? "Replay" : "The observations behind this number"}</div>
            <div className="truncate text-[15px]">{replay ? (replay.date ? `${replay.symbol} on ${fmtDate(replay.date)}` : `Trade ${replay.tradeId}`) : t?.title || "…"}</div>
          </div>
          <button className="act" onClick={onClose} aria-label="Close"><X /></button>
        </div>
        {replay ? (
          <div className="min-h-0 flex-1 lab"><ReplayView runId={replay.runId} tradeId={replay.tradeId} symbol={replay.symbol} date={replay.date} onTrade={replay.runId ? (id) => setReplay({ ...replay, tradeId: id }) : undefined} tradeCount={t?.total} /></div>
        ) : (
          <>
            <div className="border-b px-5 py-3" style={{ borderColor: "var(--rule)" }}>
              {filters.length > 0 && <div className="seg mb-3">{filters.map(([k, l]) => <button key={k} aria-pressed={flt === k} onClick={() => setFlt(k)}>{l}</button>)}</div>}
              {t && <div className="small soft"><span className="num" style={{ color: "var(--amber)" }}>{t.count.toLocaleString()}</span>{t.total != null && t.total !== t.count ? <span className="mute"> of {t.total.toLocaleString()}</span> : null} {t.caption}</div>}
            </div>
            <div className="min-h-0 flex-1 overflow-auto px-5" style={{ background: "var(--ink-1)" }}>
              {q.isLoading && <div className="small mute flex items-center gap-2 py-6"><Loader2 className="size-4 animate-spin" />Reading the stored observations…</div>}
              {q.isError && <div className="small down py-6">{String((q.error as any)?.message)}</div>}
              {t && (
                <table className="obs" data-testid="obs-table">
                  <thead><tr><th style={{ width: 24 }} />{t.columns.map((c) => <th key={c.key} className={c.unit ? "r" : ""}>{c.label}</th>)}<th /></tr></thead>
                  <tbody>
                    {rows.slice(0, 1500).map((r) => {
                      const k = keyOf(r);
                      return (
                        <tr key={k} className={sel.has(k) ? "sel" : ""}>
                          <td><input type="checkbox" aria-label={`Select ${r.date}`} checked={sel.has(k)} onChange={() => setSel((s) => { const n = new Set(s); if (n.has(k)) n.delete(k); else n.add(k); return n; })} /></td>
                          {t.columns.map((c) => <td key={c.key} className={c.unit ? "r num" : c.key === "date" ? "num" : ""}>{c.key === "date" ? fmtDate(r.date) : cellText(r[c.key], c.unit)}</td>)}
                          <td className="r">{r.date && r.date !== "..." && <button className="link tiny" onClick={() => open(r)}>replay</button>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
              {rows.length > 1500 && <div className="tiny mute py-3">Showing the first 1,500 rows.</div>}
            </div>
            <div className="border-t px-5 py-3" style={{ borderColor: "var(--rule)" }}>
              {sel.size === 0 ? <div className="tiny mute">Select days to ask what they have in common, or open one to replay it.</div> : (
                <div className="flex flex-col gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="small soft">{sel.size} selected</span>
                    {onAsk && (() => {
                      const what = t?.kind === "trades" ? "trades" : "days";
                      const ids = rows.filter((r) => sel.has(keyOf(r))).map((r) => String(t?.kind === "trades" ? r.id : r.date));
                      return <>
                        <button className="act primary" onClick={() => onAsk(ids, what, `What sets the selected ${what} apart from the others?`)} data-testid="selection-ask">What sets them apart?</button>
                        <button className="act" onClick={() => onAsk(ids, what)} data-testid="selection-keep">Ask about them in the notebook</button>
                      </>;
                    })()}
                    <button className="act" onClick={() => days.mutate(selDates)} disabled={days.isPending || !selDates.length} data-testid="days-common">Against every stored day</button>
                    <button className="link tiny" onClick={() => setSel(new Set())}>clear</button>
                  </div>
                  {days.isPending && <div className="tiny mute flex items-center gap-2"><Loader2 className="size-3.5 animate-spin" />Comparing them with every stored day…</div>}
                  {days.isError && <div className="tiny down">{String((days.error as any)?.message)}</div>}
                  {days.data && <ul className="small soft list-disc pl-5" data-testid="days-summary">{days.data.lines.map((l: string, i: number) => <li key={i}>{l}</li>)}</ul>}
                </div>
              )}
            </div>
          </>
        )}
      </aside>
    </>
  );
}
