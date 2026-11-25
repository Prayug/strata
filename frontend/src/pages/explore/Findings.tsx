/* Knowledge base: every finding recorded by any notebook, searchable. A finding is a statement derived from a
   finished run, with its evidence grade and the reasons for that grade. Nothing here is written by hand or by a
   language model: an empty knowledge base means no research has finished yet. */
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Library, Loader2, Search, Wrench } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { pp } from "@/lib/lab";
import { Empty, Skeleton } from "../lab/bits";
import { GRADE, GradeBadge } from "../lab/Graph";
import { pv } from "../lab/Research";

const USABLE = new Set(["confirmed", "holds, not independent", "supported"]);

function Numbers({ n }: { n: any }) {
  if (!n) return null;
  const parts: [string, string][] = [];
  if (n.n !== undefined && n.n !== null) parts.push(["observations", Number(n.n).toLocaleString()]);
  if (n.probability !== undefined && n.probability !== null) parts.push(["rate", `${pp(n.probability)}${n.rest_probability !== undefined && n.rest_probability !== null ? ` vs ${pp(n.rest_probability)}` : ""}`]);
  if (n.p_value !== undefined && n.p_value !== null) parts.push(["p", pv(n.p_value)]);
  if (n.matched !== undefined && n.matched !== null) parts.push(["matched difference", `${n.matched > 0 ? "+" : ""}${(n.matched * 100).toFixed(1)} pts${n.matched_verdict ? ` (${String(n.matched_verdict).replace(/_/g, " ")})` : ""}`]);
  if (n.net_pnl !== undefined && n.net_pnl !== null) parts.push(["net P&L", `$${Math.round(n.net_pnl).toLocaleString()}`]);
  if (n.trades !== undefined && n.trades !== null) parts.push(["trades", String(n.trades)]);
  if (n.holdout_p !== undefined && n.holdout_p !== null) parts.push(["final test p", pv(n.holdout_p)]);
  if (!parts.length) return null;
  return <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs">{parts.map(([k, v]) => <span key={k}><span className="dim">{k} </span><span className="num">{v}</span></span>)}</div>;
}

export default function Findings() {
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [grade, setGrade] = useState<string | null>(null);
  const [symbol, setSymbol] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null);
  const all = useQuery({ queryKey: ["findings", "all"], queryFn: () => api.get<{ findings: any[]; grades: string[] }>("/api/findings?limit=500"), staleTime: 5_000 });
  const hit = useQuery({ queryKey: ["findings", q], queryFn: () => api.get<{ findings: any[] }>(`/api/findings?limit=500&q=${encodeURIComponent(q)}`), enabled: q.trim().length > 0, staleTime: 5_000 });
  const base = q.trim() ? hit.data?.findings || [] : all.data?.findings || [];
  const rows = base.filter((f) => (!grade || f.evidence === grade) && (!symbol || f.symbol === symbol));
  const symbols = [...new Set((all.data?.findings || []).map((f) => f.symbol).filter(Boolean))].sort();
  const counts = (all.data?.findings || []).reduce<Record<string, number>>((a, f) => ((a[f.evidence] = (a[f.evidence] || 0) + 1), a), {});

  const build = async (f: any) => {
    setBusy(f.id);
    try {
      await api.post(`/api/ai/sessions/${f.session_id}/strategy-from-finding`, { finding_id: f.id });
      nav(`/lab/${f.session_id}`);
    } catch (e: any) { toast.error(e.message); setBusy(null); }
  };

  return (
    <div className="lab h-full overflow-auto">
      <div className="mx-auto w-full max-w-[980px] px-6 pb-16 pt-8">
        <div className="eyebrow mb-2 flex items-center gap-2"><Library className="size-3.5 text-accent" />Knowledge base</div>
        <h1 className="!text-[24px]">What your research has established so far</h1>
        <p className="mt-1.5 max-w-[680px] text-[14px] soft">Every finished analysis leaves a finding here with an evidence grade and the reasons for it. Findings that failed are kept too: knowing what does not hold is half of the record.</p>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <div className="relative w-72">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-faint" />
            <input className="field !pl-8" placeholder="Search statements, symbols, features" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search findings" data-testid="findings-search" />
          </div>
          {symbols.length > 1 && symbols.map((s) => <button key={s} className={cn("chip", symbol === s && "on")} onClick={() => setSymbol(symbol === s ? null : s)}>{s}</button>)}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2" role="group" aria-label="Filter by evidence grade">
          {(all.data?.grades || []).filter((g) => counts[g]).map((g) => (
            <button key={g} className={cn("chip", grade === g && "on")} onClick={() => setGrade(grade === g ? null : g)} title={GRADE[g]?.help}><span aria-hidden>{GRADE[g]?.glyph}</span>{g}<span className="num dim">{counts[g]}</span></button>
          ))}
        </div>

        <div className="mt-5 flex flex-col gap-3" data-testid="findings-list">
          {all.isLoading && <><Skeleton className="h-24" /><Skeleton className="h-24" /></>}
          {all.data && base.length === 0 && (
            <div className="card"><Empty title={q ? "No finding matches that search" : "No findings yet"} icon={Library}>{q ? "Strata only lists findings that a finished run produced. It does not guess at ones that might exist." : "Ask a market question under Ask, or explore an event under Patterns. When an analysis finishes, its findings are recorded here."}</Empty></div>
          )}
          {rows.map((f) => (
            <article key={f.id} className="card px-4 py-3.5" data-testid="finding">
              <div className="flex flex-wrap items-center gap-2">
                <GradeBadge g={f.evidence} />
                {f.symbol && <span className="chip !h-[22px]">{f.symbol}</span>}
                <span className="text-xs dim">{f.period_start} to {f.period_end}</span>
                <span className="flex-1" />
                <span className="text-xs text-faint">{ago(f.created_at)}</span>
              </div>
              <p className="mt-2 text-[14px] leading-relaxed">{f.statement}</p>
              <Numbers n={f.numbers} />
              {(f.why || []).length > 0 && <ul className="mt-2 list-disc pl-5 text-xs soft">{f.why.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button className="btn sm" onClick={() => nav(`/lab/${f.session_id}`)} title={f.notebook}>Open the notebook<ArrowRight /></button>
                {f.convertible && (
                  <button className="btn sm" disabled={busy === f.id || !USABLE.has(f.evidence)} onClick={() => build(f)}
                    title={USABLE.has(f.evidence) ? "Draft an options strategy from this finding. It still has to be backtested on option prices." : "Only findings graded supported or better are offered as a basis for a strategy. Open the notebook to build one anyway."}>
                    {busy === f.id ? <Loader2 className="animate-spin" /> : <Wrench />}Build a strategy from it
                  </button>
                )}
                <span className="flex-1" />
                <span className="truncate text-xs dim" title="The notebook this finding belongs to">{f.notebook}</span>
                {f.reproducible && <span className="text-[11px] text-faint" title={`Configuration fingerprint ${f.fingerprint}. The exact configuration is stored and can be re-run.`}>reproducible · {String(f.fingerprint || "").slice(0, 8)}</span>}
              </div>
            </article>
          ))}
          {base.length > 0 && rows.length === 0 && <div className="card px-4 py-6 text-center text-[13px] dim">No finding has that grade and symbol.</div>}
        </div>
      </div>
    </div>
  );
}
