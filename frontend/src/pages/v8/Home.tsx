/* The first screen: one question box, questions the stored data can actually answer, and earlier investigations as
   questions and what they found. No infrastructure statistics. */
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, Loader2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { startNotebook } from "./data";

function ago(iso?: string) {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function Home() {
  const nav = useNavigate();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const starters = useQuery({ queryKey: ["v8-starters"], queryFn: () => api.get<any[]>("/api/v8/starters"), staleTime: 300_000 });
  const history = useQuery({ queryKey: ["v8-history"], queryFn: () => api.get<any[]>("/api/v8/history?limit=12"), staleTime: 15_000 });
  const scope = useQuery({ queryKey: ["v8-scope", "SPY"], queryFn: () => api.get<any>("/api/v8/scope/SPY"), staleTime: 600_000, retry: false });
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies"), staleTime: 60_000 });
  const ask = async (q: string) => {
    const t = q.trim();
    if (!t || busy) return;
    setBusy(true);
    try { nav(await startNotebook(t)); } catch (e: any) { toast.error(e.message); setBusy(false); }
  };
  const recent = (history.data || []).filter((h) => h.findings?.length || h.questions > 0).slice(0, 6);
  return (
    <div className="px-5 pb-24 pt-[9vh] md:px-8" data-testid="home">
      <div className="col">
        <div className="kicker mb-5" style={{ color: "var(--amber)" }}>Strata · historical market research</div>
        <h1 className="display">Ask what the market has done before.</h1>
        <p className="lede mt-4 max-w-[620px]">Questions are answered from stored history: every trading day of SPY, QQQ and IWM, minute by minute since 2016, and real option trades since 2024. Every number opens the days behind it.</p>
        <div className="ask mt-9">
          <textarea autoFocus rows={2} value={text} onChange={(e) => setText(e.target.value)} placeholder="How often does SPY recover after a 1% down day?"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(text); } }} aria-label="Your question" data-testid="home-ask" />
          <button className="send" onClick={() => ask(text)} disabled={!text.trim() || busy} aria-label="Ask">{busy ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={16} />}</button>
        </div>
        <div className="tiny mute mt-2">Plain words are fine: "spy dump 2% before noon?", "what if I bought calls at the open instead".</div>

        <section className="mt-14" data-testid="starters">
          <div className="kicker mb-2">Try one: each runs on the stored data in a few seconds</div>
          <div className="qlist">
            {(starters.data || []).map((s) => (
              <button key={s.text} className="qrow" onClick={() => ask(s.text)} disabled={busy} data-testid="starter">
                <span className="q">{s.text}</span>
                <span className="tiny faint hidden text-right sm:block" style={{ maxWidth: 260 }}>{s.why}</span>
              </button>
            ))}
            {starters.isLoading && <div className="small mute py-3">Checking what the stored data can answer…</div>}
          </div>
        </section>

        {recent.length > 0 && (
          <section className="mt-14" data-testid="recent">
            <div className="kicker mb-2">Earlier investigations</div>
            <div className="qlist">
              {recent.map((h) => (
                <Link key={h.id} to={`/ask/${h.id}`} className="qrow" style={{ textDecoration: "none" }}>
                  <span>
                    <span className="q block">{h.question}</span>
                    {h.findings?.length > 0 && <span className="small soft mt-1 block line-clamp-2">{h.findings[h.findings.length - 1].headline}</span>}
                  </span>
                  <span className="tiny faint whitespace-nowrap">{h.questions > 1 ? `${h.questions} questions · ` : ""}{ago(h.updated)}</span>
                </Link>
              ))}
            </div>
          </section>
        )}

        {(strategies.data || []).length > 0 && (
          <section className="mt-14">
            <div className="kicker mb-2">Saved strategies</div>
            <div className="flex flex-wrap gap-x-6 gap-y-2 small">
              {(strategies.data || []).slice(0, 8).map((s) => <Link key={s.id} className="link" to={`/builder?strategy=${s.id}`}>{s.name}</Link>)}
            </div>
          </section>
        )}

        {scope.data && (
          <section className="mt-14 rule pt-5" data-testid="scope">
            <div className="kicker mb-2">What the stored data can and cannot answer</div>
            <ul className="small soft flex flex-col gap-1.5">{(scope.data.lines || []).map((l: string, i: number) => <li key={i}>{l}</li>)}</ul>
            <Link className="link tiny mt-2 inline-block" to="/data">Data sources and coverage in detail</Link>
          </section>
        )}
      </div>
    </div>
  );
}

export default Home;
