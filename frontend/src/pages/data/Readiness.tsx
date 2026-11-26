/* Research readiness: what the data stored on this computer can answer today, counted from the files themselves.
   Four things, in the order they matter:
     1. which kinds of research are possible for each symbol and period (never a provider's advertised coverage);
     2. whether the one free source of expired-contract history has been reached with a real key (and, when it has not,
        exactly what is needed and where it goes);
     3. the original question, on real observations: how many of its sessions can be priced;
     4. how trustworthy the intraday bars are. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, ExternalLink, KeyRound, Loader2, Play, RefreshCw, ShieldAlert, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { Callout, Section, Skeleton } from "../lab/bits";
import { CoverageHero } from "../lab/Coverage";

const CAP_TONE: Record<string, string> = { daily: "bg-info", intraday: "bg-accent", eod_options: "bg-up/60", option_trades: "bg-info", option_quotes: "bg-up" };

function Capabilities() {
  const q = useQuery({ queryKey: ["readiness"], queryFn: () => api.get<any>("/api/data/readiness"), staleTime: 30_000 });
  const [sym, setSym] = useState<string | null>(null);
  if (q.isLoading) return <Skeleton className="h-56" />;
  const syms: any[] = q.data?.symbols || [];
  if (!syms.length) return <Callout tone="info" title="No real market data is stored yet">Download the free daily history and end-of-day option chains under “Get free data”, and this view will show what they can answer.</Callout>;
  const cur = syms.find((s) => s.symbol === sym) || syms.find((s) => s.symbol === "SPY") || syms[0];
  const years: number[] = cur.years || [];
  const matrix: Record<number, any> = Object.fromEntries((cur.matrix || []).map((m: any) => [m.year, m]));
  return (
    <div data-testid="readiness">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-xs dim">Symbol</span>
        {syms.map((s) => <button key={s.symbol} className={cn("chip", cur.symbol === s.symbol && "on")} onClick={() => setSym(s.symbol)} data-testid={`readiness-${s.symbol}`}>{s.symbol}</button>)}
      </div>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="t" data-testid="capability-table">
          <thead><tr><th>Kind of research</th><th>Possible now</th><th className="r">Sessions</th><th>From</th><th>To</th><th>{years.length ? `${years[0]} → ${years[years.length - 1]}, by year` : ""}</th></tr></thead>
          <tbody>
            {(cur.capabilities || []).map((c: any) => (
              <tr key={c.id} data-capability={c.id} data-ready={c.ready ? "yes" : "no"}>
                <td><div className="w-[330px] whitespace-normal"><div className="font-medium">{c.title}</div><div className="mt-0.5 text-xs dim">{c.answers}</div>
                  {c.partial_sessions > 0 && <div className="mt-1 text-xs text-warn">{c.partial_sessions} partial session{c.partial_sessions === 1 ? "" : "s"} ({(c.partial_sample || []).join(", ")}): {c.partial_note}</div>}
                </div></td>
                <td><span className={cn("inline-flex items-center gap-1.5 text-[12.5px] font-medium", c.ready ? "text-up" : "text-down")}>{c.ready ? <Check className="size-3.5" strokeWidth={3} /> : <X className="size-3.5" strokeWidth={3} />}{c.ready ? "yes" : "no"}</span></td>
                <td className="r num font-medium">{(c.sessions || 0).toLocaleString()}</td>
                <td className="num dim">{c.first || "—"}</td>
                <td className="num dim">{c.last || "—"}</td>
                <td>
                  <div className="flex items-end gap-px" style={{ height: 22 }} role="img" aria-label={`${c.title}: sessions per year`}>
                    {years.map((y) => {
                      const n = matrix[y]?.[c.id] || 0;
                      return <Tip key={y} content={`${y}: ${n.toLocaleString()} session${n === 1 ? "" : "s"}`}><div className={cn("w-[7px] rounded-[1px]", n ? CAP_TONE[c.id] : "bg-bg-3")} style={{ height: n ? Math.max(3, Math.min(22, (n / 252) * 22)) : 2 }} /></Tip>;
                    })}
                  </div>
                  {c.by_resolution && <div className="mt-1 text-xs dim">{Object.entries(c.by_resolution).map(([k, v]: any) => `${v.toLocaleString()} at ${k}`).join(" · ")}</div>}
                  {!c.by_resolution && c.detail && <div className="mt-1 text-xs dim">{c.detail}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const CLASS_TONE: Record<string, string> = { ok: "text-up", authentication: "text-down", entitlement: "text-warn", plan: "text-warn", not_found: "text-warn", rate_limit: "text-warn", empty: "text-warn", bad_request: "text-down", server: "text-down", network: "text-down" };

function Access({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["alpaca-access"], queryFn: () => api.get<any>("/api/data/access/alpaca"), refetchInterval: 8000 });
  const run = useMutation({ mutationFn: () => api.post<{ run_id: string }>("/api/data/access/alpaca"), onSuccess: (r) => { onJob(r.run_id); toast.message("Access test started: read-only requests, nothing is downloaded or bought"); }, onError: (e: any) => toast.error(e.message) });
  const [raw, setRaw] = useState(false);
  if (q.isLoading) return <Skeleton className="h-40" />;
  const r = q.data || {};
  const blocked = r.status === "blocked";
  const tone = r.verified ? "border-up/40 bg-up/[0.05]" : blocked ? "border-down/45 bg-down/[0.06]" : "border-warn/40 bg-warn/[0.06]";
  return (
    <div data-testid="alpaca-access" data-status={r.status}>
      <div className={cn("rounded-[12px] border px-4 py-3.5", tone)}>
        <div className="flex flex-wrap items-center gap-2.5">
          {r.verified ? <Check className="size-4 text-up" strokeWidth={3} /> : <ShieldAlert className={cn("size-4", blocked ? "text-down" : "text-warn")} />}
          <span className="text-[14.5px] font-medium">Alpaca: {r.verified ? "verified from this installation" : blocked ? "BLOCKED, no credentials" : r.status === "not_run" ? "not tested yet" : String(r.status || "").replace(/_/g, " ")}</span>
          {r.run_at && <span className="text-xs dim">tested {ago(r.run_at)}</span>}
          <span className="flex-1" />
          <Link to="/settings#alpaca" className="btn sm" data-testid="enter-key" data-stored={r.key_stored ? "yes" : "no"}><KeyRound />{r.key_stored ? "Key stored (change it in Settings)" : "Enter the key in Settings"}</Link>
          <button className="btn primary sm" disabled={run.isPending} onClick={() => run.mutate()} data-testid="run-access">{run.isPending ? <Loader2 className="animate-spin" /> : <Play />}{r.key_stored ? "Run the access test" : "Re-check without a key"}</button>
        </div>
        <ul className="mt-2 flex flex-col gap-1 text-[13px] soft">{(r.summary || []).map((s: string, i: number) => <li key={i}>{s}</li>)}</ul>
        {r.reason && <div className="mt-1.5 text-xs dim">{r.reason}</div>}
      </div>
      {!r.verified && (r.how || []).length > 0 && (
        <div className="mt-3 rounded-[10px] border border-line px-4 py-3" data-testid="alpaca-how">
          <div className="text-[13.5px] font-medium">What is needed, and where it goes</div>
          <ol className="mt-2 flex list-decimal flex-col gap-1.5 pl-5 text-[13px] soft">{r.how.map((s: string, i: number) => <li key={i}>{s}</li>)}</ol>
          <div className="mt-2.5 text-xs dim">You create the account and type the key yourself. It is stored on this computer, never shown again, and never written to a log, an export or a language-model prompt. A free paper account places no orders: Strata has no code that can.</div>
        </div>
      )}
      {(r.matrix || []).length > 0 && (
        <div className="mt-3 overflow-x-auto rounded-lg border border-line">
          <table className="t" data-testid="access-matrix">
            <thead><tr><th>Period</th><th>Session</th><th className="r">Contracts asked</th><th className="r">Returned observations</th><th className="r">Found by listing</th><th className="r">Found by constructed id</th><th className="r">Option bars</th><th>Underlying minute bars</th><th>Notes</th></tr></thead>
            <tbody>{r.matrix.map((m: any, i: number) => {
              const cs: any[] = m.contracts || [];
              const got = cs.filter((c) => c.has_observations);
              return (
                <tr key={i}><td>{m.period}</td><td className="num">{m.date}</td><td className="r num">{cs.length}</td>
                  <td className={cn("r num font-medium", got.length ? "text-up" : "text-down")}>{got.length}</td>
                  <td className="r num">{got.filter((c) => c.expiration_source === "listed").length} / {cs.filter((c) => c.expiration_source === "listed").length}</td>
                  <td className="r num">{got.filter((c) => c.expiration_source !== "listed").length} / {cs.filter((c) => c.expiration_source !== "listed").length}</td>
                  <td className="r num">{cs.reduce((x, c) => x + (c.bars || 0), 0).toLocaleString()}</td>
                  <td className="num dim">{m.underlying ? `${m.underlying.bars} of ${m.underlying.expected} (${m.underlying.feed})` : "—"}</td>
                  <td className="w-[300px] whitespace-normal text-xs dim">{m.error || (m.bars_request && m.bars_request.class !== "ok" ? `bars: ${m.bars_request.class}` : "")}</td></tr>
              );
            })}</tbody>
          </table>
          <div className="border-t border-line px-3 py-2 text-xs dim">A constructed identifier that returns nothing proves nothing: the contract may not have existed, or may not have traded that session.</div>
        </div>
      )}
      {(r.requests || []).length > 0 && (
        <div className="mt-3">
          <button className="text-xs text-info hover:underline" onClick={() => setRaw((x) => !x)} data-testid="toggle-diagnostics">{raw ? "Hide" : "Show"} the sanitized diagnostics ({r.requests.length} requests)</button>
          {raw && (
            <div className="mt-2 overflow-x-auto rounded-lg border border-line">
              <table className="t" data-testid="access-diagnostics">
                <thead><tr><th>Asked for</th><th>Endpoint</th><th>Parameters</th><th className="r">Status</th><th>Classified as</th><th>What that means</th></tr></thead>
                <tbody>{r.requests.map((x: any, i: number) => (
                  <tr key={i}><td>{x.what}</td><td className="num text-xs">{x.endpoint}</td><td className="w-[260px] whitespace-normal break-all num text-xs dim">{Object.entries(x.params || {}).map(([k, v]) => `${k}=${v}`).join(" ")}</td><td className="r num">{x.status ?? "—"}</td>
                    <td className={cn("font-medium", CLASS_TONE[x.class] || "soft")}>{String(x.class || "").replace(/_/g, " ")}</td><td className="w-[340px] min-w-[240px] whitespace-normal text-xs dim">{x.explanation}{x.provider_code ? ` (provider code ${x.provider_code})` : ""}</td></tr>
                ))}</tbody>
              </table>
              <div className="border-t border-line px-3 py-2 text-xs dim">No key, token or account identifier is recorded in these rows.</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UseCase({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["usecase"], queryFn: () => api.get<any>("/api/research/usecase"), refetchInterval: 15000 });
  const run = useMutation({ mutationFn: () => api.post<{ run_id: string }>("/api/research/usecase"), onSuccess: (r) => { onJob(r.run_id); toast.message("Recomputing from the stored observations"); }, onError: (e: any) => toast.error(e.message) });
  const [pick, setPick] = useState(3);
  if (q.isLoading) return <Skeleton className="h-40" />;
  const u = q.data || {};
  const ths: any[] = u.thresholds || [];
  const cur = ths.find((t) => t.threshold_pct === pick) || ths[0];
  const all = cur?.periods?.[0];
  return (
    <div data-testid="usecase">
      <div className="flex flex-wrap items-start gap-3">
        <blockquote className="min-w-0 flex-1 border-l-2 border-accent/60 pl-3 text-[13.5px] soft">{u.sentence || "Buy ATM SPY 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in SPY, only on days when SPY fell at least 3% the previous trading day."}</blockquote>
        <button className="btn sm" disabled={run.isPending} onClick={() => run.mutate()} data-testid="run-usecase">{run.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}{ths.length ? "Recompute" : "Compute it"}</button>
      </div>
      {!ths.length && <Callout tone="info" className="mt-3" title="Not computed on this installation yet">It reads only what is stored here: the daily record for the signal, stored option observations for the fills.</Callout>}
      {cur && all && (
        <>
          <div className="mt-3 flex flex-wrap items-center gap-2"><span className="text-xs dim">After a fall of at least</span>{ths.map((t) => <button key={t.threshold_pct} className={cn("chip", cur.threshold_pct === t.threshold_pct && "on")} onClick={() => setPick(t.threshold_pct)}>{t.threshold_pct}%</button>)}{u.run_at && <span className="ml-auto text-xs dim">computed {ago(u.run_at)}</span>}</div>
          <CoverageHero className="mt-3" n={all.priced} of={all.qualifying} what="qualifying sessions are priced on stored option observations"
            counts={{ quotes: cur.counts?.priced_by_reference_only || 0, trades: cur.counts?.priced_by_engine || 0, missing: cur.counts?.missing || 0, no_contract: cur.counts?.no_contract || 0 }} first={u.history?.[0]} last={u.history?.[1]} testid="usecase-coverage" />
          <div className="mt-3 overflow-x-auto rounded-lg border border-line">
            <table className="t" data-testid="usecase-periods">
              <thead><tr><th>Period</th><th className="r">Rule fired</th><th className="r">No 0DTE listed</th><th className="r">Listing unknown</th><th className="r">Priced</th><th className="r">Priced, of fired</th><th className="r">No option prices</th></tr></thead>
              <tbody>{cur.periods.map((p: any, i: number) => (
                <tr key={i}><td className="w-[420px] min-w-[280px] whitespace-normal">{p.label}<div className="num text-xs dim">{p.from} to {p.to}</div></td><td className="r num">{p.qualifying.toLocaleString()}</td><td className="r num dim">{p.no_0dte_contract.toLocaleString()}</td><td className="r num dim">{p.contract_not_established.toLocaleString()}</td>
                  <td className={cn("r num font-medium", p.priced ? "text-up" : "text-down")}>{p.priced.toLocaleString()}</td><td className="r num">{p.priced_pct_of_qualifying ?? "—"}%</td><td className="r num">{p.missing.toLocaleString()}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <div className="card px-4 py-3"><div className="text-xs dim">Signal: plain arithmetic vs the engine</div><div className={cn("mt-1 text-[15px] font-medium", cur.lists_identical ? "text-up" : "text-down")}>{cur.lists_identical ? "identical lists" : "the lists differ"}</div><div className="mt-0.5 text-xs dim">{cur.qualifying.toLocaleString()} sessions each way</div></div>
            <div className="card px-4 py-3"><div className="text-xs dim">Fills: engine vs an independent recomputation</div><div className={cn("mt-1 text-[15px] font-medium", cur.counts?.disagree ? "text-down" : "text-up")}>{cur.counts?.agree ?? 0} of {(cur.counts?.agree ?? 0) + (cur.counts?.disagree ?? 0)} agree</div><div className="mt-0.5 text-xs dim">entry, exit, fees and P&amp;L recomputed from raw observations</div></div>
            <div className="card px-4 py-3"><div className="text-xs dim">The sentence vs a hand-written configuration</div><div className={cn("mt-1 text-[15px] font-medium", cur.from_sentence_identical ? "text-up" : "text-down")}>{cur.from_sentence_identical ? "the same configuration" : "they differ"}</div><div className="mt-0.5 text-xs dim">what the built-in reader produced from the words</div></div>
          </div>
          <ul className="mt-3 flex flex-col gap-1 text-xs dim">{(u.notes || []).map((n: string, i: number) => <li key={i}>· {n}</li>)}</ul>
        </>
      )}
    </div>
  );
}

function Bars() {
  const [sym, setSym] = useState("SPY");
  const q = useQuery({ queryKey: ["bars-audit", sym], queryFn: () => api.get<any>(`/api/data/bars/audit?symbol=${sym}`), staleTime: 60_000, retry: false });
  const a = q.data;
  return (
    <div data-testid="bars-audit">
      <div className="mb-3 flex items-center gap-2"><span className="text-xs dim">Symbol</span>{["SPY", "QQQ", "IWM"].map((s) => <button key={s} className={cn("chip", sym === s && "on")} onClick={() => setSym(s)}>{s}</button>)}</div>
      {q.isLoading && <Skeleton className="h-32" />}
      {q.isError && <Callout tone="info">No price data is stored for {sym}.</Callout>}
      {a && (
        <>
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))" }}>
            {Object.entries(a.finest_resolution_by_session || {}).map(([k, v]: any) => (
              <div key={k} className="card px-4 py-3"><div className="text-xs dim">Finest bars: {k}</div><div className="num mt-1 text-[20px] font-medium">{v.sessions.toLocaleString()}</div><div className="num text-xs dim">{v.first} to {v.last}</div></div>
            ))}
          </div>
          <div className="mt-3 flex flex-col gap-2.5">
            {(a.datasets || []).map((d: any) => (
              <div key={d.dataset} className="rounded-lg border border-line px-3.5 py-2.5" data-testid="bars-dataset">
                <div className="flex flex-wrap items-center gap-2 text-[13px]"><span className={cn("inline-flex items-center gap-1 font-medium", d.ok ? "text-up" : "text-warn")}>{d.ok ? <Check className="size-3.5" strokeWidth={3} /> : <ShieldAlert className="size-3.5" />}{d.provider} {d.resolution}</span><span className="num dim">{d.sessions.toLocaleString()} sessions · {d.bars.toLocaleString()} bars · {d.first} to {d.last}</span>
                  <span className="dim">· {d.intervals?.missing ?? 0} missing interval{d.intervals?.missing === 1 ? "" : "s"}</span></div>
                {(d.findings || []).length > 0 && <ul className="mt-1.5 flex flex-col gap-1 text-xs">{d.findings.map((f: any, i: number) => <li key={i} className="flex gap-2"><span className={cn("shrink-0 font-medium", f.level === "error" ? "text-down" : f.level === "warn" ? "text-warn" : "dim")}>{f.check}</span><span className="dim">{f.detail}</span></li>)}</ul>}
                <div className="mt-1 text-xs text-faint">{d.feed}</div>
              </div>
            ))}
          </div>
          {(a.mixed_feeds || []).length > 0 && <Callout tone="warn" className="mt-3" title="More than one feed is stored for this symbol">{a.mixed_feeds.join(" ")}</Callout>}
          {(a.routes_to_more || []).length > 0 && (
            <div className="mt-3"><div className="text-[13px] font-medium">Routes to more minute bars</div>
              <ul className="mt-1.5 flex flex-col gap-1.5 text-[13px] soft">{a.routes_to_more.map((x: string, i: number) => <li key={i}>· {x}</li>)}</ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function Readiness({ onJob }: { onJob: (id: string) => void }) {
  return (
    <div className="lab mx-auto flex max-w-[1180px] flex-col gap-4 px-2 pb-10 pt-3" data-testid="readiness-page">
      <div>
        <h2>What this data can answer today</h2>
        <p className="mt-1 max-w-[880px] text-[13.5px] soft">Counted from the files on this computer. A provider that advertises ten years of option history adds nothing here until its data has been downloaded with a real key and stored.</p>
      </div>
      <Section title="Kinds of research, by symbol and period"><Capabilities /></Section>
      <Section title="Access to expired-contract history" sub="the free route to intraday option prices"><Access onJob={onJob} /></Section>
      <Section title="The original question, on stored observations" sub="signal from the daily record, fills from stored option prices"><UseCase onJob={onJob} /></Section>
      <Section title="Underlying intraday bars" sub="what is stored, and what is wrong with it" open={false}><Bars /></Section>
      <div className="flex flex-wrap gap-2 text-[13px]">
        <a className="btn sm" href="https://app.alpaca.markets/signup" target="_blank" rel="noreferrer">Alpaca sign-up<ExternalLink /></a>
        <Link to="/data?tab=sources" className="btn quiet sm" reloadDocument>Every source that was found</Link>
        <Link to="/data?tab=bulk" className="btn quiet sm" reloadDocument>Import files you already have</Link>
      </div>
    </div>
  );
}
