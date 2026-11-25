/* The provider marketplace: every legitimate source of historical options data that was found, in one place.
   Two columns of truth are kept apart on every card: what the provider ADVERTISES (read from its own pages on a
   stated date) and what THIS installation has VERIFIED with a real request. A source without credentials is
   "unverified", never "working". Nothing here spends money: paid sources show a quoted cost and stop there. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { Calculator, ExternalLink, KeyRound, Loader2, Play, PlugZap, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { Callout, Section, Skeleton } from "../lab/bits";

const STATUS: Record<string, { glyph: string; word: string; tone: string; help: string; order: number }> = {
  verified: { glyph: "●", word: "verified here", tone: "text-up border-up/40 bg-up/[0.07]", help: "A real request from this installation returned the data described under Measured.", order: 0 },
  "parser verified (demo symbol only)": { glyph: "◐", word: "parser verified, demo symbol only", tone: "text-warn border-warn/40 bg-warn/[0.07]", help: "The response format was read from the provider's free demo, which serves one symbol. Access to SPY, QQQ or IWM is not verified.", order: 1 },
  unverified: { glyph: "○", word: "unverified", tone: "text-warn border-warn/40 bg-warn/[0.07]", help: "An adapter exists, but no live request has succeeded from this installation. Tests against recorded responses do not count.", order: 2 },
  "not checked yet": { glyph: "○", word: "not checked yet", tone: "text-muted border-line-2 bg-bg-2", help: "The live check has not been run on this installation.", order: 2 },
  "import only": { glyph: "▤", word: "file import only", tone: "text-fg-2 border-line-2 bg-bg-2", help: "Strata reads files you obtain from this source yourself. It makes no request to it.", order: 3 },
  "no integration": { glyph: "–", word: "no integration", tone: "text-muted border-line-2 bg-bg-2", help: "Listed so you know it exists and why it is not connected.", order: 4 },
};
const STEP: Record<string, { glyph: string; tone: string }> = { not_run: { glyph: "·", tone: "text-faint" }, ok: { glyph: "✓", tone: "text-up" }, done: { glyph: "✓", tone: "text-up" }, missing: { glyph: "○", tone: "text-warn" }, failed: { glyph: "✕", tone: "text-down" }, error: { glyph: "✕", tone: "text-down" }, skipped: { glyph: "–", tone: "text-faint" }, pending: { glyph: "·", tone: "text-faint" } };
const PRICE: Record<string, string> = { free: "free, no account", "free key": "free account", "free sample": "free sample", "free / low cost": "free or low cost", "free in the cloud": "free in their cloud", "free with account": "free with an account", usage: "pay per use", subscription: "subscription", "subscription or one-time": "subscription or one-time", "one-time": "one-time purchase", "per order": "per order", institutional: "institutional" };

function StatusBadge({ s, at }: { s: string; at?: string | null }) {
  const x = STATUS[s] || STATUS.unverified;
  return <span title={x.help} className={cn("inline-flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-[11.5px] font-medium whitespace-nowrap", x.tone)} data-testid="verification"><span aria-hidden>{x.glyph}</span>{x.word}{at ? <span className="font-normal opacity-70">· {ago(at)}</span> : null}</span>;
}

function AlpacaGuide({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["alpaca-validation"], queryFn: () => api.get<any>("/api/data/alpaca/validation"), refetchInterval: 6000 });
  const run = useMutation({ mutationFn: () => api.post<{ run_id: string }>("/api/data/alpaca/validate"), onSuccess: (r) => onJob(r.run_id), onError: (e: any) => toast.error(e.message) });
  const r = q.data;
  const plan = ["Test the stored credentials", "Discover what the account is entitled to", "Retrieve an expired SPY option contract", "Download its historical trade bars", "Run a small validation backtest on them"];
  return (
    <div className="mt-3 rounded-[10px] border border-info/30 bg-info/[0.05] px-3.5 py-3" data-testid="alpaca-guide">
      <div className="flex flex-wrap items-center gap-2">
        <PlugZap className="size-4 text-info" /><span className="text-[13.5px] font-medium">Guided connection</span>
        <span className="text-xs dim">Five real requests, in order. It stops at the first one that fails and says why.</span>
        <span className="flex-1" />
        <Link to="/settings" className="btn sm"><KeyRound />Enter the key in Settings</Link>
        <button className="btn primary sm" onClick={() => run.mutate()} disabled={run.isPending} data-testid="alpaca-run">{run.isPending ? <Loader2 className="animate-spin" /> : <Play />}Run it</button>
      </div>
      <ol className="mt-2.5 flex flex-col gap-1.5">
        {(r?.steps?.length ? r.steps : plan.map((t, i) => ({ id: i, title: t, status: "pending", detail: "" }))).map((st: any, i: number) => {
          const g = STEP[st.status] || STEP.pending;
          return (
            <li key={st.id ?? i} className="flex gap-2.5 text-[12.5px]">
              <span className={cn("w-4 shrink-0 text-center", g.tone)} title={st.status}>{g.glyph}</span>
              <div className="min-w-0"><span className="font-medium">{i + 1}. {st.title}</span><span className="dim"> · {st.status === "pending" || st.status === "not_run" ? "not run" : st.status}</span>{st.detail && <div className="soft">{st.detail}</div>}</div>
            </li>
          );
        })}
      </ol>
      {r?.summary && <div className={cn("mt-2 text-[12.5px]", r.verified ? "text-up" : "text-fg-2")}>{r.verified ? "● " : "○ "}{r.summary}{r.measured_at ? <span className="dim"> · {ago(r.measured_at)}</span> : null}</div>}
      <div className="mt-1.5 text-[11px] dim">The key is typed by you on the Settings page and stored on this computer. It is never shown again, logged, exported or sent to a language model. A free Alpaca paper account is enough for trade bars.</div>
    </div>
  );
}

function Card({ s, focus, onJob, busy, onCheck }: { s: any; focus: string; onJob: (id: string) => void; busy: boolean; onCheck: (name: string) => void }) {
  const v = s.verification || {};
  const steps: any[] = v.steps || [];
  return (
    <div className="card px-4 py-3.5" data-testid={`source-${s.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <a className="inline-flex items-center gap-1.5 text-[14.5px] font-medium hover:text-info" href={s.url} target="_blank" rel="noreferrer">{s.name}<ExternalLink className="size-3.5 dim" /></a>
        <span className="chip !h-[22px] !text-[11px]">{s.kind}</span>
        <span className="chip !h-[22px] !text-[11px]">{PRICE[s.pricing?.model] || s.pricing?.model}</span>
        <span className="flex-1" />
        <StatusBadge s={v.status} at={v.status === "verified" || String(v.status).startsWith("parser") ? v.checked_at : null} />
        {s.can_check && <button className="btn sm" disabled={busy} onClick={() => onCheck(s.check_name)} data-testid={`check-${s.id}`}><Play />Run the live check</button>}
      </div>

      <div className="mt-3 grid gap-x-8 gap-y-3 md:grid-cols-2">
        <div className="min-w-0">
          <div className="eyebrow mb-1">Advertised</div>
          <dl className="kv text-[12.5px]">
            <dt>Data</dt><dd className="soft">{s.data}</dd>
            <dt>Resolution</dt><dd className="soft">{s.resolution}</dd>
            <dt>History</dt><dd className="soft">{s.history}</dd>
            <dt>Price</dt><dd className="soft">{s.pricing?.summary}</dd>
            <dt>Account</dt><dd className="soft">{s.account}{(s.credentials || []).length > 0 && <span className="dim"> · needs {s.credentials.join(", ")}{(s.credentials_configured || []).length ? ` (${s.credentials_configured.length} of ${s.credentials.length} stored)` : " (none stored)"}</span>}</dd>
          </dl>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {Object.entries<string>(s.symbols || {}).map(([sym, what]) => (
              <span key={sym} title={what} className={cn("inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11.5px]", focus === sym ? "border-accent/60 bg-accent/[0.08]" : "border-line-2 bg-bg-2", (!what || /not offered|not available|none|no /i.test(what)) && "opacity-60")}>
                <b className="font-medium">{sym}</b><span className="dim max-w-[190px] truncate">{what || "not covered"}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="min-w-0">
          <div className="eyebrow mb-1">Verified on this installation</div>
          <div className="text-[12.5px] soft">{s.tested}</div>
          {v.detail && <div className={cn("mt-1 text-[12.5px]", v.status === "verified" ? "soft" : "text-warn")}>{v.detail}</div>}
          {steps.length > 0 && (
            <div className="mt-1.5 flex flex-col gap-0.5 text-xs">
              {steps.slice(0, 6).map((st: any, i: number) => (
                <div key={i} className={st.ok ? "soft" : "text-down"}>{st.ok ? "✓" : "✕"} {st.dataset || st.name}{st.ok ? [st.rows !== undefined && st.rows !== null && ` · ${Number(st.rows).toLocaleString()} rows`, st.first && ` · ${st.first} → ${st.last || ""}`, st.seconds !== undefined && st.seconds !== null && ` · ${Number(st.seconds).toFixed(1)}s`].filter(Boolean).join("") : ` · ${String(st.error || "failed").slice(0, 140)}`}</div>
              ))}
            </div>
          )}
          {(v.entitlements || []).map((e: any, i: number) => <div key={i} className={cn("text-xs", e.granted ? "text-up" : "text-warn")}>{e.granted ? "✓ granted" : "✕ not granted"}: {e.what}</div>)}
          {(v.depth || []).length > 0 && <div className="mt-1 text-xs dim">Measured depth: {v.depth.map((d: any) => `${d.what || d.dataset}: ${d.first || d.oldest || "?"}`).join(" · ")}</div>}
          <div className="mt-1.5 text-xs dim">Integration: {s.integration === "built-in" ? "built in (downloads from inside Strata)" : s.integration === "adapter" ? "adapter (needs your credentials)" : s.integration === "import" ? "file import (you obtain the files)" : "none"}</div>
        </div>
      </div>
      {(s.limits || []).length > 0 && <ul className="mt-2.5 list-disc pl-5 text-xs dim">{s.limits.map((l: string, i: number) => <li key={i}>{l}</li>)}</ul>}
      {s.id === "alpaca" && <AlpacaGuide onJob={onJob} />}
    </div>
  );
}

function Estimator() {
  const runs = useQuery({ queryKey: ["runs", "estimable"], queryFn: () => api.get<any[]>("/api/backtests?kind=backtest&limit=60"), staleTime: 10_000 });
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies"), staleTime: 10_000 });
  const [pick, setPick] = useState("");
  const est = useMutation({ mutationFn: () => api.post<any>("/api/data/marketplace/estimate", pick.startsWith("s:") ? { strategy_id: pick.slice(2) } : { run_id: pick.slice(2) }), onError: (e: any) => toast.error(e.message) });
  const r = est.data;
  return (
    <Section title="What would one strategy's data cost?" sub="only the observations it needs, per source" right={<Calculator className="size-4 dim" />}>
      <p className="text-[13px] soft">A strategy trades on a handful of sessions, so it needs option prices for those sessions and contracts only, not a whole archive. Pick a strategy: its signals are found by scanning the underlying (no option data is needed for that), then each source is costed for exactly those observations.</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <select className="field !w-[420px]" value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Strategy to estimate" data-testid="estimate-pick">
          <option value="">Choose a backtest or a saved strategy…</option>
          <optgroup label="Finished backtests">{(runs.data || []).filter((x) => x.status === "done").map((x) => <option key={x.id} value={`r:${x.id}`}>{x.name}</option>)}</optgroup>
          <optgroup label="Saved strategies (last two years)">{(strategies.data || []).map((x) => <option key={x.id} value={`s:${x.id}`}>{x.name} · {x.symbol}</option>)}</optgroup>
        </select>
        <button className="btn primary sm" disabled={!pick || est.isPending} onClick={() => est.mutate()} data-testid="estimate-run">{est.isPending ? <Loader2 className="animate-spin" /> : <Calculator />}Estimate</button>
      </div>
      {r && (
        <div className="mt-4" data-testid="estimate-result">
          <div className="text-[13.5px]"><b>{r.need.symbol}</b>: {r.need.signals.toLocaleString()} signals on {r.need.signal_sessions.toLocaleString()} sessions between {r.need.first} and {r.need.last} ({r.need.sessions_scanned.toLocaleString()} sessions scanned). Each needs about {r.need.contracts_per_signal} contracts over {r.need.sessions_per_signal} session{r.need.sessions_per_signal !== 1 ? "s" : ""}.
            <span className="dim"> Already stored: quotes for {r.need.sessions_with_quotes} of those sessions, trades for {r.need.sessions_with_trades}.</span></div>
          <div className="mt-3 overflow-auto rounded-lg border border-line">
            <table className="t">
              <thead><tr><th>Source</th><th>What it would supply</th><th className="r">Signal sessions covered</th><th className="r">Requests</th><th className="r">Time</th><th className="r">Cost</th><th>Before it can run</th></tr></thead>
              <tbody>
                {r.sources.map((x: any) => (
                  <tr key={x.id}>
                    <td className="font-medium">{x.name}</td><td className="soft">{x.what}<div className="text-[11px] dim">{x.window?.[0]} → {x.window?.[1]}</div></td>
                    <td className="r num">{x.covered_sessions} / {x.signal_sessions}<span className="ml-1.5 dim">{Math.round(x.coverage_pct)}%</span></td>
                    <td className="r num">{x.requests ?? "—"}</td><td className="r num">{x.minutes !== null && x.minutes !== undefined ? `${x.minutes < 1 ? "<1" : Math.round(x.minutes)} min` : "—"}</td>
                    <td className="r num">{x.dollars === 0 ? "free" : x.dollars === null || x.dollars === undefined ? "not quoted" : `$${Number(x.dollars).toLocaleString()}`}{x.dollars_note && <div className="max-w-[220px] whitespace-normal text-right font-sans text-[11px] dim">{x.dollars_note}</div>}</td>
                    <td className="max-w-[320px] whitespace-normal text-xs dim">
                      <span className={x.verification === "verified" ? "text-up" : "text-warn"}>{x.verification === "verified" ? "● verified here" : `○ ${x.verification}`}</span>
                      {(x.needs || []).length > 0 && !x.ready && <span> · needs {x.needs.join(", ")}</span>} {(x.notes || []).join(" ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(r.notes || []).map((n: string, i: number) => <div key={i} className="mt-1.5 text-xs dim">· {n}</div>)}
        </div>
      )}
    </Section>
  );
}

export function Marketplace({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["marketplace"], queryFn: () => api.get<any>("/api/data/marketplace"), refetchInterval: 8000 });
  const check = useMutation({ mutationFn: (source: string) => api.post<{ run_id: string }>("/api/data/marketplace/check", { source }), onSuccess: (r) => { onJob(r.run_id); toast.message("Live check started: a few small read-only requests"); }, onError: (e: any) => toast.error(e.message) });
  const [focus, setFocus] = useState("SPY");
  const [only, setOnly] = useState<"all" | "usable" | "free">("all");
  const groups = useMemo(() => {
    const list: any[] = (q.data?.sources || []).filter((s: any) => only === "all" || (only === "usable" ? s.integration !== "none" : /^free/.test(s.pricing?.model || "")));
    const by: Record<string, any[]> = {};
    for (const s of list) { const o = (STATUS[s.verification?.status] || STATUS.unverified).order; (by[o] ||= []).push(s); }
    return ([[0, "Verified from this installation"], [1, "Format verified, access not"], [2, "Connected in code, not yet verified here"], [3, "Files you obtain and import"], [4, "Known, not integrated"]] as const).map(([o, t]) => ({ title: t, items: by[o] || [] })).filter((g) => g.items.length);
  }, [q.data, only]);
  if (!q.data) return <div className="lab mx-auto flex max-w-[1180px] flex-col gap-3 p-4"><Skeleton className="h-20" /><Skeleton className="h-48" /></div>;
  const sm = q.data.summary || {};
  return (
    <div className="lab mx-auto flex max-w-[1180px] flex-col gap-4 px-2 pb-10 pt-3" data-testid="marketplace">
      <div>
        <h2>Where historical options data can come from</h2>
        <p className="mt-1 max-w-[860px] text-[13.5px] soft">{q.data.note}</p>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="chip" title={STATUS.verified.help}><span className="text-up">●</span>{sm.verified} verified here</span>
          <span className="chip" title={STATUS.unverified.help}><span className="text-warn">○</span>{sm.unverified} unverified</span>
          <span className="chip" title={STATUS["import only"].help}>▤ {sm.import_only} file import</span>
          <span className="chip" title={STATUS["no integration"].help}>– {sm.no_integration} not integrated</span>
          <span className="dim">Provider pages read on {q.data.checked}.</span>
          <span className="flex-1" />
          <span className="dim">Coverage for</span>
          {(q.data.priority_symbols || []).map((s: string) => <button key={s} className={cn("chip", focus === s && "on")} onClick={() => setFocus(s)}>{s}</button>)}
          <span className="mx-1 h-4 w-px bg-line-2" />
          {([["all", "All"], ["usable", "Usable here"], ["free", "Free"]] as const).map(([k, l]) => <button key={k} className={cn("chip", only === k && "on")} onClick={() => setOnly(k)}>{l}</button>)}
        </div>
      </div>
      <Callout tone="info" title={<span className="flex items-center gap-2"><ShieldCheck className="size-4" />How to read a card</span>}>
        The left column is the provider's own claim. The right column is what a real request from this computer returned. A source is only called verified when the right column has data in it; tests against recorded responses, or a key that was never entered, leave it unverified. Strata never calls a paid service by itself.
      </Callout>
      <Estimator />
      {groups.map((g) => (
        <div key={g.title} className="flex flex-col gap-3">
          <h3 className="mt-1 !text-[13px]">{g.title} <span className="font-normal dim">· {g.items.length}</span></h3>
          {g.items.map((s: any) => <Card key={s.id} s={s} focus={focus} onJob={onJob} busy={check.isPending} onCheck={(n) => check.mutate(n)} />)}
        </div>
      ))}
    </div>
  );
}
