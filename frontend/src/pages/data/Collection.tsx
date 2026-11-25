/* Data page panels added in V4: the background collector (settings, coverage calendar, history, storage, quality),
   live verification of what each source actually delivers, and streaming bulk imports with layout recognition. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, CircleAlert, ExternalLink, FolderSearch, Loader2, Play, Radio, ShieldCheck, ShieldQuestion, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Tip } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { ago, bytes, cn } from "@/lib/format";
import { Callout, Section, StatTile } from "../lab/bits";

const STATE: Record<string, { glyph: string; word: string; cls: string }> = {
  collected: { glyph: "●", word: "collected", cls: "bg-up/30 border-up/50 text-up" }, pending: { glyph: "…", word: "not due yet", cls: "bg-bg-3 border-line-2 text-muted" },
  missed: { glyph: "✕", word: "missed", cls: "bg-down/15 border-down/50 text-down" }, "not applicable": { glyph: "–", word: "no option grid", cls: "bg-transparent border-dashed border-line text-faint" },
};

/* ----------------------------------------------------------------------------- collector */
export function CollectorPanel({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["collector"], queryFn: () => api.get<any>("/api/collector"), refetchInterval: (s) => (s.state.data?.running ? 2500 : 30_000) });
  const [cfg, setCfg] = useState<any>(null);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (q.data && !dirty) setCfg(q.data.config); }, [q.data]); // eslint-disable-line
  const save = useMutation({ mutationFn: (c: any) => api.put<any>("/api/collector/config", c), onSuccess: (r) => { setDirty(false); setCfg(r.config); (r.notes || []).forEach((n: string) => toast.info(n)); queryClient.invalidateQueries({ queryKey: ["collector"] }); toast.success("Collector settings saved"); }, onError: (e: any) => toast.error(e.message) });
  const run = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/collector/run"), onSuccess: (r) => { onJob(r.id); queryClient.invalidateQueries({ queryKey: ["collector"] }); }, onError: (e: any) => toast.error(e.message) });
  const d = q.data;
  if (!d || !cfg) return <div className="lab p-4 text-[13px] dim">Loading the collector…</div>;
  const up = (patch: any) => { setCfg((c: any) => ({ ...c, ...patch })); setDirty(true); };
  const upo = (patch: any) => { setCfg((c: any) => ({ ...c, options: { ...c.options, ...patch } })); setDirty(true); };
  const last = d.last;
  const sessions: string[] = d.coverage.sessions;
  const used = d.storage.total_bytes / d.storage.limit_bytes;
  return (
    <div className="lab mx-auto flex max-w-[1180px] flex-col gap-4 p-1" data-testid="collector">
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <h2 className="flex items-center gap-2"><Radio className={cn("size-[18px]", cfg.enabled ? "text-up" : "text-faint")} />Market data collector</h2>
          <p className="mt-1 max-w-[760px] text-[13.5px] soft">Free sources drop intraday history after a few weeks, and an option contract's minute bars about a day after it expires. The collector saves them while they exist, so your own archive grows session by session.</p>
        </div>
        <label className="flex items-center gap-2.5 rounded-lg border border-line-2 bg-bg-2 px-3 py-2 text-[13px]"><input type="checkbox" className="size-4 accent-[#e8a33d]" checked={!!cfg.enabled} onChange={(e) => up({ enabled: e.target.checked })} data-testid="collector-enabled" />Collect automatically after each close</label>
        <button className="btn" disabled={run.isPending || d.running} onClick={() => run.mutate()} data-testid="collector-run">{d.running ? <Loader2 className="animate-spin" /> : <Play />}{d.running ? "Collecting…" : "Collect now"}</button>
      </div>
      <Callout tone="info" title="It only reads market data">{d.safety} {d.license[0]}</Callout>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Status" value={cfg.enabled ? "On" : "Off"} tone={cfg.enabled ? "text-up" : undefined} sub={d.next_due ? `next cycle ${new Date(d.next_due).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}` : "runs only when you press Collect now"} />
        <StatTile label="Last cycle" value={last ? ago(last.started_at) : "never"} sub={last ? `${last.status} · ${(last.report.bars || 0).toLocaleString()} option bars · ${last.report.requests || 0} requests` : "nothing collected yet"} tone={last?.status === "error" ? "text-down" : undefined} />
        <StatTile label="Missed sessions" value={d.coverage.missed.length} tone={d.coverage.missed.length ? "text-warn" : undefined} sub="in the last 45 sessions, gone from the source" hint="A session the collector did not reach while its option data was still being served. It cannot be filled in afterwards." />
        <StatTile label="Storage" value={bytes(d.storage.total_bytes)} sub={`${(used * 100).toFixed(0)}% of the ${cfg.storage_limit_gb} GB limit`} tone={used > 0.9 ? "text-warn" : undefined} />
      </div>
      <Section title="Coverage calendar" sub="option trade bars collected per session and symbol">
        <div className="overflow-auto">
          <table className="border-separate" style={{ borderSpacing: 2 }} data-testid="coverage-calendar">
            <thead><tr><th />{sessions.map((s, i) => <th key={s} className="num px-0 text-center text-[9.5px] font-normal text-faint" style={{ minWidth: 20 }}>{i === 0 || s.slice(5, 7) !== sessions[i - 1].slice(5, 7) ? s.slice(5, 7) + "/" : ""}{s.slice(8)}</th>)}</tr></thead>
            <tbody>
              {Object.entries(d.coverage.symbols as Record<string, any[]>).map(([sym, rows]) => (
                <tr key={sym}><th className="num pr-2 text-right text-xs font-medium">{sym}</th>
                  {rows.map((r) => { const s = STATE[r.state] || STATE.pending; return (
                    <td key={r.date}><Tip content={`${sym} ${r.date}: ${s.word}\nunderlying bars: ${r.underlying || "daily only"}\n${r.contracts_with_trades} contracts with trades, ${r.contracts_without_trades} without${Object.keys(r.by_bucket || {}).length ? "\nby days to expiration: " + Object.entries(r.by_bucket).map(([b, v]: any) => `${b}d ${v.ok}`).join(", ") : ""}`}>
                      <div className={cn("grid h-6 w-5 place-items-center rounded-[3px] border text-[10px] leading-none", s.cls)} data-state={r.state}>{s.glyph}</div></Tip></td>
                  ); })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs dim">{Object.entries(STATE).map(([k, s]) => <span key={k}><span className="text-fg">{s.glyph}</span> {s.word}</span>)}</div>
        {d.coverage.missed.length > 0 && <div className="mt-2 text-xs text-warn">Missed: {d.coverage.missed.slice(-6).map((m: any) => `${m.symbol} ${m.date}`).join(", ")}{d.coverage.missed.length > 6 ? ` and ${d.coverage.missed.length - 6} more` : ""}. {d.coverage.missed[0].reason}.</div>}
      </Section>
      <Section title="What to collect" right={dirty ? <button className="btn primary sm" disabled={save.isPending} onClick={() => save.mutate(cfg)} data-testid="collector-save">Save settings</button> : <span className="text-xs dim">saved</span>}>
        <div className="grid gap-x-6 gap-y-3 md:grid-cols-3">
          <label className="text-xs dim">Watch-list symbols<input className="field mt-1" value={cfg.symbols.join(", ")} onChange={(e) => up({ symbols: e.target.value.split(/[\s,]+/).filter(Boolean) })} aria-label="Watch-list symbols" /></label>
          <label className="text-xs dim">Underlying bar resolutions
            <div className="mt-1 flex gap-1.5">{["1m", "5m", "60m"].map((r) => <button type="button" key={r} className={cn("chip", cfg.bar_resolutions.includes(r) && "on")} onClick={() => up({ bar_resolutions: cfg.bar_resolutions.includes(r) ? cfg.bar_resolutions.filter((x: string) => x !== r) : [...cfg.bar_resolutions, r] })}>{r}</button>)}</div></label>
          <label className="text-xs dim">Collect after (New York time)<input className="field mt-1" value={cfg.run_after} onChange={(e) => up({ run_after: e.target.value })} aria-label="Collect after" /></label>
          <label className="text-xs dim">Option expirations (days out)
            <div className="mt-1 flex flex-wrap gap-1.5">{[0, 1, 3, 7, 14, 30].map((b) => <button type="button" key={b} className={cn("chip", cfg.options.dte.includes(b) && "on")} onClick={() => upo({ dte: cfg.options.dte.includes(b) ? cfg.options.dte.filter((x: number) => x !== b) : [...cfg.options.dte, b].sort((x, y) => x - y) })}>{b}DTE</button>)}</div></label>
          <label className="text-xs dim">Strike range around the previous close (%)<input type="number" step={0.25} min={0.25} max={10} className="field mt-1" value={cfg.options.strike_pct} onChange={(e) => upo({ strike_pct: Number(e.target.value) })} aria-label="Strike range" /></label>
          <label className="text-xs dim">Strikes on each side, at most<input type="number" min={1} max={40} className="field mt-1" value={cfg.options.max_strikes_each_side} onChange={(e) => upo({ max_strikes_each_side: Number(e.target.value) })} aria-label="Strikes each side" /></label>
          <label className="text-xs dim">Calls and puts
            <div className="mt-1 flex gap-1.5">{[["C", "Calls"], ["P", "Puts"]].map(([r, l]) => <button type="button" key={r} className={cn("chip", cfg.options.rights.includes(r) && "on")} onClick={() => upo({ rights: cfg.options.rights.includes(r) ? cfg.options.rights.filter((x: string) => x !== r) : [...cfg.options.rights, r] })}>{l}</button>)}
              <button type="button" className={cn("chip", !cfg.options.enabled && "on")} onClick={() => upo({ enabled: !cfg.options.enabled })}>{cfg.options.enabled ? "options on" : "options off"}</button></div></label>
          <label className="text-xs dim">Requests per cycle, at most<input type="number" min={50} className="field mt-1" value={cfg.max_requests_per_cycle} onChange={(e) => up({ max_requests_per_cycle: Number(e.target.value) })} aria-label="Requests per cycle" /></label>
          <label className="text-xs dim">Sessions looked back each cycle<input type="number" min={1} max={20} className="field mt-1" value={cfg.lookback_sessions} onChange={(e) => up({ lookback_sessions: Number(e.target.value) })} aria-label="Lookback sessions" /></label>
          <label className="text-xs dim">Option requests per second<input type="number" step={0.5} min={0.2} max={10} className="field mt-1" value={cfg.rate_per_second[cfg.options.provider] ?? 4} onChange={(e) => up({ rate_per_second: { ...cfg.rate_per_second, [cfg.options.provider]: Number(e.target.value) } })} aria-label="Requests per second" /></label>
          <label className="text-xs dim">Storage limit (GB)<input type="number" step={1} min={0.5} className="field mt-1" value={cfg.storage_limit_gb} onChange={(e) => up({ storage_limit_gb: Number(e.target.value) })} aria-label="Storage limit" /></label>
          <label className="text-xs dim">Automatic clean-up
            <div className="mt-1 flex items-center gap-2"><button type="button" className={cn("chip", cfg.cleanup.enabled && "on")} onClick={() => up({ cleanup: { ...cfg.cleanup, enabled: !cfg.cleanup.enabled } })}>{cfg.cleanup.enabled ? "on" : "off"}</button>
              <input type="number" min={0} className="field !w-20" value={cfg.cleanup.keep_option_trade_days} onChange={(e) => up({ cleanup: { ...cfg.cleanup, keep_option_trade_days: Number(e.target.value) } })} aria-label="Keep option days" /><span>days of option bars kept (0 = keep all)</span></div></label>
        </div>
        <div className="mt-3 text-xs dim">Strikes are chosen around the close BEFORE each session and trimmed nearest-the-money first when the request budget is reached. Nothing is deleted unless clean-up is switched on. {d.how_to_schedule}</div>
      </Section>
      <Section title="Collection history" sub={`${d.history.length} recent cycles`} open={d.history.length > 0}>
        {d.history.length === 0 ? <div className="text-[13px] dim">No cycle has run yet.</div> : (
          <div className="max-h-72 overflow-auto rounded-lg border border-line">
            <table className="t" data-testid="collector-history"><thead><tr><th>Started</th><th>Status</th><th>Trigger</th><th className="r">Option bars</th><th className="r">Requests</th><th className="r">Seconds</th><th>Notes</th></tr></thead>
              <tbody>{d.history.map((h: any) => (
                <tr key={h.id}><td className="num whitespace-nowrap">{h.started_at.slice(0, 16).replace("T", " ")}</td>
                  <td className={cn("whitespace-nowrap", h.status === "done" ? "text-up" : h.status === "error" ? "text-down" : "text-warn")}>{h.status === "done" ? "✓ " : h.status === "error" ? "✕ " : "! "}{h.status}</td>
                  <td className="dim">{h.trigger}</td><td className="r num">{(h.report.bars || 0).toLocaleString()}</td><td className="r num">{h.report.requests ?? "—"}</td><td className="r num">{h.report.seconds ?? "—"}</td>
                  <td className="max-w-[420px] truncate text-xs dim" title={[...(h.report.errors || []), ...(h.report.notes || [])].join("\n")}>{[...(h.report.errors || []), ...(h.report.notes || [])][0] || ""}</td></tr>
              ))}</tbody></table>
          </div>
        )}
      </Section>
      <Section title="Data quality of what was collected" open={false}>
        <div className="overflow-auto rounded-lg border border-line">
          <table className="t"><thead><tr><th>Symbol</th><th className="r">Contract-sessions</th><th className="r">With trades</th><th className="r">Without</th><th className="r">Thinly traded</th><th className="r">Median bars</th><th>Range</th></tr></thead>
            <tbody>{d.quality.map((x: any) => <tr key={x.symbol}><td>{x.symbol}</td><td className="r num">{x.contract_sessions.toLocaleString()}</td><td className="r num">{x.with_trades?.toLocaleString() ?? "—"}</td><td className="r num">{x.without_trades_pct != null ? `${x.without_trades_pct}%` : "—"}</td><td className="r num">{x.thinly_traded_pct != null ? `${x.thinly_traded_pct}%` : "—"}</td><td className="r num">{x.median_bars_per_contract ?? "—"}</td><td className="num dim">{x.first ? `${x.first} → ${x.last}` : x.note}</td></tr>)}</tbody></table>
        </div>
        <div className="mt-2 text-xs dim">A contract without trades did not print in the session. A thinly traded one printed in fewer than ten minutes, so its last price is often stale; the engine refuses stale prints rather than using them.</div>
      </Section>
    </div>
  );
}

/* ----------------------------------------------------------------------------- source verification */
const VER: Record<string, { Icon: any; cls: string }> = { verified: { Icon: ShieldCheck, cls: "text-up border-up/40" }, "partly verified": { Icon: CircleAlert, cls: "text-warn border-warn/40" }, failed: { Icon: CircleAlert, cls: "text-down border-down/40" } };

export function SourcesPanel({ onJob }: { onJob: (id: string) => void }) {
  const q = useQuery({ queryKey: ["provider-catalog"], queryFn: () => api.get<any>("/api/data/providers/catalog"), refetchInterval: 8000 });
  const check = useMutation({ mutationFn: (source: string) => api.post<{ id: string }>("/api/data/providers/benchmark", { source }), onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message) });
  if (!q.data) return <div className="lab p-4 text-[13px] dim">Loading sources…</div>;
  return (
    <div className="lab mx-auto flex max-w-[1180px] flex-col gap-4 p-1" data-testid="sources-panel">
      <div><h2>What each source actually delivers</h2><p className="mt-1 max-w-[780px] text-[13.5px] soft">{q.data.note} A live check makes a handful of small read-only requests and records depth, speed, missing bars and entitlements. It stores nothing in your market data.</p></div>
      <div className="flex flex-col gap-3">
        {q.data.sources.map((s: any) => {
          const b = s.benchmark, v = VER[s.verification] || { Icon: ShieldQuestion, cls: "text-warn border-warn/40" };
          return (
            <div key={s.name} className="card px-4 py-3.5" data-testid={`source-${s.name}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[14px] font-medium">{s.label}</span>
                <span className="chip !h-5 !text-[11px]">{s.tier === "free" ? "free, no account" : s.tier === "free_key" ? "free account" : "paid"}</span>
                <span className={cn("chip !h-6", v.cls)} data-testid="verification"><v.Icon className="size-3.5" />{s.verification}{b?.measured_at ? ` · ${ago(b.measured_at)}` : ""}</span>
                <span className="flex-1" />
                <button className="btn sm" disabled={check.isPending} onClick={() => check.mutate(s.name)}><Play />Run the live check</button>
              </div>
              <div className="mt-1 text-[13px] soft">{s.description}</div>
              {!s.available && <div className="mt-1.5 text-xs text-warn">{s.availability}</div>}
              {b?.reason && <div className="mt-1.5 text-xs text-warn">{b.reason}</div>}
              <div className="mt-2 grid gap-x-6 gap-y-1 text-xs md:grid-cols-2">
                <div><div className="mb-0.5 font-medium dim">Documented</div>{s.claim.map((c: any, i: number) => <div key={i} className="dim">· {c.data} {c.observation} {c.resolution}{c.first ? `, ${c.first} → ${c.last}` : ""}{c.note ? ` (${c.note})` : ""}</div>)}</div>
                <div><div className="mb-0.5 font-medium dim">Measured here</div>
                  {!b?.steps?.length ? <div className="text-faint">· nothing measured yet</div> : b.steps.map((st: any, i: number) => (
                    <div key={i} className={st.ok ? "soft" : "text-down"}>{st.ok ? "✓" : "✕"} {st.dataset || st.name}: {st.ok ? [st.rows != null && `${Number(st.rows).toLocaleString()} rows`, st.first && `${st.first} → ${st.last || ""}`, st.missing_pct != null && `${st.missing_pct}% of expected bars missing`, st.contract_coverage_pct != null && `${st.contract_coverage_pct}% of contracts had trades`, `${st.seconds}s`].filter(Boolean).join(" · ") : st.error}</div>
                  ))}
                  {(b?.entitlements || []).map((e: any, i: number) => <div key={`e${i}`} className={e.granted ? "text-up" : "text-warn"}>{e.granted ? "✓ granted" : "✕ not granted"}: {e.what} <span className="dim">({String(e.detail).slice(0, 90)})</span></div>)}
                  {b?.bars_per_second && <div className="dim">· {b.bars_per_second.toLocaleString()} bars per second{Object.keys(b.rate_limits || {}).length ? ` · rate-limit headers: ${Object.entries(b.rate_limits).map(([k, x]) => `${k}=${x}`).join(", ")}` : " · no rate-limit headers sent"}</div>}
                </div>
              </div>
              {(b?.notes || []).map((n: string, i: number) => <div key={i} className="mt-1.5 text-xs dim">· {n}</div>)}
            </div>
          );
        })}
      </div>
      <Section title="Other legitimate sources" sub="terms read from public pages, not measured by Strata">
        <div className="overflow-auto rounded-lg border border-line">
          <table className="t"><thead><tr><th>Source</th><th>Cost</th><th>Options</th><th>Underlying</th><th>How to use it here</th></tr></thead>
            <tbody>{q.data.external.map((e: any) => (
              <tr key={e.name}><td className="whitespace-nowrap"><a className="inline-flex items-center gap-1 text-info hover:underline" href={e.url} target="_blank" rel="noreferrer">{e.name}<ExternalLink className="size-3" /></a><div className="text-[11px] text-faint">{e.integrated === "import" ? "via bulk import" : e.integrated ? "built in" : "not integrated"}</div></td>
                <td className="max-w-[190px] text-xs soft">{e.cost}</td><td className="max-w-[260px] text-xs soft">{e.options}</td><td className="max-w-[180px] text-xs soft">{e.underlying}</td><td className="max-w-[260px] text-xs dim">{e.how}<div className="text-faint">{e.limits}</div></td></tr>
            ))}</tbody></table>
        </div>
        <div className="mt-2 text-xs dim">Strata never calls a paid service on its own. Paid data enters only as files you bought and import yourself, or after you approve a stated cost.</div>
      </Section>
    </div>
  );
}

/* ----------------------------------------------------------------------------- bulk import */
export function BulkImportPanel({ onJob }: { onJob: (id: string) => void }) {
  const [path, setPath] = useState("");
  const [symbol, setSymbol] = useState("");
  const [provider, setProvider] = useState("import");
  const [info, setInfo] = useState<any>(null);
  const inspect = useMutation({ mutationFn: () => api.post<any>("/api/data/import/inspect", { path }), onSuccess: setInfo, onError: (e: any) => { setInfo(null); toast.error(e.message); } });
  const go = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/import/bulk", { path, symbol: symbol || null, provider: provider || "import" }), onSuccess: (r) => { onJob(r.id); toast.success("Import started"); }, onError: (e: any) => toast.error(e.message) });
  return (
    <div className="lab mx-auto flex max-w-[1000px] flex-col gap-4 p-1" data-testid="bulk-import">
      <div><h2>Import historical files</h2><p className="mt-1 max-w-[760px] text-[13.5px] soft">Point at a file, a ZIP archive or a whole folder on this computer. Vendor layouts are recognised from their column names, and files are streamed, so several years of option data import without being loaded into memory.</p></div>
      <div className="card flex flex-col gap-3 px-4 py-3.5">
        <label className="text-xs dim">Path on this computer<input className="field mt-1" placeholder="/Users/you/Downloads/spy_options_2019_2023.zip" value={path} onChange={(e) => { setPath(e.target.value); setInfo(null); }} aria-label="Path" data-testid="import-path" /></label>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs dim">Symbol (only if the files do not name it)<input className="field mt-1 !w-36" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} aria-label="Symbol" /></label>
          <label className="text-xs dim">Dataset name<input className="field mt-1 !w-44" value={provider} onChange={(e) => setProvider(e.target.value.replace(/[^a-z0-9_]/gi, "").toLowerCase())} aria-label="Dataset name" /></label>
          <button className="btn" disabled={!path || inspect.isPending} onClick={() => inspect.mutate()} data-testid="import-inspect">{inspect.isPending ? <Loader2 className="animate-spin" /> : <FolderSearch />}Look at it first</button>
          <button className="btn primary" disabled={!info?.files?.length || go.isPending} onClick={() => go.mutate()} data-testid="import-go"><Upload />Import</button>
        </div>
      </div>
      {info && (
        <div className="card px-4 py-3.5">
          <div className="flex flex-wrap items-center gap-2 text-[13px]"><span className="font-medium">{info.files.length} importable file{info.files.length !== 1 ? "s" : ""}</span><span className="dim">{bytes(info.bytes)}</span>{info.kinds.map((k: string) => <span key={k} className="chip !h-5 !text-[11px]">{k.replace(/_/g, " ")}</span>)}</div>
          {info.note && <div className="mt-1.5 text-xs dim">{info.note}</div>}
          <div className="mt-2.5 flex flex-col gap-1.5">
            {info.files.slice(0, 30).map((f: any) => (
              <div key={f.path} className="flex items-baseline gap-2 text-xs"><Check className="size-3.5 shrink-0 translate-y-0.5 text-up" /><span className="min-w-0 flex-1 truncate" title={f.path}>{f.path.split("/").slice(-1)[0]}{f.archive ? ` (${f.members} files inside, ${bytes(f.uncompressed_bytes)} uncompressed)` : ""}</span><span className="soft">{f.label}</span></div>
            ))}
            {info.unrecognised.map((f: any) => <div key={f.path} className="flex items-baseline gap-2 text-xs text-warn"><CircleAlert className="size-3.5 shrink-0 translate-y-0.5" /><span className="min-w-0 flex-1 truncate">{f.path.split("/").slice(-1)[0]}</span><span>not recognised: {f.columns.slice(0, 6).join(", ")}</span></div>)}
          </div>
        </div>
      )}
      <Section title="Layouts recognised automatically" open={false}>
        <ul className="grid gap-x-6 gap-y-1 text-[12.5px] soft md:grid-cols-2">{Object.values((info?.formats as Record<string, string>) || FORMATS).map((l) => <li key={l}>· {l}</li>)}</ul>
        <div className="mt-2 text-xs dim">Trade bars are stored as trade prints and quotes as quotes; the two are never mixed. Rows with crossed or missing quotes are counted and reported, never repaired. Imported files form their own dataset and never overwrite downloaded data.</div>
      </Section>
    </div>
  );
}
const FORMATS = { a: "OptionsDX (wide: calls and puts side by side)", b: "Cboe DataShop end-of-day and interval option quotes", c: "historicaloptiondata.com option quotes", d: "DoltHub option_chain export", e: "ThetaData end-of-day export",
  f: "Polygon / Massive option and stock aggregates (flat files)", g: "Databento OHLCV bars with OCC symbols", h: "Stooq bars", i: "Yahoo Finance daily CSV", j: "Headerless bars (Kibot / FirstRate)", k: "Any file with expiration, strike, type, bid and ask" };
