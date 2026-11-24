import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, CircleAlert, Database, Download, ExternalLink, FlaskConical, Loader2, PlugZap, RefreshCw, ShieldCheck, Table2, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { Empty, ErrorBox, JobBar, Loading, PageHeader, Panel, SyntheticBadge, Warnings } from "@/components/common";
import { Button, Field, Input, NumInput, Select, Switch, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui";
import { api, queryClient, useJob, useMeta } from "@/lib/api";
import { bytes, cn, compact, etTime, num, pct } from "@/lib/format";

const refreshData = () => {
  for (const k of ["catalog", "coverage", "storage", "meta", "symbol", "sources"]) queryClient.invalidateQueries({ queryKey: [k] });
};

export default function DataPage() {
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage") });
  const storage = useQuery({ queryKey: ["storage"], queryFn: () => api.get<any>("/api/data/storage") });
  const [sel, setSel] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [tab, setTab] = useState("coverage");
  const job = useJob(jobId, (s) => {
    if (s.status === "done") toast.success("Data job finished");
    if (s.status === "error") toast.error(`Data job failed: ${s.message}`);
    refreshData();
  });
  const rows = cov.data || [];
  const real = rows.filter((r) => !r.synthetic);
  const sandbox = rows.filter((r) => r.synthetic);
  const cur = sel && rows.some((r) => r.symbol === sel) ? sel : real.find((r) => r.options)?.symbol || real[0]?.symbol || rows[0]?.symbol;
  useEffect(() => { if (cov.isSuccess && real.length === 0) setTab("free"); }, [cov.isSuccess]); // eslint-disable-line
  const market = Object.values(storage.data?.market || {}).reduce((a: number, b: any) => a + b, 0) as number;
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Data" sub={<span className="num text-2xs text-muted">{storage.data?.data_dir} · market {bytes(market)} · results {bytes(storage.data?.results)} · download cache {bytes(storage.data?.cache)}</span>}>
        <JobBar job={job} />
        <Button variant="ghost" onClick={() => { api.get("/api/data/catalog?refresh=true").then(refreshData); }}><RefreshCw /> Rescan</Button>
      </PageHeader>
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList>
          <TabsTrigger value="coverage"><Database className="size-3" /> Coverage</TabsTrigger>
          <TabsTrigger value="free"><Download className="size-3" /> Get free data</TabsTrigger>
          <TabsTrigger value="chain"><Table2 className="size-3" /> Option chain</TabsTrigger>
          <TabsTrigger value="providers"><PlugZap className="size-3" /> Paid providers</TabsTrigger>
          <TabsTrigger value="import"><Upload className="size-3" /> Import files</TabsTrigger>
          <TabsTrigger value="sandbox"><FlaskConical className="size-3" /> Developer sandbox</TabsTrigger>
        </TabsList>
        <div className="min-h-0 flex-1 overflow-auto">
          <TabsContent value="coverage">
            {cov.isLoading ? <Loading label="Scanning datasets" /> : cov.error ? <ErrorBox className="m-2" error={cov.error} /> : real.length === 0 ? (
              <Empty title="No real market data stored yet" icon={Database} action={<Button variant="primary" onClick={() => setTab("free")}><Download /> Get free data</Button>}>
                Daily history since 1993, recent intraday bars and end-of-day option chains are available without an account.
              </Empty>
            ) : <>
              <CoverageMatrix rows={real} cur={cur} onPick={setSel} />
              {sandbox.length > 0 && <div className="border-t border-line bg-bg-1 px-3 py-1.5 text-2xs text-muted"><FlaskConical className="mr-1 inline size-3 text-synth" />Simulated sandbox symbols also stored (never used by default): {sandbox.map((r) => <button key={r.symbol} onClick={() => setSel(r.symbol)} className="mx-1 text-synth hover:underline">{r.symbol}</button>)}</div>}
              {cur && <SymbolDetail sym={cur} />}
            </>}
          </TabsContent>
          <TabsContent value="free"><FreeData onJob={setJobId} busy={!!job && ["queued", "running"].includes(job.status)} /></TabsContent>
          <TabsContent value="chain">{rows.some((r) => r.options) ? <ChainPicker rows={rows.filter((r) => r.options)} /> : <Empty title="No option chains stored" icon={Table2}>Download free end-of-day chains under “Get free data”.</Empty>}</TabsContent>
          <TabsContent value="providers"><Providers onJob={setJobId} /></TabsContent>
          <TabsContent value="import"><ImportForm onJob={setJobId} /></TabsContent>
          <TabsContent value="sandbox"><DemoForm onJob={setJobId} /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

/** One row per symbol, one column per dataset; the last column says in words what cannot be answered. */
function CoverageMatrix({ rows, cur, onPick }: { rows: any[]; cur?: string; onPick: (s: string) => void }) {
  const RES = ["1m", "5m", "15m", "30m", "60m"];
  const cell = (ok: boolean, body: React.ReactNode) => <td className={cn("num", ok ? "text-fg-2" : "text-faint")}>{body}</td>;
  return (
    <table className="tbl">
      <thead><tr><th>Symbol</th><th>Daily bars</th><th>Intraday bars (sessions stored)</th><th>Option quotes</th><th className="r">Stored</th><th>What this cannot answer yet</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.symbol} onClick={() => onPick(r.symbol)} className={cn("cursor-pointer", cur === r.symbol && "[&>td]:bg-accent/[0.08]")}>
            <td className="font-medium text-fg">{r.symbol} <span className="ml-1 text-2xs font-normal text-faint">{r.kind}</span></td>
            {cell(!!r.daily, r.daily ? <>{r.daily.first} → {r.daily.last}{r.daily.missing ? <span className="ml-1 text-warn">({r.daily.missing} missing)</span> : null}</> : "—")}
            <td className="num">{RES.filter((k) => r.intraday?.[k]).length ? RES.filter((k) => r.intraday?.[k]).map((k) => <span key={k} title={`${k}: ${r.intraday[k].first} → ${r.intraday[k].last}`} className="mr-1.5 inline-flex items-baseline gap-1"><span className="rounded-[2px] border border-line-2 px-1 text-[9.5px] text-fg-2">{k}</span><span className="text-fg-2">{r.intraday[k].sessions}d</span><span className="text-faint">from {r.intraday[k].first.slice(2)}</span></span>) : <span className="text-faint">—</span>}</td>
            {cell(!!r.options, r.options ? <><span className={cn("mr-1 rounded-[2px] border px-1 text-[9.5px]", r.options.resolution === "EOD" ? "border-warn/50 text-warn" : "border-up/50 text-up")}>{r.options.resolution}</span>{r.options.first} → {r.options.last}<span className="ml-1 text-faint">{compact(r.options.rows)} rows · {r.options.provider}</span></> : "—")}
            <td className="r num text-muted">{bytes(r.storage_bytes)}</td>
            <td className="min-w-[260px] max-w-[420px] whitespace-normal text-2xs leading-snug text-muted">{r.gaps.length ? r.gaps.join("; ") : <span className="text-up">nothing missing for intraday option backtests</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FreeData({ onJob, busy }: { onJob: (id: string) => void; busy: boolean }) {
  const src = useQuery({ queryKey: ["sources"], queryFn: () => api.get<any[]>("/api/data/sources") });
  const [quick, setQuick] = useState("SPY, QQQ, IWM");
  const [verify, setVerify] = useState<any>(null);
  const qs = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/quickstart", { symbols: quick.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean), intraday: true }), onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message) });
  const ver = useMutation({ mutationFn: () => api.get<any>("/api/data/verify/SPY"), onSuccess: setVerify, onError: (e: any) => toast.error(e.message) });
  return (
    <div className="flex flex-col gap-px bg-line">
      <div className="grid grid-cols-[1.2fr_1fr] gap-px bg-line">
        <Panel className="border-0" title="Start here: free underlying history (no account)">
          <div className="flex flex-col gap-2 p-3">
            <p className="text-xs leading-relaxed text-fg-2">Daily bars for each symbol's full listing history with dividends and splits, recent intraday bars (60-minute for about two years, 5-minute for about two months, 1-minute for about a month) and the official VIX history. A few megabytes; enough to run market studies immediately.</p>
            <div className="flex items-end gap-2">
              <Field label="Symbols" className="flex-1"><Input className="num uppercase" value={quick} onChange={(e) => setQuick(e.target.value)} /></Field>
              <Button variant="primary" disabled={qs.isPending || busy} onClick={() => qs.mutate()}>{qs.isPending || busy ? <Loader2 className="animate-spin" /> : <Download />} Download</Button>
            </div>
            <p className="text-2xs text-faint">Sources: Yahoo Finance chart data and Cboe index history. Re-running only fetches what is new.</p>
          </div>
        </Panel>
        <Panel className="border-0" title="Verify what was downloaded">
          <div className="flex flex-col gap-2 p-3 text-xs">
            <p className="leading-relaxed text-fg-2">Compares stored SPY daily closes with an independent published price table, so you can see the data matches a second source.</p>
            <Button className="self-start" disabled={ver.isPending} onClick={() => ver.mutate()}>{ver.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />} Cross-check SPY</Button>
            {verify && (verify.compared ? (
              <div className="rounded-sm border border-up/40 bg-up/[0.06] px-2.5 py-1.5 text-2xs leading-relaxed text-fg-2">
                <span className="num text-fg">{verify.compared.toLocaleString()}</span> sessions compared ({verify.first} → {verify.last}). <span className="num text-fg">{(verify.within_0_1pct * 100).toFixed(2)}%</span> of closes agree within 0.1%; the largest difference is <span className="num text-fg">{verify.max_close_diff_pct.toFixed(3)}%</span>.
                <div className="text-faint">Reference: {verify.reference}</div>
              </div>
            ) : <div className="text-2xs text-warn">No overlapping sessions to compare - download SPY daily bars first.</div>)}
          </div>
        </Panel>
      </div>
      {src.isLoading ? <Loading /> : src.error ? <ErrorBox className="m-2" error={src.error} /> : (
        <div className="grid grid-cols-2 gap-px bg-line">{(src.data || []).map((s) => <SourceCard key={s.name} s={s} onJob={onJob} busy={busy} />)}</div>
      )}
      <div className="bg-bg-1 px-3 py-2 text-2xs leading-relaxed text-muted">
        Intraday option <b className="font-medium text-fg-2">quotes</b> (needed for 0DTE and time-of-day option backtests) are not offered free by any source we could verify. Options to add them: a free-trial or paid feed under “Paid providers” (ThetaData, Databento, Polygon/Massive), or your own files under “Import files” (the OptionsDX free monthly samples import directly). Until then the lab answers those requests with a market study, or - only if you opt in - a clearly labelled modeled estimate.
      </div>
    </div>
  );
}

function SourceCard({ s, onJob, busy }: { s: any; onJob: (id: string) => void; busy: boolean }) {
  const year = new Date().getFullYear();
  const optionSrc = s.datasets.some((d: any) => d.id === "options_eod");
  const [f, setF] = useState({ dataset: s.datasets[0].id, symbols: s.name === "cboe" ? "VIX" : "SPY", start: optionSrc ? `${year - 5}-01-01` : "", end: "" });
  const body = () => ({ source: s.name, dataset: f.dataset, symbols: f.symbols.split(",").map((x) => x.trim().toUpperCase()).filter(Boolean), start: f.start || null, end: f.end || null });
  const plan = useMutation({ mutationFn: () => api.post<any>("/api/data/sources/plan", body()) });
  const go = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/sources/ingest", body()), onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message) });
  const ds = s.datasets.find((d: any) => d.id === f.dataset) || s.datasets[0];
  const tier = { free: ["free · no account", "border-up/50 text-up"], free_key: ["free with an account key", "border-info/50 text-info"], paid: ["paid", "border-warn/50 text-warn"] }[s.tier as string] || [s.tier, "border-line-2 text-muted"];
  return (
    <Panel className="border-0" title={<span className="flex items-center gap-2">{s.label}<span className={cn("rounded-[2px] border px-1 text-[9.5px] normal-case tracking-normal", tier[1])}>{tier[0]}</span></span>}
      right={<a href={s.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-2xs text-muted hover:text-fg">source <ExternalLink className="size-3" /></a>}>
      <div className="flex flex-col gap-2 p-3">
        <p className="text-xs leading-relaxed text-fg-2">{s.description}</p>
        <div className="grid grid-cols-[1.2fr_1fr] gap-2">
          <Field label="Dataset"><Select sz="sm" value={f.dataset} onChange={(v) => { setF({ ...f, dataset: v }); plan.reset(); }} options={s.datasets.map((d: any) => ({ value: d.id, label: `${d.label} (${d.resolution})` }))} /></Field>
          <Field label="Symbols"><Input sz="sm" className="num uppercase" disabled={s.name === "cboe"} value={f.symbols} onChange={(e) => { setF({ ...f, symbols: e.target.value }); plan.reset(); }} /></Field>
          {(optionSrc || s.name === "alpaca") && <>
            <Field label="From"><Input sz="sm" type="date" className="num" value={f.start} onChange={(e) => { setF({ ...f, start: e.target.value }); plan.reset(); }} /></Field>
            <Field label="To (blank = latest)"><Input sz="sm" type="date" className="num" value={f.end} onChange={(e) => { setF({ ...f, end: e.target.value }); plan.reset(); }} /></Field>
          </>}
        </div>
        <div className="text-2xs text-muted">Coverage: {ds.coverage}{ds.quote_type === "trades" ? " · trade prints only (no bid/ask): fills use last-trade prices and are flagged" : ""} · {s.license}</div>
        {!s.available && <div className="flex items-center gap-1.5 text-2xs text-warn"><CircleAlert className="size-3" /> {s.status}</div>}
        <div className="flex items-center gap-2">
          <Button size="sm" disabled={plan.isPending || !s.available} onClick={() => plan.mutate()}>{plan.isPending ? <Loader2 className="animate-spin" /> : null} Estimate size</Button>
          <Button size="sm" variant="primary" disabled={!s.available || go.isPending || busy} onClick={() => { const big = plan.data?.est_bytes > 300e6; if (!big || confirm(`This download needs about ${bytes(plan.data.est_bytes)} of disk space. Continue?`)) go.mutate(); }}><Download /> Download</Button>
          {plan.data && <span className="num text-2xs text-fg-2">{plan.data.requests} request(s) · ~{bytes(plan.data.est_bytes)} on disk · ~{plan.data.est_seconds < 90 ? `${plan.data.est_seconds}s` : `${Math.round(plan.data.est_seconds / 60)} min`}</span>}
          {plan.error ? <span className="text-2xs text-down">{(plan.error as any).message}</span> : null}
        </div>
        {plan.data?.detail && <div className="num text-2xs text-faint">{Object.entries(plan.data.detail).map(([k, v]: any) => `${k}: ${v.years ? `${v.years[0]}–${v.years[v.years.length - 1]}` : JSON.stringify(v)}${v.years_already_stored?.length ? ` (${v.years_already_stored.length} yr already stored, skipped)` : ""}`).join(" · ")}</div>}
        {plan.data?.note && <div className="text-2xs text-faint">{plan.data.note}</div>}
      </div>
    </Panel>
  );
}

function ChainPicker({ rows }: { rows: any[] }) {
  const [sym, setSym] = useState(rows.find((r) => !r.synthetic)?.symbol || rows[0].symbol);
  const row = rows.find((r) => r.symbol === sym) || rows[0];
  return (
    <div>
      <div className="flex items-center gap-2 border-b border-line px-2.5 py-1.5">
        <Select sz="sm" className="w-28" value={row.symbol} onChange={setSym} options={rows.map((r) => r.symbol)} />
        <SyntheticBadge on={row.synthetic} />
        <span className="num text-2xs text-muted">{row.options.resolution} {row.options.quote_type} · {row.options.first} → {row.options.last} · via {row.options.provider}</span>
      </div>
      <ChainViewer key={row.symbol} sym={row.symbol} defaultTime={row.options.resolution === "EOD" ? "16:00" : "10:00"} />
    </div>
  );
}

function SymbolDetail({ sym }: { sym: string }) {
  const d = useQuery({ queryKey: ["symbol", sym], queryFn: () => api.get<any>(`/api/data/symbols/${sym}`) });
  const candles = useQuery({ queryKey: ["candles", sym, "1d"], queryFn: () => api.get<any>(`/api/data/candles/${sym}?tf=1d`) });
  const del = useMutation({ mutationFn: () => api.del(`/api/data/symbols/${sym}`), onSuccess: () => { refreshData(); toast.success(`${sym} removed`); } });
  const cal = useMemo(() => {
    const o = d.data?.options;
    if (!o || o.resolution === "EOD") return null;
    const years = Array.from(new Set(o.daily_snapshots.map((x: any) => x.date.slice(0, 4)))) as string[];
    const missing = new Set(o.missing_sessions);
    return {
      tooltip: { formatter: (p: any) => `${p.value[0]}: ${p.value[1] === 0 ? "MISSING" : `${p.value[1]} snapshots`}` },
      visualMap: { show: false, min: 0, max: o.median_snapshots, inRange: { color: ["#ef5350", "#e2b93b", "#1f6f66", "#26a69a"] } },
      calendar: years.map((y, i) => ({ range: y, top: 24 + i * 120, left: 40, right: 10, cellSize: ["auto", 12], splitLine: { lineStyle: { color: "#090b0f" } }, itemStyle: { color: "#141922", borderColor: "#090b0f" }, yearLabel: { color: C.muted, fontSize: 11 }, monthLabel: { color: C.muted, fontSize: 10 }, dayLabel: { color: C.faint, fontSize: 9 } })),
      series: years.map((y, i) => ({ type: "heatmap", coordinateSystem: "calendar", calendarIndex: i, data: [...o.daily_snapshots.filter((x: any) => x.date.startsWith(y)).map((x: any) => [x.date, x.n]), ...[...missing].filter((m: any) => m.startsWith(y)).map((m) => [m, 0])] })),
      height: years.length * 120 + 20,
    };
  }, [d.data]);
  if (d.isLoading) return <Loading />;
  if (d.error) return <ErrorBox className="m-2" error={d.error} />;
  const x = d.data;
  const o = x.options;
  return (
    <div className="grid grid-cols-[1.25fr_1fr] gap-px border-t border-line bg-line">
      <div className="flex flex-col gap-px bg-line">
        <Panel className="border-0" title={`${sym} daily bars`}>
          <LWChart height={200} series={[{ type: "candles", data: (candles.data?.bars || []).map((b: number[]) => ({ time: dayTime(new Date(b[0] * 1000).toISOString().slice(0, 10)), open: b[1], high: b[2], low: b[3], close: b[4] })) }]} />
        </Panel>
        {cal && <Panel className="border-0" title="Option snapshot coverage by session (red = missing, amber = partial)"><EChart option={cal as any} height={(cal as any).height} /></Panel>}
        <Panel className="border-0" title="Where this data came from">
          {(x.sources || []).length ? (
            <table className="tbl"><thead><tr><th>Dataset</th><th>Provider</th><th>Resolution</th><th>Type</th><th>Licence</th></tr></thead><tbody>
              {x.sources.map((s: any, i: number) => <tr key={i}><td className="text-fg-2">{s.kind}</td><td>{s.provider}</td><td className="num">{s.resolution}</td><td className="text-muted">{String(s.kind).startsWith("underlying") ? "price bars" : s.quote_type || "quotes"}</td><td className="whitespace-normal text-2xs text-muted">{s.license || "—"}</td></tr>)}
            </tbody></table>
          ) : <div className="p-2.5 text-2xs text-muted">Provider: {x.meta.provider || "unknown"}</div>}
        </Panel>
      </div>
      <Panel className="border-0" title={`${sym} quality report`} right={<Button size="sm" variant="danger" onClick={() => confirm(`Delete all stored data for ${sym}? This cannot be undone.`) && del.mutate()}><Trash2 /> Delete</Button>}>
        <table className="tbl"><tbody>
          <tr><td className="text-muted">Dataset version</td><td className="r font-mono">{x.dataset_version} ({x.files} files)</td></tr>
          <tr><td className="text-muted">Contract spec</td><td className="r text-fg-2">{x.meta.kind} · {x.meta.exercise} exercise · {x.meta.settlement} settlement · ×{x.meta.multiplier}</td></tr>
          <tr><td className="text-muted">Volatility index</td><td className="r">{x.meta.vol_index || "—"}</td></tr>
          {x.meta.extra?.price_adjustment && <tr><td className="text-muted">Price adjustment</td><td className="r text-fg-2">{x.meta.extra.price_adjustment}-adjusted by the provider; re-expressed in as-traded terms around splits</td></tr>}
          {x.daily && <>
            <tr><td className="text-muted">Daily sessions</td><td className="r num">{x.daily.sessions.toLocaleString()} ({x.daily.first} → {x.daily.last})</td></tr>
            <tr><td className="text-muted">Missing daily sessions</td><td className="r num text-2xs whitespace-normal">{x.daily.missing_sessions ? `${x.daily.missing_sessions}: ${x.daily.missing_sample.join(", ")}` : "none"}</td></tr>
          </>}
          {Object.entries(x.intraday || {}).map(([k, v]: any) => (
            <tr key={k}><td className="text-muted">{k} bars</td><td className="r num text-2xs whitespace-normal">{v.sessions} sessions, {compact(v.bars)} bars ({v.first} → {v.last}){v.missing_sessions ? ` · ${v.missing_sessions} missing` : ""}{v.incomplete_sessions ? ` · ${v.incomplete_sessions} incomplete` : ""}</td></tr>
          ))}
          {o && <>
            <tr><td className="text-muted">Option sessions</td><td className="r num">{o.sessions} / {o.expected_sessions} expected · {compact(o.rows)} rows</td></tr>
            <tr><td className="text-muted">Option resolution</td><td className="r">{o.resolution === "EOD" ? "end of day (one 16:00 ET snapshot per session)" : `${o.resolution} snapshots, median ${num(o.median_snapshots, 0)} per session`} · {o.quote_type}</td></tr>
            <tr><td className="text-muted">Missing option sessions</td><td className="r num text-2xs whitespace-normal">{o.missing_count ? `${o.missing_count}: ${o.missing_sessions.slice(-8).join(", ")}` : "none"}</td></tr>
            {o.resolution !== "EOD" && <tr><td className="text-muted">Sessions with snapshot gaps</td><td className="r num text-2xs whitespace-normal">{o.snapshot_gaps.map((g: any) => `${g.date} (${g.snapshots}/${g.expected})`).join(", ") || "none"}</td></tr>}
          </>}
          {x.iv_history && <tr><td className="text-muted">Derived IV history</td><td className="r num">{x.iv_history.first} → {x.iv_history.last} ({x.iv_history.rows})</td></tr>}
        </tbody></table>
        <div className="p-2"><Warnings items={[
          ...(x.meta.synthetic ? [{ level: "info", text: "SIMULATED sandbox symbol: generated by a stochastic-volatility model. It is not market data and every result on it is labelled." }] : []),
          ...(o?.resolution === "EOD" ? [{ level: "info", text: "End-of-day option quotes support entries and exits at the close only. Intraday entries, 0DTE trades and intraday option exits are refused rather than approximated." }] : []),
        ]} /></div>
      </Panel>
    </div>
  );
}

export function ChainViewer({ sym, defaultTime = "10:00" }: { sym: string; defaultTime?: string }) {
  const [day, setDay] = useState("");
  const [time, setTime] = useState(defaultTime);
  const [exp, setExp] = useState("");
  const q = useQuery({ queryKey: ["chain", sym, day, time], queryFn: () => api.get<any>(`/api/data/chain/${sym}?time=${time}${day ? `&day=${day}` : ""}`) });
  const d = q.data;
  const e = exp && d?.expirations.includes(exp) ? exp : d?.expirations?.[0];
  const rows = useMemo(() => {
    if (!d) return [];
    const m = new Map<number, any>();
    for (const r of d.rows.filter((x: any) => x.expiration === e)) {
      const o = m.get(r.strike) || { strike: r.strike };
      o[r.right] = r;
      m.set(r.strike, o);
    }
    return [...m.values()].sort((a, b) => a.strike - b.strike);
  }, [d, e]);
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 border-b border-line px-2.5 py-1.5">
        <Input sz="sm" type="date" className="w-36 num" value={day || d?.date || ""} onChange={(ev) => setDay(ev.target.value)} />
        <Input sz="sm" className="w-16 num" value={time} onChange={(ev) => setTime(ev.target.value)} />
        <Select sz="sm" className="w-40" value={e || ""} onChange={setExp} options={(d?.expirations || []).map((x: string) => x)} />
        {d && <span className="num text-2xs text-muted">snapshot {etTime(d.ts)} ET · spot {num(d.spot)} · {d.snapshots.length} snapshot(s) that session · observed quotes only</span>}
      </div>
      {q.isLoading ? <Loading /> : q.error ? <ErrorBox className="m-2" error={q.error} /> : (
        <table className="tbl">
          <thead><tr>{["IV", "Δ", "Θ", "Vol", "OI", "Size", "Bid", "Ask"].map((h) => <th key={`c${h}`} className="r">{h}</th>)}<th className="text-center">Strike</th>{["Bid", "Ask", "Size", "OI", "Vol", "Θ", "Δ", "IV"].map((h) => <th key={`p${h}`} className="r">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => {
            const c = r.C, p = r.P, itm = r.strike < d.spot;
            const cell = (x: any, k: string, f: (v: any) => string, shade: boolean) => <td className={cn("r num", shade && "bg-bg-2", x?.quote_age > 120 && "text-warn")}>{x ? f(x[k]) : "—"}</td>;
            return (
              <tr key={r.strike}>
                {cell(c, "iv", (v) => pct(v, 1), itm)}{cell(c, "delta", (v) => num(v, 2), itm)}{cell(c, "theta", (v) => num(v, 2), itm)}{cell(c, "volume", (v) => compact(v), itm)}{cell(c, "open_interest", (v) => compact(v), itm)}
                {cell(c, "bid_size", (v) => String(v), itm)}{cell(c, "bid", (v) => num(v), itm)}{cell(c, "ask", (v) => num(v), itm)}
                <td className="text-center num font-medium">{r.strike}</td>
                {cell(p, "bid", (v) => num(v), !itm)}{cell(p, "ask", (v) => num(v), !itm)}{cell(p, "ask_size", (v) => String(v), !itm)}{cell(p, "open_interest", (v) => compact(v), !itm)}{cell(p, "volume", (v) => compact(v), !itm)}
                {cell(p, "theta", (v) => num(v, 2), !itm)}{cell(p, "delta", (v) => num(v, 2), !itm)}{cell(p, "iv", (v) => pct(v, 1), !itm)}
              </tr>
            );
          })}</tbody>
        </table>
      )}
    </div>
  );
}

function Providers({ onJob }: { onJob: (id: string) => void }) {
  const [check, setCheck] = useState(false);
  const p = useQuery({ queryKey: ["providers", check], queryFn: () => api.get<any[]>(`/api/data/providers?check=${check}`) });
  const [f, setF] = useState({ provider: "thetadata", symbol: "SPY", start: "2024-01-02", end: "2024-01-31", interval: 15, include_options: true });
  const ingest = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/ingest", f), onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message) });
  return (
    <div className="grid grid-cols-[1.3fr_1fr] gap-px bg-line">
      <Panel className="border-0" title="Provider adapters" right={<Button size="sm" variant="ghost" onClick={() => setCheck(true)}><PlugZap /> Test connections</Button>}>
        <table className="tbl">
          <thead><tr><th>Provider</th><th>Status</th><th>Resolution</th><th className="r">Local cache</th><th>Setup</th></tr></thead>
          <tbody>{(p.data || []).map((x) => (
            <tr key={x.name}>
              <td className="text-fg">{x.label}</td>
              <td><span className={cn("text-xs", x.status.ok ? "text-up" : "text-muted")}>{x.status.ok ? "● " : "○ "}{x.status.message}</span></td>
              <td className="num text-fg-2">{x.resolution}</td><td className="r num">{bytes(x.cache_bytes)}</td>
              <td className="whitespace-normal text-2xs text-muted max-w-md">{x.docs}</td>
            </tr>
          ))}</tbody>
        </table>
        <div className="p-2.5 text-2xs text-faint leading-relaxed">
          Keys are read from environment variables (.env). Downloads are cached under <span className="font-mono">data/cache/&lt;provider&gt;</span> and sessions already in the
          columnar store are skipped, so re-running an ingest never re-downloads. Missing quotes are recorded as gaps — never filled with model prices.
        </div>
      </Panel>
      <Panel className="border-0" title="Ingest from a provider">
        <div className="grid grid-cols-2 gap-2 p-2.5">
          <Field label="Provider"><Select value={f.provider} onChange={(v) => setF({ ...f, provider: v })} options={(p.data || []).map((x) => ({ value: x.name, label: x.label }))} /></Field>
          <Field label="Symbol"><Input value={f.symbol} onChange={(e) => setF({ ...f, symbol: e.target.value.toUpperCase() })} /></Field>
          <Field label="From"><Input type="date" className="num" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" className="num" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Snapshot interval (min)"><NumInput value={f.interval} onChange={(v) => setF({ ...f, interval: v ?? 15 })} /></Field>
          <label className="flex items-center gap-2 self-end pb-1.5 text-xs text-fg-2"><Switch checked={f.include_options} onChange={(v) => setF({ ...f, include_options: v })} /> Option chains</label>
          <Button variant="primary" className="col-span-2" onClick={() => ingest.mutate()} disabled={ingest.isPending}><Download /> Start ingestion job</Button>
        </div>
      </Panel>
    </div>
  );
}

function ImportForm({ onJob }: { onJob: (id: string) => void }) {
  const file = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState("option_quotes");
  const [symbol, setSymbol] = useState("");
  const [tz, setTz] = useState("America/New_York");
  const [meta, setMeta] = useState({ settlement: "physical", exercise: "american", multiplier: 100 });
  const up = useMutation({
    mutationFn: () => {
      const f = file.current?.files?.[0];
      if (!f) throw new Error("choose a file");
      const fd = new FormData();
      fd.append("file", f);
      fd.append("kind", kind);
      fd.append("symbol", symbol);
      fd.append("tz", tz);
      fd.append("meta", JSON.stringify(meta));
      return api.form<{ id: string }>("/api/data/import", fd);
    },
    onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message),
  });
  const schemas: Record<string, string> = {
    option_quotes: "ts, expiration, strike, right (C/P), bid, ask [, bid_size, ask_size, volume, open_interest, iv, delta, gamma, theta, vega, underlying_price, quote_ts]",
    underlying_minutes: "ts (bar start), open, high, low, close [, volume]",
    underlying_daily: "date, open, high, low, close [, volume]",
    events: "date, event_type [, time, description]",
    corporate_actions: "ex_date, action (split|dividend) [, ratio, amount]",
  };
  return (
    <div className="grid grid-cols-2 gap-px bg-line">
      <Panel className="border-0" title="Upload a dataset">
        <div className="grid grid-cols-2 gap-2 p-2.5">
          <Field label="File (.csv / .parquet)" className="col-span-2"><input ref={file} type="file" accept=".csv,.parquet,.pq" className="text-xs file:mr-2 file:rounded-sm file:border file:border-line-2 file:bg-bg-3 file:px-2 file:py-1 file:text-fg" /></Field>
          <Field label="Kind"><Select value={kind} onChange={setKind} options={Object.keys(schemas)} /></Field>
          <Field label="Symbol"><Input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} placeholder="SPY" /></Field>
          <Field label="Timezone of naive timestamps"><Input value={tz} onChange={(e) => setTz(e.target.value)} /></Field>
          <Field label="Settlement"><Select value={meta.settlement} onChange={(v) => setMeta({ ...meta, settlement: v })} options={["physical", "cash"]} /></Field>
          <Field label="Exercise"><Select value={meta.exercise} onChange={(v) => setMeta({ ...meta, exercise: v })} options={["american", "european"]} /></Field>
          <Field label="Multiplier"><NumInput value={meta.multiplier} onChange={(v) => setMeta({ ...meta, multiplier: v ?? 100 })} /></Field>
          <Button variant="primary" className="col-span-2" onClick={() => up.mutate()} disabled={up.isPending}><Upload /> Import</Button>
        </div>
      </Panel>
      <Panel className="border-0" title="Expected columns">
        <div className="p-2.5 space-y-2 text-xs">
          <div className="font-mono text-2xs text-fg-2 rounded-sm bg-bg-2 border border-line p-2">{schemas[kind]}</div>
          <ul className="list-disc list-inside text-2xs text-muted space-y-1">
            <li>Common aliases are recognised (quote_datetime, option_type, strike_price, implied_volatility, underlying_last…).</li>
            <li>Rows with crossed quotes (ask &lt; bid), negative prices or unknown rights are rejected and counted — never repaired.</li>
            <li>Option files are split into one Parquet file per session; IV history and the dataset manifest are rebuilt afterwards.</li>
            <li>Missing IV/Greeks are computed from observed mid quotes at backtest time (flagged in diagnostics).</li>
          </ul>
        </div>
      </Panel>
    </div>
  );
}

function DemoForm({ onJob }: { onJob: (id: string) => void }) {
  const [f, setF] = useState({ start: "2023-01-03", end: "2024-12-31", seed: 20240101, interval: 15 });
  const go = useMutation({ mutationFn: () => api.post<{ id: string }>("/api/data/demo", f), onSuccess: (r) => onJob(r.id), onError: (e: any) => toast.error(e.message) });
  return (
    <div className="grid grid-cols-2 gap-px bg-line">
      <Panel className="border-0" title="Generate simulated symbols (developer sandbox)">
        <div className="grid grid-cols-2 gap-2 p-2.5">
          <Field label="From"><Input type="date" className="num" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" className="num" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Seed"><NumInput value={f.seed} onChange={(v) => setF({ ...f, seed: v ?? 1 })} /></Field>
          <Field label="Snapshot interval (min)"><NumInput value={f.interval} onChange={(v) => setF({ ...f, interval: v ?? 15 })} /></Field>
          <Button variant="primary" className="col-span-2" onClick={() => confirm("Generate the SIMULATED symbols DEMO, DEMOX and DEMOVIX? They are not market data and are only for testing the software.") && go.mutate()}><FlaskConical /> Generate</Button>
        </div>
      </Panel>
      <Panel className="border-0" title="What it contains">
        <ul className="list-disc list-inside p-2.5 text-2xs text-muted space-y-1">
          <li><b className="text-fg-2">DEMO</b>: ETF-like underlying, American exercise, physical settlement, daily/weekly/monthly expirations.</li>
          <li><b className="text-fg-2">DEMOX</b>: index-like underlying, European exercise, cash settlement (PM).</li>
          <li><b className="text-fg-2">DEMOVIX</b>: volatility index derived from the DEMO variance process.</li>
          <li>Stochastic variance with leverage effect, fat-tailed intraday returns, overnight gaps and jumps, scheduled synthetic CPI/NFP/FOMC events with event variance and IV crush.</li>
          <li>Quotes = Black-Scholes mids with skew, realistic spreads, sizes, volume and OI. ~0.4% stale quotes, ~0.15% missing quotes, three snapshot outages and one missing session are injected deliberately.</li>
          <li>Labelled SIMULATED everywhere, never used as a default, and never substituted for missing real data. Backtests only read them when the strategy names one of these symbols.</li>
        </ul>
      </Panel>
    </div>
  );
}
