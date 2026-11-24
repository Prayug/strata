import { useMutation, useQuery } from "@tanstack/react-query";
import { Database, Download, FlaskConical, PlugZap, RefreshCw, Trash2, Upload } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { LWChart, dayTime } from "@/components/charts/LWChart";
import { DataTable, ErrorBox, JobBar, Loading, PageHeader, Panel, SyntheticBadge, Warnings, type Col } from "@/components/common";
import { Button, Field, Input, NumInput, Select, Switch, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui";
import { api, queryClient, useJob, useMeta } from "@/lib/api";
import { bytes, cn, compact, etTime, num, pct } from "@/lib/format";

export default function DataPage() {
  const cat = useQuery({ queryKey: ["catalog"], queryFn: () => api.get<any[]>("/api/data/catalog") });
  const storage = useQuery({ queryKey: ["storage"], queryFn: () => api.get<any>("/api/data/storage") });
  const [sel, setSel] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, (s) => {
    if (s.status === "done") toast.success("Data job finished");
    if (s.status === "error") toast.error(`Data job failed: ${s.message}`);
    queryClient.invalidateQueries({ queryKey: ["catalog"] });
    queryClient.invalidateQueries({ queryKey: ["storage"] });
  });
  const rows = cat.data || [];
  const cur = sel || rows.find((r) => r.options)?.symbol || rows[0]?.symbol;
  const cols: Col<any>[] = [
    { key: "sym", label: "Symbol", render: (r) => <span className="flex items-center gap-1.5 font-medium text-fg">{r.symbol}<SyntheticBadge on={r.meta.synthetic} /></span>, sort: (r) => r.symbol },
    { key: "prov", label: "Provider", render: (r) => <span className="text-fg-2">{r.meta.provider}</span> },
    { key: "spec", label: "Contract spec", render: (r) => <span className="text-2xs text-muted">{r.meta.kind} · {r.meta.exercise} · {r.meta.settlement}{r.meta.settlement === "cash" ? ` ${r.meta.settlement_time}` : ""} · ×{r.meta.multiplier}</span> },
    { key: "u", label: "Underlying range", render: (r) => <span className="num text-fg-2">{r.underlying ? `${r.underlying.first} → ${r.underlying.last}` : "—"}</span> },
    { key: "ures", label: "Bars", align: "r", render: (r) => <span className="num">{r.underlying ? `${compact(r.underlying.bars)} @${r.underlying.resolution_seconds}s` : "—"}</span> },
    { key: "umiss", label: "Missing / partial sess.", align: "r", render: (r) => r.underlying ? <span className={cn("num", r.underlying.missing_sessions.length + r.underlying.incomplete_count ? "text-warn" : "text-muted")}>{r.underlying.missing_sessions.length} / {r.underlying.incomplete_count}</span> : "—" },
    { key: "o", label: "Options range", render: (r) => <span className="num text-fg-2">{r.options ? `${r.options.first} → ${r.options.last}` : "—"}</span> },
    { key: "orows", label: "Quote rows", align: "r", render: (r) => <span className="num">{r.options ? compact(r.options.rows) : "—"}</span>, sort: (r) => r.options?.rows },
    { key: "ores", label: "Resolution", align: "r", render: (r) => <span className="num">{r.options ? `${r.options.resolution_minutes}m · ${num(r.options.median_snapshots, 0)}/sess` : "—"}</span> },
    { key: "omiss", label: "Missing sess. / gaps", align: "r", render: (r) => r.options ? <span className={cn("num", r.options.missing_sessions.length + r.options.snapshot_gap_count ? "text-warn" : "text-muted")}>{r.options.missing_sessions.length} / {r.options.snapshot_gap_count}</span> : "—" },
    { key: "size", label: "Storage", align: "r", render: (r) => <span className="num">{bytes(r.storage_bytes)}</span>, sort: (r) => r.storage_bytes },
    { key: "ver", label: "Dataset version", render: (r) => <span className="font-mono text-2xs text-faint">{r.dataset_version}</span> },
  ];
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Data Management" sub={<span className="num text-2xs text-muted">{storage.data?.data_dir} · market {bytes(Object.values(storage.data?.market || {}).reduce((a: number, b: any) => a + b, 0) as number)} · results {bytes(storage.data?.results)} · cache {bytes(storage.data?.cache)}</span>}>
        <JobBar job={job} />
        <Button variant="ghost" onClick={() => { api.get("/api/data/catalog?refresh=true").then(() => queryClient.invalidateQueries({ queryKey: ["catalog"] })); }}><RefreshCw /> Rescan</Button>
      </PageHeader>
      <div className="h-[34%] min-h-40 border-b border-line">
        {cat.isLoading ? <Loading label="Scanning datasets" /> : cat.error ? <ErrorBox className="m-2" error={cat.error} /> : (
          <DataTable rows={rows} cols={cols} rowKey={(r) => r.symbol} onRowClick={(r) => setSel(r.symbol)} selected={cur} />
        )}
      </div>
      <Tabs defaultValue="detail" className="flex min-h-0 flex-1 flex-col">
        <TabsList>
          <TabsTrigger value="detail"><Database className="size-3" /> {cur} coverage</TabsTrigger>
          <TabsTrigger value="chain">Chain viewer</TabsTrigger>
          <TabsTrigger value="providers"><PlugZap className="size-3" /> Providers & ingestion</TabsTrigger>
          <TabsTrigger value="import"><Upload className="size-3" /> Import CSV / Parquet</TabsTrigger>
          <TabsTrigger value="demo"><FlaskConical className="size-3" /> Synthetic demo data</TabsTrigger>
        </TabsList>
        <div className="min-h-0 flex-1 overflow-auto">
          <TabsContent value="detail">{cur && <SymbolDetail sym={cur} />}</TabsContent>
          <TabsContent value="chain">{cur && <ChainViewer sym={cur} />}</TabsContent>
          <TabsContent value="providers"><Providers onJob={setJobId} /></TabsContent>
          <TabsContent value="import"><ImportForm onJob={setJobId} /></TabsContent>
          <TabsContent value="demo"><DemoForm onJob={setJobId} /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

function SymbolDetail({ sym }: { sym: string }) {
  const d = useQuery({ queryKey: ["symbol", sym], queryFn: () => api.get<any>(`/api/data/symbols/${sym}`) });
  const candles = useQuery({ queryKey: ["candles", sym, "1d"], queryFn: () => api.get<any>(`/api/data/candles/${sym}?tf=1d`) });
  const del = useMutation({ mutationFn: () => api.del(`/api/data/symbols/${sym}`), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["catalog"] }); toast.success(`${sym} removed`); } });
  const cal = useMemo(() => {
    const o = d.data?.options;
    if (!o) return null;
    const years = Array.from(new Set(o.daily_snapshots.map((x: any) => x.date.slice(0, 4)))) as string[];
    const missing = new Set(o.missing_sessions);
    const med = o.median_snapshots;
    return {
      tooltip: { formatter: (p: any) => `${p.value[0]}: ${p.value[1] === 0 ? "MISSING" : `${p.value[1]} snapshots`}` },
      visualMap: { show: false, min: 0, max: med, inRange: { color: ["#ef5350", "#e2b93b", "#1f6f66", "#26a69a"] } },
      calendar: years.map((y, i) => ({ range: y, top: 24 + i * 120, left: 40, right: 10, cellSize: ["auto", 12], splitLine: { lineStyle: { color: "#090b0f" } }, itemStyle: { color: "#141922", borderColor: "#090b0f" }, yearLabel: { color: C.muted, fontSize: 11 }, monthLabel: { color: C.muted, fontSize: 10 }, dayLabel: { color: C.faint, fontSize: 9, firstDay: 1 } })),
      series: years.map((y, i) => ({ type: "heatmap", coordinateSystem: "calendar", calendarIndex: i, data: [...o.daily_snapshots.filter((x: any) => x.date.startsWith(y)).map((x: any) => [x.date, x.n]), ...[...missing].filter((m: any) => m.startsWith(y)).map((m) => [m, 0])] })),
      height: years.length * 120 + 20,
    };
  }, [d.data]);
  if (d.isLoading) return <Loading />;
  if (d.error) return <ErrorBox className="m-2" error={d.error} />;
  const x = d.data;
  return (
    <div className="grid grid-cols-[1.4fr_1fr] gap-px bg-line">
      <Panel className="border-0" title="Option snapshot coverage by session (red = missing, amber = partial)">
        {cal ? <EChart option={cal as any} height={(cal as any).height} /> : <div className="p-3 text-xs text-faint">No option data for {sym}.</div>}
      </Panel>
      <div className="flex flex-col gap-px bg-line">
        <Panel className="border-0" title={`${sym} daily bars`}>
          <LWChart height={180} series={[{ type: "candles", data: (candles.data?.bars || []).map((b: number[]) => ({ time: dayTime(new Date(b[0] * 1000).toISOString().slice(0, 10)), open: b[1], high: b[2], low: b[3], close: b[4] })) }]} />
        </Panel>
        <Panel className="border-0" title="Quality report" right={<Button size="sm" variant="danger" onClick={() => confirm(`Delete all stored data for ${sym}? This cannot be undone.`) && del.mutate()}><Trash2 /> Delete</Button>}>
          <table className="tbl"><tbody>
            <tr><td className="text-muted">Dataset version</td><td className="r font-mono">{x.dataset_version} ({x.files} files)</td></tr>
            <tr><td className="text-muted">Description</td><td className="r whitespace-normal text-fg-2">{x.meta.description || "—"}</td></tr>
            <tr><td className="text-muted">Vol index</td><td className="r">{x.meta.vol_index || "—"}</td></tr>
            {x.underlying && <>
              <tr><td className="text-muted">Underlying sessions</td><td className="r num">{x.underlying.sessions} / {x.underlying.expected_sessions} expected</td></tr>
              <tr><td className="text-muted">Missing underlying sessions</td><td className="r num text-2xs whitespace-normal">{x.underlying.missing_sessions.join(", ") || "none"}</td></tr>
              <tr><td className="text-muted">Incomplete sessions (bars &lt; expected)</td><td className="r num">{x.underlying.incomplete_count}</td></tr>
            </>}
            {x.options && <>
              <tr><td className="text-muted">Option sessions</td><td className="r num">{x.options.sessions} / {x.options.expected_sessions} expected</td></tr>
              <tr><td className="text-muted">Missing option sessions</td><td className="r num text-2xs whitespace-normal">{x.options.missing_sessions.join(", ") || "none"}</td></tr>
              <tr><td className="text-muted">Sessions with snapshot gaps</td><td className="r num text-2xs whitespace-normal">{x.options.snapshot_gaps.map((g: any) => `${g.date} (${g.snapshots}/${g.expected})`).join(", ") || "none"}</td></tr>
            </>}
            {x.iv_history && <tr><td className="text-muted">Derived IV history</td><td className="r num">{x.iv_history.first} → {x.iv_history.last} ({x.iv_history.rows})</td></tr>}
          </tbody></table>
          <div className="p-2"><Warnings items={x.meta.synthetic ? [{ level: "info", text: "SYNTHETIC dataset: generated by a stochastic-volatility model for demonstration. Every backtest on it is labelled synthetic." }] : []} /></div>
        </Panel>
      </div>
    </div>
  );
}

function ChainViewer({ sym }: { sym: string }) {
  const [day, setDay] = useState("");
  const [time, setTime] = useState("10:00");
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
        {d && <span className="num text-2xs text-muted">snapshot {etTime(d.ts)} ET · spot {num(d.spot)} · {d.snapshots.length} snapshots that session · observed quotes only</span>}
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
      <Panel className="border-0" title="Regenerate the synthetic demonstration dataset">
        <div className="grid grid-cols-2 gap-2 p-2.5">
          <Field label="From"><Input type="date" className="num" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" className="num" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Seed"><NumInput value={f.seed} onChange={(v) => setF({ ...f, seed: v ?? 1 })} /></Field>
          <Field label="Snapshot interval (min)"><NumInput value={f.interval} onChange={(v) => setF({ ...f, interval: v ?? 15 })} /></Field>
          <Button variant="primary" className="col-span-2" onClick={() => confirm("Replace DEMO, DEMOX and DEMOVIX with a newly generated synthetic dataset? Existing results keep their old dataset version.") && go.mutate()}><FlaskConical /> Generate</Button>
        </div>
      </Panel>
      <Panel className="border-0" title="What it contains">
        <ul className="list-disc list-inside p-2.5 text-2xs text-muted space-y-1">
          <li><b className="text-fg-2">DEMO</b>: ETF-like underlying, American exercise, physical settlement, daily/weekly/monthly expirations.</li>
          <li><b className="text-fg-2">DEMOX</b>: index-like underlying, European exercise, cash settlement (PM).</li>
          <li><b className="text-fg-2">DEMOVIX</b>: volatility index derived from the DEMO variance process.</li>
          <li>Stochastic variance with leverage effect, fat-tailed intraday returns, overnight gaps and jumps, scheduled synthetic CPI/NFP/FOMC events with event variance and IV crush.</li>
          <li>Quotes = Black-Scholes mids with skew, realistic spreads, sizes, volume and OI. ~0.4% stale quotes, ~0.15% missing quotes, three snapshot outages and one missing session are injected deliberately.</li>
          <li>Labelled SYNTHETIC everywhere; never presented as market history.</li>
        </ul>
      </Panel>
    </div>
  );
}
