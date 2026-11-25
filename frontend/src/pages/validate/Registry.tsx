/* Experiment registry: everything that has been evaluated on which dates, across every notebook and tool.
   Its job is to say how much a result on a given period can still be trusted, so statuses always carry a glyph
   and a word, and a period that related work has already examined is never called independent. */
import { useQuery } from "@tanstack/react-query";
import { Scale, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { DataTable, Empty, Loading, PageHeader, Panel, Stat, type Col } from "@/components/common";
import { Dialog, Input, Select, Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { pv } from "../lab/Research";

const PHASE: Record<string, { glyph: string; help: string }> = {
  exploration: { glyph: "○", help: "Looking at the data to form an idea. Free to repeat, but every look is counted." },
  development: { glyph: "◐", help: "Tuning a specific strategy or rule." },
  validation: { glyph: "◆", help: "Scored on a final test period that was held back." },
  confirmation: { glyph: "◈", help: "A repeat of a validated configuration." },
  forward: { glyph: "▶", help: "Evaluated only on sessions after the configuration was frozen." },
};
const HOLD: Record<string, { glyph: string; tone: string }> = {
  untouched: { glyph: "●", tone: "text-up" }, used: { glyph: "◐", tone: "text-warn" }, "used once": { glyph: "◐", tone: "text-warn" }, exposed: { glyph: "✕", tone: "text-down" }, contaminated: { glyph: "✕", tone: "text-down" },
};

function HoldoutDialog({ h, onClose }: { h: any; onClose: () => void }) {
  const q = useQuery({ queryKey: ["registry-check", h.scope, h.symbol], queryFn: () => api.post<any>("/api/registry/check", { symbol: h.symbol, scope: h.scope }) });
  const mine = (q.data?.holdouts || []).find((x: any) => x.id === h.id) || q.data?.holdouts?.[0];
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={`Final test period · ${h.symbol} ${h.start} to ${h.end}`} width={720}>
      {q.isLoading ? <Loading /> : (
        <div className="flex flex-col gap-3 text-xs">
          {(q.data?.sentences || []).map((s: string, i: number) => <p key={i} className="text-sm leading-relaxed text-fg">{s}</p>)}
          {mine?.elsewhere?.by_notebook?.length > 0 && (
            <div>
              <div className="label mb-1">Who else has evaluated these dates</div>
              <table className="tbl"><thead><tr><th>Notebook or tool</th><th className="r">Configurations</th><th className="r">Related to this one</th><th>Kinds</th><th className="r">Last</th></tr></thead>
                <tbody>{mine.elsewhere.by_notebook.map((b: any) => (
                  <tr key={b.scope}><td className="max-w-[300px] truncate" title={b.title}>{b.title}</td><td className="r num">{b.configurations}</td><td className="r num">{b.related}</td>
                    <td className="text-muted">{Object.entries(b.kinds || {}).map(([k, v]) => `${v} ${k}`).join(", ")}</td><td className="r text-muted">{ago(b.last)}</td></tr>
                ))}</tbody></table>
            </div>
          )}
          <p className="text-muted">A final test period only confirms anything the first time it is used, and only if nothing related was tuned on those dates. Strata records every evaluation so this can be answered from the record instead of from memory.</p>
        </div>
      )}
    </Dialog>
  );
}

export default function Registry() {
  const nav = useNavigate();
  const ov = useQuery({ queryKey: ["registry"], queryFn: () => api.get<any>("/api/registry"), staleTime: 5_000 });
  const [symbol, setSymbol] = useState("");
  const [phase, setPhase] = useState("");
  const [q, setQ] = useState("");
  const [hold, setHold] = useState<any>(null);
  const audit = useQuery({
    queryKey: ["registry-audit", symbol, phase, q],
    queryFn: () => api.get<{ total: number; rows: any[] }>(`/api/registry/audit?limit=1000${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}${phase ? `&phase=${phase}` : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`),
    staleTime: 5_000,
  });
  const d = ov.data;
  const open = (scope: string) => { if (scope && !scope.includes(":")) nav(`/lab/${scope}`); };
  const cols: Col<any>[] = useMemo(() => [
    { key: "created_at", label: "When", render: (r) => <span className="text-muted">{ago(r.created_at)}</span>, sort: (r) => r.created_at, width: 76 },
    { key: "phase", label: "Phase", render: (r) => <Tip content={PHASE[r.phase]?.help}><span className="whitespace-nowrap"><span aria-hidden className="mr-1">{PHASE[r.phase]?.glyph || "·"}</span>{r.phase}</span></Tip>, sort: (r) => r.phase, width: 104 },
    { key: "symbol", label: "Symbol", render: (r) => r.symbol, sort: (r) => r.symbol, width: 96 },
    { key: "kind", label: "Kind", render: (r) => <span className="text-muted">{r.kind}</span>, sort: (r) => r.kind, width: 84 },
    { key: "label", label: "Configuration", render: (r) => <span className="block max-w-[520px] truncate" title={r.label}>{r.label}</span> },
    { key: "period", label: "Dates evaluated", render: (r) => <span className="num text-muted">{r.start} → {r.end}</span>, sort: (r) => r.start, width: 176 },
    { key: "n", label: "Obs.", align: "r", render: (r) => <span className="num">{r.n ?? "—"}</span>, sort: (r) => r.n, width: 60 },
    { key: "p", label: "p", align: "r", render: (r) => <span className="num">{pv(r.p_value)}</span>, sort: (r) => r.p_value, width: 60 },
    { key: "title", label: "Where", render: (r) => <span className="block max-w-[220px] truncate text-muted" title={r.title}>{r.title}</span>, sort: (r) => r.title },
    { key: "key", label: "Fingerprint", render: (r) => <span className="font-mono text-2xs text-faint">{String(r.key).slice(0, 10)}</span>, width: 90 },
  ], []);

  if (ov.isLoading) return <Loading />;
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="registry">
      <PageHeader title={<span className="flex items-center gap-2"><Scale className="size-3.5 text-up" />Experiment registry</span>} sub={<span className="text-xs text-muted">Every configuration evaluated, on which dates, in which notebook</span>} />
      {!d?.configurations ? <Empty title="Nothing has been evaluated yet" icon={Scale}>Every study, backtest, sweep and pattern exploration is recorded here as it runs, so the registry can say which dates have already been looked at.</Empty> : (
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-2">
          <div className="grid shrink-0 grid-cols-2 gap-2 lg:grid-cols-5">
            <div className="panel"><Stat label="Configurations" value={d.configurations.toLocaleString()} sub={`${d.evaluations.toLocaleString()} evaluations`} hint="Distinct configurations by fingerprint. Running the same one twice is one configuration and two evaluations." /></div>
            <div className="panel"><Stat label="Notebooks and tools" value={d.notebooks} /></div>
            <div className="panel"><Stat label="Evaluated more than once" value={d.repeated.length} hint="The same fingerprint evaluated again, here or in another notebook." /></div>
            <div className="panel"><Stat label="Final test periods" value={d.holdouts.length} sub={`${d.holdouts.filter((h: any) => h.usable).length} still usable`} /></div>
            <div className="panel"><Stat label="Not independent" value={d.not_independent} tone={d.not_independent ? "text-warn" : undefined} sub="examined elsewhere too" hint="Final test periods whose dates were also evaluated by other notebooks or tools. A result on them is not an independent confirmation." /></div>
          </div>

          <div className="grid shrink-0 gap-2 lg:grid-cols-[1.5fr_1fr]">
            <Panel title="Final test periods (holdouts)" bodyClass="overflow-auto max-h-[230px]">
              {d.holdouts.length === 0 ? <div className="p-3 text-xs text-muted">No notebook has reserved a final test period yet.</div> : (
                <table className="tbl" data-testid="holdouts">
                  <thead><tr><th>Symbol</th><th>Period held back</th><th>Status</th><th className="r">Used here</th><th className="r">Evaluated elsewhere</th><th>Independent</th><th>Notebook</th></tr></thead>
                  <tbody>
                    {d.holdouts.map((h: any) => (
                      <tr key={h.id} className="cursor-pointer" onClick={() => setHold(h)}>
                        <td>{h.symbol}</td><td className="num">{h.start} → {h.end}</td>
                        <td className={cn("whitespace-nowrap", HOLD[h.label]?.tone)}><span aria-hidden className="mr-1">{HOLD[h.label]?.glyph || "·"}</span>{h.label}</td>
                        <td className="r num">{(h.uses || []).length}</td>
                        <td className="r num">{h.elsewhere ? `${h.elsewhere} in ${h.elsewhere_notebooks}` : "0"}</td>
                        <td className={h.independent ? "text-up" : "text-warn"}>{h.independent ? "● yes" : "◈ no"}</td>
                        <td className="max-w-[240px] truncate text-muted" title={h.title}>{h.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>
            <Panel title="Where the testing has gone" bodyClass="overflow-auto max-h-[230px]">
              <table className="tbl">
                <thead><tr><th>Phase</th><th className="r">Configurations</th></tr></thead>
                <tbody>{d.by_phase.map((p: any) => <tr key={p.phase}><td><span aria-hidden className="mr-1">{PHASE[p.phase]?.glyph}</span>{p.phase}<span className="ml-2 text-muted">{PHASE[p.phase]?.help}</span></td><td className="r num">{p.configurations}</td></tr>)}</tbody>
                <thead><tr><th>Most tested inputs</th><th className="r">Configurations</th></tr></thead>
                <tbody>{d.features.slice(0, 8).map((f: any) => <tr key={f.feature}><td className="font-mono">{f.feature}</td><td className="r num">{f.configurations}</td></tr>)}</tbody>
              </table>
            </Panel>
          </div>

          {d.repeated.length > 0 && (
            <Panel title="Configurations evaluated more than once" bodyClass="overflow-auto max-h-[150px]" className="shrink-0">
              <table className="tbl"><thead><tr><th>Configuration</th><th>Symbol</th><th className="r">Notebooks</th><th className="r">Evaluations</th><th className="r">Last</th></tr></thead>
                <tbody>{d.repeated.map((r: any) => <tr key={r.key} className="cursor-pointer" onClick={() => setQ(r.key)}><td className="max-w-[640px] truncate" title={r.label}>{r.label}</td><td>{r.symbol}</td><td className="r num">{r.notebooks}</td><td className="r num">{r.evaluations}</td><td className="r text-muted">{ago(r.last)}</td></tr>)}</tbody></table>
            </Panel>
          )}

          <Panel className="min-h-[320px] flex-1" title={`Audit trail · ${audit.data?.total ?? "…"} evaluations`} bodyClass="flex flex-col"
            right={<>
              <div className="relative"><Search className="pointer-events-none absolute left-1.5 top-1/2 size-3 -translate-y-1/2 text-faint" /><Input sz="sm" className="w-56 !pl-6" placeholder="Search configuration or fingerprint" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search the audit trail" /></div>
              <Select sz="sm" className="w-28" value={symbol} onChange={setSymbol} options={[{ value: "", label: "All symbols" }, ...d.by_symbol.map((s: any) => ({ value: s.symbol, label: s.symbol }))]} />
              <Select sz="sm" className="w-32" value={phase} onChange={setPhase} options={[{ value: "", label: "All phases" }, ...Object.keys(PHASE).map((p) => ({ value: p, label: p }))]} />
            </>}>
            {audit.isLoading ? <Loading /> : <DataTable rows={audit.data?.rows || []} cols={cols} rowKey={(r) => r.id} onRowClick={(r) => open(r.scope)} defaultSort={{ key: "created_at", dir: -1 }} />}
          </Panel>
        </div>
      )}
      {hold && <HoldoutDialog h={hold} onClose={() => setHold(null)} />}
    </div>
  );
}
