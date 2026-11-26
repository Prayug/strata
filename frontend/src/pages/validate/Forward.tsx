/* Forward validation: a configuration is frozen with a hash and a date, then judged only on sessions that did
   not exist when it was frozen. Paper only: there is no order entry anywhere in Strata. A frozen experiment
   cannot be edited; a change is a new version with its own activation date. */
import { useQuery } from "@tanstack/react-query";
import { Lock, Play, Plus, RefreshCw, ShieldAlert, ShieldCheck, Snowflake, Square } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Empty, ErrorBox, Loading, PageHeader, Panel, Stat, Warnings } from "@/components/common";
import { Button, Dialog, Select, Tip } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { ago, cn, money } from "@/lib/format";
import { pp } from "@/lib/lab";

const STATUS: Record<string, { glyph: string; word: string; tone: string }> = {
  waiting: { glyph: "…", word: "waiting for the first forward session", tone: "text-muted" },
  evaluated: { glyph: "●", word: "evaluated", tone: "text-up" },
  no_data: { glyph: "○", word: "forward data missing", tone: "text-warn" },
  never: { glyph: "·", word: "not evaluated yet", tone: "text-muted" },
};
const st = (ex: any) => STATUS[ex?.latest?.status || "never"] || STATUS.never;
const n0 = (x: any) => (x === null || x === undefined ? "—" : typeof x === "number" ? x.toLocaleString() : String(x));

function Compare({ rows }: { rows: [string, React.ReactNode, React.ReactNode][] }) {
  return (
    <table className="tbl">
      <thead><tr><th /><th className="r">Since freezing</th><th className="r">Historical benchmark</th></tr></thead>
      <tbody>{rows.map(([k, a, b]) => <tr key={k}><td className="text-muted">{k}</td><td className="r num">{a}</td><td className="r num text-fg-2">{b}</td></tr>)}</tbody>
    </table>
  );
}

/** The paper account as it is stored: one state per sealed session. Each evaluation starts from the last one and runs only the
    sessions after it; a state is never rewritten. Reconcile compares sealed sessions with the data as stored now. */
function Account({ id, strategy }: { id: string; strategy: boolean }) {
  const q = useQuery({ queryKey: ["forward", id, "account"], queryFn: () => api.get<any>(`/api/forward/${id}/account`), enabled: strategy });
  const [rec, setRec] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  if (!strategy) return null;
  const rows: any[] = q.data?.states || [];
  const last = rows[rows.length - 1];
  const reconcile = async (force: boolean) => {
    setBusy(true);
    try { setRec(await api.post<any>(`/api/forward/${id}/reconcile`, { force })); queryClient.invalidateQueries({ queryKey: ["forward", id] }); } catch (e: any) { toast.error(e.message); } finally { setBusy(false); }
  };
  return (
    <Panel title={<span>Paper account, as stored <span className="ml-2 normal-case text-muted">{rows.length} session state{rows.length === 1 ? "" : "s"} · each evaluation continues from the last</span></span>}
      className="shrink-0" right={<Button variant="outline" disabled={busy || !rows.length} onClick={() => reconcile(false)} data-testid="forward-reconcile"><RefreshCw className={cn(busy && "animate-spin")} />Reconcile with the stored data</Button>}>
      <div className="p-2 text-xs" data-testid="forward-account">
        {!rows.length ? <div className="px-1 py-2 text-muted">No session has been sealed yet, so there is no stored account. The first evaluation with a closed forward session creates it.</div> : (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 px-1 pb-2 md:grid-cols-4">
              <div><span className="text-muted">Through </span><span className="num">{last.session}</span></div>
              <div><span className="text-muted">Cash </span><span className="num">{money(last.cash, 2)}</span></div>
              <div><span className="text-muted">Realised </span><span className={cn("num", last.realised > 0 ? "text-up" : last.realised < 0 ? "text-down" : "")}>{money(last.realised, 2, true)}</span></div>
              <div><span className="text-muted">Open positions </span><span className="num">{last.open_positions.length}</span>{last.open_positions.some((p: any) => p.pending_exit) && <span className="text-warn"> · an exit is pending</span>}</div>
            </div>
            {last.open_positions.length > 0 && <div className="px-1 pb-2 text-fg-2">Carried into the next session: {last.open_positions.map((p: any) => p.legs.map((l: any) => `${l.qty > 0 ? "+" : ""}${l.qty} ${l.strike}${l.right} @ ${l.entry_price}`).join(" / ")).join("; ")}</div>}
            <div className="max-h-44 overflow-auto border border-line">
              <table className="tbl"><thead><tr><th>Session</th><th>Stored</th><th className="r">Cash</th><th className="r">Open</th><th className="r">Closed</th><th>Engine</th><th>State hash</th><th>Data fingerprint</th></tr></thead>
                <tbody>{[...rows].reverse().map((r) => <tr key={r.session}><td className="num">{r.session}</td><td className="text-muted">{r.origin === "sealed" ? "sealed at evaluation" : "rebuilt from the ledger"}</td><td className="r num">{money(r.cash, 2)}</td><td className="r num">{r.open_positions.length}</td>
                  <td className="r num">{r.closed_positions ?? "—"}</td><td className="num text-muted">{r.engine_version}</td><td className="num text-muted">{String(r.state_hash).slice(0, 12)}</td><td className="num text-muted">{String(r.data_fingerprint || "").slice(0, 12)}</td></tr>)}</tbody></table>
            </div>
          </>
        )}
        {rec && (
          <div className={cn("mt-2 border px-2.5 py-2", rec.revisions_new ? "border-warn/50 bg-warn/[0.06]" : "border-line")} data-testid="forward-reconcile-result">
            {rec.note ? rec.note : <>Checked {rec.checked} sealed session{rec.checked === 1 ? "" : "s"} against the data stored now: {rec.changed.length === 0 ? "none of them has changed." : `${rec.changed.length} read different data (${rec.changed.slice(0, 6).join(", ")}).`}
              {rec.changed.length > 0 && (rec.revisions_new ? ` ${rec.revisions_new} revision${rec.revisions_new === 1 ? " was" : "s were"} added to the ledger. The sealed entries and every later decision are unchanged.` : " Re-run on the new data they give the same entries: nothing to record.")}</>}
            {rec.changed.length === 0 && !rec.note && <button className="ml-2 text-info hover:underline" onClick={() => reconcile(true)}>Replay every sealed session anyway</button>}
          </div>
        )}
      </div>
    </Panel>
  );
}

function VersionDialog({ ex, onClose, onCreated }: { ex: any; onClose: () => void; onCreated: (id: string) => void }) {
  const [text, setText] = useState(() => JSON.stringify(ex.config, null, 2));
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    let cfg: any;
    try { cfg = JSON.parse(text); } catch (e: any) { setErr(`Not valid JSON: ${e.message}`); return; }
    setBusy(true);
    try {
      const n = await api.post<any>(`/api/forward/${ex.id}/version`, { config: cfg });
      toast.success(`Version ${n.version} frozen; version ${ex.version} keeps running unchanged`);
      onCreated(n.id);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={`New version of "${ex.name}"`} width={760}
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" onClick={go} disabled={busy}><Snowflake />Freeze as version {ex.version + 1}</Button></>}>
      <div className="flex flex-col gap-2 text-xs">
        <p className="text-fg-2">Version {ex.version} cannot be changed. Edit the configuration below and it is frozen as a separate experiment that starts collecting forward evidence from today's last stored session. The two are then compared on their own records, never merged.</p>
        <textarea className="h-[46vh] w-full resize-none rounded-sm border border-line-2 bg-bg-0 p-2 font-mono text-[11.5px] leading-[1.45] outline-none focus:border-accent/60" spellCheck={false} value={text} onChange={(e) => { setText(e.target.value); setErr(null); }} aria-label="Configuration for the new version" />
        {err && <ErrorBox error={err} />}
      </div>
    </Dialog>
  );
}

function Detail({ id }: { id: string }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["forward", id], queryFn: () => api.get<any>(`/api/forward/${id}`) });
  const [busy, setBusy] = useState(false);
  const [showCfg, setShowCfg] = useState(false);
  const [versioning, setVersioning] = useState(false);
  if (q.isLoading) return <Loading />;
  if (q.isError || !q.data) return <div className="p-3"><ErrorBox error={q.error} /></div>;
  const ex = q.data;
  const rep = ex.latest;
  const b = ex.benchmark || {};
  const c = rep?.comparison || {};
  const refresh = () => { queryClient.invalidateQueries({ queryKey: ["forward"] }); };
  const act = async (path: string, ok: string) => {
    setBusy(true);
    try { await api.post(`/api/forward/${id}/${path}`); toast.success(ok); refresh(); } catch (e: any) { toast.error(e.message); } finally { setBusy(false); }
  };
  const s = st(ex);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-2" data-testid="forward-detail">
      <div className="panel flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2"><Snowflake className="size-4 shrink-0 text-info" /><h2 className="truncate text-sm font-semibold">{ex.name}</h2><span className="rounded-[2px] border border-line-2 px-1.5 text-2xs text-muted">v{ex.version}</span>
            <span className={cn("text-xs", ex.status === "active" ? "text-up" : "text-muted")}>{ex.status === "active" ? "▶ active" : "■ stopped"}</span></div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted">
            <span>{ex.kind === "strategy" ? "Option strategy (paper)" : "Market study"} · {ex.symbol}</span>
            <Tip content={<span className="block max-w-xs">{ex.activation?.rule || "The moment the configuration was frozen, from the clock."} Sessions stored before or after make no difference: a stale dataset or a later backfill cannot move this.</span>}><span data-testid="forward-activated">activated <b className="num text-fg-2">{String(ex.activated_at || "").replace("T", " ").slice(0, 16)} UTC</b></span></Tip>
            <Tip content="The first exchange session whose every decision was still in the future at activation. Nothing earlier is ever counted, whatever data arrives later."><span data-testid="forward-first">first session that counts <b className="num text-fg-2">{ex.first_session || "—"}</b></span></Tip>
            <Tip content="A hash of the full configuration, recorded at freezing. Every evaluation checks it; an experiment whose stored configuration no longer matches is refused."><span className="flex items-center gap-1"><Lock className="size-3" /><span className="font-mono">{ex.config_hash}</span></span></Tip>
            <span>{ex.evaluations} evaluation{ex.evaluations !== 1 ? "s" : ""}</span>
            <Tip content={<span className="block max-w-xs">Every decision, fill, exit and missing-data event is appended to a ledger in which each entry carries the hash of the one before it. The database refuses to update or delete an entry; this check recomputes the chain. {ex.ledger?.detail}</span>}>
              <span className={cn("flex items-center gap-1", ex.ledger?.ok === false ? "text-down" : "")} data-testid="forward-ledger" data-ok={ex.ledger?.ok === false ? "no" : "yes"}>{ex.ledger?.ok === false ? <ShieldAlert className="size-3" /> : <ShieldCheck className="size-3 text-up" />}ledger: {ex.ledger_entries ?? 0} entr{(ex.ledger_entries ?? 0) === 1 ? "y" : "ies"}, {ex.sealed_sessions ?? 0} sealed session{(ex.sealed_sessions ?? 0) === 1 ? "" : "s"}{ex.ledger?.ok === false ? `, BROKEN at entry ${ex.ledger.broken_at}` : ", chain verified"}</span>
            </Tip>
            <span className="num" title="The engine version recorded when the configuration was frozen">engine {ex.engine_version || "—"}</span>
          </div>
        </div>
        <Button onClick={() => setShowCfg(true)} variant="ghost">Frozen configuration</Button>
        <Button onClick={() => setVersioning(true)} variant="outline"><Plus />New version</Button>
        {ex.status === "active" && <Button onClick={() => act("stop", "Experiment stopped. Its record is kept.")} disabled={busy} variant="outline"><Square />Stop</Button>}
        <Button onClick={() => act("evaluate", "Evaluated on the forward sessions stored so far")} disabled={busy} variant="primary" data-testid="forward-evaluate"><RefreshCw className={cn(busy && "animate-spin")} />Evaluate now</Button>
      </div>

      {(ex.newer_versions || []).length > 0 && <Warnings items={[{ level: "info", text: `A newer version exists: ${ex.newer_versions.map((k: any) => `v${k.version} (first session ${k.first_session || k.activation_date})`).join(", ")}. This one is unchanged and keeps its own record.` }]} />}
      {ex.ledger?.ok === false && <Warnings items={[{ level: "error", text: `The ledger does not verify (${ex.ledger.detail}). Nothing further is appended to it and its figures should not be relied on.` }]} />}
      {ex.activation?.migrated && <Warnings items={[{ level: "info", text: ex.activation.migrated }]} />}
      {(ex.pending_sessions || []).length > 0 && <Warnings items={[{ level: "info", text: `${ex.pending_sessions.length} session${ex.pending_sessions.length === 1 ? " has" : "s have"} closed since the last evaluation and ${ex.pending_sessions.length === 1 ? "is" : "are"} not sealed yet: ${ex.pending_sessions.slice(0, 6).join(", ")}${ex.pending_sessions.length > 6 ? " …" : ""}. Evaluate now seals what the stored data allows.` }]} />}
      {!rep && <div className="panel"><Empty title="Not evaluated yet" icon={Snowflake}>{ex.first_session ? `The first session that counts is ${ex.first_session}. ` : ""}Press Evaluate now, or leave the collector running: every active experiment is evaluated after each collection cycle. Nothing is recorded for a session until it has closed and its data is stored.</Empty></div>}
      {rep && (
        <>
          <div className="grid shrink-0 grid-cols-2 gap-2 lg:grid-cols-5">
            <div className="panel"><Stat label="Status" value={<span className={s.tone}><span aria-hidden className="mr-1">{s.glyph}</span>{rep.status === "evaluated" ? "evaluated" : rep.status === "waiting" ? "waiting for the first session" : rep.status === "awaiting_data" ? "awaiting data" : "data missing"}</span>} sub={`checked ${ago(rep.evaluated_at)}`} /></div>
            <div className="panel"><Stat label="Forward sessions" value={n0(rep.forward_sessions)} sub={rep.first ? `${rep.first} → ${rep.last}` : "none stored yet"} hint="Sessions on or after the first session that counts, sealed in the ledger once they closed and their data was stored. Earlier sessions never count." /></div>
            {ex.kind === "strategy" ? (
              <>
                <div className="panel"><Stat label="Sessions with option prices" value={rep.data ? `${rep.data.with_option_data} / ${rep.data.sessions}` : "—"} sub={rep.data?.pricing ? `${rep.data.pricing} pricing` : undefined} tone={rep.data && rep.data.with_option_data < rep.data.sessions ? "text-warn" : undefined} hint="A forward session without stored option prices cannot produce a simulated fill. Its signals are listed as missed." /></div>
                <div className="panel"><Stat label="Signals / filled" value={`${n0(rep.signals)} / ${n0(rep.executed)}`} sub={rep.not_executed ? `${rep.not_executed} could not be filled` : undefined} /></div>
                <div className="panel"><Stat label="Forward net P&L (paper)" value={rep.metrics ? money(rep.metrics.net_pnl, 0, true) : "—"} tone={rep.metrics?.net_pnl > 0 ? "text-up" : rep.metrics?.net_pnl < 0 ? "text-down" : undefined} sub={rep.metrics ? `${rep.metrics.trades} trade${rep.metrics.trades !== 1 ? "s" : ""}` : undefined} /></div>
              </>
            ) : (
              <>
                <div className="panel"><Stat label="Sessions evaluable" value={rep.data ? `${rep.data.evaluated} / ${rep.data.sessions}` : "—"} tone={rep.data?.not_evaluable ? "text-warn" : undefined} /></div>
                <div className="panel"><Stat label="Condition occurred" value={n0(rep.signals)} sub={rep.hits !== undefined ? `outcome followed ${rep.hits} time${rep.hits !== 1 ? "s" : ""}` : undefined} /></div>
                <div className="panel"><Stat label="Forward frequency" value={pp(rep.probability)} sub={rep.probability_ci95?.[0] !== null && rep.probability_ci95 ? `95% interval ${pp(rep.probability_ci95[0], 0)} to ${pp(rep.probability_ci95[1], 0)}` : undefined} /></div>
              </>
            )}
          </div>

          {(rep.notes || []).length > 0 && <Warnings items={rep.notes.map((t: string) => ({ level: c.consistent === false ? "warn" : "info", text: t }))} />}

          <div className="grid shrink-0 gap-2 lg:grid-cols-2">
            <Panel title={<span>Forward record against the benchmark {c.enough === false && <span className="ml-2 normal-case text-warn">too few observations to compare</span>}</span>}>
              {ex.kind === "strategy" ? (
                <Compare rows={[
                  ["Period", rep.first ? `${rep.first} → ${rep.last}` : "—", b.period ? `${b.period[0]} → ${b.period[1]}` : "—"],
                  ["Trades", n0(c.forward_trades), n0(c.backtest_trades)],
                  ["Win rate", pp(c.forward_win_rate, 0), pp(c.backtest_win_rate, 0)],
                  ["Win-rate 95% interval", c.win_rate_ci95?.[0] !== null && c.win_rate_ci95 ? `${pp(c.win_rate_ci95[0], 0)} to ${pp(c.win_rate_ci95[1], 0)}` : "—", ""],
                  ["Expectancy per trade", money(c.forward_expectancy, 0, true), money(c.backtest_expectancy, 0, true)],
                  ["Signals per 100 sessions", c.forward_signals_per_100_sessions?.toFixed(1) ?? "—", c.backtest_signals_per_100_sessions?.toFixed(1) ?? "—"],
                  ["Consistent with the backtest", c.consistent === null || c.consistent === undefined ? "cannot say yet" : c.consistent ? "● yes" : "✕ no", ""],
                ]} />
              ) : (
                <Compare rows={[
                  ["Period", rep.first ? `${rep.first} → ${rep.last}` : "—", b.period ? `${b.period[0]} → ${b.period[1]}` : "—"],
                  ["Occurrences", n0(c.forward_n), n0(c.backtest_n)],
                  ["Outcome frequency", pp(c.forward_probability), pp(c.backtest_probability)],
                  ["Occurrences per 100 sessions", c.forward_signals_per_100_sessions?.toFixed(1) ?? "—", c.backtest_signals_per_100_sessions?.toFixed(1) ?? "—"],
                  ["Consistent with history", c.consistent === null || c.consistent === undefined ? "cannot say yet" : c.consistent ? "● yes" : "✕ no", ""],
                ]} />
              )}
              <div className="border-t border-line px-2.5 py-1.5 text-2xs text-muted">Benchmark: {b.source || b.note || "none recorded"}.</div>
            </Panel>
            <Panel title="Evaluation history" bodyClass="overflow-auto max-h-[230px]">
              <table className="tbl"><thead><tr><th>Evaluated</th><th>Through session</th></tr></thead>
                <tbody>{(ex.history || []).map((h: any) => <tr key={h.id}><td className="text-muted">{ago(h.evaluated_at)}</td><td className="num">{h.through_date || "no forward session yet"}</td></tr>)}</tbody></table>
            </Panel>
          </div>

          <Account id={id} strategy={ex.kind === "strategy"} />
          {ex.kind === "strategy" && (rep.trades || []).length > 0 && (
            <Panel title={`Simulated forward fills (${rep.trades.length})`} bodyClass="overflow-auto max-h-[300px]" className="shrink-0">
              <table className="tbl"><thead><tr><th>Session</th><th>Contract</th><th className="r">Entry</th><th className="r">Exit</th><th>Price evidence</th><th>Exit reason</th><th className="r">P&L</th></tr></thead>
                <tbody>{rep.trades.map((t: any, i: number) => (
                  <tr key={i}><td className="num">{t.entry_date}</td><td>{(t.legs || []).map((l: any) => `${l.strike}${String(l.right || "").slice(0, 1).toUpperCase()} ${l.expiration}`).join(" / ")}</td>
                    <td className="r num">{t.legs?.[0]?.entry_price?.toFixed(2)}</td><td className="r num">{t.legs?.[0]?.exit_price?.toFixed(2) ?? "—"}</td><td className="text-muted">{t.legs?.[0]?.price_source}</td><td className="text-muted">{String(t.exit_reason || "").replace(/_/g, " ")}</td>
                    <td className={cn("r num", t.pnl > 0 ? "text-up" : t.pnl < 0 ? "text-down" : "")}>{money(t.pnl, 0, true)}</td></tr>
                ))}</tbody></table>
            </Panel>
          )}
          {ex.kind === "strategy" && (rep.open_positions || []).length > 0 && (
            <Panel title={`Open positions carried forward (${rep.open_positions.length})`} bodyClass="overflow-auto max-h-[220px]" className="shrink-0">
              <table className="tbl" data-testid="forward-open"><thead><tr><th>Entered</th><th>Contract</th><th className="r">Entry</th><th>Price evidence</th><th className="r">Units</th><th className="r">Marked P&amp;L</th><th>Carried from the ledger</th></tr></thead>
                <tbody>{rep.open_positions.map((o: any) => (
                  <tr key={o.key}><td className="num">{o.entry_date}</td><td>{(o.legs || []).map((l: any) => `${l.strike}${String(l.right || "").slice(0, 1).toUpperCase()} ${l.expiration}`).join(" / ")}</td><td className="r num">{o.legs?.[0]?.entry_price?.toFixed(2)}</td><td className="text-muted">{o.legs?.[0]?.price_source}</td><td className="r num">{o.units}</td>
                    <td className={cn("r num", o.marked_pnl > 0 ? "text-up" : o.marked_pnl < 0 ? "text-down" : "")}>{o.marked_pnl === null || o.marked_pnl === undefined ? "—" : money(o.marked_pnl, 0, true)}</td><td className={o.reproduced ? "text-muted" : "text-warn"}>{o.reproduced ? "yes, and reproduced by this evaluation" : "yes (this evaluation did not reproduce it)"}</td></tr>
                ))}</tbody></table>
              <div className="border-t border-line px-2.5 py-1.5 text-2xs text-muted">An open position is read from the sealed entry, so it survives a restart. It enters the P&amp;L when it closes.</div>
            </Panel>
          )}
          {(rep.sealed || []).length > 0 && (
            <Panel title={`Sealed sessions (${rep.forward_sessions})`} bodyClass="overflow-auto max-h-[220px]" className="shrink-0">
              <table className="tbl" data-testid="forward-sealed"><thead><tr><th>Session</th><th>Sealed</th><th>What was recorded for it</th></tr></thead>
                <tbody>{[...rep.sealed].reverse().map((x: any) => {
                  const { session, sealed_at, ...kinds } = x;
                  return <tr key={session}><td className="num">{session}</td><td className="text-muted">{ago(sealed_at)}</td><td className="text-muted">{Object.entries(kinds).filter(([, v]) => v).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(" · ") || "no signal"}</td></tr>;
                })}</tbody></table>
            </Panel>
          )}
          {(rep.revisions || []).length > 0 && (
            <Panel title={`Revisions: stored data changed after a session was sealed (${rep.revisions.length})`} bodyClass="overflow-auto max-h-[200px]" className="shrink-0">
              <table className="tbl" data-testid="forward-revisions"><thead><tr><th>Session</th><th>Noticed</th><th>What differed</th><th>Note</th></tr></thead>
                <tbody>{rep.revisions.map((x: any, i: number) => <tr key={i}><td className="num">{x.session}</td><td className="text-muted">{ago(x.recorded_at)}</td><td>{x.what} {x.of ? `(${x.of})` : ""}</td><td className="text-muted">{x.note}</td></tr>)}</tbody></table>
              <div className="border-t border-line px-2.5 py-1.5 text-2xs text-muted">The sealed entries were not altered. A revision is a record that today's data would give a different answer for a day that was already decided.</div>
            </Panel>
          )}
          {(rep.missing_data || []).length > 0 && (
            <Panel title={`Missing-data events (${rep.missing_data.length})`} bodyClass="overflow-auto max-h-[200px]" className="shrink-0">
              <table className="tbl" data-testid="forward-missing"><thead><tr><th>Session</th><th>What was missing</th><th>Why it matters</th></tr></thead>
                <tbody>{rep.missing_data.map((x: any, i: number) => <tr key={i}><td className="num">{x.session}</td><td>{String(x.why || "").replace(/_/g, " ")}</td><td className="text-muted">{String(x.reason || "").replace(/_/g, " ")}</td></tr>)}</tbody></table>
            </Panel>
          )}
          {ex.kind === "strategy" && (rep.missed_signals || []).length > 0 && (
            <Panel title={`Forward signals that could not be filled (${rep.missed_signals.length})`} bodyClass="overflow-auto max-h-[220px]" className="shrink-0">
              <table className="tbl"><thead><tr><th>Signal time</th><th>Kind</th><th>Why no fill was simulated</th></tr></thead>
                <tbody>{rep.missed_signals.map((m: any, i: number) => <tr key={i}><td className="num">{String(m.signal_time || "").replace("T", " ").slice(0, 16)}</td><td className="text-muted">{m.kind}</td><td>{String(m.reason || "").replace(/_/g, " ")}</td></tr>)}</tbody></table>
            </Panel>
          )}
        </>
      )}
      <Dialog open={showCfg} onOpenChange={setShowCfg} title="Frozen configuration (read-only)" width={760}>
        <pre className="max-h-[60vh] overflow-auto rounded-sm border border-line bg-bg-0 p-2 font-mono text-[11.5px] leading-[1.45] text-fg-2">{JSON.stringify(ex.config, null, 2)}</pre>
        <div className="mt-2 text-xs text-muted">Hash {ex.config_hash}. Execution assumptions frozen with it: {JSON.stringify(ex.assumptions?.execution || ex.assumptions || {})}</div>
      </Dialog>
      {versioning && <VersionDialog ex={ex} onClose={() => setVersioning(false)} onCreated={(nid) => { setVersioning(false); refresh(); nav(`/forward/${nid}`); }} />}
    </div>
  );
}

function Freeze({ onCreated }: { onCreated: (id: string) => void }) {
  const runs = useQuery({ queryKey: ["runs", "freezable"], queryFn: () => api.get<any[]>("/api/backtests?kind=backtest,graph&limit=150"), staleTime: 5_000 });
  const [rid, setRid] = useState("");
  const [busy, setBusy] = useState(false);
  // a run that stopped because the data could not cover it has no result to freeze: it is not offered
  const list = (runs.data || []).filter((r) => r.status === "done" && r.summary?.outcome !== "needs_decision");
  const go = async () => {
    setBusy(true);
    try { const ex = await api.post<any>("/api/forward", { run_id: rid }); toast.success(`Frozen. The first session that counts is ${ex.first_session}.`); onCreated(ex.id); } catch (e: any) { toast.error(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="flex items-center gap-1.5">
      <Select sz="sm" className="w-[300px]" value={rid} onChange={setRid} options={[{ value: "", label: list.length ? "Choose a finished backtest or study…" : "No finished backtest or study yet" }, ...list.map((r) => ({ value: r.id, label: `${r.kind === "graph" ? "Study" : "Backtest"} · ${r.name}`.slice(0, 70) }))]} />
      <Button size="sm" variant="primary" disabled={!rid || busy} onClick={go} data-testid="forward-freeze"><Snowflake />Freeze it</Button>
    </div>
  );
}

export default function Forward() {
  const nav = useNavigate();
  const { id } = useParams();
  const q = useQuery({ queryKey: ["forward", "list"], queryFn: () => api.get<{ experiments: any[]; min_forward: number }>("/api/forward"), refetchInterval: 60_000 });
  const list = q.data?.experiments || [];
  const cur = id || list[0]?.id;
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="forward">
      <PageHeader title={<span className="flex items-center gap-2"><Snowflake className="size-3.5 text-info" />Forward validation</span>} sub={<span className="text-xs text-muted">Paper only. Frozen configurations judged on sessions that did not exist when they were frozen.</span>}>
        <Freeze onCreated={(nid) => { queryClient.invalidateQueries({ queryKey: ["forward"] }); nav(`/forward/${nid}`); }} />
      </PageHeader>
      {q.isLoading ? <Loading /> : list.length === 0 ? (
        <Empty title="No frozen experiment yet" icon={Snowflake}>
          A backtest can always be made to look good, because its author has seen the data. Freeze a finished backtest or market study here and Strata will judge it only on sessions collected afterwards. It never places an order; fills are simulated on observed prices as they are collected. At least {q.data?.min_forward ?? 10} forward observations are needed before any comparison is drawn.
        </Empty>
      ) : (
        <div className="flex min-h-0 flex-1">
          <aside className="w-[300px] shrink-0 overflow-auto border-r border-line bg-bg-1" aria-label="Experiments">
            {list.map((ex) => {
              const s = st(ex);
              return (
                <button key={ex.id} onClick={() => nav(`/forward/${ex.id}`)} className={cn("block w-full border-b border-line/70 px-3 py-2 text-left hover:bg-bg-2", cur === ex.id && "bg-bg-3 shadow-[inset_2px_0_0_var(--color-info)]")}>
                  <div className="flex items-center gap-1.5"><span className="truncate text-xs font-medium text-fg">{ex.name}</span><span className="shrink-0 text-2xs text-muted">v{ex.version}</span></div>
                  <div className="mt-0.5 flex items-center gap-2 text-2xs text-muted"><span>{ex.symbol}</span><span>counts from {ex.first_session || ex.activation_date}</span>{ex.status !== "active" && <span>■ stopped</span>}</div>
                  <div className={cn("mt-0.5 text-2xs", s.tone)}><span aria-hidden className="mr-1">{s.glyph}</span>{s.word}{ex.latest?.forward_sessions ? ` · ${ex.latest.forward_sessions} forward session${ex.latest.forward_sessions !== 1 ? "s" : ""}` : ""}</div>
                </button>
              );
            })}
            <div className="p-3 text-2xs text-muted"><Play className="mr-1 inline size-3" />Active experiments are re-evaluated after every collection cycle. Nothing is collected while Strata or this machine is off.</div>
          </aside>
          {cur ? <Detail key={cur} id={cur} /> : null}
        </div>
      )}
    </div>
  );
}
