/* ML research (optional, advanced). A model here predicts an outcome of the underlying from point-in-time
   features, scored out of sample with purged, embargoed chronological splits. Predictive skill is reported against
   the base rate; it is never presented as profitability. The only bridge to a strategy is a saved walk-forward
   signal, which the deterministic backtester then prices on observed option data. */
import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, Check, Copy, Loader2, Play, Save, ShieldAlert, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { JobBar, SyntheticBadge, cancelJob } from "@/components/common";
import { api, queryClient, useJob } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { pp } from "@/lib/lab";
import { Callout, Empty, Section, Skeleton, StatTile } from "../lab/bits";
import { pv } from "../lab/Research";

const skill = (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(1)}%`);
const VERDICT: Record<string, { glyph: string; word: string; tone: "good" | "warn" | "info" }> = {
  no_skill: { glyph: "○", word: "No out-of-sample skill", tone: "info" },
  weak: { glyph: "◐", word: "Weak, not dependable", tone: "warn" },
  some_skill: { glyph: "●", word: "Some out-of-sample skill", tone: "good" },
};

function Result({ rid }: { rid: string }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["ml", rid], queryFn: () => api.get<any>(`/api/runs/${rid}/ml`), retry: false });
  const [name, setName] = useState(`ml_${rid}`);
  const [saved, setSaved] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const r = q.data;
  const sel = r?.models?.find((m: any) => m.name === r.selected);
  const calOpt = useMemo(() => {
    if (!sel?.calibration?.length) return null;
    const pts = sel.calibration.map((c: any) => [c.predicted, c.observed, c.n]);
    const lo = Math.min(...pts.map((p: number[]) => Math.min(p[0], p[1]))), hi = Math.max(...pts.map((p: number[]) => Math.max(p[0], p[1])));
    const a = Math.max(0, lo - 0.03), b = Math.min(1, hi + 0.03);
    return {
      grid: { left: 44, right: 14, top: 14, bottom: 34 },
      tooltip: { formatter: (p: any) => (p.seriesType === "scatter" ? `predicted ${(p.value[0] * 100).toFixed(1)}%<br/>observed ${(p.value[1] * 100).toFixed(1)}%<br/>${p.value[2]} sessions` : "") },
      xAxis: { type: "value", min: a, max: b, name: "predicted probability", nameLocation: "middle", nameGap: 22, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
      yAxis: { type: "value", min: a, max: b, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
      series: [
        { type: "line", data: [[a, a], [b, b]], symbol: "none", lineStyle: { color: C.faint, type: "dashed", width: 1 }, silent: true },
        { type: "line", data: pts.map((p: number[]) => [p[0], p[1]]), symbol: "none", lineStyle: { color: C.accent, width: 1.2, opacity: 0.6 }, silent: true },
        { type: "scatter", data: pts, symbolSize: (v: number[]) => 6 + Math.sqrt(v[2]) * 0.7, itemStyle: { color: C.accent } },
      ],
    };
  }, [sel]);
  if (q.isLoading) return <><Skeleton className="h-24" /><Skeleton className="h-72" /></>;
  if (q.isError || !r) return <Callout tone="bad" title="This ML run has no result">{(q.error as any)?.message || "It may have failed or been cancelled."}</Callout>;
  const v = VERDICT[r.verdict] || VERDICT.no_skill;
  const save = async () => {
    setSaving(true);
    try {
      const s = await api.post<any>(`/api/runs/${rid}/ml/signal`, { name });
      setSaved(s);
      queryClient.invalidateQueries({ queryKey: ["signals"] });
      toast.success(`Signal "${s.name}" saved`);
    } catch (e: any) { toast.error(e.message); } finally { setSaving(false); }
  };
  return (
    <div className="flex flex-col gap-4" data-testid="ml-result">
      <div>
        <div className="flex flex-wrap items-center gap-2 text-xs dim"><span>{r.symbol}</span><span>·</span><span>decided at the {r.decision_tick}</span><span>·</span><span>{r.split.train[0]} to {r.split.test[1]}</span><SyntheticBadge on={r.synthetic} /></div>
        <h1 className="mt-1 !text-[21px]">Can these features predict: {r.outcome_text}?</h1>
      </div>
      <Callout tone={v.tone} title={<span><span aria-hidden className="mr-1.5">{v.glyph}</span>{v.word}</span>}>{(r.summary || []).join(" ")}</Callout>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatTile label="Sessions" value={r.n.toLocaleString()} sub={`${r.split.train_n.toLocaleString()} train · ${r.split.test_n.toLocaleString()} final test`} />
        <StatTile label="Base rate" value={pp(r.base_rate)} sub="how often the outcome happens" hint="A forecast that always says this number is the baseline every model has to beat." />
        <StatTile label="Brier skill, cross-validation" value={skill(sel?.cv?.brier_skill)} sub={sel ? `positive in ${sel.cv.skill_folds_positive} of ${sel.cv.folds.length} folds` : undefined} hint="Improvement in the Brier score over the base-rate forecast, on purged and embargoed folds. Zero means the model adds nothing." />
        <StatTile label="Brier skill, final test" value={skill(sel?.test?.brier_skill)} sub={`${r.split.test[0]} onward`} hint="The last fifth of the history. No model was chosen using it." />
        <StatTile label="Calibration error" value={sel?.calibration_error === null || sel?.calibration_error === undefined ? "—" : `${(sel.calibration_error * 100).toFixed(1)} pts`} sub={r.calibrated ? "after calibration" : "uncalibrated"} hint="Average gap between the predicted probability and how often the outcome then happened, on the final test set." />
      </div>

      <Section title="Models" sub={`selected by ${r.selected_by}; every test score is shown and none was used to choose`}>
        <div className="overflow-auto">
          <table className="t">
            <thead><tr><th>Model</th><th className="r">CV Brier skill</th><th className="r">Folds positive</th><th className="r">CV log-loss skill</th><th className="r">Test Brier skill</th><th className="r">Test log loss</th><th className="r">Test AUC</th><th className="r">Test accuracy</th><th className="r">Always-majority accuracy</th></tr></thead>
            <tbody>
              {r.models.map((m: any) => m.status !== "ok" ? (
                <tr key={m.name}><td>{m.label}</td><td colSpan={8} className="text-down">{m.error}</td></tr>
              ) : (
                <tr key={m.name} className={cn(m.name === r.selected && "sel")}>
                  <td>{m.label}{m.name === r.selected && <span className="ml-2 text-[11px] text-accent">◆ selected</span>}</td>
                  <td className="r num">{skill(m.cv.brier_skill)}</td><td className="r num">{m.cv.skill_folds_positive} / {m.cv.folds.length}</td><td className="r num">{skill(m.cv.log_loss_skill)}</td>
                  <td className="r num">{skill(m.test.brier_skill)}</td><td className="r num">{m.test.log_loss.toFixed(4)}</td><td className="r num">{m.test.auc === null ? "—" : m.test.auc.toFixed(3)}</td>
                  <td className="r num">{pp(m.test.accuracy)}</td><td className="r num dim">{pp(m.test.accuracy_baseline)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 text-xs dim">{r.split.method}. Embargo: {r.split.embargo_sessions} session{r.split.embargo_sessions !== 1 ? "s" : ""}. Accuracy is shown next to what always guessing the majority would score, because on an imbalanced outcome accuracy alone says little.</div>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Calibration on the final test set" sub="points on the dashed line mean the stated probability is honest">
          {calOpt ? <EChart option={calOpt} height={260} /> : <div className="text-xs dim">Too few test sessions to bin.</div>}
        </Section>
        <Section title="Skill by cross-validation fold" sub="each fold is a later stretch of history">
          {sel && <EChart height={260} option={{ grid: { left: 44, right: 10, top: 14, bottom: 40 }, tooltip: { trigger: "axis", valueFormatter: (x: number) => skill(x) },
            xAxis: { type: "category", data: sel.cv.folds.map((f: any) => `${f.start.slice(0, 4)}–${f.end.slice(2, 4)}`), axisLabel: { rotate: 30 } }, yAxis: { type: "value", axisLabel: { formatter: (x: number) => `${(x * 100).toFixed(0)}%` } },
            series: [{ type: "bar", data: sel.cv.folds.map((f: any) => ({ value: f.brier_skill, itemStyle: { color: (f.brier_skill || 0) > 0 ? C.up : C.down } })), barMaxWidth: 26, markLine: { silent: true, symbol: "none", data: [{ yAxis: 0 }], lineStyle: { color: C.faint }, label: { show: false } } }] }} />}
        </Section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="What the model relies on" sub="how much the test score worsens when one feature is shuffled">
          {(r.importance || []).length === 0 ? <div className="text-xs dim">Not available for this model.</div> : (
            <div className="flex flex-col gap-1.5">
              {r.importance.slice(0, 12).map((f: any) => {
                const top = Math.max(...r.importance.map((x: any) => Math.abs(x.importance))) || 1;
                return (
                  <div key={f.feature} className="grid grid-cols-[minmax(0,1fr)_120px_92px] items-center gap-2 text-[12.5px]">
                    <span className="truncate" title={f.feature}>{f.label}</span>
                    <span className="relative h-2 rounded-full bg-bg-3"><span className={cn("absolute inset-y-0 left-0 rounded-full", f.matters ? "bg-accent" : "bg-faint")} style={{ width: `${Math.max(2, (Math.abs(f.importance) / top) * 100)}%` }} /></span>
                    <span className="text-right text-[11px] dim">{f.matters ? "● matters" : "○ within noise"}</span>
                  </div>
                );
              })}
            </div>
          )}
          {r.coefficients && <div className="mt-3 text-xs dim">Largest coefficients per standard deviation: {r.coefficients.slice(0, 4).map((c: any) => `${c.label} ${c.coef_per_sd > 0 ? "+" : ""}${c.coef_per_sd.toFixed(2)}`).join(" · ")}.</div>}
        </Section>
        <Section title="Leakage checks" sub={r.leakage.passed ? "all passed" : r.leakage.not_run ? `${r.leakage.not_run} could not be run` : "a check failed"} right={r.leakage.passed ? <ShieldCheck className="size-4 text-up" /> : <ShieldAlert className="size-4 text-warn" />}>
          <div className="flex flex-col gap-2.5">
            {r.leakage.checks.map((c: any) => (
              <div key={c.check} className="flex gap-2.5 text-[12.5px]">
                {c.passed === true ? <Check className="mt-0.5 size-4 shrink-0 text-up" /> : c.passed === false ? <X className="mt-0.5 size-4 shrink-0 text-down" /> : <span className="mt-0.5 size-4 shrink-0 text-center text-warn">?</span>}
                <div><div className="font-medium">{c.check} <span className="font-normal dim">· {c.passed === true ? "passed" : c.passed === false ? "failed" : "not run"}</span></div><div className="dim">{c.detail}</div></div>
              </div>
            ))}
            <div className="flex gap-2.5 text-[12.5px]">
              {r.placebo.passed ? <Check className="mt-0.5 size-4 shrink-0 text-up" /> : <X className="mt-0.5 size-4 shrink-0 text-warn" />}
              <div><div className="font-medium">Beats a time-shifted placebo <span className="font-normal dim">· {r.placebo.beats} of {r.placebo.of}</span></div><div className="dim">Real skill {skill(r.placebo.real_skill)} against a best placebo of {skill(r.placebo.placebo_max)}. {r.placebo.how}</div></div>
            </div>
          </div>
        </Section>
      </div>

      {(r.regimes || []).length > 0 && (
        <div className="grid gap-4 lg:grid-cols-3">
          {r.regimes.map((g: any) => (
            <Section key={g.title} title={g.title}>
              <table className="t"><thead><tr><th>Regime</th><th className="r">Sessions</th><th className="r">Base rate</th><th className="r">Brier skill</th><th className="r">AUC</th></tr></thead>
                <tbody>{g.rows.map((x: any) => <tr key={x.regime}><td>{x.regime}{x.thin && <span className="ml-1.5 text-[11px] text-warn">thin</span>}</td><td className="r num">{x.n}</td><td className="r num">{pp(x.base_rate)}</td><td className="r num">{skill(x.brier_skill)}</td><td className="r num">{x.auc === null || x.auc === undefined ? "—" : x.auc.toFixed(2)}</td></tr>)}</tbody></table>
            </Section>
          ))}
        </div>
      )}

      {r.signal && (
        <Section title="Walk-forward signal" sub="every prediction came from a model fitted only on sessions whose outcomes were already known">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatTile label="Signal days" value={r.signal.signals.toLocaleString()} sub={`of ${r.signal.predicted.toLocaleString()} predicted sessions (top ${Math.round((1 - r.signal.quantile) * 100)}%)`} />
            <StatTile label="Outcome on signal days" value={pp(r.signal.hit_rate_on_signal)} sub={`other days ${pp(r.signal.hit_rate_otherwise)}`} />
            <StatTile label="Could this be chance?" value={pv(r.signal.p_value)} sub={r.signal.lift_ci95 ? `difference ${(r.signal.lift_ci95[0] * 100).toFixed(1)} to ${(r.signal.lift_ci95[1] * 100).toFixed(1)} pts` : undefined} />
            <StatTile label="Refitted every" value={`${r.signal.refit_every} sessions`} sub={`${r.signal.first} to ${r.signal.last}`} />
          </div>
          <Callout tone="warn" className="mt-3" title="This is not a profit estimate">A higher hit rate on the underlying says nothing about option premiums, spreads or fills. Save the signal, use it as an entry condition, and backtest that strategy on observed option prices.</Callout>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input className="field !w-64 font-mono" value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_]/g, "_"))} aria-label="Signal name" />
            <button className="btn primary sm" onClick={save} disabled={saving || !r.signal.signals} data-testid="ml-save-signal">{saving ? <Loader2 className="animate-spin" /> : <Save />}Save as a signal</button>
            {saved && (
              <>
                <code className="rounded-md border border-line-2 bg-bg-2 px-2 py-1 font-mono text-xs">signal("{saved.name}")</code>
                <button className="btn quiet sm" onClick={() => { navigator.clipboard.writeText(`signal("${saved.name}")`); toast.success("Copied"); }}><Copy />Copy</button>
                <button className="btn sm" onClick={() => nav(`/ide?signal=${encodeURIComponent(saved.name)}&symbol=${r.symbol}`)}>Use it in the Strategy IDE</button>
              </>
            )}
          </div>
        </Section>
      )}

      {(r.dropped_features || []).length > 0 && <Callout tone="info" title="Features left out">{r.dropped_features.map((d: any) => `${d.feature}: ${d.why}`).join(" · ")}</Callout>}
      <Callout tone="info" title="How this was measured">{(r.notes || []).map((n: string, i: number) => <div key={i} className={cn(i > 0 && "mt-1.5")}>{n}</div>)}</Callout>
    </div>
  );
}

export default function MlLab() {
  const nav = useNavigate();
  const { id } = useParams();
  const meta = useQuery({ queryKey: ["ml-meta"], queryFn: () => api.get<any>("/api/ml/meta") });
  const past = useQuery({ queryKey: ["runs", "ml"], queryFn: () => api.get<any[]>("/api/backtests?kind=ml&limit=30"), staleTime: 5_000 });
  const [symbol, setSymbol] = useState("");
  const [oc, setOc] = useState({ entry: "close", sessions: 5, direction: "up", target: 0, stop: 0 });
  const [models, setModels] = useState<string[]>(["logistic", "lasso", "gbt"]);
  const [feats, setFeats] = useState<string[] | null>(null);
  const [period, setPeriod] = useState({ start: "", end: "" });
  const [quantile, setQuantile] = useState(0.8);
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const job = useJob(jobId, (s) => {
    setJobId(null);
    queryClient.invalidateQueries({ queryKey: ["runs", "ml"] });
    if (s.status === "done") nav(`/ml/${s.id}`); else if (s.status === "error") toast.error(s.error || "The ML study failed");
  });
  useEffect(() => { if (meta.data && !symbol) setSymbol(meta.data.default_symbol || meta.data.symbols?.[0] || ""); }, [meta.data, symbol]);
  useEffect(() => { if (meta.data && feats === null) setFeats(meta.data.default_features); }, [meta.data, feats]);
  const groups = useMemo(() => {
    const by: Record<string, any[]> = {};
    for (const f of meta.data?.features || []) (by[f.group] ||= []).push(f);
    return Object.entries(by);
  }, [meta.data]);
  const start = async () => {
    setStarting(true);
    try {
      const outcome = { entry: oc.entry, horizon: oc.sessions > 0 ? `sessions:${oc.sessions}` : "session", direction: oc.direction, target: oc.target || 0, stop: oc.stop || null, measure: oc.target > 0 ? "touch" : "close" };
      const res = await api.post<{ run_id: string }>("/api/ml/run", { spec: { symbol, outcome, models, features: feats, start: period.start || null, end: period.end || null, signal: { quantile } } });
      setJobId(res.run_id);
    } catch (e: any) { toast.error(e.message); } finally { setStarting(false); }
  };
  const busy = starting || !!jobId;

  return (
    <div className="lab flex h-full min-h-0">
      <aside className="flex w-[304px] shrink-0 flex-col border-r border-line bg-bg-1" aria-label="ML study settings">
        <div className="border-b border-line px-4 py-3">
          <div className="eyebrow flex items-center gap-2"><BrainCircuit className="size-3.5 text-accent" />ML research</div>
          <div className="mt-1 text-xs dim">Optional. Tests whether features predict an outcome, out of sample.</div>
        </div>
        {meta.data && !meta.data.available ? (
          <div className="p-4"><Callout tone="warn" title="scikit-learn is not installed">{meta.data.detail}</Callout></div>
        ) : (
          <>
            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-auto px-4 py-3">
              <label className="flex flex-col gap-1"><span className="text-xs dim">Symbol</span><select className="field" value={symbol} onChange={(e) => setSymbol(e.target.value)}>{(meta.data?.symbols || []).map((s: string) => <option key={s}>{s}</option>)}</select></label>
              <div>
                <div className="mb-1 text-xs dim">Outcome to predict</div>
                <div className="grid grid-cols-2 gap-2">
                  <select className="field" value={oc.entry} onChange={(e) => setOc((s) => ({ ...s, entry: e.target.value }))} aria-label="Decision time"><option value="close">from the close</option><option value="open">from the open</option></select>
                  <select className="field" value={oc.sessions} onChange={(e) => setOc((s) => ({ ...s, sessions: Number(e.target.value) }))} aria-label="Horizon">{(oc.entry === "open" ? [0, 1, 2, 3, 5, 10] : [1, 2, 3, 5, 10, 20]).map((h) => <option key={h} value={h}>{h === 0 ? "to that close" : `${h} session${h > 1 ? "s" : ""} later`}</option>)}</select>
                  <select className="field" value={oc.direction} onChange={(e) => setOc((s) => ({ ...s, direction: e.target.value }))} aria-label="Direction"><option value="up">price is higher</option><option value="down">price is lower</option></select>
                  <label className="relative"><input type="number" className="field num" min={0} step={0.25} value={oc.target} onChange={(e) => setOc((s) => ({ ...s, target: Number(e.target.value) }))} aria-label="Target percent" /><span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-faint">% target</span></label>
                </div>
                <div className="mt-1 text-[11px] dim">{oc.target > 0 ? `Reaches ${oc.target}% ${oc.direction} at any point in the window.` : `Finishes ${oc.direction === "up" ? "above" : "below"} the entry price.`}</div>
              </div>
              <div>
                <div className="mb-1 text-xs dim">Models to compare</div>
                <div className="flex flex-col gap-1">{(meta.data?.models || []).map((m: any) => (
                  <label key={m.name} className="flex items-center gap-2 text-[12.5px] soft"><input type="checkbox" className="accent-[var(--color-accent)]" checked={models.includes(m.name)} onChange={(e) => setModels((s) => (e.target.checked ? [...s, m.name] : s.filter((x) => x !== m.name)))} />{m.label}</label>
                ))}</div>
              </div>
              <div>
                <div className="mb-1 flex items-center justify-between text-xs dim"><span>Features ({(feats || []).length})</span><button className="underline-offset-2 hover:underline" onClick={() => setFeats(meta.data?.default_features || [])}>reset</button></div>
                {groups.map(([g, list]) => (
                  <div key={g} className="mb-1.5">
                    <div className="text-[10.5px] font-medium uppercase tracking-wide text-faint">{g}</div>
                    {list.map((f: any) => (
                      <label key={f.name} className={cn("flex items-center gap-2 py-px text-[12px] soft", f.needs_intraday && oc.entry !== "open" && "opacity-70")} title={`${f.expr}${f.needs_intraday ? " · read during the session, so it is left out when the decision is at the open or the close" : ""}`}>
                        <input type="checkbox" className="accent-[var(--color-accent)]" checked={(feats || []).includes(f.name)} onChange={(e) => setFeats((s) => (e.target.checked ? [...(s || []), f.name] : (s || []).filter((x) => x !== f.name)))} /><span className="truncate">{f.label}</span>
                      </label>
                    ))}
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1"><span className="text-xs dim">From</span><input type="date" className="field" value={period.start} onChange={(e) => setPeriod((s) => ({ ...s, start: e.target.value }))} /></label>
                <label className="flex flex-col gap-1"><span className="text-xs dim">To</span><input type="date" className="field" value={period.end} onChange={(e) => setPeriod((s) => ({ ...s, end: e.target.value }))} /></label>
              </div>
              <label className="flex flex-col gap-1"><span className="flex justify-between text-xs dim"><span>Signal: act on the top</span><span className="num text-fg">{Math.round((1 - quantile) * 100)}% of predictions</span></span>
                <input type="range" min={0.5} max={0.95} step={0.05} value={quantile} onChange={(e) => setQuantile(Number(e.target.value))} className="accent-[var(--color-accent)]" aria-label="Signal quantile" /></label>
            </div>
            <div className="flex flex-col gap-2 border-t border-line px-4 py-3">
              <button className="btn primary" onClick={start} disabled={busy || !symbol || !models.length || !(feats || []).length} data-testid="ml-run">{busy ? <Loader2 className="animate-spin" /> : <Play />}Run the study</button>
              <JobBar job={job} onCancel={() => jobId && cancelJob(jobId)} />
              <div className="text-[11px] dim">Splits are chronological with purging and an embargo. Random splits are not offered: neighbouring sessions share information.</div>
            </div>
          </>
        )}
      </aside>
      <div className="min-w-0 flex-1 overflow-auto">
        <div className="mx-auto flex w-full max-w-[1120px] flex-col gap-4 px-6 py-6">
          {id ? <Result key={id} rid={id} /> : (
            <>
              <div className="card"><Empty title="Choose an outcome and run a study" icon={BrainCircuit}>The result reports skill against the base rate, calibration, stability across time, a time-shifted placebo and leakage checks. A model with no skill is reported as exactly that.</Empty></div>
              {(past.data || []).length > 0 && (
                <Section title="Earlier ML studies">
                  <table className="t"><thead><tr><th>Study</th><th>Status</th><th className="r">When</th></tr></thead>
                    <tbody>{(past.data || []).map((p) => <tr key={p.id} className="click" onClick={() => p.status === "done" && nav(`/ml/${p.id}`)}><td>{p.name}</td><td className="dim">{p.status}</td><td className="r dim">{ago(p.created_at)}</td></tr>)}</tbody></table>
                </Section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
