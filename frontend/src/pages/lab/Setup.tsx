/* The interpretation of a request, editable in place: what will be traded or measured, when, how it exits, on which data.
   Plain English on the left, the structured configuration on demand, and a preview that evaluates the rules on a real session. */
import { useMutation } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Code2, Eye, Play, Plus, RotateCcw, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ExprInput } from "@/components/research/ExprInput";
import { Tip } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { MODE_HELP } from "@/lib/lab";
import { Callout, Section, StatusChip, T, Term } from "./bits";
import { Rules } from "./CalendarView";
import { Timeline } from "./Pipeline";

const POSITIONS: [string, string][] = [["long_call", "Buy calls"], ["long_put", "Buy puts"], ["short_put", "Sell puts"], ["short_call", "Sell calls"], ["bull_call_spread", "Call debit spread (bullish)"], ["bear_put_spread", "Put debit spread (bearish)"],
  ["bull_put_spread", "Put credit spread (bullish)"], ["bear_call_spread", "Call credit spread (bearish)"], ["iron_condor", "Iron condor"], ["iron_butterfly", "Iron butterfly"], ["long_straddle", "Buy a straddle"], ["short_straddle", "Sell a straddle"],
  ["long_strangle", "Buy a strangle"], ["short_strangle", "Sell a strangle"], ["covered_call", "Covered call"]];
const TEMPLATES: [string, string, string][] = [
  ["The previous day fell at least 2%", "previous_session.return_pct <= -2", "previous session fell 2% or more (close-to-close)"],
  ["It opens with a gap down of at least 0.5%", "gap_pct <= -0.5", "opening gap down 0.5% or more"],
  ["VIX is below 20", "vix < 20", "VIX below 20"],
  ["Price is above the session's average price (VWAP)", "price > vwap", "price above session VWAP"],
  ["Price breaks above yesterday's high", "crosses_above(price, prev_high)", "price breaks above yesterday's high"],
  ["Three red 5-minute candles in a row", 'consecutive_bearish_candles("5m") >= 3', "3 consecutive red 5m candles"],
  ["Another symbol has not broken its own high (QQQ)", "QQQ.high <= QQQ.prev_high", "QQQ has not traded above its previous-day high today"],
];

function Row({ label, children, hint }: { label: React.ReactNode; children: React.ReactNode; hint?: string }) {
  return (
    <div className="grid grid-cols-[150px_minmax(0,1fr)] items-start gap-3 py-1.5">
      <div className="pt-1.5 text-[13px] dim">{label}{hint && <Tip content={hint}><span className="ml-1 cursor-help text-faint">ⓘ</span></Tip>}</div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
const Num = ({ value, onChange, suffix, w = 84, placeholder }: { value: number | null | undefined; onChange: (v: number | null) => void; suffix?: string; w?: number; placeholder?: string }) => (
  <span className="relative inline-block" style={{ width: w }}>
    <input className="field num" value={value ?? ""} placeholder={placeholder} inputMode="decimal" onChange={(e) => { const t = e.target.value; onChange(t === "" ? null : Number.isFinite(Number(t)) ? Number(t) : value ?? null); }} />
    {suffix && <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-xs text-faint">{suffix}</span>}
  </span>
);
const Sel = ({ value, onChange, options, w }: { value: string; onChange: (v: string) => void; options: [string, string][]; w?: number }) => (
  <select className="field cursor-pointer" style={{ width: w }} value={value} onChange={(e) => onChange(e.target.value)}>{options.map(([v, l]) => <option key={v} value={v} className="bg-bg-2">{l}</option>)}</select>
);

function ExprList({ items, onChange, kind, addLabel }: { items: any[]; onChange: (x: any[]) => void; kind: "cond" | "stage" | "add" | "exit"; addLabel: string }) {
  const [menu, setMenu] = useState(false);
  const set = (i: number, patch: any) => onChange(items.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  const move = (i: number, d: number) => { const a = [...items]; const j = i + d; if (j < 0 || j >= a.length) return; [a[i], a[j]] = [a[j], a[i]]; onChange(a); };
  return (
    <div className="flex flex-col gap-2">
      {items.map((x, i) => (
        <div key={i} className="rounded-lg border border-line bg-bg-2/50 px-3 py-2.5">
          <div className="flex items-center gap-2">
            {kind === "stage" && <span className="grid size-5 shrink-0 place-items-center rounded-full bg-synth/15 text-[11px] font-semibold text-synth">{i + 1}</span>}
            <input className="min-w-0 flex-1 bg-transparent text-[13.5px] outline-none placeholder:text-faint" value={x.text || ""} placeholder="describe it in words (optional)" onChange={(e) => set(i, { text: e.target.value })} />
            {kind === "stage" && <><input className="field !h-7 !w-24 font-mono !text-xs" value={x.name} onChange={(e) => set(i, { name: e.target.value.replace(/[^a-z0-9_]/gi, "") })} title="Short name later steps can refer to" />
              <button className="btn quiet sm !px-1" onClick={() => move(i, -1)} disabled={i === 0} aria-label="Move up"><ArrowUp /></button><button className="btn quiet sm !px-1" onClick={() => move(i, 1)} disabled={i === items.length - 1} aria-label="Move down"><ArrowDown /></button></>}
            {kind === "add" && <span className="flex items-center gap-1 text-xs dim">add <Num w={52} value={x.quantity} onChange={(v) => set(i, { quantity: Math.max(1, v || 1) })} /> up to <Num w={52} value={x.max_adds} onChange={(v) => set(i, { max_adds: Math.max(1, v || 1) })} />×</span>}
            <button className="btn quiet sm !px-1" onClick={() => onChange(items.filter((_, j) => j !== i))} aria-label="Remove"><Trash2 /></button>
          </div>
          <ExprInput className="mt-1.5" value={x.expr || ""} onChange={(v) => set(i, { expr: v, source: "stated" })} />
        </div>
      ))}
      <div className="relative">
        <button className="btn sm" onClick={() => (kind === "cond" ? setMenu((m) => !m) : onChange([...items, kind === "stage" ? { name: `step${items.length + 1}`, expr: "", text: "" } : kind === "add" ? { expr: "price <= entry.underlying_price * (1 - 0.005)", text: "the underlying is a further 0.5% lower than at the first entry", quantity: 1, max_adds: 1 } : { expr: "price >= vwap", text: "price is back at or above session VWAP" }]))}>
          <Plus />{addLabel}
        </button>
        {menu && kind === "cond" && (
          <div className="absolute left-0 top-9 z-20 w-[420px] rounded-xl border border-line-2 bg-bg-2 p-1.5 shadow-2xl">
            {TEMPLATES.map(([l, e, t]) => <button key={e} className="block w-full rounded-lg px-3 py-2 text-left text-[13px] hover:bg-bg-4" onClick={() => { onChange([...items, { expr: e, text: t, source: "stated" }]); setMenu(false); }}>{l}<span className="ml-2 font-mono text-[11px] text-faint">{e}</span></button>)}
            <button className="block w-full rounded-lg px-3 py-2 text-left text-[13px] hover:bg-bg-4" onClick={() => { onChange([...items, { expr: "", text: "", source: "stated" }]); setMenu(false); }}>Write my own rule…</button>
          </div>
        )}
      </div>
    </div>
  );
}

export function Setup({ plan, onApply, onConfirm, busy, refresh }: { plan: any; onApply: (draft: any, run: boolean) => void; onConfirm: (versionId: number) => void; busy: boolean; refresh: () => void }) {
  const [d, setD] = useState<any>(plan.draft);
  const [showJson, setShowJson] = useState(false);
  const serverDraft = JSON.stringify(plan.draft);
  useEffect(() => setD(plan.draft), [plan.version_id, serverDraft]); // eslint-disable-line
  const set = (patch: any) => setD((x: any) => ({ ...x, ...patch }));
  const dirty = useMemo(() => JSON.stringify(d) !== serverDraft, [d, serverDraft]);
  const options = d.kind === "options";
  const st = d.study || {};
  const setSt = (patch: any) => set({ study: { reference: "entry", measure: "touch", horizon_kind: "session", horizon_time: "12:00", horizon_sessions: 1, ...st, ...patch } });
  const entryKind = d.entry_time === "open" ? "open" : d.entry_time === "close" ? "close" : d.entry_time ? "time" : "signal";
  const tUnit = d.premium_target_pct !== null && d.premium_target_pct !== undefined ? "premium" : "underlying";
  const sUnit = d.premium_stop_pct !== null && d.premium_stop_pct !== undefined ? "premium" : "underlying";
  const tVal = tUnit === "premium" ? d.premium_target_pct : d.underlying_target_pct;
  const sVal = sUnit === "premium" ? d.premium_stop_pct : d.underlying_stop_pct;
  const unitOpts: [string, string][] = [["underlying", `% move in ${d.symbol}`], ["premium", "% of the option's price"]];
  const waiting = plan.state === "awaiting_confirmation" || plan.state === "awaiting_answer";
  return (
    <div className="grid gap-4 p-5 2xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]" data-testid="setup">
      <div className="flex min-w-0 flex-col gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2>How this request was read</h2>
          <span className="text-xs dim">version {plan.version_id} · read by {plan.interpreter}</span>
          <span className="flex-1" />
          <button className="btn quiet sm 2xl:hidden" onClick={() => document.getElementById("lab-preview")?.scrollIntoView({ behavior: "smooth", block: "start" })}><Eye />Check it on a real day</button>
          <button className="btn quiet sm" onClick={() => setShowJson((s) => !s)}><Code2 />{showJson ? "Hide" : "Show"} structured configuration</button>
        </div>
        {(plan.unsupported || []).length > 0 && <Callout tone="bad" title="Not represented">{plan.unsupported.map((u: string) => `"${u}"`).join("; ")}. Add a rule below if you can express it, or rephrase that part. Nothing was put in its place.</Callout>}
        {(plan.assumptions || []).length > 0 && (
          <Callout tone="warn" title={`${plan.assumptions.length} thing${plan.assumptions.length === 1 ? " was" : "s were"} assumed because the request did not say`}>
            <ul className="mt-1 flex flex-col gap-0.5">{plan.assumptions.map((a: any, i: number) => <li key={i}><span className="text-fg">{String(a.field).replace(/^study\./, "").replace(/_/g, " ").replace(/\bpct\b/, "%")}{a.value !== null && a.value !== undefined && String(a.value).length < 40 && typeof a.value !== "object" ? `: ${String(a.value)}` : ""}</span> — {a.reason}</li>)}</ul>
          </Callout>
        )}
        {showJson && (
          <div className="grid gap-3 lg:grid-cols-2">
            <div className="card px-4 py-3"><div className="eyebrow mb-2">In plain English</div><div className="flex flex-col gap-1.5">{(plan.items || []).map((x: any, i: number) => <div key={i} className="flex items-baseline gap-2 text-[13px]"><span className="w-24 shrink-0 text-xs dim">{x.label}</span><span className="flex-1"><T>{String(x.value).split("   [")[0]}</T></span><StatusChip status={x.status} /></div>)}</div></div>
            <div className="card px-4 py-3"><div className="eyebrow mb-2">What the engine receives</div><pre className="max-h-[360px] overflow-auto font-mono text-[11.5px] leading-[1.5] text-fg-2">{JSON.stringify(plan.config?.strategy || plan.config?.study || d, null, 1)}</pre></div>
          </div>
        )}
        <Section title={options ? "What to trade" : "What to measure"}>
          <Row label="Kind of research"><div className="inline-flex rounded-lg border border-line-2 bg-bg-2 p-0.5 text-[13px]">
            {([["study", "A market question"], ["options", "An option strategy"]] as const).map(([k, l]) => <button key={k} className={cn("h-7 rounded-md px-3", d.kind === k ? "bg-bg-4 text-fg" : "text-muted")} onClick={() => set({ kind: k, ...(k === "options" && !d.structure ? { structure: "long_call" } : {}), ...(k === "study" && !d.study ? { study: { target: { kind: "pct", value: 1 }, reference: "entry", measure: "touch", horizon_kind: "session", horizon_time: "12:00", horizon_sessions: 1 } } : {}) })}>{l}</button>)}
          </div></Row>
          <Row label="Symbol"><input className="field !w-28 font-mono uppercase" value={d.symbol} onChange={(e) => set({ symbol: e.target.value.toUpperCase().replace(/[^A-Z0-9.]/g, ""), symbol_stated: true })} /></Row>
          {options ? (
            <>
              <Row label="Position"><Sel w={260} value={d.structure || ""} onChange={(v) => set({ structure: v })} options={[["", "choose…"], ...POSITIONS]} /></Row>
              <Row label={<Term k="DTE">Expires in</Term>}>
                <div className="flex flex-wrap items-center gap-2"><Num value={d.dte} onChange={(v) => set({ dte: v === null ? null : Math.max(0, Math.round(v)) })} suffix="days" w={96} placeholder="auto" />
                  {[[0, "same day"], [7, "a week"], [30, "a month"]].map(([n, l]) => <button key={n} className={cn("chip", d.dte === n && "on")} onClick={() => set({ dte: n })}>{l}</button>)}</div>
              </Row>
              <Row label="Strike" hint="Which contract: at the money (closest to the current price), by delta, or a percentage away from the price.">
                <div className="flex items-center gap-2"><Sel w={190} value={d.strike?.method || "auto"} onChange={(v) => set({ strike: v === "auto" ? null : { method: v, value: v === "delta" ? 0.3 : v === "pct_otm" ? 1 : 0 } })} options={[["auto", "let Strata choose"], ["atm", "at the money"], ["delta", "by delta"], ["pct_otm", "% out of the money"]]} />
                  {d.strike && d.strike.method !== "atm" && <Num value={d.strike.value} onChange={(v) => set({ strike: { ...d.strike, value: v ?? 0 } })} suffix={d.strike.method === "pct_otm" ? "%" : "Δ"} />}</div>
              </Row>
              <Row label="Contracts"><Num value={d.quantity} onChange={(v) => set({ quantity: Math.max(1, Math.round(v || 1)) })} w={72} /></Row>
            </>
          ) : (
            <>
              <Row label="Outcome to count"><div className="flex flex-wrap items-center gap-2">
                <Sel w={170} value={st.measure || "touch"} onChange={(v) => setSt({ measure: v })} options={[["touch", "price reaches"], ["close", "the period ends"]]} />
                <Num value={st.target?.kind === "pct" ? st.target.value : null} onChange={(v) => setSt({ target: v === null ? null : { kind: "pct", value: v } })} suffix="%" placeholder="e.g. 1" />
                <span className="text-[13px] dim">{st.target?.kind === "expr" ? `the level ${st.target.expr}` : "above (+) or below (−) the starting price"}</span></div></Row>
              <Row label="Stop (optional)" hint="If set, the study counts which of the two levels is reached first."><Num value={st.stop?.kind === "pct" ? st.stop.value : null} onChange={(v) => setSt({ stop: v === null ? null : { kind: "pct", value: v } })} suffix="%" placeholder="none" /></Row>
              <Row label="Until"><div className="flex items-center gap-2"><Sel w={210} value={st.horizon_kind || "session"} onChange={(v) => setSt({ horizon_kind: v })} options={[["session", "that day's close"], ["time", "a time of day"], ["sessions", "a number of days later"]]} />
                {st.horizon_kind === "time" && <input type="time" className="field !w-28" value={st.horizon_time || "12:00"} onChange={(e) => setSt({ horizon_time: e.target.value })} />}
                {st.horizon_kind === "sessions" && <Num value={st.horizon_sessions} onChange={(v) => setSt({ horizon_sessions: Math.max(1, Math.round(v || 1)) })} suffix="days" w={96} />}</div></Row>
            </>
          )}
        </Section>
        <Section title="When" sub={`${(d.conditions || []).length + (d.sequence || []).length} rule${(d.conditions || []).length + (d.sequence || []).length === 1 ? "" : "s"}`}>
          <Row label={options ? "Enter" : "Start observing"}>
            <div className="flex flex-wrap items-center gap-2">
              <Sel w={220} value={entryKind} onChange={(v) => set({ entry_time: v === "open" ? "open" : v === "close" ? "close" : v === "time" ? "10:00" : null, entry_time_stated: null })} options={[["open", "at the open (09:30 ET)"], ["close", "at the close"], ["time", "at a set time (New York)"], ["signal", "as soon as the rules hold"]]} />
              {entryKind === "time" && <input type="time" className="field !w-28" value={d.entry_time} onChange={(e) => set({ entry_time: e.target.value, entry_time_stated: null })} />}
            </div>
          </Row>
          <Row label="Only on"><div className="flex gap-1.5">{["Mon", "Tue", "Wed", "Thu", "Fri"].map((w) => { const on = !d.weekdays || d.weekdays.includes(w); return <button key={w} className={cn("chip", on && "on")} onClick={() => { const cur = d.weekdays || ["Mon", "Tue", "Wed", "Thu", "Fri"]; const nx = cur.includes(w) ? cur.filter((x: string) => x !== w) : [...cur, w]; set({ weekdays: nx.length === 5 || nx.length === 0 ? null : nx }); }}>{w}</button>; })}</div></Row>
          <div className="mt-2"><div className="mb-2 text-[13px] dim">All of these must be true at that moment</div><ExprList kind="cond" addLabel="Add a rule" items={d.conditions || []} onChange={(x) => set({ conditions: x })} /></div>
          <div className="mt-4 border-t border-line pt-3">
            <div className="mb-2 text-[13px] dim">Things that must happen <span className="text-fg">in order</span> first <Tip content={"Each step is only looked for after the one before it has happened. The entry is the moment the last step happens. Later steps can read earlier ones, e.g. low_since(\"drop\")."}><span className="cursor-help text-faint">ⓘ</span></Tip></div>
            <ExprList kind="stage" addLabel="Add a step" items={d.sequence || []} onChange={(x) => set({ sequence: x, ...(x.length ? { entry_time: null } : {}) })} />
          </div>
          {options && <div className="mt-4 border-t border-line pt-3"><div className="mb-2 text-[13px] dim">Add to the position while it is open</div><ExprList kind="add" addLabel="Add a scale-in rule" items={d.adds || []} onChange={(x) => set({ adds: x })} /></div>}
        </Section>
        {options && (
          <Section title="When to get out">
            <Row label="Take profit at" hint="Be explicit about what the percentage refers to: the underlying's move, or the option's own price.">
              <div className="flex items-center gap-2"><Num value={tVal} onChange={(v) => set(tUnit === "premium" ? { premium_target_pct: v, underlying_target_pct: null } : { underlying_target_pct: v, premium_target_pct: null })} suffix="%" placeholder="none" />
                <Sel w={220} value={tUnit} onChange={(u) => set(u === "premium" ? { premium_target_pct: tVal, underlying_target_pct: null } : { underlying_target_pct: tVal, premium_target_pct: null })} options={unitOpts} /></div>
            </Row>
            <Row label="Stop out at"><div className="flex items-center gap-2"><Num value={sVal} onChange={(v) => set(sUnit === "premium" ? { premium_stop_pct: v, underlying_stop_pct: null } : { underlying_stop_pct: v, premium_stop_pct: null })} suffix="%" placeholder="none" />
              <Sel w={220} value={sUnit} onChange={(u) => set(u === "premium" ? { premium_stop_pct: sVal, underlying_stop_pct: null } : { underlying_stop_pct: sVal, premium_stop_pct: null })} options={unitOpts} /></div></Row>
            <Row label="Close by"><div className="flex flex-wrap items-center gap-2"><input type="time" className="field !w-28" value={d.exit_time === "close" ? "16:00" : d.exit_time || ""} onChange={(e) => set({ exit_time: e.target.value || null, exit_time_stated: null })} /><span className="text-[13px] dim">New York time, or hold for</span>
              <Num value={d.hold_sessions} onChange={(v) => set({ hold_sessions: v === null ? null : Math.max(1, Math.round(v)) })} suffix="days" w={96} placeholder="—" /></div></Row>
            <div className="mt-2"><div className="mb-2 text-[13px] dim">Also close (every lot) when any of these is true</div><ExprList kind="exit" addLabel="Add an exit rule" items={d.exit_conditions || []} onChange={(x) => set({ exit_conditions: x })} /></div>
          </Section>
        )}
        <Section title="Period, prices and costs" open={false} sub="advanced">
          <Row label="Period"><div className="flex items-center gap-2"><input type="date" className="field !w-40" value={d.start || ""} onChange={(e) => set({ start: e.target.value || null })} /><span className="dim">to</span><input type="date" className="field !w-40" value={d.end || ""} onChange={(e) => set({ end: e.target.value || null })} /><span className="text-xs dim">empty = all available history</span></div></Row>
          {options && <Row label="Option prices from"><div><Sel w={300} value={d.mode || "auto"} onChange={(v) => set({ mode: v === "auto" ? null : v })} options={[["auto", "whatever observed prices exist (recommended)"], ["historical", "bid/ask quotes"], ["trades", "trade prints"], ["study", "none: study the underlying only"], ["modeled", "a pricing model (estimates)"]]} />
            <div className="mt-1 text-xs dim">{MODE_HELP[d.mode || "auto"]}</div></div></Row>}
          {options && <Row label="Trading costs"><Sel w={300} value={d.costs?.preset || "default"} onChange={(v) => set({ costs: v === "default" ? null : { ...(d.costs || {}), preset: v } })} options={[["default", "standard: $0.65 per contract"], ["conservative", "conservative: fees, slippage and a delay"], ["none", "none (frictionless)"]]} /></Row>}
        </Section>
        <div className="sticky bottom-0 -mx-1 flex items-center gap-2 rounded-xl border border-line-2 bg-bg-1/95 px-3 py-2.5 backdrop-blur">
          {dirty ? (
            <>
              <button className="btn primary" disabled={busy} onClick={() => onApply(d, true)} data-testid="apply"><Play />Run with these changes</button>
              <button className="btn" onClick={() => setD(plan.draft)}><RotateCcw />Reset</button>
              <span className="text-xs dim">Creates a new version; the current one and its results are kept.</span>
            </>
          ) : waiting ? (
            <>
              <button className="btn primary" disabled={busy || plan.state === "awaiting_answer"} onClick={() => onConfirm(plan.version_id)}><Play />Run it</button>
              <span className="text-xs dim">{plan.state === "awaiting_answer" ? "Answer the question in the conversation first." : "Nothing has run yet."}</span>
            </>
          ) : <span className="text-xs dim">Change anything above to create a new version.</span>}
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-4">
        {(plan.runs || []).length > 0 && !plan.finalized && <div className="card px-4 py-3.5"><Timeline runId={plan.runs[plan.runs.length - 1].run_id} onDone={refresh} title="Progress" /></div>}
        <Preview draft={d} />
      </div>
    </div>
  );
}

function Preview({ draft }: { draft: any }) {
  const [day, setDay] = useState<string>("");
  const m = useMutation({ mutationFn: (dd: string | null) => api.post<any>("/api/ai/preview", { draft, day: dd || null }) });
  const p = m.data;
  useEffect(() => { m.reset(); setDay(""); }, [JSON.stringify([draft.conditions, draft.sequence, draft.entry_time, draft.symbol, draft.kind])]); // eslint-disable-line
  const run = (dd: string | null) => { setDay(dd || ""); m.mutate(dd); };
  return (
    <div className="card overflow-hidden" data-testid="preview" id="lab-preview">
      <div className="px-4 py-3.5">
        <div className="flex items-center gap-2"><Eye className="size-4 text-accent" /><h3 className="!text-[13.5px] !text-fg">Check it on a real day</h3></div>
        <div className="mt-1 text-[13px] dim">See exactly how each rule evaluates through an actual session before running anything. Nothing is traded.</div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className="btn sm" disabled={m.isPending} onClick={() => run(null)}>{m.isPending ? "Evaluating…" : p ? "Pick a day for me" : "Show me a day where it fires"}</button>
          <input type="date" className="field !h-7 !w-36 !text-xs" value={day} onChange={(e) => e.target.value && run(e.target.value)} aria-label="Preview date" />
        </div>
        {p?.samples && (
          <div className="mt-2.5 flex flex-col gap-1.5 text-xs">
            {p.samples.qualifying.length > 0 && <div className="flex flex-wrap items-center gap-1.5"><span className="dim">fires on</span>{p.samples.qualifying.map((x: string) => <button key={x} className={cn("chip !h-6", p.date === x && "on")} onClick={() => run(x)}>{x}</button>)}</div>}
            {p.samples.not_qualifying.length > 0 && <div className="flex flex-wrap items-center gap-1.5"><span className="dim">does not on</span>{p.samples.not_qualifying.map((x: string) => <button key={x} className={cn("chip !h-6", p.date === x && "on")} onClick={() => run(x)}>{x}</button>)}</div>}
            {p.samples.qualifying.length === 0 && <div className="text-warn">The rule did not fire on any session in the last few years of stored history.</div>}
          </div>
        )}
        {m.isError && <div className="mt-2 text-xs text-down">{String((m.error as any)?.message)}</div>}
        {p && p.available === false && <div className="mt-2 text-xs dim">{p.reason}</div>}
        {p?.summary && <div className={cn("mt-3 rounded-lg px-3 py-2 text-[13px]", p.signal ? "bg-up/[0.08] text-fg" : "bg-bg-2 soft")}><span className="num dim">{p.date}</span> · {p.summary}</div>}
      </div>
      {p?.rules && <Rules r={p} />}
    </div>
  );
}
