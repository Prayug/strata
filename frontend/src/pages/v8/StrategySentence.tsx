/* A strategy as a sentence whose bracketed values can be changed in place (V8 brief section 10):
     Buy [SPY] [calls] with [0 days] to expiration at [6:30 AM Pacific].
     Only enter when [yesterday's return was below -1%] [and ...].
     Take profit when [the option gains 30%]. Stop when [the option loses 15%].
     Close any remaining position at [12:45 PM Pacific].
   Changes are collected and said in words before anything runs; running creates a new version (the old one is kept).
   The rule editor and the strategy IDE stay one click away for everything a sentence cannot say. */
import { Plus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

const ZONES: [string, number, string][] = [["ET", 0, "Eastern"], ["CT", -60, "Central"], ["MT", -120, "Mountain"], ["PT", -180, "Pacific"]];
const STRUCT: Record<string, string> = { long_call: "calls", long_put: "puts", short_call: "short calls", short_put: "short puts" };

const toMin = (t: string) => (t === "open" ? 570 : t === "close" ? 960 : Number(t.slice(0, 2)) * 60 + Number(t.slice(3, 5)));
const toHHMM = (m: number) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
function clock(et: string | null | undefined, zone: number) {
  if (!et) return null;
  if (et === "close" && zone === 0) return "the close";
  const m = toMin(et) + zone;
  const h = Math.floor(m / 60), mm = m % 60;
  return `${((h + 11) % 12) + 1}:${String(mm).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
}

/* conditions a sentence can add without expression syntax */
const TEMPLATES: { id: string; label: (v: number) => string; expr: (v: number) => string; def: number; unit: string }[] = [
  { id: "prev_down", label: (v) => `yesterday's return was below −${v}%`, expr: (v) => `previous_session.return_pct <= -${v}`, def: 1, unit: "%" },
  { id: "prev_up", label: (v) => `yesterday's return was above ${v}%`, expr: (v) => `previous_session.return_pct >= ${v}`, def: 1, unit: "%" },
  { id: "gap_down", label: (v) => `it opened at least ${v}% below yesterday's close`, expr: (v) => `gap_pct <= -${v}`, def: 0.5, unit: "%" },
  { id: "gap_up", label: (v) => `it opened at least ${v}% above yesterday's close`, expr: (v) => `gap_pct >= ${v}`, def: 0.5, unit: "%" },
  { id: "open_down", label: (v) => `it is at least ${v}% below its open`, expr: (v) => `from_open_pct <= -${v}`, def: 0.5, unit: "%" },
  { id: "open_up", label: (v) => `it is at least ${v}% above its open`, expr: (v) => `from_open_pct >= ${v}`, def: 0.5, unit: "%" },
  { id: "vix_above", label: (v) => `the VIX closed above ${v} the day before`, expr: (v) => `vix > ${v}`, def: 20, unit: "" },
  { id: "vix_below", label: (v) => `the VIX closed below ${v} the day before`, expr: (v) => `vix < ${v}`, def: 15, unit: "" },
  { id: "above_vwap", label: () => "the price is above VWAP", expr: () => "price > vwap", def: 0, unit: "" },
  { id: "below_vwap", label: () => "the price is below VWAP", expr: () => "price < vwap", def: 0, unit: "" },
];

function Slot({ children, onClick, testid }: { children: React.ReactNode; onClick: () => void; testid?: string }) {
  return <button className="xnum serif" style={{ fontSize: "inherit" }} onClick={onClick} data-testid={testid}>{children}</button>;
}

function Pop({ open, onClose, children }: { open: boolean; onClose: () => void; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose(); };
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    setTimeout(() => window.addEventListener("mousedown", h), 0);
    window.addEventListener("keydown", k);
    return () => { window.removeEventListener("mousedown", h); window.removeEventListener("keydown", k); };
  }, [open, onClose]);
  if (!open) return null;
  return <div ref={ref} className="absolute z-30 mt-2 flex flex-col gap-2 p-3 text-[14px]" style={{ background: "var(--ink-2)", border: "1px solid var(--rule-2)", minWidth: 260, fontFamily: "var(--font-sans)" }}>{children}</div>;
}

function NumberEdit({ value, onSave, suffix, allowNone, step = 1 }: { value: number | null; onSave: (v: number | null) => void; suffix?: string; allowNone?: boolean; step?: number }) {
  const [v, setV] = useState(value == null ? "" : String(value));
  return (
    <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); onSave(v === "" ? null : Number(v)); }}>
      <input autoFocus type="number" step={step} value={v} onChange={(e) => setV(e.target.value)} className="h-8 w-24 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)" }} />
      {suffix && <span className="mute">{suffix}</span>}
      <button className="act primary" type="submit">Set</button>
      {allowNone && <button className="act" type="button" onClick={() => onSave(null)}>None</button>}
    </form>
  );
}

function TimeEdit({ value, onSave, zone, setZone, allowClose }: { value: string | null; onSave: (et: string | null) => void; zone: number; setZone: (z: number) => void; allowClose?: boolean }) {
  const [t, setT] = useState(value ? toHHMM(toMin(value) + zone) : "");
  return (
    <form className="flex flex-col gap-2" onSubmit={(e) => { e.preventDefault(); if (t) onSave(toHHMM(toMin(t) - zone)); }}>
      <div className="flex items-center gap-2">
        <input autoFocus type="time" value={t} onChange={(e) => setT(e.target.value)} className="h-8 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)", colorScheme: "dark" }} />
        <select value={zone} onChange={(e) => { const z = Number(e.target.value); if (t) setT(toHHMM(toMin(t) - zone + z)); setZone(z); }} className="h-8 border bg-transparent px-1" style={{ borderColor: "var(--rule-2)" }}>
          {ZONES.map(([k, off, name]) => <option key={k} value={off} style={{ background: "#191c22" }}>{name}</option>)}
        </select>
        <button className="act primary" type="submit">Set</button>
      </div>
      <div className="flex gap-3 tiny">
        <button type="button" className="link" onClick={() => onSave("open")}>at the open</button>
        {allowClose && <button type="button" className="link" onClick={() => onSave("close")}>at the close</button>}
        {!allowClose && <button type="button" className="link" onClick={() => onSave(null)}>whenever the conditions hold</button>}
      </div>
    </form>
  );
}

/* position sizing in a trader's terms (V9): what each trade buys, and what "risk" means for it */
const SIZE_MODES: [string, string, string][] = [["contracts", "contracts per trade", ""], ["dollars", "dollars of premium per trade", "$"], ["risk_dollars", "dollars at risk per trade", "$"],
  ["risk_pct", "% of the account at risk per trade", "%"], ["equity_pct", "% of the account in premium per trade", "%"]];

export function sizeText(d: any): string {
  const s = d.sizing;
  if (!s || s.mode === "contracts") return `${d.quantity || 1} contract${(d.quantity || 1) === 1 ? "" : "s"}`;
  const v = Number(s.value);
  const base = s.mode === "dollars" ? `$${v.toLocaleString()} of premium` : s.mode === "risk_dollars" ? `$${v.toLocaleString()} at risk` : s.mode === "risk_pct" ? `${v}% of the account at risk` : `${v}% of the account in premium`;
  const extra = [s.capital ? `from a $${Number(s.capital).toLocaleString()} account` : "", ["risk_pct", "equity_pct"].includes(s.mode) ? (s.reinvest ? "reinvesting profits" : "of the starting account") : "",
    s.daily_loss_stop ? `stopping for the day after losing $${Number(s.daily_loss_stop).toLocaleString()}` : ""].filter(Boolean);
  return [base, ...extra].join(", ");
}

function SizeEdit({ d, onSave }: { d: any; onSave: (patch: any) => void }) {
  const s = d.sizing || { mode: "contracts", value: d.quantity || 1 };
  const [mode, setMode] = useState<string>(s.mode);
  const [v, setV] = useState<string>(String(s.mode === "contracts" ? d.quantity || 1 : s.value));
  const [cap, setCap] = useState<string>(s.capital ? String(s.capital) : "");
  const [re, setRe] = useState<boolean>(!!s.reinvest);
  const [stop, setStop] = useState<string>(s.daily_loss_stop ? String(s.daily_loss_stop) : "");
  const unit = SIZE_MODES.find((m) => m[0] === mode)?.[2] || "";
  const save = () => {
    const n = Number(v);
    if (!(n > 0)) return;
    if (mode === "contracts" && !cap && !stop) { onSave({ quantity: Math.max(1, Math.round(n)), sizing: null }); return; }
    onSave({ quantity: mode === "contracts" ? Math.max(1, Math.round(n)) : 1, sizing: { mode, value: mode === "contracts" ? Math.round(n) : n, capital: cap ? Number(cap) : null, reinvest: re, daily_loss_stop: stop ? Number(stop) : null, text: "" } });
  };
  return (
    <form className="flex flex-col gap-2" onSubmit={(e) => { e.preventDefault(); save(); }} data-testid="size-edit">
      <select value={mode} onChange={(e) => setMode(e.target.value)} className="h-8 border bg-transparent px-1" style={{ borderColor: "var(--rule-2)" }}>
        {SIZE_MODES.map(([k, l]) => <option key={k} value={k} style={{ background: "#191c22" }}>{l}</option>)}
      </select>
      <div className="flex items-center gap-2">{unit === "$" && <span className="mute">$</span>}<input autoFocus type="number" step={mode === "contracts" ? 1 : 0.5} value={v} onChange={(e) => setV(e.target.value)} className="h-8 w-28 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)" }} />{unit === "%" && <span className="mute">%</span>}</div>
      <label className="small flex items-center gap-2"><span className="mute w-[130px]">Starting account $</span><input type="number" value={cap} placeholder="100,000" onChange={(e) => setCap(e.target.value)} className="h-8 w-28 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)" }} /></label>
      {["risk_pct", "equity_pct"].includes(mode) && <label className="small flex items-center gap-2"><input type="checkbox" checked={re} onChange={(e) => setRe(e.target.checked)} />Reinvest profits (size from the current account)</label>}
      <label className="small flex items-center gap-2"><span className="mute w-[130px]">Stop for the day after losing $</span><input type="number" value={stop} onChange={(e) => setStop(e.target.value)} className="h-8 w-28 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)" }} /></label>
      <div className="tiny mute">{mode === "risk_dollars" || mode === "risk_pct" ? (d.premium_stop_pct ? `Risk is the loss if the ${d.premium_stop_pct}% stop is hit (premium × ${d.premium_stop_pct}%); a stop can be jumped.` : "With no stop, risk is the whole premium paid.") : mode === "dollars" ? "As many whole contracts as the budget buys; none when one contract costs more." : ""}</div>
      <button className="act primary" type="submit">Set</button>
    </form>
  );
}

export function describeChange(before: any, after: any, zone: number): string[] {
  const out: string[] = [];
  const name = (k: string, v: any) => {
    if (v == null) return "none";
    if (k === "structure") return STRUCT[v] || v;
    if (k === "dte") return `${v} day${v === 1 ? "" : "s"} to expiration`;
    if (k === "entry_time" || k === "exit_time") return v === "open" ? "the open" : v === "close" ? "the close" : `${clock(v, zone)} ${ZONES.find((z) => z[1] === zone)?.[2]}`;
    if (k.endsWith("_pct")) return `${v}%`;
    return String(v);
  };
  const label: Record<string, string> = { symbol: "the ticker", structure: "the position", dte: "the expiration", entry_time: "the entry time", exit_time: "the exit time", premium_target_pct: "the profit target",
    premium_stop_pct: "the stop", underlying_target_pct: "the target on the underlying", underlying_stop_pct: "the stop on the underlying", quantity: "the number of contracts" };
  for (const k of Object.keys(label)) if (k !== "quantity" && JSON.stringify(before?.[k] ?? null) !== JSON.stringify(after?.[k] ?? null)) out.push(`Changed ${label[k]} from ${name(k, before?.[k])} to ${name(k, after?.[k])}.`);
  if (sizeText(before || {}) !== sizeText(after || {})) out.push(`Changed the size of each trade from ${sizeText(before || {})} to ${sizeText(after || {})}.`);
  const bc = (before?.conditions || []).map((c: any) => c.text || c.expr), ac = (after?.conditions || []).map((c: any) => c.text || c.expr);
  for (const c of ac) if (!bc.includes(c)) out.push(`Added: only when ${c}.`);
  for (const c of bc) if (!ac.includes(c)) out.push(`Removed: ${c}.`);
  return out;
}

export function StrategySentence({ draft, symbols, onRun, busy }: { draft: any; symbols: string[]; onRun: (d: any) => void; busy: boolean }) {
  const [d, setD] = useState<any>(draft);
  useEffect(() => setD(draft), [draft]);
  const [open, setOpen] = useState<string | null>(null);
  const stated = String(draft.entry_time_stated || "").toLowerCase();
  // the zone times are shown in: the one the strategy was stated in, else the one last chosen here (remembered)
  const [zone, setZoneRaw] = useState<number>(() => {
    if (/pacific|\bpt\b|pst|pdt/.test(stated)) return -180;
    if (/central|\bct\b|cst|cdt/.test(stated)) return -60;
    if (/mountain|\bmt\b|mst|mdt/.test(stated)) return -120;
    if (/eastern|\bet\b|est|edt/.test(stated)) return 0;
    try { return Number(localStorage.getItem("strata.v8.zone") || 0); } catch { return 0; }
  });
  const setZone = (z: number) => { setZoneRaw(z); try { localStorage.setItem("strata.v8.zone", String(z)); } catch { /* not remembered */ } };
  const zname = ZONES.find((z) => z[1] === zone)?.[2] || "Eastern";
  const set = (patch: any) => { setD((x: any) => ({ ...x, ...patch })); setOpen(null); };
  const changes = useMemo(() => describeChange(draft, d, zone), [draft, d, zone]);
  const [tpl, setTpl] = useState(TEMPLATES[0].id);
  const [tv, setTv] = useState<number>(TEMPLATES[0].def);
  const [join, setJoin] = useState<"and" | "or">("and");
  const t = TEMPLATES.find((x) => x.id === tpl)!;
  const addCond = () => {
    const c = { expr: t.expr(tv), text: t.label(tv), source: "stated" };
    const cs = [...(d.conditions || [])];
    if (join === "or" && cs.length) {
      const last = cs.pop();
      cs.push({ expr: `(${last.expr}) or (${c.expr})`, text: `${last.text || last.expr} or ${c.text}`, source: "stated" });
    } else cs.push(c);
    set({ conditions: cs });
  };
  const sym = d.symbol || "SPY";
  const entry = d.entry_time ? (d.entry_time === "open" ? "the open" : `${clock(d.entry_time, zone)} ${zname}`) : "any time the conditions hold";
  const exit = d.exit_time ? (d.exit_time === "close" ? "the close" : `${clock(d.exit_time, zone)} ${zname}`) : d.hold_minutes ? `${d.hold_minutes} minutes after entry` : d.hold_sessions ? `the end of ${d.hold_sessions} sessions` : "expiration";
  return (
    <div className="serif" style={{ fontSize: 21, lineHeight: 1.75 }} data-testid="strategy-sentence">
      <div>
        {d.structure?.startsWith("short") ? "Sell " : "Buy "}
        <span className="relative inline-block"><Slot testid="slot-symbol" onClick={() => setOpen("symbol")}>{sym}</Slot>
          <Pop open={open === "symbol"} onClose={() => setOpen(null)}>{symbols.map((s) => <button key={s} className="link text-left" onClick={() => set({ symbol: s })}>{s}</button>)}</Pop></span>{" "}
        <span className="relative inline-block"><Slot testid="slot-structure" onClick={() => setOpen("structure")}>{STRUCT[d.structure] || "calls"}</Slot>
          <Pop open={open === "structure"} onClose={() => setOpen(null)}>{Object.entries(STRUCT).map(([k, l]) => <button key={k} className="link text-left" onClick={() => set({ structure: k })}>{l}</button>)}</Pop></span>
        {" with "}
        <span className="relative inline-block"><Slot testid="slot-dte" onClick={() => setOpen("dte")}>{d.dte ?? 0} day{(d.dte ?? 0) === 1 ? "" : "s"}</Slot>
          <Pop open={open === "dte"} onClose={() => setOpen(null)}><NumberEdit value={d.dte ?? 0} onSave={(v) => set({ dte: Math.max(0, Math.round(v ?? 0)) })} suffix="days to expiration" /></Pop></span>
        {" to expiration at "}
        <span className="relative inline-block"><Slot testid="slot-entry" onClick={() => setOpen("entry")}>{entry}</Slot>
          <Pop open={open === "entry"} onClose={() => setOpen(null)}><TimeEdit value={d.entry_time && d.entry_time !== "close" ? d.entry_time : "09:30"} zone={zone} setZone={setZone} onSave={(et) => set({ entry_time: et, entry_time_stated: null })} /></Pop></span>.
      </div>
      <div className="mt-2">
        {(d.conditions || []).length ? "Only enter when " : "Enter on every session"}
        {(d.conditions || []).map((c: any, i: number) => (
          <span key={i}>{i > 0 && " and "}<span className="amber" style={{ textDecoration: "underline dotted", textUnderlineOffset: 4 }}>{c.text || c.expr}</span>
            <button className="mx-1 align-middle" style={{ background: "none", border: 0, color: "var(--faint)", cursor: "pointer" }} aria-label={`Remove ${c.text || c.expr}`} onClick={() => set({ conditions: d.conditions.filter((_: any, k: number) => k !== i) })}><X size={14} /></button></span>
        ))}
        <span className="relative inline-block">
          <button className="link align-middle text-[14px]" style={{ fontFamily: "var(--font-sans)" }} onClick={() => setOpen("add")} data-testid="add-condition"><Plus size={13} className="mr-0.5 inline" />add a condition</button>
          <Pop open={open === "add"} onClose={() => setOpen(null)}>
            {(d.conditions || []).length > 0 && <div className="flex gap-3 small"><label><input type="radio" checked={join === "and"} onChange={() => setJoin("and")} /> and</label><label><input type="radio" checked={join === "or"} onChange={() => setJoin("or")} /> or (with the last one)</label></div>}
            <select value={tpl} onChange={(e) => { setTpl(e.target.value); setTv(TEMPLATES.find((x) => x.id === e.target.value)!.def); }} className="h-8 border bg-transparent px-1" style={{ borderColor: "var(--rule-2)" }}>
              {TEMPLATES.map((x) => <option key={x.id} value={x.id} style={{ background: "#191c22" }}>{x.label(x.def).replace(/[\d.]+/, "…")}</option>)}
            </select>
            {t.unit !== "" || ["vix_above", "vix_below"].includes(t.id) ? <input type="number" step={0.1} value={tv} onChange={(e) => setTv(Number(e.target.value))} className="h-8 w-24 border bg-transparent px-2 num" style={{ borderColor: "var(--rule-2)" }} /> : null}
            <div className="small soft">Only when {t.label(tv)}.</div>
            <button className="act primary" onClick={addCond}>Add</button>
          </Pop>
        </span>.
      </div>
      <div className="mt-2">
        Take profit when <span className="relative inline-block"><Slot testid="slot-target" onClick={() => setOpen("target")}>{d.premium_target_pct != null ? `the option gains ${d.premium_target_pct}%` : "never (no target)"}</Slot>
          <Pop open={open === "target"} onClose={() => setOpen(null)}><NumberEdit value={d.premium_target_pct} allowNone onSave={(v) => set({ premium_target_pct: v })} suffix="% gain on the premium" /></Pop></span>.
        {" "}Stop when <span className="relative inline-block"><Slot testid="slot-stop" onClick={() => setOpen("stop")}>{d.premium_stop_pct != null ? `the option loses ${d.premium_stop_pct}%` : "never (no stop)"}</Slot>
          <Pop open={open === "stop"} onClose={() => setOpen(null)}><NumberEdit value={d.premium_stop_pct} allowNone onSave={(v) => set({ premium_stop_pct: v })} suffix="% loss on the premium" /></Pop></span>.
        {d.underlying_target_pct != null && <> Also exit when {sym} moves {d.underlying_target_pct}% in the trade's favour.</>}
      </div>
      <div className="mt-2">
        Close any remaining position at <span className="relative inline-block"><Slot testid="slot-exit" onClick={() => setOpen("exit")}>{exit}</Slot>
          <Pop open={open === "exit"} onClose={() => setOpen(null)}><TimeEdit value={d.exit_time && d.exit_time !== "close" ? d.exit_time : "15:55"} zone={zone} setZone={setZone} allowClose onSave={(et) => set({ exit_time: et, exit_time_stated: null, hold_minutes: null })} /></Pop></span>.
      </div>
      <div className="mt-2">
        Size each trade at <span className="relative inline-block"><Slot testid="slot-size" onClick={() => setOpen("size")}>{sizeText(d)}</Slot>
          <Pop open={open === "size"} onClose={() => setOpen(null)}><SizeEdit d={d} onSave={(patch) => set(patch)} /></Pop></span>.
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3" style={{ fontFamily: "var(--font-sans)", fontSize: 14, lineHeight: 1.5 }}>
        {changes.length > 0 ? (
          <>
            <div className="w-full small soft" data-testid="pending-changes">{changes.join(" ")} Everything else stays the same.</div>
            <button className="act primary" disabled={busy} onClick={() => onRun(d)} data-testid="run-changes">Run with these changes</button>
            <button className="act" onClick={() => setD(draft)}>Discard</button>
          </>
        ) : <span className="tiny mute">Click any underlined value to change it. Times are shown in {zname} time.</span>}
        <span className="flex-1" />
        <Link className="link tiny" to="/ide">Open in the strategy IDE</Link>
      </div>
    </div>
  );
}
