/* Strategy IDE. One strategy, seen four ways that cannot drift apart because they are the same validated document:
     structured view   the strategy as text (defaults left out), with completion, inline docs and error markers
     outline + inspect the same document as a tree of rules with a form for the selected part and, for a rule,
                       how often it was true historically
     edit by instruction   "replace this stop with an ATR-based stop" applied to the SELECTED section only; the
                           changed lines are marked and anything that cannot be expressed is refused, not guessed
     debugger          the entry rule replayed tick by tick on a real session, with watches and a breakpoint
   The visual builder and the backtester read the same store, so nothing is copied between screens. */
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Blocks, Bug, Check, ChevronLeft, ChevronRight, Code2, FolderOpen, Loader2, Pause, Play, Plus, Save, Sparkles, Trash2, Undo2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { C, EChart } from "@/components/charts/EChart";
import { JobBar, PageHeader, cancelJob } from "@/components/common";
import { ExprInput } from "@/components/research/ExprInput";
import { Button, Dialog, Input, Menu, MenuContent, MenuItem, MenuTrigger, Segmented, Switch, Tip } from "@/components/ui";
import { api, queryClient, useJob } from "@/lib/api";
import { cn, money } from "@/lib/format";
import { pp } from "@/lib/lab";
import { normalizeStrategy, useBuilder, useUi } from "@/lib/store";
import { CodeEditor, type Completion, type Diag, type EditorHandle, type Mark } from "./CodeEditor";
import { format, getAt, lineOf, parentPath, parseRanges, pathAt, setAt, tokens } from "./jsonPath";

const RULE_KEYS = new Set(["expr", "when", "break_when"]);
const HINTS: Record<string, string> = {
  legs: "The option (or stock) legs opened together as one position.",
  "entry.conditions": "The entry rule: a tree of conditions. A group holds when all (and) or any (or) of its children hold.",
  "entry.systems": "Independent entry systems. The first one whose rule holds opens the position, and may carry its own legs (for example calls after a decline, puts after a rally).",
  "entry.sequence": "Events that must happen in order, each on a later tick; the entry fires when the last one happens.",
  "entry.adds": "Scale-in rules: open another lot while the first is open and `when` holds.",
  "exits.profit_target": "Close when the position's P&L reaches this share of the premium.",
  "exits.stop_loss": "Close when the position's loss reaches this share of the premium.",
  "exits.stages": "Staged exits: close part of the position at a profit level, optionally tightening the stop for the rest.",
  "exits.underlying": "Exits measured on the underlying: a move for or against, an ATR target or stop, a trailing threshold.",
  "exits.time_exit": "Close at a clock time (New York).",
  sizing: "How many contracts each entry opens: fixed, a share of equity, by ATR risk, or an expression.",
  hedge: "Hold shares against the position's option delta and rebalance on a schedule.",
};
const EXAMPLES = ["Replace this fixed stop with an ATR-based stop.", "Only allow this entry condition between 9:30 and 11:00 Eastern.", "Add a rule requiring QQQ confirmation.", "Take half off at 50% and move the stop to breakeven.", "Size by ATR, risking 1% of equity."];

/* ----------------------------------------------------------------------------- where the caret is */
interface Ctx { container: string; valuePath: string; inString: boolean; stringStart: number; isKey: boolean; key: string | null }
function contextAt(text: string, caret: number): Ctx {
  type F = { type: "obj" | "arr"; key: string | null; idx: number; expectKey: boolean; path: string };
  const st: F[] = [];
  const child = (f?: F) => (!f ? "" : f.type === "obj" ? (f.path ? `${f.path}.${f.key}` : `${f.key}`) : `${f.path}[${f.idx}]`);
  let i = 0;
  while (i < caret) {
    const c = text[i], top = st[st.length - 1];
    if (c === '"') {
      let j = i + 1;
      while (j < text.length && text[j] !== '"' && text[j] !== "\n") { if (text[j] === "\\") j++; j++; }
      if (caret <= j) {
        const isKey = !!top && top.type === "obj" && top.expectKey;
        return { container: top?.path ?? "", valuePath: isKey ? top?.path ?? "" : child(top), inString: true, stringStart: i + 1, isKey, key: isKey || top?.type !== "obj" ? null : top.key };
      }
      if (top && top.type === "obj" && top.expectKey) top.key = text.slice(i + 1, j);
      i = j + 1;
      continue;
    }
    if (c === "{") st.push({ type: "obj", key: null, idx: 0, expectKey: true, path: child(top) });
    else if (c === "[") st.push({ type: "arr", key: null, idx: 0, expectKey: false, path: child(top) });
    else if (c === "}" || c === "]") st.pop();
    else if (c === ":" && top) top.expectKey = false;
    else if (c === "," && top) { if (top.type === "obj") { top.expectKey = true; top.key = null; } else top.idx++; }
    i++;
  }
  const top = st[st.length - 1];
  return { container: top?.path ?? "", valuePath: top ? (top.type === "obj" && top.expectKey ? top.path : child(top)) : "", inString: false, stringStart: -1, isKey: false, key: top?.type === "obj" ? top.key : null };
}

function deref(schema: any, node: any): any {
  for (let k = 0; node && k < 8; k++) {
    if (node.$ref) node = schema.$defs?.[String(node.$ref).split("/").pop()!];
    else if (node.anyOf) node = node.anyOf.map((n: any) => deref(schema, n)).find((n: any) => n && n.type !== "null") || null;
    else if (node.allOf) node = node.allOf[0];
    else break;
  }
  return node;
}
function schemaAt(schema: any, path: string): any {
  let node = schema;
  for (const t of tokens(path)) {
    node = deref(schema, node);
    node = typeof t === "number" ? node?.items : node?.properties?.[t];
    if (!node) return null;
  }
  return { raw: node, node: deref(schema, node) };
}
const typeWord = (raw: any, node: any) => (node?.enum ? node.enum.join(" | ") : raw?.anyOf ? raw.anyOf.map((x: any) => x.type || String(x.$ref || "").split("/").pop()).join(" | ") : node?.type || node?.title || "");

/* ----------------------------------------------------------------------------- outline */
interface Item { path: string; label: string; detail?: string; depth: number; head?: boolean; bad?: boolean }
function condLine(n: any): string {
  if (!n) return "";
  if (n.kind === "cond") return n.type === "expr" ? String(n.params?.expr || "(empty rule)") : `${String(n.type || "").replace(/_/g, " ")} ${Object.values(n.params || {}).map((v) => (Array.isArray(v) ? v.join("/") : v)).join(" ")}`.trim();
  return `${n.negate ? "not " : ""}${(n.op || "and") === "or" ? "any of" : "all of"} (${(n.children || []).length})`;
}
function outline(doc: any): Item[] {
  const out: Item[] = [];
  const tree = (n: any, path: string, depth: number) => {
    if (!n) return;
    out.push({ path, label: condLine(n), depth });
    (n.children || []).forEach((c: any, i: number) => tree(c, `${path}.children[${i}]`, depth + 1));
  };
  out.push({ path: "legs", label: "Legs", depth: 0, head: true });
  (doc.legs || []).forEach((l: any, i: number) => out.push({ path: `legs[${i}]`, depth: 1, label: l.kind === "stock" ? `${l.side} ${l.quantity ?? 1} shares` : `${l.side} ${l.quantity ?? 1} ${l.right === "P" ? "put" : "call"}`,
    detail: l.kind === "stock" ? "" : `${l.dte ?? 0}DTE · ${l.strike?.method ?? "delta"} ${l.strike?.value ?? ""}` }));
  out.push({ path: "entry", label: "Entry", depth: 0, head: true });
  tree(doc.entry?.conditions, "entry.conditions", 1);
  (doc.entry?.systems || []).forEach((s: any, i: number) => { out.push({ path: `entry.systems[${i}]`, depth: 1, label: `system: ${s.name || i + 1}`, detail: s.legs ? "own legs" : "" }); tree(s.conditions, `entry.systems[${i}].conditions`, 2); });
  (doc.entry?.sequence || []).forEach((s: any, i: number) => out.push({ path: `entry.sequence[${i}]`, depth: 1, label: `step ${i + 1}: ${s.name || ""}`, detail: s.when }));
  (doc.entry?.adds || []).forEach((s: any, i: number) => out.push({ path: `entry.adds[${i}]`, depth: 1, label: `scale-in ${i + 1}`, detail: s.when }));
  out.push({ path: "exits", label: "Exits", depth: 0, head: true });
  for (const [k, v] of Object.entries<any>(doc.exits || {})) {
    if (k === "stages") (v || []).forEach((s: any, i: number) => out.push({ path: `exits.stages[${i}]`, depth: 1, label: `stage: close ${s.close_pct}%`, detail: s.profit_pct !== undefined && s.profit_pct !== null ? `at +${s.profit_pct}%` : s.when }));
    else if (k === "custom" && v) tree(v, "exits.custom", 1);
    else if (v && typeof v === "object" && !Array.isArray(v)) out.push({ path: `exits.${k}`, depth: 1, label: k.replace(/_/g, " ") + (v.enabled === false ? " (off)" : ""), detail: Object.entries(v).filter(([kk, vv]) => kk !== "enabled" && vv !== null && typeof vv !== "object").map(([kk, vv]) => `${kk.replace(/_/g, " ")} ${vv}`).join(" · ") });
  }
  out.push({ path: "sizing", label: "Sizing", depth: 0, head: true, detail: doc.sizing ? `${doc.sizing.method ?? "fixed"} ${doc.sizing.contracts ?? doc.sizing.pct ?? doc.sizing.expr ?? ""}` : "1 contract" });
  if (doc.hedge) out.push({ path: "hedge", label: "Delta hedge", depth: 0, head: true, detail: doc.hedge.enabled === false ? "off" : `every ${doc.hedge.rebalance_minutes ?? 30} min` });
  return out;
}

/* ----------------------------------------------------------------------------- rule preview */
function RulePreview({ symbol, expr, at }: { symbol: string; expr: string; at: string }) {
  const q = useQuery({ queryKey: ["rule-preview", symbol, expr, at], queryFn: () => api.post<any>("/api/ide/preview-rule", { symbol, expr, at }), enabled: !!expr.trim() && !!symbol, staleTime: 60_000, retry: false });
  const r = q.data;
  if (!expr.trim()) return null;
  if (q.isLoading) return <div className="text-2xs text-muted">checking history…</div>;
  if (!r) return null;
  if (!r.ok) return <div className="text-2xs text-down">{r.error}</div>;
  if (r.position) return <div className="text-2xs text-muted">{r.note}</div>;
  if (r.kind === "condition") return (
    <div className="text-2xs text-fg-2" data-testid="rule-preview">
      True on <b className="num text-fg">{r.true.toLocaleString()}</b> of {r.evaluable.toLocaleString()} sessions ({pp(r.share)}) read at {r.at === "open" ? "the open" : r.at === "close" ? "the close" : `${r.at} ET`}{r.moved_tick ? " (it needs intraday bars, so it is read during the session)" : ""}.
      {r.evaluable < r.sessions && <span className="text-muted"> {(r.sessions - r.evaluable).toLocaleString()} sessions could not be evaluated and are left out.</span>}
      {(r.recent || []).length > 0 && <span className="text-muted"> Most recent: {r.recent.join(", ")}.</span>}
    </div>
  );
  if (r.kind === "value") return <div className="text-2xs text-fg-2">A value, not a condition. Latest {Number(r.latest).toFixed(3)} ({r.latest_date}); median {Number(r.median).toFixed(3)}, 5th to 95th percentile {Number(r.p05).toFixed(3)} to {Number(r.p95).toFixed(3)}.</div>;
  return <div className="text-2xs text-muted">{r.note}</div>;
}

/* ----------------------------------------------------------------------------- inspect: a form for the selected part */
function Inspect({ doc, sel, symbol, onSet, onSelect, leafTypes }: { doc: any; sel: string; symbol: string; onSet: (path: string, v: any) => void; onSelect: (p: string) => void; leafTypes: any }) {
  let path = sel;
  let node = path ? getAt(doc, path) : doc;
  if (node === null || typeof node !== "object") { path = parentPath(path); node = path ? getAt(doc, path) : doc; }
  if (node === null || node === undefined) return <div className="p-3 text-xs text-muted">Select a part of the strategy in the outline or in the text.</div>;
  const at = /exits|adds/.test(path) ? "open" : "open";
  const rows = Array.isArray(node) ? node.map((v: any, i: number) => [`[${i}]`, v, `${path}[${i}]`] as const) : Object.entries(node).map(([k, v]) => [k, v, path ? `${path}.${k}` : k] as const);
  const isGroup = !Array.isArray(node) && Array.isArray(node.children);
  const isCond = !Array.isArray(node) && node.kind === "cond";
  const inList = /\[\d+\]$/.test(path);
  return (
    <div className="flex flex-col gap-2 p-2.5 text-xs" data-testid="inspect">
      <div className="flex items-center gap-1.5">
        {path && <Button size="iconSm" variant="ghost" onClick={() => onSelect(parentPath(path))} title="Up one level"><ChevronLeft /></Button>}
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-accent-2" title={path}>{path || "strategy"}</span>
        {inList && <Button size="sm" variant="danger" onClick={() => { onSet(path, undefined); onSelect(parentPath(path)); }}><Trash2 />Remove</Button>}
      </div>
      {HINTS[path.replace(/\[\d+\]/g, "")] && <div className="text-2xs text-muted">{HINTS[path.replace(/\[\d+\]/g, "")]}</div>}
      {isCond && node.type !== "expr" && leafTypes?.[node.type] && <div className="text-2xs text-muted">{leafTypes[node.type].label} ({leafTypes[node.type].category})</div>}
      <div className="flex flex-col gap-1.5">
        {rows.map(([k, v, p]) => {
          if (v !== null && typeof v === "object") return (
            <button key={p} onClick={() => onSelect(p)} className="flex items-center gap-2 rounded-sm border border-line bg-bg-2 px-2 py-1.5 text-left hover:border-faint">
              <span className="w-28 shrink-0 truncate text-muted">{k}</span><span className="min-w-0 flex-1 truncate text-fg-2">{Array.isArray(v) ? `${v.length} item${v.length !== 1 ? "s" : ""}` : v.kind ? condLine(v) : Object.keys(v).slice(0, 4).join(", ")}</span><ChevronRight className="size-3 text-faint" />
            </button>
          );
          const rule = RULE_KEYS.has(k) && typeof v === "string";
          return (
            <div key={p} className={cn("flex gap-2", rule ? "flex-col" : "items-center")}>
              <span className="w-28 shrink-0 truncate text-muted" title={k}>{k}</span>
              {rule ? (<><ExprInput value={v} onChange={(x) => onSet(p, x)} /><RulePreview symbol={symbol} expr={v} at={at} /></>)
                : typeof v === "boolean" ? <Switch checked={v} onChange={(x) => onSet(p, x)} />
                : typeof v === "number" ? <input type="number" className="field field-sm num w-32" value={v} step="any" onChange={(e) => e.target.value !== "" && onSet(p, Number(e.target.value))} />
                : <Input sz="sm" className="flex-1" value={v === null ? "" : String(v)} placeholder={v === null ? "not set" : ""} onChange={(e) => onSet(p, e.target.value === "" && v === null ? null : e.target.value)} />}
            </div>
          );
        })}
      </div>
      {isGroup && <Button size="sm" variant="outline" className="self-start" onClick={() => { onSet(`${path}.children[${node.children.length}]`, { kind: "cond", type: "expr", params: { expr: "previous_session.return_pct <= -1" } }); onSelect(`${path}.children[${node.children.length}]`); }}><Plus />Add a rule to this group</Button>}
      {path === "entry" && !node.adds && <Button size="sm" variant="outline" className="self-start" onClick={() => onSet("entry.adds", [{ when: "price <= entry.underlying_price * 0.995", text: "add after a further 0.5% decline" }])}><Plus />Add a scale-in rule</Button>}
      {path === "exits" && !node.stages && <Button size="sm" variant="outline" className="self-start" onClick={() => onSet("exits.stages", [{ name: "half at +50%", profit_pct: 50, close_pct: 50, then_stop: "breakeven" }])}><Plus />Add a staged exit</Button>}
    </div>
  );
}

/* ----------------------------------------------------------------------------- debugger */
function Debugger({ doc, valid, symbol, breakWhen, setBreakWhen }: { doc: any; valid: boolean; symbol: string; breakWhen: string; setBreakWhen: (s: string) => void }) {
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 60_000 });
  const row = (cov.data || []).find((r) => r.symbol === symbol);
  const lastDay = row?.intraday?.["1m"]?.last || row?.intraday?.["5m"]?.last || row?.intraday?.["60m"]?.last || row?.daily?.last || "";
  const [day, setDay] = useState("");
  const [watch, setWatch] = useState<string[]>(["price", "from_open_pct"]);
  const [nw, setNw] = useState("");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { if (!day && lastDay) setDay(lastDay); }, [lastDay, day]);
  const tl: any[] = res?.timeline || [];
  useEffect(() => {
    if (!playing) return;
    const h = setInterval(() => setCur((c) => { if (c >= tl.length - 1) { setPlaying(false); return c; } return c + 1; }), 140);
    return () => clearInterval(h);
  }, [playing, tl.length]);
  const run = async () => {
    setBusy(true); setErr(null); setPlaying(false);
    try {
      const r = await api.post<any>("/api/ide/debug", { strategy: doc, day, watch, break_when: breakWhen.trim() || null });
      setRes(r);
      const t: any[] = r.timeline || [];
      const bi = r.breakpoint?.hit ? t.findIndex((x) => x.i === r.breakpoint.index) : -1;
      const si = t.findIndex((x) => x.passed);
      setCur(bi >= 0 ? bi : si >= 0 ? si : 0);
    } catch (e: any) { setErr(e.message); setRes(null); } finally { setBusy(false); }
  };
  const tick = tl[cur];
  const sigIdx = tl.findIndex((x) => x.passed);
  const brkIdx = res?.breakpoint?.hit ? tl.findIndex((x) => x.i === res.breakpoint.index) : -1;
  const option = useMemo(() => !tl.length ? null : ({
    grid: { left: 46, right: 10, top: 10, bottom: 22 }, tooltip: { trigger: "axis", valueFormatter: (v: number) => v?.toFixed(2) },
    xAxis: { type: "category", data: tl.map((t) => t.time), boundaryGap: false }, yAxis: { type: "value", scale: true },
    series: [{ type: "line", data: tl.map((t) => t.price), symbol: "none", lineStyle: { width: 1.4, color: C.fg }, areaStyle: { color: C.fg, opacity: 0.04 },
      markLine: { symbol: "none", silent: true, animation: false, label: { show: false }, lineStyle: { color: C.accent, width: 1 }, data: [{ xAxis: tl[cur]?.time }] },
      markPoint: { symbolSize: 26, label: { fontSize: 9, color: "#111" }, data: [
        ...(sigIdx >= 0 ? [{ name: "entry", coord: [tl[sigIdx].time, tl[sigIdx].price], value: "IN", itemStyle: { color: C.up } }] : []),
        ...(brkIdx >= 0 ? [{ name: "break", coord: [tl[brkIdx].time, tl[brkIdx].price], value: "BP", itemStyle: { color: C.accent } }] : []),
      ] } }],
  }), [tl, cur, sigIdx, brkIdx]);
  return (
    <div className="flex flex-col gap-2 p-2.5 text-xs" data-testid="debugger">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1"><span className="label">Session</span><input type="date" className="field field-sm w-[136px]" value={day} max={row?.daily?.last} onChange={(e) => setDay(e.target.value)} aria-label="Session to replay" /></label>
        <Button variant="primary" size="sm" onClick={run} disabled={busy || !valid || !day} data-testid="debug-run">{busy ? <Loader2 className="animate-spin" /> : <Bug />}Replay the entry rule</Button>
        {!valid && <span className="text-warn">Fix the errors first</span>}
      </div>
      <label className="flex flex-col gap-1"><span className="label">Breakpoint: stop at the first tick where this holds</span><ExprInput value={breakWhen} onChange={setBreakWhen} placeholder="e.g. from_open_pct < -0.3   (or click the gutter beside a rule)" /></label>
      <div>
        <div className="label mb-1">Watch</div>
        <div className="flex flex-wrap items-center gap-1">
          {watch.map((w) => <span key={w} className="inline-flex items-center gap-1 rounded-sm border border-line-2 bg-bg-2 px-1.5 py-0.5 font-mono text-[11px]">{w}<button onClick={() => setWatch((s) => s.filter((x) => x !== w))} aria-label={`Stop watching ${w}`}><X className="size-3 text-faint hover:text-fg" /></button></span>)}
          <input className="field field-sm w-40 font-mono" placeholder="add: vwap, rsi(14, &quot;5m&quot;)…" value={nw} onChange={(e) => setNw(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && nw.trim()) { setWatch((s) => [...new Set([...s, nw.trim()])].slice(0, 8)); setNw(""); } }} aria-label="Add a watch expression" />
        </div>
      </div>
      {err && <div className="rounded-sm border border-down/40 bg-down/[0.06] px-2 py-1.5">{err}</div>}
      {res && !tl.length && <div className="rounded-sm border border-warn/40 bg-warn/[0.06] px-2 py-1.5">{res.error || res.summary || (res.trading_day === false ? "That date is not a trading session." : "No intraday bars are stored for this session, so there is nothing to replay. Pick a session with intraday data (see the Data page).")}</div>}
      {tl.length > 0 && tick && (
        <>
          <div className="text-fg-2">{res.summary} <span className="text-muted">{res.evaluations} evaluations on {res.resolution} bars{res.sampled_every > 1 ? `, every ${res.sampled_every}th shown` : ""}.</span></div>
          {res.breakpoint && <div className={cn("rounded-sm border px-2 py-1", res.breakpoint.hit ? "border-accent/50 bg-accent/[0.07]" : "border-line-2 bg-bg-2 text-muted")}>{res.breakpoint.hit ? <>Breakpoint hit at <b className="num">{res.breakpoint.time} ET</b>: <span className="font-mono">{res.breakpoint.expr}</span></> : res.breakpoint.note}</div>}
          {option && <EChart option={option} height={150} onClick={(p) => typeof p.dataIndex === "number" && setCur(p.dataIndex)} />}
          <div className="flex items-center gap-1.5">
            <Button size="iconSm" variant="ghost" onClick={() => setCur((c) => Math.max(0, c - 1))} aria-label="Previous tick"><ChevronLeft /></Button>
            <Button size="iconSm" variant="ghost" onClick={() => { if (cur >= tl.length - 1) setCur(0); setPlaying((p) => !p); }} aria-label={playing ? "Pause" : "Play the session"}>{playing ? <Pause /> : <Play />}</Button>
            <Button size="iconSm" variant="ghost" onClick={() => setCur((c) => Math.min(tl.length - 1, c + 1))} aria-label="Next tick"><ChevronRight /></Button>
            <input type="range" min={0} max={tl.length - 1} value={cur} onChange={(e) => setCur(Number(e.target.value))} className="flex-1 accent-[var(--color-accent)]" aria-label="Scrub through the session" />
            <span className="num w-[92px] text-right text-fg">{tick.time} ET · {tick.price.toFixed(2)}</span>
          </div>
          <div className="rounded-sm border border-line bg-bg-0" data-testid="debug-tick">
            <div className="flex items-center gap-2 border-b border-line px-2 py-1"><span className="label">Every condition at {tick.time}</span><span className="flex-1" />
              <span className={tick.passed ? "text-up" : tick.na ? "text-warn" : "text-muted"}>{tick.passed ? "● entry rule holds" : tick.na ? "? not evaluable" : "○ entry rule does not hold"}</span></div>
            {tick.items.map((it: any, i: number) => (
              <div key={i} className="flex items-start gap-2 border-b border-line/50 px-2 py-1 last:border-0">
                <span className={cn("mt-px w-3 shrink-0 text-center", it.passed === true ? "text-up" : it.passed === false ? "text-down" : "text-warn")} title={it.passed === true ? "holds" : it.passed === false ? "does not hold" : "cannot be evaluated"}>{it.passed === true ? "✓" : it.passed === false ? "✕" : "?"}</span>
                <span className="min-w-0 break-words font-mono text-[11px] text-fg-2">{it.text}</span>
              </div>
            ))}
            {Object.keys(tick.watch || {}).length > 0 && (
              <div className="grid grid-cols-2 gap-x-3 border-t border-line px-2 py-1.5">
                {Object.entries<any>(tick.watch).map(([k, v]) => <div key={k} className="flex justify-between gap-2"><span className="truncate font-mono text-[11px] text-muted">{k}</span><span className="num">{v === null ? "n/a" : typeof v === "boolean" ? String(v) : Number(v).toFixed(4)}</span></div>)}
              </div>
            )}
          </div>
          <table className="tbl"><thead><tr><th>Rule</th><th className="r">Held on</th><th className="r">First true</th></tr></thead>
            <tbody>{(res.rules || []).map((r: any, i: number) => <tr key={i}><td className="max-w-[230px] truncate font-mono text-[11px]" title={r.label}>{r.label}</td><td className="r num">{r.held_ticks} / {r.ticks}</td><td className="r num">{r.first_true || "never"}</td></tr>)}</tbody></table>
        </>
      )}
      {!res && !err && <div className="text-muted">The replay evaluates the entry rule on every 1-minute tick of one real session and keeps each condition's state and values, so you can see exactly why it did or did not fire. It simulates no fills.</div>}
    </div>
  );
}

/* ----------------------------------------------------------------------------- backtest */
function Backtest({ valid, strategyId }: { valid: boolean; strategyId: string | null }) {
  const nav = useNavigate();
  const b = useBuilder();
  const [jobId, setJobId] = useState<string | null>(null);
  const [rid, setRid] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const job = useJob(jobId, (s) => { setJobId(null); if (s.status === "done") { setRid(s.id); useUi.getState().set({ lastRunId: s.id }); } else if (s.status === "error") toast.error(s.error || "The backtest failed"); });
  const run = useQuery({ queryKey: ["run", rid], queryFn: () => api.get<any>(`/api/runs/${rid}`), enabled: !!rid });
  const start = async () => {
    setBusy(true);
    try {
      const r = await api.post<{ id: string; cached: boolean }>("/api/backtests", { config: useBuilder.getState().backtestConfig(), strategy_id: strategyId });
      if (r.cached) { setRid(r.id); toast.message("This exact configuration was already run; showing that result."); } else { setRid(null); setJobId(r.id); }
    } catch (e: any) { toast.error(e.message); } finally { setBusy(false); }
  };
  const s = run.data?.summary;
  const mode = (b.pricing as any)?.mode || "historical";
  return (
    <div className="flex flex-col gap-2 p-2.5 text-xs" data-testid="ide-backtest">
      <div className="grid grid-cols-2 gap-2">
        <label className="flex flex-col gap-1"><span className="label">From</span><input type="date" className="field field-sm" value={b.start} onChange={(e) => b.patch({ start: e.target.value })} /></label>
        <label className="flex flex-col gap-1"><span className="label">To</span><input type="date" className="field field-sm" value={b.end} onChange={(e) => b.patch({ end: e.target.value })} /></label>
      </div>
      <div><div className="label mb-1">Option prices</div>
        <Segmented value={mode} onChange={(m) => b.patch({ pricing: { ...(b.pricing as any), mode: m } as any })} options={[{ value: "historical", label: "Observed quotes", title: "Stored historical bid/ask quotes" }, { value: "trades", label: "Observed trades", title: "Stored option trade bars; a contract with no recent trade is not filled" }, { value: "modeled", label: "Modeled", title: "Theoretical premiums. Clearly labelled, never mixed into an observed-price backtest" }]} />
      </div>
      <Button variant="primary" onClick={start} disabled={!valid || busy || !!jobId} data-testid="ide-run">{busy || jobId ? <Loader2 className="animate-spin" /> : <Play />}Run the backtest</Button>
      {!valid && <span className="text-warn">Fix the errors first.</span>}
      <JobBar job={job} onCancel={() => jobId && cancelJob(jobId)} />
      {run.data?.status === "error" && <div className="rounded-sm border border-down/40 bg-down/[0.06] px-2 py-1.5">{run.data.error}</div>}
      {s && run.data?.status === "done" && (
        <div className="rounded-sm border border-line bg-bg-0">
          <div className="grid grid-cols-3 divide-x divide-line border-b border-line">
            {[["Net P&L", money(s.net_pnl, 0, true), s.net_pnl > 0 ? "text-up" : s.net_pnl < 0 ? "text-down" : ""], ["Trades", String(Math.round(s.trades ?? 0)), ""], ["Win rate", pp(s.win_rate, 0), ""]].map(([l, v, t]) => <div key={l} className="px-2 py-1.5"><div className="label">{l}</div><div className={cn("num text-[15px]", t)}>{v}</div></div>)}
          </div>
          <div className="grid grid-cols-3 divide-x divide-line">
            {[["Expectancy", money(s.expectancy, 0, true)], ["Profit factor", s.profit_factor?.toFixed(2) ?? "—"], ["Max drawdown", pp(s.max_drawdown)]].map(([l, v]) => <div key={l} className="px-2 py-1.5"><div className="label">{l}</div><div className="num text-[15px]">{v}</div></div>)}
          </div>
          {(s.trades ?? 0) < 30 && <div className="border-t border-line px-2 py-1.5 text-warn">{Math.round(s.trades ?? 0)} trades is too few to tell skill from luck. Treat these numbers as a record of what happened, not as an estimate.</div>}
          <div className="flex gap-2 border-t border-line px-2 py-1.5"><Button size="sm" onClick={() => nav(`/results/${rid}`)}>Open the full result</Button><Button size="sm" variant="ghost" onClick={() => nav("/forward")}>Freeze for forward validation</Button></div>
        </div>
      )}
      <div className="text-muted">Capital, costs and fill assumptions are the ones set in the visual builder. Observed-price modes never substitute a modeled premium: a session without stored option prices is reported as missing.</div>
    </div>
  );
}

/* ----------------------------------------------------------------------------- the page */
export default function Ide() {
  const nav = useNavigate();
  const [qs] = useSearchParams();
  const b = useBuilder();
  const ref = useQuery({ queryKey: ["ide-reference"], queryFn: () => api.get<any>("/api/ide/reference"), staleTime: 60_000 });
  const saved = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies"), staleTime: 10_000 });
  const editor = useRef<EditorHandle>(null);
  const [text, setText] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [sel, setSel] = useState("");
  const [v, setV] = useState<any>(null);
  const [changed, setChanged] = useState<string[]>([]);
  const [notes, setNotes] = useState<{ tone: "info" | "warn"; text: string }[]>([]);
  const [undo, setUndo] = useState<string | null>(null);
  const [instr, setInstr] = useState("");
  const [applying, setApplying] = useState(false);
  const [tab, setTab] = useState<"inspect" | "debug" | "backtest" | "reference">("inspect");
  const [breakWhen, setBreakWhen] = useState("");
  const [breakLine, setBreakLine] = useState<number | null>(null);
  const [describe, setDescribe] = useState(false);
  const [desc, setDesc] = useState("");
  const [drafting, setDrafting] = useState(false);
  const [refQ, setRefQ] = useState("");
  const [caretDoc, setCaretDoc] = useState<string>("");
  const good = useRef<any>(null);

  const parsed = useMemo(() => parseRanges(text), [text]);
  if (parsed.ok) good.current = parsed.value;
  const doc = good.current || {};
  const ranges = parsed.ok ? parsed.ranges : null;
  const symbol = String(doc.symbol || "SPY").toUpperCase();

  // ----- open: the store's strategy, shown without its defaults
  const open = useCallback(async (full: any, note?: string) => {
    try {
      const r = await api.post<{ strategy: any }>("/api/ide/compact", { strategy: full });
      setText(format(r.strategy)); setSel(""); setChanged([]); setUndo(null); setNotes(note ? [{ tone: "info", text: note }] : []);
    } catch (e: any) { setText(format(full)); setNotes([{ tone: "warn", text: `This strategy does not validate, so it is shown in full: ${e.message}` }]); }
    setLoaded(true);
  }, []);
  useEffect(() => { if (!loaded) open(useBuilder.getState().strategy); }, [loaded, open]);
  // arriving from the ML page with a saved signal: add it as an entry rule and say so
  useEffect(() => {
    const sig = qs.get("signal");
    if (!sig || !loaded || !parsed.ok || JSON.stringify(parsed.value).includes(`signal(\\"${sig}\\")`)) return;
    const kids = parsed.value.entry?.conditions?.children || [];
    const next = setAt(parsed.value, `entry.conditions.children[${kids.length}]`, { kind: "cond", type: "expr", params: { expr: `signal("${sig}")`, label: `model signal ${sig}` } });
    setText(format(next)); setChanged([`entry.conditions.children[${kids.length}]`]);
    setNotes([{ tone: "info", text: `Added the saved model signal "${sig}" as an entry rule. It is true only on the dates the walk-forward model flagged; backtest it on observed option prices before reading anything into it.` }]);
  }, [qs, loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // ----- validate (debounced) and keep the shared store in step
  useEffect(() => {
    if (!loaded || !parsed.ok) return;
    const h = setTimeout(async () => {
      try {
        const r = await api.post<any>("/api/ide/validate", { strategy: parsed.value });
        setV(r);
        if (r.ok && r.full) useBuilder.setState({ strategy: normalizeStrategy(r.full), dirty: true });
      } catch (e: any) { setV({ ok: false, errors: [{ path: "", message: e.message }], expressions: [] }); }
    }, 350);
    return () => clearTimeout(h);
  }, [text, loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  const diags: Diag[] = useMemo(() => {
    if (!parsed.ok) return [{ offset: parsed.offset, message: parsed.error }];
    return (v?.errors || []).map((e: any) => {
      let p = String(e.path || "");
      while (p && !parsed.ranges.has(p)) p = parentPath(p);
      return { offset: parsed.ranges.get(p)?.start ?? 0, message: `${e.path ? `${e.path}: ` : ""}${e.message}` };
    });
  }, [parsed, v]);
  const valid = parsed.ok && !!v?.ok && diags.length === 0;
  const marks: Mark[] = useMemo(() => {
    if (!ranges) return [];
    const out: Mark[] = [];
    const r = sel ? ranges.get(sel) : null;
    if (r) out.push({ from: r.start, to: r.end, kind: "select" });
    for (const p of changed) { let q = p; while (q && !ranges.has(q)) q = parentPath(q); const x = ranges.get(q); if (x) out.push({ from: x.start, to: x.end, kind: "changed" }); }
    return out;
  }, [ranges, sel, changed]);
  const items = useMemo(() => outline(doc), [doc]); // eslint-disable-line react-hooks/exhaustive-deps
  const active = useMemo(() => items.filter((i) => sel === i.path || sel.startsWith(`${i.path}.`) || sel.startsWith(`${i.path}[`)).sort((a, c) => c.path.length - a.path.length)[0]?.path, [items, sel]);
  const badPaths = useMemo(() => new Set<string>((v?.errors || []).map((e: any) => String(e.path))), [v]);

  const select = useCallback((p: string, reveal = true) => {
    setSel(p);
    if (reveal && ranges?.get(p)) editor.current?.reveal(ranges.get(p)!.start);
  }, [ranges]);
  const setPath = useCallback((p: string, value: any) => { if (!parsed.ok) { toast.error("Fix the text first: it does not parse."); return; } setChanged([]); setText(format(setAt(parsed.value, p, value))); }, [parsed]);

  // ----- completion and inline documentation
  const fnSet = useMemo(() => new Set<string>((ref.data?.names || []).filter((n: any) => n.function).map((n: any) => n.name)), [ref.data]);
  const nameSet = useMemo(() => new Set<string>([...(ref.data?.names || []).map((n: any) => n.name), ...Object.keys(ref.data?.fields || {}), ...(ref.data?.symbols || [])]), [ref.data]);
  const complete = useCallback((t: string, caret: number): Completion | null => {
    const R = ref.data;
    if (!R) return null;
    const cx = contextAt(t, caret);
    if (!cx.inString) return null;
    const typed = t.slice(cx.stringStart, caret);
    if (cx.isKey) {
      let keys: { label: string; detail?: string }[] = Object.entries<any>(schemaAt(R.schema, cx.container)?.node?.properties || {}).map(([k, x]) => ({ label: k, detail: typeWord(x, deref(R.schema, x)) }));
      if (/\.params$/.test(cx.container)) keys = (R.leaf_types?.[getAt(good.current, `${parentPath(cx.container)}.type`)]?.params || []).map((p: any) => ({ label: p.name, detail: p.label }));
      const have = new Set(Object.keys(getAt(good.current, cx.container) || {}));
      const out = keys.filter((k) => k.label.startsWith(typed) && k.label !== typed && !have.has(k.label)).slice(0, 14);
      return out.length ? { from: cx.stringStart, to: caret, items: out.map((k) => ({ ...k, kind: "key" })) } : null;
    }
    if (cx.key && RULE_KEYS.has(cx.key)) {
      const word = /[A-Za-z_][\w.]*$/.exec(typed)?.[0] || "";
      if (!word) return null;
      const from = caret - word.length;
      if (word.includes(".")) {
        const k = word.lastIndexOf("."), obj = word.slice(0, k), part = word.slice(k + 1);
        const fields: string[] = R.fields?.[obj] || ((R.symbols || []).includes(obj) || /^[A-Z]{1,5}$/.test(obj) ? R.symbol_fields : []);
        const out = fields.filter((f) => f.startsWith(part) && f !== part).slice(0, 14).map((f) => ({ label: `${obj}.${f}`, kind: "field" }));
        return out.length ? { from, to: caret, items: out } : null;
      }
      const lw = word.toLowerCase();
      const out = [
        ...(R.names || []).filter((n: any) => n.name.toLowerCase().startsWith(lw)).map((n: any) => ({ label: n.name, insert: n.function ? `${n.name}(` : n.name, detail: `${n.signature}  ${n.description}`, kind: n.function ? "function" : "name" })),
        ...Object.keys(R.fields || {}).filter((o) => o.startsWith(lw)).map((o) => ({ label: o, insert: `${o}.`, detail: `fields: ${(R.fields[o] as string[]).slice(0, 5).join(", ")}…`, kind: "field" })),
        ...(R.symbols || []).filter((s: string) => word === word.toUpperCase() && s.startsWith(word) && s !== symbol).map((s: string) => ({ label: s, insert: `${s}.`, detail: "another symbol: price, from_open_pct, vwap…", kind: "field" })),
        ...("signal".startsWith(lw) ? (R.signals || []).map((s: string) => ({ label: `signal ${s}`, insert: `signal(\\"${s}\\")`, detail: "a saved signal list", kind: "function" })) : []),
      ].filter((x, i, all) => (x.insert ?? x.label) !== word && all.findIndex((y) => y.label === x.label) === i).slice(0, 14);
      return out.length ? { from, to: caret, items: out } : null;
    }
    const en: string[] | undefined = cx.key === "type" && /conditions|custom|children/.test(cx.container) ? Object.keys(R.leaf_types || {}) : cx.key === "symbol" ? R.symbols : schemaAt(R.schema, cx.valuePath)?.node?.enum;
    const out = (en || []).filter((x) => String(x).startsWith(typed) && String(x) !== typed).slice(0, 14);
    return out.length ? { from: cx.stringStart, to: caret, items: out.map((x) => ({ label: String(x), detail: cx.key === "type" ? R.leaf_types?.[x]?.label : undefined })) } : null;
  }, [ref.data, symbol]);

  const onCaret = useCallback((start: number) => {
    const R = ref.data;
    if (ranges) { const p = pathAt(ranges, start); if (p !== sel) setSel(p); }
    if (!R) return;
    const cx = contextAt(text, start);
    let d = "";
    if (cx.inString && cx.key && RULE_KEYS.has(cx.key)) {
      const a = /[A-Za-z_][\w.]*$/.exec(text.slice(cx.stringStart, start))?.[0] || "", z = /^[\w.]*/.exec(text.slice(start))?.[0] || "";
      const word = a + z, n = (R.names || []).find((x: any) => x.name === word || x.name === word.split(".")[0]);
      if (n) d = `${n.signature} — ${n.description}${n.needs === "intraday" ? " · needs intraday bars" : ""}`;
      else if (R.fields?.[word.split(".")[0]]) d = `${word.split(".")[0]}: ${(R.fields[word.split(".")[0]] as string[]).join(", ")}`;
      else d = "A rule. Names complete as you type; and / or / not combine conditions.";
    } else {
      const p = cx.valuePath, s = schemaAt(R.schema, p);
      const hint = HINTS[p.replace(/\[\d+\]/g, "")];
      d = [p || "strategy", s ? typeWord(s.raw, s.node) : "", s?.raw?.description || s?.node?.description || hint || "", s?.raw?.default !== undefined && typeof s.raw.default !== "object" ? `default ${JSON.stringify(s.raw.default)}` : ""].filter(Boolean).join(" · ");
    }
    setCaretDoc(d);
  }, [ref.data, ranges, sel, text]);

  // ----- a breakpoint from the gutter: the rule on that line
  const onGutter = (line: number) => {
    if (!ranges) return;
    const hit = (v?.expressions || []).find((e: any) => { const r = ranges.get(e.path); return r && lineOf(text, r.start) <= line && lineOf(text, r.end) >= line; });
    if (!hit) { toast.message("Click beside a rule (an \"expr\" or \"when\" line) to break when it first holds."); return; }
    if (breakLine === line) { setBreakLine(null); setBreakWhen(""); return; }
    setBreakLine(line); setBreakWhen(hit.expr); setTab("debug");
  };

  // ----- edit the selected section by instruction
  const apply = async () => {
    if (!instr.trim() || !parsed.ok) return;
    setApplying(true);
    try {
      const r = await api.post<any>("/api/ide/edit", { strategy: parsed.value, path: sel, instruction: instr });
      if ((r.unsupported || []).length) { setNotes(r.unsupported.map((t: string) => ({ tone: "warn" as const, text: t }))); setChanged([]); }
      else { setUndo(text); setText(format(r.strategy)); setChanged(r.changed || []); setNotes((r.notes || []).map((t: string) => ({ tone: "info" as const, text: t }))); setInstr(""); }
    } catch (e: any) { setNotes([{ tone: "warn", text: e.message }]); } finally { setApplying(false); }
  };
  const draft = async () => {
    setDrafting(true);
    try {
      const r = await api.post<any>("/api/ide/from-text", { text: desc });
      if (!r.understood) { toast.error(r.note || "That could not be read as a strategy."); return; }
      setUndo(text); setText(format(r.strategy)); setSel(""); setChanged([]); setDescribe(false);
      useBuilder.setState({ strategyId: null });
      setNotes([
        ...(r.unsupported || []).map((t: string) => ({ tone: "warn" as const, text: `Not represented, and nothing was substituted for it: "${t}"` })),
        ...(r.missing || []).map((t: string) => ({ tone: "warn" as const, text: `Missing: ${t}` })),
        ...(r.notes || []).map((t: string) => ({ tone: "info" as const, text: t })),
        ...(r.assumptions || []).map((a: any) => ({ tone: "info" as const, text: `Assumed ${a.field}: ${a.value} (${a.reason})` })),
      ]);
    } catch (e: any) { toast.error(e.message); } finally { setDrafting(false); }
  };
  const save = async () => {
    if (!valid || !v?.full) return;
    try {
      const r = b.strategyId ? await api.put<any>(`/api/strategies/${b.strategyId}`, { config: v.full }) : await api.post<any>("/api/strategies", { config: v.full });
      useBuilder.setState({ strategyId: r.id, dirty: false });
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
      toast.success(`Saved "${v.full.name}"`);
    } catch (e: any) { toast.error(e.message); }
  };
  const sectionWord = sel ? sel : "the whole strategy";

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="ide">
      <PageHeader title={<span className="flex items-center gap-2"><Code2 className="size-3.5 text-info" />Strategy IDE</span>}
        sub={<span className="flex min-w-0 items-center gap-2 text-xs"><span className="truncate text-fg-2">{doc.name || "Untitled strategy"}</span>
          {!parsed.ok ? <span className="shrink-0 text-down">✕ does not parse</span> : !v ? <span className="shrink-0 text-muted">checking…</span> : valid ? <span className="shrink-0 text-up">✓ valid</span> : <span className="shrink-0 text-down">✕ {diags.length} problem{diags.length !== 1 ? "s" : ""}</span>}
          {b.dirty && valid && <span className="shrink-0 text-muted">· unsaved</span>}</span>}>
        <Button size="sm" variant="outline" onClick={() => setDescribe(true)} data-testid="ide-describe"><Sparkles />Describe a strategy</Button>
        <Menu>
          <MenuTrigger asChild><Button size="sm" variant="ghost"><FolderOpen />Open</Button></MenuTrigger>
          <MenuContent align="end" className="max-h-[50vh] w-72 overflow-auto">
            {(saved.data || []).length === 0 && <div className="px-2 py-2 text-xs text-muted">No saved strategies yet.</div>}
            {(saved.data || []).map((s) => <MenuItem key={s.id} onSelect={() => { useBuilder.getState().load(s.config, s.id); open(s.config); }}><span className="truncate">{s.name}</span><span className="ml-auto pl-2 text-2xs text-muted">{s.symbol}</span></MenuItem>)}
          </MenuContent>
        </Menu>
        <Button size="sm" variant="ghost" onClick={() => nav("/builder")} title="The same strategy in the visual builder"><Blocks />Visual builder</Button>
        <Button size="sm" onClick={save} disabled={!valid}><Save />Save</Button>
      </PageHeader>

      <div className="flex min-h-0 flex-1">
        {/* outline */}
        <aside className="flex w-[232px] shrink-0 flex-col border-r border-line bg-bg-1" aria-label="Strategy outline">
          <div className="label flex h-7 shrink-0 items-center border-b border-line px-2.5">Outline</div>
          <div className="min-h-0 flex-1 overflow-auto py-1" data-testid="outline">
            {items.map((it) => (
              <button key={it.path} onClick={() => select(it.path)} title={it.path}
                className={cn("flex w-full items-baseline gap-1.5 py-[3px] pr-2 text-left text-xs hover:bg-bg-3", it.head ? "mt-1 font-medium text-fg" : "text-fg-2", active === it.path && "bg-bg-3 shadow-[inset_2px_0_0_var(--color-accent)]")} style={{ paddingLeft: 10 + it.depth * 11 }}>
                {[...badPaths].some((p) => p === it.path || p.startsWith(`${it.path}.`) || p.startsWith(`${it.path}[`)) && <AlertTriangle className="size-3 shrink-0 self-center text-down" />}
                <span className={cn("min-w-0 truncate", !it.head && /[<>=(]/.test(it.label) && "font-mono text-[11px]")}>{it.label}</span>
                {it.detail && <span className="min-w-0 flex-1 truncate text-right text-2xs text-muted">{it.detail}</span>}
              </button>
            ))}
          </div>
        </aside>

        {/* structured view */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1">
            <CodeEditor ref={editor} value={text} onChange={(t) => { setText(t); if (changed.length) setChanged([]); }} onCaret={onCaret} marks={marks} diags={diags} breakLine={breakLine} onGutter={onGutter} complete={complete} functions={fnSet} names={nameSet} />
          </div>
          <div className="flex h-6 shrink-0 items-center gap-2 border-t border-line bg-bg-1 px-2.5 text-2xs text-muted" data-testid="inline-doc"><span className="truncate">{caretDoc || "Place the caret on a field or inside a rule for its documentation. Names complete as you type."}</span></div>
          {(diags.length > 0 || notes.length > 0) && (
            <div className="max-h-[132px] shrink-0 overflow-auto border-t border-line bg-bg-1 px-2.5 py-1.5 text-xs" data-testid="problems">
              {diags.map((d, i) => <button key={`d${i}`} className="flex w-full items-start gap-2 py-0.5 text-left hover:text-fg" onClick={() => editor.current?.reveal(d.offset)}><X className="mt-0.5 size-3 shrink-0 text-down" /><span><span className="num text-muted">line {lineOf(text, d.offset) + 1}</span> {d.message}</span></button>)}
              {notes.map((n, i) => <div key={`n${i}`} className="flex items-start gap-2 py-0.5">{n.tone === "warn" ? <AlertTriangle className="mt-0.5 size-3 shrink-0 text-warn" /> : <Check className="mt-0.5 size-3 shrink-0 text-up" />}<span className="text-fg-2">{n.text}</span></div>)}
            </div>
          )}
          <div className="shrink-0 border-t border-line bg-bg-1 p-2">
            <div className="mb-1 flex items-center gap-2 text-2xs text-muted">
              <Wand2 className="size-3 text-accent" /><span>Edit <b className="font-mono font-normal text-accent-2">{sectionWord}</b> by instruction. Every line it changes is marked; nothing else in the strategy is touched.</span>
              <span className="flex-1" />{changed.length > 0 && <span className="text-up">{changed.length} path{changed.length !== 1 ? "s" : ""} changed, marked in green</span>}
              {undo !== null && <button className="flex items-center gap-1 hover:text-fg" onClick={() => { setText(undo); setUndo(null); setChanged([]); setNotes([]); }}><Undo2 className="size-3" />Undo</button>}
            </div>
            <div className="flex gap-1.5">
              <Input value={instr} onChange={(e) => setInstr(e.target.value)} onKeyDown={(e) => e.key === "Enter" && apply()} placeholder={sel.startsWith("exits") ? EXAMPLES[0] : sel.startsWith("entry") ? EXAMPLES[1] : sel.startsWith("sizing") ? EXAMPLES[4] : "Select a section, then say what to change: " + EXAMPLES[0]} aria-label="Instruction for the selected section" data-testid="ide-instruction" />
              <Button variant="primary" onClick={apply} disabled={applying || !instr.trim() || !parsed.ok} data-testid="ide-apply">{applying ? <Loader2 className="animate-spin" /> : <Wand2 />}Apply</Button>
            </div>
          </div>
        </div>

        {/* tools */}
        <aside className="flex w-[388px] shrink-0 flex-col border-l border-line bg-bg-1">
          <div className="flex h-7 shrink-0 items-center gap-0.5 border-b border-line px-1.5" role="tablist">
            {([["inspect", "Inspect"], ["debug", "Debugger"], ["backtest", "Backtest"], ["reference", "Reference"]] as const).map(([k, l]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)} className={cn("h-5 rounded-sm px-2 text-2xs font-medium", tab === k ? "bg-bg-4 text-fg" : "text-muted hover:text-fg")} data-testid={`ide-tab-${k}`}>{l}</button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            {tab === "inspect" && <Inspect doc={doc} sel={sel} symbol={symbol} onSet={setPath} onSelect={(p) => select(p)} leafTypes={ref.data?.leaf_types} />}
            {tab === "debug" && <Debugger doc={doc} valid={valid} symbol={symbol} breakWhen={breakWhen} setBreakWhen={(s) => { setBreakWhen(s); if (!s) setBreakLine(null); }} />}
            {tab === "backtest" && <Backtest valid={valid} strategyId={b.strategyId} />}
            {tab === "reference" && (
              <div className="flex flex-col">
                <div className="sticky top-0 border-b border-line bg-bg-1 p-1.5"><Input sz="sm" placeholder="Search the rule language: gap, vwap, atr, portfolio…" value={refQ} onChange={(e) => setRefQ(e.target.value)} /></div>
                {(ref.data?.names || []).filter((n: any) => !refQ || `${n.name} ${n.description} ${n.category}`.toLowerCase().includes(refQ.toLowerCase())).map((n: any) => (
                  <Tip key={n.name} content="Click to copy" side="left"><button className="border-b border-line/50 px-2.5 py-1.5 text-left text-xs hover:bg-bg-2" onClick={() => { navigator.clipboard?.writeText(n.signature); toast.success(`Copied ${n.signature}`); }}>
                    <div className="flex items-baseline gap-2"><span className="font-mono text-[11.5px] text-[#61afef]">{n.signature}</span><span className="ml-auto shrink-0 text-2xs text-faint">{n.category}{n.needs === "intraday" ? " · intraday" : ""}</span></div>
                    <div className="mt-0.5 text-2xs text-muted">{n.description}</div>
                  </button></Tip>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      <Dialog open={describe} onOpenChange={setDescribe} title="Describe a strategy in plain English" width={620}
        footer={<><Button variant="ghost" onClick={() => setDescribe(false)}>Cancel</Button><Button variant="primary" onClick={draft} disabled={drafting || desc.trim().length < 12} data-testid="ide-draft">{drafting ? <Loader2 className="animate-spin" /> : <Sparkles />}Write the strategy</Button></>}>
        <div className="flex flex-col gap-2 text-xs">
          <textarea className="h-32 w-full resize-none rounded-sm border border-line-2 bg-bg-2 p-2 text-sm outline-none focus:border-accent/60" value={desc} onChange={(e) => setDesc(e.target.value)} autoFocus
            placeholder="Buy a 0DTE at-the-money SPY call at 10:00 when SPY is below its open and QQQ is weak too. Take profit at 50%, stop at 30%, out by 15:30." aria-label="Strategy description" />
          <p className="text-muted">The description is read into the structured strategy you can then edit. Whatever is assumed is listed, and anything that cannot be represented is reported rather than replaced with something similar. Nothing runs until you run it.</p>
        </div>
      </Dialog>
    </div>
  );
}
