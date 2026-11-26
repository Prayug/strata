/* The AI Lab's front page: a prominent composer, example prompts grouped by intent, and the saved research notebooks. */
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ArrowUp, BookOpen, BookOpenText, Code2, Cpu, Database, Download, FlaskConical, Grid3x3, ListChecks, Loader2, Rocket, Scale, Search, Sparkles, Trash2 } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Kbd, Tip } from "@/components/ui";
import { api, queryClient, useJob } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { useAiMeta } from "@/lib/lab";
import { useUi } from "@/lib/store";
import { useHotkeys } from "./bits";

export interface ComposerHandle { focus: () => void; set: (t: string) => void }

/** The one input of the lab. Enter sends, Shift+Enter breaks the line, "/" focuses it from anywhere. */
export const Composer = forwardRef<ComposerHandle, {
  onSend: (text: string) => void; busy?: boolean; placeholder?: string; mode: "quick" | "review"; onMode: (m: "quick" | "review") => void; big?: boolean; chips?: string[]; autoFocus?: boolean; bar?: boolean;
}>(({ onSend, busy, placeholder, mode, onMode, big, chips, autoFocus, bar }, ref) => {
  const [text, setText] = useState("");
  const el = useRef<HTMLTextAreaElement>(null);
  useImperativeHandle(ref, () => ({ focus: () => el.current?.focus(), set: (t: string) => { setText(t); setTimeout(() => el.current?.focus(), 0); } }));
  useEffect(() => {
    const t = el.current;
    if (!t) return;
    t.style.height = "0px";
    t.style.height = `${Math.min(t.scrollHeight, big ? 220 : 160)}px`;
  }, [text, big]);
  useEffect(() => { if (autoFocus) el.current?.focus(); }, [autoFocus]);
  const send = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };
  if (bar) {
    // the command line of the research desk: one row across the top, the request on the left and how it runs on the right
    return (
      <div className="cmdline">
        <div className="flex items-start gap-2.5 px-3.5 py-2">
          <span className="num mt-[5px] select-none text-[15px] leading-6 text-accent" aria-hidden>❯</span>
          <textarea ref={el} rows={1} value={text} placeholder={placeholder} data-testid="composer" aria-label="Research request" className="min-w-0 flex-1"
            onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          <div className="mt-[3px] inline-flex shrink-0 border border-line-2 bg-bg-2 p-0.5 text-xs">
            {([["quick", "Run at once", "Runs straight away and tells you every assumption it made."], ["review", "Review first", "Shows how the request was read and waits for you to confirm before running."]] as const).map(([k, l, h]) => (
              <Tip key={k} content={h}><button onClick={() => onMode(k)} className={cn("h-6 px-2.5 transition-colors", mode === k ? "bg-bg-4 text-fg" : "text-muted hover:text-fg")}>{l}</button></Tip>
            ))}
          </div>
          <button className="btn primary sm mt-[3px] !rounded-[3px] !px-2.5" onClick={send} disabled={busy || !text.trim()} aria-label="Send" data-testid="send">{busy ? <Loader2 className="animate-spin" /> : <ArrowUp strokeWidth={2.6} />}</button>
        </div>
        {chips && chips.length > 0 && (
          <div className="flex gap-1.5 overflow-x-auto border-t border-line px-3.5 py-1.5" style={{ scrollbarWidth: "none" }}>
            <span className="shrink-0 self-center text-[10.5px] uppercase tracking-[0.07em] text-faint">next</span>
            {chips.map((c) => <button key={c} className="chip !h-[22px] shrink-0 !rounded-[3px] !text-[11.5px]" onClick={() => onSend(c)} disabled={busy} title="Send this follow-up">{c}</button>)}
          </div>
        )}
      </div>
    );
  }
  return (
    <div>
      {chips && chips.length > 0 && (
        <div className="mb-2 flex gap-1.5 overflow-x-auto pb-0.5" style={{ scrollbarWidth: "none" }}>
          {chips.map((c) => <button key={c} className="chip" onClick={() => onSend(c)} disabled={busy} title="Send this follow-up">{c}</button>)}
        </div>
      )}
      <div className="composer">
        <textarea ref={el} rows={big ? 2 : 1} value={text} placeholder={placeholder} data-testid="composer" aria-label="Research request"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }} />
        <div className="flex items-center gap-2 px-3 pb-2.5 pt-1">
          <div className="inline-flex rounded-lg border border-line-2 bg-bg-2 p-0.5 text-xs">
            {([["quick", "Quick research", "Runs straight away and tells you every assumption it made."], ["review", "Review first", "Shows how the request was read and waits for you to confirm before running."]] as const).map(([k, l, h]) => (
              <Tip key={k} content={h}><button onClick={() => onMode(k)} className={cn("rounded-md px-2.5 h-6 transition-colors", mode === k ? "bg-bg-4 text-fg" : "text-muted hover:text-fg")}>{l}</button></Tip>
            ))}
          </div>
          <span className="flex-1" />
          <span className="hidden sm:flex items-center gap-1 text-[11px] text-faint"><Kbd>↵</Kbd> send <Kbd>⇧↵</Kbd> new line</span>
          <button className="btn primary sm !px-2.5" onClick={send} disabled={busy || !text.trim()} aria-label="Send" data-testid="send">
            {busy ? <Loader2 className="animate-spin" /> : <ArrowUp strokeWidth={2.6} />}
          </button>
        </div>
      </div>
    </div>
  );
});
Composer.displayName = "Composer";

export function InterpreterChip() {
  const meta = useAiMeta();
  const llm = meta.data?.llm;
  if (!llm) return null;
  return (
    <Tip content={llm.active ? `Requests are read by ${llm.active}. It only fills in a structured draft; the engine calculates every number.` : llm.error ? `The configured model is unavailable: ${llm.error}` : "Requests are read by the built-in interpreter. Add a model under Settings for freer phrasing."}>
      <span className="chip"><Cpu className="size-3.5" />{llm.active ? llm.active.split(":").slice(-1)[0] : "built-in interpreter"}</span>
    </Tip>
  );
}

/* ----------------------------------------------------------------------------- first run */
function Onboarding({ ob, onAsk }: { ob: any; onAsk: (t: string) => void }) {
  const nav = useNavigate();
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, (s) => {
    setJobId(null);
    for (const k of ["onboarding", "coverage"]) queryClient.invalidateQueries({ queryKey: [k] });
    if (s.status === "done") toast.success("Free market data downloaded"); else if (s.status === "error") toast.error(s.error || "The download failed");
  });
  const fetchFree = async () => {
    try { const r = await api.post<{ id: string }>("/api/data/quickstart", { symbols: ob.free_download.symbols.filter((x: string) => x !== "VIX") }); setJobId(r.id); } catch (e: any) { toast.error(e.message); }
  };
  const dismiss = async () => { await api.put("/api/settings", { settings: { "ui.onboarded": "1" } }).catch(() => {}); queryClient.invalidateQueries({ queryKey: ["onboarding"] }); };
  const busy = !!jobId;
  return (
    <div className="card mt-6 overflow-hidden" data-testid="onboarding">
      <div className="flex items-center gap-3 border-b border-line px-4 py-3">
        <Rocket className="size-[18px] text-accent" />
        <div className="min-w-0 flex-1"><div className="text-[14px] font-medium">{ob.has_data ? "You are set up. Three things worth knowing." : "Start here: two minutes to a first real answer"}</div></div>
        {ob.has_data && <button className="btn quiet sm" onClick={dismiss} data-testid="onboarding-dismiss">Got it, hide this</button>}
      </div>
      <div className="grid gap-px bg-line md:grid-cols-3">
        {ob.steps.map((st: any, i: number) => (
          <div key={st.id} className="bg-bg-1 px-4 py-3.5">
            <div className="flex items-center gap-2 text-[13px] font-medium">
              <span className={cn("grid size-5 place-items-center rounded-full border text-[11px]", st.done ? "border-up/50 bg-up/10 text-up" : "border-line-2 text-muted")} aria-label={st.done ? "done" : "to do"}>{st.done ? "✓" : i + 1}</span>{st.title}
            </div>
            <div className="mt-1.5 text-xs soft">{st.detail}</div>
            {st.id === "data" && !st.done && (
              <div className="mt-2.5">
                <button className="btn primary sm" onClick={fetchFree} disabled={busy} data-testid="onboarding-download">{busy ? <Loader2 className="animate-spin" /> : <Download />}Get {ob.free_download.symbols.join(", ")} ({ob.free_download.size})</button>
                <div className="mt-1.5 text-[11px] dim">{ob.free_download.what}, from {ob.free_download.source}. Nothing paid, no account.</div>
                {job && <div className="mt-1.5 text-[11px] text-fg-2">{Math.round((job.progress || 0) * 100)}% {job.message}</div>}
              </div>
            )}
            {st.id === "question" && ob.has_data && <button className="btn sm mt-2.5" onClick={() => onAsk(ob.suggestions.find((x: any) => x.ready)?.text || ob.suggestions[0].text)} data-testid="onboarding-ask"><Sparkles />Try a first question</button>}
            {st.id === "options" && <button className="btn quiet sm mt-2.5 !px-0" onClick={() => nav("/data?tab=sources")}>See every option data source<ArrowRight /></button>}
          </div>
        ))}
      </div>
      {ob.has_data && (
        <div className="border-t border-line px-4 py-3 text-xs soft">
          <b className="text-fg">What you can do right now:</b> ask market questions and explore patterns on {ob.symbols.filter((x: any) => x.symbol !== "VIX").map((x: any) => x.symbol).join(", ")} ({Math.max(0, ...ob.symbols.filter((x: any) => x.symbol !== "VIX").map((x: any) => x.daily_sessions)).toLocaleString()} daily sessions at most).
          {ob.symbols.some((x: any) => x.option_quote_sessions > 0) ? " Option strategies can be backtested on the stored end-of-day quotes." : " Backtesting an option strategy needs option prices: the free end-of-day archive downloads itself the first time a backtest asks for it."}
          {" "}Intraday option fills need trade bars, which the collector saves going forward or a free Alpaca key supplies from 2024.
        </div>
      )}
    </div>
  );
}

const WORKFLOWS: { id: string; title: string; line: string; tone: string; icon: any; links: [string, string, string][] }[] = [
  { id: "explore", title: "Explore", line: "Ask a market question and get evidence, not an opinion.", tone: "var(--color-accent)", icon: Sparkles,
    links: [["Patterns", "/patterns", "What price did after an event, every time it happened"], ["Knowledge base", "/findings", "Everything your research has established"]] },
  { id: "build", title: "Build", line: "Turn an idea into a strategy you can read line by line.", tone: "var(--color-info)", icon: Code2,
    links: [["Strategy IDE", "/ide", "Describe it, edit it, debug it on a real session"], ["Visual builder", "/builder", "Assemble legs and rules with forms"]] },
  { id: "validate", title: "Validate", line: "Find out whether it survives contact with real prices.", tone: "var(--color-up)", icon: Scale,
    links: [["Backtests", "/results", "Results on observed option prices"], ["Forward validation", "/forward", "Freeze it and judge it on sessions that do not exist yet"], ["Experiment registry", "/registry", "What has already been tested on which dates"]] },
];

const TRY: [string, (s: string) => string][] = [
  ["A strategy", (s) => `Buy ATM ${s} 0DTE calls at the open after a 1% down day, exit at 3:55 PM Eastern or when ${s} has risen 1%`],
  ["An ordered rule", (s) => `Buy ${s} 0DTE calls when price reclaims VWAP after declining 0.5% from the open, with a 30% premium target and a 15% premium stop`],
  ["A grid from one sentence", (s) => `Buy ATM ${s} 0DTE calls at 10:00 AM Eastern after a 0.5% down day and compare exits after 15, 30, 60 and 120 minutes with 0DTE versus 1DTE`],
  ["Across symbols", () => "Buy ATM SPY 0DTE calls at the open after a 1% down day, sell at 3:55 PM Eastern, and compare the same signal across SPY, QQQ and IWM"],
];

/** What is stored on this machine, as one line per symbol, and sentences that run on it. Shown only when there is something stored. */
function Stored({ onPick }: { onPick: (text: string) => void }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["dataset-options"], queryFn: () => api.get<any>("/api/dataset/options"), staleTime: 60_000, retry: false });
  const d = q.data;
  const rows: any[] = (d?.symbols || []).filter((r: any, i: number, all: any[]) => all.findIndex((x) => x.symbol === r.symbol) === i).slice(0, 6);
  if (!rows.length) return null;
  const big = rows[0];
  return (
    <div className="mt-6 border border-line" data-testid="stored-tape">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-bg-1 px-3.5 py-2">
        <span className="eyebrow">Stored on this machine</span>
        <span className="num text-[12.5px]">{Number(d.bars).toLocaleString()} option trade bars</span>
        <span className="flex-1" />
        <span className={cn("text-xs", d.collector.enabled ? "text-up" : "text-warn")}>{d.collector.enabled ? `● collector on${d.collector.next_due ? `, next cycle ${new Date(d.collector.next_due).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}` : ""}` : "○ collector off: new sessions are not being added"}</span>
        <span className={cn("text-xs", d.quotes.connected ? "text-up" : "dim")}>{d.quotes.connected ? "● " : "○ "}{d.quotes.text}</span>
      </div>
      <div className="grid grid-cols-1 gap-px bg-line md:grid-cols-3">
        {rows.slice(0, 3).map((r) => (
          <button key={r.symbol} className="bg-bg-0 px-3.5 py-2.5 text-left transition-colors hover:bg-bg-2" onClick={() => nav("/replay")} title="Open a stored session in the replay">
            <div className="flex items-baseline gap-2"><span className="num text-[15px] font-medium">{r.symbol}</span><span className="num text-xs dim">{r.sessions} sessions</span></div>
            <div className="num mt-0.5 text-[11.5px] dim">{r.first} → {r.last}</div>
            <div className="num text-[11.5px] dim">{(r.bars / 1e6).toFixed(1)}M bars · {r.with_trades.toLocaleString()} contract-sessions · {r.provider}</div>
          </button>
        ))}
      </div>
      <div className="border-t border-line px-3.5 py-2.5">
        <div className="eyebrow mb-1.5">Sentences that run on it</div>
        <div className="flex flex-col gap-1">
          {TRY.map(([k, f]) => <button key={k} className="flex items-baseline gap-3 text-left text-[13px] soft hover:text-fg" onClick={() => onPick(f(big.symbol))} title="Put this on the line to edit or run"><span className="w-[150px] shrink-0 text-xs dim">{k}</span><span className="min-w-0">{f(big.symbol)}</span></button>)}
        </div>
        <div className="mt-2 text-[11.5px] dim">These are trade prints, not quotes: a result on them shows where contracts traded, and its cost assumptions are stated with it.</div>
      </div>
    </div>
  );
}

export function Landing() {
  const nav = useNavigate();
  const meta = useAiMeta();
  const advanced = useUi((u) => u.mode) === "advanced";
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"quick" | "review">("quick");
  const [q, setQ] = useState("");
  const comp = useRef<ComposerHandle>(null);
  useEffect(() => { if (meta.data?.exec_mode) setMode(meta.data.exec_mode); }, [meta.data?.exec_mode]);
  const books = useQuery({ queryKey: ["notebooks", q], queryFn: () => api.get<any[]>(`/api/ai/search?q=${encodeURIComponent(q)}`), staleTime: 5_000 });
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 60_000 });
  const ob = useQuery({ queryKey: ["onboarding"], queryFn: () => api.get<any>("/api/onboarding"), staleTime: 30_000 });
  useHotkeys({ "/": (e) => { e.preventDefault(); comp.current?.focus(); } });

  const start = async (text: string) => {
    setBusy(true);
    try {
      const s = await api.post<any>("/api/ai/sessions", { title: text.slice(0, 70) });
      if (mode !== (meta.data?.exec_mode || "quick")) await api.post(`/api/ai/sessions/${s.id}/mode`, { exec_mode: mode });
      sessionStorage.setItem(`lab-first-${s.id}`, text);
      nav(`/lab/${s.id}`);
    } catch (e: any) {
      toast.error(e.message);
      setBusy(false);
    }
  };
  const real = (cov.data || []).filter((r) => !r.synthetic && r.symbol !== "VIX");
  const groups: { title: string; items: string[] }[] = meta.data?.example_groups || [];
  const icons = [Search, FlaskConical, Sparkles];
  const firstRun = ob.data && (!ob.data.onboarded || !ob.data.has_data);
  return (
    <div className="lab h-full overflow-auto">
      <div className="mx-auto w-full max-w-[960px] px-6 pb-16 pt-[7vh]">
        <div className="eyebrow mb-3 flex items-center gap-2"><Sparkles className="size-3.5 text-accent" /> Explore</div>
        <h1>What do you want to find out about the market?</h1>
        <p className="mt-2 max-w-[660px] text-[15px] soft">
          Ask in plain English. Strata plans the analysis, runs it on real history, and shows what it assumed, what it found and how far to trust it.
        </p>
        <div className="mt-6">
          <Composer ref={comp} big autoFocus onSend={start} busy={busy} mode={mode} onMode={setMode} placeholder="e.g. Does SPY tend to recover after large down days, even after controlling for volatility?" />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs dim">
          <InterpreterChip />
          {cov.isSuccess && (real.length ? (
            <Tip content={real.map((r) => `${r.symbol}: ${r.daily ? `${r.daily.first} to ${r.daily.last}` : "no daily history"}`).join("\n")}>
              <span className="chip"><Database className="size-3.5" />Real data stored: {real.map((r) => r.symbol).slice(0, 5).join(", ")}</span>
            </Tip>
          ) : <span className="chip"><Database className="size-3.5" />No data stored yet</span>)}
          <span>Nothing is simulated, and no paid data is ever fetched without your approval.</span>
        </div>

        <Stored onPick={(t) => comp.current?.set(t)} />

        {firstRun && <Onboarding ob={ob.data} onAsk={start} />}

        {ob.data?.has_data && (
          <div className="mt-6" data-testid="suggestions">
            <div className="mb-2 text-[13px] font-medium">Questions your stored data can answer now</div>
            <div className="grid gap-2 md:grid-cols-2">
              {ob.data.suggestions.map((x: any) => (
                <button key={x.text} onClick={() => (x.ready ? start(x.text) : comp.current?.set(x.text))} disabled={busy} title={x.ready ? "Run this question" : `Needs ${x.needs}`}
                  className="card flex items-start gap-2.5 px-3.5 py-2.5 text-left text-[13px] soft transition-colors hover:border-accent/60 hover:text-fg">
                  <span className={cn("mt-0.5 shrink-0 text-[11px]", x.ready ? "text-up" : "text-warn")}>{x.ready ? "●" : "○"}</span>
                  <span className="min-w-0 flex-1">{x.text}<span className="mt-0.5 block text-[11px] dim">{x.ready ? `uses ${x.needs}` : `needs ${x.needs}, not stored yet`}</span></span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-7 grid gap-3 md:grid-cols-3" data-testid="workflows">
          {WORKFLOWS.map((w) => (
            <div key={w.id} className="card flex flex-col px-4 py-3.5" style={{ borderTopColor: w.tone, borderTopWidth: 2 }}>
              <div className="flex items-center gap-2 text-[14px] font-medium"><w.icon className="size-4" style={{ color: w.tone }} />{w.title}</div>
              <div className="mt-1 text-xs soft">{w.line}</div>
              <div className="mt-2.5 flex flex-col gap-1">
                {w.id === "explore" && <button className="flex items-baseline gap-2 text-left text-[13px] hover:text-accent-2" onClick={() => comp.current?.focus()}><ArrowRight className="size-3 shrink-0 self-center dim" /><span>Ask a question</span><span className="truncate text-[11px] dim">the box above</span></button>}
                {w.links.map(([l, to, d]) => <button key={to} className="flex items-baseline gap-2 text-left text-[13px] hover:text-accent-2" onClick={() => nav(to)}><ArrowRight className="size-3 shrink-0 self-center dim" /><span className="shrink-0">{l}</span><span className="truncate text-[11px] dim">{d}</span></button>)}
              </div>
            </div>
          ))}
        </div>

        {advanced && (
          <>
            <div className="mt-7 grid gap-3 md:grid-cols-3" data-testid="start-points">
              {([
                [ListChecks, "Investigate a hypothesis", "A broad question becomes a bounded plan: every configuration listed before it runs, a final test period kept untouched.", () => comp.current?.set("Investigate whether SPY tends to reverse following large down days.")],
                [BookOpenText, "Browse market behaviour", "Reproducible studies of gaps, opening ranges, streaks and volatility, computed on your stored data.", () => nav("/encyclopedia")],
                [Grid3x3, "Explore sensitivity", "See a result across two assumptions at once, with thin samples and stable regions marked.", () => nav("/sensitivity")],
              ] as const).map(([I, t, d, go]) => (
                <button key={t} onClick={go} className="card flex items-start gap-3 px-4 py-3 text-left transition-colors hover:border-accent/60">
                  <I className="mt-0.5 size-[18px] shrink-0 text-accent" /><span><span className="block text-[13.5px] font-medium">{t}</span><span className="mt-0.5 block text-xs dim">{d}</span></span>
                </button>
              ))}
            </div>
            <div className="mt-6 grid gap-4 md:grid-cols-3">
              {groups.map((g, gi) => {
                const I = icons[gi % 3];
                return (
                  <div key={g.title} className="min-w-0">
                    <div className="mb-2 flex items-center gap-2 text-[13px] font-medium"><I className="size-4 text-accent" />{g.title}</div>
                    <div className="flex flex-col gap-2">
                      {g.items.map((x) => (
                        <button key={x} onClick={() => comp.current?.set(x)} onDoubleClick={() => start(x)} title="Click to edit, double-click to run"
                          className="card px-3.5 py-2.5 text-left text-[13px] soft transition-colors hover:border-accent/60 hover:text-fg">{x}</button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {ob.data && (ob.data.unlock || []).length > 0 && (
          <div className="mt-8" data-testid="unlock">
            <div className="mb-2 flex items-center gap-3"><div className="text-[13px] font-medium">More data unlocks more</div><span className="flex-1" /><button className="btn quiet sm" onClick={() => nav("/data?tab=sources")}>All sources<ArrowRight /></button></div>
            <div className="grid gap-2 md:grid-cols-2">
              {ob.data.unlock.slice(0, 4).map((u: any) => (
                <button key={u.id} onClick={() => nav("/data?tab=sources")} className="card px-3.5 py-2.5 text-left transition-colors hover:border-line-2 hover:bg-bg-2">
                  <div className="flex items-center gap-2 text-[13px] font-medium"><span className="truncate">{u.name}</span><span className="flex-1" /><span className={cn("shrink-0 text-[11px] font-normal", u.status === "verified" ? "text-up" : "text-warn")}>{u.status === "verified" ? "● verified here" : `○ ${u.status}`}</span></div>
                  <div className="mt-0.5 line-clamp-2 text-xs soft">{u.what}</div>
                  <div className="mt-1 line-clamp-1 text-[11px] dim">{u.history} · {u.cost}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-10">
          <div className="mb-3 flex items-center gap-3">
            <h2 className="flex items-center gap-2"><BookOpen className="size-[18px] dim" />Your research</h2>
            <span className="flex-1" />
            <div className="relative w-64">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-faint" />
              <input className="field !pl-8" placeholder="Search questions and results" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search research" />
            </div>
          </div>
          {books.data && books.data.length === 0 && <div className="card px-4 py-6 text-center text-[13px] dim">{q ? "Nothing matches that search." : "Your research notebooks will appear here. Each one keeps its questions, versions, results and reports."}</div>}
          <div className="flex flex-col gap-2">
            {(books.data || []).map((b) => (
              <div key={b.id} className="card group flex items-center gap-3 px-4 py-3 transition-colors hover:border-line-2 hover:bg-bg-2 cursor-pointer" onClick={() => nav(`/lab/${b.id}`)}>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[14px] font-medium">{b.title}</div>
                  {b.match && b.match !== b.title && <div className="truncate text-xs dim">{b.match}</div>}
                </div>
                <span className="text-xs dim whitespace-nowrap">{b.questions} question{b.questions !== 1 ? "s" : ""} · {b.results} result{b.results !== 1 ? "s" : ""}</span>
                <span className="w-16 text-right text-xs text-faint">{ago(b.updated_at)}</span>
                <button className="btn quiet sm !px-1.5 opacity-0 group-hover:opacity-100" title="Delete this notebook" aria-label="Delete notebook"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!confirm(`Delete "${b.title}"? Its runs are kept under Backtests, and the experiment registry keeps the record of what it tested.`)) return;
                    await api.del(`/api/ai/sessions/${b.id}`);
                    queryClient.invalidateQueries({ queryKey: ["notebooks"] });
                  }}><Trash2 /></button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
