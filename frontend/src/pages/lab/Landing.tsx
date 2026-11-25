/* The AI Lab's front page: a prominent composer, example prompts grouped by intent, and the saved research notebooks. */
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, BookOpen, BookOpenText, Cpu, Database, FlaskConical, Grid3x3, ListChecks, Loader2, Search, Sparkles, Trash2 } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Kbd, Tip } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { useAiMeta } from "@/lib/lab";
import { useHotkeys } from "./bits";

export interface ComposerHandle { focus: () => void; set: (t: string) => void }

/** The one input of the lab. Enter sends, Shift+Enter breaks the line, "/" focuses it from anywhere. */
export const Composer = forwardRef<ComposerHandle, {
  onSend: (text: string) => void; busy?: boolean; placeholder?: string; mode: "quick" | "review"; onMode: (m: "quick" | "review") => void; big?: boolean; chips?: string[]; autoFocus?: boolean;
}>(({ onSend, busy, placeholder, mode, onMode, big, chips, autoFocus }, ref) => {
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

export function Landing() {
  const nav = useNavigate();
  const meta = useAiMeta();
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"quick" | "review">("quick");
  const [q, setQ] = useState("");
  const comp = useRef<ComposerHandle>(null);
  useEffect(() => { if (meta.data?.exec_mode) setMode(meta.data.exec_mode); }, [meta.data?.exec_mode]);
  const books = useQuery({ queryKey: ["notebooks", q], queryFn: () => api.get<any[]>(`/api/ai/search?q=${encodeURIComponent(q)}`), staleTime: 5_000 });
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 60_000 });
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
  return (
    <div className="lab h-full overflow-auto">
      <div className="mx-auto w-full max-w-[920px] px-6 pb-16 pt-[9vh]">
        <div className="eyebrow mb-3 flex items-center gap-2"><Sparkles className="size-3.5 text-accent" /> Research lab</div>
        <h1>What do you want to find out about the market?</h1>
        <p className="mt-2 max-w-[640px] text-[15px] soft">
          Ask in plain English. Strata works out what data that needs, downloads what is free, tests it on real history and gives you a report you can check line by line.
        </p>
        <div className="mt-6">
          <Composer ref={comp} big autoFocus onSend={start} busy={busy} mode={mode} onMode={setMode} placeholder="e.g. How often does SPY rise 1% after a 3% down day?" />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs dim">
          <InterpreterChip />
          {cov.isSuccess && (real.length ? (
            <Tip content={real.map((r) => `${r.symbol}: ${r.daily ? `${r.daily.first} to ${r.daily.last}` : "no daily history"}`).join("\n")}>
              <span className="chip"><Database className="size-3.5" />Real data stored: {real.map((r) => r.symbol).slice(0, 5).join(", ")}</span>
            </Tip>
          ) : <span className="chip"><Database className="size-3.5" />No data stored yet: the first question downloads what it needs</span>)}
          <span>Nothing is simulated, and no paid data is ever fetched without your approval.</span>
        </div>

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

        <div className="mt-8 grid gap-4 md:grid-cols-3">
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

        <div className="mt-12">
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
                    if (!confirm(`Delete "${b.title}"? Its runs are kept under Backtests.`)) return;
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
