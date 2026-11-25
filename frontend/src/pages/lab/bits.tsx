/* Small building blocks of the research workspace. */
import { AlertTriangle, Check, ChevronDown, ChevronRight, Info } from "lucide-react";
import * as React from "react";
import { Tip } from "@/components/ui";
import { cn } from "@/lib/format";
import { GLOSSARY, MODE_HELP, MODE_NAME, MODE_TONE } from "@/lib/lab";

const TERM_RE = new RegExp(`\\b(${Object.keys(GLOSSARY).sort((a, b) => b.length - a.length).map((k) => k.replace(/[/]/g, "\\/")).join("|")})\\b`, "gi");

/** Text with trading jargon underlined and explained on hover, so nobody has to know the vocabulary first. */
export function T({ children }: { children: string | null | undefined }) {
  if (!children) return null;
  const parts = String(children).split(TERM_RE);
  return (
    <>
      {parts.map((p, i) => {
        const key = Object.keys(GLOSSARY).find((k) => k.toLowerCase() === p.toLowerCase());
        return key ? <Tip key={i} content={GLOSSARY[key]}><span className="term">{p}</span></Tip> : <React.Fragment key={i}>{p}</React.Fragment>;
      })}
    </>
  );
}

export function Term({ k, children }: { k: string; children?: React.ReactNode }) {
  return <Tip content={GLOSSARY[k]}><span className="term">{children ?? k}</span></Tip>;
}

export function ModeBadge({ mode, className }: { mode?: string | null; className?: string }) {
  if (!mode) return null;
  return (
    <Tip content={MODE_HELP[mode]}>
      <span className={cn("inline-flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium whitespace-nowrap", MODE_TONE[mode] || MODE_TONE.auto, className)}>{MODE_NAME[mode] || mode}</span>
    </Tip>
  );
}

export function StatusChip({ status }: { status: string }) {
  const tone = { stated: "text-up border-up/30 bg-up/[0.08]", assumed: "text-warn border-warn/40 bg-warn/[0.08]", missing: "text-down border-down/40 bg-down/[0.08]", unsupported: "text-down border-down/40 bg-down/[0.08]" }[status] || "";
  const label = { stated: "you said", assumed: "assumed", missing: "missing", unsupported: "not represented" }[status] || status;
  return <span className={cn("inline-flex h-[18px] items-center rounded-full border px-1.5 text-[10.5px] font-medium tracking-wide whitespace-nowrap", tone)}>{label}</span>;
}

/** A headline number. When `onClick` is given the tile opens the observations it was computed from. */
export function StatTile({ label, value, sub, tone, onClick, hint }: { label: React.ReactNode; value: React.ReactNode; sub?: React.ReactNode; tone?: string; onClick?: () => void; hint?: string }) {
  const body = (
    <>
      <div className="text-xs dim flex items-center gap-1">{label}{hint && <Tip content={hint}><Info className="size-3 text-faint" /></Tip>}</div>
      <div className={cn("num mt-1 leading-7 font-medium tracking-tight whitespace-nowrap", String(value ?? "").length > 10 ? "text-[17px]" : "text-[22px]", tone)}>{value}</div>
      {sub && <div className="mt-0.5 text-xs dim truncate">{sub}</div>}
    </>
  );
  return onClick ? (
    <button onClick={onClick} className="card px-4 py-3 text-left transition-colors hover:border-accent/60 hover:bg-bg-2 min-w-0" title="Show the observations behind this number">{body}</button>
  ) : <div className="card px-4 py-3 min-w-0">{body}</div>;
}

export function Section({ title, children, right, open: openDefault = true, id, sub }: { title: React.ReactNode; children: React.ReactNode; right?: React.ReactNode; open?: boolean; id?: string; sub?: React.ReactNode }) {
  const [open, setOpen] = React.useState(openDefault);
  return (
    <section id={id} className="card overflow-hidden">
      <header className="flex items-center gap-2 px-4 h-11 cursor-pointer select-none" onClick={() => setOpen((o) => !o)}>
        {open ? <ChevronDown className="size-4 dim" /> : <ChevronRight className="size-4 dim" />}
        <h3 className="!text-[13.5px] !text-fg shrink-0 whitespace-nowrap">{title}</h3>
        {sub && <span className="min-w-0 text-xs dim truncate">{sub}</span>}
        <span className="flex-1" />
        <span onClick={(e) => e.stopPropagation()} className="flex items-center gap-2">{right}</span>
      </header>
      {open && <div className="border-t border-line px-4 py-3.5">{children}</div>}
    </section>
  );
}

export function Callout({ tone = "info", title, children, className }: { tone?: "info" | "warn" | "bad" | "good"; title?: React.ReactNode; children?: React.ReactNode; className?: string }) {
  const t = { info: "border-info/30 bg-info/[0.06]", warn: "border-warn/40 bg-warn/[0.07]", bad: "border-down/40 bg-down/[0.07]", good: "border-up/35 bg-up/[0.07]" }[tone];
  const Icon = tone === "good" ? Check : tone === "info" ? Info : AlertTriangle;
  const ic = { info: "text-info", warn: "text-warn", bad: "text-down", good: "text-up" }[tone];
  return (
    <div className={cn("flex gap-3 rounded-[10px] border px-3.5 py-3", t, className)}>
      <Icon className={cn("size-4 shrink-0 mt-0.5", ic)} />
      <div className="min-w-0 text-[13px]">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className={cn("soft", title && "mt-0.5")}>{children}</div>}
      </div>
    </div>
  );
}

export function Bar({ parts, height = 10 }: { parts: { value: number; className: string; label: string }[]; height?: number }) {
  const total = parts.reduce((a, p) => a + Math.max(p.value, 0), 0) || 1;
  return (
    <div className="flex w-full overflow-hidden rounded-full bg-bg-3" style={{ height }}>
      {parts.filter((p) => p.value > 0).map((p, i) => (
        <Tip key={i} content={`${p.label}: ${p.value.toLocaleString()}`}><div className={p.className} style={{ width: `${(p.value / total) * 100}%`, minWidth: 3 }} /></Tip>
      ))}
    </div>
  );
}

export const Skeleton = ({ className }: { className?: string }) => <div className={cn("rounded-md bg-bg-3 pulse", className)} />;

export function Empty({ title, children, icon: Icon }: { title: string; children?: React.ReactNode; icon?: any }) {
  return (
    <div className="flex h-full min-h-48 flex-col items-center justify-center gap-2 p-8 text-center">
      {Icon && <Icon className="size-6 text-faint" />}
      <div className="text-[15px] soft">{title}</div>
      {children && <div className="max-w-md text-[13px] dim">{children}</div>}
    </div>
  );
}

export function useHotkeys(map: Record<string, (e: KeyboardEvent) => void>, deps: any[] = []) {
  React.useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const typing = el?.tagName === "INPUT" || el?.tagName === "TEXTAREA" || el?.tagName === "SELECT" || el?.isContentEditable;
      const combo = `${e.metaKey || e.ctrlKey ? "mod+" : ""}${e.key.length === 1 ? e.key.toLowerCase() : e.key}`;
      const fn = map[combo];
      if (!fn) return;
      if (typing && !combo.startsWith("mod+") && e.key !== "Escape") return;
      fn(e);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, deps); // eslint-disable-line
}
