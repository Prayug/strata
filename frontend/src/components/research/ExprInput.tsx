/* Rule-expression input with live validation and a searchable function reference.
   The same registry drives the strategy builder, the market explorer and the AI lab. */
import { BookOpen } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Input, Popover, PopoverContent, PopoverTrigger } from "@/components/ui";
import { api, useMeta } from "@/lib/api";
import { cn } from "@/lib/format";

export interface ExprDoc { name: string; signature: string; description: string; category: string; needs: string }

export function ExprReference({ onInsert }: { onInsert: (sig: string) => void }) {
  const meta = useMeta().data as any;
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    const docs: ExprDoc[] = meta?.expressions || [];
    const hit = docs.filter((d) => !q || `${d.name} ${d.description} ${d.category}`.toLowerCase().includes(q.toLowerCase()));
    const by: Record<string, ExprDoc[]> = {};
    for (const d of hit) (by[d.category] ||= []).push(d);
    return by;
  }, [meta, q]);
  return (
    <Popover>
      <PopoverTrigger asChild><Button size="iconSm" variant="ghost" title="Function reference"><BookOpen /></Button></PopoverTrigger>
      <PopoverContent align="end" className="w-[440px] p-0">
        <div className="border-b border-line p-1.5"><Input sz="sm" autoFocus placeholder="Search names and functions: gap, vwap, consecutive, previous…" value={q} onChange={(e) => setQ(e.target.value)} /></div>
        <div className="max-h-[46vh] overflow-auto">
          {Object.entries(groups).map(([cat, docs]) => (
            <div key={cat}>
              <div className="label sticky top-0 bg-bg-2 px-2 py-1">{cat}</div>
              {docs.map((d) => (
                <button key={d.name} onClick={() => onInsert(d.signature)} className="block w-full px-2 py-1 text-left hover:bg-bg-3">
                  <div className="flex items-center gap-2"><span className="font-mono text-2xs text-accent">{d.signature}</span>{d.needs === "intraday" && <span className="text-[9.5px] uppercase tracking-wider text-info">intraday bars</span>}</div>
                  <div className="text-2xs leading-snug text-muted">{d.description}</div>
                </button>
              ))}
            </div>
          ))}
          {Object.keys(groups).length === 0 && <div className="p-3 text-xs text-muted">No match.</div>}
        </div>
        <div className="border-t border-line px-2 py-1 text-2xs text-faint">Combine with and / or / not and comparisons, e.g. <span className="font-mono text-fg-2">previous_session.return_pct &lt;= -3 and gap_pct &lt; 0</span>. Missing inputs exclude the session; they are never read as "false".</div>
      </PopoverContent>
    </Popover>
  );
}

export function ExprInput({ value, onChange, className, placeholder, onValid }: { value: string; onChange: (v: string) => void; className?: string; placeholder?: string; onValid?: (ok: boolean) => void }) {
  const [err, setErr] = useState<string | null>(null);
  const [intraday, setIntraday] = useState(false);
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!value.trim()) { setErr(null); onValid?.(false); return; }
    const h = setTimeout(async () => {
      try {
        const r = await api.post<any>("/api/expr/validate", { expr: value });
        setErr(r.valid ? null : r.errors.join("; "));
        setIntraday(!!r.needs_intraday_bars);
        onValid?.(r.valid);
      } catch { /* offline: leave as is */ }
    }, 280);
    return () => clearTimeout(h);
  }, [value]); // eslint-disable-line
  const insert = (sig: string) => {
    const el = ref.current;
    const pos = el?.selectionStart ?? value.length;
    onChange(value.slice(0, pos) + (value && !/[\s(]$/.test(value.slice(0, pos)) ? " " : "") + sig + value.slice(pos));
    setTimeout(() => el?.focus(), 0);
  };
  return (
    <div className={cn("flex flex-col gap-0.5", className)}>
      <div className="flex items-center gap-1">
        <Input ref={ref} sz="sm" spellCheck={false} className={cn("font-mono text-2xs", err && "border-down/60")} value={value} placeholder={placeholder || "previous_session.return_pct <= -3"} onChange={(e) => onChange(e.target.value)} />
        <ExprReference onInsert={insert} />
      </div>
      {err ? <span className="text-2xs text-down">{err}</span> : intraday ? <span className="text-2xs text-info">needs intraday bars - sessions with only daily bars are excluded and reported</span> : null}
    </div>
  );
}
