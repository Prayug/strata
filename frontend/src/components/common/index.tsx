import { useVirtualizer } from "@tanstack/react-virtual";
import { AlertTriangle, ArrowDown, ArrowUp, FlaskConical, Inbox, Loader2 } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/format";
import { Button, Tip } from "@/components/ui";
import { api, type JobState } from "@/lib/api";

export function Panel({ title, right, children, className, bodyClass, dense }: { title?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; className?: string; bodyClass?: string; dense?: boolean }) {
  return (
    <section className={cn("panel flex flex-col min-h-0 min-w-0", className)}>
      {(title || right) && (
        <header className={cn("flex items-center justify-between gap-2 px-2.5 border-b border-line shrink-0", dense ? "h-7" : "h-8")}>
          <h3 className="label text-fg-2 truncate">{title}</h3>
          <div className="flex items-center gap-1.5">{right}</div>
        </header>
      )}
      <div className={cn("min-h-0 flex-1", bodyClass)}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone, hint, className }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: string; hint?: string; className?: string }) {
  return (
    <div className={cn("px-2.5 py-1.5 min-w-0", className)}>
      <Tip content={hint}><div className="label truncate cursor-default">{label}</div></Tip>
      <div className={cn("num text-[15px] leading-5 mt-0.5 truncate", tone)}>{value}</div>
      {sub && <div className="num text-2xs text-muted truncate">{sub}</div>}
    </div>
  );
}

export function Empty({ title, children, icon: Icon = Inbox, action }: { title: string; children?: React.ReactNode; icon?: any; action?: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-32 flex-col items-center justify-center gap-2 p-6 text-center">
      <Icon className="size-5 text-faint" />
      <div className="text-sm text-fg-2">{title}</div>
      {children && <div className="max-w-md text-xs text-muted">{children}</div>}
      {action}
    </div>
  );
}

export function ErrorBox({ error, className }: { error: unknown; className?: string }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className={cn("flex gap-2 rounded-sm border border-down/40 bg-down/[0.06] px-2.5 py-2 text-xs text-fg", className)}>
      <AlertTriangle className="size-3.5 text-down shrink-0 mt-px" />
      <pre className="whitespace-pre-wrap font-sans">{msg}</pre>
    </div>
  );
}

export function Loading({ label = "Loading", className }: { label?: string; className?: string }) {
  return (
    <div className={cn("flex h-full min-h-24 items-center justify-center gap-2 text-xs text-muted", className)}>
      <Loader2 className="size-3.5 animate-spin" /> {label}
    </div>
  );
}

export function SyntheticBadge({ on, className }: { on?: boolean | number | null; className?: string }) {
  if (!on) return null;
  return (
    <Tip content="Results computed on the SYNTHETIC demonstration dataset (simulated prices, not market history).">
      <span className={cn("inline-flex items-center gap-1 rounded-[2px] border border-synth/50 bg-synth/10 px-1.5 h-4 text-[10px] font-semibold tracking-wider text-synth", className)}>
        <FlaskConical className="size-2.5" /> SYNTHETIC
      </span>
    </Tip>
  );
}

export function StatusDot({ status }: { status: string }) {
  const c = { done: "bg-up", running: "bg-accent animate-pulse", queued: "bg-muted", error: "bg-down", cancelled: "bg-faint" }[status] || "bg-faint";
  return <span className={cn("inline-block size-1.5 rounded-full", c)} title={status} />;
}

export function JobBar({ job, onCancel, className }: { job: JobState | null; onCancel?: () => void; className?: string }) {
  if (!job || !["queued", "running"].includes(job.status)) return null;
  const p = Math.round((job.progress || 0) * 100);
  return (
    <div className={cn("flex items-center gap-2 text-xs", className)}>
      <div className="relative h-1 w-40 overflow-hidden rounded-full bg-bg-4">
        <div className={cn("absolute inset-y-0 left-0 bg-accent transition-[width]", job.status === "queued" && "progress-indeterminate w-full bg-transparent")} style={{ width: job.status === "queued" ? undefined : `${p}%` }} />
      </div>
      <span className="num text-muted w-9">{job.status === "queued" ? "queued" : `${p}%`}</span>
      <span className="text-muted truncate max-w-64">{job.message}</span>
      {onCancel && <Button size="sm" variant="ghost" onClick={onCancel}>Cancel</Button>}
    </div>
  );
}

export const cancelJob = (id: string) => api.post(`/api/jobs/${id}/cancel`);

export function Pnl({ v, fmt, className }: { v: number | null | undefined; fmt: (x: number | null | undefined) => string; className?: string }) {
  return <span className={cn("num", v === null || v === undefined || v === 0 ? "text-fg-2" : v > 0 ? "text-up" : "text-down", className)}>{fmt(v)}</span>;
}

export type Col<T> = { key: string; label: React.ReactNode; render: (r: T) => React.ReactNode; sort?: (r: T) => number | string | null | undefined; align?: "r" | "l"; width?: number; title?: string };

/** Virtualised, sortable dense table. */
export function DataTable<T>({ rows, cols, rowKey, onRowClick, height = "100%", selected, defaultSort, className, rowClass }: {
  rows: T[]; cols: Col<T>[]; rowKey: (r: T) => string | number; onRowClick?: (r: T) => void; height?: number | string; selected?: string | number | null;
  defaultSort?: { key: string; dir: 1 | -1 }; className?: string; rowClass?: (r: T) => string;
}) {
  const [sort, setSort] = React.useState<{ key: string; dir: 1 | -1 } | null>(defaultSort || null);
  const sorted = React.useMemo(() => {
    if (!sort) return rows;
    const c = cols.find((x) => x.key === sort.key);
    if (!c?.sort) return rows;
    return [...rows].sort((a, b) => {
      const x = c.sort!(a), y = c.sort!(b);
      if (x === y) return 0;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x < y ? -1 : 1) * sort.dir;
    });
  }, [rows, sort, cols]);
  const parent = React.useRef<HTMLDivElement>(null);
  const v = useVirtualizer({ count: sorted.length, getScrollElement: () => parent.current, estimateSize: () => 25, overscan: 20 });
  const items = v.getVirtualItems();
  const padTop = items.length ? items[0].start : 0;
  const padBottom = items.length ? v.getTotalSize() - items[items.length - 1].end : 0;
  return (
    <div ref={parent} className={cn("overflow-auto min-h-0", className)} style={{ height }}>
      <table className="tbl">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} title={c.title} className={cn(c.align === "r" && "r", c.sort && "cursor-pointer hover:text-fg")} style={{ width: c.width }}
                onClick={() => c.sort && setSort((s) => (s?.key === c.key ? { key: c.key, dir: (-s.dir) as 1 | -1 } : { key: c.key, dir: -1 }))}>
                <span className="inline-flex items-center gap-0.5">{c.label}{sort?.key === c.key && (sort.dir > 0 ? <ArrowUp className="size-2.5" /> : <ArrowDown className="size-2.5" />)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {padTop > 0 && <tr><td colSpan={cols.length} style={{ height: padTop, padding: 0, border: 0 }} /></tr>}
          {items.map((it) => {
            const r = sorted[it.index];
            const k = rowKey(r);
            return (
              <tr key={k} onClick={() => onRowClick?.(r)} className={cn(onRowClick && "cursor-pointer", selected === k && "[&>td]:bg-accent/[0.08]", rowClass?.(r))}>
                {cols.map((c) => <td key={c.key} className={cn(c.align === "r" && "r")}>{c.render(r)}</td>)}
              </tr>
            );
          })}
          {padBottom > 0 && <tr><td colSpan={cols.length} style={{ height: padBottom, padding: 0, border: 0 }} /></tr>}
        </tbody>
      </table>
    </div>
  );
}

export function PageHeader({ title, children, sub }: { title: React.ReactNode; children?: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="flex h-10 shrink-0 items-center justify-between gap-3 border-b border-line bg-bg-1 px-3">
      <div className="flex items-center gap-2.5 min-w-0">
        <h1 className="text-sm font-semibold text-fg truncate">{title}</h1>
        {sub}
      </div>
      <div className="flex items-center gap-1.5">{children}</div>
    </div>
  );
}

export function Warnings({ items }: { items: { level?: string; text: string }[] }) {
  if (!items?.length) return null;
  return (
    <div className="flex flex-col gap-1">
      {items.map((w, i) => (
        <div key={i} className={cn("flex gap-2 rounded-sm border px-2 py-1.5 text-xs",
          w.level === "error" ? "border-down/40 bg-down/[0.06]" : w.level === "info" ? "border-line-2 bg-bg-2 text-fg-2" : "border-warn/40 bg-warn/[0.06]")}>
          <AlertTriangle className={cn("size-3.5 shrink-0 mt-px", w.level === "error" ? "text-down" : w.level === "info" ? "text-muted" : "text-warn")} />
          <span>{w.text}</span>
        </div>
      ))}
    </div>
  );
}
