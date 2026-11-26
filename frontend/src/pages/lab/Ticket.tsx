/* The reading of the request, as a ticket under the command line.

   Every field the request was read into is one cell: what the user said is plain, what was assumed is marked, what could
   not be represented is red. A cell opens the editor on that reading. A detail whose meaning is open (a target that could be
   the underlying or the premium, two readers that disagree) is asked here, in one row, before anything runs. */
import { Check, HelpCircle, Pencil, Play, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/format";
import { MODE_NAME } from "@/lib/lab";

type Item = { label: string; value: any; status: string };
type Q = { id: string; text: string; options: { label: string; detail?: string }[]; answered?: number | null; hint?: string };
const STATE: Record<string, [string, string]> = { awaiting_answer: ["waiting for your answer", "text-accent"], awaiting_confirmation: ["waiting for your go-ahead", "text-accent"], blocked: ["cannot run yet", "text-down"],
  running: ["running", "text-accent"], done: ["ran", "text-up"], needs_decision: ["the data cannot cover it", "text-warn"] };

/** The value as one glance: the reading itself, without the remark that follows a dash (the whole text is the tooltip). */
function brief(v: any): string {
  const s = String(v).split("   [")[0];
  let depth = 0;
  for (let i = 0; i < s.length - 2; i++) {
    if (s[i] === "(") depth++;
    else if (s[i] === ")") depth--;
    else if (depth === 0 && i > 8 && s.slice(i, i + 3) === " - ") return s.slice(0, i);
  }
  return s;
}

export function Ticket({ c, busy, ran, onEdit, onAnswer, onConfirm, rail }: { c: any; busy: boolean; ran?: boolean; onEdit: () => void; onAnswer: (versionId: number, qid: string, option: number) => void; onConfirm: (versionId: number) => void; rail?: React.ReactNode }) {
  const items: Item[] = c.items || [];
  const open: Q[] = (c.questions || []).filter((q: Q) => q.answered === undefined || q.answered === null);
  const assumed = items.filter((x) => x.status === "assumed").length;
  // "the data cannot cover it" is where the plan stopped to ask; after the answer and the run it is no longer where things stand
  const st = c.state === "needs_decision" && ran ? (["ran on what the data covers", "text-up"] as [string, string]) : STATE[c.state] || (c.finalized ? STATE.done : null);
  if (!items.length) return null;
  return (
    <div className="ticket shrink-0" data-testid="ticket">
      <div className="flex items-stretch">
        <div className="flex w-[148px] shrink-0 flex-col justify-center border-r border-line bg-bg-1 px-3.5 py-1.5">
          <div className="text-[10px] uppercase tracking-[0.08em] text-muted">Read as</div>
          <div className="text-[12.5px] font-medium leading-tight">{(MODE_NAME[c.mode || (c.kind === "study" ? "study" : "auto")] || "Strategy").replace("Options · ", "")}</div>
          <div className="num mt-0.5 truncate text-[10.5px] text-faint" title={`version ${c.version_id}, read by ${c.interpreter}`}>v{c.version_id} · {c.interpreter}</div>
        </div>
        <div className="flex min-w-0 flex-1 overflow-x-auto" style={{ scrollbarWidth: "thin" }} onClick={onEdit} title="Open the reading to change it">
          {items.map((it, i) => (
            <div key={i} className={cn("cell max-w-[236px] shrink-0", it.status === "assumed" && "assumed", (it.status === "unsupported" || it.status === "missing") && "bad")} data-status={it.status}>
              <div className="k">{it.label}{it.status === "assumed" ? " · assumed" : it.status === "missing" ? " · missing" : it.status === "unsupported" ? " · not represented" : ""}</div>
              <div className="v" title={String(it.value).split("   [")[0]}>{brief(it.value)}</div>
            </div>
          ))}
        </div>
        <div className="flex shrink-0 items-center gap-1.5 border-l border-line bg-bg-1 px-3">
          {st && <span className={cn("hidden whitespace-nowrap text-[11.5px] xl:inline", st[1])}>{st[0]}</span>}
          {c.state === "awaiting_confirmation" && open.length === 0 && <button className="btn primary sm !rounded-[3px]" disabled={busy} onClick={() => onConfirm(c.version_id)} data-testid="ticket-run"><Play />{(c.unsupported || []).length ? "Run without that part" : "Run"}</button>}
          <button className="btn sm !rounded-[3px]" onClick={onEdit} data-testid="ticket-edit"><Pencil />Adjust</button>
        </div>
      </div>
      {open.map((q) => (
        <div key={q.id} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-accent/40 bg-accent/[0.06] px-3.5 py-2" data-testid="ticket-question">
          <HelpCircle className="size-4 shrink-0 text-accent" />
          <span className="min-w-0 text-[13px] font-medium">{q.text}</span>
          <span className="flex flex-wrap gap-1.5">{q.options.map((o, i) => <button key={i} className="btn sm !rounded-[3px]" disabled={busy} title={o.detail} onClick={() => onAnswer(c.version_id, q.id, i)}>{o.label}</button>)}</span>
          {q.options[0]?.detail && <span className="w-full pl-7 text-[11.5px] dim">{q.options.map((o) => `${o.label}: ${o.detail}`).join(" · ")}</span>}
        </div>
      ))}
      {(c.unsupported || []).length > 0 && (
        <div className="flex items-start gap-2 border-t border-down/40 bg-down/[0.06] px-3.5 py-1.5 text-[12.5px]" data-testid="ticket-unsupported"><ShieldAlert className="mt-0.5 size-4 shrink-0 text-down" />
          <span><span className="font-medium">Not represented:</span> {c.unsupported.map((u: string) => `"${u}"`).join("; ")}. Nothing was put in its place.</span></div>
      )}
      {(rail || assumed > 0) && (
        <div className="flex items-center gap-3 border-t border-line bg-bg-1/60 px-3 py-0.5 text-[11.5px] dim">
          <div className="min-w-0 flex-1">{rail}</div>
          {assumed > 0 ? <button className="shrink-0 text-warn hover:underline" onClick={onEdit} title={items.filter((x) => x.status === "assumed").map((x) => `${x.label}: ${String(x.value).split("   [")[0]}`).join("\n")} data-testid="ticket-assumed">{assumed} assumed</button>
            : <span className="flex shrink-0 items-center gap-1 text-up"><Check className="size-3" />nothing assumed</span>}
        </div>
      )}
    </div>
  );
}
