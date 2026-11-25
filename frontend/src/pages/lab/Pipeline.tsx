/* Live progress of a research run: the seven pipeline stages, streamed from the job system. */
import { useQuery } from "@tanstack/react-query";
import { Check, CircleDashed, Loader2, Minus, OctagonAlert, Pause } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, queryClient, useJob } from "@/lib/api";
import { cn } from "@/lib/format";
import { fmtDur, type PipelineState, type Stage } from "@/lib/lab";

const FALLBACK: Stage[] = ["Interpreting strategy", "Checking historical coverage", "Identifying qualifying sessions", "Downloading missing data", "Validating contracts", "Running simulation", "Generating research report"]
  .map((label, i) => ({ id: String(i), label, status: "pending", detail: "", data: {} }));

/** Subscribes to a run's server-sent events and keeps its stage list fresh. */
export function usePipeline(runId: string | null | undefined, onDone?: () => void) {
  const done = useRef(onDone);
  done.current = onDone;
  const q = useQuery({
    queryKey: ["pipeline", runId],
    queryFn: () => api.get<PipelineState>(`/api/runs/${runId}/pipeline`),
    enabled: !!runId,
    refetchInterval: (s) => (s.state.data && ["queued", "running"].includes(s.state.data.status) ? 900 : false),
  });
  const live = !!runId && (!q.data || ["queued", "running"].includes(q.data.status));
  const job = useJob(live ? runId : null, () => {
    queryClient.invalidateQueries({ queryKey: ["pipeline", runId] });
    done.current?.();
  });
  // every progress event is a stage change or a new detail line: pull the stage list straight away
  const last = useRef("");
  useEffect(() => {
    if (job?.message && job.message !== last.current) {
      last.current = job.message;
      queryClient.invalidateQueries({ queryKey: ["pipeline", runId] });
    }
  }, [job?.message, runId]);
  return { state: q.data, job, live };
}

function Icon({ s }: { s: Stage["status"] }) {
  if (s === "done") return <span className="grid size-5 place-items-center rounded-full bg-up/15 text-up"><Check className="size-3" strokeWidth={3} /></span>;
  if (s === "running") return <span className="grid size-5 place-items-center rounded-full bg-accent/15 text-accent"><Loader2 className="size-3 animate-spin" strokeWidth={3} /></span>;
  if (s === "skipped") return <span className="grid size-5 place-items-center rounded-full bg-bg-3 text-faint"><Minus className="size-3" strokeWidth={3} /></span>;
  if (s === "blocked") return <span className="grid size-5 place-items-center rounded-full bg-warn/15 text-warn"><Pause className="size-3" strokeWidth={3} /></span>;
  if (s === "error") return <span className="grid size-5 place-items-center rounded-full bg-down/15 text-down"><OctagonAlert className="size-3" strokeWidth={3} /></span>;
  return <span className="grid size-5 place-items-center text-faint"><CircleDashed className="size-4" /></span>;
}

export function Timeline({ runId, onDone, compact, title }: { runId: string; onDone?: () => void; compact?: boolean; title?: string }) {
  const { state, job, live } = usePipeline(runId, onDone);
  const stages = state?.stages?.length ? state.stages : FALLBACK;
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setTick((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, [live]);
  const running = stages.find((s) => s.status === "running");
  const doneN = stages.filter((s) => ["done", "skipped"].includes(s.status)).length;
  const failed = state?.status === "error";
  if (compact && !live) {
    return (
      <div className="flex items-center gap-2 text-xs dim">
        {failed ? <OctagonAlert className="size-3.5 text-down" /> : <Check className="size-3.5 text-up" />}
        {failed ? (state?.error || "the run failed") : state?.outcome === "needs_decision" ? "stopped: your decision is needed" : `${stages.filter((s) => s.status === "done").length} stages completed in ${fmtDur(state?.elapsed_seconds)}`}
      </div>
    );
  }
  return (
    <div className="fade-in" data-testid="pipeline">
      {title && <div className="eyebrow mb-2">{title}</div>}
      <div className="mb-2.5 flex items-center gap-3">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-3">
          <div className={cn("h-full rounded-full transition-[width] duration-500", failed ? "bg-down" : "bg-accent")} style={{ width: `${Math.max(4, (doneN / stages.length) * 100 + (running ? 6 : 0))}%` }} />
        </div>
        <span className="num text-xs dim w-16 text-right" key={tick}>{live ? fmtDur(state?.elapsed_seconds ?? 0) : fmtDur(state?.elapsed_seconds)}</span>
      </div>
      <ol className="relative flex flex-col">
        {stages.map((s, i) => (
          <li key={s.id} className="relative flex gap-3 pb-2.5 last:pb-0">
            {i < stages.length - 1 && <span className={cn("absolute left-[9.5px] top-5 bottom-0 w-px", ["done", "skipped"].includes(s.status) ? "bg-up/30" : "bg-line-2")} />}
            <Icon s={s.status} />
            <div className="min-w-0 flex-1">
              <div className={cn("text-[13px] leading-5", s.status === "pending" ? "dim" : s.status === "running" ? "text-fg font-medium" : "soft")}>
                {s.label}
                {s.started !== undefined && s.finished !== undefined && s.finished - s.started >= 0.4 && <span className="ml-2 num text-[11px] text-faint">{fmtDur(s.finished - s.started)}</span>}
              </div>
              {s.detail && s.status !== "pending" && <div className={cn("text-xs leading-[1.45]", s.status === "error" ? "text-down" : s.status === "blocked" ? "text-warn" : "dim", s.status === "running" && "pulse")}>{s.detail}</div>}
            </div>
          </li>
        ))}
      </ol>
      {live && job?.status === "queued" && <div className="mt-2 text-xs dim">Waiting for a free worker…</div>}
      {failed && state?.error && <div className="mt-2 text-xs text-down">{state.error}</div>}
    </div>
  );
}
