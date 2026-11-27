/* Data of the V8 notebook: one research session (its messages, versions, state), what is in flight, and the
   helpers that open the observations behind a number. Everything is computed by the server; this file only asks. */
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { api, queryClient } from "@/lib/api";
import type { AiMessage, AiPayload } from "@/lib/lab";

export type Level = "answer" | "explore" | "technical";

const PENDING = ["plan", "sweep", "run", "investigation", "discovery", "hypothesis", "stress", "challenge", "graph", "experiment"];
export const isPending = (m: AiMessage) => PENDING.includes(m.kind) && m.content?.runs?.length > 0 && !m.content.finalized;

export function useNotebook(sid: string) {
  const q = useQuery({ queryKey: ["ai", sid], queryFn: () => api.get<AiPayload>(`/api/ai/sessions/${sid}`) });
  const messages = q.data?.messages || [];
  const pending = messages.some(isPending);
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["ai", sid] });
    queryClient.invalidateQueries({ queryKey: ["v8-history"] });
  };
  const act = useMutation({
    mutationFn: (fn: () => Promise<any>) => fn(),
    onSuccess: () => refresh(),
    onError: (e: any) => toast.error(e.message),
  });
  const send = (text: string) => act.mutate(() => api.post(`/api/ai/sessions/${sid}/messages`, { text }));
  // finished runs become result messages on the server: ask while anything is in flight
  const busy = useRef(false);
  useEffect(() => {
    if (!pending) return;
    let stop = false;
    const tick = async () => {
      if (busy.current) return;
      busy.current = true;
      try {
        const r = await api.post<{ messages: AiMessage[] }>(`/api/ai/sessions/${sid}/finalize`);
        if (!stop && r.messages.length) refresh();
      } catch { /* next tick */ } finally { busy.current = false; }
    };
    const h = setInterval(tick, 1000);
    tick();
    return () => { stop = true; clearInterval(h); };
  }, [pending, sid]); // eslint-disable-line
  return { q, data: q.data, messages, pending, send, act, refresh };
}

export async function startNotebook(text: string): Promise<string> {
  const s = await api.post<any>("/api/ai/sessions", { title: text.slice(0, 70) });
  sessionStorage.setItem(`v8-first-${s.id}`, text);
  return `/ask/${s.id}`;
}

export function useLevel(): [Level, (l: Level) => void] {
  const key = "strata.v8.level";
  const read = (): Level => { try { return (localStorage.getItem(key) as Level) || "answer"; } catch { return "answer"; } };
  const q = useQuery({ queryKey: ["v8-level"], queryFn: read, initialData: read, staleTime: Infinity });
  const set = (l: Level) => { try { localStorage.setItem(key, l); } catch { /* not remembered in private mode */ } queryClient.setQueryData(["v8-level"], l); };
  return [q.data as Level, set];
}

export interface ObsTable { kind: string; title: string; count: number; total?: number; columns: { key: string; label: string; unit?: string }[]; rows: any[]; caption?: string; replay?: any; symbol?: string; sum?: any }
export function useObservations(spec: any | null) {
  return useQuery({
    queryKey: ["v8-obs", JSON.stringify(spec)],
    queryFn: () => api.post<ObsTable>("/api/v8/explore", { spec }),
    enabled: !!spec && spec.kind !== "replay",
    staleTime: 5 * 60_000,
  });
}

/** The text of a user message, whatever shape its content has. */
export const said = (m: AiMessage) => (typeof m.content === "string" ? m.content : m.content?.text) || "";

/** Turns: a question and everything the assistant answered to it, in order. */
export function turns(messages: AiMessage[]) {
  const out: { q: AiMessage | null; a: AiMessage[] }[] = [];
  for (const m of messages) {
    if (m.role === "user") out.push({ q: m, a: [] });
    else if (out.length) out[out.length - 1].a.push(m);
    else out.push({ q: null, a: [m] });
  }
  return out;
}

export const fmtPct = (x: number | null | undefined, d = 1) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${(x * 100).toFixed(d)}%`);
export const fmtSigned = (x: number | null | undefined, d = 2, unit = "%") => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}${unit}`);
export const fmtUsd = (x: number | null | undefined, d = 0) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x < 0 ? "−" : x > 0 ? "+" : ""}$${Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`);
export const fmtDate = (s?: string | null) => {
  if (!s) return "";
  const d = new Date(`${s.slice(0, 10)}T12:00:00`);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
};
