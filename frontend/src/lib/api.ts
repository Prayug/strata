import { QueryClient, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import type { Meta, RunRow } from "./types";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, staleTime: 15_000, retry: 1 } },
});

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(method: string, path: string, body?: unknown, isForm = false): Promise<T> {
  const init: RequestInit = { method, headers: isForm ? undefined : { "content-type": "application/json" } };
  if (body !== undefined) init.body = isForm ? (body as FormData) : JSON.stringify(body);
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Cannot reach the Strata API. Is the backend running? (strata serve)");
  }
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    let msg = typeof data === "object" && data?.detail ? data.detail : typeof data === "string" ? data : res.statusText;
    if (Array.isArray(msg)) msg = msg.map((e: any) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
    throw new ApiError(res.status, String(msg).slice(0, 600));
  }
  return data as T;
}

export const api = {
  get: <T,>(p: string) => req<T>("GET", p),
  post: <T,>(p: string, b?: unknown) => req<T>("POST", p, b ?? {}),
  put: <T,>(p: string, b?: unknown) => req<T>("PUT", p, b ?? {}),
  del: <T,>(p: string) => req<T>("DELETE", p),
  form: <T,>(p: string, f: FormData) => req<T>("POST", p, f, true),
};

export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: () => api.get<Meta>("/api/meta"), staleTime: Infinity });
}

export function useRun(id: string | undefined, poll = false) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: () => api.get<RunRow>(`/api/runs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (poll && q.state.data && ["queued", "running"].includes(q.state.data.status) ? 2000 : false),
  });
}

export interface JobState { id: string; status: string; progress: number; message?: string; error?: string }

/** Subscribe to server-sent progress events for a background job. */
export function useJob(id: string | undefined | null, onDone?: (s: JobState) => void) {
  const [state, setState] = useState<JobState | null>(null);
  const cb = useRef(onDone);
  cb.current = onDone;
  useEffect(() => {
    if (!id) {
      setState(null);
      return;
    }
    const es = new EventSource(`/api/jobs/${id}/stream`);
    es.onmessage = (ev) => {
      const s = JSON.parse(ev.data) as JobState;
      setState((prev) => ({ ...(prev || {}), ...s, id }));
      if (["done", "error", "cancelled"].includes(s.status)) {
        es.close();
        queryClient.invalidateQueries({ queryKey: ["run", id] });
        queryClient.invalidateQueries({ queryKey: ["runs"] });
        queryClient.invalidateQueries({ queryKey: ["jobs"] });
        cb.current?.({ ...s, id });
      }
    };
    es.onerror = () => {
      es.close();
      queryClient.invalidateQueries({ queryKey: ["run", id] });
    };
    return () => es.close();
  }, [id]);
  return state;
}
