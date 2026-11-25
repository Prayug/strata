/* Types, labels and data hooks of the research workspace (AI Lab). */
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export type Mode = "study" | "historical" | "trades" | "modeled" | "auto";
export interface PlanItem { label: string; value: string; status: "stated" | "assumed" | "missing" | "unsupported"; field?: string | null }
export interface RunRef { run_id: string; mode: Mode; label: string; cached?: boolean; draft?: any; approvals?: any }
export interface Question { id: string; text: string; options: { label: string; patch: any; detail?: string }[]; default: number | null; answered?: number | null; hint?: string; assumed?: string }
export interface AiMessage { id: number; session_id: string; role: "user" | "assistant"; kind: string; content: any; created_at: string }
export interface AiVersion { id: number; parent_id: number | null; label: string; draft: any; runs: RunRef[]; created_at: string }
export interface AiSession { id: string; title: string; state: any; created_at: string; updated_at: string }
export interface AiPayload { session: AiSession; messages: AiMessage[]; versions: AiVersion[]; current_version: number | null; exec_mode: "quick" | "review"; trials: number; pinned: string[] }
export interface Fact { key: string; label: string; value: number; unit: string; shown: string }
export interface Stage { id: string; label: string; status: "pending" | "running" | "done" | "skipped" | "blocked" | "error"; detail: string; data: any; started?: number; finished?: number }
export interface PipelineState { run_id: string; stages: Stage[]; outcome: string | null; decision?: any; downloads: any[]; status: string; progress?: number; message?: string; error?: string | null; elapsed_seconds?: number }

export const MODE_NAME: Record<string, string> = {
  study: "Market study", historical: "Options · bid/ask quotes", trades: "Options · trade prices", modeled: "Options · modeled premiums", auto: "Options backtest", discovery: "Exploration",
};
export const MODE_HELP: Record<string, string> = {
  study: "Measures how the underlying itself behaved. No option prices are involved.",
  historical: "Fills use observed option bid/ask quotes.",
  trades: "Fills use observed option trade prints. This data has no bid/ask; a print is only used while it is recent.",
  modeled: "Option premiums are THEORETICAL estimates from a pricing model, not observed prices.",
  auto: "Strata picks the kind of observed option price the data can supply. Theoretical premiums are never picked automatically.",
};
export const MODE_TONE: Record<string, string> = {
  study: "text-info border-info/40 bg-info/10", historical: "text-up border-up/40 bg-up/10", trades: "text-[#56b6c2] border-[#56b6c2]/40 bg-[#56b6c2]/10", modeled: "text-warn border-warn/50 bg-warn/10",
  auto: "text-fg-2 border-line-2 bg-bg-2", discovery: "text-synth border-synth/40 bg-synth/10",
};

/** Plain-language explanations of the jargon the app cannot avoid. */
export const GLOSSARY: Record<string, string> = {
  DTE: "Days to expiration: how many days until the option contract expires. 0DTE expires the same day.",
  "0DTE": "An option that expires on the day it is bought.",
  ATM: "At the money: the strike price closest to where the underlying is trading.",
  delta: "Roughly how much the option's price moves for a $1 move in the underlying (0 to 1).",
  IV: "Implied volatility: how large a move the option market is pricing in.",
  VWAP: "Volume-weighted average price of the session so far.",
  baseline: "How often the same outcome happened on every session, whatever the condition. The comparison that tells you whether the condition matters.",
  "confidence interval": "The range the true frequency plausibly lies in, given how many observations there are. Fewer observations, wider range.",
  "p-value": "The chance of seeing a difference this large if the condition actually made no difference. Small (below 0.05) means it is unlikely to be luck.",
  "out of sample": "A later period that was not used when the idea was formed. If the result only holds in the earlier period, it was probably luck.",
  premium: "The price paid for the option.",
  slippage: "The difference between the price you see and the price you actually get.",
  "trade print": "A price at which the option actually traded. Unlike a quote, it does not tell you what you could have bought or sold at later.",
  "bid/ask": "The prices at which you could sell (bid) and buy (ask) at that moment.",
  "qualifying session": "A trading day on which your conditions were met.",
};

export const STATUS_NAME: Record<string, string> = {
  target: "Target reached", stop: "Stop reached", win: "Profitable trade", loss: "Losing trade", neither: "Qualified, neither level", ambiguous: "Ambiguous order", skipped: "Qualified, not executed",
  missing: "Missing data", none: "Did not qualify",
};

export const fmtDur = (s?: number) => (s === undefined || s === null ? "" : s < 1 ? "<1s" : s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
export const hhmmOf = (minute: number) => `${String(Math.floor(minute / 60)).padStart(2, "0")}:${String(minute % 60).padStart(2, "0")}`;
export const fact = (facts: Fact[] | undefined, key: string) => facts?.find((f) => f.key === key);
export const pp = (x: number | null | undefined, dp = 1) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${(x * 100).toFixed(dp)}%`);
export const usd = (x: number | null | undefined, dp = 0) =>
  x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x < 0 ? "-" : ""}$${Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
export const sgn = (x: number | null | undefined, dp = 2, suffix = "") => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(dp)}${suffix}`);

export function useAiMeta() {
  return useQuery({ queryKey: ["ai-meta"], queryFn: () => api.get<any>("/api/ai/meta"), staleTime: 30_000 });
}
export function useRunResult(id: string | null | undefined) {
  return useQuery({ queryKey: ["run", id], queryFn: () => api.get<any>(`/api/runs/${id}`), enabled: !!id, staleTime: 60_000 });
}

/** The newest message that carries a result for a run (single / compare views). */
export function resultMessages(messages: AiMessage[]) {
  return messages.filter((m) => m.kind === "result" && m.content?.run_id);
}
