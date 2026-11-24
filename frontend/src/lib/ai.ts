/* Types and helpers for the AI research lab and market studies. */
export type Mode = "study" | "historical" | "modeled";
export interface PlanItem { label: string; value: string; status: "stated" | "assumed" | "missing"; field?: string | null }
export interface RunRef { run_id: string; mode: Mode; label: string; cached?: boolean; draft?: any }
export interface ModeInfo { ok: boolean; reason?: string; note?: string; window?: [string, string]; resolution?: string }
export interface Coverage { modes: Record<Mode, ModeInfo>; recommended: Mode | null; messages: string[]; profile?: any; symbol?: string }
export interface AiMessage { id: number; session_id: string; role: "user" | "assistant"; kind: string; content: any; created_at: string }
export interface AiVersion { id: number; parent_id: number | null; label: string; draft: any; runs: RunRef[]; created_at: string }
export interface AiSession { id: string; title: string; state: any; created_at: string; updated_at: string }
export interface AiPayload { session: AiSession; messages: AiMessage[]; versions: AiVersion[]; current_version: number | null }
export interface Fact { key: string; label: string; value: number; unit: string; shown: string }

export const MODE_LABEL: Record<Mode, string> = { study: "Market study", historical: "Historical options", modeled: "Modeled options" };
export const MODE_LONG: Record<Mode, string> = {
  study: "Historical market study - underlying prices only, no option premiums",
  historical: "Historical options backtest - fills use observed option quotes",
  modeled: "Modeled options backtest - premiums are theoretical estimates, not observed prices",
};
export const MODE_TONE: Record<Mode, string> = { study: "border-info/50 text-info bg-info/10", historical: "border-up/50 text-up bg-up/10", modeled: "border-warn/60 text-warn bg-warn/10" };

/** Minutes since the session open rendered as a New York clock time. */
export const hhmm = (minutesFromOpen: number) => {
  const m = 570 + Math.round(minutesFromOpen);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
};
export const fact = (facts: Fact[] | undefined, key: string) => facts?.find((f) => f.key === key);
