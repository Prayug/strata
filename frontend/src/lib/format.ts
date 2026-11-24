import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

export function money(x: number | null | undefined, dp = 0, sign = false): string {
  if (!isNum(x)) return "—";
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const pre = x < 0 ? "-" : sign && x > 0 ? "+" : "";
  return `${pre}$${s}`;
}
export function pct(x: number | null | undefined, dp = 1, sign = false): string {
  if (!isNum(x)) return "—";
  const v = x * 100;
  return `${sign && v > 0 ? "+" : ""}${v.toFixed(dp)}%`;
}
export function num(x: number | null | undefined, dp = 2, sign = false): string {
  if (!isNum(x)) return "—";
  return `${sign && x > 0 ? "+" : ""}${x.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
}
export function compact(x: number | null | undefined): string {
  if (!isNum(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return `${(x / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(x / 1e3).toFixed(1)}k`;
  return x.toFixed(0);
}
export function bytes(x: number | null | undefined): string {
  if (!isNum(x)) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = x;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`;
}
export function tone(x: number | null | undefined): string {
  if (!isNum(x) || x === 0) return "text-fg-2";
  return x > 0 ? "text-up" : "text-down";
}
export function etTime(ts: number | string | null | undefined, withDate = true): string {
  if (ts === null || ts === undefined) return "—";
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  const opts: Intl.DateTimeFormatOptions = { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false };
  if (withDate) Object.assign(opts, { year: "2-digit", month: "2-digit", day: "2-digit" });
  return d.toLocaleString("en-US", opts).replace(",", "");
}
export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.max(1, Math.floor(s))}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
export function minutesLabel(m: number | null | undefined): string {
  if (!isNum(m)) return "—";
  if (m < 60) return `${m.toFixed(0)}m`;
  if (m < 60 * 24) return `${(m / 60).toFixed(1)}h`;
  return `${(m / 1440).toFixed(1)}d`;
}
export const uid = (p = "n") => `${p}${Math.random().toString(36).slice(2, 8)}`;
export const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));
export const REASON_LABEL: Record<string, string> = {
  profit_target: "Target", stop_loss: "Stop", trailing_stop: "Trail stop", time_exit: "Time", dte_exit: "DTE", max_hold: "Max hold",
  underlying_stop: "Underlying stop", underlying_target: "Underlying tgt", strike_breach: "Strike breach", greek_stop: "Greek stop",
  custom_exit: "Custom", roll: "Roll", expiration: "Expiry", assigned: "Assigned", exercised: "Exercised", margin_call: "Margin call",
  end_of_backtest: "End", leg_stop: "Leg stop", leg_target: "Leg target",
};
