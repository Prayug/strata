import { create } from "zustand";
import { persist } from "zustand/middleware";
import { clone } from "./format";
import type { BacktestConfig, Capital, CondNode, Execution, ExitRules, StrategyConfig } from "./types";

export const DEFAULT_EXITS: ExitRules = {
  profit_target: { enabled: true, mode: "pct", value: 50 },
  stop_loss: { enabled: true, mode: "pct", value: 200 },
  trailing_stop: { enabled: false, mode: "pct", activate: 25, trail: 15 },
  time_exit: { enabled: false, time: "15:45", dte_lte: 0 },
  dte_exit: { enabled: false, dte: 1, time: "09:45" },
  max_hold: { enabled: false, minutes: null, days: null },
  underlying: { enabled: false, move_against_pct: null, move_for_pct: null, strike_breach: false, breach_buffer_pct: 0 },
  greeks: { enabled: false, max_abs_delta: null, short_leg_delta: null },
  leg_rules: [],
  custom: null,
  roll: { enabled: false, trigger: "dte", value: 1, max_rolls: 3 },
  trigger_price: "close_price",
};
export const DEFAULT_CAPITAL: Capital = {
  initial_capital: 100000, margin_model: "regt", max_bp_usage_pct: 100, max_concurrent_positions: 50,
  risk_free_rate: 0, cash_interest_rate: 0, cash_benchmark_rate: 0.04, liquidate_on_margin_call: true,
};
export const DEFAULT_EXECUTION: Execution = {
  fill_model: "conservative", custom_fraction: 0.5, slippage_per_contract: 0, commission_per_contract: 0.65, fee_per_contract: 0,
  commission_per_order: 0, stock_commission_per_share: 0, stock_slippage_bps: 2, exercise_fee: 0, participation_rate: 1, max_volume_pct: null,
  allow_partial_fills: true, max_quote_age_seconds: 120, early_assignment: true, assignment_extrinsic_threshold: 0.05, physical_settlement: "close_next_open",
};

export function rootGroup(children: CondNode[] = []): CondNode {
  return { id: "root", kind: "group", op: "and", children, enabled: true, negate: false };
}

/** Fill in every field so editors never meet undefined (templates only specify a subset). */
export function normalizeStrategy(s: Partial<StrategyConfig>): StrategyConfig {
  const exits = { ...clone(DEFAULT_EXITS), ...(s.exits || {}) } as ExitRules;
  for (const k of Object.keys(DEFAULT_EXITS) as (keyof ExitRules)[]) {
    const d = (DEFAULT_EXITS as any)[k];
    if (d && typeof d === "object" && !Array.isArray(d)) (exits as any)[k] = { ...d, ...((s.exits as any)?.[k] || {}) };
  }
  const withIds = (n: CondNode): CondNode => ({ ...n, id: n.id || Math.random().toString(36).slice(2, 8), enabled: n.enabled ?? true, children: (n.children || []).map(withIds) });
  return {
    name: s.name || "Untitled strategy",
    symbol: s.symbol || "DEMO",
    legs: (s.legs || []).map((l) => ({ kind: "option", dte_mode: "nearest", min_bid: 0, ...l, quantity: l.quantity ?? 1, strike: { unit: "points", ref_leg: null, ...l.strike } })),
    entry: {
      conditions: withIds(s.entry?.conditions || rootGroup()),
      max_entries_per_day: s.entry?.max_entries_per_day ?? 1, cooldown_minutes: s.entry?.cooldown_minutes ?? 0,
      max_open_positions: s.entry?.max_open_positions ?? 1, reentry_after: s.entry?.reentry_after ?? ["any"],
      min_credit: s.entry?.min_credit ?? null, max_debit: s.entry?.max_debit ?? null,
    },
    exits: { ...exits, custom: exits.custom ? withIds(exits.custom) : null },
    sizing: { method: "fixed", contracts: 1, pct: 2, max_contracts: 100, ...(s.sizing || {}) },
    allocation: s.allocation ?? 1,
    vol_symbol: s.vol_symbol ?? null,
    regime: { vix_low: 15, vix_high: 25, trend_fast: 50, trend_slow: 200, ...(s.regime || {}) },
  };
}

const STARTER = normalizeStrategy({
  name: "0DTE put credit spread",
  symbol: "DEMO",
  legs: [
    { id: "sp", side: "sell", right: "P", quantity: 1, dte: 0, strike: { method: "delta", value: 0.15 } },
    { id: "lp", side: "buy", right: "P", quantity: 1, dte: 0, strike: { method: "leg_offset", value: -5, ref_leg: "sp", unit: "points" } },
  ],
  entry: {
    conditions: rootGroup([
      { id: "c1", kind: "cond", type: "entry_times", params: { times: "10:00" } },
      { id: "c2", kind: "cond", type: "vix", params: { cmp: "<", value: 30 } },
    ]),
  } as any,
});

interface BuilderState {
  strategy: StrategyConfig;
  strategyId: string | null;
  dirty: boolean;
  start: string;
  end: string;
  capital: Capital;
  execution: Execution;
  previewDay: string | null;
  previewTime: string;
  setStrategy: (fn: (s: StrategyConfig) => void) => void;
  load: (s: StrategyConfig, id: string | null) => void;
  patch: (p: Partial<Pick<BuilderState, "start" | "end" | "capital" | "execution" | "previewDay" | "previewTime" | "strategyId" | "dirty">>) => void;
  backtestConfig: () => BacktestConfig;
}

export const useBuilder = create<BuilderState>()(
  persist(
    (set, get) => ({
      strategy: STARTER,
      strategyId: null,
      dirty: false,
      start: "2023-03-01",
      end: "2024-12-31",
      capital: DEFAULT_CAPITAL,
      execution: DEFAULT_EXECUTION,
      previewDay: null,
      previewTime: "10:00",
      setStrategy: (fn) => {
        const s = clone(get().strategy);
        fn(s);
        set({ strategy: s, dirty: true });
      },
      load: (s, id) => set({ strategy: normalizeStrategy(clone(s)), strategyId: id, dirty: false }),
      patch: (p) => set(p as any),
      backtestConfig: () => {
        const g = get();
        return { strategies: [g.strategy], start: g.start, end: g.end, capital: g.capital, execution: g.execution, allow_synthetic: true, label: g.strategy.name };
      },
    }),
    { name: "strata-builder-v1" },
  ),
);

interface UiState {
  navCollapsed: boolean;
  lastRunId: string | null;
  returnMode: "dollars" | "pct" | "risk";
  set: (p: Partial<Omit<UiState, "set">>) => void;
}
export const useUi = create<UiState>()(
  persist((set) => ({ navCollapsed: false, lastRunId: null, returnMode: "dollars", set: (p) => set(p) }), { name: "strata-ui-v1" }),
);
