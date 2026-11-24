export type Side = "buy" | "sell";
export type Right = "C" | "P";
export type StrikeMethod = "delta" | "premium" | "pct_otm" | "atm_offset" | "fixed" | "leg_offset";

export interface StrikeSel { method: StrikeMethod; value: number; unit?: "points" | "strikes" | "pct"; ref_leg?: string | null }
export interface LegSpec {
  id: string; side: Side; kind?: "option" | "stock"; right: Right; quantity: number; dte: number;
  dte_min?: number | null; dte_max?: number | null; dte_mode?: "nearest" | "at_least" | "at_most" | "exact";
  strike: StrikeSel; iv_min?: number | null; iv_max?: number | null; max_spread_abs?: number | null; max_spread_pct?: number | null;
  min_bid?: number; min_volume?: number | null; min_open_interest?: number | null; group?: string | null;
}
export interface CondNode {
  id: string; kind: "group" | "cond"; op?: "and" | "or"; negate?: boolean; enabled?: boolean;
  children?: CondNode[]; type?: string; params?: Record<string, any>;
}
export interface Threshold { enabled: boolean; mode: "pct" | "dollars"; value: number }
export interface ExitRules {
  profit_target: Threshold; stop_loss: Threshold;
  trailing_stop: { enabled: boolean; mode: "pct" | "dollars"; activate: number; trail: number };
  time_exit: { enabled: boolean; time: string; dte_lte: number | null };
  dte_exit: { enabled: boolean; dte: number; time: string };
  max_hold: { enabled: boolean; minutes: number | null; days: number | null };
  underlying: { enabled: boolean; move_against_pct: number | null; move_for_pct: number | null; strike_breach: boolean; breach_buffer_pct: number };
  greeks: { enabled: boolean; max_abs_delta: number | null; short_leg_delta: number | null };
  leg_rules: { target: string; stop_pct: number | null; target_pct: number | null }[];
  custom: CondNode | null;
  roll: { enabled: boolean; trigger: "dte" | "short_delta" | "loss_pct" | "profit_pct"; value: number; max_rolls: number };
  trigger_price: "close_price" | "mid";
}
export interface EntryRules {
  conditions: CondNode; max_entries_per_day: number; cooldown_minutes: number; max_open_positions: number;
  reentry_after: string[]; min_credit: number | null; max_debit: number | null;
}
export interface Sizing { method: "fixed" | "risk_pct" | "bp_pct"; contracts: number; pct: number; max_contracts: number }
export interface StrategyConfig {
  name: string; symbol: string; legs: LegSpec[]; entry: EntryRules; exits: ExitRules; sizing: Sizing; allocation: number;
  vol_symbol?: string | null; regime?: { vix_low: number; vix_high: number; trend_fast: number; trend_slow: number };
}
export interface Execution {
  fill_model: "conservative" | "midpoint" | "custom"; custom_fraction: number; slippage_per_contract: number; commission_per_contract: number;
  fee_per_contract: number; commission_per_order: number; stock_commission_per_share: number; stock_slippage_bps: number; exercise_fee: number;
  participation_rate: number; max_volume_pct: number | null; allow_partial_fills: boolean; max_quote_age_seconds: number;
  early_assignment: boolean; assignment_extrinsic_threshold: number; physical_settlement: "close_next_open" | "intrinsic";
  signal_interval_minutes: number; latency_seconds: number; max_fill_delay_minutes: number; trade_bar_slippage_pct: number;
}
export interface Pricing {
  mode: "historical" | "modeled"; model: "american_baw" | "black_scholes"; iv_source: "vix" | "realized" | "fixed"; iv_fixed: number; iv_scale: number;
  skew_per_10pct: number; spread_pct: number; min_spread: number; rate: number; dividend_yield: number; snapshot_minutes: number; strike_step: number | null;
}
export interface Capital {
  initial_capital: number; margin_model: "regt" | "cash"; max_bp_usage_pct: number; max_concurrent_positions: number;
  risk_free_rate: number; cash_interest_rate: number; cash_benchmark_rate: number; liquidate_on_margin_call: boolean;
}
export interface BacktestConfig {
  strategies: StrategyConfig[]; start: string; end: string; capital: Capital; execution: Execution; pricing?: Pricing; benchmark?: string | null; allow_synthetic: boolean; label: string;
}

export interface RunRow {
  id: string; kind: string; status: "queued" | "running" | "done" | "error" | "cancelled"; name: string; strategy_id?: string | null;
  config_hash?: string; dataset_versions?: Record<string, string>; engine_version?: string; synthetic?: number; progress: number; message?: string;
  error?: string | null; summary?: any; created_at: string; started_at?: string; finished_at?: string; result?: any; config?: any;
}

export interface LegRecord {
  id: string; kind: string; right: Right | null; strike: number; expiration: string | null; ratio: number; contracts: number; side: Side; group?: string | null;
  multiplier: number; entry_price: number; entry_bid: number; entry_ask: number; entry_mid: number; entry_iv: number | null; entry_delta: number | null;
  entry_gamma: number | null; entry_theta: number | null; entry_vega: number | null; exit_price: number | null; exit_bid: number | null; exit_ask: number | null;
  exit_mid: number | null; exit_delta: number | null; exit_gamma: number | null; exit_theta: number | null; exit_vega: number | null; exit_ts: number | null; exit_reason: string; pnl: number | null;
}
export interface Trade {
  id: number; strategy: string; strategy_idx: number; symbol: string; tag: string; entry_ts: number; exit_ts: number; entry_time: string; exit_time: string | null;
  entry_date: string; entry_minute: number; entry_weekday: number; holding_minutes: number | null; units: number; entry_cash: number; exit_cash: number;
  commissions: number; slippage: number; pnl: number; gross_pnl: number; max_risk: number; return_on_risk: number | null; pnl_pct_premium: number; credit: boolean;
  basis: number; mfe: number; mae: number; exit_reason: string; entry_spot: number; exit_spot: number | null; entry_dte: number; short_delta: number | null;
  entry_vix: number | null; entry_iv: number | null; entry_regime: string; max_profit: number | null; max_loss: number | null; flags: string[];
  roll_group: number; roll_count: number; legs: LegRecord[]; entry_explain?: Explain[]; exit_explain?: Explain[];
}
export interface Explain { id?: string; depth?: number; text: string; passed: boolean; group?: boolean; type?: string; selection?: boolean; reason?: string }

export interface Meta {
  conditions: Record<string, { label: string; category: string; exit_only?: boolean; params: ParamMeta[] }>;
  indicators: Record<string, { params: Record<string, number>; fields: string[] }>;
  indicator_names: string[]; timeframes: string[]; features: string[];
  templates: { key: string; label: string; category: string; description: string; config: StrategyConfig; dte_hint: number }[];
  objectives: string[]; research: Record<string, { label: string; question: string; params: string[] }>;
  research_targets: Record<string, string>; research_filters: Record<string, string>;
  providers: { name: string; label: string; docs: string; resolution: string }[];
  execution_defaults: Execution; engine_version: string;
  expressions: { name: string; signature: string; description: string; category: string; needs: string }[]; default_symbol: string; pricing_defaults: Pricing;
}
export interface ParamMeta { name: string; type: "string" | "time" | "select" | "multiselect" | "int" | "number" | "bool" | "operand" | "expr"; label: string; default: any; options?: string[] }

export interface ChartSpec {
  type: "bar" | "bar_ci" | "line" | "scatter" | "histogram" | "band"; title: string; x?: any[];
  series: { name: string; data: any[]; ci?: [number, number][]; p25?: number[]; p75?: number[] }[];
  xLabel?: string; yLabel?: string; yFormat?: "pct"; diagonal?: boolean;
}
