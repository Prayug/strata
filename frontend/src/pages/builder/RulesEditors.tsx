import { Plus, Trash2 } from "lucide-react";
import type { ReactNode } from "react";
import { Button, Field, Input, NumInput, Segmented, Select, Switch } from "@/components/ui";
import { cn } from "@/lib/format";
import { rootGroup, useBuilder } from "@/lib/store";
import { ConditionTree } from "./ConditionTree";

function Row({ title, on, onToggle, children, desc }: { title: string; on: boolean; onToggle?: (v: boolean) => void; children?: ReactNode; desc?: string }) {
  return (
    <div className={cn("grid grid-cols-[150px_1fr] items-center gap-3 border-b border-line/70 px-1 py-1.5", !on && "[&>div:last-child]:opacity-40")}>
      <div className="flex items-center gap-2">
        {onToggle && <Switch checked={on} onChange={onToggle} />}
        <div>
          <div className="text-xs text-fg">{title}</div>
          {desc && <div className="text-2xs text-faint leading-tight">{desc}</div>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}
const L = ({ children }: { children: ReactNode }) => <span className="text-2xs text-muted">{children}</span>;

export function EntryEditor() {
  const s = useBuilder((x) => x.strategy);
  const set = useBuilder((x) => x.setStrategy);
  const e = s.entry;
  return (
    <div className="flex flex-col gap-3">
      <div>
        <div className="label mb-1.5">Entry rule</div>
        <ConditionTree value={e.conditions} onChange={(n) => set((x) => { x.entry.conditions = n; })} />
      </div>
      <div>
        <div className="label mb-1">Entry constraints</div>
        <Row title="Entries per day" on desc="Signals evaluated at every snapshot">
          <L>max</L><NumInput sz="sm" className="w-14" min={1} value={e.max_entries_per_day} onChange={(v) => set((x) => { x.entry.max_entries_per_day = Math.max(1, v ?? 1); })} />
          <L>cooldown after exit</L><NumInput sz="sm" className="w-16" min={0} value={e.cooldown_minutes} suffix="m" onChange={(v) => set((x) => { x.entry.cooldown_minutes = Math.max(0, v ?? 0); })} />
        </Row>
        <Row title="Overlap" on desc="Concurrent positions of this strategy">
          <L>max open</L><NumInput sz="sm" className="w-14" min={1} value={e.max_open_positions} onChange={(v) => set((x) => { x.entry.max_open_positions = Math.max(1, v ?? 1); })} />
        </Row>
        <Row title="Re-entry after exit" on desc="Same-session re-entry policy">
          <Segmented value={e.reentry_after.includes("any") ? "any" : e.reentry_after.includes("none") ? "none" : "some"} onChange={(v) => set((x) => { x.entry.reentry_after = v === "some" ? ["target"] : [v]; })}
            options={[{ value: "any", label: "Any exit" }, { value: "some", label: "Only after…" }, { value: "none", label: "Never" }]} />
          {!e.reentry_after.includes("any") && !e.reentry_after.includes("none") && ["target", "stop", "time"].map((r) => (
            <label key={r} className="flex items-center gap-1 text-2xs text-fg-2">
              <input type="checkbox" className="accent-[var(--color-accent)]" checked={e.reentry_after.includes(r)} onChange={(ev) => set((x) => {
                const cur = x.entry.reentry_after.filter((y) => y !== r);
                x.entry.reentry_after = ev.target.checked ? [...cur, r] : cur.length ? cur : ["none"];
              })} />{r}
            </label>
          ))}
        </Row>
        <Row title="Premium limits" on desc="Per unit, $ per share">
          <L>min credit</L><NumInput sz="sm" className="w-16" allowNull step={0.05} value={e.min_credit} onChange={(v) => set((x) => { x.entry.min_credit = v; })} />
          <L>max debit</L><NumInput sz="sm" className="w-16" allowNull step={0.05} value={e.max_debit} onChange={(v) => set((x) => { x.entry.max_debit = v; })} />
        </Row>
      </div>
    </div>
  );
}

export function ExitsEditor() {
  const s = useBuilder((x) => x.strategy);
  const set = useBuilder((x) => x.setStrategy);
  const ex = s.exits;
  const groups = Array.from(new Set([...s.legs.map((l) => l.id), ...s.legs.map((l) => l.group).filter(Boolean) as string[]]));
  return (
    <div className="flex flex-col gap-3">
      <div>
        <Row title="Profit target" on={ex.profit_target.enabled} onToggle={(v) => set((x) => { x.exits.profit_target.enabled = v; })} desc="% of premium (credit captured / debit gained)">
          <NumInput sz="sm" className="w-20" value={ex.profit_target.value} onChange={(v) => set((x) => { x.exits.profit_target.value = v ?? 0; })} />
          <Segmented value={ex.profit_target.mode} onChange={(v) => set((x) => { x.exits.profit_target.mode = v; })} options={[{ value: "pct", label: "% premium" }, { value: "dollars", label: "$ / unit" }]} />
        </Row>
        <Row title="Stop loss" on={ex.stop_loss.enabled} onToggle={(v) => set((x) => { x.exits.stop_loss.enabled = v; })} desc="Loss as % of premium (200 = 2x credit)">
          <NumInput sz="sm" className="w-20" value={ex.stop_loss.value} onChange={(v) => set((x) => { x.exits.stop_loss.value = v ?? 0; })} />
          <Segmented value={ex.stop_loss.mode} onChange={(v) => set((x) => { x.exits.stop_loss.mode = v; })} options={[{ value: "pct", label: "% premium" }, { value: "dollars", label: "$ / unit" }]} />
        </Row>
        <Row title="Trailing stop" on={ex.trailing_stop.enabled} onToggle={(v) => set((x) => { x.exits.trailing_stop.enabled = v; })} desc="Arms after profit ≥ activation">
          <L>activate at</L><NumInput sz="sm" className="w-16" value={ex.trailing_stop.activate} onChange={(v) => set((x) => { x.exits.trailing_stop.activate = v ?? 0; })} />
          <L>give back</L><NumInput sz="sm" className="w-16" value={ex.trailing_stop.trail} onChange={(v) => set((x) => { x.exits.trailing_stop.trail = v ?? 0; })} />
          <Segmented value={ex.trailing_stop.mode} onChange={(v) => set((x) => { x.exits.trailing_stop.mode = v; })} options={[{ value: "pct", label: "%" }, { value: "dollars", label: "$/unit" }]} />
        </Row>
        <Row title="Time exit" on={ex.time_exit.enabled} onToggle={(v) => set((x) => { x.exits.time_exit.enabled = v; })} desc="Close at/after a time of day">
          <L>at</L><Input sz="sm" className="w-16 num" value={ex.time_exit.time} onChange={(e) => set((x) => { x.exits.time_exit.time = e.target.value; })} />
          <L>on sessions with front DTE ≤</L><NumInput sz="sm" className="w-14" allowNull value={ex.time_exit.dte_lte} placeholder="any" onChange={(v) => set((x) => { x.exits.time_exit.dte_lte = v; })} />
        </Row>
        <Row title="Expiration exit" on={ex.dte_exit.enabled} onToggle={(v) => set((x) => { x.exits.dte_exit.enabled = v; })} desc="Close before expiration">
          <L>when DTE ≤</L><NumInput sz="sm" className="w-14" value={ex.dte_exit.dte} onChange={(v) => set((x) => { x.exits.dte_exit.dte = v ?? 0; })} />
          <L>after</L><Input sz="sm" className="w-16 num" value={ex.dte_exit.time} onChange={(e) => set((x) => { x.exits.dte_exit.time = e.target.value; })} />
        </Row>
        <Row title="Max holding period" on={ex.max_hold.enabled} onToggle={(v) => set((x) => { x.exits.max_hold.enabled = v; })}>
          <L>minutes</L><NumInput sz="sm" className="w-16" allowNull value={ex.max_hold.minutes} onChange={(v) => set((x) => { x.exits.max_hold.minutes = v; })} />
          <L>sessions</L><NumInput sz="sm" className="w-14" allowNull value={ex.max_hold.days} onChange={(v) => set((x) => { x.exits.max_hold.days = v; })} />
        </Row>
        <Row title="Underlying threshold" on={ex.underlying.enabled} onToggle={(v) => set((x) => { x.exits.underlying.enabled = v; })} desc="vs entry price; neutral = absolute move">
          <L>against %</L><NumInput sz="sm" className="w-14" allowNull step={0.1} value={ex.underlying.move_against_pct} onChange={(v) => set((x) => { x.exits.underlying.move_against_pct = v; })} />
          <L>in favour %</L><NumInput sz="sm" className="w-14" allowNull step={0.1} value={ex.underlying.move_for_pct} onChange={(v) => set((x) => { x.exits.underlying.move_for_pct = v; })} />
          <label className="flex items-center gap-1 text-2xs text-fg-2"><Switch checked={ex.underlying.strike_breach} onChange={(v) => set((x) => { x.exits.underlying.strike_breach = v; })} /> short strike breached</label>
          <L>buffer %</L><NumInput sz="sm" className="w-12" step={0.1} value={ex.underlying.breach_buffer_pct} onChange={(v) => set((x) => { x.exits.underlying.breach_buffer_pct = v ?? 0; })} />
        </Row>
        <Row title="Greek limits" on={ex.greeks.enabled} onToggle={(v) => set((x) => { x.exits.greeks.enabled = v; })}>
          <L>|Δ| per unit &gt;</L><NumInput sz="sm" className="w-16" allowNull value={ex.greeks.max_abs_delta} onChange={(v) => set((x) => { x.exits.greeks.max_abs_delta = v; })} />
          <L>short leg |Δ| &gt;</L><NumInput sz="sm" className="w-16" allowNull step={0.05} value={ex.greeks.short_leg_delta} onChange={(v) => set((x) => { x.exits.greeks.short_leg_delta = v; })} />
        </Row>
        <Row title="Roll" on={ex.roll.enabled} onToggle={(v) => set((x) => { x.exits.roll.enabled = v; })} desc="Close and reopen the same structure">
          <L>when</L>
          <Select sz="sm" className="w-28" value={ex.roll.trigger} onChange={(v) => set((x) => { x.exits.roll.trigger = v; })} options={[{ value: "dte", label: "DTE ≤" }, { value: "short_delta", label: "short |Δ| ≥" }, { value: "loss_pct", label: "loss % ≥" }, { value: "profit_pct", label: "profit % ≥" }]} />
          <NumInput sz="sm" className="w-14" value={ex.roll.value} step={ex.roll.trigger === "short_delta" ? 0.05 : 1} onChange={(v) => set((x) => { x.exits.roll.value = v ?? 0; })} />
          <L>max rolls</L><NumInput sz="sm" className="w-12" value={ex.roll.max_rolls} onChange={(v) => set((x) => { x.exits.roll.max_rolls = v ?? 1; })} />
        </Row>
        <Row title="Trigger price" on desc="Price used to evaluate P&L rules">
          <Segmented value={ex.trigger_price} onChange={(v) => set((x) => { x.exits.trigger_price = v; })} options={[{ value: "close_price", label: "Cost to close (fill model)" }, { value: "mid", label: "Mid mark" }]} />
        </Row>
      </div>
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="label">Individual leg / group rules</div>
          <Button size="sm" variant="ghost" onClick={() => set((x) => { x.exits.leg_rules.push({ target: groups[0] || "", stop_pct: 100, target_pct: null }); })}><Plus /> Leg rule</Button>
        </div>
        {ex.leg_rules.length === 0 && <div className="text-2xs text-faint px-1">e.g. close an iron condor's put side when its short put doubles, leaving the call side open.</div>}
        {ex.leg_rules.map((r, i) => (
          <div key={i} className="flex items-center gap-2 border-b border-line/70 py-1 px-1">
            <L>close</L>
            <Select sz="sm" className="w-28" value={r.target} onChange={(v) => set((x) => { x.exits.leg_rules[i].target = v; })} options={groups} />
            <L>when leg price moves against by %</L><NumInput sz="sm" className="w-14" allowNull value={r.stop_pct} onChange={(v) => set((x) => { x.exits.leg_rules[i].stop_pct = v; })} />
            <L>or in favour by %</L><NumInput sz="sm" className="w-14" allowNull value={r.target_pct} onChange={(v) => set((x) => { x.exits.leg_rules[i].target_pct = v; })} />
            <Button size="iconSm" variant="ghost" onClick={() => set((x) => { x.exits.leg_rules.splice(i, 1); })}><Trash2 /></Button>
          </div>
        ))}
      </div>
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <div className="label">Custom exit rule (strategy-wide)</div>
          {!ex.custom && <Button size="sm" variant="ghost" onClick={() => set((x) => { x.exits.custom = rootGroup([]); })}><Plus /> Add rule tree</Button>}
          {ex.custom && <Button size="sm" variant="ghost" onClick={() => set((x) => { x.exits.custom = null; })}><Trash2 /> Remove</Button>}
        </div>
        {ex.custom && <ConditionTree exitMode value={ex.custom} onChange={(n) => set((x) => { x.exits.custom = n; })} />}
        <div className="mt-2 text-2xs text-faint leading-relaxed">
          Exits are evaluated at each option snapshot after entry, using quotes observable at that time. Positions still open at expiration are settled
          (cash or physical delivery). When a loss-side and a profit-side rule fire together, the loss-side reason is recorded and the trade is flagged ambiguous.
        </div>
      </div>
    </div>
  );
}

export function RiskEditor() {
  const s = useBuilder((x) => x.strategy);
  const set = useBuilder((x) => x.setStrategy);
  const { capital, execution: ex, patch } = useBuilder();
  const cap = (k: keyof typeof capital, v: any) => patch({ capital: { ...capital, [k]: v } });
  const exe = (k: keyof typeof ex, v: any) => patch({ execution: { ...ex, [k]: v } });
  return (
    <div className="grid grid-cols-2 gap-x-5 gap-y-3">
      <section className="col-span-2">
        <div className="label mb-1.5">Position sizing</div>
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Method" className="w-44"><Select value={s.sizing.method} onChange={(v) => set((x) => { x.sizing.method = v; })} options={[{ value: "fixed", label: "Fixed contracts" }, { value: "risk_pct", label: "% equity at max loss" }, { value: "bp_pct", label: "% equity of buying power" }]} /></Field>
          {s.sizing.method === "fixed" ? (
            <Field label="Units" className="w-20"><NumInput value={s.sizing.contracts} min={1} onChange={(v) => set((x) => { x.sizing.contracts = Math.max(1, v ?? 1); })} /></Field>
          ) : (
            <Field label="% of allocated equity" className="w-32"><NumInput value={s.sizing.pct} step={0.5} onChange={(v) => set((x) => { x.sizing.pct = v ?? 1; })} suffix="%" /></Field>
          )}
          <Field label="Max units" className="w-20"><NumInput value={s.sizing.max_contracts} min={1} onChange={(v) => set((x) => { x.sizing.max_contracts = Math.max(1, v ?? 1); })} /></Field>
          <Field label="Allocation" className="w-24" hint="Fraction of portfolio equity this strategy sizes against (portfolio mode)"><NumInput value={s.allocation} step={0.05} min={0.01} max={1} onChange={(v) => set((x) => { x.allocation = v ?? 1; })} /></Field>
        </div>
      </section>
      <section>
        <div className="label mb-1.5">Capital & margin</div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Initial capital"><NumInput value={capital.initial_capital} step={1000} onChange={(v) => cap("initial_capital", v ?? 100000)} /></Field>
          <Field label="Margin model" hint="Reg-T: defined risk = max loss; naked shorts use the CBOE formula. Cash: puts cash-secured, naked calls rejected."><Select value={capital.margin_model} onChange={(v) => cap("margin_model", v)} options={[{ value: "regt", label: "Reg-T" }, { value: "cash", label: "Cash" }]} /></Field>
          <Field label="Max BP usage % equity"><NumInput value={capital.max_bp_usage_pct} onChange={(v) => cap("max_bp_usage_pct", v ?? 100)} suffix="%" /></Field>
          <Field label="Max concurrent positions"><NumInput value={capital.max_concurrent_positions} onChange={(v) => cap("max_concurrent_positions", v ?? 50)} /></Field>
          <Field label="Risk-free rate (Sharpe)"><NumInput value={capital.risk_free_rate} step={0.005} onChange={(v) => cap("risk_free_rate", v ?? 0)} /></Field>
          <Field label="Interest on idle cash"><NumInput value={capital.cash_interest_rate} step={0.005} onChange={(v) => cap("cash_interest_rate", v ?? 0)} /></Field>
          <label className="col-span-2 flex items-center gap-2 text-xs text-fg-2"><Switch checked={capital.liquidate_on_margin_call} onChange={(v) => cap("liquidate_on_margin_call", v)} /> Liquidate on margin deficit</label>
        </div>
      </section>
      <section>
        <div className="label mb-1.5">Regime definitions</div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Low-vol VIX <"><NumInput value={s.regime?.vix_low ?? 15} onChange={(v) => set((x) => { x.regime = { ...x.regime!, vix_low: v ?? 15 }; })} /></Field>
          <Field label="High-vol VIX >"><NumInput value={s.regime?.vix_high ?? 25} onChange={(v) => set((x) => { x.regime = { ...x.regime!, vix_high: v ?? 25 }; })} /></Field>
          <Field label="Trend fast SMA"><NumInput value={s.regime?.trend_fast ?? 50} onChange={(v) => set((x) => { x.regime = { ...x.regime!, trend_fast: v ?? 50 }; })} /></Field>
          <Field label="Trend slow SMA"><NumInput value={s.regime?.trend_slow ?? 200} onChange={(v) => set((x) => { x.regime = { ...x.regime!, trend_slow: v ?? 200 }; })} /></Field>
          <Field label="Vol index symbol" className="col-span-2" hint="Defaults to the symbol's configured volatility index"><Input value={s.vol_symbol || ""} placeholder="auto" onChange={(e) => set((x) => { x.vol_symbol = e.target.value.toUpperCase() || null; })} /></Field>
        </div>
      </section>
      <section className="col-span-2">
        <div className="label mb-1.5">Execution model</div>
        <div className="grid grid-cols-4 gap-2">
          <Field label="Fill model" hint="Conservative = buy at ask / sell at bid (default)"><Select value={ex.fill_model} onChange={(v) => exe("fill_model", v)} options={[{ value: "conservative", label: "Conservative" }, { value: "midpoint", label: "Midpoint" }, { value: "custom", label: "Custom" }]} /></Field>
          <Field label="Custom: fraction of half-spread"><NumInput value={ex.custom_fraction} step={0.05} min={0} max={1} onChange={(v) => exe("custom_fraction", v ?? 0.5)} /></Field>
          <Field label="Slippage $/contract-share"><NumInput value={ex.slippage_per_contract} step={0.01} onChange={(v) => exe("slippage_per_contract", v ?? 0)} /></Field>
          <Field label="Commission / contract"><NumInput value={ex.commission_per_contract} step={0.05} onChange={(v) => exe("commission_per_contract", v ?? 0)} /></Field>
          <Field label="Fees / contract"><NumInput value={ex.fee_per_contract} step={0.01} onChange={(v) => exe("fee_per_contract", v ?? 0)} /></Field>
          <Field label="Fee / order"><NumInput value={ex.commission_per_order} step={0.1} onChange={(v) => exe("commission_per_order", v ?? 0)} /></Field>
          <Field label="Stock slippage bps"><NumInput value={ex.stock_slippage_bps} step={0.5} onChange={(v) => exe("stock_slippage_bps", v ?? 0)} /></Field>
          <Field label="Max quote age (s)" hint="Older quotes are stale: rejected for entries, exits deferred"><NumInput value={ex.max_quote_age_seconds} step={30} onChange={(v) => exe("max_quote_age_seconds", v ?? 120)} /></Field>
          <Field label="Participation of displayed size"><NumInput value={ex.participation_rate} step={0.05} min={0.01} max={1} onChange={(v) => exe("participation_rate", v ?? 1)} /></Field>
          <Field label="Max % of session volume"><NumInput allowNull value={ex.max_volume_pct} onChange={(v) => exe("max_volume_pct", v)} /></Field>
          <Field label="Assignment extrinsic <"><NumInput value={ex.assignment_extrinsic_threshold} step={0.01} onChange={(v) => exe("assignment_extrinsic_threshold", v ?? 0.05)} /></Field>
          <Field label="Physical settlement"><Select value={ex.physical_settlement} onChange={(v) => exe("physical_settlement", v)} options={[{ value: "close_next_open", label: "Deliver, sell next open" }, { value: "intrinsic", label: "Cash at intrinsic (approx.)" }]} /></Field>
          <label className="flex items-center gap-2 text-xs text-fg-2"><Switch checked={ex.allow_partial_fills} onChange={(v) => exe("allow_partial_fills", v)} /> Partial fills</label>
          <label className="flex items-center gap-2 text-xs text-fg-2"><Switch checked={ex.early_assignment} onChange={(v) => exe("early_assignment", v)} /> Early assignment</label>
        </div>
      </section>
    </div>
  );
}
