import { DndContext, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, arrayMove, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Copy, GripVertical, Plus, SlidersHorizontal, Trash2 } from "lucide-react";
import { Button, Field, Input, NumInput, Popover, PopoverContent, PopoverTrigger, Select, Tip } from "@/components/ui";
import { clone, cn } from "@/lib/format";
import { useBuilder } from "@/lib/store";
import type { LegSpec, StrikeMethod } from "@/lib/types";

const METHOD_LABEL: Record<StrikeMethod, string> = { delta: "Delta", premium: "Premium $", pct_otm: "% OTM", atm_offset: "ATM ±", fixed: "Fixed strike", leg_offset: "From leg ±" };
const METHOD_HINT: Record<StrikeMethod, string> = {
  delta: "Absolute delta target, e.g. 0.16", premium: "Target mid price per share", pct_otm: "Percent out of the money (negative = ITM)",
  atm_offset: "Offset from the at-the-money strike", fixed: "Exact strike price", leg_offset: "Offset from another leg's strike (signed: -5 = five points lower)",
};

function nextId(legs: LegSpec[], base: string) {
  let i = 1;
  while (legs.some((l) => l.id === `${base}${i}`)) i++;
  return `${base}${i}`;
}

export function LegsEditor() {
  const legs = useBuilder((s) => s.strategy.legs);
  const setStrategy = useBuilder((s) => s.setStrategy);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));
  const update = (i: number, fn: (l: LegSpec) => void) => setStrategy((s) => fn(s.legs[i]));
  const add = (kind: "option" | "stock") => setStrategy((s) => {
    s.legs.push(kind === "stock"
      ? { id: nextId(s.legs, "stk"), side: "buy", kind: "stock", right: "C", quantity: 100, dte: 0, strike: { method: "fixed", value: 0 } }
      : { id: nextId(s.legs, "leg"), side: "sell", kind: "option", right: "P", quantity: 1, dte: s.legs[0]?.dte ?? 0, dte_mode: "nearest", strike: { method: "delta", value: 0.2, unit: "points", ref_leg: null } });
  });
  const dup = (i: number) => setStrategy((s) => {
    const c = clone(s.legs[i]);
    c.id = nextId(s.legs, c.id.replace(/\d+$/, "") || "leg");
    s.legs.splice(i + 1, 0, c);
  });
  const remove = (i: number) => setStrategy((s) => {
    const id = s.legs[i].id;
    s.legs.splice(i, 1);
    for (const l of s.legs) if (l.strike.ref_leg === id) l.strike = { method: "delta", value: 0.2, unit: "points", ref_leg: null };
  });
  const onDragEnd = (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return;
    setStrategy((s) => {
      const a = s.legs.findIndex((l) => l.id === e.active.id), b = s.legs.findIndex((l) => l.id === e.over!.id);
      s.legs = arrayMove(s.legs, a, b);
    });
  };
  return (
    <div className="flex flex-col gap-1.5">
      <div className="grid grid-cols-[16px_36px_56px_44px_48px_92px_1fr_auto] items-center gap-1.5 px-1.5 label">
        <span /><span>ID</span><span>Side</span><span>Type</span><span>Qty</span><span>Expiration</span><span>Strike selection</span><span />
      </div>
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
        <SortableContext items={legs.map((l) => l.id)} strategy={verticalListSortingStrategy}>
          {legs.map((l, i) => <LegRow key={l.id} leg={l} legs={legs} onChange={(fn) => update(i, fn)} onDup={() => dup(i)} onRemove={() => remove(i)} />)}
        </SortableContext>
      </DndContext>
      <div className="flex gap-1.5 pt-1">
        <Button size="sm" onClick={() => add("option")}><Plus /> Option leg</Button>
        <Button size="sm" variant="ghost" onClick={() => add("stock")}><Plus /> Stock leg</Button>
        <span className="flex-1" />
        <span className="text-2xs text-faint self-center">Drag ⋮⋮ to reorder · legs resolve top-down, "From leg" offsets after their reference</span>
      </div>
    </div>
  );
}

function LegRow({ leg, legs, onChange, onDup, onRemove }: { leg: LegSpec; legs: LegSpec[]; onChange: (fn: (l: LegSpec) => void) => void; onDup: () => void; onRemove: () => void }) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: leg.id });
  const stock = leg.kind === "stock";
  const m = leg.strike.method;
  const filters = [leg.iv_min, leg.iv_max, leg.max_spread_abs, leg.max_spread_pct, leg.min_volume, leg.min_open_interest].filter((x) => x !== null && x !== undefined).length + (leg.min_bid ? 1 : 0) + (leg.dte_min !== null && leg.dte_min !== undefined ? 1 : 0) + (leg.dte_max !== null && leg.dte_max !== undefined ? 1 : 0);
  return (
    <div ref={setNodeRef} style={{ transform: CSS.Transform.toString(transform), transition }}
      className={cn("grid grid-cols-[16px_36px_56px_44px_48px_92px_1fr_auto] items-center gap-1.5 rounded-sm border border-line-2 bg-bg-2 px-1.5 py-1", leg.side === "sell" ? "shadow-[inset_2px_0_0_var(--color-down)]" : "shadow-[inset_2px_0_0_var(--color-up)]")}>
      <button {...attributes} {...listeners} className="cursor-grab text-faint hover:text-fg-2 touch-none"><GripVertical className="size-3.5" /></button>
      <Input sz="sm" className="font-mono" value={leg.id} onChange={(e) => {
        const v = e.target.value.replace(/[^\w-]/g, "");
        if (v && !legs.some((x) => x.id === v && x !== leg)) onChange((l) => { l.id = v; });
      }} />
      <button onClick={() => onChange((l) => { l.side = l.side === "buy" ? "sell" : "buy"; })}
        className={cn("h-6 rounded-sm border text-2xs font-semibold tracking-wide", leg.side === "buy" ? "border-up/50 text-up bg-up/10" : "border-down/50 text-down bg-down/10")}>{leg.side.toUpperCase()}</button>
      {stock ? <span className="text-2xs text-muted text-center">STOCK</span> : (
        <button onClick={() => onChange((l) => { l.right = l.right === "C" ? "P" : "C"; })} className="h-6 rounded-sm border border-line-2 text-xs font-mono hover:border-faint">{leg.right === "C" ? "CALL" : "PUT"}</button>
      )}
      <NumInput sz="sm" value={leg.quantity} min={1} onChange={(v) => onChange((l) => { l.quantity = Math.max(1, Math.round(v ?? 1)); })} suffix={stock ? "sh" : "x"} />
      {stock ? <span /> : (
        <div className="flex items-center gap-0.5">
          <NumInput sz="sm" className="w-11" value={leg.dte} min={0} onChange={(v) => onChange((l) => { l.dte = Math.max(0, Math.round(v ?? 0)); })} />
          <Select sz="sm" className="w-[46px] px-1 text-2xs" value={leg.dte_mode || "nearest"} onChange={(v) => onChange((l) => { l.dte_mode = v; })}
            options={[{ value: "nearest", label: "≈" }, { value: "at_least", label: "≥" }, { value: "at_most", label: "≤" }]} />
        </div>
      )}
      {stock ? <span className="text-2xs text-faint">filled at the underlying price ± stock slippage</span> : (
        <div className="flex items-center gap-1 min-w-0">
          <Tip content={METHOD_HINT[m]}><span><Select sz="sm" className="w-[104px]" value={m} onChange={(v) => onChange((l) => {
            l.strike.method = v;
            l.strike.value = { delta: 0.2, premium: 1, pct_otm: 2, atm_offset: 0, fixed: 400, leg_offset: l.right === "P" ? -5 : 5 }[v];
            l.strike.ref_leg = v === "leg_offset" ? legs.find((x) => x.id !== l.id && x.kind !== "stock")?.id ?? null : null;
            l.strike.unit = "points";
          })} options={Object.entries(METHOD_LABEL).map(([value, label]) => ({ value: value as StrikeMethod, label }))} /></span></Tip>
          {m === "leg_offset" && (
            <Select sz="sm" className="w-16" value={leg.strike.ref_leg || ""} onChange={(v) => onChange((l) => { l.strike.ref_leg = v; })}
              options={legs.filter((x) => x.id !== leg.id && x.kind !== "stock").map((x) => x.id)} />
          )}
          <NumInput sz="sm" className="w-16" value={leg.strike.value} step={m === "delta" ? 0.01 : m === "premium" ? 0.05 : 1} onChange={(v) => onChange((l) => { l.strike.value = v ?? 0; })} />
          {(m === "leg_offset" || m === "atm_offset") && (
            <Select sz="sm" className="w-[66px]" value={leg.strike.unit || "points"} onChange={(v) => onChange((l) => { l.strike.unit = v; })} options={[{ value: "points", label: "pts" }, { value: "strikes", label: "strikes" }, { value: "pct", label: "%" }]} />
          )}
        </div>
      )}
      <div className="flex items-center gap-0.5">
        {!stock && (
          <Popover>
            <PopoverTrigger asChild>
              <Button size="iconSm" variant="ghost" className={cn(filters > 0 && "text-accent")} title="Filters: IV, spread, liquidity, DTE bounds, group"><SlidersHorizontal /></Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-72 p-2.5">
              <div className="label mb-2">Leg {leg.id} · contract filters</div>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Min DTE"><NumInput sz="sm" allowNull value={leg.dte_min ?? null} onChange={(v) => onChange((l) => { l.dte_min = v; })} /></Field>
                <Field label="Max DTE"><NumInput sz="sm" allowNull value={leg.dte_max ?? null} onChange={(v) => onChange((l) => { l.dte_max = v; })} /></Field>
                <Field label="Min IV %"><NumInput sz="sm" allowNull value={leg.iv_min ?? null} onChange={(v) => onChange((l) => { l.iv_min = v; })} /></Field>
                <Field label="Max IV %"><NumInput sz="sm" allowNull value={leg.iv_max ?? null} onChange={(v) => onChange((l) => { l.iv_max = v; })} /></Field>
                <Field label="Max spread $"><NumInput sz="sm" allowNull step={0.01} value={leg.max_spread_abs ?? null} onChange={(v) => onChange((l) => { l.max_spread_abs = v; })} /></Field>
                <Field label="Max spread % mid"><NumInput sz="sm" allowNull value={leg.max_spread_pct ?? null} onChange={(v) => onChange((l) => { l.max_spread_pct = v; })} /></Field>
                <Field label="Min bid $"><NumInput sz="sm" step={0.01} value={leg.min_bid ?? 0} onChange={(v) => onChange((l) => { l.min_bid = v ?? 0; })} /></Field>
                <Field label="Min volume"><NumInput sz="sm" allowNull value={leg.min_volume ?? null} onChange={(v) => onChange((l) => { l.min_volume = v; })} /></Field>
                <Field label="Min open interest"><NumInput sz="sm" allowNull value={leg.min_open_interest ?? null} onChange={(v) => onChange((l) => { l.min_open_interest = v; })} /></Field>
                <Field label="Group" hint="Legs sharing a group can be exited together by leg rules (e.g. put_side)"><Input sz="sm" value={leg.group || ""} onChange={(e) => onChange((l) => { l.group = e.target.value || null; })} /></Field>
              </div>
            </PopoverContent>
          </Popover>
        )}
        <Tip content="Duplicate leg"><Button size="iconSm" variant="ghost" onClick={onDup}><Copy /></Button></Tip>
        <Tip content="Remove leg"><Button size="iconSm" variant="ghost" onClick={onRemove} disabled={legs.length <= 1}><Trash2 /></Button></Tip>
      </div>
    </div>
  );
}
