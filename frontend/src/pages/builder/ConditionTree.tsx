import { DndContext, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, arrayMove, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Copy, GripVertical, ListPlus, Plus, Trash2 } from "lucide-react";
import { Fragment } from "react";
import { Button, Input, Menu, MenuContent, MenuItem, MenuLabel, MenuSep, MenuTrigger, NumInput, Segmented, Select, Switch, Tip } from "@/components/ui";
import { ExprInput } from "@/components/research/ExprInput";
import { useMeta } from "@/lib/api";
import { clone, cn, uid } from "@/lib/format";
import type { CondNode, Meta, ParamMeta } from "@/lib/types";

const CATEGORY_ORDER = ["Expression", "Time", "Price", "Indicators", "Volatility", "Regime", "Events", "Position"];

function defaults(meta: Meta, type: string): Record<string, any> {
  const p: Record<string, any> = {};
  for (const pm of meta.conditions[type].params) p[pm.name] = clone(pm.default);
  return p;
}

export function ConditionTree({ value, onChange, exitMode = false }: { value: CondNode; onChange: (n: CondNode) => void; exitMode?: boolean }) {
  const meta = useMeta().data;
  if (!meta) return null;
  return <GroupNode node={value} depth={0} meta={meta} exitMode={exitMode} onChange={onChange} />;
}

function AddMenu({ meta, exitMode, onAdd, onAddGroup }: { meta: Meta; exitMode: boolean; onAdd: (t: string) => void; onAddGroup: () => void }) {
  const byCat: Record<string, string[]> = {};
  for (const [k, v] of Object.entries(meta.conditions)) {
    if (v.exit_only && !exitMode) continue;
    (byCat[v.category] ||= []).push(k);
  }
  return (
    <Menu>
      <MenuTrigger asChild><Button size="sm" variant="ghost"><Plus /> Condition</Button></MenuTrigger>
      <MenuContent align="start" className="max-h-[60vh] overflow-auto w-60">
        {CATEGORY_ORDER.filter((c) => byCat[c]).map((c, i) => (
          <Fragment key={c}>
            {i > 0 && <MenuSep />}
            <MenuLabel>{c}</MenuLabel>
            {byCat[c].map((t) => <MenuItem key={t} onSelect={() => onAdd(t)}>{meta.conditions[t].label}</MenuItem>)}
          </Fragment>
        ))}
        <MenuSep />
        <MenuItem onSelect={onAddGroup}><ListPlus /> Nested AND/OR group</MenuItem>
      </MenuContent>
    </Menu>
  );
}

function GroupNode({ node, depth, meta, exitMode, onChange, onRemove, onDuplicate, dragHandle }: {
  node: CondNode; depth: number; meta: Meta; exitMode: boolean; onChange: (n: CondNode) => void; onRemove?: () => void; onDuplicate?: () => void; dragHandle?: React.ReactNode;
}) {
  const children = node.children || [];
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));
  const setChildren = (c: CondNode[]) => onChange({ ...node, children: c });
  const add = (t: string) => setChildren([...children, { id: uid("c"), kind: "cond", type: t, params: defaults(meta, t), enabled: true }]);
  const addGroup = () => setChildren([...children, { id: uid("g"), kind: "group", op: "or", children: [], enabled: true }]);
  const onDragEnd = (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return;
    const a = children.findIndex((c) => c.id === e.active.id), b = children.findIndex((c) => c.id === e.over!.id);
    setChildren(arrayMove(children, a, b));
  };
  const isRoot = depth === 0;
  return (
    <div className={cn("rounded-sm", !isRoot && "border border-line-2 bg-bg-2/40", node.enabled === false && "opacity-50")}>
      <div className={cn("flex items-center gap-1.5", isRoot ? "pb-1.5" : "px-1.5 py-1 border-b border-line")}>
        {dragHandle}
        {!isRoot && <Switch checked={node.enabled !== false} onChange={(v) => onChange({ ...node, enabled: v })} />}
        <Segmented value={node.op || "and"} onChange={(v) => onChange({ ...node, op: v })} options={[{ value: "and", label: "ALL (AND)" }, { value: "or", label: "ANY (OR)" }]} />
        <Tip content="Negate this group"><button onClick={() => onChange({ ...node, negate: !node.negate })} className={cn("h-6 px-1.5 rounded-sm text-2xs font-mono border", node.negate ? "border-accent text-accent" : "border-line-2 text-faint hover:text-fg")}>NOT</button></Tip>
        <span className="text-2xs text-faint">{children.length} item{children.length === 1 ? "" : "s"}</span>
        <span className="flex-1" />
        <AddMenu meta={meta} exitMode={exitMode} onAdd={add} onAddGroup={addGroup} />
        {onDuplicate && <Tip content="Duplicate group"><Button size="icon" variant="ghost" onClick={onDuplicate}><Copy /></Button></Tip>}
        {onRemove && <Tip content="Remove group"><Button size="icon" variant="ghost" onClick={onRemove}><Trash2 /></Button></Tip>}
      </div>
      <div className={cn("flex flex-col gap-1", !isRoot && "p-1.5", isRoot && "pl-2 border-l-2 border-line-2")}>
        {children.length === 0 && <div className="px-2 py-2 text-xs text-faint">{isRoot ? "No conditions — the rule is always true. Add a condition (e.g. an entry time)." : "Empty group (ignored)."}</div>}
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
          <SortableContext items={children.map((c) => c.id)} strategy={verticalListSortingStrategy}>
            {children.map((c, i) => (
              <Sortable key={c.id} id={c.id}>
                {(handle) => {
                  const upd = (n: CondNode) => setChildren(children.map((x, j) => (j === i ? n : x)));
                  const rm = () => setChildren(children.filter((_, j) => j !== i));
                  const dup = () => {
                    const cp = clone(c);
                    const reid = (n: CondNode): CondNode => ({ ...n, id: uid(n.kind === "group" ? "g" : "c"), children: (n.children || []).map(reid) });
                    setChildren([...children.slice(0, i + 1), reid(cp), ...children.slice(i + 1)]);
                  };
                  return (
                    <div className="flex items-stretch gap-1">
                      {i > 0 && <div className="w-7 shrink-0 pt-1.5 text-right font-mono text-[9.5px] text-accent/80">{node.op === "or" ? "OR" : "AND"}</div>}
                      {i === 0 && <div className="w-7 shrink-0 pt-1.5 text-right font-mono text-[9.5px] text-faint">IF</div>}
                      <div className="min-w-0 flex-1">
                        {c.kind === "group"
                          ? <GroupNode node={c} depth={depth + 1} meta={meta} exitMode={exitMode} onChange={upd} onRemove={rm} onDuplicate={dup} dragHandle={handle} />
                          : <LeafNode node={c} meta={meta} onChange={upd} onRemove={rm} onDuplicate={dup} dragHandle={handle} />}
                      </div>
                    </div>
                  );
                }}
              </Sortable>
            ))}
          </SortableContext>
        </DndContext>
      </div>
    </div>
  );
}

function Sortable({ id, children }: { id: string; children: (handle: React.ReactNode) => React.ReactNode }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const handle = <button {...attributes} {...listeners} className="cursor-grab text-faint hover:text-fg-2 touch-none" title="Drag to reorder"><GripVertical className="size-3.5" /></button>;
  return <div ref={setNodeRef} style={{ transform: CSS.Transform.toString(transform), transition, zIndex: isDragging ? 10 : undefined, position: "relative" }}>{children(handle)}</div>;
}

function LeafNode({ node, meta, onChange, onRemove, onDuplicate, dragHandle }: { node: CondNode; meta: Meta; onChange: (n: CondNode) => void; onRemove: () => void; onDuplicate: () => void; dragHandle: React.ReactNode }) {
  const def = meta.conditions[node.type || ""];
  const params = node.params || {};
  const setP = (k: string, v: any) => onChange({ ...node, params: { ...params, [k]: v } });
  if (!def) return <div className="text-xs text-down">Unknown condition {node.type}</div>;
  return (
    <div className={cn("group flex flex-wrap items-center gap-x-2 gap-y-1 rounded-sm border border-line-2 bg-bg-2 px-1.5 py-1", node.enabled === false && "opacity-45")}>
      {dragHandle}
      <Switch checked={node.enabled !== false} onChange={(v) => onChange({ ...node, enabled: v })} />
      <span className="text-xs font-medium text-fg min-w-28">{def.label}</span>
      {def.params.map((pm) => <ParamControl key={pm.name} pm={pm} value={params[pm.name]} onChange={(v) => setP(pm.name, v)} meta={meta} params={params} />)}
      <span className="flex-1" />
      <div className="flex items-center gap-0.5 opacity-60 group-hover:opacity-100">
        <Tip content="Negate"><button onClick={() => onChange({ ...node, negate: !node.negate })} className={cn("h-5 px-1 rounded-sm text-[9.5px] font-mono border", node.negate ? "border-accent text-accent" : "border-line-2 text-faint hover:text-fg")}>NOT</button></Tip>
        <Tip content="Duplicate"><Button size="iconSm" variant="ghost" onClick={onDuplicate}><Copy /></Button></Tip>
        <Tip content="Remove"><Button size="iconSm" variant="ghost" onClick={onRemove}><Trash2 /></Button></Tip>
      </div>
    </div>
  );
}

function ParamControl({ pm, value, onChange, meta, params }: { pm: ParamMeta; value: any; onChange: (v: any) => void; meta: Meta; params: Record<string, any> }) {
  if (pm.name === "minutes" && params.ref && params.ref !== "minutes_ago") return null;
  if (pm.name === "n" && params.when && params.when !== "within_n") return null;
  const lab = <span className="text-2xs text-muted">{pm.label}</span>;
  switch (pm.type) {
    case "select":
      return <span className="flex items-center gap-1">{pm.name !== "cmp" && lab}<Select sz="sm" className={pm.name === "cmp" ? "w-24" : "w-28"} value={value ?? pm.default} onChange={onChange} options={pm.options!.map((o) => ({ value: o, label: o.replace(/_/g, " ") }))} /></span>;
    case "multiselect":
      return (
        <span className="flex items-center gap-0.5">
          {pm.options!.map((o) => {
            const on = (value || []).includes(o);
            return <button key={o} onClick={() => onChange(on ? value.filter((x: string) => x !== o) : [...(value || []), o])}
              className={cn("h-5 px-1.5 rounded-[2px] text-2xs border", on ? "border-accent/60 bg-accent/10 text-accent" : "border-line-2 text-faint hover:text-fg")}>{o}</button>;
          })}
        </span>
      );
    case "int":
    case "number":
      return <span className="flex items-center gap-1">{lab}<NumInput sz="sm" className="w-16" value={value ?? pm.default} step={pm.type === "int" ? 1 : 0.1} onChange={(v) => onChange(v ?? 0)} /></span>;
    case "bool":
      return <span className="flex items-center gap-1">{lab}<Switch checked={!!value} onChange={onChange} /></span>;
    case "time":
      return <span className="flex items-center gap-1">{lab}<Input sz="sm" className="w-16 num" value={value ?? pm.default} onChange={(e) => onChange(e.target.value)} placeholder="HH:MM" /></span>;
    case "operand":
      return <OperandEditor value={value ?? pm.default} onChange={onChange} meta={meta} />;
    case "expr":
      return <ExprInput className="min-w-[340px] flex-1" value={value ?? ""} onChange={onChange} />;
    default:
      return <span className="flex items-center gap-1">{lab}<Input sz="sm" className="w-40 num" value={value ?? ""} onChange={(e) => onChange(e.target.value)} /></span>;
  }
}

function OperandEditor({ value, onChange, meta }: { value: any; onChange: (v: any) => void; meta: Meta }) {
  const kind = value?.kind || "const";
  const set = (p: any) => onChange({ ...value, ...p });
  return (
    <span className="inline-flex items-center gap-1 rounded-sm border border-line px-1 py-0.5 bg-bg-1">
      <Select sz="sm" className="w-[86px]" value={kind} onChange={(k) => onChange(k === "const" ? { kind: "const", value: 0 } : k === "indicator" ? { kind: "indicator", name: "rsi", params: { period: 14 }, tf: "5m" } : k === "feature" ? { kind: "feature", name: "vix" } : { kind: "price" })}
        options={[{ value: "indicator", label: "indicator" }, { value: "feature", label: "feature" }, { value: "price", label: "price" }, { value: "const", label: "value" }]} />
      {kind === "const" && <NumInput sz="sm" className="w-16" value={value.value} step={0.5} onChange={(v) => set({ value: v ?? 0 })} />}
      {kind === "feature" && <Select sz="sm" className="w-28" value={value.name} onChange={(n) => set({ name: n })} options={meta.features} />}
      {kind === "indicator" && (
        <>
          <Select sz="sm" className="w-20" value={value.name} onChange={(n) => set({ name: n, params: { ...meta.indicators[n].params }, field: null })} options={meta.indicator_names.map((n) => ({ value: n, label: n.toUpperCase() }))} />
          {Object.entries(meta.indicators[value.name]?.params || {}).map(([k, d]) => (
            <Tip key={k} content={k}><span><NumInput sz="sm" className="w-11" value={value.params?.[k] ?? d} onChange={(v) => set({ params: { ...(value.params || {}), [k]: v ?? d } })} /></span></Tip>
          ))}
          <Select sz="sm" className="w-14" value={value.tf || "5m"} onChange={(tf) => set({ tf })} options={meta.timeframes} />
          {(meta.indicators[value.name]?.fields.length || 0) > 1 && (
            <Select sz="sm" className="w-20" value={value.field || meta.indicators[value.name].fields[0]} onChange={(f) => set({ field: f })} options={meta.indicators[value.name].fields} />
          )}
        </>
      )}
    </span>
  );
}
