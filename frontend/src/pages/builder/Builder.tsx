import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, Blocks, ChevronRight, Copy, FilePlus2, Play, Save, Sigma, Briefcase, FolderOpen } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Group, Panel as RPanel, Separator } from "react-resizable-panels";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ErrorBox } from "@/components/common";
import { Button, Input, Kbd, Menu, MenuContent, MenuItem, MenuLabel, MenuSep, MenuTrigger, Select, Tip } from "@/components/ui";
import { api, queryClient, useMeta } from "@/lib/api";
import { cn } from "@/lib/format";
import { normalizeStrategy, useBuilder } from "@/lib/store";
import type { CondNode, StrategyConfig } from "@/lib/types";
import { LegsEditor } from "./LegsEditor";
import { PreviewPanel } from "./PreviewPanel";
import { EntryEditor, ExitsEditor, RiskEditor } from "./RulesEditors";

type Section = "legs" | "entry" | "exits" | "risk";

function countConds(n: CondNode | null | undefined): number {
  if (!n) return 0;
  if (n.kind === "cond") return n.enabled === false ? 0 : 1;
  return (n.children || []).reduce((a, c) => a + countConds(c), 0);
}

function legSummary(s: StrategyConfig) {
  return s.legs.map((l) => l.kind === "stock" ? `${l.side === "buy" ? "+" : "-"}${l.quantity}sh` : `${l.side === "buy" ? "+" : "-"}${l.quantity}${l.right} ${l.dte}d ${l.strike.method === "delta" ? `Δ${l.strike.value}` : l.strike.method === "leg_offset" ? `${l.strike.ref_leg}${l.strike.value > 0 ? "+" : ""}${l.strike.value}` : `${l.strike.method}:${l.strike.value}`}`);
}

export default function Builder() {
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const meta = useMeta().data;
  const b = useBuilder();
  const { strategy: s, setStrategy, load, patch } = b;
  const [section, setSection] = useState<Section>("legs");
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage") });
  // real symbols first; simulated sandbox symbols last and labelled
  const tradable = (cov.data || []).filter((c) => c.kind !== "vol_index" && c.symbol !== "VIX" && (c.daily || Object.keys(c.intraday || {}).length || c.options));
  const symbolOpts = [...tradable.filter((c) => !c.synthetic), ...tradable.filter((c) => c.synthetic)].map((c) => ({ value: c.symbol as string, label: c.synthetic ? `${c.symbol} (simulated)` : (c.symbol as string) }));
  const symRow = (cov.data || []).find((c) => c.symbol === s.symbol);
  const tpl = useQuery({ queryKey: ["templates", s.symbol], queryFn: () => api.get<{ symbol: string; eod: boolean; templates: NonNullable<typeof meta>["templates"] }>(`/api/templates?symbol=${s.symbol}`), staleTime: 60_000 });
  const modeled = (b.pricing?.mode || "historical") === "modeled";
  const entryTimes = (function walk(n: any): string[] { return n.kind === "cond" ? (n.type === "entry_times" && n.enabled !== false ? String(n.params?.times || "").split(/[,;]/).map((x: string) => x.trim()).filter(Boolean) : []) : (n.children || []).flatMap(walk); })(s.entry.conditions);
  const dataHint = !symRow ? `No data is stored for ${s.symbol}. Download it under Data - nothing is simulated in its place.`
    : modeled ? null
      : !symRow.options ? `No option quotes are stored for ${s.symbol}: a historical backtest cannot run. Download chains under Data, or switch Option prices to Modeled (estimates).`
        : symRow.options.resolution === "EOD" && (entryTimes.some((t) => t !== "close" && t !== "16:00") || s.legs.some((l) => (l.kind || "option") === "option" && l.dte === 0))
          ? `${s.symbol} option quotes are end-of-day only: entries must be at the close and 0DTE legs cannot be traded. Orders at other times expire unfilled and are reported.` : null;

  // load a saved strategy from ?strategy=
  useEffect(() => {
    const id = params.get("strategy");
    if (!id) return;
    api.get<any>(`/api/strategies/${id}`).then((r) => {
      load(r.config, r.id);
      toast.success(`Loaded "${r.name}"`);
      setParams({}, { replace: true });
    }).catch((e) => toast.error(String(e.message || e)));
  }, [params]); // eslint-disable-line

  const validation = useQuery({
    queryKey: ["validate", JSON.stringify(s)],
    queryFn: () => api.post<{ ok: boolean; errors: { loc: string; msg: string }[] }>("/api/strategies/validate", s),
    placeholderData: (p) => p,
  });

  const save = useMutation({
    mutationFn: async (asNew: boolean) => {
      const id = useBuilder.getState().strategyId;
      const body = { config: useBuilder.getState().strategy };
      return id && !asNew ? api.put<any>(`/api/strategies/${id}`, body) : api.post<any>("/api/strategies", body);
    },
    onSuccess: (r) => {
      patch({ strategyId: r.id, dirty: false });
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
      toast.success(`Saved "${r.name}"`);
    },
    onError: (e: any) => toast.error(e.message),
  });

  const run = useMutation({
    mutationFn: () => api.post<{ id: string; cached: boolean }>("/api/backtests", { config: useBuilder.getState().backtestConfig(), strategy_id: useBuilder.getState().strategyId }),
    onSuccess: (r) => {
      if (r.cached) toast.message("Identical run found (same config, dataset version and engine) — opening cached result");
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      nav(`/results/${r.id}`);
    },
    onError: (e: any) => toast.error(e.message),
  });

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        run.mutate();
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        save.mutate(false);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [run, save]);

  const templatesByCat = useMemo(() => {
    const m: Record<string, NonNullable<typeof meta>["templates"]> = {};
    for (const t of tpl.data?.templates || meta?.templates || []) (m[t.category] ||= []).push(t);
    return m;
  }, [meta, tpl.data]);

  const ex = s.exits;
  const exitBits = [ex.profit_target.enabled && `PT ${ex.profit_target.value}${ex.profit_target.mode === "pct" ? "%" : "$"}`, ex.stop_loss.enabled && `SL ${ex.stop_loss.value}${ex.stop_loss.mode === "pct" ? "%" : "$"}`,
    ex.trailing_stop.enabled && "trail", ex.time_exit.enabled && `@${ex.time_exit.time}`, ex.dte_exit.enabled && `DTE≤${ex.dte_exit.dte}`, ex.underlying.enabled && "underlying", ex.greeks.enabled && "greeks",
    ex.leg_rules.length && `${ex.leg_rules.length} leg rules`, ex.custom && `custom(${countConds(ex.custom)})`, ex.roll.enabled && "roll"].filter(Boolean) as string[];
  const steps: { key: Section; title: string; lines: string[] }[] = [
    { key: "legs", title: `Legs · ${s.legs.length}`, lines: legSummary(s) },
    { key: "entry", title: `Entry · ${countConds(s.entry.conditions)} conditions`, lines: [`≤${s.entry.max_entries_per_day}/day · ≤${s.entry.max_open_positions} open`, s.entry.cooldown_minutes ? `cooldown ${s.entry.cooldown_minutes}m` : "no cooldown"] },
    { key: "exits", title: "Exits", lines: exitBits.length ? exitBits : ["hold to expiration"] },
    { key: "risk", title: "Sizing, capital & execution", lines: [s.sizing.method === "fixed" ? `${s.sizing.contracts} unit(s)` : `${s.sizing.pct}% ${s.sizing.method === "risk_pct" ? "risk" : "BP"}`, `${b.capital.margin_model === "regt" ? "Reg-T" : "cash"} · $${(b.capital.initial_capital / 1000).toFixed(0)}k`, `${b.execution.fill_model} fills · $${b.execution.commission_per_contract}/ct`] },
  ];

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line bg-bg-1 px-3">
        <Blocks className="size-4 text-accent" />
        <Input className="h-7 w-64 bg-transparent border-transparent hover:border-line-2 text-sm font-semibold" value={s.name} onChange={(e) => setStrategy((x) => { x.name = e.target.value; })} />
        {b.dirty && <span className="text-2xs text-warn">● unsaved</span>}
        <Select className="w-32" value={s.symbol} onChange={(v) => setStrategy((x) => { x.symbol = v; })} options={symbolOpts.some((o) => o.value === s.symbol) ? symbolOpts : [{ value: s.symbol, label: s.symbol }, ...symbolOpts]} />
        {modeled && <span className="rounded-[2px] border border-warn/60 bg-warn/10 px-1.5 py-px text-[10px] font-semibold tracking-wider text-warn">MODELED PREMIUMS</span>}
        <Menu>
          <MenuTrigger asChild><Button><FilePlus2 /> Template</Button></MenuTrigger>
          <MenuContent align="start" className="max-h-[70vh] overflow-auto w-72">
            {Object.entries(templatesByCat).map(([cat, ts], i) => (
              <div key={cat}>
                {i > 0 && <MenuSep />}
                <MenuLabel>{cat}</MenuLabel>
                {ts.map((t) => (
                  <MenuItem key={t.key} onSelect={() => { load(normalizeStrategy({ ...t.config, symbol: s.symbol }), null); patch({ dirty: true }); setSection("legs"); }}>
                    <div className="flex flex-col"><span className="text-fg">{t.label}</span><span className="text-2xs text-faint">{t.description}</span></div>
                  </MenuItem>
                ))}
              </div>
            ))}
          </MenuContent>
        </Menu>
        <div className="flex items-center gap-1 ml-2">
          <span className="label">Range</span>
          <Input sz="sm" type="date" className="w-[118px] num" value={b.start} onChange={(e) => patch({ start: e.target.value })} />
          <span className="text-faint">→</span>
          <Input sz="sm" type="date" className="w-[118px] num" value={b.end} onChange={(e) => patch({ end: e.target.value })} />
        </div>
        <span className="flex-1" />
        <Menu>
          <MenuTrigger asChild><Button variant="ghost">More</Button></MenuTrigger>
          <MenuContent align="end">
            <MenuItem onSelect={() => nav("/strategies")}><FolderOpen /> Saved strategies</MenuItem>
            <MenuItem onSelect={() => save.mutate(true)}><Copy /> Save as new strategy</MenuItem>
            <MenuItem onSelect={() => nav("/optimize?from=builder")}><Sigma /> Optimise these parameters</MenuItem>
            <MenuItem onSelect={() => nav("/portfolio?add=builder")}><Briefcase /> Add to portfolio test</MenuItem>
          </MenuContent>
        </Menu>
        <Tip content={<span>Save strategy <Kbd>⌘S</Kbd></span>}><Button onClick={() => save.mutate(false)} disabled={save.isPending}><Save /> Save</Button></Tip>
        <Tip content={<span>Run backtest <Kbd>⌘↵</Kbd></span>}>
          <Button variant="primary" onClick={() => run.mutate()} disabled={run.isPending || validation.data?.ok === false}><Play /> Run backtest</Button>
        </Tip>
      </div>
      {dataHint && <div className="flex shrink-0 items-center gap-2 border-b border-warn/30 bg-warn/[0.06] px-3 py-1 text-2xs text-warn"><AlertTriangle className="size-3 shrink-0" />{dataHint}<Link to="/data" className="ml-1 underline">Data</Link></div>}
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <RPanel defaultSize="17%" minSize="12%" maxSize="28%" className="bg-bg-1">
          <div className="flex h-full flex-col overflow-auto">
            <div className="label px-3 pt-2.5 pb-1.5">Algorithm</div>
            <div className="flex items-start gap-2 px-3 pb-2 text-xs">
              <span className="mt-1 size-1.5 rounded-full bg-info shrink-0" />
              <div><div className="text-fg">Universe</div><div className="num text-2xs text-muted">{s.symbol} · {b.start} → {b.end}</div></div>
            </div>
            {steps.map((st) => (
              <button key={st.key} onClick={() => setSection(st.key)} className={cn("relative flex items-start gap-2 px-3 py-2 text-left text-xs border-l-2 transition-colors",
                section === st.key ? "border-accent bg-bg-3" : "border-transparent hover:bg-bg-2")}>
                <ChevronRight className={cn("mt-0.5 size-3 shrink-0 text-faint", section === st.key && "text-accent")} />
                <div className="min-w-0">
                  <div className="text-fg">{st.title}</div>
                  {st.lines.map((l, i) => <div key={i} className="num text-2xs text-muted truncate">{l}</div>)}
                </div>
              </button>
            ))}
            <div className="mt-auto border-t border-line p-3 text-2xs text-faint leading-relaxed">
              Contracts are selected from the stored chain snapshot at each decision time — the preview on the right resolves the same legs against a real session.
            </div>
          </div>
        </RPanel>
        <Separator />
        <RPanel defaultSize="50%" minSize="35%">
          <div className="flex h-full flex-col">
            <div className="flex h-8 shrink-0 items-center gap-0 border-b border-line px-2">
              {steps.map((st) => (
                <button key={st.key} onClick={() => setSection(st.key)} className={cn("h-8 px-3 text-xs border-b-2 -mb-px", section === st.key ? "border-accent text-fg" : "border-transparent text-muted hover:text-fg")}>
                  {st.key === "risk" ? "Sizing & execution" : st.title.split(" ·")[0]}
                </button>
              ))}
            </div>
            <div className="min-h-0 flex-1 overflow-auto p-3">
              {validation.data?.ok === false && <ErrorBox className="mb-2" error={validation.data.errors.map((e) => `${e.loc}: ${e.msg}`).join("\n")} />}
              {section === "legs" && <LegsEditor />}
              {section === "entry" && <EntryEditor />}
              {section === "exits" && <ExitsEditor />}
              {section === "risk" && <RiskEditor />}
            </div>
          </div>
        </RPanel>
        <Separator />
        <RPanel defaultSize="33%" minSize="22%" className="bg-bg-1">
          <PreviewPanel />
        </RPanel>
      </Group>
    </div>
  );
}
