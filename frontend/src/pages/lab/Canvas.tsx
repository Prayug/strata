/* The research canvas: the same notebook as a graph. Every node is a stored object of the session (a question,
   a plan, a study, a backtest, a set of variants, a review...) and opens the real result behind it. Any node can
   be branched from. Positions, collapsed groups and the viewport are saved with the notebook. */
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { Background, Controls, Handle, MiniMap, Panel, Position, ReactFlow, useNodesState, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { BarChart3, ChevronDown, ChevronRight, CornerDownRight, FileText, FlaskConical, GitBranch, GitCompareArrows, HelpCircle, LayoutGrid, Layers, ListChecks, Maximize2, Microscope, Minimize2, Network, ShieldAlert, Target, X, Zap } from "lucide-react";
import { createContext, memo, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { Empty } from "./bits";
import type { OpenTarget } from "./Messages";

interface GNode { id: string; type: string; title: string; subtitle?: string; status?: string; run_id?: string; version_id?: number; message_id?: number; result_message?: number; key?: string; group?: boolean; parent_group?: string;
  stats?: { label: string; value: string }[]; assumptions?: string[]; items?: { label: string; value: string; status: string }[]; summary?: string; mode?: string; default_position: { x: number; y: number } }
interface Graph { nodes: GNode[]; edges: { id: string; source: string; target: string }[]; layout: { positions: Record<string, { x: number; y: number }>; collapsed: string[]; viewport: any } }

const KIND: Record<string, { label: string; Icon: any; tone: string }> = {
  question: { label: "Question", Icon: HelpCircle, tone: "#7a8496" }, plan: { label: "Research plan", Icon: ListChecks, tone: "#e8a33d" }, tested: { label: "Tested configurations", Icon: LayoutGrid, tone: "#7a8496" },
  candidate: { label: "Selected hypothesis", Icon: Target, tone: "#e8a33d" }, study: { label: "Market study", Icon: Microscope, tone: "#4c8dff" }, backtest: { label: "Option backtest", Icon: BarChart3, tone: "#26a69a" },
  sweep: { label: "Parameter sweep", Icon: Layers, tone: "#c678dd" }, investigation: { label: "Related studies", Icon: Layers, tone: "#c678dd" }, structures: { label: "Option structures", Icon: FlaskConical, tone: "#56b6c2" },
  discovery: { label: "Exploration", Icon: Network, tone: "#c678dd" }, comparison: { label: "Comparison", Icon: GitCompareArrows, tone: "#d19a66" }, validation: { label: "Robustness", Icon: ListChecks, tone: "#98c379" },
  analysis: { label: "Reasoning", Icon: Zap, tone: "#61afef" }, stress: { label: "Execution stress", Icon: Zap, tone: "#d19a66" }, challenge: { label: "Adversarial review", Icon: ShieldAlert, tone: "#ef5350" },
  report: { label: "Report", Icon: FileText, tone: "#d6dce5" },
};
const STATUS_WORD: Record<string, string> = { done: "finished", running: "running", queued: "queued", error: "failed", idle: "not run", awaiting_confirmation: "waiting for you", awaiting_answer: "needs an answer", blocked: "blocked", ready: "ready", needs_decision: "data missing" };

const Ctx = createContext<{ collapsed: Set<string>; toggle: (id: string) => void; hidden: Record<string, number> }>({ collapsed: new Set(), toggle: () => {}, hidden: {} });

const ResearchNode = memo(({ data, selected }: NodeProps<Node<{ n: GNode }>>) => {
  const n = data.n;
  const k = KIND[n.type] || KIND.study;
  const { collapsed, toggle, hidden } = useContext(Ctx);
  const isCol = collapsed.has(n.id);
  const st = n.status || "done";
  return (
    <div className={cn("w-[264px] rounded-[10px] border bg-bg-1 text-left shadow-[0_2px_10px_rgba(0,0,0,0.35)] transition-colors", selected ? "border-accent" : "border-line-2 hover:border-fg-2/40")} data-testid={`node-${n.type}`}>
      <Handle type="target" position={Position.Left} className="!size-1.5 !border-0 !bg-line-2" />
      <div className="flex items-center gap-1.5 rounded-t-[9px] px-2.5 py-1.5" style={{ borderLeft: `3px solid ${k.tone}`, background: `${k.tone}14` }}>
        <k.Icon className="size-3.5 shrink-0" style={{ color: k.tone }} />
        <span className="text-[10.5px] font-medium uppercase tracking-wide" style={{ color: k.tone }}>{k.label}</span>
        <span className="flex-1" />
        <span className={cn("flex items-center gap-1 text-[10.5px]", st === "error" ? "text-down" : st === "running" || st === "queued" ? "text-accent" : st === "done" ? "text-muted" : "text-warn")}>
          <span className={cn("dot", st === "error" ? "bg-down" : st === "running" || st === "queued" ? "bg-accent pulse" : st === "done" ? "bg-up" : "bg-warn")} />{STATUS_WORD[st] || st}
        </span>
      </div>
      <div className="px-2.5 pb-2 pt-1.5">
        <div className="line-clamp-2 text-[12.5px] font-medium leading-snug">{n.title}</div>
        {n.subtitle && <div className="mt-0.5 truncate text-[11px] dim">{n.subtitle}</div>}
        {n.stats && n.stats.length > 0 && (
          <div className="mt-1.5 grid grid-cols-3 gap-1">
            {n.stats.slice(0, 3).map((s) => <div key={s.label} className="min-w-0 rounded bg-bg-2 px-1.5 py-1"><div className="truncate text-[9.5px] dim">{s.label}</div><div className="num truncate text-[11.5px]" title={String(s.value)}>{s.value ?? "—"}</div></div>)}
          </div>
        )}
        {n.group && (
          <button className="nodrag mt-1.5 flex w-full items-center gap-1 rounded border border-line px-1.5 py-1 text-[11px] dim hover:border-accent/60 hover:text-fg" onClick={(e) => { e.stopPropagation(); toggle(n.id); }} data-testid="node-collapse">
            {isCol ? <ChevronRight className="size-3" /> : <ChevronDown className="size-3" />}{isCol ? `Show ${hidden[n.id] || 0} hidden` : "Collapse this group"}
          </button>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!size-1.5 !border-0 !bg-line-2" />
    </div>
  );
});
ResearchNode.displayName = "ResearchNode";
const nodeTypes = { research: ResearchNode };

const BRANCHES: Record<string, string[]> = {
  study: ["Compare that with 2% and 4% down days", "Test option structures on this", "Challenge this result", "How much of the result is explained by volatility?", "Test whether the results hold up out of sample"],
  backtest: ["Stress-test the execution", "Challenge this strategy", "Compare the results with 7DTE calls", "Include commissions and conservative slippage", "Are most profits concentrated in a handful of sessions?"],
};

export function Canvas({ sid, stamp, onOpen, onBranch, onSelect, onAnalyse, busy, wide, onWide }: {
  sid: string; stamp: number; busy?: boolean; wide: boolean; onWide: (w: boolean) => void;
  onOpen: (t: OpenTarget) => void; onBranch: (text: string, versionId?: number) => void; onSelect: (runId: string, key: string, what: "study" | "options") => void; onAnalyse: (kind: "stress" | "challenge", runId: string) => void;
}) {
  const q = useQuery({ queryKey: ["canvas", sid, stamp], queryFn: () => api.get<Graph>(`/api/ai/sessions/${sid}/canvas`), placeholderData: (p) => p, refetchInterval: (s) => (s.state.data?.nodes.some((n) => n.status === "running" || n.status === "queued") ? 2500 : false) });
  const g = q.data;
  const [layout, setLayout] = useState<Graph["layout"] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [text, setText] = useState("");
  const timer = useRef<any>(null);
  // the saved layout is read once per mount (the workspace remounts this component for another notebook)
  useEffect(() => { if (g && !layout) setLayout({ positions: g.layout.positions || {}, collapsed: g.layout.collapsed || [], viewport: g.layout.viewport || null }); }, [g, layout]);
  const save = (next: Graph["layout"]) => {
    setLayout(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => { api.put(`/api/ai/sessions/${sid}/canvas`, next).catch(() => {}); }, 500);
  };
  const collapsed = useMemo(() => new Set(layout?.collapsed || []), [layout?.collapsed]);
  const { visible, hidden } = useMemo(() => {
    const by = Object.fromEntries((g?.nodes || []).map((n) => [n.id, n]));
    const parent: Record<string, string> = {};
    for (const e of g?.edges || []) if (!parent[e.target]) parent[e.target] = e.source;
    const hiddenCount: Record<string, number> = {};
    const vis: GNode[] = [];
    for (const n of g?.nodes || []) {
      let p = parent[n.id], gone: string | null = null;
      for (let i = 0; p && i < 40; i++) { if (collapsed.has(p) && by[p]?.group) gone = p; p = parent[p]; }
      if (gone) hiddenCount[gone] = (hiddenCount[gone] || 0) + 1; else vis.push(n);
    }
    return { visible: vis, hidden: hiddenCount };
  }, [g, collapsed]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<{ n: GNode }>>([]);
  useEffect(() => {
    if (!layout) return;
    setNodes((cur) => {
      const pos = Object.fromEntries(cur.map((c) => [c.id, c.position]));
      return visible.map((n) => ({ id: n.id, type: "research", data: { n }, position: layout.positions[n.id] || pos[n.id] || n.default_position, selected: n.id === sel }));
    });
  }, [visible, layout?.positions, sel]); // eslint-disable-line
  const ids = useMemo(() => new Set(visible.map((n) => n.id)), [visible]);
  const edges: Edge[] = useMemo(() => (g?.edges || []).filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => {
    const t = visible.find((n) => n.id === e.target);
    return { id: e.id, source: e.source, target: e.target, type: "smoothstep", animated: t?.status === "running" || t?.status === "queued", style: { stroke: "#3a4556", strokeWidth: 1.4 } };
  }), [g, ids, visible]);
  const toggle = (id: string) => { if (!layout) return; const c = new Set(layout.collapsed); if (c.has(id)) c.delete(id); else c.add(id); save({ ...layout, collapsed: [...c] }); };
  const node = g?.nodes.find((n) => n.id === sel) || null;
  const open = (n: GNode) => {
    if ((n.type === "study" || n.type === "backtest") && n.run_id && n.status === "done") return onOpen({ type: "run", runId: n.run_id, messageId: n.result_message });
    if (n.type === "candidate" || n.type === "tested") { const p = g?.nodes.find((x) => x.type === "plan" && x.run_id === n.run_id); if (p?.result_message) return onOpen({ type: "message", messageId: p.result_message }); }
    if (n.result_message) return onOpen({ type: "message", messageId: n.result_message });
    if (n.message_id && ["analysis", "validation", "comparison"].includes(n.type)) return onOpen(n.run_id && n.type !== "analysis" ? { type: "run", runId: n.run_id, tab: n.type === "validation" ? "robustness" : "overview" } : { type: "message", messageId: n.message_id });
    if (n.version_id) return onOpen({ type: "setup", versionId: n.version_id });
  };
  if (q.isLoading || !layout) return <div className="p-6 text-[13px] dim">Laying out the research…</div>;
  if (!g || g.nodes.length === 0) return <Empty icon={Network} title="The canvas fills in as you research">Every question, plan, study, backtest and review of this notebook appears here as a node you can open and branch from.</Empty>;
  const canBranch = node && (node.version_id !== undefined || node.type === "candidate");
  return (
    <div className="relative h-full w-full" data-testid="canvas">
      <Ctx.Provider value={{ collapsed, toggle, hidden }}>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} forceColorMode="dark" minZoom={0.15} maxZoom={1.6} nodesConnectable={false} deleteKeyCode={null}
          defaultViewport={layout.viewport || undefined} fitView={!layout.viewport} fitViewOptions={{ padding: 0.12, minZoom: 0.55, maxZoom: 1 }} proOptions={{ hideAttribution: true }}
          onNodeClick={(_, n) => setSel(n.id)} onNodeDoubleClick={(_, n) => open((n.data as any).n)} onPaneClick={() => setSel(null)}
          onNodeDragStop={(_, n) => save({ ...layout, positions: { ...layout.positions, [n.id]: n.position } })}
          onMoveEnd={(_, vp) => { if (Math.abs((layout.viewport?.x ?? 1e9) - vp.x) > 2 || Math.abs((layout.viewport?.zoom ?? 0) - vp.zoom) > 0.01 || Math.abs((layout.viewport?.y ?? 1e9) - vp.y) > 2) save({ ...layout, viewport: vp }); }}>
          <Background gap={22} size={1} color="#232b38" />
          <Controls showInteractive={false} position="bottom-left" />
          <MiniMap pannable zoomable position="bottom-right" nodeColor={(n: any) => KIND[n.data?.n?.type]?.tone || "#7a8496"} maskColor="rgba(10,13,18,0.7)" style={{ background: "#11151c", border: "1px solid #232b38", width: 150, height: 96 }} />
          <Panel position="top-left">
            <div className="flex items-center gap-1.5 rounded-lg border border-line bg-bg-1/95 px-2 py-1.5 text-xs dim">
              <Network className="size-3.5 text-accent" /><span>{g.nodes.length} nodes</span>
              <span className="text-faint">·</span><span>drag to arrange, scroll to zoom, double-click to open</span>
              <button className="btn quiet sm !h-6 !px-1.5" onClick={() => save({ positions: {}, collapsed: [], viewport: null })} title="Return every node to the automatic layout">Reset layout</button>
              <button className="btn quiet sm !h-6 !px-1.5" onClick={() => onWide(!wide)} title={wide ? "Show the conversation again" : "Give the canvas the whole window"} data-testid="canvas-wide">{wide ? <Minimize2 /> : <Maximize2 />}</button>
            </div>
          </Panel>
        </ReactFlow>
      </Ctx.Provider>
      {node && (
        <div className="absolute right-3 top-3 z-10 flex max-h-[calc(100%-24px)] w-[330px] flex-col overflow-hidden rounded-[12px] border border-line-2 bg-bg-1 shadow-xl" data-testid="node-detail">
          <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5">
            <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: KIND[node.type]?.tone }}>{KIND[node.type]?.label}</span><span className="flex-1" />
            <button className="btn quiet sm !px-1" onClick={() => setSel(null)} aria-label="Close"><X /></button>
          </div>
          <div className="min-h-0 flex-1 overflow-auto px-3.5 py-3">
            <div className="text-[13.5px] font-medium leading-snug">{node.title}</div>
            {node.summary && <div className="mt-1.5 text-[12.5px] soft">{node.summary}</div>}
            {node.stats && node.stats.length > 0 && <div className="mt-2.5 flex flex-col gap-1">{node.stats.map((s) => <div key={s.label} className="flex justify-between gap-3 text-xs"><span className="dim">{s.label}</span><span className="num text-right">{s.value ?? "—"}</span></div>)}</div>}
            {node.items && node.items.length > 0 && (
              <div className="mt-3"><div className="mb-1 text-[11px] font-medium uppercase tracking-wide dim">Configuration</div>
                {node.items.map((it, i) => <div key={i} className="flex gap-2 py-0.5 text-xs"><span className="w-[84px] shrink-0 dim">{it.label}</span><span className="min-w-0 flex-1">{String(it.value).split("   [")[0]}{it.status === "assumed" && <span className="ml-1 text-warn">(assumed)</span>}</span></div>)}</div>
            )}
            {node.assumptions && node.assumptions.length > 0 && (
              <div className="mt-3"><div className="mb-1 text-[11px] font-medium uppercase tracking-wide dim">Assumptions</div>{node.assumptions.map((a, i) => <div key={i} className="py-0.5 text-xs soft">· {a}</div>)}</div>
            )}
            <div className="mt-3 flex flex-wrap gap-1.5">
              <button className="btn sm" onClick={() => open(node)} data-testid="node-open"><FileText />Open</button>
              {node.type === "candidate" && node.run_id && node.key && (
                <>
                  <button className="btn sm" disabled={busy} onClick={() => onSelect(node.run_id!, node.key!, "study")}><Layers />Investigate further</button>
                  <button className="btn sm" disabled={busy} onClick={() => onSelect(node.run_id!, node.key!, "options")}><FlaskConical />Option structures</button>
                </>
              )}
              {(node.type === "study" || node.type === "backtest") && node.run_id && node.status === "done" && <button className="btn sm" disabled={busy} onClick={() => onAnalyse("challenge", node.run_id!)}><ShieldAlert />Challenge</button>}
              {node.type === "backtest" && node.run_id && node.status === "done" && <button className="btn sm" disabled={busy} onClick={() => onAnalyse("stress", node.run_id!)}><Zap />Stress execution</button>}
            </div>
            {canBranch && node.version_id !== undefined && (
              <div className="mt-3.5 border-t border-line pt-3">
                <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide dim"><GitBranch className="size-3.5" />Branch from here</div>
                <div className="flex flex-col gap-1">
                  {(BRANCHES[node.type] || []).map((b) => <button key={b} className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-xs soft hover:bg-bg-2 hover:text-fg" disabled={busy} onClick={() => onBranch(b, node.version_id)}><CornerDownRight className="size-3 shrink-0 text-faint" />{b}</button>)}
                </div>
                <div className="mt-2 flex gap-1.5">
                  <input className="field !h-8 flex-1" placeholder="or describe the change, e.g. hold for 3 sessions" value={text} onChange={(e) => setText(e.target.value)} aria-label="Branch request" data-testid="branch-input"
                    onKeyDown={(e) => { if (e.key === "Enter" && text.trim()) { onBranch(text.trim(), node.version_id); setText(""); } }} />
                  <button className="btn primary sm" disabled={busy || !text.trim()} onClick={() => { onBranch(text.trim(), node.version_id); setText(""); }}>Branch</button>
                </div>
                <div className="mt-1.5 text-[11px] text-faint">The new node starts from this node's configuration and stays connected to it.</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
