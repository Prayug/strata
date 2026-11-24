import { useQuery } from "@tanstack/react-query";
import { Command } from "cmdk";
import {
  Activity, BarChart3, Blocks, Briefcase, ChevronsLeft, ChevronsRight, Database, FlaskConical, FolderOpen, LayoutDashboard, Microscope, PlayCircle, Sigma,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { Kbd, Tip } from "@/components/ui";
import { StatusDot } from "@/components/common";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { useUi } from "@/lib/store";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, key: "o" },
  { to: "/builder", label: "Strategy Builder", icon: Blocks, key: "b" },
  { to: "/results", label: "Backtest Results", icon: BarChart3, key: "r" },
  { to: "/replay", label: "Trade Replay", icon: PlayCircle, key: "t" },
  { to: "/optimize", label: "Optimization Lab", icon: Sigma, key: "l" },
  { to: "/research", label: "Market Research", icon: Microscope, key: "m" },
  { to: "/portfolio", label: "Portfolio", icon: Briefcase, key: "p" },
  { to: "/strategies", label: "Saved Strategies", icon: FolderOpen, key: "s" },
  { to: "/data", label: "Data Management", icon: Database, key: "d" },
];

export function Shell() {
  const nav = useNavigate();
  const { navCollapsed, set } = useUi();
  const [palette, setPalette] = useState(false);
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.get<{ active: any[]; recent: any[] }>("/api/jobs"), refetchInterval: 3000 });
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<any>("/api/health"), refetchInterval: 15000, retry: false });

  useEffect(() => {
    let g = false;
    let timer: any;
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (e.target as HTMLElement)?.isContentEditable;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette((p) => !p);
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
        e.preventDefault();
        set({ navCollapsed: !useUi.getState().navCollapsed });
        return;
      }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "g") {
        g = true;
        clearTimeout(timer);
        timer = setTimeout(() => (g = false), 900);
        return;
      }
      if (g) {
        const n = NAV.find((x) => x.key === e.key);
        if (n) {
          nav(n.to);
          g = false;
        }
      }
      if (e.key === "?") setPalette(true);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [nav, set]);

  const active = jobs.data?.active || [];
  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <aside className={cn("flex shrink-0 flex-col border-r border-line bg-bg-1 transition-[width] duration-150", navCollapsed ? "w-11" : "w-48")}>
          <div className="flex h-10 items-center gap-2 border-b border-line px-3">
            <svg viewBox="0 0 32 32" className="size-4 shrink-0"><path d="M4 24h24M4 16h16M4 8h8" stroke="#e8a33d" strokeWidth="3.5" /></svg>
            {!navCollapsed && <span className="text-sm font-semibold tracking-wide">STRATA</span>}
          </div>
          <nav className="flex flex-1 flex-col gap-px p-1.5">
            {NAV.map((n) => (
              <Tip key={n.to} content={navCollapsed ? `${n.label}  (g ${n.key})` : null} side="right">
                <NavLink to={n.to} end={n.to === "/"}
                  className={({ isActive }) => cn("flex h-7 items-center gap-2.5 rounded-sm px-2 text-xs text-muted hover:bg-bg-3 hover:text-fg transition-colors",
                    isActive && "bg-bg-3 text-fg shadow-[inset_2px_0_0_var(--color-accent)]")}>
                  <n.icon className="size-3.5 shrink-0" />
                  {!navCollapsed && <span className="flex-1 truncate">{n.label}</span>}
                  {!navCollapsed && <span className="font-mono text-[9.5px] text-faint">g{n.key}</span>}
                </NavLink>
              </Tip>
            ))}
          </nav>
          <button className="flex h-8 items-center gap-2 border-t border-line px-3 text-2xs text-faint hover:text-fg" onClick={() => set({ navCollapsed: !navCollapsed })}>
            {navCollapsed ? <ChevronsRight className="size-3.5" /> : <><ChevronsLeft className="size-3.5" /> Collapse <Kbd>⌘\</Kbd></>}
          </button>
        </aside>
        <main className="flex min-w-0 flex-1 flex-col bg-bg-0">
          <Outlet />
        </main>
      </div>
      <footer className="flex h-6 shrink-0 items-center gap-4 border-t border-line bg-bg-1 px-3 text-2xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className={cn("size-1.5 rounded-full", health.isError ? "bg-down" : "bg-up")} />
          {health.isError ? "API offline" : `engine ${health.data?.engine_version ?? ""}`}
        </span>
        {active.length > 0 ? (
          <span className="flex items-center gap-1.5 text-fg-2">
            <Activity className="size-3 text-accent" />
            {active.map((j) => (
              <span key={j.id} className="flex items-center gap-1"><StatusDot status={j.status} />{j.kind} {j.status === "running" ? `${Math.round((j.progress || 0) * 100)}%` : j.status}</span>
            ))}
          </span>
        ) : <span>no running jobs</span>}
        <span className="flex-1" />
        <span className="flex items-center gap-1"><FlaskConical className="size-3 text-synth" /> demo symbols DEMO / DEMOX / DEMOVIX are synthetic</span>
        <button onClick={() => setPalette(true)} className="flex items-center gap-1 hover:text-fg"><Kbd>⌘K</Kbd> commands</button>
      </footer>
      {palette && <Palette onClose={() => setPalette(false)} />}
    </div>
  );
}

function Palette({ onClose }: { onClose: () => void }) {
  const nav = useNavigate();
  const runs = useQuery({ queryKey: ["runs", "palette"], queryFn: () => api.get<any[]>("/api/backtests?kind=backtest,portfolio,optimization&limit=30") });
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies") });
  const go = (to: string) => {
    nav(to);
    onClose();
  };
  const shortcuts = useMemo(() => [
    ["⌘K", "Command palette"], ["⌘↵", "Run backtest (builder)"], ["⌘S", "Save strategy (builder)"], ["⌘\\", "Collapse sidebar"], ["g + key", "Go to page"],
    ["space / ← →", "Play / step (trade replay)"],
  ], []);
  return (
    <div className="cmdk-overlay" onClick={onClose}>
      <Command className="mx-auto mt-[14vh] w-[560px] overflow-hidden rounded-md border border-line-2 bg-bg-1 shadow-2xl" onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === "Escape" && onClose()}>
        <Command.Input autoFocus placeholder="Jump to a page, run or strategy…" className="h-10 w-full border-b border-line bg-transparent px-3 text-sm outline-none placeholder:text-faint" />
        <Command.List className="max-h-[52vh] overflow-auto p-1 text-xs">
          <Command.Empty className="p-3 text-muted">No matches.</Command.Empty>
          <Command.Group heading="Pages" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {NAV.map((n) => (
              <Command.Item key={n.to} onSelect={() => go(n.to)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <n.icon className="size-3.5 text-muted" /> {n.label} <span className="ml-auto font-mono text-faint">g {n.key}</span>
              </Command.Item>
            ))}
          </Command.Group>
          <Command.Group heading="Recent runs" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {(runs.data || []).map((r) => (
              <Command.Item key={r.id} value={`${r.name} ${r.id}`} onSelect={() => go(r.kind === "optimization" ? `/optimize/${r.id}` : `/results/${r.id}`)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <StatusDot status={r.status} /> <span className="truncate">{r.name}</span> <span className="ml-auto text-faint">{r.kind}</span>
              </Command.Item>
            ))}
          </Command.Group>
          <Command.Group heading="Strategies" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {(strategies.data || []).map((s) => (
              <Command.Item key={s.id} value={`strategy ${s.name}`} onSelect={() => go(`/builder?strategy=${s.id}`)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <Blocks className="size-3.5 text-muted" /> {s.name} <span className="ml-auto text-faint">{s.symbol}</span>
              </Command.Item>
            ))}
          </Command.Group>
          <Command.Group heading="Keyboard shortcuts" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {shortcuts.map(([k, l]) => (
              <Command.Item key={k} value={`shortcut ${l}`} className="flex items-center gap-2 rounded-sm px-2 py-1 text-muted data-[selected=true]:bg-bg-3">
                <span className="w-28 font-mono text-fg-2">{k}</span> {l}
              </Command.Item>
            ))}
          </Command.Group>
        </Command.List>
      </Command>
    </div>
  );
}
