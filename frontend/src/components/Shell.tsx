import { useQuery } from "@tanstack/react-query";
import { Command } from "cmdk";
import {
  Activity, BarChart3, Blocks, Briefcase, ChevronsLeft, ChevronsRight, Database, FlaskConical, Microscope, PlayCircle, Settings as SettingsIcon, Sigma, Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Kbd, Tip } from "@/components/ui";
import { StatusDot } from "@/components/common";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import { useUi } from "@/lib/store";

const NAV = [
  { to: "/", label: "AI Lab", icon: Sparkles, key: "a", match: ["/lab"] },
  { to: "/explore", label: "Market Explorer", icon: Microscope, key: "m" },
  { to: "/builder", label: "Strategy Builder", icon: Blocks, key: "b", match: ["/strategies"] },
  { to: "/results", label: "Backtests", icon: BarChart3, key: "r" },
  { to: "/optimize", label: "Optimization", icon: Sigma, key: "o" },
  { to: "/replay", label: "Trade Replay", icon: PlayCircle, key: "t" },
  { to: "/portfolio", label: "Portfolio", icon: Briefcase, key: "p" },
  { to: "/data", label: "Data", icon: Database, key: "d" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, key: "s" },
];

export function Shell() {
  const nav = useNavigate();
  const loc = useLocation();
  const { navCollapsed, set } = useUi();
  const [palette, setPalette] = useState(false);
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 60_000 });
  const real = (cov.data || []).filter((r) => !r.synthetic);
  const sandbox = (cov.data || []).filter((r) => r.synthetic);
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
                    (isActive || (n.match || []).some((m) => loc.pathname.startsWith(m))) && "bg-bg-3 text-fg shadow-[inset_2px_0_0_var(--color-accent)]")}>
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
        {cov.isSuccess && (real.length ? (
          <span className="flex items-center gap-1"><Database className="size-3" /> real market data: {real.filter((r) => r.symbol !== "VIX").map((r) => r.symbol).slice(0, 6).join(" · ")}</span>
        ) : <NavLink to="/data" className="flex items-center gap-1 text-warn hover:underline"><Database className="size-3" /> no market data stored - get free data</NavLink>)}
        {sandbox.length > 0 && <span className="flex items-center gap-1" title="Simulated sandbox symbols are stored. They are never used unless a strategy names them."><FlaskConical className="size-3 text-synth" /> {sandbox.length} simulated sandbox symbol{sandbox.length > 1 ? "s" : ""}</span>}
        <button onClick={() => setPalette(true)} className="flex items-center gap-1 hover:text-fg"><Kbd>⌘K</Kbd> commands</button>
      </footer>
      {palette && <Palette onClose={() => setPalette(false)} />}
    </div>
  );
}

function Palette({ onClose }: { onClose: () => void }) {
  const nav = useNavigate();
  // every run and every research notebook is searchable here: by name, symbol, id, pricing basis, or a question that was asked
  const runs = useQuery({ queryKey: ["runs", "palette"], queryFn: () => api.get<any[]>("/api/runs/search?limit=300") });
  const sessions = useQuery({ queryKey: ["notebooks", ""], queryFn: () => api.get<any[]>("/api/ai/search?q=&limit=200") });
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => api.get<any[]>("/api/strategies") });
  const go = (to: string) => {
    nav(to);
    onClose();
  };
  const shortcuts = useMemo(() => [
    ["⌘K", "Command palette"], ["↵", "Send a request (AI Lab)"], ["⌘↵", "Run backtest (builder)"], ["⌘S", "Save strategy (builder)"], ["⌘\\", "Collapse sidebar"], ["g + key", "Go to page"],
    ["space / ← →", "Play / step (trade replay)"],
  ], []);
  return (
    <div className="cmdk-overlay" onClick={onClose}>
      <Command className="mx-auto mt-[14vh] w-[560px] overflow-hidden rounded-md border border-line-2 bg-bg-1 shadow-2xl" onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === "Escape" && onClose()}>
        <Command.Input autoFocus placeholder="Search research, runs and strategies, or jump to a page…" className="h-10 w-full border-b border-line bg-transparent px-3 text-sm outline-none placeholder:text-faint" />
        <Command.List className="max-h-[52vh] overflow-auto p-1 text-xs">
          <Command.Empty className="p-3 text-muted">No matches.</Command.Empty>
          <Command.Group heading="Pages" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {NAV.map((n) => (
              <Command.Item key={n.to} onSelect={() => go(n.to)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <n.icon className="size-3.5 text-muted" /> {n.label} <span className="ml-auto font-mono text-faint">g {n.key}</span>
              </Command.Item>
            ))}
          </Command.Group>
          <Command.Group heading="Research sessions" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {(sessions.data || []).map((x) => (
              <Command.Item key={x.id} value={`research ${x.title} ${x.id} ${x.match || ""}`} onSelect={() => go(`/lab/${x.id}`)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <Sparkles className="size-3.5 text-muted" /> <span className="truncate">{x.title}</span>
              </Command.Item>
            ))}
          </Command.Group>
          <Command.Group heading="Runs" className="[&_[cmdk-group-heading]]:label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
            {(runs.data || []).map((r) => (
              <Command.Item key={r.id} value={`run ${r.name} ${r.id} ${(r.symbols || []).join(" ")} ${r.mode || ""} ${r.kind}`} onSelect={() => go(r.kind === "portfolio" ? `/portfolio/${r.id}` : `/results/${r.id}`)} className="flex items-center gap-2 rounded-sm px-2 py-1.5 data-[selected=true]:bg-bg-3">
                <StatusDot status={r.status} /> <span className="truncate">{r.name}</span> <span className={cn("ml-auto whitespace-nowrap", r.mode === "modeled" ? "text-warn" : "text-faint")}>{r.mode === "modeled" ? "theoretical" : r.mode === "trades" ? "trade prints" : r.mode === "historical" ? "quotes" : r.kind}</span>
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
