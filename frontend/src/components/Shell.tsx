import { useQuery } from "@tanstack/react-query";
import { Command } from "cmdk";
import {
  Activity, BarChart3, Bell, Blocks, BookOpenText, BrainCircuit, Briefcase, ChevronsLeft, ChevronsRight, Code2, Database, FlaskConical, Grid3x3, Library, Microscope, PlayCircle, Scale, Settings as SettingsIcon, Sigma, Snowflake,
  Sparkles, Waves,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Kbd, Popover, PopoverContent, PopoverTrigger, Tip } from "@/components/ui";
import { StatusDot } from "@/components/common";
import { api, queryClient } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { useUi } from "@/lib/store";

/* The application is organised around three things people come to do. Everything else supports them.
     Explore   ask market questions, investigate hypotheses, browse patterns
     Build     turn an idea into a strategy: in words, with the visual builder, or in the IDE
     Validate  backtest, challenge, compare, and follow a frozen configuration forward
   `adv` items are tools for people who want them; simple mode keeps them out of the way (one engine either way). */
type Item = { to: string; label: string; icon: any; key: string; match?: string[]; adv?: boolean };
export const GROUPS: { id: string; label: string; hint: string; tone: string; items: Item[] }[] = [
  { id: "explore", label: "Explore", hint: "Ask questions, investigate, discover", tone: "var(--color-accent)", items: [
    { to: "/", label: "Ask", icon: Sparkles, key: "a", match: ["/lab"] },
    { to: "/patterns", label: "Patterns", icon: Waves, key: "w" },
    { to: "/findings", label: "Knowledge base", icon: Library, key: "k" },
    { to: "/encyclopedia", label: "Market behaviour", icon: BookOpenText, key: "e", adv: true },
    { to: "/sensitivity", label: "Sensitivity", icon: Grid3x3, key: "n", adv: true },
    { to: "/ml", label: "ML research", icon: BrainCircuit, key: "l", adv: true },
  ] },
  { id: "build", label: "Build", hint: "Describe, assemble or code a strategy", tone: "var(--color-info)", items: [
    { to: "/ide", label: "Strategy IDE", icon: Code2, key: "i" },
    { to: "/builder", label: "Visual builder", icon: Blocks, key: "b", match: ["/strategies"] },
    { to: "/explore", label: "Data explorer", icon: Microscope, key: "m", adv: true },
  ] },
  { id: "validate", label: "Validate", hint: "Backtest, challenge, follow forward", tone: "var(--color-up)", items: [
    { to: "/results", label: "Backtests", icon: BarChart3, key: "r" },
    { to: "/forward", label: "Forward validation", icon: Snowflake, key: "f" },
    { to: "/registry", label: "Experiment registry", icon: Scale, key: "x" },
    { to: "/optimize", label: "Optimization", icon: Sigma, key: "o", adv: true },
    { to: "/replay", label: "Trade replay", icon: PlayCircle, key: "t", adv: true },
    { to: "/portfolio", label: "Portfolio", icon: Briefcase, key: "p", adv: true },
  ] },
  { id: "support", label: "", hint: "", tone: "var(--color-muted)", items: [
    { to: "/data", label: "Data", icon: Database, key: "d" },
    { to: "/settings", label: "Settings", icon: SettingsIcon, key: "s" },
  ] },
];
const NAV: Item[] = GROUPS.flatMap((g) => g.items);

function Notifications() {
  const q = useQuery({ queryKey: ["notifications"], queryFn: () => api.get<{ notifications: any[]; unread: number }>("/api/notifications?limit=30"), refetchInterval: 30_000 });
  const n = q.data?.unread || 0;
  const read = async () => { if (n) { await api.post("/api/notifications/read", {}); queryClient.invalidateQueries({ queryKey: ["notifications"] }); } };
  return (
    <Popover onOpenChange={(o) => { if (!o) read(); }}>
      <PopoverTrigger asChild>
        <button className="relative flex items-center gap-1 hover:text-fg" aria-label={`Notifications${n ? `, ${n} unread` : ""}`} data-testid="notifications">
          <Bell className={cn("size-3", n > 0 && "text-accent")} />{n > 0 && <span className="num text-accent">{n}</span>}
        </button>
      </PopoverTrigger>
      <PopoverContent side="top" align="end" className="w-[380px] !p-0">
        <div className="border-b border-line px-3 py-2 text-xs font-medium">What happened while you were away</div>
        <div className="max-h-[340px] overflow-auto">
          {(q.data?.notifications || []).length === 0 && <div className="px-3 py-5 text-xs text-muted">Nothing yet. Collection cycles and forward evaluations leave a note here. Strata sends nothing to any outside service.</div>}
          {(q.data?.notifications || []).map((x) => (
            <a key={x.id} href={x.link || "#"} className={cn("block border-b border-line/60 px-3 py-2 text-xs hover:bg-bg-2", !x.read && "bg-accent/[0.04]")}>
              <div className="flex items-center gap-2"><span className={cn("size-1.5 rounded-full", x.level === "error" ? "bg-down" : x.level === "warn" ? "bg-warn" : "bg-up")} /><span className="font-medium text-fg">{x.title}</span><span className="ml-auto shrink-0 text-faint">{ago(x.created_at)}</span></div>
              {x.body && <div className="mt-0.5 pl-3.5 text-muted">{x.body}</div>}
            </a>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function Shell() {
  const nav = useNavigate();
  const loc = useLocation();
  const { navCollapsed, mode, set } = useUi();
  const [palette, setPalette] = useState(false);
  const cov = useQuery({ queryKey: ["coverage"], queryFn: () => api.get<any[]>("/api/data/coverage"), staleTime: 60_000 });
  const real = (cov.data || []).filter((r) => !r.synthetic);
  const sandbox = (cov.data || []).filter((r) => r.synthetic);
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.get<{ active: any[]; recent: any[] }>("/api/jobs"), refetchInterval: 3000 });
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<any>("/api/health"), refetchInterval: 15000, retry: false });
  const advanced = mode === "advanced";
  const isOn = (n: Item) => (n.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(n.to)) || (n.match || []).some((m) => loc.pathname.startsWith(m));

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
        <aside className={cn("flex shrink-0 flex-col border-r border-line bg-bg-1 transition-[width] duration-150", navCollapsed ? "w-11" : "w-[196px]")}>
          <div className="flex h-10 items-center gap-2 border-b border-line px-3">
            <svg viewBox="0 0 32 32" className="size-4 shrink-0"><path d="M4 24h24M4 16h16M4 8h8" stroke="#e8a33d" strokeWidth="3.5" /></svg>
            {!navCollapsed && <span className="text-sm font-semibold tracking-wide">STRATA</span>}
          </div>
          <nav className="flex flex-1 flex-col overflow-y-auto p-1.5" aria-label="Workflows">
            {GROUPS.map((grp) => {
              const items = grp.items.filter((n) => advanced || !n.adv || isOn(n));
              const here = grp.items.some(isOn);
              return (
                <div key={grp.id} className={cn(grp.id === "support" ? "mt-auto pt-2" : "mb-2")} data-testid={`nav-${grp.id}`}>
                  {grp.label && !navCollapsed && (
                    <Tip content={grp.hint} side="right">
                      <div className="flex h-6 items-center gap-1.5 px-2 text-[10.5px] font-semibold uppercase tracking-[0.09em]" style={{ color: here ? grp.tone : "var(--color-faint)" }}>
                        <span className="size-1 rounded-full" style={{ background: grp.tone, opacity: here ? 1 : 0.5 }} />{grp.label}
                      </div>
                    </Tip>
                  )}
                  {grp.label && navCollapsed && <div className="mx-auto my-1 h-px w-5" style={{ background: grp.tone, opacity: here ? 0.9 : 0.3 }} />}
                  <div className="flex flex-col gap-px">
                    {items.map((n) => (
                      <Tip key={n.to} content={navCollapsed ? `${n.label}  (g ${n.key})` : null} side="right">
                        <NavLink to={n.to} end={n.to === "/"}
                          className={cn("flex h-7 items-center gap-2.5 rounded-sm px-2 text-xs text-muted transition-colors hover:bg-bg-3 hover:text-fg", isOn(n) && "bg-bg-3 text-fg")}
                          style={isOn(n) ? { boxShadow: `inset 2px 0 0 ${grp.tone}` } : undefined}>
                          <n.icon className="size-3.5 shrink-0" />
                          {!navCollapsed && <span className="flex-1 truncate">{n.label}</span>}
                          {!navCollapsed && <span className="font-mono text-[9.5px] text-faint">g{n.key}</span>}
                        </NavLink>
                      </Tip>
                    ))}
                  </div>
                </div>
              );
            })}
          </nav>
          {!navCollapsed && (
            <div className="border-t border-line px-2 py-1.5">
              <Tip side="right" content="Simple shows the main path through each workflow. Advanced adds the specialist tools. The calculations are identical in both.">
                <div className="flex rounded-sm border border-line-2 bg-bg-2 p-0.5 text-2xs" role="group" aria-label="Interface detail" data-testid="mode-toggle">
                  {(["simple", "advanced"] as const).map((m) => (
                    <button key={m} onClick={() => { set({ mode: m }); api.put("/api/settings", { settings: { "ui.mode": m } }).catch(() => {}); }}
                      className={cn("h-5 flex-1 rounded-[2px] capitalize transition-colors", mode === m ? "bg-bg-4 text-fg" : "text-muted hover:text-fg")} aria-pressed={mode === m}>{m}</button>
                  ))}
                </div>
              </Tip>
            </div>
          )}
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
        <Notifications />
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
