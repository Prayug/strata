import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { DataTable, Empty, Loading, PageHeader, Pnl, StatusDot, SyntheticBadge, type Col } from "@/components/common";
import { Input, Segmented } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, money, num, pct } from "@/lib/format";
import type { RunRow } from "@/lib/types";
import { RunDetail } from "./RunDetail";

export default function Results() {
  const { id } = useParams();
  if (id) return <RunDetail id={id} />;
  return <RunList />;
}

function Spark({ data }: { data: number[] }) {
  if (!data?.length) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * 80},${18 - ((v - min) / (max - min || 1)) * 16}`).join(" ");
  const up = data[data.length - 1] >= data[0];
  return <svg width="80" height="20"><polyline points={pts} fill="none" stroke={up ? "#26a69a" : "#ef5350"} strokeWidth="1.2" /></svg>;
}

const thin = (r: RunRow) => r.kind === "backtest" && (r.summary?.trades ?? 0) < 10;
const PRICED: Record<string, string> = { study: "underlying only", historical: "bid/ask quotes", trades: "trade prints", modeled: "THEORETICAL", auto: "observed prices" };

function RunList() {
  const nav = useNavigate();
  const [kind, setKind] = useState<"backtest,portfolio,study" | "backtest" | "portfolio" | "study">("backtest,portfolio,study");
  const [q, setQ] = useState("");
  const runs = useQuery({ queryKey: ["runs", kind], queryFn: () => api.get<RunRow[]>(`/api/backtests?kind=${kind}&limit=500`), refetchInterval: 4000 });
  // every word must match somewhere: name, id, symbol, pricing basis, kind or configuration hash
  const rows = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (runs.data || []).filter((r) => {
      const s: any = r.summary || {};
      const hay = `${r.name} ${r.id} ${(s.symbols || []).join(" ")} ${s.mode || ""} ${PRICED[s.mode] || ""} ${r.kind} ${r.config_hash || ""}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    });
  }, [runs.data, q]);
  const cols: Col<RunRow>[] = [
    { key: "st", label: "", render: (r) => <StatusDot status={r.status} />, width: 14 },
    { key: "name", label: "Run", render: (r) => <span className="flex items-center gap-1.5"><span className="truncate max-w-[340px] text-fg">{r.name}</span>{r.kind === "portfolio" && <span className="text-2xs text-info">PORTFOLIO</span>}{r.kind === "study" && <span className="text-2xs text-info">STUDY {r.summary?.probability !== null && r.summary?.probability !== undefined ? `${(r.summary.probability * 100).toFixed(1)}% of ${r.summary.qualifying_sessions}` : ""}</span>}<SyntheticBadge on={r.synthetic} /></span>, sort: (r) => r.name },
    { key: "mode", label: "Priced on", render: (r) => { const m = (r.summary as any)?.mode || (r.kind === "study" ? "study" : null); return m ? <span className={m === "modeled" ? "text-warn" : m === "trades" ? "text-[#56b6c2]" : "text-fg-2"} title={m === "modeled" ? "Option premiums are theoretical estimates, not observed prices" : undefined}>{PRICED[m] || m}</span> : <span className="text-faint">—</span>; } },
    { key: "period", label: "Period", render: (r) => <span className="num text-fg-2">{r.summary?.start} → {r.summary?.end}</span> },
    { key: "spark", label: "Equity", render: (r) => <Spark data={r.summary?.equity_spark} /> },
    { key: "trades", label: "Trades", align: "r", render: (r) => <span className="num">{r.summary?.trades ?? "—"}</span>, sort: (r) => r.summary?.trades },
    { key: "pnl", label: "Net P&L", align: "r", render: (r) => <Pnl v={r.summary?.net_pnl} fmt={(x) => money(x, 0, true)} />, sort: (r) => r.summary?.net_pnl },
    // annualised figures from a handful of trades describe nothing: they are withheld below ten trades
    { key: "cagr", label: "CAGR", align: "r", render: (r) => (thin(r) ? <span className="text-faint" title="Fewer than 10 trades: an annualised return would be meaningless">—</span> : <Pnl v={r.summary?.cagr} fmt={(x) => pct(x, 1)} />), sort: (r) => (thin(r) ? undefined : r.summary?.cagr) },
    { key: "sharpe", label: "Sharpe", align: "r", render: (r) => (thin(r) ? <span className="text-faint" title="Fewer than 10 trades: a Sharpe ratio would be meaningless">—</span> : <span className="num">{num(r.summary?.sharpe)}</span>), sort: (r) => (thin(r) ? undefined : r.summary?.sharpe) },
    { key: "dd", label: "Max DD", align: "r", render: (r) => <span className="num text-down">{pct(r.summary?.max_drawdown, 1)}</span>, sort: (r) => r.summary?.max_drawdown },
    { key: "wr", label: "Win %", align: "r", render: (r) => <span className="num">{pct(r.summary?.win_rate, 0)}</span>, sort: (r) => r.summary?.win_rate },
    { key: "pf", label: "PF", align: "r", render: (r) => <span className="num">{num(r.summary?.profit_factor)}</span>, sort: (r) => r.summary?.profit_factor },
    { key: "hash", label: "Config", render: (r) => <span className="font-mono text-2xs text-faint">{r.config_hash}</span> },
    { key: "created", label: "Created", align: "r", render: (r) => <span className="text-muted">{r.status === "running" ? `${Math.round(r.progress * 100)}%` : ago(r.created_at)}</span>, sort: (r) => r.created_at },
  ];
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Backtests" sub={<span className="text-2xs text-muted">{rows.length} runs</span>}>
        <Input sz="sm" className="w-52" placeholder="Search: name, symbol, id, pricing…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Segmented value={kind} onChange={setKind} options={[{ value: "backtest,portfolio,study", label: "All" }, { value: "backtest", label: "Option backtests" }, { value: "study", label: "Market studies" }, { value: "portfolio", label: "Portfolios" }]} />
      </PageHeader>
      <div className="min-h-0 flex-1">
        {runs.isLoading ? <Loading /> : rows.length === 0 ? (
          <Empty icon={BarChart3} title="No runs yet">Ask a question in the AI Lab, or build a strategy in the Strategy Builder and press Run (⌘↵). Every run is stored with its configuration hash, dataset version and engine version so it can be reproduced.</Empty>
        ) : (
          <DataTable rows={rows} cols={cols} rowKey={(r) => r.id} onRowClick={(r) => nav(r.kind === "portfolio" ? `/portfolio/${r.id}` : `/results/${r.id}`)} defaultSort={{ key: "created", dir: -1 }} />
        )}
      </div>
    </div>
  );
}
