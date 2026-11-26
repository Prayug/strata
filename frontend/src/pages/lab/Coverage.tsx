/* Coverage comes before profitability.
   How many of the sessions a strategy would have traded can actually be priced on stored observations is shown first,
   large, wherever a result is shown: a profit computed on one session out of a hundred is a statement about one session. */
import { ArrowRight, Check, CircleDashed, CircleDot, Database, FlaskConical, Minus, ShieldAlert, X } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Tip } from "@/components/ui";
import { cn } from "@/lib/format";
import { Callout, Section } from "./bits";

type Counts = { quotes?: number; trades?: number; eod?: number; exit_uncertain?: number; stale?: number; contract?: number; expiration?: number; partial?: number; no_contract?: number; missing?: number };
/** Every check passed on stored observations: the contract the strategy would select, a usable print at entry, the exit window. */
export const PRICED = ["quotes", "trades", "eod"];
export interface DataCheck {
  symbol?: string; signal_sessions: number; priced: number; priced_pct?: number | null; counts: Counts; first?: string; last?: string; intraday?: boolean; text?: string;
  contract_existence?: { known_sessions: number; unknown_sessions: number; no_contract: number; note?: string } | null;
  alternatives?: Alt[]; budget?: Budget | null; blocking?: boolean; potential?: number; fetchable?: number;
  sessions?: { date: string; status: string; stored: string[]; time?: string | null; detail?: string | null; downloadable?: { source: string; label: string } | null;
    entry?: { time: string; contracts: string[]; print_age_seconds: number } | null; exit?: { planned: string; longest_gap_minutes: number } | null }[];
  settings?: { max_trade_age_seconds?: number; max_fill_delay_minutes?: number; signal_interval_minutes?: number; latency_seconds?: number; hold_sessions?: number };
}
export interface Alt { id: string; label: string; detail?: string; available?: boolean; warning?: boolean; cost?: number }
export interface Budget {
  checked?: string; testable_sessions?: number; no_contract_sessions?: number; earliest_source_date?: string | null; before_any_source?: number; reachable_sessions?: number; denominator_note?: string;
  budgets: { budget: number; best: null | { label: string; cost: number; sessions: number; sessions_with_bid_ask: number; sessions_trade_prints_only: number; of: number; needs?: string[] } }[];
}

const PARTS: { key: keyof Counts; label: string; cls: string; help: string }[] = [
  { key: "quotes", label: "priced on bid/ask quotes", cls: "bg-up", help: "Entry and exit can be taken from stored bid/ask quotes." },
  { key: "trades", label: "priced on trade prints", cls: "bg-info", help: "The contract this strategy would select is stored, it traded inside the entry window, and a print exists near the planned exit. No bid/ask exists for these, so the spread is unknown." },
  { key: "eod", label: "priced on end-of-day quotes", cls: "bg-up/55", help: "End-of-day bid/ask only. Enough for a position entered and exited at the close; never used for an intraday entry." },
  { key: "exit_uncertain", label: "entry priced, exit not observed", cls: "bg-info/45", help: "The entry can be priced, but no observation is stored near the planned exit, or the contract is not stored on a later session of the holding period. A run would defer the exit or settle the contract at expiration, and flags the trade." },
  { key: "stale", label: "no trade at the entry", cls: "bg-warn", help: "The contract is stored, but it did not trade inside the entry window: no print recent enough to fill at. An older or a later print is not used." },
  { key: "contract", label: "strike not stored", cls: "bg-warn/65", help: "The strike this strategy needs is outside the strikes stored for the session, typically after the underlying opened far from the previous close. A neighbouring strike is not substituted." },
  { key: "expiration", label: "expiration not stored", cls: "bg-warn/45", help: "Option data is stored for the session, but not an expiration this strategy can use. A different expiration is not substituted." },
  { key: "partial", label: "session held only in part", cls: "bg-warn/30", help: "The provider holds option prices for part of the session only, so the engine does not run it." },
  { key: "missing", label: "no option prices stored", cls: "bg-down/75", help: "The rule fired, and nothing is stored that could price the option. Not tested; not counted as a win, a loss or 'no signal'." },
  { key: "no_contract", label: "no such contract was listed", cls: "bg-faint/60", help: "No contract with the required expiration was listed that day. No data source can make this session testable." },
];

const pct = (a: number, b: number) => (b > 0 ? (a / b) * 100 : 0);
const fmtPct = (x: number) => (x === 0 ? "0%" : x < 0.1 ? "<0.1%" : x < 10 ? `${x.toFixed(1)}%` : `${Math.round(x)}%`);

/** The fraction itself. `n` of `of`, as large as anything else on the page. */
export function CoverageHero({ n, of, counts, first, last, what = "qualifying sessions pass every check on stored option observations", onOpen, className, testid = "coverage" }:
  { n: number; of: number; counts?: Counts; first?: string; last?: string; what?: string; onOpen?: () => void; className?: string; testid?: string }) {
  const p = pct(n, of);
  const tone = of === 0 ? "none" : p < 20 ? "bad" : p < 80 ? "warn" : "good";
  const box = { bad: "border-down/50 bg-down/[0.07]", warn: "border-warn/45 bg-warn/[0.06]", good: "border-up/35 bg-up/[0.05]", none: "border-line" }[tone];
  const num = { bad: "text-down", warn: "text-warn", good: "text-up", none: "" }[tone];
  const parts = counts ? PARTS.filter((x) => (counts[x.key] || 0) > 0) : [];
  return (
    <div className={cn("rounded-[12px] border px-4 py-3.5", box, className)} data-testid={testid} data-priced={n} data-of={of}>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-1">
        <div className={cn("num text-[34px] font-semibold leading-none tracking-tight", num)}>{n.toLocaleString()}<span className="mx-1.5 text-[22px] font-normal text-faint">of</span>{of.toLocaleString()}</div>
        <div className="min-w-[220px] flex-1 pb-0.5">
          <div className="text-[13.5px] font-medium">{what}</div>
          <div className="text-xs dim">{of === 0 ? "The entry rule never fired in the stored history." : `${fmtPct(p)} of them${first && last ? ` · rule fired ${first} to ${last}` : ""}`}</div>
        </div>
      </div>
      {parts.length > 0 && (
        <>
          <div className="mt-3 flex h-2.5 w-full overflow-hidden rounded-full bg-bg-3" role="img" aria-label={parts.map((x) => `${counts![x.key]} ${x.label}`).join(", ")}>
            {parts.map((x) => <Tip key={x.key} content={`${(counts![x.key] || 0).toLocaleString()} ${x.label}. ${x.help}`}><div className={x.cls} style={{ width: `${pct(counts![x.key] || 0, of)}%`, minWidth: 4 }} /></Tip>)}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
            {parts.map((x) => <Tip key={x.key} content={x.help}><span className="flex items-center gap-1.5"><span className={cn("size-2 rounded-sm", x.cls)} /><span className="num font-medium">{(counts![x.key] || 0).toLocaleString()}</span><span className="dim">{x.label}</span></span></Tip>)}
          </div>
        </>
      )}
      {onOpen && <button className="mt-2.5 flex items-center gap-1 text-xs text-info hover:underline" onClick={onOpen} data-testid="open-data-check"><Database className="size-3" />Which sessions, and what each one is missing<ArrowRight className="size-3" /></button>}
      {tone === "bad" && of > 0 && (
        <div className="mt-2.5 flex gap-2 text-[12.5px] text-down"><ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
          <span>{n === 0 ? "Nothing could be priced, so there is no strategy result at all." : n === 1 ? "Any profit or loss shown below is one trade. It says what happened that day and nothing about the strategy." : `Any profit or loss shown below describes ${n.toLocaleString()} of ${of.toLocaleString()} sessions. It is not a result for the strategy as asked.`}</span>
        </div>
      )}
    </div>
  );
}

/** From a plan's data check (computed before anything runs). */
export function CoverageMeter({ check, onOpen, className }: { check: DataCheck; onOpen?: () => void; className?: string }) {
  return <CoverageHero n={check.priced} of={check.signal_sessions} counts={check.counts} first={check.first} last={check.last} onOpen={onOpen} className={className} testid="data-check" />;
}

/** From a finished backtest's own coverage record. Returns nothing when every qualifying session was priced or the run is a study. */
export function ResultCoverage({ cov, mode, onOpen, className, always }: { cov: any; mode?: string; onOpen?: () => void; className?: string; always?: boolean }) {
  const of = cov?.signals?.qualifying, n = cov?.executable?.signals;
  if (of === undefined || of === null || n === undefined || n === null || mode === "study") return null;
  if (!always && n >= of) return null;
  const issues: any[] = cov?.options?.issues || [];
  return (
    <div className={className}>
      <CoverageHero n={n} of={of} what={mode === "modeled" ? "qualifying sessions were priced, with THEORETICAL premiums" : "qualifying sessions were executed on observed option prices"} onOpen={onOpen} testid="result-coverage" />
      {issues.length > 0 && n < of && <div className="mt-1.5 px-1 text-xs dim">{issues.slice(0, 3).map((x) => `${x.signals.toLocaleString()} not tested: ${x.reason}`).join(" · ")}</div>}
    </div>
  );
}

const STAGE_LABEL: Record<string, string> = { understand: "Understand", validate: "Validate", data: "Check data", plan: "Plan", execute: "Execute", explain: "Explain", followup: "Follow up" };
const STAGE_ORDER = ["understand", "validate", "data", "plan", "execute", "explain", "followup"];
export interface StageInfo { stage: string; status: string; detail?: string; priced?: number; of?: number }

/** The one path every request takes, and where this one is on it. A stage that did not happen is not shown as done. */
export function StageRail({ stages, onStage, className }: { stages: StageInfo[]; onStage?: (id: string) => void; className?: string }) {
  const by = Object.fromEntries(stages.map((s) => [s.stage, s]));
  return (
    <div className={cn("flex items-center gap-0.5 overflow-x-auto", className)} data-testid="stage-rail" aria-label="Where this request is in the research workflow">
      {STAGE_ORDER.map((id, i) => {
        const s: StageInfo | undefined = by[id];
        const st = s?.status || "pending";
        const Icon = st === "done" ? Check : st === "running" ? CircleDot : st === "attention" ? ShieldAlert : st === "blocked" || st === "failed" ? X : st === "skipped" ? Minus : CircleDashed;
        const tone = st === "done" ? "text-up" : st === "running" ? "text-accent" : st === "attention" ? "text-warn" : st === "blocked" || st === "failed" ? "text-down" : "text-faint";
        const body = (
          <button disabled={!onStage || !s} onClick={() => onStage?.(id)} data-stage={id} data-status={st}
            className={cn("flex h-7 shrink-0 items-center gap-1.5 rounded-md px-2 text-xs whitespace-nowrap transition-colors", s ? "hover:bg-bg-2" : "cursor-default opacity-60", st === "attention" && "bg-warn/[0.08]")}>
            <Icon className={cn("size-3.5", tone, st === "running" && "pulse")} strokeWidth={2.4} />
            <span className={cn(s ? "soft" : "dim")}>{STAGE_LABEL[id]}</span>
            {id === "data" && s?.of !== undefined && s?.of !== null && <span className={cn("num rounded-full px-1.5 text-[11px]", st === "attention" ? "bg-warn/20 text-warn" : "bg-bg-3 dim")}>{(s.priced ?? 0).toLocaleString()}/{s.of.toLocaleString()}</span>}
          </button>
        );
        return (
          <span key={id} className="flex items-center gap-0.5">
            {i > 0 && <span className="h-px w-2.5 shrink-0 bg-line-2" />}
            {s?.detail ? <Tip content={<span className="block max-w-xs">{s.detail}</span>}>{body}</Tip> : body}
          </span>
        );
      })}
    </div>
  );
}

/** Other honest ways to get an answer when the data does not cover the request. Nothing here changes the strategy silently. */
export function Alternatives({ alts, onPick, busy }: { alts: Alt[]; onPick: (a: Alt) => void; busy?: boolean }) {
  if (!alts?.length) return null;
  return (
    <div className="flex flex-col gap-1.5" data-testid="alternatives">
      {alts.map((x) => (
        <button key={x.id} disabled={busy} onClick={() => onPick(x)} data-alt={x.id}
          className={cn("flex items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors hover:bg-bg-2", x.warning ? "border-warn/40 hover:border-warn/70" : "border-line hover:border-accent/60")}>
          {x.id === "modeled" ? <ShieldAlert className="mt-0.5 size-4 shrink-0 text-warn" /> : x.id === "study" || x.id === "subset" ? <FlaskConical className="mt-0.5 size-4 shrink-0 text-accent" /> : <Database className="mt-0.5 size-4 shrink-0 dim" />}
          <span className="min-w-0 flex-1">
            <span className="block text-[13px] font-medium">{x.label}</span>
            {x.detail && <span className="mt-0.5 block text-xs dim">{x.detail}</span>}
          </span>
          <ArrowRight className="mt-1 size-3.5 shrink-0 text-faint" />
        </button>
      ))}
    </div>
  );
}

/** What each budget reaches, for this strategy's own sessions. */
export function BudgetTable({ b, pricedNow, of }: { b: Budget; pricedNow?: number; of?: number }) {
  const rows = b.budgets || [];
  return (
    <div data-testid="budget-table">
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="t">
          <thead><tr><th>Budget</th><th>What it buys</th><th className="r">Sessions reached</th><th className="r">With bid/ask</th><th className="r">Cost</th></tr></thead>
          <tbody>
            {pricedNow !== undefined && <tr><td className="dim">stored now</td><td className="soft">What is on this computer already</td><td className="r num">{pricedNow.toLocaleString()}</td><td className="r num dim">—</td><td className="r num">$0</td></tr>}
            {rows.map((r, i) => {
              const same = i > 0 && rows[i - 1].best && r.best && rows[i - 1].best!.label === r.best.label;
              return (
                <tr key={r.budget}>
                  <td className="num font-medium">${r.budget}</td>
                  <td className={cn(same && "dim")}><div className="w-[340px] whitespace-normal">{r.best ? (same ? "nothing more than the row above" : r.best.label) : "nothing reaches these sessions"}{r.best && !same && r.best.needs?.length ? <div className="mt-0.5 text-xs dim">needs: {r.best.needs.join("; ")}</div> : null}</div></td>
                  <td className="r num">{r.best ? `up to ${r.best.sessions.toLocaleString()}` : "0"}</td>
                  <td className="r num">{r.best ? r.best.sessions_with_bid_ask.toLocaleString() : "0"}</td>
                  <td className="r num">{r.best ? `$${r.best.cost.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-col gap-1 text-xs dim">
        {of !== undefined && b.reachable_sessions !== undefined && <div>Of {of.toLocaleString()} qualifying sessions: {b.no_contract_sessions ? `${b.no_contract_sessions.toLocaleString()} had no such contract listed; ` : ""}{b.before_any_source ? `${b.before_any_source.toLocaleString()} fall before ${b.earliest_source_date}, which no source compared here reaches at any price; ` : ""}<span className="soft">{b.reachable_sessions.toLocaleString()} could be reached by data at all.</span></div>}
        {b.denominator_note && <div>{b.denominator_note}</div>}
        <div>Prices and coverage are the providers' own published figures, read on {b.checked}. None of these routes has been run with a real key from this installation. Strata never buys anything.</div>
      </div>
    </div>
  );
}

const WHY: Record<string, string> = { missing: "no option prices stored", no_contract: "no such contract was listed", partial: "session held only in part", quotes: "bid/ask quotes", trades: "trade prints", eod: "end-of-day quotes",
  exit_uncertain: "entry priced, exit not observed", stale: "no trade at the entry", contract: "strike not stored", expiration: "expiration not stored" };
const TONE: Record<string, string> = { missing: "text-down", no_contract: "dim", partial: "text-warn", stale: "text-warn", contract: "text-warn", expiration: "text-warn", exit_uncertain: "text-info" };

/** The right-hand view of a data check: the fraction, every session and what is stored for it, what a budget would reach. */
export function DataCheckView({ check, onPick, busy }: { check: DataCheck; onPick: (a: Alt) => void; busy?: boolean }) {
  const nav = useNavigate();
  const [show, setShow] = useState<string>("untested");
  const rows = (check.sessions || []).filter((s) => show === "all" || (show === "untested" ? !PRICED.includes(s.status) : PRICED.includes(s.status)));
  const byYear: Record<string, { n: number; ok: number }> = {};
  for (const s of check.sessions || []) {
    const y = s.date.slice(0, 4);
    byYear[y] = byYear[y] || { n: 0, ok: 0 };
    byYear[y].n += 1;
    if (PRICED.includes(s.status)) byYear[y].ok += 1;
  }
  const years = Object.keys(byYear).sort();
  const maxN = Math.max(1, ...years.map((y) => byYear[y].n));
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="data-check-view">
      <div>
        <div className="eyebrow mb-2">Data check · before anything runs</div>
        <CoverageMeter check={check} />
        {check.text && <div className="mt-2.5 text-[13px] soft">{check.text}</div>}
      </div>
      {check.contract_existence?.note && <Callout tone="info" title="Whether the contract existed">{check.contract_existence.note}</Callout>}
      {years.length > 0 && (
        <Section title="By year" sub="sessions the rule fired on; the filled part can be priced">
          <div className="flex items-end gap-1 overflow-x-auto pb-1" style={{ height: 96 }}>
            {years.map((y) => (
              <Tip key={y} content={`${y}: the rule fired on ${byYear[y].n} session${byYear[y].n === 1 ? "" : "s"}; ${byYear[y].ok} can be priced`}>
                <div className="flex w-6 shrink-0 flex-col items-center justify-end gap-1" style={{ height: "100%" }}>
                  <div className="relative w-full overflow-hidden rounded-sm bg-down/25" style={{ height: `${Math.max(6, (byYear[y].n / maxN) * 64)}px` }}>
                    <div className="absolute inset-x-0 bottom-0 bg-up" style={{ height: `${pct(byYear[y].ok, byYear[y].n)}%` }} />
                  </div>
                  <span className="num text-[9.5px] text-faint">{y.slice(2)}</span>
                </div>
              </Tip>
            ))}
          </div>
        </Section>
      )}
      <Section title="Every session the rule fired on" sub={`${(check.sessions || []).length.toLocaleString()} listed`} right={
        <div className="flex gap-1">{[["untested", "Not priced"], ["priced", "Pass every check"], ["all", "All"]].map(([k, l]) => <button key={k} className={cn("btn quiet sm", show === k && "!bg-bg-3 !text-fg")} onClick={() => setShow(k)}>{l}</button>)}</div>}>
        <div className="max-h-[360px] overflow-auto rounded-lg border border-line">
          <table className="t" data-testid="session-list">
            <thead><tr><th>Session</th><th>Signal</th><th>Status</th><th>What was checked</th><th>Can be fetched</th></tr></thead>
            <tbody>
              {rows.length === 0 && <tr><td colSpan={5} className="dim">None.</td></tr>}
              {[...rows].reverse().map((s) => (
                <tr key={s.date} data-status={s.status}>
                  <td className="num">{s.date}</td>
                  <td className="num dim">{s.time || "—"}</td>
                  <td><span className={cn("inline-flex items-center gap-1.5", TONE[s.status] || "text-up")}><span className={cn("size-1.5 rounded-full", PARTS.find((x) => x.key === s.status)?.cls || "bg-faint")} />{WHY[s.status] || s.status}</span></td>
                  <td><div className="w-[380px] whitespace-normal text-xs soft">
                    {s.entry ? <><span className="num text-fg">{s.entry.contracts.join(" + ")}</span> entered {s.entry.time}{s.entry.print_age_seconds ? `, print ${Math.round(s.entry.print_age_seconds / 60)} min old` : ""}{s.exit ? `; exit ${s.exit.planned}, longest gap between prints ${s.exit.longest_gap_minutes} min` : ""}{s.status !== "trades" && s.detail ? `. ${s.detail}` : ""}</>
                      : (s.detail || (s.stored || []).join(", ") || "nothing stored")}
                  </div></td>
                  <td className="text-xs">{s.downloadable ? <span className="text-info">{s.downloadable.label}</span> : <span className="text-faint">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 text-xs dim">Each signal is taken through the engine's own contract selection{check.settings?.max_trade_age_seconds ? `: a print older than ${Math.round(check.settings.max_trade_age_seconds / 60)} minutes is not a fill, and an order waits ${check.settings.max_fill_delay_minutes} minutes` : ""}. Passing every check is still not a trade: a run confirms it. A session that could not be priced is not tested; it is never counted as a win, a loss or "no signal", and no price is invented for it.</div>
      </Section>
      {check.budget && <Section title="What a data budget would reach" sub="for these sessions only"><BudgetTable b={check.budget} pricedNow={check.priced} of={check.signal_sessions} /></Section>}
      {(check.alternatives || []).length > 0 && <Section title="Ways to get an answer" sub="the strategy is never changed to fit the data"><Alternatives alts={check.alternatives!} busy={busy} onPick={onPick} /></Section>}
      <div className="flex gap-2"><button className="btn sm" onClick={() => nav("/data?tab=readiness")}><Database />Open data readiness</button></div>
    </div>
  );
}

const money = (x: number | null | undefined) => (x === null || x === undefined ? "—" : (x < 0 ? "-$" : "$") + Math.abs(x).toFixed(2));

/** Every option fill of a backtest with the observation it came from. */
export function FillsView({ c }: { c: any }) {
  const f = c.fills || {};
  const rows: any[] = f.rows || [];
  const SRC: Record<string, string> = { quote: "bid/ask quote", trade_print: "trade print", model: "THEORETICAL", stock: "underlying" };
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="fills-view">
      <div><div className="eyebrow mb-1.5">Every option fill and where its price came from</div><div className="text-[13.5px] soft">{c.text}</div></div>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="t">
          <thead><tr><th>Session</th><th>Contract</th><th>Side</th><th className="r">Qty</th><th>In</th><th className="r">Fill</th><th className="r">Bid / ask then</th><th>Out</th><th className="r">Fill</th><th className="r">Bid / ask then</th><th>Closed by</th><th>Price source</th><th className="r">Leg P&amp;L</th></tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={13} className="dim">No fills: no qualifying session had stored option prices that could be used.</td></tr>}
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="num">{r.date}</td><td className="num">{r.contract}</td><td>{r.side}</td><td className="r num">{r.contracts}</td>
                <td className="num">{r.entry_time}</td><td className="r num">{money(r.entry_price)}</td>
                <td className="r num dim">{r.entry_bid !== null && r.entry_bid !== undefined ? `${money(r.entry_bid)} / ${money(r.entry_ask)}` : "none observed"}</td>
                <td className="num">{r.exit_time}</td><td className="r num">{money(r.exit_price)}</td>
                <td className="r num dim">{r.exit_bid !== null && r.exit_bid !== undefined ? `${money(r.exit_bid)} / ${money(r.exit_ask)}` : "none observed"}</td>
                <td>{String(r.exit_reason || "").replace(/_/g, " ")}</td>
                <td><span className={cn("chip !h-5 !text-[11px]", r.source === "model" && "!border-warn/50 !text-warn")}>{SRC[r.source] || r.source}</span></td>
                <td className={cn("r num", (r.pnl ?? 0) >= 0 ? "text-up" : "text-down")}>{money(r.pnl)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {f.count > rows.length && <div className="text-xs dim">Showing the first {rows.length.toLocaleString()} of {f.count.toLocaleString()} fills. The trade log export holds all of them.</div>}
      <Section title="Datasets these prices were read from">
        <div className="flex flex-col gap-2 text-[13px]">
          {Object.entries(f.datasets || {}).map(([sym, d]: any) => (
            <div key={sym}>
              <div><span className="font-medium">{sym}</span> <span className="num dim">{d.dataset}</span> · {d.observation} · {d.resolution}{d.license ? ` · ${d.license}` : ""}</div>
              {(d.limitations || []).length > 0 && <ul className="mt-1 flex flex-col gap-0.5 text-xs dim">{d.limitations.map((l: string, i: number) => <li key={i}>· {l}</li>)}</ul>}
            </div>
          ))}
          {Object.keys(f.datasets || {}).length === 0 && <div className="dim">No dataset was read.</div>}
        </div>
      </Section>
    </div>
  );
}

/** Exactly which qualifying sessions could not be tested, and why each one could not. */
export function UntestedView({ c }: { c: any }) {
  const u = c.untested || {};
  const rows: any[] = u.rows || [];
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="untested-view">
      <div><div className="eyebrow mb-2">Sessions that could not be tested</div>
        <CoverageHero n={u.priced ?? 0} of={u.of ?? 0} counts={u.counts} first={u.first} last={u.last} />
        <div className="mt-2.5 text-[13.5px] soft">{c.text}</div>
      </div>
      {u.contract_existence?.note && <Callout tone="info" title="Whether the contract existed">{u.contract_existence.note}</Callout>}
      <div className="max-h-[520px] overflow-auto rounded-lg border border-line">
        <table className="t" data-testid="untested-list">
          <thead><tr><th>Session</th><th>Why it could not be tested</th><th>Stored for it</th></tr></thead>
          <tbody>
            {[...rows].reverse().map((r) => <tr key={r.date}><td className="num">{r.date}</td><td className={cn(r.status === "no_contract" ? "dim" : r.status === "partial" ? "text-warn" : "soft")}>{r.reason}</td><td className="dim">{(r.stored || []).join(", ") || "nothing"}</td></tr>)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** What the missing observations would cost, per budget. */
export function CostView({ c, onPick, busy }: { c: any; onPick: (a: Alt) => void; busy?: boolean }) {
  const b = c.cost || {};
  return (
    <div className="flex flex-col gap-4 p-5" data-testid="cost-view">
      <div><div className="eyebrow mb-1.5">What the missing observations would cost</div><div className="text-[13.5px] soft">{c.text}</div></div>
      <BudgetTable b={b} pricedNow={b.priced_now} of={b.signal_sessions} />
      {(c.alternatives || []).length > 0 && <Section title="Ways to get an answer"><Alternatives alts={c.alternatives} busy={busy} onPick={onPick} /></Section>}
    </div>
  );
}
