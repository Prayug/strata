/* The one picture an answer gets, chosen by what was asked (V8 brief: "different questions require different visualizations").
   Hand-drawn SVG: every mark is data the server computed, every clickable mark opens the observations behind it. */
import { useEffect, useMemo, useRef, useState } from "react";
import { fmtDate, fmtPct, fmtSigned, fmtUsd } from "./data";

const C = { paper: "#ece8e0", paper2: "#c4beb3", mute: "#8f897f", faint: "#615d56", rule: "#262a32", rule2: "#353a44", amber: "#e8a33d", amber2: "#f3c071", up: "#45b58f", down: "#e3654f", ink: "#0c0d10" };

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setW(Math.max(280, Math.floor(e[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

type Open = (spec: any) => void;

/* ---------------------------------------------------------------------------------------------- how often */
export function FrequencyViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const n = v.n || 1;
  const hw = (v.hits / n) * w;
  const bx = v.baseline != null ? v.baseline * w : null;
  const years = (v.by_year || []).filter((y: any) => y.n > 0);
  const maxN = Math.max(1, ...years.map((y: any) => y.n));
  return (
    <div ref={ref} data-testid="viz-frequency">
      <div className="mb-2 flex items-baseline justify-between gap-4 text-[13px]">
        <span><span className="num" style={{ color: C.amber }}>{v.hits.toLocaleString()}</span> <span className="soft">days it happened</span></span>
        <span className="text-right"><span className="num soft">{v.misses.toLocaleString()}</span> <span className="mute">days it didn't</span></span>
      </div>
      <svg width={w} height={58} role="img" aria-label={`${v.hits} of ${v.n} days`}>
        <rect x={0} y={8} width={hw} height={30} fill={C.amber} opacity={0.88} style={{ cursor: "pointer" }} onClick={() => onOpen({ kind: "sessions", run_id: v.run_id, filter: "hit" })}><title>Show the {v.hits} days it happened</title></rect>
        <rect x={hw} y={8} width={Math.max(0, w - hw)} height={30} fill={C.rule2} style={{ cursor: "pointer" }} onClick={() => onOpen({ kind: "sessions", run_id: v.run_id, filter: "miss" })}><title>Show the {v.misses} days it did not</title></rect>
        {bx != null && (
          <g>
            <line x1={bx} x2={bx} y1={2} y2={44} stroke={C.paper} strokeWidth={1.5} strokeDasharray="3 3" />
            <text x={Math.min(Math.max(bx, 70), w - 70)} y={56} fill={C.mute} fontSize={11.5} textAnchor="middle">on any day: {fmtPct(v.baseline)}</text>
          </g>
        )}
      </svg>
      {years.length > 3 && (
        <div className="mt-5">
          <div className="kicker mb-2">Year by year</div>
          <svg width={w} height={64}>
            {years.map((y: any, i: number) => {
              const bw = w / years.length;
              const h = 40 * (y.probability ?? 0);
              const op = 0.35 + 0.65 * (y.n / maxN);
              return (
                <g key={y.year}>
                  <rect x={i * bw + 1} y={44 - h} width={Math.max(1, bw - 2)} height={h} fill={C.amber} opacity={op}><title>{`${y.year}: ${fmtPct(y.probability)} of ${y.n} days`}</title></rect>
                  {(years.length <= 16 || i % 2 === 0) && <text x={i * bw + bw / 2} y={60} fill={C.faint} fontSize={10.5} textAnchor="middle">{String(y.year).slice(-2)}</text>}
                </g>
              );
            })}
            {v.rate != null && <line x1={0} x2={w} y1={44 - 40 * v.rate} y2={44 - 40 * v.rate} stroke={C.paper2} strokeDasharray="2 4" />}
          </svg>
          <div className="tiny faint">Bar height is the share that year; fainter bars had fewer days.</div>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- outcome matrix */
export function MatrixViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const rows: any[] = v.rows || [];
  const mins = Array.from(new Set(rows.map((r) => r.minutes))).sort((a, b) => a - b);
  const scopes = Array.from(new Set(rows.map((r) => r.scope)));
  const [m, setM] = useState<number>(v.lead ?? mins[0]);
  const [s, setS] = useState<string>(scopes.includes("rest") ? "rest" : scopes[0]);
  const r = rows.find((x) => x.minutes === m && x.scope === s) || rows[0];
  if (!r) return null;
  const runs = [r.up_run, r.down_run];
  const later = s === "rest" ? "rest of the session" : v.predictor === "prev_direction" ? "next session" : "whole session";
  const first = v.predictor === "prev_direction" ? "Previous session" : `First ${r.minutes}-minute candle`;
  const cell = (k: string, n: number, agree: boolean, rowTotal: number, filter: string) => (
    <button key={k} onClick={() => onOpen({ kind: "classification", runs, filter })} className="flex flex-col items-start gap-1 border-0 p-4 text-left"
      style={{ background: agree ? "color-mix(in srgb, #e8a33d 14%, transparent)" : "transparent", outline: `1px solid ${C.rule}`, cursor: "pointer" }} data-testid={`cell-${k}`}>
      <span className="num text-[26px] leading-none" style={{ color: agree ? C.amber2 : C.paper2 }}>{n.toLocaleString()}</span>
      <span className="tiny mute">{rowTotal ? `${((n / rowTotal) * 100).toFixed(1)}% of these days` : ""}</span>
    </button>
  );
  return (
    <div data-testid="viz-matrix">
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px]">
        {mins.length > 1 && <span className="flex items-center gap-2"><span className="mute">Candle</span>{mins.map((x) => <button key={x} className="link" style={x === m ? { color: C.paper, textDecorationColor: C.amber } : undefined} onClick={() => setM(x)}>{x} min</button>)}</span>}
        {scopes.length > 1 && <span className="flex items-center gap-2"><span className="mute">Compared with</span>{scopes.map((x) => <button key={x} className="link" style={x === s ? { color: C.paper, textDecorationColor: C.amber } : undefined} onClick={() => setS(x)}>{x === "rest" ? "the rest of the day" : "the whole day"}</button>)}</span>}
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(92px, 150px) 1fr 1fr" }}>
        <div />
        <div className="tiny mute px-4 pb-2">{later[0].toUpperCase() + later.slice(1)} went up</div>
        <div className="tiny mute px-4 pb-2">{later[0].toUpperCase() + later.slice(1)} went down</div>
        <div className="small soft flex items-center pr-3">{first} up</div>
        {cell("up_up", r.up_up, true, r.up_up + r.up_down, "right")}
        {cell("up_down", r.up_down, false, r.up_up + r.up_down, "wrong")}
        <div className="small soft flex items-center pr-3">{first} down</div>
        {cell("down_up", r.down_up, false, r.down_up + r.down_down, "wrong")}
        {cell("down_down", r.down_down, true, r.down_up + r.down_down, "right")}
      </div>
      <div className="mt-4 grid gap-x-8 gap-y-1 text-[13px] sm:grid-cols-3">
        <div><span className="mute">Same way: </span><span className="num" style={{ color: C.amber }}>{fmtPct(r.agreement)}</span></div>
        {r.if_unrelated != null && <div><span className="mute">If unrelated: </span><span className="num">{fmtPct(r.if_unrelated)}</span></div>}
        {r.always_up != null && s === "rest" && <div><span className="mute">Always guessing up: </span><span className="num">{fmtPct(r.always_up)}</span></div>}
      </div>
      <div className="tiny faint mt-1">Highlighted boxes are the days both went the same way. "If unrelated" is what two independent directions with these frequencies would give, not 50%.</div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- price paths */
export function PathsViz({ v }: { v: any }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const ps = v.paths || {};
  const [hover, setHover] = useState<any>(null);
  const h = 260;
  const pad = { l: 44, r: 12, t: 10, b: 26 };
  const mins: number[] = ps.minutes || [];
  const all = useMemo(() => [...(ps.p10 || []), ...(ps.p90 || []), ...((ps.sample || []).filter((s: any) => !s.extreme).flatMap((s: any) => s.values) as number[])].filter((x) => x != null && Number.isFinite(x)), [ps]);
  if (!mins.length) return <div className="small mute">No intraday paths are stored for these days (daily bars only), so only the end results can be shown.</div>;
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const span = hi - lo || 1;
  lo -= span * 0.06; hi += span * 0.06;
  const X = (i: number) => pad.l + (mins[i] / mins[mins.length - 1]) * (w - pad.l - pad.r);
  const Y = (y: number) => pad.t + (1 - (y - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const line = (arr: (number | null)[]) => arr.map((y, i) => (y == null ? "" : `${i === 0 || arr[i - 1] == null ? "M" : "L"}${X(i).toFixed(1)},${Y(y).toFixed(1)}`)).join("");
  const band = (a: number[], b: number[]) => `M${a.map((y, i) => `${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join("L")}L${b.map((_, i) => b.length - 1 - i).map((i) => `${X(i).toFixed(1)},${Y(b[i]).toFixed(1)}`).join("L")}Z`;
  const ticks = [lo, 0, hi].map((t) => Math.round(t * 10) / 10);
  const tmax = mins[mins.length - 1];
  const fmtT = (m: number) => (tmax > 420 ? `${Math.round(m / 60)}h` : `${Math.round(m)}m`);
  return (
    <div ref={ref} data-testid="viz-paths">
      <svg width={w} height={h} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => <g key={t}><line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} stroke={t === 0 ? C.rule2 : C.rule} strokeDasharray={t === 0 ? undefined : "2 4"} /><text x={pad.l - 6} y={Y(t) + 4} fontSize={11} fill={C.faint} textAnchor="end">{t > 0 ? "+" : ""}{t}%</text></g>)}
        {ps.p10 && <path d={band(ps.p10, ps.p90)} fill={C.paper} opacity={0.06} />}
        {ps.p25 && <path d={band(ps.p25, ps.p75)} fill={C.paper} opacity={0.1} />}
        {(ps.sample || []).map((s: any, i: number) => (
          <path key={i} d={line(s.values)} fill="none" stroke={s.extreme ? (s.end >= 0 ? C.up : C.down) : C.paper2} strokeWidth={hover === s ? 2 : 1} opacity={hover === s ? 1 : s.extreme ? 0.5 : 0.16}
            onMouseEnter={() => setHover(s)} style={{ cursor: "default" }} />
        ))}
        {ps.median && <path d={line(ps.median)} fill="none" stroke={C.amber} strokeWidth={2.2} />}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => { const i = Math.round(f * (mins.length - 1)); return <text key={f} x={X(i)} y={h - 6} fontSize={11} fill={C.faint} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}>{fmtT(mins[i])}</text>; })}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px]">
        <span><span className="inline-block h-[2px] w-4 align-middle" style={{ background: C.amber }} /> <span className="mute">median day</span></span>
        <span><span className="inline-block h-2 w-4 align-middle" style={{ background: "rgba(236,232,224,.16)" }} /> <span className="mute">middle half / middle 80% of days</span></span>
        <span className="mute">{ps.days?.toLocaleString()} days, time from the start of the observation</span>
        {hover && <span className="soft">{fmtDate(hover.date)}: ended {fmtSigned(hover.end)}</span>}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- time of day */
export function TimingViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const b: any[] = v.buckets || [];
  const val = (x: any) => (v.measure === "abs_move" ? x.mean_abs_move_pct : x.share);
  const max = Math.max(...b.map(val), v.even || 0) || 1;
  const h = 200, pad = 26;
  const bw = w / Math.max(1, b.length);
  const top = b.reduce((a, x) => (val(x) > val(a) ? x : a), b[0] || {});
  return (
    <div ref={ref} data-testid="viz-timing">
      <svg width={w} height={h + pad}>
        {b.map((x, i) => {
          const hh = (val(x) / max) * (h - 18);
          return (
            <g key={i} onClick={() => onOpen({ kind: "timing", filter: x.start })} style={{ cursor: "pointer" }}>
              <rect x={i * bw + 2} y={h - hh} width={Math.max(2, bw - 4)} height={hh} fill={x === top ? C.amber : C.paper2} opacity={x === top ? 0.95 : 0.45}><title>{`${x.start}–${x.end}: ${v.measure === "abs_move" ? `${val(x).toFixed(3)}% average move` : `${val(x).toFixed(1)}% of days`}`}</title></rect>
              {(b.length <= 14 || i % 2 === 0) && <text x={i * bw + bw / 2} y={h + 16} fontSize={10.5} fill={C.faint} textAnchor="middle">{x.start}</text>}
            </g>
          );
        })}
        {v.measure !== "abs_move" && v.even ? <g><line x1={0} x2={w} y1={h - (v.even / max) * (h - 18)} y2={h - (v.even / max) * (h - 18)} stroke={C.paper} strokeDasharray="3 4" opacity={0.6} /><text x={w - 4} y={h - (v.even / max) * (h - 18) - 5} fontSize={11} fill={C.mute} textAnchor="end">if every half hour were equal</text></g> : null}
      </svg>
      <div className="tiny faint">New York time. Click a bar to see those days.</div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- groups (weekday, month) */
export function BreakdownViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const g: any[] = v.groups || [];
  // V10: a share of days per group ("how often Mondays close red") is drawn against the share on all days; an average
  // move is drawn against zero, as before
  const share = !!v.share;
  const val = (x: any) => (share ? x.share : x.mean_pct);
  const ci = (x: any) => (share ? x.share_ci95 : x.ci95);
  const base = share ? (v.overall_share ?? 50) : 0;
  const lo = Math.min(base, ...g.map((x) => ci(x)?.[0] ?? val(x))), hi = Math.max(base, ...g.map((x) => ci(x)?.[1] ?? val(x)));
  const h = 220, padB = 26;
  const Y = (y: number) => 8 + (1 - (y - lo) / (hi - lo || 1)) * (h - 16);
  const bw = w / Math.max(1, g.length);
  return (
    <div ref={ref} data-testid="viz-breakdown">
      <svg width={w} height={h + padB}>
        <line x1={0} x2={w} y1={Y(base)} y2={Y(base)} stroke={C.rule2} strokeDasharray={share ? "3 4" : undefined} />
        {share && <text x={w - 4} y={Y(base) - 5} fontSize={11} fill={C.mute} textAnchor="end">all days: {base.toFixed(1)}%</text>}
        {g.map((x, i) => {
          const cx = i * bw + bw / 2;
          const y0 = Y(base), y1 = Y(val(x));
          const c = ci(x);
          return (
            <g key={x.group} onClick={() => onOpen({ kind: "group", filter: x.group })} style={{ cursor: "pointer" }}>
              <rect x={cx - Math.min(28, bw * 0.3)} y={Math.min(y0, y1)} width={Math.min(56, bw * 0.6)} height={Math.max(1, Math.abs(y1 - y0))} fill={share ? C.amber : x.mean_pct >= 0 ? C.up : C.down} opacity={0.75}>
                <title>{share ? `${x.group}: ${x.share.toFixed(1)}% of ${x.sessions} days` : `${x.group}: ${x.mean_pct.toFixed(3)}% average over ${x.sessions} days`}</title></rect>
              {c && <g stroke={C.paper2}><line x1={cx} x2={cx} y1={Y(c[0])} y2={Y(c[1])} /><line x1={cx - 5} x2={cx + 5} y1={Y(c[0])} y2={Y(c[0])} /><line x1={cx - 5} x2={cx + 5} y1={Y(c[1])} y2={Y(c[1])} /></g>}
              {share && <text x={cx} y={Math.min(Y(c?.[1] ?? val(x)), y1) - 5} fontSize={11} fill={C.paper2} textAnchor="middle">{x.share.toFixed(1)}%</text>}
              <text x={cx} y={h + 16} fontSize={11} fill={C.mute} textAnchor="middle">{String(x.group).slice(0, bw < 50 ? 3 : 9)}</text>
            </g>
          );
        })}
      </svg>
      <div className="tiny faint">{share ? `Share of days ${v.label || ""} per group, drawn against the share on all days; the whisker is the range the share is likely in. Overlapping whiskers mean the difference is within chance.`
        : "Average move per group; the whisker is the range the average is likely in. Overlapping whiskers mean the difference is within chance."}</div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- account (a backtest) */
export function StrategyViz({ v, onTrade, onOpen }: { v: any; onTrade: (id: number) => void; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const t: any[] = v.trades || [];
  const [hover, setHover] = useState<any>(null);
  if (!t.length) return null;
  const h = 250, dotH = 70, padL = 56;
  const tot = t.map((x) => x.total);
  const lo = Math.min(0, ...tot), hi = Math.max(0, ...tot);
  const X = (i: number) => padL + (t.length === 1 ? 0.5 : i / (t.length - 1)) * (w - padL - 10);
  const Y = (y: number) => 10 + (1 - (y - lo) / (hi - lo || 1)) * (h - 20);
  const pmax = Math.max(...t.map((x) => Math.abs(x.pnl))) || 1;
  const D = (p: number) => h + dotH / 2 - (p / pmax) * (dotH / 2 - 4);
  return (
    <div ref={ref} data-testid="viz-strategy">
      <svg width={w} height={h + dotH + 18} onMouseLeave={() => setHover(null)}>
        {[lo, 0, hi].map((y, i) => <g key={i}><line x1={padL} x2={w - 10} y1={Y(y)} y2={Y(y)} stroke={y === 0 ? C.rule2 : C.rule} strokeDasharray={y === 0 ? undefined : "2 4"} /><text x={padL - 6} y={Y(y) + 4} fontSize={11} fill={C.faint} textAnchor="end">{fmtUsd(y)}</text></g>)}
        <path d={t.map((x, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(x.total).toFixed(1)}`).join("")} fill="none" stroke={C.amber} strokeWidth={2} />
        <line x1={padL} x2={w - 10} y1={h + dotH / 2} y2={h + dotH / 2} stroke={C.rule} />
        {t.map((x, i) => (
          <circle key={x.id} cx={X(i)} cy={D(x.pnl)} r={x.id === v.worst || x.id === v.best ? 4.2 : 2.6} fill={x.pnl >= 0 ? C.up : C.down} opacity={hover && hover !== x ? 0.35 : 0.9}
            onMouseEnter={() => setHover(x)} onClick={() => onTrade(x.id)} style={{ cursor: "pointer" }} />
        ))}
        <text x={padL} y={h + dotH + 14} fontSize={11} fill={C.faint}>{fmtDate(t[0].date)}</text>
        <text x={w - 10} y={h + dotH + 14} fontSize={11} fill={C.faint} textAnchor="end">{fmtDate(t[t.length - 1].date)}</text>
      </svg>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12.5px]">
        <span className="mute">Line: running total after fees. Dots: each trade, click one to replay it.</span>
        {hover ? <span className="soft">{fmtDate(hover.date)}: {fmtUsd(hover.pnl, 2)} · total {fmtUsd(hover.total, 2)}</span> :
          <button className="link" onClick={() => onOpen({ kind: "trades", run_id: v.run_id, filter: "losers" })}>Show the losing trades</button>}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------- how results were spread */
export function WinLossViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const t: any[] = v.trades || [];
  const [hover, setHover] = useState<number | null>(null);
  const bins = useMemo(() => {
    if (t.length < 4) return [];
    const p = t.map((x) => x.pnl);
    const lo = Math.min(...p), hi = Math.max(...p);
    const nb = Math.min(24, Math.max(8, Math.round(Math.sqrt(t.length) * 1.5)));
    const step = (hi - lo) / nb || 1;
    const out = Array.from({ length: nb }, (_, i) => ({ a: lo + i * step, b: lo + (i + 1) * step, n: 0 }));
    for (const x of p) out[Math.min(nb - 1, Math.floor((x - lo) / step))].n += 1;
    return out;
  }, [t]);
  if (!bins.length) return null;
  const h = 120, padL = 8, bw = (w - padL * 2) / bins.length;
  const mx = Math.max(...bins.map((b) => b.n));
  const wins = t.filter((x) => x.pnl > 0).length;
  return (
    <div ref={ref} className="mt-6" data-testid="viz-winloss">
      <div className="kicker mb-2">How the trades were spread</div>
      <svg width={w} height={h + 22} onMouseLeave={() => setHover(null)}>
        {bins.map((b, i) => {
          const bh = (b.n / mx) * h;
          const mid = (b.a + b.b) / 2;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onClick={() => onOpen({ kind: "trades", run_id: v.run_id, filter: mid > 0 ? "winners" : "losers" })} style={{ cursor: "pointer" }}>
              <rect x={padL + i * bw + 1} y={h - bh} width={Math.max(1, bw - 2)} height={bh} fill={mid > 0 ? C.up : C.down} opacity={hover === null || hover === i ? 0.85 : 0.35} />
              <rect x={padL + i * bw} y={0} width={bw} height={h} fill="transparent" />
            </g>
          );
        })}
        <line x1={padL} x2={w - padL} y1={h} y2={h} stroke={C.rule2} />
        <text x={padL} y={h + 15} fontSize={11} fill={C.faint}>{fmtUsd(bins[0].a)}</text>
        <text x={w - padL} y={h + 15} fontSize={11} fill={C.faint} textAnchor="end">{fmtUsd(bins[bins.length - 1].b)}</text>
      </svg>
      <div className="text-[12.5px] mute">{hover !== null ? <span className="soft">{bins[hover].n} trade{bins[hover].n === 1 ? "" : "s"} between {fmtUsd(bins[hover].a, 0)} and {fmtUsd(bins[hover].b, 0)}. Click to list them.</span> :
        <span>{wins} of {t.length} trades made money. Each bar counts trades by their result after fees; click one to list them.</span>}</div>
    </div>
  );
}

/* --------------------------------------------------------------------------- a group against the rest of it */
export function ContrastViz({ v }: { v: any }) {
  const rows: any[] = (v.rows || []).slice(0, 6);
  if (!rows.length) return null;
  const fmt = (r: any, x: number) => r.kind === "share" ? `${(x * 100).toFixed(0)}%` : r.feature === "premium" ? `$${x.toFixed(2)}` : r.feature === "vix" ? x.toFixed(1) : r.feature === "held" ? `${x.toFixed(0)} min`
    : r.feature === "entry_minute" ? `${String(Math.floor(x / 60)).padStart(2, "0")}:${String(Math.round(x % 60)).padStart(2, "0")}` : `${x > 0 ? "+" : ""}${x.toFixed(2)}%`;
  return (
    <div data-testid="viz-contrast">
      <table className="obs">
        <thead><tr><th>What was compared</th><th className="r">{v.group}</th><th className="r">{v.rest}</th><th className="r" title="How often random splits of the same observations give a difference at least this large">Random splits this different</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i}><td>{r.label}{r.category ? <span className="faint"> · {r.category}</span> : null}</td><td className="r num amber">{fmt(r, r.group)}</td><td className="r num">{fmt(r, r.rest)}</td>
            <td className="r num mute">{Number.isFinite(r.p) ? (r.p < 0.01 ? "under 1%" : `${(r.p * 100).toFixed(0)}%`) : "—"}</td></tr>
        ))}</tbody>
      </table>
      <div className="tiny mute mt-2">Each row: the average (or share) in the group and in the rest of the same result. The last column is how often splitting the same observations at random gives a difference at least as large; it is not the chance that the difference is real.</div>
    </div>
  );
}

/* ------------------------------------------------------------------------- the question before and after a change */
export function BeforeAfter({ before, after }: { before: any; after: any }) {
  const num = (a: any) => {
    const v = a?.viz || {};
    if (v.kind === "frequency" && v.n) return { value: v.hits / v.n, text: `${((v.hits / v.n) * 100).toFixed(1)}%`, n: `${v.hits.toLocaleString()} of ${v.n.toLocaleString()}` };
    if (v.kind === "strategy" && (v.trades || []).length) { const t = v.trades; const tot = t[t.length - 1].total; return { value: tot, text: fmtUsd(tot), n: `${t.length} trades` }; }
    return null;
  };
  const a = num(before), b = num(after);
  if (!a || !b || before?.viz?.kind !== after?.viz?.kind) return null;
  const mx = Math.max(Math.abs(a.value), Math.abs(b.value)) || 1;
  const bar = (x: any, label: string, now: boolean) => (
    <div className="flex items-center gap-3">
      <span className="w-[48px] shrink-0 tiny mute">{label}</span>
      <div className="h-[10px] min-w-[90px] flex-1" style={{ background: "var(--rule)" }}><div className="h-full" style={{ width: `${(Math.abs(x.value) / mx) * 100}%`, background: now ? C.amber : C.mute }} /></div>
      <span className={`num shrink-0 text-right ${now ? "amber" : "soft"}`}>{x.text}</span><span className="tiny faint hidden w-[120px] sm:inline">{x.n}</span>
    </div>
  );
  return <div className="mt-6 flex flex-col gap-1.5" data-testid="before-after"><div className="kicker mb-1">Before and after the change</div>{bar(a, "Before", false)}{bar(b, "After", true)}</div>;
}

/* --------------------------------------------------------------- after an up candle, after a down candle */
export function CandleOverlay({ o }: { o: any }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  if (!o?.up?.minutes || !o?.down?.minutes) return null;
  const h = 220, padL = 46, padB = 22;
  const all = [o.up, o.down].flatMap((x: any) => [...(x.p25 || []), ...(x.p75 || [])]).filter((v: any) => v != null && Number.isFinite(v));
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const mx = Math.max(o.up.minutes[o.up.minutes.length - 1], o.down.minutes[o.down.minutes.length - 1]) || 1;
  const X = (m: number) => padL + (m / mx) * (w - padL - 10);
  const Y = (v: number) => 8 + (1 - (v - lo) / (hi - lo || 1)) * (h - padB - 8);
  const line = (x: any, k: string) => x.minutes.map((m: number, i: number) => (x[k][i] == null ? null : `${X(m).toFixed(1)},${Y(x[k][i]).toFixed(1)}`)).filter(Boolean).join(" L");
  const band = (x: any) => {
    const top = x.minutes.map((m: number, i: number) => (x.p75[i] == null ? null : `${X(m).toFixed(1)},${Y(x.p75[i]).toFixed(1)}`)).filter(Boolean);
    const bot = x.minutes.map((m: number, i: number) => (x.p25[i] == null ? null : `${X(m).toFixed(1)},${Y(x.p25[i]).toFixed(1)}`)).filter(Boolean).reverse();
    return `M${top.join(" L")} L${bot.join(" L")} Z`;
  };
  const hours = Array.from({ length: Math.floor(mx / 60) + 1 }, (_, i) => i * 60);
  return (
    <div ref={ref} className="mt-8" data-testid="viz-candle-overlay">
      <div className="kicker mb-2">The rest of the session, after an up candle and after a down candle</div>
      <svg width={w} height={h}>
        {[lo, 0, hi].map((v, i) => <g key={i}><line x1={padL} x2={w - 10} y1={Y(v)} y2={Y(v)} stroke={v === 0 ? C.rule2 : C.rule} strokeDasharray={v === 0 ? undefined : "2 4"} /><text x={padL - 6} y={Y(v) + 4} fontSize={11} fill={C.faint} textAnchor="end">{v > 0 ? "+" : ""}{v.toFixed(2)}%</text></g>)}
        {hours.map((m) => <text key={m} x={X(m)} y={h - 6} fontSize={11} fill={C.faint} textAnchor="middle">{m === 0 ? "candle close" : `+${m / 60}h`}</text>)}
        <path d={band(o.up)} fill={C.up} opacity={0.12} />
        <path d={band(o.down)} fill={C.down} opacity={0.12} />
        <path d={`M${line(o.up, "median")}`} fill="none" stroke={C.up} strokeWidth={2} />
        <path d={`M${line(o.down, "median")}`} fill="none" stroke={C.down} strokeWidth={2} />
      </svg>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12.5px]">
        <span style={{ color: C.up }}>After an up first {o.minutes_candle}-minute candle ({(o.up.days || 0).toLocaleString()} days)</span>
        <span style={{ color: C.down }}>After a down one ({(o.down.days || 0).toLocaleString()} days)</span>
        <span className="mute">Lines: the median day, from the candle's close (0%). Shading: the middle half of the days. If the candle predicted the rest of the session, the two would pull apart.</span>
      </div>
    </div>
  );
}

export function CompareViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const ser: any[] = v.series || [];
  const all = ser.flatMap((s) => s.trades.map((t: any) => t.total));
  if (!all.length) return null;
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), h = 230, padL = 56;
  const dates = Array.from(new Set(ser.flatMap((s) => s.trades.map((t: any) => t.date)))).sort();
  const X = (d: string) => padL + (dates.length > 1 ? dates.indexOf(d) / (dates.length - 1) : 0.5) * (w - padL - 10);
  const Y = (y: number) => 10 + (1 - (y - lo) / (hi - lo || 1)) * (h - 20);
  const cols = [C.amber, C.paper2];
  return (
    <div ref={ref} data-testid="viz-compare">
      <svg width={w} height={h}>
        {[lo, 0, hi].map((y, i) => <g key={i}><line x1={padL} x2={w - 10} y1={Y(y)} y2={Y(y)} stroke={y === 0 ? C.rule2 : C.rule} strokeDasharray={y === 0 ? undefined : "2 4"} /><text x={padL - 6} y={Y(y) + 4} fontSize={11} fill={C.faint} textAnchor="end">{fmtUsd(y)}</text></g>)}
        {ser.map((s, k) => <path key={k} d={s.trades.map((t: any, i: number) => `${i ? "L" : "M"}${X(t.date).toFixed(1)},${Y(t.total).toFixed(1)}`).join("")} fill="none" stroke={cols[k % 2]} strokeWidth={2} strokeDasharray={k ? "5 4" : undefined} />)}
      </svg>
      <div className="flex flex-wrap gap-x-6 text-[12.5px]">
        {ser.map((s, k) => <button key={k} className="link" onClick={() => onOpen({ kind: "trades", run_id: s.run_id, filter: "all" })}><span className="mr-2 inline-block h-[2px] w-5 align-middle" style={{ background: cols[k % 2] }} />{s.label}</button>)}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- settings tried */
export function SweepViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const rows: any[] = v.rows || [];
  const vals = rows.map((r) => r.value as number);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), h = 190;
  const bw = w / Math.max(1, rows.length);
  const Y = (y: number) => 8 + (1 - (y - lo) / (hi - lo || 1)) * (h - 16);
  return (
    <div ref={ref} data-testid="viz-sweep">
      <svg width={w} height={h + 34}>
        <line x1={0} x2={w} y1={Y(0)} y2={Y(0)} stroke={C.rule2} />
        {rows.map((r, i) => (
          <g key={i} onClick={() => onOpen({ kind: v.study ? "sessions" : "trades", run_id: r.run_id, filter: "all" })} style={{ cursor: "pointer" }}>
            <rect x={i * bw + bw * 0.2} y={Math.min(Y(0), Y(r.value))} width={bw * 0.6} height={Math.abs(Y(r.value) - Y(0))} fill={i === v.best ? C.amber : r.value >= 0 ? C.paper2 : C.down} opacity={i === v.best ? 0.95 : 0.55}><title>{r.label}</title></rect>
            <text x={i * bw + bw / 2} y={h + 14} fontSize={10.5} fill={C.mute} textAnchor="middle">{String(r.label).slice(0, Math.max(4, Math.floor(bw / 6.5)))}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function VariantsViz({ v, onOpen }: { v: any; onOpen: Open }) {
  const rows: any[] = v.rows || [];
  return (
    <div className="flex flex-col gap-3" data-testid="viz-variants">
      {rows.map((r) => (
        <button key={r.run_id} onClick={() => onOpen({ kind: "sessions", run_id: r.run_id, filter: "all" })} className="grid items-center gap-3 border-0 bg-transparent p-0 text-left" style={{ gridTemplateColumns: "minmax(140px, 300px) 1fr 70px", cursor: "pointer" }}>
          <span className="small soft">{r.label}</span>
          <span className="relative h-[14px]" style={{ background: C.rule }}>
            <span className="absolute inset-y-0 left-0" style={{ width: `${(r.rate || 0) * 100}%`, background: C.amber, opacity: 0.85 }} />
            {r.baseline != null && <span className="absolute -inset-y-1 w-[2px]" style={{ left: `${r.baseline * 100}%`, background: C.paper }} />}
          </span>
          <span className="num text-right text-[13px]">{fmtPct(r.rate)} <span className="faint">{r.n}</span></span>
        </button>
      ))}
      <div className="tiny faint">White mark: the same outcome on any day. Click a row to see its days.</div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- illustration (not data) */
export function PayoffViz({ ill }: { ill: any }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const { strike: K, premium: p, multiplier: m = 100, contracts: n = 1, spot: S0, shares, right } = ill;
  const lo = S0 * 0.97, hi = S0 * 1.03, h = 200, padL = 56;
  const xs = Array.from({ length: 61 }, (_, i) => lo + ((hi - lo) * i) / 60);
  const opt = (s: number) => ((right === "C" ? Math.max(s - K, 0) : Math.max(K - s, 0)) - p) * m * n;
  const sh = (s: number) => (s - S0) * shares;
  const ys = xs.flatMap((s) => [opt(s), sh(s)]);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const X = (s: number) => padL + ((s - lo) / (hi - lo)) * (w - padL - 10);
  const Y = (y: number) => 10 + (1 - (y - ymin) / (ymax - ymin || 1)) * (h - 20);
  return (
    <div ref={ref} className="border p-4" style={{ borderColor: C.rule2, borderStyle: "dashed" }} data-testid="viz-payoff">
      <div className="kicker mb-1" style={{ color: C.amber }}>Illustration, not historical data</div>
      <div className="tiny mute mb-3">{ill.note}</div>
      <svg width={w - 32} height={h}>
        <line x1={padL} x2={w - 42} y1={Y(0)} y2={Y(0)} stroke={C.rule2} />
        <line x1={X(S0)} x2={X(S0)} y1={6} y2={h - 6} stroke={C.rule} strokeDasharray="2 3" />
        <path d={xs.map((s, i) => `${i ? "L" : "M"}${X(s).toFixed(1)},${Y(opt(s)).toFixed(1)}`).join("")} stroke={C.amber} strokeWidth={2} fill="none" />
        <path d={xs.map((s, i) => `${i ? "L" : "M"}${X(s).toFixed(1)},${Y(sh(s)).toFixed(1)}`).join("")} stroke={C.paper2} strokeWidth={1.5} strokeDasharray="5 4" fill="none" />
        <text x={padL - 6} y={Y(0) + 4} fontSize={11} fill={C.faint} textAnchor="end">$0</text>
        <text x={X(lo)} y={h - 2} fontSize={11} fill={C.faint}>${lo.toFixed(0)}</text>
        <text x={X(hi) - 4} y={h - 2} fontSize={11} fill={C.faint} textAnchor="end">${hi.toFixed(0)}</text>
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-5 text-[12.5px]">
        <span><span className="mr-2 inline-block h-[2px] w-5 align-middle" style={{ background: C.amber }} /><span className="mute">the {right === "C" ? "call" : "put"} at expiry</span></span>
        <span><span className="mr-2 inline-block h-[2px] w-5 align-middle" style={{ background: C.paper2 }} /><span className="mute">the same money in shares</span></span>
        <span className="mute">across the price at expiry (dotted line: price when bought)</span>
      </div>
    </div>
  );
}

export function Viz({ a, onOpen, onTrade }: { a: any; onOpen: Open; onTrade: (id: number) => void }) {
  const v = a?.viz;
  if (!v) return null;
  switch (v.kind) {
    case "frequency": return <FrequencyViz v={v} onOpen={onOpen} />;
    case "matrix": return <MatrixViz v={v} onOpen={onOpen} />;
    case "paths": return <PathsViz v={v} />;
    case "timing": return <TimingViz v={v} onOpen={onOpen} />;
    case "breakdown": return <BreakdownViz v={v} onOpen={onOpen} />;
    case "strategy": return <div><StrategyViz v={v} onTrade={onTrade} onOpen={onOpen} /><WinLossViz v={v} onOpen={onOpen} /></div>;
    case "contrast": return <ContrastViz v={v} />;
    case "compare_strategies": return <CompareViz v={v} onOpen={onOpen} />;
    case "sweep": return <SweepViz v={v} onOpen={onOpen} />;
    case "variants": return <VariantsViz v={v} onOpen={onOpen} />;
    default: return null;
  }
}
